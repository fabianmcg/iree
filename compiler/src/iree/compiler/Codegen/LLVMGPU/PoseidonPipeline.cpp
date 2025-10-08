// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "poseidon/Dialect/Poseidon/IR/PoseidonOps.h"
#include "poseidon/Transforms/Passes.h"
#include "poseidon/Transforms/TritonPipelines.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define DEBUG_TYPE "iree-poseidon"

namespace mlir::iree_compiler {
#define GEN_PASS_DEF_POSEIDONPIPELINE
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

static llvm::cl::opt<int>
    clIndexingBits("iree-poseidon-index-bits",
                   llvm::cl::desc("Set the bit width of indices in ROCm."),
                   llvm::cl::init(64));

static llvm::cl::opt<int> clNumStages(
    "iree-poseidon-num-stages",
    llvm::cl::desc("Number of pipeline stages for Triton compilation."),
    llvm::cl::init(3));

static llvm::cl::opt<int> clMatrixInstrNonKDim(
    "iree-poseidon-matrix-instr-non-k-dim",
    llvm::cl::desc("Matrix instruction non-K dimension size."),
    llvm::cl::init(16));

static llvm::cl::opt<int>
    clKpack("iree-poseidon-kpack",
            llvm::cl::desc("K-dimension packing factor for matrix operations."),
            llvm::cl::init(4));

static llvm::cl::opt<std::string>
    clScheduleHint("iree-poseidon-schedule-hint",
                   llvm::cl::desc("Schedule hint for Triton compilation."),
                   llvm::cl::init("none"));

static llvm::cl::opt<bool>
    clUseAsyncCopy("iree-poseidon-use-async-copy",
                   llvm::cl::desc("Enable asynchronous copy operations."),
                   llvm::cl::init(false));

static llvm::cl::opt<bool>
    clUseBufferOps("iree-poseidon-use-buffer-ops",
                   llvm::cl::desc("Enable buffer operations."),
                   llvm::cl::init(true));

static llvm::cl::opt<bool>
    clUseBufferAtomics("iree-poseidon-use-buffer-atomics",
                       llvm::cl::desc("Enable buffer atomic operations."),
                       llvm::cl::init(true));

static llvm::cl::opt<bool>
    clUseBlockPingpong("iree-poseidon-use-block-pingpong",
                       llvm::cl::desc("Enable block ping-pong optimization."),
                       llvm::cl::init(false));

static llvm::cl::opt<bool> clIsInThreadTransposeEnabled(
    "iree-poseidon-in-thread-transpose",
    llvm::cl::desc("Enable in-thread transpose optimization."),
    llvm::cl::init(true));

static llvm::cl::opt<bool>
    clHipFTZ("iree-poseidon-hip-ftz",
             llvm::cl::desc("Enable HIP flush-to-zero for denormal numbers."),
             llvm::cl::init(true));

namespace {
struct IREEConf {
  std::string chip = "gfx942";
  int32_t warpSize = 64;
  int32_t numWarps = 1;
  LogicalResult init(ModuleOp op);
};

LogicalResult IREEConf::init(ModuleOp op) {
  int32_t numFuncs = 0;
  for (func::FuncOp func : op.getOps<func::FuncOp>()) {
    numFuncs++;
    std::optional<SmallVector<int64_t>> wgSz = getWorkgroupSize(func);
    if (!wgSz)
      return failure();
    std::optional<int64_t> warpSz = getSubgroupSize(func);
    if (!warpSz)
      return failure();
    warpSize = *warpSz;
    int64_t numW = 1;
    for (int64_t i : *wgSz)
      numW *= i;
    assert(numW % warpSize == 0);
    assert(llvm::count(*wgSz, 1) == (wgSz->size() - 1));
    numWarps = numW / warpSize;
  }
  IREE::GPU::TargetAttr tgt = getGPUTargetAttr(op);
  if (!tgt)
    return failure();
  chip = tgt.getArch().str();
  return success(numFuncs == 1);
}

static FailureOr<int64_t> setWorkgroupCount(ModuleOp mod) {
  int32_t numEntries = 0, numForalls = 0;
  scf::ForallOp forall;
  func::FuncOp entryPoint;
  for (func::FuncOp entry : mod.getOps<func::FuncOp>()) {
    entryPoint = entry;
    entry.walk([&](scf::ForallOp op) {
      forall = op;
      ++numForalls;
    });
    ++numEntries;
  }
  if (!(numEntries == 1 && numForalls == 1)) {
    return mod.emitError() << "expected a single dispatch with a single forall";
  }
  IRRewriter rewriter(mod);
  rewriter.setInsertionPoint(forall);
  SmallVector<OpFoldResult> sizes;

  AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
  AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
  AffineExpr s2 = rewriter.getAffineSymbolExpr(2);

  int64_t numWorkgroups = 1;
  for (auto [upper, lower, step] :
       llvm::zip(forall.getMixedUpperBound(), forall.getMixedLowerBound(),
                 forall.getMixedStep())) {
    OpFoldResult size = affine::makeComposedFoldedAffineApply(
        rewriter, forall.getLoc(), (s0 - s1).ceilDiv(s2), {upper, lower, step});
    sizes.push_back(size);

    if (numWorkgroups <= 0)
      continue;

    std::optional<int64_t> sV = getConstantIntValue(size);
    if (!sV) {
      numWorkgroups = -1;
      continue;
    }
    numWorkgroups *= *sV;
  }

  if (failed(lowerWorkgroupCountFromSliceOp(rewriter, entryPoint, sizes, 1)))
    return failure();
  return numWorkgroups;
}

struct PoseidonPipeline final : impl::PoseidonPipelineBase<PoseidonPipeline> {
  using Base::Base;

  LogicalResult runToTTIR(ModuleOp mod, IREEConf &conf,
                          int64_t countWorkgroups) {
    OpPassManager modulePassManager("builtin.module");
    modulePassManager.addPass(createPoseidonOutlineForall());
    OpPassManager &ttPm = modulePassManager.nest<ModuleOp>();
    OpPassManager &ttFnPm = ttPm.nest<func::FuncOp>();
    ttFnPm.addPass(poseidon::createForallToTriton());
    ttFnPm.addPass(createCanonicalizerPass());
    ttFnPm.addPass(createCSEPass());
    poseidon::SetStaticProgramInfoOptions opts;
    if (countWorkgroups > 0) {
      opts.staticProgramDims.push_back(countWorkgroups);
    }
    ttFnPm.addPass(poseidon::createSetStaticProgramInfo(opts));
    ttFnPm.addPass(createCanonicalizerPass());
    ttFnPm.addPass(createCSEPass());
    ttFnPm.addPass(createLowerAffinePass());
    ttFnPm.addPass(createCanonicalizerPass());
    ttFnPm.addPass(createCSEPass());
    ttPm.addPass(poseidon::createOptimizeArith());
    ttPm.addPass(createCanonicalizerPass());
    ttPm.addPass(createCSEPass());
    ttPm.nest<func::FuncOp>().addPass(poseidon::createSimplifyVectorOps());
    ttPm.addPass(poseidon::createConvertToTriton({clIndexingBits == 32}));
    OpPassManager &ttirFnPm = ttPm.nest<triton::FuncOp>();
    ttirFnPm.addPass(createCanonicalizerPass());
    ttirFnPm.addPass(createCSEPass());
    ttirFnPm.addPass(createLoopInvariantCodeMotionPass());
    ttirFnPm.addPass(createCanonicalizerPass());
    ttirFnPm.addPass(createCSEPass());
    ttirFnPm.addPass(poseidon::createPoseidonToTriton());
    ttirFnPm.addPass(createCanonicalizerPass());
    ttirFnPm.addPass(createCSEPass());
    ttirFnPm.addPass(createLoopInvariantCodeMotionPass());
    ttirFnPm.addPass(createCanonicalizerPass());
    ttirFnPm.addPass(createCSEPass());
    return runPipeline(modulePassManager, mod);
  }

  LogicalResult runTTIR(ModuleOp mod, IREEConf &conf) {
    OpPassManager modulePassManager("builtin.module");
    modulePassManager.addPass(createPoseidonOutlineForall());
    OpPassManager &ttPm = modulePassManager.nest<ModuleOp>();
    triton::populateTritonToTTIR(ttPm);
    triton::populateTritonToTTGIR(
        ttPm, /*arch=*/conf.chip,
        /*numWarps=*/conf.numWarps,
        /*warpSize=*/conf.warpSize,
        /*numCTAs=*/1,
        /*numStages=*/clNumStages,
        /*matrixInstrNonKDim=*/clMatrixInstrNonKDim,
        /*kpack=*/clKpack,
        /*scheduleHint=*/clScheduleHint,
        /*useAsyncCopy=*/clUseAsyncCopy,
        /*useBufferOps=*/clUseBufferOps,
        /*useBufferAtomics=*/clUseBufferAtomics,
        /*useBlockPingpong=*/clUseBlockPingpong,
        /*isInThreadTransposeEnabled=*/clIsInThreadTransposeEnabled);
    triton::populateTritonToLLVM(ttPm, /*arch=*/conf.chip,
                                 /*numStages=*/clNumStages,
                                 /*scheduleHint=*/clScheduleHint,
                                 /*disableLineInfo=*/true,
                                 /*hipFTZ=*/clHipFTZ);
    modulePassManager.addPass(
        createPoseidonInlineKernel({clIndexingBits == 32}));
    return runPipeline(modulePassManager, mod);
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!llvm::all_of(mod.getOps<func::FuncOp>(), [](func::FuncOp funcOp) {
          IREE::Codegen::TranslationInfoAttr translationInfo =
              getTranslationInfo(funcOp);
          return translationInfo
                     ? translationInfo.getPassPipeline().getValue() ==
                           IREE::Codegen::DispatchLoweringPassPipeline::
                               LLVMGPUPoseidon
                     : false;
        }))
      return;
    LDBG() << "Input module:\n" << mod;
    FailureOr<int64_t> countWorkgroups = setWorkgroupCount(mod);
    if (failed(countWorkgroups))
      return signalPassFailure();

    IREEConf conf;
    if (failed(conf.init(mod)))
      return signalPassFailure();

    if (failed(runToTTIR(mod, conf, *countWorkgroups)))
      return signalPassFailure();

    LDBG() << "ToTTIR:\n" << mod;

    if (failed(runTTIR(mod, conf)))
      return signalPassFailure();

    LDBG(2) << "Output module:\n" << mod;
  }
};
} // namespace
} // namespace mlir::iree_compiler
