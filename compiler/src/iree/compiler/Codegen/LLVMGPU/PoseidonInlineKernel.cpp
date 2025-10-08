// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Passes.h"
#include "poseidon/Dialect/Poseidon/IR/PoseidonOps.h"

namespace mlir::iree_compiler {
#define GEN_PASS_DEF_POSEIDONINLINEKERNEL
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

static void handleCall(poseidon::CallOp call, IRRewriter &rewriter,
                       const LLVMTypeConverter &converter) {
  rewriter.setInsertionPoint(call);
  SmallVector<Value> args;
  for (Value arg : call.getOperands()) {
    auto mTy = dyn_cast<MemRefType>(arg.getType());
    if (!mTy) {
      args.push_back(arg);
      continue;
    }
    {
      FailureOr<PtrLikeTypeInterface> tmpTy =
          mTy.clonePtrWith(rewriter.getI64IntegerAttr(1), std::nullopt);
      assert(succeeded(tmpTy) && "Failed to create clone");
      mTy = cast<MemRefType>(*tmpTy);
    }
    Value mVal =
        memref::MemorySpaceCastOp::create(rewriter, arg.getLoc(), mTy, arg);
    auto descTy = converter.convertType(mTy);
    assert(descTy && "Failed to convert memory type");
    mVal = UnrealizedConversionCastOp::create(rewriter, arg.getLoc(),
                                              TypeRange({descTy}), mVal)
               ->getResult(0);
    MemRefDescriptor descriptor(mVal);
    args.push_back(descriptor.alignedPtr(rewriter, arg.getLoc()));
    auto [strides, offset] = mTy.getStridesAndOffset();
    if (ShapedType::isDynamic(offset)) {
      args.push_back(descriptor.offset(rewriter, arg.getLoc()));
    }
    for (auto [i, dim] : llvm::enumerate(mTy.getShape())) {
      if (!ShapedType::isDynamic(dim))
        continue;
      args.push_back(descriptor.size(rewriter, arg.getLoc(), i));
    }
    for (auto [i, dim] : llvm::enumerate(strides)) {
      if (!ShapedType::isDynamic(dim))
        continue;
      args.push_back(descriptor.stride(rewriter, arg.getLoc(), i));
    }
  }
  SymbolRefAttr name = call.getCalleeAttr();
  FlatSymbolRefAttr kerName = name.getNestedReferences().front();
  std::string callee =
      (name.getRootReference().getValue() + "$_" + kerName.getValue()).str();
  auto nullPtr = LLVM::ZeroOp::create(
      rewriter, call.getLoc(),
      LLVM::LLVMPointerType::get(rewriter.getContext(), 1));
  args.push_back(nullPtr);
  args.push_back(nullPtr);
  auto newCall =
      LLVM::CallOp::create(rewriter, call.getLoc(), TypeRange(), callee, args);
  rewriter.replaceOp(call, newCall);
}

static SymbolOpInterface handleSharedMem(SymbolTable &table,
                                         LLVM::GlobalOp global,
                                         IRRewriter &rewriter,
                                         const LLVMTypeConverter &converter) {
  int64_t smem = cast<IntegerAttr>(
                     global->getParentOfType<ModuleOp>()->getAttr("ttg.shared"))
                     .getValue()
                     .getSExtValue();
  std::optional<SymbolTable::UseRange> uses =
      table.getSymbolUses(global.getSymNameAttr(), table.getOp());
  if (smem == 0 || !uses || uses->empty()) {
    rewriter.eraseOp(global);
    return nullptr;
  }
  auto arrayTy = LLVM::LLVMArrayType::get(
      converter.convertType(rewriter.getIntegerType(8)), smem);
  rewriter.setInsertionPoint(global);
  auto globalStatic = LLVM::GlobalOp::create(
      rewriter, global.getLoc(), arrayTy, /*isConstant=*/false,
      LLVM::Linkage::Internal, "global_smem", /*value=*/Attribute(),
      /*alignment=*/16,
      // Add ROCm support.
      static_cast<unsigned>(3));
  table.remove(global);
  rewriter.replaceOp(global, globalStatic);
  table.insert(globalStatic);
  return globalStatic;
}

namespace {
struct PoseidonInlineKernel final
    : impl::PoseidonInlineKernelBase<PoseidonInlineKernel> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleOp kernelMod;
    int32_t numMods = 0;
    for (ModuleOp modOp : mod.getOps<ModuleOp>()) {
      kernelMod = modOp;
      ++numMods;
    }
    if (numMods > 1) {
      return signalPassFailure();
    } else if (numMods == 0) {
      return;
    }
    IRRewriter rewriter(&getContext());
    LowerToLLVMOptions opts(&getContext());
    if (use32BitIndex) {
      opts.overrideIndexBitwidth(32);
    }
    mlir::LLVMTypeConverter converter(&getContext(), opts);
    std::optional<StringRef> modName = kernelMod.getName();
    SymbolTable symbolTable(kernelMod);
    assert(modName && "Expected module name");
    for (Operation &op :
         llvm::make_early_inc_range(kernelMod.getBody()->getOperations())) {
      auto symbol = dyn_cast<SymbolOpInterface>(op);
      assert(symbol && "Expected SymbolOpInterface");
      if (auto fnOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        fnOp->removeAttr("nvvm.kernel");
        fnOp->removeAttr("nvvm.reqntid");
        fnOp.setAlwaysInline(true);
        symbol.setVisibility(SymbolTable::Visibility::Private);
      }
      if (symbol.getName() == "global_smem") {
        auto op = handleSharedMem(symbolTable, cast<LLVM::GlobalOp>(symbol),
                                  rewriter, converter);
        if (!op)
          continue;
        symbol = op;
      }
      if (failed(symbolTable.rename(
              symbol, (*modName + "$_" + symbol.getName()).str())))
        return signalPassFailure();
    }
    for (Operation &op :
         llvm::make_early_inc_range(kernelMod.getBody()->getOperations())) {
      auto symbol = dyn_cast<SymbolOpInterface>(op);
      assert(symbol && "Expected SymbolOpInterface");
      rewriter.moveOpBefore(&op, kernelMod);
    }
    mod.walk(
        [&](poseidon::CallOp call) { handleCall(call, rewriter, converter); });
    rewriter.eraseOp(kernelMod);
  }
};
} // namespace
} // namespace mlir::iree_compiler
