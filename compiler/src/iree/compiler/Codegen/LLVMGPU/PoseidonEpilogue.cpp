// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {
#define GEN_PASS_DEF_POSEIDONEPILOGUE
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {
struct PoseidonEpilogue final : impl::PoseidonEpilogueBase<PoseidonEpilogue> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpPassManager modulePassManager("builtin.module");
    modulePassManager.addPass(createCanonicalizerPass());
    modulePassManager.addPass(createCSEPass());
    modulePassManager.addPass(createCanonicalizerPass());
    modulePassManager.addPass(createInlinerPass());
    modulePassManager.addPass(createCanonicalizerPass());
    modulePassManager.addPass(createCSEPass());
    modulePassManager.addPass(createCanonicalizerPass());
    modulePassManager.addPass(createSymbolDCEPass());
    if (failed(runPipeline(modulePassManager, mod)))
      return signalPassFailure();
  }
};
} // namespace
} // namespace mlir::iree_compiler
