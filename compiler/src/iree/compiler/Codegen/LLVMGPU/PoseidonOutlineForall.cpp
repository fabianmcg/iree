// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "poseidon/Dialect/Poseidon/IR/PoseidonOps.h"

namespace mlir::iree_compiler {
#define GEN_PASS_DEF_POSEIDONOUTLINEFORALL
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

/// Wraps the given operation `op` into an `scf.execute_region` operation.
static scf::ExecuteRegionOp wrapInExecuteRegion(RewriterBase &b,
                                                Operation *op) {
  if (op->getNumRegions() != 1)
    return nullptr;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  scf::ExecuteRegionOp executeRegionOp =
      scf::ExecuteRegionOp::create(b, op->getLoc(), op->getResultTypes());
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToEnd(&executeRegionOp.getRegion().emplaceBlock());
    IRMapping mapping;
    SmallVector<Operation *> constLike;
    op->getParentOfType<FunctionOpInterface>().walk(
        [&](Operation *cOp) -> WalkResult {
          if (op == cOp)
            return WalkResult::skip();
          if (cOp->hasTrait<OpTrait::ConstantLike>())
            constLike.push_back(cOp);
          return WalkResult::advance();
        });
    for (Operation *op : constLike) {
      b.clone(*op, mapping);
    }
    Operation *clonedOp = b.clone(*op, mapping);
    scf::YieldOp::create(b, op->getLoc(), clonedOp->getResults());
  }
  b.replaceOp(op, executeRegionOp.getResults());
  return executeRegionOp;
}

namespace {
struct PoseidonOutlineForall final
    : impl::PoseidonOutlineForallBase<PoseidonOutlineForall> {
  using Base::Base;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, scf::SCFDialect,
                    poseidon::PoseidonDialect>();
  }

  void runOnOperation() override {
    Operation *rootOp = getOperation();
    IRRewriter rewriter(rootOp->getContext());
    llvm::DenseMap<Operation *, SymbolTable> symbolTables;

    // Walk the operation to find all scf::ForallOp instances
    SmallVector<scf::ForallOp> forallOps;
    rootOp->walk(
        [&](scf::ForallOp forallOp) { forallOps.push_back(forallOp); });

    // Outline each ForallOp
    for (scf::ForallOp forallOp : forallOps) {
      Location location = forallOp.getLoc();
      Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(forallOp);

      // Wrap the ForallOp in an ExecuteRegionOp
      rewriter.setInsertionPoint(forallOp);
      scf::ExecuteRegionOp exec = wrapInExecuteRegion(rewriter, forallOp);
      if (!exec) {
        forallOp.emitError("failed to wrap ForallOp in ExecuteRegionOp");
        return signalPassFailure();
      }

      // Outline the region into a function
      func::CallOp call;
      FailureOr<func::FuncOp> outlined = outlineSingleBlockRegion(
          rewriter, location, exec.getRegion(), "forall", &call);

      if (failed(outlined)) {
        forallOp.emitError("failed to outline ForallOp");
        return signalPassFailure();
      }

      // Move the outlined function into the module.
      rewriter.setInsertionPoint(call->getParentOfType<FunctionOpInterface>());
      ModuleOp module =
          ModuleOp::create(rewriter, rootOp->getLoc(), "_$outlined");
      rewriter.moveOpBefore(*outlined, module.getBody(),
                            module.getBody()->begin());

      assert(symbolTableOp);
      // Insert the function into the module symbol table
      SymbolTable &symbolTable =
          symbolTables.try_emplace(symbolTableOp, symbolTableOp)
              .first->getSecond();
      symbolTable.insert(module);

      // Amend the call operation.
      auto calleeName =
          SymbolRefAttr::get(FlatSymbolRefAttr::get(module).getAttr(),
                             {FlatSymbolRefAttr::get(*outlined)});
      rewriter.setInsertionPoint(call);
      poseidon::CallOp::create(rewriter, location, calleeName,
                               TypeRange(call.getResults()),
                               call.getOperands());
      rewriter.eraseOp(call);
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
