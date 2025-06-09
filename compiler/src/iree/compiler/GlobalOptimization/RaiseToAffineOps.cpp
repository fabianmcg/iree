// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_RAISETOAFFINEOPSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

static void populateRaiseToAffinePatterns(RewritePatternSet &patterns);

namespace {
class RaiseToAffineOpsPass
    : public impl::RaiseToAffineOpsPassBase<RaiseToAffineOpsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateRaiseToAffinePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

template <typename Op>
struct GenericRaiseToAffinePattern : public OpRewritePattern<Op> {
  GenericRaiseToAffinePattern(AffineExpr expr, PatternBenefit benefit = 1,
                              ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern<Op>(expr.getContext(), benefit, generatedNames),
        expr(expr) {}
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    if (!isa<IndexType>(op.getType()))
      return failure();
    SmallVector<OpFoldResult> operands;
    operands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands())
      operands.push_back(operand);
    affine::AffineApplyOp result =
        affine::makeComposedAffineApply(rewriter, op.getLoc(), expr, operands);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  AffineExpr expr;
};
} // namespace

template <typename ArithOp, typename IndexOp>
static void addPattern(RewritePatternSet &patterns, AffineExpr expr) {
  patterns.template add<GenericRaiseToAffinePattern<ArithOp>,
                        GenericRaiseToAffinePattern<IndexOp>>(expr);
}

void populateRaiseToAffinePatterns(RewritePatternSet &patterns) {
  AffineExpr lhs, rhs;
  bindSymbols(patterns.getContext(), lhs, rhs);
  addPattern<arith::AddIOp, index::AddOp>(patterns, lhs + rhs);
  addPattern<arith::SubIOp, index::SubOp>(patterns, lhs - rhs);
  addPattern<arith::MulIOp, index::MulOp>(patterns, lhs * rhs);
  // TODO: Add an attribute to the following arith and index ops to guruantee
  // the RHS is positive, and lower affine to the ops below.
  addPattern<arith::RemSIOp, index::RemSOp>(patterns, lhs % rhs);
  addPattern<arith::CeilDivSIOp, index::CeilDivSOp>(patterns, lhs.ceilDiv(rhs));
  addPattern<arith::FloorDivSIOp, index::FloorDivSOp>(patterns,
                                                      lhs.floorDiv(rhs));
}
} // namespace mlir::iree_compiler::GlobalOptimization
