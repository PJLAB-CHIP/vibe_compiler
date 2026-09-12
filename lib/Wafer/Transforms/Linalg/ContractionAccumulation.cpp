//===- ContractionAccumulation.cpp - Explicit wide accumulation --------===//

#include "Wafer/Transforms/Linalg/ContractionAccumulation.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

#include <iterator>

namespace wafer {

#define GEN_PASS_DEF_PROMOTECONTRACTIONACCUMULATIONPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"

namespace {

bool isNarrowContraction(mlir::linalg::LinalgOp op) {
  if (!op.hasPureTensorSemantics() || op.getNumDpsInputs() != 2 ||
      op.getNumDpsInits() != 1 || op->getNumResults() != 1 ||
      !mlir::linalg::isaContractionOpInterface(op))
    return false;
  auto type =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!type ||
      !(type.getElementType().isF16() || type.getElementType().isBF16()))
    return false;
  auto &body = op->getRegion(0).front();
  if (body.getNumArguments() != 3 ||
      llvm::range_size(body.without_terminator()) != 2 ||
      llvm::any_of(body.getArguments(), [&](mlir::BlockArgument arg) {
        return arg.getType() != type.getElementType();
      }))
    return false;
  auto multiply = mlir::dyn_cast<mlir::arith::MulFOp>(body.front());
  auto add = mlir::dyn_cast<mlir::arith::AddFOp>(*std::next(body.begin()));
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  auto pair = [](mlir::Value a, mlir::Value b, mlir::Value x, mlir::Value y) {
    return (a == x && b == y) || (a == y && b == x);
  };
  return multiply && add && yield && yield.getNumOperands() == 1 &&
         pair(multiply.getLhs(), multiply.getRhs(), body.getArgument(0),
              body.getArgument(1)) &&
         pair(add.getLhs(), add.getRhs(), multiply, body.getArgument(2)) &&
         yield.getOperand(0) == add.getResult();
}

mlir::Value emptyLike(mlir::IRRewriter &rewriter, mlir::Location loc,
                      mlir::Value source, mlir::Type element) {
  auto type = mlir::cast<mlir::RankedTensorType>(source.getType());
  llvm::SmallVector<mlir::Value> dynamic;
  for (int64_t dim = 0; dim < type.getRank(); ++dim)
    if (type.isDynamicDim(dim))
      dynamic.push_back(rewriter.create<mlir::tensor::DimOp>(loc, source, dim));
  return rewriter.create<mlir::tensor::EmptyOp>(loc, type.getShape(), element,
                                                dynamic, type.getEncoding());
}

mlir::Value castTensor(mlir::IRRewriter &rewriter, mlir::Location loc,
                       mlir::Value input, mlir::Type element) {
  auto type = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto empty = emptyLike(rewriter, loc, input, element);
  auto map = rewriter.getMultiDimIdentityMap(type.getRank());
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      type.getRank(), mlir::utils::IteratorType::parallel);
  return rewriter
      .create<mlir::linalg::GenericOp>(
          loc, mlir::TypeRange{empty.getType()}, mlir::ValueRange{input},
          mlir::ValueRange{empty}, llvm::ArrayRef<mlir::AffineMap>{map, map},
          iterators,
          [&](mlir::OpBuilder &builder, mlir::Location nested,
              mlir::ValueRange arguments) {
            mlir::Value value = element.isF32()
                                    ? builder
                                          .create<mlir::arith::ExtFOp>(
                                              nested, element, arguments[0])
                                          .getResult()
                                    : builder
                                          .create<mlir::arith::TruncFOp>(
                                              nested, element, arguments[0])
                                          .getResult();
            builder.create<mlir::linalg::YieldOp>(nested, value);
          })
      .getResult(0);
}

struct PromoteContractionAccumulationPass final
    : impl::PromoteContractionAccumulationPassBase<
          PromoteContractionAccumulationPass> {
  void runOnOperation() override {
    if (mlir::failed(promoteContractionAccumulation(getOperation())))
      signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult
promoteContractionAccumulation(mlir::func::FuncOp function) {
  if (mlir::failed(mlir::verify(function)))
    return mlir::failure();
  llvm::SmallVector<mlir::linalg::LinalgOp> contractions;
  function.walk([&](mlir::linalg::LinalgOp op) {
    if (isNarrowContraction(op))
      contractions.push_back(op);
  });
  mlir::IRRewriter rewriter(function.getContext());
  for (auto op : contractions) {
    rewriter.setInsertionPoint(op);
    auto loc = op.getLoc();
    auto &body = op->getRegion(0).front();
    auto multiply = mlir::cast<mlir::arith::MulFOp>(body.front());
    auto add = mlir::cast<mlir::arith::AddFOp>(*std::next(body.begin()));
    auto init = op.getDpsInits()[0];
    mlir::Value wideInit;
    if (auto fill = init.getDefiningOp<mlir::linalg::FillOp>()) {
      auto scalar = rewriter.createOrFold<mlir::arith::ExtFOp>(
          loc, rewriter.getF32Type(), fill.getInputs()[0]);
      auto empty = emptyLike(rewriter, loc, init, rewriter.getF32Type());
      wideInit = rewriter
                     .create<mlir::linalg::FillOp>(
                         loc, mlir::ValueRange{scalar}, mlir::ValueRange{empty})
                     .getResult(0);
    } else {
      wideInit = castTensor(rewriter, loc, init, rewriter.getF32Type());
    }
    auto wide = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{wideInit.getType()}, op.getDpsInputs(),
        mlir::ValueRange{wideInit}, op.getIndexingMapsArray(),
        op.getIteratorTypesArray(),
        [&](mlir::OpBuilder &builder, mlir::Location nested,
            mlir::ValueRange arguments) {
          auto lhs = builder.create<mlir::arith::ExtFOp>(
              nested, builder.getF32Type(), arguments[0]);
          auto rhs = builder.create<mlir::arith::ExtFOp>(
              nested, builder.getF32Type(), arguments[1]);
          auto product = builder.create<mlir::arith::MulFOp>(nested, lhs, rhs);
          product.setFastmathAttr(multiply.getFastmathAttr());
          auto sum = builder.create<mlir::arith::AddFOp>(nested, arguments[2],
                                                         product);
          sum.setFastmathAttr(add.getFastmathAttr());
          builder.create<mlir::linalg::YieldOp>(nested, sum.getResult());
        });
    auto outputType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto output = castTensor(rewriter, loc, wide.getResult(0),
                             outputType.getElementType());
    rewriter.replaceOp(op, output);
    // A replaced fill is no longer a structured root. Preserve it when an
    // independent user still observes the original narrow initialization.
    if (auto fill = init.getDefiningOp<mlir::linalg::FillOp>())
      if (fill->use_empty())
        rewriter.eraseOp(fill);
  }
  return mlir::verify(function);
}

} // namespace wafer
