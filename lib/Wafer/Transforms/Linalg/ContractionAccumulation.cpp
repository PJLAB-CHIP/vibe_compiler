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
  if (!op || !op.hasPureTensorSemantics() || op.getNumDpsInputs() != 2 ||
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

mlir::Value emptyLike(mlir::OpBuilder &rewriter, mlir::Location loc,
                      mlir::Value source, mlir::Type element) {
  auto type = mlir::cast<mlir::RankedTensorType>(source.getType());
  llvm::SmallVector<mlir::Value> dynamic;
  for (int64_t dim = 0; dim < type.getRank(); ++dim)
    if (type.isDynamicDim(dim))
      dynamic.push_back(rewriter.create<mlir::tensor::DimOp>(loc, source, dim));
  return rewriter.create<mlir::tensor::EmptyOp>(loc, type.getShape(), element,
                                                dynamic, type.getEncoding());
}

mlir::Value castTensor(mlir::OpBuilder &rewriter, mlir::Location loc,
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

mlir::LogicalResult foldContractionInitializers(mlir::func::FuncOp function) {
  if (!function || mlir::failed(mlir::verify(function)))
    return mlir::failure();
  llvm::SmallVector<mlir::linalg::FillOp> fills;
  function.walk([&](mlir::linalg::FillOp fill) {
    if (!fill.hasPureTensorSemantics() || fill.getNumResults() != 1)
      return;
    auto type =
        mlir::dyn_cast<mlir::RankedTensorType>(fill.getResult(0).getType());
    auto scalar = fill.getInputs()[0].getDefiningOp<mlir::arith::ConstantOp>();
    if (!type || !type.hasStaticShape() || !scalar ||
        !mlir::isa<mlir::FloatAttr, mlir::IntegerAttr>(scalar.getValue()))
      return;
    if (llvm::any_of(fill.getResult(0).getUses(), [&](mlir::OpOperand &use) {
          auto contraction =
              mlir::dyn_cast<mlir::linalg::LinalgOp>(use.getOwner());
          return isNarrowContraction(contraction) &&
                 contraction.isDpsInit(&use);
        }))
      fills.push_back(fill);
  });
  mlir::IRRewriter rewriter(function.getContext());
  for (auto fill : fills) {
    rewriter.setInsertionPoint(fill);
    auto type = mlir::cast<mlir::RankedTensorType>(fill.getResult(0).getType());
    auto scalar = fill.getInputs()[0].getDefiningOp<mlir::arith::ConstantOp>();
    auto value = mlir::DenseElementsAttr::get(type, scalar.getValue());
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(fill, value);
  }
  return mlir::verify(function);
}

bool requiresWideContractionState(mlir::Operation *operation) {
  return isNarrowContraction(
      mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(operation));
}

mlir::Value convertContractionState(mlir::Value state, mlir::Type element,
                                    mlir::OpBuilder &builder) {
  return castTensor(builder, state.getLoc(), state, element);
}

mlir::FailureOr<mlir::linalg::GenericOp>
materializeContractionState(mlir::linalg::LinalgOp op,
                            mlir::OpBuilder &builder) {
  if (!isNarrowContraction(op))
    return mlir::failure();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);
  auto loc = op.getLoc();
  auto &body = op->getRegion(0).front();
  auto multiply = mlir::cast<mlir::arith::MulFOp>(body.front());
  auto add = mlir::cast<mlir::arith::AddFOp>(*std::next(body.begin()));
  auto init = op.getDpsInits()[0];
  mlir::Value wideInit;
  if (auto fill = init.getDefiningOp<mlir::linalg::FillOp>()) {
    auto scalar = builder.createOrFold<mlir::arith::ExtFOp>(
        loc, builder.getF32Type(), fill.getInputs()[0]);
    auto empty = emptyLike(builder, loc, init, builder.getF32Type());
    wideInit = builder
                   .create<mlir::linalg::FillOp>(loc, mlir::ValueRange{scalar},
                                                 mlir::ValueRange{empty})
                   .getResult(0);
  } else {
    wideInit = castTensor(builder, loc, init, builder.getF32Type());
  }
  auto wide = builder.create<mlir::linalg::GenericOp>(
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
        auto sum =
            builder.create<mlir::arith::AddFOp>(nested, arguments[2], product);
        sum.setFastmathAttr(add.getFastmathAttr());
        builder.create<mlir::linalg::YieldOp>(nested, sum.getResult());
      });
  return wide;
}

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
    auto init = op.getDpsInits()[0];
    auto wide = materializeContractionState(op, rewriter);
    if (mlir::failed(wide))
      return mlir::failure();
    rewriter.setInsertionPointAfter(*wide);
    auto outputType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto output = castTensor(rewriter, op.getLoc(), wide->getResult(0),
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
