//===- BooleanReduction.cpp - Exact boolean reduction representation -----===//

#include "BooleanReduction.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <utility>

namespace wafer::compiler::detail {
namespace {

enum class BooleanCombiner { All, Any };

std::optional<BooleanCombiner> matchReduction(mlir::linalg::GenericOp op) {
  if (!op->getParentOfType<TileRegionOp>() || !op.hasPureTensorSemantics() ||
      op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op.getNumResults() != 1 || op.getNumReductionLoops() == 0)
    return std::nullopt;
  for (mlir::Value operand : op->getOperands()) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isInteger(1))
      return std::nullopt;
  }
  auto &body = op.getRegion().front();
  if (body.getNumArguments() != 2 ||
      std::distance(body.begin(), body.end()) != 2)
    return std::nullopt;
  auto *combine = &body.front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!mlir::isa<mlir::arith::AndIOp, mlir::arith::OrIOp>(combine) || !yield ||
      yield.getNumOperands() != 1 ||
      yield.getOperand(0) != combine->getResult(0))
    return std::nullopt;
  auto lhs = combine->getOperand(0), rhs = combine->getOperand(1);
  if (!((lhs == body.getArgument(0) && rhs == body.getArgument(1)) ||
        (rhs == body.getArgument(0) && lhs == body.getArgument(1))))
    return std::nullopt;
  return mlir::isa<mlir::arith::AndIOp>(combine) ? BooleanCombiner::All
                                                 : BooleanCombiner::Any;
}

mlir::Value encodePredicate(mlir::Value predicate, mlir::IRRewriter &rewriter) {
  auto type = mlir::cast<mlir::RankedTensorType>(predicate.getType());
  auto encodedType =
      mlir::RankedTensorType::get(type.getShape(), rewriter.getF16Type());
  auto loc = predicate.getLoc();
  mlir::Value empty = rewriter.create<mlir::tensor::EmptyOp>(
      loc, encodedType.getShape(), encodedType.getElementType());
  // Preserve a scalar fill as a fill so the ordinary reduction path can
  // consume its exact initial value without loading an uninitialized buffer.
  if (auto fill = predicate.getDefiningOp<mlir::linalg::FillOp>()) {
    auto constant =
        fill.getDpsInputs().front().getDefiningOp<mlir::arith::ConstantOp>();
    auto value = constant
                     ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                     : mlir::IntegerAttr{};
    if (value) {
      auto number = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getF16FloatAttr(value.getValue().isZero() ? 0.0 : 1.0));
      return rewriter
          .create<mlir::linalg::FillOp>(loc, number.getResult(), empty)
          .getResult(0);
    }
  }
  auto identity = rewriter.getMultiDimIdentityMap(type.getRank());
  return rewriter
      .create<mlir::linalg::GenericOp>(
          loc, mlir::TypeRange{encodedType}, mlir::ValueRange{predicate},
          mlir::ValueRange{empty},
          llvm::ArrayRef<mlir::AffineMap>{identity, identity},
          llvm::SmallVector<mlir::utils::IteratorType>(
              type.getRank(), mlir::utils::IteratorType::parallel),
          [&](mlir::OpBuilder &builder, mlir::Location location,
              mlir::ValueRange args) {
            auto zero = builder.create<mlir::arith::ConstantOp>(
                location, builder.getF16FloatAttr(0.0));
            auto one = builder.create<mlir::arith::ConstantOp>(
                location, builder.getF16FloatAttr(1.0));
            auto selected = builder.create<mlir::arith::SelectOp>(
                location, args[0], one, zero);
            builder.create<mlir::linalg::YieldOp>(location,
                                                  selected.getResult());
          })
      .getResult(0);
}

} // namespace

mlir::LogicalResult
lowerBooleanReductions(mlir::ModuleOp module,
                       StructuredMaterializationRelations &relations) {
  llvm::SmallVector<std::pair<mlir::linalg::GenericOp, BooleanCombiner>>
      matched;
  module.walk([&](mlir::linalg::GenericOp op) {
    if (auto kind = matchReduction(op))
      matched.emplace_back(op, *kind);
  });
  if (matched.empty())
    return mlir::success();
  StructuredBufferReplacementListener listener(relations);
  mlir::IRRewriter rewriter(module.getContext(), &listener);
  for (auto [op, kind] : matched) {
    rewriter.setInsertionPoint(op);
    auto loc = op.getLoc();
    mlir::Value input = encodePredicate(op.getDpsInputs().front(), rewriter);
    mlir::Value init = encodePredicate(op.getDpsInits().front(), rewriter);
    auto reduction = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{init.getType()}, mlir::ValueRange{input},
        mlir::ValueRange{init}, op.getIndexingMapsArray(),
        op.getIteratorTypesArray(),
        [&](mlir::OpBuilder &builder, mlir::Location location,
            mlir::ValueRange args) {
          mlir::Value value = kind == BooleanCombiner::All
                                  ? builder
                                        .create<mlir::arith::MinimumFOp>(
                                            location, args[0], args[1])
                                        .getResult()
                                  : builder
                                        .create<mlir::arith::MaximumFOp>(
                                            location, args[0], args[1])
                                        .getResult();
          builder.create<mlir::linalg::YieldOp>(location, value);
        });
    auto type = mlir::cast<mlir::RankedTensorType>(op.getResult(0).getType());
    auto empty = rewriter.create<mlir::tensor::EmptyOp>(loc, type.getShape(),
                                                        type.getElementType());
    auto identity = rewriter.getMultiDimIdentityMap(type.getRank());
    auto decoded = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{type}, reduction->getResults(),
        mlir::ValueRange{empty},
        llvm::ArrayRef<mlir::AffineMap>{identity, identity},
        llvm::SmallVector<mlir::utils::IteratorType>(
            type.getRank(), mlir::utils::IteratorType::parallel),
        [&](mlir::OpBuilder &builder, mlir::Location location,
            mlir::ValueRange args) {
          auto zero = builder.create<mlir::arith::ConstantOp>(
              location, builder.getF16FloatAttr(0.0));
          auto predicate = builder.create<mlir::arith::CmpFOp>(
              location, mlir::arith::CmpFPredicate::OGT, args[0], zero);
          builder.create<mlir::linalg::YieldOp>(location,
                                                predicate.getResult());
        });
    listener.recordLoweredOperation(op, reduction);
    rewriter.replaceOp(op, decoded->getResults());
  }
  return mlir::success(listener.finalizeAfterRewrite() &&
                       mlir::succeeded(mlir::verify(module)) &&
                       mlir::succeeded(checkStructuredBufferRelationsCurrent(
                           module, relations)));
}

} // namespace wafer::compiler::detail
