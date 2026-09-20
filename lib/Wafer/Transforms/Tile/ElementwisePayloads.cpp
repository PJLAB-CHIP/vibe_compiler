//===- ElementwisePayloads.cpp - Materialize pointwise SSA ----------------===//

#include "ElementwisePayloads.h"
#include "StructuredToTile.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

namespace wafer::compiler::detail {
namespace {

struct PayloadValue {
  mlir::Value value;
  mlir::AffineMap map;
};

bool needsMaterialization(mlir::linalg::GenericOp op) {
  if (!op->getParentOfType<TileRegionOp>() || !op.hasPureTensorSemantics() ||
      op.getNumResults() != 1 || op.getNumDpsInits() != 1 ||
      op.getNumReductionLoops() || isScalarIntegerMap(op))
    return false;
  if (llvm::any_of(op.getStaticLoopRanges(),
                   [](int64_t size) {
                     return size <= 0 || mlir::ShapedType::isDynamic(size);
                   }) ||
      llvm::any_of(op.getIndexingMapsArray(),
                   [](mlir::AffineMap map) {
                     return map.getNumSymbols() ||
                            !map.isProjectedPermutation(
                                /*allowZeroInResults=*/true);
                   }) ||
      !op.getIndexingMapsArray().back().isPermutation())
    return false;
  unsigned steps = 0;
  llvm::SetVector<mlir::Value> captures;
  mlir::getUsedValuesDefinedAbove(op.getRegion(), captures);
  if (llvm::any_of(captures, [](mlir::Value value) {
        return mlir::isa<mlir::ShapedType>(value.getType());
      }))
    return false;
  for (mlir::Operation &scalar : op.getBody()->without_terminator()) {
    if (mlir::isOpTriviallyDead(&scalar))
      continue;
    if (!isSupportedElementwiseScalarOperation(&scalar))
      return false;
    steps += !mlir::isa<mlir::arith::ConstantOp>(scalar);
  }
  if (steps > 1 || (steps && op.getBody()->getOperations().size() > 2))
    return true;
  if (steps == 1 && mlir::isa<mlir::arith::ExtFOp, mlir::arith::TruncFOp>(
                        op.getBody()->front())) {
    auto inputMap = op.getIndexingMapsArray().front();
    if (inputMap != op.getIndexingMapsArray().back() ||
        !op.getDpsInits().front().getDefiningOp<mlir::tensor::EmptyOp>())
      return true;
  }
  // A single projected step can also hide a compact intermediate followed
  // by a broadcast. Expose that value to the layout query as well.
  llvm::SmallBitVector axes(op.getNumLoops());
  for (auto *input : op.getDpsInputOperands()) {
    if (op.getMatchingBlockArgument(input).use_empty())
      continue;
    for (auto expr : op.getMatchingIndexingMap(input).getResults())
      if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr))
        axes.set(dim.getPosition());
  }
  if (!op.getRegionOutputArgs().front().use_empty())
    return false;
  return steps != 0 && axes.count() < op.getNumLoops();
}

void materialize(mlir::linalg::GenericOp op, mlir::IRRewriter &rewriter,
                 StructuredBufferReplacementListener &listener) {
  rewriter.setInsertionPoint(op);
  auto context = op.getContext();
  unsigned rank = op.getNumLoops();
  auto scalarMap = mlir::AffineMap::get(rank, 0, {}, context);
  auto extents = op.getStaticLoopRanges();
  auto outputMap = op.getIndexingMapsArray().back();
  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op.getResult(0).getType());
  auto yield = mlir::cast<mlir::linalg::YieldOp>(op.getBody()->getTerminator());
  llvm::DenseMap<mlir::Value, PayloadValue> values;
  for (auto [argument, operand] :
       llvm::zip_equal(op.getBody()->getArguments(), op->getOpOperands()))
    values.try_emplace(
        argument,
        PayloadValue{operand.get(), op.getMatchingIndexingMap(&operand)});
  auto lookup = [&](mlir::Value value) {
    auto found = values.find(value);
    return found == values.end() ? PayloadValue{value, scalarMap}
                                 : found->second;
  };
  bool reusedDestination = false;
  llvm::SmallVector<mlir::Value> destinationValues{op.getDpsInits().front()};
  for (mlir::Operation &scalar : op.getBody()->without_terminator()) {
    if (mlir::isOpTriviallyDead(&scalar))
      continue;
    llvm::SmallVector<PayloadValue> inputs;
    llvm::SmallBitVector used(rank);
    bool scalarOnly = true;
    for (auto operand : scalar.getOperands()) {
      auto value = lookup(operand);
      inputs.push_back(value);
      scalarOnly &= !mlir::isa<mlir::ShapedType>(value.value.getType());
      for (auto expr : value.map.getResults())
        if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr))
          used.set(dim.getPosition());
    }
    // Keep the scalar constants/casts that lowering already handles on Kcore.
    // Other scalar expressions retain rank-zero CT storage and execution;
    // moving math to RISC-V merely because it is invariant is not free.
    if (scalarOnly && mlir::isa<mlir::arith::ConstantOp, mlir::arith::ExtFOp,
                                mlir::arith::TruncFOp>(scalar)) {
      mlir::IRMapping mapping;
      for (auto [operand, input] :
           llvm::zip_equal(scalar.getOperands(), inputs))
        mapping.map(operand, input.value);
      auto *clone = rewriter.clone(scalar, mapping);
      values[scalar.getResult(0)] = {clone->getResult(0), scalarMap};
      continue;
    }
    llvm::SmallVector<int64_t> shape;
    llvm::SmallVector<mlir::AffineExpr> coordinates,
        inverse(rank, rewriter.getAffineConstantExpr(0));
    for (int axis : used.set_bits()) {
      inverse[axis] = rewriter.getAffineDimExpr(shape.size());
      coordinates.push_back(rewriter.getAffineDimExpr(axis));
      shape.push_back(extents[axis]);
    }
    auto map = mlir::AffineMap::get(rank, 0, coordinates, context);
    bool cast = mlir::isa<mlir::arith::ExtFOp, mlir::arith::TruncFOp>(scalar);
    if (cast) {
      // Convert preserves the operand's logical coordinates. Any permutation
      // or broadcast is a separate current SSA use, with its own layout cost.
      auto inputType =
          mlir::cast<mlir::RankedTensorType>(inputs.front().value.getType());
      shape.assign(inputType.getShape().begin(), inputType.getShape().end());
      map = inputs.front().map;
    }
    auto type =
        mlir::RankedTensorType::get(shape, scalar.getResult(0).getType());
    llvm::SmallVector<mlir::Value> operands;
    llvm::SmallVector<mlir::AffineMap> maps;
    for (auto input : inputs) {
      if (!mlir::isa<mlir::ShapedType>(input.value.getType()))
        continue;
      operands.push_back(input.value);
      maps.push_back(
          cast ? rewriter.getMultiDimIdentityMap(shape.size())
               : input.map.replaceDimsAndSymbols(inverse, {}, shape.size(), 0));
    }
    maps.push_back(rewriter.getMultiDimIdentityMap(shape.size()));
    bool final =
        yield.getOperand(0) == scalar.getResult(0) && map == outputMap &&
        type == outputType &&
        (!cast ||
         op.getDpsInits().front().getDefiningOp<mlir::tensor::EmptyOp>());
    reusedDestination |= final;
    mlir::Value destination = final ? op.getDpsInits().front() : mlir::Value{};
    if (!destination)
      for (auto input : inputs)
        if (llvm::is_contained(destinationValues, input.value) &&
            input.value.getType() == type && input.map == map) {
          // Preserve the original generic's destination relation. One-Shot owns
          // the alias decision, including copies when an old value is live.
          destination = input.value;
          break;
        }
    if (!destination)
      destination = rewriter.create<mlir::tensor::EmptyOp>(
          scalar.getLoc(), shape, type.getElementType());
    auto step = rewriter.create<mlir::linalg::GenericOp>(
        scalar.getLoc(), mlir::TypeRange{type}, operands,
        mlir::ValueRange{destination}, maps,
        llvm::SmallVector<mlir::utils::IteratorType>(
            shape.size(), mlir::utils::IteratorType::parallel),
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange args) {
          mlir::IRMapping mapping;
          unsigned argument = 0;
          for (auto [operand, input] :
               llvm::zip_equal(scalar.getOperands(), inputs))
            mapping.map(operand,
                        mlir::isa<mlir::ShapedType>(input.value.getType())
                            ? args[argument++]
                            : input.value);
          auto *clone = builder.clone(scalar, mapping);
          builder.create<mlir::linalg::YieldOp>(loc, clone->getResults());
        });
    listener.recordLoweredOperation(op, step);
    if (llvm::is_contained(destinationValues, destination))
      destinationValues.push_back(step.getResult(0));
    values[scalar.getResult(0)] = {step.getResult(0), map};
  }
  auto result = lookup(yield.getOperand(0));
  if (!reusedDestination) {
    bool shaped = mlir::isa<mlir::ShapedType>(result.value.getType());
    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<mlir::AffineMap> maps;
    if (shaped) {
      inputs.push_back(result.value);
      maps.push_back(result.map);
    }
    maps.push_back(outputMap);
    auto broadcast = rewriter.create<mlir::linalg::GenericOp>(
        op.getLoc(), mlir::TypeRange{outputType}, inputs, op.getDpsInits(),
        maps, op.getIteratorTypesArray(),
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange args) {
          builder.create<mlir::linalg::YieldOp>(loc, shaped ? args[0]
                                                            : result.value);
        });
    listener.recordLoweredOperation(op, broadcast);
    result.value = broadcast.getResult(0);
  }
  rewriter.replaceOp(op, result.value);
}
} // namespace

bool hasUnmaterializedElementwisePayloads(mlir::ModuleOp module) {
  return module
      .walk([](mlir::linalg::GenericOp op) {
        return needsMaterialization(op) ? mlir::WalkResult::interrupt()
                                        : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

mlir::LogicalResult
materializeElementwisePayloads(mlir::ModuleOp module,
                               StructuredMaterializationRelations &relations) {
  llvm::SmallVector<mlir::linalg::GenericOp> operations;
  module.walk([&](mlir::linalg::GenericOp op) {
    if (needsMaterialization(op))
      operations.push_back(op);
  });
  StructuredBufferReplacementListener listener(relations);
  mlir::IRRewriter rewriter(module.getContext(), &listener);
  for (auto op : operations)
    materialize(op, rewriter, listener);
  return mlir::success(listener.finalizeAfterRewrite() &&
                       mlir::succeeded(mlir::verify(module)) &&
                       mlir::succeeded(checkStructuredBufferRelationsCurrent(
                           module, relations)));
}
} // namespace wafer::compiler::detail
