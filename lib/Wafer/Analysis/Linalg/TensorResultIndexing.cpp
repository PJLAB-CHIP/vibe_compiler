//===- TensorResultIndexing.cpp - Static tensor support relations -------===//

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/SubsetOpInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>

namespace wafer::analysis {
namespace {

TensorResultIndexingResult fail(TensorResultIndexingStatus status,
                                llvm::Twine detail) {
  return {status, std::nullopt, detail.str()};
}

TensorResultIndexingResult
fromRelationFailure(const IndexRelationResult &result) {
  if (result.status == IndexRelationStatus::ResourceExhausted)
    return fail(TensorResultIndexingStatus::ResourceExhausted, result.reason);
  if (result.status == IndexRelationStatus::Invalid)
    return fail(TensorResultIndexingStatus::BrokenContract, result.reason);
  return fail(TensorResultIndexingStatus::Unsupported, result.reason);
}

} // namespace

namespace {

struct LinearLoopIndex {
  mlir::Value induction;
  int64_t base = 0;
  int64_t scale = 1;
};

std::optional<LinearLoopIndex>
parseLinearLoopIndex(mlir::Value value, const IndexRelationLimits &limits,
                     TensorResultIndexingStatus &status, unsigned depth = 0) {
  if (!value)
    return std::nullopt;
  if (depth >= limits.maxVariables) {
    status = TensorResultIndexingStatus::ResourceExhausted;
    return std::nullopt;
  }
  if (auto apply = value.getDefiningOp<mlir::affine::AffineApplyOp>()) {
    auto map = apply.getAffineMap();
    if (map.getNumDims() != 1 || map.getNumSymbols() != 0 ||
        apply.getMapOperands().size() != 1)
      return std::nullopt;
    auto dim = mlir::getAffineDimExpr(0, value.getContext());
    auto zero = mlir::getAffineConstantExpr(0, value.getContext());
    auto one = mlir::getAffineConstantExpr(1, value.getContext());
    auto atZero = mlir::dyn_cast<mlir::AffineConstantExpr>(
        mlir::simplifyAffineExpr(map.getResult(0).replace(dim, zero), 1, 0));
    auto atOne = mlir::dyn_cast<mlir::AffineConstantExpr>(
        mlir::simplifyAffineExpr(map.getResult(0).replace(dim, one), 1, 0));
    int64_t scale = 0;
    if (!atZero || !atOne ||
        llvm::SubOverflow(atOne.getValue(), atZero.getValue(), scale) ||
        scale <= 0)
      return std::nullopt;
    int64_t base = atZero.getValue();
    auto expected =
        dim * mlir::getAffineConstantExpr(scale, value.getContext()) +
        mlir::getAffineConstantExpr(base, value.getContext());
    if (mlir::simplifyAffineExpr(map.getResult(0) - expected, 1, 0) != zero)
      return std::nullopt;
    auto inner = parseLinearLoopIndex(apply.getMapOperands().front(), limits,
                                      status, depth + 1);
    int64_t scaledBase = 0, composedBase = 0, composedScale = 0;
    if (!inner || llvm::MulOverflow(scale, inner->base, scaledBase) ||
        llvm::AddOverflow(base, scaledBase, composedBase) ||
        llvm::MulOverflow(scale, inner->scale, composedScale))
      return std::nullopt;
    return LinearLoopIndex{inner->induction, composedBase, composedScale};
  }
  if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
    auto constant = mlir::getConstantIntValue(add.getLhs());
    mlir::Value other = add.getRhs();
    if (!constant) {
      constant = mlir::getConstantIntValue(add.getRhs());
      other = add.getLhs();
    }
    auto inner = constant
                     ? parseLinearLoopIndex(other, limits, status, depth + 1)
                     : std::nullopt;
    int64_t base = 0;
    if (!inner || llvm::AddOverflow(inner->base, *constant, base))
      return std::nullopt;
    return LinearLoopIndex{inner->induction, base, inner->scale};
  }
  if (auto multiply = value.getDefiningOp<mlir::arith::MulIOp>()) {
    auto constant = mlir::getConstantIntValue(multiply.getLhs());
    mlir::Value other = multiply.getRhs();
    if (!constant) {
      constant = mlir::getConstantIntValue(multiply.getRhs());
      other = multiply.getLhs();
    }
    auto inner = constant && *constant > 0
                     ? parseLinearLoopIndex(other, limits, status, depth + 1)
                     : std::nullopt;
    int64_t base = 0, scale = 0;
    if (!inner || llvm::MulOverflow(inner->base, *constant, base) ||
        llvm::MulOverflow(inner->scale, *constant, scale))
      return std::nullopt;
    return LinearLoopIndex{inner->induction, base, scale};
  }
  return LinearLoopIndex{value, 0, 1};
}

} // namespace

TensorLoopGridResult queryTensorLoopGrid(mlir::OpFoldResult offset,
                                         const IndexRelationLimits &limits) {
  auto value = mlir::dyn_cast<mlir::Value>(offset);
  auto status = TensorResultIndexingStatus::Unsupported;
  auto linear = parseLinearLoopIndex(value, limits, status);
  if (!linear)
    return {status, std::nullopt,
            status == TensorResultIndexingStatus::ResourceExhausted
                ? "loop index expression exceeded its work bound"
                : "offset is not one supported linear induction expression"};
  int64_t base = linear->base;
  int64_t scale = linear->scale;
  mlir::Value induction = linear->induction;
  auto argument = induction ? mlir::dyn_cast<mlir::BlockArgument>(induction)
                            : mlir::BlockArgument{};
  auto loop = argument && argument.getOwner()
                  ? mlir::dyn_cast_or_null<mlir::scf::ForOp>(
                        argument.getOwner()->getParentOp())
                  : mlir::scf::ForOp{};
  if (!loop || argument != loop.getInductionVar())
    return {TensorResultIndexingStatus::Unsupported, std::nullopt,
            "loop grid requires a positive bounded static induction range"};
  auto lower = mlir::getConstantIntValue(loop.getLowerBound());
  auto upper = mlir::getConstantIntValue(loop.getUpperBound());
  auto step = mlir::getConstantIntValue(loop.getStep());
  int64_t scaledLower = 0, scaledUpper = 0, coordinateStep = 0;
  int64_t coordinateLower = 0, coordinateUpper = 0;
  if (!lower || !upper || !step || *step <= 0 ||
      llvm::MulOverflow(*lower, scale, scaledLower) ||
      llvm::MulOverflow(*upper, scale, scaledUpper) ||
      llvm::MulOverflow(*step, scale, coordinateStep) ||
      llvm::AddOverflow(scaledLower, base, coordinateLower) ||
      llvm::AddOverflow(scaledUpper, base, coordinateUpper) ||
      coordinateLower < 0 || coordinateUpper <= coordinateLower)
    return {TensorResultIndexingStatus::Unsupported, std::nullopt,
            "loop grid requires a positive bounded static induction range"};
  return {TensorResultIndexingStatus::Exact,
          TensorLoopGrid{value, induction, base, scale, coordinateLower,
                         coordinateUpper, coordinateStep},
          {}};
}

std::optional<int64_t> getTensorLoopIndex(const TensorLoopGrid &grid,
                                          int64_t coordinate) {
  int64_t relative = 0;
  if (llvm::SubOverflow(coordinate, grid.base, relative) ||
      relative % grid.scale != 0)
    return std::nullopt;
  return relative / grid.scale;
}

std::optional<int64_t> getTensorLoopGridFloor(const TensorLoopGrid &grid,
                                              int64_t coordinate) {
  if (coordinate < grid.lower || coordinate >= grid.upper)
    return std::nullopt;
  int64_t delta = coordinate - grid.lower;
  int64_t multiple = 0;
  int64_t value = 0;
  if (llvm::MulOverflow(delta / grid.step, grid.step, multiple) ||
      llvm::AddOverflow(grid.lower, multiple, value) || value < grid.lower ||
      value >= grid.upper)
    return std::nullopt;
  return value;
}

std::optional<int64_t> getTensorLoopGridCeil(const TensorLoopGrid &grid,
                                             int64_t coordinate) {
  if (coordinate <= grid.lower)
    return grid.lower;
  if (coordinate >= grid.upper)
    return std::nullopt;
  int64_t delta = coordinate - grid.lower;
  int64_t rounded = 0;
  int64_t multiple = 0;
  int64_t value = 0;
  if (llvm::AddOverflow(delta, grid.step - 1, rounded) ||
      llvm::MulOverflow(rounded / grid.step, grid.step, multiple) ||
      llvm::AddOverflow(grid.lower, multiple, value) || value >= grid.upper)
    return std::nullopt;
  return value;
}

TensorViewIndexingResult
deriveTensorViewIndexing(mlir::Value value, const IndexRelationLimits &limits) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape())
    return {TensorResultIndexingStatus::Unsupported, std::nullopt,
            "view chain requires a static ranked tensor"};
  auto relation = IndexRelation::identity(type.getShape(), limits);
  mlir::Value source = value;
  unsigned steps = 0;
  while (auto result = mlir::dyn_cast<mlir::OpResult>(source)) {
    if (++steps > limits.maxRectangularPieces)
      return {TensorResultIndexingStatus::ResourceExhausted, std::nullopt,
              "view chain exceeded its relation work budget"};
    auto step = deriveTensorResultIndexing(result, limits);
    if (step.status == TensorResultIndexingStatus::Unsupported)
      break;
    if (!step.isExact())
      return {step.status, std::nullopt, step.detail};
    const auto *operand = step.indexing->getTransparentSource();
    if (!operand)
      break;
    if (relation.isExact())
      relation = relation.get()->compose(operand->resultToOperand, limits);
    if (!relation.isExact()) {
      auto failure = fromRelationFailure(relation);
      return {failure.status, std::nullopt, failure.detail};
    }
    source = result.getOwner()->getOperand(operand->operand);
  }
  if (source == value)
    return {TensorResultIndexingStatus::Unsupported, std::nullopt,
            "value has no transparent view chain"};
  return {TensorResultIndexingStatus::Exact,
          TensorViewIndexing{source, std::move(*relation.get())},
          {}};
}

StaticRectangularIndexSetResult
getTensorViewTileSource(const TensorViewIndexing &indexing,
                        llvm::ArrayRef<int64_t> resultShape,
                        const StaticRectangularIndexSet &requested,
                        const IndexRelationLimits &limits) {
  auto image = indexing.resultToSource.getExactStaticRectangularImage(
      requested.offsets, requested.sizes, limits);
  if (!image.isExact())
    return image;
  auto sourceType =
      mlir::cast<mlir::RankedTensorType>(indexing.source.getType());
  llvm::SmallVector<int64_t, 4> resultStrides(resultShape.size(), 1);
  llvm::SmallVector<int64_t, 4> sourceStrides(sourceType.getRank(), 1);
  auto tile = IndexRelation::staticSlice(
      requested.sizes, resultShape, requested.offsets, resultStrides, limits);
  auto source =
      IndexRelation::staticSlice(image.domain->sizes, sourceType.getShape(),
                                 image.domain->offsets, sourceStrides, limits);
  auto reshape = IndexRelation::staticReshape(requested.sizes,
                                              image.domain->sizes, limits);
  for (const auto *part : {&tile, &source, &reshape})
    if (!part->isExact())
      return {part->status, std::nullopt, part->reason};
  auto actual = tile.get()->compose(indexing.resultToSource, limits);
  auto expected = reshape.get()->compose(*source.get(), limits);
  for (const auto *part : {&actual, &expected})
    if (!part->isExact())
      return {part->status, std::nullopt, part->reason};
  auto order = actual.get()->isEquivalentTo(*expected.get(), limits);
  if (!order.isProvenTrue())
    return {order.status == IndexRelationStatus::Exact
                ? IndexRelationStatus::Unsupported
                : order.status,
            std::nullopt,
            "selected source tile does not preserve row-major order"};
  return image;
}

TensorViewTilePieceResult
getTensorViewTilePiece(const TensorViewIndexing &indexing,
                       llvm::ArrayRef<int64_t> resultShape,
                       const StaticRectangularIndexSet &requested,
                       const StaticRectangularIndexSet &sourceWindow,
                       const IndexRelationLimits &limits) {
  if (resultShape.size() != indexing.resultToSource.getDestinationRank() ||
      requested.offsets.size() != resultShape.size() ||
      sourceWindow.offsets.size() != indexing.resultToSource.getSourceRank())
    return {IndexRelationStatus::Invalid, std::nullopt,
            "view piece has inconsistent coordinate ranks"};
  auto request = IndexRelation::staticRectangularDomain(
      requested.offsets, requested.sizes, limits);
  auto source = IndexRelation::staticRectangularDomain(
      sourceWindow.offsets, sourceWindow.sizes, limits);
  for (const auto *domain : {&request, &source})
    if (!domain->isExact())
      return {domain->status, std::nullopt, domain->reason};
  // Preserve the relation's exact reshape construction when inverting a
  // rectangle. This removes quotient/remainder coordinates before set
  // intersection; the generic preimage remains available for other views.
  auto inverse = indexing.resultToSource.inverse(limits);
  if (!inverse.isExact())
    return {inverse.status, std::nullopt, inverse.reason};
  auto inverseImage = inverse.get()->getExactStaticRectangularImage(
      sourceWindow.offsets, sourceWindow.sizes, limits);
  IndexSetResult result;
  if (inverseImage.isExact())
    result = IndexRelation::staticRectangularDomain(
        inverseImage.domain->offsets, inverseImage.domain->sizes, limits);
  else if (inverseImage.status == IndexRelationStatus::Unsupported) {
    auto restricted = indexing.resultToSource.intersectDestinationDomain(
        *request.set, limits);
    if (!restricted.isExact())
      return {restricted.status, std::nullopt, restricted.reason};
    result = restricted.get()->preimage(*source.set, limits);
  } else
    return {inverseImage.status, std::nullopt, inverseImage.reason};
  if (!result.isExact())
    return {result.status, std::nullopt, result.reason};
  if (inverseImage.isExact()) {
    StaticRectangularIndexSet intersection;
    for (unsigned axis = 0; axis < resultShape.size(); ++axis) {
      int64_t lower =
          std::max(requested.offsets[axis], inverseImage.domain->offsets[axis]);
      int64_t upper = std::min(requested.offsets[axis] + requested.sizes[axis],
                               inverseImage.domain->offsets[axis] +
                                   inverseImage.domain->sizes[axis]);
      if (lower >= upper)
        return {IndexRelationStatus::Exact, std::nullopt, {}};
      intersection.offsets.push_back(lower);
      intersection.sizes.push_back(upper - lower);
    }
    result = IndexRelation::staticRectangularDomain(intersection.offsets,
                                                    intersection.sizes, limits);
    if (!result.isExact())
      return {result.status, std::nullopt, result.reason};
  }
  if (result.set->isIntegerEmpty())
    return {IndexRelationStatus::Exact, std::nullopt, {}};
  auto rectangle = result.getExactStaticRectangularDomain(limits);
  if (!rectangle.isExact())
    return {rectangle.status, std::nullopt,
            "view preimage: " + rectangle.reason};
  auto image =
      getTensorViewTileSource(indexing, resultShape, *rectangle.domain, limits);
  if (!image.isExact())
    return {image.status, std::nullopt, "view local image: " + image.reason};
  return {IndexRelationStatus::Exact,
          TensorViewTilePiece{std::move(*rectangle.domain),
                              std::move(*image.domain)},
          {}};
}

mlir::FailureOr<mlir::AffineMap>
getStructuredOperandMap(mlir::OpOperand &operand) {
  if (auto gather = mlir::dyn_cast<mlir::tensor::GatherOp>(operand.getOwner())) {
    if (operand.getOperandNumber() != 1 || gather.getGatherDims().size() != 1)
      return mlir::failure();
    unsigned batchRank = gather.getIndicesType().getRank() - 1;
    llvm::SmallVector<mlir::AffineExpr> coordinates;
    for (unsigned dimension = 0; dimension < batchRank; ++dimension)
      coordinates.push_back(mlir::getAffineDimExpr(dimension, gather.getContext()));
    coordinates.push_back(mlir::getAffineConstantExpr(0, gather.getContext()));
    return mlir::AffineMap::get(gather.getResultType().getRank(), 0,
                                coordinates, gather.getContext());
  }
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operand.getOwner()))
    return linalg.getMatchingIndexingMap(&operand);
  if (auto attention =
          mlir::dyn_cast<LinalgExtAttentionOp>(operand.getOwner())) {
    auto maps = attention.getIndexingMapsArray();
    if (operand.getOperandNumber() < maps.size())
      return maps[operand.getOperandNumber()];
  }
  if (auto online =
          mlir::dyn_cast<LinalgExtOnlineAttentionOp>(operand.getOwner())) {
    auto maps = online.getIndexingMapsArray();
    if (operand.getOperandNumber() < maps.size())
      return maps[operand.getOperandNumber()];
  }
  return mlir::failure();
}

mlir::FailureOr<mlir::AffineMap> getStructuredResultMap(mlir::OpResult result) {
  if (!result)
    return mlir::failure();
  if (auto gather = mlir::dyn_cast<mlir::tensor::GatherOp>(result.getOwner()))
    return mlir::AffineMap::getMultiDimIdentityMap(
        gather.getResultType().getRank(), gather.getContext());
  auto dps =
      mlir::dyn_cast<mlir::DestinationStyleOpInterface>(result.getOwner());
  if (!dps || result.getResultNumber() >= dps.getNumDpsInits())
    return mlir::failure();
  return getStructuredOperandMap(
      *dps.getDpsInitOperand(result.getResultNumber()));
}

IndexRelationResult
deriveIterationOperandRelation(mlir::OpOperand &operand,
                               llvm::ArrayRef<int64_t> iterationShape,
                               const IndexRelationLimits &limits) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(operand.get().getType());
  if (auto pack = mlir::dyn_cast<mlir::tensor::PackOp>(operand.getOwner());
      pack && operand.getOperandNumber() == 0 && type &&
      type.hasStaticShape()) {
    // Pack's TilingInterface iterates the outer packed coordinates. The exact
    // source demand is the inverse of source-point -> outer-tile, including
    // source bounds for a partially filled last inner tile.
    auto inner = pack.getStaticInnerTiles();
    if (llvm::any_of(inner, [](int64_t value) { return value <= 0; }))
      return {IndexRelationStatus::Unsupported, std::nullopt,
              "pack access requires static positive inner tiles"};
    llvm::SmallVector<int64_t, 4> factors(type.getRank(), 1);
    for (auto [axis, tile] : llvm::zip_equal(pack.getInnerDimsPos(), inner))
      factors[axis] = tile;
    llvm::SmallVector<mlir::AffineExpr, 4> outer;
    for (int64_t position = 0; position < type.getRank(); ++position) {
      int64_t axis = pack.getOuterDimsPerm().empty()
                         ? position
                         : pack.getOuterDimsPerm()[position];
      outer.push_back(mlir::getAffineDimExpr(axis, pack.getContext())
                          .floorDiv(factors[axis]));
    }
    auto points = IndexRelation::fromAffineMap(
        mlir::AffineMap::get(type.getRank(), 0, outer, pack.getContext()),
        type.getShape(), iterationShape, limits);
    return points.isExact() ? points.get()->inverse(limits) : points;
  }
  auto map = getStructuredOperandMap(operand);
  if (!type || !type.hasStaticShape() || mlir::failed(map))
    return {IndexRelationStatus::Unsupported, std::nullopt,
            "operand has no static structured indexing contract"};
  return IndexRelation::fromAffineMap(*map, iterationShape, type.getShape(),
                                      limits);
}

IndexRelationResult deriveIterationProducerRelation(
    mlir::OpOperand &operand, llvm::ArrayRef<int64_t> iterationShape,
    mlir::OpResult producer, const IndexRelationLimits &limits) {
  auto relation =
      deriveIterationOperandRelation(operand, iterationShape, limits);
  mlir::Value value = operand.get();
  while (relation.isExact() && value != producer) {
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return {IndexRelationStatus::Unsupported, std::nullopt,
              "operand path does not reach the current producer"};
    auto step = deriveTensorResultIndexing(result, limits);
    if (!step.isExact()) {
      auto status = step.status == TensorResultIndexingStatus::ResourceExhausted
                        ? IndexRelationStatus::ResourceExhausted
                    : step.status == TensorResultIndexingStatus::BrokenContract
                        ? IndexRelationStatus::Invalid
                        : IndexRelationStatus::Unsupported;
      return {status, std::nullopt, step.detail};
    }
    if (step.indexing->operands.size() != 1 ||
        step.indexing->operands.front().role !=
            TensorIndexingOperandRole::Source)
      return {IndexRelationStatus::Unsupported, std::nullopt,
              "operand path is not one transparent source relation"};
    const auto &source = step.indexing->operands.front();
    relation = relation.get()->compose(source.resultToOperand, limits);
    value = result.getOwner()->getOperand(source.operand);
  }
  return relation;
}

TensorResultIndexingResult
deriveTensorResultIndexing(mlir::OpResult result,
                           const IndexRelationLimits &limits) {
  mlir::Operation *operation = result ? result.getOwner() : nullptr;
  if (!operation || !mlir::isMemoryEffectFree(operation))
    return fail(TensorResultIndexingStatus::Unsupported,
                "tensor support result lacks a pure exact indexing contract");
  if (result.getResultNumber() > std::numeric_limits<uint32_t>::max())
    return fail(TensorResultIndexingStatus::BrokenContract,
                "tensor support result index is not representable");
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  if (!resultType || !resultType.hasStaticShape())
    return fail(TensorResultIndexingStatus::Unsupported,
                "tensor support result requires a static ranked tensor");

  auto interface =
      mlir::dyn_cast<wafer::WaferTensorIndexingOpInterface>(operation);
  if (!interface)
    return fail(TensorResultIndexingStatus::Unsupported,
                llvm::Twine("tensor support operation ") +
                    operation->getName().getStringRef() +
                    " has no exact indexing interface");
  mlir::FailureOr<wafer::TensorIndexingDescription> description =
      interface.getTensorIndexingDescription(result.getResultNumber());
  if (mlir::failed(description))
    return fail(TensorResultIndexingStatus::Unsupported,
                "tensor indexing interface has no static exact description");
  if (description->result != result.getResultNumber() ||
      description->operands.empty() ||
      !llvm::is_sorted(description->operands,
                       [](const auto &lhs, const auto &rhs) {
                         return lhs.operand < rhs.operand;
                       }))
    return fail(TensorResultIndexingStatus::BrokenContract,
                "tensor indexing interface returned a malformed description");

  const size_t sourceCount =
      llvm::count_if(description->operands, [](const auto &operand) {
        return operand.role == TensorIndexingOperandRole::Source;
      });
  const size_t destinationCount =
      llvm::count_if(description->operands, [](const auto &operand) {
        return operand.role == TensorIndexingOperandRole::Destination;
      });
  const bool isInsert =
      description->kind == TensorIndexingTransformKind::InsertSlice;
  if (sourceCount != 1 || destinationCount != (isInsert ? 1u : 0u) ||
      description->operands.size() != (isInsert ? 2u : 1u))
    return fail(TensorResultIndexingStatus::BrokenContract,
                "tensor indexing interface returned invalid operand roles");

  TensorResultIndexing indexing;
  indexing.result = result;
  indexing.kind = description->kind;
  llvm::SmallBitVector seenOperands(operation->getNumOperands());
  for (const TensorIndexingOperandDescription &operand :
       description->operands) {
    if (operand.operand >= operation->getNumOperands() ||
        seenOperands.test(operand.operand))
      return fail(TensorResultIndexingStatus::BrokenContract,
                  "tensor indexing interface returned invalid operands");
    seenOperands.set(operand.operand);
    auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(
        operation->getOperand(operand.operand).getType());
    if (!operandType || !operandType.hasStaticShape())
      return fail(TensorResultIndexingStatus::Unsupported,
                  "tensor indexing operand requires a static ranked tensor");

    IndexRelationResult relation;
    switch (description->kind) {
    case TensorIndexingTransformKind::ExpandShape:
    case TensorIndexingTransformKind::CollapseShape:
    case TensorIndexingTransformKind::Cast:
      relation = IndexRelation::staticReshape(resultType.getShape(),
                                              operandType.getShape(), limits);
      break;
    case TensorIndexingTransformKind::ExtractSlice:
      if (!mlir::computeRankReductionMask(operand.sizes, resultType.getShape()))
        return fail(TensorResultIndexingStatus::BrokenContract,
                    "slice result does not match its current subset sizes");
      relation =
          IndexRelation::staticSlice(operand.sizes, operandType.getShape(),
                                     operand.offsets, operand.strides, limits);
      if (relation.isExact() && resultType.getRank() != operandType.getRank()) {
        auto expansion = IndexRelation::staticReshape(resultType.getShape(),
                                                      operand.sizes, limits);
        if (!expansion.isExact())
          return fromRelationFailure(expansion);
        relation = expansion.relation->compose(*relation.relation, limits);
      }
      break;
    case TensorIndexingTransformKind::InsertSlice:
      if (operand.role == TensorIndexingOperandRole::Destination) {
        relation = IndexRelation::identity(operandType.getShape(), limits);
      } else {
        if (!llvm::all_of(operand.strides,
                          [](int64_t stride) { return stride == 1; }))
          return fail(TensorResultIndexingStatus::Unsupported,
                      "insert_slice requires unit-stride exact semantics");
        if (!mlir::computeRankReductionMask(operand.sizes,
                                            operandType.getShape()))
          return fail(TensorResultIndexingStatus::BrokenContract,
                      "insert source does not match its current subset sizes");
        relation = IndexRelation::staticInsertSlice(
            resultType.getShape(), operand.sizes, operand.offsets, limits);
        if (relation.isExact() &&
            resultType.getRank() != operandType.getRank()) {
          auto reduction = IndexRelation::staticReshape(
              operand.sizes, operandType.getShape(), limits);
          if (!reduction.isExact())
            return fromRelationFailure(reduction);
          relation = relation.relation->compose(*reduction.relation, limits);
        }
      }
      break;
    case TensorIndexingTransformKind::Pad:
      relation = IndexRelation::staticInsertSlice(resultType.getShape(),
                                                  operandType.getShape(),
                                                  operand.offsets, limits);
      break;
    }
    if (!relation.isExact())
      return fromRelationFailure(relation);
    indexing.operands.push_back({operand.operand, operand.role, operand.offsets,
                                 operand.sizes, operand.strides,
                                 std::move(*relation.relation)});
  }
  return {TensorResultIndexingStatus::Exact, std::move(indexing), {}};
}

StaticRectangularIndexSetPiecesResult
getTensorOperandDemand(const TensorResultIndexing &indexing,
                       const TensorOperandIndexing &operand,
                       llvm::ArrayRef<StaticRectangularIndexSet> demand,
                       const IndexRelationLimits &limits) {
  auto fail = [](IndexRelationStatus status, llvm::StringRef reason) {
    return StaticRectangularIndexSetPiecesResult{status, {}, reason.str()};
  };
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(indexing.result.getType());
  if (!type || !type.hasStaticShape())
    return fail(IndexRelationStatus::Invalid,
                "tensor demand needs a static result");
  if (demand.size() > limits.maxRectangularPieces)
    return fail(IndexRelationStatus::ResourceExhausted,
                "tensor demand exceeds rectangle work limit");
  const bool inserted =
      indexing.kind == TensorIndexingTransformKind::InsertSlice;
  const bool padded = indexing.kind == TensorIndexingTransformKind::Pad;
  const TensorOperandIndexing *source = nullptr;
  mlir::RankedTensorType sourceType;
  if (inserted || padded) {
    auto found = llvm::find_if(indexing.operands, [](const auto &candidate) {
      return candidate.role == TensorIndexingOperandRole::Source;
    });
    if (found == indexing.operands.end() ||
        found->operand >= indexing.result.getOwner()->getNumOperands())
      return fail(IndexRelationStatus::Invalid,
                  "tensor insertion has no source");
    source = &*found;
    sourceType = mlir::dyn_cast<mlir::RankedTensorType>(
        indexing.result.getOwner()->getOperand(source->operand).getType());
    if (!sourceType || !sourceType.hasStaticShape() ||
        source->sizes.size() != static_cast<size_t>(type.getRank()) ||
        !mlir::computeRankReductionMask(source->sizes, sourceType.getShape()) ||
        source->offsets.size() != static_cast<size_t>(type.getRank()))
      return fail(IndexRelationStatus::Invalid,
                  "tensor insertion has an inconsistent source window");
    for (auto [offset, size, extent] :
         llvm::zip_equal(source->offsets, source->sizes, type.getShape()))
      if (offset < 0 || offset > extent || size < 0 || size > extent - offset)
        return fail(IndexRelationStatus::Invalid,
                    "tensor insertion is out of bounds");
  }
  StaticRectangularIndexSetPiecesResult result{
      IndexRelationStatus::Exact, {}, {}};
  for (const auto &rectangle : demand) {
    if (rectangle.offsets.size() != static_cast<size_t>(type.getRank()) ||
        rectangle.sizes.size() != rectangle.offsets.size())
      return fail(IndexRelationStatus::Invalid,
                  "tensor demand rank is inconsistent");
    for (auto [offset, size, extent] :
         llvm::zip_equal(rectangle.offsets, rectangle.sizes, type.getShape()))
      if (offset < 0 || offset > extent || size < 0 || size > extent - offset)
        return fail(IndexRelationStatus::Invalid,
                    "tensor demand is out of bounds");
    if (llvm::is_contained(rectangle.sizes, 0))
      continue;
    auto read = rectangle;
    if (source) {
      bool overlaps = true;
      for (unsigned d = 0; d < rectangle.offsets.size(); ++d) {
        const int64_t begin =
            std::max(rectangle.offsets[d], source->offsets[d]);
        const int64_t end = std::min(rectangle.offsets[d] + rectangle.sizes[d],
                                     source->offsets[d] + source->sizes[d]);
        overlaps &= begin < end;
        read.offsets[d] = begin;
        read.sizes[d] = std::max<int64_t>(0, end - begin);
      }
      if (operand.role == TensorIndexingOperandRole::Destination) {
        if (!overlaps) {
          result.domains.push_back(rectangle);
        } else {
          // Peel disjoint slabs from a shrinking core. Only the intersection
          // remains at the end, and that is supplied by the inserted source.
          auto core = rectangle;
          for (unsigned d = 0; d < core.offsets.size(); ++d) {
            const int64_t end = core.offsets[d] + core.sizes[d];
            const int64_t readEnd = read.offsets[d] + read.sizes[d];
            if (core.offsets[d] < read.offsets[d]) {
              auto lower = core;
              lower.sizes[d] = read.offsets[d] - core.offsets[d];
              result.domains.push_back(std::move(lower));
              core.offsets[d] = read.offsets[d];
              core.sizes[d] = end - core.offsets[d];
            }
            if (readEnd < end) {
              auto upper = core;
              upper.offsets[d] = readEnd;
              upper.sizes[d] = end - readEnd;
              result.domains.push_back(std::move(upper));
              core.sizes[d] = readEnd - core.offsets[d];
            }
            if (result.domains.size() > limits.maxRectangularPieces)
              return fail(IndexRelationStatus::ResourceExhausted,
                          "tensor insertion exceeds rectangle work limit");
          }
        }
      } else if (overlaps) {
        auto image =
            operand.resultToOperand.getExactStaticRectangularImagePieces(
                read.offsets, read.sizes, limits);
        if (!image.isExact())
          return image;
        result.domains.append(std::move(image.domains));
      }
    } else {
      auto image = operand.resultToOperand.getExactStaticRectangularImagePieces(
          read.offsets, read.sizes, limits);
      if (!image.isExact())
        return image;
      result.domains.append(std::move(image.domains));
    }
    if (result.domains.size() > limits.maxRectangularPieces)
      return fail(IndexRelationStatus::ResourceExhausted,
                  "tensor demand image exceeds rectangle work limit");
  }
  return result;
}

TensorAssemblyResult queryTensorAssembly(mlir::Value value,
                                         const IndexRelationLimits &limits) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape())
    return {TensorAssemblyStatus::Unsupported, {},
            "insert assembly requires a static ranked tensor"};

  llvm::SmallVector<TensorAssemblySegment, 4> reverseSegments;
  mlir::Value current = value;
  while (auto insert =
             current.getDefiningOp<mlir::SubsetInsertionOpInterface>()) {
    if (reverseSegments.size() >= limits.maxRectangularPieces)
      return {TensorAssemblyStatus::ResourceExhausted, {},
              "insert assembly exceeded its segment work bound"};
    auto subset = mlir::cast<mlir::SubsetOpInterface>(insert.getOperation())
                      .getAccessedHyperrectangularSlice();
    if (mlir::failed(subset) || insert.getUpdatedDestination() != current ||
        insert.getUpdatedDestination().getType() != type ||
        llvm::any_of(subset->getMixedStrides(), [](auto stride) {
          return !mlir::isConstantIntValue(stride, 1);
        }))
      return {TensorAssemblyStatus::Unsupported, {},
              "insert assembly is strided or type-inconsistent"};

    llvm::SmallVector<int64_t, 4> offsets;
    llvm::SmallVector<int64_t, 4> sizes;
    for (mlir::OpFoldResult offset : subset->getMixedOffsets()) {
      std::optional<int64_t> constant = mlir::getConstantIntValue(offset);
      if (!constant)
        return {TensorAssemblyStatus::Unsupported, {},
                "insert assembly requires static offsets"};
      offsets.push_back(*constant);
    }
    for (mlir::OpFoldResult size : subset->getMixedSizes()) {
      std::optional<int64_t> constant = mlir::getConstantIntValue(size);
      if (!constant)
        return {TensorAssemblyStatus::Unsupported, {},
                "insert assembly requires static sizes"};
      sizes.push_back(*constant);
    }
    auto source = insert.getSourceOperand().get();
    auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
    if (!sourceType || !sourceType.hasStaticShape() ||
        offsets.size() != static_cast<size_t>(type.getRank()) ||
        sizes.size() != static_cast<size_t>(type.getRank()) ||
        !mlir::computeRankReductionMask(sizes, sourceType.getShape()))
      return {TensorAssemblyStatus::BrokenContract, {},
              "insert assembly source does not match its rectangle"};
    reverseSegments.push_back(
        {source, &insert.getSourceOperand(), std::move(offsets),
         std::move(sizes)});
    current = insert.getDestinationOperand().get();
  }
  if (reverseSegments.size() < 2)
    return {TensorAssemblyStatus::NotAssembly, {}, {}};
  int64_t fullVolume = 1;
  for (int64_t extent : type.getShape()) {
    int64_t next = 0;
    if (extent <= 0 || llvm::MulOverflow(fullVolume, extent, next))
      return {TensorAssemblyStatus::ResourceExhausted, {},
              "insert assembly volume is not representable"};
    fullVolume = next;
  }
  int64_t coveredVolume = 0;
  for (auto [index, segment] : llvm::enumerate(reverseSegments)) {
    int64_t volume = 1;
    for (auto [offset, size, extent] : llvm::zip_equal(
             segment.offsets, segment.sizes, type.getShape())) {
      int64_t end = 0;
      int64_t next = 0;
      if (offset < 0 || size <= 0 || llvm::AddOverflow(offset, size, end) ||
          end > extent || llvm::MulOverflow(volume, size, next))
        return {TensorAssemblyStatus::BrokenContract, {},
                "insert assembly rectangle is invalid"};
      volume = next;
    }
    int64_t nextCovered = 0;
    if (llvm::AddOverflow(coveredVolume, volume, nextCovered))
      return {TensorAssemblyStatus::ResourceExhausted, {},
              "insert assembly covered volume is not representable"};
    coveredVolume = nextCovered;
    for (size_t previous = 0; previous < index; ++previous) {
      bool overlaps = true;
      for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
           llvm::zip_equal(segment.offsets, segment.sizes,
                           reverseSegments[previous].offsets,
                           reverseSegments[previous].sizes))
        overlaps &= lhsOffset < rhsOffset + rhsSize &&
                    rhsOffset < lhsOffset + lhsSize;
      if (overlaps)
        return {TensorAssemblyStatus::Unsupported, {},
                "insert assembly rectangles overlap"};
    }
  }
  if (coveredVolume != fullVolume)
    return {TensorAssemblyStatus::Unsupported, {},
            "insert assembly does not exactly cover its result"};
  return {TensorAssemblyStatus::Exact, std::move(reverseSegments), {}};
}

TensorAssemblyDemandResult
queryTensorAssemblyDemand(mlir::Value value,
                          const StaticRectangularIndexSet &requested,
                          const IndexRelationLimits &limits) {
  auto fail = [](TensorAssemblyStatus status, llvm::StringRef detail) {
    return TensorAssemblyDemandResult{status, {}, detail.str()};
  };
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() ||
      requested.offsets.size() != static_cast<size_t>(type.getRank()) ||
      requested.sizes.size() != requested.offsets.size())
    return fail(TensorAssemblyStatus::BrokenContract,
                "assembly demand has an inconsistent static shape");
  for (auto [offset, size, extent] : llvm::zip_equal(
           requested.offsets, requested.sizes, type.getShape()))
    if (offset < 0 || offset > extent || size < 0 || size > extent - offset)
      return fail(TensorAssemblyStatus::BrokenContract,
                  "assembly demand is outside its result");
  if (!value.getDefiningOp<mlir::SubsetInsertionOpInterface>())
    return fail(TensorAssemblyStatus::NotAssembly, "value has no insert chain");

  auto fromRelation = [&](const StaticRectangularIndexSetPiecesResult &query) {
    return fail(query.status == IndexRelationStatus::ResourceExhausted
                    ? TensorAssemblyStatus::ResourceExhausted
                : query.status == IndexRelationStatus::Invalid
                    ? TensorAssemblyStatus::BrokenContract
                    : TensorAssemblyStatus::Unsupported,
                query.reason);
  };
  llvm::SmallVector<StaticRectangularIndexSet, 4> pending;
  if (!llvm::is_contained(requested.sizes, 0))
    pending.push_back(requested);
  TensorAssemblyDemandResult result;
  result.status = TensorAssemblyStatus::Exact;
  mlir::Value current = value;
  uint64_t steps = 0;
  while (auto insert =
             current.getDefiningOp<mlir::SubsetInsertionOpInterface>()) {
    if (pending.empty())
      break;
    if (++steps > limits.maxConstraintWork)
      return fail(TensorAssemblyStatus::ResourceExhausted,
                  "insert demand exceeded its SSA traversal budget");
    auto indexing = deriveTensorResultIndexing(
        mlir::cast<mlir::OpResult>(current), limits);
    if (!indexing.isExact())
      return fail(indexing.status == TensorResultIndexingStatus::ResourceExhausted
                      ? TensorAssemblyStatus::ResourceExhausted
                  : indexing.status == TensorResultIndexingStatus::BrokenContract
                      ? TensorAssemblyStatus::BrokenContract
                      : TensorAssemblyStatus::Unsupported,
                  indexing.detail);
    const TensorOperandIndexing *source = nullptr;
    const TensorOperandIndexing *destination = nullptr;
    for (const auto &operand : indexing.indexing->operands) {
      if (operand.role == TensorIndexingOperandRole::Source)
        source = &operand;
      else if (operand.role == TensorIndexingOperandRole::Destination)
        destination = &operand;
    }
    if (!source || !destination ||
        source->operand >= insert->getNumOperands() ||
        destination->operand >= insert->getNumOperands())
      return fail(TensorAssemblyStatus::BrokenContract,
                  "insert chain has no typed source and destination");
    llvm::SmallVector<StaticRectangularIndexSet, 4> remaining;
    for (const auto &window : pending) {
      auto sourceDemand =
          getTensorOperandDemand(*indexing.indexing, *source, {window}, limits);
      if (!sourceDemand.isExact())
        return fromRelation(sourceDemand);
      if (!sourceDemand.domains.empty()) {
        if (sourceDemand.domains.size() != 1)
          return fail(TensorAssemblyStatus::Unsupported,
                      "insert source needs multiple rectangular images");
        StaticRectangularIndexSet read;
        for (auto [offset, size, insertedOffset, insertedSize] :
             llvm::zip_equal(window.offsets, window.sizes, source->offsets,
                             source->sizes)) {
          int64_t begin = std::max(offset, insertedOffset);
          int64_t end = std::min(offset + size,
                                 insertedOffset + insertedSize);
          if (begin >= end)
            return fail(TensorAssemblyStatus::BrokenContract,
                        "insert source image has no result intersection");
          read.offsets.push_back(begin);
          read.sizes.push_back(end - begin);
        }
        mlir::OpOperand *sourceOperand = &insert->getOpOperand(source->operand);
        if (sourceOperand->get().getDefiningOp<mlir::tensor::EmptyOp>())
          return fail(TensorAssemblyStatus::Unsupported,
                      "insert source is undefined tensor.empty data");
        result.pieces.push_back({sourceOperand->get(), sourceOperand,
                                 std::move(read),
                                 std::move(sourceDemand.domains.front())});
      }
      auto oldDemand = getTensorOperandDemand(
          *indexing.indexing, *destination, {window}, limits);
      if (!oldDemand.isExact())
        return fromRelation(oldDemand);
      remaining.append(std::move(oldDemand.domains));
      if (remaining.size() + result.pieces.size() >
          limits.maxRectangularPieces)
        return fail(TensorAssemblyStatus::ResourceExhausted,
                    "insert demand exceeded its rectangle work bound");
    }
    pending = std::move(remaining);
    current = insert.getDestinationOperand().get();
  }
  if (!pending.empty() && current.getDefiningOp<mlir::tensor::EmptyOp>())
    return fail(TensorAssemblyStatus::Unsupported,
                "insert demand reads undefined tensor.empty data");
  for (auto &window : pending)
    result.pieces.push_back({current, nullptr, window, std::move(window)});
  return result;
}

} // namespace wafer::analysis
