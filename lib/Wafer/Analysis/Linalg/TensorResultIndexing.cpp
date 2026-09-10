//===- TensorResultIndexing.cpp - Static tensor support relations -------===//

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/Twine.h"

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

mlir::FailureOr<mlir::AffineMap>
getStructuredOperandMap(mlir::OpOperand &operand) {
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operand.getOwner()))
    return linalg.getMatchingIndexingMap(&operand);
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
      relation = IndexRelation::staticSlice(
          resultType.getShape(), operandType.getShape(), operand.offsets,
          operand.strides, limits);
      break;
    case TensorIndexingTransformKind::InsertSlice:
      if (operand.role == TensorIndexingOperandRole::Destination) {
        relation = IndexRelation::identity(operandType.getShape(), limits);
      } else {
        if (!llvm::all_of(operand.strides,
                          [](int64_t stride) { return stride == 1; }))
          return fail(TensorResultIndexingStatus::Unsupported,
                      "insert_slice requires unit-stride exact semantics");
        relation = IndexRelation::staticInsertSlice(resultType.getShape(),
                                                    operandType.getShape(),
                                                    operand.offsets, limits);
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
                                 operand.strides,
                                 std::move(*relation.relation)});
  }
  return {TensorResultIndexingStatus::Exact, std::move(indexing), {}};
}

} // namespace wafer::analysis
