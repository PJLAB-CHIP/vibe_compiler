//===- TensorResultIndexing.cpp - Static tensor support relations -------===//

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"

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
