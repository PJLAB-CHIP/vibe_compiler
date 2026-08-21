//===- AttentionSpatialConstraints.cpp - Attention spatial legality -----===//

#include "Wafer/Planning/PhysicalDataflow/AttentionSpatialConstraints.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"

namespace wafer::compiler::detail {

mlir::FailureOr<AttentionSpatialConstraints>
deriveAttentionSpatialConstraints(LinalgExtAttentionOp attention) {
  if (!attention)
    return mlir::failure();
  mlir::FailureOr<AttentionIterationRoles> roles =
      attention.getIterationRoles();
  if (mlir::failed(roles) || roles->queryKeyReduction.empty() ||
      roles->keyValueReduction.empty())
    return mlir::failure();

  AttentionSpatialConstraints constraints;
  constraints.queryKeyReductionIterators = roles->queryKeyReduction;
  constraints.keyValueReductionIterators = roles->keyValueReduction;
  switch (attention.getAlgorithm()) {
  case AttentionAlgorithm::FlashAttention:
    constraints.keyValuePartition =
        AttentionKeyValuePartitionRequirement::SingleInterval;
    break;
  case AttentionAlgorithm::FlashDecoding:
    constraints.keyValuePartition =
        AttentionKeyValuePartitionRequirement::MultipleIntervals;
    break;
  }
  return constraints;
}

AttentionSpatialConstraintViolation
checkAttentionSpatialConstraints(const AttentionSpatialConstraints &constraints,
                                 llvm::ArrayRef<int64_t> iteratorExtents,
                                 llvm::ArrayRef<IteratorPartition> partitions) {
  if (iteratorExtents.empty() || partitions.size() != iteratorExtents.size())
    return AttentionSpatialConstraintViolation::IteratorDomainMismatch;

  llvm::SmallBitVector queryKey(iteratorExtents.size(), false);
  llvm::SmallBitVector keyValue(iteratorExtents.size(), false);
  for (unsigned iterator : constraints.queryKeyReductionIterators) {
    if (iterator >= iteratorExtents.size() || queryKey.test(iterator) ||
        keyValue.test(iterator))
      return AttentionSpatialConstraintViolation::IteratorDomainMismatch;
    queryKey.set(iterator);
  }
  for (unsigned iterator : constraints.keyValueReductionIterators) {
    if (iterator >= iteratorExtents.size() || keyValue.test(iterator) ||
        queryKey.test(iterator))
      return AttentionSpatialConstraintViolation::IteratorDomainMismatch;
    keyValue.set(iterator);
  }
  if (constraints.queryKeyReductionIterators.empty() ||
      constraints.keyValueReductionIterators.empty())
    return AttentionSpatialConstraintViolation::IteratorDomainMismatch;

  bool partitionsKeyValue = false;
  for (auto [iterator, partition] : llvm::enumerate(partitions)) {
    if (partition.iterator != iterator || iteratorExtents[iterator] <= 0)
      return AttentionSpatialConstraintViolation::IteratorDomainMismatch;
    mlir::FailureOr<int64_t> intervalCount =
        getIteratorPartitionIntervalCount(iteratorExtents[iterator], partition);
    if (mlir::failed(intervalCount))
      return AttentionSpatialConstraintViolation::IteratorDomainMismatch;
    if (keyValue.test(iterator) && *intervalCount > 1)
      partitionsKeyValue = true;
  }

  switch (constraints.keyValuePartition) {
  case AttentionKeyValuePartitionRequirement::SingleInterval:
    return partitionsKeyValue ? AttentionSpatialConstraintViolation::
                                    FlashAttentionPartitionsKeyValue
                              : AttentionSpatialConstraintViolation::None;
  case AttentionKeyValuePartitionRequirement::MultipleIntervals:
    return partitionsKeyValue ? AttentionSpatialConstraintViolation::None
                              : AttentionSpatialConstraintViolation::
                                    FlashDecodingLeavesKeyValueUnpartitioned;
  }
  return AttentionSpatialConstraintViolation::IteratorDomainMismatch;
}

} // namespace wafer::compiler::detail
