//===- AttentionSpatialConstraints.h - Attention spatial constraints ----===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_ATTENTIONSPATIALCONSTRAINTS_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_ATTENTIONSPATIALCONSTRAINTS_H

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace wafer::compiler::detail {

enum class AttentionKeyValuePartitionRequirement : uint8_t {
  SingleInterval,
  MultipleIntervals,
};

/// Query-local constraints derived entirely from one verifier-valid attention
/// op. The value contains no operation handle, physical placement, or plan.
struct AttentionSpatialConstraints {
  llvm::SmallVector<unsigned, 2> queryKeyReductionIterators;
  llvm::SmallVector<unsigned, 2> keyValueReductionIterators;
  AttentionKeyValuePartitionRequirement keyValuePartition =
      AttentionKeyValuePartitionRequirement::SingleInterval;
};

enum class AttentionSpatialConstraintViolation : uint8_t {
  None,
  IteratorDomainMismatch,
  FlashAttentionPartitionsKeyValue,
  FlashDecodingLeavesKeyValueUnpartitioned,
};

mlir::FailureOr<AttentionSpatialConstraints>
deriveAttentionSpatialConstraints(LinalgExtAttentionOp attention);

AttentionSpatialConstraintViolation
checkAttentionSpatialConstraints(const AttentionSpatialConstraints &constraints,
                                 llvm::ArrayRef<int64_t> iteratorExtents,
                                 llvm::ArrayRef<IteratorPartition> partitions);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_ATTENTIONSPATIALCONSTRAINTS_H
