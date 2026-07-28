//===- SoftwarePipelining.h - Static fixed-slot candidates ------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_SOFTWAREPIPELINING_H
#define WAFER_TRANSFORMS_SOFTWAREPIPELINING_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace wafer {

/// One independently owned, fully materialized static software-pipeline
/// candidate. The stage and slot counts are derivation diagnostics only; the
/// module's ordinary allocation/SSA/control-flow IR is the semantic result.
struct StaticFixedSlotPipelineCandidate {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  unsigned stageCount = 0;
  unsigned slotAllocationCount = 0;
};

/// Derive a fixed-slot software-pipeline candidate for `sourceLoop`.
///
/// The source module is never modified and must be unplaced: no allocation may
/// already carry a Wafer SPM or DDR physical-offset fact. The first
/// implementation accepts only a statically positive-trip, non-nested,
/// single-block `scf.for` whose direct body has a dependency DAG provable from
/// Wafer instruction interfaces, SSA, and value-associated MemoryEffects.
/// Direct DTE, synchronous completion, unknown effects/aliases, dynamic
/// allocation and unsupported recurrence are rejected.
///
/// A single-iteration loop returns an independently owned identity clone with
/// `stageCount == 1` and no slot allocations. It is not a buffering
/// alternative; callers may discard that identity when enumerating a
/// candidate frontier.
///
/// On success, loop-local static allocations become real loop-external slot
/// allocations, slot permutation is represented by `scf.for` iter_args/yield,
/// and MLIR's SCF pipeliner mechanically emits prologue, steady kernel and
/// epilogue. Pure pointer permutation is independent of memory-use stages;
/// slot counts come from each allocation's stage live span and never from
/// target queue depth.
mlir::FailureOr<StaticFixedSlotPipelineCandidate>
deriveStaticFixedSlotPipelineCandidate(mlir::ModuleOp sourceModule,
                                       mlir::scf::ForOp sourceLoop,
                                       std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_TRANSFORMS_SOFTWAREPIPELINING_H
