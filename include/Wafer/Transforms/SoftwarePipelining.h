//===- SoftwarePipelining.h - Static fixed-slot candidates ------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_SOFTWAREPIPELINING_H
#define WAFER_TRANSFORMS_SOFTWAREPIPELINING_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

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
/// Non-overlapping Direct-DTE issue windows with following exact token waits
/// are supported; every wait remains independent of NCC completion and extends
/// its endpoint buffers' lifetimes. An NCC-produced DTE buffer requires a
/// preceding typed participant join whose workers exactly own pending source
/// issues; the join remains an explicit synchronization stage. A leading join
/// may represent the exact loop-tail pending issue set only when every
/// recurrence conflict is a loop-local allocation that slot rotation can
/// separate; that transient backedge completion is rebuilt from the pipelined
/// kernel rather than copied as a steady-state observer. Overlapping endpoint
/// windows, extra join observers, synchronous NCC writeback, unknown
/// effects/aliases, dynamic allocation and unsupported recurrence are rejected.
///
/// A single-iteration loop returns an independently owned identity clone with
/// `stageCount == 1` and no slot allocations. It is not a buffering
/// alternative; callers may discard that clone when enumerating candidates.
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

/// Specialize rotating Direct-DTE buffers in a complete fixed-slot rank tuple.
///
/// Every selected loop has static positive bounds and a bounded, purely
/// periodic loop-carried allocation relation for each Direct-DTE endpoint.
/// All affected loops are modulo-unrolled by one common period so paired rank
/// sites retain an isomorphic static execution shape; a residual tail is fully
/// unrolled. Each surviving DTE site must then resolve through an exact
/// allocation recurrence to one planned SPM allocation. The modules are updated
/// atomically only on success.
mlir::LogicalResult
specializePeriodicDirectDTESites(llvm::ArrayRef<mlir::ModuleOp> rankModules,
                                 std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_TRANSFORMS_SOFTWAREPIPELINING_H
