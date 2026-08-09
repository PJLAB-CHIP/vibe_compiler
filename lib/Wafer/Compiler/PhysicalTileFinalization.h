//===- ScheduledRankFinalization.h - Exact rank finalization ----*- C++ -*-===//

#ifndef WAFER_COMPILER_SCHEDULEDRANKFINALIZATION_H
#define WAFER_COMPILER_SCHEDULEDRANKFINALIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace wafer::compiler::detail {

enum class RankFinalizationFailureKind : uint8_t {
  None,
  Contract,
  WholeVariantFacts,
  Completion,
  Verification,
  FunctionBoundaryBufferization,
  SPMAllocation,
  ExactCost,
};

/// Typed, invocation-local rejection evidence for one finalized rank request.
/// It never carries a partially placed module or becomes an artifact.
struct RankFinalizationFailure {
  RankFinalizationFailureKind kind = RankFinalizationFailureKind::None;
};

/// Consumes one coordinated action's owned canonical Instr module and runs the
/// complete current hard-gate sequence: function-boundary bufferization, fresh
/// completion rebuilt from current effects, SPM replanning, verification, and
/// exact cost closure. Tile-to-Instr conversion belongs to canonical-parent
/// construction and must already be complete. Action identity and baseline
/// policy remain owned by the all-rank coordinator.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
finalizeCoordinatedRankModule(mlir::OwningOpRef<mlir::ModuleOp> module,
                              RankFinalizationFailure *failure = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_SCHEDULEDRANKFINALIZATION_H
