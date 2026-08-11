//===- PhysicalTileFinalization.h - Exact Tile finalization ----*- C++ -*-===//

#ifndef WAFER_COMPILER_PHYSICALTILEFINALIZATION_H
#define WAFER_COMPILER_PHYSICALTILEFINALIZATION_H

#include "SelectedBufferMaterialization.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

enum class PhysicalTileFinalizationFailureKind : uint8_t {
  None,
  Contract,
  PrematureWholeCardFacts,
  Completion,
  Verification,
  FunctionBoundaryBufferization,
  SelectedBufferMaterialization,
  SPMAllocation,
};

/// Typed, invocation-local rejection evidence for one finalized Tile request.
/// It never carries a partially placed module or becomes an artifact.
struct PhysicalTileFinalizationFailure {
  PhysicalTileFinalizationFailureKind kind =
      PhysicalTileFinalizationFailureKind::None;
  bool spmCapacityOverflow = false;
  mlir::LocationAttr spmLargestDemandLocation;
  mlir::Type spmLargestDemandType;
  uint64_t spmLargestDemandBytes = 0;
  uint64_t spmDemandCount = 0;
  struct SPMDemandEvidence {
    mlir::LocationAttr location;
    mlir::Type type;
    uint64_t bytes = 0;
    llvm::SmallVector<mlir::LocationAttr, 4> userLocations;
  };
  llvm::SmallVector<SPMDemandEvidence, 4> spmLargestDemands;
  /// Exact over-capacity clique certificate propagated from the final SPM
  /// allocator.  Empty means that the allocator proved failure by another
  /// exact mechanism and only the largest-demand cohort is available.
  llvm::SmallVector<SPMDemandEvidence, 8> spmCapacityConflictDemands;
  /// Actual allocations that are each larger than usable SPM, propagated as
  /// independent exact rejection evidence.
  llvm::SmallVector<SPMDemandEvidence, 8> spmIndividuallyOversizedDemands;
};

/// Consumes one physical Tile's owned canonical Instr module and runs the
/// complete current hard-gate sequence: function-boundary bufferization,
/// completion rebuilt from current effects, SPM replanning and verification.
/// Tile-to-Instr conversion belongs to the caller and must already be complete.
/// Performance-cost availability is not a legality or finalization gate.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> finalizePhysicalTileModule(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    PhysicalTileFinalizationFailure *failure = nullptr,
    llvm::ArrayRef<SelectedBufferRequest> selectedBufferRequests = {},
    unsigned *materializedSlotAllocationCount = nullptr,
    SelectedBufferMaterializationFailure *selectedBufferFailure = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PHYSICALTILEFINALIZATION_H
