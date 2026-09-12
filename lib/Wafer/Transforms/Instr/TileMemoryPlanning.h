//===- TileMemoryPlanning.h - Tile memory planning -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_INSTR_TILEMEMORYPLANNING_H
#define WAFER_TRANSFORMS_INSTR_TILEMEMORYPLANNING_H

#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "Wafer/Transforms/Instr/MemoryPlanning.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

enum class TileMemoryPlanningFailureKind : uint8_t {
  None,
  Contract,
  PreexistingPlacementFacts,
  Verification,
  SPMAllocation,
};

/// Typed, invocation-local rejection evidence for one Tile module.
/// It never carries a partially placed module.
struct TileMemoryPlanningFailure {
  TileMemoryPlanningFailureKind kind = TileMemoryPlanningFailureKind::None;
  bool spmCapacityOverflow = false;
  SPMMemoryPlanningFailureKind spmPlanningFailureKind =
      SPMMemoryPlanningFailureKind::None;
  mlir::LocationAttr spmLargestDemandLocation;
  mlir::Type spmLargestDemandType;
  uint64_t spmLargestDemandBytes = 0;
  uint64_t spmDemandCount = 0;
  struct SPMDemandEvidence {
    mlir::LocationAttr location;
    mlir::Type type;
    uint64_t bytes = 0;
    llvm::SmallVector<mlir::LocationAttr, 4> userLocations;
    /// Diagnostic-only; control flow never parses these names.
    llvm::SmallVector<mlir::OperationName, 4> userOperationNames;
    llvm::SmallVector<unsigned, 2> outputIndices;
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

/// Synchronous read-only observation of an exact capacity certificate while
/// its allocation/owner IR is still alive. Invoked independently by parallel
/// Tile leaves; callers synchronize their own choice work data. No IR handle
/// may escape the callback or participate in the allocator's decision.
using SPMCapacityObserver =
    llvm::function_ref<void(const SPMMemoryPlanningFailure &,
                            const StructuredMaterializationRelations &)>;

/// Converts one raw SPM planning failure into Tile-local planning evidence.
/// Actual allocation/user/type/location evidence comes directly from the SPM
/// planner. Observable output attribution is joined only through current
/// output endpoints; no source-node parity relation is reconstructed.
TileMemoryPlanningFailure convertSPMMemoryPlanningFailure(
    const SPMMemoryPlanningFailure &spmFailure,
    const StructuredMaterializationRelations &relations);

/// Consumes one Tile's owned canonical Instr module and runs the
/// complete current hard-gate sequence: validate the completion-closed,
/// function-boundary-bufferized canonical Instr input, assign SPM offsets and
/// verify the resulting Tile module. Selected execution structure, rotating
/// storage, Tile-to-Instr conversion, worker/order and completion must already
/// be present in the input IR. This leaf never runs those upstream mutations.
/// Whole-device candidate evaluation may suppress only the redundant per-Tile
/// diagnostic for a typed capacity rejection; all other failures still emit
/// their ordinary diagnostics and remain typed failures.
/// Performance-cost availability is not a memory-planning requirement.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> planTileMemory(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    TileMemoryPlanningFailure *failure = nullptr,
    StructuredMaterializationRelations *materializationRelations = nullptr,
    bool emitSPMCapacityDiagnostics = true,
    SPMCapacityObserver capacityObserver = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_INSTR_TILEMEMORYPLANNING_H
