//===- TileMemoryPlanning.h - Tile memory planning -*- C++
//-*-===//

#ifndef WAFER_COMPILER_TILEMEMORYPLANNING_H
#define WAFER_COMPILER_TILEMEMORYPLANNING_H

#include "SelectedBufferMaterialization.h"
#include "StructuredBufferRelations.h"

#include "Wafer/Transforms/MemoryPlanning.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

enum class TileMemoryPlanningFailureKind : uint8_t {
  None,
  Contract,
  PreexistingPlacementFacts,
  Verification,
  InstrMemoryPlanningPreparation,
  SelectedBufferMaterialization,
  SPMAllocation,
};

/// Typed, invocation-local rejection evidence for one Tile module.
/// It never carries a partially placed module.
struct TileMemoryPlanningFailure {
  TileMemoryPlanningFailureKind kind =
      TileMemoryPlanningFailureKind::None;
  bool spmCapacityOverflow = false;
  SPMMemoryPlanningFailureKind spmPlanningFailureKind =
      SPMMemoryPlanningFailureKind::None;
  mlir::LocationAttr spmLargestDemandLocation;
  mlir::Type spmLargestDemandType;
  uint64_t spmLargestDemandBytes = 0;
  uint64_t spmDemandCount = 0;
  struct SPMDemandEvidence {
    mlir::LocationAttr location;
    mlir::Value allocation;
    mlir::Type type;
    uint64_t bytes = 0;
    llvm::SmallVector<mlir::LocationAttr, 4> userLocations;
    llvm::SmallVector<uint32_t, 2> operationResultNodes;
    llvm::SmallVector<uint32_t, 2> operandDemandNodes;
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

/// Converts one raw SPM planning failure into Tile-local planning evidence,
/// attributing every demand to structured operation nodes, operand-demand
/// nodes and output indices through the materialization relations of the IR
/// the failure was produced on. The relation buffers must be current values
/// of that IR; a probe may therefore call this with clone-remapped relations.
TileMemoryPlanningFailure convertSPMMemoryPlanningFailure(
    const SPMMemoryPlanningFailure &spmFailure,
    const StructuredMaterializationRelations &relations);

/// Consumes one Tile's owned canonical Instr module and runs the
/// complete current hard-gate sequence: prepare Instr IR for memory planning,
/// materialize requested rotating buffers, assign SPM offsets and verify the
/// resulting Tile module.
/// Tile-to-Instr conversion belongs to the caller and must already be complete.
/// Performance-cost availability is not a memory-planning requirement.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> planTileMemory(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    TileMemoryPlanningFailure *failure = nullptr,
    llvm::ArrayRef<SelectedBufferRequest> selectedBufferRequests = {},
    StructuredMaterializationRelations *materializationRelations = nullptr,
    unsigned *materializedSlotAllocationCount = nullptr,
    SelectedBufferMaterializationFailure *selectedBufferFailure = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_TILEMEMORYPLANNING_H
