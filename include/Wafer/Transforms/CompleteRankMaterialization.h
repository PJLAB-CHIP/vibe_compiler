//===- CompleteRankMaterialization.h - Conservative rank IR ---*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_COMPLETERANKMATERIALIZATION_H
#define WAFER_TRANSFORMS_COMPLETERANKMATERIALIZATION_H

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer {

/// Returns true when complete-rank materialization cannot observe the logical
/// rank from the current typed tensor program. Typed collective operations are
/// the only rank-dependent source operations at this boundary.
bool isCompleteRankTensorProgramRankInvariant(mlir::ModuleOp sourceModule);

/// Clone one tensor-program module and construct its unique conservative
/// complete-rank baseline for `logicalRank`. Dense tensor constants first gain
/// compiler-owned read-only storage. If the clone contains an eligible
/// structured root, every root is lowered into one outer Tile residency
/// region; otherwise the verified source-level clone is returned after
/// compiler-derived physical facts are cleared.
///
/// This utility performs no candidate enumeration, selection, instruction
/// lowering, buffering or worker placement, and carries no frontier metadata.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeConservativeCompleteRankBaseline(mlir::ModuleOp sourceModule,
                                            int64_t logicalRank);

/// One bounded physical residency action applied directly to an actual
/// complete-rank Tile clone. It is invocation-local search policy, not an IR
/// attribute, artifact kind, or commit input.
enum class CandidateTileResidencyAction : uint8_t {
  KeepSingleRegion,
  SplitAtExplicitDDRBoundary,
  SelectiveSpill,
};

/// Boundary movement realization consumed while constructing one actual Tile
/// clone. ExactDirectMapped is admitted only when the current physical
/// relation proves the transfer; the request itself never survives in IR.
enum class CandidateBoundaryMovementAction : uint8_t {
  Staged,
  ExactDirectMapped,
};

/// Loop-local movement realization for one actual Tile clone. The hoisted
/// form moves only a proven read-only boundary load and its invariant physical
/// layout chain; all other movement remains in the constructed traversal.
enum class CandidateLoopMovementAction : uint8_t {
  AsConstructed,
  HoistInvariantReadOnlyBoundary,
};

/// Prepares compiler-owned constant storage, then materializes one actual
/// complete-rank structured traversal directly as unplaced Tile IR. This is a
/// construction mechanism for the coordinated frontier; it performs no
/// rank-local selection, instruction lowering, or memory placement.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeCompleteRankCandidateTileProgram(
    mlir::ModuleOp sourceModule, int64_t logicalRank,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    CandidateTileTraversalKind traversalKind,
    CompleteRankTraversalComposition composition,
    CandidateTileResidencyAction residencyAction,
    CandidateBoundaryMovementAction boundaryMovementAction,
    CandidateLoopMovementAction loopMovementAction,
    std::string *failureReason = nullptr,
    std::optional<TargetImplementationKind> selectedImplementation =
        std::nullopt,
    std::optional<unsigned> physicalLayoutProposalOrdinal = std::nullopt);

/// Materializes one complete-rank Tile program from a full vector of
/// current-SSA connection actions. The vector has exactly the count returned
/// by getCompleteRankCandidateConnectionCount. No connection plan is retained
/// after the actual clone is built. Empty tile sizes select each traversal's
/// full static result shape independently.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeCompleteRankConnectionTileProgram(
    mlir::ModuleOp sourceModule, int64_t logicalRank,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<CandidateTraversalConnectionAction> connectionActions,
    CandidateBoundaryMovementAction boundaryMovementAction,
    CandidateLoopMovementAction loopMovementAction,
    std::string *failureReason = nullptr,
    std::optional<TargetImplementationKind> selectedImplementation =
        std::nullopt,
    std::optional<unsigned> physicalLayoutProposalOrdinal = std::nullopt);

/// As above, with explicit producer- and consumer-side tile vectors for each
/// connection. The choice vector is destroyed after the actual clone is built.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeCompleteRankConnectionChoicesTileProgram(
    mlir::ModuleOp sourceModule, int64_t logicalRank,
    llvm::ArrayRef<CandidateTraversalConnectionChoice> connectionChoices,
    CandidateBoundaryMovementAction boundaryMovementAction,
    CandidateLoopMovementAction loopMovementAction,
    std::string *failureReason = nullptr,
    std::optional<TargetImplementationKind> selectedImplementation =
        std::nullopt,
    std::optional<unsigned> physicalLayoutProposalOrdinal = std::nullopt);

/// Clones an already materialized, unplaced complete-rank Tile parent and
/// applies one bounded residency action directly to that clone. This is the
/// structured coordinator's executable-admission feedback mechanism: it never
/// replays source lowering and never consumes failed Instr/placement state.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeCompleteRankTileResidencySibling(
    mlir::ModuleOp tileParent, CandidateTileResidencyAction residencyAction,
    std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_TRANSFORMS_COMPLETERANKMATERIALIZATION_H
