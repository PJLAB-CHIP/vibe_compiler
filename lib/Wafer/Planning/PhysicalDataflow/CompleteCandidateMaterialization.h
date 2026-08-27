//===- CompleteCandidateMaterialization.h - Card construction -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_COMPLETECANDIDATEMATERIALIZATION_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_COMPLETECANDIDATEMATERIALIZATION_H

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/SelectedAttentionDecomposition.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/CoupledTileRegion.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

struct CandidateMaterializationStatistics {
  uint64_t spatialCoordinateQueries = 0;
  uint64_t exactDemandSatisfiedEdges = 0;
  uint64_t sourcePreparations = 0;
  uint64_t materializationPreparations = 0;
  uint64_t cardModuleMaterializations = 0;
  uint64_t tileEntryMaterializations = 0;
  uint64_t maximumTileMaterializationWorkers = 1;
};

struct CandidateNodeRootRelation {
  uint32_t structuredNodeId = 0;
  SemanticRootKey root;
};

/// Complete semantic carrier used by the policy-free Card materializer. It
/// contains no score, policy identity, candidate ordinal, failure history,
/// resource estimate, or accepted offset.
struct CompleteCandidatePlan {
  SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
  std::vector<analysis::RootRegionWork> rootWorks;
  RegionPlan regions;
  TemporalPlan temporal;
  RepresentationPlan representations;
  MovementPlan movement;
  std::vector<MovementResourceDescription> movementResources;
  BufferPlan buffers;
  std::vector<StorageResourceDescription> storageResources;
  ExecutionStructurePlan structure;
  PreparedAttentionDecomposition preparedAttention;
};

struct CardMaterializationPlan {
  SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
  llvm::SmallVector<StructuredDAGNodePlacement, 16> nodePlacements;
  TileMapping mapping;
  /// Present only when RegionPlan, rather than the canonical source mapping,
  /// directly constructed the actual outer TileRegion groups.
  std::optional<RegionPlan> selectedRegions;
  std::optional<RepresentationPlan> selectedRepresentations;
  /// Candidate-local execution instance to structured node relation used to
  /// verify actual groups and attribute replica buffers. It is never persisted
  /// or used as a semantic ordering key.
  std::vector<std::pair<RegionExecutionId, uint32_t>> selectedRegionExecutions;
  /// Exact query-local construction descriptors retained only until the
  /// selected actual CardModule has been verified. They are not serialized or
  /// consumed by later planning stages.
  std::vector<StructuredNodeShardGroup> selectedRegionGroups;
};

struct MaterializedCardCandidate {
  /// Keeps a selected TensorProgram alive when Card construction was driven
  /// by an actual attention decomposition. Empty for ordinary source
  /// materialization.
  mlir::OwningOpRef<mlir::ModuleOp> materializationSource;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
  llvm::SmallVector<CandidateNodeRootRelation, 64> nodeRoots;
  CardMaterializationPlan assignment;
};

/// Invocation-local structural input. It contains only already-selected
/// spatial/region/temporal semantics and no future physical value, buffer,
/// movement, event, execution-structure or schedule identity.
struct CurrentStructuralCandidatePlan {
  SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
  std::vector<analysis::RootRegionWork> rootWorks;
  RegionPlan regions;
  TemporalPlan temporal;
};

mlir::FailureOr<CardMaterializationPlan>
buildCardMaterializationPlan(const CardProgramAnalysis &program,
                             const CompleteCandidatePlan &plan,
                             SpatialDataflowMaterializationMode mode,
                             CandidateMaterializationStatistics *statistics,
                             llvm::raw_ostream &diagnostics);

/// Test-only plan/materialization correspondence oracle. Production
/// correctness is established by direct construction plus the local and
/// downstream stage verifiers.
mlir::LogicalResult verifyMaterializedCardCandidate(
    mlir::ModuleOp cardModule, const CardMaterializationPlan &assignment,
    const StructuredDAGAnalysis &dag,
    const StructuredMaterializationRelations &relations,
    llvm::ArrayRef<TileId> expectedTileIds, std::string &failureReason);

mlir::FailureOr<MaterializedCardCandidate>
materializeCardCandidate(mlir::ModuleOp tensorProgram, CardId cardId,
                         const CardProgramAnalysis &program,
                         const CompleteCandidatePlan &plan,
                         SpatialDataflowMaterializationMode mode,
                         CandidateMaterializationStatistics *statistics,
                         llvm::raw_ostream &diagnostics);

/// Materializes one complete search assignment through the selected
/// execution/RegionPlan builder. Canonical and non-canonical assignments use
/// this same path; it never invokes the edge-driven Tile materialization
/// session or chooses a different builder from plan equality.
mlir::FailureOr<MaterializedCardCandidate>
materializeSearchCardCandidate(
    mlir::ModuleOp tensorProgram, CardId cardId,
    const CardProgramAnalysis &program, const CompleteCandidatePlan &plan,
    CandidateMaterializationStatistics *statistics,
    llvm::raw_ostream &diagnostics);

/// Materializes one search structural choice directly into candidate-owned
/// Card/TileRegion IR. Layout, movement, execution structure, Instr order,
/// completion and memory are deliberately absent from the input and are read
/// or chosen only by their current-IR downstream owners.
mlir::FailureOr<MaterializedCardCandidate>
materializeSearchStructuralCandidate(
    mlir::ModuleOp tensorProgram, CardId cardId,
    const CardProgramAnalysis &program,
    const CurrentStructuralCandidatePlan &plan,
    CandidateMaterializationStatistics *statistics,
    llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_COMPLETECANDIDATEMATERIALIZATION_H
