//===- CompleteCandidateMaterialization.h - Card construction -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_COMPLETECANDIDATEMATERIALIZATION_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_COMPLETECANDIDATEMATERIALIZATION_H

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/SelectedAttentionDecomposition.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

#include <cstdint>
#include <string>
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
  TemporalPlan temporal;
  PreparedAttentionDecomposition preparedAttention;
};

struct CardMaterializationPlan {
  SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
  llvm::SmallVector<StructuredDAGNodePlacement, 16> nodePlacements;
  TileMapping mapping;
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

mlir::FailureOr<CardMaterializationPlan>
buildCardMaterializationPlan(const CardProgramAnalysis &program,
                             const CompleteCandidatePlan &plan,
                             CandidateMaterializationStatistics *statistics,
                             llvm::raw_ostream &diagnostics);

mlir::LogicalResult verifyMaterializedCardCandidate(
    mlir::ModuleOp cardModule, const CardMaterializationPlan &assignment,
    const StructuredDAGAnalysis &dag,
    const StructuredMaterializationRelations &relations,
    llvm::ArrayRef<TileId> expectedTileIds, std::string &failureReason);

mlir::FailureOr<MaterializedCardCandidate>
materializeCardCandidate(mlir::ModuleOp tensorProgram, CardId cardId,
                         const CardProgramAnalysis &program,
                         const CompleteCandidatePlan &plan,
                         CandidateMaterializationStatistics *statistics,
                         llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_COMPLETECANDIDATEMATERIALIZATION_H
