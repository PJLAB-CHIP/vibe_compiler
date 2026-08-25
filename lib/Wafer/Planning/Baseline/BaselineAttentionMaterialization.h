//===- BaselineAttentionMaterialization.h -------------------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_BASELINE_BASELINEATTENTIONMATERIALIZATION_H
#define WAFER_COMPILER_PLANNING_BASELINE_BASELINEATTENTIONMATERIALIZATION_H

#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"

namespace wafer::compiler::detail {

/// One invocation-owned expansion of a fixed attention algorithm. The source
/// module is unchanged; every structured operation in `operationNodes`
/// belongs to `module`. This leaf consumes explicit physical choices but does
/// not select a baseline/search candidate or construct a CardModule.
struct AttentionMaterializationSource {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  CardMaterializationPlan assignment;
  llvm::SmallVector<StructuredOperationNodeMapping, 64> operationNodes;
  llvm::SmallVector<CandidateNodeRootRelation, 64> nodeRoots;
};

mlir::FailureOr<AttentionMaterializationSource>
expandSelectedAttentionAlgorithm(
    mlir::ModuleOp source, CardId cardId,
    const CardProgramAnalysis &sourceProgram,
    const SpatialAssignment &sourceSpatial,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    const TemporalPlan &temporal, const MovementPlan &movement,
    const PreparedAttentionDecomposition &preparedAttention,
    llvm::ArrayRef<TileId> availableTiles,
    SpatialDataflowMaterializationMode mode,
    CandidateMaterializationStatistics *statistics = nullptr,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_BASELINE_BASELINEATTENTIONMATERIALIZATION_H
