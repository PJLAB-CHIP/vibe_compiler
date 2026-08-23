//===- BaselineAttentionMaterialization.h -------------------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_BASELINE_BASELINEATTENTIONMATERIALIZATION_H
#define WAFER_COMPILER_PLANNING_BASELINE_BASELINEATTENTIONMATERIALIZATION_H

#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"

namespace wafer::compiler::detail {

/// Candidate-owned TensorProgram used only by one actual transaction. The
/// source module is unchanged; every structured operation in
/// `operationNodes` belongs to `module` and is covered by `assignment`.
struct AttentionMaterializationSource {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  CardMaterializationPlan assignment;
  llvm::SmallVector<StructuredOperationNodeMapping, 64> operationNodes;
  llvm::SmallVector<CandidateNodeRootRelation, 64> nodeRoots;
};

mlir::FailureOr<AttentionMaterializationSource>
prepareAttentionMaterializationSource(
    mlir::ModuleOp source, CardId cardId,
    const CardProgramAnalysis &sourceProgram, const CompleteCandidatePlan &plan,
    llvm::ArrayRef<TileId> availableTiles,
    CandidateMaterializationStatistics *statistics = nullptr,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_BASELINE_BASELINEATTENTIONMATERIALIZATION_H
