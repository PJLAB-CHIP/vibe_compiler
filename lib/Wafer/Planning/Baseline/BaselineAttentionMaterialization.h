//===- BaselineAttentionMaterialization.h -------------------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_BASELINE_BASELINEATTENTIONMATERIALIZATION_H
#define WAFER_COMPILER_PLANNING_BASELINE_BASELINEATTENTIONMATERIALIZATION_H

#include "Wafer/Planning/Baseline/CanonicalBaselinePlan.h"
#include "Wafer/Planning/Baseline/CardBaselineAssignment.h"

namespace wafer::compiler::detail {

/// Winner-owned TensorProgram used only by the single actual baseline
/// transaction. The source module is unchanged; every structured operation in
/// `operationNodes` belongs to `module` and is covered by `assignment`.
struct BaselineAttentionMaterializationSource {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  CardBaselineAssignment assignment;
  llvm::SmallVector<StructuredOperationNodeMapping, 64> operationNodes;
  llvm::SmallVector<BaselineNodeRootRelation, 64> nodeRoots;
};

mlir::FailureOr<BaselineAttentionMaterializationSource>
prepareBaselineAttentionMaterializationSource(
    mlir::ModuleOp source, CardId cardId,
    const CardProgramAnalysis &sourceProgram,
    const CanonicalBaselinePlan &plan, llvm::ArrayRef<TileId> availableTiles,
    BaselineStatistics *statistics = nullptr,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_BASELINE_BASELINEATTENTIONMATERIALIZATION_H
