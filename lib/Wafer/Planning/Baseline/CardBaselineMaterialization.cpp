//===- CardBaselineMaterialization.cpp --------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineAssignment.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {

mlir::FailureOr<CardBaselineModule> materializeCardBaseline(
    mlir::ModuleOp tensorProgram, CardId cardId,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    const CardBaselineAssignment &assignment,
    BaselineStatistics *statistics, llvm::raw_ostream &diagnostics) {
  std::string failureReason;
  mlir::FailureOr<std::unique_ptr<TileMaterializationSourceSession>> source =
      TileMaterializationSourceSession::create(
          tensorProgram, cardId, operationNodes, &failureReason);
  if (mlir::failed(source)) {
    diagnostics << "wafer-compile: baseline source analysis failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics)
    ++statistics->baselineSourcePreparations;

  mlir::FailureOr<std::unique_ptr<TileMaterializationSession>> materializer =
      TileMaterializationSession::create(**source, assignment.mapping,
                                         &failureReason);
  if (mlir::failed(materializer)) {
    diagnostics << "wafer-compile: baseline mapping validation failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics)
    ++statistics->baselineMaterializationPreparations;

  CardBaselineModule result;
  CardModuleMaterializationStatistics materializationStatistics;
  wafer::support::ScopedCompileTimingSpan timing(
      "conversion", "deterministic-baseline",
      "tensor-program-to-card-module");
  if (mlir::failed((*materializer)->lowerCardModule(
          result.module, &result.relations, &failureReason,
          statistics ? &materializationStatistics : nullptr))) {
    diagnostics << "wafer-compile: baseline CardModule materialization "
                   "failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics) {
    ++statistics->baselineCardModuleMaterializations;
    statistics->baselineTileEntryMaterializations +=
        materializationStatistics.tileEntryMaterializations;
    statistics->baselineMaximumTileMaterializationWorkers =
        materializationStatistics.maximumTileMaterializationWorkers;
  }
  return result;
}

mlir::LogicalResult verifyCardBaselineMaterialization(
    mlir::ModuleOp cardModule, const CardBaselineAssignment &assignment,
    const StructuredDAGAnalysis &dag,
    const StructuredMaterializationRelations &relations,
    llvm::ArrayRef<TileId> expectedTileIds, std::string &failureReason) {
  (void)dag;
  (void)relations;
  if (llvm::any_of(assignment.mapping.edgeStrategies,
                   [](const SpatialEdgeStrategy &strategy) {
                     return strategy.action ==
                            SpatialEdgeAction::CoupledFusion;
                   })) {
    failureReason = "baseline assignment contains coupled fusion";
    return mlir::failure();
  }

  llvm::SmallVector<TileModuleOp, 16> tileModules;
  cardModule.walk(
      [&](TileModuleOp tileModule) { tileModules.push_back(tileModule); });
  llvm::sort(tileModules, [](TileModuleOp lhs, TileModuleOp rhs) {
    return lhs.getTileIdAttr().getInt() < rhs.getTileIdAttr().getInt();
  });
  if (tileModules.size() != expectedTileIds.size()) {
    failureReason = "baseline CardModule has an incomplete Tile domain";
    return mlir::failure();
  }
  for (auto [index, tileModule] : llvm::enumerate(tileModules)) {
    if (TileId(tileModule.getTileIdAttr().getInt()) != expectedTileIds[index]) {
      failureReason = "baseline CardModule changed the Tile identity domain";
      return mlir::failure();
    }
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
