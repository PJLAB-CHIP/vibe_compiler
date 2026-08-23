//===- CardBaselineMaterialization.cpp --------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineAssignment.h"

#include "Wafer/Planning/Baseline/BaselineAttentionMaterialization.h"
#include "Wafer/Planning/Baseline/CanonicalBaselinePlan.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/STLExtras.h"

#include <map>

namespace wafer::compiler::detail {
namespace {

mlir::FailureOr<llvm::SmallVector<BaselineNodeRootRelation, 64>>
buildSourceNodeRoots(
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  std::map<mlir::Operation *, SemanticRootKey> rootsByOperation;
  for (const analysis::RootRegionWork &work : rootWorks) {
    if (!work.rootOperation)
      continue;
    auto [position, inserted] =
        rootsByOperation.try_emplace(work.rootOperation, work.id.root);
    if (!inserted && position->second != work.id.root)
      return mlir::failure();
  }
  std::map<uint32_t, SemanticRootKey> rootsByNode;
  for (const StructuredOperationNodeMapping &mapping : operationNodes) {
    auto root = rootsByOperation.find(mapping.operation);
    if (root == rootsByOperation.end())
      return mlir::failure();
    auto [position, inserted] =
        rootsByNode.try_emplace(mapping.structuredNodeId, root->second);
    if (!inserted && position->second != root->second)
      return mlir::failure();
  }
  llvm::SmallVector<BaselineNodeRootRelation, 64> result;
  for (const auto &[node, root] : rootsByNode)
    result.push_back({node, root});
  return result;
}

mlir::FailureOr<llvm::SmallVector<StructuredNodeRootGroup, 64>>
buildMaterializationRootGroups(
    llvm::ArrayRef<BaselineNodeRootRelation> nodeRoots) {
  std::map<SemanticRootKey, uint32_t> groupsByRoot;
  for (const BaselineNodeRootRelation &relation : nodeRoots)
    groupsByRoot.try_emplace(relation.root, 0);
  uint32_t nextGroup = 0;
  for (auto &[root, group] : groupsByRoot) {
    (void)root;
    group = nextGroup++;
  }

  std::map<uint32_t, uint32_t> groupsByNode;
  for (const BaselineNodeRootRelation &relation : nodeRoots) {
    auto root = groupsByRoot.find(relation.root);
    if (root == groupsByRoot.end() ||
        !groupsByNode
             .try_emplace(relation.structuredNodeId, root->second)
             .second)
      return mlir::failure();
  }
  llvm::SmallVector<StructuredNodeRootGroup, 64> result;
  result.reserve(groupsByNode.size());
  for (const auto &[node, group] : groupsByNode)
    result.push_back({node, group});
  return result;
}

} // namespace

mlir::FailureOr<CardBaselineModule> materializeCardBaseline(
    mlir::ModuleOp tensorProgram, CardId cardId,
    const CardProgramAnalysis &program, const CanonicalBaselinePlan &plan,
    BaselineStatistics *statistics, llvm::raw_ostream &diagnostics) {
  std::string failureReason;
  mlir::ModuleOp materializationSource = tensorProgram;
  mlir::OwningOpRef<mlir::ModuleOp> selectedSource;
  CardBaselineAssignment assignment;
  llvm::SmallVector<StructuredOperationNodeMapping, 64> selectedNodes;
  llvm::SmallVector<BaselineNodeRootRelation, 64> nodeRoots;
  llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes =
      program.operationNodes;
  if (plan.preparedAttention.work.roots.empty()) {
    mlir::FailureOr<CardBaselineAssignment> built =
        buildCardBaselineMaterializationAssignment(program, plan, statistics,
                                                   diagnostics);
    if (mlir::failed(built))
      return mlir::failure();
    assignment = std::move(*built);
  } else {
    mlir::FailureOr<BaselineAttentionMaterializationSource> selected =
        prepareBaselineAttentionMaterializationSource(
            tensorProgram, cardId, program, plan, program.availableTileIds,
            statistics, &failureReason);
    if (mlir::failed(selected)) {
      diagnostics << "wafer-compile: selected attention preparation failed: "
                  << failureReason << '\n';
      return mlir::failure();
    }
    selectedSource = std::move(selected->module);
    materializationSource = *selectedSource;
    assignment = std::move(selected->assignment);
    selectedNodes = std::move(selected->operationNodes);
    nodeRoots = std::move(selected->nodeRoots);
    operationNodes = selectedNodes;
  }
  if (nodeRoots.empty()) {
    auto sourceNodeRoots = buildSourceNodeRoots(operationNodes, plan.rootWorks);
    if (mlir::failed(sourceNodeRoots)) {
      diagnostics << "wafer-compile: baseline node/root relation failed\n";
      return mlir::failure();
    }
    nodeRoots = std::move(*sourceNodeRoots);
  }
  mlir::FailureOr<llvm::SmallVector<StructuredNodeRootGroup, 64>> rootGroups =
      buildMaterializationRootGroups(nodeRoots);
  if (mlir::failed(rootGroups)) {
    diagnostics << "wafer-compile: baseline node/root group failed\n";
    return mlir::failure();
  }
  mlir::FailureOr<std::unique_ptr<TileMaterializationSourceSession>> source =
      [&]() {
        wafer::support::ScopedCompileTimingSpan timing(
            "query", "deterministic-baseline",
            "prepare-tile-materialization-source");
        return TileMaterializationSourceSession::create(
            materializationSource, cardId, operationNodes, *rootGroups,
            &failureReason);
      }();
  if (mlir::failed(source)) {
    diagnostics << "wafer-compile: baseline source analysis failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics)
    ++statistics->baselineSourcePreparations;

  mlir::FailureOr<std::unique_ptr<TileMaterializationSession>> materializer =
      [&]() {
        wafer::support::ScopedCompileTimingSpan timing(
            "query", "deterministic-baseline",
            "prepare-tile-materialization");
        return TileMaterializationSession::create(**source, assignment.mapping,
                                                  &failureReason);
      }();
  if (mlir::failed(materializer)) {
    diagnostics << "wafer-compile: baseline mapping validation failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics)
    ++statistics->baselineMaterializationPreparations;

  CardBaselineModule result;
  result.materializationSource = std::move(selectedSource);
  result.assignment = std::move(assignment);
  result.nodeRoots = std::move(nodeRoots);
  CardModuleMaterializationStatistics materializationStatistics;
  wafer::support::ScopedCompileTimingSpan timing(
      "conversion", "deterministic-baseline", "tensor-program-to-card-module");
  if (mlir::failed((*materializer)
                       ->lowerCardModule(result.module, &result.relations,
                                         &failureReason,
                                         statistics ? &materializationStatistics
                                                    : nullptr))) {
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
                            SpatialEdgeAction::RecursiveProducerTiling;
                   })) {
    failureReason = "baseline assignment contains recursive producer tiling";
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
