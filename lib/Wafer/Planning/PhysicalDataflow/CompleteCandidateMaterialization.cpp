//===- CompleteCandidateMaterialization.cpp - Selected Card IR --------===//

#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"

#include "Wafer/Planning/Baseline/BaselineAttentionMaterialization.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"
#include "Wafer/Planning/PhysicalDataflow/PhysicalVersionBuilder.h"
#include "Wafer/Planning/PhysicalDataflow/RepresentationDomain.h"
#include "Wafer/Planning/PhysicalDataflow/SelectedRegionMaterialization.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

mlir::FailureOr<llvm::SmallVector<CandidateNodeRootRelation, 64>>
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
  llvm::SmallVector<CandidateNodeRootRelation, 64> result;
  for (const auto &[node, root] : rootsByNode)
    result.push_back({node, root});
  return result;
}

mlir::FailureOr<llvm::SmallVector<StructuredNodeRootGroup, 64>>
buildMaterializationRootGroups(
    llvm::ArrayRef<CandidateNodeRootRelation> nodeRoots) {
  std::map<SemanticRootKey, uint32_t> groupsByRoot;
  for (const CandidateNodeRootRelation &relation : nodeRoots)
    groupsByRoot.try_emplace(relation.root, 0);
  uint32_t nextGroup = 0;
  for (auto &[root, group] : groupsByRoot) {
    (void)root;
    group = nextGroup++;
  }

  std::map<uint32_t, uint32_t> groupsByNode;
  for (const CandidateNodeRootRelation &relation : nodeRoots) {
    auto root = groupsByRoot.find(relation.root);
    if (root == groupsByRoot.end() ||
        !groupsByNode.try_emplace(relation.structuredNodeId, root->second)
             .second)
      return mlir::failure();
  }
  llvm::SmallVector<StructuredNodeRootGroup, 64> result;
  result.reserve(groupsByNode.size());
  for (const auto &[node, group] : groupsByNode)
    result.push_back({node, group});
  return result;
}

mlir::LogicalResult appendRequiredExecutionNodeRelations(
    const CompleteCandidatePlan &plan,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    CardMaterializationPlan &assignment) {
  std::map<mlir::Operation *, uint32_t> nodesByOperation;
  for (const StructuredOperationNodeMapping &mapping : operationNodes)
    if (!mapping.operation ||
        !nodesByOperation
             .try_emplace(mapping.operation, mapping.structuredNodeId)
             .second)
      return mlir::failure();
  std::map<analysis::RootRegionWorkId, mlir::Operation *> rootsByWork;
  for (const analysis::RootRegionWork &work : plan.rootWorks)
    if (!work.rootOperation ||
        !rootsByWork.try_emplace(work.id, work.rootOperation).second)
      return mlir::failure();
  for (const RegionGroupPlan &group : plan.regions.groups) {
    if (!group.replicas.empty())
      return mlir::failure();
    for (const ExecutionInstancePlan &execution : group.executions) {
      analysis::RootRegionWorkId work = std::visit(
          [](const auto &source) { return source.work; }, execution.id.source);
      auto root = rootsByWork.find(work);
      auto node = root == rootsByWork.end()
                      ? nodesByOperation.end()
                      : nodesByOperation.find(root->second);
      if (node == nodesByOperation.end())
        return mlir::failure();
      assignment.selectedRegionExecutions.push_back(
          {execution.id, node->second});
    }
  }
  llvm::sort(assignment.selectedRegionExecutions, [](const auto &lhs,
                                                     const auto &rhs) {
    return std::tie(lhs.first, lhs.second) < std::tie(rhs.first, rhs.second);
  });
  for (size_t index = 1; index < assignment.selectedRegionExecutions.size();
       ++index)
    if (assignment.selectedRegionExecutions[index - 1].first ==
        assignment.selectedRegionExecutions[index].first)
      return mlir::failure();
  return mlir::success();
}

} // namespace

mlir::FailureOr<MaterializedCardCandidate>
materializeCardCandidate(mlir::ModuleOp tensorProgram, CardId cardId,
                         const CardProgramAnalysis &program,
                         const CompleteCandidatePlan &plan,
                         SpatialDataflowMaterializationMode mode,
                         CandidateMaterializationStatistics *statistics,
                         llvm::raw_ostream &diagnostics) {
  std::string failureReason;
  mlir::ModuleOp materializationSource = tensorProgram;
  mlir::OwningOpRef<mlir::ModuleOp> selectedSource;
  CardMaterializationPlan assignment;
  llvm::SmallVector<StructuredOperationNodeMapping, 64> selectedNodes;
  llvm::SmallVector<CandidateNodeRootRelation, 64> nodeRoots;
  llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes =
      program.operationNodes;
  if (plan.preparedAttention.work.roots.empty()) {
    mlir::FailureOr<CardMaterializationPlan> built =
        buildCardMaterializationPlan(program, plan, mode, statistics,
                                     diagnostics);
    if (mlir::failed(built))
      return mlir::failure();
    assignment = std::move(*built);
  } else {
    mlir::FailureOr<AttentionMaterializationSource> selected =
        prepareAttentionMaterializationSource(tensorProgram, cardId, program,
                                              plan, program.availableTileIds,
                                              mode, statistics, &failureReason);
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
      diagnostics << "wafer-compile: candidate node/root relation failed\n";
      return mlir::failure();
    }
    nodeRoots = std::move(*sourceNodeRoots);
  }
  CanonicalRegionPlanOutcome canonicalRegions =
      buildCanonicalRegionPlan(plan.rootWorks);
  const RegionPlan *canonical = getRegionPlan(canonicalRegions);
  if (!canonical) {
    diagnostics << "wafer-compile: candidate canonical region validation "
                   "failed\n";
    return mlir::failure();
  }
  const bool usesSelectedRegions = !(plan.regions == *canonical);
  CanonicalRepresentationPlanOutcome canonicalRepresentation =
      buildCanonicalRepresentationPlan(plan.regions, plan.temporal,
                                       plan.rootWorks);
  const CanonicalRepresentationCoordinate *representationCoordinate =
      getCanonicalRepresentationCoordinate(canonicalRepresentation);
  if (!representationCoordinate) {
    diagnostics << "wafer-compile: candidate representation inventory "
                   "validation failed\n";
    return mlir::failure();
  }
  RepresentationDomainResult representationDomain =
      buildRepresentationDomain(*representationCoordinate);
  if (!representationDomain.succeeded() ||
      !representationDomain.domain->contains(plan.representations)) {
    diagnostics << "wafer-compile: candidate representation plan is outside "
                   "its current domain\n";
    return mlir::failure();
  }
  auto preparedRepresentation = prepareRepresentationPlan(
      *representationDomain.domain, plan.representations, &failureReason);
  if (mlir::failed(preparedRepresentation)) {
    diagnostics << "wafer-compile: candidate representation preflight failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  const bool usesSelectedRepresentations =
      !(plan.representations == representationCoordinate->plan);
  if (usesSelectedRegions || usesSelectedRepresentations) {
    if (!plan.preparedAttention.work.roots.empty()) {
      diagnostics << "wafer-compile: selected attention region construction "
                     "is not yet representable\n";
      return mlir::failure();
    }
    auto selectedSource = prepareSelectedRegionMaterializationSource(
        tensorProgram, program, plan.regions, plan.rootWorks, &failureReason);
    if (mlir::failed(selectedSource)) {
      diagnostics << "wafer-compile: selected region source preparation "
                     "failed: "
                  << failureReason << '\n';
      return mlir::failure();
    }
    mlir::FailureOr<std::vector<StructuredNodeShardGroup>> groups =
        prepareSelectedRegionGroups(
            program, plan.regions, plan.rootWorks, plan.temporal,
            selectedSource->executionNodes, &failureReason);
    if (mlir::failed(groups)) {
      diagnostics << "wafer-compile: selected region preparation failed: "
                  << failureReason << '\n';
      return mlir::failure();
    }
    if (wafer::support::getActiveCompileTimingSession() && !groups->empty()) {
      auto firstNonEmptyGroup =
          llvm::find_if(*groups, [](const StructuredNodeShardGroup &group) {
            return !group.shards.empty();
          });
      if (firstNonEmptyGroup != groups->end()) {
        TileId firstTile = firstNonEmptyGroup->shards.front().tile;
        uint64_t totalShards = 0;
        for (const StructuredNodeShardGroup &group : *groups)
          totalShards += group.shards.size();
        diagnostics << "wafer-compile: ir-region-plan groups=" << groups->size()
                    << " shards=" << totalShards
                    << " first_tile=" << firstTile.getValue() << '\n';
        for (const StructuredNodeShardGroup &group : *groups)
          for (const StructuredNodeIterationShard &shard : group.shards) {
            if (shard.tile != firstTile)
              continue;
            auto temporal =
                llvm::find_if(group.temporalTiles, [&](const auto &candidate) {
                  return candidate.structuredNodeId == shard.structuredNodeId;
                });
            uint64_t leafVariants = 1;
            if (temporal != group.temporalTiles.end())
              for (auto [extent, tile] :
                   llvm::zip_equal(shard.sizes, temporal->iteratorTileSizes)) {
                const int64_t boundedTile = std::min(extent, tile);
                if (boundedTile <= 0) {
                  leafVariants = 0;
                  break;
                }
                uint64_t variants = 1;
                if (extent - extent % boundedTile > boundedTile)
                  ++variants;
                if (extent % boundedTile != 0)
                  ++variants;
                if (leafVariants <=
                    std::numeric_limits<uint64_t>::max() / variants)
                  leafVariants *= variants;
              }
            diagnostics << "wafer-compile: ir-region-node tile="
                        << shard.tile.getValue()
                        << " node=" << shard.structuredNodeId
                        << " leaf_variants=" << leafVariants
                        << " shard_sizes=[";
            llvm::interleaveComma(shard.sizes, diagnostics);
            diagnostics << "] temporal_tiles=[";
            if (temporal != group.temporalTiles.end())
              llvm::interleaveComma(temporal->iteratorTileSizes, diagnostics);
            diagnostics << "]\n";
          }
      }
    }
    if (mlir::failed(applySelectedRegionRepresentations(
            plan.regions, plan.rootWorks, plan.representations,
            selectedSource->executionNodes, *groups, &failureReason))) {
      diagnostics << "wafer-compile: selected representation preparation "
                     "failed: "
                  << failureReason << '\n';
      return mlir::failure();
    }
    MaterializedCardCandidate result;
    result.assignment = std::move(assignment);
    result.assignment.selectedRegions = plan.regions;
    result.assignment.selectedRepresentations = plan.representations;
    result.assignment.selectedRegionGroups = *groups;
    std::map<uint32_t, SemanticRootKey> rootsByNode;
    for (const SelectedRegionExecutionNode &execution :
         selectedSource->executionNodes) {
      auto [position, inserted] =
          rootsByNode.try_emplace(execution.structuredNodeId, execution.root);
      if (!inserted && position->second != execution.root) {
        diagnostics << "wafer-compile: selected execution node has several "
                       "semantic roots\n";
        return mlir::failure();
      }
      result.assignment.selectedRegionExecutions.push_back(
          {execution.execution, execution.structuredNodeId});
    }
    for (const auto &[node, root] : rootsByNode)
      result.nodeRoots.push_back({node, root});
    result.materializationSource = std::move(selectedSource->module);
    if (mlir::failed(lowerStructuredNodeGroupsToCardModule(
            *result.materializationSource, cardId, program.availableTileIds,
            selectedSource->operationNodes, *groups, result.module,
            &result.relations, &failureReason))) {
      diagnostics << "wafer-compile: selected region CardModule "
                     "materialization failed: "
                  << failureReason << '\n';
      return mlir::failure();
    }
    if (statistics) {
      ++statistics->sourcePreparations;
      ++statistics->materializationPreparations;
      ++statistics->cardModuleMaterializations;
      statistics->tileEntryMaterializations += program.availableTileIds.size();
      statistics->maximumTileMaterializationWorkers = 1;
    }
    return result;
  }
  mlir::FailureOr<llvm::SmallVector<StructuredNodeRootGroup, 64>> rootGroups =
      buildMaterializationRootGroups(nodeRoots);
  if (mlir::failed(rootGroups)) {
    diagnostics << "wafer-compile: candidate node/root group failed\n";
    return mlir::failure();
  }
  mlir::FailureOr<std::unique_ptr<TileMaterializationSourceSession>> source =
      [&]() {
        wafer::support::ScopedCompileTimingSpan timing(
            "query", "complete-candidate",
            "prepare-tile-materialization-source");
        return TileMaterializationSourceSession::create(
            materializationSource, cardId, operationNodes, *rootGroups,
            &failureReason);
      }();
  if (mlir::failed(source)) {
    diagnostics << "wafer-compile: candidate source analysis failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics)
    ++statistics->sourcePreparations;

  mlir::FailureOr<std::unique_ptr<TileMaterializationSession>> materializer =
      [&]() {
        wafer::support::ScopedCompileTimingSpan timing(
            "query", "complete-candidate", "prepare-tile-materialization");
        return TileMaterializationSession::create(**source, assignment.mapping,
                                                  &failureReason);
      }();
  if (mlir::failed(materializer)) {
    diagnostics << "wafer-compile: candidate mapping validation failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics)
    ++statistics->materializationPreparations;

  MaterializedCardCandidate result;
  result.materializationSource = std::move(selectedSource);
  result.assignment = std::move(assignment);
  result.nodeRoots = std::move(nodeRoots);
  CardModuleMaterializationStatistics materializationStatistics;
  wafer::support::ScopedCompileTimingSpan timing(
      "conversion", "complete-candidate", "tensor-program-to-card-module");
  if (mlir::failed((*materializer)
                       ->lowerCardModule(result.module, &result.relations,
                                         &failureReason,
                                         statistics ? &materializationStatistics
                                                    : nullptr))) {
    diagnostics << "wafer-compile: candidate CardModule materialization "
                   "failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics) {
    ++statistics->cardModuleMaterializations;
    statistics->tileEntryMaterializations +=
        materializationStatistics.tileEntryMaterializations;
    statistics->maximumTileMaterializationWorkers =
        materializationStatistics.maximumTileMaterializationWorkers;
  }
  if (result.assignment.selectedRegionExecutions.empty() &&
      mlir::failed(appendRequiredExecutionNodeRelations(
          plan, program.operationNodes, result.assignment))) {
    diagnostics << "wafer-compile: candidate execution/node relation failed\n";
    return mlir::failure();
  }
  return result;
}

mlir::LogicalResult verifyMaterializedCardCandidate(
    mlir::ModuleOp cardModule, const CardMaterializationPlan &assignment,
    const StructuredDAGAnalysis &dag,
    const StructuredMaterializationRelations &relations,
    llvm::ArrayRef<TileId> expectedTileIds, std::string &failureReason) {
  (void)dag;
  if (llvm::any_of(assignment.mapping.edgeStrategies,
                   [](const SpatialEdgeStrategy &strategy) {
                     return strategy.action ==
                            SpatialEdgeAction::RecursiveProducerTiling;
                   })) {
    failureReason = "candidate assignment contains recursive producer tiling";
    return mlir::failure();
  }

  llvm::SmallVector<TileModuleOp, 16> tileModules;
  cardModule.walk(
      [&](TileModuleOp tileModule) { tileModules.push_back(tileModule); });
  llvm::sort(tileModules, [](TileModuleOp lhs, TileModuleOp rhs) {
    return lhs.getTileIdAttr().getInt() < rhs.getTileIdAttr().getInt();
  });
  if (tileModules.size() != expectedTileIds.size()) {
    failureReason = "candidate CardModule has an incomplete Tile domain";
    return mlir::failure();
  }
  for (auto [index, tileModule] : llvm::enumerate(tileModules)) {
    if (TileId(tileModule.getTileIdAttr().getInt()) != expectedTileIds[index]) {
      failureReason = "candidate CardModule changed the Tile identity domain";
      return mlir::failure();
    }
  }
  if (assignment.selectedRegions) {
    std::map<RegionExecutionId, uint32_t> nodesByExecution;
    for (const auto &[execution, node] : assignment.selectedRegionExecutions)
      if (!nodesByExecution.try_emplace(execution, node).second) {
        failureReason =
            "selected region verifier has a duplicate execution/node relation";
        return mlir::failure();
      }
    using RegionKey = std::pair<int64_t, std::vector<uint32_t>>;
    std::vector<RegionKey> expected;
    for (const RegionGroupPlan &group : assignment.selectedRegions->groups) {
      std::set<uint32_t> nodes;
      for (const ExecutionInstancePlan &execution : group.executions) {
        auto node = nodesByExecution.find(execution.id);
        if (node == nodesByExecution.end()) {
          failureReason =
              "selected region verifier lost one required execution node";
          return mlir::failure();
        }
        nodes.insert(node->second);
      }
      for (const ReplicaExecutionPlan &replica : group.replicas) {
        auto node = nodesByExecution.find(replica.id);
        if (node == nodesByExecution.end()) {
          failureReason =
              "selected region verifier lost one replica execution node";
          return mlir::failure();
        }
        nodes.insert(node->second);
      }
      if (!nodes.empty())
        expected.push_back({group.tile.getValue(),
                            std::vector<uint32_t>(nodes.begin(), nodes.end())});
    }
    llvm::sort(expected);

    std::map<mlir::Operation *, std::set<uint32_t>> actualNodes;
    for (const StructuredOperationEmissionRelation &relation :
         relations.operationEmissions) {
      TileRegionOp region =
          relation.operation
              ? relation.operation->getParentOfType<TileRegionOp>()
              : TileRegionOp{};
      if (region)
        actualNodes[region.getOperation()].insert(relation.structuredNodeId);
    }
    for (const StructuredOperationResultBufferRelation &relation :
         relations.operationResultBuffers) {
      mlir::Operation *owner = relation.buffer.getDefiningOp();
      if (!owner && mlir::isa<mlir::BlockArgument>(relation.buffer))
        owner = mlir::cast<mlir::BlockArgument>(relation.buffer)
                    .getOwner()
                    ->getParentOp();
      TileRegionOp region =
          owner ? owner->getParentOfType<TileRegionOp>() : TileRegionOp{};
      if (region)
        actualNodes[region.getOperation()].insert(relation.structuredNodeId);
    }
    std::vector<RegionKey> actual;
    std::map<RegionKey, TileRegionOp> actualRegions;
    for (const auto &[regionOperation, nodes] : actualNodes) {
      auto region = mlir::cast<TileRegionOp>(regionOperation);
      TileModuleOp tile = region->getParentOfType<TileModuleOp>();
      if (!tile || nodes.empty()) {
        failureReason = "selected region verifier found an unowned region";
        return mlir::failure();
      }
      RegionKey key{tile.getTileIdAttr().getInt(),
                    std::vector<uint32_t>(nodes.begin(), nodes.end())};
      if (!actualRegions.try_emplace(key, region).second) {
        failureReason =
            "selected region verifier found duplicate actual groups";
        return mlir::failure();
      }
      actual.push_back(std::move(key));
    }
    llvm::sort(actual);
    if (actual != expected) {
      failureReason = "actual TileRegion/node groups differ from RegionPlan";
      return mlir::failure();
    }

    auto hasExpectedLoopChain = [&](TileRegionOp region, uint32_t node,
                                    llvm::ArrayRef<int64_t> expectedSteps) {
      bool sawEmission = false;
      bool sawChain = expectedSteps.empty();
      for (const StructuredOperationEmissionRelation &relation :
           relations.operationEmissions) {
        mlir::Operation *operation = relation.operation;
        if (!operation || relation.structuredNodeId != node ||
            operation->getParentOfType<TileRegionOp>() != region)
          continue;
        sawEmission = true;
        llvm::SmallVector<int64_t, 8> enclosingSteps;
        for (mlir::Operation *parent = operation->getParentOp();
             parent && parent != region.getOperation();
             parent = parent->getParentOp())
          if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
            std::optional<int64_t> step =
                mlir::getConstantIntValue(loop.getStep());
            if (step)
              enclosingSteps.push_back(*step);
          }
        std::reverse(enclosingSteps.begin(), enclosingSteps.end());
        if (std::search(enclosingSteps.begin(), enclosingSteps.end(),
                        expectedSteps.begin(),
                        expectedSteps.end()) != enclosingSteps.end())
          sawChain = true;
      }
      for (const StructuredOperationResultBufferRelation &relation :
           relations.operationResultBuffers) {
        if (relation.structuredNodeId != node || !relation.buffer)
          continue;
        mlir::Operation *owner = relation.buffer.getDefiningOp();
        if (!owner && mlir::isa<mlir::BlockArgument>(relation.buffer))
          owner = mlir::cast<mlir::BlockArgument>(relation.buffer)
                      .getOwner()
                      ->getParentOp();
        if (owner && owner->getParentOfType<TileRegionOp>() == region)
          sawEmission = true;
      }
      if (!sawChain)
        region.walk([&](mlir::scf::ForOp loop) {
          llvm::SmallVector<int64_t, 8> enclosingSteps;
          for (mlir::Operation *parent = loop.getOperation();
               parent && parent != region.getOperation();
               parent = parent->getParentOp())
            if (auto enclosing = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
              std::optional<int64_t> step =
                  mlir::getConstantIntValue(enclosing.getStep());
              if (step)
                enclosingSteps.push_back(*step);
            }
          std::reverse(enclosingSteps.begin(), enclosingSteps.end());
          if (std::search(enclosingSteps.begin(), enclosingSteps.end(),
                          expectedSteps.begin(),
                          expectedSteps.end()) != enclosingSteps.end())
            sawChain = true;
        });
      return sawEmission && sawChain;
    };
    auto getSteadySteps = [](llvm::ArrayRef<int64_t> extents,
                             llvm::ArrayRef<int64_t> tileSizes,
                             llvm::ArrayRef<uint32_t> selectedOrder) {
      llvm::SmallVector<uint32_t, 4> order(selectedOrder.begin(),
                                           selectedOrder.end());
      if (order.empty())
        for (auto [dimension, extent, tile] :
             llvm::enumerate(extents, tileSizes))
          if (tile < extent)
            order.push_back(static_cast<uint32_t>(dimension));
      llvm::SmallVector<int64_t, 4> steps;
      for (uint32_t dimension : order) {
        if (dimension >= extents.size() || dimension >= tileSizes.size())
          continue;
        const int64_t extent = extents[dimension];
        const int64_t tile = tileSizes[dimension];
        if (tile > 0 && extent - extent % tile > tile)
          steps.push_back(tile);
      }
      return steps;
    };

    for (const StructuredNodeShardGroup &group :
         assignment.selectedRegionGroups) {
      std::vector<uint32_t> nodes;
      for (const StructuredNodeIterationShard &shard : group.shards)
        nodes.push_back(shard.structuredNodeId);
      llvm::sort(nodes);
      nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
      RegionKey key{group.shards.front().tile.getValue(), std::move(nodes)};
      auto actualRegion = actualRegions.find(key);
      if (actualRegion == actualRegions.end()) {
        failureReason =
            "selected temporal verifier lost its actual TileRegion group";
        return mlir::failure();
      }
      for (const StructuredNodeTemporalTile &temporal : group.temporalTiles) {
        auto shard = llvm::find_if(group.shards, [&](const auto &candidate) {
          return candidate.structuredNodeId == temporal.structuredNodeId;
        });
        if (shard == group.shards.end() ||
            !hasExpectedLoopChain(
                actualRegion->second, temporal.structuredNodeId,
                getSteadySteps(shard->sizes, temporal.iteratorTileSizes,
                               temporal.waveLoopOrder))) {
          std::string detail;
          llvm::raw_string_ostream diagnostic(detail);
          diagnostic << "actual top-level temporal loops differ from the "
                        "selected plan; tile="
                     << group.shards.front().tile.getValue()
                     << ", node=" << temporal.structuredNodeId
                     << ", expected_steps=[";
          if (shard != group.shards.end())
            llvm::interleaveComma(getSteadySteps(shard->sizes,
                                                 temporal.iteratorTileSizes,
                                                 temporal.waveLoopOrder),
                                  diagnostic);
          diagnostic << ']';
          failureReason = diagnostic.str();
          return mlir::failure();
        }
      }
      for (const StructuredNodeNestedTemporalTile &temporal :
           group.nestedTemporalTiles) {
        llvm::SmallVector<int64_t, 4> expectedSteps =
            getSteadySteps(temporal.producerIterationExtents,
                           temporal.iteratorTileSizes, temporal.waveLoopOrder);
        if (!hasExpectedLoopChain(actualRegion->second, temporal.producerNodeId,
                                  expectedSteps)) {
          std::string detail;
          llvm::raw_string_ostream diagnostic(detail);
          diagnostic << "actual nested temporal loops differ from the "
                        "selected class; tile="
                     << group.shards.front().tile.getValue()
                     << ", node=" << temporal.producerNodeId
                     << ", expected_steps=[";
          llvm::interleaveComma(expectedSteps, diagnostic);
          diagnostic << "], extents=[";
          llvm::interleaveComma(temporal.producerIterationExtents, diagnostic);
          diagnostic << "], tile_sizes=[";
          llvm::interleaveComma(temporal.iteratorTileSizes, diagnostic);
          diagnostic << "], order=[";
          llvm::interleaveComma(temporal.waveLoopOrder, diagnostic);
          diagnostic << "], actual_chains=[";
          bool firstChain = true;
          actualRegion->second.walk([&](mlir::scf::ForOp loop) {
            llvm::SmallVector<int64_t, 8> chain;
            for (mlir::Operation *parent = loop.getOperation();
                 parent && parent != actualRegion->second.getOperation();
                 parent = parent->getParentOp())
              if (auto enclosing = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
                std::optional<int64_t> step =
                    mlir::getConstantIntValue(enclosing.getStep());
                if (step)
                  chain.push_back(*step);
              }
            std::reverse(chain.begin(), chain.end());
            if (!firstChain)
              diagnostic << ';';
            firstChain = false;
            llvm::interleaveComma(chain, diagnostic);
          });
          diagnostic << ']';
          failureReason = diagnostic.str();
          return mlir::failure();
        }
      }
    }
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
