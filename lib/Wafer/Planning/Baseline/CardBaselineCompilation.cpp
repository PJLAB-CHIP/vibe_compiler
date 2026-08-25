//===- CardBaselineCompilation.cpp -----------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineCompilation.h"

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"
#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"
#include "Wafer/Planning/Baseline/BaselineAttentionMaterialization.h"
#include "Wafer/Planning/Baseline/BaselineTemporalPlan.h"
#include "Wafer/Planning/Baseline/CanonicalBaselinePlan.h"
#include "Wafer/Planning/PhysicalDataflow/CompleteCandidateMaterialization.h"
#include "Wafer/Planning/PhysicalDataflow/SelectedRegionMaterialization.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"
#include "Wafer/Support/CompileTiming.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/CoupledTileRegion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

const analysis::RootRegionWork *findRootWork(
    llvm::ArrayRef<analysis::RootRegionWork> works,
    const analysis::RootRegionWorkId &id) {
  auto work = llvm::find_if(
      works, [&](const analysis::RootRegionWork &candidate) {
        return candidate.id == id;
      });
  return work == works.end() ? nullptr : &*work;
}

const TemporalScopePlan *findBaselineTemporalScope(
    const TemporalPlan &temporal, const ExecutionInstanceId &execution) {
  auto scope = llvm::find_if(
      temporal.scopes, [&](const TemporalScopePlan &candidate) {
        return candidate.id.execution == RegionExecutionId(execution) &&
               isTopLevelScope(candidate.id);
      });
  return scope == temporal.scopes.end() ? nullptr : &*scope;
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getBaselineAttentionIterationExtents(mlir::Operation *operation,
                                     std::string *failureReason) {
  auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(operation);
  if (!tiling) {
    if (failureReason)
      *failureReason = "baseline attention operation has no TilingInterface";
    return mlir::failure();
  }
  mlir::OpBuilder builder(operation);
  llvm::SmallVector<int64_t, 4> extents;
  for (const mlir::Range &range : tiling.getIterationDomain(builder)) {
    std::optional<int64_t> offset =
        mlir::getConstantIntValue(range.offset);
    std::optional<int64_t> size = mlir::getConstantIntValue(range.size);
    std::optional<int64_t> stride =
        mlir::getConstantIntValue(range.stride);
    if (!offset || *offset != 0 || !size || *size <= 0 || !stride ||
        *stride != 1) {
      if (failureReason)
        *failureReason =
            "baseline attention operation has a dynamic iteration domain";
      return mlir::failure();
    }
    extents.push_back(*size);
  }
  if (extents.size() != tiling.getLoopIteratorTypes().size()) {
    if (failureReason)
      *failureReason =
          "baseline attention operation iteration rank is inconsistent";
    return mlir::failure();
  }
  return extents;
}

bool isBaselineIdentityBroadcastGeneric(mlir::Operation *operation) {
  auto generic = mlir::dyn_cast_or_null<mlir::linalg::GenericOp>(operation);
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || generic->getNumResults() != 1 ||
      generic.getNumLoops() == 0 ||
      !llvm::all_of(generic.getIteratorTypesArray(),
                    mlir::linalg::isParallelIterator))
    return false;
  llvm::ArrayRef<mlir::AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 2 || !maps.front().isProjectedPermutation() ||
      !maps.back().isIdentity() || !generic.getRegion().hasOneBlock())
    return false;
  mlir::Block &body = generic.getRegion().front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  return yield && yield.getNumOperands() == 1 &&
         yield.getOperand(0) == body.getArgument(0);
}

std::optional<MemLayout> findBaselineAttentionComponentEncoding(
    const RepresentationPlan &representations,
    const ExecutionInstanceId &execution,
    wafer::CoupledReductionComponentKind component) {
  std::optional<MemLayout> encoding;
  for (const LogicalRepresentationPlan &logical :
       representations.logicalValues) {
    const auto *value =
        std::get_if<CoupledComponentValueId>(&logical.value);
    if (!value || !(value->execution == execution) ||
        value->component != component)
      continue;
    auto physical = llvm::find_if(
        representations.physicalVersions,
        [&](const PhysicalVersionPlan &candidate) {
          return candidate.id == logical.primary;
        });
    if (physical == representations.physicalVersions.end() ||
        (encoding && *encoding != physical->encoding))
      return std::nullopt;
    encoding = physical->encoding;
  }
  return encoding;
}

void remapBaselineAttentionRelations(
    StructuredMaterializationRelations &relations,
    const std::map<uint32_t, uint32_t> &semanticNodeByActualNode) {
  auto remap = [&](auto &entries) {
    for (auto &entry : entries) {
      auto semantic = semanticNodeByActualNode.find(entry.structuredNodeId);
      if (semantic != semanticNodeByActualNode.end())
        entry.structuredNodeId = semantic->second;
    }
  };
  remap(relations.operationEmissions);
  remap(relations.operationResultBuffers);
  remap(relations.operandBuffers);
  remap(relations.scratchBuffers);
  remap(relations.partialReductionContributions);
  remap(relations.partialReductionMergeInputs);
}

mlir::FailureOr<MaterializedCardCandidate>
materializeBaselineAttentionCandidate(
    mlir::ModuleOp tensorProgram, CardId cardId,
    const CardProgramAnalysis &program, const CanonicalBaselinePlan &plan,
    CandidateMaterializationStatistics *statistics,
    llvm::raw_ostream &diagnostics) {
  std::string failureReason;
  mlir::FailureOr<AttentionMaterializationSource> attention =
      expandSelectedAttentionAlgorithm(
          tensorProgram, cardId, program, plan.spatial, plan.rootWorks,
          plan.temporal, plan.movements.plan, plan.preparedAttention,
          program.availableTileIds,
          SpatialDataflowMaterializationMode::IndependentDDRStages, statistics,
          &failureReason);
  if (mlir::failed(attention)) {
    diagnostics << "wafer-compile: baseline attention source preparation "
                   "failed: "
                << failureReason << '\n';
    return mlir::failure();
  }

  mlir::func::FuncOp selectedProgram;
  for (mlir::func::FuncOp function :
       attention->module->getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (selectedProgram) {
      diagnostics << "wafer-compile: baseline attention source has several "
                     "program functions\n";
      return mlir::failure();
    }
    selectedProgram = function;
  }
  if (!selectedProgram) {
    diagnostics << "wafer-compile: baseline attention source has no program "
                   "function\n";
    return mlir::failure();
  }
  mlir::FailureOr<StructuredDAGAnalysis> dag =
      StructuredDAGAnalysis::create(selectedProgram, &failureReason);
  if (mlir::failed(dag)) {
    diagnostics << "wafer-compile: baseline attention DAG failed: "
                << failureReason << '\n';
    return mlir::failure();
  }

  std::map<mlir::Operation *, const StructuredOperationNodeMapping *>
      sourceMappings;
  std::map<uint32_t, SemanticRootKey> rootsBySemanticNode;
  for (const CandidateNodeRootRelation &relation : attention->nodeRoots)
    if (!rootsBySemanticNode
             .try_emplace(relation.structuredNodeId, relation.root)
             .second) {
      diagnostics << "wafer-compile: baseline attention has duplicate "
                     "semantic node roots\n";
      return mlir::failure();
    }
  for (const StructuredOperationNodeMapping &mapping :
       attention->operationNodes)
    if (!mapping.operation ||
        !sourceMappings.try_emplace(mapping.operation, &mapping).second) {
      diagnostics << "wafer-compile: baseline attention has malformed "
                     "operation mappings\n";
      return mlir::failure();
    }

  std::map<uint32_t, const StructuredDAGNode *> dagNodes;
  std::map<uint32_t, const StructuredDAGNodePlacement *> placements;
  std::map<uint32_t, llvm::SmallVector<uint32_t, 8>>
      actualNodesBySemanticNode;
  llvm::SmallVector<StructuredOperationNodeMapping, 64> actualOperationNodes;
  std::map<uint32_t, uint32_t> semanticNodeByActualNode;
  for (const StructuredDAGNode &node : dag->getNodes()) {
    auto source = sourceMappings.find(node.operation);
    if (source == sourceMappings.end() ||
        !dagNodes.try_emplace(node.id, &node).second) {
      diagnostics << "wafer-compile: baseline attention DAG lost an "
                     "operation mapping\n";
      return mlir::failure();
    }
    actualOperationNodes.push_back(
        {node.operation, node.id, source->second->coupledComponentIndices});
    semanticNodeByActualNode.emplace(node.id,
                                     source->second->structuredNodeId);
    actualNodesBySemanticNode[source->second->structuredNodeId].push_back(
        node.id);
  }
  for (const StructuredDAGNodePlacement &placement :
       attention->assignment.nodePlacements)
    if (!placements.try_emplace(placement.node, &placement).second) {
      diagnostics << "wafer-compile: baseline attention has duplicate actual "
                     "node placement\n";
      return mlir::failure();
    }

  std::map<mlir::Operation *, const StructuredOpTemporalTile *>
      temporalByOperation;
  for (const StructuredOpTemporalTile &temporal :
       attention->assignment.mapping.operationTemporalTiles)
    if (!temporal.operation ||
        !temporalByOperation.try_emplace(temporal.operation, &temporal)
             .second) {
      diagnostics << "wafer-compile: baseline attention has duplicate "
                     "temporal operation\n";
      return mlir::failure();
    }

  std::map<RegionExecutionId, uint32_t> semanticNodeByExecution;
  for (const auto &[execution, node] :
       attention->assignment.selectedRegionExecutions)
    if (!semanticNodeByExecution.try_emplace(execution, node).second) {
      diagnostics << "wafer-compile: baseline attention has duplicate "
                     "execution identity\n";
      return mlir::failure();
    }

  std::map<std::pair<SemanticRootKey, int64_t>, StructuredNodeShardGroup>
      groupsByRootAndTile;
  std::set<uint32_t> recomputableActualNodes;
  std::set<std::pair<uint32_t, int64_t>> coveredActualNodes;
  for (const RegionGroupPlan &region : plan.regions.groups) {
    if (!region.replicas.empty() || region.executions.empty() ||
        region.mandatoryRoots.size() != 1) {
      diagnostics << "wafer-compile: baseline attention RegionPlan does not "
                     "contain one semantic root\n";
      return mlir::failure();
    }
    StructuredNodeShardGroup &group = groupsByRootAndTile[
        {region.mandatoryRoots.front().root, region.tile.getValue()}];
    for (const ExecutionInstancePlan &selected : region.executions) {
      const ExecutionInstanceId &execution = selected.id;
      const analysis::RootRegionWorkId work = std::visit(
          [](const auto &source) { return source.work; }, execution.source);
      const SemanticRootKey root = work.root;
      if (work != region.mandatoryRoots.front()) {
        diagnostics << "wafer-compile: baseline attention region crosses "
                       "semantic roots\n";
        return mlir::failure();
      }
      auto semantic =
          semanticNodeByExecution.find(RegionExecutionId(execution));
      if (semantic == semanticNodeByExecution.end()) {
        llvm::SmallVector<uint32_t, 2> candidates;
        for (const auto &[node, candidateRoot] : rootsBySemanticNode)
          if (candidateRoot == root)
            candidates.push_back(node);
        if (candidates.size() != 1) {
          diagnostics << "wafer-compile: baseline attention execution has no "
                         "unique semantic node\n";
          return mlir::failure();
        }
        semantic = semanticNodeByExecution
                       .try_emplace(RegionExecutionId(execution),
                                    candidates.front())
                       .first;
      }
      auto actualNodes = actualNodesBySemanticNode.find(semantic->second);
      if (actualNodes == actualNodesBySemanticNode.end()) {
        diagnostics << "wafer-compile: baseline attention execution has no "
                       "actual structured nodes\n";
        return mlir::failure();
      }

      for (uint32_t actualNode : actualNodes->second) {
        auto placement = placements.find(actualNode);
        auto dagNode = dagNodes.find(actualNode);
        const StructuredDAGNode *node =
            dagNode == dagNodes.end() ? nullptr : dagNode->second;
        if (placement == placements.end() || !node ||
            !llvm::is_contained(placement->second->tiles, region.tile))
          continue;
        if (!coveredActualNodes
                 .emplace(actualNode, region.tile.getValue())
                 .second)
          continue;
        mlir::FailureOr<llvm::SmallVector<int64_t, 4>> extents =
            getBaselineAttentionIterationExtents(node->operation,
                                                 &failureReason);
        auto temporal = temporalByOperation.find(node->operation);
        if (mlir::failed(extents) || temporal == temporalByOperation.end() ||
            temporal->second->iteratorTileSizes.size() != extents->size()) {
          diagnostics << "wafer-compile: baseline attention node has no "
                         "complete temporal plan: "
                      << failureReason << '\n';
          return mlir::failure();
        }
        if (mlir::isa<mlir::linalg::FillOp>(node->operation) ||
            isBaselineIdentityBroadcastGeneric(node->operation)) {
          recomputableActualNodes.insert(actualNode);
          continue;
        }

        StructuredNodeIterationShard shard;
        shard.structuredNodeId = actualNode;
        shard.tile = region.tile;
        shard.offsets.assign(extents->size(), 0);
        shard.sizes = *extents;
        group.shards.push_back(std::move(shard));
        group.temporalTiles.push_back(
            {actualNode, temporal->second->iteratorTileSizes,
             temporal->second->waveLoopOrder});
        group.independentlyMaterializedNodes.push_back(actualNode);

        StructuredNodePhysicalRepresentation representation;
        representation.structuredNodeId = actualNode;
        representation.preserveNaturalOperands = true;
        for (mlir::Value operand : node->operation->getOperands()) {
          (void)operand;
          representation.operandLayouts.push_back(std::nullopt);
          representation.sharedOperands.push_back(0);
        }
        const StructuredOperationNodeMapping *sourceMapping =
            sourceMappings.at(node->operation);
        if (!sourceMapping->coupledComponentIndices.empty() &&
            sourceMapping->coupledComponentIndices.size() !=
                node->operation->getNumResults()) {
          diagnostics << "wafer-compile: baseline attention "
                         "component/result mapping is incomplete\n";
          return mlir::failure();
        }
        for (auto [resultNumber, value] :
             llvm::enumerate(node->operation->getResults())) {
          if (!mlir::isa<mlir::RankedTensorType>(value.getType())) {
            representation.resultLayouts.push_back(std::nullopt);
            continue;
          }
          MemLayout layout = MemLayout::Tensor;
          if (!sourceMapping->coupledComponentIndices.empty()) {
            const unsigned raw =
                sourceMapping->coupledComponentIndices[resultNumber];
            if (raw > static_cast<unsigned>(
                          wafer::CoupledReductionComponentKind::Accumulator)) {
              diagnostics << "wafer-compile: baseline attention component "
                             "kind is outside the IR contract\n";
              return mlir::failure();
            }
            std::optional<MemLayout> componentEncoding =
                findBaselineAttentionComponentEncoding(
                    plan.representations.plan, execution,
                    static_cast<wafer::CoupledReductionComponentKind>(raw));
            if (!componentEncoding) {
              diagnostics << "wafer-compile: baseline attention component "
                             "has no canonical physical version\n";
              return mlir::failure();
            }
            layout = *componentEncoding;
          }
          representation.resultLayouts.push_back(layout);
        }
        group.representations.push_back(std::move(representation));
      }
    }
  }

  std::vector<StructuredNodeShardGroup> groups;
  groups.reserve(groupsByRootAndTile.size());
  for (auto &[rootAndTile, group] : groupsByRootAndTile) {
    const TileId tile(rootAndTile.second);
    std::set<uint32_t> selectedNodes;
    for (const StructuredNodeIterationShard &shard : group.shards)
      selectedNodes.insert(shard.structuredNodeId);
    std::set<uint32_t> requiredRecomputations;
    std::set<uint32_t> visitedDependencies;
    std::function<void(uint32_t)> collectRecomputableDependencies =
        [&](uint32_t consumer) {
          if (!visitedDependencies.insert(consumer).second)
            return;
          const StructuredDAGNode *consumerNode = dag->getNode(consumer);
          if (!consumerNode)
            return;
          for (StructuredDAGEdgeID edgeId : consumerNode->incomingEdges) {
            const StructuredDAGEdge *edge = dag->getEdge(edgeId);
            if (!edge)
              continue;
            if (selectedNodes.count(edge->producer)) {
              collectRecomputableDependencies(edge->producer);
              continue;
            }
            if (!recomputableActualNodes.count(edge->producer))
              continue;
            auto placement = placements.find(edge->producer);
            if (placement == placements.end() ||
                !llvm::is_contained(placement->second->tiles, tile))
              continue;
            requiredRecomputations.insert(edge->producer);
            collectRecomputableDependencies(edge->producer);
          }
        };
    for (uint32_t node : selectedNodes)
      collectRecomputableDependencies(node);
    group.recomputedProducerNodes.assign(requiredRecomputations.begin(),
                                         requiredRecomputations.end());
    llvm::sort(group.shards, [](const auto &lhs, const auto &rhs) {
      return lhs.structuredNodeId < rhs.structuredNodeId;
    });
    llvm::sort(group.temporalTiles, [](const auto &lhs, const auto &rhs) {
      return lhs.structuredNodeId < rhs.structuredNodeId;
    });
    llvm::sort(group.representations, [](const auto &lhs, const auto &rhs) {
      return lhs.structuredNodeId < rhs.structuredNodeId;
    });
    llvm::sort(group.independentlyMaterializedNodes);
    llvm::sort(group.recomputedProducerNodes);
    if (!group.shards.empty())
      groups.push_back(std::move(group));
  }

  std::set<std::pair<uint32_t, int64_t>> expectedActualNodes;
  for (const auto &[node, placement] : placements)
    for (TileId tile : placement->tiles)
      expectedActualNodes.emplace(node, tile.getValue());
  if (coveredActualNodes != expectedActualNodes || groups.empty()) {
    diagnostics << "wafer-compile: baseline attention RegionPlan does not "
                   "cover every actual node/Tile\n";
    return mlir::failure();
  }

  MaterializedCardCandidate result;
  result.assignment = std::move(attention->assignment);
  result.assignment.selectedRegions = plan.regions;
  result.assignment.selectedRepresentations = plan.representations.plan;
  result.assignment.selectedRegionGroups = groups;
  result.nodeRoots = std::move(attention->nodeRoots);
  result.materializationSource = std::move(attention->module);
  wafer::support::ScopedCompileTimingSpan timing(
      "conversion", "deterministic-baseline",
      "attention-root-groups-to-card-module");
  if (mlir::failed(lowerStructuredNodeGroupsToCardModule(
          *result.materializationSource, cardId, program.availableTileIds,
          actualOperationNodes, groups, result.module, &result.relations,
          &failureReason))) {
    diagnostics << "wafer-compile: baseline attention CardModule "
                   "materialization failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  remapBaselineAttentionRelations(result.relations,
                                  semanticNodeByActualNode);
  if (statistics) {
    ++statistics->sourcePreparations;
    ++statistics->materializationPreparations;
    ++statistics->cardModuleMaterializations;
    statistics->tileEntryMaterializations += program.availableTileIds.size();
    statistics->maximumTileMaterializationWorkers = 1;
  }
  return result;
}

mlir::FailureOr<MaterializedCardCandidate> materializeBaselineCandidate(
    mlir::ModuleOp tensorProgram, CardId cardId,
    const CardProgramAnalysis &program, const CanonicalBaselinePlan &plan,
    CandidateMaterializationStatistics *statistics,
    llvm::raw_ostream &diagnostics) {
  if (!plan.preparedAttention.work.roots.empty()) {
    return materializeBaselineAttentionCandidate(
        tensorProgram, cardId, program, plan, statistics, diagnostics);
  }

  std::map<mlir::Operation *, uint32_t> nodesByOperation;
  for (const StructuredOperationNodeMapping &mapping : program.operationNodes)
    if (!mapping.operation ||
        !nodesByOperation
             .try_emplace(mapping.operation, mapping.structuredNodeId)
             .second) {
      diagnostics << "wafer-compile: baseline root materializer has a "
                     "malformed operation/node relation\n";
      return mlir::failure();
    }

  std::vector<StructuredNodeShardGroup> groups;
  std::map<uint32_t, SemanticRootKey> rootsByNode;
  std::vector<SelectedRegionExecutionNode> executionNodes;
  for (const RegionGroupPlan &region : plan.regions.groups) {
    if (!region.replicas.empty() || !region.localBindings.empty() ||
        region.executions.size() != 1) {
      diagnostics << "wafer-compile: baseline RegionPlan is not singleton\n";
      return mlir::failure();
    }
    const ExecutionInstancePlan &selected = region.executions.front();
    const auto *rootExecution =
        std::get_if<RequiredRootExecution>(&selected.id.source);
    if (!rootExecution) {
      if (std::holds_alternative<RequiredMergeExecution>(selected.id.source))
        continue;
      diagnostics << "wafer-compile: baseline execution has no root work\n";
      return mlir::failure();
    }
    const analysis::RootRegionWork *work =
        findRootWork(plan.rootWorks, rootExecution->work);
    auto node = work ? nodesByOperation.find(work->rootOperation)
                     : nodesByOperation.end();
    const analysis::RootExecutionWork *piece = nullptr;
    if (work) {
      auto found =
          llvm::find_if(work->execution, [&](const auto &candidate) {
            return candidate.shard == rootExecution->shard;
          });
      if (found != work->execution.end())
        piece = &*found;
    }
    const TemporalScopePlan *temporal =
        findBaselineTemporalScope(plan.temporal, selected.id);
    if (!work || node == nodesByOperation.end() ||
        !piece || work->id.tile != region.tile || !temporal) {
      diagnostics << "wafer-compile: baseline root execution is incomplete\n";
      return mlir::failure();
    }

    StructuredNodeIterationShard shard;
    shard.structuredNodeId = node->second;
    shard.tile = region.tile;
    for (const IteratorInterval &interval : piece->iterationDomain) {
      shard.offsets.push_back(interval.offset);
      shard.sizes.push_back(interval.size);
    }
    for (const analysis::RootContributionWork &contribution :
         work->contributions)
      if (contribution.contribution.shard == rootExecution->shard)
        shard.reductionGroups.push_back(
            {contribution.group, contribution.mergeTile});
    StructuredNodeShardGroup group;
    group.shards.push_back(std::move(shard));
    group.temporalTiles.push_back(
        {node->second, temporal->iteratorTileSizes,
         temporal->waveLoopOrder});
    group.independentlyMaterializedNodes.push_back(node->second);
    groups.push_back(std::move(group));
    executionNodes.push_back(
        {RegionExecutionId(selected.id), node->second, work->id.root});
    auto [root, inserted] =
        rootsByNode.try_emplace(node->second, work->id.root);
    if (!inserted && root->second != work->id.root) {
      diagnostics << "wafer-compile: baseline node has several semantic "
                     "roots\n";
      return mlir::failure();
    }
  }
  if (groups.empty()) {
    diagnostics << "wafer-compile: baseline root materializer has no work\n";
    return mlir::failure();
  }
  std::string failureReason;
  if (mlir::failed(applySelectedRegionRepresentations(
          plan.regions, plan.rootWorks, plan.representations.plan,
          executionNodes, groups, &failureReason))) {
    diagnostics << "wafer-compile: baseline representation preparation "
                   "failed: "
                << failureReason << '\n';
    return mlir::failure();
  }

  MaterializedCardCandidate result;
  result.assignment.spatial = plan.spatial;
  result.assignment.demand = plan.demand;
  for (const SelectedRegionExecutionNode &execution : executionNodes)
    result.assignment.selectedRegionExecutions.push_back(
        {execution.execution, execution.structuredNodeId});
  for (const auto &[node, root] : rootsByNode)
    result.nodeRoots.push_back({node, root});
  wafer::support::ScopedCompileTimingSpan timing(
      "conversion", "deterministic-baseline",
      "single-root-groups-to-card-module");
  if (mlir::failed(lowerStructuredNodeGroupsToCardModule(
          tensorProgram, cardId, program.availableTileIds,
          program.operationNodes, groups, result.module, &result.relations,
          &failureReason))) {
    diagnostics << "wafer-compile: baseline root CardModule materialization "
                   "failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics) {
    statistics->spatialCoordinateQueries = 1;
    statistics->exactDemandSatisfiedEdges = program.dag.getEdges().size();
    ++statistics->sourcePreparations;
    ++statistics->materializationPreparations;
    ++statistics->cardModuleMaterializations;
    statistics->tileEntryMaterializations += program.availableTileIds.size();
    statistics->maximumTileMaterializationWorkers = 1;
  }
  return result;
}

mlir::FailureOr<llvm::SmallVector<SemanticRootKey, 8>>
collectActualSPMRejectedRoots(
    const CardExecutableCompilationResult &compilation,
    const CardProgramAnalysis &program, const CanonicalBaselinePlan &plan,
    llvm::ArrayRef<CandidateNodeRootRelation> nodeRoots,
    std::string *failureReason) {
  if (!compilation.isProvenExactRejection() ||
      compilation.tileFailures.empty()) {
    if (failureReason)
      *failureReason =
          "candidate rejection is not an actual Tile SPM capacity result";
    return mlir::failure();
  }

  std::map<uint32_t, SemanticRootKey> rootsByNode;
  for (const CandidateNodeRootRelation &relation : nodeRoots)
    if (!rootsByNode.try_emplace(relation.structuredNodeId, relation.root)
             .second) {
      if (failureReason)
        *failureReason = "candidate node/root relation is duplicated";
      return mlir::failure();
    }

  std::map<uint32_t, SemanticRootKey> sourceRoots;
  for (const analysis::RootRegionWork &work : plan.rootWorks) {
    if (!work.rootOperation)
      continue;
    auto node = llvm::find_if(program.dag.getNodes(), [&](const auto &entry) {
      return entry.operation == work.rootOperation;
    });
    if (node == program.dag.getNodes().end())
      continue;
    auto [position, inserted] = sourceRoots.try_emplace(node->id, work.id.root);
    if (!inserted && position->second != work.id.root) {
      if (failureReason)
        *failureReason = "source node maps to several semantic roots";
      return mlir::failure();
    }
  }

  std::set<SemanticRootKey> affected;
  auto collectDemand =
      [&](const TileMemoryPlanningFailure::SPMDemandEvidence &demand)
      -> mlir::LogicalResult {
    std::set<SemanticRootKey> demandRoots;
    auto collectNode = [&](uint32_t node) {
      auto root = rootsByNode.find(node);
      if (root == rootsByNode.end())
        return false;
      demandRoots.insert(root->second);
      return true;
    };
    for (uint32_t node : demand.operationResultNodes)
      if (!collectNode(node))
        return mlir::failure();
    for (uint32_t node : demand.operandDemandNodes)
      if (!collectNode(node))
        return mlir::failure();
    for (uint32_t node : demand.scratchNodes)
      if (!collectNode(node))
        return mlir::failure();
    for (unsigned output : demand.outputIndices) {
      auto outputRoots = program.dag.getObservableOutputRootNodes();
      if (output >= outputRoots.size() || outputRoots[output].empty())
        return mlir::failure();
      for (uint32_t node : outputRoots[output]) {
        auto root = sourceRoots.find(node);
        if (root == sourceRoots.end())
          return mlir::failure();
        demandRoots.insert(root->second);
      }
    }
    if (demandRoots.empty())
      return mlir::failure();
    affected.insert(demandRoots.begin(), demandRoots.end());
    return mlir::success();
  };

  bool sawCapacityDemand = false;
  for (const CardExecutableTileFailure &tile : compilation.tileFailures) {
    if (!isProvenExactTileMemoryPlanningFailure(tile.memoryPlanning)) {
      if (failureReason)
        *failureReason =
            "candidate rejection mixes SPM capacity with another failure";
      return mlir::failure();
    }
    for (const auto &demand : tile.memoryPlanning.spmCapacityConflictDemands) {
      sawCapacityDemand = true;
      if (mlir::failed(collectDemand(demand))) {
        if (failureReason)
          *failureReason =
              "actual SPM conflict demand has no exact semantic owner";
        return mlir::failure();
      }
    }
    for (const auto &demand :
         tile.memoryPlanning.spmIndividuallyOversizedDemands) {
      sawCapacityDemand = true;
      if (mlir::failed(collectDemand(demand))) {
        if (failureReason)
          *failureReason =
              "actual oversized SPM demand has no exact semantic owner";
        return mlir::failure();
      }
    }
  }
  if (!sawCapacityDemand || affected.empty()) {
    if (failureReason)
      *failureReason = "actual SPM rejection has no causal conflict demand";
    return mlir::failure();
  }
  return llvm::SmallVector<SemanticRootKey, 8>(affected.begin(),
                                               affected.end());
}

void printSPMFailure(const CardExecutableCompilationResult &compilation,
                     llvm::raw_ostream &diagnostics) {
  if (compilation.tileFailures.empty())
    return;
  const CardExecutableTileFailure &tile = compilation.tileFailures.front();
  const TileMemoryPlanningFailure &memory = tile.memoryPlanning;
  if (!memory.spmCapacityOverflow)
    return;
  diagnostics << "wafer-compile: baseline actual SPM conflict tile="
              << tile.tileId.getValue();
  for (const auto &demand : memory.spmCapacityConflictDemands) {
    diagnostics << " demand(bytes=" << demand.bytes << ", type=" << demand.type
                << ", result_nodes=[";
    llvm::interleaveComma(demand.operationResultNodes, diagnostics);
    diagnostics << "], operand_nodes=[";
    llvm::interleaveComma(demand.operandDemandNodes, diagnostics);
    diagnostics << "], scratch_nodes=[";
    llvm::interleaveComma(demand.scratchNodes, diagnostics);
    diagnostics << "], outputs=[";
    llvm::interleaveComma(demand.outputIndices, diagnostics);
    diagnostics << "], users=[";
    llvm::interleaveComma(
        demand.userOperationNames, diagnostics,
        [&](mlir::OperationName name) { diagnostics << name.getStringRef(); });
    diagnostics << "])";
  }
  diagnostics << '\n';
}

void accumulateMaterializationStatistics(
    BaselineStatistics &baseline,
    const CandidateMaterializationStatistics &candidate) {
  baseline.spatialCoordinateQueries += candidate.spatialCoordinateQueries;
  baseline.exactDemandSatisfiedEdges = std::max(
      baseline.exactDemandSatisfiedEdges, candidate.exactDemandSatisfiedEdges);
  baseline.baselineSourcePreparations += candidate.sourcePreparations;
  baseline.baselineMaterializationPreparations +=
      candidate.materializationPreparations;
  baseline.baselineCardModuleMaterializations +=
      candidate.cardModuleMaterializations;
  baseline.baselineTileEntryMaterializations +=
      candidate.tileEntryMaterializations;
  baseline.baselineMaximumTileMaterializationWorkers =
      std::max(baseline.baselineMaximumTileMaterializationWorkers,
               candidate.maximumTileMaterializationWorkers);
}

} // namespace

mlir::FailureOr<CardBaselineCompilationResult> compileCardBaseline(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData, BaselineStatistics *baselineStatistics,
    unsigned tilePipelineParallelism, bool captureTileDataflowIRTrace) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "stage", "tensor-program-to-executable", "card-baseline-compilation");
  if (baselineStatistics)
    *baselineStatistics = {};

  mlir::FailureOr<std::unique_ptr<CardProgramAnalysis>> analysis = [&]() {
    wafer::support::ScopedCompileTimingSpan timing(
        "query", "deterministic-baseline", "analyze-card-program");
    return analyzeCardProgram(tensorProgram, program, executionConfig,
                              diagnostics);
  }();
  if (mlir::failed(analysis))
    return mlir::failure();

  constexpr CardId cardId(0);
  std::string failureReason;
  mlir::FailureOr<CanonicalBaselinePlan> resolved = [&]() {
    wafer::support::ScopedCompileTimingSpan timing(
        "query", "deterministic-baseline", "build-initial-candidate");
    return buildCanonicalBaselinePlan(**analysis, &failureReason);
  }();
  if (mlir::failed(resolved)) {
    diagnostics << "wafer-compile: canonical baseline planning failed: "
                << failureReason << '\n';
    return mlir::failure();
  }

  while (true) {
    CandidateMaterializationStatistics candidateStatistics;
    mlir::FailureOr<MaterializedCardCandidate> materialized =
        materializeBaselineCandidate(
            tensorProgram, cardId, **analysis, *resolved,
            baselineStatistics ? &candidateStatistics : nullptr, diagnostics);
    if (baselineStatistics)
      accumulateMaterializationStatistics(*baselineStatistics,
                                          candidateStatistics);
    if (mlir::failed(materialized))
      return mlir::failure();

    llvm::SmallVector<CandidateNodeRootRelation, 64> nodeRoots =
        materialized->nodeRoots;
    CardExecutablePreparation preparation;
    CardExecutableCompilationResult compilation = compileCardModuleToExecutable(
        std::move(materialized->module), cardId, (*analysis)->availableTileIds,
        materialized->relations, preparation, program, executionConfig,
        diagnostics, programData,
        baselineStatistics ? &baselineStatistics->exactGates : nullptr,
        tilePipelineParallelism, captureTileDataflowIRTrace);
    if (compilation.isAccepted()) {
      if (baselineStatistics)
        baselineStatistics->baselineTileIRPrints +=
            compilation.tileDataflowIRTrace.size();
      return CardBaselineCompilationResult(
          compilation.takeExecutable(),
          std::move(compilation.tileDataflowIRTrace));
    }

    if (baselineStatistics) {
      ++baselineStatistics->materializationRejections;
      if (compilation.status ==
          CardExecutableCompilationStatus::IndeterminateFailure)
        ++baselineStatistics->indeterminateCompilationFailures;
    }
    printSPMFailure(compilation, diagnostics);

    std::string feedbackFailure;
    mlir::FailureOr<llvm::SmallVector<SemanticRootKey, 8>> affected =
        collectActualSPMRejectedRoots(compilation, **analysis, *resolved,
                                      nodeRoots, &feedbackFailure);
    if (mlir::succeeded(affected)) {
      if (baselineStatistics)
        ++baselineStatistics->actualSPMCapacityRejections;
      mlir::FailureOr<bool> refined = refineTemporalPlanFromActualSPMFeedback(
          resolved->temporal, resolved->rootWorks, *affected, &feedbackFailure);
      if (mlir::succeeded(refined) && *refined) {
        if (mlir::failed(
                recloseCanonicalBaselinePlan(*resolved, &feedbackFailure))) {
          diagnostics << "wafer-compile: baseline candidate reclose failed: "
                      << feedbackFailure << '\n';
          return mlir::failure();
        }
        if (baselineStatistics)
          ++baselineStatistics->actualTemporalRefinements;
        continue;
      }
    }

    diagnostics << "wafer-compile: baseline CardExecutable gate failed: gate="
                << compilation.gate << " detail=" << compilation.detail;
    if (!feedbackFailure.empty())
      diagnostics << " feedback=" << feedbackFailure;
    diagnostics << '\n';
    return mlir::failure();
  }
}

} // namespace wafer::compiler::detail
