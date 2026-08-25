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

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

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

struct SearchAttentionExecutionSource {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SmallVector<StructuredOperationNodeMapping, 64> operationNodes;
  std::vector<SelectedRegionExecutionNode> executionNodes;
  std::vector<StructuredNodeShardGroup> groups;
  llvm::SmallVector<CandidateNodeRootRelation, 64> nodeRoots;
  CardMaterializationPlan assignment;
  std::map<uint32_t, uint32_t> semanticNodeByActualNode;
};

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getStaticIterationExtents(mlir::Operation *operation,
                          std::string *failureReason) {
  auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(operation);
  if (!tiling) {
    if (failureReason)
      *failureReason =
          "search attention operation has no TilingInterface";
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
            "search attention operation has a dynamic iteration domain";
      return mlir::failure();
    }
    extents.push_back(*size);
  }
  if (extents.size() != tiling.getLoopIteratorTypes().size()) {
    if (failureReason)
      *failureReason =
          "search attention operation iteration rank is inconsistent";
    return mlir::failure();
  }
  return extents;
}

analysis::RootRegionWorkId
getRegionExecutionWork(const RegionExecutionId &execution) {
  if (const auto *required = std::get_if<ExecutionInstanceId>(&execution))
    return std::visit([](const auto &source) { return source.work; },
                      required->source);
  return std::get<ReplicaExecutionId>(execution).producer.work;
}

bool isIdentityBroadcastGeneric(mlir::Operation *operation) {
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

mlir::FailureOr<SearchAttentionExecutionSource>
prepareSearchAttentionExecutionSource(
    mlir::ModuleOp tensorProgram, CardId cardId,
    const CardProgramAnalysis &program, const CompleteCandidatePlan &plan,
    CandidateMaterializationStatistics *statistics,
    std::string *failureReason) {
  mlir::FailureOr<AttentionMaterializationSource> attention =
      prepareAttentionMaterializationSource(
          tensorProgram, cardId, program, plan, program.availableTileIds,
          SpatialDataflowMaterializationMode::JointDataflow, statistics,
          failureReason);
  if (mlir::failed(attention))
    return mlir::failure();

  mlir::func::FuncOp selectedProgram;
  for (mlir::func::FuncOp function :
       attention->module->getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (selectedProgram) {
      if (failureReason)
        *failureReason =
            "search attention source has several program functions";
      return mlir::failure();
    }
    selectedProgram = function;
  }
  if (!selectedProgram) {
    if (failureReason)
      *failureReason = "search attention source has no program function";
    return mlir::failure();
  }
  mlir::FailureOr<StructuredDAGAnalysis> dag =
      StructuredDAGAnalysis::create(selectedProgram, failureReason);
  if (mlir::failed(dag))
    return mlir::failure();

  std::map<mlir::Operation *, const StructuredOperationNodeMapping *>
      sourceMappings;
  std::map<uint32_t, SemanticRootKey> rootsBySemanticNode;
  for (const CandidateNodeRootRelation &relation : attention->nodeRoots)
    if (!rootsBySemanticNode.try_emplace(relation.structuredNodeId,
                                         relation.root)
             .second) {
      if (failureReason)
        *failureReason =
            "search attention source has duplicate semantic node roots";
      return mlir::failure();
    }
  for (const StructuredOperationNodeMapping &mapping :
       attention->operationNodes)
    if (!mapping.operation ||
        !sourceMappings.try_emplace(mapping.operation, &mapping).second) {
      if (failureReason)
        *failureReason =
            "search attention source has malformed operation mappings";
      return mlir::failure();
    }

  std::map<uint32_t, const StructuredDAGNode *> dagNodes;
  std::map<uint32_t, const StructuredDAGNodePlacement *> placements;
  std::map<uint32_t, llvm::SmallVector<uint32_t, 8>>
      actualNodesBySemanticNode;
  SearchAttentionExecutionSource result;
  result.module = std::move(attention->module);
  result.assignment = std::move(attention->assignment);
  result.nodeRoots = std::move(attention->nodeRoots);
  for (const StructuredDAGNode &node : dag->getNodes()) {
    auto source = sourceMappings.find(node.operation);
    if (source == sourceMappings.end() ||
        !dagNodes.try_emplace(node.id, &node).second) {
      if (failureReason)
        *failureReason =
            "search attention DAG lost one selected operation mapping";
      return mlir::failure();
    }
    result.operationNodes.push_back(
        {node.operation, node.id, source->second->coupledComponentIndices});
    result.semanticNodeByActualNode.emplace(
        node.id, source->second->structuredNodeId);
    actualNodesBySemanticNode[source->second->structuredNodeId].push_back(
        node.id);
  }
  for (const StructuredDAGNodePlacement &placement :
       result.assignment.nodePlacements)
    if (!placements.try_emplace(placement.node, &placement).second) {
      if (failureReason)
        *failureReason =
            "search attention source has duplicate actual node placement";
      return mlir::failure();
    }

  std::map<mlir::Operation *, const StructuredOpTemporalTile *>
      temporalByOperation;
  for (const StructuredOpTemporalTile &temporal :
       result.assignment.mapping.operationTemporalTiles)
    if (!temporal.operation ||
        !temporalByOperation.try_emplace(temporal.operation, &temporal)
             .second) {
      if (failureReason)
        *failureReason =
            "search attention source has duplicate temporal operation";
      return mlir::failure();
    }

  std::map<RegionExecutionId, uint32_t> semanticNodeByExecution;
  for (const auto &[execution, node] :
       result.assignment.selectedRegionExecutions)
    if (!semanticNodeByExecution.try_emplace(execution, node).second) {
      if (failureReason)
        *failureReason =
            "search attention source has duplicate execution identity";
      return mlir::failure();
    }

  auto findComponentEncoding =
      [&](const RegionExecutionId &execution,
          wafer::CoupledReductionComponentKind component)
      -> std::optional<MemLayout> {
    const auto *required = std::get_if<ExecutionInstanceId>(&execution);
    if (!required)
      return std::nullopt;
    std::optional<MemLayout> encoding;
    for (const LogicalRepresentationPlan &logical :
         plan.representations.logicalValues) {
      const auto *value =
          std::get_if<CoupledComponentValueId>(&logical.value);
      if (!value || !(value->execution == *required) ||
          value->component != component)
        continue;
      auto physical = llvm::find_if(
          plan.representations.physicalVersions,
          [&](const PhysicalVersionPlan &candidate) {
            return candidate.id == logical.primary;
          });
      if (physical == plan.representations.physicalVersions.end() ||
          (encoding && *encoding != physical->encoding))
        return std::nullopt;
      encoding = physical->encoding;
    }
    return encoding;
  };

  std::set<std::pair<uint32_t, int64_t>> coveredActualNodes;
  for (const RegionGroupPlan &region : plan.regions.groups) {
    StructuredNodeShardGroup group;
    llvm::SmallVector<RegionExecutionId, 8> executions;
    for (const ExecutionInstancePlan &execution : region.executions)
      executions.emplace_back(execution.id);
    for (const ReplicaExecutionPlan &replica : region.replicas)
      executions.emplace_back(replica.id);
    for (const RegionExecutionId &execution : executions) {
      auto semantic = semanticNodeByExecution.find(execution);
      if (semantic == semanticNodeByExecution.end()) {
        const SemanticRootKey root = getRegionExecutionWork(execution).root;
        llvm::SmallVector<uint32_t, 2> candidates;
        for (const auto &[node, candidateRoot] : rootsBySemanticNode)
          if (candidateRoot == root)
            candidates.push_back(node);
        if (candidates.size() != 1) {
          if (failureReason)
            *failureReason =
                "search attention execution has no unique semantic node";
          return mlir::failure();
        }
        semantic = semanticNodeByExecution
                       .try_emplace(execution, candidates.front())
                       .first;
      }
      auto actualNodes = actualNodesBySemanticNode.find(semantic->second);
      if (actualNodes == actualNodesBySemanticNode.end()) {
        if (failureReason)
          *failureReason =
              "search attention execution has no actual structured nodes";
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
        if (!coveredActualNodes.emplace(actualNode, region.tile.getValue())
                 .second)
          continue;
        mlir::FailureOr<llvm::SmallVector<int64_t, 4>> extents =
            getStaticIterationExtents(node->operation, failureReason);
        auto temporal = temporalByOperation.find(node->operation);
        if (mlir::failed(extents) || temporal == temporalByOperation.end() ||
            temporal->second->iteratorTileSizes.size() != extents->size()) {
          if (failureReason && failureReason->empty())
            *failureReason =
                "search attention actual node has no complete temporal plan";
          return mlir::failure();
        }
        if (mlir::isa<mlir::linalg::FillOp>(node->operation) ||
            isIdentityBroadcastGeneric(node->operation)) {
          group.recomputedProducerNodes.push_back(actualNode);
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
          if (failureReason)
            *failureReason =
                "search attention component/result mapping is incomplete";
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
              if (failureReason)
                *failureReason =
                    "search attention component kind is outside the IR "
                    "contract";
              return mlir::failure();
            }
            auto selected = findComponentEncoding(
                execution,
                static_cast<wafer::CoupledReductionComponentKind>(raw));
            if (!selected) {
              if (failureReason)
                *failureReason =
                    "search attention component has no selected physical "
                    "version";
              return mlir::failure();
            }
            layout = *selected;
          }
          representation.resultLayouts.push_back(layout);
        }
        group.representations.push_back(std::move(representation));
      }
    }
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
      result.groups.push_back(std::move(group));
  }

  std::set<std::pair<uint32_t, int64_t>> expectedActualNodes;
  for (const auto &[node, placement] : placements)
    for (TileId tile : placement->tiles)
      expectedActualNodes.emplace(node, tile.getValue());
  if (coveredActualNodes != expectedActualNodes) {
    if (failureReason)
      *failureReason =
          "search attention RegionPlan does not cover every actual node/Tile";
    return mlir::failure();
  }

  for (const auto &[execution, node] : semanticNodeByExecution) {
    auto root = rootsBySemanticNode.find(node);
    if (root == rootsBySemanticNode.end()) {
      if (failureReason)
        *failureReason =
            "search attention execution has no semantic root relation";
      return mlir::failure();
    }
    result.executionNodes.push_back({execution, node, root->second});
  }
  llvm::sort(result.executionNodes, [](const auto &lhs, const auto &rhs) {
    return lhs.execution < rhs.execution;
  });
  result.assignment.selectedRegionExecutions.clear();
  for (const SelectedRegionExecutionNode &execution : result.executionNodes)
    result.assignment.selectedRegionExecutions.push_back(
        {execution.execution, execution.structuredNodeId});
  return result;
}

void remapAttentionMaterializationRelations(
    StructuredMaterializationRelations &relations,
    const std::map<uint32_t, uint32_t> &semanticNodeByActualNode) {
  auto remap = [&](auto &entries) {
    for (auto &entry : entries) {
      auto semantic =
          semanticNodeByActualNode.find(entry.structuredNodeId);
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

} // namespace

mlir::FailureOr<MaterializedCardCandidate>
materializeSearchCardCandidate(
    mlir::ModuleOp tensorProgram, CardId cardId,
    const CardProgramAnalysis &program, const CompleteCandidatePlan &plan,
    CandidateMaterializationStatistics *statistics,
    llvm::raw_ostream &diagnostics) {
  std::string failureReason;
  CanonicalRepresentationPlanOutcome canonicalRepresentation =
      buildCanonicalRepresentationPlan(plan.regions, plan.temporal,
                                       plan.rootWorks);
  const CanonicalRepresentationCoordinate *representationCoordinate =
      getCanonicalRepresentationCoordinate(canonicalRepresentation);
  if (!representationCoordinate) {
    diagnostics << "wafer-compile: search representation inventory "
                   "validation failed\n";
    return mlir::failure();
  }
  RepresentationDomainResult representationDomain =
      buildRepresentationDomain(*representationCoordinate);
  if (!representationDomain.succeeded() ||
      !representationDomain.domain->contains(plan.representations)) {
    diagnostics << "wafer-compile: search representation plan is outside "
                   "its current domain\n";
    return mlir::failure();
  }
  if (mlir::failed(prepareRepresentationPlan(
          *representationDomain.domain, plan.representations,
          &failureReason))) {
    diagnostics << "wafer-compile: search representation preflight failed: "
                << failureReason << '\n';
    return mlir::failure();
  }

  if (!plan.preparedAttention.work.roots.empty()) {
    mlir::FailureOr<SearchAttentionExecutionSource> source =
        prepareSearchAttentionExecutionSource(
            tensorProgram, cardId, program, plan, statistics, &failureReason);
    if (mlir::failed(source)) {
      diagnostics << "wafer-compile: search attention execution "
                     "preparation failed: "
                  << failureReason << '\n';
      return mlir::failure();
    }
    if (statistics) {
      ++statistics->sourcePreparations;
      ++statistics->materializationPreparations;
    }
    MaterializedCardCandidate result;
    result.assignment = std::move(source->assignment);
    result.assignment.selectedRegions = plan.regions;
    result.assignment.selectedRepresentations = plan.representations;
    result.assignment.selectedRegionGroups = source->groups;
    result.nodeRoots = std::move(source->nodeRoots);
    result.materializationSource = std::move(source->module);
    wafer::support::ScopedCompileTimingSpan timing(
        "conversion", "search-execution",
        "selected-attention-to-card-module");
    if (mlir::failed(lowerStructuredNodeGroupsToCardModule(
            *result.materializationSource, cardId, program.availableTileIds,
            source->operationNodes, source->groups, result.module,
            &result.relations, &failureReason))) {
      diagnostics << "wafer-compile: search attention CardModule "
                     "materialization failed: "
                  << failureReason << '\n';
      return mlir::failure();
    }
    remapAttentionMaterializationRelations(
        result.relations, source->semanticNodeByActualNode);
    if (statistics) {
      ++statistics->cardModuleMaterializations;
      statistics->tileEntryMaterializations +=
          program.availableTileIds.size();
      statistics->maximumTileMaterializationWorkers = 1;
    }
    return result;
  }

  mlir::FailureOr<SelectedRegionMaterializationSource> source =
      prepareSelectedRegionMaterializationSource(
          tensorProgram, program, plan.regions, plan.rootWorks,
          &failureReason);
  if (mlir::failed(source)) {
    diagnostics << "wafer-compile: search execution source preparation "
                   "failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics)
    ++statistics->sourcePreparations;

  mlir::FailureOr<std::vector<StructuredNodeShardGroup>> groups =
      prepareSelectedRegionGroups(program, plan.regions, plan.rootWorks,
                                  plan.temporal, source->executionNodes,
                                  &failureReason);
  if (mlir::failed(groups)) {
    diagnostics << "wafer-compile: search execution preparation failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics)
    ++statistics->materializationPreparations;

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
    }
  }

  if (mlir::failed(applySelectedRegionRepresentations(
          plan.regions, plan.rootWorks, plan.representations,
          source->executionNodes, *groups, &failureReason))) {
    diagnostics << "wafer-compile: search representation preparation "
                   "failed: "
                << failureReason << '\n';
    return mlir::failure();
  }

  MaterializedCardCandidate result;
  result.assignment.spatial = plan.spatial;
  result.assignment.demand = plan.demand;
  result.assignment.selectedRegions = plan.regions;
  result.assignment.selectedRepresentations = plan.representations;
  result.assignment.selectedRegionGroups = *groups;
  std::map<uint32_t, SemanticRootKey> rootsByNode;
  for (const SelectedRegionExecutionNode &execution : source->executionNodes) {
    auto [position, inserted] = rootsByNode.try_emplace(
        execution.structuredNodeId, execution.root);
    if (!inserted && position->second != execution.root) {
      diagnostics << "wafer-compile: search execution node has several "
                     "semantic roots\n";
      return mlir::failure();
    }
    result.assignment.selectedRegionExecutions.push_back(
        {execution.execution, execution.structuredNodeId});
  }
  for (const auto &[node, root] : rootsByNode)
    result.nodeRoots.push_back({node, root});
  result.materializationSource = std::move(source->module);

  wafer::support::ScopedCompileTimingSpan timing(
      "conversion", "search-execution", "selected-region-to-card-module");
  if (mlir::failed(lowerStructuredNodeGroupsToCardModule(
          *result.materializationSource, cardId, program.availableTileIds,
          source->operationNodes, *groups, result.module, &result.relations,
          &failureReason))) {
    diagnostics << "wafer-compile: search execution CardModule "
                   "materialization failed: "
                << failureReason << '\n';
    return mlir::failure();
  }
  if (statistics) {
    ++statistics->cardModuleMaterializations;
    statistics->tileEntryMaterializations += program.availableTileIds.size();
    statistics->maximumTileMaterializationWorkers = 1;
  }
  return result;
}

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
