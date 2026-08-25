//===- SelectedRegionMaterialization.cpp - RegionPlan apply -----------===//

#include "Wafer/Planning/PhysicalDataflow/SelectedRegionMaterialization.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
  return mlir::failure();
}

const analysis::RootRegionWork *
findWork(llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
         const analysis::RootRegionWorkId &id) {
  auto work = llvm::find_if(
      rootWorks, [&](const auto &candidate) { return candidate.id == id; });
  return work == rootWorks.end() ? nullptr : &*work;
}

const TemporalScopePlan *
findTopLevelTemporalScope(const TemporalPlan &temporal,
                          const RegionExecutionId &id) {
  auto scope = llvm::find_if(temporal.scopes, [&](const auto &candidate) {
    return candidate.id.execution == id && isTopLevelScope(candidate.id);
  });
  return scope == temporal.scopes.end() ? nullptr : &*scope;
}

const StructuredOperationNodeMapping *
findNode(const CardProgramAnalysis &program, mlir::Operation *operation) {
  auto node = llvm::find_if(program.operationNodes, [&](const auto &candidate) {
    return candidate.operation == operation;
  });
  return node == program.operationNodes.end() ? nullptr : &*node;
}

const analysis::RootExecutionWork *
findExecution(const analysis::RootRegionWork &work, LogicalShardId shard) {
  auto execution = llvm::find_if(work.execution, [&](const auto &candidate) {
    return candidate.shard == shard;
  });
  return execution == work.execution.end() ? nullptr : &*execution;
}

bool isStructuredMember(const RegionGroupPlan &group,
                        const DemandFragmentId &fragment) {
  if (fragment.source.kind != analysis::RootBoundaryKind::StructuredResult)
    return false;
  return llvm::any_of(group.mandatoryRoots, [&](const auto &work) {
    return work.root == fragment.source.semantic;
  });
}

std::optional<analysis::RootRegionWorkId>
getExecutionWork(const RegionExecutionId &execution) {
  if (const auto *required = std::get_if<ExecutionInstanceId>(&execution))
    return std::visit([](const auto &source) { return source.work; },
                      required->source);
  return std::get<ReplicaExecutionId>(execution).producer.work;
}

const LogicalShardId *getExecutionShard(const RegionExecutionId &execution) {
  if (const auto *required = std::get_if<ExecutionInstanceId>(&execution)) {
    const auto *root = std::get_if<RequiredRootExecution>(&required->source);
    return root ? &root->shard : nullptr;
  }
  return &std::get<ReplicaExecutionId>(execution).producer.shard;
}

const PhysicalVersionPlan *findPhysicalVersion(const RepresentationPlan &plan,
                                               const PhysicalVersionId &id) {
  auto version = llvm::find_if(
      plan.physicalVersions,
      [&](const PhysicalVersionPlan &candidate) { return candidate.id == id; });
  return version == plan.physicalVersions.end() ? nullptr : &*version;
}

const PhysicalVersionPlan *
findPrimaryVersion(const RepresentationPlan &plan,
                   const RegionValueVersionId &logical) {
  auto value = llvm::find_if(plan.logicalValues,
                             [&](const LogicalRepresentationPlan &candidate) {
                               return candidate.value == logical;
                             });
  return value == plan.logicalValues.end()
             ? nullptr
             : findPhysicalVersion(plan, value->primary);
}

} // namespace

mlir::FailureOr<SelectedRegionMaterializationSource>
prepareSelectedRegionMaterializationSource(
    mlir::ModuleOp source, const CardProgramAnalysis &program,
    const RegionPlan &regions,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    std::string *failureReason) {
  if (!source)
    return fail<SelectedRegionMaterializationSource>(
        failureReason, "selected region source is null");
  mlir::IRMapping cloneMapping;
  mlir::OwningOpRef<mlir::ModuleOp> selected =
      mlir::cast<mlir::ModuleOp>(source->clone(cloneMapping));

  SelectedRegionMaterializationSource result;
  result.module = std::move(selected);
  uint32_t nextNode = 0;
  for (const StructuredOperationNodeMapping &mapping : program.operationNodes) {
    mlir::Operation *cloned = cloneMapping.lookupOrNull(mapping.operation);
    if (!cloned ||
        mapping.structuredNodeId == std::numeric_limits<uint32_t>::max())
      return fail<SelectedRegionMaterializationSource>(
          failureReason,
          "selected region source cannot clone one structured node");
    result.operationNodes.push_back(
        {cloned, mapping.structuredNodeId, mapping.coupledComponentIndices});
    nextNode = std::max(nextNode, mapping.structuredNodeId + 1);
  }

  std::set<ExecutionInstanceId> requiredExecutions;
  std::set<ReplicaExecutionId> replicas;
  for (const RegionGroupPlan &group : regions.groups) {
    for (const ExecutionInstancePlan &execution : group.executions) {
      if (!requiredExecutions.insert(execution.id).second)
        return fail<SelectedRegionMaterializationSource>(
            failureReason,
            "selected region source repeats one required execution");
      const analysis::RootRegionWorkId workId = std::visit(
          [](const auto &entry) { return entry.work; }, execution.id.source);
      const analysis::RootRegionWork *work = findWork(rootWorks, workId);
      const StructuredOperationNodeMapping *node =
          work ? findNode(program, work->rootOperation) : nullptr;
      if (!work || !node)
        return fail<SelectedRegionMaterializationSource>(
            failureReason,
            "selected region source cannot map one required execution");
      result.executionNodes.push_back(
          {execution.id, node->structuredNodeId, work->id.root});
    }
  }

  for (const RegionGroupPlan &group : regions.groups) {
    for (const ReplicaExecutionPlan &replica : group.replicas) {
      if (!replicas.insert(replica.id).second ||
          nextNode == std::numeric_limits<uint32_t>::max())
        return fail<SelectedRegionMaterializationSource>(
            failureReason,
            "selected region source repeats or overflows a replica identity");
      const analysis::RootRegionWork *producerWork =
          findWork(rootWorks, replica.id.producer.work);
      const StructuredOperationNodeMapping *producerNode =
          producerWork ? findNode(program, producerWork->rootOperation)
                       : nullptr;
      if (!producerWork || !producerNode ||
          replica.id.fragment.source.kind !=
              analysis::RootBoundaryKind::StructuredResult ||
          replica.id.fragment.source.index >=
              producerWork->rootOperation->getNumResults())
        return fail<SelectedRegionMaterializationSource>(
            failureReason,
            "selected replica has no direct structured producer result");
      auto consumerExecution = llvm::find_if(
          group.executions, [&](const ExecutionInstancePlan &execution) {
            const auto *root =
                std::get_if<RequiredRootExecution>(&execution.id.source);
            return root &&
                   root->shard == replica.id.fragment.use.destinationShard;
          });
      if (consumerExecution == group.executions.end())
        return fail<SelectedRegionMaterializationSource>(
            failureReason,
            "selected replica has no destination execution in its group");
      const auto &consumerRoot =
          std::get<RequiredRootExecution>(consumerExecution->id.source);
      const analysis::RootRegionWork *consumerWork =
          findWork(rootWorks, consumerRoot.work);
      mlir::Operation *clonedConsumer =
          consumerWork ? cloneMapping.lookupOrNull(consumerWork->rootOperation)
                       : nullptr;
      mlir::Operation *originalProducer = producerWork->rootOperation;
      if (!clonedConsumer ||
          replica.id.fragment.use.operand >= clonedConsumer->getNumOperands())
        return fail<SelectedRegionMaterializationSource>(
            failureReason,
            "selected replica consumer operand is not representable");
      mlir::Value expectedSource = cloneMapping.lookupOrNull(
          originalProducer->getResult(replica.id.fragment.source.index));
      mlir::OpOperand &consumerOperand =
          clonedConsumer->getOpOperand(replica.id.fragment.use.operand);
      if (!expectedSource || consumerOperand.get() != expectedSource)
        return fail<SelectedRegionMaterializationSource>(
            failureReason,
            "selected replica requires support reconstruction before direct "
            "operand rewiring");

      mlir::IRMapping replicaMapping;
      for (mlir::Value operand : originalProducer->getOperands()) {
        mlir::Value mapped = cloneMapping.lookupOrNull(operand);
        if (!mapped)
          return fail<SelectedRegionMaterializationSource>(
              failureReason,
              "selected replica producer operand left the candidate clone");
        replicaMapping.map(operand, mapped);
      }
      mlir::OpBuilder builder(clonedConsumer);
      mlir::Operation *clonedProducer =
          builder.clone(*originalProducer, replicaMapping);
      if (!clonedProducer ||
          replica.id.fragment.source.index >= clonedProducer->getNumResults())
        return fail<SelectedRegionMaterializationSource>(
            failureReason, "selected replica producer clone is malformed");
      consumerOperand.set(
          clonedProducer->getResult(replica.id.fragment.source.index));
      result.operationNodes.push_back({clonedProducer, nextNode});
      result.executionNodes.push_back(
          {replica.id, nextNode, producerWork->id.root});
      ++nextNode;
    }
  }
  llvm::sort(result.executionNodes, [](const auto &lhs, const auto &rhs) {
    return lhs.execution < rhs.execution;
  });
  if (mlir::failed(mlir::verify(*result.module)))
    return fail<SelectedRegionMaterializationSource>(
        failureReason, "selected region source clone failed verification");
  return result;
}

mlir::FailureOr<std::vector<StructuredNodeShardGroup>>
prepareSelectedRegionGroups(
    const CardProgramAnalysis &program, const RegionPlan &regions,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    const TemporalPlan &temporal,
    llvm::ArrayRef<SelectedRegionExecutionNode> executionNodes,
    std::string *failureReason) {
  std::vector<StructuredNodeShardGroup> result;
  result.reserve(regions.groups.size());
  std::set<ExecutionInstanceId> seenRequired;
  std::set<ReplicaExecutionId> seenReplicas;
  std::set<TraversalScopeId> seenTemporalScopes;
  std::map<RegionExecutionId, const SelectedRegionExecutionNode *>
      selectedNodes;
  for (const SelectedRegionExecutionNode &node : executionNodes)
    if (!selectedNodes.try_emplace(node.execution, &node).second)
      return fail<std::vector<StructuredNodeShardGroup>>(
          failureReason,
          "selected execution/node relation contains a duplicate identity");

  for (const RegionGroupPlan &group : regions.groups) {
    StructuredNodeShardGroup selected;
    std::map<uint32_t, const ExecutionInstancePlan *> executionsByNode;
    std::map<ExecutionInstanceId, uint32_t> nodesByExecution;
    std::map<ReplicaExecutionId, uint32_t> nodesByReplica;
    std::map<RegionExecutionId, uint32_t> nodesByRegionExecution;
    std::set<RegionExecutionId> temporalExecutions;
    for (const ExecutionInstancePlan &execution : group.executions) {
      if (!seenRequired.insert(execution.id).second)
        return fail<std::vector<StructuredNodeShardGroup>>(
            failureReason,
            "selected region repeats one required execution instance");
      const auto *root =
          std::get_if<RequiredRootExecution>(&execution.id.source);
      if (!root)
        continue;
      const analysis::RootRegionWork *work = findWork(rootWorks, root->work);
      const analysis::RootExecutionWork *piece =
          work ? findExecution(*work, root->shard) : nullptr;
      const StructuredOperationNodeMapping *node =
          work ? findNode(program, work->rootOperation) : nullptr;
      const bool topLevel =
          std::holds_alternative<ExecutionInstancePlan::TopLevel>(
              execution.placement);
      const TemporalScopePlan *scope =
          topLevel ? findTopLevelTemporalScope(temporal,
                                               RegionExecutionId(execution.id))
                   : nullptr;
      if (!work || !piece || !node || (topLevel && !scope) ||
          work->id.tile != group.tile)
        return fail<std::vector<StructuredNodeShardGroup>>(
            failureReason,
            "selected region execution has no matching work, node, or "
            "temporal scope");
      auto selectedNode = selectedNodes.find(execution.id);
      const uint32_t structuredNodeId =
          selectedNode == selectedNodes.end()
              ? node->structuredNodeId
              : selectedNode->second->structuredNodeId;
      StructuredNodeIterationShard shard;
      shard.structuredNodeId = structuredNodeId;
      shard.tile = group.tile;
      for (const IteratorInterval &interval : piece->iterationDomain) {
        shard.offsets.push_back(interval.offset);
        shard.sizes.push_back(interval.size);
      }
      for (const analysis::RootContributionWork &contribution :
           work->contributions)
        if (contribution.contribution.shard == root->shard)
          shard.reductionGroups.push_back(
              {contribution.group, contribution.mergeTile});
      selected.shards.push_back(std::move(shard));
      if (scope) {
        if (!seenTemporalScopes.insert(scope->id).second)
          return fail<std::vector<StructuredNodeShardGroup>>(
              failureReason, "selected top-level temporal scope is duplicated");
        selected.temporalTiles.push_back(StructuredNodeTemporalTile{
            structuredNodeId, scope->iteratorTileSizes, scope->waveLoopOrder});
        temporalExecutions.insert(RegionExecutionId(execution.id));
      }
      if (!executionsByNode.try_emplace(structuredNodeId, &execution).second ||
          !nodesByExecution.try_emplace(execution.id, structuredNodeId)
               .second ||
          !nodesByRegionExecution
               .try_emplace(RegionExecutionId(execution.id), structuredNodeId)
               .second)
        return fail<std::vector<StructuredNodeShardGroup>>(
            failureReason,
            "selected region maps one source node to several required "
            "executions in one group");
      if (topLevel &&
          !llvm::is_contained(selected.independentlyMaterializedNodes,
                              structuredNodeId))
        selected.independentlyMaterializedNodes.push_back(structuredNodeId);
    }

    for (const ReplicaExecutionPlan &replica : group.replicas) {
      if (!seenReplicas.insert(replica.id).second)
        return fail<std::vector<StructuredNodeShardGroup>>(
            failureReason, "selected region repeats one replica execution");
      const analysis::RootRegionWork *work =
          findWork(rootWorks, replica.id.producer.work);
      const analysis::RootExecutionWork *piece =
          work ? findExecution(*work, replica.id.producer.shard) : nullptr;
      auto node = selectedNodes.find(replica.id);
      const bool topLevel =
          std::holds_alternative<ExecutionInstancePlan::TopLevel>(
              replica.placement);
      const TemporalScopePlan *scope =
          topLevel ? findTopLevelTemporalScope(temporal,
                                               RegionExecutionId(replica.id))
                   : nullptr;
      if (!work || !piece || node == selectedNodes.end() ||
          (topLevel && !scope) ||
          llvm::any_of(selected.shards, [&](const auto &shard) {
            return shard.structuredNodeId == node->second->structuredNodeId;
          }))
        return fail<std::vector<StructuredNodeShardGroup>>(
            failureReason,
            "selected replica has no distinct candidate-local execution "
            "node or temporal scope");
      StructuredNodeIterationShard replicaShard;
      replicaShard.structuredNodeId = node->second->structuredNodeId;
      replicaShard.tile = group.tile;
      for (const IteratorInterval &interval : piece->iterationDomain) {
        replicaShard.offsets.push_back(interval.offset);
        replicaShard.sizes.push_back(interval.size);
      }
      selected.shards.push_back(std::move(replicaShard));
      if (scope) {
        if (!seenTemporalScopes.insert(scope->id).second)
          return fail<std::vector<StructuredNodeShardGroup>>(
              failureReason, "selected replica temporal scope is duplicated");
        selected.temporalTiles.push_back(StructuredNodeTemporalTile{
            node->second->structuredNodeId, scope->iteratorTileSizes,
            scope->waveLoopOrder});
        temporalExecutions.insert(RegionExecutionId(replica.id));
      }
      if (!nodesByReplica
               .try_emplace(replica.id, node->second->structuredNodeId)
               .second ||
          !nodesByRegionExecution
               .try_emplace(RegionExecutionId(replica.id),
                            node->second->structuredNodeId)
               .second)
        return fail<std::vector<StructuredNodeShardGroup>>(
            failureReason,
            "selected replica maps to a duplicate execution node");
      if (topLevel)
        selected.independentlyMaterializedNodes.push_back(
            node->second->structuredNodeId);
    }

    for (const LocalUseBinding &binding : group.localBindings) {
      if (const auto *required =
              std::get_if<ExecutionInstanceId>(&binding.producer)) {
        auto node = nodesByExecution.find(*required);
        if (node == nodesByExecution.end())
          return fail<std::vector<StructuredNodeShardGroup>>(
              failureReason, "local required binding has no group execution");
        auto executionPosition = executionsByNode.find(node->second);
        const ExecutionInstancePlan *execution =
            executionPosition == executionsByNode.end()
                ? nullptr
                : executionPosition->second;
        const bool topLevel =
            execution &&
            std::holds_alternative<ExecutionInstancePlan::TopLevel>(
                execution->placement);
        const bool stored =
            binding.delivery == LocalUseDelivery::StoredRegionValue;
        if (stored != topLevel)
          return fail<std::vector<StructuredNodeShardGroup>>(
              failureReason,
              "local required delivery disagrees with execution placement");
        if (stored &&
            !llvm::is_contained(selected.independentlyMaterializedNodes,
                                node->second))
          selected.independentlyMaterializedNodes.push_back(node->second);
      } else {
        const auto &replica = std::get<ReplicaExecutionId>(binding.producer);
        auto node = nodesByReplica.find(replica);
        auto replicaPlan = llvm::find_if(
            group.replicas, [&](const ReplicaExecutionPlan &candidate) {
              return candidate.id == replica;
            });
        if (node == nodesByReplica.end() || replicaPlan == group.replicas.end())
          return fail<std::vector<StructuredNodeShardGroup>>(
              failureReason, "local replica binding has no group execution");
        const bool topLevel =
            std::holds_alternative<ExecutionInstancePlan::TopLevel>(
                replicaPlan->placement);
        const bool stored =
            binding.delivery == LocalUseDelivery::StoredRegionValue;
        if (stored != topLevel)
          return fail<std::vector<StructuredNodeShardGroup>>(
              failureReason,
              "local replica delivery disagrees with execution placement");
        if (stored &&
            !llvm::is_contained(selected.independentlyMaterializedNodes,
                                node->second))
          selected.independentlyMaterializedNodes.push_back(node->second);
      }
    }

    for (const ExternalUseBinding &binding : group.externalBindings)
      if (isStructuredMember(group, binding.fragment))
        return fail<std::vector<StructuredNodeShardGroup>>(
            failureReason,
            "current selected construction requires same-group external "
            "delivery from the movement stage");

    for (const TemporalScopePlan &scope : temporal.scopes) {
      const auto *invocation =
          std::get_if<NestedInvocationClassId>(&scope.id.invocation);
      if (!invocation)
        continue;
      auto producer = nodesByRegionExecution.find(scope.id.execution);
      auto parent = nodesByRegionExecution.find(invocation->parent);
      if (producer == nodesByRegionExecution.end() ||
          parent == nodesByRegionExecution.end())
        continue;
      if (!seenTemporalScopes.insert(scope.id).second)
        return fail<std::vector<StructuredNodeShardGroup>>(
            failureReason, "selected nested temporal scope is duplicated");
      temporalExecutions.insert(scope.id.execution);
      if (scope.iteratorTileSizes.size() !=
              invocation->producerExtents.size() ||
          invocation->producerOffsets.size() !=
              invocation->producerExtents.size())
        return fail<std::vector<StructuredNodeShardGroup>>(
            failureReason,
            "selected nested temporal scope has inconsistent iterator work");
      for (const NestedUseClassId &use : invocation->uses) {
        auto binding = llvm::find_if(
            group.localBindings, [&](const LocalUseBinding &candidate) {
              return candidate.fragment == use.relation &&
                     candidate.producer == scope.id.execution &&
                     candidate.delivery == LocalUseDelivery::DirectNestedValue;
            });
        if (binding == group.localBindings.end() ||
            use.requestedOffsets.size() != use.requestedExtents.size())
          return fail<std::vector<StructuredNodeShardGroup>>(
              failureReason,
              "nested temporal class has no exact direct use binding");
        selected.nestedTemporalTiles.push_back(
            {producer->second, parent->second, use.relation.source.index,
             use.relation.use.operand, use.requestedExtents,
             invocation->producerExtents, scope.iteratorTileSizes,
             scope.waveLoopOrder});
      }
    }

    for (const ExecutionInstancePlan &execution : group.executions)
      if (std::holds_alternative<RequiredRootExecution>(execution.id.source) &&
          !temporalExecutions.count(RegionExecutionId(execution.id)))
        return fail<std::vector<StructuredNodeShardGroup>>(
            failureReason,
            "selected root execution has no temporal traversal scope");
    for (const ReplicaExecutionPlan &replica : group.replicas)
      if (!temporalExecutions.count(RegionExecutionId(replica.id)))
        return fail<std::vector<StructuredNodeShardGroup>>(
            failureReason,
            "selected replica execution has no temporal traversal scope");

    llvm::sort(selected.shards, [](const auto &lhs, const auto &rhs) {
      return lhs.structuredNodeId < rhs.structuredNodeId;
    });
    llvm::sort(selected.temporalTiles, [](const auto &lhs, const auto &rhs) {
      return lhs.structuredNodeId < rhs.structuredNodeId;
    });
    llvm::sort(selected.nestedTemporalTiles, [](const auto &lhs,
                                                const auto &rhs) {
      return std::tie(lhs.producerNodeId, lhs.parentNodeId, lhs.producerResult,
                      lhs.parentOperand, lhs.requestedResultExtents,
                      lhs.producerIterationExtents, lhs.iteratorTileSizes,
                      lhs.waveLoopOrder) <
             std::tie(rhs.producerNodeId, rhs.parentNodeId, rhs.producerResult,
                      rhs.parentOperand, rhs.requestedResultExtents,
                      rhs.producerIterationExtents, rhs.iteratorTileSizes,
                      rhs.waveLoopOrder);
    });
    llvm::sort(selected.independentlyMaterializedNodes);
    if (!selected.shards.empty())
      result.push_back(std::move(selected));
  }
  if (seenTemporalScopes.size() != temporal.scopes.size())
    return fail<std::vector<StructuredNodeShardGroup>>(
        failureReason,
        "selected temporal plan contains a scope outside its RegionPlan");
  return result;
}

mlir::LogicalResult applySelectedRegionRepresentations(
    const RegionPlan &regions,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    const RepresentationPlan &representations,
    llvm::ArrayRef<SelectedRegionExecutionNode> executionNodes,
    std::vector<StructuredNodeShardGroup> &groups, std::string *failureReason) {
  auto groupContainsExecution = [](const RegionGroupPlan &group,
                                   const RegionExecutionId &execution) {
    if (const auto *required = std::get_if<ExecutionInstanceId>(&execution))
      return llvm::any_of(group.executions, [&](const auto &candidate) {
        return candidate.id == *required;
      });
    const auto &replica = std::get<ReplicaExecutionId>(execution);
    return llvm::any_of(group.replicas, [&](const auto &candidate) {
      return candidate.id == replica;
    });
  };

  auto getUseFragment = [](const RepresentationUseId &use)
      -> std::pair<analysis::RootRegionWorkId, const DemandFragmentId *> {
    if (const auto *boundary = std::get_if<BoundaryRepresentationUseId>(&use))
      return {boundary->value.work, &boundary->value.fragment};
    const auto &local = std::get<LocalRepresentationUseId>(use);
    return {local.consumerWork, &local.fragment};
  };

  for (StructuredNodeShardGroup &group : groups) {
    std::vector<uint32_t> actualNodes;
    for (const StructuredNodeIterationShard &shard : group.shards)
      actualNodes.push_back(shard.structuredNodeId);
    llvm::sort(actualNodes);
    const RegionGroupPlan *regionGroup = nullptr;
    for (const RegionGroupPlan &candidate : regions.groups) {
      if (candidate.tile != group.shards.front().tile)
        continue;
      std::vector<uint32_t> candidateNodes;
      for (const SelectedRegionExecutionNode &execution : executionNodes)
        if (groupContainsExecution(candidate, execution.execution))
          candidateNodes.push_back(execution.structuredNodeId);
      llvm::sort(candidateNodes);
      candidateNodes.erase(
          std::unique(candidateNodes.begin(), candidateNodes.end()),
          candidateNodes.end());
      if (candidateNodes != actualNodes)
        continue;
      if (regionGroup) {
        if (failureReason)
          *failureReason =
              "selected representation group matches several RegionPlan groups";
        return mlir::failure();
      }
      regionGroup = &candidate;
    }
    if (!regionGroup) {
      if (failureReason)
        *failureReason =
            "selected representation group has no RegionPlan owner";
      return mlir::failure();
    }
    group.representations.clear();
    for (const StructuredNodeIterationShard &shard : group.shards) {
      const SelectedRegionExecutionNode *execution = nullptr;
      for (const SelectedRegionExecutionNode &candidate : executionNodes)
        if (candidate.structuredNodeId == shard.structuredNodeId &&
            groupContainsExecution(*regionGroup, candidate.execution)) {
          if (execution) {
            if (failureReason)
              *failureReason =
                  "selected node has several execution identities in one group";
            return mlir::failure();
          }
          execution = &candidate;
        }
      std::optional<analysis::RootRegionWorkId> workId =
          execution ? getExecutionWork(execution->execution) : std::nullopt;
      const LogicalShardId *executionShard =
          execution ? getExecutionShard(execution->execution) : nullptr;
      const analysis::RootRegionWork *work =
          workId ? findWork(rootWorks, *workId) : nullptr;
      mlir::Operation *operation = work ? work->rootOperation : nullptr;
      if (!operation || !executionShard) {
        if (failureReason)
          *failureReason =
              "selected representation has no execution/root operation";
        return mlir::failure();
      }

      StructuredNodePhysicalRepresentation selected;
      selected.structuredNodeId = shard.structuredNodeId;
      for (auto [operandNumber, operand] :
           llvm::enumerate(operation->getOperands())) {
        const bool ranked =
            mlir::isa<mlir::RankedTensorType>(operand.getType());
        const bool used = !mlir::isa<mlir::linalg::LinalgOp>(operation) ||
                          mlir::cast<mlir::linalg::LinalgOp>(operation)
                              .payloadUsesValueFromOperand(
                                  &operation->getOpOperand(operandNumber));
        if (!ranked || !used) {
          selected.operandLayouts.push_back(std::nullopt);
          selected.sharedOperands.push_back(0);
          continue;
        }
        std::optional<MemLayout> layout;
        std::optional<bool> shared;
        for (const PhysicalUseBinding &binding : representations.uses) {
          auto [consumerWork, fragment] = getUseFragment(binding.use);
          if (consumerWork != *workId || !fragment ||
              fragment->use.destinationShard != *executionShard ||
              fragment->use.operand != operandNumber)
            continue;
          const PhysicalVersionPlan *version =
              findPhysicalVersion(representations, binding.version);
          const bool bindingShared =
              !binding.version.derivation.empty() &&
              std::holds_alternative<SharedRepresentationAnchor>(
                  binding.version.derivation.back().anchor);
          if (!version || (layout && *layout != version->encoding) ||
              (shared && *shared != bindingShared)) {
            if (failureReason)
              *failureReason =
                  "selected operand fragments disagree on physical layout";
            return mlir::failure();
          }
          layout = version->encoding;
          shared = bindingShared;
        }
        if (!layout) {
          if (failureReason)
            *failureReason =
                "selected shaped operand has no physical use binding";
          return mlir::failure();
        }
        selected.operandLayouts.push_back(*layout);
        selected.sharedOperands.push_back(shared.value_or(false) ? 1 : 0);
      }

      for (unsigned result = 0; result < operation->getNumResults(); ++result) {
        if (!mlir::isa<mlir::RankedTensorType>(
                operation->getResult(result).getType())) {
          selected.resultLayouts.push_back(std::nullopt);
          continue;
        }
        const PhysicalVersionPlan *version = nullptr;
        if (!shard.reductionGroups.empty()) {
          RegionValueVersionId logical = ReductionPartialValueId{
              std::get<ExecutionInstanceId>(execution->execution),
              shard.reductionGroups.front().group, result};
          version = findPrimaryVersion(representations, logical);
        } else {
          RegionValueVersionId logical =
              ExecutionResultValueId{execution->execution, result};
          version = findPrimaryVersion(representations, logical);
        }
        if (!version) {
          if (failureReason)
            *failureReason =
                "selected shaped result has no primary physical version";
          return mlir::failure();
        }
        selected.resultLayouts.push_back(version->encoding);
      }
      group.representations.push_back(std::move(selected));
    }
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
