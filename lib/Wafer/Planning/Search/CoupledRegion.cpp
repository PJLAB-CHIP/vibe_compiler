//===- CoupledRegion.cpp - Connected node-shard partitions ------------===//

#include "Wafer/Planning/Search/CoupledRegion.h"

#include "Wafer/Planning/Search/DataMovement.h"
#include "Wafer/Planning/Search/DataMovementApply.h"
#include "Wafer/Planning/Search/PhysicalRepresentation.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Verifier.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

using TileDomain = CoupledRegionDomain::TileDomain;

const analysis::LogicalNodeTrial *
findNodeTrial(const analysis::LogicalShardTrial &trial,
              StructuredDAGNodeID node) {
  auto found =
      llvm::find_if(trial.nodes, [&](const analysis::LogicalNodeTrial &entry) {
        return entry.node == node;
      });
  return found == trial.nodes.end() ? nullptr : &*found;
}

bool retreatRestrictedGrowth(llvm::MutableArrayRef<uint32_t> labels) {
  if (labels.size() < 2)
    return false;
  for (size_t reverse = 0; reverse + 1 < labels.size(); ++reverse) {
    const size_t index = labels.size() - reverse - 1;
    if (labels[index] == 0)
      continue;
    --labels[index];
    uint32_t maximumPrefix = 0;
    for (uint32_t label : labels.take_front(index + 1))
      maximumPrefix = std::max(maximumPrefix, label);
    for (size_t suffix = index + 1; suffix < labels.size(); ++suffix)
      labels[suffix] = ++maximumPrefix;
    return true;
  }
  return false;
}

bool isCanonicalRestrictedGrowth(llvm::ArrayRef<uint32_t> labels) {
  if (labels.empty())
    return true;
  if (labels.front() != 0)
    return false;
  uint32_t maximum = 0;
  for (uint32_t label : llvm::drop_begin(labels)) {
    if (label > maximum + 1)
      return false;
    maximum = std::max(maximum, label);
  }
  return true;
}

bool isLegalPartition(const TileDomain &domain,
                      llvm::ArrayRef<uint32_t> labels) {
  const size_t count = domain.nodes.size();
  if (labels.size() != count || !isCanonicalRestrictedGrowth(labels) ||
      domain.fusableEdges.size() != count * count ||
      domain.forbiddenInternalEdges.size() != count * count)
    return false;
  const uint32_t groupCount =
      labels.empty() ? 0 : *llvm::max_element(labels) + 1;
  for (uint32_t group = 0; group < groupCount; ++group) {
    llvm::SmallVector<size_t, 16> members;
    for (auto [index, label] : llvm::enumerate(labels))
      if (label == group)
        members.push_back(index);
    if (members.empty())
      return false;
    for (size_t lhs : members)
      for (size_t rhs : members)
        if (domain.forbiddenInternalEdges[lhs * count + rhs])
          return false;
    if (members.size() == 1)
      continue;

    llvm::DenseSet<size_t> reached;
    llvm::SmallVector<size_t, 16> worklist{members.front()};
    reached.insert(members.front());
    while (!worklist.empty()) {
      const size_t current = worklist.pop_back_val();
      for (size_t candidate : members) {
        if (reached.contains(candidate))
          continue;
        if (!domain.fusableEdges[current * count + candidate] &&
            !domain.fusableEdges[candidate * count + current])
          continue;
        reached.insert(candidate);
        worklist.push_back(candidate);
      }
    }
    if (reached.size() != members.size())
      return false;
  }
  return true;
}

llvm::SmallVector<uint32_t, 16> getFirstLabels(const TileDomain &domain) {
  llvm::SmallVector<uint32_t, 16> labels;
  labels.reserve(domain.nodes.size());
  for (size_t index = 0; index < domain.nodes.size(); ++index)
    labels.push_back(static_cast<uint32_t>(index));
  assert(isLegalPartition(domain, labels) &&
         "all-singleton coupled partition must be legal");
  return labels;
}

std::optional<llvm::SmallVector<uint32_t, 16>>
getNextLabels(const TileDomain &domain, llvm::ArrayRef<uint32_t> current) {
  llvm::SmallVector<uint32_t, 16> labels(current.begin(), current.end());
  while (retreatRestrictedGrowth(labels))
    if (isLegalPartition(domain, labels))
      return labels;
  return std::nullopt;
}

void appendGroups(const TileDomain &domain, llvm::ArrayRef<uint32_t> labels,
                  CoupledRegionAssignment &assignment) {
  const uint32_t groupCount =
      labels.empty() ? 0 : *llvm::max_element(labels) + 1;
  for (uint32_t group = 0; group < groupCount; ++group) {
    CoupledRegionGroup entry;
    entry.tile = domain.tile;
    for (auto [node, label] : llvm::zip_equal(domain.nodes, labels))
      if (label == group)
        entry.nodes.push_back(node);
    assignment.groups.push_back(std::move(entry));
  }
}

std::optional<llvm::SmallVector<uint32_t, 16>>
getLabels(const TileDomain &domain, const CoupledRegionAssignment &assignment) {
  llvm::SmallVector<const CoupledRegionGroup *, 16> groups;
  for (const CoupledRegionGroup &group : assignment.groups) {
    if (group.tile != domain.tile)
      continue;
    const bool intersects = llvm::any_of(group.nodes, [&](uint32_t node) {
      return llvm::is_contained(domain.nodes, node);
    });
    if (!intersects)
      continue;
    if (llvm::any_of(group.nodes, [&](uint32_t node) {
          return !llvm::is_contained(domain.nodes, node);
        }))
      return std::nullopt;
    groups.push_back(&group);
  }
  llvm::SmallVector<uint32_t, 16> labels(domain.nodes.size(), 0);
  llvm::SmallVector<uint8_t, 16> seen(domain.nodes.size(), 0);
  for (auto [groupIndex, group] : llvm::enumerate(groups)) {
    if (group->nodes.empty() || !llvm::is_sorted(group->nodes))
      return std::nullopt;
    for (StructuredDAGNodeID node : group->nodes) {
      auto found = llvm::find(domain.nodes, node);
      if (found == domain.nodes.end())
        return std::nullopt;
      const size_t index = static_cast<size_t>(found - domain.nodes.begin());
      if (seen[index])
        return std::nullopt;
      seen[index] = 1;
      labels[index] = static_cast<uint32_t>(groupIndex);
    }
  }
  if (llvm::any_of(seen, [](uint8_t value) { return value == 0; }) ||
      !isCanonicalRestrictedGrowth(labels) || !isLegalPartition(domain, labels))
    return std::nullopt;
  return labels;
}

} // namespace

mlir::FailureOr<CoupledRegionDomain>
CoupledRegionDomain::create(const StructuredDAGAnalysis &dag,
                            const analysis::LogicalShardTrial &trial,
                            std::string *failureReason) {
  auto fail =
      [&](llvm::StringRef message) -> mlir::FailureOr<CoupledRegionDomain> {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  };
  if (!trial.epoch.isValid() || trial.nodes.size() != dag.getNodes().size())
    return fail("coupled-region domain requires one closed logical trial");

  std::map<int64_t, TileDomain> byTile;
  for (const StructuredDAGNode &node : dag.getNodes()) {
    const analysis::LogicalNodeTrial *nodeTrial = findNodeTrial(trial, node.id);
    if (!nodeTrial || nodeTrial->executionShards.empty())
      return fail("coupled-region trial omitted one structured node");
    llvm::DenseSet<int64_t> seenTiles;
    for (const analysis::LogicalExecutionShard &shard :
         nodeTrial->executionShards) {
      if (!shard.executionDomain ||
          !seenTiles.insert(shard.tile.getValue()).second)
        return fail("coupled-region trial has a malformed node shard");
      TileDomain &tile = byTile[shard.tile.getValue()];
      tile.tile = shard.tile;
      tile.nodes.push_back(node.id);
    }
  }

  llvm::SmallVector<TileDomain, 16> wholeTiles;
  for (auto &[tileId, domain] : byTile) {
    (void)tileId;
    llvm::sort(domain.nodes);
    const size_t count = domain.nodes.size();
    domain.fusableEdges.assign(count * count, 0);
    domain.forbiddenInternalEdges.assign(count * count, 0);
    wholeTiles.push_back(std::move(domain));
  }

  StructuredDAGExactDemandQuery query(dag, trial.epoch);
  for (const StructuredDAGEdge &edge : dag.getEdges()) {
    analysis::ExactDemandResult demand = query.query(edge.id, trial);
    if (demand.status != analysis::ExactDemandStatus::Satisfied)
      return fail("coupled-region input trial is not exact-demand satisfied");
    const StructuredDAGNode *producer = dag.getNode(edge.producer);
    const StructuredDAGNode *consumer = dag.getNode(edge.consumer);
    if (!producer || !consumer || !producer->operation || !consumer->operation)
      return fail("coupled-region edge references an unknown node");

    std::string relationFailure;
    const bool supportedChain = mlir::succeeded(traceProducerToConsumerChain(
        producer->operation, edge.producerResult, consumer->operation,
        edge.consumerOperand, &relationFailure));
    const analysis::LogicalNodeTrial *producerTrial =
        findNodeTrial(trial, edge.producer);
    const analysis::LogicalNodeTrial *consumerTrial =
        findNodeTrial(trial, edge.consumer);
    if (!producerTrial || !consumerTrial)
      return fail("coupled-region edge lost its endpoint trial");

    for (TileDomain &tile : wholeTiles) {
      auto producerPosition = llvm::find(tile.nodes, edge.producer);
      auto consumerPosition = llvm::find(tile.nodes, edge.consumer);
      if (producerPosition == tile.nodes.end() ||
          consumerPosition == tile.nodes.end())
        continue;
      auto destination =
          llvm::find_if(demand.perDestination,
                        [&](const analysis::ExactDestinationDemand &entry) {
                          return entry.destinationTile == tile.tile;
                        });
      if (destination == demand.perDestination.end() ||
          !destination->producerDemand)
        return fail("coupled-region edge omitted one destination demand");
      if (destination->producerDemand->isIntegerEmpty())
        continue;

      bool hasOwner = false;
      bool localOnly = true;
      for (const analysis::ExactOwnershipIntersection &intersection :
           destination->ownershipIntersections) {
        if (!intersection.set)
          return fail("coupled-region ownership intersection is unavailable");
        if (intersection.set->isIntegerEmpty())
          continue;
        hasOwner = true;
        localOnly &= intersection.tile == tile.tile;
      }
      const size_t producerIndex =
          static_cast<size_t>(producerPosition - tile.nodes.begin());
      const size_t consumerIndex =
          static_cast<size_t>(consumerPosition - tile.nodes.begin());
      const size_t index = producerIndex * tile.nodes.size() + consumerIndex;
      const bool partialEndpoint = producerTrial->reductionMergeTile ||
                                   consumerTrial->reductionMergeTile;
      if (supportedChain && hasOwner && localOnly && !partialEndpoint)
        tile.fusableEdges[index] = 1;
      else
        tile.forbiddenInternalEdges[index] = 1;
    }
  }
  llvm::SmallVector<TileDomain, 16> components;
  for (const TileDomain &whole : wholeTiles) {
    const size_t wholeCount = whole.nodes.size();
    llvm::SmallVector<uint8_t, 16> assigned(wholeCount, 0);
    for (size_t start = 0; start < wholeCount; ++start) {
      if (assigned[start])
        continue;
      llvm::SmallVector<size_t, 16> indices{start};
      llvm::SmallVector<size_t, 16> worklist{start};
      assigned[start] = 1;
      while (!worklist.empty()) {
        size_t current = worklist.pop_back_val();
        for (size_t candidate = 0; candidate < wholeCount; ++candidate) {
          if (assigned[candidate] ||
              (!whole.fusableEdges[current * wholeCount + candidate] &&
               !whole.fusableEdges[candidate * wholeCount + current]))
            continue;
          assigned[candidate] = 1;
          indices.push_back(candidate);
          worklist.push_back(candidate);
        }
      }
      llvm::sort(indices);
      TileDomain component;
      component.tile = whole.tile;
      for (size_t index : indices)
        component.nodes.push_back(whole.nodes[index]);
      const size_t count = indices.size();
      component.fusableEdges.assign(count * count, 0);
      component.forbiddenInternalEdges.assign(count * count, 0);
      for (size_t lhs = 0; lhs < count; ++lhs)
        for (size_t rhs = 0; rhs < count; ++rhs) {
          component.fusableEdges[lhs * count + rhs] =
              whole.fusableEdges[indices[lhs] * wholeCount + indices[rhs]];
          component.forbiddenInternalEdges[lhs * count + rhs] =
              whole.forbiddenInternalEdges[indices[lhs] * wholeCount +
                                           indices[rhs]];
        }
      components.push_back(std::move(component));
    }
  }
  llvm::sort(components, [](const TileDomain &lhs, const TileDomain &rhs) {
    return std::tuple(lhs.tile.getValue(), lhs.nodes.front()) <
           std::tuple(rhs.tile.getValue(), rhs.nodes.front());
  });
  return CoupledRegionDomain(std::move(components));
}

CoupledRegionAssignment CoupledRegionDomain::getFirstAssignment() const {
  CoupledRegionAssignment assignment;
  for (const TileDomain &tile : tiles)
    appendGroups(tile, getFirstLabels(tile), assignment);
  return assignment;
}

mlir::FailureOr<std::optional<CoupledRegionAssignment>>
CoupledRegionDomain::getNextAssignment(
    const CoupledRegionAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  llvm::SmallVector<llvm::SmallVector<uint32_t, 16>, 16> labels;
  for (const TileDomain &tile : tiles)
    labels.push_back(*getLabels(tile, assignment));
  for (size_t reverse = 0; reverse < tiles.size(); ++reverse) {
    const size_t index = tiles.size() - reverse - 1;
    std::optional<llvm::SmallVector<uint32_t, 16>> next =
        getNextLabels(tiles[index], labels[index]);
    if (!next)
      continue;
    labels[index] = std::move(*next);
    for (size_t reset = index + 1; reset < tiles.size(); ++reset)
      labels[reset] = getFirstLabels(tiles[reset]);
    CoupledRegionAssignment result;
    for (size_t tile = 0; tile < tiles.size(); ++tile)
      appendGroups(tiles[tile], labels[tile], result);
    return std::optional<CoupledRegionAssignment>(std::move(result));
  }
  return std::optional<CoupledRegionAssignment>{};
}

bool CoupledRegionDomain::contains(
    const CoupledRegionAssignment &assignment) const {
  if (!llvm::is_sorted(assignment.groups))
    return false;
  size_t expectedGroups = 0;
  for (const TileDomain &tile : tiles) {
    std::optional<llvm::SmallVector<uint32_t, 16>> labels =
        getLabels(tile, assignment);
    if (!labels)
      return false;
    expectedGroups += *llvm::max_element(*labels) + 1;
  }
  return assignment.groups.size() == expectedGroups;
}

mlir::FailureOr<CardCoupledRegionMaterialization> materializeCardCoupledRegions(
    mlir::ModuleOp tensorProgram, const CardProgramAnalysis &program,
    CardId cardId, const analysis::LogicalShardTrial &trial,
    const CoupledRegionDomain &domain,
    const CoupledRegionAssignment &assignment,
    const CardTemporalDomain &temporalDomain,
    const CardTemporalAssignment &temporalAssignment,
    const CardPhysicalRepresentationDomain &representationDomain,
    const CardPhysicalRepresentationAssignment &representationAssignment,
    const CardDataMovementDomain &movementDomain,
    const CardDataMovementAssignment &movementAssignment,
    std::string *failureReason) {
  auto fail = [&](llvm::StringRef message)
      -> mlir::FailureOr<CardCoupledRegionMaterialization> {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  };
  if (!tensorProgram || trial.epoch != program.epoch ||
      !domain.contains(assignment) ||
      !temporalDomain.contains(temporalAssignment) ||
      !representationDomain.contains(representationAssignment) ||
      !movementDomain.contains(movementAssignment))
    return fail("coupled-region apply received a stale or unknown assignment");

  for (const DataMovementChoice &movement : movementAssignment.edges) {
    const StructuredDAGEdge *edge = program.dag.getEdge(movement.edge);
    auto producerGroup =
        edge ? llvm::find_if(assignment.groups,
                             [&](const auto &group) {
                               return group.tile == movement.destinationTile &&
                                      llvm::is_contained(group.nodes,
                                                         edge->producer);
                             })
             : assignment.groups.end();
    auto consumerGroup =
        edge ? llvm::find_if(assignment.groups,
                             [&](const auto &group) {
                               return group.tile == movement.destinationTile &&
                                      llvm::is_contained(group.nodes,
                                                         edge->consumer);
                             })
             : assignment.groups.end();
    const bool sameGroup = producerGroup != assignment.groups.end() &&
                           consumerGroup != assignment.groups.end() &&
                           producerGroup == consumerGroup;
    if (((movement.kind == DataMovementKind::Retained ||
          movement.kind == DataMovementKind::Refetch) &&
         !sameGroup) ||
        (movement.kind == DataMovementKind::DDR && sameGroup))
      return fail("movement choice disagrees with selected group boundary");
    if (movement.kind != DataMovementKind::Retained &&
        movement.kind != DataMovementKind::Refetch &&
        movement.kind != DataMovementKind::DDR &&
        movement.kind != DataMovementKind::Recompute &&
        movement.kind != DataMovementKind::Peer)
      return fail("selected movement kind has no current actual apply");
  }

  llvm::SmallVector<StructuredNodeShardGroup, 32> groups;
  for (const CoupledRegionGroup &selected : assignment.groups) {
    StructuredNodeShardGroup group;
    for (StructuredDAGNodeID node : selected.nodes) {
      const analysis::LogicalNodeTrial *nodeTrial = findNodeTrial(trial, node);
      if (!nodeTrial)
        return fail("coupled-region apply lost one node trial");
      auto execution =
          llvm::find_if(nodeTrial->executionShards,
                        [&](const analysis::LogicalExecutionShard &shard) {
                          return shard.tile == selected.tile;
                        });
      if (execution == nodeTrial->executionShards.end() ||
          !execution->executionDomain)
        return fail("coupled-region apply lost one selected node shard");
      analysis::StaticRectangularIndexSetResult rectangle =
          analysis::IndexSetResult{analysis::IndexRelationStatus::Exact,
                                   *execution->executionDomain,
                                   {}}
              .getExactStaticRectangularDomain();
      if (!rectangle.isExact() || !rectangle.domain)
        return fail("coupled-region shard is not one exact rectangle");
      const bool partial = llvm::any_of(
          nodeTrial->bindings, [](const analysis::LogicalTileBinding &binding) {
            return binding.role ==
                   analysis::TileRole::PartialReductionContribution;
          });
      group.shards.push_back(StructuredNodeIterationShard{
          node, selected.tile, std::move(rectangle.domain->offsets),
          std::move(rectangle.domain->sizes),
          partial
              ? StructuredNodeIterationShardRole::PartialReductionContribution
              : StructuredNodeIterationShardRole::Complete,
          nodeTrial->reductionMergeTile});
      auto temporal =
          llvm::find_if(temporalAssignment.nodes,
                        [&](const TemporalNodeAssignment &candidate) {
                          return candidate.node == node;
                        });
      if (temporal == temporalAssignment.nodes.end())
        return fail("coupled-region apply lost one temporal assignment");
      group.temporalTiles.push_back(StructuredNodeTemporalTile{
          node, temporal->iteratorTileSizes, temporal->waveLoopOrder});

      mlir::Operation *operation = program.dag.getNode(node)->operation;
      StructuredNodePhysicalRepresentation representation;
      representation.structuredNodeId = node;
      representation.operandLayouts.resize(operation->getNumOperands());
      representation.resultLayouts.resize(operation->getNumResults());
      for (const PhysicalRepresentationChoice &choice :
           representationAssignment.values) {
        if (choice.tile != selected.tile || choice.node != node)
          continue;
        if (choice.role == PhysicalValueRole::Operand) {
          if (choice.index >= representation.operandLayouts.size())
            return fail("physical operand choice is outside node arity");
          representation.operandLayouts[choice.index] = choice.layout;
        } else {
          if (choice.index >= representation.resultLayouts.size())
            return fail("physical result choice is outside node arity");
          representation.resultLayouts[choice.index] = choice.layout;
        }
      }
      for (auto [operandNumber, layout] :
           llvm::enumerate(representation.operandLayouts)) {
        mlir::OpOperand &operand = operation->getOpOperand(operandNumber);
        if ((mlir::isa<mlir::RankedTensorType>(operand.get().getType()) &&
             (!mlir::isa<mlir::linalg::LinalgOp>(operation) ||
              mlir::cast<mlir::linalg::LinalgOp>(operation)
                  .payloadUsesValueFromOperand(&operand))) !=
            layout.has_value())
          return fail("physical operand assignment is incomplete");
      }
      for (auto [result, layout] : llvm::zip_equal(
               operation->getResults(), representation.resultLayouts))
        if (mlir::isa<mlir::RankedTensorType>(result.getType()) !=
            layout.has_value())
          return fail("physical result assignment is incomplete");
      group.representations.push_back(std::move(representation));
    }
    groups.push_back(std::move(group));
  }

  for (const DataMovementChoice &movement : movementAssignment.edges) {
    if (movement.kind != DataMovementKind::Recompute)
      continue;
    const StructuredDAGEdge *edge = program.dag.getEdge(movement.edge);
    auto destination =
        edge ? llvm::find_if(
                   groups,
                   [&](const auto &group) {
                     return group.shards.front().tile ==
                                movement.destinationTile &&
                            llvm::any_of(group.shards, [&](const auto &shard) {
                              return shard.structuredNodeId == edge->consumer;
                            });
                   })
             : groups.end();
    if (!edge || destination == groups.end() ||
        llvm::any_of(destination->shards, [&](const auto &shard) {
          return shard.structuredNodeId == edge->producer;
        }))
      return fail("recompute movement has no separate consumer group");
    if (!llvm::is_contained(destination->recomputedProducerNodes,
                            edge->producer))
      destination->recomputedProducerNodes.push_back(edge->producer);
    llvm::sort(destination->recomputedProducerNodes);
  }

  CardCoupledRegionMaterialization result;
  if (mlir::failed(lowerStructuredNodeGroupsToCardModule(
          tensorProgram, cardId, program.availableTileIds,
          program.operationNodes, groups, result.module, &result.relations,
          failureReason)))
    return mlir::failure();
  if (mlir::failed(
          applySelectedDataMovement(*result.module, program, movementAssignment,
                                    result.relations, failureReason)) ||
      mlir::failed(mlir::verify(*result.module)))
    return mlir::failure();
  return result;
}

} // namespace wafer::compiler::detail
