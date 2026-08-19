//===- DataMovement.cpp - Explicit physical data movement -------------===//

#include "Wafer/Planning/Search/DataMovement.h"

#include "Wafer/Planning/Search/SimpleRoute.h"

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"
#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGEdgeDemandPlan.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>

namespace wafer::compiler::detail {
namespace {

using namespace simple_route;

uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  return product > std::numeric_limits<uint64_t>::max()
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(product);
}

const analysis::LogicalNodeTrial *
findNodeTrial(const analysis::LogicalShardTrial &trial,
              StructuredDAGNodeID node) {
  auto found = llvm::find_if(trial.nodes, [&](const auto &candidate) {
    return candidate.node == node;
  });
  return found == trial.nodes.end() ? nullptr : &*found;
}

const CoupledRegionGroup *findGroup(const CoupledRegionAssignment &assignment,
                                    StructuredDAGNodeID node, TileId tile) {
  auto found = llvm::find_if(assignment.groups, [&](const auto &group) {
    return group.tile == tile && llvm::is_contained(group.nodes, node);
  });
  return found == assignment.groups.end() ? nullptr : &*found;
}

std::optional<MemLayout>
findLayout(const CardPhysicalRepresentationAssignment &assignment, TileId tile,
           StructuredDAGNodeID node, PhysicalValueRole role, unsigned index) {
  auto found = llvm::find_if(assignment.values, [&](const auto &choice) {
    return choice.tile == tile && choice.node == node && choice.role == role &&
           choice.index == index;
  });
  return found == assignment.values.end()
             ? std::nullopt
             : std::optional<MemLayout>(found->layout);
}

llvm::SmallVector<TileId, 16>
getNodeTiles(const analysis::LogicalNodeTrial &trial) {
  llvm::SmallVector<TileId, 16> result;
  for (const analysis::LogicalExecutionShard &shard : trial.executionShards)
    result.push_back(shard.tile);
  llvm::sort(result, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  return result;
}

mlir::FailureOr<StructuredDAGEdgeDemandPlan>
getDemandPlan(const CardProgramAnalysis &program,
              const analysis::LogicalShardTrial &trial,
              std::string *failureReason) {
  StructuredDAGEdgeDemandPlan result;
  StructuredDAGExactDemandQuery query(program.dag, program.epoch);
  for (const StructuredDAGEdge &edge : program.dag.getEdges()) {
    const analysis::LogicalNodeTrial *producer =
        findNodeTrial(trial, edge.producer);
    const analysis::LogicalNodeTrial *consumer =
        findNodeTrial(trial, edge.consumer);
    if (!producer || !consumer)
      return mlir::failure();
    StructuredDAGNodePlacement producerPlacement{edge.producer,
                                                 {},
                                                 getNodeTiles(*producer),
                                                 producer->reductionMergeTile};
    StructuredDAGNodePlacement consumerPlacement{edge.consumer,
                                                 {},
                                                 getNodeTiles(*consumer),
                                                 consumer->reductionMergeTile};
    analysis::ExactDemandResult demand = query.query(edge.id, trial);
    if (demand.status == analysis::ExactDemandStatus::Satisfied) {
      if (mlir::failed(assembleStructuredDAGEdgeDemandPlan(
              program.dag, edge, producerPlacement, consumerPlacement, trial,
              demand, &result, failureReason)))
        return mlir::failure();
      continue;
    }
    if (demand.status != analysis::ExactDemandStatus::Satisfied &&
        !demand.perDestination.empty()) {
      if (failureReason)
        *failureReason = demand.detail;
      return mlir::failure();
    }
  }
  return result;
}

mlir::FailureOr<llvm::SmallVector<DataMovementFragment, 4>>
getFragments(const StructuredDAGEdgeDemand &demand,
             mlir::RankedTensorType tensorType, const StructuredDAGEdge &edge,
             MemLayout consumerLayout,
             const CardPhysicalRepresentationAssignment &representations) {
  llvm::SmallVector<DataMovementFragment, 4> result;
  for (const StructuredDAGEdgeProducerShardOwnership &ownership :
       demand.producerShardOwnership) {
    std::optional<MemLayout> layout =
        findLayout(representations, ownership.tile, edge.producer,
                   PhysicalValueRole::Result, edge.producerResult);
    if (!layout)
      return mlir::failure();
    analysis::IndexSetResult ownerSet{
        analysis::IndexRelationStatus::Exact, ownership.logicalDomain, {}};
    auto ownerRectangle = ownerSet.getExactStaticRectangularDomain();
    if (!ownerRectangle.isExact())
      return mlir::failure();
    mlir::presburger::PresburgerSet intersection =
        demand.producerDemand.intersect(ownership.logicalDomain);
    analysis::IndexSetResult exact{
        analysis::IndexRelationStatus::Exact, intersection, {}};
    auto rectangles = exact.getExactStaticRectangularDisjuncts();
    if (!rectangles.isExact())
      return mlir::failure();
    for (const analysis::StaticRectangularIndexSet &rectangle :
         rectangles.domains) {
      uint64_t elements = 1;
      for (int64_t size : rectangle.sizes)
        elements = saturatingMultiply(elements, size);
      unsigned elementBits = 0;
      if (auto integer =
              mlir::dyn_cast<mlir::IntegerType>(tensorType.getElementType()))
        elementBits = integer.getWidth();
      else if (auto floating =
                   mlir::dyn_cast<mlir::FloatType>(tensorType.getElementType()))
        elementBits = floating.getWidth();
      else
        return mlir::failure();
      uint64_t logicalBits = saturatingMultiply(elements, elementBits);
      auto type = mlir::MemRefType::get(
          rectangle.sizes, tensorType.getElementType(),
          mlir::MemRefLayoutAttrInterface{},
          MemoryAttr::get(tensorType.getContext(), MemorySpace::SPM,
                          consumerLayout));
      auto physical = analysis::PhysicalLayoutRelation::create(type);
      if (mlir::failed(physical))
        return mlir::failure();
      result.push_back(DataMovementFragment{
          ownership.tile,
          *layout,
          consumerLayout,
          tensorType.getElementType(),
          rectangle.offsets,
          rectangle.sizes,
          ownerRectangle.domain->offsets,
          ownerRectangle.domain->sizes,
          {},
          logicalBits / 8 + (logicalBits % 8 != 0),
          static_cast<uint64_t>(physical->getPhysicalFootprintBytes()),
          {}});
    }
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return std::tuple(lhs.sourceTile.getValue(), lhs.offsets, lhs.sizes) <
           std::tuple(rhs.sourceTile.getValue(), rhs.offsets, rhs.sizes);
  });
  if (!result.empty()) {
    llvm::SmallVector<int64_t, 4> base = result.front().offsets;
    for (const DataMovementFragment &fragment : result)
      for (auto [dimension, offset] : llvm::enumerate(fragment.offsets))
        base[dimension] = std::min(base[dimension], offset);
    for (DataMovementFragment &fragment : result)
      for (auto [offset, lower] : llvm::zip_equal(fragment.offsets, base))
        fragment.destinationOffsets.push_back(offset - lower);
  }
  return result;
}

} // namespace

mlir::FailureOr<CardDataMovementDomain> CardDataMovementDomain::create(
    const CardProgramAnalysis &program, CardId cardId,
    const analysis::LogicalShardTrial &trial,
    const CoupledRegionDomain &coupledDomain,
    const CoupledRegionAssignment &coupledAssignment,
    const CardTemporalDomain &temporalDomain,
    const CardTemporalAssignment &temporalAssignment,
    const CardPhysicalRepresentationDomain &representationDomain,
    const CardPhysicalRepresentationAssignment &representationAssignment,
    std::string *failureReason) {
  if (trial.epoch != program.epoch ||
      !coupledDomain.contains(coupledAssignment) ||
      !temporalDomain.contains(temporalAssignment) ||
      !representationDomain.contains(representationAssignment))
    return mlir::failure();
  auto plan = getDemandPlan(program, trial, failureReason);
  if (mlir::failed(plan))
    return mlir::failure();
  llvm::SmallVector<DemandDomain, 32> demands;
  for (const StructuredDAGEdgeDemand &demand : plan->demands) {
    const StructuredDAGEdge *edge = program.dag.getEdge(demand.edge);
    const StructuredDAGNode *producer =
        edge ? program.dag.getNode(edge->producer) : nullptr;
    if (!edge || !producer || !producer->operation)
      return mlir::failure();
    std::optional<MemLayout> consumerLayout = findLayout(
        representationAssignment, demand.destinationTile, edge->consumer,
        PhysicalValueRole::Operand, edge->consumerOperand);
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(
        producer->operation->getResult(edge->producerResult).getType());
    if (!consumerLayout || !tensor)
      return mlir::failure();
    auto fragments = getFragments(demand, tensor, *edge, *consumerLayout,
                                  representationAssignment);
    if (mlir::failed(fragments) || fragments->empty())
      return mlir::failure();
    const CoupledRegionGroup *producerGroup =
        findGroup(coupledAssignment, edge->producer, demand.destinationTile);
    const CoupledRegionGroup *consumerGroup =
        findGroup(coupledAssignment, edge->consumer, demand.destinationTile);
    bool allLocal = llvm::all_of(*fragments, [&](const auto &fragment) {
      return fragment.sourceTile == demand.destinationTile;
    });
    bool retained = allLocal && producerGroup && consumerGroup &&
                    producerGroup == consumerGroup;
    bool peer = llvm::any_of(*fragments, [&](const auto &fragment) {
      return fragment.sourceTile != demand.destinationTile;
    });
    if (peer)
      for (const DataMovementFragment &fragment : *fragments)
        if (fragment.sourceTile != demand.destinationTile &&
            !getFirstSimpleRoute(program.topology, cardId, fragment.sourceTile,
                                 demand.destinationTile))
          return mlir::failure();
    llvm::SmallVector<TileId, 4> equivalentDestinations;
    for (const StructuredDAGEdgeDemand &other : plan->demands)
      if (other.edge == demand.edge &&
          other.producerDemand.isEqual(demand.producerDemand))
        equivalentDestinations.push_back(other.destinationTile);
    llvm::sort(equivalentDestinations, [](TileId lhs, TileId rhs) {
      return lhs.getValue() < rhs.getValue();
    });
    llvm::SmallVector<uint32_t, 4> invariantIterators;
    const StructuredDAGNode *consumer = program.dag.getNode(edge->consumer);
    auto linalg =
        consumer ? mlir::dyn_cast<mlir::linalg::LinalgOp>(consumer->operation)
                 : mlir::linalg::LinalgOp{};
    if (linalg && edge->consumerOperand < linalg->getNumOperands()) {
      mlir::AffineMap map = linalg.getMatchingIndexingMap(
          &linalg->getOpOperand(edge->consumerOperand));
      llvm::DenseSet<unsigned> used;
      for (mlir::AffineExpr expression : map.getResults())
        expression.walk([&](mlir::AffineExpr nested) {
          if (auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(nested))
            used.insert(dimension.getPosition());
        });
      for (unsigned dimension = 0; dimension < map.getNumDims(); ++dimension)
        if (!used.contains(dimension))
          invariantIterators.push_back(dimension);
    }
    demands.push_back(DemandDomain{
        demand.edge, demand.destinationTile, *consumerLayout, retained,
        mlir::isMemoryEffectFree(producer->operation), peer,
        std::move(*fragments), std::move(equivalentDestinations),
        std::move(invariantIterators)});
  }
  llvm::sort(demands, [](const auto &lhs, const auto &rhs) {
    return std::tuple(lhs.edge, lhs.destinationTile.getValue()) <
           std::tuple(rhs.edge, rhs.destinationTile.getValue());
  });
  llvm::SmallVector<ReductionDomain, 4> reductions;
  for (const analysis::LogicalNodeTrial &nodeTrial : trial.nodes) {
    if (!nodeTrial.reductionMergeTile)
      continue;
    const StructuredDAGNode *node = program.dag.getNode(nodeTrial.node);
    if (!node || !node->operation)
      return mlir::failure();
    for (unsigned resultIndex = 0;
         resultIndex < node->operation->getNumResults(); ++resultIndex) {
      auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(
          node->operation->getResult(resultIndex).getType());
      std::optional<MemLayout> selectedMergeLayout =
          findLayout(representationAssignment, *nodeTrial.reductionMergeTile,
                     nodeTrial.node, PhysicalValueRole::Result, resultIndex);
      if (!tensor || !selectedMergeLayout ||
          nodeTrial.executionShards.empty() ||
          !nodeTrial.executionShards.front().executionDomain)
        return mlir::failure();
      const MemLayout mergeLayout = *selectedMergeLayout;
      llvm::SmallVector<DataMovementFragment, 8> fragments;
      for (const analysis::LogicalExecutionShard &execution :
           nodeTrial.executionShards) {
        if (!execution.executionDomain)
          return mlir::failure();
        analysis::IndexSetResult exact{analysis::IndexRelationStatus::Exact,
                                       *execution.executionDomain,
                                       {}};
        auto rectangle = exact.getExactStaticRectangularDomain();
        std::optional<MemLayout> sourceLayout =
            findLayout(representationAssignment, execution.tile, nodeTrial.node,
                       PhysicalValueRole::Result, resultIndex);
        if (!rectangle.isExact() || !sourceLayout)
          return mlir::failure();
        uint64_t elements = 1;
        for (int64_t size : rectangle.domain->sizes)
          elements = saturatingMultiply(elements, size);
        unsigned elementBits = 0;
        if (auto integer =
                mlir::dyn_cast<mlir::IntegerType>(tensor.getElementType()))
          elementBits = integer.getWidth();
        else if (auto floating =
                     mlir::dyn_cast<mlir::FloatType>(tensor.getElementType()))
          elementBits = floating.getWidth();
        else
          return mlir::failure();
        auto transportType = mlir::MemRefType::get(
            rectangle.domain->sizes, tensor.getElementType(),
            mlir::MemRefLayoutAttrInterface{},
            MemoryAttr::get(tensor.getContext(), MemorySpace::SPM,
                            mergeLayout));
        auto physical = analysis::PhysicalLayoutRelation::create(transportType);
        if (mlir::failed(physical))
          return mlir::failure();
        DataMovementFragment fragment{
            execution.tile,
            *sourceLayout,
            mergeLayout,
            tensor.getElementType(),
            rectangle.domain->offsets,
            rectangle.domain->sizes,
            rectangle.domain->offsets,
            rectangle.domain->sizes,
            llvm::SmallVector<int64_t, 4>(rectangle.domain->sizes.size(), 0),
            (saturatingMultiply(elements, elementBits) + 7) / 8,
            static_cast<uint64_t>(physical->getPhysicalFootprintBytes()),
            {}};
        if (execution.tile != *nodeTrial.reductionMergeTile) {
          auto route =
              getFirstSimpleRoute(program.topology, cardId, execution.tile,
                                  *nodeTrial.reductionMergeTile);
          if (!route)
            return mlir::failure();
          fragment.route = std::move(*route);
        }
        fragments.push_back(std::move(fragment));
      }
      llvm::sort(fragments, [](const auto &lhs, const auto &rhs) {
        return lhs.sourceTile.getValue() < rhs.sourceTile.getValue();
      });
      reductions.push_back(ReductionDomain{nodeTrial.node, resultIndex,
                                           *nodeTrial.reductionMergeTile,
                                           mergeLayout, std::move(fragments)});
    }
  }
  return CardDataMovementDomain(program.topology, cardId, std::move(demands),
                                std::move(reductions));
}

DataMovementChoice
CardDataMovementDomain::getFirstChoice(const DemandDomain &domain) const {
  DataMovementKind kind =
      domain.retained ? DataMovementKind::Retained : DataMovementKind::DDR;
  return DataMovementChoice{
      domain.edge, domain.destinationTile, kind, domain.consumerLayout, {}};
}

bool CardDataMovementDomain::contains(const DemandDomain &domain,
                                      const DataMovementChoice &choice) const {
  if (choice.edge != domain.edge ||
      choice.destinationTile != domain.destinationTile ||
      choice.consumerLayout != domain.consumerLayout)
    return false;
  if (choice.kind == DataMovementKind::Retained)
    return domain.retained && choice.fragments.empty();
  if (choice.kind == DataMovementKind::Refetch)
    return domain.retained && choice.fragments == domain.fragments;
  if (choice.kind == DataMovementKind::DDR)
    return choice.fragments.empty();
  if (choice.kind == DataMovementKind::Recompute)
    return domain.recompute && choice.fragments.empty();
  if (choice.kind != DataMovementKind::Peer || !domain.peer ||
      choice.fragments.size() != domain.fragments.size())
    return false;
  for (auto [selected, expected] :
       llvm::zip_equal(choice.fragments, domain.fragments)) {
    if (selected.sourceTile != expected.sourceTile ||
        selected.sourceLayout != expected.sourceLayout ||
        selected.transportLayout != expected.transportLayout ||
        selected.elementType != expected.elementType ||
        selected.offsets != expected.offsets ||
        selected.sizes != expected.sizes ||
        selected.sourceBaseOffsets != expected.sourceBaseOffsets ||
        selected.sourceShape != expected.sourceShape ||
        selected.destinationOffsets != expected.destinationOffsets ||
        selected.logicalBytes != expected.logicalBytes ||
        selected.physicalBytes != expected.physicalBytes ||
        !isValidSimpleRoute(topology, cardId, selected.sourceTile,
                            domain.destinationTile, selected.route))
      return false;
  }
  return true;
}

mlir::FailureOr<std::optional<DataMovementChoice>>
CardDataMovementDomain::getNextChoice(const DemandDomain &domain,
                                      const DataMovementChoice &choice) const {
  if (!contains(domain, choice))
    return mlir::failure();
  auto firstPeer = [&]() -> DataMovementChoice {
    DataMovementChoice result{domain.edge, domain.destinationTile,
                              DataMovementKind::Peer, domain.consumerLayout,
                              domain.fragments};
    for (DataMovementFragment &fragment : result.fragments) {
      if (fragment.sourceTile == domain.destinationTile)
        continue;
      fragment.route = *getFirstSimpleRoute(
          topology, cardId, fragment.sourceTile, domain.destinationTile);
    }
    return result;
  };
  switch (choice.kind) {
  case DataMovementKind::Retained:
    return std::optional<DataMovementChoice>(DataMovementChoice{
        domain.edge, domain.destinationTile, DataMovementKind::Refetch,
        domain.consumerLayout, domain.fragments});
  case DataMovementKind::Refetch:
    return std::optional<DataMovementChoice>{};
  case DataMovementKind::DDR:
    if (domain.recompute)
      return std::optional<DataMovementChoice>(
          DataMovementChoice{domain.edge,
                             domain.destinationTile,
                             DataMovementKind::Recompute,
                             domain.consumerLayout,
                             {}});
    if (domain.peer)
      return std::optional<DataMovementChoice>(firstPeer());
    return std::optional<DataMovementChoice>{};
  case DataMovementKind::Recompute:
    if (domain.peer)
      return std::optional<DataMovementChoice>(firstPeer());
    return std::optional<DataMovementChoice>{};
  case DataMovementKind::Peer: {
    DataMovementChoice next = choice;
    for (size_t reverse = 0; reverse < next.fragments.size(); ++reverse) {
      size_t index = next.fragments.size() - reverse - 1;
      DataMovementFragment &fragment = next.fragments[index];
      if (fragment.sourceTile == domain.destinationTile)
        continue;
      RouteSuccessor route =
          getNextSimpleRoute(topology, cardId, fragment.sourceTile,
                             domain.destinationTile, fragment.route);
      if (!route.currentFound)
        return mlir::failure();
      if (!route.next)
        continue;
      fragment.route = std::move(*route.next);
      for (size_t reset = index + 1; reset < next.fragments.size(); ++reset) {
        DataMovementFragment &later = next.fragments[reset];
        if (later.sourceTile == domain.destinationTile)
          continue;
        later.route = *getFirstSimpleRoute(topology, cardId, later.sourceTile,
                                           domain.destinationTile);
      }
      return std::optional<DataMovementChoice>(std::move(next));
    }
    return std::optional<DataMovementChoice>{};
  }
  }
  llvm_unreachable("unknown data movement kind");
}

CardDataMovementAssignment CardDataMovementDomain::getFirstAssignment() const {
  CardDataMovementAssignment result;
  for (const DemandDomain &demand : demands)
    result.edges.push_back(getFirstChoice(demand));
  for (const ReductionDomain &reduction : reductions)
    result.reductions.push_back(ReductionGatherChoice{reduction.node,
                                                      reduction.resultIndex,
                                                      reduction.mergeTile,
                                                      ReductionGatherKind::DDR,
                                                      reduction.mergeLayout,
                                                      {}});
  return result;
}

llvm::SmallVector<DataReuseFact, 32>
CardDataMovementDomain::getReuseFacts() const {
  llvm::SmallVector<DataReuseFact, 32> result;
  for (const DemandDomain &demand : demands)
    result.push_back(DataReuseFact{demand.edge, demand.destinationTile,
                                   demand.equivalentDestinationTiles,
                                   demand.temporalInvariantIterators});
  return result;
}

bool CardDataMovementDomain::contains(
    const CardDataMovementAssignment &assignment) const {
  if (assignment.edges.size() != demands.size())
    return false;
  if (assignment.reductions.size() != reductions.size())
    return false;
  for (auto [domain, choice] : llvm::zip_equal(demands, assignment.edges))
    if (!contains(domain, choice))
      return false;
  for (auto [domain, choice] :
       llvm::zip_equal(reductions, assignment.reductions)) {
    if (choice.node != domain.node ||
        choice.resultIndex != domain.resultIndex ||
        choice.mergeTile != domain.mergeTile ||
        choice.mergeLayout != domain.mergeLayout)
      return false;
    if (choice.kind == ReductionGatherKind::DDR) {
      if (!choice.fragments.empty())
        return false;
      continue;
    }
    if (choice.kind != ReductionGatherKind::Peer ||
        choice.fragments.size() != domain.fragments.size())
      return false;
    for (auto [selected, expected] :
         llvm::zip_equal(choice.fragments, domain.fragments)) {
      if (selected.sourceTile != expected.sourceTile ||
          selected.sourceLayout != expected.sourceLayout ||
          selected.transportLayout != expected.transportLayout ||
          selected.elementType != expected.elementType ||
          selected.offsets != expected.offsets ||
          selected.sizes != expected.sizes ||
          selected.sourceBaseOffsets != expected.sourceBaseOffsets ||
          selected.sourceShape != expected.sourceShape ||
          selected.logicalBytes != expected.logicalBytes ||
          selected.physicalBytes != expected.physicalBytes ||
          !isValidSimpleRoute(topology, cardId, selected.sourceTile,
                              domain.mergeTile, selected.route))
        return false;
    }
  }
  std::map<int64_t, llvm::SmallVector<size_t, 8>> multicastGroups;
  for (auto [index, choice] : llvm::enumerate(assignment.edges)) {
    if (choice.multicastGroup < 0)
      continue;
    if (choice.kind != DataMovementKind::Peer || choice.fragments.size() != 1)
      return false;
    multicastGroups[choice.multicastGroup].push_back(index);
  }
  for (const auto &[groupId, indices] : multicastGroups) {
    if (indices.size() < 2)
      return false;
    const DataMovementChoice &first = assignment.edges[indices.front()];
    const DataMovementFragment &firstFragment = first.fragments.front();
    int64_t minimumDestination = first.destinationTile.getValue();
    llvm::DenseMap<int64_t, int64_t> parent;
    llvm::DenseSet<int64_t> destinations;
    for (size_t index : indices) {
      const DataMovementChoice &choice = assignment.edges[index];
      const DataMovementFragment &fragment = choice.fragments.front();
      minimumDestination =
          std::min(minimumDestination, choice.destinationTile.getValue());
      destinations.insert(choice.destinationTile.getValue());
      if (choice.edge != first.edge ||
          fragment.sourceTile != firstFragment.sourceTile ||
          fragment.sourceLayout != firstFragment.sourceLayout ||
          fragment.transportLayout != firstFragment.transportLayout ||
          fragment.elementType != firstFragment.elementType ||
          fragment.offsets != firstFragment.offsets ||
          fragment.sizes != firstFragment.sizes ||
          fragment.logicalBytes != firstFragment.logicalBytes ||
          fragment.physicalBytes != firstFragment.physicalBytes)
        return false;
      for (const TileLink &link : fragment.route) {
        auto [entry, inserted] = parent.try_emplace(link.destination.getValue(),
                                                    link.source.getValue());
        if (!inserted && entry->second != link.source.getValue())
          return false;
      }
    }
    if (groupId != minimumDestination)
      return false;
    (void)destinations;
  }
  return true;
}

mlir::FailureOr<std::optional<CardDataMovementAssignment>>
CardDataMovementDomain::getNextAssignment(
    const CardDataMovementAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  auto multicastCompatible = [](const DataMovementChoice &lhs,
                                const DataMovementChoice &rhs) {
    if (lhs.kind != DataMovementKind::Peer ||
        rhs.kind != DataMovementKind::Peer || lhs.edge != rhs.edge ||
        lhs.fragments.size() != 1 || rhs.fragments.size() != 1)
      return false;
    const DataMovementFragment &left = lhs.fragments.front();
    const DataMovementFragment &right = rhs.fragments.front();
    return left.sourceTile == right.sourceTile &&
           left.sourceLayout == right.sourceLayout &&
           left.transportLayout == right.transportLayout &&
           left.elementType == right.elementType &&
           left.offsets == right.offsets && left.sizes == right.sizes &&
           left.logicalBytes == right.logicalBytes &&
           left.physicalBytes == right.physicalBytes;
  };
  llvm::SmallVector<llvm::SmallVector<size_t, 8>, 8> multicastClasses;
  for (size_t index = 0; index < assignment.edges.size(); ++index) {
    const DataMovementChoice &choice = assignment.edges[index];
    if (choice.kind != DataMovementKind::Peer || choice.fragments.size() != 1)
      continue;
    auto found = llvm::find_if(multicastClasses, [&](const auto &indices) {
      return multicastCompatible(choice, assignment.edges[indices.front()]);
    });
    if (found == multicastClasses.end())
      multicastClasses.push_back({index});
    else
      found->push_back(index);
  }

  CardDataMovementAssignment multicast = assignment;
  for (DataMovementChoice &choice : multicast.edges)
    choice.multicastGroup = -1;
  bool foundCurrentMulticast = false;
  std::optional<CardDataMovementAssignment> nextMulticast;
  std::function<void(size_t)> generateClasses;
  generateClasses = [&](size_t classIndex) {
    if (nextMulticast)
      return;
    if (classIndex == multicastClasses.size()) {
      if (!contains(multicast))
        return;
      if (multicast == assignment) {
        foundCurrentMulticast = true;
      } else if (foundCurrentMulticast) {
        nextMulticast = multicast;
      }
      return;
    }
    llvm::SmallVector<size_t, 8> initial = multicastClasses[classIndex];
    std::function<void(llvm::SmallVector<size_t, 8>)> partition;
    partition = [&](llvm::SmallVector<size_t, 8> remaining) {
      if (nextMulticast)
        return;
      if (remaining.empty()) {
        generateClasses(classIndex + 1);
        return;
      }
      const size_t first = remaining.front();
      llvm::SmallVector<size_t, 8> tail(remaining.begin() + 1, remaining.end());
      multicast.edges[first].multicastGroup = -1;
      partition(tail);
      const uint64_t subsetCount = uint64_t{1} << tail.size();
      for (uint64_t mask = 1; mask < subsetCount && !nextMulticast; ++mask) {
        llvm::SmallVector<size_t, 8> group{first};
        llvm::SmallVector<size_t, 8> nextRemaining;
        for (auto [bit, index] : llvm::enumerate(tail)) {
          if (mask & (uint64_t{1} << bit))
            group.push_back(index);
          else
            nextRemaining.push_back(index);
        }
        int64_t groupId = std::numeric_limits<int64_t>::max();
        for (size_t index : group)
          groupId = std::min(groupId,
                             multicast.edges[index].destinationTile.getValue());
        for (size_t index : group)
          multicast.edges[index].multicastGroup = groupId;
        partition(std::move(nextRemaining));
        for (size_t index : group)
          multicast.edges[index].multicastGroup = -1;
      }
    };
    partition(std::move(initial));
  };
  generateClasses(/*classIndex=*/0);
  if (!foundCurrentMulticast)
    return mlir::failure();
  if (nextMulticast)
    return std::optional<CardDataMovementAssignment>(std::move(*nextMulticast));

  CardDataMovementAssignment next = assignment;
  for (DataMovementChoice &choice : next.edges)
    choice.multicastGroup = -1;
  for (size_t reverse = 0; reverse < reductions.size(); ++reverse) {
    size_t index = reductions.size() - reverse - 1;
    const ReductionDomain &domain = reductions[index];
    ReductionGatherChoice &choice = next.reductions[index];
    if (choice.kind == ReductionGatherKind::DDR) {
      choice.kind = ReductionGatherKind::Peer;
      choice.fragments = domain.fragments;
      for (size_t reset = index + 1; reset < reductions.size(); ++reset) {
        next.reductions[reset].kind = ReductionGatherKind::DDR;
        next.reductions[reset].fragments.clear();
      }
      return std::optional<CardDataMovementAssignment>(std::move(next));
    }
    bool advancedRoute = false;
    for (size_t fragmentReverse = 0; fragmentReverse < choice.fragments.size();
         ++fragmentReverse) {
      size_t fragmentIndex = choice.fragments.size() - fragmentReverse - 1;
      DataMovementFragment &fragment = choice.fragments[fragmentIndex];
      if (fragment.sourceTile == domain.mergeTile)
        continue;
      RouteSuccessor route =
          getNextSimpleRoute(topology, cardId, fragment.sourceTile,
                             domain.mergeTile, fragment.route);
      if (!route.currentFound)
        return mlir::failure();
      if (!route.next)
        continue;
      fragment.route = std::move(*route.next);
      for (size_t reset = fragmentIndex + 1; reset < choice.fragments.size();
           ++reset) {
        DataMovementFragment &later = choice.fragments[reset];
        if (later.sourceTile == domain.mergeTile)
          continue;
        later.route = *getFirstSimpleRoute(topology, cardId, later.sourceTile,
                                           domain.mergeTile);
      }
      advancedRoute = true;
      break;
    }
    if (advancedRoute) {
      for (size_t reset = index + 1; reset < reductions.size(); ++reset) {
        next.reductions[reset].kind = ReductionGatherKind::DDR;
        next.reductions[reset].fragments.clear();
      }
      return std::optional<CardDataMovementAssignment>(std::move(next));
    }
  }
  for (ReductionGatherChoice &reduction : next.reductions) {
    reduction.kind = ReductionGatherKind::DDR;
    reduction.fragments.clear();
  }
  for (size_t reverse = 0; reverse < demands.size(); ++reverse) {
    size_t index = demands.size() - reverse - 1;
    auto advanced = getNextChoice(demands[index], next.edges[index]);
    if (mlir::failed(advanced))
      return mlir::failure();
    if (!*advanced)
      continue;
    next.edges[index] = std::move(**advanced);
    for (size_t reset = index + 1; reset < demands.size(); ++reset)
      next.edges[reset] = getFirstChoice(demands[reset]);
    return std::optional<CardDataMovementAssignment>(std::move(next));
  }
  return std::optional<CardDataMovementAssignment>{};
}

} // namespace wafer::compiler::detail
