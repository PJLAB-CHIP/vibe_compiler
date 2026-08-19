//===- Buffering.cpp - Rotating-buffer candidate domain ---------------===//

#include "Wafer/Planning/Search/Buffering.h"

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"
#include "Wafer/Analysis/Structured/StructuredOperationTileFootprint.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

namespace wafer::compiler::detail {
namespace {

const analysis::LogicalNodeTrial *
findNodeTrial(const analysis::LogicalShardTrial &trial,
              StructuredDAGNodeID node) {
  auto found = llvm::find_if(trial.nodes, [&](const auto &candidate) {
    return candidate.node == node;
  });
  return found == trial.nodes.end() ? nullptr : &*found;
}

const TemporalNodeAssignment *
findTemporal(const CardTemporalAssignment &assignment,
             StructuredDAGNodeID node) {
  auto found = llvm::find_if(assignment.nodes, [&](const auto &candidate) {
    return candidate.node == node;
  });
  return found == assignment.nodes.end() ? nullptr : &*found;
}

std::optional<analysis::StaticRectangularIndexSet>
findShard(const analysis::LogicalNodeTrial &trial, TileId tile) {
  auto found = llvm::find_if(trial.executionShards, [&](const auto &shard) {
    return shard.tile == tile;
  });
  if (found == trial.executionShards.end() || !found->executionDomain)
    return std::nullopt;
  analysis::StaticRectangularIndexSetResult rectangle =
      analysis::IndexSetResult{
          analysis::IndexRelationStatus::Exact, *found->executionDomain, {}}
          .getExactStaticRectangularDomain();
  return rectangle.isExact() ? rectangle.domain : std::nullopt;
}

uint64_t
getMaximumSteadyTripCount(const CoupledRegionGroup &group,
                          const analysis::LogicalShardTrial &trial,
                          const CardTemporalAssignment &temporalAssignment) {
  uint64_t result = 0;
  for (StructuredDAGNodeID node : group.nodes) {
    const analysis::LogicalNodeTrial *nodeTrial = findNodeTrial(trial, node);
    const TemporalNodeAssignment *temporal =
        findTemporal(temporalAssignment, node);
    std::optional<analysis::StaticRectangularIndexSet> shard =
        nodeTrial ? findShard(*nodeTrial, group.tile) : std::nullopt;
    if (!temporal || !shard ||
        shard->sizes.size() != temporal->iteratorTileSizes.size())
      return 0;
    for (auto [extent, tile] :
         llvm::zip_equal(shard->sizes, temporal->iteratorTileSizes)) {
      if (extent <= 0 || tile <= 0)
        return 0;
      // TemporalWaveLoop emits the first full wave before its steady scf.for
      // and a partial tail after it. Only the remaining full waves form the
      // static loop on which rotating slots can be materialized.
      const uint64_t fullWaves = static_cast<uint64_t>(extent / tile);
      if (fullWaves > 1)
        result = std::max(result, fullWaves - 1);
    }
  }
  return result;
}

const DataMovementChoice *
findMovement(const CardDataMovementAssignment &assignment,
             StructuredDAGEdgeID edge, TileId tile) {
  auto found = llvm::find_if(assignment.edges, [&](const auto &choice) {
    return choice.edge == edge && choice.destinationTile == tile;
  });
  return found == assignment.edges.end() ? nullptr : &*found;
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

std::optional<llvm::SmallVector<int64_t, 4>>
getLeafSizes(const analysis::LogicalShardTrial &trial,
             const CardTemporalAssignment &temporalAssignment,
             StructuredDAGNodeID node, TileId tile) {
  const analysis::LogicalNodeTrial *nodeTrial = findNodeTrial(trial, node);
  const TemporalNodeAssignment *temporal =
      findTemporal(temporalAssignment, node);
  std::optional<analysis::StaticRectangularIndexSet> shard =
      nodeTrial ? findShard(*nodeTrial, tile) : std::nullopt;
  if (!temporal || !shard ||
      shard->sizes.size() != temporal->iteratorTileSizes.size())
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> result;
  for (auto [extent, size] :
       llvm::zip_equal(shard->sizes, temporal->iteratorTileSizes)) {
    if (extent <= 0 || size <= 0)
      return std::nullopt;
    result.push_back(std::min(extent, size));
  }
  return result;
}

std::optional<uint64_t> getSingleSlotPhysicalLowerBound(
    const CardProgramAnalysis &program,
    const analysis::LogicalShardTrial &trial,
    const CardTemporalAssignment &temporalAssignment,
    const CardPhysicalRepresentationAssignment &representationAssignment,
    const StructuredDAGEdge &edge, TileId tile) {
  const StructuredDAGNode *producer = program.dag.getNode(edge.producer);
  const StructuredDAGNode *consumer = program.dag.getNode(edge.consumer);
  auto producerLeaf =
      getLeafSizes(trial, temporalAssignment, edge.producer, tile);
  auto consumerLeaf =
      getLeafSizes(trial, temporalAssignment, edge.consumer, tile);
  std::optional<MemLayout> producerLayout =
      findLayout(representationAssignment, tile, edge.producer,
                 PhysicalValueRole::Result, edge.producerResult);
  std::optional<MemLayout> consumerLayout =
      findLayout(representationAssignment, tile, edge.consumer,
                 PhysicalValueRole::Operand, edge.consumerOperand);
  if (!producer || !producer->operation || !consumer || !consumer->operation ||
      !producerLeaf || !consumerLeaf || !producerLayout || !consumerLayout)
    return std::nullopt;
  auto producerShape = getStructuredResultTileShape(
      producer->operation, edge.producerResult, *producerLeaf);
  auto consumerShape = getStructuredOperandTileShape(
      consumer->operation, edge.consumerOperand, *consumerLeaf);
  auto producerTensor = mlir::dyn_cast<mlir::RankedTensorType>(
      producer->operation->getResult(edge.producerResult).getType());
  auto consumerTensor = mlir::dyn_cast<mlir::RankedTensorType>(
      consumer->operation->getOperand(edge.consumerOperand).getType());
  if (!producerShape || !consumerShape || !producerTensor || !consumerTensor)
    return std::nullopt;
  auto getBytes = [](mlir::RankedTensorType tensor,
                     llvm::ArrayRef<int64_t> shape,
                     MemLayout layout) -> std::optional<uint64_t> {
    auto type = mlir::MemRefType::get(
        shape, tensor.getElementType(), mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(tensor.getContext(), MemorySpace::SPM, layout));
    auto physical = analysis::PhysicalLayoutRelation::create(type);
    if (mlir::failed(physical) || physical->getPhysicalFootprintBytes() <= 0)
      return std::nullopt;
    return static_cast<uint64_t>(physical->getPhysicalFootprintBytes());
  };
  std::optional<uint64_t> producerBytes =
      getBytes(producerTensor, *producerShape, *producerLayout);
  std::optional<uint64_t> consumerBytes =
      getBytes(consumerTensor, *consumerShape, *consumerLayout);
  if (!producerBytes || !consumerBytes)
    return std::nullopt;
  return std::max(*producerBytes, *consumerBytes);
}

} // namespace

mlir::FailureOr<CardBufferingDomain> CardBufferingDomain::create(
    const CardProgramAnalysis &program,
    const analysis::LogicalShardTrial &trial,
    const CoupledRegionDomain &coupledDomain,
    const CoupledRegionAssignment &coupledAssignment,
    const CardTemporalDomain &temporalDomain,
    const CardTemporalAssignment &temporalAssignment,
    const CardPhysicalRepresentationDomain &representationDomain,
    const CardPhysicalRepresentationAssignment &representationAssignment,
    const CardDataMovementDomain &movementDomain,
    const CardDataMovementAssignment &movementAssignment,
    const TargetMemoryPolicy &memory, std::string *failureReason) {
  auto fail =
      [&](llvm::StringRef message) -> mlir::FailureOr<CardBufferingDomain> {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  };
  if (trial.epoch != program.epoch ||
      !coupledDomain.contains(coupledAssignment) ||
      !temporalDomain.contains(temporalAssignment) ||
      !representationDomain.contains(representationAssignment) ||
      !movementDomain.contains(movementAssignment) || memory.spmBase < 0 ||
      memory.spmLimit <= memory.spmBase)
    return fail("buffering received a stale assignment or invalid memory "
                "capacity");

  const uint64_t capacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);
  llvm::SmallVector<GroupDomain, 16> groups;
  groups.reserve(coupledAssignment.groups.size());
  for (const CoupledRegionGroup &selected : coupledAssignment.groups) {
    GroupDomain group;
    group.tile = selected.tile;
    group.nodes = selected.nodes;
    const uint64_t steadyTripCount =
        getMaximumSteadyTripCount(selected, trial, temporalAssignment);
    for (const StructuredDAGEdge &edge : program.dag.getEdges()) {
      if (!llvm::is_contained(selected.nodes, edge.producer) ||
          !llvm::is_contained(selected.nodes, edge.consumer))
        continue;
      const DataMovementChoice *movement =
          findMovement(movementAssignment, edge.id, selected.tile);
      if (!movement || movement->kind != DataMovementKind::Retained)
        continue;
      std::optional<uint64_t> singleSlotBytes = getSingleSlotPhysicalLowerBound(
          program, trial, temporalAssignment, representationAssignment, edge,
          selected.tile);
      if (!singleSlotBytes || *singleSlotBytes == 0)
        return fail("retained buffering edge has no physical leaf payload");
      const uint64_t capacitySlots = capacity / *singleSlotBytes;
      const uint64_t maximum = std::min(steadyTripCount, capacitySlots);
      if (maximum < 2)
        continue;
      group.edges.push_back(EdgeDomain{
          edge.id, static_cast<uint32_t>(std::min<uint64_t>(
                       maximum, std::numeric_limits<uint32_t>::max()))});
    }
    llvm::sort(group.edges, [](const EdgeDomain &lhs, const EdgeDomain &rhs) {
      return lhs.edge < rhs.edge;
    });
    groups.push_back(std::move(group));
  }
  return CardBufferingDomain(std::move(groups));
}

BufferingChoice
CardBufferingDomain::getFirstChoice(const GroupDomain &group) const {
  return BufferingChoice{group.tile, group.nodes, {}, 1};
}

bool CardBufferingDomain::contains(const GroupDomain &group,
                                   const BufferingChoice &choice) const {
  if (choice.tile != group.tile || choice.groupNodes != group.nodes ||
      choice.slotCount == 0 || !llvm::is_sorted(choice.pipelinedEdges) ||
      std::adjacent_find(choice.pipelinedEdges.begin(),
                         choice.pipelinedEdges.end()) !=
          choice.pipelinedEdges.end())
    return false;
  if (choice.slotCount == 1)
    return choice.pipelinedEdges.empty();
  if (choice.pipelinedEdges.empty())
    return false;
  return llvm::all_of(choice.pipelinedEdges, [&](StructuredDAGEdgeID edge) {
    auto found = llvm::find_if(group.edges, [&](const EdgeDomain &candidate) {
      return candidate.edge == edge;
    });
    return found != group.edges.end() &&
           found->maximumSlots >= choice.slotCount;
  });
}

mlir::FailureOr<std::optional<BufferingChoice>>
CardBufferingDomain::getNextChoice(const GroupDomain &group,
                                   const BufferingChoice &choice) const {
  if (!contains(group, choice))
    return mlir::failure();
  auto getEligibleEdges = [&](uint32_t count) {
    llvm::SmallVector<StructuredDAGEdgeID, 4> result;
    for (const EdgeDomain &edge : group.edges)
      if (edge.maximumSlots >= count)
        result.push_back(edge.edge);
    return result;
  };
  uint32_t slotCount = choice.slotCount;
  llvm::SmallVector<StructuredDAGEdgeID, 4> eligible;
  llvm::SmallVector<uint8_t, 8> selected;
  if (slotCount == 1) {
    slotCount = 2;
  } else {
    eligible = getEligibleEdges(slotCount);
    selected.assign(eligible.size(), 0);
    for (StructuredDAGEdgeID edge : choice.pipelinedEdges) {
      auto found = llvm::find(eligible, edge);
      if (found == eligible.end())
        return mlir::failure();
      selected[static_cast<size_t>(found - eligible.begin())] = 1;
    }
    for (size_t index = 0; index < selected.size(); ++index) {
      selected[index] ^= 1;
      if (selected[index]) {
        BufferingChoice next{group.tile, group.nodes, {}, slotCount};
        for (auto [edge, enabled] : llvm::zip_equal(eligible, selected))
          if (enabled)
            next.pipelinedEdges.push_back(edge);
        return std::optional<BufferingChoice>(std::move(next));
      }
    }
    ++slotCount;
  }

  while (true) {
    eligible = getEligibleEdges(slotCount);
    if (!eligible.empty())
      return std::optional<BufferingChoice>(BufferingChoice{
          group.tile, group.nodes, {eligible.front()}, slotCount});
    uint32_t maximum = 1;
    for (const EdgeDomain &edge : group.edges)
      maximum = std::max(maximum, edge.maximumSlots);
    if (slotCount >= maximum ||
        slotCount == std::numeric_limits<uint32_t>::max())
      break;
    ++slotCount;
  }
  return std::optional<BufferingChoice>{};
}

CardBufferingAssignment CardBufferingDomain::getFirstAssignment() const {
  CardBufferingAssignment result;
  for (const GroupDomain &group : groups)
    result.groups.push_back(getFirstChoice(group));
  return result;
}

bool CardBufferingDomain::contains(
    const CardBufferingAssignment &assignment) const {
  if (assignment.groups.size() != groups.size())
    return false;
  for (auto [domain, choice] : llvm::zip_equal(groups, assignment.groups))
    if (!contains(domain, choice))
      return false;
  return true;
}

mlir::FailureOr<std::optional<CardBufferingAssignment>>
CardBufferingDomain::getNextAssignment(
    const CardBufferingAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  CardBufferingAssignment next = assignment;
  for (size_t reverse = 0; reverse < groups.size(); ++reverse) {
    const size_t index = groups.size() - reverse - 1;
    auto advanced = getNextChoice(groups[index], next.groups[index]);
    if (mlir::failed(advanced))
      return mlir::failure();
    if (!*advanced)
      continue;
    next.groups[index] = std::move(**advanced);
    for (size_t reset = index + 1; reset < groups.size(); ++reset)
      next.groups[reset] = getFirstChoice(groups[reset]);
    return std::optional<CardBufferingAssignment>(std::move(next));
  }
  return std::optional<CardBufferingAssignment>{};
}

} // namespace wafer::compiler::detail
