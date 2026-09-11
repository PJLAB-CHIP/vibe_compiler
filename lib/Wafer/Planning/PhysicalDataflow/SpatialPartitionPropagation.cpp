//===- SpatialPartitionPropagation.cpp - SSA-coordinated proposals --------===//

#include "Wafer/Planning/PhysicalDataflow/SpatialPartitionPropagation.h"
#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

uint64_t getOperandBytes(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return 0;
  uint64_t bytes = llvm::divideCeil(type.getElementTypeBitWidth(), 8u);
  for (int64_t extent : type.getShape())
    bytes = llvm::SaturatingMultiply(bytes, static_cast<uint64_t>(extent));
  return bytes;
}

std::optional<size_t> getParallelParts(const SpatialRootDomainFacts &facts,
                                       const NodeSpatialPlan &plan) {
  size_t count = 1;
  for (size_t axis = 0; axis < facts.iteratorKinds.size(); ++axis) {
    if (facts.iteratorKinds[axis] != SpatialIteratorKind::Parallel)
      continue;
    auto parts = getIteratorPartitionIntervalCount(facts.iteratorExtents[axis],
                                                   plan.axes[axis]);
    if (mlir::failed(parts) || *parts <= 0 ||
        count >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(*parts))
      return std::nullopt;
    count *= static_cast<size_t>(*parts);
  }
  return count;
}

bool isRequiredReductionAxis(const SpatialRootDomainFacts &facts, size_t axis) {
  return facts.attention &&
         facts.attention->keyValuePartition ==
             AttentionKeyValuePartitionRequirement::MultipleIntervals &&
         llvm::is_contained(facts.attention->keyValueReductionIterators,
                            static_cast<unsigned>(axis));
}

// Compose the actual operand path. A multi-source support op is deliberately
// not treated as a reshape or a whole-tensor identity.
std::optional<analysis::IndexRelation>
getConsumerToProducer(mlir::Operation *consumer, unsigned operand,
                      const SpatialRootDomainFacts &consumerFacts,
                      mlir::Operation *producer, unsigned result,
                      const SpatialRootDomainFacts &producerFacts,
                      const analysis::IndexRelationLimits &limits) {
  mlir::Value value = consumer->getOperand(operand);
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  auto output = producer->getResult(result);
  auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(output.getType());
  if (!type || !type.hasStaticShape() || !outputType ||
      !outputType.hasStaticShape())
    return std::nullopt;
  auto relation = analysis::deriveIterationProducerRelation(
      consumer->getOpOperand(operand), consumerFacts.iteratorExtents, output,
      limits);
  if (!relation.isExact())
    return std::nullopt;
  auto map = analysis::getStructuredResultMap(output);
  if (mlir::failed(map))
    return std::nullopt;
  auto iteration = analysis::IndexRelation::fromAffineMap(
      *map, producerFacts.iteratorExtents, outputType.getShape(), limits);
  if (!iteration.isExact())
    return std::nullopt;
  auto inverse = iteration.get()->inverse(limits);
  if (!inverse.isExact())
    return std::nullopt;
  auto composed = relation.get()->compose(*inverse.get(), limits);
  return composed.isExact() ? std::move(composed.relation) : std::nullopt;
}

std::optional<NodeSpatialPlan>
mapPartition(const NodeExecutionPartition &source,
             const SpatialRootDomainFacts &target, const NodeSpatialPlan &seed,
             llvm::ArrayRef<TileId> availableTiles,
             const analysis::IndexRelation &relation,
             const analysis::IndexRelationLimits &limits) {
  const size_t rank = target.iteratorExtents.size();
  auto imageLimits = limits;
  imageLimits.rectangleProof = analysis::RectangleProofMode::Construction;
  auto inverse = relation.inverse(limits);
  if (!inverse.isExact())
    return std::nullopt;
  llvm::SmallVector<analysis::StaticRectangularIndexSet, 16> images;
  llvm::SmallVector<llvm::SmallVector<IteratorInterval, 16>, 4> intervals(rank);
  for (const auto &shard : source.shards) {
    llvm::SmallVector<int64_t> offsets, sizes;
    for (const auto &axis : shard.iterationDomain) {
      offsets.push_back(axis.offset);
      sizes.push_back(axis.size);
    }
    auto image =
        relation.getExactStaticRectangularImage(offsets, sizes, imageLimits);
    if (!image.isExact() || image.domain->sizes.size() != rank)
      return std::nullopt;
    for (size_t axis = 0; axis < rank; ++axis)
      intervals[axis].push_back(
          {image.domain->offsets[axis], image.domain->sizes[axis]});
    images.push_back(std::move(*image.domain));
  }
  // A whole-target demand supplies no partition boundary to propagate.
  // Preserve the target choice; exact-demand placement can still co-locate it.
  if (llvm::all_of(images, [&](const auto &image) {
        return llvm::all_of(image.offsets,
                            [](int64_t offset) { return offset == 0; }) &&
               image.sizes == target.iteratorExtents;
      }))
    return seed;
  NodeSpatialPlan result;
  result.root = target.root;
  llvm::SmallVector<size_t, 4> optionalRetainedReductions;
  for (auto [axis, pieces] : llvm::enumerate(intervals)) {
    llvm::sort(pieces,
               [](const IteratorInterval &a, const IteratorInterval &b) {
                 return std::tie(a.offset, a.size) < std::tie(b.offset, b.size);
               });
    pieces.erase(std::unique(pieces.begin(), pieces.end()), pieces.end());
    if (pieces.empty())
      return std::nullopt;
    IteratorPartition partition{static_cast<uint32_t>(axis),
                                IteratorPartitionScheme::BalancedParts,
                                static_cast<int64_t>(pieces.size())};
    auto exact = getIteratorPartitionIntervals(target.iteratorExtents[axis],
                                               partition, source.shards.size());
    if (mlir::failed(exact) || *exact != pieces) {
      partition.scheme = IteratorPartitionScheme::UniformExtent;
      partition.parameter = pieces.front().size;
      exact = getIteratorPartitionIntervals(target.iteratorExtents[axis],
                                            partition, source.shards.size());
      if (mlir::failed(exact) || *exact != pieces)
        return std::nullopt;
    }
    // The relation does not constrain a full-extent target axis. Keep its
    // existing choice (including independent reduction axes) and refine these
    // demand rectangles by that partition instead of silently erasing it.
    if (pieces.size() == 1 && pieces.front().offset == 0 &&
        pieces.front().size == target.iteratorExtents[axis]) {
      auto retained = getIteratorPartitionIntervals(
          target.iteratorExtents[axis], seed.axes[axis], availableTiles.size());
      if (mlir::failed(retained))
        return std::nullopt;
      if (retained->size() > 1) {
        auto independent = inverse.get()->isInvariantOnDestinationDimension(
            target.iteratorExtents, axis, limits);
        if (independent.status != analysis::IndexRelationStatus::Exact ||
            !independent.value)
          return std::nullopt;
        if (*independent.value) {
          partition = seed.axes[axis];
          pieces.assign(retained->begin(), retained->end());
          if (target.iteratorKinds[axis] == SpatialIteratorKind::Reduction &&
              !isRequiredReductionAxis(target, axis))
            optionalRetainedReductions.push_back(axis);
        }
      }
    }
    intervals[axis] = pieces;
    result.axes.push_back(partition);
  }
  const auto beforeParallel = getParallelParts(target, seed);
  const auto afterParallel = getParallelParts(target, result);
  if (!beforeParallel || !afterParallel || *afterParallel < *beforeParallel)
    return std::nullopt;
  auto cellCount = [&]() -> std::optional<size_t> {
    size_t count = 1;
    for (const auto &pieces : intervals) {
      if (count > availableTiles.size() / pieces.size())
        return std::nullopt;
      count *= pieces.size();
    }
    return count;
  };
  auto count = cellCount();
  // Output parallelism outranks optional reduction alignment. Drop only
  // independently retained, optional reduction axes, in semantic axis order.
  if (!count && *afterParallel > *beforeParallel)
    for (size_t axis : optionalRetainedReductions) {
      result.axes[axis] = {static_cast<uint32_t>(axis),
                           IteratorPartitionScheme::BalancedParts, 1};
      intervals[axis] = {{0, target.iteratorExtents[axis]}};
      count = cellCount();
      if (count)
        break;
    }
  if (!count)
    return std::nullopt;
  const size_t cells = *count;
  llvm::SmallVector<llvm::SmallVector<TileId, 4>, 16> candidates(cells);
  llvm::SmallVector<llvm::SmallVector<uint32_t, 4>, 16> coordinates(cells);
  for (size_t cell = 0; cell < cells; ++cell) {
    size_t remainder = cell;
    coordinates[cell].resize(rank);
    for (size_t reverse = 0; reverse < rank; ++reverse) {
      size_t axis = rank - reverse - 1;
      coordinates[cell][axis] = remainder % intervals[axis].size();
      remainder /= intervals[axis].size();
    }
  }
  llvm::SmallVector<std::optional<TileId>, 16> embedding(cells);
  for (auto [shardIndex, image] : llvm::enumerate(images)) {
    for (size_t cell = 0; cell < cells; ++cell) {
      bool contained = true;
      for (size_t axis = 0; axis < rank; ++axis) {
        const auto &piece = intervals[axis][coordinates[cell][axis]];
        contained &= piece.offset >= image.offsets[axis] &&
                     piece.offset + piece.size <=
                         image.offsets[axis] + image.sizes[axis];
      }
      if (contained)
        candidates[cell].push_back(source.shards[shardIndex].tile);
    }
  }
  llvm::SmallVector<TileId, 16> used;
  for (size_t cell = 0; cell < cells; ++cell) {
    auto &owners = candidates[cell];
    if (owners.empty())
      return std::nullopt;
    llvm::sort(owners,
               [](TileId a, TileId b) { return a.getValue() < b.getValue(); });
    for (TileId tile : owners)
      if (!llvm::is_contained(used, tile)) {
        embedding[cell] = tile;
        used.push_back(tile);
        break;
      }
  }
  // Retained independent axes can require more cells than source owners.
  // These are explicit placement choices, not replicated source executions.
  for (size_t cell = 0; cell < cells; ++cell) {
    if (!embedding[cell]) {
      auto tile = llvm::find_if(availableTiles, [&](TileId candidate) {
        return !llvm::is_contained(used, candidate);
      });
      if (tile == availableTiles.end())
        return std::nullopt;
      embedding[cell] = *tile;
      used.push_back(*tile);
    }
    result.embedding.push_back(*embedding[cell]);
  }
  auto groups = deriveSpatialReductionGroups(target, result.axes);
  if (mlir::failed(groups))
    return std::nullopt;
  for (const ReductionGroupId &group : *groups) {
    auto previous = llvm::find_if(seed.reductionMerges, [&](const auto &merge) {
      return merge.group == group;
    });
    if (previous != seed.reductionMerges.end()) {
      result.reductionMerges.push_back(*previous);
      continue;
    }
    if (group.resultGroup >= target.resultParallelIteratorsByGroup.size())
      return std::nullopt;
    const auto &parallel =
        target.resultParallelIteratorsByGroup[group.resultGroup];
    std::optional<TileId> owner;
    for (size_t cell = 0; cell < cells; ++cell) {
      llvm::SmallVector<uint32_t, 4> projected;
      for (int axis = parallel.find_first(); axis >= 0;
           axis = parallel.find_next(axis))
        projected.push_back(coordinates[cell][axis]);
      if (projected == group.parallelCoordinate) {
        owner = result.embedding[cell];
        break;
      }
    }
    if (!owner)
      return std::nullopt;
    result.reductionMerges.push_back({group, *owner});
  }
  return result;
}

} // namespace

mlir::FailureOr<SpatialPlan> propagateSpatialPartitions(
    const SpatialPlanDomain &domain, const StructuredDAGAnalysis &dag,
    const SpatialPlan &seed, const analysis::IndexRelationLimits &limits,
    SpatialPropagationOrder order) {
  SpatialPlan result = seed;
  auto assignment = domain.close(result);
  if (mlir::failed(assignment))
    return mlir::failure();
  const auto &problem = domain.getProblem();
  std::map<SemanticRootKey, size_t> positions;
  for (auto [index, node] : llvm::enumerate(result.nodes))
    positions.emplace(node.root, index);
  auto getFacts = [&](mlir::Operation *operation) {
    const auto *binding = problem.getSemanticRoots().find(operation);
    return binding ? problem.findRoot(binding->key) : nullptr;
  };
  auto propose = [&](const StructuredDAGEdge &edge,
                     bool forward) -> std::optional<NodeSpatialPlan> {
    auto *producer = dag.getNode(edge.producer)->operation;
    auto *consumer = dag.getNode(edge.consumer)->operation;
    if (!producer || !consumer)
      return std::nullopt;
    const auto *p = getFacts(producer);
    const auto *c = getFacts(consumer);
    if (!p || !c)
      return std::nullopt;
    auto relation =
        getConsumerToProducer(consumer, edge.consumerOperand, *c, producer,
                              edge.producerResult, *p, limits);
    if (!relation)
      return std::nullopt;
    if (forward) {
      auto inverse = relation->inverse(limits);
      if (!inverse.isExact())
        return std::nullopt;
      relation = std::move(inverse.relation);
    }
    const auto &sourceRoot = forward ? p->root : c->root;
    auto source = llvm::find_if(assignment->nodes, [&](const auto &node) {
      return node.root == sourceRoot;
    });
    if (source == assignment->nodes.end())
      return std::nullopt;
    const auto &target = forward ? *c : *p;
    return mapPartition(
        *source, target, result.nodes[positions.at(target.root)],
        problem.getStructuralProblem().getAvailableTiles(), *relation, limits);
  };
  enum class Application { Unchanged, Applied };
  auto apply = [&](NodeSpatialPlan proposed) -> mlir::FailureOr<Application> {
    auto position = positions.find(proposed.root);
    if (position == positions.end())
      return mlir::failure();
    SpatialPlan next = result;
    next.nodes[position->second] = std::move(proposed);
    if (next == result || !domain.contains(next))
      return Application::Unchanged;
    auto closed = domain.close(next);
    if (mlir::failed(closed))
      return mlir::failure();
    result = std::move(next);
    assignment = std::move(closed);
    return Application::Applied;
  };
  auto forwardSweep = [&]() -> mlir::LogicalResult {
    for (const auto &node : dag.getNodes()) {
      auto *consumer = node.operation;
      if (!getFacts(consumer))
        continue;
      llvm::SmallVector<const StructuredDAGEdge *> edges;
      for (auto edgeId : node.incomingEdges) {
        const auto *edge = dag.getEdge(edgeId);
        auto &operand = consumer->getOpOperand(edge->consumerOperand);
        if (auto dps =
                mlir::dyn_cast<mlir::DestinationStyleOpInterface>(consumer);
            dps && dps.isDpsInit(&operand))
          continue;
        bool reads = false;
        if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(consumer))
          reads = linalg.payloadUsesValueFromOperand(&operand);
        else if (auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
                     consumer))
          reads = dps.isDpsInput(&operand);
        if (reads && getOperandBytes(operand.get()) != 0)
          edges.push_back(edge);
      }
      llvm::sort(edges, [&](const auto *a, const auto *b) {
        auto aBytes = getOperandBytes(consumer->getOperand(a->consumerOperand));
        auto bBytes = getOperandBytes(consumer->getOperand(b->consumerOperand));
        return aBytes != bBytes ? aBytes > bBytes
                                : std::tie(a->consumerOperand, a->id) <
                                      std::tie(b->consumerOperand, b->id);
      });
      for (const auto *edge : edges)
        if (auto proposed = propose(*edge, true)) {
          auto application = apply(std::move(*proposed));
          if (mlir::failed(application))
            return mlir::failure();
          if (*application == Application::Applied)
            break;
        }
    }
    return mlir::success();
  };
  auto backwardSweep = [&]() -> mlir::LogicalResult {
    for (const auto &node : llvm::reverse(dag.getNodes())) {
      if (!getFacts(node.operation) || node.outgoingEdges.empty())
        continue;
      std::optional<NodeSpatialPlan> common;
      for (auto edgeId : node.outgoingEdges) {
        auto proposed = propose(*dag.getEdge(edgeId), false);
        if (!proposed || (common && !(*common == *proposed))) {
          common.reset();
          break;
        }
        common = std::move(proposed);
      }
      if (common && mlir::failed(apply(std::move(*common))))
        return mlir::failure();
    }
    return mlir::success();
  };
  if (order == SpatialPropagationOrder::ProducersFirst) {
    if (mlir::failed(forwardSweep()) || mlir::failed(backwardSweep()))
      return mlir::failure();
  } else if (mlir::failed(backwardSweep()) || mlir::failed(forwardSweep())) {
    return mlir::failure();
  }
  return result;
}

} // namespace wafer::compiler::detail
