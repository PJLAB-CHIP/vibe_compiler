//===- SpatialPartitionPropagation.cpp - SSA-coordinated proposals --------===//

#include "Wafer/Planning/PhysicalDataflow/SpatialPartitionPropagation.h"
#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <map>
#include <optional>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

bool isParallel(const SpatialRootDomainFacts &root) {
  return llvm::all_of(root.iteratorKinds, [](SpatialIteratorKind kind) {
    return kind == SpatialIteratorKind::Parallel;
  });
}

uint64_t getOperandBytes(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return 0;
  uint64_t bytes = llvm::divideCeil(type.getElementTypeBitWidth(), 8u);
  for (int64_t extent : type.getShape())
    bytes = llvm::SaturatingMultiply(bytes, static_cast<uint64_t>(extent));
  return bytes;
}

// Compose the actual operand path. A multi-source support op is deliberately
// not treated as a reshape or a whole-tensor identity.
std::optional<analysis::IndexRelation>
getConsumerToProducer(mlir::linalg::LinalgOp consumer, unsigned operand,
                      const SpatialRootDomainFacts &consumerFacts,
                      mlir::linalg::LinalgOp producer, unsigned result,
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
  auto iteration = analysis::IndexRelation::fromAffineMap(
      producer.getIndexingMapMatchingResult(output),
      producerFacts.iteratorExtents, outputType.getShape(), limits);
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
             const SpatialRootDomainFacts &target,
             const analysis::IndexRelation &relation,
             const analysis::IndexRelationLimits &limits) {
  const size_t rank = target.iteratorExtents.size();
  llvm::SmallVector<analysis::StaticRectangularIndexSet, 16> images;
  llvm::SmallVector<llvm::SmallVector<IteratorInterval, 16>, 4> intervals(rank);
  for (const auto &shard : source.shards) {
    llvm::SmallVector<int64_t> offsets, sizes;
    for (const auto &axis : shard.iterationDomain) {
      offsets.push_back(axis.offset);
      sizes.push_back(axis.size);
    }
    auto image =
        relation.getExactStaticRectangularImage(offsets, sizes, limits);
    if (!image.isExact() || image.domain->sizes.size() != rank)
      return std::nullopt;
    for (size_t axis = 0; axis < rank; ++axis)
      intervals[axis].push_back(
          {image.domain->offsets[axis], image.domain->sizes[axis]});
    images.push_back(std::move(*image.domain));
  }
  NodeSpatialPlan result;
  result.root = target.root;
  size_t cells = 1;
  for (auto [axis, pieces] : llvm::enumerate(intervals)) {
    llvm::sort(pieces,
               [](const IteratorInterval &a, const IteratorInterval &b) {
                 return std::tie(a.offset, a.size) < std::tie(b.offset, b.size);
               });
    pieces.erase(std::unique(pieces.begin(), pieces.end()), pieces.end());
    if (pieces.empty() || cells > source.shards.size() / pieces.size())
      return std::nullopt;
    cells *= pieces.size();
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
      if (mlir::failed(exact) || *exact != pieces) {
        // Neither parameterized scheme reproduces the mapped rectangles, but
        // their interior boundaries are exactly the cut points of an explicit
        // partition. This is the only producer of that scheme: the raw
        // successor never enumerates it.
        partition.scheme = IteratorPartitionScheme::ExplicitBounds;
        partition.parameter = 1;
        partition.bounds.clear();
        for (size_t index = 0; index + 1 < pieces.size(); ++index)
          partition.bounds.push_back(pieces[index].getEnd());
        exact = getIteratorPartitionIntervals(target.iteratorExtents[axis],
                                              partition, source.shards.size());
        if (mlir::failed(exact) || *exact != pieces)
          return std::nullopt;
      }
    }
    result.axes.push_back(partition);
  }
  if (cells != source.shards.size())
    return std::nullopt;
  llvm::SmallVector<std::optional<TileId>, 16> embedding(cells);
  for (auto [shardIndex, image] : llvm::enumerate(images)) {
    size_t cell = 0;
    for (size_t axis = 0; axis < rank; ++axis) {
      auto &pieces = intervals[axis];
      auto found = llvm::find(
          pieces, IteratorInterval{image.offsets[axis], image.sizes[axis]});
      if (found == pieces.end())
        return std::nullopt;
      cell = cell * pieces.size() + std::distance(pieces.begin(), found);
    }
    if (embedding[cell])
      return std::nullopt;
    embedding[cell] = source.shards[shardIndex].tile;
  }
  for (auto tile : embedding) {
    if (!tile || llvm::is_contained(result.embedding, *tile))
      return std::nullopt;
    result.embedding.push_back(*tile);
  }
  return result;
}

} // namespace

mlir::FailureOr<SpatialPlan> propagateSpatialPartitions(
    const SpatialPlanDomain &domain, const StructuredDAGAnalysis &dag,
    const SpatialPlan &seed, const analysis::IndexRelationLimits &limits) {
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
    auto producer = mlir::dyn_cast<mlir::linalg::LinalgOp>(
        dag.getNode(edge.producer)->operation);
    auto consumer = mlir::dyn_cast<mlir::linalg::LinalgOp>(
        dag.getNode(edge.consumer)->operation);
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
    return mapPartition(*source, forward ? *c : *p, *relation, limits);
  };
  auto apply = [&](NodeSpatialPlan proposed) -> mlir::LogicalResult {
    auto position = positions.find(proposed.root);
    if (position == positions.end())
      return mlir::failure();
    SpatialPlan next = result;
    next.nodes[position->second] = std::move(proposed);
    if (!domain.contains(next))
      return mlir::success();
    auto closed = domain.close(next);
    if (mlir::failed(closed))
      return mlir::failure();
    result = std::move(next);
    assignment = std::move(closed);
    return mlir::success();
  };
  for (const auto &node : dag.getNodes()) {
    auto consumer = mlir::dyn_cast<mlir::linalg::LinalgOp>(node.operation);
    const auto *facts = getFacts(node.operation);
    if (!consumer || !facts || !isParallel(*facts))
      continue;
    llvm::SmallVector<const StructuredDAGEdge *> edges;
    for (auto edgeId : node.incomingEdges) {
      const auto *edge = dag.getEdge(edgeId);
      auto &operand = consumer->getOpOperand(edge->consumerOperand);
      if (consumer.payloadUsesValueFromOperand(&operand) &&
          getOperandBytes(operand.get()) != 0)
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
        if (mlir::failed(apply(std::move(*proposed))))
          return mlir::failure();
        break;
      }
  }
  for (const auto &node : llvm::reverse(dag.getNodes())) {
    auto producer = mlir::dyn_cast<mlir::linalg::LinalgOp>(node.operation);
    const auto *facts = getFacts(node.operation);
    if (!producer || !facts || !isParallel(*facts) ||
        node.outgoingEdges.empty())
      continue;
    if (llvm::any_of(producer->getOpOperands(), [&](mlir::OpOperand &operand) {
          return producer.payloadUsesValueFromOperand(&operand) &&
                 mlir::isa<mlir::ShapedType>(operand.get().getType());
        }))
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
  return result;
}

} // namespace wafer::compiler::detail
