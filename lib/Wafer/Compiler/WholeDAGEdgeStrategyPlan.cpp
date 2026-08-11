//===- WholeDAGEdgeStrategyPlan.cpp - Exact edge actions ---------------===//

#include "WholeDAGEdgeStrategyPlan.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>

namespace wafer::compiler::detail {
namespace {

struct StaticRectangularDomain {
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
};

void setFailure(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
}

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef message) {
  setFailure(failureReason, message);
  return mlir::failure();
}

std::optional<uint64_t> multiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  if (product > std::numeric_limits<uint64_t>::max())
    return std::nullopt;
  return static_cast<uint64_t>(product);
}

uint64_t getElementCount(const StaticRectangularDomain &domain) {
  if (domain.offsets.size() != domain.sizes.size())
    return 0;
  uint64_t elements = 1;
  for (int64_t size : domain.sizes) {
    if (size <= 0)
      return 0;
    std::optional<uint64_t> next = multiply(elements, size);
    if (!next)
      return std::numeric_limits<uint64_t>::max();
    elements = *next;
  }
  return elements;
}

std::optional<unsigned> getElementBitWidth(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  return std::nullopt;
}

std::optional<MemLayout> getStructuredResultLayout(mlir::Operation *operation,
                                                   unsigned resultNumber) {
  if (!operation || resultNumber >= operation->getNumResults())
    return std::nullopt;
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getResult(resultNumber).getType());
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!resultType || !linalg)
    return std::nullopt;
  if (mlir::succeeded(mlir::linalg::inferConvolutionDims(linalg)) ||
      mlir::isa<mlir::linalg::BatchMatmulOp,
                mlir::linalg::BatchMatmulTransposeAOp,
                mlir::linalg::BatchMatmulTransposeBOp>(operation))
    return MemLayout::NCx;
  if (mlir::isa<mlir::linalg::MatmulOp,
                mlir::linalg::MatmulTransposeAOp,
                mlir::linalg::MatmulTransposeBOp>(operation))
    return MemLayout::Cx;
  if (llvm::is_contained(linalg.getIteratorTypesArray(),
                         mlir::utils::IteratorType::reduction))
    return resultType.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  return MemLayout::Tensor;
}

std::optional<MemLayout> getStructuredOperandLayout(mlir::Operation *operation,
                                                    unsigned operandNumber) {
  if (!operation || operandNumber >= operation->getNumOperands())
    return std::nullopt;
  auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getOperand(operandNumber).getType());
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!operandType || !linalg)
    return std::nullopt;
  if (mlir::succeeded(mlir::linalg::inferConvolutionDims(linalg)) ||
      mlir::isa<mlir::linalg::BatchMatmulOp,
                mlir::linalg::BatchMatmulTransposeAOp,
                mlir::linalg::BatchMatmulTransposeBOp>(operation))
    return MemLayout::NCx;
  if (mlir::isa<mlir::linalg::MatmulOp,
                mlir::linalg::MatmulTransposeAOp,
                mlir::linalg::MatmulTransposeBOp>(operation))
    return MemLayout::Cx;
  if (llvm::is_contained(linalg.getIteratorTypesArray(),
                         mlir::utils::IteratorType::reduction))
    return operandType.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  return MemLayout::Tensor;
}

void assignPhysicalLayouts(SpatialEdgeStrategy &strategy) {
  std::optional<MemLayout> producer = getStructuredResultLayout(
      strategy.producer, strategy.producerResult);
  std::optional<MemLayout> consumer = getStructuredOperandLayout(
      strategy.consumer, strategy.consumerOperand);
  if (!producer || !consumer)
    return;
  strategy.hasLayoutAssignment = true;
  strategy.producerLayout = *producer;
  strategy.consumerLayout = *consumer;
}

mlir::FailureOr<StaticRectangularDomain>
getBalancedShard(mlir::RankedTensorType type, unsigned shardDimension,
                 llvm::ArrayRef<PhysicalTileId> group, PhysicalTileId tile,
                 std::string *failureReason) {
  if (!type || !type.hasStaticShape() || type.getRank() == 0 ||
      shardDimension >= static_cast<unsigned>(type.getRank()) || group.empty())
    return fail<StaticRectangularDomain>(
        failureReason, "dependent edge placement has an invalid static shard");
  auto found = llvm::find(group, tile);
  if (found == group.end())
    return fail<StaticRectangularDomain>(
        failureReason, "dependent edge shard Tile is outside its node group");
  const int64_t extent = type.getDimSize(shardDimension);
  if (extent <= 0 || group.size() > static_cast<size_t>(extent))
    return fail<StaticRectangularDomain>(
        failureReason,
        "dependent edge placement cannot form nonempty balanced shards");
  const int64_t ordinal = std::distance(group.begin(), found);
  const int64_t participants = static_cast<int64_t>(group.size());
  const int64_t base = extent / participants;
  const int64_t larger = extent % participants;

  StaticRectangularDomain result;
  result.offsets.assign(type.getRank(), 0);
  result.sizes.assign(type.getShape().begin(), type.getShape().end());
  result.offsets[shardDimension] =
      ordinal * base + std::min<int64_t>(ordinal, larger);
  result.sizes[shardDimension] = base + (ordinal < larger);
  return result;
}

std::optional<StaticRectangularDomain>
intersectDomains(const StaticRectangularDomain &lhs,
                 const StaticRectangularDomain &rhs) {
  if (lhs.offsets.size() != rhs.offsets.size() ||
      lhs.sizes.size() != rhs.sizes.size() ||
      lhs.offsets.size() != lhs.sizes.size())
    return std::nullopt;
  StaticRectangularDomain result;
  for (size_t dimension = 0; dimension < lhs.offsets.size(); ++dimension) {
    if (lhs.offsets[dimension] < 0 || rhs.offsets[dimension] < 0 ||
        lhs.sizes[dimension] <= 0 || rhs.sizes[dimension] <= 0)
      return std::nullopt;
    if (lhs.sizes[dimension] >
            std::numeric_limits<int64_t>::max() - lhs.offsets[dimension] ||
        rhs.sizes[dimension] >
            std::numeric_limits<int64_t>::max() - rhs.offsets[dimension])
      return std::nullopt;
    const int64_t begin =
        std::max(lhs.offsets[dimension], rhs.offsets[dimension]);
    const int64_t lhsEnd = lhs.offsets[dimension] + lhs.sizes[dimension];
    const int64_t rhsEnd = rhs.offsets[dimension] + rhs.sizes[dimension];
    const int64_t end = std::min(lhsEnd, rhsEnd);
    if (begin >= end)
      return std::nullopt;
    result.offsets.push_back(begin);
    result.sizes.push_back(end - begin);
  }
  return result;
}

struct ExactEdgeRelation {
  mlir::RankedTensorType producerType;
  mlir::RankedTensorType consumerResultType;
  mlir::AffineMap resultToProducer;
};

mlir::FailureOr<ExactEdgeRelation>
deriveExactEdgeRelation(const CardDAGAnalysis &dag, const CardDAGEdge &edge,
                        std::string *failureReason) {
  const CardDAGNode *producerNode = dag.getNode(edge.producer);
  const CardDAGNode *consumerNode = dag.getNode(edge.consumer);
  if (!producerNode || !consumerNode || !producerNode->operation ||
      !consumerNode->operation)
    return fail<ExactEdgeRelation>(failureReason,
                                   "dependent edge references an unknown node");
  mlir::Operation *producer = producerNode->operation;
  mlir::Operation *consumer = consumerNode->operation;
  if (edge.producerResult >= producer->getNumResults() ||
      edge.consumerOperand >= consumer->getNumOperands() ||
      consumer->getOperand(edge.consumerOperand) !=
          producer->getResult(edge.producerResult))
    return fail<ExactEdgeRelation>(
        failureReason,
        "dependent peer placement requires a direct current-SSA edge");
  if (!mlir::isa<mlir::TilingInterface>(producer) ||
      !mlir::isa<mlir::TilingInterface>(consumer) ||
      !mlir::isa<mlir::DestinationStyleOpInterface>(consumer))
    return fail<ExactEdgeRelation>(
        failureReason,
        "dependent peer placement requires TilingInterface DPS nodes");

  auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
      producer->getResult(edge.producerResult).getType());
  auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(
      consumer->getOperand(edge.consumerOperand).getType());
  auto consumerLinalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(consumer);
  if (!producerType || !operandType || producerType != operandType ||
      !producerType.hasStaticShape() || !consumerLinalg ||
      consumer->getNumResults() != 1 || consumerLinalg.getNumDpsInits() != 1)
    return fail<ExactEdgeRelation>(
        failureReason,
        "dependent peer placement requires one static direct Linalg result");
  auto consumerResultType =
      mlir::dyn_cast<mlir::RankedTensorType>(consumer->getResult(0).getType());
  llvm::SmallVector<int64_t, 4> loopShape =
      consumerLinalg.getStaticLoopRanges();
  if (!consumerResultType || !consumerResultType.hasStaticShape() ||
      llvm::any_of(loopShape, [](int64_t extent) { return extent <= 0; }))
    return fail<ExactEdgeRelation>(
        failureReason, "dependent peer relation requires static loop ranges");

  llvm::SmallVector<mlir::AffineMap, 4> maps =
      consumerLinalg.getIndexingMapsArray();
  const unsigned outputMapIndex = consumerLinalg.getNumDpsInputs();
  if (edge.consumerOperand >= maps.size() || outputMapIndex >= maps.size())
    return fail<ExactEdgeRelation>(failureReason,
                                   "dependent edge indexing map is missing");
  analysis::IndexRelationResult iterationToResult =
      analysis::IndexRelation::fromAffineMap(maps[outputMapIndex], loopShape,
                                             consumerResultType.getShape());
  analysis::IndexRelationResult iterationToProducer =
      analysis::IndexRelation::fromAffineMap(
          maps[edge.consumerOperand], loopShape, producerType.getShape());
  if (!iterationToResult.isExact() || !iterationToProducer.isExact())
    return fail<ExactEdgeRelation>(
        failureReason, "dependent edge indexing relation is not exact");
  analysis::IndexRelationResult resultToIteration =
      iterationToResult.get()->inverse();
  if (!resultToIteration.isExact())
    return fail<ExactEdgeRelation>(
        failureReason,
        "consumer result does not invert to its iteration domain");
  analysis::IndexRelationResult resultToProducer =
      resultToIteration.get()->compose(*iterationToProducer.get());
  if (!resultToProducer.isExact() ||
      !resultToProducer.get()->isFunctional().isProvenTrue())
    return fail<ExactEdgeRelation>(
        failureReason,
        "consumer shard does not have one exact producer demand relation");
  // Recover the executable rectangular map directly from the same structured
  // indexing maps after the Presburger composition above proved the relation
  // exact.  PresburgerRelation does not always canonicalize even an identity
  // composition back into a projected AffineMap, so treating that printer
  // limitation as semantic uncertainty would reject ordinary pointwise DAGs.
  mlir::AffineMap resultToIterationMap =
      mlir::inversePermutation(maps[outputMapIndex]);
  if (!resultToIterationMap)
    return fail<ExactEdgeRelation>(
        failureReason,
        "consumer result indexing does not invert to its iteration domain");
  mlir::AffineMap affine =
      maps[edge.consumerOperand].compose(resultToIterationMap);
  if (!affine || !affine.isProjectedPermutation(/*allowZeroInResults=*/true))
    return fail<ExactEdgeRelation>(failureReason,
                                   "dependent producer demand is not one "
                                   "rectangular projected permutation");
  analysis::IndexRelationResult executableRelation =
      analysis::IndexRelation::fromAffineMap(
          affine, consumerResultType.getShape(), producerType.getShape());
  if (!executableRelation.isExact() ||
      !executableRelation.get()
           ->isEquivalentTo(*resultToProducer.get())
           .isProvenTrue())
    return fail<ExactEdgeRelation>(
        failureReason,
        "structured affine demand differs from the exact SSA index relation");
  return ExactEdgeRelation{producerType, consumerResultType, affine};
}

class ExactEdgeRelationCache {
public:
  mlir::FailureOr<const ExactEdgeRelation *>
  get(const CardDAGAnalysis &dag, const CardDAGEdge &edge,
      std::string *failureReason) {
    auto [iterator, inserted] = entries.try_emplace(edge.id);
    Entry &entry = iterator->second;
    if (inserted) {
      std::string relationFailure;
      mlir::FailureOr<ExactEdgeRelation> relation =
          deriveExactEdgeRelation(dag, edge, &relationFailure);
      if (mlir::succeeded(relation))
        entry.relation.emplace(std::move(*relation));
      else
        entry.failureReason = std::move(relationFailure);
    }
    if (!entry.relation) {
      setFailure(failureReason, entry.failureReason);
      return mlir::failure();
    }
    return &*entry.relation;
  }

private:
  struct Entry {
    std::optional<ExactEdgeRelation> relation;
    std::string failureReason;
  };
  std::unordered_map<CardDAGEdgeID, Entry> entries;
};

mlir::FailureOr<StaticRectangularDomain>
mapConsumerShardToProducer(const ExactEdgeRelation &relation,
                           const StaticRectangularDomain &consumerShard,
                           std::string *failureReason) {
  if (consumerShard.offsets.size() !=
          static_cast<size_t>(relation.consumerResultType.getRank()) ||
      consumerShard.sizes.size() != consumerShard.offsets.size() ||
      relation.resultToProducer.getNumDims() != consumerShard.offsets.size() ||
      relation.resultToProducer.getNumResults() !=
          static_cast<unsigned>(relation.producerType.getRank()))
    return fail<StaticRectangularDomain>(
        failureReason, "dependent producer demand rank is inconsistent");

  StaticRectangularDomain demand;
  for (mlir::AffineExpr expression : relation.resultToProducer.getResults()) {
    if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
      demand.offsets.push_back(consumerShard.offsets[dim.getPosition()]);
      demand.sizes.push_back(consumerShard.sizes[dim.getPosition()]);
      continue;
    }
    auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression);
    if (!constant || constant.getValue() != 0)
      return fail<StaticRectangularDomain>(failureReason,
                                           "dependent producer demand is not a "
                                           "rectangular zero-based broadcast");
    demand.offsets.push_back(0);
    demand.sizes.push_back(1);
  }
  return demand;
}

/// Returns true only when equal physical Tile groups also own the same
/// semantic shard.  Numeric result dimensions are not a sufficient proof:
/// for a transpose, producer dimension 0 and consumer result dimension 0 are
/// carried by different structured iterators and therefore require peer
/// fragments even though both placements spell `shardDimension = 0`.
///
/// This iterator-level check intentionally remains weaker than the exact
/// non-local relation below.  In particular, it admits a reduction when its
/// result-parallel iterator is aligned with the producer shard while the
/// reduction iterators remain wholly local.  Such an edge cannot be inverted
/// from the consumer result to one producer point, but it is still a valid
/// same-Tile residency.
bool hasSemanticallyAlignedLocalShard(
    const CardDAGEdge &edge, const CardDAGNode &producerNode,
    const CardDAGNode &consumerNode,
    const WholeDAGNodePlacement &producerPlacement,
    const WholeDAGNodePlacement &consumerPlacement) {
  if (producerPlacement.tiles != consumerPlacement.tiles)
    return false;

  auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
      producerNode.operation->getResult(edge.producerResult).getType());
  auto consumerResultType =
      consumerNode.operation->getNumResults() == 1
          ? mlir::dyn_cast<mlir::RankedTensorType>(
                consumerNode.operation->getResult(0).getType())
          : mlir::RankedTensorType{};
  auto consumer =
      mlir::dyn_cast<mlir::linalg::LinalgOp>(consumerNode.operation);
  if (!producerType || !consumerResultType || !consumer ||
      producerPlacement.shardDimension >=
          static_cast<unsigned>(producerType.getRank()) ||
      consumerPlacement.shardDimension >=
          static_cast<unsigned>(consumerResultType.getRank()))
    return false;

  llvm::SmallVector<mlir::AffineMap, 4> maps =
      consumer.getIndexingMapsArray();
  const unsigned outputMapIndex = consumer.getNumDpsInputs();
  if (edge.consumerOperand >= maps.size() || outputMapIndex >= maps.size())
    return false;
  mlir::AffineMap operandMap = maps[edge.consumerOperand];
  mlir::AffineMap outputMap = maps[outputMapIndex];
  if (producerPlacement.shardDimension >= operandMap.getNumResults() ||
      consumerPlacement.shardDimension >= outputMap.getNumResults())
    return false;

  auto consumerIterator = mlir::dyn_cast<mlir::AffineDimExpr>(
      outputMap.getResult(consumerPlacement.shardDimension));
  auto producerIterator = mlir::dyn_cast<mlir::AffineDimExpr>(
      operandMap.getResult(producerPlacement.shardDimension));
  return consumerIterator && producerIterator &&
         consumerIterator.getPosition() == producerIterator.getPosition();
}

mlir::LogicalResult
appendEdgeStrategyPlan(const CardDAGAnalysis &dag, const CardDAGEdge &edge,
                       const WholeDAGNodePlacement &producerPlacement,
                       const WholeDAGNodePlacement &consumerPlacement,
                       WholeDAGEdgeStrategyPlan *result,
                       ExactEdgeRelationCache *relationCache,
                       std::string *failureReason) {
  if (producerPlacement.node != edge.producer ||
      consumerPlacement.node != edge.consumer ||
      producerPlacement.tiles.empty() || consumerPlacement.tiles.empty()) {
    setFailure(failureReason,
               "dependent edge plan has inconsistent node placements");
    return mlir::failure();
  }
  const CardDAGNode *producerNode = dag.getNode(edge.producer);
  const CardDAGNode *consumerNode = dag.getNode(edge.consumer);
  if (!producerNode || !producerNode->operation || !consumerNode ||
      !consumerNode->operation ||
      edge.producerResult >= producerNode->operation->getNumResults() ||
      edge.consumerOperand >= consumerNode->operation->getNumOperands()) {
    setFailure(failureReason,
               "dependent edge source or destination is unavailable");
    return mlir::failure();
  }
  // Only direct current-SSA edges carry a spatial action.  DPS init
  // operands are initialization state, and DAG edges that reach the nearest
  // structured producer through a pure support chain (view ops) are resolved
  // by the consumer's typed lowering inside its own traversal.  Both are
  // still ordered through wave readiness by the scheduler, but the
  // card-program materialization contract gives them no edge strategy.
  if (consumerNode->operation->getOperand(edge.consumerOperand) !=
      producerNode->operation->getResult(edge.producerResult))
    return mlir::success();
  auto consumerDps =
      mlir::dyn_cast<mlir::DestinationStyleOpInterface>(consumerNode->operation);
  const bool isDataInput =
      consumerDps &&
      llvm::any_of(consumerDps.getDpsInputOperands(),
                   [&](mlir::OpOperand *operand) {
                     return operand->getOperandNumber() == edge.consumerOperand;
                   });
  if (!isDataInput)
    return mlir::success();
  auto directProducerType = mlir::dyn_cast<mlir::RankedTensorType>(
      producerNode->operation->getResult(edge.producerResult).getType());
  std::optional<unsigned> directElementBits =
      directProducerType
          ? getElementBitWidth(directProducerType.getElementType())
          : std::nullopt;
  if (!directProducerType || !directProducerType.hasStaticShape() ||
      !directElementBits || *directElementBits == 0 ||
      *directElementBits % 8 != 0) {
    setFailure(failureReason,
               "dependent edge residency requires a static byte-addressable "
               "producer result");
    return mlir::failure();
  }

  if (hasSemanticallyAlignedLocalShard(edge, *producerNode, *consumerNode,
                                       producerPlacement,
                                       consumerPlacement)) {
    for (PhysicalTileId tile : producerPlacement.tiles) {
      mlir::FailureOr<StaticRectangularDomain> producerShard =
          getBalancedShard(directProducerType, producerPlacement.shardDimension,
                           producerPlacement.tiles, tile, failureReason);
      if (mlir::failed(producerShard))
        return mlir::failure();
      std::optional<uint64_t> residentBytes =
          multiply(getElementCount(*producerShard), *directElementBits / 8);
      if (!residentBytes || *residentBytes == 0) {
        setFailure(failureReason,
                   "dependent local residency footprint is not representable");
        return mlir::failure();
      }
      if (result) {
        SpatialEdgeStrategy strategy;
        strategy.producer = producerNode->operation;
        strategy.producerResult = edge.producerResult;
        strategy.consumer = consumerNode->operation;
        strategy.consumerOperand = edge.consumerOperand;
        strategy.producerOffsets = producerShard->offsets;
        strategy.producerSizes = producerShard->sizes;
        strategy.sourceTile = tile;
        strategy.destinationTile = tile;
        strategy.action = SpatialEdgeAction::CoupledFusion;
        assignPhysicalLayouts(strategy);
        result->strategies.push_back(std::move(strategy));
      }
    }
    return mlir::success();
  }

  std::optional<ExactEdgeRelation> ownedRelation;
  const ExactEdgeRelation *relation = nullptr;
  if (relationCache) {
    mlir::FailureOr<const ExactEdgeRelation *> cached =
        relationCache->get(dag, edge, failureReason);
    if (mlir::failed(cached))
      return mlir::failure();
    relation = *cached;
  } else {
    mlir::FailureOr<ExactEdgeRelation> derived =
        deriveExactEdgeRelation(dag, edge, failureReason);
    if (mlir::failed(derived))
      return mlir::failure();
    ownedRelation.emplace(std::move(*derived));
    relation = &*ownedRelation;
  }
  if (producerPlacement.shardDimension >=
          static_cast<unsigned>(relation->producerType.getRank()) ||
      consumerPlacement.shardDimension >=
          static_cast<unsigned>(relation->consumerResultType.getRank())) {
    setFailure(failureReason,
               "dependent edge shard dimension is outside its node");
    return mlir::failure();
  }
  std::optional<unsigned> elementBits =
      getElementBitWidth(relation->producerType.getElementType());
  if (!elementBits || *elementBits == 0 || *elementBits % 8 != 0) {
    setFailure(failureReason,
               "dependent peer transfer requires a byte-addressable element "
               "type");
    return mlir::failure();
  }

  uint32_t nextPayloadSlice = 0;
  uint64_t totalPeerBytes = result ? result->totalPeerBytes : 0;
  for (PhysicalTileId destination : consumerPlacement.tiles) {
    mlir::FailureOr<StaticRectangularDomain> consumerShard = getBalancedShard(
        relation->consumerResultType, consumerPlacement.shardDimension,
        consumerPlacement.tiles, destination, failureReason);
    if (mlir::failed(consumerShard))
      return mlir::failure();
    mlir::FailureOr<StaticRectangularDomain> demand =
        mapConsumerShardToProducer(*relation, *consumerShard, failureReason);
    if (mlir::failed(demand))
      return mlir::failure();

    SpatialEdgeStrategy strategy;
    if (result) {
      strategy.producer = producerNode->operation;
      strategy.producerResult = edge.producerResult;
      strategy.consumer = consumerNode->operation;
      strategy.consumerOperand = edge.consumerOperand;
      strategy.consumerOffsets = consumerShard->offsets;
      strategy.consumerSizes = consumerShard->sizes;
      strategy.producerOffsets = demand->offsets;
      strategy.producerSizes = demand->sizes;
      strategy.sourceTile = destination;
      strategy.destinationTile = destination;
      strategy.action = SpatialEdgeAction::PeerFragments;
      assignPhysicalLayouts(strategy);
    }

    uint64_t coveredElements = 0;
    uint64_t fragmentCount = 0;
    for (PhysicalTileId source : producerPlacement.tiles) {
      mlir::FailureOr<StaticRectangularDomain> sourceShard = getBalancedShard(
          relation->producerType, producerPlacement.shardDimension,
          producerPlacement.tiles, source, failureReason);
      if (mlir::failed(sourceShard))
        return mlir::failure();
      std::optional<StaticRectangularDomain> intersection =
          intersectDomains(*demand, *sourceShard);
      if (!intersection)
        continue;
      const uint64_t elements = getElementCount(*intersection);
      if (elements == 0 ||
          elements > std::numeric_limits<uint64_t>::max() - coveredElements) {
        setFailure(failureReason,
                   "dependent edge rectangular coverage is not representable");
        return mlir::failure();
      }
      coveredElements += elements;
      ++fragmentCount;
      if (source == destination) {
        std::optional<uint64_t> residentBytes =
            multiply(elements, *elementBits / 8);
        if (!residentBytes || *residentBytes == 0) {
          setFailure(
              failureReason,
              "dependent local residency footprint is not representable");
          return mlir::failure();
        }
        if (result)
          strategy.fragments.push_back(SpatialEdgeFragment{
              SpatialEdgeFragmentKind::Resident, intersection->offsets,
              intersection->sizes, source, /*bytes=*/0,
              /*communicationId=*/0, /*payloadSlice=*/0});
        continue;
      }
      std::optional<uint64_t> bytes = multiply(elements, *elementBits / 8);
      if (!bytes || *bytes == 0 ||
          *bytes > std::numeric_limits<uint32_t>::max() ||
          nextPayloadSlice == std::numeric_limits<uint32_t>::max()) {
        setFailure(failureReason,
                   "dependent peer payload is not representable by target DTE");
        return mlir::failure();
      }
      if (result)
        strategy.fragments.push_back(SpatialEdgeFragment{
            SpatialEdgeFragmentKind::Peer, intersection->offsets,
            intersection->sizes, source, *bytes,
            static_cast<int64_t>(edge.id),
            static_cast<int64_t>(nextPayloadSlice)});
      ++nextPayloadSlice;
      if (*bytes >
          std::numeric_limits<uint64_t>::max() - totalPeerBytes) {
        setFailure(failureReason, "dependent peer byte work overflows");
        return mlir::failure();
      }
      totalPeerBytes += *bytes;
      if (result)
        result->totalPeerBytes = totalPeerBytes;
    }
    if (coveredElements != getElementCount(*demand)) {
      setFailure(
          failureReason,
          "dependent source shards do not exactly cover consumer demand");
      return mlir::failure();
    }
    if (fragmentCount == 0) {
      setFailure(failureReason,
                 "dependent edge strategy has no exact source fragment");
      return mlir::failure();
    }
    if (result)
      result->strategies.push_back(std::move(strategy));
  }
  return mlir::success();
}

} // namespace

class WholeDAGEdgeStrategyPlanner::Impl {
public:
  explicit Impl(const CardDAGAnalysis &dag) : dag(dag) {}

  const CardDAGAnalysis &dag;
  ExactEdgeRelationCache relationCache;
};

WholeDAGEdgeStrategyPlanner::WholeDAGEdgeStrategyPlanner(
    const CardDAGAnalysis &dag)
    : impl(std::make_unique<Impl>(dag)) {}

WholeDAGEdgeStrategyPlanner::~WholeDAGEdgeStrategyPlanner() = default;
WholeDAGEdgeStrategyPlanner::WholeDAGEdgeStrategyPlanner(
    WholeDAGEdgeStrategyPlanner &&) noexcept = default;
WholeDAGEdgeStrategyPlanner &WholeDAGEdgeStrategyPlanner::operator=(
    WholeDAGEdgeStrategyPlanner &&) noexcept = default;

mlir::FailureOr<WholeDAGEdgeStrategyPlan>
WholeDAGEdgeStrategyPlanner::derive(
    CardDAGEdgeID edgeID,
    const WholeDAGNodePlacement &producerPlacement,
    const WholeDAGNodePlacement &consumerPlacement,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  const CardDAGEdge *edge = impl->dag.getEdge(edgeID);
  if (!edge)
    return fail<WholeDAGEdgeStrategyPlan>(
        failureReason, "dependent edge plan references an unknown edge");
  WholeDAGEdgeStrategyPlan result;
  if (mlir::failed(appendEdgeStrategyPlan(
          impl->dag, *edge, producerPlacement, consumerPlacement, &result,
          &impl->relationCache, failureReason)))
    return mlir::failure();
  return result;
}

mlir::LogicalResult WholeDAGEdgeStrategyPlanner::verify(
    CardDAGEdgeID edgeID,
    const WholeDAGNodePlacement &producerPlacement,
    const WholeDAGNodePlacement &consumerPlacement,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  const CardDAGEdge *edge = impl->dag.getEdge(edgeID);
  if (!edge) {
    setFailure(failureReason,
               "dependent edge plan references an unknown edge");
    return mlir::failure();
  }
  return appendEdgeStrategyPlan(impl->dag, *edge, producerPlacement,
                                consumerPlacement, nullptr,
                                &impl->relationCache, failureReason);
}

uint64_t countPeerFragments(const WholeDAGEdgeStrategyPlan &plan) {
  uint64_t result = 0;
  for (const SpatialEdgeStrategy &strategy : plan.strategies)
    for (const SpatialEdgeFragment &fragment : strategy.fragments)
      if (fragment.kind == SpatialEdgeFragmentKind::Peer &&
          result != std::numeric_limits<uint64_t>::max())
        ++result;
  return result;
}

mlir::LogicalResult verifyWholeDAGEdgeNonLocalRelation(
    const CardDAGAnalysis &dag, CardDAGEdgeID edgeID,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  const CardDAGEdge *edge = dag.getEdge(edgeID);
  if (!edge) {
    setFailure(failureReason,
               "dependent edge plan references an unknown edge");
    return mlir::failure();
  }
  const CardDAGNode *producerNode = dag.getNode(edge->producer);
  const CardDAGNode *consumerNode = dag.getNode(edge->consumer);
  if (!producerNode || !producerNode->operation || !consumerNode ||
      !consumerNode->operation ||
      edge->producerResult >= producerNode->operation->getNumResults() ||
      edge->consumerOperand >= consumerNode->operation->getNumOperands()) {
    setFailure(failureReason,
               "dependent edge source or destination is unavailable");
    return mlir::failure();
  }
  if (consumerNode->operation->getOperand(edge->consumerOperand) !=
      producerNode->operation->getResult(edge->producerResult))
    return mlir::success();
  auto consumerDps =
      mlir::dyn_cast<mlir::DestinationStyleOpInterface>(consumerNode->operation);
  const bool isDataInput =
      consumerDps &&
      llvm::any_of(consumerDps.getDpsInputOperands(),
                   [&](mlir::OpOperand *operand) {
                     return operand->getOperandNumber() == edge->consumerOperand;
                   });
  if (!isDataInput)
    return mlir::success();
  auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
      producerNode->operation->getResult(edge->producerResult).getType());
  std::optional<unsigned> elementBits =
      producerType ? getElementBitWidth(producerType.getElementType())
                   : std::nullopt;
  if (!producerType || !producerType.hasStaticShape() || !elementBits ||
      *elementBits == 0 || *elementBits % 8 != 0) {
    setFailure(failureReason,
               "dependent edge residency requires a static byte-addressable "
               "producer result");
    return mlir::failure();
  }
  return mlir::succeeded(deriveExactEdgeRelation(dag, *edge, failureReason))
             ? mlir::success()
             : mlir::failure();
}

mlir::FailureOr<WholeDAGEdgeStrategyPlan>
deriveWholeDAGEdgeStrategyPlan(const CardDAGAnalysis &dag, CardDAGEdgeID edgeID,
                               const WholeDAGNodePlacement &producerPlacement,
                               const WholeDAGNodePlacement &consumerPlacement,
                               std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  const CardDAGEdge *edge = dag.getEdge(edgeID);
  if (!edge)
    return fail<WholeDAGEdgeStrategyPlan>(
        failureReason, "dependent edge plan references an unknown edge");
  WholeDAGEdgeStrategyPlan result;
  if (mlir::failed(appendEdgeStrategyPlan(dag, *edge, producerPlacement,
                                          consumerPlacement, &result, nullptr,
                                          failureReason)))
    return mlir::failure();
  return result;
}

mlir::FailureOr<WholeDAGEdgeStrategyPlan> deriveWholeDAGEdgeStrategyPlan(
    const CardDAGAnalysis &dag,
    llvm::ArrayRef<WholeDAGNodePlacement> requestedPlacements,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (requestedPlacements.size() != dag.getNodes().size())
    return fail<WholeDAGEdgeStrategyPlan>(
        failureReason,
        "dependent edge strategy plan must place every DAG node once");
  llvm::SmallVector<const WholeDAGNodePlacement *, 16> placements(
      dag.getNodes().size(), nullptr);
  for (const WholeDAGNodePlacement &placement : requestedPlacements) {
    if (placement.node >= placements.size() || placements[placement.node] ||
        placement.tiles.empty())
      return fail<WholeDAGEdgeStrategyPlan>(
          failureReason,
          "dependent edge strategy plan has an invalid or duplicate node "
          "placement");
    placements[placement.node] = &placement;
  }

  WholeDAGEdgeStrategyPlan result;
  for (const CardDAGEdge &edge : dag.getEdges()) {
    const WholeDAGNodePlacement &producerPlacement = *placements[edge.producer];
    const WholeDAGNodePlacement &consumerPlacement = *placements[edge.consumer];
    if (mlir::failed(appendEdgeStrategyPlan(dag, edge, producerPlacement,
                                            consumerPlacement, &result, nullptr,
                                            failureReason)))
      return mlir::failure();
  }
  return result;
}

} // namespace wafer::compiler::detail
