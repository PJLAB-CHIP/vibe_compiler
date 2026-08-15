//===- StructuredDAGEdgeDemandPlan.cpp - Exact logical edge demand ---------===//

#include "StructuredDAGEdgeDemandPlan.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

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

mlir::FailureOr<StaticRectangularDomain>
getBalancedShard(mlir::RankedTensorType type, unsigned shardDimension,
                 llvm::ArrayRef<TileId> group, TileId tile,
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

struct ExactEdgeRelation {
  mlir::RankedTensorType producerType;
  mlir::RankedTensorType consumerResultType;
  analysis::IndexRelation resultToProducer;
  struct DemandCacheEntry {
    std::optional<mlir::presburger::PresburgerSet> demand;
    std::string failureReason;
  };
  mutable std::map<std::vector<int64_t>, DemandCacheEntry> demandCache;
};

mlir::FailureOr<ExactEdgeRelation>
deriveExactEdgeRelation(const StructuredDAGAnalysis &dag, const StructuredDAGEdge &edge,
                        std::string *failureReason) {
  const StructuredDAGNode *producerNode = dag.getNode(edge.producer);
  const StructuredDAGNode *consumerNode = dag.getNode(edge.consumer);
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
        "dependent demand requires a direct current-SSA edge");
  if (!mlir::isa<mlir::TilingInterface>(producer) ||
      !mlir::isa<mlir::TilingInterface>(consumer) ||
      !mlir::isa<mlir::DestinationStyleOpInterface>(consumer))
    return fail<ExactEdgeRelation>(
        failureReason,
        "dependent demand requires TilingInterface DPS nodes");

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
        "dependent demand requires one static direct Linalg result");
  auto consumerResultType =
      mlir::dyn_cast<mlir::RankedTensorType>(consumer->getResult(0).getType());
  llvm::SmallVector<int64_t, 4> loopShape =
      consumerLinalg.getStaticLoopRanges();
  if (!consumerResultType || !consumerResultType.hasStaticShape() ||
      llvm::any_of(loopShape, [](int64_t extent) { return extent <= 0; }))
    return fail<ExactEdgeRelation>(
        failureReason, "dependent demand requires static loop ranges");

  llvm::SmallVector<mlir::AffineMap, 4> maps =
      consumerLinalg.getIndexingMapsArray();
  const unsigned outputMapIndex = consumerLinalg.getNumDpsInputs();
  if (edge.consumerOperand >= maps.size() || outputMapIndex >= maps.size())
    return fail<ExactEdgeRelation>(failureReason,
                                   "dependent edge indexing map is missing");
  analysis::IndexRelationResult resultToProducer =
      analysis::IndexRelation::fromCommonIterationDomain(
          maps[outputMapIndex], consumerResultType.getShape(),
          maps[edge.consumerOperand], producerType.getShape(), loopShape);
  if (!resultToProducer.isExact())
    return fail<ExactEdgeRelation>(
        failureReason,
        "consumer shard does not have an exact producer demand relation");
  return ExactEdgeRelation{producerType, consumerResultType,
                           std::move(*resultToProducer.relation)};
}

class ExactEdgeRelationCache {
public:
  mlir::FailureOr<const ExactEdgeRelation *> get(const StructuredDAGAnalysis &dag,
                                                 const StructuredDAGEdge &edge,
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
  std::unordered_map<StructuredDAGEdgeID, Entry> entries;
};

mlir::FailureOr<mlir::presburger::PresburgerSet>
mapConsumerShardToProducer(const ExactEdgeRelation &relation,
                           const StaticRectangularDomain &consumerShard,
                           std::string *failureReason) {
  if (consumerShard.offsets.size() !=
          static_cast<size_t>(relation.consumerResultType.getRank()) ||
      consumerShard.sizes.size() != consumerShard.offsets.size() ||
      relation.resultToProducer.getDestinationRank() !=
          consumerShard.offsets.size() ||
      relation.resultToProducer.getSourceRank() !=
          static_cast<unsigned>(relation.producerType.getRank()))
    return fail<mlir::presburger::PresburgerSet>(
        failureReason, "dependent producer demand rank is inconsistent");

  std::vector<int64_t> cacheKey(consumerShard.offsets.begin(),
                                consumerShard.offsets.end());
  cacheKey.insert(cacheKey.end(), consumerShard.sizes.begin(),
                  consumerShard.sizes.end());
  auto cached = relation.demandCache.find(cacheKey);
  if (cached != relation.demandCache.end()) {
    if (cached->second.demand)
      return *cached->second.demand;
    return fail<mlir::presburger::PresburgerSet>(failureReason,
                                                 cached->second.failureReason);
  }
  auto cacheFailure = [&](llvm::StringRef reason)
      -> mlir::FailureOr<mlir::presburger::PresburgerSet> {
    relation.demandCache.emplace(
        std::move(cacheKey),
        ExactEdgeRelation::DemandCacheEntry{std::nullopt, reason.str()});
    return fail<mlir::presburger::PresburgerSet>(failureReason, reason);
  };

  analysis::IndexSetResult consumerDomain =
      analysis::IndexRelation::staticRectangularDomain(consumerShard.offsets,
                                                       consumerShard.sizes);
  if (!consumerDomain.isExact())
    return cacheFailure(consumerDomain.reason);
  analysis::IndexSetResult demand =
      relation.resultToProducer.image(*consumerDomain.set);
  if (!demand.isExact())
    return cacheFailure(demand.reason.empty()
                            ? "consumer shard has no exact producer demand"
                            : demand.reason);
  relation.demandCache.emplace(
      std::move(cacheKey),
      ExactEdgeRelation::DemandCacheEntry{*demand.set, /*failureReason=*/{}});
  return *demand.set;
}

mlir::LogicalResult
appendEdgeDemandPlan(const StructuredDAGAnalysis &dag, const StructuredDAGEdge &edge,
                     const StructuredDAGNodePlacement &producerPlacement,
                     const StructuredDAGNodePlacement &consumerPlacement,
                     StructuredDAGEdgeDemandPlan *result,
                     ExactEdgeRelationCache *relationCache,
                     std::string *failureReason) {
  if (producerPlacement.node != edge.producer ||
      consumerPlacement.node != edge.consumer ||
      producerPlacement.tiles.empty() || consumerPlacement.tiles.empty()) {
    setFailure(failureReason,
               "dependent edge demand has inconsistent node placements");
    return mlir::failure();
  }
  const StructuredDAGNode *producerNode = dag.getNode(edge.producer);
  const StructuredDAGNode *consumerNode = dag.getNode(edge.consumer);
  if (!producerNode || !producerNode->operation || !consumerNode ||
      !consumerNode->operation ||
      edge.producerResult >= producerNode->operation->getNumResults() ||
      edge.consumerOperand >= consumerNode->operation->getNumOperands()) {
    setFailure(failureReason,
               "dependent edge source or destination is unavailable");
    return mlir::failure();
  }
  if (consumerNode->operation->getOperand(edge.consumerOperand) !=
      producerNode->operation->getResult(edge.producerResult))
    return mlir::success();
  auto consumerDps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
      consumerNode->operation);
  const bool isDataInput =
      consumerDps &&
      llvm::any_of(consumerDps.getDpsInputOperands(),
                   [&](mlir::OpOperand *operand) {
                     return operand->getOperandNumber() == edge.consumerOperand;
                   });
  if (!isDataInput)
    return mlir::success();

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

  llvm::SmallVector<StructuredDAGEdgeProducerShardOwnership, 4> ownership;
  std::optional<mlir::presburger::PresburgerSet> ownedDomain;
  for (TileId source : producerPlacement.tiles) {
    mlir::FailureOr<StaticRectangularDomain> sourceShard = getBalancedShard(
        relation->producerType, producerPlacement.shardDimension,
        producerPlacement.tiles, source, failureReason);
    if (mlir::failed(sourceShard))
      return mlir::failure();
    analysis::IndexSetResult sourceDomain =
        analysis::IndexRelation::staticRectangularDomain(sourceShard->offsets,
                                                         sourceShard->sizes);
    if (!sourceDomain.isExact()) {
      setFailure(failureReason, sourceDomain.reason);
      return mlir::failure();
    }
    ownedDomain = ownedDomain ? ownedDomain->unionSet(*sourceDomain.set)
                              : *sourceDomain.set;
    ownership.push_back(StructuredDAGEdgeProducerShardOwnership{
        source, std::move(*sourceDomain.set)});
  }

  for (TileId destination : consumerPlacement.tiles) {
    mlir::FailureOr<StaticRectangularDomain> consumerShard = getBalancedShard(
        relation->consumerResultType, consumerPlacement.shardDimension,
        consumerPlacement.tiles, destination, failureReason);
    if (mlir::failed(consumerShard))
      return mlir::failure();
    analysis::IndexSetResult consumerDomain =
        analysis::IndexRelation::staticRectangularDomain(consumerShard->offsets,
                                                         consumerShard->sizes);
    if (!consumerDomain.isExact()) {
      setFailure(failureReason, consumerDomain.reason);
      return mlir::failure();
    }
    mlir::FailureOr<mlir::presburger::PresburgerSet> demand =
        mapConsumerShardToProducer(*relation, *consumerShard, failureReason);
    if (mlir::failed(demand))
      return mlir::failure();
    if (!ownedDomain || !demand->subtract(*ownedDomain).isIntegerEmpty()) {
      setFailure(failureReason,
                 "producer shard ownership does not cover exact demand");
      return mlir::failure();
    }
    if (result)
      result->demands.push_back(StructuredDAGEdgeDemand{
          edge.id, destination, std::move(*consumerDomain.set),
          std::move(*demand), ownership});
  }
  return mlir::success();
}

} // namespace

class StructuredDAGEdgeDemandPlanner::Impl {
public:
  explicit Impl(const StructuredDAGAnalysis &dag) : dag(dag) {}

  const StructuredDAGAnalysis &dag;
  ExactEdgeRelationCache relationCache;
};

StructuredDAGEdgeDemandPlanner::StructuredDAGEdgeDemandPlanner(const StructuredDAGAnalysis &dag)
    : impl(std::make_unique<Impl>(dag)) {}

StructuredDAGEdgeDemandPlanner::~StructuredDAGEdgeDemandPlanner() = default;
StructuredDAGEdgeDemandPlanner::StructuredDAGEdgeDemandPlanner(
    StructuredDAGEdgeDemandPlanner &&) noexcept = default;
StructuredDAGEdgeDemandPlanner &StructuredDAGEdgeDemandPlanner::operator=(
    StructuredDAGEdgeDemandPlanner &&) noexcept = default;

mlir::FailureOr<StructuredDAGEdgeDemandPlan> StructuredDAGEdgeDemandPlanner::derive(
    StructuredDAGEdgeID edgeID, const StructuredDAGNodePlacement &producerPlacement,
    const StructuredDAGNodePlacement &consumerPlacement,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  const StructuredDAGEdge *edge = impl->dag.getEdge(edgeID);
  if (!edge)
    return fail<StructuredDAGEdgeDemandPlan>(
        failureReason, "dependent edge demand references an unknown edge");
  StructuredDAGEdgeDemandPlan result;
  if (mlir::failed(appendEdgeDemandPlan(impl->dag, *edge, producerPlacement,
                                        consumerPlacement, &result,
                                        &impl->relationCache, failureReason)))
    return mlir::failure();
  return result;
}

mlir::FailureOr<StructuredDAGEdgeDemandPlan> StructuredDAGEdgeDemandPlanner::derive(
    llvm::ArrayRef<StructuredDAGNodePlacement> requestedPlacements,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (requestedPlacements.size() != impl->dag.getNodes().size())
    return fail<StructuredDAGEdgeDemandPlan>(
        failureReason,
        "dependent edge demand plan must place every DAG node once");
  llvm::SmallVector<const StructuredDAGNodePlacement *, 16> placements(
      impl->dag.getNodes().size(), nullptr);
  for (const StructuredDAGNodePlacement &placement : requestedPlacements) {
    if (placement.node >= placements.size() || placements[placement.node] ||
        placement.tiles.empty())
      return fail<StructuredDAGEdgeDemandPlan>(
          failureReason,
          "dependent edge demand plan has an invalid or duplicate node "
          "placement");
    placements[placement.node] = &placement;
  }

  StructuredDAGEdgeDemandPlan result;
  for (const StructuredDAGEdge &edge : impl->dag.getEdges()) {
    const StructuredDAGNodePlacement &producerPlacement = *placements[edge.producer];
    const StructuredDAGNodePlacement &consumerPlacement = *placements[edge.consumer];
    if (mlir::failed(appendEdgeDemandPlan(impl->dag, edge, producerPlacement,
                                          consumerPlacement, &result,
                                          &impl->relationCache, failureReason)))
      return mlir::failure();
  }
  return result;
}

mlir::FailureOr<StructuredDAGEdgeDemandPlan>
deriveStructuredDAGEdgeDemandPlan(const StructuredDAGAnalysis &dag, StructuredDAGEdgeID edgeID,
                             const StructuredDAGNodePlacement &producerPlacement,
                             const StructuredDAGNodePlacement &consumerPlacement,
                             std::string *failureReason) {
  StructuredDAGEdgeDemandPlanner planner(dag);
  return planner.derive(edgeID, producerPlacement, consumerPlacement,
                        failureReason);
}

mlir::FailureOr<StructuredDAGEdgeDemandPlan> deriveStructuredDAGEdgeDemandPlan(
    const StructuredDAGAnalysis &dag,
    llvm::ArrayRef<StructuredDAGNodePlacement> requestedPlacements,
    std::string *failureReason) {
  StructuredDAGEdgeDemandPlanner planner(dag);
  return planner.derive(requestedPlacements, failureReason);
}

} // namespace wafer::compiler::detail
