//===- StructuredDAGEdgeStrategyPlan.cpp - Exact edge actions ---------------===//

#include "StructuredDAGEdgeStrategyPlan.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <memory>
#include <optional>
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
  if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::MatmulTransposeAOp,
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
  if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::MatmulTransposeAOp,
                mlir::linalg::MatmulTransposeBOp>(operation))
    return MemLayout::Cx;
  if (llvm::is_contained(linalg.getIteratorTypesArray(),
                         mlir::utils::IteratorType::reduction))
    return operandType.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  return MemLayout::Tensor;
}

void assignPhysicalLayouts(SpatialEdgeStrategy &strategy) {
  std::optional<MemLayout> producer =
      getStructuredResultLayout(strategy.producer, strategy.producerResult);
  std::optional<MemLayout> consumer =
      getStructuredOperandLayout(strategy.consumer, strategy.consumerOperand);
  if (!producer || !consumer)
    return;
  strategy.hasLayoutAssignment = true;
  strategy.producerLayout = *producer;
  strategy.consumerLayout = *consumer;
}

mlir::FailureOr<StructuredDAGEdgeStrategyPlan>
lowerDemandPlanToCanonicalStrategies(const StructuredDAGAnalysis &dag,
                                     const StructuredDAGEdgeDemandPlan &demandPlan,
                                     std::string *failureReason) {
  StructuredDAGEdgeStrategyPlan result;
  std::unordered_map<StructuredDAGEdgeID, uint32_t> nextPayloadSlice;
  for (const StructuredDAGEdgeDemand &demand : demandPlan.demands) {
    const StructuredDAGEdge *edge = dag.getEdge(demand.edge);
    const StructuredDAGNode *producerNode =
        edge ? dag.getNode(edge->producer) : nullptr;
    const StructuredDAGNode *consumerNode =
        edge ? dag.getNode(edge->consumer) : nullptr;
    if (!edge || !producerNode || !producerNode->operation || !consumerNode ||
        !consumerNode->operation ||
        edge->producerResult >= producerNode->operation->getNumResults() ||
        edge->consumerOperand >= consumerNode->operation->getNumOperands())
      return fail<StructuredDAGEdgeStrategyPlan>(
          failureReason,
          "canonical edge lowering cannot resolve current SSA edge");
    analysis::IndexSetResult consumerSet{
        analysis::IndexRelationStatus::Exact, demand.consumerDomain, {}};
    analysis::IndexSetResult producerSet{
        analysis::IndexRelationStatus::Exact, demand.producerDemand, {}};
    analysis::StaticRectangularIndexSetResult consumerRectangle =
        consumerSet.getExactStaticRectangularDomain();
    analysis::StaticRectangularIndexSetResult producerRectangle =
        producerSet.getExactStaticRectangularDomain();
    if (!consumerRectangle.isExact() || !producerRectangle.isExact())
      return fail<StructuredDAGEdgeStrategyPlan>(
          failureReason,
          "canonical edge lowering requires one dense logical rectangle");

    auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
        producerNode->operation->getResult(edge->producerResult).getType());
    std::optional<unsigned> elementBits =
        producerType ? getElementBitWidth(producerType.getElementType())
                     : std::nullopt;
    if (!producerType || !elementBits || *elementBits == 0 ||
        *elementBits % 8 != 0)
      return fail<StructuredDAGEdgeStrategyPlan>(
          failureReason,
          "canonical edge lowering requires a byte-addressable tensor");

    SpatialEdgeStrategy strategy;
    strategy.producer = producerNode->operation;
    strategy.producerResult = edge->producerResult;
    strategy.consumer = consumerNode->operation;
    strategy.consumerOperand = edge->consumerOperand;
    strategy.consumerOffsets = consumerRectangle.domain->offsets;
    strategy.consumerSizes = consumerRectangle.domain->sizes;
    strategy.producerOffsets = producerRectangle.domain->offsets;
    strategy.producerSizes = producerRectangle.domain->sizes;
    strategy.sourceTile = demand.destinationTile;
    strategy.destinationTile = demand.destinationTile;
    assignPhysicalLayouts(strategy);

    std::optional<mlir::presburger::PresburgerSet> covered;
    bool hasRemoteFragment = false;
    uint64_t fragmentCount = 0;
    for (const StructuredDAGEdgeProducerShardOwnership &ownership :
         demand.producerShardOwnership) {
      mlir::presburger::PresburgerSet intersection =
          demand.producerDemand.intersect(ownership.logicalDomain);
      if (intersection.isIntegerEmpty())
        continue;
      analysis::IndexSetResult intersectionSet{
          analysis::IndexRelationStatus::Exact, intersection, {}};
      analysis::StaticRectangularIndexSetResult rectangle =
          intersectionSet.getExactStaticRectangularDomain();
      if (!rectangle.isExact())
        return fail<StructuredDAGEdgeStrategyPlan>(
            failureReason,
            "canonical edge lowering cannot represent an exact shard "
            "intersection");
      StaticRectangularDomain domain{rectangle.domain->offsets,
                                     rectangle.domain->sizes};
      const uint64_t elements = getElementCount(domain);
      std::optional<uint64_t> bytes = multiply(elements, *elementBits / 8);
      if (!bytes || *bytes == 0)
        return fail<StructuredDAGEdgeStrategyPlan>(
            failureReason,
            "canonical edge fragment byte size is not representable");
      covered = covered ? covered->unionSet(intersection) : intersection;
      ++fragmentCount;
      if (ownership.tile == demand.destinationTile) {
        strategy.fragments.push_back(SpatialEdgeFragment{
            SpatialEdgeFragmentKind::Resident, rectangle.domain->offsets,
            rectangle.domain->sizes, ownership.tile, /*bytes=*/0,
            /*communicationId=*/0, /*payloadSlice=*/0});
        continue;
      }
      uint32_t &payloadSlice = nextPayloadSlice[demand.edge];
      if (*bytes > std::numeric_limits<uint32_t>::max() ||
          payloadSlice == std::numeric_limits<uint32_t>::max())
        return fail<StructuredDAGEdgeStrategyPlan>(
            failureReason,
            "canonical peer payload is not representable by target DTE");
      strategy.fragments.push_back(SpatialEdgeFragment{
          SpatialEdgeFragmentKind::Peer, rectangle.domain->offsets,
          rectangle.domain->sizes, ownership.tile, *bytes,
          static_cast<int64_t>(demand.edge),
          static_cast<int64_t>(payloadSlice++)});
      hasRemoteFragment = true;
      if (*bytes > std::numeric_limits<uint64_t>::max() - result.totalPeerBytes)
        return fail<StructuredDAGEdgeStrategyPlan>(failureReason,
                                              "peer byte work overflows");
      result.totalPeerBytes += *bytes;
    }
    if (fragmentCount == 0 || !covered ||
        !covered->isEqual(demand.producerDemand))
      return fail<StructuredDAGEdgeStrategyPlan>(
          failureReason,
          "canonical edge fragments do not exactly cover logical demand");
    if (hasRemoteFragment) {
      strategy.action = SpatialEdgeAction::PeerFragments;
    } else {
      strategy.action = SpatialEdgeAction::LocalShardResidency;
      strategy.fragments.clear();
    }
    result.strategies.push_back(std::move(strategy));
  }
  return result;
}

} // namespace

mlir::FailureOr<StructuredDAGEdgeStrategyPlan>
lowerStructuredDAGEdgeDemandPlanToCanonicalStrategies(
    const StructuredDAGAnalysis &dag, const StructuredDAGEdgeDemandPlan &demandPlan,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  return lowerDemandPlanToCanonicalStrategies(dag, demandPlan, failureReason);
}

class StructuredDAGEdgeStrategyPlanner::Impl {
public:
  explicit Impl(const StructuredDAGAnalysis &dag) : dag(dag), demandPlanner(dag) {}

  const StructuredDAGAnalysis &dag;
  StructuredDAGEdgeDemandPlanner demandPlanner;
};

StructuredDAGEdgeStrategyPlanner::StructuredDAGEdgeStrategyPlanner(
    const StructuredDAGAnalysis &dag)
    : impl(std::make_unique<Impl>(dag)) {}

StructuredDAGEdgeStrategyPlanner::~StructuredDAGEdgeStrategyPlanner() = default;
StructuredDAGEdgeStrategyPlanner::StructuredDAGEdgeStrategyPlanner(
    StructuredDAGEdgeStrategyPlanner &&) noexcept = default;
StructuredDAGEdgeStrategyPlanner &StructuredDAGEdgeStrategyPlanner::operator=(
    StructuredDAGEdgeStrategyPlanner &&) noexcept = default;

mlir::FailureOr<StructuredDAGEdgeStrategyPlan> StructuredDAGEdgeStrategyPlanner::derive(
    StructuredDAGEdgeID edgeID, const StructuredDAGNodePlacement &producerPlacement,
    const StructuredDAGNodePlacement &consumerPlacement,
    std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  mlir::FailureOr<StructuredDAGEdgeDemandPlan> demandPlan =
      impl->demandPlanner.derive(edgeID, producerPlacement, consumerPlacement,
                                 failureReason);
  if (mlir::failed(demandPlan))
    return mlir::failure();
  return lowerStructuredDAGEdgeDemandPlanToCanonicalStrategies(
      impl->dag, *demandPlan, failureReason);
}

mlir::FailureOr<StructuredDAGEdgeStrategyPlan> StructuredDAGEdgeStrategyPlanner::derive(
    llvm::ArrayRef<StructuredDAGNodePlacement> requestedPlacements,
    std::string *failureReason) {
  mlir::FailureOr<StructuredDAGEdgeDemandPlan> demandPlan =
      impl->demandPlanner.derive(requestedPlacements, failureReason);
  if (mlir::failed(demandPlan))
    return mlir::failure();
  return lowerStructuredDAGEdgeDemandPlanToCanonicalStrategies(
      impl->dag, *demandPlan, failureReason);
}

uint64_t countPeerFragments(const StructuredDAGEdgeStrategyPlan &plan) {
  uint64_t result = 0;
  for (const SpatialEdgeStrategy &strategy : plan.strategies)
    for (const SpatialEdgeFragment &fragment : strategy.fragments)
      if (fragment.kind == SpatialEdgeFragmentKind::Peer &&
          result != std::numeric_limits<uint64_t>::max())
        ++result;
  return result;
}

mlir::FailureOr<StructuredDAGEdgeStrategyPlan>
deriveStructuredDAGEdgeStrategyPlan(const StructuredDAGAnalysis &dag, StructuredDAGEdgeID edgeID,
                               const StructuredDAGNodePlacement &producerPlacement,
                               const StructuredDAGNodePlacement &consumerPlacement,
                               std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  mlir::FailureOr<StructuredDAGEdgeDemandPlan> demandPlan =
      deriveStructuredDAGEdgeDemandPlan(dag, edgeID, producerPlacement,
                                   consumerPlacement, failureReason);
  if (mlir::failed(demandPlan))
    return mlir::failure();
  return lowerStructuredDAGEdgeDemandPlanToCanonicalStrategies(dag, *demandPlan,
                                                          failureReason);
}

mlir::FailureOr<StructuredDAGEdgeStrategyPlan> deriveStructuredDAGEdgeStrategyPlan(
    const StructuredDAGAnalysis &dag,
    llvm::ArrayRef<StructuredDAGNodePlacement> requestedPlacements,
    std::string *failureReason) {
  StructuredDAGEdgeStrategyPlanner planner(dag);
  return planner.derive(requestedPlacements, failureReason);
}

} // namespace wafer::compiler::detail
