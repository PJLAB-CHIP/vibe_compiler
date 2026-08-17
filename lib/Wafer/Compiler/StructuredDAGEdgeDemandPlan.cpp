//===- StructuredDAGEdgeDemandPlan.cpp - Exact logical edge demand
//---------===//

#include "StructuredDAGEdgeDemandPlan.h"

#include "StructuredDAGExactDemandQuery.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <memory>

namespace wafer::compiler::detail {
namespace {

void setFailure(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
}

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef message) {
  setFailure(failureReason, message);
  return mlir::failure();
}

mlir::LogicalResult appendEdgeDemandPlan(
    const StructuredDAGAnalysis &dag, const StructuredDAGEdge &edge,
    const StructuredDAGNodePlacement &producerPlacement,
    const StructuredDAGNodePlacement &consumerPlacement,
    StructuredDAGEdgeDemandPlan *result, StructuredDAGExactDemandQuery *query,
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

  const bool direct =
      consumerNode->operation->getOperand(edge.consumerOperand) ==
      producerNode->operation->getResult(edge.producerResult);
  auto consumerDps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
      consumerNode->operation);
  const bool isDataInput =
      consumerDps &&
      llvm::any_of(consumerDps.getDpsInputOperands(),
                   [&](mlir::OpOperand *operand) {
                     return operand->getOperandNumber() == edge.consumerOperand;
                   });

  // Typed logical gate: the four-state exact-demand query proves whether the
  // edge carries a legal demand for this closed trial. Only Satisfied direct
  // data-input edges are assembled into canonical per-Tile demands below;
  // init and support dependencies remain owned by the consumer typed
  // lowering, and their logical verdicts are consumed by the placement
  // legality owner, not by this carrier adapter.
  std::string trialFailure;
  mlir::FailureOr<analysis::LogicalShardTrial> trial =
      buildEdgeShardTrial(dag, producerPlacement, consumerPlacement,
                          query->getEpoch(), &trialFailure);
  if (mlir::failed(trial)) {
    setFailure(failureReason, trialFailure);
    return mlir::failure();
  }
  analysis::ExactDemandResult demand = query->query(edge.id, *trial);
  if (direct && isDataInput &&
      demand.status != analysis::ExactDemandStatus::Satisfied) {
    setFailure(failureReason,
               demand.detail.empty()
                   ? "edge demand does not form an exact coverage proof"
                   : demand.detail);
    return mlir::failure();
  }
  if (!direct || !isDataInput)
    return mlir::success();

  return assembleStructuredDAGEdgeDemandPlan(dag, edge, producerPlacement,
                                             consumerPlacement, *trial, demand,
                                             result, failureReason);
}

} // namespace

mlir::LogicalResult assembleStructuredDAGEdgeDemandPlan(
    const StructuredDAGAnalysis &dag, const StructuredDAGEdge &edge,
    const StructuredDAGNodePlacement &producerPlacement,
    const StructuredDAGNodePlacement &consumerPlacement,
    const analysis::LogicalShardTrial &trial,
    const analysis::ExactDemandResult &demand,
    StructuredDAGEdgeDemandPlan *result, std::string *failureReason) {
  if (demand.status != analysis::ExactDemandStatus::Satisfied) {
    setFailure(failureReason,
               "canonical edge assembly requires a satisfied demand verdict");
    return mlir::failure();
  }
  const StructuredDAGNode *producerNode = dag.getNode(edge.producer);
  const StructuredDAGNode *consumerNode = dag.getNode(edge.consumer);
  if (!producerNode || !producerNode->operation || !consumerNode ||
      !consumerNode->operation) {
    setFailure(failureReason, "dependent edge endpoint is unavailable");
    return mlir::failure();
  }
  // Init and support dependencies stay owned by the consumer typed
  // lowering; the carrier assembles no spatial action for them.
  const bool direct =
      consumerNode->operation->getOperand(edge.consumerOperand) ==
      producerNode->operation->getResult(edge.producerResult);
  auto consumerDps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
      consumerNode->operation);
  const bool isDataInput =
      consumerDps &&
      llvm::any_of(consumerDps.getDpsInputOperands(),
                   [&](mlir::OpOperand *operand) {
                     return operand->getOperandNumber() == edge.consumerOperand;
                   });
  if (!direct || !isDataInput)
    return mlir::success();
  // Canonical carrier assembly consumes the query's typed facts only: the
  // producer's full per-result ownership from the trial and the exact
  // per-destination demand from the query. The carrier never re-derives the
  // edge relation; the query owns the logical boundary.
  const analysis::LogicalNodeTrial *producerTrial = nullptr;
  const analysis::LogicalNodeTrial *consumerTrial = nullptr;
  for (const analysis::LogicalNodeTrial &nodeTrial : trial.nodes) {
    if (nodeTrial.node == edge.producer)
      producerTrial = &nodeTrial;
    if (nodeTrial.node == edge.consumer)
      consumerTrial = &nodeTrial;
  }
  if (!producerTrial || !consumerTrial) {
    setFailure(failureReason, "dependent edge trial misses an endpoint");
    return mlir::failure();
  }
  if (demand.perDestination.empty()) {
    setFailure(failureReason,
               "canonical edge demand has no destination shard facts");
    return mlir::failure();
  }
  llvm::SmallVector<TileId, 16> expectedDestinations(
      consumerPlacement.tiles.begin(), consumerPlacement.tiles.end());
  llvm::SmallVector<TileId, 16> actualDestinations;
  actualDestinations.reserve(demand.perDestination.size());
  for (const analysis::ExactDestinationDemand &destination :
       demand.perDestination)
    actualDestinations.push_back(destination.destinationTile);
  auto tileLess = [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  };
  llvm::sort(expectedDestinations, tileLess);
  llvm::sort(actualDestinations, tileLess);
  if (expectedDestinations != actualDestinations ||
      std::adjacent_find(expectedDestinations.begin(),
                         expectedDestinations.end()) !=
          expectedDestinations.end()) {
    setFailure(failureReason,
               "canonical edge demand does not cover every destination Tile");
    return mlir::failure();
  }

  llvm::SmallVector<StructuredDAGEdgeProducerShardOwnership, 4> ownership;
  for (const analysis::LogicalTileBinding &binding : producerTrial->bindings) {
    if (binding.resultIndex != edge.producerResult || !binding.ownedDomain)
      continue;
    ownership.push_back(StructuredDAGEdgeProducerShardOwnership{
        binding.tile, *binding.ownedDomain});
  }
  if (ownership.empty()) {
    setFailure(failureReason,
               "dependent edge producer ownership is unavailable");
    return mlir::failure();
  }

  for (const analysis::ExactDestinationDemand &destination :
       demand.perDestination) {
    // Satisfied implies the per-shard witness is empty; a non-empty witness
    // here would be a query defect, so fail closed.
    if (destination.uncoveredWitness &&
        !destination.uncoveredWitness->isIntegerEmpty()) {
      setFailure(failureReason,
                 "producer shard ownership does not cover exact demand");
      return mlir::failure();
    }
    const analysis::LogicalTileBinding *consumerShard = nullptr;
    for (const analysis::LogicalTileBinding &binding :
         consumerTrial->bindings) {
      if (binding.tile == destination.destinationTile &&
          binding.resultIndex == 0 && binding.ownedDomain) {
        consumerShard = &binding;
        break;
      }
    }
    if (!consumerShard || !destination.producerDemand) {
      setFailure(failureReason,
                 "dependent edge destination shard is unavailable");
      return mlir::failure();
    }
    if (result)
      result->demands.push_back(StructuredDAGEdgeDemand{
          edge.id, destination.destinationTile, *consumerShard->ownedDomain,
          *destination.producerDemand, ownership});
  }
  return mlir::success();
}

class StructuredDAGEdgeDemandPlanner::Impl {
public:
  Impl(const StructuredDAGAnalysis &dag, analysis::IREpoch epoch)
      : dag(dag), query(dag, epoch) {}

  const StructuredDAGAnalysis &dag;
  StructuredDAGExactDemandQuery query;
};

StructuredDAGEdgeDemandPlanner::StructuredDAGEdgeDemandPlanner(
    const StructuredDAGAnalysis &dag, analysis::IREpoch epoch)
    : impl(std::make_unique<Impl>(dag, epoch)) {}

StructuredDAGEdgeDemandPlanner::~StructuredDAGEdgeDemandPlanner() = default;
StructuredDAGEdgeDemandPlanner::StructuredDAGEdgeDemandPlanner(
    StructuredDAGEdgeDemandPlanner &&) noexcept = default;
StructuredDAGEdgeDemandPlanner &StructuredDAGEdgeDemandPlanner::operator=(
    StructuredDAGEdgeDemandPlanner &&) noexcept = default;

mlir::FailureOr<StructuredDAGEdgeDemandPlan>
StructuredDAGEdgeDemandPlanner::derive(
    StructuredDAGEdgeID edgeID,
    const StructuredDAGNodePlacement &producerPlacement,
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
                                        &impl->query, failureReason)))
    return mlir::failure();
  return result;
}

mlir::FailureOr<StructuredDAGEdgeDemandPlan>
StructuredDAGEdgeDemandPlanner::derive(
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
    const StructuredDAGNodePlacement &producerPlacement =
        *placements[edge.producer];
    const StructuredDAGNodePlacement &consumerPlacement =
        *placements[edge.consumer];
    if (mlir::failed(appendEdgeDemandPlan(impl->dag, edge, producerPlacement,
                                          consumerPlacement, &result,
                                          &impl->query, failureReason)))
      return mlir::failure();
  }
  return result;
}

mlir::FailureOr<StructuredDAGEdgeDemandPlan> deriveStructuredDAGEdgeDemandPlan(
    const StructuredDAGAnalysis &dag, StructuredDAGEdgeID edgeID,
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
