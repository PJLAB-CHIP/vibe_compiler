//===- DependentDataflow.cpp - Selected edge-action lowering ------------===//

#include "SelectedEdgeLoweringInternal.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

using namespace wafer;
using namespace wafer::tensor_program_to_tile_region;

namespace {

struct StaticBoxRef {
  llvm::ArrayRef<int64_t> offsets;
  llvm::ArrayRef<int64_t> sizes;
};

std::optional<uint64_t> getBoxVolume(const StaticBoxRef &box) {
  if (box.offsets.size() != box.sizes.size())
    return std::nullopt;
  uint64_t result = 1;
  for (auto [offset, size] : llvm::zip_equal(box.offsets, box.sizes)) {
    int64_t end = 0;
    if (offset < 0 || size <= 0 || llvm::AddOverflow(offset, size, end) ||
        static_cast<uint64_t>(size) >
            std::numeric_limits<uint64_t>::max() / result)
      return std::nullopt;
    result *= static_cast<uint64_t>(size);
  }
  return result;
}

std::optional<uint64_t> getIntersectionVolume(const StaticBoxRef &lhs,
                                              const StaticBoxRef &rhs) {
  if (lhs.offsets.size() != rhs.offsets.size() ||
      lhs.sizes.size() != rhs.sizes.size())
    return std::nullopt;
  uint64_t result = 1;
  for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
       llvm::zip_equal(lhs.offsets, lhs.sizes, rhs.offsets, rhs.sizes)) {
    int64_t lhsEnd = 0;
    int64_t rhsEnd = 0;
    if (llvm::AddOverflow(lhsOffset, lhsSize, lhsEnd) ||
        llvm::AddOverflow(rhsOffset, rhsSize, rhsEnd))
      return std::nullopt;
    const int64_t begin = std::max(lhsOffset, rhsOffset);
    const int64_t end = std::min(lhsEnd, rhsEnd);
    if (begin >= end)
      return uint64_t{0};
    const uint64_t size = static_cast<uint64_t>(end - begin);
    if (size > std::numeric_limits<uint64_t>::max() / result)
      return std::nullopt;
    result *= size;
  }
  return result;
}

mlir::LogicalResult
validateStrategyDemand(const SpatialEdgeStrategy &strategy,
                       const analysis::ExactIndexSet &requiredDomain,
                       std::string *failureReason) {
  const analysis::IndexRelationLimits limits;
  mlir::FailureOr<analysis::ExactIndexSet> normalizedRequired =
      analysis::normalizeFiniteExactIndexSet(requiredDomain);
  if (mlir::failed(normalizedRequired)) {
    setFailureReason(failureReason,
                     "selected carrier demand is not a finite box union");
    return mlir::failure();
  }
  if (normalizedRequired->getBoxes().size() > limits.maxRectangularPieces) {
    setFailureReason(failureReason,
                     "selected carrier demand exceeds comparison work limit");
    return mlir::failure();
  }
  llvm::SmallVector<StaticBoxRef, 16> pieces;
  if (strategy.action == SpatialEdgeAction::PeerFragments ||
      strategy.action == SpatialEdgeAction::CardDDRTransfer) {
    if (strategy.fragments.size() > limits.maxRectangularPieces) {
      setFailureReason(failureReason,
                       "selected carrier fragment count exceeds work limit");
      return mlir::failure();
    }
    for (const SpatialEdgeFragment &fragment : strategy.fragments)
      pieces.push_back({fragment.offsets, fragment.sizes});
  } else {
    pieces.push_back({strategy.producerOffsets, strategy.producerSizes});
  }
  if (pieces.empty()) {
    setFailureReason(failureReason,
                     "selected physical carrier has no demand domain");
    return mlir::failure();
  }

  uint64_t requiredVolume = 0;
  for (const analysis::StaticRectangularIndexSet &box :
       normalizedRequired->getBoxes()) {
    std::optional<uint64_t> volume = getBoxVolume({box.offsets, box.sizes});
    if (!volume ||
        requiredVolume > std::numeric_limits<uint64_t>::max() - *volume)
      return mlir::failure();
    requiredVolume += *volume;
  }
  uint64_t coveredVolume = 0;
  for (const StaticBoxRef &piece : pieces) {
    std::optional<uint64_t> pieceVolume = getBoxVolume(piece);
    if (!pieceVolume) {
      setFailureReason(
          failureReason,
          "dependent fragment extends outside its consumer demand");
      return mlir::failure();
    }
    uint64_t insideVolume = 0;
    for (const analysis::StaticRectangularIndexSet &required :
         normalizedRequired->getBoxes()) {
      std::optional<uint64_t> overlap =
          getIntersectionVolume(piece, {required.offsets, required.sizes});
      if (!overlap) {
        setFailureReason(
            failureReason,
            "dependent fragment extends outside its consumer demand");
        return mlir::failure();
      }
      if (insideVolume > std::numeric_limits<uint64_t>::max() - *overlap)
        return mlir::failure();
      insideVolume += *overlap;
    }
    if (insideVolume != *pieceVolume) {
      setFailureReason(
          failureReason,
          "dependent fragment extends outside its consumer demand");
      return mlir::failure();
    }
    if (coveredVolume > std::numeric_limits<uint64_t>::max() - *pieceVolume)
      return mlir::failure();
    coveredVolume += *pieceVolume;
  }

  if (pieces.front().offsets.empty()) {
    if (pieces.size() != 1) {
      setFailureReason(failureReason, "zero-rank carrier domain is duplicated");
      return mlir::failure();
    }
  } else {
    unsigned sweepDimension = 0;
    for (unsigned dimension = 1; dimension < pieces.front().offsets.size();
         ++dimension) {
      llvm::SmallDenseSet<int64_t, 16> currentOffsets;
      llvm::SmallDenseSet<int64_t, 16> bestOffsets;
      for (const StaticBoxRef &piece : pieces) {
        currentOffsets.insert(piece.offsets[dimension]);
        bestOffsets.insert(piece.offsets[sweepDimension]);
      }
      if (currentOffsets.size() > bestOffsets.size())
        sweepDimension = dimension;
    }
    llvm::SmallVector<unsigned, 16> order(pieces.size());
    std::iota(order.begin(), order.end(), 0);
    llvm::sort(order, [&](unsigned lhs, unsigned rhs) {
      if (pieces[lhs].offsets[sweepDimension] !=
          pieces[rhs].offsets[sweepDimension])
        return pieces[lhs].offsets[sweepDimension] <
               pieces[rhs].offsets[sweepDimension];
      return lhs < rhs;
    });
    for (size_t left = 0; left < order.size(); ++left) {
      const StaticBoxRef lhs = pieces[order[left]];
      const int64_t lhsEnd =
          lhs.offsets[sweepDimension] + lhs.sizes[sweepDimension];
      for (size_t right = left + 1; right < order.size(); ++right) {
        const StaticBoxRef rhs = pieces[order[right]];
        if (rhs.offsets[sweepDimension] >= lhsEnd)
          break;
        std::optional<uint64_t> overlap = getIntersectionVolume(lhs, rhs);
        if (!overlap)
          return mlir::failure();
        if (*overlap != 0) {
          setFailureReason(
              failureReason,
              strategy.action == SpatialEdgeAction::PeerFragments ||
                      strategy.action == SpatialEdgeAction::CardDDRTransfer
                  ? "dependent fragments overlap within one consumer demand"
                  : "selected producer domain overlaps itself");
          return mlir::failure();
        }
      }
    }
  }
  if (coveredVolume != requiredVolume) {
    setFailureReason(failureReason,
                     strategy.action == SpatialEdgeAction::PeerFragments ||
                             strategy.action ==
                                 SpatialEdgeAction::CardDDRTransfer
                         ? "dependent fragments do not exactly cover the "
                           "consumer demand"
                         : "selected producer domain differs from the exact "
                           "relation image");
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult
wafer::tensor_program_to_tile_region::reportSelectedEdgeFailure(
    std::string *failureReason, llvm::StringRef message) {
  setFailureReason(failureReason, message);
  return mlir::failure();
}

bool wafer::isSpatialEdgeStrategyIncidentOnTile(
    const SpatialEdgeStrategy &strategy, TileId tile) {
  if (strategy.destinationTile == tile)
    return true;
  if (strategy.action == SpatialEdgeAction::CardDDRTransfer)
    return strategy.sourceTile == tile;
  if (strategy.action != SpatialEdgeAction::PeerFragments)
    return false;
  return llvm::any_of(strategy.fragments, [&](const SpatialEdgeFragment &item) {
    return (item.kind == SpatialEdgeFragmentKind::Peer ||
            item.kind == SpatialEdgeFragmentKind::CardDDR) &&
           item.sourceTile == tile;
  });
}

mlir::FailureOr<llvm::SmallVector<SpatialEdgeMaterializationFacts, 16>>
wafer::deriveSpatialEdgeMaterializationFacts(
    mlir::Block &sourceBody, llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    llvm::ArrayRef<analysis::DependencyDemand> operandDemands,
    std::string *failureReason) {
  llvm::SmallVector<SpatialEdgeMaterializationFacts, 16> facts;
  facts.reserve(edgeStrategies.size());
  llvm::DenseMap<mlir::Operation *, uint64_t> scheduleOrdinals;
  uint64_t ordinal = 0;
  for (mlir::Operation &operation : sourceBody)
    scheduleOrdinals.try_emplace(&operation, ordinal++);
  std::map<std::pair<mlir::Operation *, uint32_t>,
           const analysis::DependencyDemand *>
      dependencies;
  for (const analysis::DependencyDemand &demand : operandDemands)
    if (!dependencies
             .try_emplace(std::make_pair(demand.consumerOperation,
                                         demand.consumerOperand),
                          &demand)
             .second) {
      setFailureReason(failureReason,
                       "grouped exact-demand dependency is duplicated");
      return mlir::failure();
    }
  for (const SpatialEdgeStrategy &strategy : edgeStrategies) {
    auto consumerPosition = scheduleOrdinals.find(strategy.consumer);
    if (consumerPosition == scheduleOrdinals.end()) {
      setFailureReason(failureReason,
                       "selected edge consumer is outside source body");
      return mlir::failure();
    }
    auto dependency = dependencies.find(
        std::make_pair(strategy.consumer, strategy.consumerOperand));
    if (dependency == dependencies.end()) {
      setFailureReason(failureReason,
                       "selected edge has no grouped exact-demand dependency");
      return mlir::failure();
    }
    auto destination = llvm::find_if(
        dependency->second->perDestination,
        [&](const analysis::DestinationDemand &candidate) {
          return candidate.destinationTile == strategy.destinationTile;
        });
    if (destination == dependency->second->perDestination.end()) {
      setFailureReason(failureReason,
                       "selected edge has no destination demand recipe");
      return mlir::failure();
    }
    auto source = llvm::find_if(
        destination->sources, [&](const analysis::SourceDemand &candidate) {
          const auto *structured =
              std::get_if<analysis::StructuredResultSource>(&candidate.source);
          return structured && structured->operation == strategy.producer &&
                 structured->result == strategy.producerResult;
        });
    if (source == destination->sources.end() ||
        source->requiredDomain.isEmpty()) {
      setFailureReason(
          failureReason,
          "selected edge has no nonempty structured source demand");
      return mlir::failure();
    }
    if (mlir::failed(validateStrategyDemand(strategy, source->requiredDomain,
                                            failureReason)))
      return mlir::failure();
    facts.push_back(SpatialEdgeMaterializationFacts{
        /*requiresConsumerInputReconstruction=*/
        !destination->reconstruction.steps.empty(),
        /*consumerScheduleOrdinal=*/
        consumerPosition->second});
  }
  return facts;
}

mlir::LogicalResult wafer::lowerSpatialEdgeStrategiesToTileRegionModule(
    mlir::ModuleOp sourceModule, unsigned functionalArgumentCount,
    llvm::ArrayRef<SpatialOutputShard> outputShards, TileId currentTile,
    SpatialDataflowMaterializationMode materializationMode,
    llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalPartition,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    StructuredMaterializationRelations *materializationRelations,
    llvm::ArrayRef<SpatialEdgeMaterializationFacts> edgeFacts,
    llvm::ArrayRef<analysis::DependencyDemand> operandDemands,
    bool requireOneStructuredRootPerRegion) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModule || currentLogicalPartition < 0)
    return reportSelectedEdgeFailure(
        failureReason, "edge-action lowering requires a source module and "
                       "logical card partition");
  if (!edgeFacts.empty() && edgeFacts.size() != edgeStrategies.size())
    return reportSelectedEdgeFailure(
        failureReason,
        "edge materialization facts must cover every selected edge");

  mlir::IRMapping cloneMapping;
  mlir::OwningOpRef<mlir::ModuleOp> candidate =
      mlir::cast<mlir::ModuleOp>(sourceModule->clone(cloneMapping));
  mlir::func::FuncOp function = findSingleStandaloneTensorProgram(*candidate);
  mlir::func::FuncOp sourceFunction =
      findSingleStandaloneTensorProgram(sourceModule);
  if (!function || !sourceFunction || sourceFunction.isExternal() ||
      !sourceFunction.getBody().hasOneBlock())
    return reportSelectedEdgeFailure(
        failureReason,
        "edge-action lowering requires one pristine support template body");
  if (mlir::failed(appendTileOutputDestinations(function, failureReason)))
    return mlir::failure();
  TensorProgramScope scope(function, functionalArgumentCount);
  if (mlir::failed(verifyTensorProgramScope(function, functionalArgumentCount,
                                            failureReason,
                                            scope.getBoundaryArgumentCount())))
    return mlir::failure();
  mlir::Block &sourceBody = sourceFunction.getBody().front();

  mlir::FailureOr<SelectedEdgeProgramMapping> mappedEdges =
      mapSelectedEdgesToCandidate(sourceBody, cloneMapping, scope, currentTile,
                                  materializationMode, edgeStrategies,
                                  operationTemporalTiles, operationNodes,
                                  edgeFacts, operandDemands, failureReason);
  if (mlir::failed(mappedEdges))
    return mlir::failure();
  SelectedEdgeLoweringState state(
      sourceModule, functionalArgumentCount, outputShards, currentTile,
      currentLogicalPartition, requireOneStructuredRootPerRegion,
      operationNodes, std::move(candidate), function, std::move(*mappedEdges),
      failureReason);
  if (mlir::failed(materializeSelectedDirectActions(state)) ||
      mlir::failed(materializeSelectedPeerReceives(state)) ||
      mlir::failed(verifySelectedReceiveOwners(state)) ||
      mlir::failed(materializeSelectedSourceStages(state)) ||
      mlir::failed(materializeSelectedConsumerStages(state)) ||
      mlir::failed(requireLiveSelectedReceives(
          state, "internal consumer materialization")) ||
      mlir::failed(materializeSelectedPeerSends(state)) ||
      mlir::failed(requireLiveSelectedReceives(
          state, "outgoing peer materialization")) ||
      mlir::failed(materializeSelectedOutputs(state)) ||
      mlir::failed(requireLiveSelectedReceives(
          state, "output traversal materialization")) ||
      mlir::failed(verifySelectedReceiveExecutionSinks(state)))
    return mlir::failure();
  return finishSelectedEdgeLowering(state, module, materializationRelations);
}
