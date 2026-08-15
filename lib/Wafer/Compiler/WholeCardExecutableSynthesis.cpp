//===- WholeCardExecutableSynthesis.cpp - Whole-card search seam --------===//

#include "WholeCardExecutableSynthesis.h"

#include "BoundedTileExecutor.h"
#include "CardExecutableCompilation.h"
#include "SelectedBufferMaterialization.h"
#include "StructuredBufferRelations.h"
#include "TileRegionSPMCapacityEvaluation.h"
#include "WholeDAGCandidateSchedule.h"
#include "WholeDAGEdgeStrategyPlan.h"
#include "WholeDAGPlacementEnumeration.h"
#include "WholeDAGSchedule.h"
#include "WholeDAGSchedulePlan.h"

#include "Wafer/Conversion/WaferTensorProgramToCardProgram/WaferTensorProgramToCardProgram.h"
#include "Wafer/IR/Target/PhysicalTopology.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/Tx81InstructionLimits.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

static llvm::StringRef getSpatialEdgeActionName(SpatialEdgeAction action) {
  switch (action) {
  case SpatialEdgeAction::CoupledFusion:
    return "coupled-fusion";
  case SpatialEdgeAction::LocalShardResidency:
    return "local-shard-residency";
  case SpatialEdgeAction::PeerFragments:
    return "peer-fragments";
  case SpatialEdgeAction::SpillReload:
    return "spill-reload";
  case SpatialEdgeAction::Recompute:
    return "recompute";
  case SpatialEdgeAction::RegionCut:
    return "region-cut";
  case SpatialEdgeAction::LocalPhysicalConversion:
    return "local-physical-conversion";
  }
  llvm_unreachable("unknown spatial edge action");
}

struct SPMCapacityDemandEvidence {
  PhysicalTileId physicalTileId{-1};
  std::optional<CardDAGNodeID> operationNode;
  std::optional<unsigned> outputIndex;
  std::optional<CardDAGNodeID> operandDemandNode;
  bool relationFromAllocation = false;
  mlir::Type type;
  uint64_t bytes = 0;
  uint64_t demandCount = 0;
  uint64_t exactConflictBytes = 0;
  uint64_t exactConflictDemandCount = 0;
};

struct WholeCardCandidateAssignment {
  CardSpatialMapping mapping;
  llvm::SmallVector<WholeDAGNodePlacement, 16> nodePlacements;
};

/// Facts derived from the candidate assignment and the current TensorProgram
/// epoch. They are recomputed after an assignment change and never serve as
/// cross-epoch identity.
struct WholeCardCandidateEvaluation {
  uint64_t criticalStructuredElementWork = 0;
  uint64_t shardImbalance = 0;
  uint64_t queryExpansionWork = 0;
  uint64_t parallelComponentCount = 1;
  uint64_t temporalWaveLowerBound = 0;
  uint64_t instructionExecutionLowerBound = 0;
  uint64_t peakOutputTileFootprintEstimate = 0;
  uint64_t peakAlignedResidencyEstimate = 0;
  uint64_t scheduledEventCount = 0;
  uint64_t scheduledMakespan = 0;
  uint64_t scheduledPeakLiveSPMBytes = 0;
  uint64_t scheduledSPMMovementWork = 0;
  uint64_t scheduledDDRMovementWork = 0;
  uint64_t peerBytes = 0;
  uint64_t topologyHopByteWork = 0;
  uint32_t distinctTileGroupCount = 1;
  uint32_t partialOverlapEdgeCount = 0;
  uint32_t disjointEdgeCount = 0;
};

struct SPMDemandRelation {
  std::optional<CardDAGNodeID> operationNode;
  std::optional<unsigned> outputIndex;
  std::optional<CardDAGNodeID> operandDemandNode;
  mlir::Type type;
  uint64_t bytes = 0;

  bool matches(const SPMCapacityDemandEvidence &evidence) const {
    return operationNode == evidence.operationNode &&
           outputIndex == evidence.outputIndex &&
           operandDemandNode == evidence.operandDemandNode &&
           type == evidence.type && bytes == evidence.bytes;
  }
};

/// Query-local controller bookkeeping. It may order or generate another
/// assignment but cannot establish legality or become selected IR semantics.
struct WholeCardCandidateTransition {
  uint64_t stableOrdinal = 0;
  /// Stable identity of one candidate emitted by the initial factorized
  /// joint search. Exact allocator-directed temporal and hard-capacity edge
  /// neighbors inherit it so a successful refinement closes that search branch
  /// without merging distinct spatial/layout/action/buffer seeds.
  uint64_t feedbackRootOrdinal = 0;
  bool nodePlacementCandidate = false;
  unsigned temporalRefinementDepth = 0;
  bool bufferFeedbackBoundary = false;
  /// This exact state is an additional two-breakpoint probe created only
  /// after the allocator reported the same limiting allocation on adjacent
  /// states.  It remains part of the ordinary joint-search domain, but an
  /// a successful check must not close its family until the adjacent state
  /// establishes the coarser exact boundary.
  bool allocationFeedbackLookahead = false;
  /// Exact allocator signature that caused the immediately preceding
  /// feedback transition.  It is query-local evidence used only to detect
  /// that the chosen adjacent coordinate left the limiting actual allocation
  /// unchanged; it never accepts or rejects a candidate by estimate.
  std::optional<SPMDemandRelation> previousSPMDemand;
  /// Number of consecutive exact failures with the same limiting allocator
  /// signature along the current feedback lane.  It controls only how far an
  /// additional probe looks ahead; the adjacent state remains in the common
  /// candidate queue.
  uint64_t unchangedSPMDemandStreak = 0;
  /// Exact allocator result of the parent state that generated this pending
  /// feedback candidate.  These fields order already-legal search states;
  /// they are not estimates and never decide SPM acceptance.
  uint64_t parentSPMConflictBytes = 0;
  uint64_t parentSPMConflictDemandCount = 0;
  uint64_t parentSPMTotalDemandCount = 0;
  /// Query-local ordering memory for progressive multi-demand allocator
  /// feedback. It is not part of exact-state identity or selected IR.
  llvm::SmallVector<uint64_t, 8> progressiveSPMDemandHistory;
};

struct WholeCardCandidate {
  WholeCardCandidateAssignment assignment;
  WholeCardCandidateEvaluation evaluation;
  WholeCardCandidateTransition transition;
};

static SPMDemandRelation
getSPMDemandRelation(const SPMCapacityDemandEvidence &evidence) {
  return {evidence.operationNode, evidence.outputIndex,
          evidence.operandDemandNode, evidence.type, evidence.bytes};
}

struct AdmittedWholeCardCandidate {
  AdmittedWholeCardCandidate(
      WholeCardCandidate candidate, WholeCardExecutable executable,
      llvm::SmallVector<AcceptedOperationNodeRelation, 64> operationNodes,
      uint64_t actualFusedLogicalEdges)
      : candidate(std::move(candidate)), executable(std::move(executable)),
        operationNodes(std::move(operationNodes)),
        actualFusedLogicalEdges(actualFusedLogicalEdges) {}
  AdmittedWholeCardCandidate(AdmittedWholeCardCandidate &&) noexcept = default;
  AdmittedWholeCardCandidate &
  operator=(AdmittedWholeCardCandidate &&) noexcept = default;
  AdmittedWholeCardCandidate(const AdmittedWholeCardCandidate &) = delete;
  AdmittedWholeCardCandidate &
  operator=(const AdmittedWholeCardCandidate &) = delete;

  WholeCardCandidate candidate;
  WholeCardExecutable executable;
  llvm::SmallVector<AcceptedOperationNodeRelation, 64> operationNodes;
  llvm::SmallVector<analysis::WholeCardInstructionProgramCost, 16> phaseCosts;
  std::optional<analysis::StaticSchedulePlan> schedulePlan;
  uint64_t actualFusedLogicalEdges = 0;
};

static uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  if (product > std::numeric_limits<uint64_t>::max())
    return std::numeric_limits<uint64_t>::max();
  return static_cast<uint64_t>(product);
}

static uint64_t saturatingAdd(uint64_t lhs, uint64_t rhs) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return std::numeric_limits<uint64_t>::max();
  return lhs + rhs;
}

static uint64_t ceilDivide(uint64_t numerator, uint64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

static uint64_t saturatingAlignTo(uint64_t value, uint64_t alignment) {
  if (alignment == 0)
    return std::numeric_limits<uint64_t>::max();
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : saturatingAdd(value, alignment - remainder);
}

static mlir::FailureOr<mlir::func::FuncOp>
getStructuredProgram(mlir::ModuleOp module, std::string &failureReason) {
  mlir::func::FuncOp program;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (program) {
      failureReason =
          "whole-card synthesis requires exactly one defined tensor program";
      return mlir::failure();
    }
    program = function;
  }
  if (!program) {
    failureReason =
        "whole-card synthesis requires exactly one defined tensor program";
    return mlir::failure();
  }
  return program;
}

using StaticOutputDomains = llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>;

static mlir::FailureOr<StaticOutputDomains>
getStaticOutputDomains(mlir::func::FuncOp program, std::string &failureReason) {
  if (program.getNumResults() == 0) {
    failureReason = "whole-card synthesis requires tensor output domains";
    return mlir::failure();
  }
  StaticOutputDomains domains;
  domains.reserve(program.getNumResults());
  for (mlir::Type resultType : program.getResultTypes()) {
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!ranked || !ranked.hasStaticShape() || ranked.getRank() == 0 ||
        llvm::any_of(ranked.getShape(),
                     [](int64_t extent) { return extent <= 0; })) {
      failureReason =
          "whole-card synthesis requires nonempty static tensor output "
          "domains";
      return mlir::failure();
    }
    domains.emplace_back(ranked.getShape());
  }
  return domains;
}

static uint64_t getOutputElementCount(llvm::ArrayRef<int64_t> domain) {
  uint64_t elements = 1;
  for (int64_t extent : domain)
    elements = saturatingMultiply(elements, static_cast<uint64_t>(extent));
  return elements;
}

static size_t getUniqueActiveTileCount(
    const CardSpatialMapping &mapping,
    llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements = {}) {
  llvm::SmallVector<int64_t, 16> ids;
  for (const WholeDAGNodePlacement &placement : nodePlacements)
    for (PhysicalTileId tile : placement.tiles)
      ids.push_back(tile.getValue());
  for (const CardOutputSpatialMapping &output : mapping.outputs)
    for (PhysicalTileId tile : output.activeTileIds)
      ids.push_back(tile.getValue());
  llvm::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids.size();
}

static size_t getActiveTileAssignmentCount(
    const CardSpatialMapping &mapping,
    llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements = {}) {
  size_t count = 0;
  if (!nodePlacements.empty()) {
    for (const WholeDAGNodePlacement &placement : nodePlacements)
      count += placement.tiles.size();
  } else {
    for (const CardOutputSpatialMapping &output : mapping.outputs)
      count += output.activeTileIds.size();
  }
  return count;
}

static size_t getPeerFragmentCount(const CardSpatialMapping &mapping) {
  size_t count = 0;
  for (const SpatialEdgeStrategy &strategy : mapping.edgeStrategies)
    count += llvm::count_if(
        strategy.fragments, [](const SpatialEdgeFragment &fragment) {
          return fragment.kind == SpatialEdgeFragmentKind::Peer;
        });
  return count;
}

static uint64_t getDisjointDependencyComponentCount(
    const CardDAGAnalysis &dag,
    llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements) {
  llvm::ArrayRef<CardDAGDependencyComponent> components =
      dag.getObservableDependencyComponents();
  if (!dag.supportsIndependentComponentPlacement() || components.size() < 2 ||
      nodePlacements.size() != dag.getNodes().size())
    return 1;
  llvm::SmallVector<llvm::SmallVector<PhysicalTileId, 16>, 4> componentTiles;
  componentTiles.resize(components.size());
  for (auto [componentIndex, component] : llvm::enumerate(components)) {
    for (CardDAGNodeID node : component.nodes) {
      if (node >= nodePlacements.size())
        return 1;
      componentTiles[componentIndex].append(nodePlacements[node].tiles.begin(),
                                            nodePlacements[node].tiles.end());
    }
    llvm::sort(componentTiles[componentIndex],
               [](PhysicalTileId lhs, PhysicalTileId rhs) {
                 return lhs.getValue() < rhs.getValue();
               });
    componentTiles[componentIndex].erase(
        std::unique(componentTiles[componentIndex].begin(),
                    componentTiles[componentIndex].end()),
        componentTiles[componentIndex].end());
  }
  for (size_t left = 0; left < componentTiles.size(); ++left)
    for (size_t right = left + 1; right < componentTiles.size(); ++right)
      if (llvm::any_of(componentTiles[left], [&](PhysicalTileId tile) {
            return llvm::is_contained(componentTiles[right], tile);
          }))
        return 1;
  return components.size();
}

static WholeCardCandidate
makeJointCandidate(uint64_t stableOrdinal, size_t activeTileCount,
                   llvm::ArrayRef<unsigned> dimensions,
                   llvm::ArrayRef<PhysicalTileId> availableTileIds,
                   const StaticOutputDomains &outputDomains,
                   const CardDAGAnalysis &dag) {
  WholeCardCandidate candidate;
  candidate.transition.stableOrdinal = stableOrdinal;
  uint64_t totalOutputElements = 0;
  for (auto [outputIndex, domain] : llvm::enumerate(outputDomains)) {
    CardOutputSpatialMapping output;
    output.outputIndex = static_cast<unsigned>(outputIndex);
    output.shardDimension = dimensions[outputIndex];
    output.activeTileIds.append(availableTileIds.begin(),
                                availableTileIds.begin() + activeTileCount);
    output.temporalTileSizes.assign(domain.begin(), domain.end());
    candidate.assignment.mapping.outputs.push_back(std::move(output));
    totalOutputElements =
        saturatingAdd(totalOutputElements, getOutputElementCount(domain));
    candidate.evaluation.shardImbalance =
        saturatingAdd(candidate.evaluation.shardImbalance,
                      static_cast<uint64_t>(domain[dimensions[outputIndex]]) %
                          activeTileCount);
  }
  candidate.evaluation.criticalStructuredElementWork = saturatingMultiply(
      ceilDivide(totalOutputElements, activeTileCount),
      std::max<uint64_t>(1, static_cast<uint64_t>(dag.getNodes().size())));
  for (const CardDAGNode &node : dag.getNodes()) {
    WholeDAGNodePlacement placement;
    placement.node = node.id;
    placement.shardDimension = dimensions.front();
    if (auto tiling = mlir::dyn_cast<mlir::TilingInterface>(node.operation)) {
      const size_t iteratorCount = tiling.getLoopIteratorTypes().size();
      placement.iteratorPartitionFactors.assign(iteratorCount, 1);
      unsigned iteratorDimension = std::min<unsigned>(
          placement.shardDimension,
          iteratorCount == 0 ? 0 : static_cast<unsigned>(iteratorCount - 1));
      if (auto linalg =
              mlir::dyn_cast<mlir::linalg::LinalgOp>(node.operation)) {
        mlir::AffineMap resultMap =
            linalg.getIndexingMapMatchingResult(node.operation->getResult(0));
        if (resultMap && placement.shardDimension < resultMap.getNumResults())
          if (auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(
                  resultMap.getResult(placement.shardDimension)))
            iteratorDimension = dimension.getPosition();
      }
      placement.spatialIteratorDimension = iteratorDimension;
      if (iteratorDimension < placement.iteratorPartitionFactors.size())
        placement.iteratorPartitionFactors[iteratorDimension] =
            static_cast<uint32_t>(activeTileCount);
    }
    placement.tiles.append(availableTileIds.begin(),
                           availableTileIds.begin() + activeTileCount);
    candidate.assignment.nodePlacements.push_back(std::move(placement));
  }
  // This is compile-time query expansion, not a runtime communication model.
  candidate.evaluation.queryExpansionWork = saturatingMultiply(
      getActiveTileAssignmentCount(candidate.assignment.mapping,
                                   candidate.assignment.nodePlacements),
      saturatingAdd(dag.getNodes().size(), dag.getEdges().size()));
  return candidate;
}

static bool prepareEdgeStrategies(WholeCardCandidate &candidate,
                                  const CardDAGAnalysis &dag) {
  if (!candidate.assignment.mapping.edgeStrategies.empty())
    return true;
  std::string failureReason;
  mlir::FailureOr<WholeDAGEdgeStrategyPlan> edgePlan =
      deriveWholeDAGEdgeStrategyPlan(dag, candidate.assignment.nodePlacements,
                                     &failureReason);
  if (mlir::failed(edgePlan))
    return false;
  candidate.assignment.mapping.edgeStrategies = std::move(edgePlan->strategies);
  candidate.evaluation.peerBytes = edgePlan->totalPeerBytes;
  candidate.evaluation.queryExpansionWork = saturatingMultiply(
      saturatingAdd(dag.getNodes().size(), dag.getEdges().size()),
      saturatingAdd(
          getActiveTileAssignmentCount(candidate.assignment.mapping,
                                       candidate.assignment.nodePlacements),
          getPeerFragmentCount(candidate.assignment.mapping)));
  return true;
}

static bool deriveBalancedBaselineShard(mlir::RankedTensorType type,
                                        unsigned shardDimension,
                                        llvm::ArrayRef<PhysicalTileId> tiles,
                                        PhysicalTileId tile,
                                        llvm::SmallVectorImpl<int64_t> &offsets,
                                        llvm::SmallVectorImpl<int64_t> &sizes) {
  auto found = llvm::find(tiles, tile);
  if (!type || !type.hasStaticShape() ||
      shardDimension >= static_cast<unsigned>(type.getRank()) ||
      tiles.empty() || found == tiles.end())
    return false;
  const int64_t extent = type.getDimSize(shardDimension);
  const int64_t participants = static_cast<int64_t>(tiles.size());
  if (extent <= 0 || extent < participants)
    return false;
  const int64_t ordinal = std::distance(tiles.begin(), found);
  const int64_t base = extent / participants;
  const int64_t larger = extent % participants;
  offsets.assign(type.getRank(), 0);
  sizes.assign(type.getShape().begin(), type.getShape().end());
  offsets[shardDimension] = ordinal * base + std::min(ordinal, larger);
  sizes[shardDimension] = base + (ordinal < larger ? 1 : 0);
  return true;
}

static bool staticDomainsOverlap(llvm::ArrayRef<int64_t> lhsOffsets,
                                 llvm::ArrayRef<int64_t> lhsSizes,
                                 llvm::ArrayRef<int64_t> rhsOffsets,
                                 llvm::ArrayRef<int64_t> rhsSizes) {
  if (lhsOffsets.size() != lhsSizes.size() ||
      lhsOffsets.size() != rhsOffsets.size() ||
      lhsOffsets.size() != rhsSizes.size())
    return false;
  return llvm::all_of(
      llvm::zip_equal(lhsOffsets, lhsSizes, rhsOffsets, rhsSizes),
      [](auto values) {
        auto [lhsOffset, lhsSize, rhsOffset, rhsSize] = values;
        return lhsOffset >= 0 && rhsOffset >= 0 && lhsSize > 0 && rhsSize > 0 &&
               lhsOffset < rhsOffset + rhsSize &&
               rhsOffset < lhsOffset + lhsSize;
      });
}

/// The general edge planner leaves dependencies through pure support ops to
/// consumer traversal because search policy may choose to keep them resident.
/// The no-fusion baseline has a stricter contract: dependencies through a
/// reshape/view support graph cannot remain implicit fusion. Map each
/// producer spatial shard forward through the exact support/index relation
/// and retain only shards that contribute to the destination consumer shard;
/// the consumer side then rebuilds the support graph from those explicit
/// fragments. Direct local dependencies remain DDR RegionCuts; direct
/// nonlocal dependencies retain their exact peer plan.
static bool appendBaselineSupportChainTransfers(WholeCardCandidate &candidate,
                                                const CardDAGAnalysis &dag,
                                                std::string *failureReason) {
  llvm::SmallVector<const WholeDAGNodePlacement *, 16> placements(
      dag.getNodes().size(), nullptr);
  for (const WholeDAGNodePlacement &placement :
       candidate.assignment.nodePlacements) {
    if (placement.node >= placements.size() || placements[placement.node]) {
      if (failureReason)
        *failureReason = "baseline has an invalid structured placement";
      return false;
    }
    placements[placement.node] = &placement;
  }

  for (const CardDAGEdge &edge : dag.getEdges()) {
    const CardDAGNode *producerNode = dag.getNode(edge.producer);
    const CardDAGNode *consumerNode = dag.getNode(edge.consumer);
    if (!producerNode || !consumerNode || !producerNode->operation ||
        !consumerNode->operation || !placements[edge.producer] ||
        !placements[edge.consumer]) {
      if (failureReason)
        *failureReason = "baseline dependency references an unknown node";
      return false;
    }
    mlir::Value producerResult =
        producerNode->operation->getResult(edge.producerResult);
    if (consumerNode->operation->getOperand(edge.consumerOperand) ==
        producerResult)
      continue;

    mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>> supportChain =
        deriveUnaryPureSupportChain(
            producerNode->operation, edge.producerResult,
            consumerNode->operation, edge.consumerOperand, failureReason);
    if (mlir::failed(supportChain) || supportChain->empty())
      return false;
    const WholeDAGNodePlacement &producerPlacement = *placements[edge.producer];
    const WholeDAGNodePlacement &consumerPlacement = *placements[edge.consumer];
    auto producerType =
        mlir::dyn_cast<mlir::RankedTensorType>(producerResult.getType());
    auto consumerType = mlir::dyn_cast<mlir::RankedTensorType>(
        consumerNode->operation->getResult(0).getType());
    mlir::Type elementType =
        producerType ? producerType.getElementType() : mlir::Type{};
    const unsigned elementBits = elementType && elementType.isIntOrFloat()
                                     ? elementType.getIntOrFloatBitWidth()
                                     : 0;
    if (!producerType || !consumerType || !producerType.hasStaticShape() ||
        !consumerType.hasStaticShape() || elementBits == 0 ||
        elementBits % 8 != 0 ||
        producerPlacement.shardDimension >=
            static_cast<unsigned>(producerType.getRank()) ||
        consumerPlacement.shardDimension >=
            static_cast<unsigned>(consumerType.getRank())) {
      if (failureReason)
        *failureReason =
            "baseline support dependency has no static byte-addressable "
            "producer/consumer shard";
      return false;
    }
    int64_t payloadSlice = 0;
    for (PhysicalTileId destination : consumerPlacement.tiles) {
      SpatialEdgeStrategy strategy;
      strategy.producer = producerNode->operation;
      strategy.producerResult = edge.producerResult;
      strategy.consumer = consumerNode->operation;
      strategy.consumerOperand = edge.consumerOperand;
      if (!deriveBalancedBaselineShard(
              consumerType, consumerPlacement.shardDimension,
              consumerPlacement.tiles, destination, strategy.consumerOffsets,
              strategy.consumerSizes)) {
        if (failureReason)
          *failureReason =
              "baseline support consumer cannot form nonempty shards";
        return false;
      }
      strategy.sourceTile = destination;
      strategy.destinationTile = destination;
      strategy.action = SpatialEdgeAction::PeerFragments;
      strategy.bufferCount = 1;
      struct ContributingShard {
        PhysicalTileId source{0};
        llvm::SmallVector<int64_t, 4> offsets;
        llvm::SmallVector<int64_t, 4> sizes;
      };
      llvm::SmallVector<ContributingShard, 16> contributing;
      for (PhysicalTileId source : producerPlacement.tiles) {
        llvm::SmallVector<int64_t, 4> offsets;
        llvm::SmallVector<int64_t, 4> sizes;
        if (!deriveBalancedBaselineShard(
                producerType, producerPlacement.shardDimension,
                producerPlacement.tiles, source, offsets, sizes)) {
          if (failureReason)
            *failureReason =
                "baseline support producer cannot form nonempty shards";
          return false;
        }
        SpatialEdgeStrategy sourceRelation = strategy;
        sourceRelation.producerOffsets = offsets;
        sourceRelation.producerSizes = sizes;
        llvm::SmallVector<int64_t, 4> imageOffsets;
        llvm::SmallVector<int64_t, 4> imageSizes;
        std::string relationFailure;
        if (mlir::failed(deriveSpatialEdgeConsumerResultDomain(
                sourceRelation, imageOffsets, imageSizes, &relationFailure))) {
          if (failureReason)
            *failureReason =
                "baseline support shard has no exact forward relation: " +
                relationFailure;
          return false;
        }
        if (staticDomainsOverlap(imageOffsets, imageSizes,
                                 strategy.consumerOffsets,
                                 strategy.consumerSizes))
          contributing.push_back(
              ContributingShard{source, std::move(offsets), std::move(sizes)});
      }
      if (contributing.empty()) {
        if (failureReason)
          *failureReason =
              "baseline support dependency has no producer contribution for "
              "one consumer shard";
        return false;
      }

      strategy.producerOffsets = contributing.front().offsets;
      strategy.producerSizes = contributing.front().sizes;
      const unsigned shardDimension = producerPlacement.shardDimension;
      int64_t demandBegin = strategy.producerOffsets[shardDimension];
      int64_t demandEnd = demandBegin + strategy.producerSizes[shardDimension];
      uint64_t coveredElements = 0;
      for (const ContributingShard &shard : contributing) {
        demandBegin = std::min(demandBegin, shard.offsets[shardDimension]);
        demandEnd = std::max(demandEnd, shard.offsets[shardDimension] +
                                            shard.sizes[shardDimension]);
        uint64_t elements = 1;
        for (int64_t size : shard.sizes)
          elements = saturatingMultiply(elements, static_cast<uint64_t>(size));
        coveredElements = saturatingAdd(coveredElements, elements);
      }
      strategy.producerOffsets[shardDimension] = demandBegin;
      strategy.producerSizes[shardDimension] = demandEnd - demandBegin;
      uint64_t demandedElements = 1;
      for (int64_t size : strategy.producerSizes)
        demandedElements =
            saturatingMultiply(demandedElements, static_cast<uint64_t>(size));
      if (coveredElements != demandedElements) {
        if (failureReason)
          *failureReason =
              "baseline support contribution is not one exact rectangular "
              "producer demand";
        return false;
      }

      for (const ContributingShard &shard : contributing) {
        PhysicalTileId source = shard.source;
        const llvm::SmallVector<int64_t, 4> &offsets = shard.offsets;
        const llvm::SmallVector<int64_t, 4> &sizes = shard.sizes;
        if (source == destination) {
          strategy.fragments.push_back(SpatialEdgeFragment{
              SpatialEdgeFragmentKind::Resident, offsets, sizes, source,
              /*bytes=*/0, /*communicationId=*/0, /*payloadSlice=*/0});
          continue;
        }
        uint64_t elements = 1;
        for (int64_t size : sizes)
          elements = saturatingMultiply(elements, static_cast<uint64_t>(size));
        const uint64_t bytes = saturatingMultiply(elements, elementBits / 8);
        if (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max()) {
          if (failureReason)
            *failureReason =
                "baseline support peer fragment exceeds target payload";
          return false;
        }
        strategy.fragments.push_back(SpatialEdgeFragment{
            SpatialEdgeFragmentKind::Peer, offsets, sizes, source, bytes,
            static_cast<int64_t>(edge.id), payloadSlice++});
        candidate.evaluation.peerBytes =
            saturatingAdd(candidate.evaluation.peerBytes, bytes);
      }

      candidate.assignment.mapping.edgeStrategies.push_back(
          std::move(strategy));
    }
  }
  return true;
}

static std::optional<WholeCardCandidate> makeNodePlacementCandidate(
    uint64_t stableOrdinal, const WholeDAGPlacementCandidate &placement,
    const StaticOutputDomains &outputDomains, const CardDAGAnalysis &dag) {
  if (placement.outputPlacements.size() != outputDomains.size())
    return std::nullopt;

  WholeCardCandidate candidate;
  candidate.transition.stableOrdinal = stableOrdinal;
  candidate.transition.nodePlacementCandidate = true;
  candidate.evaluation.peerBytes = placement.edgePlan.totalPeerBytes;
  candidate.evaluation.topologyHopByteWork = placement.topologyHopByteWork;
  candidate.evaluation.scheduledEventCount = placement.schedule.eventCount;
  candidate.evaluation.scheduledMakespan = placement.schedule.makespan;
  candidate.evaluation.scheduledPeakLiveSPMBytes =
      placement.schedule.peakLiveSPMBytes;
  candidate.evaluation.criticalStructuredElementWork =
      placement.schedule.makespan;
  candidate.evaluation.distinctTileGroupCount =
      placement.distinctTileGroupCount;
  candidate.evaluation.partialOverlapEdgeCount =
      placement.partialOverlapEdgeCount;
  candidate.evaluation.disjointEdgeCount = placement.disjointEdgeCount;
  candidate.assignment.nodePlacements = placement.nodePlacements;
  candidate.evaluation.parallelComponentCount =
      getDisjointDependencyComponentCount(dag,
                                          candidate.assignment.nodePlacements);
  for (const WholeDAGObservablePlacement &selected :
       placement.outputPlacements) {
    if (selected.outputIndex >= outputDomains.size() ||
        selected.shardDimension >= outputDomains[selected.outputIndex].size() ||
        selected.tiles.empty())
      return std::nullopt;
    CardOutputSpatialMapping output;
    output.outputIndex = selected.outputIndex;
    output.shardDimension = selected.shardDimension;
    output.activeTileIds = selected.tiles;
    output.temporalTileSizes.assign(outputDomains[selected.outputIndex].begin(),
                                    outputDomains[selected.outputIndex].end());
    candidate.evaluation.shardImbalance = saturatingAdd(
        candidate.evaluation.shardImbalance,
        static_cast<uint64_t>(
            outputDomains[selected.outputIndex][selected.shardDimension]) %
            selected.tiles.size());
    candidate.assignment.mapping.outputs.push_back(std::move(output));
  }
  llvm::sort(candidate.assignment.mapping.outputs,
             [](const CardOutputSpatialMapping &lhs,
                const CardOutputSpatialMapping &rhs) {
               return lhs.outputIndex < rhs.outputIndex;
             });
  candidate.assignment.mapping.edgeStrategies.assign(
      placement.edgePlan.strategies.begin(),
      placement.edgePlan.strategies.end());
  candidate.evaluation.queryExpansionWork = saturatingMultiply(
      saturatingAdd(dag.getNodes().size(), dag.getEdges().size()),
      saturatingAdd(
          getActiveTileAssignmentCount(candidate.assignment.mapping,
                                       candidate.assignment.nodePlacements),
          getPeerFragmentCount(candidate.assignment.mapping)));
  return candidate;
}

static std::optional<uint64_t> getElementByteWidth(mlir::Type type) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped)
    return std::nullopt;
  mlir::Type elementType = shaped.getElementType();
  unsigned bits = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(elementType))
    bits = integer.getWidth();
  else if (auto floating = mlir::dyn_cast<mlir::FloatType>(elementType))
    bits = floating.getWidth();
  else
    return std::nullopt;
  return std::max<uint64_t>(1, ceilDivide(bits, 8));
}

static llvm::SmallVector<int64_t, 4>
getMaximumSpatialShardShape(const CardOutputSpatialMapping &mapping,
                            llvm::ArrayRef<int64_t> outputDomain) {
  llvm::SmallVector<int64_t, 4> shape(outputDomain.begin(), outputDomain.end());
  shape[mapping.shardDimension] = static_cast<int64_t>(
      ceilDivide(static_cast<uint64_t>(outputDomain[mapping.shardDimension]),
                 mapping.activeTileIds.size()));
  return shape;
}

static std::optional<uint64_t> getStaticTensorBytes(mlir::Type type) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  std::optional<uint64_t> elementBytes = getElementByteWidth(type);
  if (!shaped || !shaped.hasStaticShape() || !elementBytes)
    return std::nullopt;
  uint64_t elements = 1;
  for (int64_t extent : shaped.getShape())
    elements = saturatingMultiply(elements, static_cast<uint64_t>(extent));
  return saturatingMultiply(elements, *elementBytes);
}

/// A type/SSA-derived upper estimate for a fused dependency component's live
/// tensor volume relative to one observable output.  Counting only the
/// largest operand/result arity of one op badly underestimates DAGs whose
/// intermediates expand a dimension (for example scores followed by
/// reduction and broadcast).  Count each tensor SSA value once across the
/// current dependency component and normalize by the output's physical byte
/// volume.  This remains a cheap candidate-derivation bound; exact lifetime
/// and fixed-capacity packing stay the legality owner.
static uint64_t getOutputFusedTensorByteScale(const CardDAGAnalysis &dag,
                                              unsigned outputIndex) {
  if (outputIndex >= dag.getFunction().getNumResults())
    return 1;
  std::optional<uint64_t> outputBytes =
      getStaticTensorBytes(dag.getFunction().getResultTypes()[outputIndex]);
  if (!outputBytes || *outputBytes == 0)
    return 1;

  llvm::DenseSet<mlir::Value> countedValues;
  uint64_t componentTensorBytes = 0;
  for (const CardDAGDependencyComponent &component :
       dag.getObservableDependencyComponents()) {
    if (!llvm::is_contained(component.observableOutputs, outputIndex))
      continue;
    for (CardDAGNodeID nodeID : component.nodes) {
      const CardDAGNode *node = dag.getNode(nodeID);
      if (!node || !node->operation)
        continue;
      auto count = [&](mlir::Value value) {
        if (!mlir::isa<mlir::RankedTensorType>(value.getType()) ||
            !countedValues.insert(value).second)
          return;
        if (std::optional<uint64_t> bytes =
                getStaticTensorBytes(value.getType()))
          componentTensorBytes = saturatingAdd(componentTensorBytes, *bytes);
      };
      for (mlir::Value operand : node->operation->getOperands())
        count(operand);
      for (mlir::Value result : node->operation->getResults())
        count(result);
    }
    break;
  }
  return std::max<uint64_t>(1, ceilDivide(componentTensorBytes, *outputBytes));
}

static uint64_t getElementProduct(llvm::ArrayRef<int64_t> shape) {
  uint64_t elements = 1;
  for (int64_t extent : shape)
    elements = saturatingMultiply(elements, static_cast<uint64_t>(extent));
  return elements;
}

static uint64_t estimateAlignedResidencyBytes(
    llvm::ArrayRef<int64_t> temporalShape, uint64_t elementBytes,
    uint64_t tensorMultiplicity, const TargetMemoryPolicy &memory) {
  const uint64_t allocationBytes =
      saturatingMultiply(getElementProduct(temporalShape), elementBytes);
  const uint64_t alignedAllocation = saturatingAlignTo(
      allocationBytes, static_cast<uint64_t>(memory.spmAlignment));
  return saturatingMultiply(alignedAllocation, tensorMultiplicity);
}

/// Returns the lower end of the ceilDiv-wave equivalence class containing
/// `requested`.  This is a real wave-count breakpoint, not an arbitrary size
/// decrement.
static int64_t getWaveBreakpointAtOrBelow(int64_t extent, int64_t requested) {
  requested = std::clamp<int64_t>(requested, 1, extent);
  const uint64_t waves = ceilDivide(static_cast<uint64_t>(extent),
                                    static_cast<uint64_t>(requested));
  return static_cast<int64_t>(ceilDivide(static_cast<uint64_t>(extent), waves));
}

/// Returns the canonical breakpoint of the next strictly smaller tile-size
/// class. Incrementing the wave count by one is not sufficient: ceilDiv can
/// map several adjacent wave counts back to the same tile extent (for example
/// extent 32, tile 2, waves 16 -> 17 still yields tile 2).
static int64_t getNextLowerWaveBreakpoint(int64_t extent, int64_t current) {
  assert(extent > 0 && current > 1 && current <= extent);
  const uint64_t currentWaves =
      ceilDivide(static_cast<uint64_t>(extent), static_cast<uint64_t>(current));
  // ceilDiv(extent, tile) exceeds currentWaves exactly when tile is no
  // greater than floor((extent - 1) / currentWaves). This jumps directly to
  // the adjacent distinct wave-count class instead of probing wave counts.
  const uint64_t nextClassUpperBound =
      (static_cast<uint64_t>(extent) - 1) / currentWaves;
  assert(nextClassUpperBound >= 1 &&
         nextClassUpperBound < static_cast<uint64_t>(current));
  const int64_t next = getWaveBreakpointAtOrBelow(
      extent, static_cast<int64_t>(nextClassUpperBound));
  assert(next >= 1 && next < current &&
         "wave refinement must make strict tile-size progress");
  return next;
}

/// Returns the largest strictly smaller tile that divides the full iterator
/// extent. Actual allocation feedback prefers this structurally compact
/// coordinate: it removes a proven failed wave while avoiding a new static
/// tail class in every recursively fused producer. The common coordinate
/// search still evaluates the complete ceilDiv breakpoint domain; this only
/// changes which exact state is probed first after allocator rejection.
static int64_t getNextLowerPackingFeedbackBreakpoint(int64_t extent,
                                                     int64_t current) {
  assert(extent > 0 && current > 1 && current <= extent);
  int64_t best = 0;
  for (int64_t divisor = 1; divisor <= extent / divisor; ++divisor) {
    if (extent % divisor != 0)
      continue;
    if (divisor < current)
      best = std::max(best, divisor);
    const int64_t quotient = extent / divisor;
    if (quotient < current)
      best = std::max(best, quotient);
  }
  assert(best > 0 && best < current);
  return best;
}

/// Selects one adjacent finite wave breakpoint by the theoretical residency
/// reduction obtained per added runtime wave.  Every quantity comes from the
/// current static shape and known element bytes; ties use the iterator order.
/// This keeps one deterministic path through the complete iterator domain
/// instead of constructing a dimension Cartesian product.
static bool isOrderPreservingReductionRefinement(
    mlir::Operation *operation, unsigned candidateDimension,
    llvm::ArrayRef<int64_t> fullShape, llvm::ArrayRef<int64_t> currentShape);

static std::optional<unsigned>
selectTemporalRefinementAxis(llvm::ArrayRef<int64_t> fullShape,
                             llvm::ArrayRef<int64_t> currentShape,
                             uint64_t knownBytesPerIterationPoint,
                             mlir::Operation *operation = nullptr) {
  assert(fullShape.size() == currentShape.size());
  struct Selection {
    unsigned dimension = 0;
    uint64_t residencyReduction = 0;
    uint64_t addedWaves = 1;
  };
  std::optional<Selection> selected;
  for (auto [dimension, current] : llvm::enumerate(currentShape)) {
    if (current <= 1 ||
        (operation && !isOrderPreservingReductionRefinement(
                          operation, static_cast<unsigned>(dimension),
                          fullShape, currentShape)))
      continue;
    const int64_t next =
        getNextLowerWaveBreakpoint(fullShape[dimension], current);
    uint64_t otherElements = 1;
    for (auto [otherDimension, extent] : llvm::enumerate(currentShape))
      if (otherDimension != dimension)
        otherElements =
            saturatingMultiply(otherElements, static_cast<uint64_t>(extent));
    const uint64_t removedElements = saturatingMultiply(
        otherElements, static_cast<uint64_t>(current - next));
    const uint64_t reduction = saturatingMultiply(
        removedElements, std::max<uint64_t>(1, knownBytesPerIterationPoint));
    const uint64_t currentAxisWaves =
        ceilDivide(static_cast<uint64_t>(fullShape[dimension]),
                   static_cast<uint64_t>(current));
    const uint64_t nextAxisWaves =
        ceilDivide(static_cast<uint64_t>(fullShape[dimension]),
                   static_cast<uint64_t>(next));
    const uint64_t addedWaves = nextAxisWaves - currentAxisWaves;
    Selection candidate{static_cast<unsigned>(dimension), reduction,
                        addedWaves};
    if (!selected) {
      selected = candidate;
      continue;
    }
    const unsigned __int128 candidateBenefit =
        static_cast<unsigned __int128>(candidate.residencyReduction) *
        selected->addedWaves;
    const unsigned __int128 selectedBenefit =
        static_cast<unsigned __int128>(selected->residencyReduction) *
        candidate.addedWaves;
    if (candidateBenefit > selectedBenefit ||
        (candidateBenefit == selectedBenefit &&
         (candidate.residencyReduction > selected->residencyReduction ||
          (candidate.residencyReduction == selected->residencyReduction &&
           (candidate.addedWaves < selected->addedWaves ||
            (candidate.addedWaves == selected->addedWaves &&
             candidate.dimension < selected->dimension))))))
      selected = candidate;
  }
  return selected ? std::optional<unsigned>(selected->dimension) : std::nullopt;
}

static llvm::SmallVector<int64_t, 4> deriveCapacityTemporalShapeImpl(
    llvm::ArrayRef<int64_t> maximumShardShape, uint64_t elementBytes,
    uint64_t tensorMultiplicity, const TargetMemoryPolicy &memory,
    unsigned additionalWaveRefinements) {
  llvm::SmallVector<int64_t, 4> shape(maximumShardShape.begin(),
                                      maximumShardShape.end());
  if (shape.size() <= 4) {
    constexpr std::array<uint32_t, 4> nativeDimensionLimits = {
        Tx81InstructionLimits::dataShapeOuterMax,
        Tx81InstructionLimits::dataShapeOuterMax,
        Tx81InstructionLimits::dataShapeOuterMax,
        Tx81InstructionLimits::dataShapeChannelMax};
    const size_t leadingDimensions = 4 - shape.size();
    for (auto [dimension, extent] : llvm::enumerate(shape))
      shape[dimension] = std::min<int64_t>(
          extent, nativeDimensionLimits[leadingDimensions + dimension]);
  }
  const uint64_t capacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);
  const uint64_t knownBytesPerIterationPoint =
      saturatingMultiply(elementBytes, tensorMultiplicity);
  while (estimateAlignedResidencyBytes(shape, elementBytes, tensorMultiplicity,
                                       memory) > capacity) {
    std::optional<unsigned> selectedDimension = selectTemporalRefinementAxis(
        maximumShardShape, shape, knownBytesPerIterationPoint);
    if (!selectedDimension)
      break;
    const unsigned dimension = *selectedDimension;
    shape[dimension] = getNextLowerWaveBreakpoint(maximumShardShape[dimension],
                                                  shape[dimension]);
  }

  for (unsigned refinement = 0; refinement < additionalWaveRefinements;
       ++refinement) {
    std::optional<unsigned> selectedDimension = selectTemporalRefinementAxis(
        maximumShardShape, shape, knownBytesPerIterationPoint);
    if (!selectedDimension)
      break;
    const unsigned dimension = *selectedDimension;
    shape[dimension] = getNextLowerWaveBreakpoint(maximumShardShape[dimension],
                                                  shape[dimension]);
  }
  return shape;
}

static std::optional<llvm::SmallVector<int64_t, 4>>
getStaticIteratorRanges(mlir::Operation *operation) {
  auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(operation);
  if (!tiling)
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> ranges;
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation))
    ranges = linalg.getStaticLoopRanges();
  else if (operation->getNumResults() == 1) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(
        operation->getResult(0).getType());
    if (!type || !type.hasStaticShape() ||
        type.getRank() !=
            static_cast<int64_t>(tiling.getLoopIteratorTypes().size()))
      return std::nullopt;
    ranges.assign(type.getShape().begin(), type.getShape().end());
  } else {
    return std::nullopt;
  }
  if (ranges.size() != tiling.getLoopIteratorTypes().size() ||
      llvm::any_of(ranges, [](int64_t range) {
        return mlir::ShapedType::isDynamic(range) || range <= 0;
      }))
    return std::nullopt;
  return ranges;
}

/// Returns the largest iterator box that any Tile assigned to `placement`
/// can observe.  Temporal sizes are interpreted inside the spatially sharded
/// operation, so their legal upper bound is the balanced spatial shard, not
/// the unpartitioned source iteration domain.
static std::optional<llvm::SmallVector<int64_t, 4>>
getMaximumSpatialIteratorRanges(mlir::Operation *operation,
                                const WholeDAGNodePlacement &placement) {
  std::optional<llvm::SmallVector<int64_t, 4>> ranges =
      getStaticIteratorRanges(operation);
  if (!ranges || placement.iteratorPartitionFactors.size() != ranges->size())
    return std::nullopt;
  for (auto [dimension, factor] :
       llvm::enumerate(placement.iteratorPartitionFactors)) {
    if (factor == 0)
      return std::nullopt;
    (*ranges)[dimension] = static_cast<int64_t>(
        ceilDivide(static_cast<uint64_t>((*ranges)[dimension]), factor));
  }
  return ranges;
}

using SignedRange = std::pair<int64_t, int64_t>;

static std::optional<int64_t> narrowSigned(__int128 value) {
  if (value > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  if (value < std::numeric_limits<int64_t>::min())
    return std::nullopt;
  return static_cast<int64_t>(value);
}

/// Computes a constant bounding interval for an affine operand subscript over
/// the local iterator box [0, tileSize). Symbolic expressions are deliberately
/// omitted: an unavailable theoretical term must not become an invented cost.
static std::optional<SignedRange>
getAffineTileRange(mlir::AffineExpr expression,
                   llvm::ArrayRef<int64_t> iteratorTiles) {
  if (auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression))
    return SignedRange{constant.getValue(), constant.getValue()};
  if (auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
    if (dimension.getPosition() >= iteratorTiles.size() ||
        iteratorTiles[dimension.getPosition()] <= 0)
      return std::nullopt;
    return SignedRange{0, iteratorTiles[dimension.getPosition()] - 1};
  }
  if (mlir::isa<mlir::AffineSymbolExpr>(expression))
    return std::nullopt;
  auto binary = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression);
  if (!binary)
    return std::nullopt;
  std::optional<SignedRange> lhs =
      getAffineTileRange(binary.getLHS(), iteratorTiles);
  std::optional<SignedRange> rhs =
      getAffineTileRange(binary.getRHS(), iteratorTiles);
  if (!lhs || !rhs)
    return std::nullopt;

  switch (expression.getKind()) {
  case mlir::AffineExprKind::Add: {
    std::optional<int64_t> lower =
        narrowSigned(static_cast<__int128>(lhs->first) + rhs->first);
    std::optional<int64_t> upper =
        narrowSigned(static_cast<__int128>(lhs->second) + rhs->second);
    if (!lower || !upper)
      return std::nullopt;
    return SignedRange{*lower, *upper};
  }
  case mlir::AffineExprKind::Mul: {
    const std::array<__int128, 4> products{
        static_cast<__int128>(lhs->first) * rhs->first,
        static_cast<__int128>(lhs->first) * rhs->second,
        static_cast<__int128>(lhs->second) * rhs->first,
        static_cast<__int128>(lhs->second) * rhs->second};
    auto [minimum, maximum] =
        std::minmax_element(products.begin(), products.end());
    std::optional<int64_t> lower = narrowSigned(*minimum);
    std::optional<int64_t> upper = narrowSigned(*maximum);
    if (!lower || !upper)
      return std::nullopt;
    return SignedRange{*lower, *upper};
  }
  case mlir::AffineExprKind::FloorDiv:
  case mlir::AffineExprKind::CeilDiv:
  case mlir::AffineExprKind::Mod:
    // Without the candidate's absolute iterator offset, quotient/remainder
    // boundaries can cross inside a tile even when the zero-based interval
    // does not. Omit this operand relation instead of underestimating it.
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

/// Theoretical SPM residency of the structured operand tiles described by
/// the operation's current indexing maps. Each known buffer is aligned
/// independently. Unsupported/symbolic operand relations are omitted rather
/// than represented by an "unknown" sentinel or a guessed multiplier.
static std::optional<uint64_t>
estimateKnownIteratorResidencyBytes(mlir::Operation *operation,
                                    llvm::ArrayRef<int64_t> iteratorTiles,
                                    const TargetMemoryPolicy &memory) {
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!linalg)
    return std::nullopt;
  llvm::SmallVector<mlir::AffineMap, 4> maps = linalg.getIndexingMapsArray();
  if (maps.size() != operation->getNumOperands())
    return std::nullopt;

  uint64_t residency = 0;
  bool hasKnownBuffer = false;
  for (auto [operand, map] : llvm::zip_equal(operation->getOperands(), maps)) {
    auto shaped = mlir::dyn_cast<mlir::ShapedType>(operand.getType());
    std::optional<uint64_t> elementBytes =
        getElementByteWidth(operand.getType());
    if (!shaped || !shaped.hasRank() || !elementBytes ||
        map.getNumDims() != iteratorTiles.size() ||
        map.getNumResults() != static_cast<unsigned>(shaped.getRank()))
      continue;
    uint64_t elements = 1;
    bool knownRelation = true;
    for (mlir::AffineExpr expression : map.getResults()) {
      std::optional<SignedRange> range =
          getAffineTileRange(expression, iteratorTiles);
      if (!range || range->second < range->first) {
        knownRelation = false;
        break;
      }
      const unsigned __int128 extent =
          static_cast<unsigned __int128>(static_cast<__int128>(range->second) -
                                         range->first) +
          1;
      elements = saturatingMultiply(
          elements, extent > std::numeric_limits<uint64_t>::max()
                        ? std::numeric_limits<uint64_t>::max()
                        : static_cast<uint64_t>(extent));
    }
    if (!knownRelation)
      continue;
    const uint64_t bytes = saturatingMultiply(elements, *elementBytes);
    residency = saturatingAdd(
        residency,
        saturatingAlignTo(bytes, static_cast<uint64_t>(memory.spmAlignment)));
    hasKnownBuffer = true;
  }
  return hasKnownBuffer ? std::optional<uint64_t>(residency) : std::nullopt;
}

/// Exact result-window bytes for one selected iterator tile when the current
/// Linalg result map is rectangular.  An unsupported affine projection is an
/// omitted search term; it is never converted into zero or a guessed scale.
static std::optional<uint64_t>
estimateKnownResultTileBytes(mlir::Operation *operation, unsigned resultNumber,
                             llvm::ArrayRef<int64_t> iteratorTiles,
                             const TargetMemoryPolicy &memory) {
  auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(operation);
  if (!linalg || resultNumber >= operation->getNumResults())
    return std::nullopt;
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getResult(resultNumber).getType());
  std::optional<uint64_t> elementBytes =
      getElementByteWidth(operation->getResult(resultNumber).getType());
  if (!type || !type.hasStaticShape() || !elementBytes)
    return std::nullopt;
  mlir::AffineMap map =
      linalg.getIndexingMapMatchingResult(operation->getResult(resultNumber));
  if (!map || map.getNumDims() != iteratorTiles.size() ||
      map.getNumResults() != static_cast<unsigned>(type.getRank()))
    return std::nullopt;
  uint64_t elements = 1;
  for (mlir::AffineExpr expression : map.getResults()) {
    std::optional<SignedRange> range =
        getAffineTileRange(expression, iteratorTiles);
    if (!range || range->second < range->first)
      return std::nullopt;
    elements = saturatingMultiply(
        elements, static_cast<uint64_t>(range->second - range->first + 1));
  }
  return saturatingAlignTo(saturatingMultiply(elements, *elementBytes),
                           static_cast<uint64_t>(memory.spmAlignment));
}

static bool
requiresLexicographicFloatReductionOrder(mlir::Operation *operation) {
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!linalg || linalg.getRegionOutputArgs().size() != 1)
    return false;
  llvm::SmallVector<mlir::Operation *, 1> combinerOps;
  mlir::Value reducedValue = mlir::matchReduction(linalg.getRegionOutputArgs(),
                                                  /*redPos=*/0, combinerOps);
  if (!reducedValue || combinerOps.size() != 1)
    return false;
  auto addf = mlir::dyn_cast<mlir::arith::AddFOp>(combinerOps.front());
  return addf && !static_cast<bool>(addf.getFastmath() &
                                    mlir::arith::FastMathFlags::reassoc);
}

static bool isOrderPreservingReductionRefinement(
    mlir::Operation *operation, unsigned candidateDimension,
    llvm::ArrayRef<int64_t> fullShape, llvm::ArrayRef<int64_t> currentShape) {
  if (!requiresLexicographicFloatReductionOrder(operation))
    return true;
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(operation);
  if (!tiling)
    return false;
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      tiling.getLoopIteratorTypes();
  if (candidateDimension >= iteratorTypes.size() ||
      iteratorTypes[candidateDimension] != mlir::utils::IteratorType::reduction)
    return true;
  for (unsigned dimension = 0; dimension < candidateDimension; ++dimension)
    if (iteratorTypes[dimension] == mlir::utils::IteratorType::reduction &&
        fullShape[dimension] > 1 && currentShape[dimension] > 1)
      return false;
  return true;
}

static std::optional<unsigned> selectOperationTemporalRefinementAxis(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> fullShape,
    llvm::ArrayRef<int64_t> currentShape, const TargetMemoryPolicy &memory) {
  std::optional<uint64_t> currentResidency =
      estimateKnownIteratorResidencyBytes(operation, currentShape, memory);
  if (!currentResidency)
    return selectTemporalRefinementAxis(fullShape, currentShape,
                                        /*knownBytesPerIterationPoint=*/1,
                                        operation);

  struct Selection {
    unsigned dimension = 0;
    uint64_t residencyReduction = 0;
    uint64_t addedWaves = 1;
  };
  std::optional<Selection> selected;
  for (auto [dimension, current] : llvm::enumerate(currentShape)) {
    if (current <= 1 || !isOrderPreservingReductionRefinement(
                            operation, static_cast<unsigned>(dimension),
                            fullShape, currentShape))
      continue;
    llvm::SmallVector<int64_t, 4> nextShape(currentShape.begin(),
                                            currentShape.end());
    nextShape[dimension] =
        getNextLowerWaveBreakpoint(fullShape[dimension], current);
    std::optional<uint64_t> nextResidency =
        estimateKnownIteratorResidencyBytes(operation, nextShape, memory);
    if (!nextResidency || *nextResidency >= *currentResidency)
      continue;
    const uint64_t currentAxisWaves =
        ceilDivide(static_cast<uint64_t>(fullShape[dimension]),
                   static_cast<uint64_t>(current));
    const uint64_t nextAxisWaves =
        ceilDivide(static_cast<uint64_t>(fullShape[dimension]),
                   static_cast<uint64_t>(nextShape[dimension]));
    Selection candidate{static_cast<unsigned>(dimension),
                        *currentResidency - *nextResidency,
                        nextAxisWaves - currentAxisWaves};
    if (!selected) {
      selected = candidate;
      continue;
    }
    const unsigned __int128 candidateBenefit =
        static_cast<unsigned __int128>(candidate.residencyReduction) *
        selected->addedWaves;
    const unsigned __int128 selectedBenefit =
        static_cast<unsigned __int128>(selected->residencyReduction) *
        candidate.addedWaves;
    if (candidateBenefit > selectedBenefit ||
        (candidateBenefit == selectedBenefit &&
         (candidate.residencyReduction > selected->residencyReduction ||
          (candidate.residencyReduction == selected->residencyReduction &&
           (candidate.addedWaves < selected->addedWaves ||
            (candidate.addedWaves == selected->addedWaves &&
             candidate.dimension < selected->dimension))))))
      selected = candidate;
  }
  if (selected)
    return selected->dimension;
  // Alignment can hide the benefit of an early breakpoint. Preserve finite
  // progress using only iterator geometry until a modeled buffer shrinks.
  return selectTemporalRefinementAxis(fullShape, currentShape,
                                      /*knownBytesPerIterationPoint=*/1,
                                      operation);
}

static llvm::SmallVector<int64_t, 4> deriveOperationTemporalShape(
    mlir::Operation *operation, llvm::ArrayRef<int64_t> iteratorRanges,
    const TargetMemoryPolicy &memory, unsigned additionalWaveRefinements,
    bool minimumEndpoint, uint64_t coexistingProducerResidencyBytes = 0) {
  llvm::SmallVector<int64_t, 4> shape(iteratorRanges.begin(),
                                      iteratorRanges.end());
  if (minimumEndpoint) {
    std::fill(shape.begin(), shape.end(), 1);
    return shape;
  }

  const uint64_t capacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);
  while (true) {
    std::optional<uint64_t> residency =
        estimateKnownIteratorResidencyBytes(operation, shape, memory);
    if (!residency || *residency + coexistingProducerResidencyBytes <= capacity)
      break;
    std::optional<unsigned> dimension = selectOperationTemporalRefinementAxis(
        operation, iteratorRanges, shape, memory);
    if (!dimension)
      break;
    shape[*dimension] = getNextLowerWaveBreakpoint(iteratorRanges[*dimension],
                                                   shape[*dimension]);
  }
  for (unsigned refinement = 0; refinement < additionalWaveRefinements;
       ++refinement) {
    std::optional<unsigned> dimension = selectOperationTemporalRefinementAxis(
        operation, iteratorRanges, shape, memory);
    if (!dimension)
      break;
    shape[*dimension] = getNextLowerWaveBreakpoint(iteratorRanges[*dimension],
                                                   shape[*dimension]);
  }
  return shape;
}

/// Initializes the complete per-operation temporal coordinate vector.  No
/// footprint estimate decides feasibility here: search begins from the
/// full iterator extents and actual SPM packing remains the capacity owner.
/// `additionalWaveRefinements` is only the deterministic recurrence used
/// after an exact failure; it orders legal finite states and cannot accept one.
static bool populateOperationTemporalTiles(WholeCardCandidate &candidate,
                                           const CardDAGAnalysis &dag,
                                           unsigned additionalWaveRefinements,
                                           bool minimumEndpoint = false) {
  candidate.assignment.mapping.operationTemporalTiles.clear();
  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 16> rangesByNode;
  rangesByNode.reserve(dag.getNodes().size());
  for (const CardDAGNode &node : dag.getNodes()) {
    if (node.id >= candidate.assignment.nodePlacements.size() ||
        candidate.assignment.nodePlacements[node.id].node != node.id)
      return false;
    std::optional<llvm::SmallVector<int64_t, 4>> ranges =
        getMaximumSpatialIteratorRanges(
            node.operation, candidate.assignment.nodePlacements[node.id]);
    if (!ranges)
      return false;
    rangesByNode.push_back(std::move(*ranges));
    StructuredOpTemporalTile tile;
    tile.operation = node.operation;
    tile.iteratorTileSizes = rangesByNode.back();
    if (minimumEndpoint)
      std::fill(tile.iteratorTileSizes.begin(), tile.iteratorTileSizes.end(),
                1);
    candidate.assignment.mapping.operationTemporalTiles.push_back(
        std::move(tile));
  }
  if (minimumEndpoint)
    return true;
  for (unsigned refinement = 0; refinement < additionalWaveRefinements;
       ++refinement) {
    bool changed = false;
    for (auto [node, tile] :
         llvm::enumerate(candidate.assignment.mapping.operationTemporalTiles)) {
      std::optional<unsigned> dimension = selectOperationTemporalRefinementAxis(
          tile.operation, rangesByNode[node], tile.iteratorTileSizes, memory);
      if (!dimension)
        continue;
      tile.iteratorTileSizes[*dimension] = getNextLowerWaveBreakpoint(
          rangesByNode[node][*dimension], tile.iteratorTileSizes[*dimension]);
      changed = true;
    }
    if (!changed)
      break;
  }
  return true;
}

[[maybe_unused]] static bool populateEstimatedOperationTemporalTiles(
    WholeCardCandidate &candidate, const CardDAGAnalysis &dag,
    unsigned additionalWaveRefinements, bool minimumEndpoint = false) {
  candidate.assignment.mapping.operationTemporalTiles.clear();
  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  const size_t nodeCount = dag.getNodes().size();

  // Node ids follow direct structured-operation order in the single-block
  // function, so producers always derive before their consumers.
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 16> nodeRanges;
  nodeRanges.reserve(nodeCount);
  for (const CardDAGNode &node : dag.getNodes()) {
    std::optional<llvm::SmallVector<int64_t, 4>> ranges =
        getStaticIteratorRanges(node.operation);
    auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(node.operation);
    if (!ranges || !tiling)
      return false;
    nodeRanges.push_back(std::move(*ranges));
  }

  // In-region producer relations: a producer traversed inside the consumer's
  // TileRegion keeps its load/output transients live while its slices are
  // staged into the consumer's buffer, so those transients are part of the
  // consumer's SPM residency.  Three sources make a producer in-region:
  //   * an edge strategy that runs the producer traversal inside the
  //     consumer's TileRegion (CoupledFusion, LocalShardResidency, Recompute);
  //     PeerFragments runs the producer in its own Tile, SpillReload stores
  //     it to DDR, and RegionCut releases SPM at the region boundary;
  //   * a DAG edge that reaches the producer through a pure view chain
  //     (reshape / transpose / expand / collapse) gets no edge strategy --
  //     the materialization contract resolves it inside the consumer's typed
  //     lowering -- yet its load/output transients still coexist with the
  //     consumer staging;
  //   * a direct DPS-init operand is the consumer's own output buffer and is
  //     already part of the consumer's residency estimate.
  // Deduplicate by producer node: one materialization carries one transient
  // pair no matter how many consumer operands it feeds.
  llvm::SmallVector<llvm::SmallVector<CardDAGNodeID, 4>, 16> inRegionProducers;
  inRegionProducers.resize(nodeCount);
  llvm::DenseMap<mlir::Operation *, CardDAGNodeID> nodeByOperation;
  for (const CardDAGNode &node : dag.getNodes())
    nodeByOperation.try_emplace(node.operation, node.id);
  auto isInRegionAction = [](SpatialEdgeAction action) {
    switch (action) {
    case SpatialEdgeAction::CoupledFusion:
    case SpatialEdgeAction::LocalShardResidency:
    case SpatialEdgeAction::Recompute:
      return true;
    default:
      return false;
    }
  };
  auto addInRegionProducer = [&](mlir::Operation *consumer,
                                 mlir::Operation *producer) {
    auto consumerIt = nodeByOperation.find(consumer);
    auto producerIt = nodeByOperation.find(producer);
    if (consumerIt == nodeByOperation.end() ||
        producerIt == nodeByOperation.end())
      return;
    llvm::SmallVector<CardDAGNodeID, 4> &producers =
        inRegionProducers[consumerIt->second];
    if (!llvm::is_contained(producers, producerIt->second))
      producers.push_back(producerIt->second);
  };
  for (const SpatialEdgeStrategy &strategy :
       candidate.assignment.mapping.edgeStrategies) {
    if (!strategy.consumer || !strategy.producer ||
        !isInRegionAction(strategy.action))
      continue;
    addInRegionProducer(strategy.consumer, strategy.producer);
  }
  // Strategy-less DAG edges are either direct DPS-init operands (the
  // consumer's own output buffer) or view chains resolved inside the
  // consumer's traversal; only the latter contribute coexisting transients.
  for (const CardDAGEdge &edge : dag.getEdges()) {
    const CardDAGNode *producerNode = dag.getNode(edge.producer);
    const CardDAGNode *consumerNode = dag.getNode(edge.consumer);
    if (!producerNode || !consumerNode ||
        consumerNode->operation->getOperand(edge.consumerOperand) ==
            producerNode->operation->getResult(edge.producerResult))
      continue;
    addInRegionProducer(consumerNode->operation, producerNode->operation);
  }

  llvm::SmallVector<StructuredOpTemporalTile, 16> selected;
  selected.reserve(nodeCount);
  // `selected` is indexed by node id and grows in node order, so a producer
  // tile is always available when its consumers derive.
  auto coexistingProducerBytes = [&](CardDAGNodeID consumer) -> uint64_t {
    uint64_t bytes = 0;
    for (CardDAGNodeID producer : inRegionProducers[consumer]) {
      std::optional<uint64_t> residency = estimateKnownIteratorResidencyBytes(
          selected[producer].operation, selected[producer].iteratorTileSizes,
          memory);
      if (residency)
        bytes = saturatingAdd(bytes, *residency);
    }
    return bytes;
  };
  auto isShrinkable = [&](CardDAGNodeID nodeID) {
    const StructuredOpTemporalTile &tile = selected[nodeID];
    return selectOperationTemporalRefinementAxis(tile.operation,
                                                 nodeRanges[nodeID],
                                                 tile.iteratorTileSizes, memory)
        .has_value();
  };

  const uint64_t capacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);
  for (const CardDAGNode &node : dag.getNodes()) {
    StructuredOpTemporalTile nodeTile;
    nodeTile.operation = node.operation;
    if (minimumEndpoint) {
      nodeTile.iteratorTileSizes = nodeRanges[node.id];
      std::fill(nodeTile.iteratorTileSizes.begin(),
                nodeTile.iteratorTileSizes.end(), 1);
    } else {
      nodeTile.iteratorTileSizes = deriveOperationTemporalShape(
          node.operation, nodeRanges[node.id], memory,
          additionalWaveRefinements, /*minimumEndpoint=*/false,
          coexistingProducerBytes(node.id));
      // A consumer that still cannot fit after its own refinement must not
      // collapse to a degenerate tile while its in-region producer keeps a
      // large transient pair.  Refine the largest shrinkable producer one
      // breakpoint at a time; the shrink also relaxes every other consumer
      // of the same output.  The exact packer remains the legality owner.
      while (true) {
        std::optional<uint64_t> ownResidency =
            estimateKnownIteratorResidencyBytes(
                node.operation, nodeTile.iteratorTileSizes, memory);
        if (!ownResidency ||
            *ownResidency + coexistingProducerBytes(node.id) <= capacity)
          break;
        CardDAGNodeID largestProducer =
            std::numeric_limits<CardDAGNodeID>::max();
        uint64_t largestBytes = 0;
        for (CardDAGNodeID producer : inRegionProducers[node.id]) {
          if (!isShrinkable(producer))
            continue;
          std::optional<uint64_t> producerResidency =
              estimateKnownIteratorResidencyBytes(
                  selected[producer].operation,
                  selected[producer].iteratorTileSizes, memory);
          if (producerResidency && *producerResidency > largestBytes) {
            largestBytes = *producerResidency;
            largestProducer = producer;
          }
        }
        if (largestProducer == std::numeric_limits<CardDAGNodeID>::max())
          break;
        StructuredOpTemporalTile &producerTile = selected[largestProducer];
        std::optional<unsigned> axis = selectOperationTemporalRefinementAxis(
            producerTile.operation, nodeRanges[largestProducer],
            producerTile.iteratorTileSizes, memory);
        if (!axis)
          break;
        producerTile.iteratorTileSizes[*axis] =
            getNextLowerWaveBreakpoint(nodeRanges[largestProducer][*axis],
                                       producerTile.iteratorTileSizes[*axis]);
        nodeTile.iteratorTileSizes = deriveOperationTemporalShape(
            node.operation, nodeRanges[node.id], memory,
            additionalWaveRefinements, /*minimumEndpoint=*/false,
            coexistingProducerBytes(node.id));
      }
    }
    selected.push_back(std::move(nodeTile));
  }
  candidate.assignment.mapping.operationTemporalTiles = std::move(selected);
  return candidate.assignment.mapping.operationTemporalTiles.size() ==
         nodeCount;
}

static void populateTemporalMetrics(WholeCardCandidate &candidate,
                                    const StaticOutputDomains &outputDomains,
                                    const CardDAGAnalysis &dag) {
  candidate.evaluation.temporalWaveLowerBound = 0;
  candidate.evaluation.instructionExecutionLowerBound = 0;
  candidate.evaluation.peakOutputTileFootprintEstimate = 0;
  candidate.evaluation.peakAlignedResidencyEstimate = 0;
  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  for (const CardOutputSpatialMapping &output :
       candidate.assignment.mapping.outputs) {
    llvm::SmallVector<int64_t, 4> shardShape =
        getMaximumSpatialShardShape(output, outputDomains[output.outputIndex]);
    uint64_t waves = 1;
    llvm::SmallVector<int64_t, 4> effectiveTemporalShape;
    for (auto [shardExtent, temporalExtent] :
         llvm::zip_equal(shardShape, output.temporalTileSizes)) {
      const int64_t effective = std::min(shardExtent, temporalExtent);
      effectiveTemporalShape.push_back(effective);
      waves = saturatingMultiply(waves,
                                 ceilDivide(static_cast<uint64_t>(shardExtent),
                                            static_cast<uint64_t>(effective)));
    }
    candidate.evaluation.temporalWaveLowerBound =
        std::max(candidate.evaluation.temporalWaveLowerBound, waves);
    candidate.evaluation.instructionExecutionLowerBound =
        saturatingAdd(candidate.evaluation.instructionExecutionLowerBound,
                      saturatingMultiply(
                          waves, std::max<uint64_t>(1, dag.getNodes().size())));
    std::optional<uint64_t> elementBytes = getElementByteWidth(
        dag.getFunction().getResultTypes()[output.outputIndex]);
    if (!elementBytes)
      continue;
    const uint64_t tileBytes = saturatingMultiply(
        getElementProduct(effectiveTemporalShape), *elementBytes);
    candidate.evaluation.peakOutputTileFootprintEstimate = std::max(
        candidate.evaluation.peakOutputTileFootprintEstimate, tileBytes);
    candidate.evaluation.peakAlignedResidencyEstimate = std::max(
        candidate.evaluation.peakAlignedResidencyEstimate,
        estimateAlignedResidencyBytes(
            effectiveTemporalShape, *elementBytes,
            getOutputFusedTensorByteScale(dag, output.outputIndex), memory));
  }
  for (auto [node, selected] :
       llvm::enumerate(candidate.assignment.mapping.operationTemporalTiles)) {
    if (node >= candidate.assignment.nodePlacements.size())
      continue;
    std::optional<llvm::SmallVector<int64_t, 4>> ranges =
        getMaximumSpatialIteratorRanges(
            selected.operation, candidate.assignment.nodePlacements[node]);
    auto tiling =
        mlir::dyn_cast_or_null<mlir::TilingInterface>(selected.operation);
    if (!ranges || !tiling ||
        ranges->size() != selected.iteratorTileSizes.size())
      continue;
    uint64_t operationWaves = 1;
    for (unsigned dimension = 0; dimension < ranges->size(); ++dimension)
      operationWaves = saturatingMultiply(
          operationWaves,
          ceilDivide(
              static_cast<uint64_t>((*ranges)[dimension]),
              static_cast<uint64_t>(selected.iteratorTileSizes[dimension])));
    candidate.evaluation.temporalWaveLowerBound =
        std::max(candidate.evaluation.temporalWaveLowerBound, operationWaves);
    candidate.evaluation.instructionExecutionLowerBound = saturatingAdd(
        candidate.evaluation.instructionExecutionLowerBound, operationWaves);
  }
}

static std::optional<CardDAGEdgeID>
resolveCandidateStrategyEdge(const CardDAGAnalysis &dag,
                             const SpatialEdgeStrategy &strategy) {
  for (const CardDAGEdge &edge : dag.getEdges()) {
    const CardDAGNode *producer = dag.getNode(edge.producer);
    const CardDAGNode *consumer = dag.getNode(edge.consumer);
    if (producer && consumer && producer->operation == strategy.producer &&
        consumer->operation == strategy.consumer &&
        edge.producerResult == strategy.producerResult &&
        edge.consumerOperand == strategy.consumerOperand)
      return edge.id;
  }
  return std::nullopt;
}

static std::optional<uint64_t> getStrategyTemporalBytes(
    const SpatialEdgeStrategy &strategy, const CardDAGAnalysis &dag,
    const WholeCardCandidate &candidate, const TargetMemoryPolicy &memory) {
  std::optional<CardDAGEdgeID> edgeID =
      resolveCandidateStrategyEdge(dag, strategy);
  const CardDAGEdge *edge = edgeID ? dag.getEdge(*edgeID) : nullptr;
  if (!edge || edge->producer >=
                   candidate.assignment.mapping.operationTemporalTiles.size())
    return std::nullopt;
  const StructuredOpTemporalTile &tile =
      candidate.assignment.mapping.operationTemporalTiles[edge->producer];
  if (tile.operation != strategy.producer)
    return std::nullopt;
  return estimateKnownResultTileBytes(strategy.producer,
                                      strategy.producerResult,
                                      tile.iteratorTileSizes, memory);
}

struct CandidateResourceScheduleSummary {
  uint64_t eventCount = 0;
  uint64_t makespan = 0;
  uint64_t peakLiveSPMBytes = 0;
  uint64_t spmMovementWork = 0;
  uint64_t ddrMovementWork = 0;
};

/// Query-local memo for the exact input of `scheduleWholeDAGCandidate`.
/// Output traversal coordinates and reduction iterator coordinates that do
/// not change an edge footprint intentionally map to the same entry.  Every
/// coordinate is still evaluated and retains its own temporal metrics; only
/// repeated deterministic resource-calendar work is shared.
class CandidateResourceScheduleMemo {
public:
  struct Entry {
    std::vector<uint64_t> key;
    CandidateResourceScheduleSummary summary;
    std::string failureReason;
    std::condition_variable readyCondition;
    bool legal = false;
    bool ready = false;
  };

  struct Lookup {
    std::shared_ptr<Entry> entry;
    bool owner = false;
  };

  Lookup lookupOrCreate(std::vector<uint64_t> key) {
    const size_t hash =
        static_cast<size_t>(llvm::hash_combine_range(key.begin(), key.end()));
    std::unique_lock<std::mutex> lock(guard);
    std::vector<std::shared_ptr<Entry>> &bucket = entries[hash];
    auto found = llvm::find_if(
        bucket, [&](const auto &entry) { return entry->key == key; });
    if (found != bucket.end()) {
      ++hits;
      std::shared_ptr<Entry> entry = *found;
      while (!entry->ready)
        entry->readyCondition.wait(lock);
      return {std::move(entry), false};
    }
    ++misses;
    auto entry = std::make_shared<Entry>();
    entry->key = std::move(key);
    bucket.push_back(entry);
    return {std::move(entry), true};
  }

  void storeResult(const std::shared_ptr<Entry> &entry, bool legal,
                   CandidateResourceScheduleSummary summary,
                   std::string failureReason) {
    {
      std::lock_guard<std::mutex> lock(guard);
      entry->legal = legal;
      entry->summary = summary;
      entry->failureReason = std::move(failureReason);
      entry->ready = true;
    }
    entry->readyCondition.notify_all();
  }

  std::pair<uint64_t, uint64_t> getCounts() const {
    std::lock_guard<std::mutex> lock(guard);
    return {hits, misses};
  }

private:
  mutable std::mutex guard;
  std::unordered_map<size_t, std::vector<std::shared_ptr<Entry>>> entries;
  uint64_t hits = 0;
  uint64_t misses = 0;
};

static std::vector<uint64_t> buildCandidateResourceScheduleKey(
    llvm::ArrayRef<WholeDAGNodePlacement> nodePlacements,
    llvm::ArrayRef<WholeDAGLocalResidency> localResidencies,
    llvm::ArrayRef<WholeDAGPeerMovement> peerMovements,
    llvm::ArrayRef<WholeDAGLocalMovement> localMovements) {
  std::vector<uint64_t> key;
  key.reserve(nodePlacements.size() * 8 + localResidencies.size() * 3 +
              peerMovements.size() * 8 + localMovements.size() * 5 + 4);
  key.push_back(nodePlacements.size());
  for (const WholeDAGNodePlacement &placement : nodePlacements) {
    key.push_back(placement.node);
    key.push_back(placement.shardDimension);
    key.push_back(placement.spatialIteratorDimension);
    key.push_back(placement.tiles.size());
    for (PhysicalTileId tile : placement.tiles)
      key.push_back(static_cast<uint64_t>(tile.getValue()));
    key.push_back(placement.iteratorPartitionFactors.size());
    key.insert(key.end(), placement.iteratorPartitionFactors.begin(),
               placement.iteratorPartitionFactors.end());
  }
  key.push_back(localResidencies.size());
  for (const WholeDAGLocalResidency &residency : localResidencies) {
    key.push_back(residency.edge);
    key.push_back(static_cast<uint64_t>(residency.tile.getValue()));
    key.push_back(residency.footprintBytes);
  }
  key.push_back(peerMovements.size());
  for (const WholeDAGPeerMovement &movement : peerMovements) {
    key.push_back(movement.edge);
    key.push_back(static_cast<uint64_t>(movement.sourceTile.getValue()));
    key.push_back(static_cast<uint64_t>(movement.destinationTile.getValue()));
    key.push_back(movement.duration);
    key.push_back(movement.bufferCount);
    key.push_back(movement.route.size());
    for (const PhysicalTileDirectedLink &link : movement.route) {
      key.push_back(static_cast<uint64_t>(link.source.getValue()));
      key.push_back(static_cast<uint64_t>(link.destination.getValue()));
    }
  }
  key.push_back(localMovements.size());
  for (const WholeDAGLocalMovement &movement : localMovements) {
    key.push_back(movement.edge);
    key.push_back(static_cast<uint64_t>(movement.resource));
    key.push_back(static_cast<uint64_t>(movement.tile.getValue()));
    key.push_back(movement.duration);
    key.push_back(movement.bufferCount);
  }
  return key;
}

static void applyCandidateResourceScheduleSummary(
    WholeCardCandidate &candidate,
    const CandidateResourceScheduleSummary &summary) {
  candidate.evaluation.scheduledEventCount = summary.eventCount;
  candidate.evaluation.scheduledMakespan = summary.makespan;
  candidate.evaluation.scheduledPeakLiveSPMBytes = summary.peakLiveSPMBytes;
  candidate.evaluation.scheduledSPMMovementWork = summary.spmMovementWork;
  candidate.evaluation.scheduledDDRMovementWork = summary.ddrMovementWork;
  candidate.evaluation.criticalStructuredElementWork = summary.makespan;
}

static bool refreshCandidateResourceSchedule(
    WholeCardCandidate &candidate, const CardDAGAnalysis &dag,
    const PhysicalTopology &topology, PhysicalCardId cardId,
    llvm::ArrayRef<PhysicalTileId> availableTiles,
    std::string *failureReason = nullptr,
    CandidateResourceScheduleMemo *memo = nullptr) {
  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  std::map<std::pair<CardDAGEdgeID, int64_t>, uint64_t> residencyBytes;
  llvm::SmallVector<WholeDAGPeerMovement, 32> peerMovements;
  llvm::SmallVector<WholeDAGLocalMovement, 32> localMovements;

  auto addResidency = [&](CardDAGEdgeID edge, PhysicalTileId tile,
                          uint64_t bytes) {
    uint64_t &current = residencyBytes[{edge, tile.getValue()}];
    current = saturatingAdd(current, bytes);
  };
  for (const SpatialEdgeStrategy &strategy :
       candidate.assignment.mapping.edgeStrategies) {
    std::optional<CardDAGEdgeID> edgeID =
        resolveCandidateStrategyEdge(dag, strategy);
    std::optional<uint64_t> temporalBytes =
        getStrategyTemporalBytes(strategy, dag, candidate, memory);
    if (!edgeID || !temporalBytes || *temporalBytes == 0)
      continue;
    const uint64_t bufferedBytes = saturatingMultiply(
        *temporalBytes, std::max<uint8_t>(1, strategy.bufferCount));
    switch (strategy.action) {
    case SpatialEdgeAction::CoupledFusion:
    case SpatialEdgeAction::LocalShardResidency:
    case SpatialEdgeAction::Recompute:
    case SpatialEdgeAction::LocalPhysicalConversion:
      addResidency(*edgeID, strategy.destinationTile, bufferedBytes);
      break;
    case SpatialEdgeAction::PeerFragments:
      if (llvm::any_of(
              strategy.fragments, [](const SpatialEdgeFragment &fragment) {
                return fragment.kind == SpatialEdgeFragmentKind::Resident;
              }))
        addResidency(*edgeID, strategy.destinationTile, bufferedBytes);
      break;
    case SpatialEdgeAction::SpillReload:
    case SpatialEdgeAction::RegionCut:
      break;
    }

    const bool needsLayoutMovement =
        strategy.hasLayoutAssignment &&
        strategy.producerLayout != strategy.consumerLayout;
    if (needsLayoutMovement ||
        strategy.action == SpatialEdgeAction::LocalShardResidency ||
        strategy.action == SpatialEdgeAction::LocalPhysicalConversion ||
        strategy.action == SpatialEdgeAction::PeerFragments) {
      localMovements.push_back(WholeDAGLocalMovement{
          *edgeID, WholeDAGLocalMovementResource::TileSPM,
          strategy.destinationTile, *temporalBytes, strategy.bufferCount});
    }
    if (strategy.action == SpatialEdgeAction::SpillReload ||
        strategy.action == SpatialEdgeAction::RegionCut) {
      localMovements.push_back(WholeDAGLocalMovement{
          *edgeID, WholeDAGLocalMovementResource::CardDDR,
          strategy.destinationTile, saturatingMultiply(*temporalBytes, 2),
          strategy.bufferCount});
    }
    for (const SpatialEdgeFragment &fragment : strategy.fragments) {
      if (fragment.kind != SpatialEdgeFragmentKind::Peer || fragment.bytes == 0)
        continue;
      mlir::FailureOr<llvm::SmallVector<PhysicalTileDirectedLink, 8>> route =
          topology.getCanonicalOnCardPath(cardId, fragment.sourceTile,
                                          strategy.destinationTile);
      if (mlir::failed(route) || route->empty()) {
        if (failureReason)
          *failureReason = "candidate peer movement has no topology route";
        return false;
      }
      peerMovements.push_back(WholeDAGPeerMovement{
          *edgeID, fragment.sourceTile, strategy.destinationTile,
          fragment.bytes, strategy.bufferCount, std::move(*route)});
    }
  }

  llvm::SmallVector<WholeDAGLocalResidency, 32> localResidencies;
  localResidencies.reserve(residencyBytes.size());
  for (const auto &[key, bytes] : residencyBytes)
    if (bytes != 0)
      localResidencies.push_back(
          WholeDAGLocalResidency{key.first, PhysicalTileId(key.second), bytes});

  CandidateResourceScheduleMemo::Lookup memoLookup;
  if (memo) {
    memoLookup = memo->lookupOrCreate(buildCandidateResourceScheduleKey(
        candidate.assignment.nodePlacements, localResidencies, peerMovements,
        localMovements));
    if (!memoLookup.owner) {
      if (!memoLookup.entry->legal) {
        if (failureReason)
          *failureReason = memoLookup.entry->failureReason;
        return false;
      }
      applyCandidateResourceScheduleSummary(candidate,
                                            memoLookup.entry->summary);
      return true;
    }
  }

  std::string scheduleFailure;
  mlir::FailureOr<WholeDAGCandidateSchedule> schedule =
      scheduleWholeDAGCandidate(
          dag, availableTiles, candidate.assignment.nodePlacements,
          localResidencies, &scheduleFailure, peerMovements, localMovements,
          /*enforceSPMCapacity=*/false);
  if (mlir::failed(schedule)) {
    if (memo)
      memo->storeResult(memoLookup.entry, /*legal=*/false, {}, scheduleFailure);
    if (failureReason)
      *failureReason = std::move(scheduleFailure);
    return false;
  }
  CandidateResourceScheduleSummary summary{
      schedule->eventCount, schedule->makespan, schedule->peakLiveSPMBytes,
      schedule->spmMovementWork, schedule->ddrMovementWork};
  if (memo)
    memo->storeResult(memoLookup.entry, /*legal=*/true, summary, {});
  applyCandidateResourceScheduleSummary(candidate, summary);
  return true;
}

static std::optional<WholeCardCandidate>
makeTemporalVariant(const WholeCardCandidate &spatial,
                    const StaticOutputDomains &outputDomains,
                    const CardDAGAnalysis &dag, uint64_t stableOrdinal,
                    unsigned additionalWaveRefinements) {
  WholeCardCandidate candidate = spatial;
  candidate.transition.stableOrdinal = stableOrdinal;
  candidate.transition.temporalRefinementDepth = additionalWaveRefinements;
  for (CardOutputSpatialMapping &output :
       candidate.assignment.mapping.outputs) {
    std::optional<uint64_t> elementBytes = getElementByteWidth(
        dag.getFunction().getResultTypes()[output.outputIndex]);
    if (!elementBytes)
      return std::nullopt;
    llvm::SmallVector<int64_t, 4> shardShape =
        getMaximumSpatialShardShape(output, outputDomains[output.outputIndex]);
    output.temporalTileSizes = shardShape;
    for (unsigned refinement = 0; refinement < additionalWaveRefinements;
         ++refinement) {
      std::optional<unsigned> dimension = selectTemporalRefinementAxis(
          shardShape, output.temporalTileSizes,
          saturatingMultiply(*elementBytes, getOutputFusedTensorByteScale(
                                                dag, output.outputIndex)));
      if (!dimension)
        break;
      output.temporalTileSizes[*dimension] = getNextLowerWaveBreakpoint(
          shardShape[*dimension], output.temporalTileSizes[*dimension]);
    }
  }
  if (!populateOperationTemporalTiles(candidate, dag,
                                      additionalWaveRefinements))
    return std::nullopt;
  populateTemporalMetrics(candidate, outputDomains, dag);
  return candidate;
}

/// Advances one operation coordinate selected from an actual SPM packing
/// failure. Refining every output and every operation at once composes
/// prologue/steady/tail classes on a producer chain and can duplicate an
/// exponential number of IR classes before the next pack. The largest failed
/// demand carries source
/// structured-node relation when lowering preserved it. An internal assembled
/// producer is refined through an exact downstream operand relation; a terminal
/// producer is refined together with the exact observable traversal coordinate
/// that materializes its result.
///
/// An allocation without a recoverable structured-node relation cannot identify
/// a related temporal coordinate. That complete candidate is rejected and the
/// candidate queue continues with its other placement/action states; the
/// search must not retile an unrelated modeled operation from an estimate.
static bool refineOperationTemporalVariantOnce(
    WholeCardCandidate &candidate, const StaticOutputDomains &outputDomains,
    const CardDAGAnalysis &dag, mlir::Operation *preferredOperation,
    bool preferredIsOperandDemand, mlir::Operation **refinedOperation = nullptr,
    size_t *refinedNode = nullptr, unsigned *refinedDimension = nullptr,
    int64_t *previousExtent = nullptr, int64_t *refinedExtent = nullptr,
    bool explicitProducerCanStreamToDDR = false) {
  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;

  // Only CoupledFusion recursively maps a consumer wave into its
  // producer. Ordinary search actions materialize the strategy's exact
  // spatial producer window as one explicit retained, peer, layout or DDR
  // value, so refining the producer op cannot shrink that allocation. The
  // independent baseline is different: its producer wave is streamed into a
  // compiler-owned DDR destination and only the compact temporal wave is live
  // in SPM. Its deterministic controller therefore explicitly enables the
  // producer coordinate below. This is a materialization contract, not an
  // estimate-based exception to allocator legality.
  auto hasOversizedExplicitProducerWindow =
      [&](const SpatialEdgeStrategy &strategy) {
        if (strategy.producer != preferredOperation ||
            strategy.action == SpatialEdgeAction::CoupledFusion)
          return false;
        if (!preferredIsOperandDemand)
          return true;
        if (strategy.producerResult >= preferredOperation->getNumResults())
          return true;
        std::optional<uint64_t> elementBytes = getElementByteWidth(
            preferredOperation->getResult(strategy.producerResult).getType());
        if (!elementBytes || strategy.producerSizes.empty() ||
            llvm::any_of(strategy.producerSizes,
                         [](int64_t extent) { return extent <= 0; }))
          return true;
        const uint64_t capacity = memory.spmLimit - memory.spmBase;
        return saturatingMultiply(getElementProduct(strategy.producerSizes),
                                  *elementBytes) > capacity;
      };
  if (!explicitProducerCanStreamToDDR && preferredOperation &&
      llvm::any_of(candidate.assignment.mapping.edgeStrategies,
                   hasOversizedExplicitProducerWindow))
    return false;

  struct Refinement {
    size_t node = 0;
    unsigned dimension = 0;
    uint64_t modeledResidency = 0;
    bool hasModeledResidency = false;
    uint64_t demandedOperandReduction = 0;
  };
  std::optional<Refinement> downstream;
  std::optional<Refinement> preferred;
  for (auto [node, tile] :
       llvm::enumerate(candidate.assignment.mapping.operationTemporalTiles)) {
    if (node >= candidate.assignment.nodePlacements.size())
      continue;
    std::optional<llvm::SmallVector<int64_t, 4>> ranges =
        getMaximumSpatialIteratorRanges(
            tile.operation, candidate.assignment.nodePlacements[node]);
    if (!ranges || ranges->size() != tile.iteratorTileSizes.size())
      continue;
    std::optional<unsigned> dimension = selectOperationTemporalRefinementAxis(
        tile.operation, *ranges, tile.iteratorTileSizes, memory);
    if (!dimension)
      continue;
    std::optional<uint64_t> residency = estimateKnownIteratorResidencyBytes(
        tile.operation, tile.iteratorTileSizes, memory);
    Refinement option{node, *dimension, residency.value_or(0),
                      residency.has_value(), 0};
    if (tile.operation == preferredOperation)
      preferred = option;
  }

  // A large allocation related to a producer is commonly an assembled
  // full producer value. Shrinking that producer's own result traversal still
  // assembles the same full tensor before its consumer and therefore cannot
  // reduce the allocation. Use the current SSA edge relation to find a
  // consumer iterator coordinate that strictly shrinks the demanded operand
  // window. This is an exact relation check on the failed state's structured
  // IR, not a name or shape heuristic.
  if (preferredOperation && !preferredIsOperandDemand) {
    for (const CardDAGEdge &edge : dag.getEdges()) {
      const CardDAGNode *producer = dag.getNode(edge.producer);
      const CardDAGNode *consumer = dag.getNode(edge.consumer);
      if (!producer || !consumer || producer->operation != preferredOperation ||
          edge.consumer >=
              candidate.assignment.mapping.operationTemporalTiles.size())
        continue;
      StructuredOpTemporalTile &consumerTile =
          candidate.assignment.mapping.operationTemporalTiles[edge.consumer];
      auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(
          consumerTile.operation);
      if (!linalg || edge.consumerOperand >= linalg->getNumOperands())
        continue;
      std::optional<llvm::SmallVector<int64_t, 4>> ranges =
          getMaximumSpatialIteratorRanges(
              consumerTile.operation,
              candidate.assignment.nodePlacements[edge.consumer]);
      if (!ranges || ranges->size() != consumerTile.iteratorTileSizes.size())
        continue;
      mlir::AffineMap operandMap = linalg.getMatchingIndexingMap(
          &linalg->getOpOperand(edge.consumerOperand));
      auto getElements = [&](llvm::ArrayRef<int64_t> iteratorShape)
          -> std::optional<uint64_t> {
        if (!operandMap || operandMap.getNumDims() != iteratorShape.size())
          return std::nullopt;
        uint64_t elements = 1;
        for (mlir::AffineExpr expression : operandMap.getResults()) {
          std::optional<SignedRange> range =
              getAffineTileRange(expression, iteratorShape);
          if (!range || range->second < range->first)
            return std::nullopt;
          elements = saturatingMultiply(
              elements,
              static_cast<uint64_t>(range->second - range->first + 1));
        }
        return elements;
      };
      std::optional<uint64_t> currentElements =
          getElements(consumerTile.iteratorTileSizes);
      if (!currentElements)
        continue;
      for (auto [dimension, current] :
           llvm::enumerate(consumerTile.iteratorTileSizes)) {
        if (current <= 1 ||
            !isOrderPreservingReductionRefinement(
                consumerTile.operation, static_cast<unsigned>(dimension),
                *ranges, consumerTile.iteratorTileSizes))
          continue;
        llvm::SmallVector<int64_t, 4> nextShape(
            consumerTile.iteratorTileSizes.begin(),
            consumerTile.iteratorTileSizes.end());
        nextShape[dimension] = getNextLowerPackingFeedbackBreakpoint(
            (*ranges)[dimension], nextShape[dimension]);
        std::optional<uint64_t> nextElements = getElements(nextShape);
        if (!nextElements || *nextElements >= *currentElements)
          continue;
        Refinement option{edge.consumer, static_cast<unsigned>(dimension),
                          *currentElements, true,
                          *currentElements - *nextElements};
        if (!downstream ||
            std::tuple(option.demandedOperandReduction, option.modeledResidency,
                       std::numeric_limits<size_t>::max() - option.node,
                       std::numeric_limits<unsigned>::max() -
                           option.dimension) >
                std::tuple(downstream->demandedOperandReduction,
                           downstream->modeledResidency,
                           std::numeric_limits<size_t>::max() -
                               downstream->node,
                           std::numeric_limits<unsigned>::max() -
                               downstream->dimension))
          downstream = option;
      }
    }
  }
  const std::optional<Refinement> &selected =
      downstream ? downstream : preferred;
  if (!selected)
    return false;
  StructuredOpTemporalTile &tile =
      candidate.assignment.mapping.operationTemporalTiles[selected->node];
  std::optional<llvm::SmallVector<int64_t, 4>> ranges =
      getMaximumSpatialIteratorRanges(
          tile.operation, candidate.assignment.nodePlacements[selected->node]);
  if (!ranges || selected->dimension >= ranges->size())
    return false;
  const int64_t previous = tile.iteratorTileSizes[selected->dimension];
  tile.iteratorTileSizes[selected->dimension] =
      getNextLowerPackingFeedbackBreakpoint(
          (*ranges)[selected->dimension],
          tile.iteratorTileSizes[selected->dimension]);

  // A function result is materialized from CardOutputSpatialMapping's
  // traversal, not solely from the operation iterator vector. When the
  // refined operation result is returned directly, carry the selected
  // iterator coordinate through the Linalg result indexing map and advance
  // the matching observable coordinate as part of the same joint transition.
  // Reduction iterators have no result dimension and therefore correctly do
  // not change the observable traversal here.
  auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(tile.operation);
  mlir::func::FuncOp function = dag.getFunction();
  auto returnOp = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(
      function.getBody().empty() ? nullptr
                                 : function.getBody().front().getTerminator());
  if (linalg && returnOp) {
    for (mlir::OpResult result : tile.operation->getResults()) {
      if (result.getResultNumber() >= linalg.getNumDpsInits())
        continue;
      mlir::AffineMap resultMap = linalg.getMatchingIndexingMap(
          linalg.getDpsInitOperand(result.getResultNumber()));
      if (!resultMap)
        continue;
      for (auto [returnIndex, returned] :
           llvm::enumerate(returnOp.getOperands())) {
        if (returned != result)
          continue;
        auto output =
            llvm::find_if(candidate.assignment.mapping.outputs,
                          [&](const CardOutputSpatialMapping &mapping) {
                            return mapping.outputIndex == returnIndex;
                          });
        if (output == candidate.assignment.mapping.outputs.end())
          continue;
        llvm::SmallVector<int64_t, 4> shardShape =
            getMaximumSpatialShardShape(*output, outputDomains[returnIndex]);
        for (auto [outputDimension, expression] :
             llvm::enumerate(resultMap.getResults())) {
          auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
          if (!dimension || dimension.getPosition() != selected->dimension ||
              outputDimension >= output->temporalTileSizes.size() ||
              outputDimension >= shardShape.size())
            continue;
          int64_t &outputTile = output->temporalTileSizes[outputDimension];
          // The observable traversal is an independent joint coordinate and
          // can already be finer than the returned operation coordinate.
          // Projecting another operation refinement must never enlarge it or
          // ask for a breakpoint below the terminal extent-one state.
          outputTile = std::min(outputTile, shardShape[outputDimension]);
          if (outputTile > 1)
            outputTile = getNextLowerWaveBreakpoint(shardShape[outputDimension],
                                                    outputTile);
        }
      }
    }
  }
  if (refinedOperation)
    *refinedOperation = tile.operation;
  if (refinedNode)
    *refinedNode = selected->node;
  if (refinedDimension)
    *refinedDimension = selected->dimension;
  if (previousExtent)
    *previousExtent = previous;
  if (refinedExtent)
    *refinedExtent = tile.iteratorTileSizes[selected->dimension];
  if (candidate.transition.temporalRefinementDepth !=
      std::numeric_limits<unsigned>::max())
    ++candidate.transition.temporalRefinementDepth;
  populateTemporalMetrics(candidate, outputDomains, dag);
  return true;
}

/// Advances only the observable traversal named by an actual allocation
/// failure.  This covers view/update roots (for example a stateful output
/// assembled from DDR-resident tensors) that intentionally are not structured
/// DAG nodes. The allocation's current-IR output relation is the authority;
/// result shape is used only to enumerate the next finite coordinate value.
static bool refineOutputTemporalVariantOnce(
    WholeCardCandidate &candidate, const StaticOutputDomains &outputDomains,
    const CardDAGAnalysis &dag, unsigned outputIndex,
    unsigned *refinedDimension = nullptr, int64_t *previousExtent = nullptr,
    int64_t *refinedExtent = nullptr) {
  auto output = llvm::find_if(candidate.assignment.mapping.outputs,
                              [&](const CardOutputSpatialMapping &mapping) {
                                return mapping.outputIndex == outputIndex;
                              });
  if (output == candidate.assignment.mapping.outputs.end() ||
      outputIndex >= outputDomains.size())
    return false;
  std::optional<uint64_t> elementBytes =
      getElementByteWidth(dag.getFunction().getResultTypes()[outputIndex]);
  if (!elementBytes)
    return false;
  llvm::SmallVector<int64_t, 4> shardShape =
      getMaximumSpatialShardShape(*output, outputDomains[outputIndex]);
  std::optional<unsigned> dimension = selectTemporalRefinementAxis(
      shardShape, output->temporalTileSizes,
      saturatingMultiply(*elementBytes,
                         getOutputFusedTensorByteScale(dag, outputIndex)));
  if (!dimension)
    return false;
  const int64_t previous = output->temporalTileSizes[*dimension];
  output->temporalTileSizes[*dimension] = getNextLowerPackingFeedbackBreakpoint(
      shardShape[*dimension], output->temporalTileSizes[*dimension]);
  if (output->temporalTileSizes[*dimension] >= previous)
    return false;
  if (refinedDimension)
    *refinedDimension = *dimension;
  if (previousExtent)
    *previousExtent = previous;
  if (refinedExtent)
    *refinedExtent = output->temporalTileSizes[*dimension];
  if (candidate.transition.temporalRefinementDepth !=
      std::numeric_limits<unsigned>::max())
    ++candidate.transition.temporalRefinementDepth;
  populateTemporalMetrics(candidate, outputDomains, dag);
  return true;
}

/// Advances every refinable temporal coordinate by one finite wave class.
/// This recurrence preserves reduction-order legality through
/// selectOperationTemporalRefinementAxis and does not infer SPM capacity.
/// `none` uses one step after resource-calendar rejection.
[[maybe_unused]] static bool refineResourceScheduleTemporalVariantOnce(
    WholeCardCandidate &candidate, const StaticOutputDomains &outputDomains,
    const CardDAGAnalysis &dag) {
  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  bool changed = false;
  for (CardOutputSpatialMapping &output :
       candidate.assignment.mapping.outputs) {
    std::optional<uint64_t> elementBytes = getElementByteWidth(
        dag.getFunction().getResultTypes()[output.outputIndex]);
    if (!elementBytes)
      return false;
    llvm::SmallVector<int64_t, 4> shardShape =
        getMaximumSpatialShardShape(output, outputDomains[output.outputIndex]);
    std::optional<unsigned> dimension = selectTemporalRefinementAxis(
        shardShape, output.temporalTileSizes,
        saturatingMultiply(*elementBytes, getOutputFusedTensorByteScale(
                                              dag, output.outputIndex)));
    if (!dimension)
      continue;
    output.temporalTileSizes[*dimension] = getNextLowerWaveBreakpoint(
        shardShape[*dimension], output.temporalTileSizes[*dimension]);
    changed = true;
  }
  for (auto [node, tile] :
       llvm::enumerate(candidate.assignment.mapping.operationTemporalTiles)) {
    if (node >= candidate.assignment.nodePlacements.size())
      return false;
    std::optional<llvm::SmallVector<int64_t, 4>> ranges =
        getMaximumSpatialIteratorRanges(
            tile.operation, candidate.assignment.nodePlacements[node]);
    if (!ranges || ranges->size() != tile.iteratorTileSizes.size())
      return false;
    std::optional<unsigned> dimension = selectOperationTemporalRefinementAxis(
        tile.operation, *ranges, tile.iteratorTileSizes, memory);
    if (!dimension)
      continue;
    tile.iteratorTileSizes[*dimension] = getNextLowerWaveBreakpoint(
        (*ranges)[*dimension], tile.iteratorTileSizes[*dimension]);
    changed = true;
  }
  if (!changed)
    return false;
  populateTemporalMetrics(candidate, outputDomains, dag);
  return true;
}

static unsigned
refineOversizedExplicitProducerEdges(WholeCardCandidate &candidate,
                                     mlir::Operation *producer);

/// Returns true when an allocation related only to a generic operation
/// cannot be that operation's selected result tile.  In that case its exact
/// byte size is evidence for an operand or internal working-set demand even
/// if the more specific operand marker was lost while composing movement
/// locations.
static bool
isLargerThanKnownOperationResultTile(const WholeCardCandidate &candidate,
                                     mlir::Operation *operation,
                                     uint64_t demandBytes) {
  if (!operation || demandBytes == 0)
    return false;
  auto selected =
      llvm::find_if(candidate.assignment.mapping.operationTemporalTiles,
                    [&](const StructuredOpTemporalTile &tile) {
                      return tile.operation == operation;
                    });
  if (selected == candidate.assignment.mapping.operationTemporalTiles.end())
    return false;
  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  uint64_t largestKnownResultBytes = 0;
  bool hasKnownResult = false;
  for (unsigned result = 0; result < operation->getNumResults(); ++result) {
    std::optional<uint64_t> resultBytes = estimateKnownResultTileBytes(
        operation, result, selected->iteratorTileSizes, memory);
    if (!resultBytes)
      continue;
    hasKnownResult = true;
    largestKnownResultBytes = std::max(largestKnownResultBytes, *resultBytes);
  }
  return hasKnownResult && demandBytes > largestKnownResultBytes;
}

/// Builds one conflict-directed exact probe from the allocator's measured
/// allocation size. The byte projection only chooses how many adjacent
/// finite coordinate transitions to compose before the next materialization;
/// it never accepts capacity. The resulting candidate still crosses the real
/// fixed-capacity allocator and every downstream executable verification.
static bool refineAllocationDemandTowardCapacityProbe(
    WholeCardCandidate &candidate, const StaticOutputDomains &outputDomains,
    const CardDAGAnalysis &dag, const SPMCapacityDemandEvidence &demand,
    uint64_t *temporalTransitions = nullptr,
    uint64_t *edgeTransitions = nullptr) {
  mlir::Operation *demandOperation =
      demand.operandDemandNode
          ? dag.getNodes()[*demand.operandDemandNode].operation
      : demand.operationNode ? dag.getNodes()[*demand.operationNode].operation
                             : nullptr;
  const bool repeatedUnchangedDemand =
      candidate.transition.previousSPMDemand &&
      candidate.transition.previousSPMDemand->matches(demand);
  const bool inferredOperandDemand =
      !demand.operandDemandNode && !demand.outputIndex &&
      isLargerThanKnownOperationResultTile(candidate, demandOperation,
                                           demand.bytes);
  const unsigned changedEdges =
      refineOversizedExplicitProducerEdges(candidate, demandOperation);
  if (changedEdges != 0) {
    if (edgeTransitions)
      *edgeTransitions = saturatingAdd(*edgeTransitions, changedEdges);
    return true;
  }

  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  const uint64_t capacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);
  uint64_t projectedBytes = std::max<uint64_t>(1, demand.bytes);
  bool changed = false;
  bool requireAdjacentTransition = true;
  while (requireAdjacentTransition || projectedBytes > capacity) {
    requireAdjacentTransition = false;
    unsigned refinedDimension = 0;
    int64_t previousExtent = 0;
    int64_t refinedExtent = 0;
    bool refined = false;
    if (demand.operandDemandNode) {
      refined = refineOperationTemporalVariantOnce(
          candidate, outputDomains, dag, demandOperation,
          /*preferredIsOperandDemand=*/true, nullptr, nullptr,
          &refinedDimension, &previousExtent, &refinedExtent);
    } else if (demand.outputIndex) {
      refined = refineOutputTemporalVariantOnce(
          candidate, outputDomains, dag, *demand.outputIndex, &refinedDimension,
          &previousExtent, &refinedExtent);
    } else if (demand.operationNode) {
      refined = refineOperationTemporalVariantOnce(
          candidate, outputDomains, dag, demandOperation,
          // A first producer-result failure may be an assembled producer whose
          // demanded window is controlled by a downstream consumer. If that
          // exact allocation survives the downstream transition unchanged,
          // the measured result disproves that causal direction for this
          // state: advance the allocation owner's own coordinate next.
          /*preferredIsOperandDemand=*/repeatedUnchangedDemand ||
              inferredOperandDemand,
          nullptr, nullptr, &refinedDimension, &previousExtent, &refinedExtent);
    }
    if (!refined)
      break;
    changed = true;
    if (temporalTransitions)
      *temporalTransitions = saturatingAdd(*temporalTransitions, 1);
    if (previousExtent <= 0 || refinedExtent <= 0 ||
        refinedExtent >= previousExtent)
      break;
    projectedBytes =
        ceilDivide(saturatingMultiply(projectedBytes,
                                      static_cast<uint64_t>(refinedExtent)),
                   static_cast<uint64_t>(previousExtent));
  }
  return changed;
}

static bool sameMapping(const WholeCardCandidate &lhs,
                        const WholeCardCandidate &rhs) {
  if (lhs.assignment.mapping.materializationMode !=
          rhs.assignment.mapping.materializationMode ||
      lhs.assignment.mapping.outputs.size() !=
          rhs.assignment.mapping.outputs.size() ||
      lhs.assignment.mapping.operationTemporalTiles.size() !=
          rhs.assignment.mapping.operationTemporalTiles.size() ||
      lhs.assignment.mapping.edgeStrategies.size() !=
          rhs.assignment.mapping.edgeStrategies.size())
    return false;
  for (auto [lhsOutput, rhsOutput] : llvm::zip_equal(
           lhs.assignment.mapping.outputs, rhs.assignment.mapping.outputs))
    if (lhsOutput.outputIndex != rhsOutput.outputIndex ||
        lhsOutput.shardDimension != rhsOutput.shardDimension ||
        lhsOutput.activeTileIds != rhsOutput.activeTileIds ||
        lhsOutput.temporalTileSizes != rhsOutput.temporalTileSizes)
      return false;
  for (auto [lhsTile, rhsTile] :
       llvm::zip_equal(lhs.assignment.mapping.operationTemporalTiles,
                       rhs.assignment.mapping.operationTemporalTiles))
    if (lhsTile.operation != rhsTile.operation ||
        lhsTile.iteratorTileSizes != rhsTile.iteratorTileSizes)
      return false;
  for (auto [lhsStrategy, rhsStrategy] :
       llvm::zip_equal(lhs.assignment.mapping.edgeStrategies,
                       rhs.assignment.mapping.edgeStrategies)) {
    if (lhsStrategy.producer != rhsStrategy.producer ||
        lhsStrategy.producerResult != rhsStrategy.producerResult ||
        lhsStrategy.consumer != rhsStrategy.consumer ||
        lhsStrategy.consumerOperand != rhsStrategy.consumerOperand ||
        lhsStrategy.consumerOffsets != rhsStrategy.consumerOffsets ||
        lhsStrategy.consumerSizes != rhsStrategy.consumerSizes ||
        lhsStrategy.producerOffsets != rhsStrategy.producerOffsets ||
        lhsStrategy.producerSizes != rhsStrategy.producerSizes ||
        lhsStrategy.sourceTile != rhsStrategy.sourceTile ||
        lhsStrategy.destinationTile != rhsStrategy.destinationTile ||
        lhsStrategy.action != rhsStrategy.action ||
        lhsStrategy.hasLayoutAssignment != rhsStrategy.hasLayoutAssignment ||
        lhsStrategy.producerLayout != rhsStrategy.producerLayout ||
        lhsStrategy.consumerLayout != rhsStrategy.consumerLayout ||
        lhsStrategy.bufferCount != rhsStrategy.bufferCount ||
        lhsStrategy.fragments.size() != rhsStrategy.fragments.size())
      return false;
    for (auto [lhsFragment, rhsFragment] :
         llvm::zip_equal(lhsStrategy.fragments, rhsStrategy.fragments))
      if (lhsFragment.kind != rhsFragment.kind ||
          lhsFragment.offsets != rhsFragment.offsets ||
          lhsFragment.sizes != rhsFragment.sizes ||
          lhsFragment.sourceTile != rhsFragment.sourceTile ||
          lhsFragment.bytes != rhsFragment.bytes ||
          lhsFragment.communicationId != rhsFragment.communicationId ||
          lhsFragment.payloadSlice != rhsFragment.payloadSlice)
        return false;
  }
  return true;
}

/// Coarse content hash for exact candidate-equivalence lookup.  Every field
/// mixed here is also compared by `sameMapping`; omitted fields only enlarge a
/// collision bucket and therefore cannot merge distinct semantic states.
static size_t getMappingHash(const WholeCardCandidate &candidate,
                             bool includeBufferCount = true) {
  llvm::hash_code hash = llvm::hash_combine(
      static_cast<unsigned>(candidate.assignment.mapping.materializationMode),
      candidate.assignment.mapping.outputs.size(),
      candidate.assignment.mapping.operationTemporalTiles.size(),
      candidate.assignment.mapping.edgeStrategies.size());
  auto mix = [&](auto value) { hash = llvm::hash_combine(hash, value); };
  for (const CardOutputSpatialMapping &output :
       candidate.assignment.mapping.outputs) {
    mix(output.outputIndex);
    mix(output.shardDimension);
    for (PhysicalTileId tile : output.activeTileIds)
      mix(tile.getValue());
    for (int64_t extent : output.temporalTileSizes)
      mix(extent);
  }
  for (const StructuredOpTemporalTile &tile :
       candidate.assignment.mapping.operationTemporalTiles) {
    mix(tile.operation);
    for (int64_t extent : tile.iteratorTileSizes)
      mix(extent);
  }
  for (const SpatialEdgeStrategy &strategy :
       candidate.assignment.mapping.edgeStrategies) {
    mix(strategy.producer);
    mix(strategy.producerResult);
    mix(strategy.consumer);
    mix(strategy.consumerOperand);
    mix(strategy.sourceTile.getValue());
    mix(strategy.destinationTile.getValue());
    mix(static_cast<unsigned>(strategy.action));
    mix(strategy.hasLayoutAssignment);
    mix(static_cast<unsigned>(strategy.producerLayout));
    mix(static_cast<unsigned>(strategy.consumerLayout));
    if (includeBufferCount)
      mix(strategy.bufferCount);
    mix(strategy.fragments.size());
  }
  return static_cast<size_t>(hash);
}

static bool sameNodePlacements(const WholeCardCandidate &lhs,
                               const WholeCardCandidate &rhs) {
  return lhs.assignment.nodePlacements.size() ==
             rhs.assignment.nodePlacements.size() &&
         llvm::all_of(llvm::zip_equal(lhs.assignment.nodePlacements,
                                      rhs.assignment.nodePlacements),
                      [](auto values) {
                        const auto &[left, right] = values;
                        return left.node == right.node &&
                               left.shardDimension == right.shardDimension &&
                               left.tiles == right.tiles &&
                               left.spatialIteratorDimension ==
                                   right.spatialIteratorDimension &&
                               left.iteratorPartitionFactors ==
                                   right.iteratorPartitionFactors;
                      });
}

static bool samePreBufferMapping(const WholeCardCandidate &lhs,
                                 const WholeCardCandidate &rhs) {
  if (!sameNodePlacements(lhs, rhs))
    return false;
  WholeCardCandidate normalizedLhs = lhs;
  WholeCardCandidate normalizedRhs = rhs;
  for (SpatialEdgeStrategy &strategy :
       normalizedLhs.assignment.mapping.edgeStrategies)
    strategy.bufferCount = 1;
  for (SpatialEdgeStrategy &strategy :
       normalizedRhs.assignment.mapping.edgeStrategies)
    strategy.bufferCount = 1;
  return sameMapping(normalizedLhs, normalizedRhs);
}

/// Buffer multiplicity changes slot count and the minimum steady-state trip
/// count, but it cannot create a static loop that contains the selected
/// producer/consumer/message endpoints.  Preserve which logical edges are
/// buffered while canonicalizing every selected multiplicity to two so an
/// exact NoExactLoop failure can be shared by the 2/3-buffer siblings only.
static WholeCardCandidate
normalizeBufferStructure(const WholeCardCandidate &candidate) {
  WholeCardCandidate normalized = candidate;
  for (SpatialEdgeStrategy &strategy :
       normalized.assignment.mapping.edgeStrategies)
    if (strategy.bufferCount > 1)
      strategy.bufferCount = 2;
  return normalized;
}

static size_t getBufferStructureHash(const WholeCardCandidate &candidate) {
  return getMappingHash(normalizeBufferStructure(candidate));
}

static bool sameBufferStructure(const WholeCardCandidate &lhs,
                                const WholeCardCandidate &rhs) {
  if (!sameNodePlacements(lhs, rhs))
    return false;
  return sameMapping(normalizeBufferStructure(lhs),
                     normalizeBufferStructure(rhs));
}

static bool sameNonTemporalState(const WholeCardCandidate &lhs,
                                 const WholeCardCandidate &rhs) {
  if (!sameNodePlacements(lhs, rhs))
    return false;
  WholeCardCandidate normalizedLhs = lhs;
  WholeCardCandidate normalizedRhs = rhs;
  for (CardOutputSpatialMapping &output :
       normalizedLhs.assignment.mapping.outputs)
    output.temporalTileSizes.clear();
  for (CardOutputSpatialMapping &output :
       normalizedRhs.assignment.mapping.outputs)
    output.temporalTileSizes.clear();
  for (StructuredOpTemporalTile &tile :
       normalizedLhs.assignment.mapping.operationTemporalTiles)
    tile.iteratorTileSizes.clear();
  for (StructuredOpTemporalTile &tile :
       normalizedRhs.assignment.mapping.operationTemporalTiles)
    tile.iteratorTileSizes.clear();
  return sameMapping(normalizedLhs, normalizedRhs);
}

static bool isStrictTemporalRefinementOf(const WholeCardCandidate &candidate,
                                         const WholeCardCandidate &coarser) {
  if (candidate.assignment.mapping.outputs.size() !=
          coarser.assignment.mapping.outputs.size() ||
      candidate.assignment.mapping.operationTemporalTiles.size() !=
          coarser.assignment.mapping.operationTemporalTiles.size())
    return false;
  bool strict = false;
  for (auto [fine, coarse] :
       llvm::zip_equal(candidate.assignment.mapping.outputs,
                       coarser.assignment.mapping.outputs)) {
    if (fine.temporalTileSizes.size() != coarse.temporalTileSizes.size())
      return false;
    for (auto [fineExtent, coarseExtent] :
         llvm::zip_equal(fine.temporalTileSizes, coarse.temporalTileSizes)) {
      if (fineExtent > coarseExtent)
        return false;
      strict |= fineExtent < coarseExtent;
    }
  }
  for (auto [fine, coarse] :
       llvm::zip_equal(candidate.assignment.mapping.operationTemporalTiles,
                       coarser.assignment.mapping.operationTemporalTiles)) {
    if (fine.operation != coarse.operation ||
        fine.iteratorTileSizes.size() != coarse.iteratorTileSizes.size())
      return false;
    for (auto [fineExtent, coarseExtent] :
         llvm::zip_equal(fine.iteratorTileSizes, coarse.iteratorTileSizes)) {
      if (fineExtent > coarseExtent)
        return false;
      strict |= fineExtent < coarseExtent;
    }
  }
  return strict;
}

static bool isStrictlyDominatedByAdmittedTemporalState(
    const WholeCardCandidate &candidate,
    const WholeCardCandidate &admittedCoarser) {
  return sameNonTemporalState(candidate, admittedCoarser) &&
         isStrictTemporalRefinementOf(candidate, admittedCoarser) &&
         admittedCoarser.evaluation.temporalWaveLowerBound <=
             candidate.evaluation.temporalWaveLowerBound &&
         admittedCoarser.evaluation.instructionExecutionLowerBound <=
             candidate.evaluation.instructionExecutionLowerBound &&
         admittedCoarser.evaluation.scheduledMakespan <=
             candidate.evaluation.scheduledMakespan &&
         admittedCoarser.evaluation.scheduledSPMMovementWork <=
             candidate.evaluation.scheduledSPMMovementWork &&
         admittedCoarser.evaluation.scheduledDDRMovementWork <=
             candidate.evaluation.scheduledDDRMovementWork &&
         admittedCoarser.evaluation.peerBytes <= candidate.evaluation.peerBytes;
}

static bool hasMultiReductionAxisSplit(const WholeCardCandidate &candidate) {
  for (auto [node, selected] :
       llvm::enumerate(candidate.assignment.mapping.operationTemporalTiles)) {
    if (node >= candidate.assignment.nodePlacements.size())
      continue;
    std::optional<llvm::SmallVector<int64_t, 4>> ranges =
        getMaximumSpatialIteratorRanges(
            selected.operation, candidate.assignment.nodePlacements[node]);
    auto tiling =
        mlir::dyn_cast_or_null<mlir::TilingInterface>(selected.operation);
    if (!ranges || !tiling ||
        ranges->size() != selected.iteratorTileSizes.size())
      continue;
    unsigned splitReductionAxes = 0;
    for (auto [dimension, iteratorType] :
         llvm::enumerate(tiling.getLoopIteratorTypes()))
      if (iteratorType == mlir::utils::IteratorType::reduction &&
          selected.iteratorTileSizes[dimension] < (*ranges)[dimension])
        ++splitReductionAxes;
    if (splitReductionAxes > 1)
      return true;
  }
  return false;
}

static bool hasAlternativeEdgeAction(const WholeCardCandidate &candidate) {
  return llvm::any_of(
      candidate.assignment.mapping.edgeStrategies,
      [](const SpatialEdgeStrategy &strategy) {
        return strategy.action != SpatialEdgeAction::CoupledFusion &&
               strategy.action != SpatialEdgeAction::PeerFragments;
      });
}

static bool hasLayoutAssignment(const WholeCardCandidate &candidate) {
  return llvm::any_of(candidate.assignment.mapping.edgeStrategies,
                      [](const SpatialEdgeStrategy &strategy) {
                        return strategy.hasLayoutAssignment;
                      });
}

static bool hasLayoutConversion(const WholeCardCandidate &candidate) {
  return llvm::any_of(candidate.assignment.mapping.edgeStrategies,
                      [](const SpatialEdgeStrategy &strategy) {
                        return strategy.action ==
                               SpatialEdgeAction::LocalPhysicalConversion;
                      });
}

static uint8_t getMaximumBufferCount(const WholeCardCandidate &candidate) {
  uint8_t count = 1;
  for (const SpatialEdgeStrategy &strategy :
       candidate.assignment.mapping.edgeStrategies)
    count = std::max(count, strategy.bufferCount);
  return count;
}

static bool hasBufferedEdge(const WholeCardCandidate &candidate) {
  return getMaximumBufferCount(candidate) > 1;
}

static bool hasLayoutBufferedEdge(const WholeCardCandidate &candidate) {
  return hasLayoutConversion(candidate) && hasBufferedEdge(candidate);
}

static bool outputDependsOnOperation(const CardDAGAnalysis &dag,
                                     unsigned outputIndex,
                                     mlir::Operation *operation) {
  if (!operation || outputIndex >= dag.getObservableOutputRootNodes().size())
    return false;
  std::optional<CardDAGNodeID> target;
  for (const CardDAGNode &node : dag.getNodes())
    if (node.operation == operation) {
      target = node.id;
      break;
    }
  if (!target)
    return false;

  llvm::BitVector visited(dag.getNodes().size());
  llvm::SmallVector<CardDAGNodeID, 16> worklist(
      dag.getObservableOutputRootNodes()[outputIndex].begin(),
      dag.getObservableOutputRootNodes()[outputIndex].end());
  while (!worklist.empty()) {
    const CardDAGNodeID current = worklist.pop_back_val();
    if (current >= visited.size() || visited.test(current))
      continue;
    if (current == *target)
      return true;
    visited.set(current);
    const CardDAGNode *node = dag.getNode(current);
    if (!node)
      return false;
    for (CardDAGEdgeID edgeId : node->incomingEdges) {
      const CardDAGEdge *edge = dag.getEdge(edgeId);
      if (!edge)
        return false;
      worklist.push_back(edge->producer);
    }
  }
  return false;
}

static bool outputContainsBufferedConsumer(
    const WholeCardCandidate &candidate, const CardDAGAnalysis &dag,
    unsigned outputIndex,
    std::optional<CardDAGNodeID> consumerNode = std::nullopt) {
  if (consumerNode && *consumerNode < dag.getNodes().size())
    return outputDependsOnOperation(dag, outputIndex,
                                    dag.getNodes()[*consumerNode].operation);
  return llvm::any_of(candidate.assignment.mapping.edgeStrategies,
                      [&](const SpatialEdgeStrategy &strategy) {
                        return strategy.bufferCount > 1 &&
                               outputDependsOnOperation(dag, outputIndex,
                                                        strategy.consumer);
                      });
}

static std::vector<WholeCardCandidate> refineSelectedBufferConsumerWaveOnce(
    const WholeCardCandidate &candidate, const CardDAGAnalysis &dag,
    const StaticOutputDomains &outputDomains,
    std::optional<CardDAGNodeID> consumerNode, uint64_t &nextStableOrdinal) {
  std::vector<WholeCardCandidate> result;
  if (!consumerNode || *consumerNode >= dag.getNodes().size())
    return result;
  mlir::Operation *consumer = dag.getNodes()[*consumerNode].operation;
  for (size_t outputIndex = 0;
       outputIndex < candidate.assignment.mapping.outputs.size();
       ++outputIndex) {
    const unsigned resultIndex =
        candidate.assignment.mapping.outputs[outputIndex].outputIndex;
    if (!outputDependsOnOperation(dag, resultIndex, consumer))
      continue;
    llvm::SmallVector<int64_t, 4> ranges = getMaximumSpatialShardShape(
        candidate.assignment.mapping.outputs[outputIndex],
        outputDomains[resultIndex]);
    const llvm::SmallVector<int64_t, 4> &current =
        candidate.assignment.mapping.outputs[outputIndex].temporalTileSizes;
    if (ranges.size() != current.size())
      continue;
    for (size_t dimension = 0; dimension < ranges.size(); ++dimension) {
      if (current[dimension] <= 1 || current[dimension] >= ranges[dimension])
        continue;
      int64_t next = current[dimension];
      do {
        next = getNextLowerWaveBreakpoint(ranges[dimension], next);
      } while (next > 1 && ranges[dimension] % next != 0);
      if (next >= current[dimension])
        continue;
      WholeCardCandidate child = candidate;
      child.transition.stableOrdinal = nextStableOrdinal++;
      child.assignment.mapping.outputs[outputIndex]
          .temporalTileSizes[dimension] = next;
      child.evaluation.queryExpansionWork =
          saturatingAdd(child.evaluation.queryExpansionWork, 1);
      populateTemporalMetrics(child, outputDomains, dag);
      result.push_back(std::move(child));
    }
  }
  return result;
}

static bool sameLogicalEdge(const SpatialEdgeStrategy &lhs,
                            const SpatialEdgeStrategy &rhs) {
  return lhs.producer == rhs.producer &&
         lhs.producerResult == rhs.producerResult &&
         lhs.consumer == rhs.consumer &&
         lhs.consumerOperand == rhs.consumerOperand;
}

static llvm::SmallVector<llvm::SmallVector<size_t, 16>, 32>
getLogicalEdgeStrategyGroups(const WholeCardCandidate &candidate) {
  llvm::SmallVector<llvm::SmallVector<size_t, 16>, 32> groups;
  for (auto [index, strategy] :
       llvm::enumerate(candidate.assignment.mapping.edgeStrategies)) {
    auto group = llvm::find_if(groups, [&](llvm::ArrayRef<size_t> members) {
      return !members.empty() &&
             sameLogicalEdge(
                 candidate.assignment.mapping.edgeStrategies[members.front()],
                 strategy);
    });
    if (group == groups.end())
      groups.push_back({index});
    else
      group->push_back(index);
  }
  return groups;
}

static uint64_t countBufferedLogicalEdges(const WholeCardCandidate &candidate) {
  return llvm::count_if(getLogicalEdgeStrategyGroups(candidate),
                        [&](llvm::ArrayRef<size_t> group) {
                          return llvm::any_of(group, [&](size_t index) {
                            return candidate.assignment.mapping
                                       .edgeStrategies[index]
                                       .bufferCount > 1;
                          });
                        });
}

static bool isLegalLocalEdgeAction(const WholeCardCandidate &candidate,
                                   llvm::ArrayRef<size_t> group,
                                   SpatialEdgeAction action) {
  if (group.empty())
    return false;
  for (size_t index : group) {
    const SpatialEdgeStrategy &strategy =
        candidate.assignment.mapping.edgeStrategies[index];
    if (!strategy.fragments.empty() ||
        strategy.sourceTile != strategy.destinationTile)
      return false;
    if (action == SpatialEdgeAction::Recompute &&
        (!strategy.producer || !mlir::isMemoryEffectFree(strategy.producer)))
      return false;
    if (action == SpatialEdgeAction::LocalPhysicalConversion &&
        (!strategy.hasLayoutAssignment ||
         strategy.producerLayout == strategy.consumerLayout))
      return false;
  }
  return action != SpatialEdgeAction::PeerFragments;
}

static uint64_t getLogicalEdgeBytes(const WholeCardCandidate &candidate,
                                    llvm::ArrayRef<size_t> group) {
  uint64_t result = 0;
  for (size_t index : group) {
    const SpatialEdgeStrategy &strategy =
        candidate.assignment.mapping.edgeStrategies[index];
    if (!strategy.producer || strategy.producerSizes.empty())
      continue;
    std::optional<uint64_t> elementBytes = getElementByteWidth(
        strategy.producer->getResult(strategy.producerResult).getType());
    if (elementBytes)
      result = saturatingAdd(
          result, saturatingMultiply(getElementProduct(strategy.producerSizes),
                                     *elementBytes));
  }
  return result;
}

static std::optional<WholeCardCandidate>
makeSingleEdgeActionVariant(const WholeCardCandidate &base,
                            SpatialEdgeAction action, uint64_t stableOrdinal) {
  std::optional<llvm::SmallVector<size_t, 16>> selected;
  uint64_t selectedBytes = 0;
  for (const llvm::SmallVector<size_t, 16> &group :
       getLogicalEdgeStrategyGroups(base)) {
    if (!llvm::all_of(
            group,
            [&](size_t index) {
              return base.assignment.mapping.edgeStrategies[index].action ==
                     SpatialEdgeAction::CoupledFusion;
            }) ||
        !isLegalLocalEdgeAction(base, group, action))
      continue;
    const uint64_t bytes = getLogicalEdgeBytes(base, group);
    if (!selected || bytes > selectedBytes) {
      selected = group;
      selectedBytes = bytes;
    }
  }
  if (!selected)
    return std::nullopt;
  WholeCardCandidate candidate = base;
  candidate.transition.stableOrdinal = stableOrdinal;
  for (size_t index : *selected)
    candidate.assignment.mapping.edgeStrategies[index].action = action;
  candidate.evaluation.queryExpansionWork =
      saturatingAdd(candidate.evaluation.queryExpansionWork, 1);
  return candidate;
}

static std::optional<WholeCardCandidate>
makeSingleEdgeBufferVariant(const WholeCardCandidate &base, uint8_t bufferCount,
                            uint64_t stableOrdinal) {
  if (bufferCount < 2 || bufferCount > 3)
    return std::nullopt;
  std::optional<llvm::SmallVector<size_t, 16>> selected;
  uint64_t selectedBytes = 0;
  for (const llvm::SmallVector<size_t, 16> &group :
       getLogicalEdgeStrategyGroups(base)) {
    if (group.empty() || llvm::any_of(group, [&](size_t index) {
          const SpatialEdgeStrategy &strategy =
              base.assignment.mapping.edgeStrategies[index];
          return !strategy.producer || strategy.producerSizes.empty() ||
                 strategy.action == SpatialEdgeAction::RegionCut;
        }))
      continue;
    const uint64_t bytes = getLogicalEdgeBytes(base, group);
    if (!selected || bytes > selectedBytes) {
      selected = group;
      selectedBytes = bytes;
    }
  }
  if (!selected)
    return std::nullopt;
  WholeCardCandidate candidate = base;
  candidate.transition.stableOrdinal = stableOrdinal;
  for (size_t index : *selected)
    candidate.assignment.mapping.edgeStrategies[index].bufferCount =
        bufferCount;
  candidate.evaluation.queryExpansionWork =
      saturatingAdd(candidate.evaluation.queryExpansionWork, 1);
  return candidate;
}

static std::optional<WholeCardCandidate>
makeLogicalEdgeVariant(const WholeCardCandidate &base,
                       llvm::ArrayRef<size_t> group, SpatialEdgeAction action,
                       uint8_t bufferCount, uint64_t stableOrdinal) {
  if (group.empty() || bufferCount < 1 || bufferCount > 3)
    return std::nullopt;
  const bool peer = llvm::any_of(group, [&](size_t index) {
    const SpatialEdgeStrategy &strategy =
        base.assignment.mapping.edgeStrategies[index];
    return !strategy.fragments.empty() ||
           strategy.sourceTile != strategy.destinationTile;
  });
  if ((peer && action != SpatialEdgeAction::PeerFragments) ||
      (!peer && !isLegalLocalEdgeAction(base, group, action)) ||
      (action == SpatialEdgeAction::RegionCut && bufferCount != 1))
    return std::nullopt;

  WholeCardCandidate candidate = base;
  candidate.transition.stableOrdinal = stableOrdinal;
  for (size_t index : group) {
    SpatialEdgeStrategy &strategy =
        candidate.assignment.mapping.edgeStrategies[index];
    strategy.action = action;
    strategy.bufferCount = bufferCount;
  }
  candidate.evaluation.queryExpansionWork =
      saturatingAdd(candidate.evaluation.queryExpansionWork, 1);
  return candidate;
}

/// Advances the exact edge-action coordinate that keeps an allocator-proven
/// oversized producer window explicit in one local Tile.  Under the current
/// lowering contract, retained/recomputed/layout/spill boundaries all
/// materialize that selected producer window before their boundary action;
/// only CoupledFusion maps the consumer wave recursively into the
/// producer.  This is therefore a structural neighbor justified by the
/// failed actual state, not a byte-estimate acceptance decision.
static unsigned
refineOversizedExplicitProducerEdges(WholeCardCandidate &candidate,
                                     mlir::Operation *producer) {
  if (!producer)
    return 0;
  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  const uint64_t capacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);
  unsigned refinedGroups = 0;
  for (llvm::ArrayRef<size_t> group : getLogicalEdgeStrategyGroups(candidate)) {
    const bool hasProvenOversizedWindow =
        llvm::any_of(group, [&](size_t index) {
          const SpatialEdgeStrategy &strategy =
              candidate.assignment.mapping.edgeStrategies[index];
          if (strategy.producer != producer ||
              strategy.action == SpatialEdgeAction::CoupledFusion ||
              strategy.producerResult >= producer->getNumResults() ||
              strategy.producerSizes.empty())
            return false;
          std::optional<uint64_t> elementBytes = getElementByteWidth(
              producer->getResult(strategy.producerResult).getType());
          return elementBytes &&
                 saturatingMultiply(getElementProduct(strategy.producerSizes),
                                    *elementBytes) > capacity;
        });
    if (!hasProvenOversizedWindow ||
        !isLegalLocalEdgeAction(candidate, group,
                                SpatialEdgeAction::CoupledFusion))
      continue;
    for (size_t index : group) {
      SpatialEdgeStrategy &strategy =
          candidate.assignment.mapping.edgeStrategies[index];
      strategy.action = SpatialEdgeAction::CoupledFusion;
      strategy.bufferCount = 1;
    }
    ++refinedGroups;
  }
  candidate.evaluation.queryExpansionWork =
      saturatingAdd(candidate.evaluation.queryExpansionWork, refinedGroups);
  return refinedGroups;
}

static uint64_t countFusedLogicalEdges(const WholeCardCandidate &candidate) {
  uint64_t result = 0;
  for (const llvm::SmallVector<size_t, 16> &group :
       getLogicalEdgeStrategyGroups(candidate))
    if (!group.empty() && llvm::all_of(group, [&](size_t index) {
          return candidate.assignment.mapping.edgeStrategies[index].action ==
                 SpatialEdgeAction::CoupledFusion;
        }))
      ++result;
  return result;
}

static bool cheapCandidateLess(const WholeCardCandidate &lhs,
                               const WholeCardCandidate &rhs) {
  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  const uint64_t capacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);
  const uint64_t lhsEstimatedResidency =
      saturatingAdd(lhs.evaluation.peakAlignedResidencyEstimate,
                    lhs.evaluation.scheduledPeakLiveSPMBytes);
  const uint64_t rhsEstimatedResidency =
      saturatingAdd(rhs.evaluation.peakAlignedResidencyEstimate,
                    rhs.evaluation.scheduledPeakLiveSPMBytes);
  const bool lhsEstimatedFits = lhsEstimatedResidency <= capacity;
  const bool rhsEstimatedFits = rhsEstimatedResidency <= capacity;
  return std::tuple(
             !lhsEstimatedFits, lhs.evaluation.criticalStructuredElementWork,
             lhs.evaluation.instructionExecutionLowerBound,
             lhs.evaluation.temporalWaveLowerBound,
             lhs.evaluation.shardImbalance, lhsEstimatedResidency,
             lhs.evaluation.topologyHopByteWork, lhs.evaluation.peerBytes,
             std::numeric_limits<size_t>::max() -
                 getUniqueActiveTileCount(lhs.assignment.mapping,
                                          lhs.assignment.nodePlacements),
             lhs.evaluation.queryExpansionWork,
             std::numeric_limits<uint64_t>::max() -
                 lhs.evaluation.parallelComponentCount,
             lhs.transition.stableOrdinal) <
         std::tuple(
             !rhsEstimatedFits, rhs.evaluation.criticalStructuredElementWork,
             rhs.evaluation.instructionExecutionLowerBound,
             rhs.evaluation.temporalWaveLowerBound,
             rhs.evaluation.shardImbalance, rhsEstimatedResidency,
             rhs.evaluation.topologyHopByteWork, rhs.evaluation.peerBytes,
             std::numeric_limits<size_t>::max() -
                 getUniqueActiveTileCount(rhs.assignment.mapping,
                                          rhs.assignment.nodePlacements),
             rhs.evaluation.queryExpansionWork,
             std::numeric_limits<uint64_t>::max() -
                 rhs.evaluation.parallelComponentCount,
             rhs.transition.stableOrdinal);
}

static std::optional<WholeCardCandidate>
deriveBaseline(const PhysicalTopology &topology, PhysicalCardId cardId,
               llvm::ArrayRef<PhysicalTileId> availableTileIds,
               const StaticOutputDomains &outputDomains,
               const CardDAGAnalysis &dag, std::string *failureReason) {
  // The deterministic baseline is a full-card SPMD execution, not a
  // single-Tile escape hatch. Pick the largest participant count that every
  // structured node can use (all available Tiles for ordinary Attention),
  // then solve only the finite iterator-axis compatibility problem on one
  // common physical group. A smaller group is chosen only when a node has no
  // parallel result extent large enough to form nonempty shards; an edge
  // incompatibility never reduces the participant count.
  mlir::FailureOr<WholeDAGPlacementSearchDomain> domain =
      deriveWholeDAGPlacementSearchDomain(dag, topology, cardId, failureReason);
  if (mlir::failed(domain) ||
      domain->nodeOptions.size() != dag.getNodes().size())
    return std::nullopt;

  size_t participantCount = availableTileIds.size();
  for (const auto &options : domain->nodeOptions) {
    if (options.empty())
      return std::nullopt;
    participantCount = std::min(participantCount, options.front().tiles.size());
  }
  if (participantCount == 0) {
    if (failureReason)
      *failureReason = "whole-card baseline has no nonempty spatial group";
    return std::nullopt;
  }

  std::optional<llvm::SmallVector<PhysicalTileId, 16>> participantGroup;
  for (const WholeDAGNodePlacement &option : domain->nodeOptions.front()) {
    if (option.tiles.size() != participantCount)
      continue;
    const bool sharedByEveryNode =
        llvm::all_of(domain->nodeOptions, [&](const auto &nodeOptions) {
          return llvm::any_of(nodeOptions, [&](const auto &nodeOption) {
            return nodeOption.tiles == option.tiles;
          });
        });
    if (sharedByEveryNode) {
      participantGroup = option.tiles;
      break;
    }
  }
  if (!participantGroup) {
    if (failureReason)
      *failureReason =
          "whole-card baseline has no common maximum-participant Tile group";
    return std::nullopt;
  }

  std::vector<llvm::SmallVector<WholeDAGNodePlacement, 4>> nodeOptions;
  nodeOptions.reserve(domain->nodeOptions.size());
  for (const auto &options : domain->nodeOptions) {
    llvm::SmallVector<WholeDAGNodePlacement, 4> selected;
    for (const WholeDAGNodePlacement &option : options)
      if (option.tiles == *participantGroup)
        selected.push_back(option);
    if (selected.empty()) {
      if (failureReason)
        *failureReason =
            "whole-card baseline cannot bind a node to its participant group";
      return std::nullopt;
    }
    nodeOptions.push_back(std::move(selected));
  }

  struct EdgeCompatibility {
    CardDAGNodeID producer = 0;
    CardDAGNodeID consumer = 0;
    size_t consumerOptionCount = 0;
    std::vector<uint8_t> legal;

    bool contains(size_t producerOption, size_t consumerOption) const {
      return legal[producerOption * consumerOptionCount + consumerOption] != 0;
    }
  };

  WholeDAGEdgeStrategyPlanner edgePlanner(dag);
  std::vector<EdgeCompatibility> compatibility(dag.getEdges().size());
  for (const CardDAGEdge &edge : dag.getEdges()) {
    EdgeCompatibility &relation = compatibility[edge.id];
    relation.producer = edge.producer;
    relation.consumer = edge.consumer;
    relation.consumerOptionCount = nodeOptions[edge.consumer].size();
    relation.legal.assign(nodeOptions[edge.producer].size() *
                              nodeOptions[edge.consumer].size(),
                          0);
    std::string lastRelationFailure;
    for (auto [producerIndex, producer] :
         llvm::enumerate(nodeOptions[edge.producer])) {
      for (auto [consumerIndex, consumer] :
           llvm::enumerate(nodeOptions[edge.consumer])) {
        const CardDAGNode *producerNode = dag.getNode(edge.producer);
        const CardDAGNode *consumerNode = dag.getNode(edge.consumer);
        const bool direct =
            producerNode && consumerNode && producerNode->operation &&
            consumerNode->operation &&
            consumerNode->operation->getOperand(edge.consumerOperand) ==
                producerNode->operation->getResult(edge.producerResult);
        bool local = false;
        std::string ignoredFailure;
        if (direct) {
          mlir::FailureOr<WholeDAGEdgeStrategyPlan> plan =
              edgePlanner.derive(edge.id, producer, consumer, &ignoredFailure);
          // A baseline disables fusion, not required data movement. Exact
          // peer fragments are a legal op boundary when two 16-Tile spatial
          // axes differ; local edges become DDR RegionCuts below.
          local = mlir::succeeded(plan);
        } else if (producerNode && consumerNode &&
                   producer.tiles == consumer.tiles) {
          local = mlir::succeeded(deriveUnaryPureSupportChain(
              producerNode->operation, edge.producerResult,
              consumerNode->operation, edge.consumerOperand, &ignoredFailure));
        }
        if (!local)
          lastRelationFailure = ignoredFailure;
        relation.legal[producerIndex * relation.consumerOptionCount +
                       consumerIndex] = local ? 1 : 0;
      }
    }
    if (llvm::none_of(relation.legal,
                      [](uint8_t value) { return value != 0; })) {
      if (failureReason) {
        llvm::raw_string_ostream stream(*failureReason);
        stream << "whole-card baseline dependency " << edge.id << " ("
               << dag.getNode(edge.producer)->operation->getName() << ':'
               << dag.getNode(edge.producer)
                      ->operation->getResult(edge.producerResult)
                      .getType()
               << " -> " << dag.getNode(edge.consumer)->operation->getName()
               << ':'
               << dag.getNode(edge.consumer)
                      ->operation->getOperand(edge.consumerOperand)
                      .getType()
               << ") has no legal 16-Tile axis relation";
        if (!lastRelationFailure.empty())
          stream << ": " << lastRelationFailure;
      }
      return std::nullopt;
    }
  }

  WholeDAGPlacementEvaluator evaluator(dag, topology, cardId);
  using BaselineDomains = std::vector<std::vector<size_t>>;
  BaselineDomains initialDomains;
  initialDomains.reserve(nodeOptions.size());
  for (const auto &options : nodeOptions) {
    std::vector<size_t> values(options.size());
    std::iota(values.begin(), values.end(), 0);
    initialDomains.push_back(std::move(values));
  }

  auto propagate = [&](BaselineDomains &domains) {
    bool changed = false;
    do {
      changed = false;
      for (const EdgeCompatibility &edge : compatibility) {
        auto &producer = domains[edge.producer];
        auto &consumer = domains[edge.consumer];
        const size_t producerSize = producer.size();
        const size_t consumerSize = consumer.size();
        llvm::erase_if(producer, [&](size_t producerOption) {
          return llvm::none_of(consumer, [&](size_t consumerOption) {
            return edge.contains(producerOption, consumerOption);
          });
        });
        if (producer.empty())
          return false;
        llvm::erase_if(consumer, [&](size_t consumerOption) {
          return llvm::none_of(producer, [&](size_t producerOption) {
            return edge.contains(producerOption, consumerOption);
          });
        });
        if (consumer.empty())
          return false;
        changed |=
            producer.size() != producerSize || consumer.size() != consumerSize;
      }
    } while (changed);
    return true;
  };

  std::optional<WholeDAGPlacementCandidate> evaluated;
  std::string evaluationFailure;
  std::function<bool(BaselineDomains)> solve = [&](BaselineDomains domains) {
    if (!propagate(domains))
      return false;
    size_t selectedNode = domains.size();
    for (size_t node = 0; node < domains.size(); ++node)
      if (domains[node].size() > 1 &&
          (selectedNode == domains.size() ||
           domains[node].size() < domains[selectedNode].size()))
        selectedNode = node;
    if (selectedNode == domains.size()) {
      llvm::SmallVector<WholeDAGNodePlacement, 16> placements;
      placements.reserve(domains.size());
      for (size_t node = 0; node < domains.size(); ++node)
        placements.push_back(nodeOptions[node][domains[node].front()]);
      mlir::FailureOr<WholeDAGPlacementCandidate> result =
          evaluator.evaluate(placements, &evaluationFailure);
      if (mlir::failed(result))
        return false;
      evaluated = std::move(*result);
      return true;
    }
    for (size_t option : domains[selectedNode]) {
      BaselineDomains child = domains;
      child[selectedNode] = {option};
      if (solve(std::move(child)))
        return true;
    }
    return false;
  };
  if (!solve(std::move(initialDomains)) || !evaluated) {
    if (failureReason)
      *failureReason = evaluationFailure.empty()
                           ? "whole-card baseline has no legal "
                             "maximum-participant spatial assignment"
                           : evaluationFailure;
    return std::nullopt;
  }
  return makeNodePlacementCandidate(/*stableOrdinal=*/0, *evaluated,
                                    outputDomains, dag);
}

static std::optional<WholeCardCandidate> deriveDeterministicBaseline(
    const PhysicalTopology &topology, PhysicalCardId cardId,
    llvm::ArrayRef<PhysicalTileId> availableTileIds,
    const StaticOutputDomains &outputDomains, const CardDAGAnalysis &dag,
    WholeCardExecutableSynthesisStatistics &statistics,
    llvm::raw_ostream &diagnostics) {
  std::string failure;
  std::optional<WholeCardCandidate> spatial = deriveBaseline(
      topology, cardId, availableTileIds, outputDomains, dag, &failure);
  if (!spatial || !prepareEdgeStrategies(*spatial, dag)) {
    diagnostics << "wafer-compile: deterministic whole-card baseline failed: "
                << (failure.empty() ? "no legal spatial coordinate" : failure)
                << '\n';
    return std::nullopt;
  }
  spatial->assignment.mapping.materializationMode =
      SpatialDataflowMaterializationMode::IndependentDDRStages;
  for (SpatialEdgeStrategy &strategy :
       spatial->assignment.mapping.edgeStrategies) {
    if (strategy.fragments.empty()) {
      if (strategy.sourceTile != strategy.destinationTile) {
        diagnostics << "wafer-compile: deterministic whole-card baseline has "
                       "a nonlocal edge without exact fragments\n";
        return std::nullopt;
      }
      strategy.action = SpatialEdgeAction::RegionCut;
    }
    strategy.bufferCount = 1;
  }
  if (!appendBaselineSupportChainTransfers(*spatial, dag, &failure)) {
    diagnostics << "wafer-compile: deterministic whole-card baseline cannot "
                   "isolate a structured support dependency: "
                << failure << '\n';
    return std::nullopt;
  }
  std::optional<WholeCardCandidate> baseline =
      makeTemporalVariant(*spatial, outputDomains, dag, /*stableOrdinal=*/0,
                          /*additionalWaveRefinements=*/0);
  if (!baseline)
    return std::nullopt;
  baseline->transition.feedbackRootOrdinal = baseline->transition.stableOrdinal;
  statistics.candidateProposals = 1;
  statistics.shortlistedCandidates = 0;
  statistics.multiReductionAxisCandidateProposals =
      hasMultiReductionAxisSplit(*baseline) ? 1 : 0;
  statistics.layoutAssignedCandidateProposals =
      hasLayoutAssignment(*baseline) ? 1 : 0;
  statistics.layoutConversionCandidateProposals =
      hasLayoutConversion(*baseline) ? 1 : 0;
  statistics.bufferedCandidateProposals = hasBufferedEdge(*baseline) ? 1 : 0;
  statistics.applicableFusionLogicalEdges = countFusedLogicalEdges(*baseline);
  statistics.fusedEdgeCandidateProposals =
      statistics.applicableFusionLogicalEdges != 0 ? 1 : 0;
  return baseline;
}

static std::vector<WholeCardCandidate>
deriveShortlist(const PhysicalTopology &topology, PhysicalCardId cardId,
                llvm::ArrayRef<PhysicalTileId> availableTileIds,
                const StaticOutputDomains &outputDomains,
                const CardDAGAnalysis &dag,
                WholeCardExecutableSynthesisStatistics &statistics,
                llvm::raw_ostream &diagnostics,
                CandidateResourceScheduleMemo &resourceScheduleMemo) {
  std::string baselineFailure;
  std::optional<WholeCardCandidate> derivedBaseline = deriveBaseline(
      topology, cardId, availableTileIds, outputDomains, dag, &baselineFailure);
  if (!derivedBaseline) {
    statistics.candidateProposals = 0;
    statistics.shortlistedCandidates = 0;
    diagnostics << "wafer-compile: deterministic whole-card baseline failed: "
                << (baselineFailure.empty() ? "no legal spatial coordinate"
                                            : baselineFailure)
                << '\n';
    return {};
  }
  WholeCardCandidate baselineSpatial = std::move(*derivedBaseline);
  if (!prepareEdgeStrategies(baselineSpatial, dag)) {
    statistics.candidateProposals = 0;
    statistics.shortlistedCandidates = 0;
    return {};
  }
  std::optional<WholeCardCandidate> baselineProposal = makeTemporalVariant(
      baselineSpatial, outputDomains, dag, /*stableOrdinal=*/0,
      /*additionalWaveRefinements=*/0);
  if (!baselineProposal) {
    statistics.candidateProposals = 0;
    statistics.shortlistedCandidates = 0;
    return {};
  }
  WholeCardCandidate baseline = std::move(*baselineProposal);
  std::vector<WholeCardCandidate> spatialSeeds;
  spatialSeeds.push_back(baselineSpatial);
  std::optional<WholeCardCandidate> bestSpatialOpposing;
  std::optional<WholeCardCandidate> bestMultiStageSpatial;
  std::optional<WholeCardCandidate> bestIndependentSpatial;
  std::optional<WholeCardCandidate> bestPartialOverlapSpatial;
  std::optional<WholeCardCandidate> bestDisjointSpatial;
  uint64_t nextStableOrdinal = 1;

  // Keep the spatial domain factorized. The former prefix expansion
  // materialized a Cartesian product before temporal/layout/resource choices
  // existed (53-node decode reached 1,136,800 states at node 3 and would have
  // reached 168 million at node 4). Here every node still exposes every legal
  // iterator-axis/rectangle option, but search evaluates it as one
  // coordinate of a complete whole-card state. Strict improvement is measured
  // after exact edge closure and the common resource calendar. Repeating
  // sweeps until no coordinate improves has structural finite termination and
  // no width, candidate-count, or elapsed-time policy.
  std::string spatialFailure;
  mlir::FailureOr<WholeDAGPlacementSearchDomain> spatialDomain = [&] {
    wafer::support::ScopedCompileTimingSpan timing(
        "search-phase", "whole-card-executable-synthesis",
        "spatial-placement-enumeration");
    return deriveWholeDAGPlacementSearchDomain(dag, topology, cardId,
                                               &spatialFailure);
  }();
  if (mlir::failed(spatialDomain)) {
    ++statistics.nodePlacementEnumerationFailures;
    diagnostics << "wafer-compile: whole-DAG placement domain failed: "
                << spatialFailure << '\n';
  } else {
    statistics.nodePlacementGroups = spatialDomain->placementGroupCount;
    WholeDAGPlacementEvaluator placementEvaluator(dag, topology, cardId);
    WholeCardCandidate incumbent = baselineSpatial;
    mlir::FailureOr<WholeDAGPlacementCandidate> baselinePlacement =
        placementEvaluator.evaluate(incumbent.assignment.nodePlacements,
                                    &spatialFailure);
    if (mlir::succeeded(baselinePlacement)) {
      if (std::optional<WholeCardCandidate> evaluated =
              makeNodePlacementCandidate(
                  nextStableOrdinal++, *baselinePlacement, outputDomains, dag))
        incumbent = std::move(*evaluated);
    }

    auto retainSpatialWitness = [&](std::optional<WholeCardCandidate> &slot,
                                    const WholeCardCandidate &candidate) {
      if (!slot || cheapCandidateLess(candidate, *slot))
        slot = candidate;
    };
    auto observeSpatialCandidate = [&](const WholeCardCandidate &candidate) {
      if (sameNodePlacements(baselineSpatial, candidate))
        return;
      retainSpatialWitness(bestSpatialOpposing, candidate);
      if (candidate.evaluation.distinctTileGroupCount >= 3)
        retainSpatialWitness(bestMultiStageSpatial, candidate);
      if (candidate.evaluation.parallelComponentCount > 1)
        retainSpatialWitness(bestIndependentSpatial, candidate);
      if (candidate.evaluation.partialOverlapEdgeCount > 0)
        retainSpatialWitness(bestPartialOverlapSpatial, candidate);
      if (candidate.evaluation.disjointEdgeCount > 0)
        retainSpatialWitness(bestDisjointSpatial, candidate);
    };

    bool improved = false;
    wafer::support::ScopedCompileTimingSpan coordinateTiming(
        "search-phase", "whole-card-executable-synthesis",
        "spatial-coordinate-search");
    do {
      improved = false;
      for (size_t node = 0; node < spatialDomain->nodeOptions.size(); ++node) {
        WholeCardCandidate best = incumbent;
        std::optional<WholeCardCandidate> bestDifferentPlacement;
        struct PlacementQuery {
          llvm::SmallVector<WholeDAGNodePlacement, 16> placements;
          std::optional<WholeDAGPlacementCandidate> result;
          std::string failure;
        };
        std::vector<PlacementQuery> queries;
        queries.reserve(spatialDomain->nodeOptions[node].size());
        for (const WholeDAGNodePlacement &option :
             spatialDomain->nodeOptions[node]) {
          const WholeDAGNodePlacement &currentPlacement =
              incumbent.assignment.nodePlacements[node];
          if (option.spatialIteratorDimension ==
                  currentPlacement.spatialIteratorDimension &&
              option.iteratorPartitionFactors ==
                  currentPlacement.iteratorPartitionFactors &&
              option.shardDimension == currentPlacement.shardDimension &&
              option.tiles == currentPlacement.tiles)
            continue;
          PlacementQuery query;
          query.placements = incumbent.assignment.nodePlacements;
          query.placements[node] = option;
          queries.push_back(std::move(query));
        }

        // With all other node coordinates fixed, every value of this one
        // spatial coordinate is an independent exact query. Evaluate those
        // values concurrently, including their complete edge fragments,
        // routes, residency and common resource calendar. Results are folded
        // below in domain order, so the exhaustive domain, strict-improvement
        // walk and deterministic tie-breaking are unchanged.
        mlir::MLIRContext *context = dag.getFunction().getContext();
        const unsigned placementWorkers =
            getBoundedTilePipelineWorkerCount(context, queries.size());
        std::vector<std::unique_ptr<WholeDAGPlacementEvaluator>>
            workerEvaluators;
        workerEvaluators.reserve(placementWorkers);
        for (unsigned worker = 0; worker < placementWorkers; ++worker)
          workerEvaluators.push_back(
              std::make_unique<WholeDAGPlacementEvaluator>(dag, topology,
                                                           cardId));
        for (size_t batchBegin = 0; batchBegin < queries.size();
             batchBegin += placementWorkers) {
          const size_t batchSize =
              std::min<size_t>(placementWorkers, queries.size() - batchBegin);
          runBoundedTilePipelines(
              context, batchSize,
              [&](size_t localIndex) {
                PlacementQuery &query = queries[batchBegin + localIndex];
                mlir::FailureOr<WholeDAGPlacementCandidate> evaluated =
                    workerEvaluators[localIndex]->evaluate(query.placements,
                                                           &query.failure);
                if (mlir::succeeded(evaluated))
                  query.result = std::move(*evaluated);
              },
              placementWorkers);
        }

        statistics.nodePlacementStatesExpanded = saturatingAdd(
            statistics.nodePlacementStatesExpanded, queries.size());
        for (PlacementQuery &query : queries) {
          if (!query.result) {
            statistics.nodePlacementTransitionsRejected =
                saturatingAdd(statistics.nodePlacementTransitionsRejected, 1);
            spatialFailure = std::move(query.failure);
            continue;
          }
          std::optional<WholeCardCandidate> candidate =
              makeNodePlacementCandidate(nextStableOrdinal++, *query.result,
                                         outputDomains, dag);
          if (!candidate)
            continue;
          statistics.nodePlacementCandidateProposals =
              saturatingAdd(statistics.nodePlacementCandidateProposals, 1);
          if (!bestDifferentPlacement ||
              cheapCandidateLess(*candidate, *bestDifferentPlacement))
            bestDifferentPlacement = *candidate;
          if (cheapCandidateLess(*candidate, best))
            best = std::move(*candidate);
        }
        // Preserve the best opposing value of this coordinate for actual
        // feedback. This is a measured-search trade-off, not a domain cap: all
        // values above were evaluated, and later sweeps can accumulate changes
        // from any strictly improving coordinate.
        if (bestDifferentPlacement)
          observeSpatialCandidate(*bestDifferentPlacement);
        if (cheapCandidateLess(best, incumbent)) {
          incumbent = std::move(best);
          observeSpatialCandidate(incumbent);
          improved = true;
        }
      }
    } while (improved);
    observeSpatialCandidate(incumbent);

    // A best-response walk from the all-overlapping baseline cannot cross a
    // temporarily non-improving two-group state to reach a useful three-stage
    // placement.  Seed that basin constructively from the same complete
    // domain: assign successive DAG nodes to distinct singleton physical
    // Tiles when those exact options exist, then evaluate the resulting whole
    // state through the ordinary edge planner and resource calendar.  This is
    // a topology-derived multi-start, not a cap or a special-case placement.
    if (spatialDomain->nodeOptions.size() >= 3 && !availableTileIds.empty()) {
      llvm::SmallVector<WholeDAGNodePlacement, 16> stagedPlacements =
          incumbent.assignment.nodePlacements;
      bool constructed = true;
      for (size_t node = 0; node < spatialDomain->nodeOptions.size(); ++node) {
        PhysicalTileId desired =
            availableTileIds[node % availableTileIds.size()];
        auto option =
            llvm::find_if(spatialDomain->nodeOptions[node],
                          [&](const WholeDAGNodePlacement &placement) {
                            return placement.tiles.size() == 1 &&
                                   placement.tiles.front() == desired;
                          });
        if (option == spatialDomain->nodeOptions[node].end()) {
          constructed = false;
          break;
        }
        stagedPlacements[node] = *option;
      }
      if (constructed) {
        statistics.nodePlacementStatesExpanded =
            saturatingAdd(statistics.nodePlacementStatesExpanded, 1);
        mlir::FailureOr<WholeDAGPlacementCandidate> staged =
            placementEvaluator.evaluate(stagedPlacements, &spatialFailure);
        if (mlir::succeeded(staged)) {
          if (std::optional<WholeCardCandidate> candidate =
                  makeNodePlacementCandidate(nextStableOrdinal++, *staged,
                                             outputDomains, dag)) {
            statistics.nodePlacementCandidateProposals =
                saturatingAdd(statistics.nodePlacementCandidateProposals, 1);
            observeSpatialCandidate(*candidate);
          }
        } else {
          statistics.nodePlacementTransitionsRejected =
              saturatingAdd(statistics.nodePlacementTransitionsRejected, 1);
        }
      }
    }
  }

  auto appendDistinctSpatialWitness =
      [&](std::optional<WholeCardCandidate> &witness) {
        if (!witness || llvm::any_of(spatialSeeds, [&](const auto &existing) {
              return sameNodePlacements(existing, *witness);
            }))
          return;
        spatialSeeds.push_back(std::move(*witness));
      };
  appendDistinctSpatialWitness(bestSpatialOpposing);
  appendDistinctSpatialWitness(bestMultiStageSpatial);
  appendDistinctSpatialWitness(bestIndependentSpatial);
  appendDistinctSpatialWitness(bestPartialOverlapSpatial);
  appendDistinctSpatialWitness(bestDisjointSpatial);

  // Every spatial seed enters temporal refinement with a complete edge-action
  // carrier. Unsupported exact redistribution removes only that seed; it does
  // not restore an implicit fusion fallback or mutate its placement.
  std::vector<WholeCardCandidate> preparedSpatialSeeds;
  preparedSpatialSeeds.reserve(spatialSeeds.size());
  for (WholeCardCandidate &seed : spatialSeeds)
    if (prepareEdgeStrategies(seed, dag))
      preparedSpatialSeeds.push_back(std::move(seed));
  spatialSeeds = std::move(preparedSpatialSeeds);

  // All non-spatial coordinates now stay factorized as well.  A complete
  // candidate owns the temporal vector for every operation/output and one
  // action/layout/buffer tuple for every logical SSA edge.  Repeated
  // best-response sweeps inspect the full finite value domain of each
  // coordinate conditional on the current complete state.  Strict total-order
  // improvement gives structural termination without a beam width, candidate
  // count, temporal depth or elapsed-time policy.  This is the measured
  // trade-off for the decode Cartesian product: the domain is unchanged, but
  // it is queried as a factor graph instead of materialized as a product.
  std::vector<WholeCardCandidate> proposals;
  std::unordered_map<size_t, llvm::SmallVector<size_t, 2>> proposalHashBuckets;
  auto appendUnique = [&](WholeCardCandidate candidate) {
    const size_t hash = getMappingHash(candidate);
    llvm::SmallVector<size_t, 2> &bucket = proposalHashBuckets[hash];
    if (llvm::any_of(bucket, [&](size_t index) {
          return sameNodePlacements(candidate, proposals[index]) &&
                 sameMapping(candidate, proposals[index]);
        }))
      return;
    bucket.push_back(proposals.size());
    proposals.push_back(std::move(candidate));
  };
  constexpr size_t kEdgeActionCount = 7;
  std::array<std::optional<WholeCardCandidate>, kEdgeActionCount>
      actionWitnesses;
  std::array<std::array<std::optional<WholeCardCandidate>, 4>, kEdgeActionCount>
      actionBufferWitnesses;
  std::optional<WholeCardCandidate> multiReductionWitness;
  uint64_t evaluatedCoordinates = 0;

  auto retainWitness = [&](std::optional<WholeCardCandidate> &slot,
                           const WholeCardCandidate &candidate) {
    if (!slot || cheapCandidateLess(candidate, *slot))
      slot = candidate;
  };
  auto retainActionWitness = [&](std::optional<WholeCardCandidate> &slot,
                                 const WholeCardCandidate &candidate) {
    const bool candidateSingleBuffer = !hasBufferedEdge(candidate);
    const bool slotSingleBuffer = slot && !hasBufferedEdge(*slot);
    if (!slot || (candidateSingleBuffer && !slotSingleBuffer) ||
        (candidateSingleBuffer == slotSingleBuffer &&
         cheapCandidateLess(candidate, *slot)))
      slot = candidate;
  };
  auto retainBufferWitness = [&](std::optional<WholeCardCandidate> &slot,
                                 const WholeCardCandidate &candidate) {
    if (!slot || std::tuple(countBufferedLogicalEdges(candidate),
                            candidate.evaluation.temporalWaveLowerBound,
                            candidate.transition.stableOrdinal) <
                     std::tuple(countBufferedLogicalEdges(*slot),
                                slot->evaluation.temporalWaveLowerBound,
                                slot->transition.stableOrdinal))
      slot = candidate;
  };
  auto observeJointAxes = [&](const WholeCardCandidate &candidate) {
    for (const SpatialEdgeStrategy &strategy :
         candidate.assignment.mapping.edgeStrategies) {
      const size_t action = static_cast<size_t>(strategy.action);
      if (action < actionWitnesses.size())
        retainActionWitness(actionWitnesses[action], candidate);
      const size_t buffers = std::min<size_t>(strategy.bufferCount, 3);
      if (action < actionBufferWitnesses.size() && buffers > 1)
        retainBufferWitness(actionBufferWitnesses[action][buffers], candidate);
    }
    if (hasMultiReductionAxisSplit(candidate))
      retainWitness(multiReductionWitness, candidate);
  };
  auto evaluateCoordinate = [&](WholeCardCandidate &candidate) {
    ++evaluatedCoordinates;
    populateTemporalMetrics(candidate, outputDomains, dag);
    std::string ignoredFailure;
    if (!refreshCandidateResourceSchedule(candidate, dag, topology, cardId,
                                          availableTileIds, &ignoredFailure,
                                          &resourceScheduleMemo)) {
      ++statistics.resourceScheduleRejections;
      return false;
    }
    observeJointAxes(candidate);
    return true;
  };
  auto evaluateCoordinates = [&](std::vector<WholeCardCandidate> &candidates) {
    std::vector<uint8_t> accepted(candidates.size(), uint8_t{0});
    evaluatedCoordinates =
        saturatingAdd(evaluatedCoordinates, candidates.size());
    runBoundedTilePipelines(
        dag.getFunction().getContext(), candidates.size(), [&](size_t index) {
          WholeCardCandidate &candidate = candidates[index];
          populateTemporalMetrics(candidate, outputDomains, dag);
          std::string ignoredFailure;
          accepted[index] = refreshCandidateResourceSchedule(
              candidate, dag, topology, cardId, availableTileIds,
              &ignoredFailure, &resourceScheduleMemo);
        });
    for (size_t index = 0; index < candidates.size(); ++index) {
      if (!accepted[index]) {
        ++statistics.resourceScheduleRejections;
        continue;
      }
      observeJointAxes(candidates[index]);
    }
    return accepted;
  };

  auto getWaveBreakpoints = [&](int64_t extent) {
    llvm::SmallVector<int64_t, 16> result;
    if (extent <= 0)
      return result;
    int64_t current = extent;
    result.push_back(current);
    while (current > 1) {
      current = getNextLowerWaveBreakpoint(extent, current);
      result.push_back(current);
    }
    return result;
  };

  {
    wafer::support::ScopedCompileTimingSpan timing(
        "search-phase", "whole-card-executable-synthesis",
        "joint-coordinate-search");
    for (WholeCardCandidate &seed : spatialSeeds) {
      std::optional<WholeCardCandidate> initial = makeTemporalVariant(
          seed, outputDomains, dag, seed.transition.stableOrdinal,
          /*additionalWaveRefinements=*/0);
      if (!initial || !evaluateCoordinate(*initial))
        continue;
      WholeCardCandidate incumbent = std::move(*initial);
      appendUnique(incumbent);

      // Reduction-order legality creates a second measured basin: a later
      // non-reassociated reduction axis becomes splittable only after every
      // preceding reduction axis is already at its unit endpoint.  Construct
      // that finite boundary state in lexical iterator order and send it
      // through the same evaluator.  Without this multi-start, coordinate
      // descent can falsely report that the later axis is absent.
      WholeCardCandidate reductionBoundary = incumbent;
      bool splitMultipleReductionAxes = false;
      for (auto [node, selected] : llvm::enumerate(
               reductionBoundary.assignment.mapping.operationTemporalTiles)) {
        if (node >= reductionBoundary.assignment.nodePlacements.size())
          continue;
        std::optional<llvm::SmallVector<int64_t, 4>> ranges =
            getMaximumSpatialIteratorRanges(
                selected.operation,
                reductionBoundary.assignment.nodePlacements[node]);
        auto tiling =
            mlir::dyn_cast_or_null<mlir::TilingInterface>(selected.operation);
        if (!ranges || !tiling ||
            ranges->size() != selected.iteratorTileSizes.size())
          continue;
        unsigned splitAxes = 0;
        for (auto [dimension, iteratorType] :
             llvm::enumerate(tiling.getLoopIteratorTypes())) {
          if (iteratorType != mlir::utils::IteratorType::reduction ||
              (*ranges)[dimension] <= 1)
            continue;
          if (!isOrderPreservingReductionRefinement(
                  selected.operation, static_cast<unsigned>(dimension), *ranges,
                  selected.iteratorTileSizes))
            continue;
          selected.iteratorTileSizes[dimension] = 1;
          ++splitAxes;
        }
        splitMultipleReductionAxes |= splitAxes > 1;
      }
      if (splitMultipleReductionAxes) {
        reductionBoundary.transition.stableOrdinal = nextStableOrdinal++;
        if (evaluateCoordinate(reductionBoundary)) {
          retainWitness(multiReductionWitness, reductionBoundary);
          appendUnique(std::move(reductionBoundary));
        }
      }

      bool improved = false;
      do {
        improved = false;

        // Observable traversal coordinates participate for every output and
        // every finite ceilDiv class.  They are separate from operation
        // iterator coordinates because the output store traversal is an
        // actual downstream obligation.
        for (size_t outputIndex = 0;
             outputIndex < incumbent.assignment.mapping.outputs.size();
             ++outputIndex) {
          const unsigned resultIndex =
              incumbent.assignment.mapping.outputs[outputIndex].outputIndex;
          llvm::SmallVector<int64_t, 4> shardShape =
              getMaximumSpatialShardShape(
                  incumbent.assignment.mapping.outputs[outputIndex],
                  outputDomains[resultIndex]);
          for (size_t dimension = 0; dimension < shardShape.size();
               ++dimension) {
            WholeCardCandidate best = incumbent;
            std::vector<WholeCardCandidate> candidates;
            for (int64_t value : getWaveBreakpoints(shardShape[dimension])) {
              if (value == incumbent.assignment.mapping.outputs[outputIndex]
                               .temporalTileSizes[dimension])
                continue;
              WholeCardCandidate candidate = incumbent;
              candidate.transition.stableOrdinal = nextStableOrdinal++;
              candidate.assignment.mapping.outputs[outputIndex]
                  .temporalTileSizes[dimension] = value;
              candidates.push_back(std::move(candidate));
            }
            std::vector<uint8_t> accepted = evaluateCoordinates(candidates);
            for (size_t index = 0; index < candidates.size(); ++index) {
              if (accepted[index] &&
                  cheapCandidateLess(candidates[index], best))
                best = std::move(candidates[index]);
            }
            if (cheapCandidateLess(best, incumbent)) {
              incumbent = std::move(best);
              improved = true;
            }
          }
        }

        // Every parallel and reduction iterator has its complete finite
        // breakpoint coordinate.  Non-reassociated floating reductions keep
        // their proven lexicographic split order.
        for (size_t node = 0;
             node < incumbent.assignment.mapping.operationTemporalTiles.size();
             ++node) {
          mlir::Operation *operation =
              incumbent.assignment.mapping.operationTemporalTiles[node]
                  .operation;
          std::optional<llvm::SmallVector<int64_t, 4>> ranges =
              getMaximumSpatialIteratorRanges(
                  operation, incumbent.assignment.nodePlacements[node]);
          if (!ranges || ranges->size() != incumbent.assignment.mapping
                                               .operationTemporalTiles[node]
                                               .iteratorTileSizes.size())
            continue;
          for (size_t dimension = 0; dimension < ranges->size(); ++dimension) {
            WholeCardCandidate best = incumbent;
            std::vector<WholeCardCandidate> candidates;
            for (int64_t value : getWaveBreakpoints((*ranges)[dimension])) {
              if (value ==
                  incumbent.assignment.mapping.operationTemporalTiles[node]
                      .iteratorTileSizes[dimension])
                continue;
              WholeCardCandidate candidate = incumbent;
              candidate.transition.stableOrdinal = nextStableOrdinal++;
              StructuredOpTemporalTile &candidateTile =
                  candidate.assignment.mapping.operationTemporalTiles[node];
              if (value < candidateTile.iteratorTileSizes[dimension] &&
                  !isOrderPreservingReductionRefinement(
                      candidateTile.operation, static_cast<unsigned>(dimension),
                      *ranges, candidateTile.iteratorTileSizes))
                continue;
              candidateTile.iteratorTileSizes[dimension] = value;
              candidates.push_back(std::move(candidate));
            }
            std::vector<uint8_t> accepted = evaluateCoordinates(candidates);
            for (size_t index = 0; index < candidates.size(); ++index) {
              if (accepted[index] &&
                  cheapCandidateLess(candidates[index], best))
                best = std::move(candidates[index]);
            }
            if (cheapCandidateLess(best, incumbent)) {
              incumbent = std::move(best);
              improved = true;
            }
          }
        }

        // One coordinate is one logical SSA edge, not one destination-Tile
        // fragment.  Thus action, typed layout relation and buffer count are
        // changed atomically across every destination strategy of that edge.
        llvm::SmallVector<llvm::SmallVector<size_t, 16>, 32> edgeGroups =
            getLogicalEdgeStrategyGroups(incumbent);
        for (llvm::ArrayRef<size_t> group : edgeGroups) {
          WholeCardCandidate best = incumbent;
          std::vector<WholeCardCandidate> candidates;
          const bool peer = llvm::any_of(group, [&](size_t index) {
            const SpatialEdgeStrategy &strategy =
                incumbent.assignment.mapping.edgeStrategies[index];
            return !strategy.fragments.empty() ||
                   strategy.sourceTile != strategy.destinationTile;
          });
          constexpr std::array<SpatialEdgeAction, 6> localActions = {
              SpatialEdgeAction::CoupledFusion,
              SpatialEdgeAction::LocalShardResidency,
              SpatialEdgeAction::SpillReload,
              SpatialEdgeAction::Recompute,
              SpatialEdgeAction::RegionCut,
              SpatialEdgeAction::LocalPhysicalConversion};
          auto considerAction = [&](SpatialEdgeAction action) {
            for (uint8_t buffers : {uint8_t{1}, uint8_t{2}, uint8_t{3}}) {
              std::optional<WholeCardCandidate> candidate =
                  makeLogicalEdgeVariant(incumbent, group, action, buffers,
                                         nextStableOrdinal++);
              if (!candidate || sameMapping(*candidate, incumbent))
                continue;
              candidates.push_back(std::move(*candidate));
            }
          };
          if (peer)
            considerAction(SpatialEdgeAction::PeerFragments);
          else
            for (SpatialEdgeAction action : localActions)
              considerAction(action);
          std::vector<uint8_t> accepted = evaluateCoordinates(candidates);
          for (size_t index = 0; index < candidates.size(); ++index) {
            if (accepted[index] && cheapCandidateLess(candidates[index], best))
              best = std::move(candidates[index]);
          }
          if (cheapCandidateLess(best, incumbent)) {
            incumbent = std::move(best);
            improved = true;
          }
        }
      } while (improved);
      appendUnique(std::move(incumbent));
    }
  }

  for (std::optional<WholeCardCandidate> &witness : actionWitnesses)
    if (witness)
      appendUnique(std::move(*witness));
  for (auto &actionWitnesses : actionBufferWitnesses)
    for (size_t buffers : {size_t{2}, size_t{3}})
      if (actionWitnesses[buffers])
        appendUnique(std::move(*actionWitnesses[buffers]));
  if (multiReductionWitness)
    appendUnique(std::move(*multiReductionWitness));

  statistics.candidateProposals = evaluatedCoordinates;
  statistics.multiReductionAxisCandidateProposals =
      llvm::count_if(proposals, hasMultiReductionAxisSplit);
  statistics.multiStagePlacementCandidateProposals =
      llvm::count_if(proposals, [](const WholeCardCandidate &candidate) {
        return candidate.transition.nodePlacementCandidate &&
               candidate.evaluation.distinctTileGroupCount >= 3;
      });
  statistics.independentComponentCandidateProposals =
      llvm::count_if(proposals, [](const WholeCardCandidate &candidate) {
        return candidate.transition.nodePlacementCandidate &&
               candidate.evaluation.parallelComponentCount > 1;
      });
  statistics.alternativeEdgeActionCandidateProposals =
      llvm::count_if(proposals, hasAlternativeEdgeAction);
  statistics.layoutAssignedCandidateProposals =
      llvm::count_if(proposals, hasLayoutAssignment);
  statistics.layoutConversionCandidateProposals =
      llvm::count_if(proposals, hasLayoutConversion);
  statistics.layoutBufferedCandidateProposals =
      llvm::count_if(proposals, hasLayoutBufferedEdge);
  statistics.bufferedCandidateProposals =
      llvm::count_if(proposals, hasBufferedEdge);
  statistics.fusedEdgeCandidateProposals =
      llvm::count_if(proposals, [](const WholeCardCandidate &candidate) {
        return countFusedLogicalEdges(candidate) != 0;
      });
  for (const WholeCardCandidate &candidate : proposals)
    statistics.applicableFusionLogicalEdges =
        std::max(statistics.applicableFusionLogicalEdges,
                 countFusedLogicalEdges(candidate));

  // Observable result shape is not a capacity lower bound. Stateful outputs
  // such as a KV cache can remain a DDR view and update only one slice, while
  // ordinary output traversal can stream compact tiles. Keep the derived
  // footprint as an ordering/diagnostic estimate and let actual Instr-IR
  // lifetime plus fixed-capacity packing own SPM feasibility.
  for (WholeCardCandidate &proposal : proposals)
    proposal.transition.feedbackRootOrdinal = proposal.transition.stableOrdinal;
  llvm::sort(proposals, cheapCandidateLess);
  statistics.cheapPrunedCandidates = 0;
  statistics.shortlistedCandidates = proposals.size();
  statistics.feedbackBeamDeferredCandidates = 0;
  return proposals;
}

static bool operationCarriesLayout(mlir::Operation *operation,
                                   MemLayout layout) {
  auto hasLayout = [&](mlir::Type type) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
    MemoryAttr memory = memref ? getWaferMemoryAttr(memref) : MemoryAttr{};
    return memory && memory.getSpace() == MemorySpace::SPM &&
           memory.getLayout() == layout;
  };
  return llvm::any_of(operation->getOperandTypes(), hasLayout) ||
         llvm::any_of(operation->getResultTypes(), hasLayout);
}

static std::optional<CardDAGNodeID>
findStructuredNode(const CardDAGAnalysis &dag,
                   mlir::Operation *sourceOperation) {
  for (const CardDAGNode &node : dag.getNodes())
    if (node.operation == sourceOperation)
      return node.id;
  return std::nullopt;
}

static mlir::FailureOr<llvm::SmallVector<SelectedBufferRequest, 4>>
buildSelectedBufferRequestsForTile(const WholeCardCandidate &candidate,
                                   PhysicalTileId tile,
                                   const CardDAGAnalysis &dag,
                                   std::string &failureReason) {
  llvm::SmallVector<SelectedBufferRequest, 4> requests;
  for (llvm::ArrayRef<size_t> group : getLogicalEdgeStrategyGroups(candidate)) {
    if (group.empty())
      continue;
    const SpatialEdgeStrategy &representative =
        candidate.assignment.mapping.edgeStrategies[group.front()];
    if (representative.bufferCount <= 1)
      continue;
    if (llvm::any_of(group, [&](size_t index) {
          return candidate.assignment.mapping.edgeStrategies[index]
                     .bufferCount != representative.bufferCount;
        })) {
      failureReason =
          "one selected logical edge has inconsistent buffer multiplicity";
      return mlir::failure();
    }

    SelectedBufferRequest request;
    request.producerNode = findStructuredNode(dag, representative.producer);
    request.consumerNode = findStructuredNode(dag, representative.consumer);
    request.bufferCount = representative.bufferCount;
    if (!request.producerNode || !request.consumerNode) {
      failureReason = "selected buffering lost its structured DAG nodes";
      return mlir::failure();
    }

    auto appendMessage = [&](SelectedBufferMessage message) {
      if (!llvm::any_of(
              request.messages, [&](const SelectedBufferMessage &existing) {
                return existing.direction == message.direction &&
                       existing.communicationId == message.communicationId &&
                       existing.payloadSlice == message.payloadSlice;
              }))
        request.messages.push_back(message);
    };
    for (size_t index : group) {
      const SpatialEdgeStrategy &strategy =
          candidate.assignment.mapping.edgeStrategies[index];
      if (strategy.fragments.empty()) {
        request.requireLocalDataflow |=
            strategy.sourceTile == tile && strategy.destinationTile == tile;
        continue;
      }
      for (const SpatialEdgeFragment &fragment : strategy.fragments) {
        if (fragment.kind == SpatialEdgeFragmentKind::Resident) {
          request.requireLocalDataflow |=
              fragment.sourceTile == tile && strategy.destinationTile == tile;
          continue;
        }
        if (fragment.sourceTile == tile)
          appendMessage({SelectedBufferMessageDirection::Send,
                         fragment.communicationId, fragment.payloadSlice});
        if (strategy.destinationTile == tile)
          appendMessage({SelectedBufferMessageDirection::Receive,
                         fragment.communicationId, fragment.payloadSlice});
      }
    }
    if (!request.requireLocalDataflow && request.messages.empty())
      continue;
    llvm::sort(request.messages, [](const SelectedBufferMessage &lhs,
                                    const SelectedBufferMessage &rhs) {
      return std::tuple(static_cast<uint8_t>(lhs.direction),
                        lhs.communicationId, lhs.payloadSlice) <
             std::tuple(static_cast<uint8_t>(rhs.direction),
                        rhs.communicationId, rhs.payloadSlice);
    });
    requests.push_back(std::move(request));
  }
  return requests;
}

static bool rootContainsNodeLayout(
    mlir::Operation *root, CardDAGNodeID node, MemLayout layout,
    const StructuredMaterializationRelations &materializationRelations) {
  bool found = false;
  root->walk([&](mlir::Operation *operation) {
    if (!found &&
        operationUsesStructuredNode(operation, node,
                                    materializationRelations) &&
        operationCarriesLayout(operation, layout))
      found = true;
  });
  return found;
}

static bool rootContainsLayout(mlir::Operation *root, MemLayout layout) {
  bool found = false;
  root->walk([&](mlir::Operation *operation) {
    found |= operationCarriesLayout(operation, layout);
  });
  return found;
}

static bool rootContainsLayoutConversion(mlir::Operation *root,
                                         MemLayout source,
                                         MemLayout destination) {
  bool found = false;
  root->walk([&](LayoutMaterializeOp materialize) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(materialize.getSource().getType());
    auto destinationType =
        mlir::dyn_cast<mlir::MemRefType>(materialize.getResult().getType());
    MemoryAttr sourceMemory =
        sourceType ? getWaferMemoryAttr(sourceType) : MemoryAttr{};
    MemoryAttr destinationMemory =
        destinationType ? getWaferMemoryAttr(destinationType) : MemoryAttr{};
    found |= sourceMemory && destinationMemory &&
             sourceMemory.getSpace() == MemorySpace::SPM &&
             destinationMemory.getSpace() == MemorySpace::SPM &&
             sourceMemory.getLayout() == source &&
             destinationMemory.getLayout() == destination;
  });
  return found;
}

static bool isSPMValue(mlir::Value value) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  MemoryAttr memory = memref ? getWaferMemoryAttr(memref) : MemoryAttr{};
  return memory && memory.getSpace() == MemorySpace::SPM;
}

/// Proves that work for one selected producer/consumer node pair is
/// connected inside one actual TileRegion.  A direct/transitive SSA path, a
/// shared typed SPM value, or one fused-location operation is accepted; a DDR
/// boundary or mere co-location without dataflow is not.
static bool rootContainsActualFusedEdge(
    mlir::Operation *root, CardDAGNodeID producerNode,
    CardDAGNodeID consumerNode,
    const StructuredMaterializationRelations &materializationRelations) {
  llvm::SmallVector<mlir::Operation *, 16> producers;
  llvm::SmallVector<mlir::Operation *, 16> consumers;
  root->walk([&](mlir::Operation *operation) {
    const bool producer = operationUsesStructuredNode(operation, producerNode,
                                                      materializationRelations);
    const bool consumer = operationUsesStructuredNode(operation, consumerNode,
                                                      materializationRelations);
    if (producer && consumer && operation->getParentOfType<TileRegionOp>())
      producers.push_back(operation), consumers.push_back(operation);
    else {
      if (producer)
        producers.push_back(operation);
      if (consumer)
        consumers.push_back(operation);
    }
  });

  for (mlir::Operation *producer : producers) {
    TileRegionOp region = producer->getParentOfType<TileRegionOp>();
    if (!region)
      continue;
    for (mlir::Operation *consumer : consumers) {
      if (consumer->getParentOfType<TileRegionOp>() != region)
        continue;
      if (producer == consumer)
        return true;
      for (mlir::Value producerValue : producer->getOperands()) {
        if (!isSPMValue(producerValue))
          continue;
        if (llvm::is_contained(consumer->getOperands(), producerValue) ||
            llvm::is_contained(consumer->getResults(), producerValue))
          return true;
      }
      for (mlir::Value producerValue : producer->getResults()) {
        if (isSPMValue(producerValue) &&
            (llvm::is_contained(consumer->getOperands(), producerValue) ||
             llvm::is_contained(consumer->getResults(), producerValue)))
          return true;
      }

      llvm::SmallVector<mlir::Operation *, 16> worklist{producer};
      llvm::DenseSet<mlir::Operation *> visited;
      while (!worklist.empty()) {
        mlir::Operation *current = worklist.pop_back_val();
        if (!visited.insert(current).second)
          continue;
        for (mlir::Value result : current->getResults())
          for (mlir::Operation *user : result.getUsers()) {
            if (user == consumer)
              return true;
            if (user->getParentOfType<TileRegionOp>() == region)
              worklist.push_back(user);
          }
      }
    }
  }
  return false;
}

static mlir::LogicalResult validateSelectedFusion(
    mlir::ModuleOp cardProgram, const WholeCardCandidate &candidate,
    const CardDAGAnalysis &dag,
    const StructuredMaterializationRelations &materializationRelations,
    uint64_t &actualFusedLogicalEdges, std::string &failureReason) {
  actualFusedLogicalEdges = 0;
  for (const llvm::SmallVector<size_t, 16> &group :
       getLogicalEdgeStrategyGroups(candidate)) {
    if (group.empty() || !llvm::all_of(group, [&](size_t index) {
          return candidate.assignment.mapping.edgeStrategies[index].action ==
                 SpatialEdgeAction::CoupledFusion;
        }))
      continue;
    const SpatialEdgeStrategy &strategy =
        candidate.assignment.mapping.edgeStrategies[group.front()];
    std::optional<CardDAGNodeID> producer =
        findStructuredNode(dag, strategy.producer);
    std::optional<CardDAGNodeID> consumer =
        findStructuredNode(dag, strategy.consumer);
    if (!producer || !consumer) {
      failureReason = "selected fusion lost its structured DAG nodes";
      return mlir::failure();
    }
    const bool witnessed =
        rootContainsActualFusedEdge(cardProgram.getOperation(), *producer,
                                    *consumer, materializationRelations);
    if (!witnessed) {
      failureReason =
          "selected coupled fusion has no actual in-region dataflow witness";
      return mlir::failure();
    }
    ++actualFusedLogicalEdges;
  }
  return mlir::success();
}

static mlir::LogicalResult validateSelectedTileLayouts(
    mlir::Operation *root, const WholeCardCandidate &candidate,
    PhysicalTileId tile, const CardDAGAnalysis &dag,
    const StructuredMaterializationRelations &materializationRelations,
    std::string &failureReason) {
  for (const SpatialEdgeStrategy &strategy :
       candidate.assignment.mapping.edgeStrategies) {
    if (!strategy.hasLayoutAssignment ||
        !isSpatialEdgeStrategyIncidentOnTile(strategy, tile))
      continue;
    std::optional<CardDAGNodeID> producer =
        findStructuredNode(dag, strategy.producer);
    std::optional<CardDAGNodeID> consumer =
        findStructuredNode(dag, strategy.consumer);
    if (!producer || !consumer) {
      failureReason = "selected layout lost its structured DAG nodes";
      return mlir::failure();
    }
    const bool producerWitness =
        rootContainsNodeLayout(root, *producer, strategy.producerLayout,
                               materializationRelations) ||
        rootContainsLayout(root, strategy.producerLayout);
    const bool consumerWitness =
        rootContainsNodeLayout(root, *consumer, strategy.consumerLayout,
                               materializationRelations) ||
        rootContainsLayout(root, strategy.consumerLayout);
    const bool materializesProducer =
        strategy.action == SpatialEdgeAction::PeerFragments
            ? llvm::any_of(strategy.fragments,
                           [&](const SpatialEdgeFragment &fragment) {
                             return fragment.sourceTile == tile;
                           })
            : strategy.sourceTile == tile;
    if (materializesProducer && !producerWitness) {
      failureReason = (llvm::Twine("selected producer layout ") +
                       stringifyMemLayout(strategy.producerLayout) +
                       " for structured node " + llvm::Twine(*producer) +
                       " has no typed actual IR witness on Tile " +
                       llvm::Twine(tile.getValue()))
                          .str();
      return mlir::failure();
    }
    if (strategy.destinationTile == tile && !consumerWitness) {
      failureReason = (llvm::Twine("selected consumer layout ") +
                       stringifyMemLayout(strategy.consumerLayout) +
                       " for structured node " + llvm::Twine(*consumer) +
                       " has no typed actual IR witness on Tile " +
                       llvm::Twine(tile.getValue()))
                          .str();
      return mlir::failure();
    }
    if (strategy.destinationTile == tile &&
        strategy.action == SpatialEdgeAction::LocalPhysicalConversion &&
        !rootContainsLayoutConversion(root, strategy.producerLayout,
                                      strategy.consumerLayout)) {
      failureReason =
          "selected local layout conversion has no actual movement witness";
      return mlir::failure();
    }
  }
  return mlir::success();
}

static PhysicalTileMemoryPlanningFailure::SPMDemandEvidence
makeSPMDemandEvidence(
    const SPMMemoryPlanningFailure::DemandEvidence &demand,
    const StructuredMaterializationRelations &materializationRelations) {
  PhysicalTileMemoryPlanningFailure::SPMDemandEvidence evidence{
      demand.location, demand.allocation, demand.type, demand.bytes, {}};
  evidence.userLocations.append(demand.userLocations.begin(),
                                demand.userLocations.end());
  auto appendNode = [](auto &nodes, uint32_t node) {
    if (!llvm::is_contained(nodes, node))
      nodes.push_back(node);
  };
  for (const auto &relation : materializationRelations.operationResultBuffers)
    if (shareStructuredBufferStorage(demand.allocation, relation.buffer))
      appendNode(evidence.operationResultNodes, relation.structuredNodeId);
  for (const auto &relation : materializationRelations.operandBuffers)
    if (shareStructuredBufferStorage(demand.allocation, relation.buffer))
      appendNode(evidence.operandDemandNodes, relation.structuredNodeId);
  for (const auto &relation : materializationRelations.outputBuffers)
    if (shareStructuredBufferStorage(demand.allocation, relation.buffer) &&
        !llvm::is_contained(evidence.outputIndices, relation.outputIndex))
      evidence.outputIndices.push_back(relation.outputIndex);
  return evidence;
}

static CardExecutableTileFailure makeTileSPMCapacityFailure(
    PhysicalTileId tileId, const SPMMemoryPlanningFailure &planningFailure,
    const StructuredMaterializationRelations &materializationRelations) {
  CardExecutableTileFailure tileFailure;
  tileFailure.tileId = tileId;
  PhysicalTileMemoryPlanningFailure &failure = tileFailure.memoryPlanning;
  failure.kind = PhysicalTileMemoryPlanningFailureKind::SPMAllocation;
  failure.spmCapacityOverflow =
      planningFailure.kind == SPMMemoryPlanningFailureKind::CapacityOverflow;
  failure.spmPlanningFailureKind = planningFailure.kind;
  failure.spmLargestDemandLocation = planningFailure.largestDemandLocation;
  failure.spmLargestDemandType = planningFailure.largestDemandType;
  failure.spmLargestDemandBytes = planningFailure.largestDemandBytes;
  failure.spmDemandCount = planningFailure.demandCount;
  auto appendEvidence = [&](const auto &from, auto &to) {
    to.push_back(makeSPMDemandEvidence(from, materializationRelations));
  };
  for (const auto &demand : planningFailure.largestDemands)
    appendEvidence(demand, failure.spmLargestDemands);
  for (const auto &demand : planningFailure.capacityConflictDemands)
    appendEvidence(demand, failure.spmCapacityConflictDemands);
  for (const auto &demand : planningFailure.individuallyOversizedDemands)
    appendEvidence(demand, failure.spmIndividuallyOversizedDemands);
  return tileFailure;
}

static void collectSPMCapacityDemandEvidence(
    llvm::ArrayRef<CardExecutableTileFailure> tileFailures,
    bool composePrimaryAllocationOwner, llvm::raw_ostream &diagnostics,
    llvm::SmallVectorImpl<SPMCapacityDemandEvidence> &spmCapacityDemands) {
  uint64_t largestSPMCapacityDemandBytes = 0;
  bool selectedExactSPMCausalSet = false;
  for (const CardExecutableTileFailure &tileFailure : tileFailures) {
    const PhysicalTileMemoryPlanningFailure &failure =
        tileFailure.memoryPlanning;
    if (!failure.spmCapacityOverflow)
      continue;
    llvm::SmallVector<PhysicalTileMemoryPlanningFailure::SPMDemandEvidence, 4>
        largestDemands = failure.spmLargestDemands;
    const bool hasIndividuallyOversizedDemands =
        !failure.spmIndividuallyOversizedDemands.empty();
    const bool hasExactCapacityConflict =
        !failure.spmCapacityConflictDemands.empty();
    const bool hasExactCausalSet =
        hasIndividuallyOversizedDemands || hasExactCapacityConflict;
    uint64_t exactConflictBytes = 0;
    uint64_t exactConflictDemandCount = 0;
    if (hasExactCausalSet) {
      // One physical Tile's independently oversized set or over-capacity
      // clique is already a complete rejection proof. Choose the first
      // failing Tile deterministically; mixing independent Tile
      // certificates would create unrelated joint coordinates and cannot
      // strengthen legality.
      if (selectedExactSPMCausalSet)
        continue;
      selectedExactSPMCausalSet = true;
      spmCapacityDemands.clear();
      llvm::ArrayRef<PhysicalTileMemoryPlanningFailure::SPMDemandEvidence>
          exactCausalDemands(failure.spmCapacityConflictDemands);
      if (hasIndividuallyOversizedDemands)
        exactCausalDemands = failure.spmIndividuallyOversizedDemands;
      exactConflictDemandCount = exactCausalDemands.size();
      for (const auto &demand : exactCausalDemands)
        exactConflictBytes = saturatingAdd(exactConflictBytes, demand.bytes);
      diagnostics << "wafer-compile: whole-card-exact-spm-"
                  << (hasIndividuallyOversizedDemands
                          ? "individually-oversized-demands"
                          : "conflict-certificate")
                  << " physical_tile_id=" << tileFailure.tileId.getValue()
                  << " causal_demands=" << exactCausalDemands.size()
                  << " causal_bytes=" << exactConflictBytes
                  << " demand_count=" << failure.spmDemandCount << '\n';
    } else if (selectedExactSPMCausalSet) {
      continue;
    }
    llvm::ArrayRef<PhysicalTileMemoryPlanningFailure::SPMDemandEvidence>
        feedbackDemands(largestDemands);
    if (hasIndividuallyOversizedDemands)
      feedbackDemands = failure.spmIndividuallyOversizedDemands;
    else if (hasExactCapacityConflict)
      feedbackDemands = failure.spmCapacityConflictDemands;
    for (const auto &demand : feedbackDemands) {
      const bool attributed = !demand.operationResultNodes.empty() ||
                              !demand.operandDemandNodes.empty() ||
                              !demand.outputIndices.empty();
      if (!attributed) {
        diagnostics << "wafer-compile: whole-card-unattributed-spm-demand"
                    << " physical_tile_id=" << tileFailure.tileId.getValue()
                    << " demand_bytes=" << demand.bytes
                    << " demand_count=" << failure.spmDemandCount
                    << " demand_type=" << demand.type
                    << " demand_location=" << demand.location << '\n';
      }
      if (!hasExactCausalSet) {
        if (demand.bytes < largestSPMCapacityDemandBytes)
          continue;
        if (demand.bytes > largestSPMCapacityDemandBytes) {
          largestSPMCapacityDemandBytes = demand.bytes;
          spmCapacityDemands.clear();
        }
      }
      auto appendEvidence = [&](std::optional<CardDAGNodeID> operationNode,
                                std::optional<unsigned> outputIndex,
                                std::optional<CardDAGNodeID> operandNode) {
        SPMCapacityDemandEvidence evidence;
        evidence.physicalTileId = tileFailure.tileId;
        evidence.operationNode = operationNode;
        evidence.outputIndex = outputIndex;
        evidence.operandDemandNode = operandNode;
        evidence.relationFromAllocation = true;
        evidence.type = demand.type;
        evidence.bytes = demand.bytes;
        evidence.demandCount = failure.spmDemandCount;
        evidence.exactConflictBytes =
            hasExactCausalSet ? exactConflictBytes : demand.bytes;
        evidence.exactConflictDemandCount =
            hasExactCausalSet ? exactConflictDemandCount : uint64_t{1};
        if (!llvm::any_of(spmCapacityDemands, [&](const auto &known) {
              return known.physicalTileId == evidence.physicalTileId &&
                     known.operationNode == evidence.operationNode &&
                     known.outputIndex == evidence.outputIndex &&
                     known.operandDemandNode == evidence.operandDemandNode;
            }))
          spmCapacityDemands.push_back(std::move(evidence));
      };
      // Baseline region cuts may expose both the producer result allocation
      // and the downstream operand demand. Compose both current-IR relations;
      // search policy may still prioritize one coordinate below.
      if (composePrimaryAllocationOwner) {
        for (uint32_t node : demand.operationResultNodes)
          appendEvidence(node, std::nullopt, std::nullopt);
      }
      if (!demand.operandDemandNodes.empty()) {
        for (uint32_t node : demand.operandDemandNodes)
          appendEvidence(std::nullopt, std::nullopt, node);
      } else if (!demand.outputIndices.empty()) {
        for (unsigned output : demand.outputIndices)
          appendEvidence(std::nullopt, output, std::nullopt);
      } else {
        for (uint32_t node : demand.operationResultNodes)
          appendEvidence(node, std::nullopt, std::nullopt);
      }
      if (!attributed)
        appendEvidence(std::nullopt, std::nullopt, std::nullopt);
    }
  }
}

static mlir::FailureOr<WholeCardExecutable> materializeCandidate(
    mlir::ModuleOp tensorProgram, const WholeCardCandidate &candidate,
    PhysicalCardId physicalCardId,
    llvm::ArrayRef<PhysicalTileId> expectedTileIds,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    const CardDAGAnalysis &dag,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeCardSynthesisStatistics &gateStatistics,
    WholeCardExecutableSynthesisStatistics *searchStatistics,
    uint64_t &rotatingSlotAllocationsMaterialized,
    uint64_t &actualFusedLogicalEdges, unsigned tilePipelineParallelism,
    std::optional<PhysicalTileId> preferredFailureProbeTileId,
    std::string &failureGate, std::string &failureReason,
    SelectedBufferMaterializationFailure &selectedBufferFailure,
    bool &indeterminateFailure, bool &spmCapacityOverflow,
    llvm::SmallVectorImpl<SPMCapacityDemandEvidence> &spmCapacityDemands,
    llvm::SmallVectorImpl<AcceptedOperationNodeRelation>
        &acceptedOperationNodes,
    std::vector<std::string> &tileDataflowIRTrace) {
  selectedBufferFailure = {};
  indeterminateFailure = false;
  spmCapacityOverflow = false;
  spmCapacityDemands.clear();
  acceptedOperationNodes.clear();
  tileDataflowIRTrace.clear();
  (void)preferredFailureProbeTileId;
  (void)searchStatistics;
  mlir::OwningOpRef<mlir::ModuleOp> cardProgram;
  StructuredMaterializationRelations materializationRelations;
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "conversion", "whole-card-executable-synthesis",
        "tensor-program-to-card-program");
    mlir::LogicalResult loweredCard = lowerTensorProgramToCardProgram(
        tensorProgram, physicalCardId, candidate.assignment.mapping,
        cardProgram, &failureReason, operationNodes, &materializationRelations);
    if (mlir::failed(loweredCard)) {
      failureGate = "card-program-materialization";
      return mlir::failure();
    }
  }

  if (mlir::failed(validateSelectedFusion(
          *cardProgram, candidate, dag, materializationRelations,
          actualFusedLogicalEdges, failureReason))) {
    failureGate = "selected-fusion-materialization";
    return mlir::failure();
  }

  llvm::SmallVector<TileProgramOp, 16> tilePrograms;
  cardProgram->walk(
      [&](TileProgramOp tileProgram) { tilePrograms.push_back(tileProgram); });
  llvm::sort(tilePrograms, [](TileProgramOp lhs, TileProgramOp rhs) {
    return lhs.getTileIdAttr().getInt() < rhs.getTileIdAttr().getInt();
  });
  if (tilePrograms.size() != expectedTileIds.size()) {
    failureGate = "card-program-to-tile-modules";
    failureReason = "CardProgram physical Tile domain is incomplete";
    return mlir::failure();
  }
  std::vector<llvm::SmallVector<SelectedBufferRequest, 4>>
      selectedBufferRequests;
  selectedBufferRequests.reserve(tilePrograms.size());
  size_t selectedBufferRequestCount = 0;
  for (auto [index, tileProgram] : llvm::enumerate(tilePrograms)) {
    const PhysicalTileId tileId(tileProgram.getTileIdAttr().getInt());
    if (tileId != expectedTileIds[index]) {
      failureGate = "card-program-to-tile-modules";
      failureReason = "CardProgram changed the verified physical Tile domain";
      return mlir::failure();
    }
    if (mlir::failed(validateSelectedTileLayouts(
            tileProgram.getOperation(), candidate, tileId, dag,
            materializationRelations, failureReason))) {
      failureGate = "selected-layout-materialization";
      return mlir::failure();
    }
    mlir::FailureOr<llvm::SmallVector<SelectedBufferRequest, 4>> requests =
        buildSelectedBufferRequestsForTile(candidate, tileId, dag,
                                           failureReason);
    if (mlir::failed(requests)) {
      failureGate = "selected-buffer-request";
      return mlir::failure();
    }
    selectedBufferRequestCount += requests->size();
    selectedBufferRequests.push_back(std::move(*requests));
  }
  if (hasBufferedEdge(candidate) && selectedBufferRequestCount == 0) {
    failureGate = "selected-buffer-request";
    failureReason =
        "selected buffering has no incident logical-edge request on any "
        "physical Tile";
    return mlir::failure();
  }

  CardExecutableCompilationResult compilation = compileCardProgramToExecutable(
      std::move(cardProgram), physicalCardId, expectedTileIds,
      selectedBufferRequests, materializationRelations, program,
      executionConfig, diagnostics, &gateStatistics, tilePipelineParallelism);
  rotatingSlotAllocationsMaterialized +=
      compilation.rotatingSlotAllocationsMaterialized;
  if (compilation.isAccepted()) {
    acceptedOperationNodes.append(compilation.operationNodeRelations.begin(),
                                  compilation.operationNodeRelations.end());
    tileDataflowIRTrace = std::move(compilation.tileDataflowIRTrace);
    return compilation.takeExecutable();
  }

  indeterminateFailure = compilation.status ==
                         CardExecutableCompilationStatus::IndeterminateFailure;
  failureGate = compilation.gate;
  failureReason = compilation.detail;
  for (const CardExecutableTileFailure &failure : compilation.tileFailures)
    if (failure.selectedBuffer.kind !=
        SelectedBufferMaterializationFailureKind::None) {
      selectedBufferFailure = failure.selectedBuffer;
      break;
    }
  spmCapacityOverflow = llvm::any_of(
      compilation.tileFailures, [](const CardExecutableTileFailure &failure) {
        return failure.memoryPlanning.spmCapacityOverflow;
      });
  collectSPMCapacityDemandEvidence(compilation.tileFailures,
                                   /*composePrimaryAllocationOwner=*/false,
                                   diagnostics, spmCapacityDemands);
  return mlir::failure();
}

static bool refineDeterministicBaselineFromSPM(
    WholeCardCandidate &candidate, const StaticOutputDomains &outputDomains,
    const CardDAGAnalysis &dag,
    llvm::ArrayRef<SPMCapacityDemandEvidence> spmCapacityDemands,
    uint64_t &temporalTransitions, uint64_t &causalOperationCount,
    uint64_t &causalOutputCount, int64_t &causalNode,
    unsigned &refinedDimension, int64_t &previousExtent,
    int64_t &refinedExtent) {
  struct CausalCoordinate {
    mlir::Operation *operation = nullptr;
    std::optional<unsigned> output;
    uint32_t node = std::numeric_limits<uint32_t>::max();
    bool operandDemand = false;
  };
  llvm::SmallVector<CausalCoordinate, 8> coordinates;
  for (const SPMCapacityDemandEvidence &demand : spmCapacityDemands) {
    CausalCoordinate coordinate;
    if (demand.operandDemandNode) {
      coordinate.node = *demand.operandDemandNode;
      if (coordinate.node < dag.getNodes().size())
        coordinate.operation = dag.getNodes()[coordinate.node].operation;
      coordinate.operandDemand = true;
    } else if (demand.outputIndex) {
      coordinate.output = *demand.outputIndex;
    } else if (demand.operationNode) {
      coordinate.node = *demand.operationNode;
      if (coordinate.node < dag.getNodes().size())
        coordinate.operation = dag.getNodes()[coordinate.node].operation;
    }
    if (!coordinate.operation && !coordinate.output)
      continue;
    if (!llvm::any_of(coordinates, [&](const CausalCoordinate &existing) {
          return existing.operation == coordinate.operation &&
                 existing.output == coordinate.output &&
                 existing.operandDemand == coordinate.operandDemand;
        }))
      coordinates.push_back(coordinate);
  }
  llvm::stable_sort(coordinates, [](const CausalCoordinate &lhs,
                                    const CausalCoordinate &rhs) {
    auto priority = [](const CausalCoordinate &coordinate) {
      return coordinate.operandDemand ? 0 : coordinate.output ? 1 : 2;
    };
    return std::tuple(
               priority(lhs), std::numeric_limits<uint32_t>::max() - lhs.node,
               lhs.output.value_or(std::numeric_limits<unsigned>::max())) <
           std::tuple(
               priority(rhs), std::numeric_limits<uint32_t>::max() - rhs.node,
               rhs.output.value_or(std::numeric_limits<unsigned>::max()));
  });
  temporalTransitions = 0;
  causalOperationCount = 0;
  causalOutputCount = 0;
  causalNode = -1;
  refinedDimension = 0;
  previousExtent = 0;
  refinedExtent = 0;
  WholeCardCandidate refined = candidate;
  for (const CausalCoordinate &coordinate : coordinates) {
    unsigned localDimension = 0;
    int64_t localPreviousExtent = 0;
    int64_t localRefinedExtent = 0;
    bool changed = false;
    if (coordinate.operation) {
      size_t refinedNode = 0;
      changed = refineOperationTemporalVariantOnce(
          refined, outputDomains, dag, coordinate.operation,
          // In the independent baseline, a producer-result allocation is an
          // explicit producer wave streamed to DDR, while an operand relation
          // is the exact consumer wave that requested it. Both are direct
          // local coordinates; search uses the more general downstream
          // fallback in its separate controller.
          /*preferredIsOperandDemand=*/true,
          /*refinedOperation=*/nullptr, &refinedNode, &localDimension,
          &localPreviousExtent, &localRefinedExtent,
          /*explicitProducerCanStreamToDDR=*/true);
      if (changed) {
        ++causalOperationCount;
        if (temporalTransitions == 0)
          causalNode = static_cast<int64_t>(refinedNode);
      }
    } else {
      changed = refineOutputTemporalVariantOnce(
          refined, outputDomains, dag, *coordinate.output, &localDimension,
          &localPreviousExtent, &localRefinedExtent);
      if (changed)
        ++causalOutputCount;
    }
    if (!changed)
      continue;
    if (temporalTransitions == 0) {
      refinedDimension = localDimension;
      previousExtent = localPreviousExtent;
      refinedExtent = localRefinedExtent;
    }
    ++temporalTransitions;
  }
  if (temporalTransitions == 0 || sameMapping(refined, candidate))
    return false;
  candidate = std::move(refined);
  return true;
}

static mlir::FailureOr<WholeCardCompilationResult>
synthesizeDeterministicBaseline(
    mlir::ModuleOp tensorProgram, const PhysicalTopology &topology,
    PhysicalCardId physicalCardId,
    llvm::ArrayRef<PhysicalTileId> expectedTileIds,
    const StaticOutputDomains &outputDomains, const CardDAGAnalysis &dag,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    WholeCardExecutableSynthesisStatistics &statistics,
    unsigned tilePipelineParallelism) {
  std::optional<WholeCardCandidate> candidate =
      deriveDeterministicBaseline(topology, physicalCardId, expectedTileIds,
                                  outputDomains, dag, statistics, diagnostics);
  if (!candidate)
    return mlir::failure();

  uint64_t controllerIterations = 0;
  std::optional<PhysicalTileId> scopedProbeTileId;
  while (true) {
    ++controllerIterations;
    const bool scopedProbe = scopedProbeTileId.has_value();
    mlir::OwningOpRef<mlir::ModuleOp> cardProgram;
    StructuredMaterializationRelations materializationRelations;
    std::string failureReason;
    {
      wafer::support::ScopedCompileTimingSpan timing(
          "conversion", "deterministic-baseline",
          "tensor-program-to-card-program");
      mlir::LogicalResult lowered =
          scopedProbe
              ? lowerTensorProgramToCardProgramForPhysicalTile(
                    tensorProgram, physicalCardId, *scopedProbeTileId,
                    candidate->assignment.mapping, cardProgram, &failureReason,
                    operationNodes, &materializationRelations)
              : lowerTensorProgramToCardProgram(
                    tensorProgram, physicalCardId,
                    candidate->assignment.mapping, cardProgram, &failureReason,
                    operationNodes, &materializationRelations);
      if (mlir::failed(lowered)) {
        diagnostics << "wafer-compile: deterministic whole-card baseline "
                       "failed "
                    << (scopedProbe ? "scoped " : "")
                    << "CardProgram materialization: " << failureReason << '\n';
        return mlir::failure();
      }
    }
    if (scopedProbe)
      ++statistics.baselineScopedCardProgramMaterializations;
    else
      ++statistics.baselineCardProgramMaterializations;

    uint64_t actualFusedLogicalEdges = 0;
    if (!scopedProbe &&
        (mlir::failed(validateSelectedFusion(
             *cardProgram, *candidate, dag, materializationRelations,
             actualFusedLogicalEdges, failureReason)) ||
         actualFusedLogicalEdges != 0)) {
      diagnostics << "wafer-compile: deterministic whole-card baseline lost "
                     "its zero-fusion contract: "
                  << failureReason << '\n';
      return mlir::failure();
    }

    llvm::SmallVector<TileProgramOp, 16> tilePrograms;
    cardProgram->walk([&](TileProgramOp tileProgram) {
      tilePrograms.push_back(tileProgram);
    });
    llvm::sort(tilePrograms, [](TileProgramOp lhs, TileProgramOp rhs) {
      return lhs.getTileIdAttr().getInt() < rhs.getTileIdAttr().getInt();
    });
    if (tilePrograms.size() != expectedTileIds.size()) {
      diagnostics << "wafer-compile: deterministic whole-card baseline "
                     "CardProgram has an incomplete physical Tile domain\n";
      return mlir::failure();
    }

    struct RegionEvaluation {
      PhysicalTileId tileId;
      TileRegionOp region;
      llvm::SmallVector<CardDAGNodeID, 2> structuredNodes;
    };
    llvm::SmallVector<RegionEvaluation, 32> regionEvaluations;
    for (auto [tileIndex, tileProgram] : llvm::enumerate(tilePrograms)) {
      const PhysicalTileId tileId(tileProgram.getTileIdAttr().getInt());
      if (tileId != expectedTileIds[tileIndex]) {
        diagnostics << "wafer-compile: deterministic whole-card baseline "
                       "lost its complete physical Tile domain\n";
        return mlir::failure();
      }
      if (scopedProbe && tileId != *scopedProbeTileId)
        continue;
      if (mlir::failed(validateSelectedTileLayouts(
              tileProgram.getOperation(), *candidate, tileId, dag,
              materializationRelations, failureReason))) {
        diagnostics << "wafer-compile: deterministic whole-card baseline "
                       "lost its selected physical assignment: "
                    << failureReason << '\n';
        return mlir::failure();
      }
      llvm::SmallVector<TileRegionOp, 16> regions;
      tileProgram.walk([&](TileRegionOp region) {
        if (!region->getParentOfType<TileRegionOp>())
          regions.push_back(region);
      });
      for (TileRegionOp region : regions) {
        RegionEvaluation evaluation{tileId, region, {}};
        region.walk([&](mlir::Operation *operation) {
          if (!mlir::isa<ComputeFillOp, ComputeConvertOp, ComputeGemmOp,
                         ComputeConvOp, ComputeElementwiseOp, ComputeReduceOp>(
                  operation))
            return;
          for (uint32_t node : collectStructuredNodesUsedByOperation(
                   operation, materializationRelations))
            if (!llvm::is_contained(evaluation.structuredNodes, node))
              evaluation.structuredNodes.push_back(node);
        });
        regionEvaluations.push_back(std::move(evaluation));
      }
    }

    std::vector<TileRegionSPMCapacityEvaluation> capacityResults(
        regionEvaluations.size());
    std::vector<std::string> capacityDiagnostics(regionEvaluations.size());
    TileRegionToInstrLoweringSession loweringSession(
        *cardProgram->getContext());
    auto evaluateRegionSPM = [&](size_t index) {
      llvm::raw_string_ostream stream(capacityDiagnostics[index]);
      capacityResults[index] = evaluateTileRegionSPMCapacity(
          regionEvaluations[index].region, loweringSession, stream);
      stream.flush();
    };
    const unsigned requestedWorkers = tilePipelineParallelism == 0
                                          ? kMaximumBoundedTilePipelineWorkers
                                          : tilePipelineParallelism;
    const unsigned workers = runBoundedTilePipelines(
        cardProgram->getContext(), regionEvaluations.size(), evaluateRegionSPM,
        requestedWorkers);
    statistics.baselineMaximumRegionSPMQueryWorkers = std::max<uint64_t>(
        statistics.baselineMaximumRegionSPMQueryWorkers, workers);
    statistics.baselineRegionSPMCapacityChecks += regionEvaluations.size();

    llvm::SmallVector<size_t, 16> causalFailureIndices;
    for (auto [index, capacity] : llvm::enumerate(capacityResults)) {
      diagnostics << capacityDiagnostics[index];
      if (capacity.fits())
        continue;
      if (capacity.requiresFunctionScope()) {
        ++statistics.baselineRegionSPMChecksRequiringFunctionScope;
        continue;
      }
      if (!capacity.capacityExceeded()) {
        ++statistics.baselineRegionSPMCapacityAnalysisFailures;
        diagnostics << "wafer-compile: deterministic whole-card baseline "
                       "region SPM query is indeterminate phase="
                    << capacity.getPhaseDiagnosticLabel()
                    << " detail=" << capacity.detail << '\n';
        return mlir::failure();
      }
      ++statistics.baselineRegionSPMCapacityOverflowProofs;
      const bool capacityOverflow =
          capacity.phase == TileRegionSPMCapacityPhase::StaticPacking &&
          capacity.planningFailure.kind ==
              SPMMemoryPlanningFailureKind::CapacityOverflow;
      if (!capacityOverflow) {
        diagnostics << "wafer-compile: deterministic whole-card baseline "
                       "received a non-capacity infeasibility proof phase="
                    << capacity.getPhaseDiagnosticLabel()
                    << " detail=" << capacity.detail << '\n';
        return mlir::failure();
      }
      causalFailureIndices.push_back(index);
    }
    if (!causalFailureIndices.empty()) {
      bool sawAttributedConflict = false;
      bool refinedConflict = false;
      for (size_t causalIndex : causalFailureIndices) {
        llvm::SmallVector<SPMCapacityDemandEvidence, 8> demands;
        llvm::SmallVector<CardExecutableTileFailure, 1> localFailures = {
            makeTileSPMCapacityFailure(
                regionEvaluations[causalIndex].tileId,
                capacityResults[causalIndex].planningFailure,
                materializationRelations)};
        collectSPMCapacityDemandEvidence(localFailures,
                                         /*composePrimaryAllocationOwner=*/true,
                                         diagnostics, demands);

        // Some producer-side allocations are created through a downstream
        // support/materialization path whose location retains only the
        // requesting consumer relation. Recover no general semantics from
        // that location: instead, use the typed CardDAG edge and the exact
        // failed memref shape/element type to identify direct structured
        // producers whose result is the allocated DDR-stage wave. Add every
        // exact match to the same deterministic state. Ambiguous equal-shaped
        // operands are all refined together; no branch or guessed winner is
        // introduced.
        const size_t attributedDemandCount = demands.size();
        for (const SPMCapacityDemandEvidence &demand :
             llvm::ArrayRef<SPMCapacityDemandEvidence>(demands).take_front(
                 attributedDemandCount)) {
          if (!demand.operandDemandNode || !demand.type)
            continue;
          auto failedType = mlir::dyn_cast<mlir::ShapedType>(demand.type);
          if (!failedType || !failedType.hasRank())
            continue;
          if (*demand.operandDemandNode >= dag.getNodes().size())
            continue;
          const CardDAGNode &consumerNode =
              dag.getNodes()[*demand.operandDemandNode];
          for (CardDAGEdgeID edgeId : consumerNode.incomingEdges) {
            const CardDAGEdge *edge = dag.getEdge(edgeId);
            const CardDAGNode *producer =
                edge ? dag.getNode(edge->producer) : nullptr;
            if (!edge || !producer || !producer->operation ||
                edge->producerResult >= producer->operation->getNumResults())
              continue;
            auto producerType = mlir::dyn_cast<mlir::ShapedType>(
                producer->operation->getResult(edge->producerResult).getType());
            if (!producerType || !producerType.hasRank() ||
                producerType.getElementType() != failedType.getElementType() ||
                !llvm::equal(producerType.getShape(), failedType.getShape()))
              continue;
            if (llvm::any_of(demands, [&](const auto &known) {
                  return known.operationNode == producer->id;
                }))
              continue;
            SPMCapacityDemandEvidence producerDemand = demand;
            producerDemand.operationNode = producer->id;
            producerDemand.outputIndex.reset();
            producerDemand.operandDemandNode.reset();
            demands.push_back(std::move(producerDemand));
          }
        }

        // The exact probe covers one materialized TileRegion. RegionCut keeps
        // dependent structured stages DDR-separated; source owners with no
        // selected dependency can remain as sequential work in the same
        // region. The allocator's exact conflict certificate names boundary
        // values, while every directly inherited compute owner names a
        // traversal that can contribute an unreported result allocation.
        // Compose all missing owners in the same deterministic local state;
        // this remains one actual-region probe, not a candidate family or a
        // branch.
        for (CardDAGNodeID node :
             regionEvaluations[causalIndex].structuredNodes) {
          if (llvm::any_of(demands, [&](const auto &demand) {
                return demand.operandDemandNode == node;
              }))
            continue;
          SPMCapacityDemandEvidence regionDemand;
          regionDemand.physicalTileId = regionEvaluations[causalIndex].tileId;
          regionDemand.operandDemandNode = node;
          regionDemand.relationFromAllocation = true;
          demands.push_back(std::move(regionDemand));
        }
        if (demands.empty())
          continue;
        sawAttributedConflict = true;

        uint64_t temporalTransitions = 0;
        uint64_t causalOperationCount = 0;
        uint64_t causalOutputCount = 0;
        int64_t causalNode = -1;
        unsigned refinedDimension = 0;
        int64_t previousExtent = 0;
        int64_t refinedExtent = 0;
        if (!refineDeterministicBaselineFromSPM(
                *candidate, outputDomains, dag, demands, temporalTransitions,
                causalOperationCount, causalOutputCount, causalNode,
                refinedDimension, previousExtent, refinedExtent))
          continue;

        ++statistics.allocationFeedbackTransitions;
        diagnostics << "wafer-compile: whole-card-baseline-temporal-refinement"
                    << " state=single source=tile-region-spm-capacity"
                    << " scope=" << (scopedProbe ? "physical-tile" : "card")
                    << " capacity_overflow_regions="
                    << causalFailureIndices.size()
                    << " selected_overflow_region=" << causalIndex
                    << " causal_operations=" << causalOperationCount
                    << " causal_outputs=" << causalOutputCount
                    << " temporal_transitions=" << temporalTransitions
                    << " causal_node=" << causalNode
                    << " refined_dimension=" << refinedDimension
                    << " previous_extent=" << previousExtent
                    << " refined_extent=" << refinedExtent
                    << " placement_changed=0 edge_action_changed=0"
                    << " layout_changed=0 buffer_count_changed=0\n";
        scopedProbeTileId = regionEvaluations[causalIndex].tileId;
        refinedConflict = true;
        break;
      }
      if (refinedConflict)
        continue;
      diagnostics << "wafer-compile: deterministic whole-card baseline "
                  << (sawAttributedConflict
                          ? "exhausted its temporal domain for all proven "
                            "region SPM capacity conflicts\n"
                          : "cannot refine an unattributed region SPM "
                            "capacity conflict\n");
      return mlir::failure();
    }

    if (scopedProbe) {
      diagnostics << "wafer-compile: whole-card-baseline-region-capacity-check"
                  << " physical_tile_id=" << scopedProbeTileId->getValue()
                  << " outcome=within-capacity"
                     " full_card_materialization_required=1\n";
      scopedProbeTileId.reset();
      continue;
    }

    // Every independent region is within the local SPM capacity. The baseline
    // contract (DDR boundaries, zero fusion and one buffer) makes those local
    // proofs complete for SPM;
    // run the unique full CardExecutable seam exactly once for DDR,
    // transport, resource, ABI and program-output executable facts.
    ++statistics.materializedCandidates;
    CardExecutableCompilationResult compilation =
        compileCardProgramToExecutable(
            std::move(cardProgram), physicalCardId, expectedTileIds,
            /*selectedBufferRequests=*/{}, materializationRelations, program,
            executionConfig, diagnostics, &statistics.exactGates,
            tilePipelineParallelism);
    statistics.rotatingSlotAllocationsMaterialized +=
        compilation.rotatingSlotAllocationsMaterialized;
    if (!compilation.isAccepted()) {
      ++statistics.materializationRejections;
      if (compilation.status ==
          CardExecutableCompilationStatus::IndeterminateFailure)
        ++statistics.indeterminateCompilationFailures;
      diagnostics << "wafer-compile: deterministic whole-card baseline "
                     "failed final CardExecutable gate="
                  << compilation.gate << " detail=" << compilation.detail
                  << '\n';
      return mlir::failure();
    }

    WholeCardExecutable executable = compilation.takeExecutable();
    llvm::SmallVector<analysis::WholeCardInstructionProgramCost, 16> phaseCosts;
    mlir::FailureOr<analysis::StaticSchedulePlan> plan =
        buildAcceptedWholeDAGSchedulePlan(
            dag, candidate->assignment.nodePlacements,
            compilation.operationNodeRelations, executable, phaseCosts,
            &failureReason);
    if (mlir::failed(plan)) {
      ++statistics.schedulePlanRejections;
      diagnostics << "wafer-compile: deterministic whole-card baseline "
                     "failed accepted schedule plan: "
                  << failureReason << '\n';
      return mlir::failure();
    }
    llvm::SmallVector<const analysis::StaticSchedulePlan *, 1> plans = {&*plan};
    llvm::SmallVector<analysis::WholeCardResourceDurationEstimate, 16>
        estimates = analysis::estimateStaticSchedulePlanDurations(
            plans, analysis::getTargetScheduleCostPolicy());
    if (estimates.size() != 1) {
      diagnostics << "wafer-compile: deterministic whole-card baseline "
                     "duration estimation failed\n";
      return mlir::failure();
    }

    statistics.acceptedCandidates = 1;
    statistics.plannedCandidates = 1;
    statistics.selectedStableOrdinal = candidate->transition.stableOrdinal;
    statistics.selectedOutputMappingCount =
        candidate->assignment.mapping.outputs.size();
    statistics.selectedUniqueActiveTileCount = getUniqueActiveTileCount(
        candidate->assignment.mapping, candidate->assignment.nodePlacements);
    statistics.selectedParallelComponentCount =
        candidate->evaluation.parallelComponentCount;
    statistics.selectedTemporalWaveLowerBound =
        candidate->evaluation.temporalWaveLowerBound;
    statistics.selectedInstructionExecutionLowerBound =
        candidate->evaluation.instructionExecutionLowerBound;
    statistics.selectedPeakOutputTileFootprintEstimate =
        candidate->evaluation.peakOutputTileFootprintEstimate;
    statistics.selectedPeakAlignedResidencyEstimate =
        candidate->evaluation.peakAlignedResidencyEstimate;
    statistics.selectedPeerTransferCount =
        getPeerFragmentCount(candidate->assignment.mapping);
    statistics.selectedPeerBytes = candidate->evaluation.peerBytes;
    statistics.selectedBufferCount = getMaximumBufferCount(*candidate);
    statistics.selectedActualFusedLogicalEdges = 0;
    statistics.selectedSPMMovementWork =
        candidate->evaluation.scheduledSPMMovementWork;
    statistics.selectedDDRMovementWork =
        candidate->evaluation.scheduledDDRMovementWork;
    statistics.selectedMakespanPicoseconds =
        estimates.front().makespan.picoseconds;
    statistics.enabledDurationTerms = estimates.front().enabledTerms;

    diagnostics << "wafer-compile: whole-card-baseline-controller"
                << " search_states=0 placement_enumeration=0 candidate_family=0"
                << " controller_iterations=" << controllerIterations
                << " card_program_materializations="
                << statistics.baselineCardProgramMaterializations
                << " scoped_card_program_materializations="
                << statistics.baselineScopedCardProgramMaterializations
                << " region_spm_capacity_checks="
                << statistics.baselineRegionSPMCapacityChecks
                << " region_spm_capacity_overflow_proofs="
                << statistics.baselineRegionSPMCapacityOverflowProofs
                << " region_spm_function_scope_checks="
                << statistics.baselineRegionSPMChecksRequiringFunctionScope
                << " region_spm_query_workers="
                << statistics.baselineMaximumRegionSPMQueryWorkers
                << " full_compilations="
                << statistics.exactGates.cardProgramCompilationInvocations
                << '\n';
    diagnostics << "wafer-compile: whole-card-selection"
                << " admitted=1 planned=1 selected_stable_ordinal="
                << statistics.selectedStableOrdinal
                << " output_mappings=" << statistics.selectedOutputMappingCount
                << " active_tiles=" << statistics.selectedUniqueActiveTileCount
                << " parallel_components="
                << statistics.selectedParallelComponentCount
                << " temporal_wave_lb="
                << statistics.selectedTemporalWaveLowerBound
                << " instruction_execution_lb="
                << statistics.selectedInstructionExecutionLowerBound
                << " output_tile_footprint_estimate="
                << statistics.selectedPeakOutputTileFootprintEstimate
                << " aligned_residency_estimate="
                << statistics.selectedPeakAlignedResidencyEstimate
                << " peer_transfers=" << statistics.selectedPeerTransferCount
                << " peer_bytes=" << statistics.selectedPeerBytes
                << " buffer_count=1 actual_fused_edges=0"
                << " spm_movement_work=" << statistics.selectedSPMMovementWork
                << " ddr_movement_work=" << statistics.selectedDDRMovementWork
                << " makespan_ps=" << statistics.selectedMakespanPicoseconds
                << " enabled_terms=" << statistics.enabledDurationTerms << '\n';
    diagnostics
        << "wafer-compile: selected baseline reuses exact-admitted executable"
        << " stable_ordinal=" << statistics.selectedStableOrdinal
        << " selected_executable_rematerializations=0\n";
    return WholeCardCompilationResult(
        std::move(executable), std::move(compilation.tileDataflowIRTrace));
  }
}

static bool
selectionLess(const AdmittedWholeCardCandidate &lhs,
              const analysis::WholeCardResourceDurationEstimate &lhsEstimate,
              const AdmittedWholeCardCandidate &rhs,
              const analysis::WholeCardResourceDurationEstimate &rhsEstimate) {
  if (lhsEstimate.makespan.picoseconds != rhsEstimate.makespan.picoseconds)
    return lhsEstimate.makespan.picoseconds < rhsEstimate.makespan.picoseconds;
  const size_t lhsActive =
      getUniqueActiveTileCount(lhs.candidate.assignment.mapping,
                               lhs.candidate.assignment.nodePlacements);
  const size_t rhsActive =
      getUniqueActiveTileCount(rhs.candidate.assignment.mapping,
                               rhs.candidate.assignment.nodePlacements);
  if (lhsActive != rhsActive)
    return lhsActive > rhsActive;
  if (lhs.candidate.evaluation.parallelComponentCount !=
      rhs.candidate.evaluation.parallelComponentCount)
    return lhs.candidate.evaluation.parallelComponentCount >
           rhs.candidate.evaluation.parallelComponentCount;
  if (lhs.candidate.evaluation.temporalWaveLowerBound !=
      rhs.candidate.evaluation.temporalWaveLowerBound)
    return lhs.candidate.evaluation.temporalWaveLowerBound <
           rhs.candidate.evaluation.temporalWaveLowerBound;
  return lhs.candidate.transition.stableOrdinal <
         rhs.candidate.transition.stableOrdinal;
}

} // namespace

llvm::SmallVector<int64_t, 4>
deriveCapacityTemporalShape(llvm::ArrayRef<int64_t> maximumShardShape,
                            uint64_t elementBytes, uint64_t tensorMultiplicity,
                            const TargetMemoryPolicy &memory,
                            unsigned additionalWaveRefinements) {
  return deriveCapacityTemporalShapeImpl(maximumShardShape, elementBytes,
                                         tensorMultiplicity, memory,
                                         additionalWaveRefinements);
}

mlir::FailureOr<WholeCardCompilationResult> synthesizeWholeCardExecutable(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics,
    WholeCardExecutableSynthesisStatistics *statistics,
    unsigned tilePipelineParallelism) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "stage", "tensor-program-to-executable",
      "whole-card-executable-synthesis");
  if (!tensorProgram || executionConfig.getNumPartitions() != 1 ||
      program.numPartitions != executionConfig.getNumPartitions()) {
    diagnostics << "wafer-compile: whole-card synthesis requires one logical "
                   "card partition\n";
    return mlir::failure();
  }

  WholeCardExecutableSynthesisStatistics localStatistics;
  WholeCardExecutableSynthesisStatistics &resultStatistics =
      statistics ? *statistics : localStatistics;
  resultStatistics = {};

  std::string failureReason;
  mlir::FailureOr<PhysicalTopology> topology =
      PhysicalTopology::create(tensorProgram, &failureReason);
  if (mlir::failed(topology)) {
    diagnostics << "wafer-compile: cannot construct physical topology: "
                << failureReason << '\n';
    return mlir::failure();
  }
  constexpr PhysicalCardId physicalCardId(0);
  std::optional<llvm::ArrayRef<PhysicalTileId>> availableTileIds =
      topology->getAvailableTileIds(physicalCardId);
  if (!availableTileIds ||
      availableTileIds->size() !=
          static_cast<size_t>(executionConfig.getPhysicalTileCount())) {
    diagnostics << "wafer-compile: target topology does not provide the exact "
                   "configured physical Tile domain\n";
    return mlir::failure();
  }

  mlir::FailureOr<mlir::func::FuncOp> structuredProgram =
      getStructuredProgram(tensorProgram, failureReason);
  if (mlir::failed(structuredProgram)) {
    diagnostics << "wafer-compile: " << failureReason << '\n';
    return mlir::failure();
  }
  mlir::FailureOr<CardDAGAnalysis> dag =
      CardDAGAnalysis::create(*structuredProgram, &failureReason);
  if (mlir::failed(dag)) {
    diagnostics << "wafer-compile: cannot derive whole-card structured DAG: "
                << failureReason << '\n';
    return mlir::failure();
  }
  resultStatistics.structuredNodeCount = dag->getNodes().size();
  resultStatistics.structuredEdgeCount = dag->getEdges().size();
  resultStatistics.dependencyComponentCount =
      dag->getObservableDependencyComponents().size();
  resultStatistics.independentComponentPlacementProven =
      dag->supportsIndependentComponentPlacement();

  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
  operationNodes.reserve(dag->getNodes().size());
  for (const CardDAGNode &node : dag->getNodes())
    operationNodes.push_back({node.operation, node.id});

  mlir::FailureOr<StaticOutputDomains> outputDomains =
      getStaticOutputDomains(*structuredProgram, failureReason);
  if (mlir::failed(outputDomains)) {
    diagnostics << "wafer-compile: " << failureReason << '\n';
    return mlir::failure();
  }

  // The deterministic baseline exits before any search memo, shortlist,
  // placement enumeration, candidate family or admitted cohort is constructed.
  if (optimizations.isNone())
    return synthesizeDeterministicBaseline(
        tensorProgram, *topology, physicalCardId, *availableTileIds,
        *outputDomains, *dag, operationNodes, program, executionConfig,
        diagnostics, resultStatistics, tilePipelineParallelism);
  assert(optimizations.isSearch() &&
         "non-baseline synthesis must use the search controller");

  CandidateResourceScheduleMemo resourceScheduleMemo;
  const TargetMemoryPolicy targetMemory = getDefaultWaferTargetPolicy().memory;
  const uint64_t usableSPMCapacity =
      static_cast<uint64_t>(targetMemory.spmLimit - targetMemory.spmBase);
  std::vector<WholeCardCandidate> shortlist = deriveShortlist(
      *topology, physicalCardId, *availableTileIds, *outputDomains, *dag,
      resultStatistics, diagnostics, resourceScheduleMemo);
  std::tie(resultStatistics.resourceScheduleMemoHits,
           resultStatistics.resourceScheduleMemoMisses) =
      resourceScheduleMemo.getCounts();
  diagnostics << "wafer-compile: whole-card-search"
              << " policy=search"
              << " structured_nodes=" << resultStatistics.structuredNodeCount
              << " structured_edges=" << resultStatistics.structuredEdgeCount
              << " dependency_components="
              << resultStatistics.dependencyComponentCount
              << " independent_components="
              << resultStatistics.independentComponentPlacementProven
              << " proposals=" << resultStatistics.candidateProposals
              << " multi_reduction_axis_proposals="
              << resultStatistics.multiReductionAxisCandidateProposals
              << " node_placement_groups="
              << resultStatistics.nodePlacementGroups
              << " node_placement_states="
              << resultStatistics.nodePlacementStatesExpanded
              << " node_placement_rejected="
              << resultStatistics.nodePlacementTransitionsRejected
              << " node_placement_resource_equivalent="
              << resultStatistics.nodePlacementResourceEquivalentStates
              << " node_placement_enumeration_failures="
              << resultStatistics.nodePlacementEnumerationFailures
              << " node_placement_proposals="
              << resultStatistics.nodePlacementCandidateProposals
              << " multi_stage_placement_proposals="
              << resultStatistics.multiStagePlacementCandidateProposals
              << " independent_component_proposals="
              << resultStatistics.independentComponentCandidateProposals
              << " alternative_edge_action_proposals="
              << resultStatistics.alternativeEdgeActionCandidateProposals
              << " layout_assigned_proposals="
              << resultStatistics.layoutAssignedCandidateProposals
              << " layout_conversion_proposals="
              << resultStatistics.layoutConversionCandidateProposals
              << " layout_buffered_proposals="
              << resultStatistics.layoutBufferedCandidateProposals
              << " buffered_proposals="
              << resultStatistics.bufferedCandidateProposals
              << " applicable_fusion_edges="
              << resultStatistics.applicableFusionLogicalEdges
              << " fused_edge_proposals="
              << resultStatistics.fusedEdgeCandidateProposals
              << " resource_schedule_rejections="
              << resultStatistics.resourceScheduleRejections
              << " resource_schedule_memo_hits="
              << resultStatistics.resourceScheduleMemoHits
              << " resource_schedule_memo_misses="
              << resultStatistics.resourceScheduleMemoMisses
              << " cheap_pruned=" << resultStatistics.cheapPrunedCandidates
              << " feedback_beam_deferred="
              << resultStatistics.feedbackBeamDeferredCandidates
              << " shortlist=" << resultStatistics.shortlistedCandidates
              << " mapping=whole-dag-joint-resource\n";

  // Accepted executable modules are the dominant live-memory owner.  Keep
  // only their exact cost/schedule summaries during cohort comparison, then
  // rematerialize the selected state once.  unique_ptr keeps every summary's
  // plan pointers stable while the vector grows; clearing `tiles` releases
  // the actual IR before the next candidate enters the exact pipeline.
  std::vector<std::unique_ptr<AdmittedWholeCardCandidate>> admitted;
  admitted.reserve(shortlist.size());
  struct CachedPreBufferFailure {
    WholeCardCandidate candidate;
    std::string gate;
    std::string reason;
  };
  std::unordered_map<size_t, llvm::SmallVector<CachedPreBufferFailure, 1>>
      preBufferFailureCache;
  struct CachedBufferStructureFailure {
    WholeCardCandidate candidate;
    std::string reason;
  };
  std::unordered_map<size_t, llvm::SmallVector<CachedBufferStructureFailure, 1>>
      bufferStructureFailureCache;
  std::vector<WholeCardCandidate> admittedTemporalRepresentatives;
  // Exact states already materialized or deliberately retired by a measured
  // feedback beam. Retired states remain here so another branch cannot
  // regenerate work that the same query has already deferred.
  std::vector<WholeCardCandidate> seenExactStates;
  std::unordered_map<uint64_t, size_t> allocationFeedbackBeamWidths;
  std::unordered_map<uint64_t, uint64_t> feedbackRootMaterializations;
  struct FeedbackRootDemandWork {
    size_t initialNeighborCount = 0;
    uint64_t initialDemandCount = 0;
  };
  std::unordered_map<uint64_t, FeedbackRootDemandWork> feedbackRootDemandWork;
  // An exact SPM failure is also a measured ordering hint for the next
  // candidate. Probe that physical Tile first; this changes no candidate or
  // legality decision and completes the remaining Tile domain whenever the
  // probe passes.
  std::optional<PhysicalTileId> preferredSPMFailureProbeTileId;
  uint64_t nextFeedbackStableOrdinal = 1;
  for (const WholeCardCandidate &candidate : shortlist)
    nextFeedbackStableOrdinal = std::max(
        nextFeedbackStableOrdinal, candidate.transition.stableOrdinal + 1);
  auto matchesExactState = [&](const WholeCardCandidate &candidate,
                               const WholeCardCandidate &existing) {
    return sameNodePlacements(candidate, existing) &&
           sameMapping(candidate, existing);
  };
  auto findPendingExactState = [&](const WholeCardCandidate &candidate) {
    return llvm::find_if(shortlist, [&](const WholeCardCandidate &existing) {
      return matchesExactState(candidate, existing);
    });
  };
  auto containsAttemptedExactState = [&](const WholeCardCandidate &candidate) {
    auto matches = [&](const WholeCardCandidate &existing) {
      return matchesExactState(candidate, existing);
    };
    return llvm::any_of(seenExactStates, matches) ||
           llvm::any_of(admittedTemporalRepresentatives, matches);
  };
  auto findAdmittedExactState = [&](const WholeCardCandidate &candidate) {
    return llvm::find_if(admittedTemporalRepresentatives,
                         [&](const WholeCardCandidate &existing) {
                           return matchesExactState(candidate, existing);
                         });
  };
  auto containsExactState = [&](const WholeCardCandidate &candidate) {
    return findPendingExactState(candidate) != shortlist.end() ||
           containsAttemptedExactState(candidate);
  };
  auto mergeBetterSPMFeedbackPriority = [](WholeCardCandidate &destination,
                                           const WholeCardCandidate &source) {
    if (source.transition.parentSPMConflictBytes == 0)
      return;
    const auto sourceScore =
        std::tuple(source.transition.parentSPMConflictBytes,
                   source.transition.parentSPMConflictDemandCount,
                   source.transition.parentSPMTotalDemandCount);
    const auto destinationScore =
        std::tuple(destination.transition.parentSPMConflictBytes,
                   destination.transition.parentSPMConflictDemandCount,
                   destination.transition.parentSPMTotalDemandCount);
    if (destination.transition.parentSPMConflictBytes != 0 &&
        destinationScore <= sourceScore)
      return;
    destination.transition.parentSPMConflictBytes =
        source.transition.parentSPMConflictBytes;
    destination.transition.parentSPMConflictDemandCount =
        source.transition.parentSPMConflictDemandCount;
    destination.transition.parentSPMTotalDemandCount =
        source.transition.parentSPMTotalDemandCount;
  };
  auto getFeedbackWaveBreakpoints = [&](int64_t extent) {
    llvm::SmallVector<int64_t, 16> result;
    if (extent <= 0)
      return result;
    result.push_back(extent);
    while (result.back() > 1)
      result.push_back(getNextLowerWaveBreakpoint(extent, result.back()));
    return result;
  };
  while (!shortlist.empty()) {
    WholeCardCandidate candidate = std::move(shortlist.front());
    shortlist.erase(shortlist.begin());
    // Actual model-scale decode exposed the remaining feedback explosion: one
    // allocator conflict with several tied structured relations can enqueue
    // every single-coordinate child plus their joint child, even after this
    // exact initial factorized-search basin already has a verifier-legal actual
    // executable. Feedback changes only temporal coordinates or an
    // allocator-proven impossible explicit edge to its coupled neighbor.
    // Close that basin at its first exact-admitted schedule-plan incumbent.
    // The other initial spatial, layout, fusion, communication and buffer
    // seeds remain independent contenders, and no estimate can admit a
    // candidate: the incumbent below has crossed every actual gate,
    // including fixed-capacity SPM packing.
    const bool needsMultiReductionAxisWitness =
        hasMultiReductionAxisSplit(candidate) &&
        resultStatistics.multiReductionAxisCandidateMaterializations == 0;
    if (!needsMultiReductionAxisWitness &&
        llvm::any_of(
            admittedTemporalRepresentatives,
            [&](const WholeCardCandidate &representative) {
              return !representative.transition.allocationFeedbackLookahead &&
                     (candidate.transition.feedbackRootOrdinal ==
                          representative.transition.feedbackRootOrdinal ||
                      sameNonTemporalState(candidate, representative));
            })) {
      // Remember the retired exact state so another outstanding feedback
      // branch cannot regenerate it before it observes the closed family.
      seenExactStates.push_back(std::move(candidate));
      ++resultStatistics.incumbentClosedFeedbackCandidates;
      continue;
    }
    if (llvm::any_of(admittedTemporalRepresentatives,
                     [&](const WholeCardCandidate &representative) {
                       return isStrictlyDominatedByAdmittedTemporalState(
                           candidate, representative);
                     }) &&
        !needsMultiReductionAxisWitness) {
      ++resultStatistics.strictDominatedCandidates;
      continue;
    }
    seenExactStates.push_back(candidate);
    const size_t preBufferHash =
        getMappingHash(candidate, /*includeBufferCount=*/false);
    auto cachedBucket = preBufferFailureCache.find(preBufferHash);
    if (cachedBucket != preBufferFailureCache.end()) {
      auto cached = llvm::find_if(
          cachedBucket->second, [&](const CachedPreBufferFailure &failure) {
            return samePreBufferMapping(candidate, failure.candidate);
          });
      if (cached != cachedBucket->second.end()) {
        ++resultStatistics.preBufferEquivalentRejections;
        ++resultStatistics.materializationRejections;
        diagnostics << "wafer-compile: whole-card-candidate rejection"
                    << " stable_ordinal=" << candidate.transition.stableOrdinal
                    << " gate=" << cached->gate << " detail=" << cached->reason
                    << " exact_pre_buffer_equivalent=1\n";
        continue;
      }
    }
    const size_t bufferStructureHash = getBufferStructureHash(candidate);
    auto bufferStructureBucket =
        bufferStructureFailureCache.find(bufferStructureHash);
    if (bufferStructureBucket != bufferStructureFailureCache.end()) {
      auto cached = llvm::find_if(
          bufferStructureBucket->second,
          [&](const CachedBufferStructureFailure &failure) {
            return sameBufferStructure(candidate, failure.candidate);
          });
      if (cached != bufferStructureBucket->second.end()) {
        ++resultStatistics.bufferStructureEquivalentRejections;
        ++resultStatistics.materializationRejections;
        diagnostics << "wafer-compile: whole-card-candidate rejection"
                    << " stable_ordinal=" << candidate.transition.stableOrdinal
                    << " gate=selected-buffer-materialization"
                    << " detail=" << cached->reason
                    << " exact_buffer_structure_equivalent=1\n";
        continue;
      }
    }
    ++feedbackRootMaterializations[candidate.transition.feedbackRootOrdinal];
    ++resultStatistics.materializedCandidates;
    if (hasMultiReductionAxisSplit(candidate))
      ++resultStatistics.multiReductionAxisCandidateMaterializations;
    if (candidate.transition.nodePlacementCandidate)
      ++resultStatistics.nodePlacementCandidateMaterializations;
    if (candidate.transition.nodePlacementCandidate &&
        candidate.evaluation.distinctTileGroupCount >= 3)
      ++resultStatistics.multiStagePlacementCandidateMaterializations;
    if (candidate.transition.nodePlacementCandidate &&
        candidate.evaluation.parallelComponentCount > 1)
      ++resultStatistics.independentComponentCandidateMaterializations;
    if (hasAlternativeEdgeAction(candidate))
      ++resultStatistics.alternativeEdgeActionCandidateMaterializations;
    if (hasLayoutAssignment(candidate))
      ++resultStatistics.layoutAssignedCandidateMaterializations;
    if (hasLayoutConversion(candidate))
      ++resultStatistics.layoutConversionCandidateMaterializations;
    if (hasLayoutBufferedEdge(candidate))
      ++resultStatistics.layoutBufferedCandidateMaterializations;
    if (hasBufferedEdge(candidate))
      ++resultStatistics.bufferedCandidateMaterializations;
    std::string failureGate;
    failureReason.clear();
    uint64_t candidateActualFusedLogicalEdges = 0;
    SelectedBufferMaterializationFailure selectedBufferFailure;
    bool indeterminateFailure = false;
    bool spmCapacityOverflow = false;
    llvm::SmallVector<SPMCapacityDemandEvidence, 8> spmCapacityDemands;
    llvm::SmallVector<AcceptedOperationNodeRelation, 64> acceptedOperationNodes;
    std::vector<std::string> tileDataflowIRTrace;
    mlir::FailureOr<WholeCardExecutable> accepted = materializeCandidate(
        tensorProgram, candidate, physicalCardId, *availableTileIds,
        operationNodes, *dag, program, executionConfig, diagnostics,
        resultStatistics.exactGates, &resultStatistics,
        resultStatistics.rotatingSlotAllocationsMaterialized,
        candidateActualFusedLogicalEdges, tilePipelineParallelism,
        preferredSPMFailureProbeTileId, failureGate, failureReason,
        selectedBufferFailure, indeterminateFailure, spmCapacityOverflow,
        spmCapacityDemands, acceptedOperationNodes, tileDataflowIRTrace);
    if (mlir::failed(accepted)) {
      if (indeterminateFailure) {
        ++resultStatistics.indeterminateCompilationFailures;
        diagnostics << "wafer-compile: whole-card candidate compilation is "
                       "indeterminate gate="
                    << failureGate << " detail=" << failureReason << '\n';
        // An indeterminate result proves neither feasibility nor
        // infeasibility. It therefore cannot enter an exact-rejection cache or
        // prune related states, but it also cannot invalidate independently
        // admitted candidates in the same finite cohort.
        continue;
      }
      ++resultStatistics.materializationRejections;
      if (failureGate == "card-program-materialization" ||
          failureGate == "card-program-to-tile-modules" ||
          failureGate == "selected-fusion-materialization" ||
          failureGate == "selected-layout-materialization" ||
          failureGate == "tile-region-to-instr") {
        preBufferFailureCache[preBufferHash].push_back(
            CachedPreBufferFailure{candidate, failureGate, failureReason});
      }
      if (failureGate == "selected-buffer-materialization" &&
          selectedBufferFailure.kind ==
              SelectedBufferMaterializationFailureKind::NoExactLoop) {
        bufferStructureFailureCache[bufferStructureHash].push_back(
            CachedBufferStructureFailure{candidate, failureReason});
      }
      diagnostics
          << "wafer-compile: whole-card-candidate rejection"
          << " stable_ordinal=" << candidate.transition.stableOrdinal
          << " output_mappings=" << candidate.assignment.mapping.outputs.size()
          << " active_tiles="
          << getUniqueActiveTileCount(candidate.assignment.mapping,
                                      candidate.assignment.nodePlacements)
          << " tile_groups=" << candidate.evaluation.distinctTileGroupCount
          << " partial_overlap_edges="
          << candidate.evaluation.partialOverlapEdgeCount
          << " disjoint_edges=" << candidate.evaluation.disjointEdgeCount
          << " parallel_components="
          << candidate.evaluation.parallelComponentCount << " peer_fragments="
          << getPeerFragmentCount(candidate.assignment.mapping)
          << " layout_conversion=" << hasLayoutConversion(candidate)
          << " buffer_count="
          << static_cast<unsigned>(getMaximumBufferCount(candidate))
          << " buffered_edges=" << countBufferedLogicalEdges(candidate)
          << " temporal_wave_lb=" << candidate.evaluation.temporalWaveLowerBound
          << " output_tile_footprint_estimate="
          << candidate.evaluation.peakOutputTileFootprintEstimate
          << " gate=" << failureGate << " detail=" << failureReason << '\n';

      // SPM acceptance is owned exclusively by the fixed-capacity allocator
      // on actual Instr IR.  Its failure creates the next legal temporal
      // state in the same joint search; the estimate above merely orders that
      // recurrence. Every transition strictly decreases at least one finite
      // tile extent and exact-state deduplication prevents cycles. Production
      // bounds repeated feedback work below from the allocator's actual
      // conflict neighborhood and demand count; it does not use a shape
      // estimate, wall-clock timeout, or a predeclared semantic search cap.
      bool enqueuedAllocationFeedback = false;
      bool closedAtAcceptedLookaheadBoundary = false;
      size_t allocationFeedbackNeighborCount = 0;
      uint64_t allocationFeedbackDemandCount = 0;
      llvm::SmallVector<const SPMCapacityDemandEvidence *, 8>
          allocationFeedbackEndpointDemands;
      if (failureGate == "spm-allocation" && spmCapacityOverflow) {
        if (!spmCapacityDemands.empty()) {
          preferredSPMFailureProbeTileId =
              spmCapacityDemands.front().physicalTileId;
          allocationFeedbackDemandCount =
              llvm::max_element(spmCapacityDemands,
                                [](const SPMCapacityDemandEvidence &lhs,
                                   const SPMCapacityDemandEvidence &rhs) {
                                  return lhs.demandCount < rhs.demandCount;
                                })
                  ->demandCount;
        }
        llvm::SmallVector<const SPMCapacityDemandEvidence *, 8> orderedDemands;
        for (const SPMCapacityDemandEvidence &demand : spmCapacityDemands) {
          if (!demand.operationNode && !demand.outputIndex &&
              !demand.operandDemandNode)
            continue;
          const bool duplicate = llvm::any_of(
              orderedDemands, [&](const SPMCapacityDemandEvidence *existing) {
                // One causal coordinate can own several tied allocations of
                // different exact types. It still has only one adjacent
                // search state; the next allocator run will report whichever
                // allocation remains limiting.
                return existing->operationNode == demand.operationNode &&
                       existing->outputIndex == demand.outputIndex &&
                       existing->operandDemandNode == demand.operandDemandNode;
              });
          if (!duplicate)
            orderedDemands.push_back(&demand);
        }
        // Insert-at-front below makes the most causal operand-demand
        // transition the next common-search state.  Lower-priority operation
        // and output evidence is still explored when it produces a distinct
        // exact coordinate.
        llvm::stable_sort(orderedDemands, [](const auto *lhs, const auto *rhs) {
          auto priority = [](const SPMCapacityDemandEvidence *demand) {
            return demand->operandDemandNode ? 2 : demand->outputIndex ? 1 : 0;
          };
          return priority(lhs) < priority(rhs);
        });
        uint64_t exactConflictBytes = 0;
        uint64_t exactConflictDemandCount = 0;
        for (const SPMCapacityDemandEvidence *demand : orderedDemands) {
          exactConflictBytes =
              std::max(exactConflictBytes, demand->exactConflictBytes);
          exactConflictDemandCount = std::max(exactConflictDemandCount,
                                              demand->exactConflictDemandCount);
        }
        allocationFeedbackEndpointDemands.assign(orderedDemands.begin(),
                                                 orderedDemands.end());
        for (auto [demandIndex, demand] : llvm::enumerate(orderedDemands)) {
          WholeCardCandidate refined = candidate;
          refined.transition.parentSPMConflictBytes = exactConflictBytes;
          refined.transition.parentSPMConflictDemandCount =
              exactConflictDemandCount;
          refined.transition.parentSPMTotalDemandCount =
              allocationFeedbackDemandCount;
          mlir::Operation *demandOperation =
              demand->operandDemandNode &&
                      *demand->operandDemandNode < dag->getNodes().size()
                  ? dag->getNodes()[*demand->operandDemandNode].operation
              : demand->operationNode &&
                      *demand->operationNode < dag->getNodes().size()
                  ? dag->getNodes()[*demand->operationNode].operation
                  : nullptr;
          const bool repeatedUnchangedDemand =
              candidate.transition.previousSPMDemand &&
              candidate.transition.previousSPMDemand->matches(*demand);
          const bool inferredOperandDemand =
              !demand->operandDemandNode && !demand->outputIndex &&
              isLargerThanKnownOperationResultTile(candidate, demandOperation,
                                                   demand->bytes);
          mlir::Operation *refinementAnchor = demandOperation;
          const bool anchorIsOperandDemand =
              demand->operandDemandNode.has_value() || inferredOperandDemand ||
              repeatedUnchangedDemand;
          // Each tied maximum is independent allocator evidence.  Generate
          // every distinct adjacent causal state; exact-state memoization
          // merges equal temporal and edge-action transitions.
          mlir::Operation *refinedOperation = nullptr;
          size_t refinedNode = 0;
          unsigned refinedDimension = 0;
          int64_t previousExtent = 0;
          int64_t refinedExtent = 0;
          unsigned refinedEdgeGroups = 0;
          auto refineActualDemandCoordinate = [&]() {
            // An explicit producer window larger than the physical capacity
            // is independently impossible: changing the producer op's
            // temporal loop does not change strategy.producerSizes for
            // retained/recompute/spill/layout boundaries.  Consume that hard
            // allocator fact first by moving the affected edge to its legal
            // coupled traversal neighbor.  Model-scale decode otherwise
            // repeated the same 32 MiB demand across every op-tile
            // breakpoint before reaching this identical transition.
            const unsigned changed =
                refineOversizedExplicitProducerEdges(refined, refinementAnchor);
            refinedEdgeGroups += changed;
            if (changed != 0)
              return true;

            bool refinedCoordinate = false;
            if (demand->operandDemandNode) {
              refinedCoordinate = refineOperationTemporalVariantOnce(
                  refined, *outputDomains, *dag, refinementAnchor,
                  anchorIsOperandDemand, &refinedOperation, &refinedNode,
                  &refinedDimension, &previousExtent, &refinedExtent);
            } else if (demand->outputIndex) {
              refinedCoordinate = refineOutputTemporalVariantOnce(
                  refined, *outputDomains, *dag, *demand->outputIndex,
                  &refinedDimension, &previousExtent, &refinedExtent);
            } else {
              refinedCoordinate = refineOperationTemporalVariantOnce(
                  refined, *outputDomains, *dag,
                  dag->getNodes()[*demand->operationNode].operation,
                  /*preferredIsOperandDemand=*/anchorIsOperandDemand,
                  &refinedOperation, &refinedNode, &refinedDimension,
                  &previousExtent, &refinedExtent);
            }
            return refinedCoordinate;
          };
          bool refinedCausalCoordinate = refineActualDemandCoordinate();
          unsigned skippedKnownRejectedStates = 0;
          while (refinedCausalCoordinate &&
                 containsAttemptedExactState(refined)) {
            auto admittedExact = findAdmittedExactState(refined);
            if (!candidate.transition.allocationFeedbackLookahead &&
                orderedDemands.size() == 1 &&
                admittedExact != admittedTemporalRepresentatives.end() &&
                admittedExact->transition.allocationFeedbackLookahead &&
                admittedExact->transition.feedbackRootOrdinal ==
                    candidate.transition.feedbackRootOrdinal) {
              // A failed coarse state followed immediately by an admitted
              // lookahead is the exact coarsest legal boundary on this
              // recurrence.  The adjacent state was not skipped: it was
              // materialized above and failed in the real allocator.
              closedAtAcceptedLookaheadBoundary = true;
              refinedCausalCoordinate = false;
              break;
            }
            ++skippedKnownRejectedStates;
            refinedCausalCoordinate = refineActualDemandCoordinate();
          }
          if (refinedCausalCoordinate) {
            refined.transition.previousSPMDemand =
                getSPMDemandRelation(*demand);
            refined.transition.unchangedSPMDemandStreak =
                repeatedUnchangedDemand
                    ? candidate.transition.unchangedSPMDemandStreak + 1
                    : 0;
            const uint64_t refinedUnchangedDemandStreak =
                refined.transition.unchangedSPMDemandStreak;
            // Children of a lookahead probe remain on the fast probe lane;
            // they cannot close the ordinary adjacent family by themselves.
            refined.transition.allocationFeedbackLookahead =
                candidate.transition.allocationFeedbackLookahead;

            std::optional<WholeCardCandidate> lookahead;
            size_t lookaheadNode = 0;
            unsigned lookaheadDimension = 0;
            int64_t lookaheadPreviousExtent = 0;
            int64_t lookaheadRefinedExtent = 0;
            uint64_t lookaheadBreakpoints = 0;
            if (repeatedUnchangedDemand && refinedEdgeGroups == 0 &&
                skippedKnownRejectedStates == 0) {
              WholeCardCandidate probe = refined;
              uint64_t probeStride = 1;
              for (uint64_t level = 1; level < refinedUnchangedDemandStreak;
                   ++level)
                probeStride = saturatingMultiply(probeStride, 2);
              for (uint64_t step = 0; step < probeStride; ++step) {
                WholeCardCandidate trial = probe;
                bool advancedSameCoordinate = false;
                size_t trialNode = 0;
                unsigned trialDimension = 0;
                int64_t trialPreviousExtent = 0;
                int64_t trialRefinedExtent = 0;
                if (demand->outputIndex) {
                  advancedSameCoordinate = refineOutputTemporalVariantOnce(
                      trial, *outputDomains, *dag, *demand->outputIndex,
                      &trialDimension, &trialPreviousExtent,
                      &trialRefinedExtent);
                  advancedSameCoordinate = advancedSameCoordinate &&
                                           trialDimension == refinedDimension;
                } else if (refinedOperation) {
                  mlir::Operation *trialOperation = nullptr;
                  advancedSameCoordinate = refineOperationTemporalVariantOnce(
                      trial, *outputDomains, *dag, refinementAnchor,
                      anchorIsOperandDemand, &trialOperation, &trialNode,
                      &trialDimension, &trialPreviousExtent,
                      &trialRefinedExtent);
                  advancedSameCoordinate = advancedSameCoordinate &&
                                           trialOperation == refinedOperation &&
                                           trialNode == refinedNode &&
                                           trialDimension == refinedDimension;
                }
                if (!advancedSameCoordinate)
                  break;
                probe = std::move(trial);
                lookaheadNode = trialNode;
                lookaheadDimension = trialDimension;
                if (lookaheadBreakpoints == 0)
                  lookaheadPreviousExtent = trialPreviousExtent;
                lookaheadRefinedExtent = trialRefinedExtent;
                ++lookaheadBreakpoints;
              }
              if (lookaheadBreakpoints != 0 && !containsExactState(probe)) {
                probe.transition.previousSPMDemand =
                    getSPMDemandRelation(*demand);
                probe.transition.unchangedSPMDemandStreak =
                    refinedUnchangedDemandStreak;
                probe.transition.allocationFeedbackLookahead = true;
                lookahead.emplace(std::move(probe));
              }
            }
            refined.transition.stableOrdinal = nextFeedbackStableOrdinal++;
            bool promotedPendingState = false;
            auto pending = findPendingExactState(refined);
            if (pending != shortlist.end()) {
              mergeBetterSPMFeedbackPriority(*pending, refined);
              if (pending != shortlist.begin())
                std::rotate(shortlist.begin(), pending, std::next(pending));
              promotedPendingState = true;
            }
            std::string scheduleFailure;
            const bool refreshedSchedule = refreshCandidateResourceSchedule(
                refined, *dag, *topology, physicalCardId, *availableTileIds,
                &scheduleFailure, &resourceScheduleMemo);
            const bool duplicateExactState = containsExactState(refined);
            if (refreshedSchedule && !duplicateExactState) {
              const uint64_t nextOrdinal = refined.transition.stableOrdinal;
              shortlist.insert(shortlist.begin(), std::move(refined));
              ++resultStatistics.allocationFeedbackCandidates;
              ++resultStatistics.shortlistedCandidates;
              enqueuedAllocationFeedback = true;
              ++allocationFeedbackNeighborCount;
              llvm::SmallVector<llvm::StringRef, 4> demandOutgoingActions;
              llvm::SmallVector<llvm::StringRef, 4> demandIncomingActions;
              for (const SpatialEdgeStrategy &strategy :
                   candidate.assignment.mapping.edgeStrategies) {
                if (strategy.producer == demandOperation)
                  demandOutgoingActions.push_back(
                      getSpatialEdgeActionName(strategy.action));
                if (strategy.consumer == demandOperation)
                  demandIncomingActions.push_back(
                      getSpatialEdgeActionName(strategy.action));
              }
              diagnostics
                  << "wafer-compile: whole-card-allocation-feedback"
                  << " from=" << candidate.transition.stableOrdinal
                  << " next=" << nextOrdinal << " source=actual-spm-packing"
                  << " tied_demand=" << demandIndex + 1 << '/'
                  << orderedDemands.size() << " demand_bytes=" << demand->bytes
                  << " demand_count=" << demand->demandCount
                  << " physical_tile_id=" << demand->physicalTileId.getValue()
                  << " structured_relation=1 relation_kind="
                  << (demand->operandDemandNode
                          ? "operand-demand"
                          : (demand->outputIndex ? "output" : "operation"))
                  << " relation_source="
                  << (demand->relationFromAllocation ? "allocation"
                                                     : "operation-use")
                  << " operation_node="
                  << (demand->operationNode
                          ? static_cast<int64_t>(*demand->operationNode)
                          : int64_t{-1})
                  << " operand_demand_node="
                  << (demand->operandDemandNode
                          ? static_cast<int64_t>(*demand->operandDemandNode)
                          : int64_t{-1})
                  << " structured_node="
                  << (demand->operandDemandNode
                          ? static_cast<int64_t>(*demand->operandDemandNode)
                      : demand->operationNode
                          ? static_cast<int64_t>(*demand->operationNode)
                          : int64_t{-1})
                  << " output_index="
                  << (demand->outputIndex
                          ? static_cast<int64_t>(*demand->outputIndex)
                          : int64_t{-1})
                  << " repeated_unchanged_demand=" << repeatedUnchangedDemand
                  << " inferred_operand_demand=" << inferredOperandDemand
                  << " refinement_mode="
                  << (refinedEdgeGroups != 0
                          ? "edge-action"
                          : (anchorIsOperandDemand
                                 ? "direct"
                                 : (demand->outputIndex ? "output"
                                                        : "downstream")))
                  << " refined_edge_groups=" << refinedEdgeGroups
                  << " skipped_known_rejected_states="
                  << skippedKnownRejectedStates
                  << " promoted_pending_state=" << promotedPendingState
                  << " demand_incoming_actions=[";
              llvm::interleaveComma(demandIncomingActions, diagnostics);
              diagnostics << "]"
                          << " demand_outgoing_actions=[";
              llvm::interleaveComma(demandOutgoingActions, diagnostics);
              diagnostics << "] refined_operation="
                          << (refinedOperation
                                  ? refinedOperation->getName().getStringRef()
                                  : llvm::StringRef("none"))
                          << " refined_node=" << refinedNode
                          << " refined_dimension=" << refinedDimension
                          << " previous_extent=" << previousExtent
                          << " refined_extent=" << refinedExtent << '\n';

              if (lookahead) {
                lookahead->transition.stableOrdinal =
                    nextFeedbackStableOrdinal++;
                std::string lookaheadScheduleFailure;
                const bool refreshedLookaheadSchedule =
                    refreshCandidateResourceSchedule(
                        *lookahead, *dag, *topology, physicalCardId,
                        *availableTileIds, &lookaheadScheduleFailure,
                        &resourceScheduleMemo);
                const bool duplicateLookaheadState =
                    containsExactState(*lookahead);
                if (refreshedLookaheadSchedule && !duplicateLookaheadState) {
                  const uint64_t lookaheadOrdinal =
                      lookahead->transition.stableOrdinal;
                  shortlist.insert(shortlist.begin(), std::move(*lookahead));
                  ++resultStatistics.allocationFeedbackCandidates;
                  ++resultStatistics.allocationFeedbackLookaheadCandidates;
                  ++resultStatistics.shortlistedCandidates;
                  ++allocationFeedbackNeighborCount;
                  diagnostics
                      << "wafer-compile: whole-card-allocation-feedback-"
                         "lookahead"
                      << " from=" << candidate.transition.stableOrdinal
                      << " next=" << lookaheadOrdinal
                      << " source=actual-unchanged-spm-demand"
                      << " unchanged_streak=" << refinedUnchangedDemandStreak
                      << " probe_breakpoints=" << lookaheadBreakpoints
                      << " refined_node=" << lookaheadNode
                      << " refined_dimension=" << lookaheadDimension
                      << " previous_extent=" << lookaheadPreviousExtent
                      << " refined_extent=" << lookaheadRefinedExtent
                      << " adjacent_state_preserved=1\n";
                } else if (!refreshedLookaheadSchedule) {
                  ++resultStatistics.resourceScheduleRejections;
                  diagnostics
                      << "wafer-compile: whole-card-allocation-feedback-"
                         "lookahead-discarded"
                      << " from=" << candidate.transition.stableOrdinal
                      << " schedule_legal=0 detail="
                      << (lookaheadScheduleFailure.empty()
                              ? "none"
                              : lookaheadScheduleFailure)
                      << '\n';
                }
              }
            } else {
              if (!refreshedSchedule)
                ++resultStatistics.resourceScheduleRejections;
              diagnostics
                  << "wafer-compile: whole-card-allocation-feedback-discarded"
                  << " from=" << candidate.transition.stableOrdinal
                  << " tied_demand=" << demandIndex + 1 << '/'
                  << orderedDemands.size()
                  << " schedule_legal=" << refreshedSchedule
                  << " duplicate_exact_state=" << duplicateExactState
                  << " detail="
                  << (scheduleFailure.empty() ? "none" : scheduleFailure)
                  << '\n';
            }
          } else {
            diagnostics
                << "wafer-compile: whole-card-allocation-feedback-exhausted"
                << " from=" << candidate.transition.stableOrdinal
                << " tied_demand=" << demandIndex + 1 << '/'
                << orderedDemands.size() << " demand_bytes=" << demand->bytes
                << " demand_count=" << demand->demandCount
                << " physical_tile_id=" << demand->physicalTileId.getValue()
                << " skipped_attempted_states=" << skippedKnownRejectedStates
                << " anchor_operation="
                << (refinementAnchor
                        ? refinementAnchor->getName().getStringRef()
                        : llvm::StringRef("none"))
                << " anchor_is_operand_demand=" << anchorIsOperandDemand;
            auto anchorTile = llvm::find_if(
                candidate.assignment.mapping.operationTemporalTiles,
                [&](const StructuredOpTemporalTile &tile) {
                  return tile.operation == refinementAnchor;
                });
            if (anchorTile !=
                candidate.assignment.mapping.operationTemporalTiles.end()) {
              diagnostics << " anchor_tile=[";
              llvm::interleaveComma(anchorTile->iteratorTileSizes, diagnostics);
              diagnostics << "]";
              const size_t anchorNode = static_cast<size_t>(std::distance(
                  candidate.assignment.mapping.operationTemporalTiles.begin(),
                  anchorTile));
              if (anchorNode < candidate.assignment.nodePlacements.size()) {
                std::optional<llvm::SmallVector<int64_t, 4>> ranges =
                    getMaximumSpatialIteratorRanges(
                        anchorTile->operation,
                        candidate.assignment.nodePlacements[anchorNode]);
                if (ranges) {
                  diagnostics << " anchor_ranges=[";
                  llvm::interleaveComma(*ranges, diagnostics);
                  diagnostics << "] anchor_direct_axis=";
                  std::optional<unsigned> directAxis =
                      selectOperationTemporalRefinementAxis(
                          anchorTile->operation, *ranges,
                          anchorTile->iteratorTileSizes,
                          getDefaultWaferTargetPolicy().memory);
                  if (directAxis)
                    diagnostics << *directAxis;
                  else
                    diagnostics << "none";
                }
              }
            }
            diagnostics << " anchor_outgoing_actions=[";
            bool firstAction = true;
            for (const SpatialEdgeStrategy &strategy :
                 candidate.assignment.mapping.edgeStrategies) {
              if (strategy.producer != refinementAnchor)
                continue;
              if (!firstAction)
                diagnostics << ',';
              firstAction = false;
              diagnostics << getSpatialEdgeActionName(strategy.action) << ':';
              std::optional<uint64_t> elementBytes = getElementByteWidth(
                  strategy.producer->getResult(strategy.producerResult)
                      .getType());
              if (elementBytes && !strategy.producerSizes.empty())
                diagnostics << saturatingMultiply(
                    getElementProduct(strategy.producerSizes), *elementBytes);
              else
                diagnostics << "unknown";
            }
            diagnostics << ']';
            diagnostics << '\n';
          }
        }
        if (closedAtAcceptedLookaheadBoundary) {
          size_t retired = 0;
          for (auto pending = shortlist.begin(); pending != shortlist.end();) {
            if (pending->transition.feedbackRootOrdinal !=
                candidate.transition.feedbackRootOrdinal) {
              ++pending;
              continue;
            }
            seenExactStates.push_back(std::move(*pending));
            pending = shortlist.erase(pending);
            ++retired;
          }
          ++resultStatistics.allocationFeedbackLookaheadBoundaryClosures;
          resultStatistics.feedbackBeamDeferredCandidates += retired;
          diagnostics
              << "wafer-compile: whole-card-allocation-feedback-lookahead-"
                 "boundary"
              << " feedback_root=" << candidate.transition.feedbackRootOrdinal
              << " retired=" << retired
              << " source=exact-failed-adjacent-plus-admitted-lookahead\n";
        }
        // Retain every single-coordinate child above so the finite domain
        // stays complete.  A multi-demand conflict is composed progressively:
        // one measured-capacity probe is put on the fast lane, then the next
        // actual allocator result chooses another still-limiting coordinate.
        // Applying every independently necessary transition in one clone made
        // model-scale IR grow combinatorially, while a fixed batch size would
        // merely hide that semantic state.  Sequential conflict-directed
        // composition reaches the same joint states without materializing the
        // cross product before it is demanded.  The allocator remains the
        // sole capacity authority.
        if (orderedDemands.size() > 1) {
          auto getDemandOperation =
              [&](const SPMCapacityDemandEvidence *demand) {
                std::optional<CardDAGNodeID> node =
                    demand->operandDemandNode ? demand->operandDemandNode
                                              : demand->operationNode;
                return node && *node < dag->getNodes().size()
                           ? dag->getNodes()[*node].operation
                           : nullptr;
              };
          auto repeatsPreviousDemand = [&](const auto *demand) {
            return candidate.transition.previousSPMDemand &&
                   candidate.transition.previousSPMDemand->matches(*demand);
          };
          auto getDemandKey =
              [](const SPMCapacityDemandEvidence *demand) -> uint64_t {
            if (demand->operandDemandNode)
              return (uint64_t{1} << 32) | *demand->operandDemandNode;
            if (demand->outputIndex)
              return (uint64_t{2} << 32) | *demand->outputIndex;
            return (uint64_t{3} << 32) |
                   demand->operationNode.value_or(
                       std::numeric_limits<uint32_t>::max());
          };
          const bool hasUnseenDemand =
              llvm::any_of(orderedDemands, [&](const auto *demand) {
                return !llvm::is_contained(
                    candidate.transition.progressiveSPMDemandHistory,
                    getDemandKey(demand));
              });
          const SPMCapacityDemandEvidence *progressiveDemand =
              *llvm::max_element(
                  orderedDemands, [&](const auto *lhs, const auto *rhs) {
                    auto key = [&](const auto *demand) {
                      const unsigned specificity = demand->operandDemandNode ? 2
                                                   : demand->outputIndex     ? 1
                                                                         : 0;
                      const bool unseen = !llvm::is_contained(
                          candidate.transition.progressiveSPMDemandHistory,
                          getDemandKey(demand));
                      return std::tuple(!hasUnseenDemand || unseen,
                                        !repeatsPreviousDemand(demand),
                                        demand->bytes, specificity);
                    };
                    return key(lhs) < key(rhs);
                  });
          WholeCardCandidate progressive = candidate;
          progressive.transition.parentSPMConflictBytes = exactConflictBytes;
          progressive.transition.parentSPMConflictDemandCount =
              exactConflictDemandCount;
          progressive.transition.parentSPMTotalDemandCount =
              allocationFeedbackDemandCount;
          if (!hasUnseenDemand)
            progressive.transition.progressiveSPMDemandHistory.clear();
          progressive.transition.progressiveSPMDemandHistory.push_back(
              getDemandKey(progressiveDemand));
          uint64_t temporalTransitions = 0;
          uint64_t edgeTransitions = 0;
          const bool refined = refineAllocationDemandTowardCapacityProbe(
              progressive, *outputDomains, *dag, *progressiveDemand,
              &temporalTransitions, &edgeTransitions);
          if (refined) {
            const bool repeated = repeatsPreviousDemand(progressiveDemand);
            progressive.transition.previousSPMDemand =
                getSPMDemandRelation(*progressiveDemand);
            progressive.transition.unchangedSPMDemandStreak =
                repeated ? candidate.transition.unchangedSPMDemandStreak + 1
                         : 0;
            progressive.transition.allocationFeedbackLookahead = true;

            // Every causal-set probe also exists as one of the exact
            // single-coordinate children retained above.  Once those states
            // coincide, exact-state deduplication must promote and annotate
            // the already-pending child instead of silently leaving the last
            // (usually smallest) causal demand at the front.  Otherwise the
            // candidate queue can walk increasingly expanded temporal
            // compositions
            // while its exact packing conflict remains unchanged.  This
            // changes only candidate-priority order and traversal metadata: no
            // semantic state is removed, synthesized, or accepted without the
            // fixed-capacity allocator.
            auto pending = findPendingExactState(progressive);
            if (pending != shortlist.end()) {
              const uint64_t nextOrdinal = pending->transition.stableOrdinal;
              mergeBetterSPMFeedbackPriority(*pending, progressive);
              pending->transition.previousSPMDemand =
                  progressive.transition.previousSPMDemand;
              pending->transition.unchangedSPMDemandStreak =
                  progressive.transition.unchangedSPMDemandStreak;
              pending->transition.progressiveSPMDemandHistory =
                  progressive.transition.progressiveSPMDemandHistory;
              pending->transition.allocationFeedbackLookahead = true;
              const bool reordered = pending != shortlist.begin();
              if (reordered) {
                std::rotate(shortlist.begin(), pending, std::next(pending));
                ++resultStatistics.candidatePriorityReorders;
              }
              ++resultStatistics.allocationFeedbackProgressivePromotions;
              enqueuedAllocationFeedback = true;
              diagnostics << "wafer-compile: whole-card-allocation-feedback-"
                             "progressive-promoted"
                          << " from=" << candidate.transition.stableOrdinal
                          << " next=" << nextOrdinal
                          << " source=actual-spm-packing-causal-set"
                          << " causal_demands=" << orderedDemands.size()
                          << " selected_demand_bytes="
                          << progressiveDemand->bytes
                          << " repeated_selected_demand=" << repeated
                          << " temporal_transitions=" << temporalTransitions
                          << " edge_transitions=" << edgeTransitions
                          << " exact_pending_state=1"
                          << " candidate_queue_reordered=" << reordered
                          << " search_domain_preserved=1\n";
            } else if (!containsAttemptedExactState(progressive)) {
              progressive.transition.stableOrdinal =
                  nextFeedbackStableOrdinal++;
              std::string scheduleFailure;
              const bool refreshedSchedule = refreshCandidateResourceSchedule(
                  progressive, *dag, *topology, physicalCardId,
                  *availableTileIds, &scheduleFailure, &resourceScheduleMemo);
              if (refreshedSchedule) {
                const uint64_t nextOrdinal =
                    progressive.transition.stableOrdinal;
                shortlist.insert(shortlist.begin(), std::move(progressive));
                ++resultStatistics.allocationFeedbackCandidates;
                ++resultStatistics.shortlistedCandidates;
                enqueuedAllocationFeedback = true;
                ++allocationFeedbackNeighborCount;
                diagnostics
                    << "wafer-compile: "
                       "whole-card-allocation-feedback-progressive"
                    << " from=" << candidate.transition.stableOrdinal
                    << " next=" << nextOrdinal
                    << " source=actual-spm-packing-causal-set"
                    << " causal_demands=" << orderedDemands.size()
                    << " selected_demand_bytes=" << progressiveDemand->bytes
                    << " repeated_selected_demand=" << repeated
                    << " temporal_transitions=" << temporalTransitions
                    << " edge_transitions=" << edgeTransitions
                    << " capacity_check=1"
                    << " exact_resource_verification_required=1\n";
              } else {
                ++resultStatistics.resourceScheduleRejections;
                diagnostics
                    << "wafer-compile: whole-card-allocation-feedback-"
                       "progressive-discarded"
                    << " from=" << candidate.transition.stableOrdinal
                    << " schedule_legal=0 detail="
                    << (scheduleFailure.empty() ? "none" : scheduleFailure)
                    << '\n';
              }
            }
          }
        }
        ++resultStatistics.allocationFeedbackTransitions;
      }
      bool enqueuedBufferFeedback = false;
      if (failureGate == "selected-buffer-materialization" &&
          hasBufferedEdge(candidate) &&
          candidate.transition.bufferFeedbackBoundary &&
          (selectedBufferFailure.kind ==
               SelectedBufferMaterializationFailureKind::TripCountTooSmall ||
           selectedBufferFailure.kind ==
               SelectedBufferMaterializationFailureKind::NoExactLoop)) {
        std::vector<WholeCardCandidate> refinements =
            refineSelectedBufferConsumerWaveOnce(
                candidate, *dag, *outputDomains,
                selectedBufferFailure.consumerNode, nextFeedbackStableOrdinal);
        for (WholeCardCandidate &refined : llvm::reverse(refinements)) {
          std::string scheduleFailure;
          if (!refreshCandidateResourceSchedule(
                  refined, *dag, *topology, physicalCardId, *availableTileIds,
                  &scheduleFailure, &resourceScheduleMemo)) {
            ++resultStatistics.resourceScheduleRejections;
            continue;
          }
          if (containsExactState(refined))
            continue;
          ++resultStatistics.bufferFeedbackCandidates;
          ++resultStatistics.shortlistedCandidates;
          shortlist.insert(shortlist.begin(), std::move(refined));
          enqueuedBufferFeedback = true;
        }
        ++resultStatistics.bufferFeedbackTransitions;
        if (enqueuedBufferFeedback)
          diagnostics << "wafer-compile: whole-card-buffer-feedback"
                      << " from=" << candidate.transition.stableOrdinal
                      << " generated=" << refinements.size()
                      << " source=exact-edge-temporal-recurrence\n";
      }
      if (closedAtAcceptedLookaheadBoundary)
        continue;
      if (failureGate == "selected-buffer-materialization" &&
          hasBufferedEdge(candidate) &&
          !candidate.transition.bufferFeedbackBoundary) {
        std::vector<WholeCardCandidate> children;
        auto retainChild = [&](WholeCardCandidate child) {
          populateTemporalMetrics(child, *outputDomains, *dag);
          std::string scheduleFailure;
          if (!refreshCandidateResourceSchedule(
                  child, *dag, *topology, physicalCardId, *availableTileIds,
                  &scheduleFailure, &resourceScheduleMemo)) {
            ++resultStatistics.resourceScheduleRejections;
            return;
          }
          if (containsExactState(child) ||
              llvm::any_of(children, [&](const WholeCardCandidate &existing) {
                return sameNodePlacements(child, existing) &&
                       sameMapping(child, existing);
              }))
            return;
          children.push_back(std::move(child));
        };

        std::optional<WholeCardCandidate> fullTemporal = makeTemporalVariant(
            candidate, *outputDomains, *dag, nextFeedbackStableOrdinal++,
            /*additionalWaveRefinements=*/0);
        std::vector<WholeCardCandidate> actionBoundaries;
        if (fullTemporal) {
          fullTemporal->transition.bufferFeedbackBoundary = true;
          actionBoundaries.push_back(std::move(*fullTemporal));
        } else {
          WholeCardCandidate boundary = candidate;
          boundary.transition.stableOrdinal = nextFeedbackStableOrdinal++;
          boundary.transition.bufferFeedbackBoundary = true;
          actionBoundaries.push_back(std::move(boundary));
        }
        // The joint coordinate search already retains one exact witness for
        // every legal (edge action, buffer count) cell.  Feedback therefore
        // keeps the failed cell fixed and composes only the consumer's
        // observable temporal boundary.  Re-enumerating all actions/counts at
        // every actualization failure is duplicate work, not search coverage.
        for (const WholeCardCandidate &base : actionBoundaries) {
          const uint8_t requestedBuffers =
              selectedBufferFailure.bufferCount > 1
                  ? selectedBufferFailure.bufferCount
                  : getMaximumBufferCount(base);
          auto getMinimumViableBufferWave =
              [&](int64_t extent) -> std::optional<int64_t> {
            if (extent <= 1 || requestedBuffers <= 1)
              return std::nullopt;
            std::optional<int64_t> tailBearingFallback;
            for (int64_t value : getFeedbackWaveBreakpoints(extent))
              // Temporal lowering peels one full wave as the prologue.  The
              // remaining static scf.for must still contain enough
              // iterations to carry every selected slot through the steady
              // pipeline. Prefer an exact divisor: a nondivisible boundary
              // introduces an scf.if tail inside the owner loop, which the
              // exact one-loop software pipeline cannot legally reorder.
              if (ceilDivide(static_cast<uint64_t>(extent),
                             static_cast<uint64_t>(value)) > requestedBuffers) {
                if (extent % value == 0)
                  return value;
                if (!tailBearingFallback)
                  tailBearingFallback = value;
              }
            return tailBearingFallback;
          };
          for (size_t outputIndex = 0;
               outputIndex < base.assignment.mapping.outputs.size();
               ++outputIndex) {
            const unsigned resultIndex =
                base.assignment.mapping.outputs[outputIndex].outputIndex;
            if (!outputContainsBufferedConsumer(
                    base, *dag, resultIndex,
                    selectedBufferFailure.consumerNode))
              continue;
            llvm::SmallVector<int64_t, 4> ranges = getMaximumSpatialShardShape(
                base.assignment.mapping.outputs[outputIndex],
                (*outputDomains)[resultIndex]);
            for (size_t dimension = 0; dimension < ranges.size(); ++dimension) {
              std::optional<int64_t> value =
                  getMinimumViableBufferWave(ranges[dimension]);
              if (!value)
                continue;
              WholeCardCandidate child = base;
              child.transition.stableOrdinal = nextFeedbackStableOrdinal++;
              child.assignment.mapping.outputs[outputIndex]
                  .temporalTileSizes[dimension] = *value;
              retainChild(std::move(child));
            }
          }
        }
        auto actionPriority = [](const WholeCardCandidate &child) {
          if (hasLayoutConversion(child))
            return 0;
          if (llvm::any_of(child.assignment.mapping.edgeStrategies,
                           [](const SpatialEdgeStrategy &strategy) {
                             return strategy.action ==
                                    SpatialEdgeAction::LocalShardResidency;
                           }))
            return 1;
          return 2;
        };
        llvm::sort(children, [&](const WholeCardCandidate &lhs,
                                 const WholeCardCandidate &rhs) {
          const int lhsPriority = actionPriority(lhs);
          const int rhsPriority = actionPriority(rhs);
          if (lhsPriority != rhsPriority)
            return lhsPriority < rhsPriority;
          return cheapCandidateLess(lhs, rhs);
        });
        for (WholeCardCandidate &child : llvm::reverse(children)) {
          if (hasLayoutBufferedEdge(child))
            ++resultStatistics.layoutBufferedCandidateProposals;
          ++resultStatistics.bufferedCandidateProposals;
          ++resultStatistics.bufferFeedbackCandidates;
          ++resultStatistics.shortlistedCandidates;
          shortlist.insert(shortlist.begin(), std::move(child));
          enqueuedBufferFeedback = true;
        }
        ++resultStatistics.bufferFeedbackTransitions;
        if (enqueuedBufferFeedback)
          diagnostics << "wafer-compile: whole-card-buffer-feedback"
                      << " from=" << candidate.transition.stableOrdinal
                      << " generated=" << children.size()
                      << " source=actual-buffer-materialization\n";
      }
      // A single-causal-coordinate recurrence may use measured lookahead and
      // root closure: all of its finite states are linearly ordered.  A
      // multi-coordinate conflict must stay in the common candidate queue;
      // dropping siblings or replacing them with one simultaneous endpoint
      // would remove legal compositions, and model-scale IR showed that the
      // simultaneous endpoint can construct the full temporal cross product
      // before the allocator observes it.
      if (enqueuedAllocationFeedback && allocationFeedbackNeighborCount != 0 &&
          allocationFeedbackEndpointDemands.size() == 1) {
        size_t &beamWidth =
            allocationFeedbackBeamWidths[candidate.transition
                                             .feedbackRootOrdinal];
        beamWidth = std::max(beamWidth, allocationFeedbackNeighborCount);
        FeedbackRootDemandWork &demandWork =
            feedbackRootDemandWork[candidate.transition.feedbackRootOrdinal];
        if (demandWork.initialNeighborCount == 0) {
          demandWork.initialNeighborCount = allocationFeedbackNeighborCount;
          demandWork.initialDemandCount = allocationFeedbackDemandCount;
        }
        size_t retained = 0;
        size_t deferred = 0;
        for (auto pending = shortlist.begin(); pending != shortlist.end();) {
          if (pending->transition.feedbackRootOrdinal !=
              candidate.transition.feedbackRootOrdinal) {
            ++pending;
            continue;
          }
          if (retained++ < beamWidth) {
            ++pending;
            continue;
          }
          seenExactStates.push_back(std::move(*pending));
          pending = shortlist.erase(pending);
          ++deferred;
        }
        if (deferred != 0) {
          resultStatistics.feedbackBeamDeferredCandidates += deferred;
          diagnostics
              << "wafer-compile: whole-card-candidate-priority"
              << " feedback_root=" << candidate.transition.feedbackRootOrdinal
              << " adaptive_conflict_beam=" << beamWidth
              << " exact_neighbors=" << allocationFeedbackNeighborCount
              << " current_exact_demand_count=" << allocationFeedbackDemandCount
              << " initial_exact_demand_count=" << demandWork.initialDemandCount
              << " retained=" << std::min(retained, beamWidth)
              << " deferred=" << deferred << " source=actual-spm-conflict\n";
        }
        const uint64_t expanded =
            feedbackRootMaterializations[candidate.transition
                                             .feedbackRootOrdinal];
        const uint64_t rootWorkBudget = std::max<uint64_t>(
            saturatingMultiply(demandWork.initialNeighborCount, 2),
            availableTileIds->size());
        const bool candidateBudgetReached = expanded >= rootWorkBudget;
        // Model-scale decode showed that a root whose exact allocation count
        // grows by almost six times while temporal extents shrink spends most
        // of its work rebuilding deeper loop/materialization structure rather
        // than approaching packing legality. Scale the measured amplification
        // guard with the square root of the physical Tile domain: this keeps
        // it query-local (4x on the current 16-Tile card), lets several exact
        // conflict-set reductions run, and switches to the other spatial,
        // edge, layout and buffer seeds before one temporal family monopolizes
        // the common search.
        uint64_t demandAmplificationFactor = 2;
        while (saturatingMultiply(demandAmplificationFactor + 1,
                                  demandAmplificationFactor + 1) <=
               availableTileIds->size())
          ++demandAmplificationFactor;
        const uint64_t demandAmplificationLimit = saturatingMultiply(
            demandWork.initialDemandCount, demandAmplificationFactor);
        const bool demandAmplificationReached =
            expanded >= 2 && demandAmplificationLimit != 0 &&
            allocationFeedbackDemandCount > demandAmplificationLimit;
        if (candidateBudgetReached || demandAmplificationReached) {
          // The measured root is no longer worth walking one adjacent state
          // at a time, but that is not evidence that its spatial/layout/
          // edge/buffer cell is infeasible. Preserve one exact feasibility
          // anchor by composing a measured-capacity probe only for temporal
          // coordinates named by the current allocator conflict set.
          // Refining every operation to one created hundreds of unrelated
          // loops, while one adjacent transition per 30x oversized allocation
          // repeated the same expensive compiler gates. The probe chooses an
          // intermediate exact state; it decides no capacity fact and wins
          // only on admitted actual cost.
          std::optional<WholeCardCandidate> temporalEndpoint;
          uint64_t endpointTransitions = 0;
          uint64_t endpointEdgeTransitions = 0;
          WholeCardCandidate endpoint = candidate;
          endpoint.transition.parentSPMConflictBytes =
              allocationFeedbackEndpointDemands.front()->exactConflictBytes;
          endpoint.transition.parentSPMConflictDemandCount =
              allocationFeedbackEndpointDemands.front()
                  ->exactConflictDemandCount;
          endpoint.transition.parentSPMTotalDemandCount =
              allocationFeedbackDemandCount;
          for (const SPMCapacityDemandEvidence *demand :
               allocationFeedbackEndpointDemands) {
            refineAllocationDemandTowardCapacityProbe(
                endpoint, *outputDomains, *dag, *demand, &endpointTransitions,
                &endpointEdgeTransitions);
          }
          if ((endpointTransitions != 0 || endpointEdgeTransitions != 0) &&
              !containsAttemptedExactState(endpoint)) {
            auto pendingEndpoint = findPendingExactState(endpoint);
            if (pendingEndpoint != shortlist.end())
              shortlist.erase(pendingEndpoint);
            if (allocationFeedbackEndpointDemands.size() == 1) {
              const SPMCapacityDemandEvidence &demand =
                  *allocationFeedbackEndpointDemands.front();
              const bool repeatedUnchangedDemand =
                  candidate.transition.previousSPMDemand &&
                  candidate.transition.previousSPMDemand->matches(demand);
              endpoint.transition.previousSPMDemand =
                  getSPMDemandRelation(demand);
              endpoint.transition.unchangedSPMDemandStreak =
                  repeatedUnchangedDemand
                      ? candidate.transition.unchangedSPMDemandStreak + 1
                      : 0;
            } else {
              endpoint.transition.previousSPMDemand.reset();
              endpoint.transition.unchangedSPMDemandStreak = 0;
            }
            endpoint.transition.allocationFeedbackLookahead = true;
            endpoint.transition.stableOrdinal = nextFeedbackStableOrdinal++;
            std::string endpointScheduleFailure;
            if (refreshCandidateResourceSchedule(
                    endpoint, *dag, *topology, physicalCardId,
                    *availableTileIds, &endpointScheduleFailure,
                    &resourceScheduleMemo)) {
              temporalEndpoint.emplace(std::move(endpoint));
            } else {
              ++resultStatistics.resourceScheduleRejections;
              diagnostics
                  << "wafer-compile: whole-card-allocation-feedback-endpoint-"
                     "discarded"
                  << " feedback_root="
                  << candidate.transition.feedbackRootOrdinal
                  << " temporal_transitions=" << endpointTransitions
                  << " schedule_legal=0 detail="
                  << (endpointScheduleFailure.empty() ? "none"
                                                      : endpointScheduleFailure)
                  << " source=actual-feedback-root-closure\n";
            }
          }
          size_t budgetDeferred = 0;
          for (auto pending = shortlist.begin(); pending != shortlist.end();) {
            if (pending->transition.feedbackRootOrdinal !=
                candidate.transition.feedbackRootOrdinal) {
              ++pending;
              continue;
            }
            seenExactStates.push_back(std::move(*pending));
            pending = shortlist.erase(pending);
            ++budgetDeferred;
          }
          if (temporalEndpoint) {
            const uint64_t endpointOrdinal =
                temporalEndpoint->transition.stableOrdinal;
            shortlist.insert(shortlist.begin(), std::move(*temporalEndpoint));
            ++resultStatistics.allocationFeedbackCandidates;
            ++resultStatistics.allocationFeedbackEndpointCandidates;
            ++resultStatistics.shortlistedCandidates;
            diagnostics
                << "wafer-compile: whole-card-allocation-feedback-endpoint"
                << " feedback_root=" << candidate.transition.feedbackRootOrdinal
                << " next=" << endpointOrdinal
                << " temporal_transitions=" << endpointTransitions
                << " edge_transitions=" << endpointEdgeTransitions
                << " conflict_temporal_endpoint=1"
                << " conflict_relations="
                << allocationFeedbackEndpointDemands.size()
                << " non_temporal_state_preserved=1"
                << " exact_resource_verification_required=1"
                << " source=actual-feedback-root-closure\n";
          }
          if (budgetDeferred != 0 || temporalEndpoint) {
            resultStatistics.feedbackBeamDeferredCandidates += budgetDeferred;
            ++resultStatistics.feedbackRootBudgetClosures;
            diagnostics << "wafer-compile: whole-card-candidate-priority"
                        << " feedback_root="
                        << candidate.transition.feedbackRootOrdinal
                        << " adaptive_root_work_budget=" << rootWorkBudget
                        << " expanded=" << expanded
                        << " exact_demand_amplification_limit="
                        << demandAmplificationLimit
                        << " current_exact_demand_count="
                        << allocationFeedbackDemandCount
                        << " initial_exact_demand_count="
                        << demandWork.initialDemandCount
                        << " initial_exact_neighbors="
                        << demandWork.initialNeighborCount << " closure_reason="
                        << (demandAmplificationReached ? "demand-amplification"
                                                       : "candidate-work")
                        << " deferred=" << budgetDeferred
                        << " source=actual-spm-conflict\n";
          }
        }
      }
      if (enqueuedAllocationFeedback && !shortlist.empty()) {
        // Children of a newly worse allocator result must not hide retained
        // siblings whose parent had a cheaper measured path to legality.
        // Exact conflict bytes first form capacity-sized units. Above two
        // units, keep the conflict-directed progressive lane in front so it
        // crosses large capacity gaps instead of sweeping every sibling.
        // Inside the final two-unit class every failure is one capacity-scale
        // transition from legality, so prefer the smaller actual allocation
        // problem before the progressive lane and exact-byte tie-breaks. This
        // avoids trading a tiny byte improvement for thousands of repeated
        // loop demands. Stable sort preserves order for equal evidence.
        // Unknown initial states remain after measured feedback states, but
        // every state stays in the same common candidate queue.
        const uint64_t previousFront =
            shortlist.front().transition.stableOrdinal;
        auto getCapacityUnits = [&](const WholeCardCandidate &state) {
          return ceilDivide(state.transition.parentSPMConflictBytes,
                            usableSPMCapacity);
        };
        llvm::stable_sort(shortlist, [&](const WholeCardCandidate &lhs,
                                         const WholeCardCandidate &rhs) {
          const bool lhsMeasured = lhs.transition.parentSPMConflictBytes != 0;
          const bool rhsMeasured = rhs.transition.parentSPMConflictBytes != 0;
          if (lhsMeasured != rhsMeasured)
            return lhsMeasured;
          if (!lhsMeasured)
            return false;
          const uint64_t lhsUnits = getCapacityUnits(lhs);
          const uint64_t rhsUnits = getCapacityUnits(rhs);
          if (lhsUnits != rhsUnits)
            return lhsUnits < rhsUnits;
          if (lhsUnits > 2) {
            if (lhs.transition.allocationFeedbackLookahead !=
                rhs.transition.allocationFeedbackLookahead)
              return lhs.transition.allocationFeedbackLookahead;
            return std::tuple(lhs.transition.parentSPMConflictBytes,
                              lhs.transition.parentSPMTotalDemandCount,
                              lhs.transition.parentSPMConflictDemandCount) <
                   std::tuple(rhs.transition.parentSPMConflictBytes,
                              rhs.transition.parentSPMTotalDemandCount,
                              rhs.transition.parentSPMConflictDemandCount);
          }
          return std::tuple(lhs.transition.parentSPMTotalDemandCount,
                            !lhs.transition.allocationFeedbackLookahead,
                            lhs.transition.parentSPMConflictBytes,
                            lhs.transition.parentSPMConflictDemandCount) <
                 std::tuple(rhs.transition.parentSPMTotalDemandCount,
                            !rhs.transition.allocationFeedbackLookahead,
                            rhs.transition.parentSPMConflictBytes,
                            rhs.transition.parentSPMConflictDemandCount);
        });
        const bool reordered =
            shortlist.front().transition.stableOrdinal != previousFront;
        if (reordered)
          ++resultStatistics.candidatePriorityReorders;
        ++resultStatistics.allocationFeedbackPrioritySelections;
        diagnostics << "wafer-compile: whole-card-allocation-feedback-priority"
                    << " from=" << candidate.transition.stableOrdinal
                    << " next=" << shortlist.front().transition.stableOrdinal
                    << " exact_conflict_bytes="
                    << shortlist.front().transition.parentSPMConflictBytes
                    << " capacity_units=" << getCapacityUnits(shortlist.front())
                    << " priority_mode="
                    << (getCapacityUnits(shortlist.front()) > 2
                            ? "progressive-gap"
                            : "near-capacity-work")
                    << " exact_conflict_demands="
                    << shortlist.front().transition.parentSPMConflictDemandCount
                    << " total_demands="
                    << shortlist.front().transition.parentSPMTotalDemandCount
                    << " candidate_queue_reordered=" << reordered
                    << " search_domain_preserved=1\n";
      }
      if (!enqueuedAllocationFeedback && !enqueuedBufferFeedback &&
          shortlist.size() > 1) {
        const bool preserveMultiStageAxis =
            candidate.transition.nodePlacementCandidate &&
            candidate.evaluation.distinctTileGroupCount >= 3;
        const bool preserveIndependentAxis =
            candidate.transition.nodePlacementCandidate &&
            candidate.evaluation.parallelComponentCount > 1;
        const bool preserveLayoutAxis =
            hasLayoutConversion(candidate) &&
            failureGate != "selected-layout-materialization";
        // Buffer multiplicity is an optional overlap axis.  A failure before
        // or during its actualization must keep the structural spatial state
        // but immediately expose the corresponding single-buffer sibling;
        // otherwise a failed buffered representative can consume the entire
        // actual-work budget on buffer-count variants of one placement.
        const bool preserveBufferAxis = false;
        auto sameSpatialState = [&](const WholeCardCandidate &alternative) {
          if (alternative.assignment.nodePlacements.size() !=
                  candidate.assignment.nodePlacements.size() ||
              alternative.assignment.mapping.outputs.size() !=
                  candidate.assignment.mapping.outputs.size())
            return false;
          const bool sameNodes = llvm::all_of(
              llvm::zip_equal(alternative.assignment.nodePlacements,
                              candidate.assignment.nodePlacements),
              [](auto values) {
                const auto &[lhs, rhs] = values;
                return lhs.node == rhs.node &&
                       lhs.shardDimension == rhs.shardDimension &&
                       lhs.tiles == rhs.tiles &&
                       lhs.spatialIteratorDimension ==
                           rhs.spatialIteratorDimension &&
                       lhs.iteratorPartitionFactors ==
                           rhs.iteratorPartitionFactors;
              });
          const bool sameOutputs = llvm::all_of(
              llvm::zip_equal(alternative.assignment.mapping.outputs,
                              candidate.assignment.mapping.outputs),
              [](auto values) {
                const auto &[lhs, rhs] = values;
                return lhs.outputIndex == rhs.outputIndex &&
                       lhs.shardDimension == rhs.shardDimension &&
                       lhs.activeTileIds == rhs.activeTileIds;
              });
          return sameNodes && sameOutputs;
        };
        auto sameFailedAxes = [&](const WholeCardCandidate &alternative) {
          return (!preserveMultiStageAxis ||
                  (alternative.transition.nodePlacementCandidate &&
                   alternative.evaluation.distinctTileGroupCount >= 3)) &&
                 (!preserveIndependentAxis ||
                  (alternative.transition.nodePlacementCandidate &&
                   alternative.evaluation.parallelComponentCount > 1)) &&
                 (!preserveLayoutAxis || hasLayoutConversion(alternative)) &&
                 (!preserveBufferAxis || hasBufferedEdge(alternative));
        };
        if (failureGate == "spm-allocation") {
          llvm::stable_sort(shortlist, [&](const WholeCardCandidate &lhs,
                                           const WholeCardCandidate &rhs) {
            if (sameSpatialState(lhs) != sameSpatialState(rhs))
              return sameSpatialState(lhs);
            if (sameFailedAxes(lhs) != sameFailedAxes(rhs))
              return sameFailedAxes(lhs);
            // Best-first within the allocator-directed family: estimated fit
            // is considered before work, waves and movement by
            // cheapCandidateLess.  The estimate only orders exact attempts;
            // the fixed-capacity allocator remains the acceptance authority.
            return cheapCandidateLess(lhs, rhs);
          });
        } else if (failureGate == "tile-region-to-instr" ||
                   failureGate == "card-program-materialization" ||
                   failureGate == "selected-buffer-materialization" ||
                   failureGate == "selected-layout-materialization") {
          llvm::stable_sort(shortlist, [&](const WholeCardCandidate &lhs,
                                           const WholeCardCandidate &rhs) {
            if (sameFailedAxes(lhs) != sameFailedAxes(rhs))
              return sameFailedAxes(lhs);
            if (sameSpatialState(lhs) != sameSpatialState(rhs))
              return sameSpatialState(lhs);
            return cheapCandidateLess(lhs, rhs);
          });
        }
        auto sameTemporalState = [&](const WholeCardCandidate &alternative) {
          return alternative.assignment.mapping.operationTemporalTiles.size() ==
                     candidate.assignment.mapping.operationTemporalTiles
                         .size() &&
                 llvm::all_of(
                     llvm::zip_equal(
                         alternative.assignment.mapping.operationTemporalTiles,
                         candidate.assignment.mapping.operationTemporalTiles),
                     [](auto values) {
                       const auto &[lhs, rhs] = values;
                       return lhs.operation == rhs.operation &&
                              lhs.iteratorTileSizes == rhs.iteratorTileSizes;
                     });
        };
        // First feedback transition removes only the failed buffering choice
        // while holding spatial and temporal decisions fixed.  If that exact
        // sibling is absent, fall back to the least-refined single-buffer
        // state of the same placement before exploring another placement.
        bool promotedBufferedTemporalSibling = false;
        if (failureGate == "selected-buffer-materialization" &&
            hasBufferedEdge(candidate)) {
          auto deeper = shortlist.end();
          for (auto alternative = shortlist.begin();
               alternative != shortlist.end(); ++alternative) {
            if (!sameSpatialState(*alternative) ||
                !sameFailedAxes(*alternative) ||
                getMaximumBufferCount(*alternative) !=
                    getMaximumBufferCount(candidate) ||
                alternative->evaluation.temporalWaveLowerBound <=
                    candidate.evaluation.temporalWaveLowerBound)
              continue;
            if (deeper == shortlist.end() ||
                std::tuple(alternative->evaluation.temporalWaveLowerBound,
                           alternative->transition.stableOrdinal) <
                    std::tuple(deeper->evaluation.temporalWaveLowerBound,
                               deeper->transition.stableOrdinal))
              deeper = alternative;
          }
          if (deeper != shortlist.end()) {
            if (deeper != shortlist.begin())
              std::rotate(shortlist.begin(), deeper, std::next(deeper));
            promotedBufferedTemporalSibling = true;
          }
        }
        if (!promotedBufferedTemporalSibling) {
          auto singleBufferSibling = llvm::find_if(
              shortlist, [&](const WholeCardCandidate &alternative) {
                return sameSpatialState(alternative) &&
                       sameFailedAxes(alternative) &&
                       !hasBufferedEdge(alternative) &&
                       (!hasBufferedEdge(candidate) ||
                        sameTemporalState(alternative));
              });
          if (singleBufferSibling == shortlist.end())
            singleBufferSibling = llvm::find_if(
                shortlist, [&](const WholeCardCandidate &alternative) {
                  return sameSpatialState(alternative) &&
                         sameFailedAxes(alternative) &&
                         !hasBufferedEdge(alternative);
                });
          if (singleBufferSibling != shortlist.end() &&
              singleBufferSibling != shortlist.begin())
            std::rotate(shortlist.begin(), singleBufferSibling,
                        std::next(singleBufferSibling));
        }
        if (!shortlist.empty())
          diagnostics
              << "wafer-compile: whole-card-candidate-priority feedback_from="
              << candidate.transition.stableOrdinal
              << " next=" << shortlist.front().transition.stableOrdinal
              << " preserve_layout=" << preserveLayoutAxis << '\n';
        ++resultStatistics.candidatePriorityReorders;
      }
      continue;
    }
    if (candidate.transition.nodePlacementCandidate)
      ++resultStatistics.nodePlacementCandidateAcceptances;
    if (candidate.transition.nodePlacementCandidate &&
        candidate.evaluation.distinctTileGroupCount >= 3)
      ++resultStatistics.multiStagePlacementCandidateAcceptances;
    if (candidate.transition.nodePlacementCandidate &&
        candidate.evaluation.parallelComponentCount > 1)
      ++resultStatistics.independentComponentCandidateAcceptances;
    if (hasAlternativeEdgeAction(candidate))
      ++resultStatistics.alternativeEdgeActionCandidateAcceptances;
    if (hasLayoutAssignment(candidate))
      ++resultStatistics.layoutAssignedCandidateAcceptances;
    if (hasLayoutConversion(candidate))
      ++resultStatistics.layoutConversionCandidateAcceptances;
    if (hasLayoutBufferedEdge(candidate))
      ++resultStatistics.layoutBufferedCandidateAcceptances;
    if (hasBufferedEdge(candidate))
      ++resultStatistics.bufferedCandidateAcceptances;
    if (candidateActualFusedLogicalEdges != 0)
      ++resultStatistics.actualFusionCandidateAcceptances;
    auto admittedCandidate = std::make_unique<AdmittedWholeCardCandidate>(
        std::move(candidate), std::move(*accepted),
        std::move(acceptedOperationNodes), candidateActualFusedLogicalEdges);
    failureReason.clear();
    mlir::FailureOr<analysis::StaticSchedulePlan> plan =
        buildAcceptedWholeDAGSchedulePlan(
            *dag, admittedCandidate->candidate.assignment.nodePlacements,
            admittedCandidate->operationNodes, admittedCandidate->executable,
            admittedCandidate->phaseCosts, &failureReason);
    if (mlir::failed(plan)) {
      ++resultStatistics.schedulePlanRejections;
      diagnostics << "wafer-compile: whole-card-candidate rejection"
                  << " stable_ordinal="
                  << admittedCandidate->candidate.transition.stableOrdinal
                  << " gate=accepted-static-schedule-plan"
                  << " detail=" << failureReason << '\n';
    } else {
      admittedCandidate->schedulePlan.emplace(std::move(*plan));
      ++resultStatistics.plannedCandidates;
      admittedTemporalRepresentatives.push_back(admittedCandidate->candidate);
      if (admittedCandidate->candidate.transition.allocationFeedbackLookahead) {
        // The lookahead established a legal lower endpoint.  Test the
        // retained ordinary adjacent state next instead of draining deeper
        // speculative probes.  If that adjacent state fails, the exact-state
        // check above recognizes the admitted endpoint and closes the
        // coarsest legal boundary; if it succeeds, it becomes the better
        // ordinary incumbent.
        auto adjacent =
            llvm::find_if(shortlist, [&](const WholeCardCandidate &pending) {
              return pending.transition.feedbackRootOrdinal ==
                         admittedCandidate->candidate.transition
                             .feedbackRootOrdinal &&
                     !pending.transition.allocationFeedbackLookahead;
            });
        if (adjacent != shortlist.end()) {
          if (adjacent != shortlist.begin())
            std::rotate(shortlist.begin(), adjacent, std::next(adjacent));
          ++resultStatistics.candidatePriorityReorders;
          diagnostics
              << "wafer-compile: whole-card-candidate-priority feedback_root="
              << admittedCandidate->candidate.transition.feedbackRootOrdinal
              << " next=" << shortlist.front().transition.stableOrdinal
              << " source=admitted-lookahead-adjacent-boundary\n";
        }
      }
      diagnostics
          << "wafer-compile: whole-card-candidate admitted"
          << " stable_ordinal="
          << admittedCandidate->candidate.transition.stableOrdinal
          << " feedback_root="
          << admittedCandidate->candidate.transition.feedbackRootOrdinal
          << " actual_fused_edges="
          << admittedCandidate->actualFusedLogicalEdges << " temporal_wave_lb="
          << admittedCandidate->candidate.evaluation.temporalWaveLowerBound
          << " allocation_feedback_lookahead="
          << admittedCandidate->candidate.transition.allocationFeedbackLookahead
          << " non_temporal_family_closed="
          << !admittedCandidate->candidate.transition
                  .allocationFeedbackLookahead
          << '\n';
    }
    // Search retains several comparison summaries, so release their large
    // Tile modules and rematerialize only the selected state below.
    admittedCandidate->executable.tiles.clear();
    admitted.push_back(std::move(admittedCandidate));
  }
  resultStatistics.acceptedCandidates = admitted.size();
  if (admitted.empty()) {
    diagnostics << "wafer-compile: whole-card search has no exact-admitted "
                   "candidate\n";
    return mlir::failure();
  }

  llvm::SmallVector<size_t, 4> selectableIndices;
  llvm::SmallVector<const analysis::StaticSchedulePlan *, 4> plans;
  selectableIndices.reserve(admitted.size());
  plans.reserve(admitted.size());
  for (auto [index, candidate] : llvm::enumerate(admitted)) {
    if (candidate->schedulePlan &&
        (resultStatistics.applicableFusionLogicalEdges == 0 ||
         candidate->actualFusedLogicalEdges != 0)) {
      selectableIndices.push_back(index);
      plans.push_back(&*candidate->schedulePlan);
    }
  }
  if (selectableIndices.empty()) {
    diagnostics << "wafer-compile: whole-card search has no candidate with "
                   "an exact accepted schedule plan\n";
    return mlir::failure();
  }
  llvm::SmallVector<analysis::WholeCardResourceDurationEstimate, 16> estimates =
      analysis::estimateStaticSchedulePlanDurations(
          plans, analysis::getTargetScheduleCostPolicy());
  if (estimates.size() != selectableIndices.size()) {
    diagnostics << "wafer-compile: theoretical accepted schedule-plan cohort "
                   "estimation failed\n";
    return mlir::failure();
  }

  size_t selectedEstimateIndex = 0;
  for (size_t index = 1; index < selectableIndices.size(); ++index)
    if (selectionLess(*admitted[selectableIndices[index]], estimates[index],
                      *admitted[selectableIndices[selectedEstimateIndex]],
                      estimates[selectedEstimateIndex]))
      selectedEstimateIndex = index;
  const size_t selectedIndex = selectableIndices[selectedEstimateIndex];
  resultStatistics.selectedStableOrdinal =
      admitted[selectedIndex]->candidate.transition.stableOrdinal;
  resultStatistics.selectedOutputMappingCount =
      admitted[selectedIndex]->candidate.assignment.mapping.outputs.size();
  resultStatistics.selectedUniqueActiveTileCount = getUniqueActiveTileCount(
      admitted[selectedIndex]->candidate.assignment.mapping,
      admitted[selectedIndex]->candidate.assignment.nodePlacements);
  resultStatistics.selectedParallelComponentCount =
      admitted[selectedIndex]->candidate.evaluation.parallelComponentCount;
  resultStatistics.selectedTemporalWaveLowerBound =
      admitted[selectedIndex]->candidate.evaluation.temporalWaveLowerBound;
  resultStatistics.selectedInstructionExecutionLowerBound =
      admitted[selectedIndex]
          ->candidate.evaluation.instructionExecutionLowerBound;
  resultStatistics.selectedPeakOutputTileFootprintEstimate =
      admitted[selectedIndex]
          ->candidate.evaluation.peakOutputTileFootprintEstimate;
  resultStatistics.selectedPeakAlignedResidencyEstimate =
      admitted[selectedIndex]
          ->candidate.evaluation.peakAlignedResidencyEstimate;
  resultStatistics.selectedPeerTransferCount = getPeerFragmentCount(
      admitted[selectedIndex]->candidate.assignment.mapping);
  resultStatistics.selectedPeerBytes =
      admitted[selectedIndex]->candidate.evaluation.peerBytes;
  resultStatistics.selectedBufferCount =
      getMaximumBufferCount(admitted[selectedIndex]->candidate);
  resultStatistics.selectedActualFusedLogicalEdges =
      admitted[selectedIndex]->actualFusedLogicalEdges;
  resultStatistics.selectedSPMMovementWork =
      admitted[selectedIndex]->candidate.evaluation.scheduledSPMMovementWork;
  resultStatistics.selectedDDRMovementWork =
      admitted[selectedIndex]->candidate.evaluation.scheduledDDRMovementWork;
  resultStatistics.selectedMakespanPicoseconds =
      estimates[selectedEstimateIndex].makespan.picoseconds;
  resultStatistics.enabledDurationTerms =
      estimates[selectedEstimateIndex].enabledTerms;
  std::tie(resultStatistics.resourceScheduleMemoHits,
           resultStatistics.resourceScheduleMemoMisses) =
      resourceScheduleMemo.getCounts();

  diagnostics
      << "wafer-compile: whole-card-selection"
      << " admitted=" << admitted.size()
      << " planned=" << resultStatistics.plannedCandidates
      << " plan_rejections=" << resultStatistics.schedulePlanRejections
      << " multi_stage_placement_acceptances="
      << resultStatistics.multiStagePlacementCandidateAcceptances
      << " independent_component_acceptances="
      << resultStatistics.independentComponentCandidateAcceptances
      << " layout_assigned_acceptances="
      << resultStatistics.layoutAssignedCandidateAcceptances
      << " layout_conversion_acceptances="
      << resultStatistics.layoutConversionCandidateAcceptances
      << " layout_buffered_acceptances="
      << resultStatistics.layoutBufferedCandidateAcceptances
      << " buffered_acceptances="
      << resultStatistics.bufferedCandidateAcceptances
      << " actual_fusion_acceptances="
      << resultStatistics.actualFusionCandidateAcceptances
      << " spm_probe_attempts=" << resultStatistics.spmFailureProbeAttempts
      << " spm_probe_early_rejections="
      << resultStatistics.spmFailureProbeEarlyRejections
      << " spm_probe_tiles_skipped="
      << resultStatistics.spmFailureProbeTilesSkipped
      << " exact_feedback_reorders="
      << resultStatistics.candidatePriorityReorders
      << " resource_schedule_memo_hits="
      << resultStatistics.resourceScheduleMemoHits
      << " resource_schedule_memo_misses="
      << resultStatistics.resourceScheduleMemoMisses
      << " strict_dominated=" << resultStatistics.strictDominatedCandidates
      << " incumbent_closed_feedback="
      << resultStatistics.incumbentClosedFeedbackCandidates
      << " feedback_beam_deferred="
      << resultStatistics.feedbackBeamDeferredCandidates
      << " feedback_root_budget_closures="
      << resultStatistics.feedbackRootBudgetClosures
      << " allocation_feedback_lookahead_candidates="
      << resultStatistics.allocationFeedbackLookaheadCandidates
      << " allocation_feedback_progressive_promotions="
      << resultStatistics.allocationFeedbackProgressivePromotions
      << " allocation_feedback_priority_selections="
      << resultStatistics.allocationFeedbackPrioritySelections
      << " allocation_feedback_endpoint_candidates="
      << resultStatistics.allocationFeedbackEndpointCandidates
      << " allocation_feedback_lookahead_boundary_closures="
      << resultStatistics.allocationFeedbackLookaheadBoundaryClosures
      << " pre_buffer_equivalent_rejections="
      << resultStatistics.preBufferEquivalentRejections
      << " buffer_structure_equivalent_rejections="
      << resultStatistics.bufferStructureEquivalentRejections
      << " selected_stable_ordinal=" << resultStatistics.selectedStableOrdinal
      << " output_mappings=" << resultStatistics.selectedOutputMappingCount
      << " active_tiles=" << resultStatistics.selectedUniqueActiveTileCount
      << " parallel_components="
      << resultStatistics.selectedParallelComponentCount
      << " temporal_wave_lb=" << resultStatistics.selectedTemporalWaveLowerBound
      << " instruction_execution_lb="
      << resultStatistics.selectedInstructionExecutionLowerBound
      << " output_tile_footprint_estimate="
      << resultStatistics.selectedPeakOutputTileFootprintEstimate
      << " aligned_residency_estimate="
      << resultStatistics.selectedPeakAlignedResidencyEstimate
      << " peer_transfers=" << resultStatistics.selectedPeerTransferCount
      << " peer_bytes=" << resultStatistics.selectedPeerBytes
      << " buffer_count=" << resultStatistics.selectedBufferCount
      << " actual_fused_edges="
      << resultStatistics.selectedActualFusedLogicalEdges
      << " spm_movement_work=" << resultStatistics.selectedSPMMovementWork
      << " ddr_movement_work=" << resultStatistics.selectedDDRMovementWork
      << " makespan_ps=" << resultStatistics.selectedMakespanPicoseconds
      << " enabled_terms=" << resultStatistics.enabledDurationTerms << '\n';
  // The comparison summaries intentionally own no actual Tile modules.  Run
  // the exact pipeline once more for the selected semantic state and return
  // only that fresh executable.  This is an equivalence-preserving memory
  // optimization discovered from measured peak RSS, not a candidate limit.
  std::string selectedFailureGate;
  failureReason.clear();
  uint64_t selectedRotatingSlots = 0;
  uint64_t selectedActualFusedLogicalEdges = 0;
  SelectedBufferMaterializationFailure selectedBufferFailure;
  bool selectedIndeterminateFailure = false;
  bool selectedSPMCapacityOverflow = false;
  llvm::SmallVector<SPMCapacityDemandEvidence, 8> selectedSPMCapacityDemands;
  llvm::SmallVector<AcceptedOperationNodeRelation, 64> selectedOperationNodes;
  std::vector<std::string> selectedTileDataflowIRTrace;
  mlir::FailureOr<WholeCardExecutable> selectedExecutable =
      materializeCandidate(
          tensorProgram, admitted[selectedIndex]->candidate, physicalCardId,
          *availableTileIds, operationNodes, *dag, program, executionConfig,
          diagnostics,
          resultStatistics.selectedExecutableRematerializationGates,
          /*searchStatistics=*/nullptr, selectedRotatingSlots,
          selectedActualFusedLogicalEdges, tilePipelineParallelism,
          /*preferredFailureProbeTileId=*/std::nullopt, selectedFailureGate,
          failureReason, selectedBufferFailure, selectedIndeterminateFailure,
          selectedSPMCapacityOverflow, selectedSPMCapacityDemands,
          selectedOperationNodes, selectedTileDataflowIRTrace);
  ++resultStatistics.selectedExecutableRematerializations;
  if (mlir::failed(selectedExecutable)) {
    diagnostics << "wafer-compile: selected whole-card state failed exact "
                   "rematerialization gate="
                << selectedFailureGate << " detail=" << failureReason << '\n';
    return mlir::failure();
  }
  if (selectedActualFusedLogicalEdges !=
      resultStatistics.selectedActualFusedLogicalEdges) {
    diagnostics << "wafer-compile: selected whole-card state changed actual "
                   "fusion witness count during rematerialization\n";
    return mlir::failure();
  }
  return WholeCardCompilationResult(std::move(*selectedExecutable),
                                    std::move(selectedTileDataflowIRTrace));
}

} // namespace wafer::compiler::detail
