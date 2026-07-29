//===- NoCResidentDataflow.cpp - All-rank resident tile synthesis -------===//

#include "NoCResidentDataflow.h"

#include "DirectDTETransport.h"
#include "NoCIntermediateDataflow.h"
#include "NoCPartialDataflow.h"
#include "Wafer/Compiler/GlobalTileRelation.h"
#include "WholeVariantAttemptPlan.h"
#include "WholeVariantResourceAcceptance.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/Common/OpVerifierUtils.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/TargetSchedulingCapability.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/PhysicalDataflow.h"
#include "Wafer/Transforms/SoftwarePipelining.h"
#include "Wafer/Transforms/WorkerPlacement.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

namespace wafer::compiler::detail {
namespace {

struct TypedBoundary {
  int64_t argumentIndex = -1;
  int64_t programIndex = -1;
  frontend::ProgramDistributionKind distribution =
      frontend::ProgramDistributionKind::Replicated;
  llvm::ArrayRef<int64_t> globalShape;
  llvm::ArrayRef<int64_t> localShape;
  llvm::ArrayRef<frontend::ProgramRankSlice> rankSlices;
};

struct BoundaryTileLoad {
  int64_t logicalRank = -1;
  InstrRDMAOp load;
  ResolvedBoundaryTileView view;
  const frontend::ProgramRankSlice *rankSlice = nullptr;
};

struct ResidentTuple {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  wafer::RankBufferingKind bufferingKind = wafer::RankBufferingKind::Single;
  uint32_t bufferingPlanOrdinal = 0;
  wafer::RankWorkerPlacementKind workerPlacementKind =
      wafer::RankWorkerPlacementKind::Unplaced;
  uint32_t workerPlacementPlanOrdinal = 0;
};

struct ResidentSeed {
  llvm::SmallVector<const RankVariantCandidate *, 16> candidates;
  bool reservedBaseline = false;
};

static void setFailureReason(std::string *failureReason,
                             llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
}

static mlir::LogicalResult fail(std::string *failureReason,
                                llvm::StringRef detail) {
  setFailureReason(failureReason, detail);
  return mlir::failure();
}

static const frontend::ProgramRankSlice *
findRankSlice(llvm::ArrayRef<frontend::ProgramRankSlice> slices, int64_t rank) {
  const frontend::ProgramRankSlice *result = nullptr;
  for (const frontend::ProgramRankSlice &slice : slices) {
    if (slice.logicalRank != rank)
      continue;
    if (result)
      return nullptr;
    result = &slice;
  }
  return result;
}

static StaticTileRegion fullLocalTile(llvm::ArrayRef<int64_t> localShape) {
  StaticTileRegion tile;
  tile.offsets.assign(localShape.size(), 0);
  tile.sizes.assign(localShape.begin(), localShape.end());
  tile.strides.assign(localShape.size(), 1);
  return tile;
}

static bool checkedAddNonNegative(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMulNonNegative(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 ||
      (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs))
    return false;
  result = lhs * rhs;
  return true;
}

static std::optional<int64_t>
getStaticTileElementCount(const StaticTileRegion &tile) {
  int64_t count = 1;
  for (int64_t size : tile.sizes)
    if (!checkedMulNonNegative(count, size, count))
      return std::nullopt;
  return count;
}

static mlir::LogicalResult
verifyTypedBoundaryRelation(const TypedBoundary &boundary, int64_t rankCount,
                            std::string *failureReason) {
  if (boundary.argumentIndex < 0 || boundary.programIndex < 0)
    return fail(failureReason,
                "typed boundary has an invalid argument or program index");
  if (boundary.rankSlices.size() != static_cast<size_t>(rankCount))
    return fail(failureReason,
                "typed boundary rank slice domain is incomplete");

  llvm::SmallVector<StaticTileRegion, 16> globalSlices;
  llvm::SmallVector<bool, 16> seenRanks(static_cast<size_t>(rankCount), false);
  llvm::SmallVector<bool, 16> seenReplicaIds(static_cast<size_t>(rankCount),
                                             false);
  globalSlices.reserve(static_cast<size_t>(rankCount));
  for (int64_t rank = 0; rank < rankCount; ++rank) {
    const frontend::ProgramRankSlice *slice =
        findRankSlice(boundary.rankSlices, rank);
    if (!slice)
      return fail(failureReason,
                  "typed boundary lacks one typed slice per rank");
    if (slice->logicalRank < 0 || slice->logicalRank >= rankCount ||
        seenRanks[static_cast<size_t>(slice->logicalRank)])
      return fail(failureReason,
                  "typed boundary rank slice domain is not unique");
    seenRanks[static_cast<size_t>(slice->logicalRank)] = true;
    if (boundary.distribution ==
        frontend::ProgramDistributionKind::Replicated) {
      if (slice->replicaId < 0 || slice->replicaId >= rankCount ||
          seenReplicaIds[static_cast<size_t>(slice->replicaId)])
        return fail(failureReason,
                    "replicated boundary replica domain is not unique");
      seenReplicaIds[static_cast<size_t>(slice->replicaId)] = true;
    } else if (slice->replicaId != 0) {
      return fail(failureReason,
                  "partitioned boundary has a non-zero replica id");
    }

    llvm::Expected<StaticTileRegion> global = mapRankLocalTileToGlobal(
        *slice, boundary.globalShape, boundary.localShape,
        fullLocalTile(boundary.localShape));
    if (!global) {
      llvm::consumeError(global.takeError());
      return fail(failureReason,
                  "typed boundary tile relation is not representable");
    }
    llvm::Expected<StaticTileRegion> ownerProjection = mapGlobalTileToRankLocal(
        *slice, boundary.globalShape, boundary.localShape, *global);
    if (!ownerProjection ||
        *ownerProjection != fullLocalTile(boundary.localShape)) {
      if (!ownerProjection)
        llvm::consumeError(ownerProjection.takeError());
      return fail(failureReason,
                  "typed boundary shard lacks exact owner coverage");
    }
    globalSlices.push_back(std::move(*global));
  }

  if (boundary.distribution == frontend::ProgramDistributionKind::Replicated) {
    if (globalSlices.front() != fullLocalTile(boundary.globalShape))
      return fail(failureReason,
                  "replicated boundary does not cover the full global tile");
    for (const StaticTileRegion &global : llvm::drop_begin(globalSlices))
      if (compareStaticTiles(globalSlices.front(), global) !=
          StaticTileRelation::Equivalent)
        return fail(failureReason,
                    "replicated rank slices do not denote one global tile");
    return mlir::success();
  }

  int64_t globalElements = 1;
  for (int64_t extent : boundary.globalShape)
    if (!checkedMulNonNegative(globalElements, extent, globalElements))
      return fail(failureReason,
                  "partitioned boundary global coverage overflows int64");
  int64_t coveredElements = 0;
  for (size_t lhs = 0; lhs < globalSlices.size(); ++lhs) {
    const StaticTileRegion &slice = globalSlices[lhs];
    if (llvm::any_of(slice.strides, [](int64_t stride) { return stride != 1; }))
      return fail(failureReason, "partitioned boundary relation is not dense");
    std::optional<int64_t> elements = getStaticTileElementCount(slice);
    if (!elements ||
        !checkedAddNonNegative(coveredElements, *elements, coveredElements))
      return fail(failureReason,
                  "partitioned boundary coverage overflows int64");
    for (size_t rhs = lhs + 1; rhs < globalSlices.size(); ++rhs)
      if (compareStaticTiles(slice, globalSlices[rhs]) !=
          StaticTileRelation::Disjoint)
        return fail(failureReason,
                    "partitioned boundary slices overlap or have an "
                    "unrepresentable relation");
  }
  if (coveredElements != globalElements)
    return fail(failureReason,
                "partitioned boundary slices do not cover the global shape");
  return mlir::success();
}

static bool isZeroOffset(mlir::IntegerAttr attr) {
  return !attr || attr.getInt() == 0;
}

static bool isCompleteCompactBoundaryLoad(InstrRDMAOp load, int64_t bytes) {
  return load.getByteCountAttr().getInt() == bytes &&
         load.getInnerBytesAttr().getInt() == bytes &&
         load.getSrcStrides().size() == 3 &&
         load.getSrcIterations().size() == 3 &&
         llvm::all_of(load.getSrcStrides(),
                      [](int64_t stride) { return stride == 0; }) &&
         llvm::all_of(load.getSrcIterations(),
                      [](int64_t iteration) { return iteration == 1; });
}

static std::optional<BoundaryTileLoad>
matchBoundaryTileLoad(InstrRDMAOp load, const TypedBoundary &boundary,
                      const frontend::ProgramRankSlice &rankSlice,
                      int64_t logicalRank) {
  std::optional<ResolvedBoundaryTileView> resolved =
      resolveBoundaryTileView(load.getSource(), boundary.localShape);
  if (!resolved ||
      static_cast<int64_t>(resolved->argumentIndex) != boundary.argumentIndex ||
      !isZeroOffset(load.getSrcOffsetAttr()) ||
      !isZeroOffset(load.getDstOffsetAttr()) ||
      load.getByteCountAttr().getInt() <= 0)
    return std::nullopt;

  auto sourceType =
      mlir::dyn_cast<mlir::MemRefType>(load.getSource().getType());
  auto destType = mlir::dyn_cast<mlir::MemRefType>(load.getDest().getType());
  if (!sourceType || !destType ||
      !wafer::detail::hasWaferMemorySpace(sourceType, MemorySpace::DDR) ||
      !wafer::detail::hasWaferMemorySpace(destType, MemorySpace::SPM) ||
      resolved->leafType != sourceType ||
      sourceType.getShape() != destType.getShape() ||
      sourceType.getElementType() != destType.getElementType())
    return std::nullopt;

  std::optional<WaferPhysicalTensorInfo> destInfo =
      computeWaferPhysicalTensorInfo(destType);
  std::optional<mlir::RankedTensorType> sourceTensor =
      wafer::detail::getLogicalTensorType(sourceType);
  std::optional<int64_t> sourceBytes =
      sourceTensor ? wafer::detail::getCompactTensorByteSize(*sourceTensor)
                   : std::nullopt;
  const int64_t bytes = load.getByteCountAttr().getInt();
  if (!sourceBytes || !destInfo || *sourceBytes != bytes ||
      destInfo->physicalBytes != bytes ||
      !isCompleteCompactBoundaryLoad(load, bytes))
    return std::nullopt;

  return BoundaryTileLoad{logicalRank, load, std::move(*resolved), &rankSlice};
}

static StaticTileRelation
compareLogicalBoundaryTileViews(const BoundaryTileLoad &lhs,
                                const BoundaryTileLoad &rhs,
                                const TypedBoundary &boundary) {
  ResolvedBoundaryTileView normalizedRhs = rhs.view;
  // Payload compatibility is validated atomically by the materializer. It
  // must not split one logical tile occurrence into an apparently valid rank
  // subset when one rank has a different physical leaf type.
  normalizedRhs.leafType = lhs.view.leafType;
  return compareRankBoundaryTileViews(lhs.view, *lhs.rankSlice, normalizedRhs,
                                      *rhs.rankSlice, boundary.globalShape,
                                      boundary.localShape);
}

static llvm::SmallVector<BoundaryTileLoad, 8>
findBoundaryTileLoads(mlir::ModuleOp module, const TypedBoundary &boundary,
                      int64_t rank, bool &ambiguous) {
  llvm::SmallVector<BoundaryTileLoad, 8> matches;
  const frontend::ProgramRankSlice *rankSlice =
      findRankSlice(boundary.rankSlices, rank);
  if (!rankSlice)
    return matches;
  module.walk([&](InstrRDMAOp load) {
    std::optional<BoundaryTileLoad> match =
        matchBoundaryTileLoad(load, boundary, *rankSlice, rank);
    if (match)
      matches.push_back(std::move(*match));
  });
  for (size_t lhs = 0; lhs < matches.size(); ++lhs)
    for (size_t rhs = lhs + 1; rhs < matches.size(); ++rhs) {
      if (compareLogicalBoundaryTileViews(matches[lhs], matches[rhs],
                                          boundary) !=
          StaticTileRelation::Equivalent)
        continue;
      ambiguous = true;
      return {};
    }
  return matches;
}

static int64_t findNextCommunicationId(llvm::ArrayRef<mlir::ModuleOp> modules) {
  int64_t maximum = -1;
  for (mlir::ModuleOp module : modules)
    module.walk([&](mlir::Operation *operation) {
      if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
        maximum = std::max(maximum, send.getMessage().getCommunicationId());
      if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation))
        maximum = std::max(maximum, recv.getMessage().getCommunicationId());
    });
  return maximum == std::numeric_limits<int64_t>::max() ? -1 : maximum + 1;
}

static size_t chooseTileOwnerOrdinal(const TypedBoundary &boundary,
                                     size_t tileGroupOrdinal,
                                     size_t ownerCount) {
  return (static_cast<size_t>(boundary.programIndex) + tileGroupOrdinal) %
         ownerCount;
}

static mlir::LogicalResult materializeBoundaryTileFanout(
    llvm::ArrayRef<BoundaryTileLoad> loads, const TypedBoundary &boundary,
    size_t tileGroupOrdinal, int64_t logicalRankCount, int64_t communicationId,
    NoCFanoutKind kind, std::string *failureReason) {
  if (loads.size() < 2)
    return fail(failureReason,
                "boundary tile has no verified cross-rank consumer reuse");
  for (const BoundaryTileLoad &load : llvm::drop_begin(loads))
    if (!load.rankSlice || !loads.front().rankSlice ||
        compareRankBoundaryTileViews(
            loads.front().view, *loads.front().rankSlice, load.view,
            *load.rankSlice, boundary.globalShape,
            boundary.localShape) != StaticTileRelation::Equivalent)
      return fail(failureReason,
                  "boundary tile fanout has mismatched typed view relations");

  llvm::SmallVector<size_t, 16> sourceCoveredLoads;
  llvm::SmallVector<bool, 16> seenConsumers(
      static_cast<size_t>(logicalRankCount), false);
  for (auto [index, candidate] : llvm::enumerate(loads)) {
    if (candidate.logicalRank < 0 ||
        candidate.logicalRank >= logicalRankCount ||
        seenConsumers[static_cast<size_t>(candidate.logicalRank)])
      return fail(failureReason,
                  "boundary tile consumer rank domain is not unique");
    seenConsumers[static_cast<size_t>(candidate.logicalRank)] = true;
    if (!candidate.rankSlice)
      return fail(failureReason,
                  "boundary tile consumer lacks a typed rank slice");
    if (candidate.view.staticLocalTile) {
      llvm::Expected<StaticTileRegion> global = mapRankLocalTileToGlobal(
          *candidate.rankSlice, boundary.globalShape, boundary.localShape,
          *candidate.view.staticLocalTile);
      if (!global)
        return fail(failureReason,
                    "static boundary tile lost its verified global relation");
      llvm::Expected<StaticTileRegion> projected =
          mapGlobalTileToRankLocal(*candidate.rankSlice, boundary.globalShape,
                                   boundary.localShape, *global);
      if (!projected || *projected != *candidate.view.staticLocalTile) {
        if (!projected)
          llvm::consumeError(projected.takeError());
        continue;
      }
    }
    sourceCoveredLoads.push_back(index);
  }
  if (sourceCoveredLoads.empty())
    return fail(failureReason,
                "boundary tile has no actual load with owner coverage");
  const size_t selectedOwner = chooseTileOwnerOrdinal(
      boundary, tileGroupOrdinal, sourceCoveredLoads.size());
  const size_t ownerIndex = sourceCoveredLoads[selectedOwner];
  const int64_t owner = loads[ownerIndex].logicalRank;

  auto getByteCount = [](const BoundaryTileLoad &candidate) {
    return candidate.load->getAttrOfType<mlir::IntegerAttr>("byte_count")
        .getInt();
  };
  auto getDestType = [](const BoundaryTileLoad &candidate) {
    return candidate.load->getOperand(1).getType();
  };
  const int64_t bytes = getByteCount(loads[ownerIndex]);
  for (const BoundaryTileLoad &candidate : loads)
    if (getByteCount(candidate) != bytes ||
        getDestType(candidate) != getDestType(loads.front()))
      return fail(failureReason,
                  "boundary tile fanout has mismatched physical payloads");

  llvm::SmallVector<size_t, 16> forwardingOrder;
  forwardingOrder.reserve(loads.size());
  forwardingOrder.push_back(ownerIndex);
  for (auto [index, candidate] : llvm::enumerate(loads))
    if (index != ownerIndex)
      forwardingOrder.push_back(index);
  llvm::sort(llvm::drop_begin(forwardingOrder), [&](size_t lhs, size_t rhs) {
    const int64_t lhsDistance =
        (loads[lhs].logicalRank - owner + logicalRankCount) % logicalRankCount;
    const int64_t rhsDistance =
        (loads[rhs].logicalRank - owner + logicalRankCount) % logicalRankCount;
    return lhsDistance < rhsDistance;
  });

  for (auto [orderIndex, loadIndex] : llvm::enumerate(forwardingOrder)) {
    const BoundaryTileLoad &candidate = loads[loadIndex];
    const int64_t rank = candidate.logicalRank;
    InstrRDMAOp load = candidate.load;
    mlir::OpBuilder builder(load);
    auto message =
        DTEMessageAttr::get(load.getContext(), communicationId,
                            DTEProtocolPhase::PeerDataflow, /*round=*/0,
                            /*payloadSlice=*/boundary.programIndex);
    if (rank == owner) {
      builder.setInsertionPointAfter(load);
      if (kind == NoCFanoutKind::ReceiveForward) {
        const int64_t peer = loads[forwardingOrder[1]].logicalRank;
        auto send = builder.create<CommPeerSendOp>(
            load.getLoc(), builder.getType<mlir::async::TokenType>(),
            load.getDest(), builder.getI64IntegerAttr(peer),
            builder.getI64IntegerAttr(bytes), message);
        builder.create<mlir::async::AwaitOp>(load.getLoc(), send.getToken());
      } else {
        for (const BoundaryTileLoad &peerLoad : loads) {
          const int64_t peer = peerLoad.logicalRank;
          if (peer == owner)
            continue;
          auto send = builder.create<CommPeerSendOp>(
              load.getLoc(), builder.getType<mlir::async::TokenType>(),
              load.getDest(), builder.getI64IntegerAttr(peer),
              builder.getI64IntegerAttr(bytes), message);
          builder.create<mlir::async::AwaitOp>(load.getLoc(), send.getToken());
        }
      }
      continue;
    }

    const int64_t source =
        kind == NoCFanoutKind::Direct
            ? owner
            : loads[forwardingOrder[orderIndex - 1]].logicalRank;
    auto recv = builder.create<CommPeerRecvOp>(
        load.getLoc(), builder.getType<mlir::async::TokenType>(),
        load.getDest(), builder.getI64IntegerAttr(source),
        builder.getI64IntegerAttr(bytes), message);
    auto await =
        builder.create<mlir::async::AwaitOp>(load.getLoc(), recv.getToken());
    if (kind == NoCFanoutKind::ReceiveForward &&
        orderIndex + 1 < forwardingOrder.size()) {
      builder.setInsertionPointAfter(await);
      const int64_t peer = loads[forwardingOrder[orderIndex + 1]].logicalRank;
      auto send = builder.create<CommPeerSendOp>(
          load.getLoc(), builder.getType<mlir::async::TokenType>(),
          load.getDest(), builder.getI64IntegerAttr(peer),
          builder.getI64IntegerAttr(bytes), message);
      builder.create<mlir::async::AwaitOp>(load.getLoc(), send.getToken());
    }
    load.erase();
  }
  return mlir::success();
}

static unsigned
materializeTypedBoundaries(llvm::MutableArrayRef<mlir::ModuleOp> modules,
                           llvm::ArrayRef<TypedBoundary> boundaries,
                           int64_t &communicationId, NoCFanoutKind kind) {
  unsigned materialized = 0;
  const int64_t logicalRankCount = static_cast<int64_t>(modules.size());
  for (const TypedBoundary &boundary : boundaries) {
    llvm::SmallVector<BoundaryTileLoad, 32> loads;
    bool ambiguous = false;
    for (auto [rank, module] : llvm::enumerate(modules))
      llvm::append_range(
          loads, findBoundaryTileLoads(module, boundary,
                                       static_cast<int64_t>(rank), ambiguous));
    if (loads.empty() || ambiguous)
      continue;

    // Form exact equivalence classes in rank-major IR walk order. LLVM
    // semantic hashes are process-seeded in supported builds, so neither
    // group order nor owner selection may depend on their numeric value.
    llvm::SmallVector<llvm::SmallVector<BoundaryTileLoad, 16>, 8> tileGroups;
    for (BoundaryTileLoad &load : loads) {
      auto group = llvm::find_if(tileGroups, [&](const auto &candidate) {
        const BoundaryTileLoad &anchor = candidate.front();
        return compareLogicalBoundaryTileViews(anchor, load, boundary) ==
               StaticTileRelation::Equivalent;
      });
      if (group == tileGroups.end()) {
        tileGroups.emplace_back();
        tileGroups.back().push_back(std::move(load));
      } else {
        group->push_back(std::move(load));
      }
    }

    for (auto [groupOrdinal, consumers] : llvm::enumerate(tileGroups)) {
      // A verified partitioned boundary normally contributes one consumer
      // load per unique global shard. That singleton is already the minimum
      // legal DDR coverage and is not a fan-out opportunity. Peer traffic is
      // materialized only when actual rank-local loads prove reuse of the
      // exact same global tile.
      if (consumers.size() < 2)
        continue;
      if (communicationId < 0)
        return materialized;
      std::string ignoredFailure;
      if (mlir::failed(materializeBoundaryTileFanout(
              consumers, boundary, groupOrdinal, logicalRankCount,
              communicationId, kind, &ignoredFailure)))
        continue;
      ++materialized;
      communicationId = communicationId == std::numeric_limits<int64_t>::max()
                            ? -1
                            : communicationId + 1;
    }
  }
  return materialized;
}

using ResidentSeedCorrespondenceKey =
    std::tuple<int64_t, wafer::RankArtifactKind, wafer::RankBufferingKind,
               uint32_t, wafer::RankWorkerPlacementKind, uint32_t>;

static std::optional<ResidentSeed>
getEligibleResidentSeed(llvm::ArrayRef<size_t> candidateIndices,
                        const std::vector<RankVariantFrontier> &frontiers) {
  if (candidateIndices.empty() || candidateIndices.size() != frontiers.size())
    return std::nullopt;

  ResidentSeed seed;
  seed.candidates.reserve(frontiers.size());
  std::optional<ResidentSeedCorrespondenceKey> correspondence;
  std::optional<bool> reservedBaseline;
  for (auto [rank, candidateIndex] : llvm::enumerate(candidateIndices)) {
    if (candidateIndex >= frontiers[rank].size())
      return std::nullopt;
    const RankVariantCandidate &candidate = frontiers[rank][candidateIndex];
    const bool eligibleSingle =
        candidate.bufferingKind == wafer::RankBufferingKind::Single &&
        candidate.bufferingPlanOrdinal == 0;
    const bool eligibleFixedSlot =
        candidate.bufferingKind == wafer::RankBufferingKind::StaticFixedSlot &&
        candidate.bufferingPlanOrdinal != 0;
    const bool eligibleUnplaced =
        candidate.workerPlacementKind ==
            wafer::RankWorkerPlacementKind::Unplaced &&
        candidate.workerPlacementPlanOrdinal == 0;
    const bool eligibleWorkerPlacement =
        candidate.workerPlacementKind ==
            wafer::RankWorkerPlacementKind::DisjointComponents &&
        candidate.workerPlacementPlanOrdinal != 0;
    if (!candidate.module || (!eligibleSingle && !eligibleFixedSlot) ||
        (!eligibleUnplaced && !eligibleWorkerPlacement))
      return std::nullopt;
    std::set<uint32_t> issueWorkers;
    candidate.module.get()->walk([&](mlir::Operation *operation) {
      if (std::optional<NCCWorker> worker = getNCCIssueWorker(operation))
        issueWorkers.insert(static_cast<uint32_t>(*worker));
    });
    const bool hasNonzeroWorker =
        llvm::any_of(issueWorkers, [](uint32_t worker) { return worker != 0; });
    if ((eligibleUnplaced && hasNonzeroWorker) ||
        (eligibleWorkerPlacement && issueWorkers.size() < 2))
      return std::nullopt;

    ResidentSeedCorrespondenceKey current{
        candidate.stableOrdinal,       candidate.artifactKind,
        candidate.bufferingKind,       candidate.bufferingPlanOrdinal,
        candidate.workerPlacementKind, candidate.workerPlacementPlanOrdinal};
    if ((correspondence && *correspondence != current) ||
        (reservedBaseline && *reservedBaseline != candidate.reservedBaseline))
      return std::nullopt;
    correspondence = current;
    reservedBaseline = candidate.reservedBaseline;
    seed.candidates.push_back(&candidate);
  }
  seed.reservedBaseline = *reservedBaseline;
  return seed;
}

static mlir::FailureOr<llvm::SmallVector<ResidentSeed, 8>>
collectResidentSeeds(const std::vector<RankVariantFrontier> &frontiers,
                     int64_t expectedRankCount, std::string *failureReason) {
  std::vector<RankVariantMetadataFrontier> metadata;
  metadata.reserve(frontiers.size());
  for (const RankVariantFrontier &frontier : frontiers) {
    RankVariantMetadataFrontier rankMetadata;
    rankMetadata.reserve(frontier.size());
    for (const RankVariantCandidate &candidate : frontier)
      rankMetadata.push_back(
          {candidate.stableOrdinal, candidate.artifactKind,
           candidate.reservedBaseline, candidate.bufferingKind,
           candidate.bufferingPlanOrdinal, candidate.workerPlacementKind,
           candidate.workerPlacementPlanOrdinal});
    metadata.push_back(std::move(rankMetadata));
  }

  WholeVariantAttemptPlan plan =
      buildWholeVariantAttemptPlan(metadata, expectedRankCount);
  if (plan.failure == WholeVariantAttemptPlanFailure::CandidateDomain) {
    setFailureReason(failureReason,
                     "rank frontiers do not form a canonical candidate domain");
    return mlir::failure();
  }
  if (plan.failure == WholeVariantAttemptPlanFailure::ReservedBaseline) {
    setFailureReason(failureReason,
                     "rank frontiers do not contain one canonical baseline");
    return mlir::failure();
  }

  std::optional<ResidentSeed> baseline =
      getEligibleResidentSeed(plan.reservedBaselineIndices, frontiers);
  if (!baseline || !baseline->reservedBaseline) {
    setFailureReason(failureReason,
                     "reserved baseline is not a complete materialized "
                     "Single or StaticFixedSlot tuple with canonical typed "
                     "worker-placement metadata");
    return mlir::failure();
  }

  llvm::SmallVector<ResidentSeed, 8> seeds;
  seeds.push_back(std::move(*baseline));
  std::set<std::vector<size_t>> visited;
  visited.insert(plan.reservedBaselineIndices);
  auto appendSeed = [&](const std::vector<size_t> &candidateIndices,
                        size_t limit) {
    if (seeds.size() >= limit)
      return;
    if (!visited.insert(candidateIndices).second)
      return;
    std::optional<ResidentSeed> seed =
        getEligibleResidentSeed(candidateIndices, frontiers);
    if (!seed || seed->reservedBaseline)
      return;
    seeds.push_back(std::move(*seed));
  };

  // Keep the original total seed cap, including the baseline. Interleave the
  // two orthogonal typed bands with the generic attempt order so neither
  // worker/fixed composition nor ordinary NoC opportunities can consume the
  // entire bounded budget by itself.
  const size_t maximumBandSize =
      std::max({plan.workerPlacedCandidateIndices.size(),
                plan.fixedSlotCandidateIndices.size(),
                plan.genericCandidateIndices.size()});
  for (size_t index = 0; index < maximumBandSize &&
                         seeds.size() < kNoCResidentCompleteTupleSeedLimit;
       ++index) {
    if (index < plan.workerPlacedCandidateIndices.size())
      appendSeed(plan.workerPlacedCandidateIndices[index],
                 kNoCResidentCompleteTupleSeedLimit);
    if (index < plan.fixedSlotCandidateIndices.size())
      appendSeed(plan.fixedSlotCandidateIndices[index],
                 kNoCResidentCompleteTupleSeedLimit);
    if (index < plan.genericCandidateIndices.size())
      appendSeed(plan.genericCandidateIndices[index],
                 kNoCResidentCompleteTupleSeedLimit);
  }
  // Borrow any remaining capacity from the ordinary bounded attempt order.
  // Explicit correspondence bands above establish fairness; this fallback
  // preserves pre-existing opportunities when one or more bands are sparse.
  for (const std::vector<size_t> &candidateIndices :
       plan.optimizedCandidateIndices)
    appendSeed(candidateIndices, kNoCResidentCompleteTupleSeedLimit);
  return seeds;
}

static llvm::SmallVector<mlir::ModuleOp, 16>
getModuleViews(ResidentTuple &tuple) {
  llvm::SmallVector<mlir::ModuleOp, 16> modules;
  modules.reserve(tuple.modules.size());
  for (mlir::OwningOpRef<mlir::ModuleOp> &owner : tuple.modules)
    modules.push_back(*owner);
  return modules;
}

static ResidentTuple cloneResidentTuple(const ResidentTuple &source) {
  ResidentTuple clone;
  clone.bufferingKind = source.bufferingKind;
  clone.bufferingPlanOrdinal = source.bufferingPlanOrdinal;
  clone.workerPlacementKind = source.workerPlacementKind;
  clone.workerPlacementPlanOrdinal = source.workerPlacementPlanOrdinal;
  clone.modules.reserve(source.modules.size());
  for (const mlir::OwningOpRef<mlir::ModuleOp> &owner : source.modules)
    clone.modules.push_back(
        mlir::cast<mlir::ModuleOp>(owner.get().getOperation()->clone()));
  return clone;
}

static bool finalizeResidentTuple(ResidentTuple &tuple,
                                  const ExecutionConfig &executionConfig) {
  if (tuple.modules.empty())
    return false;
  const TargetMemoryPolicy memory =
      getDefaultWaferTargetPolicy(TileSearchEffort::Default).memory;
  mlir::MLIRContext *context = tuple.modules.front()->getContext();
  bool accepted = true;
  mlir::ScopedDiagnosticHandler handler(
      context, [&](mlir::Diagnostic &diagnostic) {
        if (diagnostic.getSeverity() == mlir::DiagnosticSeverity::Error)
          accepted = false;
        return mlir::success();
      });

  for (mlir::OwningOpRef<mlir::ModuleOp> &owner : tuple.modules) {
    mlir::ModuleOp module = *owner;
    uint64_t terminalOperationCount = 0;
    if (wafer::detail::checkStaticTerminalOperationBudget(
            module.getOperation(), terminalOperationCount) !=
            wafer::detail::StaticTerminalOperationBudgetStatus::WithinBudget ||
        mlir::failed(normalizeMinimumNCCJoins(module)) ||
        mlir::failed(mlir::verify(module)) ||
        mlir::failed(planSPMMemoryModule(
            module, memory.spmBase, memory.spmLimit, memory.spmAlignment)) ||
        mlir::failed(mlir::verify(module))) {
      accepted = false;
      break;
    }
  }
  if (!accepted)
    return false;

  llvm::SmallVector<mlir::ModuleOp, 16> modules = getModuleViews(tuple);
  return mlir::succeeded(verifyDirectDTETransportSchedule(modules)) &&
         mlir::succeeded(acceptWholeVariantResources(modules, executionConfig));
}

static bool planResidentTupleSPM(ResidentTuple &tuple) {
  if (tuple.modules.empty())
    return false;
  const TargetMemoryPolicy memory =
      getDefaultWaferTargetPolicy(TileSearchEffort::Default).memory;
  mlir::MLIRContext *context = tuple.modules.front()->getContext();
  bool accepted = true;
  mlir::ScopedDiagnosticHandler handler(
      context, [&](mlir::Diagnostic &diagnostic) {
        if (diagnostic.getSeverity() == mlir::DiagnosticSeverity::Error)
          accepted = false;
        return mlir::success();
      });
  for (mlir::OwningOpRef<mlir::ModuleOp> &module : tuple.modules)
    if (mlir::failed(planSPMMemoryModule(
            *module, memory.spmBase, memory.spmLimit, memory.spmAlignment))) {
      accepted = false;
      break;
    }
  return accepted;
}

static std::optional<ResidentTuple>
deriveWorkerPlacedResidentTuple(const ResidentTuple &unplaced,
                                const ExecutionConfig &executionConfig) {
  if (unplaced.modules.empty() ||
      unplaced.workerPlacementKind !=
          wafer::RankWorkerPlacementKind::Unplaced ||
      unplaced.workerPlacementPlanOrdinal != 0)
    return std::nullopt;

  llvm::Expected<TargetSchedulingCapabilityRegistry> registry =
      getTargetSchedulingCapabilityRegistry(
          executionConfig.getTargetProfileId());
  if (!registry) {
    llvm::consumeError(registry.takeError());
    return std::nullopt;
  }

  ResidentTuple placed;
  placed.bufferingKind = unplaced.bufferingKind;
  placed.bufferingPlanOrdinal = unplaced.bufferingPlanOrdinal;
  placed.workerPlacementKind =
      wafer::RankWorkerPlacementKind::DisjointComponents;
  placed.workerPlacementPlanOrdinal = 1;
  placed.modules.reserve(unplaced.modules.size());
  for (const mlir::OwningOpRef<mlir::ModuleOp> &owner : unplaced.modules) {
    std::string ignoredFailure;
    mlir::FailureOr<NCCWorkerPlacementCandidate> candidate =
        deriveDisjointNCCWorkerPlacementCandidate(*owner, &ignoredFailure);
    if (mlir::failed(candidate) ||
        llvm::popcount(candidate->participantMask) < 2)
      return std::nullopt;

    bool hasDirectDTE = false;
    candidate->module->walk([&](mlir::Operation *operation) {
      hasDirectDTE |= mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation);
    });
    TargetSchedulingMechanism mechanism =
        placed.bufferingKind == wafer::RankBufferingKind::StaticFixedSlot
            ? TargetSchedulingMechanism::StaticFixedSlot
        : hasDirectDTE ? TargetSchedulingMechanism::DirectDTEOverlap
                       : TargetSchedulingMechanism::WorkerPlacement;
    llvm::Expected<TargetSchedulingWindowQuery> query =
        analyzeTargetSchedulingWindow(*candidate->module,
                                      executionConfig.getTargetProfileId(),
                                      mechanism);
    if (!query) {
      llvm::consumeError(query.takeError());
      return std::nullopt;
    }
    llvm::Expected<TargetSchedulingWindowDecision> decision =
        registry->query(*query);
    if (!decision) {
      llvm::consumeError(decision.takeError());
      return std::nullopt;
    }
    if (decision->legality != TargetSchedulingCapabilityState::Supported)
      return std::nullopt;
    placed.modules.push_back(std::move(candidate->module));
  }
  if (placed.modules.size() != unplaced.modules.size())
    return std::nullopt;
  return placed;
}

static bool finalizeFixedResidentTuple(ResidentTuple &tuple,
                                       const ExecutionConfig &executionConfig) {
  if (!planResidentTupleSPM(tuple))
    return false;
  llvm::SmallVector<mlir::ModuleOp, 16> views = getModuleViews(tuple);
  std::string specializationFailure;
  return mlir::succeeded(
             specializePeriodicDirectDTESites(views, &specializationFailure)) &&
         finalizeResidentTuple(tuple, executionConfig);
}

template <typename Effect>
static bool hasValueEffect(mlir::Operation *operation, mlir::Value value) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [&](const auto &instance) {
    return instance.getValue() == value &&
           mlir::isa<Effect>(instance.getEffect());
  });
}

static bool
orderPeerReceivePreparationBeforeSends(llvm::ArrayRef<mlir::ModuleOp> modules) {
  for (mlir::ModuleOp module : modules) {
    llvm::SmallPtrSet<mlir::Block *, 16> communicationBlocks;
    module.walk([&](mlir::Operation *operation) {
      if (mlir::isa<CommPeerSendOp, CommPeerRecvOp>(operation))
        communicationBlocks.insert(operation->getBlock());
    });
    if (communicationBlocks.empty())
      continue;

    // Preparation is moved only within its own structured occurrence. The
    // final instruction tuple is then admitted by the call-expanded global
    // message wait-graph verifier in finalizeResidentTuple; block identity is
    // not used as a surrogate for whole-program acyclicity.
    for (mlir::Block *block : communicationBlocks) {
      mlir::Operation *firstTransportIssue = nullptr;
      llvm::SmallVector<CommPeerRecvOp, 8> receives;
      for (mlir::Operation &operation : *block) {
        // A seed may already contain lowered Direct-DTE traffic (for example
        // a collective partial protocol).  Preparing only before newly
        // materialized peer ops can leave:
        //
        //   existing send + wait -> new receive preparation
        //
        // on one rank while its peer has the reverse dependency.  Direct-DTE
        // matching is per message and cannot prove that cross-message wait
        // graph acyclic.  Put every new receive preparation ahead of all
        // transport issues in its structured block, including issues carried
        // by the seed.
        if (!firstTransportIssue &&
            mlir::isa<CommPeerSendOp, CommPeerRecvOp, InstrDTESendOp,
                      InstrDTERecvOp>(operation))
          firstTransportIssue = &operation;
        if (auto receive = mlir::dyn_cast<CommPeerRecvOp>(operation))
          receives.push_back(receive);
      }
      if (!firstTransportIssue || receives.empty())
        continue;

      for (CommPeerRecvOp receive : receives) {
        for (mlir::OpOperand &use : receive.getBuffer().getUses()) {
          mlir::Operation *owner = use.getOwner();
          if (owner == receive.getOperation())
            continue;
          if (owner->getBlock() != block ||
              (owner->isBeforeInBlock(receive) &&
               (hasValueEffect<mlir::MemoryEffects::Read>(
                    owner, receive.getBuffer()) ||
                hasValueEffect<mlir::MemoryEffects::Write>(
                    owner, receive.getBuffer()))))
            return false;
        }

        mlir::Operation *definition = receive.getBuffer().getDefiningOp();
        if (!definition || definition->getBlock() != block ||
            definition->isBeforeInBlock(firstTransportIssue))
          continue;
        auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(definition);
        if (!allocation || !allocation.getType().hasStaticShape() ||
            allocation.getNumOperands() != 0)
          return false;
        allocation->moveBefore(firstTransportIssue);
      }

      mlir::Operation *lastPreparation = nullptr;
      for (CommPeerRecvOp receive : receives) {
        if (!lastPreparation && receive.getOperation() == firstTransportIssue) {
          lastPreparation = receive;
          continue;
        }
        if (!lastPreparation)
          receive->moveBefore(firstTransportIssue);
        else
          receive->moveAfter(lastPreparation);
        lastPreparation = receive;
      }
    }
  }
  return true;
}

struct StaticLoopBounds {
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 0;

  bool operator==(const StaticLoopBounds &other) const {
    return std::tie(lower, upper, step) ==
           std::tie(other.lower, other.upper, other.step);
  }
};

static std::optional<StaticLoopBounds>
getExactStaticLoopBounds(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step)
    return std::nullopt;
  return StaticLoopBounds{*lower, *upper, *step};
}

} // namespace

mlir::FailureOr<llvm::SmallVector<mlir::scf::ForOp, 16>>
resolveExactStaticLoopCorrespondence(llvm::ArrayRef<mlir::ModuleOp> modules,
                                     mlir::scf::ForOp anchor) {
  if (modules.empty() || !anchor ||
      anchor->getParentOfType<mlir::ModuleOp>() != modules.front())
    return mlir::failure();
  std::optional<StaticLoopBounds> anchorBounds =
      getExactStaticLoopBounds(anchor);
  if (!anchorBounds)
    return mlir::failure();

  llvm::SmallVector<mlir::scf::ForOp, 16> correspondingLoops{anchor};
  for (mlir::ModuleOp module : llvm::drop_begin(modules)) {
    llvm::SmallVector<mlir::scf::ForOp, 2> matches;
    module.walk([&](mlir::scf::ForOp loop) {
      std::optional<StaticLoopBounds> bounds = getExactStaticLoopBounds(loop);
      if (bounds && *bounds == *anchorBounds &&
          haveEquivalentStructuredOperationPaths(anchor, loop))
        matches.push_back(loop);
    });
    if (matches.size() != 1)
      return mlir::failure();
    correspondingLoops.push_back(matches.front());
  }
  return correspondingLoops;
}

namespace {

static void
appendFixedSlotNeighbors(ResidentTuple &source,
                         const ExecutionConfig &executionConfig,
                         size_t &workerPlacedTupleCount,
                         llvm::SmallVectorImpl<ResidentTuple> &neighbors) {
  llvm::Expected<TargetSchedulingCapabilityRegistry> registry =
      getTargetSchedulingCapabilityRegistry(
          executionConfig.getTargetProfileId());
  if (!registry) {
    llvm::consumeError(registry.takeError());
    return;
  }

  if (source.modules.empty())
    return;
  llvm::SmallVector<mlir::scf::ForOp, 8> anchorLoops;
  source.modules.front()->walk(
      [&](mlir::scf::ForOp loop) { anchorLoops.push_back(loop); });
  if (anchorLoops.empty())
    return;

  static constexpr size_t kFixedSlotNeighborLimit = 8;
  size_t appended = 0;
  for (size_t structuralOrdinal = 0;
       structuralOrdinal < anchorLoops.size() &&
       appended < kFixedSlotNeighborLimit &&
       structuralOrdinal < std::numeric_limits<uint32_t>::max();
       ++structuralOrdinal) {
    mlir::scf::ForOp anchor = anchorLoops[structuralOrdinal];
    llvm::SmallVector<mlir::ModuleOp, 16> modules;
    modules.reserve(source.modules.size());
    for (const mlir::OwningOpRef<mlir::ModuleOp> &module : source.modules)
      modules.push_back(*module);
    mlir::FailureOr<llvm::SmallVector<mlir::scf::ForOp, 16>>
        correspondingLoops =
            resolveExactStaticLoopCorrespondence(modules, anchor);
    if (mlir::failed(correspondingLoops))
      continue;

    ResidentTuple candidate;
    candidate.bufferingKind = wafer::RankBufferingKind::StaticFixedSlot;
    candidate.bufferingPlanOrdinal =
        static_cast<uint32_t>(structuralOrdinal) + 1;
    candidate.workerPlacementKind = source.workerPlacementKind;
    candidate.workerPlacementPlanOrdinal = source.workerPlacementPlanOrdinal;
    candidate.modules.reserve(source.modules.size());
    bool valid = true;
    for (size_t rank = 0; rank < source.modules.size(); ++rank) {
      std::string ignoredFailure;
      mlir::FailureOr<StaticFixedSlotPipelineCandidate> derived =
          deriveStaticFixedSlotPipelineCandidate(*source.modules[rank],
                                                 (*correspondingLoops)[rank],
                                                 &ignoredFailure);
      if (mlir::failed(derived) || derived->stageCount < 2 ||
          derived->slotAllocationCount < 2 ||
          mlir::failed(normalizeMinimumNCCJoins(*derived->module))) {
        valid = false;
        break;
      }

      llvm::Expected<TargetSchedulingWindowQuery> query =
          analyzeTargetSchedulingWindow(
              *derived->module, executionConfig.getTargetProfileId(),
              TargetSchedulingMechanism::StaticFixedSlot);
      if (!query) {
        llvm::consumeError(query.takeError());
        valid = false;
        break;
      }
      llvm::Expected<TargetSchedulingWindowDecision> decision =
          registry->query(*query);
      if (!decision) {
        llvm::consumeError(decision.takeError());
        valid = false;
        break;
      }
      if (decision->legality != TargetSchedulingCapabilityState::Supported) {
        valid = false;
        break;
      }
      candidate.modules.push_back(std::move(derived->module));
    }
    if (!valid || candidate.modules.size() != source.modules.size())
      continue;
    bool retained = false;
    const bool isWorkerPlaced =
        candidate.workerPlacementKind ==
        wafer::RankWorkerPlacementKind::DisjointComponents;
    if ((!isWorkerPlaced ||
         workerPlacedTupleCount <
             wafer::kWorkerPlacementRankFrontierAdmissionLimit) &&
        finalizeFixedResidentTuple(candidate, executionConfig)) {
      workerPlacedTupleCount += isWorkerPlaced ? 1u : 0u;
      neighbors.push_back(std::move(candidate));
      retained = true;
    }
    appended += retained ? 1u : 0u;
  }
}

template <typename Materializer>
static mlir::FailureOr<llvm::SmallVector<ResidentTuple, 8>>
buildResidentTuples(llvm::ArrayRef<const RankVariantCandidate *> seeds,
                    const ExecutionConfig &executionConfig,
                    size_t &workerPlacedTupleCount, Materializer &&materialize,
                    std::string *failureReason) {
  if (seeds.empty())
    return mlir::failure();
  ResidentTuple tuple;
  tuple.bufferingKind = seeds.front()->bufferingKind;
  tuple.bufferingPlanOrdinal = seeds.front()->bufferingPlanOrdinal;
  tuple.workerPlacementKind = seeds.front()->workerPlacementKind;
  tuple.workerPlacementPlanOrdinal = seeds.front()->workerPlacementPlanOrdinal;
  llvm::SmallVector<mlir::ModuleOp, 16> modules;
  tuple.modules.reserve(seeds.size());
  modules.reserve(seeds.size());
  for (const RankVariantCandidate *seed : seeds) {
    tuple.modules.push_back(
        mlir::cast<mlir::ModuleOp>(seed->module.get()->clone()));
    clearRankCandidatePhysicalFacts(*tuple.modules.back());
    modules.push_back(*tuple.modules.back());
  }

  if (materialize(modules) == 0 ||
      !orderPeerReceivePreparationBeforeSends(modules))
    return mlir::failure();

  for (mlir::ModuleOp module : modules) {
    std::string conversionFailure;
    if (mlir::failed(wafer::convertTileRegionToInstrModule(
            module, &conversionFailure)) ||
        mlir::failed(mlir::verify(module))) {
      if (failureReason)
        *failureReason = "peer tile lowering rejected the all-rank candidate";
      return mlir::failure();
    }
  }

  if (tuple.bufferingKind != wafer::RankBufferingKind::Single &&
      tuple.bufferingKind != wafer::RankBufferingKind::StaticFixedSlot)
    return mlir::failure();

  // Worker placement consumes the materialized Instr graph before fixed-slot
  // scheduling changes operation order and before periodic DTE specialization
  // rewrites rotating endpoint recurrences to physical allocation roots.  The
  // resulting typed worker assignment is then preserved by fixed-slot
  // derivation and independent physical replanning.
  std::optional<ResidentTuple> workerPlaced;
  if (tuple.workerPlacementKind == wafer::RankWorkerPlacementKind::Unplaced &&
      workerPlacedTupleCount <
          wafer::kWorkerPlacementRankFrontierAdmissionLimit)
    workerPlaced = deriveWorkerPlacedResidentTuple(tuple, executionConfig);

  std::optional<ResidentTuple> baseWorkerFixedSource;
  std::optional<ResidentTuple> derivedWorkerFixedSource;
  llvm::SmallVector<ResidentTuple, 8> unplacedFixedSlotNeighbors;
  llvm::SmallVector<ResidentTuple, 8> workerFixedSlotNeighbors;
  if (tuple.bufferingKind == wafer::RankBufferingKind::Single) {
    if (tuple.workerPlacementKind == wafer::RankWorkerPlacementKind::Unplaced)
      appendFixedSlotNeighbors(tuple, executionConfig, workerPlacedTupleCount,
                               unplacedFixedSlotNeighbors);
    else
      baseWorkerFixedSource = cloneResidentTuple(tuple);
    if (workerPlaced)
      derivedWorkerFixedSource = cloneResidentTuple(*workerPlaced);
  }

  llvm::SmallVector<ResidentTuple, 8> alternatives;
  const bool tupleFinalized =
      tuple.bufferingKind == wafer::RankBufferingKind::StaticFixedSlot
          ? finalizeFixedResidentTuple(tuple, executionConfig)
          : finalizeResidentTuple(tuple, executionConfig);
  if (tupleFinalized &&
      (tuple.workerPlacementKind == wafer::RankWorkerPlacementKind::Unplaced ||
       workerPlacedTupleCount <
           wafer::kWorkerPlacementRankFrontierAdmissionLimit)) {
    if (tuple.workerPlacementKind ==
        wafer::RankWorkerPlacementKind::DisjointComponents)
      ++workerPlacedTupleCount;
    alternatives.push_back(std::move(tuple));
  }

  if (workerPlaced && workerPlacedTupleCount <
                          wafer::kWorkerPlacementRankFrontierAdmissionLimit) {
    const bool finalized =
        workerPlaced->bufferingKind == wafer::RankBufferingKind::StaticFixedSlot
            ? finalizeFixedResidentTuple(*workerPlaced, executionConfig)
            : finalizeResidentTuple(*workerPlaced, executionConfig);
    if (finalized) {
      ++workerPlacedTupleCount;
      alternatives.push_back(std::move(*workerPlaced));
    }
  }

  // Reserve the generic worker-placed realization above before admitting its
  // fixed-slot siblings, so a saturated structural loop band cannot starve the
  // direct worker alternative under the shared bounded cap.
  if (baseWorkerFixedSource)
    appendFixedSlotNeighbors(*baseWorkerFixedSource, executionConfig,
                             workerPlacedTupleCount, workerFixedSlotNeighbors);
  if (derivedWorkerFixedSource)
    appendFixedSlotNeighbors(*derivedWorkerFixedSource, executionConfig,
                             workerPlacedTupleCount, workerFixedSlotNeighbors);

  alternatives.append(
      std::make_move_iterator(unplacedFixedSlotNeighbors.begin()),
      std::make_move_iterator(unplacedFixedSlotNeighbors.end()));
  alternatives.append(std::make_move_iterator(workerFixedSlotNeighbors.begin()),
                      std::make_move_iterator(workerFixedSlotNeighbors.end()));
  if (alternatives.empty())
    return mlir::failure();
  return alternatives;
}

} // namespace

mlir::LogicalResult appendNoCResidentDataflowCandidates(
    std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, std::string *failureReason) {
  if (frontiers.size() != static_cast<size_t>(executionConfig.getRankCount()) ||
      frontiers.empty() ||
      program.logicalRankCount != executionConfig.getRankCount())
    return fail(failureReason,
                "all-rank resident dataflow synthesis received an incomplete "
                "domain");
  if (executionConfig.getRankCount() <= 1)
    return mlir::success();

  mlir::FailureOr<llvm::SmallVector<ResidentSeed, 8>> seeds =
      collectResidentSeeds(frontiers, executionConfig.getRankCount(),
                           failureReason);
  if (mlir::failed(seeds))
    return mlir::failure();

  llvm::SmallVector<TypedBoundary, 8> boundaries;
  for (const frontend::ProgramBoundaryBinding &binding :
       program.distributedInputs)
    boundaries.push_back({binding.index, binding.programIndex,
                          binding.distribution, binding.globalShape,
                          binding.localShape, binding.rankSlices});
  for (const frontend::ProgramParameterBinding &binding : program.parameters)
    boundaries.push_back({binding.argumentIndex, binding.argumentIndex,
                          binding.distribution, binding.globalShape,
                          binding.localShape, binding.rankSlices});
  llvm::SmallVector<TypedBoundary, 8> verifiedBoundaries;
  for (TypedBoundary &boundary : boundaries) {
    std::string relationFailure;
    if (mlir::succeeded(verifyTypedBoundaryRelation(
            boundary, executionConfig.getRankCount(), &relationFailure)))
      verifiedBoundaries.push_back(std::move(boundary));
  }

  llvm::SmallVector<llvm::SmallVector<ResidentTuple, 8>, 16> generations;
  size_t workerPlacedTupleCount = 0;
  for (const ResidentSeed &seed : *seeds) {
    auto appendGeneration = [&](auto &&materialize) {
      std::string localFailure;
      size_t tentativeWorkerPlacedTupleCount = workerPlacedTupleCount;
      mlir::FailureOr<llvm::SmallVector<ResidentTuple, 8>> generation =
          buildResidentTuples(
              seed.candidates, executionConfig, tentativeWorkerPlacedTupleCount,
              std::forward<decltype(materialize)>(materialize), &localFailure);
      if (mlir::succeeded(generation)) {
        workerPlacedTupleCount = tentativeWorkerPlacedTupleCount;
        generations.push_back(std::move(*generation));
      }
    };

    for (NoCFanoutKind kind :
         {NoCFanoutKind::Direct, NoCFanoutKind::ReceiveForward}) {
      appendGeneration([&](llvm::MutableArrayRef<mlir::ModuleOp> modules) {
        int64_t communicationId = findNextCommunicationId(modules);
        if (communicationId < 0)
          return 0u;

        // Materializers consume and rewrite only the current all-rank IR.
        // Run them in dependency order on one discardable clone so a single
        // bounded generation can accumulate multiple DDR/compute cuts without
        // a role enum or shadow schedule:
        //
        //   existing collective cut -> required output -> intermediate
        //   producer handoff -> remaining boundary loads.
        //
        // Output routing runs before intermediate/boundary rewrites because
        // produced-value equivalence must be derived from the unmodified
        // producer SSA/effects. Partial keeps its typed collective identity.
        unsigned partial = materializeNoCPartialReductions(modules, program);
        unsigned output = materializeNoCOutputPublications(
            modules, program, communicationId, kind);
        unsigned intermediate = materializeNoCIntermediateHandoffs(
            modules, program, communicationId, kind);
        unsigned boundary = materializeTypedBoundaries(
            modules, verifiedBoundaries, communicationId, kind);
        unsigned fanoutMaterialized = output + intermediate + boundary;
        // If no fan-out/output role applied, discard this strategy-specific
        // clone. A single partial-only generation below preserves the
        // collective cut without creating Direct/Forward duplicates.
        return fanoutMaterialized == 0 ? 0u : partial + fanoutMaterialized;
      });
    }
    appendGeneration([&](llvm::MutableArrayRef<mlir::ModuleOp> modules) {
      // An explicit collective owns its typed communication identity. This
      // callback only removes the proven partial spill/reload cut.
      return materializeNoCPartialReductions(modules, program);
    });
  }
  if (generations.empty())
    return mlir::success();

  int64_t stableOrdinal = 0;
  for (const RankVariantFrontier &frontier : frontiers)
    for (const RankVariantCandidate &candidate : frontier) {
      if (candidate.stableOrdinal == std::numeric_limits<int64_t>::max())
        return mlir::success();
      stableOrdinal = std::max(stableOrdinal, candidate.stableOrdinal + 1);
    }

  // Validate the complete commit domain before moving even one rank module
  // into a frontier. No construction invariant may turn a later append into a
  // partial all-rank commit.
  for (const auto &generation : generations)
    for (const ResidentTuple &tuple : generation)
      if (tuple.modules.size() != frontiers.size())
        return fail(failureReason,
                    "NoC-resident candidate lost its complete rank domain");

  // One bounded composed strategy is one semantic generation. Its
  // single-buffer and fixed-slot realizations share the stable ordinal while
  // buffering metadata keeps their all-rank correspondence independent.
  for (auto &generation : generations) {
    if (stableOrdinal == std::numeric_limits<int64_t>::max())
      break;
    for (ResidentTuple &tuple : generation) {
      for (size_t rank = 0; rank < frontiers.size(); ++rank)
        frontiers[rank].push_back(
            {std::move(tuple.modules[rank]), stableOrdinal,
             wafer::RankArtifactKind::Resident,
             /*reservedBaseline=*/false, tuple.bufferingKind,
             tuple.bufferingPlanOrdinal, tuple.workerPlacementKind,
             tuple.workerPlacementPlanOrdinal});
    }
    ++stableOrdinal;
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
