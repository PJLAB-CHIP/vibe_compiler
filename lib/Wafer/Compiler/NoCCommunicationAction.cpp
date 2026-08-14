//===- NoCCommunicationAction.cpp - Typed NoC action provider ------------===//

#include "NoCCommunicationAction.h"

#include "NoCPartialDataflow.h"
#include "Wafer/Compiler/GlobalTileRelation.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/Common/OpVerifierUtils.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace wafer::compiler::detail {
namespace {

enum class NoCCommunicationRecipe : uint32_t {
  PartialOnly = 0,
  DirectFanout = 1,
  ReceiveForwardFanout = 2,
};

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

struct BoundaryFanoutPlan {
  size_t ownerIndex = 0;
  int64_t ownerRank = -1;
  int64_t bytes = 0;
  llvm::SmallVector<size_t, 16> forwardingOrder;
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

static llvm::SmallVector<TypedBoundary, 8> collectVerifiedTypedBoundaries(
    const frontend::FrontendProgramVerificationResult &program,
    int64_t rankCount) {
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

  llvm::SmallVector<TypedBoundary, 8> verified;
  verified.reserve(boundaries.size());
  for (TypedBoundary &boundary : boundaries)
    if (mlir::succeeded(
            verifyTypedBoundaryRelation(boundary, rankCount, nullptr)))
      verified.push_back(std::move(boundary));
  return verified;
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
  // Payload compatibility is validated atomically by the exact plan proof. It
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

static llvm::SmallVector<llvm::SmallVector<BoundaryTileLoad, 16>, 8>
collectBoundaryTileGroups(llvm::ArrayRef<mlir::ModuleOp> modules,
                          const TypedBoundary &boundary) {
  llvm::SmallVector<BoundaryTileLoad, 32> loads;
  bool ambiguous = false;
  for (auto [rank, module] : llvm::enumerate(modules))
    llvm::append_range(loads, findBoundaryTileLoads(module, boundary,
                                                    static_cast<int64_t>(rank),
                                                    ambiguous));
  if (loads.empty() || ambiguous)
    return {};

  // Form exact equivalence classes in rank-major IR walk order. LLVM semantic
  // hashes are process-seeded in supported builds, so group order and owner
  // selection do not depend on their numeric value.
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
  return tileGroups;
}

static size_t chooseTileOwnerOrdinal(const TypedBoundary &boundary,
                                     size_t tileGroupOrdinal,
                                     size_t ownerCount) {
  return (static_cast<size_t>(boundary.programIndex) + tileGroupOrdinal) %
         ownerCount;
}

static std::optional<BoundaryFanoutPlan>
proveBoundaryTileFanout(llvm::ArrayRef<BoundaryTileLoad> loads,
                        const TypedBoundary &boundary, size_t tileGroupOrdinal,
                        int64_t logicalRankCount, std::string *failureReason) {
  if (loads.size() < 2) {
    setFailureReason(failureReason,
                     "boundary tile has no verified cross-rank consumer reuse");
    return std::nullopt;
  }
  for (const BoundaryTileLoad &load : llvm::drop_begin(loads))
    if (!load.rankSlice || !loads.front().rankSlice ||
        compareRankBoundaryTileViews(
            loads.front().view, *loads.front().rankSlice, load.view,
            *load.rankSlice, boundary.globalShape,
            boundary.localShape) != StaticTileRelation::Equivalent) {
      setFailureReason(
          failureReason,
          "boundary tile fanout has mismatched typed view relations");
      return std::nullopt;
    }

  llvm::SmallVector<size_t, 16> sourceCoveredLoads;
  llvm::SmallVector<bool, 16> seenConsumers(
      static_cast<size_t>(logicalRankCount), false);
  for (auto [index, candidate] : llvm::enumerate(loads)) {
    if (candidate.logicalRank < 0 ||
        candidate.logicalRank >= logicalRankCount ||
        seenConsumers[static_cast<size_t>(candidate.logicalRank)]) {
      setFailureReason(failureReason,
                       "boundary tile consumer rank domain is not unique");
      return std::nullopt;
    }
    seenConsumers[static_cast<size_t>(candidate.logicalRank)] = true;
    if (!candidate.rankSlice) {
      setFailureReason(failureReason,
                       "boundary tile consumer lacks a typed rank slice");
      return std::nullopt;
    }
    if (candidate.view.staticLocalTile) {
      llvm::Expected<StaticTileRegion> global = mapRankLocalTileToGlobal(
          *candidate.rankSlice, boundary.globalShape, boundary.localShape,
          *candidate.view.staticLocalTile);
      if (!global) {
        llvm::consumeError(global.takeError());
        setFailureReason(
            failureReason,
            "static boundary tile lost its verified global relation");
        return std::nullopt;
      }
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
  if (sourceCoveredLoads.empty()) {
    setFailureReason(failureReason,
                     "boundary tile has no actual load with owner coverage");
    return std::nullopt;
  }

  BoundaryFanoutPlan plan;
  const size_t selectedOwner = chooseTileOwnerOrdinal(
      boundary, tileGroupOrdinal, sourceCoveredLoads.size());
  plan.ownerIndex = sourceCoveredLoads[selectedOwner];
  plan.ownerRank = loads[plan.ownerIndex].logicalRank;
  auto getByteCount = [](const BoundaryTileLoad &candidate) {
    return candidate.load->getAttrOfType<mlir::IntegerAttr>("byte_count")
        .getInt();
  };
  auto getDestType = [](const BoundaryTileLoad &candidate) {
    return candidate.load->getOperand(1).getType();
  };
  plan.bytes = getByteCount(loads[plan.ownerIndex]);
  for (const BoundaryTileLoad &candidate : loads)
    if (getByteCount(candidate) != plan.bytes ||
        getDestType(candidate) != getDestType(loads.front())) {
      setFailureReason(failureReason,
                       "boundary tile fanout has mismatched physical payloads");
      return std::nullopt;
    }

  plan.forwardingOrder.reserve(loads.size());
  plan.forwardingOrder.push_back(plan.ownerIndex);
  for (auto [index, candidate] : llvm::enumerate(loads))
    if (index != plan.ownerIndex)
      plan.forwardingOrder.push_back(index);
  llvm::sort(
      llvm::drop_begin(plan.forwardingOrder), [&](size_t lhs, size_t rhs) {
        const int64_t lhsDistance =
            (loads[lhs].logicalRank - plan.ownerRank + logicalRankCount) %
            logicalRankCount;
        const int64_t rhsDistance =
            (loads[rhs].logicalRank - plan.ownerRank + logicalRankCount) %
            logicalRankCount;
        return lhsDistance < rhsDistance;
      });
  return plan;
}

static mlir::LogicalResult materializeBoundaryTileFanout(
    llvm::ArrayRef<BoundaryTileLoad> loads, const TypedBoundary &boundary,
    const BoundaryFanoutPlan &plan, int64_t communicationId, NoCFanoutKind kind,
    std::string *failureReason) {
  if (plan.forwardingOrder.size() != loads.size() ||
      plan.ownerIndex >= loads.size() || plan.ownerRank < 0 || plan.bytes <= 0)
    return fail(failureReason, "boundary tile fanout plan is invalid");

  for (auto [orderIndex, loadIndex] : llvm::enumerate(plan.forwardingOrder)) {
    const BoundaryTileLoad &candidate = loads[loadIndex];
    const int64_t rank = candidate.logicalRank;
    InstrRDMAOp load = candidate.load;
    mlir::OpBuilder builder(load);
    auto message =
        DTEMessageAttr::get(load.getContext(), communicationId,
                            DTEProtocolPhase::PeerDataflow, /*round=*/0,
                            /*payloadSlice=*/boundary.programIndex);
    if (rank == plan.ownerRank) {
      builder.setInsertionPointAfter(load);
      if (kind == NoCFanoutKind::ReceiveForward) {
        const int64_t peer = loads[plan.forwardingOrder[1]].logicalRank;
        auto send = builder.create<CommPeerSendOp>(
            load.getLoc(), builder.getType<mlir::async::TokenType>(),
            load.getDest(), builder.getI64IntegerAttr(peer),
            builder.getI64IntegerAttr(plan.bytes), message);
        builder.create<mlir::async::AwaitOp>(load.getLoc(), send.getToken());
      } else {
        for (const BoundaryTileLoad &peerLoad : loads) {
          const int64_t peer = peerLoad.logicalRank;
          if (peer == plan.ownerRank)
            continue;
          auto send = builder.create<CommPeerSendOp>(
              load.getLoc(), builder.getType<mlir::async::TokenType>(),
              load.getDest(), builder.getI64IntegerAttr(peer),
              builder.getI64IntegerAttr(plan.bytes), message);
          builder.create<mlir::async::AwaitOp>(load.getLoc(), send.getToken());
        }
      }
      continue;
    }

    const int64_t source =
        kind == NoCFanoutKind::Direct
            ? plan.ownerRank
            : loads[plan.forwardingOrder[orderIndex - 1]].logicalRank;
    auto recv = builder.create<CommPeerRecvOp>(
        load.getLoc(), builder.getType<mlir::async::TokenType>(),
        load.getDest(), builder.getI64IntegerAttr(source),
        builder.getI64IntegerAttr(plan.bytes), message);
    auto await =
        builder.create<mlir::async::AwaitOp>(load.getLoc(), recv.getToken());
    if (kind == NoCFanoutKind::ReceiveForward &&
        orderIndex + 1 < plan.forwardingOrder.size()) {
      builder.setInsertionPointAfter(await);
      const int64_t peer =
          loads[plan.forwardingOrder[orderIndex + 1]].logicalRank;
      auto send = builder.create<CommPeerSendOp>(
          load.getLoc(), builder.getType<mlir::async::TokenType>(),
          load.getDest(), builder.getI64IntegerAttr(peer),
          builder.getI64IntegerAttr(plan.bytes), message);
      builder.create<mlir::async::AwaitOp>(load.getLoc(), send.getToken());
    }
    load.erase();
  }
  return mlir::success();
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

class NoCCommunicationActionPoint final
    : public CoordinatedCommunicationActionPoint {
public:
  NoCCommunicationActionPoint(llvm::StringRef providerKey,
                              NoCCommunicationRecipe recipe)
      : CoordinatedCommunicationActionPoint(
            {providerKey.str(), static_cast<uint32_t>(recipe)}),
        recipe(recipe) {}

  mlir::LogicalResult materialize(
      llvm::MutableArrayRef<mlir::ModuleOp> isolatedCanonicalInstrModules,
      const frontend::FrontendProgramVerificationResult &program,
      std::string *failureReason) const final;

private:
  NoCCommunicationRecipe recipe;
};

} // namespace

int64_t findNextNoCCommunicationId(llvm::ArrayRef<mlir::ModuleOp> rankModules) {
  int64_t maximum = -1;
  for (mlir::ModuleOp module : rankModules)
    module.walk([&](mlir::Operation *operation) {
      if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
        maximum = std::max(maximum, send.getMessage().getCommunicationId());
      else if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation))
        maximum = std::max(maximum, recv.getMessage().getCommunicationId());
      else if (auto send = mlir::dyn_cast<CommPeerSendOp>(operation))
        maximum = std::max(maximum, send.getMessage().getCommunicationId());
      else if (auto recv = mlir::dyn_cast<CommPeerRecvOp>(operation))
        maximum = std::max(maximum, recv.getMessage().getCommunicationId());
      else if (auto collective = mlir::dyn_cast<CommAllGatherOp>(operation))
        maximum =
            std::max(maximum, collective.getCommunicationIdAttr().getInt());
      else if (auto collective = mlir::dyn_cast<CommReduceScatterOp>(operation))
        maximum =
            std::max(maximum, collective.getCommunicationIdAttr().getInt());
      else if (auto collective = mlir::dyn_cast<CommAllReduceOp>(operation))
        maximum =
            std::max(maximum, collective.getCommunicationIdAttr().getInt());
    });
  return maximum == std::numeric_limits<int64_t>::max() ? -1 : maximum + 1;
}

bool hasNoCTypedBoundaryFanoutOpportunity(
    llvm::ArrayRef<mlir::ModuleOp> rankModules,
    const frontend::FrontendProgramVerificationResult &program) {
  if (rankModules.size() < 2 ||
      rankModules.size() != static_cast<size_t>(program.logicalRankCount))
    return false;
  const int64_t logicalRankCount = static_cast<int64_t>(rankModules.size());
  for (const TypedBoundary &boundary :
       collectVerifiedTypedBoundaries(program, logicalRankCount))
    for (auto [groupOrdinal, consumers] :
         llvm::enumerate(collectBoundaryTileGroups(rankModules, boundary)))
      if (consumers.size() >= 2 &&
          proveBoundaryTileFanout(consumers, boundary, groupOrdinal,
                                  logicalRankCount, nullptr))
        return true;
  return false;
}

unsigned materializeNoCTypedBoundaryFanouts(
    llvm::MutableArrayRef<mlir::ModuleOp> rankModules,
    const frontend::FrontendProgramVerificationResult &program,
    int64_t &communicationId, NoCFanoutKind kind) {
  wafer::support::ScopedCompileTimingSpan timing(
      "transformation", "materializeNoCTypedBoundaryFanouts", "total");
  if (rankModules.size() < 2 ||
      rankModules.size() != static_cast<size_t>(program.logicalRankCount))
    return 0;

  unsigned materialized = 0;
  const int64_t logicalRankCount = static_cast<int64_t>(rankModules.size());
  for (const TypedBoundary &boundary :
       collectVerifiedTypedBoundaries(program, logicalRankCount)) {
    auto tileGroups = collectBoundaryTileGroups(rankModules, boundary);
    for (auto [groupOrdinal, consumers] : llvm::enumerate(tileGroups)) {
      // A verified partitioned boundary normally contributes one consumer
      // load per unique global shard. That singleton is already the minimum
      // legal DDR coverage and is not a fanout opportunity.
      if (consumers.size() < 2 || communicationId < 0)
        continue;
      std::optional<BoundaryFanoutPlan> plan = proveBoundaryTileFanout(
          consumers, boundary, groupOrdinal, logicalRankCount, nullptr);
      if (!plan)
        continue;
      if (mlir::failed(materializeBoundaryTileFanout(
              consumers, boundary, *plan, communicationId, kind, nullptr)))
        continue;
      ++materialized;
      communicationId = communicationId == std::numeric_limits<int64_t>::max()
                            ? -1
                            : communicationId + 1;
    }
  }
  return materialized;
}

mlir::LogicalResult
normalizeNoCPeerReceivePreparation(llvm::ArrayRef<mlir::ModuleOp> rankModules,
                                   std::string *failureReason) {
  for (mlir::ModuleOp module : rankModules) {
    llvm::SmallPtrSet<mlir::Block *, 16> communicationBlocks;
    module.walk([&](mlir::Operation *operation) {
      if (mlir::isa<CommPeerSendOp, CommPeerRecvOp>(operation))
        communicationBlocks.insert(operation->getBlock());
    });
    if (communicationBlocks.empty())
      continue;

    // Preparation is moved only within its own structured occurrence. The
    // later Direct-DTE all-rank gate proves whole-program message progress;
    // block identity is not used as a surrogate for global acyclicity.
    for (mlir::Block *block : communicationBlocks) {
      mlir::Operation *firstTransportIssue = nullptr;
      llvm::SmallVector<CommPeerRecvOp, 8> receives;
      for (mlir::Operation &operation : *block) {
        // Existing lowered Direct-DTE traffic can already be present in a
        // partial-only seed. Every new receive preparation must precede all
        // transport issues in this occurrence, not only newly inserted peers.
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
            return fail(
                failureReason,
                "peer receive preparation crosses an unproved buffer effect");
        }

        mlir::Operation *definition = receive.getBuffer().getDefiningOp();
        if (!definition || definition->getBlock() != block ||
            definition->isBeforeInBlock(firstTransportIssue))
          continue;
        auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(definition);
        if (!allocation || !allocation.getType().hasStaticShape() ||
            allocation.getNumOperands() != 0)
          return fail(failureReason,
                      "peer receive lacks movable static buffer preparation");
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
  if (failureReason)
    failureReason->clear();
  return mlir::success();
}

mlir::LogicalResult NoCCommunicationActionPoint::materialize(
    llvm::MutableArrayRef<mlir::ModuleOp> isolatedCanonicalInstrModules,
    const frontend::FrontendProgramVerificationResult &program,
    std::string *failureReason) const {
  if (mlir::failed(verifyCoordinatedCommunicationActionDomain(
          isolatedCanonicalInstrModules, program, failureReason)))
    return mlir::failure();

  unsigned materialized = 0;
  if (recipe == NoCCommunicationRecipe::PartialOnly) {
    materialized =
        materializeNoCPartialReductions(isolatedCanonicalInstrModules, program);
  } else {
    int64_t communicationId =
        findNextNoCCommunicationId(isolatedCanonicalInstrModules);
    if (communicationId < 0)
      return fail(failureReason, "NoC communication identity space exhausted");

    // Every composed fanout recipe follows the same current-IR dependency
    // order. Each exact helper re-proves its own capability on this one
    // discardable complete-rank clone.
    const NoCFanoutKind kind = recipe == NoCCommunicationRecipe::DirectFanout
                                   ? NoCFanoutKind::Direct
                                   : NoCFanoutKind::ReceiveForward;
    const unsigned partial =
        materializeNoCPartialReductions(isolatedCanonicalInstrModules, program);
    const unsigned output = materializeNoCOutputPublications(
        isolatedCanonicalInstrModules, program, communicationId, kind);
    const unsigned intermediate = materializeNoCIntermediateHandoffs(
        isolatedCanonicalInstrModules, program, communicationId, kind);
    const unsigned boundary = materializeNoCTypedBoundaryFanouts(
        isolatedCanonicalInstrModules, program, communicationId, kind);
    const unsigned fanout = output + intermediate + boundary;
    materialized = fanout == 0 ? 0 : partial + fanout;
  }
  if (materialized == 0)
    return fail(failureReason,
                "NoC action point no longer matches its canonical parent");

  if (mlir::failed(normalizeNoCPeerReceivePreparation(
          isolatedCanonicalInstrModules, failureReason)))
    return mlir::failure();
  for (mlir::ModuleOp module : isolatedCanonicalInstrModules) {
    if (wafer::containsTileDataflowOperations(module.getOperation()) &&
        mlir::failed(wafer::convertTileRegionToInstrModule(module)))
      return fail(failureReason, "NoC peer lowering failed");
    if (wafer::containsTileDataflowOperations(module.getOperation()) ||
        mlir::failed(wafer::placeRequiredNCCJoins(module)) ||
        mlir::failed(mlir::verify(module)))
      return fail(failureReason,
                  "NoC action did not produce normalized canonical Instr IR");
  }

  if (failureReason)
    failureReason->clear();
  return mlir::success();
}

llvm::StringRef NoCCommunicationActionProvider::getStableKey() const {
  return "typed-noc-dataflow";
}

mlir::LogicalResult NoCCommunicationActionProvider::query(
    llvm::ArrayRef<mlir::ModuleOp> currentCanonicalInstrModules,
    const frontend::FrontendProgramVerificationResult &program,
    CoordinatedCommunicationActionPoints &points,
    std::string *failureReason) const {
  if (mlir::failed(verifyCoordinatedCommunicationActionDomain(
          currentCanonicalInstrModules, program, failureReason)))
    return mlir::failure();
  if (currentCanonicalInstrModules.size() <= 1) {
    if (failureReason)
      failureReason->clear();
    return mlir::success();
  }

  const bool hasPartial =
      hasNoCPartialReductionOpportunity(currentCanonicalInstrModules, program);
  const bool hasFanout =
      findNextNoCCommunicationId(currentCanonicalInstrModules) >= 0 &&
      (hasNoCOutputPublicationOpportunity(currentCanonicalInstrModules,
                                          program) ||
       hasNoCIntermediateHandoffOpportunity(currentCanonicalInstrModules,
                                            program) ||
       hasNoCTypedBoundaryFanoutOpportunity(currentCanonicalInstrModules,
                                            program));
  if (hasPartial)
    points.push_back(std::make_unique<NoCCommunicationActionPoint>(
        getStableKey(), NoCCommunicationRecipe::PartialOnly));
  if (hasFanout) {
    points.push_back(std::make_unique<NoCCommunicationActionPoint>(
        getStableKey(), NoCCommunicationRecipe::DirectFanout));
    points.push_back(std::make_unique<NoCCommunicationActionPoint>(
        getStableKey(), NoCCommunicationRecipe::ReceiveForwardFanout));
  }

  if (failureReason)
    failureReason->clear();
  return mlir::success();
}

} // namespace wafer::compiler::detail
