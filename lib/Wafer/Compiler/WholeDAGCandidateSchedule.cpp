//===- WholeDAGCandidateSchedule.cpp - Candidate event scheduling ------===//

#include "WholeDAGCandidateSchedule.h"

#include "Wafer/Support/TargetPolicy.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

void setFailure(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
}

uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  return product > std::numeric_limits<uint64_t>::max()
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(product);
}

uint64_t getStaticElementWork(const CardDAGNode &node) {
  if (auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(
          node.operation)) {
    llvm::SmallVector<int64_t, 4> ranges = linalg.getStaticLoopRanges();
    if (!ranges.empty() &&
        llvm::all_of(ranges, [](int64_t extent) { return extent > 0; })) {
      uint64_t iteratorWork = 1;
      for (int64_t extent : ranges)
        iteratorWork =
            saturatingMultiply(iteratorWork, static_cast<uint64_t>(extent));
      return iteratorWork;
    }
  }
  uint64_t work = 1;
  bool foundTensor = false;
  for (mlir::Type type : node.operation->getResultTypes()) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape())
      continue;
    uint64_t elements = 1;
    for (int64_t extent : tensor.getShape()) {
      if (extent <= 0)
        return 1;
      elements = saturatingMultiply(elements, static_cast<uint64_t>(extent));
    }
    work = std::max(work, elements);
    foundTensor = true;
  }
  return foundTensor ? work : 1;
}

WholeDAGTime getWaveDuration(const CardDAGNode &node, SymbolicWaveKind kind,
                             size_t participants) {
  const uint64_t work = getStaticElementWork(node);
  const uint64_t parallel = std::max<size_t>(1, participants);
  const uint64_t steady = work / parallel + (work % parallel != 0);
  switch (kind) {
  case SymbolicWaveKind::Prologue:
  case SymbolicWaveKind::Tail:
    return 1;
  case SymbolicWaveKind::Steady:
    return std::max<uint64_t>(1, steady);
  }
  llvm_unreachable("unknown symbolic wave kind");
}

bool tileSetsIntersect(llvm::ArrayRef<PhysicalTileId> lhs,
                       llvm::ArrayRef<PhysicalTileId> rhs) {
  for (PhysicalTileId left : lhs)
    if (llvm::is_contained(rhs, left))
      return true;
  return false;
}

} // namespace

mlir::FailureOr<WholeDAGCandidateSchedule> scheduleWholeDAGCandidate(
    const CardDAGAnalysis &dag, llvm::ArrayRef<PhysicalTileId> availableTiles,
    llvm::ArrayRef<WholeDAGNodePlacement> requestedPlacements,
    llvm::ArrayRef<WholeDAGLocalResidency> localResidencies,
    std::string *failureReason,
    llvm::ArrayRef<WholeDAGPeerMovement> requestedPeerMovements,
    llvm::ArrayRef<WholeDAGLocalMovement> requestedLocalMovements,
    bool enforceSPMCapacity) {
  if (failureReason)
    failureReason->clear();
  if (requestedPlacements.size() != dag.getNodes().size()) {
    setFailure(failureReason,
               "whole-DAG candidate must place every structured node once");
    return mlir::failure();
  }

  llvm::DenseSet<int64_t> available;
  for (PhysicalTileId tile : availableTiles) {
    if (!available.insert(tile.getValue()).second) {
      setFailure(failureReason,
                 "whole-DAG available physical Tile domain is duplicated");
      return mlir::failure();
    }
  }

  WholeDAGCandidateSchedule result;
  result.nodePlacements.resize(dag.getNodes().size());
  llvm::BitVector seenNodes(dag.getNodes().size());
  for (const WholeDAGNodePlacement &requested : requestedPlacements) {
    if (requested.node >= dag.getNodes().size() ||
        seenNodes.test(requested.node)) {
      setFailure(failureReason,
                 "whole-DAG candidate node placement is unknown or duplicated");
      return mlir::failure();
    }
    if (requested.tiles.empty()) {
      setFailure(failureReason,
                 "whole-DAG candidate node placement has no physical Tiles");
      return mlir::failure();
    }
    WholeDAGNodePlacement placement = requested;
    llvm::sort(placement.tiles, [](PhysicalTileId lhs, PhysicalTileId rhs) {
      return lhs.getValue() < rhs.getValue();
    });
    if (std::adjacent_find(placement.tiles.begin(), placement.tiles.end()) !=
        placement.tiles.end()) {
      setFailure(failureReason,
                 "whole-DAG candidate node placement contains duplicate Tiles");
      return mlir::failure();
    }
    for (PhysicalTileId tile : placement.tiles) {
      if (!available.contains(tile.getValue())) {
        setFailure(failureReason,
                   "whole-DAG candidate references an unavailable Tile");
        return mlir::failure();
      }
    }
    seenNodes.set(requested.node);
    result.nodePlacements[requested.node] = std::move(placement);
  }
  if (!seenNodes.all()) {
    setFailure(failureReason,
               "whole-DAG candidate does not cover every structured node");
    return mlir::failure();
  }

  llvm::SmallVector<WholeDAGPeerMovement, 32> peerMovements(
      requestedPeerMovements.begin(), requestedPeerMovements.end());
  for (const WholeDAGPeerMovement &movement : peerMovements) {
    const CardDAGEdge *edge = dag.getEdge(movement.edge);
    if (!edge || movement.duration == 0 || movement.bufferCount == 0 ||
        movement.bufferCount > 3 ||
        movement.sourceTile == movement.destinationTile ||
        movement.route.empty() ||
        movement.route.front().source != movement.sourceTile ||
        movement.route.back().destination != movement.destinationTile ||
        llvm::any_of(llvm::enumerate(movement.route), [&](auto indexed) {
          return indexed.index() != 0 &&
                 movement.route[indexed.index() - 1].destination !=
                     indexed.value().source;
        }) ||
        !available.contains(movement.sourceTile.getValue()) ||
        !available.contains(movement.destinationTile.getValue()) ||
        !llvm::is_contained(result.nodePlacements[edge->producer].tiles,
                            movement.sourceTile) ||
        !llvm::is_contained(result.nodePlacements[edge->consumer].tiles,
                            movement.destinationTile)) {
      setFailure(failureReason,
                 "whole-DAG peer movement is invalid for its edge placement");
      return mlir::failure();
    }
  }
  llvm::sort(peerMovements, [](const WholeDAGPeerMovement &lhs,
                               const WholeDAGPeerMovement &rhs) {
    return std::tuple(lhs.edge, lhs.sourceTile.getValue(),
                      lhs.destinationTile.getValue(), lhs.duration,
                      lhs.bufferCount) <
           std::tuple(rhs.edge, rhs.sourceTile.getValue(),
                      rhs.destinationTile.getValue(), rhs.duration,
                      rhs.bufferCount);
  });

  llvm::SmallVector<WholeDAGLocalMovement, 32> localMovements(
      requestedLocalMovements.begin(), requestedLocalMovements.end());
  for (const WholeDAGLocalMovement &movement : localMovements) {
    const CardDAGEdge *edge = dag.getEdge(movement.edge);
    if (!edge || movement.duration == 0 || movement.bufferCount == 0 ||
        movement.bufferCount > 3 ||
        !available.contains(movement.tile.getValue()) ||
        (!llvm::is_contained(result.nodePlacements[edge->producer].tiles,
                             movement.tile) &&
         !llvm::is_contained(result.nodePlacements[edge->consumer].tiles,
                             movement.tile))) {
      setFailure(failureReason,
                 "whole-DAG local movement is invalid for its edge placement");
      return mlir::failure();
    }
  }
  llvm::sort(localMovements, [](const WholeDAGLocalMovement &lhs,
                                const WholeDAGLocalMovement &rhs) {
    return std::tuple(lhs.edge, static_cast<uint8_t>(lhs.resource),
                      lhs.tile.getValue(), lhs.duration, lhs.bufferCount) <
           std::tuple(rhs.edge, static_cast<uint8_t>(rhs.resource),
                      rhs.tile.getValue(), rhs.duration, rhs.bufferCount);
  });

  std::set<std::pair<CardDAGEdgeID, int64_t>> residencyKeys;
  for (const WholeDAGLocalResidency &residency : localResidencies) {
    const CardDAGEdge *edge = dag.getEdge(residency.edge);
    if (!edge || residency.footprintBytes == 0 ||
        !available.contains(residency.tile.getValue()) ||
        !residencyKeys.insert({residency.edge, residency.tile.getValue()})
             .second) {
      setFailure(failureReason,
                 "whole-DAG local residency is invalid or duplicated");
      return mlir::failure();
    }
    if (!llvm::is_contained(result.nodePlacements[edge->producer].tiles,
                            residency.tile) ||
        !llvm::is_contained(result.nodePlacements[edge->consumer].tiles,
                            residency.tile)) {
      setFailure(failureReason,
                 "whole-DAG local residency Tile is not shared by its edge");
      return mlir::failure();
    }
  }

  const TargetMemoryPolicy memory = getDefaultWaferTargetPolicy().memory;
  if (memory.spmBase < 0 || memory.spmLimit <= memory.spmBase) {
    setFailure(failureReason,
               "whole-DAG scheduler has an invalid target SPM capacity");
    return mlir::failure();
  }
  const uint64_t spmCapacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);

  mlir::FailureOr<WholeDAGScheduleState> state =
      WholeDAGScheduleState::create(dag, availableTiles, failureReason);
  if (mlir::failed(state))
    return mlir::failure();

  std::map<std::pair<int64_t, int64_t>, WholeDAGTime> linkAvailableTimes;
  std::map<int64_t, WholeDAGTime> spmAvailableTimes;
  WholeDAGTime ddrAvailableTime = 0;
  llvm::SmallVector<WholeDAGTime, 32> pendingMovementFinishes;

  auto delayProducerSlotReuse = [&](const CardDAGEdge &edge,
                                    SymbolicWaveClass completed,
                                    WholeDAGTime finish,
                                    uint8_t bufferCount) -> mlir::LogicalResult {
    if (bufferCount != 1 || completed.kind == SymbolicWaveKind::Tail)
      return mlir::success();
    SymbolicWaveKind next = completed.kind == SymbolicWaveKind::Prologue
                                ? SymbolicWaveKind::Steady
                                : SymbolicWaveKind::Tail;
    return state->delayWaveReadinessUntil(
        SymbolicWaveClass{edge.producer, next}, finish, failureReason);
  };

  // At one event, greedily dispatch the stable ready order whenever its
  // placement is disjoint from work already running at this same event.  A
  // skipped wave remains Ready and is reconsidered after the next completion.
  while (!state->isComplete()) {
    bool dispatched = false;
    llvm::SmallVector<SymbolicWaveClass, 8> ready = state->getReadyWaves();
    for (SymbolicWaveClass wave : ready) {
      const WholeDAGNodePlacement &placement = result.nodePlacements[wave.node];
      bool conflicts = llvm::any_of(
          state->getRunningWaves(), [&](const RunningOpWave &running) {
            return tileSetsIntersect(placement.tiles, running.tiles);
          });
      if (conflicts)
        continue;
      const CardDAGNode *node = dag.getNode(wave.node);
      if (!node) {
        setFailure(failureReason,
                   "whole-DAG ready wave references an unknown node");
        return mlir::failure();
      }
      WholeDAGTime duration =
          getWaveDuration(*node, wave.kind, placement.tiles.size());
      const WholeDAGTime start = state->getCurrentTime();
      if (mlir::failed(state->dispatchReadyWave(wave, placement.tiles, duration,
                                                failureReason)))
        return mlir::failure();
      RunningOpWave dispatchedWave;
      dispatchedWave.wave = wave;
      dispatchedWave.tiles.append(placement.tiles.begin(),
                                  placement.tiles.end());
      dispatchedWave.startTime = start;
      dispatchedWave.finishTime = start + duration;
      result.dispatchedWaves.push_back(std::move(dispatchedWave));
      for (PhysicalTileId tile : placement.tiles)
        result.resourceReservations.push_back(WholeDAGResourceReservation{
            WholeDAGReservationResource::Compute,
            std::numeric_limits<CardDAGEdgeID>::max(), tile, tile, start,
            start + duration});
      dispatched = true;
    }

    pendingMovementFinishes.erase(
        llvm::remove_if(pendingMovementFinishes, [&](WholeDAGTime finish) {
          return finish <= state->getCurrentTime();
        }),
        pendingMovementFinishes.end());
    std::optional<WholeDAGTime> nextCompute;
    if (!state->getRunningWaves().empty())
      nextCompute = state->getRunningWaves().front().finishTime;
    std::optional<WholeDAGTime> nextMovement;
    if (!pendingMovementFinishes.empty())
      nextMovement = *llvm::min_element(pendingMovementFinishes);
    if (!nextCompute && !nextMovement) {
      setFailure(failureReason,
                 dispatched
                     ? "whole-DAG dispatch produced no running wave"
                     : "whole-DAG candidate reached a scheduler deadlock");
      return mlir::failure();
    }
    const bool advanceMovement =
        nextMovement && (!nextCompute || *nextMovement < *nextCompute);
    mlir::FailureOr<WholeDAGScheduleEvent> event =
        advanceMovement
            ? state->advanceToExternalEvent(*nextMovement, failureReason)
            : state->advanceToNextEvent(failureReason);
    if (mlir::failed(event))
      return mlir::failure();
    ++result.eventCount;
    result.makespan = event->time;
    if (advanceMovement) {
      ++result.movementEventCount;
      continue;
    }

    // A remote fragment starts after its producer wave and the prior shared
    // NoC transaction. It does not occupy compute Tiles, so unrelated running
    // work may overlap. The consumer wave remains invisible to dispatch until
    // every incoming fragment's data-ready event has completed.
    for (SymbolicWaveClass completed : event->completedWaves) {
      const CardDAGNode *producer = dag.getNode(completed.node);
      if (!producer)
        return mlir::failure();
      for (const WholeDAGPeerMovement &movement : peerMovements) {
        const CardDAGEdge *edge = dag.getEdge(movement.edge);
        if (!edge || edge->producer != completed.node)
          continue;
        WholeDAGTime start = event->time;
        for (const PhysicalTileDirectedLink &link : movement.route)
          start = std::max(
              start,
              linkAvailableTimes[{link.source.getValue(),
                                  link.destination.getValue()}]);
        if (movement.duration >
            std::numeric_limits<WholeDAGTime>::max() - start) {
          setFailure(failureReason,
                     "whole-DAG peer movement finish time overflows");
          return mlir::failure();
        }
        const WholeDAGTime finish = start + movement.duration;
        for (const PhysicalTileDirectedLink &link : movement.route) {
          linkAvailableTimes[{link.source.getValue(),
                              link.destination.getValue()}] = finish;
          result.resourceReservations.push_back(WholeDAGResourceReservation{
              WholeDAGReservationResource::NoCLink, movement.edge,
              link.source, link.destination, start, finish});
        }
        pendingMovementFinishes.push_back(finish);
        result.peerMovementWork = saturatingMultiply(
            movement.duration, movement.route.size()) >
                    std::numeric_limits<uint64_t>::max() -
                        result.peerMovementWork
                ? std::numeric_limits<uint64_t>::max()
                : result.peerMovementWork +
                      saturatingMultiply(movement.duration,
                                         movement.route.size());
        if (mlir::failed(state->delayWaveReadinessUntil(
                SymbolicWaveClass{edge->consumer, completed.kind}, finish,
                failureReason)))
          return mlir::failure();
        if (mlir::failed(delayProducerSlotReuse(
                *edge, completed, finish, movement.bufferCount)))
          return mlir::failure();
      }

      for (const WholeDAGLocalMovement &movement : localMovements) {
        const CardDAGEdge *edge = dag.getEdge(movement.edge);
        if (!edge || edge->producer != completed.node)
          continue;
        WholeDAGTime start = event->time;
        if (movement.resource == WholeDAGLocalMovementResource::TileSPM)
          start = std::max(start,
                           spmAvailableTimes[movement.tile.getValue()]);
        else
          start = std::max(start, ddrAvailableTime);
        if (movement.duration >
            std::numeric_limits<WholeDAGTime>::max() - start) {
          setFailure(failureReason,
                     "whole-DAG local movement finish time overflows");
          return mlir::failure();
        }
        const WholeDAGTime finish = start + movement.duration;
        if (movement.resource == WholeDAGLocalMovementResource::TileSPM) {
          spmAvailableTimes[movement.tile.getValue()] = finish;
          result.spmMovementWork =
              movement.duration >
                      std::numeric_limits<uint64_t>::max() -
                          result.spmMovementWork
                  ? std::numeric_limits<uint64_t>::max()
                  : result.spmMovementWork + movement.duration;
        } else {
          ddrAvailableTime = finish;
          result.ddrMovementWork =
              movement.duration >
                      std::numeric_limits<uint64_t>::max() -
                          result.ddrMovementWork
                  ? std::numeric_limits<uint64_t>::max()
                  : result.ddrMovementWork + movement.duration;
        }
        result.resourceReservations.push_back(WholeDAGResourceReservation{
            movement.resource == WholeDAGLocalMovementResource::TileSPM
                ? WholeDAGReservationResource::TileSPM
                : WholeDAGReservationResource::CardDDR,
            movement.edge, movement.tile, movement.tile, start, finish});
        pendingMovementFinishes.push_back(finish);
        if (mlir::failed(state->delayWaveReadinessUntil(
                SymbolicWaveClass{edge->consumer, completed.kind}, finish,
                failureReason)) ||
            mlir::failed(delayProducerSlotReuse(
                *edge, completed, finish, movement.bufferCount)))
          return mlir::failure();
      }
    }

    // Publish all values produced at this event before releasing values whose
    // consumers complete at the same event.  This preserves the conservative
    // instantaneous high-water when two class buffers exchange ownership.
    for (SymbolicWaveClass completed : event->completedWaves) {
      for (const WholeDAGLocalResidency &residency : localResidencies) {
        const CardDAGEdge *edge = dag.getEdge(residency.edge);
        if (!edge || edge->producer != completed.node)
          continue;
        if (mlir::failed(state->retainLocalEdgeValue(
                residency.tile, residency.edge, completed.kind,
                residency.footprintBytes, failureReason)))
          return mlir::failure();
      }
    }
    for (const PhysicalTileScheduleState &tileState : state->getTileStates()) {
      uint64_t liveBytes = 0;
      for (const LiveSPMEntry &entry : tileState.liveSPM) {
        if (entry.footprintBytes >
            std::numeric_limits<uint64_t>::max() - liveBytes) {
          setFailure(failureReason, "whole-DAG local SPM high-water overflows");
          return mlir::failure();
        }
        liveBytes += entry.footprintBytes;
      }
      result.peakLiveSPMBytes = std::max(result.peakLiveSPMBytes, liveBytes);
      if (enforceSPMCapacity && liveBytes > spmCapacity) {
        setFailure(failureReason,
                   "whole-DAG local SPM high-water exceeds capacity");
        return mlir::failure();
      }
    }
    for (SymbolicWaveClass completed : event->completedWaves) {
      for (const WholeDAGLocalResidency &residency : localResidencies) {
        const CardDAGEdge *edge = dag.getEdge(residency.edge);
        if (!edge || edge->consumer != completed.node)
          continue;
        if (mlir::failed(state->releaseLocalEdgeValue(
                residency.tile, residency.edge, completed.kind, failureReason)))
          return mlir::failure();
      }
    }
  }

  if (llvm::any_of(state->getTileStates(), [](const auto &tileState) {
        return !tileState.liveSPM.empty();
      })) {
    setFailure(failureReason,
               "whole-DAG schedule completed with live local SPM values");
    return mlir::failure();
  }

  return result;
}

} // namespace wafer::compiler::detail
