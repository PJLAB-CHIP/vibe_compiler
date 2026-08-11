//===- WholeDAGSchedule.h - Whole-card DAG scheduler core -----*- C++ -*-===//

#pragma once

#include "Wafer/Target/PhysicalIds.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace wafer::compiler::detail {

using CardDAGNodeID = uint32_t;
using CardDAGEdgeID = uint32_t;
using CardDAGComponentID = uint32_t;
using WholeDAGTime = uint64_t;

enum class SymbolicWaveKind : uint8_t {
  Prologue = 0,
  Steady = 1,
  Tail = 2,
};

llvm::StringRef stringifySymbolicWaveKind(SymbolicWaveKind kind);

/// One finite temporal class for an operation.  A steady class represents the
/// repeated steady-state pattern; it is not expanded into one record per loop
/// iteration or temporal tile.
struct SymbolicWaveClass {
  CardDAGNodeID node = 0;
  SymbolicWaveKind kind = SymbolicWaveKind::Prologue;

  uint64_t getStableOrdinal() const;

  friend bool operator==(const SymbolicWaveClass &lhs,
                         const SymbolicWaveClass &rhs) {
    return lhs.node == rhs.node && lhs.kind == rhs.kind;
  }
  friend bool operator!=(const SymbolicWaveClass &lhs,
                         const SymbolicWaveClass &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const SymbolicWaveClass &lhs,
                        const SymbolicWaveClass &rhs) {
    return lhs.getStableOrdinal() < rhs.getStableOrdinal();
  }
};

struct CardDAGNode {
  CardDAGNodeID id = 0;
  mlir::Operation *operation = nullptr;
  std::array<SymbolicWaveClass, 3> waveClasses;
  llvm::SmallVector<CardDAGEdgeID, 4> incomingEdges;
  llvm::SmallVector<CardDAGEdgeID, 4> outgoingEdges;
};

/// A stable SSA dependency between two direct structured operations.  Pure
/// top-level support operations may be traversed between the producer result
/// and consumer operand, but they never become hidden scheduler nodes.
struct CardDAGEdge {
  CardDAGEdgeID id = 0;
  CardDAGNodeID producer = 0;
  uint32_t producerResult = 0;
  CardDAGNodeID consumer = 0;
  uint32_t consumerOperand = 0;
};

/// One SSA-connected structured dependency component that reaches one or more
/// observable function results.  Components are numbered by their smallest
/// node ID and carry output indices in function-result order.  They are a
/// query-local analysis result, not a selected placement or serialized plan.
struct CardDAGDependencyComponent {
  CardDAGComponentID id = 0;
  llvm::SmallVector<CardDAGNodeID, 4> nodes;
  llvm::SmallVector<uint32_t, 2> observableOutputs;
};

/// Query-local analysis of one defined single-block structured function.
/// Node IDs follow direct structured-operation order.  Edge IDs follow a
/// lexicographic order over producer/result/consumer/operand, so neither ID
/// depends on pointer values or hash-table iteration.  Any IR mutation
/// invalidates the analysis.
class CardDAGAnalysis {
public:
  static mlir::FailureOr<CardDAGAnalysis>
  create(mlir::func::FuncOp function, std::string *failureReason = nullptr);

  mlir::func::FuncOp getFunction() const { return function; }
  llvm::ArrayRef<CardDAGNode> getNodes() const { return nodes; }
  llvm::ArrayRef<CardDAGEdge> getEdges() const { return edges; }
  llvm::ArrayRef<CardDAGDependencyComponent>
  getObservableDependencyComponents() const {
    return observableComponents;
  }
  /// Nearest structured roots for each function result, in function-result
  /// order.  More than one root is possible when a pure support operation
  /// combines values.  An empty entry means that placement cannot be recovered
  /// from the current structured SSA and must fail closed.
  llvm::ArrayRef<llvm::SmallVector<CardDAGNodeID, 2>>
  getObservableOutputRootNodes() const {
    return observableOutputRootNodes;
  }
  /// True only when tensor SSA and effect contracts prove that the observable
  /// components may execute independently.  A false result keeps the normal
  /// joint mapping; it is not a whole-program legality failure.
  bool supportsIndependentComponentPlacement() const {
    return independentComponentPlacement;
  }
  const CardDAGNode *getNode(CardDAGNodeID id) const;
  const CardDAGEdge *getEdge(CardDAGEdgeID id) const;

private:
  mlir::func::FuncOp function;
  llvm::SmallVector<CardDAGNode, 16> nodes;
  llvm::SmallVector<CardDAGEdge, 32> edges;
  llvm::SmallVector<CardDAGDependencyComponent, 4> observableComponents;
  llvm::SmallVector<llvm::SmallVector<CardDAGNodeID, 2>, 4>
      observableOutputRootNodes;
  bool independentComponentPlacement = false;
};

enum class DAGScheduleStatus : uint8_t {
  Waiting,
  Ready,
  Running,
  Completed,
};

/// Minimal search-time SPM residency fact.  It is deliberately keyed by a DAG
/// edge, finite wave class and physical Tile, and never claims a selected
/// allocation or offset.
struct LiveSPMEntry {
  CardDAGEdgeID edge = 0;
  SymbolicWaveKind waveKind = SymbolicWaveKind::Prologue;
  uint64_t footprintBytes = 0;
};

struct PhysicalTileScheduleState {
  PhysicalTileId tile{0};
  WholeDAGTime availableTime = 0;
  std::optional<SymbolicWaveClass> runningWave;
  llvm::SmallVector<LiveSPMEntry, 4> liveSPM;
};

struct RunningOpWave {
  SymbolicWaveClass wave;
  llvm::SmallVector<PhysicalTileId, 4> tiles;
  WholeDAGTime startTime = 0;
  WholeDAGTime finishTime = 0;
};

struct WholeDAGScheduleEvent {
  WholeDAGTime time = 0;
  llvm::SmallVector<SymbolicWaveClass, 4> completedWaves;
  llvm::SmallVector<SymbolicWaveClass, 4> readyWaves;
};

/// Copyable query-local event state.  Dispatch does not advance time, allowing
/// independent waves to be placed on disjoint Tile sets at the same event.
/// advanceToNextEvent completes all waves at the earliest finish time before
/// publishing newly-ready classes.  Within one node the dependency order is
/// prologue -> steady -> tail.  Across every DAG edge, data readiness is
/// conservatively class-matched (producer prologue -> consumer prologue,
/// steady -> steady, tail -> tail), so a consumer pipeline does not wait for
/// the producer's entire node to finish.
class WholeDAGScheduleState {
public:
  static mlir::FailureOr<WholeDAGScheduleState>
  create(const CardDAGAnalysis &dag, llvm::ArrayRef<PhysicalTileId> tiles,
         std::string *failureReason = nullptr);

  WholeDAGTime getCurrentTime() const { return currentTime; }
  llvm::ArrayRef<PhysicalTileScheduleState> getTileStates() const {
    return tileStates;
  }
  llvm::ArrayRef<RunningOpWave> getRunningWaves() const { return runningWaves; }

  const PhysicalTileScheduleState *getTileState(PhysicalTileId tile) const;
  DAGScheduleStatus getWaveStatus(SymbolicWaveClass wave) const;
  DAGScheduleStatus getNodeStatus(CardDAGNodeID node) const;
  llvm::SmallVector<SymbolicWaveClass, 8> getReadyWaves() const;
  bool isComplete() const;

  mlir::LogicalResult dispatchReadyWave(SymbolicWaveClass wave,
                                        llvm::ArrayRef<PhysicalTileId> tiles,
                                        WholeDAGTime duration,
                                        std::string *failureReason = nullptr);

  mlir::FailureOr<WholeDAGScheduleEvent>
  advanceToNextEvent(std::string *failureReason = nullptr);

  /// Delays a dependency-satisfied wave until an external movement/data-ready
  /// event. This does not create a second dependency graph: it only adds a
  /// finite not-before time to the current wave state.
  mlir::LogicalResult delayWaveReadinessUntil(
      SymbolicWaveClass wave, WholeDAGTime time,
      std::string *failureReason = nullptr);

  /// Advances to an external event strictly before the next running compute
  /// completion (or while no compute is running). Running Tile work remains in
  /// flight, allowing communication and independent compute to overlap.
  mlir::FailureOr<WholeDAGScheduleEvent>
  advanceToExternalEvent(WholeDAGTime time,
                         std::string *failureReason = nullptr);

  mlir::LogicalResult
  retainLocalEdgeValue(PhysicalTileId tile, CardDAGEdgeID edge,
                       SymbolicWaveKind waveKind, uint64_t footprintBytes,
                       std::string *failureReason = nullptr);

  mlir::LogicalResult
  releaseLocalEdgeValue(PhysicalTileId tile, CardDAGEdgeID edge,
                        SymbolicWaveKind waveKind,
                        std::string *failureReason = nullptr);

private:
  struct WaveState {
    DAGScheduleStatus status = DAGScheduleStatus::Waiting;
    uint32_t unsatisfiedDependencies = 0;
    WholeDAGTime notBeforeTime = 0;
  };

  struct NodeState {
    std::array<WaveState, 3> waves;
  };

  WholeDAGScheduleState(const CardDAGAnalysis &dag,
                        llvm::SmallVector<PhysicalTileScheduleState, 16> tiles,
                        llvm::SmallVector<NodeState, 16> nodes)
      : dag(&dag), tileStates(std::move(tiles)), nodeStates(std::move(nodes)) {}

  PhysicalTileScheduleState *findTileState(PhysicalTileId tile);
  const NodeState *getNodeState(CardDAGNodeID node) const;
  NodeState *getNodeState(CardDAGNodeID node);
  WaveState *getWaveState(SymbolicWaveClass wave);
  const WaveState *getWaveState(SymbolicWaveClass wave) const;
  void refreshReady(SymbolicWaveClass wave);

  const CardDAGAnalysis *dag = nullptr;
  WholeDAGTime currentTime = 0;
  llvm::SmallVector<PhysicalTileScheduleState, 16> tileStates;
  llvm::SmallVector<NodeState, 16> nodeStates;
  llvm::SmallVector<RunningOpWave, 16> runningWaves;
};

} // namespace wafer::compiler::detail
