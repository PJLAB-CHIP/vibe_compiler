//===- StructuredDAGSchedule.h - Card DAG scheduler core -----*- C++ -*-===//

#pragma once

#include "Wafer/Target/TopologyIds.h"

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

using StructuredDAGNodeID = uint32_t;
using StructuredDAGEdgeID = uint32_t;
using StructuredDAGComponentID = uint32_t;
using ScheduleTime = uint64_t;

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
  StructuredDAGNodeID node = 0;
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

struct StructuredDAGNode {
  StructuredDAGNodeID id = 0;
  mlir::Operation *operation = nullptr;
  std::array<SymbolicWaveClass, 3> waveClasses;
  llvm::SmallVector<StructuredDAGEdgeID, 4> incomingEdges;
  llvm::SmallVector<StructuredDAGEdgeID, 4> outgoingEdges;
};

/// A stable SSA dependency between two direct structured operations.  Pure
/// top-level support operations may be traversed between the producer result
/// and consumer operand, but they never become hidden scheduler nodes.
struct StructuredDAGEdge {
  StructuredDAGEdgeID id = 0;
  StructuredDAGNodeID producer = 0;
  uint32_t producerResult = 0;
  StructuredDAGNodeID consumer = 0;
  uint32_t consumerOperand = 0;
};

/// One SSA-connected structured dependency component that reaches one or more
/// observable function results.  Components are numbered by their smallest
/// node ID and carry output indices in function-result order.  They are a
/// query-local analysis result, not a selected placement or serialized plan.
struct StructuredDAGDependencyComponent {
  StructuredDAGComponentID id = 0;
  llvm::SmallVector<StructuredDAGNodeID, 4> nodes;
  llvm::SmallVector<uint32_t, 2> observableOutputs;
};

/// Query-local analysis of one defined single-block structured function.
/// Node IDs follow direct structured-operation order.  Edge IDs follow a
/// lexicographic order over producer/result/consumer/operand, so neither ID
/// depends on pointer values or hash-table iteration.  Any IR mutation
/// invalidates the analysis.
class StructuredDAGAnalysis {
public:
  static mlir::FailureOr<StructuredDAGAnalysis>
  create(mlir::func::FuncOp function, std::string *failureReason = nullptr);

  mlir::func::FuncOp getFunction() const { return function; }
  llvm::ArrayRef<StructuredDAGNode> getNodes() const { return nodes; }
  llvm::ArrayRef<StructuredDAGEdge> getEdges() const { return edges; }
  llvm::ArrayRef<StructuredDAGDependencyComponent>
  getObservableDependencyComponents() const {
    return observableComponents;
  }
  /// Nearest structured roots for each function result, in function-result
  /// order.  More than one root is possible when a pure support operation
  /// combines values.  An empty entry means that placement cannot be recovered
  /// from the current structured SSA and must fail closed.
  llvm::ArrayRef<llvm::SmallVector<StructuredDAGNodeID, 2>>
  getObservableOutputRootNodes() const {
    return observableOutputRootNodes;
  }
  /// True only when tensor SSA and effect contracts prove that the observable
  /// components may execute independently.  A false result keeps the normal
  /// joint mapping; it is not a whole-program legality failure.
  bool supportsIndependentComponentPlacement() const {
    return independentComponentPlacement;
  }
  const StructuredDAGNode *getNode(StructuredDAGNodeID id) const;
  const StructuredDAGEdge *getEdge(StructuredDAGEdgeID id) const;

private:
  mlir::func::FuncOp function;
  llvm::SmallVector<StructuredDAGNode, 16> nodes;
  llvm::SmallVector<StructuredDAGEdge, 32> edges;
  llvm::SmallVector<StructuredDAGDependencyComponent, 4> observableComponents;
  llvm::SmallVector<llvm::SmallVector<StructuredDAGNodeID, 2>, 4>
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
/// edge, finite wave class and Tile, and never claims a selected
/// allocation or offset.
struct LiveSPMEntry {
  StructuredDAGEdgeID edge = 0;
  SymbolicWaveKind waveKind = SymbolicWaveKind::Prologue;
  uint64_t footprintBytes = 0;
};

struct TileScheduleState {
  TileId tile{0};
  ScheduleTime availableTime = 0;
  std::optional<SymbolicWaveClass> runningWave;
  llvm::SmallVector<LiveSPMEntry, 4> liveSPM;
};

struct RunningOpWave {
  SymbolicWaveClass wave;
  llvm::SmallVector<TileId, 4> tiles;
  ScheduleTime startTime = 0;
  ScheduleTime finishTime = 0;
};

struct StructuredDAGScheduleEvent {
  ScheduleTime time = 0;
  llvm::SmallVector<SymbolicWaveClass, 4> completedWaves;
  llvm::SmallVector<SymbolicWaveClass, 4> readyWaves;
};

/// Copyable query-local event state.  Dispatch does not advance time, allowing
/// independent waves to be placed on disjoint Tile sets at the same event.
/// advanceToNextEvent completes all waves at the earliest finish time before
/// adding newly-ready classes. Within one node the dependency order is
/// prologue -> steady -> tail.  Across every DAG edge, data readiness is
/// conservatively class-matched (producer prologue -> consumer prologue,
/// steady -> steady, tail -> tail), so a consumer pipeline does not wait for
/// the producer's entire node to finish.
class StructuredDAGScheduleState {
public:
  static mlir::FailureOr<StructuredDAGScheduleState>
  create(const StructuredDAGAnalysis &dag, llvm::ArrayRef<TileId> tiles,
         std::string *failureReason = nullptr);

  ScheduleTime getCurrentTime() const { return currentTime; }
  llvm::ArrayRef<TileScheduleState> getTileStates() const {
    return tileStates;
  }
  llvm::ArrayRef<RunningOpWave> getRunningWaves() const { return runningWaves; }

  const TileScheduleState *getTileState(TileId tile) const;
  DAGScheduleStatus getWaveStatus(SymbolicWaveClass wave) const;
  DAGScheduleStatus getNodeStatus(StructuredDAGNodeID node) const;
  llvm::SmallVector<SymbolicWaveClass, 8> getReadyWaves() const;
  bool isComplete() const;

  mlir::LogicalResult dispatchReadyWave(SymbolicWaveClass wave,
                                        llvm::ArrayRef<TileId> tiles,
                                        ScheduleTime duration,
                                        std::string *failureReason = nullptr);

  mlir::FailureOr<StructuredDAGScheduleEvent>
  advanceToNextEvent(std::string *failureReason = nullptr);

  /// Delays a dependency-satisfied wave until an external movement/data-ready
  /// event. This does not create a second dependency graph: it only adds a
  /// finite not-before time to the current wave state.
  mlir::LogicalResult
  delayWaveReadinessUntil(SymbolicWaveClass wave, ScheduleTime time,
                          std::string *failureReason = nullptr);

  /// Advances to an external event strictly before the next running compute
  /// completion (or while no compute is running). Running Tile work remains in
  /// flight, allowing communication and independent compute to overlap.
  mlir::FailureOr<StructuredDAGScheduleEvent>
  advanceToExternalEvent(ScheduleTime time,
                         std::string *failureReason = nullptr);

  mlir::LogicalResult
  retainLocalEdgeValue(TileId tile, StructuredDAGEdgeID edge,
                       SymbolicWaveKind waveKind, uint64_t footprintBytes,
                       std::string *failureReason = nullptr);

  mlir::LogicalResult
  releaseLocalEdgeValue(TileId tile, StructuredDAGEdgeID edge,
                        SymbolicWaveKind waveKind,
                        std::string *failureReason = nullptr);

private:
  struct WaveState {
    DAGScheduleStatus status = DAGScheduleStatus::Waiting;
    uint32_t unsatisfiedDependencies = 0;
    ScheduleTime notBeforeTime = 0;
  };

  struct NodeState {
    std::array<WaveState, 3> waves;
  };

  StructuredDAGScheduleState(const StructuredDAGAnalysis &dag,
                        llvm::SmallVector<TileScheduleState, 16> tiles,
                        llvm::SmallVector<NodeState, 16> nodes)
      : dag(&dag), tileStates(std::move(tiles)), nodeStates(std::move(nodes)) {}

  TileScheduleState *findTileState(TileId tile);
  const NodeState *getNodeState(StructuredDAGNodeID node) const;
  NodeState *getNodeState(StructuredDAGNodeID node);
  WaveState *getWaveState(SymbolicWaveClass wave);
  const WaveState *getWaveState(SymbolicWaveClass wave) const;
  void refreshReady(SymbolicWaveClass wave);

  const StructuredDAGAnalysis *dag = nullptr;
  ScheduleTime currentTime = 0;
  llvm::SmallVector<TileScheduleState, 16> tileStates;
  llvm::SmallVector<NodeState, 16> nodeStates;
  llvm::SmallVector<RunningOpWave, 16> runningWaves;
};

} // namespace wafer::compiler::detail
