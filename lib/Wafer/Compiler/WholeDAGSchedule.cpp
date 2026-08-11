//===- WholeDAGSchedule.cpp - Whole-card DAG scheduler core -------------===//

#include "WholeDAGSchedule.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

namespace wafer::compiler::detail {
namespace {

void setFailureReason(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
}

bool isStructuredScheduleNode(mlir::Operation *operation) {
  return operation && mlir::isa<mlir::DestinationStyleOpInterface>(operation) &&
         mlir::isa<mlir::TilingInterface>(operation);
}

using EdgeKey = std::tuple<CardDAGNodeID, uint32_t, CardDAGNodeID, uint32_t>;

class StableNodeUnionFind {
public:
  explicit StableNodeUnionFind(size_t size) : parents(size) {
    for (size_t index = 0; index < size; ++index)
      parents[index] = static_cast<CardDAGNodeID>(index);
  }

  CardDAGNodeID find(CardDAGNodeID node) {
    CardDAGNodeID root = node;
    while (parents[root] != root)
      root = parents[root];
    while (parents[node] != node) {
      CardDAGNodeID next = parents[node];
      parents[node] = root;
      node = next;
    }
    return root;
  }

  void unite(CardDAGNodeID lhs, CardDAGNodeID rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    // The smallest node ID is always the representative, making component
    // ordering independent of traversal or hash-table iteration.
    if (rhs < lhs)
      std::swap(lhs, rhs);
    parents[rhs] = lhs;
  }

private:
  llvm::SmallVector<CardDAGNodeID, 16> parents;
};

bool hasOnlyFunctionalTensorSemantics(mlir::Operation *operation) {
  if (!mlir::isMemoryEffectFree(operation))
    return false;
  auto isFunctionalType = [](mlir::Type type) {
    return !mlir::isa<mlir::ShapedType>(type) ||
           mlir::isa<mlir::RankedTensorType>(type);
  };
  return llvm::all_of(operation->getOperandTypes(), isFunctionalType) &&
         llvm::all_of(operation->getResultTypes(), isFunctionalType);
}

mlir::LogicalResult collectNearestStructuredProducerNodes(
    mlir::Value value, mlir::Block &body,
    const llvm::DenseMap<mlir::Operation *, CardDAGNodeID> &nodeIDs,
    llvm::DenseSet<mlir::Value> &visited,
    llvm::SmallVectorImpl<CardDAGNodeID> &producers,
    std::string *failureReason) {
  if (!value || !visited.insert(value).second)
    return mlir::success();

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return mlir::success();
  mlir::Operation *owner = result.getOwner();
  if (!owner || owner->getBlock() != &body)
    return mlir::success();

  auto node = nodeIDs.find(owner);
  if (node != nodeIDs.end()) {
    producers.push_back(node->second);
    return mlir::success();
  }
  if (!mlir::isMemoryEffectFree(owner)) {
    setFailureReason(failureReason,
                     "observable result crosses an effectful support op");
    return mlir::failure();
  }
  for (mlir::Value operand : owner->getOperands())
    if (mlir::failed(collectNearestStructuredProducerNodes(
            operand, body, nodeIDs, visited, producers, failureReason)))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult collectNearestStructuredProducers(
    mlir::Value value, mlir::Block &body,
    const llvm::DenseMap<mlir::Operation *, CardDAGNodeID> &nodeIDs,
    CardDAGNodeID consumer, uint32_t consumerOperand,
    llvm::DenseSet<mlir::Value> &visited, std::set<EdgeKey> &edgeKeys,
    std::string *failureReason) {
  if (!value || !visited.insert(value).second)
    return mlir::success();

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return mlir::success();
  mlir::Operation *owner = result.getOwner();
  if (!owner || owner->getBlock() != &body)
    return mlir::success();

  auto node = nodeIDs.find(owner);
  if (node != nodeIDs.end()) {
    if (node->second >= consumer) {
      setFailureReason(failureReason,
                       "structured SSA dependency is not in block order");
      return mlir::failure();
    }
    if (result.getResultNumber() > std::numeric_limits<uint32_t>::max()) {
      setFailureReason(failureReason,
                       "structured producer result index is not representable");
      return mlir::failure();
    }
    edgeKeys.emplace(node->second,
                     static_cast<uint32_t>(result.getResultNumber()), consumer,
                     consumerOperand);
    return mlir::success();
  }

  // Non-scheduled operations in the direct block are admitted only when they
  // are pure.  Conservatively traverse all their operands: this preserves
  // dependencies through views and other target-independent support ops
  // without assigning those ops a hidden execution identity.
  for (mlir::Value operand : owner->getOperands())
    if (mlir::failed(collectNearestStructuredProducers(
            operand, body, nodeIDs, consumer, consumerOperand, visited,
            edgeKeys, failureReason)))
      return mlir::failure();
  return mlir::success();
}

bool runningWaveLess(const RunningOpWave &lhs, const RunningOpWave &rhs) {
  if (lhs.finishTime != rhs.finishTime)
    return lhs.finishTime < rhs.finishTime;
  if (lhs.wave != rhs.wave)
    return lhs.wave < rhs.wave;
  return std::lexicographical_compare(
      lhs.tiles.begin(), lhs.tiles.end(), rhs.tiles.begin(), rhs.tiles.end(),
      [](PhysicalTileId lhs, PhysicalTileId rhs) {
        return lhs.getValue() < rhs.getValue();
      });
}

} // namespace

llvm::StringRef stringifySymbolicWaveKind(SymbolicWaveKind kind) {
  switch (kind) {
  case SymbolicWaveKind::Prologue:
    return "prologue";
  case SymbolicWaveKind::Steady:
    return "steady";
  case SymbolicWaveKind::Tail:
    return "tail";
  }
  return "invalid";
}

uint64_t SymbolicWaveClass::getStableOrdinal() const {
  return static_cast<uint64_t>(node) * 3 + static_cast<uint8_t>(kind);
}

mlir::FailureOr<CardDAGAnalysis>
CardDAGAnalysis::create(mlir::func::FuncOp function,
                        std::string *failureReason) {
  if (!function || function.isExternal() || !function.getBody().hasOneBlock()) {
    setFailureReason(
        failureReason,
        "whole-DAG analysis requires one defined single-block function");
    return mlir::failure();
  }

  CardDAGAnalysis analysis;
  analysis.function = function;
  mlir::Block &body = function.getBody().front();
  llvm::DenseMap<mlir::Operation *, CardDAGNodeID> nodeIDs;

  for (mlir::Operation &operation : body.without_terminator()) {
    if (isStructuredScheduleNode(&operation)) {
      if (analysis.nodes.size() >=
          static_cast<size_t>(std::numeric_limits<CardDAGNodeID>::max())) {
        setFailureReason(failureReason,
                         "whole-DAG node count is not representable");
        return mlir::failure();
      }
      CardDAGNodeID id = static_cast<CardDAGNodeID>(analysis.nodes.size());
      CardDAGNode node;
      node.id = id;
      node.operation = &operation;
      node.waveClasses = {{{id, SymbolicWaveKind::Prologue},
                           {id, SymbolicWaveKind::Steady},
                           {id, SymbolicWaveKind::Tail}}};
      analysis.nodes.push_back(std::move(node));
      nodeIDs[&operation] = id;
      continue;
    }

    if (!mlir::isMemoryEffectFree(&operation)) {
      setFailureReason(
          failureReason,
          (llvm::Twine("unsupported effectful top-level operation: ") +
           operation.getName().getStringRef())
              .str());
      return mlir::failure();
    }
    // tensor.pad is a pure typed support relation, not an independently
    // scheduled compute node. Its region supplies the explicit padding value
    // and is materialized together with the consuming tile. Other region ops
    // must not become hidden execution units merely because they are pure.
    if (operation.getNumRegions() != 0 &&
        !mlir::isa<mlir::tensor::PadOp>(&operation)) {
      setFailureReason(
          failureReason,
          (llvm::Twine("unsupported non-scheduled region operation: ") +
           operation.getName().getStringRef())
              .str());
      return mlir::failure();
    }
  }

  if (analysis.nodes.empty()) {
    setFailureReason(failureReason,
                     "whole-DAG analysis found no structured operations");
    return mlir::failure();
  }

  std::set<EdgeKey> edgeKeys;
  for (const CardDAGNode &consumer : analysis.nodes) {
    mlir::Operation *operation = consumer.operation;
    for (auto indexedOperand : llvm::enumerate(operation->getOperands())) {
      if (indexedOperand.index() > std::numeric_limits<uint32_t>::max()) {
        setFailureReason(
            failureReason,
            "structured consumer operand index is not representable");
        return mlir::failure();
      }
      llvm::DenseSet<mlir::Value> visited;
      if (mlir::failed(collectNearestStructuredProducers(
              indexedOperand.value(), body, nodeIDs, consumer.id,
              static_cast<uint32_t>(indexedOperand.index()), visited, edgeKeys,
              failureReason)))
        return mlir::failure();
    }
  }

  if (edgeKeys.size() >
      static_cast<size_t>(std::numeric_limits<CardDAGEdgeID>::max())) {
    setFailureReason(failureReason,
                     "whole-DAG edge count is not representable");
    return mlir::failure();
  }

  for (const EdgeKey &key : edgeKeys) {
    CardDAGEdge edge;
    edge.id = static_cast<CardDAGEdgeID>(analysis.edges.size());
    std::tie(edge.producer, edge.producerResult, edge.consumer,
             edge.consumerOperand) = key;
    analysis.nodes[edge.producer].outgoingEdges.push_back(edge.id);
    analysis.nodes[edge.consumer].incomingEdges.push_back(edge.id);
    analysis.edges.push_back(edge);
  }

  StableNodeUnionFind components(analysis.nodes.size());
  for (const CardDAGEdge &edge : analysis.edges)
    components.unite(edge.producer, edge.consumer);

  bool provesIndependentPlacement =
      llvm::all_of(analysis.nodes, [](const CardDAGNode &node) {
        return hasOnlyFunctionalTensorSemantics(node.operation);
      });
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(body.getTerminator());
  if (!returnOp ||
      returnOp.getNumOperands() > std::numeric_limits<uint32_t>::max()) {
    setFailureReason(failureReason,
                     "whole-DAG analysis requires representable function "
                     "results");
    return mlir::failure();
  }

  llvm::SmallVector<llvm::SmallVector<CardDAGNodeID, 2>, 4> outputProducers;
  outputProducers.reserve(returnOp.getNumOperands());
  for (mlir::Value output : returnOp.getOperands()) {
    llvm::DenseSet<mlir::Value> visited;
    llvm::SmallVector<CardDAGNodeID, 2> producers;
    std::string outputFailure;
    if (mlir::failed(collectNearestStructuredProducerNodes(
            output, body, nodeIDs, visited, producers, &outputFailure))) {
      // The function remains a legal joint scheduling input, but no branch
      // placement may be inferred through an effectful support operation.
      provesIndependentPlacement = false;
      producers.clear();
    }
    llvm::sort(producers);
    producers.erase(std::unique(producers.begin(), producers.end()),
                    producers.end());
    if (producers.empty()) {
      provesIndependentPlacement = false;
    } else {
      for (CardDAGNodeID producer : llvm::drop_begin(producers))
        components.unite(producers.front(), producer);
    }
    outputProducers.push_back(std::move(producers));
  }

  llvm::DenseMap<CardDAGNodeID, size_t> componentIndices;
  for (const CardDAGNode &node : analysis.nodes) {
    CardDAGNodeID root = components.find(node.id);
    auto [entry, inserted] = componentIndices.try_emplace(
        root, analysis.observableComponents.size());
    if (inserted) {
      CardDAGDependencyComponent component;
      component.id =
          static_cast<CardDAGComponentID>(analysis.observableComponents.size());
      analysis.observableComponents.push_back(std::move(component));
    }
    analysis.observableComponents[entry->second].nodes.push_back(node.id);
  }
  for (auto [outputIndex, producers] : llvm::enumerate(outputProducers)) {
    if (producers.empty())
      continue;
    CardDAGNodeID root = components.find(producers.front());
    auto component = componentIndices.find(root);
    if (component == componentIndices.end()) {
      setFailureReason(failureReason,
                       "observable DAG component construction is "
                       "inconsistent");
      return mlir::failure();
    }
    analysis.observableComponents[component->second]
        .observableOutputs.push_back(static_cast<uint32_t>(outputIndex));
  }
  if (llvm::any_of(analysis.observableComponents,
                   [](const CardDAGDependencyComponent &component) {
                     return component.observableOutputs.empty();
                   }))
    provesIndependentPlacement = false;
  analysis.independentComponentPlacement =
      provesIndependentPlacement && analysis.observableComponents.size() > 1;
  analysis.observableOutputRootNodes = std::move(outputProducers);
  return analysis;
}

const CardDAGNode *CardDAGAnalysis::getNode(CardDAGNodeID id) const {
  return id < nodes.size() ? &nodes[id] : nullptr;
}

const CardDAGEdge *CardDAGAnalysis::getEdge(CardDAGEdgeID id) const {
  return id < edges.size() ? &edges[id] : nullptr;
}

mlir::FailureOr<WholeDAGScheduleState>
WholeDAGScheduleState::create(const CardDAGAnalysis &dag,
                              llvm::ArrayRef<PhysicalTileId> tiles,
                              std::string *failureReason) {
  if (dag.getNodes().empty()) {
    setFailureReason(failureReason,
                     "whole-DAG schedule requires a non-empty DAG");
    return mlir::failure();
  }
  if (tiles.empty()) {
    setFailureReason(failureReason,
                     "whole-DAG schedule requires physical Tiles");
    return mlir::failure();
  }

  llvm::SmallVector<PhysicalTileId, 16> canonicalTiles(tiles.begin(),
                                                       tiles.end());
  llvm::sort(canonicalTiles, [](PhysicalTileId lhs, PhysicalTileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  if (std::adjacent_find(canonicalTiles.begin(), canonicalTiles.end()) !=
      canonicalTiles.end()) {
    setFailureReason(failureReason,
                     "whole-DAG physical Tile domain contains duplicates");
    return mlir::failure();
  }

  llvm::SmallVector<PhysicalTileScheduleState, 16> tileStates;
  tileStates.reserve(canonicalTiles.size());
  for (PhysicalTileId tile : canonicalTiles) {
    PhysicalTileScheduleState state;
    state.tile = tile;
    tileStates.push_back(std::move(state));
  }

  llvm::SmallVector<NodeState, 16> nodeStates;
  nodeStates.reserve(dag.getNodes().size());
  for (const CardDAGNode &node : dag.getNodes()) {
    if (node.incomingEdges.size() >= std::numeric_limits<uint32_t>::max()) {
      setFailureReason(failureReason,
                       "whole-DAG incoming edge count is not representable");
      return mlir::failure();
    }
    NodeState state;
    const uint32_t externalDependencies =
        static_cast<uint32_t>(node.incomingEdges.size());
    for (size_t waveIndex = 0; waveIndex < state.waves.size(); ++waveIndex) {
      WaveState &wave = state.waves[waveIndex];
      wave.unsatisfiedDependencies =
          externalDependencies + (waveIndex == 0 ? 0 : 1);
      wave.status = wave.unsatisfiedDependencies == 0
                        ? DAGScheduleStatus::Ready
                        : DAGScheduleStatus::Waiting;
    }
    nodeStates.push_back(state);
  }

  return WholeDAGScheduleState(dag, std::move(tileStates),
                               std::move(nodeStates));
}

PhysicalTileScheduleState *
WholeDAGScheduleState::findTileState(PhysicalTileId tile) {
  auto it = llvm::lower_bound(
      tileStates, tile,
      [](const PhysicalTileScheduleState &state, PhysicalTileId value) {
        return state.tile.getValue() < value.getValue();
      });
  return it != tileStates.end() && it->tile == tile ? &*it : nullptr;
}

const PhysicalTileScheduleState *
WholeDAGScheduleState::getTileState(PhysicalTileId tile) const {
  auto it = llvm::lower_bound(
      tileStates, tile,
      [](const PhysicalTileScheduleState &state, PhysicalTileId value) {
        return state.tile.getValue() < value.getValue();
      });
  return it != tileStates.end() && it->tile == tile ? &*it : nullptr;
}

const WholeDAGScheduleState::NodeState *
WholeDAGScheduleState::getNodeState(CardDAGNodeID node) const {
  return node < nodeStates.size() ? &nodeStates[node] : nullptr;
}

WholeDAGScheduleState::NodeState *
WholeDAGScheduleState::getNodeState(CardDAGNodeID node) {
  return node < nodeStates.size() ? &nodeStates[node] : nullptr;
}

WholeDAGScheduleState::WaveState *
WholeDAGScheduleState::getWaveState(SymbolicWaveClass wave) {
  NodeState *node = getNodeState(wave.node);
  const size_t index = static_cast<size_t>(wave.kind);
  return node && index < node->waves.size() ? &node->waves[index] : nullptr;
}

const WholeDAGScheduleState::WaveState *
WholeDAGScheduleState::getWaveState(SymbolicWaveClass wave) const {
  const NodeState *node = getNodeState(wave.node);
  const size_t index = static_cast<size_t>(wave.kind);
  return node && index < node->waves.size() ? &node->waves[index] : nullptr;
}

void WholeDAGScheduleState::refreshReady(SymbolicWaveClass wave) {
  WaveState *state = getWaveState(wave);
  if (state && state->status == DAGScheduleStatus::Waiting &&
      state->unsatisfiedDependencies == 0)
    state->status = DAGScheduleStatus::Ready;
}

DAGScheduleStatus
WholeDAGScheduleState::getWaveStatus(SymbolicWaveClass wave) const {
  const WaveState *state = getWaveState(wave);
  if (!state)
    return DAGScheduleStatus::Waiting;
  if (state->status == DAGScheduleStatus::Ready &&
      state->notBeforeTime > currentTime)
    return DAGScheduleStatus::Waiting;
  return state->status;
}

DAGScheduleStatus
WholeDAGScheduleState::getNodeStatus(CardDAGNodeID node) const {
  const NodeState *state = getNodeState(node);
  const CardDAGNode *dagNode = dag ? dag->getNode(node) : nullptr;
  if (!state)
    return DAGScheduleStatus::Waiting;
  if (llvm::all_of(state->waves, [](const WaveState &wave) {
        return wave.status == DAGScheduleStatus::Completed;
      }))
    return DAGScheduleStatus::Completed;
  if (llvm::any_of(state->waves, [](const WaveState &wave) {
        return wave.status == DAGScheduleStatus::Running;
      }))
    return DAGScheduleStatus::Running;
  if (dagNode && llvm::any_of(dagNode->waveClasses,
                              [&](SymbolicWaveClass wave) {
                                return getWaveStatus(wave) ==
                                       DAGScheduleStatus::Ready;
                              }))
    return DAGScheduleStatus::Ready;
  return DAGScheduleStatus::Waiting;
}

llvm::SmallVector<SymbolicWaveClass, 8>
WholeDAGScheduleState::getReadyWaves() const {
  llvm::SmallVector<SymbolicWaveClass, 8> ready;
  for (const CardDAGNode &node : dag->getNodes()) {
    for (SymbolicWaveClass wave : node.waveClasses)
      if (getWaveStatus(wave) == DAGScheduleStatus::Ready)
        ready.push_back(wave);
  }
  return ready;
}

bool WholeDAGScheduleState::isComplete() const {
  return runningWaves.empty() &&
         llvm::all_of(nodeStates, [](const NodeState &node) {
           return llvm::all_of(node.waves, [](const WaveState &wave) {
             return wave.status == DAGScheduleStatus::Completed;
           });
         });
}

mlir::LogicalResult WholeDAGScheduleState::dispatchReadyWave(
    SymbolicWaveClass wave, llvm::ArrayRef<PhysicalTileId> tiles,
    WholeDAGTime duration, std::string *failureReason) {
  const CardDAGNode *node = dag->getNode(wave.node);
  WaveState *waveState = getWaveState(wave);
  if (!node || !waveState) {
    setFailureReason(failureReason, "dispatch references an unknown DAG node");
    return mlir::failure();
  }
  if (waveState->status != DAGScheduleStatus::Ready) {
    setFailureReason(failureReason,
                     "dispatch wave is not ready at the current event");
    return mlir::failure();
  }
  if (waveState->notBeforeTime > currentTime) {
    setFailureReason(failureReason,
                     "dispatch wave data is not ready at the current event");
    return mlir::failure();
  }
  if (tiles.empty()) {
    setFailureReason(failureReason,
                     "dispatch requires a non-empty physical Tile set");
    return mlir::failure();
  }
  if (duration == 0 ||
      duration > std::numeric_limits<WholeDAGTime>::max() - currentTime) {
    setFailureReason(failureReason,
                     "dispatch duration must be positive and representable");
    return mlir::failure();
  }

  llvm::SmallVector<PhysicalTileId, 4> canonicalTiles(tiles.begin(),
                                                      tiles.end());
  llvm::sort(canonicalTiles, [](PhysicalTileId lhs, PhysicalTileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  if (std::adjacent_find(canonicalTiles.begin(), canonicalTiles.end()) !=
      canonicalTiles.end()) {
    setFailureReason(failureReason,
                     "dispatch physical Tile set contains duplicates");
    return mlir::failure();
  }
  for (PhysicalTileId tile : canonicalTiles) {
    PhysicalTileScheduleState *tileState = findTileState(tile);
    if (!tileState) {
      setFailureReason(
          failureReason,
          "dispatch references a Tile outside the physical domain");
      return mlir::failure();
    }
    if (tileState->runningWave || tileState->availableTime > currentTime) {
      setFailureReason(failureReason,
                       "dispatch references a Tile busy at the current event");
      return mlir::failure();
    }
  }

  RunningOpWave running;
  running.wave = wave;
  running.tiles = canonicalTiles;
  running.startTime = currentTime;
  running.finishTime = currentTime + duration;
  waveState->status = DAGScheduleStatus::Running;
  for (PhysicalTileId tile : canonicalTiles) {
    PhysicalTileScheduleState *tileState = findTileState(tile);
    tileState->availableTime = running.finishTime;
    tileState->runningWave = wave;
  }
  runningWaves.push_back(std::move(running));
  llvm::sort(runningWaves, runningWaveLess);
  return mlir::success();
}

mlir::FailureOr<WholeDAGScheduleEvent>
WholeDAGScheduleState::advanceToNextEvent(std::string *failureReason) {
  if (runningWaves.empty()) {
    setFailureReason(failureReason,
                     "cannot advance a schedule with no running wave");
    return mlir::failure();
  }

  WholeDAGTime nextTime = runningWaves.front().finishTime;
  auto completedEnd =
      llvm::find_if(runningWaves, [&](const RunningOpWave &wave) {
        return wave.finishTime != nextTime;
      });
  llvm::SmallVector<RunningOpWave, 4> completed(runningWaves.begin(),
                                                completedEnd);
  runningWaves.erase(runningWaves.begin(), completedEnd);
  currentTime = nextTime;

  WholeDAGScheduleEvent event;
  event.time = nextTime;
  for (const RunningOpWave &running : completed) {
    WaveState *waveState = getWaveState(running.wave);
    const CardDAGNode *node = dag->getNode(running.wave.node);
    if (!waveState || !node ||
        waveState->status != DAGScheduleStatus::Running) {
      setFailureReason(failureReason,
                       "whole-DAG running-wave state is inconsistent");
      return mlir::failure();
    }
    for (PhysicalTileId tile : running.tiles) {
      PhysicalTileScheduleState *tileState = findTileState(tile);
      if (!tileState || tileState->runningWave != running.wave) {
        setFailureReason(failureReason,
                         "whole-DAG Tile running-wave state is inconsistent");
        return mlir::failure();
      }
      tileState->runningWave.reset();
    }

    waveState->status = DAGScheduleStatus::Completed;
    event.completedWaves.push_back(running.wave);
    const size_t waveIndex = static_cast<size_t>(running.wave.kind);
    if (waveIndex + 1 < node->waveClasses.size()) {
      SymbolicWaveClass nextWave = node->waveClasses[waveIndex + 1];
      WaveState *nextState = getWaveState(nextWave);
      if (!nextState || nextState->unsatisfiedDependencies == 0) {
        setFailureReason(failureReason,
                         "whole-DAG intra-node dependency is inconsistent");
        return mlir::failure();
      }
      --nextState->unsatisfiedDependencies;
      refreshReady(nextWave);
    }

    for (CardDAGEdgeID edgeID : node->outgoingEdges) {
      const CardDAGEdge *edge = dag->getEdge(edgeID);
      SymbolicWaveClass consumerWave{edge ? edge->consumer : 0,
                                     running.wave.kind};
      WaveState *consumerState = edge ? getWaveState(consumerWave) : nullptr;
      if (!edge || !consumerState ||
          consumerState->unsatisfiedDependencies == 0) {
        setFailureReason(failureReason,
                         "whole-DAG dependency state is inconsistent");
        return mlir::failure();
      }
      --consumerState->unsatisfiedDependencies;
      refreshReady(consumerWave);
    }
  }

  llvm::sort(event.completedWaves);
  event.readyWaves = getReadyWaves();
  return event;
}

mlir::LogicalResult WholeDAGScheduleState::delayWaveReadinessUntil(
    SymbolicWaveClass wave, WholeDAGTime time, std::string *failureReason) {
  WaveState *state = getWaveState(wave);
  if (!state) {
    setFailureReason(failureReason,
                     "data-ready delay references an unknown DAG wave");
    return mlir::failure();
  }
  if (state->status == DAGScheduleStatus::Running ||
      state->status == DAGScheduleStatus::Completed) {
    setFailureReason(failureReason,
                     "data-ready delay references an already dispatched wave");
    return mlir::failure();
  }
  state->notBeforeTime = std::max(state->notBeforeTime, time);
  return mlir::success();
}

mlir::FailureOr<WholeDAGScheduleEvent>
WholeDAGScheduleState::advanceToExternalEvent(WholeDAGTime time,
                                              std::string *failureReason) {
  if (time <= currentTime) {
    setFailureReason(failureReason,
                     "external event must advance whole-DAG time");
    return mlir::failure();
  }
  if (!runningWaves.empty() && runningWaves.front().finishTime <= time) {
    setFailureReason(
        failureReason,
        "external event cannot pass a pending compute completion");
    return mlir::failure();
  }
  currentTime = time;
  WholeDAGScheduleEvent event;
  event.time = time;
  event.readyWaves = getReadyWaves();
  return event;
}

mlir::LogicalResult WholeDAGScheduleState::retainLocalEdgeValue(
    PhysicalTileId tile, CardDAGEdgeID edge, SymbolicWaveKind waveKind,
    uint64_t footprintBytes, std::string *failureReason) {
  PhysicalTileScheduleState *tileState = findTileState(tile);
  if (!tileState) {
    setFailureReason(failureReason,
                     "SPM residency references an unknown physical Tile");
    return mlir::failure();
  }
  if (!dag->getEdge(edge)) {
    setFailureReason(failureReason,
                     "SPM residency references an unknown DAG edge");
    return mlir::failure();
  }
  if (footprintBytes == 0) {
    setFailureReason(failureReason,
                     "SPM residency requires a positive footprint");
    return mlir::failure();
  }
  const auto key = std::pair(edge, static_cast<uint8_t>(waveKind));
  auto entry = llvm::lower_bound(
      tileState->liveSPM, key,
      [](const LiveSPMEntry &candidate,
         std::pair<CardDAGEdgeID, uint8_t> value) {
        return std::pair(candidate.edge,
                         static_cast<uint8_t>(candidate.waveKind)) < value;
      });
  if (entry != tileState->liveSPM.end() && entry->edge == edge &&
      entry->waveKind == waveKind) {
    setFailureReason(failureReason,
                     "SPM residency already contains the DAG edge wave");
    return mlir::failure();
  }
  tileState->liveSPM.insert(entry,
                            LiveSPMEntry{edge, waveKind, footprintBytes});
  return mlir::success();
}

mlir::LogicalResult WholeDAGScheduleState::releaseLocalEdgeValue(
    PhysicalTileId tile, CardDAGEdgeID edge, SymbolicWaveKind waveKind,
    std::string *failureReason) {
  PhysicalTileScheduleState *tileState = findTileState(tile);
  if (!tileState) {
    setFailureReason(failureReason,
                     "SPM consumption references an unknown physical Tile");
    return mlir::failure();
  }
  const auto key = std::pair(edge, static_cast<uint8_t>(waveKind));
  auto entry = llvm::lower_bound(
      tileState->liveSPM, key,
      [](const LiveSPMEntry &candidate,
         std::pair<CardDAGEdgeID, uint8_t> value) {
        return std::pair(candidate.edge,
                         static_cast<uint8_t>(candidate.waveKind)) < value;
      });
  if (entry == tileState->liveSPM.end() || entry->edge != edge ||
      entry->waveKind != waveKind) {
    setFailureReason(failureReason,
                     "SPM release references a non-resident DAG edge wave");
    return mlir::failure();
  }
  tileState->liveSPM.erase(entry);
  return mlir::success();
}

} // namespace wafer::compiler::detail
