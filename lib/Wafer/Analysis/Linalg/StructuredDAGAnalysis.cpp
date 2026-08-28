//===- StructuredDAGAnalysis.cpp - Structured SSA dependency facts -------===//

#include "Wafer/Analysis/Linalg/StructuredDAGAnalysis.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

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

bool isStructuredNode(mlir::Operation *operation) {
  return operation && mlir::isa<mlir::DestinationStyleOpInterface>(operation) &&
         mlir::isa<mlir::TilingInterface>(operation);
}

using EdgeKey =
    std::tuple<StructuredDAGNodeID, uint32_t, StructuredDAGNodeID, uint32_t>;
using ProducerRef = std::pair<StructuredDAGNodeID, uint32_t>;
using ProducerMemo =
    llvm::DenseMap<mlir::Value, llvm::SmallVector<ProducerRef, 2>>;

class StableNodeUnionFind {
public:
  explicit StableNodeUnionFind(size_t size) : parents(size) {
    for (size_t index = 0; index < size; ++index)
      parents[index] = static_cast<StructuredDAGNodeID>(index);
  }

  StructuredDAGNodeID find(StructuredDAGNodeID node) {
    StructuredDAGNodeID root = node;
    while (parents[root] != root)
      root = parents[root];
    while (parents[node] != node) {
      StructuredDAGNodeID next = parents[node];
      parents[node] = root;
      node = next;
    }
    return root;
  }

  void unite(StructuredDAGNodeID lhs, StructuredDAGNodeID rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (rhs < lhs)
      std::swap(lhs, rhs);
    parents[rhs] = lhs;
  }

private:
  llvm::SmallVector<StructuredDAGNodeID, 16> parents;
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
    const llvm::DenseMap<mlir::Operation *, StructuredDAGNodeID> &nodeIDs,
    llvm::DenseSet<mlir::Value> &visited,
    llvm::SmallVectorImpl<StructuredDAGNodeID> &producers,
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

mlir::LogicalResult collectNearestStructuredProducerRefs(
    mlir::Value value, mlir::Block &body,
    const llvm::DenseMap<mlir::Operation *, StructuredDAGNodeID> &nodeIDs,
    ProducerMemo &memo, llvm::DenseSet<mlir::Value> &active,
    llvm::SmallVectorImpl<ProducerRef> &producers, std::string *failureReason) {
  if (!value)
    return mlir::success();
  auto cached = memo.find(value);
  if (cached != memo.end()) {
    llvm::append_range(producers, cached->second);
    return mlir::success();
  }
  if (!active.insert(value).second) {
    setFailureReason(failureReason,
                     "structured support dependency contains an SSA cycle");
    return mlir::failure();
  }

  llvm::SmallVector<ProducerRef, 2> local;
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  mlir::Operation *owner = result ? result.getOwner() : nullptr;
  if (owner && owner->getBlock() == &body) {
    auto node = nodeIDs.find(owner);
    if (node != nodeIDs.end()) {
      if (result.getResultNumber() > std::numeric_limits<uint32_t>::max()) {
        setFailureReason(
            failureReason,
            "structured producer result index is not representable");
        active.erase(value);
        return mlir::failure();
      }
      local.emplace_back(node->second,
                         static_cast<uint32_t>(result.getResultNumber()));
    } else {
      for (mlir::Value operand : owner->getOperands()) {
        if (mlir::failed(collectNearestStructuredProducerRefs(
                operand, body, nodeIDs, memo, active, local, failureReason))) {
          active.erase(value);
          return mlir::failure();
        }
      }
    }
  }
  llvm::sort(local);
  local.erase(std::unique(local.begin(), local.end()), local.end());
  active.erase(value);
  auto [entry, inserted] = memo.try_emplace(value, std::move(local));
  if (!inserted) {
    setFailureReason(failureReason,
                     "structured producer memo was populated recursively");
    return mlir::failure();
  }
  llvm::append_range(producers, entry->second);
  return mlir::success();
}

} // namespace

mlir::FailureOr<StructuredDAGAnalysis>
StructuredDAGAnalysis::create(mlir::func::FuncOp function,
                              std::string *failureReason) {
  if (!function || function.isExternal() || !function.getBody().hasOneBlock()) {
    setFailureReason(
        failureReason,
        "structured-DAG analysis requires one defined single-block function");
    return mlir::failure();
  }

  StructuredDAGAnalysis analysis;
  analysis.function = function;
  mlir::Block &body = function.getBody().front();
  llvm::DenseMap<mlir::Operation *, StructuredDAGNodeID> nodeIDs;

  for (mlir::Operation &operation : body.without_terminator()) {
    if (isStructuredNode(&operation)) {
      if (analysis.nodes.size() >=
          static_cast<size_t>(
              std::numeric_limits<StructuredDAGNodeID>::max())) {
        setFailureReason(failureReason,
                         "structured-DAG node count is not representable");
        return mlir::failure();
      }
      StructuredDAGNodeID id =
          static_cast<StructuredDAGNodeID>(analysis.nodes.size());
      analysis.nodes.push_back(StructuredDAGNode{id, &operation});
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
                     "structured-DAG analysis found no structured operations");
    return mlir::failure();
  }

  std::set<EdgeKey> edgeKeys;
  ProducerMemo producerMemo;
  for (const StructuredDAGNode &consumer : analysis.nodes) {
    for (auto indexedOperand :
         llvm::enumerate(consumer.operation->getOperands())) {
      if (indexedOperand.index() > std::numeric_limits<uint32_t>::max()) {
        setFailureReason(
            failureReason,
            "structured consumer operand index is not representable");
        return mlir::failure();
      }
      llvm::DenseSet<mlir::Value> active;
      llvm::SmallVector<ProducerRef, 2> producers;
      if (mlir::failed(collectNearestStructuredProducerRefs(
              indexedOperand.value(), body, nodeIDs, producerMemo, active,
              producers, failureReason)))
        return mlir::failure();
      for (const auto &[producer, producerResult] : producers) {
        if (producer >= consumer.id) {
          setFailureReason(failureReason,
                           "structured SSA dependency is not in block order");
          return mlir::failure();
        }
        edgeKeys.emplace(producer, producerResult, consumer.id,
                         static_cast<uint32_t>(indexedOperand.index()));
      }
    }
  }
  if (edgeKeys.size() >
      static_cast<size_t>(std::numeric_limits<StructuredDAGEdgeID>::max())) {
    setFailureReason(failureReason,
                     "structured-DAG edge count is not representable");
    return mlir::failure();
  }
  for (const EdgeKey &key : edgeKeys) {
    StructuredDAGEdge edge;
    edge.id = static_cast<StructuredDAGEdgeID>(analysis.edges.size());
    std::tie(edge.producer, edge.producerResult, edge.consumer,
             edge.consumerOperand) = key;
    analysis.nodes[edge.producer].outgoingEdges.push_back(edge.id);
    analysis.nodes[edge.consumer].incomingEdges.push_back(edge.id);
    analysis.edges.push_back(edge);
  }

  StableNodeUnionFind components(analysis.nodes.size());
  for (const StructuredDAGEdge &edge : analysis.edges)
    components.unite(edge.producer, edge.consumer);
  bool provesIndependentPlacement =
      llvm::all_of(analysis.nodes, [](const StructuredDAGNode &node) {
        return hasOnlyFunctionalTensorSemantics(node.operation);
      });
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(body.getTerminator());
  if (!returnOp ||
      returnOp.getNumOperands() > std::numeric_limits<uint32_t>::max()) {
    setFailureReason(failureReason,
                     "structured-DAG analysis requires representable function "
                     "results");
    return mlir::failure();
  }

  llvm::SmallVector<llvm::SmallVector<StructuredDAGNodeID, 2>, 4>
      outputProducers;
  outputProducers.reserve(returnOp.getNumOperands());
  for (mlir::Value output : returnOp.getOperands()) {
    llvm::DenseSet<mlir::Value> visited;
    llvm::SmallVector<StructuredDAGNodeID, 2> producers;
    std::string outputFailure;
    if (mlir::failed(collectNearestStructuredProducerNodes(
            output, body, nodeIDs, visited, producers, &outputFailure))) {
      provesIndependentPlacement = false;
      producers.clear();
    }
    llvm::sort(producers);
    producers.erase(std::unique(producers.begin(), producers.end()),
                    producers.end());
    if (producers.empty()) {
      provesIndependentPlacement = false;
    } else {
      for (StructuredDAGNodeID producer : llvm::drop_begin(producers))
        components.unite(producers.front(), producer);
    }
    outputProducers.push_back(std::move(producers));
  }

  llvm::DenseMap<StructuredDAGNodeID, size_t> componentIndices;
  for (const StructuredDAGNode &node : analysis.nodes) {
    StructuredDAGNodeID root = components.find(node.id);
    auto [entry, inserted] = componentIndices.try_emplace(
        root, analysis.observableComponents.size());
    if (inserted) {
      StructuredDAGDependencyComponent component;
      component.id = static_cast<StructuredDAGComponentID>(
          analysis.observableComponents.size());
      analysis.observableComponents.push_back(std::move(component));
    }
    analysis.observableComponents[entry->second].nodes.push_back(node.id);
  }
  for (auto [outputIndex, producers] : llvm::enumerate(outputProducers)) {
    if (producers.empty())
      continue;
    auto component = componentIndices.find(components.find(producers.front()));
    if (component == componentIndices.end()) {
      setFailureReason(failureReason,
                       "observable DAG component construction is inconsistent");
      return mlir::failure();
    }
    analysis.observableComponents[component->second]
        .observableOutputs.push_back(static_cast<uint32_t>(outputIndex));
  }
  if (llvm::any_of(analysis.observableComponents,
                   [](const StructuredDAGDependencyComponent &component) {
                     return component.observableOutputs.empty();
                   }))
    provesIndependentPlacement = false;
  analysis.independentComponentPlacement =
      provesIndependentPlacement && analysis.observableComponents.size() > 1;
  analysis.observableOutputRootNodes = std::move(outputProducers);
  return analysis;
}

const StructuredDAGNode *
StructuredDAGAnalysis::getNode(StructuredDAGNodeID id) const {
  return id < nodes.size() ? &nodes[id] : nullptr;
}

const StructuredDAGEdge *
StructuredDAGAnalysis::getEdge(StructuredDAGEdgeID id) const {
  return id < edges.size() ? &edges[id] : nullptr;
}

} // namespace wafer::compiler::detail
