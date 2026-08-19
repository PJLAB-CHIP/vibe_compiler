//===- StructuredDAGAnalysis.h - Structured SSA dependency facts -*- C++ -*-===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

using StructuredDAGNodeID = uint32_t;
using StructuredDAGEdgeID = uint32_t;
using StructuredDAGComponentID = uint32_t;

struct StructuredDAGNode {
  StructuredDAGNodeID id = 0;
  mlir::Operation *operation = nullptr;
  llvm::SmallVector<StructuredDAGEdgeID, 4> incomingEdges;
  llvm::SmallVector<StructuredDAGEdgeID, 4> outgoingEdges;
};

/// A stable SSA dependency between two direct structured operations. Pure
/// top-level support operations may be traversed between the producer result
/// and consumer operand, but they never become hidden execution nodes.
struct StructuredDAGEdge {
  StructuredDAGEdgeID id = 0;
  StructuredDAGNodeID producer = 0;
  uint32_t producerResult = 0;
  StructuredDAGNodeID consumer = 0;
  uint32_t consumerOperand = 0;
};

/// One SSA-connected structured dependency component that reaches one or more
/// observable function results. Components are numbered by their smallest
/// node ID and carry output indices in function-result order.
struct StructuredDAGDependencyComponent {
  StructuredDAGComponentID id = 0;
  llvm::SmallVector<StructuredDAGNodeID, 4> nodes;
  llvm::SmallVector<uint32_t, 2> observableOutputs;
};

/// Query-local analysis of one defined single-block structured function. Node
/// IDs follow direct structured-operation order. Edge IDs follow a
/// producer/result/consumer/operand lexicographic order. Any IR mutation
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
  llvm::ArrayRef<llvm::SmallVector<StructuredDAGNodeID, 2>>
  getObservableOutputRootNodes() const {
    return observableOutputRootNodes;
  }
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

} // namespace wafer::compiler::detail
