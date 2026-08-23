//===- SemanticRootAnalysis.cpp - Observable semantic root keys --------===//

#include "Wafer/Planning/PhysicalDataflow/SemanticRootAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <limits>
#include <utility>

namespace wafer::compiler::detail {
namespace {

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
  return mlir::failure();
}

bool updateIfLess(llvm::DenseMap<mlir::Value, SemanticRootKey> &paths,
                  mlir::Value value, const SemanticRootKey &candidate) {
  auto existing = paths.find(value);
  if (existing != paths.end() && !(candidate < existing->second))
    return false;
  paths[value] = candidate;
  return true;
}

bool updateIfLess(llvm::DenseMap<mlir::Operation *, SemanticRootKey> &paths,
                  mlir::Operation *operation,
                  const SemanticRootKey &candidate) {
  auto existing = paths.find(operation);
  if (existing != paths.end() && !(candidate < existing->second))
    return false;
  paths[operation] = candidate;
  return true;
}

} // namespace

mlir::FailureOr<SemanticRootAnalysis>
SemanticRootAnalysis::create(const StructuredDAGAnalysis &dag,
                             std::string *failureReason) {
  mlir::func::FuncOp function = dag.getFunction();
  if (!function || function.isExternal() || !function.getBody().hasOneBlock())
    return fail<SemanticRootAnalysis>(
        failureReason,
        "semantic root analysis requires one defined single-block function");
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  if (!returnOp ||
      returnOp.getNumOperands() > std::numeric_limits<uint32_t>::max())
    return fail<SemanticRootAnalysis>(
        failureReason, "semantic root analysis requires function results");

  llvm::DenseSet<mlir::Operation *> structuredOperations;
  for (const StructuredDAGNode &node : dag.getNodes()) {
    if (!node.operation || !structuredOperations.insert(node.operation).second)
      return fail<SemanticRootAnalysis>(
          failureReason, "structured DAG contains an invalid duplicate node");
  }

  llvm::DenseMap<mlir::Value, SemanticRootKey> bestValuePaths;
  llvm::DenseMap<mlir::Operation *, SemanticRootKey> bestRootPaths;

  auto enqueue = [&](mlir::Value value, uint64_t consumerOperand,
                     const SemanticRootKey &suffix) -> mlir::LogicalResult {
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return mlir::success();
    if (result.getResultNumber() > std::numeric_limits<uint32_t>::max() ||
        consumerOperand > std::numeric_limits<uint32_t>::max())
      return mlir::failure();
    SemanticRootKey candidate = suffix;
    candidate.path.push_back({SemanticRootPathRelation::SSAUseDef,
                              static_cast<uint32_t>(result.getResultNumber()),
                              static_cast<uint32_t>(consumerOperand)});
    updateIfLess(bestValuePaths, value, candidate);
    return mlir::success();
  };

  for (auto [outputIndex, output] : llvm::enumerate(returnOp.getOperands())) {
    SemanticRootKey anchor;
    anchor.anchorKind = SemanticRootAnchorKind::FunctionResult;
    anchor.anchorIndex = static_cast<uint32_t>(outputIndex);
    if (mlir::failed(enqueue(output, outputIndex, anchor)))
      return fail<SemanticRootAnalysis>(
          failureReason, "semantic root path index is not representable");
  }

  // The admitted function has one SSA block. Every result is defined before
  // all of its users, so a single reverse operation traversal observes the
  // final lexicographically-smallest path of every result before propagating
  // it to that operation's operands. This avoids repeatedly copying growing
  // observable paths through a worklist while preserving the exact key.
  mlir::Block &body = function.getBody().front();
  for (mlir::Operation &owner : llvm::reverse(body.without_terminator())) {
    for (mlir::OpResult result : owner.getResults()) {
      auto current = bestValuePaths.find(result);
      if (current == bestValuePaths.end())
        continue;
      SemanticRootKey currentKey = current->second;
      if (owner.getParentOfType<mlir::func::FuncOp>() != function)
        return fail<SemanticRootAnalysis>(
            failureReason, "semantic root path leaves the current function");
      if (structuredOperations.contains(&owner))
        updateIfLess(bestRootPaths, &owner, currentKey);

      for (mlir::OpOperand &operand : owner.getOpOperands()) {
        if (mlir::failed(enqueue(operand.get(), operand.getOperandNumber(),
                                 currentKey)))
          return fail<SemanticRootAnalysis>(
              failureReason,
              "semantic root path index is not representable");
      }
    }
  }

  llvm::SmallVector<SemanticRootBinding, 16> bindings;
  bindings.reserve(dag.getNodes().size());
  for (const StructuredDAGNode &node : dag.getNodes()) {
    auto path = bestRootPaths.find(node.operation);
    if (path == bestRootPaths.end())
      return fail<SemanticRootAnalysis>(
          failureReason,
          "structured root has no typed path to an observable boundary");
    bindings.push_back({path->second, node.operation});
  }
  llvm::sort(bindings,
             [](const SemanticRootBinding &lhs,
                const SemanticRootBinding &rhs) { return lhs.key < rhs.key; });
  for (auto [lhs, rhs] : llvm::zip(bindings, llvm::drop_begin(bindings))) {
    if (lhs.key == rhs.key)
      return fail<SemanticRootAnalysis>(
          failureReason, "structured roots have duplicate semantic keys");
  }
  llvm::SmallVector<SemanticValueBinding, 32> valueBindings;
  valueBindings.reserve(bestValuePaths.size());
  for (const auto &[value, key] : bestValuePaths)
    valueBindings.push_back({key, value});
  llvm::sort(valueBindings,
             [](const SemanticValueBinding &lhs,
                const SemanticValueBinding &rhs) { return lhs.key < rhs.key; });
  return SemanticRootAnalysis(std::move(bindings), std::move(valueBindings));
}

const SemanticRootBinding *
SemanticRootAnalysis::find(mlir::Operation *operation) const {
  auto binding =
      llvm::find_if(roots, [operation](const SemanticRootBinding &v) {
        return v.operation == operation;
      });
  return binding == roots.end() ? nullptr : &*binding;
}

const SemanticRootBinding *
SemanticRootAnalysis::find(const SemanticRootKey &key) const {
  auto binding = llvm::lower_bound(
      roots, key,
      [](const SemanticRootBinding &value, const SemanticRootKey &candidate) {
        return value.key < candidate;
      });
  return binding == roots.end() || binding->key != key ? nullptr : &*binding;
}

const SemanticValueBinding *SemanticRootAnalysis::find(mlir::Value value) const {
  auto binding =
      llvm::find_if(values, [value](const SemanticValueBinding &candidate) {
        return candidate.value == value;
      });
  return binding == values.end() ? nullptr : &*binding;
}

} // namespace wafer::compiler::detail
