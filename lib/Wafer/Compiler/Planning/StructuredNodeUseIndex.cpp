//===- StructuredNodeUseIndex.cpp - Structured node use lookup --------===//

#include "Wafer/Compiler/Planning/StructuredNodeUseIndex.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {
namespace {

static llvm::SmallVector<mlir::Value, 8>
collectOperationBufferValues(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Value, 8> values;
  if (!operation)
    return values;
  auto append = [&](mlir::Value value) {
    if (value && mlir::isa<mlir::BaseMemRefType>(value.getType()) &&
        !llvm::is_contained(values, value))
      values.push_back(value);
  };
  for (mlir::Value operand : operation->getOperands())
    append(operand);
  for (mlir::Value result : operation->getResults())
    append(result);
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation)) {
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> instances;
    effects.getEffects(instances);
    for (const auto &instance : instances)
      append(instance.getValue());
  }
  return values;
}

} // namespace

StructuredNodeUseIndex::StructuredNodeUseIndex(
    const StructuredMaterializationRelations &relations) {
  auto index = [&](const StructuredOperationBufferRelation &relation) {
    for (mlir::Value root : storageRoots.getStorageRoots(relation.buffer))
      nodesByRoot[root].push_back(relation.structuredNodeId);
  };
  for (const auto &relation : relations.operationResultBuffers)
    index(relation);
  for (const auto &relation : relations.operandBuffers)
    index(relation);
  for (auto &entry : nodesByRoot) {
    llvm::sort(entry.second);
    entry.second.erase(std::unique(entry.second.begin(), entry.second.end()),
                       entry.second.end());
  }
}

llvm::SmallVector<uint32_t, 4>
StructuredNodeUseIndex::collectNodesUsedBy(mlir::Operation *operation) {
  llvm::SmallVector<uint32_t, 4> nodes;
  for (mlir::Value value : collectOperationBufferValues(operation))
    for (mlir::Value root : storageRoots.getStorageRoots(value)) {
      auto entry = nodesByRoot.find(root);
      if (entry != nodesByRoot.end())
        nodes.append(entry->second.begin(), entry->second.end());
    }
  llvm::sort(nodes);
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  return nodes;
}

} // namespace wafer::compiler::detail
