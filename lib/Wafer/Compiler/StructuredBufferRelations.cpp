//===- StructuredBufferRelations.cpp - Current-IR buffer queries -------===//

#include "StructuredBufferRelations.h"

#include "Wafer/Analysis/SingleExecutionRegionFlow.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {
namespace {

static void collectStorageRoots(mlir::Value value,
                                llvm::DenseSet<mlir::Value> &roots,
                                llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return;

  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Operation *parent = argument.getOwner()->getParentOp();
    if (mlir::Value entry =
            analysis::getSingleExecutionRegionEntryOperand(argument)) {
      collectStorageRoots(entry, roots, visited);
      return;
    }
    if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
      if (argument.getOwner() == loop.getBody() &&
          argument.getArgNumber() > 0 &&
          argument.getArgNumber() - 1 < loop.getInitArgs().size()) {
        collectStorageRoots(loop.getInitArgs()[argument.getArgNumber() - 1],
                            roots, visited);
        return;
      }
    }
    roots.insert(value);
    return;
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  mlir::Operation *definition = result ? result.getOwner() : nullptr;
  if (!definition) {
    roots.insert(value);
    return;
  }
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(definition)) {
    collectStorageRoots(cast.getSource(), roots, visited);
    return;
  }
  if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(definition)) {
    collectStorageRoots(view.getViewSource(), roots, visited);
    return;
  }
  if (mlir::Value exit =
          analysis::getSingleExecutionRegionExitOperand(result)) {
    collectStorageRoots(exit, roots, visited);
    return;
  }
  if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(definition)) {
    if (result.getResultNumber() < loop.getInitArgs().size()) {
      collectStorageRoots(loop.getInitArgs()[result.getResultNumber()], roots,
                          visited);
      return;
    }
  }
  if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(definition)) {
    const unsigned index = result.getResultNumber();
    bool forwarded = false;
    for (mlir::Region *region :
         {&branch.getThenRegion(), &branch.getElseRegion()}) {
      if (region->empty())
        continue;
      auto yield =
          mlir::dyn_cast<mlir::scf::YieldOp>(region->front().getTerminator());
      if (!yield || index >= yield.getNumOperands())
        continue;
      collectStorageRoots(yield.getOperand(index), roots, visited);
      forwarded = true;
    }
    if (forwarded)
      return;
  }
  roots.insert(value);
}

static llvm::DenseSet<mlir::Value> collectStorageRoots(mlir::Value value) {
  llvm::DenseSet<mlir::Value> roots;
  llvm::DenseSet<mlir::Value> visited;
  collectStorageRoots(value, roots, visited);
  return roots;
}

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

template <typename RelationT>
static void appendRemapped(llvm::ArrayRef<RelationT> source,
                           const mlir::IRMapping &mapping,
                           llvm::SmallVectorImpl<RelationT> &destination) {
  for (const RelationT &relation : source)
    if (mlir::Value mapped = mapping.lookupOrNull(relation.buffer)) {
      RelationT copy = relation;
      copy.buffer = mapped;
      destination.push_back(copy);
    }
}

} // namespace

struct StructuredBufferReplacementListener::Impl {
  struct RelationReference {
    mlir::Value *buffer = nullptr;
    unsigned resultNumber = 0;
  };

  explicit Impl(StructuredMaterializationRelations &relations) {
    auto record = [&](auto &entries) {
      for (auto &entry : entries) {
        auto result = mlir::dyn_cast<mlir::OpResult>(entry.buffer);
        if (!result)
          continue;
        references[result.getOwner()].push_back(
            RelationReference{&entry.buffer, result.getResultNumber()});
      }
    };
    record(relations.operationResultBuffers);
    record(relations.operandBuffers);
    record(relations.outputBuffers);
  }

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<RelationReference, 2>>
      references;
  bool preservedAll = true;
};

StructuredBufferReplacementListener::StructuredBufferReplacementListener(
    StructuredMaterializationRelations &relations)
    : impl(std::make_unique<Impl>(relations)) {}

StructuredBufferReplacementListener::~StructuredBufferReplacementListener() =
    default;

void StructuredBufferReplacementListener::notifyOperationReplaced(
    mlir::Operation *operation, mlir::ValueRange replacements) {
  auto iterator = impl->references.find(operation);
  if (iterator == impl->references.end())
    return;
  for (const Impl::RelationReference &reference : iterator->second) {
    if (reference.resultNumber >= replacements.size() ||
        !replacements[reference.resultNumber]) {
      impl->preservedAll = false;
      continue;
    }
    *reference.buffer = replacements[reference.resultNumber];
  }
  impl->references.erase(iterator);
}

void StructuredBufferReplacementListener::notifyOperationErased(
    mlir::Operation *operation) {
  auto iterator = impl->references.find(operation);
  if (iterator == impl->references.end())
    return;
  impl->preservedAll = false;
  impl->references.erase(iterator);
}

bool StructuredBufferReplacementListener::preservedAllRelations() const {
  return impl->preservedAll;
}

mlir::LogicalResult checkStructuredBufferRelationsCurrent(
    mlir::Operation *root,
    const StructuredMaterializationRelations &relations) {
  if (!root)
    return mlir::failure();

  llvm::DenseSet<const void *> liveValues;
  root->walk([&](mlir::Operation *operation) {
    for (mlir::Value result : operation->getResults())
      liveValues.insert(result.getAsOpaquePointer());
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument argument : block.getArguments())
          liveValues.insert(argument.getAsOpaquePointer());
  });

  auto allCurrent = [&](const auto &entries) {
    return llvm::all_of(entries, [&](const auto &entry) {
      return entry.buffer &&
             liveValues.contains(entry.buffer.getAsOpaquePointer());
    });
  };
  return mlir::success(allCurrent(relations.operationResultBuffers) &&
                       allCurrent(relations.operandBuffers) &&
                       allCurrent(relations.outputBuffers));
}

void retainCurrentStructuredBufferRelations(
    mlir::Operation *root, StructuredMaterializationRelations &relations) {
  if (!root) {
    relations.clear();
    return;
  }

  llvm::DenseSet<const void *> liveValues;
  root->walk([&](mlir::Operation *operation) {
    for (mlir::Value result : operation->getResults())
      liveValues.insert(result.getAsOpaquePointer());
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument argument : block.getArguments())
          liveValues.insert(argument.getAsOpaquePointer());
  });
  auto retain = [&](auto &entries) {
    llvm::erase_if(entries, [&](const auto &entry) {
      return !entry.buffer ||
             !liveValues.contains(entry.buffer.getAsOpaquePointer());
    });
  };
  retain(relations.operationResultBuffers);
  retain(relations.operandBuffers);
  retain(relations.outputBuffers);
}

StructuredMaterializationRelations
remapStructuredBufferRelations(const StructuredMaterializationRelations &source,
                               const mlir::IRMapping &mapping) {
  StructuredMaterializationRelations result;
  appendRemapped(llvm::ArrayRef<StructuredOperationBufferRelation>(
                     source.operationResultBuffers),
                 mapping, result.operationResultBuffers);
  appendRemapped(
      llvm::ArrayRef<StructuredOperationBufferRelation>(source.operandBuffers),
      mapping, result.operandBuffers);
  appendRemapped(
      llvm::ArrayRef<SpatialOutputBufferRelation>(source.outputBuffers),
      mapping, result.outputBuffers);
  return result;
}

bool shareStructuredBufferStorage(mlir::Value lhs, mlir::Value rhs) {
  llvm::DenseSet<mlir::Value> lhsRoots = collectStorageRoots(lhs);
  llvm::DenseSet<mlir::Value> rhsRoots = collectStorageRoots(rhs);
  return llvm::any_of(
      lhsRoots, [&](mlir::Value root) { return rhsRoots.contains(root); });
}

llvm::SmallVector<uint32_t, 4> collectStructuredNodesUsedByOperation(
    mlir::Operation *operation,
    const StructuredMaterializationRelations &relations) {
  llvm::SmallVector<mlir::Value, 8> values =
      collectOperationBufferValues(operation);
  llvm::SmallVector<uint32_t, 4> nodes;
  auto collect = [&](const StructuredOperationBufferRelation &relation) {
    if (llvm::any_of(values, [&](mlir::Value value) {
          return shareStructuredBufferStorage(value, relation.buffer);
        }))
      nodes.push_back(relation.structuredNodeId);
  };
  for (const auto &relation : relations.operationResultBuffers)
    collect(relation);
  for (const auto &relation : relations.operandBuffers)
    collect(relation);
  llvm::sort(nodes);
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  return nodes;
}

bool operationUsesStructuredNode(
    mlir::Operation *operation, uint32_t structuredNodeId,
    const StructuredMaterializationRelations &relations) {
  llvm::SmallVector<uint32_t, 4> nodes =
      collectStructuredNodesUsedByOperation(operation, relations);
  return llvm::is_contained(nodes, structuredNodeId);
}

} // namespace wafer::compiler::detail
