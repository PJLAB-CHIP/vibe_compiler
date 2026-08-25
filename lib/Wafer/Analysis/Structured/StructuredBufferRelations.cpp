//===- StructuredBufferRelations.cpp - Current-IR buffer queries -------===//

#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"

#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace wafer::compiler::detail {
namespace {

static bool
collectStoragePredecessors(mlir::Value value,
                           llvm::SmallVectorImpl<mlir::Value> &predecessors) {
  if (!value)
    return false;

  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Operation *parent = argument.getOwner()->getParentOp();
    if (mlir::Value entry =
            analysis::getSingleExecutionRegionEntryOperand(argument)) {
      predecessors.push_back(entry);
      return true;
    }
    if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
      if (argument.getOwner() == loop.getBody() &&
          argument.getArgNumber() > 0 &&
          argument.getArgNumber() - 1 < loop.getInitArgs().size()) {
        predecessors.push_back(loop.getInitArgs()[argument.getArgNumber() - 1]);
        return true;
      }
    }
    return false;
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  mlir::Operation *definition = result ? result.getOwner() : nullptr;
  if (!definition)
    return false;
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(definition)) {
    predecessors.push_back(cast.getSource());
    return true;
  }
  if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(definition)) {
    predecessors.push_back(view.getViewSource());
    return true;
  }
  if (auto toMemref =
          mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(definition)) {
    predecessors.push_back(toMemref.getTensor());
    return true;
  }
  if (auto toTensor =
          mlir::dyn_cast<mlir::bufferization::ToTensorOp>(definition)) {
    predecessors.push_back(toTensor.getMemref());
    return true;
  }
  if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(definition)) {
    predecessors.push_back(select.getTrueValue());
    predecessors.push_back(select.getFalseValue());
    return true;
  }
  if (mlir::Value exit =
          analysis::getSingleExecutionRegionExitOperand(result)) {
    predecessors.push_back(exit);
    return true;
  }
  if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(definition)) {
    if (result.getResultNumber() < loop.getInitArgs().size()) {
      predecessors.push_back(loop.getInitArgs()[result.getResultNumber()]);
      return true;
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
      predecessors.push_back(yield.getOperand(index));
      forwarded = true;
    }
    if (forwarded)
      return true;
  }
  return false;
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

struct StructuredBufferOwners {
  llvm::SmallVector<uint32_t, 4> nodes;
  llvm::SmallVector<unsigned, 2> outputs;

  bool empty() const { return nodes.empty() && outputs.empty(); }

  void normalize() {
    llvm::sort(nodes);
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    llvm::sort(outputs);
    outputs.erase(std::unique(outputs.begin(), outputs.end()), outputs.end());
  }
};

static StructuredBufferOwners collectBufferOwnersUsedByOperation(
    mlir::Operation *operation,
    const StructuredMaterializationRelations &relations) {
  StructuredBufferOwners owners;
  for (const StructuredOperationEmissionRelation &relation :
       relations.operationEmissions)
    if (relation.operation == operation)
      owners.nodes.push_back(relation.structuredNodeId);
  if (owners.nodes.empty()) {
    llvm::SmallVector<uint32_t, 4> bufferNodes =
        collectStructuredNodesUsedByOperation(operation, relations);
    owners.nodes.append(bufferNodes.begin(), bufferNodes.end());
  }

  StorageRootMemo memo;
  llvm::SmallVector<mlir::Value, 8> values =
      collectOperationBufferValues(operation);
  for (const SpatialOutputBufferRelation &relation : relations.outputBuffers)
    if (llvm::any_of(values, [&](mlir::Value value) {
          return shareStructuredBufferStorage(value, relation.buffer, memo);
        }))
      owners.outputs.push_back(relation.outputIndex);
  owners.normalize();
  return owners;
}

static StructuredBufferOwners collectOwnersInTileDataflowComponent(
    mlir::Operation *seed,
    const StructuredMaterializationRelations &relations) {
  StructuredBufferOwners owners;
  struct WorkItem {
    mlir::Operation *operation = nullptr;
    unsigned distance = 0;
  };
  llvm::SmallVector<WorkItem, 16> worklist;
  llvm::DenseSet<mlir::Operation *> visited;
  if (seed)
    worklist.push_back({seed, 0});
  std::optional<unsigned> ownerDistance;
  for (size_t index = 0; index < worklist.size(); ++index) {
    WorkItem item = worklist[index];
    if (ownerDistance && item.distance > *ownerDistance)
      break;
    mlir::Operation *operation = item.operation;
    if (!operation || !visited.insert(operation).second)
      continue;
    StructuredBufferOwners current =
        collectBufferOwnersUsedByOperation(operation, relations);
    if (!current.empty()) {
      ownerDistance = item.distance;
      owners.nodes.append(current.nodes.begin(), current.nodes.end());
      owners.outputs.append(current.outputs.begin(), current.outputs.end());
      continue;
    }

    for (mlir::Value value : collectOperationBufferValues(operation)) {
      auto enqueue = [&](mlir::Operation *candidate) {
        if (candidate && (mlir::isa<WaferTileDataflowOpInterface>(candidate) ||
                          mlir::isa<WaferInstructionOpInterface>(candidate) ||
                          mlir::isa<mlir::ViewLikeOpInterface>(candidate)))
          worklist.push_back({candidate, item.distance + 1});
      };
      enqueue(value.getDefiningOp());
      for (mlir::Operation *user : value.getUsers())
        enqueue(user);
    }
  }
  owners.normalize();
  return owners;
}

template <typename RelationT>
static void appendRemapped(llvm::ArrayRef<RelationT> source,
                           const mlir::IRMapping &mapping,
                           llvm::SmallVectorImpl<RelationT> &destination,
                           llvm::SmallVectorImpl<RelationT> *unmapped) {
  for (const RelationT &relation : source)
    if (mlir::Value mapped = mapping.lookupOrNull(relation.buffer)) {
      RelationT copy = relation;
      copy.buffer = mapped;
      destination.push_back(copy);
    } else if (unmapped) {
      unmapped->push_back(relation);
    }
}

} // namespace

const llvm::DenseSet<mlir::Value> &
StorageRootMemo::getStorageRoots(mlir::Value value) {
  auto [iterator, inserted] = memo.try_emplace(value);
  if (inserted) {
    iterator->second = std::make_unique<llvm::DenseSet<mlir::Value>>();
    llvm::DenseSet<mlir::Value> *valueRoots = iterator->second.get();
    llvm::SmallVector<mlir::Value, 4> predecessors;
    if (!collectStoragePredecessors(value, predecessors)) {
      if (value)
        valueRoots->insert(value);
    } else {
      // The empty map entry is installed before recursion, so a malformed
      // forwarding cycle terminates conservatively. Every acyclic predecessor
      // is memoized in turn, making shared view/region prefixes linear in the
      // number of current SSA values.
      for (mlir::Value predecessor : predecessors) {
        const llvm::DenseSet<mlir::Value> &roots = getStorageRoots(predecessor);
        valueRoots->insert(roots.begin(), roots.end());
      }
    }
    return *valueRoots;
  }
  return *iterator->second;
}

struct StructuredBufferReplacementListener::Impl {
  enum class RelationKind : uint8_t {
    OperationResult,
    Operand,
    Scratch,
    Output,
    CardDDR,
  };

  struct RelationReference {
    RelationKind kind = RelationKind::OperationResult;
    unsigned index = 0;
    unsigned resultNumber = 0;
  };

  explicit Impl(StructuredMaterializationRelations &relations)
      : relations(relations) {
    auto record = [&](auto &entries, RelationKind kind) {
      for (auto [index, entry] : llvm::enumerate(entries)) {
        auto result = mlir::dyn_cast<mlir::OpResult>(entry.buffer);
        if (!result)
          continue;
        references[result.getOwner()].push_back(RelationReference{
            kind, static_cast<unsigned>(index), result.getResultNumber()});
      }
    };
    record(relations.operationResultBuffers, RelationKind::OperationResult);
    record(relations.operandBuffers, RelationKind::Operand);
    record(relations.scratchBuffers, RelationKind::Scratch);
    record(relations.outputBuffers, RelationKind::Output);
    record(relations.cardDDRBuffers, RelationKind::CardDDR);
  }

  mlir::Value &getBuffer(const RelationReference &reference) {
    switch (reference.kind) {
    case RelationKind::OperationResult:
      return relations.operationResultBuffers[reference.index].buffer;
    case RelationKind::Operand:
      return relations.operandBuffers[reference.index].buffer;
    case RelationKind::Scratch:
      return relations.scratchBuffers[reference.index].buffer;
    case RelationKind::Output:
      return relations.outputBuffers[reference.index].buffer;
    case RelationKind::CardDDR:
      return relations.cardDDRBuffers[reference.index].buffer;
    }
    llvm_unreachable("unknown structured buffer relation kind");
  }

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<RelationReference, 2>>
      references;
  StructuredMaterializationRelations &relations;
  std::string failureReason;
  bool preservedAll = true;
};

StructuredBufferReplacementListener::StructuredBufferReplacementListener(
    StructuredMaterializationRelations &relations)
    : impl(std::make_unique<Impl>(relations)) {}

StructuredBufferReplacementListener::~StructuredBufferReplacementListener() =
    default;

void StructuredBufferReplacementListener::recordScratchAllocation(
    mlir::Operation *sourceOperation, mlir::Value allocation) {
  StructuredBufferOwners owners =
      collectBufferOwnersUsedByOperation(sourceOperation, impl->relations);
  if (owners.empty())
    owners =
        collectOwnersInTileDataflowComponent(sourceOperation, impl->relations);
  if (owners.empty() || !allocation ||
      !isWaferSPMMemRefType(allocation.getType())) {
    impl->preservedAll = false;
    if (impl->failureReason.empty()) {
      llvm::raw_string_ostream diagnostic(impl->failureReason);
      diagnostic << "lowering scratch allocation has no typed owner; source=";
      if (sourceOperation)
        diagnostic << sourceOperation->getName();
      else
        diagnostic << "<null>";
    }
    return;
  }
  auto registerAllocation = [&](Impl::RelationKind kind, unsigned index) {
    if (auto result = mlir::dyn_cast<mlir::OpResult>(allocation))
      impl->references[result.getOwner()].push_back(
          Impl::RelationReference{kind, index, result.getResultNumber()});
  };
  for (uint32_t node : owners.nodes) {
    if (!llvm::any_of(impl->relations.scratchBuffers,
                      [&](const StructuredOperationBufferRelation &relation) {
                        return relation.structuredNodeId == node &&
                               relation.buffer == allocation;
                      })) {
      impl->relations.scratchBuffers.push_back({node, allocation});
      registerAllocation(Impl::RelationKind::Scratch,
                         impl->relations.scratchBuffers.size() - 1);
    }
  }
  for (unsigned output : owners.outputs)
    if (!llvm::any_of(impl->relations.outputBuffers,
                      [&](const SpatialOutputBufferRelation &relation) {
                        return relation.outputIndex == output &&
                               relation.buffer == allocation;
                      })) {
      impl->relations.outputBuffers.push_back({output, allocation});
      registerAllocation(Impl::RelationKind::Output,
                         impl->relations.outputBuffers.size() - 1);
    }
}

void StructuredBufferReplacementListener::recordLoweredOperation(
    mlir::Operation *sourceOperation, mlir::Operation *loweredOperation) {
  StructuredBufferOwners owners =
      collectBufferOwnersUsedByOperation(sourceOperation, impl->relations);
  if (owners.empty())
    owners =
        collectOwnersInTileDataflowComponent(sourceOperation, impl->relations);
  if (owners.nodes.empty())
    return;
  if (!loweredOperation) {
    impl->preservedAll = false;
    if (impl->failureReason.empty())
      impl->failureReason = "typed lowered operation owner has no target op";
    return;
  }
  for (uint32_t node : owners.nodes)
    if (!llvm::any_of(impl->relations.operationEmissions,
                      [&](const StructuredOperationEmissionRelation &relation) {
                        return relation.structuredNodeId == node &&
                               relation.operation == loweredOperation;
                      }))
      impl->relations.operationEmissions.push_back({node, loweredOperation});
}

void StructuredBufferReplacementListener::notifyOperationReplaced(
    mlir::Operation *operation, mlir::ValueRange replacements) {
  auto iterator = impl->references.find(operation);
  if (iterator == impl->references.end())
    return;
  llvm::SmallVector<Impl::RelationReference, 2> references =
      std::move(iterator->second);
  impl->references.erase(iterator);
  for (const Impl::RelationReference &reference : references) {
    if (reference.resultNumber >= replacements.size() ||
        !replacements[reference.resultNumber]) {
      impl->preservedAll = false;
      continue;
    }
    mlir::Value replacement = replacements[reference.resultNumber];
    impl->getBuffer(reference) = replacement;
    // A replacement can itself be replaced by a later pattern in the same
    // rewrite driver. Re-register the relation under the current defining op
    // so multi-hop legalization never leaves attribution on an erased value.
    if (auto result = mlir::dyn_cast<mlir::OpResult>(replacement))
      impl->references[result.getOwner()].push_back(Impl::RelationReference{
          reference.kind, reference.index, result.getResultNumber()});
  }
}

void StructuredBufferReplacementListener::notifyOperationErased(
    mlir::Operation *operation) {
  llvm::erase_if(impl->relations.operationEmissions,
                 [&](const StructuredOperationEmissionRelation &relation) {
                   return relation.operation == operation;
                 });
  auto iterator = impl->references.find(operation);
  if (iterator == impl->references.end())
    return;
  if (mlir::isa<mlir::memref::AllocOp>(operation) && operation->use_empty()) {
    for (const Impl::RelationReference &reference : iterator->second)
      impl->getBuffer(reference) = {};
    impl->references.erase(iterator);
    return;
  }
  impl->preservedAll = false;
  impl->references.erase(iterator);
}

bool StructuredBufferReplacementListener::finalizeAfterRewrite() {
  auto dropErased = [](auto &entries) {
    llvm::erase_if(entries, [](const auto &entry) { return !entry.buffer; });
  };
  dropErased(impl->relations.operationResultBuffers);
  dropErased(impl->relations.operandBuffers);
  dropErased(impl->relations.scratchBuffers);
  dropErased(impl->relations.outputBuffers);
  dropErased(impl->relations.cardDDRBuffers);
  return impl->preservedAll;
}

llvm::StringRef StructuredBufferReplacementListener::getFailureReason() const {
  return impl->failureReason;
}

mlir::LogicalResult checkStructuredBufferRelationsCurrent(
    mlir::Operation *root,
    const StructuredMaterializationRelations &relations) {
  if (!root)
    return mlir::failure();

  llvm::DenseSet<const void *> liveValues;
  llvm::DenseSet<mlir::Operation *> liveOperations;
  root->walk([&](mlir::Operation *operation) {
    liveOperations.insert(operation);
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
  const bool currentOperations =
      llvm::all_of(relations.operationEmissions, [&](const auto &relation) {
        return relation.operation &&
               liveOperations.contains(relation.operation);
      });
  return mlir::success(currentOperations &&
                       allCurrent(relations.operationResultBuffers) &&
                       allCurrent(relations.operandBuffers) &&
                       allCurrent(relations.scratchBuffers) &&
                       allCurrent(relations.outputBuffers) &&
                       allCurrent(relations.cardDDRBuffers));
}

void retainCurrentStructuredBufferRelations(
    mlir::Operation *root, StructuredMaterializationRelations &relations) {
  llvm::DenseSet<const void *> liveValues;
  llvm::DenseSet<mlir::Operation *> liveOperations;
  if (root)
    root->walk([&](mlir::Operation *operation) {
      liveOperations.insert(operation);
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
  retain(relations.scratchBuffers);
  retain(relations.outputBuffers);
  retain(relations.cardDDRBuffers);
  retain(relations.partialReductionContributions);
  retain(relations.partialReductionMergeInputs);
  llvm::erase_if(relations.operationEmissions,
                 [&](const StructuredOperationEmissionRelation &relation) {
                   return !relation.operation ||
                          !liveOperations.contains(relation.operation);
                 });
}

mlir::LogicalResult rebaseStructuredBufferRelationsToStorageRoots(
    StructuredMaterializationRelations &relations) {
  StorageRootMemo memo;
  auto rebase = [&](auto &entries) {
    for (auto &entry : entries) {
      const llvm::DenseSet<mlir::Value> &roots =
          memo.getStorageRoots(entry.buffer);
      if (roots.size() != 1)
        return false;
      entry.buffer = *roots.begin();
    }
    return true;
  };
  return mlir::success(
      rebase(relations.operationResultBuffers) &&
      rebase(relations.operandBuffers) && rebase(relations.scratchBuffers) &&
      rebase(relations.outputBuffers) && rebase(relations.cardDDRBuffers) &&
      rebase(relations.partialReductionContributions) &&
      rebase(relations.partialReductionMergeInputs));
}

StructuredMaterializationRelations
remapStructuredBufferRelations(const StructuredMaterializationRelations &source,
                               const mlir::IRMapping &mapping) {
  StructuredMaterializationRelations result;
  llvm::SmallVectorImpl<StructuredOperationResultBufferRelation>
      *noUnmappedResults = nullptr;
  llvm::SmallVectorImpl<StructuredOperationBufferRelation> *noUnmappedOps =
      nullptr;
  llvm::SmallVectorImpl<SpatialOutputBufferRelation> *noUnmappedOutputs =
      nullptr;
  llvm::SmallVectorImpl<CardDDRBufferRelation> *noUnmappedCardDDR = nullptr;
  appendRemapped<StructuredOperationResultBufferRelation>(
      source.operationResultBuffers, mapping, result.operationResultBuffers,
      noUnmappedResults);
  appendRemapped(
      llvm::ArrayRef<StructuredOperationBufferRelation>(source.operandBuffers),
      mapping, result.operandBuffers, noUnmappedOps);
  appendRemapped(
      llvm::ArrayRef<StructuredOperationBufferRelation>(source.scratchBuffers),
      mapping, result.scratchBuffers, noUnmappedOps);
  appendRemapped(
      llvm::ArrayRef<SpatialOutputBufferRelation>(source.outputBuffers),
      mapping, result.outputBuffers, noUnmappedOutputs);
  appendRemapped(llvm::ArrayRef<CardDDRBufferRelation>(source.cardDDRBuffers),
                 mapping, result.cardDDRBuffers, noUnmappedCardDDR);
  return result;
}

StructuredMaterializationRelations scopeStructuredBufferRelations(
    mlir::Operation *root,
    const StructuredMaterializationRelations &relations) {
  llvm::DenseSet<const void *> inScope;
  auto insert = [&](mlir::Value value) {
    if (value)
      inScope.insert(value.getAsOpaquePointer());
  };
  for (mlir::Value operand : root->getOperands())
    insert(operand);
  for (mlir::Value result : root->getResults())
    insert(result);
  for (mlir::Region &region : root->getRegions())
    for (mlir::Block &block : region)
      for (mlir::BlockArgument argument : block.getArguments())
        insert(argument);
  root->walk([&](mlir::Operation *operation) {
    for (mlir::Value result : operation->getResults())
      insert(result);
  });
  auto inScopeEntry = [&](const auto &entry) {
    return entry.buffer && inScope.contains(entry.buffer.getAsOpaquePointer());
  };
  StructuredMaterializationRelations result;
  for (const auto &entry : relations.operationResultBuffers)
    if (inScopeEntry(entry))
      result.operationResultBuffers.push_back(entry);
  for (const auto &entry : relations.operandBuffers)
    if (inScopeEntry(entry))
      result.operandBuffers.push_back(entry);
  for (const auto &entry : relations.scratchBuffers)
    if (inScopeEntry(entry))
      result.scratchBuffers.push_back(entry);
  for (const auto &entry : relations.outputBuffers)
    if (inScopeEntry(entry))
      result.outputBuffers.push_back(entry);
  for (const auto &entry : relations.cardDDRBuffers)
    if (inScopeEntry(entry))
      result.cardDDRBuffers.push_back(entry);
  return result;
}

mlir::FailureOr<StructuredMaterializationRelations>
remapStructuredBufferRelationsComplete(
    const StructuredMaterializationRelations &source,
    const mlir::IRMapping &mapping, StructuredRelationRemapIssue *issue) {
  StructuredRelationRemapIssue localIssue;
  StructuredRelationRemapIssue &reported = issue ? *issue : localIssue;
  reported = {};
  StructuredMaterializationRelations result;
  appendRemapped<StructuredOperationResultBufferRelation>(
      source.operationResultBuffers, mapping, result.operationResultBuffers,
      &reported.unmappedResultBuffers);
  appendRemapped(
      llvm::ArrayRef<StructuredOperationBufferRelation>(source.operandBuffers),
      mapping, result.operandBuffers, &reported.unmappedOperandBuffers);
  appendRemapped(
      llvm::ArrayRef<StructuredOperationBufferRelation>(source.scratchBuffers),
      mapping, result.scratchBuffers, &reported.unmappedScratchBuffers);
  appendRemapped(
      llvm::ArrayRef<SpatialOutputBufferRelation>(source.outputBuffers),
      mapping, result.outputBuffers, &reported.unmappedOutputBuffers);
  appendRemapped(llvm::ArrayRef<CardDDRBufferRelation>(source.cardDDRBuffers),
                 mapping, result.cardDDRBuffers,
                 &reported.unmappedCardDDRBuffers);
  if (!reported.empty())
    return mlir::failure();
  return result;
}

bool shareStructuredBufferStorage(mlir::Value lhs, mlir::Value rhs) {
  StorageRootMemo memo;
  return shareStructuredBufferStorage(lhs, rhs, memo);
}

bool shareStructuredBufferStorage(mlir::Value lhs, mlir::Value rhs,
                                  StorageRootMemo &memo) {
  const llvm::DenseSet<mlir::Value> &lhsRoots = memo.getStorageRoots(lhs);
  const llvm::DenseSet<mlir::Value> &rhsRoots = memo.getStorageRoots(rhs);
  return llvm::any_of(
      lhsRoots, [&](mlir::Value root) { return rhsRoots.contains(root); });
}

llvm::SmallVector<uint32_t, 4> collectStructuredNodesUsedByOperation(
    mlir::Operation *operation,
    const StructuredMaterializationRelations &relations) {
  StorageRootMemo memo;
  return collectStructuredNodesUsedByOperation(operation, relations, memo);
}

llvm::SmallVector<uint32_t, 4> collectStructuredNodesUsedByOperation(
    mlir::Operation *operation,
    const StructuredMaterializationRelations &relations,
    StorageRootMemo &memo) {
  llvm::SmallVector<mlir::Value, 8> values =
      collectOperationBufferValues(operation);
  llvm::SmallVector<uint32_t, 4> nodes;
  auto collect = [&](const auto &relation) {
    if (llvm::any_of(values, [&](mlir::Value value) {
          return shareStructuredBufferStorage(value, relation.buffer, memo);
        }))
      nodes.push_back(relation.structuredNodeId);
  };
  for (const auto &relation : relations.operationResultBuffers)
    collect(relation);
  for (const auto &relation : relations.operandBuffers)
    collect(relation);
  for (const auto &relation : relations.scratchBuffers)
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

bool operationUsesStructuredNode(
    mlir::Operation *operation, uint32_t structuredNodeId,
    const StructuredMaterializationRelations &relations,
    StorageRootMemo &memo) {
  llvm::SmallVector<uint32_t, 4> nodes =
      collectStructuredNodesUsedByOperation(operation, relations, memo);
  return llvm::is_contained(nodes, structuredNodeId);
}

} // namespace wafer::compiler::detail
