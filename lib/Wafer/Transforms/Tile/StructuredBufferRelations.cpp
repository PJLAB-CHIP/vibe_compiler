//===- StructuredBufferRelations.cpp - Current-IR buffer queries -------===//

#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include <set>
#include <string>
#include <utility>

namespace wafer::compiler::detail {
namespace {

static TileModuleOp getTileOwner(mlir::Value value) {
  mlir::Operation *operation = nullptr;
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value))
    operation = result.getOwner();
  else if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    operation =
        argument.getOwner() ? argument.getOwner()->getParentOp() : nullptr;
  return operation ? operation->getParentOfType<TileModuleOp>()
                   : TileModuleOp{};
}

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

static llvm::SmallVector<mlir::Value, 4>
collectOperationDefinedBufferValues(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Value, 4> values;
  if (!operation)
    return values;
  auto append = [&](mlir::Value value) {
    if (value && mlir::isa<mlir::BaseMemRefType>(value.getType()) &&
        !llvm::is_contained(values, value))
      values.push_back(value);
  };
  for (mlir::Value result : operation->getResults())
    append(result);
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation)) {
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> instances;
    effects.getEffects(instances);
    for (const auto &instance : instances)
      if (mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect()))
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

static bool shareStructuredBufferStorage(mlir::Value lhs, mlir::Value rhs,
                                         StorageRootMemo &memo) {
  const llvm::DenseSet<mlir::Value> &lhsRoots = memo.getStorageRoots(lhs);
  const llvm::DenseSet<mlir::Value> &rhsRoots = memo.getStorageRoots(rhs);
  return llvm::any_of(
      lhsRoots, [&](mlir::Value root) { return rhsRoots.contains(root); });
}

static StructuredBufferOwners collectBufferOwnersUsedByOperation(
    mlir::Operation *operation,
    const StructuredMaterializationRelations &relations) {
  StructuredBufferOwners owners;
  for (const StructuredOperationEmissionRelation &relation :
       relations.operationEmissions)
    if (relation.operation == operation)
      owners.nodes.push_back(relation.structuredNodeId);
  StorageRootMemo memo;
  llvm::SmallVector<mlir::Value, 4> values =
      collectOperationDefinedBufferValues(operation);
  if (mlir::isa<mlir::memref::CopyOp>(operation)) {
    llvm::SmallVector<mlir::Value, 8> copyValues =
        collectOperationBufferValues(operation);
    values.assign(copyValues.begin(), copyValues.end());
  }
  auto collectNodeOwners = [&](const auto &entries) {
    for (const auto &relation : entries)
      if (llvm::any_of(values, [&](mlir::Value value) {
            return shareStructuredBufferStorage(value, relation.buffer, memo);
          }))
        owners.nodes.push_back(relation.structuredNodeId);
  };
  if (owners.nodes.empty())
    collectNodeOwners(relations.operationResultBuffers);
  if (owners.nodes.empty())
    collectNodeOwners(relations.operandBuffers);
  if (owners.nodes.empty())
    collectNodeOwners(relations.scratchBuffers);
  if (owners.nodes.empty())
    for (mlir::Value value : values) {
      auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
      auto function = argument ? mlir::dyn_cast<mlir::func::FuncOp>(
                                     argument.getOwner()->getParentOp())
                               : mlir::func::FuncOp{};
      if (!function || argument.getOwner() != &function.getBody().front())
        continue;
      auto binding = function.getArgAttrOfType<DDRBindingAttr>(
          argument.getArgNumber(), kWaferDDRBindingAttrName);
      if (!binding)
        continue;
      for (const DDRTransferRelation &transfer : relations.ddrTransfers)
        if (transfer.resourceId == binding.getResourceId()) {
          owners.nodes.push_back(transfer.producerNodeId);
        }
    }
  for (const SpatialOutputBufferRelation &relation : relations.outputBuffers)
    if (llvm::any_of(values, [&](mlir::Value value) {
          return shareStructuredBufferStorage(value, relation.buffer, memo);
        }))
      owners.outputs.push_back(relation.outputIndex);
  owners.normalize();
  return owners;
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
    DDR,
    StructuralOutput,
    BoundarySource,
    BoundaryDestination,
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
    record(relations.ddrBuffers, RelationKind::DDR);
    for (auto [index, entry] : llvm::enumerate(relations.structuralOutputs)) {
      auto result = mlir::dyn_cast<mlir::OpResult>(entry.endpoint);
      if (!result)
        continue;
      references[result.getOwner()].push_back(RelationReference{
          RelationKind::StructuralOutput, static_cast<unsigned>(index),
          result.getResultNumber()});
    }
    for (auto [index, entry] : llvm::enumerate(relations.boundaryRelations)) {
      auto recordEndpoint = [&](mlir::Value endpoint, RelationKind kind) {
        auto result = mlir::dyn_cast<mlir::OpResult>(endpoint);
        if (!result)
          return;
        references[result.getOwner()].push_back(RelationReference{
            kind, static_cast<unsigned>(index), result.getResultNumber()});
      };
      recordEndpoint(entry.sourceEndpoint, RelationKind::BoundarySource);
      recordEndpoint(entry.destinationEndpoint,
                     RelationKind::BoundaryDestination);
    }
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
    case RelationKind::DDR:
      return relations.ddrBuffers[reference.index].buffer;
    case RelationKind::StructuralOutput:
      return relations.structuralOutputs[reference.index].endpoint;
    case RelationKind::BoundarySource:
      return relations.boundaryRelations[reference.index].sourceEndpoint;
    case RelationKind::BoundaryDestination:
      return relations.boundaryRelations[reference.index].destinationEndpoint;
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
  if (owners.empty() || !allocation ||
      !isWaferSPMMemRefType(allocation.getType())) {
    impl->preservedAll = false;
    if (impl->failureReason.empty()) {
      llvm::raw_string_ostream diagnostic(impl->failureReason);
      diagnostic << "lowering scratch allocation has no typed owner; source=";
      if (sourceOperation) {
        diagnostic << sourceOperation->getName();
        diagnostic << "; source_ir=";
        sourceOperation->print(diagnostic,
                               mlir::OpPrintingFlags().skipRegions());
        diagnostic << "; result_users=[";
        bool first = true;
        for (mlir::Value result : sourceOperation->getResults())
          for (mlir::Operation *user : result.getUsers()) {
            if (!first)
              diagnostic << ',';
            first = false;
            diagnostic << user->getName();
          }
        diagnostic << ']';
      } else
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
  dropErased(impl->relations.ddrBuffers);
  llvm::erase_if(impl->relations.structuralOutputs,
                 [](const auto &entry) { return !entry.endpoint; });
  llvm::erase_if(impl->relations.boundaryRelations, [](const auto &entry) {
    return !entry.sourceEndpoint || !entry.destinationEndpoint;
  });
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
  const bool currentBoundaries =
      llvm::all_of(relations.boundaryRelations, [&](const auto &relation) {
        if (!relation.sourceEndpoint || !relation.destinationEndpoint ||
            !liveValues.contains(
                relation.sourceEndpoint.getAsOpaquePointer()) ||
            !liveValues.contains(
                relation.destinationEndpoint.getAsOpaquePointer()))
          return false;
        TileModuleOp sourceOwner = getTileOwner(relation.sourceEndpoint);
        TileModuleOp destinationOwner =
            getTileOwner(relation.destinationEndpoint);
        return sourceOwner && destinationOwner &&
               sourceOwner.getTileIdAttr().getInt() ==
                   relation.sourceTile.getValue() &&
               destinationOwner.getTileIdAttr().getInt() ==
                   relation.destinationTile.getValue();
      });
  std::set<DemandFragmentId> boundaryFragments;
  const bool uniqueBoundaries =
      llvm::all_of(relations.boundaryRelations, [&](const auto &relation) {
        return boundaryFragments.insert(relation.fragment).second;
      });
  const bool currentOutputs =
      llvm::all_of(relations.structuralOutputs, [&](const auto &relation) {
        if (!relation.endpoint ||
            !liveValues.contains(relation.endpoint.getAsOpaquePointer()))
          return false;
        TileModuleOp owner = getTileOwner(relation.endpoint);
        return owner &&
               owner.getTileIdAttr().getInt() == relation.tile.getValue();
      });
  std::set<std::pair<unsigned, int64_t>> outputOwners;
  const bool uniqueOutputs =
      llvm::all_of(relations.structuralOutputs, [&](const auto &relation) {
        return outputOwners
            .insert({relation.outputIndex, relation.tile.getValue()})
            .second;
      });
  return mlir::success(
      currentOperations && allCurrent(relations.operationResultBuffers) &&
      allCurrent(relations.operandBuffers) &&
      allCurrent(relations.scratchBuffers) &&
      allCurrent(relations.outputBuffers) && allCurrent(relations.ddrBuffers) &&
      currentBoundaries && uniqueBoundaries && currentOutputs && uniqueOutputs);
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
  retain(relations.ddrBuffers);
  retain(relations.partialReductionContributions);
  retain(relations.partialReductionMergeInputs);
  llvm::erase_if(relations.structuralOutputs, [&](const auto &relation) {
    return !relation.endpoint ||
           !liveValues.contains(relation.endpoint.getAsOpaquePointer());
  });
  llvm::erase_if(relations.boundaryRelations, [&](const auto &relation) {
    return !relation.sourceEndpoint || !relation.destinationEndpoint ||
           !liveValues.contains(relation.sourceEndpoint.getAsOpaquePointer()) ||
           !liveValues.contains(
               relation.destinationEndpoint.getAsOpaquePointer());
  });
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
      rebase(relations.outputBuffers) && rebase(relations.ddrBuffers) &&
      rebase(relations.partialReductionContributions) &&
      rebase(relations.partialReductionMergeInputs));
}

} // namespace wafer::compiler::detail
