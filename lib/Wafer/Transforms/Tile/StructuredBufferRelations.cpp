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

#include <memory>
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

static bool shareStorage(mlir::Value lhs, mlir::Value rhs,
                         StorageRootMemo &memo) {
  const llvm::DenseSet<mlir::Value> &lhsRoots = memo.getStorageRoots(lhs);
  const llvm::DenseSet<mlir::Value> &rhsRoots = memo.getStorageRoots(rhs);
  return llvm::any_of(
      lhsRoots, [&](mlir::Value root) { return rhsRoots.contains(root); });
}

static void appendBufferRelation(StructuredMaterializationRelations &relations,
                                 mlir::Operation *owner, mlir::Value buffer,
                                 MaterializedBufferRole role) {
  if (!owner || !buffer || !mlir::isa<mlir::BaseMemRefType>(buffer.getType()))
    return;
  if (llvm::any_of(relations.buffers, [&](const auto &relation) {
        return relation.owner == owner && relation.buffer == buffer &&
               relation.role == role;
      }))
    return;
  relations.buffers.push_back({owner, buffer, role});
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
      // Install the empty entry before recursion so malformed forwarding
      // cycles terminate conservatively.
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
  enum class ValueKind : uint8_t {
    Buffer,
    StructuralOutput,
    BoundarySource,
    BoundaryDestination,
  };

  struct ValueReference {
    ValueKind kind = ValueKind::Buffer;
    unsigned index = 0;
    unsigned resultNumber = 0;
  };

  explicit Impl(StructuredMaterializationRelations &relations)
      : relations(relations) {
    auto recordValue = [&](mlir::Value value, ValueKind kind, unsigned index) {
      auto result = mlir::dyn_cast<mlir::OpResult>(value);
      if (!result)
        return;
      references[result.getOwner()].push_back(
          {kind, index, result.getResultNumber()});
    };
    for (auto [index, relation] : llvm::enumerate(relations.buffers))
      recordValue(relation.buffer, ValueKind::Buffer, index);
    for (auto [index, relation] : llvm::enumerate(relations.structuralOutputs))
      recordValue(relation.endpoint, ValueKind::StructuralOutput, index);
    for (auto [index, relation] :
         llvm::enumerate(relations.boundaryRelations)) {
      recordValue(relation.sourceEndpoint, ValueKind::BoundarySource, index);
      recordValue(relation.destinationEndpoint, ValueKind::BoundaryDestination,
                  index);
    }
  }

  mlir::Value &getValue(const ValueReference &reference) {
    switch (reference.kind) {
    case ValueKind::Buffer:
      return relations.buffers[reference.index].buffer;
    case ValueKind::StructuralOutput:
      return relations.structuralOutputs[reference.index].endpoint;
    case ValueKind::BoundarySource:
      return relations.boundaryRelations[reference.index].sourceEndpoint;
    case ValueKind::BoundaryDestination:
      return relations.boundaryRelations[reference.index].destinationEndpoint;
    }
    llvm_unreachable("unknown structured relation value kind");
  }

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<ValueReference, 2>>
      references;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 2>>
      loweredOwners;
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
  if (!sourceOperation || !allocation ||
      !isWaferSPMMemRefType(allocation.getType())) {
    impl->preservedAll = false;
    if (impl->failureReason.empty())
      impl->failureReason =
          "lowering scratch allocation has no current typed owner";
    return;
  }
  const unsigned index = impl->relations.buffers.size();
  appendBufferRelation(impl->relations, sourceOperation, allocation,
                       MaterializedBufferRole::Scratch);
  if (index == impl->relations.buffers.size())
    return;
  if (auto result = mlir::dyn_cast<mlir::OpResult>(allocation))
    impl->references[result.getOwner()].push_back(
        {Impl::ValueKind::Buffer, index, result.getResultNumber()});
}

void StructuredBufferReplacementListener::recordLoweredOperation(
    mlir::Operation *sourceOperation, mlir::Operation *loweredOperation) {
  if (!sourceOperation || !loweredOperation) {
    impl->preservedAll = false;
    if (impl->failureReason.empty())
      impl->failureReason = "lowered operation has no current typed owner";
    return;
  }
  auto &owners = impl->loweredOwners[sourceOperation];
  if (!llvm::is_contained(owners, loweredOperation))
    owners.push_back(loweredOperation);
  for (mlir::Value operand : loweredOperation->getOperands())
    appendBufferRelation(impl->relations, loweredOperation, operand,
                         MaterializedBufferRole::Operand);
  for (mlir::Value result : loweredOperation->getResults())
    appendBufferRelation(impl->relations, loweredOperation, result,
                         MaterializedBufferRole::Result);
}

void StructuredBufferReplacementListener::notifyOperationReplaced(
    mlir::Operation *operation, mlir::ValueRange replacements) {
  auto iterator = impl->references.find(operation);
  if (iterator == impl->references.end())
    return;
  llvm::SmallVector<Impl::ValueReference, 2> references =
      std::move(iterator->second);
  impl->references.erase(iterator);
  for (const Impl::ValueReference &reference : references) {
    if (reference.resultNumber >= replacements.size() ||
        !replacements[reference.resultNumber]) {
      impl->preservedAll = false;
      continue;
    }
    mlir::Value replacement = replacements[reference.resultNumber];
    impl->getValue(reference) = replacement;
    if (auto result = mlir::dyn_cast<mlir::OpResult>(replacement))
      impl->references[result.getOwner()].push_back(
          {reference.kind, reference.index, result.getResultNumber()});
  }
}

void StructuredBufferReplacementListener::notifyOperationErased(
    mlir::Operation *operation) {
  auto iterator = impl->references.find(operation);
  if (iterator != impl->references.end()) {
    if (mlir::isa<mlir::memref::AllocOp>(operation) && operation->use_empty()) {
      for (const Impl::ValueReference &reference : iterator->second)
        impl->getValue(reference) = {};
    } else {
      impl->preservedAll = false;
    }
    impl->references.erase(iterator);
  }
}

bool StructuredBufferReplacementListener::finalizeAfterRewrite() {
  llvm::SmallVector<MaterializedBufferRelation, 32> currentBuffers;
  for (const MaterializedBufferRelation &relation : impl->relations.buffers) {
    if (!relation.buffer)
      continue;
    auto lowered = impl->loweredOwners.find(relation.owner);
    if (lowered == impl->loweredOwners.end()) {
      currentBuffers.push_back(relation);
      continue;
    }
    for (mlir::Operation *owner : lowered->second)
      currentBuffers.push_back({owner, relation.buffer, relation.role});
  }
  impl->relations.buffers.clear();
  for (const MaterializedBufferRelation &relation : currentBuffers)
    appendBufferRelation(impl->relations, relation.owner, relation.buffer,
                         relation.role);
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

  StorageRootMemo storageRoots;
  for (const MaterializedBufferRelation &relation : relations.buffers) {
    if (!relation.owner || !relation.buffer ||
        !liveOperations.contains(relation.owner) ||
        !liveValues.contains(relation.buffer.getAsOpaquePointer()))
      return root->emitError("buffer relation has no live owner or endpoint");
    llvm::SmallVector<mlir::Value, 8> owned =
        collectOperationBufferValues(relation.owner);
    if (!llvm::any_of(owned, [&](mlir::Value value) {
          return shareStorage(value, relation.buffer, storageRoots);
        }))
      return relation.owner->emitOpError("buffer relation endpoint does not "
                                         "share storage with its owner: ")
             << relation.buffer;
  }

  std::set<std::pair<const void *, const void *>> boundaryEndpoints;
  for (const StructuredBoundaryRelation &relation :
       relations.boundaryRelations) {
    if (!relation.sourceEndpoint || !relation.destinationEndpoint ||
        !liveValues.contains(relation.sourceEndpoint.getAsOpaquePointer()) ||
        !liveValues.contains(relation.destinationEndpoint.getAsOpaquePointer()))
      return root->emitError("boundary relation has no live endpoint");
    TileModuleOp sourceOwner = getTileOwner(relation.sourceEndpoint);
    TileModuleOp destinationOwner = getTileOwner(relation.destinationEndpoint);
    if (!sourceOwner || !destinationOwner || sourceOwner == destinationOwner ||
        !boundaryEndpoints
             .insert({relation.sourceEndpoint.getAsOpaquePointer(),
                      relation.destinationEndpoint.getAsOpaquePointer()})
             .second)
      return root->emitError(
          "boundary relation has invalid or duplicate Tile endpoints");
  }

  std::set<std::pair<unsigned, const void *>> outputEndpoints;
  for (const StructuredOutputRelation &relation : relations.structuralOutputs) {
    if (!relation.endpoint ||
        !liveValues.contains(relation.endpoint.getAsOpaquePointer()))
      return root->emitError("output relation has no live endpoint");
    TileModuleOp owner = getTileOwner(relation.endpoint);
    if (!owner || !outputEndpoints
                       .insert({relation.outputIndex,
                                relation.endpoint.getAsOpaquePointer()})
                       .second)
      return root->emitError(
          "output relation has an invalid Tile owner or duplicate endpoint");
  }
  return mlir::success();
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
  llvm::erase_if(relations.buffers, [&](const auto &relation) {
    return !relation.owner || !relation.buffer ||
           !liveOperations.contains(relation.owner) ||
           !liveValues.contains(relation.buffer.getAsOpaquePointer());
  });
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
}

void rebuildCurrentBufferOwnerRelations(
    mlir::Operation *root, StructuredMaterializationRelations &relations) {
  relations.buffers.clear();
  if (!root)
    return;
  auto isMovement = [](mlir::Operation *operation) {
    return mlir::isa<LayoutMaterializeOp, MoveExtractSliceOp, MoveInsertSliceOp,
                     MoveCopyOp, MoveCopyIntoOp, MoveReshapeOp, MoveTransposeOp,
                     MoveBroadcastOp, StorageLoadOp, StorageStoreOp,
                     CommPeerSendOp, CommPeerRecvOp, mlir::memref::CopyOp>(
        operation);
  };
  root->walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::ModuleOp, TileModuleOp, TileRegionOp,
                  mlir::func::FuncOp>(operation))
      return;
    for (mlir::Value operand : operation->getOperands())
      appendBufferRelation(relations, operation, operand,
                           isMovement(operation)
                               ? MaterializedBufferRole::Movement
                               : MaterializedBufferRole::Operand);
    for (mlir::Value result : operation->getResults())
      appendBufferRelation(relations, operation, result,
                           mlir::isa<mlir::memref::AllocOp>(operation)
                               ? MaterializedBufferRole::Scratch
                               : MaterializedBufferRole::Result);
  });
}

} // namespace wafer::compiler::detail
