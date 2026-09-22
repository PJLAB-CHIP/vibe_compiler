//===- RotatingBuffers.cpp - Tile RotatingBuffers -------------------===//

#include "Wafer/Transforms/Tile/RotatingBuffers.h"
#include "LoopPipeliningInternal.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <utility>

namespace wafer::compiler::detail {
namespace {
RotatingAllocationMaterializationResult
rotationFailure(LoopPipeliningFailureKind kind, llvm::StringRef detail) {
  return {{}, LoopPipeliningFailure{kind, {}, detail.str()}};
}

// A rotating slot must be fully defined within its own iteration before any
// read. A pre-loop initialization alone only initializes slot0.
bool hasCompleteIterationDefinition(mlir::Value root, mlir::scf::ForOp loop,
                                    bool observedAfter) {
  llvm::SmallVector<mlir::Value> pending{root};
  llvm::DenseSet<mlir::Value> seen;
  llvm::SmallVector<mlir::Operation *> uses;
  llvm::SmallVector<mlir::Operation *> definitions;
  while (!pending.empty()) {
    auto value = pending.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (auto &use : value.getUses()) {
      auto *operation = use.getOwner();
      if (mlir::isa<mlir::memref::DeallocOp>(operation))
        continue;
      if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(operation)) {
        if (!loop->isAncestor(operation) ||
            !mlir::isMemoryEffectFree(operation))
          return false;
        pending.append(operation->getResults().begin(),
                       operation->getResults().end());
        continue;
      }
      auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
      if (!interface)
        return false;
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
      interface.getEffectsOnValue(value, effects);
      if (effects.empty() || llvm::any_of(effects, [](const auto &effect) {
            return !mlir::isa<mlir::MemoryEffects::Read,
                              mlir::MemoryEffects::Write>(effect.getEffect());
          }))
        return false;
      if (!loop->isAncestor(operation))
        continue;
      uses.push_back(operation);
      mlir::Value dest;
      if (auto load = mlir::dyn_cast<StorageLoadOp>(operation))
        dest = load.getDest();
      else if (auto fill = mlir::dyn_cast<ComputeFillOp>(operation))
        dest = fill.getDest();
      else if (auto fill = mlir::dyn_cast<InstrFillOp>(operation))
        dest = fill.getDest();
      else if (auto copy = mlir::dyn_cast<MoveCopyIntoOp>(operation))
        dest = copy.getDest();
      if (dest == root && llvm::none_of(effects, [](const auto &effect) {
            return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
          }))
        definitions.push_back(operation);
    }
  }
  mlir::DominanceInfo dominance(loop);
  return llvm::any_of(definitions, [&](mlir::Operation *definition) {
    if (observedAfter && definition->getBlock() != loop.getBody())
      return false;
    return llvm::all_of(uses, [&](mlir::Operation *use) {
      return definition == use || dominance.properlyDominates(definition, use);
    });
  });
}

} // namespace

RotatingAllocationMaterializationResult materializeRotatingAllocations(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    llvm::ArrayRef<RotatingAllocationBinding> bindings,
    StructuredMaterializationRelations &relations) {
  if (!module)
    return rotationFailure(LoopPipeliningFailureKind::BrokenContract,
                           "rotating allocation requires an owned module");

  struct PreparedRotation {
    RotatingAllocationBinding binding;
    std::optional<uint64_t> tripCount;
    llvm::SmallVector<mlir::Operation *, 8> insideUses;
    llvm::SmallVector<mlir::Operation *, 4> beforeUses;
    llvm::SmallVector<mlir::Operation *, 4> afterUses;
    mlir::memref::DeallocOp deallocation;
  };
  llvm::SmallVector<PreparedRotation, 4> prepared;
  llvm::DenseSet<mlir::Operation *> allocations;
  for (const RotatingAllocationBinding &binding : bindings) {
    mlir::memref::AllocOp allocation = binding.allocation;
    mlir::scf::ForOp loop = binding.loop;
    std::optional<uint64_t> tripCount = getStaticTripCount(loop);
    TileRegionOp allocationRegion =
        allocation ? allocation->getParentOfType<TileRegionOp>()
                   : TileRegionOp{};
    if (!allocation || !loop ||
        !module->getOperation()->isAncestor(allocation) ||
        !module->getOperation()->isAncestor(loop) || !allocationRegion ||
        loop->getParentOfType<TileRegionOp>() != allocationRegion ||
        allocation->getBlock() != loop->getBlock() ||
        !allocation->isBeforeInBlock(loop) || binding.multiplicity < 2 ||
        !allocation.getDynamicSizes().empty() ||
        !allocation.getSymbolOperands().empty() ||
        !allocations.insert(allocation).second)
      return rotationFailure(LoopPipeliningFailureKind::BrokenContract,
                             "rotating allocation requires a static-shaped "
                             "pre-loop TileRegion root");

    if (auto failure = checkLoopPipeliningDomain(loop, binding.multiplicity))
      return {{}, std::move(failure)};

    const bool hasRelation =
        llvm::any_of(relations.buffers, [&](const auto &entry) {
          return entry.buffer == allocation.getResult();
        });
    if (!hasRelation)
      return rotationFailure(
          LoopPipeliningFailureKind::BrokenContract,
          "rotating allocation has no current typed owner relation");

    PreparedRotation rotation{binding, tripCount};
    for (mlir::Operation *user : allocation.getResult().getUsers()) {
      if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(user)) {
        if (rotation.deallocation || dealloc->getBlock() != loop->getBlock() ||
            !loop->isBeforeInBlock(dealloc))
          return rotationFailure(
              LoopPipeliningFailureKind::BrokenContract,
              "rotating allocation has several deallocations");
        rotation.deallocation = dealloc;
        continue;
      }
      if (loop->isAncestor(user)) {
        rotation.insideUses.push_back(user);
        continue;
      }
      if (user->getBlock() != loop->getBlock())
        return rotationFailure(
            LoopPipeliningFailureKind::Unsupported,
            "rotating allocation use is outside structured control");
      (user->isBeforeInBlock(loop) ? rotation.beforeUses : rotation.afterUses)
          .push_back(user);
    }
    if (!hasCompleteIterationDefinition(allocation, loop,
                                        !rotation.afterUses.empty()))
      return rotationFailure(LoopPipeliningFailureKind::Unsupported,
                             "rotating root lacks a complete iteration-local "
                             "definition or has an escaping alias");
    prepared.push_back(std::move(rotation));
  }

  RotatingAllocationMaterialization result;
  result.module = std::move(module);
  for (PreparedRotation &rotation : prepared) {
    mlir::memref::AllocOp allocation = rotation.binding.allocation;
    mlir::scf::ForOp loop = rotation.binding.loop;
    llvm::SmallVector<mlir::Value, 4> slots{allocation.getResult()};
    mlir::OpBuilder allocationBuilder(allocation);
    mlir::Operation *lastAllocation = allocation.getOperation();
    for (uint32_t index = 1; index < rotation.binding.multiplicity; ++index) {
      allocationBuilder.setInsertionPointAfter(lastAllocation);
      mlir::IRMapping mapping;
      auto clone = mlir::cast<mlir::memref::AllocOp>(
          allocationBuilder.clone(*allocation.getOperation(), mapping));
      slots.push_back(clone.getResult());
      lastAllocation = clone.getOperation();
    }

    mlir::OpBuilder loopBuilder = mlir::OpBuilder::atBlockBegin(loop.getBody());
    auto constant = [&](mlir::OpBuilder &builder, int64_t value) {
      return builder.create<mlir::arith::ConstantOp>(
          loop.getLoc(),
          builder.getIntegerAttr(loop.getInductionVar().getType(), value));
    };
    mlir::Value delta = loopBuilder.create<mlir::arith::SubIOp>(
        loop.getLoc(), loop.getInductionVar(), loop.getLowerBound());
    mlir::Value iteration = loopBuilder.create<mlir::arith::DivUIOp>(
        loop.getLoc(), delta, loop.getStep());
    mlir::Value divisor = constant(loopBuilder, rotation.binding.multiplicity);
    mlir::Value slotIndex = loopBuilder.create<mlir::arith::RemUIOp>(
        loop.getLoc(), iteration, divisor);
    mlir::Value selected = slots.front();
    for (uint32_t index = 1; index < rotation.binding.multiplicity; ++index) {
      mlir::Value expected = constant(loopBuilder, index);
      mlir::Value condition = loopBuilder.create<mlir::arith::CmpIOp>(
          loop.getLoc(), mlir::arith::CmpIPredicate::eq, slotIndex, expected);
      selected = loopBuilder.create<mlir::arith::SelectOp>(
          loop.getLoc(), condition, slots[index], selected);
    }
    mlir::Value finalSlot = slots.front();
    if (!rotation.afterUses.empty()) {
      if (rotation.tripCount) {
        if (*rotation.tripCount)
          finalSlot =
              slots[(*rotation.tripCount - 1) % rotation.binding.multiplicity];
      } else {
        // Last valid ordinal is floor((upper - lower - 1) / step), evaluated
        // only on the nonempty path. Zero iterations preserve original slot0.
        mlir::OpBuilder after(loop);
        after.setInsertionPointAfter(loop);
        auto nonempty = after.create<mlir::arith::CmpIOp>(
            loop.getLoc(), mlir::arith::CmpIPredicate::slt,
            loop.getLowerBound(), loop.getUpperBound());
        auto last = after.create<mlir::scf::IfOp>(
            loop.getLoc(), mlir::TypeRange{allocation.getType()}, nonempty,
            /*withElseRegion=*/true);
        auto chosen =
            mlir::OpBuilder::atBlockBegin(&last.getThenRegion().front());
        auto one = constant(chosen, 1);
        auto span = chosen.create<mlir::arith::SubIOp>(
            loop.getLoc(), loop.getUpperBound(), loop.getLowerBound());
        auto distance =
            chosen.create<mlir::arith::SubIOp>(loop.getLoc(), span, one);
        auto ordinal = chosen.create<mlir::arith::DivUIOp>(
            loop.getLoc(), distance, loop.getStep());
        auto count = constant(chosen, rotation.binding.multiplicity);
        auto index =
            chosen.create<mlir::arith::RemUIOp>(loop.getLoc(), ordinal, count);
        mlir::Value value = slots.front();
        for (uint32_t slot = 1; slot < rotation.binding.multiplicity; ++slot) {
          auto number = constant(chosen, slot);
          auto matches = chosen.create<mlir::arith::CmpIOp>(
              loop.getLoc(), mlir::arith::CmpIPredicate::eq, index, number);
          value = chosen.create<mlir::arith::SelectOp>(loop.getLoc(), matches,
                                                       slots[slot], value);
        }
        chosen.create<mlir::scf::YieldOp>(loop.getLoc(), value);
        auto empty =
            mlir::OpBuilder::atBlockBegin(&last.getElseRegion().front());
        empty.create<mlir::scf::YieldOp>(loop.getLoc(), slots.front());
        finalSlot = last.getResult(0);
      }
    }
    auto replace = [&](llvm::ArrayRef<mlir::Operation *> users,
                       mlir::Value replacement) {
      for (mlir::Operation *user : users)
        for (mlir::OpOperand &operand : user->getOpOperands())
          if (operand.get() == allocation.getResult())
            operand.set(replacement);
    };
    replace(rotation.beforeUses, slots.front());
    replace(rotation.insideUses, selected);
    replace(rotation.afterUses, finalSlot);

    if (rotation.deallocation) {
      mlir::OpBuilder deallocBuilder(rotation.deallocation);
      for (mlir::Value slot : llvm::ArrayRef<mlir::Value>(slots).drop_front())
        deallocBuilder.create<mlir::memref::DeallocOp>(
            rotation.deallocation.getLoc(), slot);
    }

    const size_t originalSize = relations.buffers.size();
    for (size_t index = 0; index < originalSize; ++index) {
      if (relations.buffers[index].buffer != allocation.getResult())
        continue;
      auto original = relations.buffers[index];
      for (mlir::Value slot : llvm::ArrayRef<mlir::Value>(slots).drop_front()) {
        auto copy = original;
        copy.buffer = slot;
        relations.buffers.push_back(std::move(copy));
      }
    }
    result.slots.append(slots.begin(), slots.end());
  }
  if (mlir::failed(mlir::verify(*result.module)))
    return rotationFailure(LoopPipeliningFailureKind::CompilerBug,
                           "rotating allocation produced verifier-invalid IR");
  return {std::move(result), {}};
}

} // namespace wafer::compiler::detail
