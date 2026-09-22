//===- RotatingBuffers.cpp - Tile RotatingBuffers -------------------===//

#include "Wafer/Transforms/Tile/RotatingBuffers.h"
#include "LoopPipeliningInternal.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
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
    uint64_t tripCount = 0;
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
        !tripCount || *tripCount < binding.multiplicity ||
        !allocation.getDynamicSizes().empty() ||
        !allocation.getSymbolOperands().empty() ||
        !allocations.insert(allocation).second)
      return rotationFailure(
          LoopPipeliningFailureKind::BrokenContract,
          "rotating allocation requires a static pre-loop TileRegion root");

    const bool hasRelation =
        llvm::any_of(relations.buffers, [&](const auto &entry) {
          return entry.buffer == allocation.getResult();
        });
    if (!hasRelation)
      return rotationFailure(
          LoopPipeliningFailureKind::BrokenContract,
          "rotating allocation has no current typed owner relation");

    PreparedRotation rotation{binding, *tripCount};
    for (mlir::Operation *user : allocation.getResult().getUsers()) {
      if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(user)) {
        if (rotation.deallocation)
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
    mlir::Value delta = loopBuilder.create<mlir::arith::SubIOp>(
        loop.getLoc(), loop.getInductionVar(), loop.getLowerBound());
    mlir::Value iteration = loopBuilder.create<mlir::arith::DivUIOp>(
        loop.getLoc(), delta, loop.getStep());
    mlir::Value divisor = loopBuilder.create<mlir::arith::ConstantIndexOp>(
        loop.getLoc(), rotation.binding.multiplicity);
    mlir::Value slotIndex = loopBuilder.create<mlir::arith::RemUIOp>(
        loop.getLoc(), iteration, divisor);
    mlir::Value selected = slots.front();
    for (uint32_t index = 1; index < rotation.binding.multiplicity; ++index) {
      mlir::Value expected = loopBuilder.create<mlir::arith::ConstantIndexOp>(
          loop.getLoc(), index);
      mlir::Value condition = loopBuilder.create<mlir::arith::CmpIOp>(
          loop.getLoc(), mlir::arith::CmpIPredicate::eq, slotIndex, expected);
      selected = loopBuilder.create<mlir::arith::SelectOp>(
          loop.getLoc(), condition, slots[index], selected);
    }
    mlir::Value finalSlot =
        slots[(rotation.tripCount - 1) % rotation.binding.multiplicity];
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
