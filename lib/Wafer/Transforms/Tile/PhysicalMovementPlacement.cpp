//===- PhysicalMovementPlacement.cpp - Reuse actual physical copies -------===//

#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>

namespace wafer::compiler::detail {
namespace {

bool hasPrivateReadOnlyUses(mlir::Value result, mlir::scf::ForOp loop) {
  llvm::SmallVector<mlir::Value> pending{result};
  llvm::DenseSet<mlir::Value> visited;
  while (!pending.empty()) {
    mlir::Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (mlir::Operation *user : value.getUsers()) {
      if (!loop->isAncestor(user))
        return false;
      if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user)) {
        if (view.getViewSource() != value)
          return false;
        pending.append(user->getResults().begin(), user->getResults().end());
        continue;
      }
      auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user);
      if (!interface)
        return false;
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
      interface.getEffectsOnValue(value, effects);
      if (effects.empty() || llvm::any_of(effects, [](const auto &effect) {
            return !mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
          }))
        return false;
    }
  }
  return true;
}

struct HoistPlan {
  llvm::SmallVector<mlir::Operation *> metadata;
};

std::optional<HoistPlan> planHoist(mlir::Operation *copy, mlir::scf::ForOp loop,
                                   mlir::AliasAnalysis &aliases) {
  auto lower = mlir::getConstantIntValue(loop.getLowerBound());
  auto upper = mlir::getConstantIntValue(loop.getUpperBound());
  auto step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower ||
      !hasPrivateReadOnlyUses(copy->getResult(0), loop))
    return std::nullopt;
  // Views and scalar offset arithmetic can be nested in a visibility branch.
  // Move their actual SSA dependency slice with the copy, never reconstruct
  // equivalent metadata from a name or a predicted load.
  HoistPlan plan;
  const bool crossesCondition = copy->getParentOp() != loop;
  llvm::DenseSet<mlir::Operation *> visited;
  std::function<bool(mlir::Value)> invariant = [&](mlir::Value value) {
    if (loop.isDefinedOutsideOfLoop(value))
      return true;
    auto *definition = value.getDefiningOp();
    if (!definition || definition->getNumRegions() ||
        !mlir::isMemoryEffectFree(definition))
      return false;
    if (crossesCondition && !mlir::isSpeculatable(definition))
      return false;
    if (crossesCondition && mlir::isa<mlir::MemRefType>(value.getType())) {
      if (auto view = mlir::dyn_cast<mlir::memref::SubViewOp>(definition)) {
        auto sourceType = view.getSourceType();
        for (auto [axis, offset] : llvm::enumerate(view.getMixedOffsets())) {
          int64_t size = view.getStaticSizes()[axis];
          int64_t stride = view.getStaticStrides()[axis];
          if (size <= 0 || stride != 1 || sourceType.isDynamicDim(axis))
            return false;
          if (auto constant = mlir::getConstantIntValue(offset)) {
            if (*constant < 0 || *constant > sourceType.getDimSize(axis) - size)
              return false;
          } else {
            using Bounds = mlir::ValueBoundsConstraintSet;
            auto variable = Bounds::Variable(mlir::cast<mlir::Value>(offset));
            auto lo = Bounds::computeConstantBound(
                mlir::presburger::BoundType::LB, variable);
            auto hi = Bounds::computeConstantBound(
                mlir::presburger::BoundType::UB, variable, nullptr, true);
            if (mlir::failed(lo) || mlir::failed(hi) || *lo < 0 ||
                *hi > sourceType.getDimSize(axis) - size)
              return false;
          }
        }
      } else if (!mlir::isa<mlir::memref::GetGlobalOp, mlir::memref::CastOp>(
                     definition)) {
        return false;
      }
    }
    if (!visited.insert(definition).second)
      return true;
    if (!llvm::all_of(definition->getOperands(), invariant))
      return false;
    plan.metadata.push_back(definition);
    return true;
  };
  if (!llvm::all_of(copy->getOperands(), invariant))
    return std::nullopt;
  mlir::Value source = copy->getOperand(0);
  StorageRootMemo roots;
  auto walk = loop.walk([&](mlir::Operation *operation) {
    if (operation->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>())
      return mlir::WalkResult::advance();
    auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (!interface)
      return mlir::WalkResult::interrupt();
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
    interface.getEffects(effects);
    for (const auto &effect : effects) {
      if (!mlir::isa<mlir::MemoryEffects::Write, mlir::MemoryEffects::Free>(
              effect.getEffect()))
        continue;
      if (!effect.getValue()) {
        // Typed Tile operations separate resource occupancy from their
        // value-associated address effects, just like final Instr operations.
        if (mlir::isa<WaferTileDataflowOpInterface>(operation) &&
            effect.getResource() != mlir::SideEffects::DefaultResource::get())
          continue;
        return mlir::WalkResult::interrupt();
      }
      const auto &sources = roots.getStorageRoots(source);
      const auto &destinations = roots.getStorageRoots(effect.getValue());
      if (sources.empty() || destinations.empty())
        return mlir::WalkResult::interrupt();
      for (mlir::Value sourceRoot : sources)
        for (mlir::Value destinationRoot : destinations)
          if (!aliases.alias(sourceRoot, destinationRoot).isNo())
            return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return walk.wasInterrupted() ? std::nullopt
                               : std::optional<HoistPlan>(std::move(plan));
}

mlir::scf::ForOp getEnclosingLoop(mlir::Operation *copy) {
  auto *parent = copy->getParentOp();
  while (mlir::isa_and_nonnull<mlir::scf::IfOp>(parent))
    parent = parent->getParentOp();
  return mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent);
}

llvm::SmallVector<mlir::Operation *>
collectPhysicalCopies(mlir::Operation *root) {
  llvm::SmallVector<mlir::Operation *> copies;
  root->walk([&](mlir::Operation *operation) {
    // These operations allocate their result. A DPS load mutates an existing
    // destination and needs a separate allocation/lifetime proof.
    if (mlir::isa<LayoutMaterializeOp, MoveReshapeOp, MoveTransposeOp,
                  MoveBroadcastOp, MoveCopyOp, MoveExtractSliceOp>(operation))
      copies.push_back(operation);
  });
  return copies;
}

} // namespace

bool hasInvariantPhysicalMovement(mlir::Operation *root) {
  mlir::AliasAnalysis aliases(root);
  for (auto *copy : collectPhysicalCopies(root))
    if (auto loop = getEnclosingLoop(copy))
      if (planHoist(copy, loop, aliases))
        return true;
  return false;
}

mlir::FailureOr<uint64_t>
optimizePhysicalMovementPlacement(mlir::Operation *root,
                                  StructuredMaterializationRelations &relations,
                                  LayoutMaterializationPlacement placement) {
  if (placement == LayoutMaterializationPlacement::FirstUse)
    return uint64_t{0};
  auto copies = collectPhysicalCopies(root);
  uint64_t moved = 0;
  mlir::IRRewriter rewriter(root->getContext());
  for (mlir::Operation *copy : copies) {
    bool changed = false;
    while (auto loop = getEnclosingLoop(copy)) {
      // Each query reads the current epoch, including previously moved copies.
      mlir::AliasAnalysis aliases(root);
      auto plan = planHoist(copy, loop, aliases);
      if (!plan)
        break;
      for (auto *metadata : plan->metadata)
        rewriter.moveOpBefore(metadata, loop);
      rewriter.moveOpBefore(copy, loop);
      changed = true;
    }
    moved += changed;
  }
  if (moved) {
    rebuildCurrentBufferOwnerRelations(root, relations);
    if (mlir::failed(mlir::verify(root)) ||
        mlir::failed(checkStructuredBufferRelationsCurrent(root, relations)))
      return mlir::failure();
  }
  return moved;
}

} // namespace wafer::compiler::detail

namespace wafer {
#define GEN_PASS_DEF_HOISTINVARIANTPHYSICALMOVEMENTPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"
namespace {
struct HoistInvariantPhysicalMovementPass
    : impl::HoistInvariantPhysicalMovementPassBase<
          HoistInvariantPhysicalMovementPass> {
  void runOnOperation() final {
    StructuredMaterializationRelations relations;
    if (mlir::failed(compiler::detail::optimizePhysicalMovementPlacement(
            getOperation(), relations,
            compiler::detail::LayoutMaterializationPlacement::LoopInvariant)))
      signalPassFailure();
  }
};
} // namespace
} // namespace wafer
