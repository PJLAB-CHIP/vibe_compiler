//===- PhysicalMovementPlacement.cpp - Reuse actual physical copies -------===//

#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

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

bool canHoist(mlir::Operation *copy, mlir::scf::ForOp loop,
              mlir::AliasAnalysis &aliases) {
  auto lower = mlir::getConstantIntValue(loop.getLowerBound());
  auto upper = mlir::getConstantIntValue(loop.getUpperBound());
  auto step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower ||
      llvm::any_of(copy->getOperands(),
                   [&](mlir::Value operand) {
                     return !loop.isDefinedOutsideOfLoop(operand);
                   }) ||
      !hasPrivateReadOnlyUses(copy->getResult(0), loop))
    return false;
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
  return !walk.wasInterrupted();
}

} // namespace

mlir::FailureOr<uint64_t>
optimizePhysicalMovementPlacement(mlir::Operation *root,
                                  StructuredMaterializationRelations &relations,
                                  LayoutMaterializationPlacement placement) {
  if (placement == LayoutMaterializationPlacement::FirstUse)
    return uint64_t{0};
  llvm::SmallVector<mlir::Operation *> copies;
  root->walk([&](mlir::Operation *operation) {
    if (mlir::isa<LayoutMaterializeOp, MoveReshapeOp, MoveTransposeOp,
                  MoveBroadcastOp, MoveCopyOp, MoveExtractSliceOp>(operation))
      copies.push_back(operation);
  });
  uint64_t moved = 0;
  mlir::IRRewriter rewriter(root->getContext());
  for (mlir::Operation *copy : copies) {
    bool changed = false;
    while (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(copy->getParentOp())) {
      // Each query reads the current epoch, including previously moved copies.
      mlir::AliasAnalysis aliases(root);
      if (!canHoist(copy, loop, aliases))
        break;
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
