//===- StorageInitialization.cpp - Private buffer initialization --------===//

#include "StorageInitialization.h"

#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer::compiler::detail {
namespace {

bool isExcluded(mlir::Operation *operation,
                const llvm::DenseSet<mlir::Operation *> &excluded) {
  for (; operation; operation = operation->getParentOp())
    if (excluded.contains(operation))
      return true;
  return false;
}

mlir::Value getCompleteWriteDestination(mlir::Operation *operation) {
  if (auto fill = mlir::dyn_cast<ComputeFillOp>(operation))
    return fill.getDest();
  if (auto gemm = mlir::dyn_cast<ComputeGemmIntoOp>(operation))
    return gemm.getDest();
  if (auto copy = mlir::dyn_cast<MoveCopyIntoOp>(operation))
    return copy.getDest();
  return {};
}

// Views do not observe contents. Follow all their aliases before selecting the
// first non-view user; escaping and unknown operations remain ordinary users.
mlir::Operation *findFirstStorageUse(
    mlir::Value value, llvm::SmallVectorImpl<mlir::Value> &aliases,
    const llvm::DenseSet<mlir::Operation *> &excluded, uint64_t &useCount) {
  mlir::Block *block = value.getDefiningOp()->getBlock();
  aliases.push_back(value);
  llvm::DenseSet<mlir::Value> visited;
  mlir::Operation *first = nullptr;
  for (size_t index = 0; index < aliases.size(); ++index) {
    mlir::Value alias = aliases[index];
    if (!visited.insert(alias).second)
      continue;
    for (mlir::OpOperand &use : alias.getUses()) {
      ++useCount;
      mlir::Operation *user = use.getOwner();
      auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user);
      if (view && view.getViewSource() == alias &&
          mlir::isMemoryEffectFree(user) && user->getBlock() == block) {
        if (isExcluded(user, excluded))
          return nullptr;
        aliases.append(user->getResults().begin(), user->getResults().end());
        continue;
      }
      // A use inside a conditional or loop is observed at that containing op.
      // Such an op cannot establish the same-block complete overwrite below.
      user = block->findAncestorOpInBlock(*user);
      if (!user)
        return nullptr;
      if (!first || user->isBeforeInBlock(first))
        first = user;
    }
  }
  return first;
}

bool completelyOverwrites(mlir::Operation *writer, mlir::Value storage,
                          llvm::ArrayRef<mlir::Value> aliases) {
  mlir::Value destination = getCompleteWriteDestination(writer);
  if (!destination)
    return false;
  while (destination != storage) {
    auto view = destination.getDefiningOp<ViewReshapeOp>();
    if (!view ||
        mlir::failed(
            analysis::TransferRealizability::proveStaticReshapeMetadataView(
                mlir::cast<mlir::MemRefType>(view.getSource().getType()),
                mlir::cast<mlir::MemRefType>(view.getResult().getType()),
                /*destinationMayWrite=*/true)))
      return false;
    destination = view.getSource();
  }
  auto effects = mlir::cast<mlir::MemoryEffectOpInterface>(writer);
  for (mlir::Value alias : aliases) {
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
    effects.getEffectsOnValue(alias, instances);
    if (llvm::any_of(instances, [](const auto &effect) {
          return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
        }))
      return false;
  }
  return true;
}

// Keep this narrower than generic DCE: only private storage, fills and pure
// aliases are removed. In particular a GEMM/elementwise input is a live read.
bool collectDeadInitialization(
    mlir::Value storage, llvm::SmallVectorImpl<mlir::Operation *> &dead,
    const llvm::DenseSet<mlir::Operation *> &excluded, uint64_t &useCount) {
  for (mlir::OpOperand &use : storage.getUses()) {
    ++useCount;
    mlir::Operation *user = use.getOwner();
    if (isExcluded(user, excluded))
      return false;
    if (auto fill = mlir::dyn_cast<ComputeFillOp>(user)) {
      if (fill.getDest() != storage)
        return false;
      dead.push_back(user);
      continue;
    }
    auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user);
    if (!view || view.getViewSource() != storage ||
        !mlir::isMemoryEffectFree(user))
      return false;
    for (mlir::Value result : user->getResults())
      if (!collectDeadInitialization(result, dead, excluded, useCount))
        return false;
    dead.push_back(user);
  }
  return true;
}

} // namespace

void eliminateUnusedStorageInitialization(
    mlir::Operation *root, mlir::RewriterBase &rewriter,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations) {
  support::ScopedCompileTimingSpan timing("transformation", "tile-storage",
                                          "eliminate-unused-initialization");
  uint64_t useCount = 0, removedCopies = 0, removedFills = 0;
  llvm::SmallVector<LayoutMaterializeOp> layouts;
  root->walk([&](LayoutMaterializeOp op) { layouts.push_back(op); });
  for (auto layout : layouts) {
    auto type = mlir::cast<mlir::MemRefType>(layout.getResult().getType());
    if (isExcluded(layout, pipelineOperations) || !type.hasStaticShape() ||
        !type.getLayout().isIdentity() ||
        llvm::any_of(type.getShape(),
                     [](int64_t extent) { return extent <= 0; }))
      continue;
    llvm::SmallVector<mlir::Value> aliases;
    mlir::Operation *first = findFirstStorageUse(layout.getResult(), aliases,
                                                 pipelineOperations, useCount);
    if (!first || isExcluded(first, pipelineOperations) ||
        !completelyOverwrites(first, layout.getResult(), aliases))
      continue;
    rewriter.setInsertionPoint(layout);
    auto allocation =
        rewriter.create<mlir::memref::AllocOp>(layout.getLoc(), type);
    rewriter.replaceOp(layout, allocation.getResult());
    ++removedCopies;
  }

  // Reverse def order lets a dead layout result expose its source's dead fill
  // without repeated whole-root fixed-point scans. These roots are not views,
  // so erasing a root's fill/view users cannot invalidate another listed root.
  llvm::SmallVector<mlir::Operation *> roots;
  root->walk([&](mlir::Operation *op) {
    if (!mlir::isa<mlir::memref::AllocOp, LayoutMaterializeOp>(op))
      return;
    auto memory = getWaferMemoryAttr(
        mlir::cast<mlir::MemRefType>(op->getResult(0).getType()));
    if (memory && memory.getSpace() == MemorySpace::SPM)
      roots.push_back(op);
  });
  for (auto *allocation : llvm::reverse(roots)) {
    if (isExcluded(allocation, pipelineOperations))
      continue;
    llvm::SmallVector<mlir::Operation *> dead;
    if (!collectDeadInitialization(allocation->getResult(0), dead,
                                   pipelineOperations, useCount))
      continue;
    for (auto *operation : dead) {
      removedFills += mlir::isa<ComputeFillOp>(operation);
      rewriter.eraseOp(operation);
    }
    removedCopies += mlir::isa<LayoutMaterializeOp>(allocation);
    rewriter.eraseOp(allocation);
  }
  support::addCompileCounter("storage-initialization", "examined-uses",
                             useCount);
  support::addCompileCounter("storage-initialization", "removed-copies",
                             removedCopies);
  support::addCompileCounter("storage-initialization", "removed-fills",
                             removedFills);
}

} // namespace wafer::compiler::detail
