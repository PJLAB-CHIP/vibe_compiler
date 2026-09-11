//===- TiledOutputStores.cpp - Forward collected tiles to DDR ------------===//

#include "TiledOutputStores.h"

#include "BoundaryMovement.h"
#include "StructuredBufferRelations.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace wafer::compiler::detail {
namespace {

bool isSameSubview(mlir::Value lhs, mlir::Value rhs) {
  if (lhs == rhs)
    return true;
  auto left = lhs.getDefiningOp<mlir::memref::SubViewOp>();
  auto right = rhs.getDefiningOp<mlir::memref::SubViewOp>();
  return left && right && left.getSource() == right.getSource() &&
         left.getType() == right.getType() &&
         left.getMixedOffsets() == right.getMixedOffsets() &&
         left.getMixedSizes() == right.getMixedSizes() &&
         left.getMixedStrides() == right.getMixedStrides();
}

// Follow only exact SCF identity forwarding. A nested loop result can be
// the same buffer as an outer iter_arg even though the SSA values differ.
bool forwardsArgument(mlir::Value value, mlir::BlockArgument argument) {
  if (value == argument)
    return true;
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  auto loop = result ? mlir::dyn_cast<mlir::scf::ForOp>(result.getOwner())
                     : mlir::scf::ForOp{};
  if (!loop)
    return false;
  unsigned index = result.getResultNumber();
  auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  return forwardsArgument(yield.getOperand(index),
                          loop.getRegionIterArg(index)) &&
         forwardsArgument(loop.getInitArgs()[index], argument);
}

struct OutputWrites {
  mlir::memref::AllocOp allocation;
  llvm::SmallVector<mlir::Value, 16> aliases;
  llvm::SmallVector<mlir::Operation *, 8> writes;
  llvm::SmallVector<mlir::memref::CopyOp, 8> identityCopies;
  llvm::SmallVector<llvm::SmallVector<mlir::memref::SubViewOp, 2>, 2>
      destinationViews;
};

// This is a proof over existing buffers and effects, not an output inventory.
// Any read, changed loop state or escape leaves the current storage unchanged.
static bool isWritableDDRRoot(mlir::Value root) {
  if (root.getDefiningOp<mlir::memref::AllocOp>())
    return true;
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(root);
  auto function = argument ? mlir::dyn_cast<mlir::func::FuncOp>(
                                 argument.getOwner()->getParentOp())
                           : mlir::func::FuncOp{};
  if (!function)
    return false;
  auto binding = function.getArgAttrOfType<DDRBindingAttr>(
      argument.getArgNumber(), kWaferDDRBindingAttrName);
  return binding && binding.getAccess() == DDRAccess::Write;
}

std::optional<OutputWrites>
collectOutputWrites(llvm::ArrayRef<StorageStoreOp> terminals) {
  if (terminals.empty())
    return std::nullopt;
  auto terminal = terminals.front();
  auto region = terminal->getParentOfType<TileRegionOp>();
  if (!region || terminal->getBlock() != &region.getBody().front())
    return std::nullopt;
  StorageRootMemo roots;
  const auto &sourceRoots = roots.getStorageRoots(terminal.getSource());
  if (sourceRoots.size() != 1)
    return std::nullopt;
  OutputWrites result;
  result.allocation =
      sourceRoots.begin()->getDefiningOp<mlir::memref::AllocOp>();
  if (!result.allocation ||
      result.allocation->getBlock() != terminal->getBlock())
    return std::nullopt;
  mlir::DominanceInfo dominance(region);
  for (auto store : terminals) {
    if (store->getParentOfType<TileRegionOp>() != region ||
        store->getBlock() != terminal->getBlock() ||
        result.allocation.getType().getShape() !=
            mlir::cast<mlir::MemRefType>(store.getSource().getType())
                .getShape())
      return std::nullopt;
    llvm::SmallVector<mlir::memref::SubViewOp, 2> views;
    llvm::DenseSet<mlir::Value> destinationAliases;
    mlir::Value destination = store.getDest();
    while (true) {
      if (!destination.hasOneUse())
        return std::nullopt;
      destinationAliases.insert(destination);
      auto view = destination.getDefiningOp<mlir::memref::SubViewOp>();
      if (!view)
        break;
      for (mlir::Value parameter : llvm::drop_begin(view->getOperands()))
        if (!dominance.dominates(parameter, result.allocation))
          return std::nullopt;
      views.push_back(view);
      destination = view.getSource();
    }
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(destination);
    if (!argument || argument.getOwner() != &region.getBody().front())
      return std::nullopt;
    result.destinationViews.push_back(std::move(views));
    const auto &destinationRoots = roots.getStorageRoots(destination);
    if (destinationRoots.size() != 1 ||
        !isWritableDDRRoot(*destinationRoots.begin()))
      return std::nullopt;
    bool aliased = false;
    region.walk([&](mlir::Operation *operation) {
      if (operation == region.getOperation())
        return;
      for (auto operand : operation->getOperands())
        if (!destinationAliases.contains(operand) &&
            mlir::isa<mlir::BaseMemRefType>(operand.getType()) &&
            roots.getStorageRoots(operand).contains(*destinationRoots.begin()))
          aliased = true;
    });
    if (aliased)
      return std::nullopt;
    mlir::Value source = store.getSource();
    while (auto loop = source.getDefiningOp<mlir::scf::ForOp>())
      source = loop.getInitArgs()[mlir::cast<mlir::OpResult>(source)
                                      .getResultNumber()];
    if (source != result.allocation.getResult())
      return std::nullopt;
  }
  llvm::DenseSet<mlir::Value> seen;
  auto append = [&](mlir::Value value) {
    if (seen.insert(value).second)
      result.aliases.push_back(value);
  };
  append(result.allocation.getResult());
  llvm::DenseSet<mlir::Operation *> seenCopies;
  for (size_t index = 0; index < result.aliases.size(); ++index) {
    mlir::Value value = result.aliases[index];
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (use.getOperandNumber() == 0 &&
          llvm::any_of(terminals, [&](auto store) {
            return store.getOperation() == user;
          }))
        continue;
      if (auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(user)) {
        if (use.getOperandNumber() != 0)
          return std::nullopt;
        append(subview.getResult());
        continue;
      }
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(user)) {
        if (use.getOperandNumber() < loop.getNumControlOperands())
          return std::nullopt;
        unsigned argument =
            use.getOperandNumber() - loop.getNumControlOperands();
        auto yield =
            mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
        mlir::BlockArgument iterArg = loop.getRegionIterArg(argument);
        if (!forwardsArgument(yield.getOperand(argument), iterArg))
          return std::nullopt;
        append(iterArg);
        append(loop.getResult(argument));
        continue;
      }
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(user)) {
        auto loop = mlir::dyn_cast<mlir::scf::ForOp>(yield->getParentOp());
        if (!loop || !forwardsArgument(
                         value, loop.getRegionIterArg(use.getOperandNumber())))
          return std::nullopt;
        continue;
      }
      if (auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(user)) {
        if (isSameSubview(copy.getSource(), copy.getTarget())) {
          if (seenCopies.insert(user).second)
            result.identityCopies.push_back(copy);
          continue;
        }
      }
      if (mlir::isa<MoveCopyIntoOp, mlir::memref::CopyOp>(user)) {
        if (use.getOperandNumber() != 1 ||
            !isWaferSPMMemRefType(user->getOperand(0).getType()))
          return std::nullopt;
        mlir::Operation *beforeStore = user;
        while (beforeStore && beforeStore->getBlock() != terminal->getBlock())
          beforeStore = beforeStore->getParentOp();
        if (!beforeStore || llvm::any_of(terminals, [&](auto store) {
              return !beforeStore->isBeforeInBlock(store);
            }))
          return std::nullopt;
        result.writes.push_back(user);
        continue;
      }
      return std::nullopt;
    }
  }
  if (result.writes.empty())
    return std::nullopt;
  return result;
}

// The carrier has no observable reads and every removed state is proved to
// forward its init. Transfer the body as in pinned SCF's iter-arg folder.
void removeOutputLoopArguments(mlir::scf::ForOp loop,
                               const llvm::SmallBitVector &removed,
                               mlir::IRRewriter &rewriter) {
  llvm::SmallVector<mlir::Value> inits;
  for (auto [index, init] : llvm::enumerate(loop.getInitArgs()))
    if (!removed[index])
      inits.push_back(init);
  rewriter.setInsertionPoint(loop);
  auto replacement = rewriter.create<mlir::scf::ForOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      inits);
  replacement->setAttrs(loop->getAttrs());
  if (inits.empty())
    rewriter.eraseOp(replacement.getBody()->getTerminator());
  llvm::SmallVector<mlir::Value> arguments{replacement.getInductionVar()};
  llvm::SmallVector<mlir::Value> results;
  unsigned retained = 0;
  for (auto [index, init] : llvm::enumerate(loop.getInitArgs())) {
    arguments.push_back(
        removed[index] ? init : replacement.getRegionIterArg(retained));
    results.push_back(removed[index] ? init
                                     : replacement.getResult(retained++));
  }
  rewriter.mergeBlocks(loop.getBody(), replacement.getBody(), arguments);
  auto yield =
      mlir::cast<mlir::scf::YieldOp>(replacement.getBody()->getTerminator());
  llvm::SmallVector<mlir::Value> values;
  for (auto [index, value] : llvm::enumerate(yield.getOperands()))
    if (!removed[index])
      values.push_back(value);
  rewriter.modifyOpInPlace(yield, [&] { yield->setOperands(values); });
  rewriter.replaceOp(loop, results);
}

void applyOutputWrites(llvm::ArrayRef<StorageStoreOp> terminals,
                       const OutputWrites &writes, mlir::IRRewriter &rewriter,
                       BoundaryMovementStatistics &statistics) {
  llvm::SmallVector<mlir::Value, 2> destinations;
  rewriter.setInsertionPoint(writes.allocation);
  for (auto [index, views] : llvm::enumerate(writes.destinationViews)) {
    StorageStoreOp store = terminals[index];
    mlir::IRMapping mapping;
    for (auto view : llvm::reverse(views))
      rewriter.clone(*view, mapping);
    destinations.push_back(mapping.lookupOrDefault(store.getDest()));
  }
  for (mlir::memref::CopyOp copy : writes.identityCopies)
    rewriter.eraseOp(copy);
  llvm::SmallVector<mlir::scf::ForOp> loops;
  llvm::DenseMap<mlir::Operation *, llvm::SmallBitVector> removedArguments;
  llvm::SmallVector<mlir::memref::SubViewOp> oldViews;
  for (mlir::Value value : writes.aliases) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      auto loop =
          mlir::cast<mlir::scf::ForOp>(argument.getOwner()->getParentOp());
      auto [found, inserted] =
          removedArguments.try_emplace(loop, loop.getNumRegionIterArgs());
      if (inserted)
        loops.push_back(loop);
      found->second.set(argument.getArgNumber() - 1);
    } else if (auto view = value.getDefiningOp<mlir::memref::SubViewOp>()) {
      oldViews.push_back(view);
    }
  }
  auto retarget = [&](auto &&self, mlir::Value view,
                      mlir::Value destination) -> mlir::Value {
    if (view == writes.allocation->getResult(0))
      return destination;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(view)) {
      auto loop =
          mlir::cast<mlir::scf::ForOp>(argument.getOwner()->getParentOp());
      return self(self, loop.getInitArgs()[argument.getArgNumber() - 1],
                  destination);
    }
    if (auto loop = view.getDefiningOp<mlir::scf::ForOp>())
      return self(self,
                  loop.getInitArgs()[mlir::cast<mlir::OpResult>(view)
                                         .getResultNumber()],
                  destination);
    auto subview = view.getDefiningOp<mlir::memref::SubViewOp>();
    assert(subview && "collected output alias is a subview or identity loop");
    auto source = self(self, subview.getSource(), destination);
    auto type = mlir::memref::SubViewOp::inferRankReducedResultType(
        subview.getType().getShape(),
        mlir::cast<mlir::MemRefType>(source.getType()),
        subview.getMixedOffsets(), subview.getMixedSizes(),
        subview.getMixedStrides());
    return rewriter.create<mlir::memref::SubViewOp>(
        subview.getLoc(), mlir::cast<mlir::MemRefType>(type), source,
        subview.getMixedOffsets(), subview.getMixedSizes(),
        subview.getMixedStrides());
  };
  for (auto *operation : writes.writes) {
    mlir::Value source, destination;
    if (auto copy = mlir::dyn_cast<MoveCopyIntoOp>(operation)) {
      source = copy.getSource();
      destination = copy.getDest();
    } else {
      auto memrefCopy = mlir::cast<mlir::memref::CopyOp>(operation);
      source = memrefCopy.getSource();
      destination = memrefCopy.getTarget();
    }
    rewriter.setInsertionPoint(operation);
    for (auto outputDestination : destinations) {
      auto output = retarget(retarget, destination, outputDestination);
      rewriter.create<StorageStoreOp>(operation->getLoc(), source, output);
    }
    rewriter.eraseOp(operation);
  }
  for (auto store : terminals)
    rewriter.eraseOp(store);
  for (const auto &views : writes.destinationViews)
    for (auto view : views)
      rewriter.eraseOp(view);
  for (auto loop : llvm::reverse(loops))
    removeOutputLoopArguments(loop, removedArguments.find(loop)->second,
                              rewriter);
  for (auto view : llvm::reverse(oldViews)) {
    assert(view->use_empty() && "output alias lost an actual use");
    rewriter.eraseOp(view);
  }
  rewriter.eraseOp(writes.allocation);
  statistics.ddrStores += terminals.size() * (writes.writes.size() - 1);
  ++statistics.streamedOutputCarriers;
}

} // namespace

void materializeTiledOutputStores(mlir::ModuleOp module,
                                  BoundaryMovementStatistics &statistics) {
  // Every successful round removes an actual allocation. New terminal stores
  // can expose an inner carrier, so its proof must use the updated IR.
  while (true) {
    llvm::SmallVector<llvm::SmallVector<StorageStoreOp, 2>, 16> groups;
    llvm::DenseMap<mlir::Value, unsigned> byAllocation;
    StorageRootMemo roots;
    module.walk([&](StorageStoreOp store) {
      const auto &sourceRoots = roots.getStorageRoots(store.getSource());
      if (sourceRoots.size() != 1 ||
          !sourceRoots.begin()->getDefiningOp<mlir::memref::AllocOp>())
        return;
      auto [found, inserted] =
          byAllocation.try_emplace(*sourceRoots.begin(), groups.size());
      if (inserted)
        groups.emplace_back();
      groups[found->second].push_back(store);
    });
    mlir::IRRewriter rewriter(module.getContext());
    bool changed = false;
    for (const auto &stores : groups)
      if (auto writes = collectOutputWrites(stores)) {
        applyOutputWrites(stores, *writes, rewriter, statistics);
        changed = true;
      }
    if (!changed)
      break;
  }
}

} // namespace wafer::compiler::detail
