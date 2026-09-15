//===- GemmFinalization.cpp - Native GEMM output formats ------------------===//
#include "GemmFinalization.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer {
namespace {
bool sameView(mlir::Value a, mlir::Value b) {
  if (a == b)
    return true;
  auto av = a.getDefiningOp<mlir::memref::SubViewOp>();
  auto bv = b.getDefiningOp<mlir::memref::SubViewOp>();
  return av && bv && a.getType() == b.getType() &&
         av.getSource() == bv.getSource() &&
         av.getMixedOffsets() == bv.getMixedOffsets() &&
         av.getMixedSizes() == bv.getMixedSizes() &&
         av.getMixedStrides() == bv.getMixedStrides();
}

struct OutputChain {
  ComputeGemmOp gemm;
  StorageLoadOp load;
  mlir::scf::ForOp loop;
  llvm::SmallVector<mlir::Operation *> copies;
  llvm::SmallVector<mlir::Value> storage;
};

// Follow the value at the time of its actual read. Every copied view must
// cover the complete producer output; unknown writes terminate the proof.
bool traceOutput(mlir::Value value, mlir::Operation *read,
                 mlir::AliasAnalysis &aliases, OutputChain &chain) {
  chain.storage.push_back(value);
  mlir::Value base = value;
  while (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
             base.getDefiningOp())) {
    base = view.getViewSource();
    chain.storage.push_back(base);
  }
  for (auto *previous = read->getPrevNode(); previous;
       previous = previous->getPrevNode()) {
    if (previous == value.getDefiningOp()) {
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(previous)) {
        if (!value.hasOneUse())
          return false;
        auto result = mlir::cast<mlir::OpResult>(value);
        auto yield =
            mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
        OutputChain nested;
        if (traceOutput(yield.getOperand(result.getResultNumber()), yield,
                        aliases, nested) &&
            nested.gemm && !nested.loop &&
            mlir::cast<mlir::MemRefType>(nested.gemm.getResult().getType())
                    .getShape() ==
                mlir::cast<mlir::MemRefType>(value.getType()).getShape()) {
          chain.loop = loop;
          return true;
        }
        return false;
      }
      if (auto gemm = mlir::dyn_cast<ComputeGemmOp>(previous)) {
        chain.gemm = gemm;
        return true;
      }
      // Constructing an alias does not write its storage. The last write may
      // precede this view and use a separately constructed equivalent view.
      if (mlir::isa<mlir::ViewLikeOpInterface>(previous))
        continue;
      mlir::Value source;
      if (auto layout = mlir::dyn_cast<LayoutMaterializeOp>(previous))
        source = layout.getSource();
      if (auto copy = mlir::dyn_cast<MoveCopyOp>(previous))
        source = copy.getSource();
      if (!source ||
          mlir::cast<mlir::MemRefType>(source.getType()).getShape() !=
              mlir::cast<mlir::MemRefType>(value.getType()).getShape())
        return false;
      if (!traceOutput(source, previous, aliases, chain))
        return false;
      chain.copies.push_back(previous);
      return true;
    }
    auto effects = mlir::getEffectsRecursively(previous);
    if (!effects)
      return false;
    bool writes = llvm::any_of(*effects, [&](const auto &effect) {
      return mlir::isa<mlir::MemoryEffects::Write, mlir::MemoryEffects::Free>(
                 effect.getEffect()) &&
             (!effect.getValue() ||
              !aliases.alias(value, effect.getValue()).isNo());
    });
    if (!writes)
      continue;
    if (auto load = mlir::dyn_cast<StorageLoadOp>(previous)) {
      if (load.getDest() != value)
        return false;
      chain.load = load;
      return true;
    }
    mlir::Value source, destination;
    if (auto copy = mlir::dyn_cast<MoveCopyIntoOp>(previous)) {
      source = copy.getSource();
      destination = copy.getDest();
    } else if (auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(previous)) {
      source = copy.getSource();
      destination = copy.getTarget();
    } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(previous)) {
      // A loop with no carried SSA values still has explicit memory effects.
      // Peeling moves its final writes into this block without guessing state.
      if (loop.getNumResults() == 0) {
        OutputChain nested;
        if (traceOutput(value, loop.getBody()->getTerminator(), aliases,
                        nested) &&
            nested.gemm && !nested.loop) {
          chain.loop = loop;
          return true;
        }
      }
    }
    if (!source || !sameView(destination, value))
      return false;
    if (!traceOutput(source, previous, aliases, chain))
      return false;
    chain.storage.push_back(destination);
    chain.copies.push_back(previous);
    return true;
  }
  return false;
}

bool isPrivateStorage(mlir::Value value) {
  while (value) {
    auto *definition = value.getDefiningOp();
    if (mlir::isa_and_nonnull<mlir::memref::AllocOp, LayoutMaterializeOp,
                              MoveCopyOp, ComputeGemmOp>(definition))
      return true;
    if (auto view =
            mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(definition)) {
      value = view.getViewSource();
      continue;
    }
    mlir::scf::ForOp loop;
    unsigned index = 0;
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
      loop = mlir::dyn_cast<mlir::scf::ForOp>(definition);
      index = result.getResultNumber();
    } else if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      loop =
          mlir::dyn_cast<mlir::scf::ForOp>(argument.getOwner()->getParentOp());
      if (!loop || argument.getArgNumber() == 0)
        return false;
      index = argument.getArgNumber() - 1;
    }
    if (!loop)
      return false;
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    if (index >= loop.getNumResults() ||
        yield.getOperand(index) != loop.getRegionIterArgs()[index])
      return false;
    value = loop.getInitArgs()[index];
  }
  return false;
}

bool exclusivelyFinal(OutputChain &chain, mlir::Operation *terminal) {
  for (auto *operation : chain.copies) {
    mlir::Value destination;
    if (auto copy = mlir::dyn_cast<MoveCopyIntoOp>(operation))
      destination = copy.getDest();
    if (auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(operation))
      destination = copy.getTarget();
    if (destination && !isPrivateStorage(destination))
      return false;
  }
  llvm::SmallPtrSet<mlir::Operation *, 16> allowed(chain.copies.begin(),
                                                   chain.copies.end());
  allowed.insert(terminal);
  mlir::Operation *origin =
      chain.gemm ? chain.gemm.getOperation() : chain.load.getOperation();
  allowed.insert(origin);
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
  auto check = [&](auto &&self, mlir::Value value) -> bool {
    for (auto *user : value.getUsers()) {
      if (allowed.contains(user))
        continue;
      if (user->getBlock() == origin->getBlock() &&
          user->isBeforeInBlock(origin)) {
        // Materializing a prior state into distinct storage observes the old
        // value before the final GEMM. Removing the final writeback does not
        // change that snapshot; aliases still follow the recursive view check.
        if (mlir::isa<ComputeFillOp, StorageLoadOp, MoveCopyIntoOp,
                      mlir::memref::CopyOp, LayoutMaterializeOp, MoveCopyOp>(
                user))
          continue;
      }
      if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user)) {
        if (visited.insert(user).second)
          for (auto result : user->getResults())
            if (!self(self, result))
              return false;
        continue;
      }
      return false;
    }
    return true;
  };
  for (auto value : chain.storage)
    if (!check(check, value))
      return false;
  return true;
}

bool peelLast(mlir::scf::ForOp loop, mlir::IRRewriter &rewriter) {
  auto lb = mlir::getConstantIntValue(loop.getLowerBound());
  auto ub = mlir::getConstantIntValue(loop.getUpperBound());
  auto step = mlir::getConstantIntValue(loop.getStep());
  if (!lb || !ub || !step || *step <= 0 || *lb < 0 || *ub <= *lb ||
      !loop.getInductionVar().getType().isIndex() ||
      llvm::any_of(
          loop.getResultTypes(),
          [](mlir::Type type) { return !mlir::isa<mlir::MemRefType>(type); }))
    return false;
  const int64_t last = *lb + ((*ub - *lb - 1) / *step) * *step;
  llvm::SmallVector<llvm::SmallVector<mlir::OpOperand *>> originalUses;
  for (auto result : loop.getResults()) {
    originalUses.emplace_back();
    for (auto &use : result.getUses())
      originalUses.back().push_back(&use);
  }
  rewriter.setInsertionPoint(loop);
  auto bound =
      rewriter.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), last);
  rewriter.modifyOpInPlace(loop, [&] { loop.setUpperBound(bound); });
  rewriter.setInsertionPointAfter(loop);
  mlir::IRMapping mapping;
  mapping.map(loop.getInductionVar(), bound);
  for (auto [argument, result] :
       llvm::zip_equal(loop.getRegionIterArgs(), loop.getResults()))
    mapping.map(argument, result);
  for (auto &operation : loop.getBody()->without_terminator())
    rewriter.clone(operation, mapping);
  auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  for (auto [uses, value] : llvm::zip_equal(originalUses, yield.getOperands()))
    for (auto *use : uses)
      rewriter.modifyOpInPlace(
          use->getOwner(), [&] { use->set(mapping.lookupOrDefault(value)); });
  return true;
}

void eraseDeadViews(mlir::Value value, mlir::IRRewriter &rewriter) {
  for (auto *user : llvm::make_early_inc_range(value.getUsers())) {
    if (!mlir::isa<mlir::ViewLikeOpInterface>(user))
      continue;
    for (auto result : user->getResults())
      eraseDeadViews(result, rewriter);
    if (user->use_empty())
      rewriter.eraseOp(user);
  }
}
// The actual private DDR allocation and its exact region argument bindings
// own this transport. No alias, partial view, external observer, or second
// writer may participate in changing its format.
struct PrivateTransport {
  StorageStoreOp store;
  llvm::SmallVector<mlir::Value> storage;
};

bool tracePrivateTransport(StorageLoadOp load, PrivateTransport &transport) {
  auto consumer = mlir::dyn_cast<TileRegionOp>(load->getParentOp());
  if (!consumer)
    return false;
  mlir::Value source = load.getSource();
  while (auto argument = mlir::dyn_cast<mlir::BlockArgument>(source)) {
    auto region =
        mlir::dyn_cast<TileRegionOp>(argument.getOwner()->getParentOp());
    if (!region || argument.getArgNumber() >= region.getInputs().size())
      return false;
    source = region.getInputs()[argument.getArgNumber()];
  }
  auto allocation = source.getDefiningOp<mlir::memref::AllocOp>();
  if (!allocation || !allocation.getType().hasStaticShape() ||
      allocation->getBlock() != consumer->getBlock())
    return false;
  llvm::SmallVector<mlir::Value> pending{source};
  unsigned reads = 0;
  while (!pending.empty()) {
    auto value = pending.pop_back_val();
    transport.storage.push_back(value);
    for (auto &use : value.getUses()) {
      auto *user = use.getOwner();
      if (auto region = mlir::dyn_cast<TileRegionOp>(user)) {
        if (region->getBlock() != consumer->getBlock() ||
            use.getOperandNumber() >= region.getInputs().size())
          return false;
        pending.push_back(
            region.getBody().front().getArgument(use.getOperandNumber()));
      } else if (user == load && use.getOperandNumber() == 0) {
        ++reads;
      } else if (auto store = mlir::dyn_cast<StorageStoreOp>(user)) {
        if (use.getOperandNumber() != 1 || transport.store)
          return false;
        transport.store = store;
      } else {
        return false;
      }
    }
  }
  if (!transport.store || reads != 1)
    return false;
  auto producer = mlir::dyn_cast<TileRegionOp>(transport.store->getParentOp());
  return producer && producer != consumer &&
         producer->getBlock() == consumer->getBlock() &&
         producer->isBeforeInBlock(consumer);
}

bool findFinalGemm(mlir::Value value, mlir::Operation *read,
                   mlir::ModuleOp module, mlir::IRRewriter &rewriter,
                   OutputChain &chain) {
  {
    mlir::AliasAnalysis aliases(module);
    if (!traceOutput(value, read, aliases, chain))
      return false;
  }
  if (chain.loop) {
    if (!peelLast(chain.loop, rewriter))
      return false;
    chain = {};
    mlir::AliasAnalysis aliases(module);
    if (!traceOutput(read->getOperand(0), read, aliases, chain))
      return false;
  }
  return !chain.loop;
}

bool canNarrowGemm(OutputChain &chain, mlir::MemRefType outputType,
                   mlir::Operation *terminal) {
  return chain.gemm &&
         mlir::cast<mlir::MemRefType>(chain.gemm.getLhs().getType())
                 .getElementType() == outputType.getElementType() &&
         mlir::cast<mlir::MemRefType>(chain.gemm.getResult().getType())
                 .getShape() == outputType.getShape() &&
         exclusivelyFinal(chain, terminal);
}

mlir::Value materializeLayout(mlir::Value result, mlir::MemRefType type,
                              mlir::Operation *read,
                              mlir::IRRewriter &rewriter) {
  rewriter.setInsertionPoint(read);
  if (result.getType() != type)
    result = rewriter.create<LayoutMaterializeOp>(read->getLoc(), type, result);
  return result;
}

mlir::Value narrowGemm(OutputChain &chain, mlir::MemRefType outputType,
                       mlir::Operation *read, mlir::IRRewriter &rewriter) {
  auto nativeType =
      mlir::cast<mlir::MemRefType>(chain.gemm.getResult().getType())
          .clone(outputType.getElementType());
  rewriter.setInsertionPoint(chain.gemm);
  auto native = mlir::cast<ComputeGemmOp>(rewriter.clone(*chain.gemm));
  rewriter.modifyOpInPlace(native,
                           [&] { native.getResult().setType(nativeType); });
  return materializeLayout(native.getResult(), outputType, read, rewriter);
}

void eraseCopies(OutputChain &chain, mlir::IRRewriter &rewriter) {
  for (auto *copy : llvm::reverse(chain.copies)) {
    for (auto result : copy->getResults())
      eraseDeadViews(result, rewriter);
    rewriter.eraseOp(copy);
  }
  if (chain.gemm) {
    eraseDeadViews(chain.gemm.getResult(), rewriter);
    rewriter.eraseOp(chain.gemm);
  }
}

bool narrowPrivateTransport(OutputChain &consumer, ComputeConvertOp convert,
                            mlir::ModuleOp module, mlir::IRRewriter &rewriter) {
  auto load = consumer.load;
  auto destination = load.getDest().getDefiningOp<mlir::memref::AllocOp>();
  if (!destination || !exclusivelyFinal(consumer, convert))
    return false;
  PrivateTransport transport;
  if (!tracePrivateTransport(load, transport))
    return false;
  auto store = transport.store;
  OutputChain producer;
  auto outputType = mlir::cast<mlir::MemRefType>(convert.getResult().getType());
  if (!findFinalGemm(store.getSource(), store, module, rewriter, producer) ||
      !canNarrowGemm(producer, outputType, store))
    return false;
  // All affected values and observers have been proved before changing types.
  // Keep the region and transport choices, and keep every earlier K state F32.
  auto dtype = outputType.getElementType();
  auto storeType =
      mlir::cast<mlir::MemRefType>(store.getSource().getType()).clone(dtype);
  auto native = narrowGemm(producer, mlir::cast<mlir::MemRefType>(storeType),
                           store, rewriter);
  rewriter.modifyOpInPlace(store,
                           [&] { store.getSourceMutable().assign(native); });
  for (auto value : transport.storage) {
    auto *owner = value.getDefiningOp();
    if (!owner)
      owner = mlir::cast<mlir::BlockArgument>(value).getOwner()->getParentOp();
    rewriter.modifyOpInPlace(owner, [&] {
      value.setType(mlir::cast<mlir::MemRefType>(value.getType()).clone(dtype));
    });
  }
  // Recreate the private load allocation instead of changing existing view
  // types. Only the proved terminal chain is bypassed; its dead views keep
  // their original valid types until ordinary DCE removes them.
  rewriter.setInsertionPoint(destination);
  auto narrowDestination =
      mlir::cast<mlir::memref::AllocOp>(rewriter.clone(*destination));
  rewriter.modifyOpInPlace(narrowDestination, [&] {
    narrowDestination.getResult().setType(
        mlir::cast<mlir::MemRefType>(destination.getType().clone(dtype)));
  });
  rewriter.modifyOpInPlace(
      load, [&] { load.getDestMutable().assign(narrowDestination); });
  auto result =
      materializeLayout(narrowDestination, outputType, convert, rewriter);
  rewriter.replaceOp(convert, result);
  eraseCopies(consumer, rewriter);
  eraseDeadViews(destination.getResult(), rewriter);
  if (destination->use_empty())
    rewriter.eraseOp(destination);
  eraseCopies(producer, rewriter);
  return true;
}
} // namespace

mlir::LogicalResult foldGemmOutputConversions(mlir::ModuleOp module) {
  llvm::SmallVector<ComputeConvertOp> conversions;
  module.walk([&](ComputeConvertOp op) { conversions.push_back(op); });
  mlir::IRRewriter rewriter(module.getContext());
  for (auto convert : conversions) {
    auto sourceType =
        mlir::cast<mlir::MemRefType>(convert.getSource().getType());
    auto outputType =
        mlir::cast<mlir::MemRefType>(convert.getResult().getType());
    if (!sourceType.getElementType().isF32() ||
        !(outputType.getElementType().isF16() ||
          outputType.getElementType().isBF16()))
      continue;
    OutputChain chain;
    if (!findFinalGemm(convert.getSource(), convert, module, rewriter, chain))
      continue;
    if (chain.load) {
      narrowPrivateTransport(chain, convert, module, rewriter);
      continue;
    }
    if (!canNarrowGemm(chain, outputType, convert))
      continue;
    auto result = narrowGemm(chain, outputType, convert, rewriter);
    rewriter.replaceOp(convert, result);
    eraseCopies(chain, rewriter);
  }
  return mlir::verify(module);
}
} // namespace wafer
