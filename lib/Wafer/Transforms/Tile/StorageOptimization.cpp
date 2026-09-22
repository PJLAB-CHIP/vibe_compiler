//===- StorageOptimization.cpp - Tile StorageOptimization
//-------------------===//

#include "Wafer/Transforms/Tile/StorageOptimization.h"
#include "StorageInitialization.h"
#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace wafer::compiler::detail {
namespace {
static mlir::BlockArgument
getDeadLoopCarriedDestination(mlir::Operation *operation, mlir::Value result,
                              bool allowCurrentOperationRead = false) {
  auto loop = operation->getParentOfType<mlir::scf::ForOp>();
  if (!loop || operation->getBlock() != loop.getBody())
    return {};
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || !result.hasOneUse() ||
      *result.user_begin() != yield.getOperation())
    return {};
  auto yielded = llvm::find(yield.getOperands(), result);
  if (yielded == yield.getOperands().end())
    return {};
  unsigned index = static_cast<unsigned>(
      std::distance(yield.getOperands().begin(), yielded));
  if (index >= loop.getRegionIterArgs().size())
    return {};
  mlir::BlockArgument destination = loop.getRegionIterArgs()[index];
  if (destination.getType() != result.getType())
    return {};
  for (mlir::Operation *user : destination.getUsers()) {
    mlir::Operation *anchor = loop.getBody()->findAncestorOpInBlock(*user);
    if (!anchor ||
        (anchor != operation && !anchor->isBeforeInBlock(operation)) ||
        (anchor == operation && !allowCurrentOperationRead))
      return {};
  }
  return destination;
}

static bool hasMapFreeEquivalent(ComputeElementwiseOp elementwise) {
  mlir::Type resultType = elementwise.getResult().getType();
  if (llvm::any_of(elementwise.getInputs(), [&](mlir::Value input) {
        return !mlir::isa<mlir::FloatType>(input.getType()) &&
               input.getType() != resultType;
      }))
    return false;
  mlir::ArrayAttr maps = elementwise.getIndexingMapsAttr();
  if (!maps)
    return true;
  auto resultMemref = mlir::dyn_cast<mlir::MemRefType>(resultType);
  if (!resultMemref || maps.size() != elementwise.getInputs().size() + 1)
    return false;
  return llvm::all_of(maps, [&](mlir::Attribute attribute) {
    auto map = mlir::dyn_cast<mlir::AffineMapAttr>(attribute);
    return map && map.getValue().getNumDims() == resultMemref.getRank() &&
           map.getValue().getNumSymbols() == 0 &&
           (map.getValue().isIdentity() || map.getValue().getNumResults() == 0);
  });
}

struct CompleteCopy {
  mlir::Operation *operation;
  mlir::Value source;
  mlir::Value dest;
};

static std::optional<CompleteCopy> getCompleteCopy(mlir::Operation *operation) {
  CompleteCopy copy;
  if (auto into = mlir::dyn_cast_or_null<MoveCopyIntoOp>(operation))
    copy = {operation, into.getSource(), into.getDest()};
  else if (auto memref =
               mlir::dyn_cast_or_null<mlir::memref::CopyOp>(operation))
    copy = {operation, memref.getSource(), memref.getTarget()};
  else
    return std::nullopt;
  if (copy.source.getType() != copy.dest.getType() ||
      !isWaferSPMMemRefType(copy.dest.getType()))
    return std::nullopt;
  return copy;
}

// Earlier reads (including psum) still need the intermediate allocation.
// Only its overwritten contents may disappear: after this write, the next
// copy must be its sole observer, including through pre-existing aliases.
static bool isPrivateCopyIntermediate(
    const CompleteCopy &write, const CompleteCopy &read,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations) {
  auto allocation = write.dest.getDefiningOp<mlir::memref::AllocOp>();
  if (!allocation || allocation->getBlock() != write.operation->getBlock() ||
      pipelineOperations.contains(allocation))
    return false;
  llvm::SmallVector<mlir::Value> pending{write.dest};
  llvm::DenseSet<mlir::Value> visited;
  while (!pending.empty()) {
    mlir::Value alias = pending.pop_back_val();
    if (!visited.insert(alias).second)
      continue;
    for (mlir::OpOperand &use : alias.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (alias == write.dest &&
          ((user == write.operation && use.getOperandNumber() == 1) ||
           (user == read.operation && use.getOperandNumber() == 0)))
        continue;
      if (user->getBlock() != write.operation->getBlock() ||
          !user->isBeforeInBlock(write.operation) || user->getNumRegions() ||
          pipelineOperations.contains(user))
        return false;
      if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user)) {
        if (view.getViewSource() != alias || !mlir::isMemoryEffectFree(user))
          return false;
        pending.append(user->getResults().begin(), user->getResults().end());
        continue;
      }
      auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user);
      if (!effects)
        return false;
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
      effects.getEffectsOnValue(alias, instances);
      if (instances.empty() || llvm::any_of(instances, [](const auto &effect) {
            return !mlir::isa<mlir::MemoryEffects::Read,
                              mlir::MemoryEffects::Write>(effect.getEffect());
          }))
        return false;
      effects.getEffects(instances);
      if (llvm::any_of(instances, [](const auto &effect) {
            return !effect.getValue() &&
                   effect.getResource() ==
                       mlir::SideEffects::DefaultResource::get();
          }))
        return false;
      for (mlir::Value result : user->getResults())
        if (mlir::isa<mlir::ShapedType>(result.getType()) &&
            !effects.getEffectOnValue<mlir::MemoryEffects::Allocate>(result))
          return false;
    }
  }
  return true;
}

static void eliminateGemmWritebacks(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations) {
  llvm::SmallVector<ComputeGemmOp> gemms;
  module.walk([&](ComputeGemmOp gemm) { gemms.push_back(gemm); });
  for (auto gemm : gemms) {
    if (!gemm.getResult().hasOneUse() || pipelineOperations.contains(gemm))
      continue;
    // Follow only the actual adjacent result chain. Inverting a proven view
    // keeps the final destination's storage identity and all of its readers.
    mlir::Value source = gemm.getResult();
    mlir::Operation *next = gemm->getNextNode();
    llvm::SmallVector<ViewReshapeOp> views;
    while (auto view = mlir::dyn_cast_or_null<ViewReshapeOp>(next)) {
      if (view.getSource() != source || !view.getResult().hasOneUse() ||
          pipelineOperations.contains(view) ||
          mlir::failed(
              analysis::TransferRealizability::proveStaticReshapeMetadataView(
                  mlir::cast<mlir::MemRefType>(view.getSource().getType()),
                  mlir::cast<mlir::MemRefType>(view.getResult().getType()),
                  /*destinationMayWrite=*/false)))
        break;
      views.push_back(view);
      source = view.getResult();
      next = view->getNextNode();
    }
    auto first = getCompleteCopy(next);
    if (!first || first->source != source ||
        pipelineOperations.contains(first->operation))
      continue;
    llvm::SmallVector<CompleteCopy> copies{*first};
    while (auto following =
               getCompleteCopy(copies.back().operation->getNextNode())) {
      if (following->source != copies.back().dest ||
          pipelineOperations.contains(following->operation) ||
          !isPrivateCopyIntermediate(copies.back(), *following,
                                     pipelineOperations))
        break;
      copies.push_back(*following);
    }
    size_t selected = 0;
    {
      mlir::AliasAnalysis aliases(module);
      mlir::DominanceInfo dominance(module);
      for (auto [index, copy] : llvm::enumerate(copies)) {
        if (!dominance.dominates(copy.dest, gemm) ||
            llvm::any_of(gemm->getOperands(),
                         [&](mlir::Value input) {
                           return !aliases.alias(input, copy.dest).isNo();
                         }) ||
            llvm::any_of(
                llvm::ArrayRef(copies).take_front(index),
                [&](const CompleteCopy &previous) {
                  return !aliases.alias(previous.dest, copy.dest).isNo();
                }))
          continue;
        if (copy.dest.getType() != gemm.getResult().getType() &&
            mlir::failed(
                analysis::TransferRealizability::proveStaticReshapeMetadataView(
                    mlir::cast<mlir::MemRefType>(copy.dest.getType()),
                    mlir::cast<mlir::MemRefType>(gemm.getResult().getType()),
                    /*destinationMayWrite=*/true)))
          continue;
        selected = index + 1;
      }
    }
    if (!selected)
      continue;
    rewriter.setInsertionPoint(gemm);
    mlir::Value dest = copies[selected - 1].dest;
    if (dest.getType() != gemm.getResult().getType())
      dest = rewriter.create<ViewReshapeOp>(gemm.getLoc(),
                                            gemm.getResult().getType(), dest);
    rewriter.create<ComputeGemmIntoOp>(
        gemm.getLoc(), gemm.getLhs(), gemm.getRhs(), dest, gemm.getPsum(),
        gemm.getLhsOrientationAttr(), gemm.getRhsOrientationAttr(),
        gemm.getBatchCountAttr(), gemm.getLhsBatchDimsAttr(),
        gemm.getLhsMDimAttr(), gemm.getLhsContractingDimAttr(),
        gemm.getRhsBatchDimsAttr(), gemm.getRhsContractingDimAttr(),
        gemm.getRhsNDimAttr(), gemm.getResultBatchDimsAttr(),
        gemm.getResultMDimAttr(), gemm.getResultNDimAttr());
    for (const auto &copy :
         llvm::reverse(llvm::ArrayRef(copies).take_front(selected)))
      rewriter.eraseOp(copy.operation);
    for (auto view : llvm::reverse(views))
      rewriter.eraseOp(view);
    rewriter.eraseOp(gemm);
  }
}

// A private DPS publication is not an observable storage identity. Remove
// its exact copy before choosing destinations, so the existing last-use
// optimization can still see the allocation-producing expression chain.
static void eliminatePrivatePointwisePublications(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations) {
  llvm::SmallVector<MoveCopyIntoOp> copies;
  module.walk([&](MoveCopyIntoOp copy) { copies.push_back(copy); });
  for (auto copy : copies) {
    auto source = copy.getSource();
    auto *producer = source.getDefiningOp();
    auto allocation = copy.getDest().getDefiningOp<mlir::memref::AllocOp>();
    if (!mlir::isa_and_nonnull<ComputeElementwiseOp, ComputeConvertOp>(
            producer) ||
        !allocation || !source.hasOneUse() ||
        allocation.getResult().hasOneUse() ||
        source.getType() != allocation.getType() ||
        producer->getBlock() != copy->getBlock() ||
        allocation->getBlock() != copy->getBlock() ||
        !producer->isBeforeInBlock(copy) ||
        pipelineOperations.contains(producer) ||
        pipelineOperations.contains(copy) ||
        pipelineOperations.contains(allocation))
      continue;
    auto producerEffects =
        mlir::dyn_cast<mlir::MemoryEffectOpInterface>(producer);
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance> allocationEffects;
    if (!producerEffects)
      continue;
    producerEffects.getEffectsOnValue(source, allocationEffects);
    if (!llvm::any_of(allocationEffects, [](const auto &effect) {
          return mlir::isa<mlir::MemoryEffects::Allocate>(effect.getEffect());
        }))
      continue;
    bool privateReads = llvm::all_of(
        allocation.getResult().getUsers(), [&](mlir::Operation *user) {
          if (user == copy)
            return true;
          if (user->getBlock() != copy->getBlock() ||
              !copy->isBeforeInBlock(user) ||
              pipelineOperations.contains(user) || user->getNumRegions() ||
              mlir::isa<mlir::ViewLikeOpInterface>(user))
            return false;
          auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user);
          if (!effects)
            return false;
          llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
          effects.getEffects(instances);
          bool reads = false;
          for (const auto &effect : instances) {
            if (!effect.getValue() &&
                effect.getResource() ==
                    mlir::SideEffects::DefaultResource::get())
              return false;
            if (effect.getValue() == allocation.getResult()) {
              if (!mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()))
                return false;
              reads = true;
            }
          }
          // A read effect alone does not prove that a returned view cannot
          // escape. Buffer results must own fresh storage, not alias the read.
          for (mlir::Value result : user->getResults())
            if (mlir::isa<mlir::MemRefType>(result.getType()) &&
                !effects.getEffectOnValue<mlir::MemoryEffects::Allocate>(
                    result))
              return false;
          return reads;
        });
    if (!privateReads)
      continue;
    rewriter.replaceAllUsesWith(allocation.getResult(), source);
    rewriter.eraseOp(copy);
    rewriter.eraseOp(allocation);
  }
}

static void eliminateElementwiseWritebacks(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations) {
  module.walk([&](MoveCopyIntoOp copy) {
    auto elementwise = copy.getSource().getDefiningOp<ComputeElementwiseOp>();
    if (!elementwise || !elementwise.getResult().hasOneUse() ||
        elementwise->getNextNode() != copy.getOperation() ||
        copy.getSource().getType() != copy.getDest().getType() ||
        pipelineOperations.contains(elementwise) ||
        pipelineOperations.contains(copy))
      return;

    // The explicit write already happens immediately after the computation.
    // Exact destination reads are supported by elementwise_into; a different
    // view may overlap only part of an input and cannot be updated in place.
    // Construct fresh analysis for this IR epoch, before making any mutation.
    {
      mlir::AliasAnalysis aliases(module);
      auto maps = elementwise.getIndexingMapsAttr();
      for (auto [index, input] : llvm::enumerate(elementwise.getInputs())) {
        // Only the aliased operand needs identity coordinates. Other inputs
        // may have independent broadcast or permutation maps; lowering
        // materializes those reads before issuing the destination update.
        const bool allowExactDestination =
            elementwise.getKind() != ComputeElementwiseKind::Select &&
            (!maps || mlir::cast<mlir::AffineMapAttr>(maps[index])
                          .getValue()
                          .isIdentity());
        if (mlir::isa<mlir::MemRefType>(input.getType()) &&
            !(allowExactDestination && input == copy.getDest()) &&
            !aliases.alias(input, copy.getDest()).isNo())
          return;
      }
    }
    rewriter.setInsertionPoint(elementwise);
    rewriter.create<ComputeElementwiseIntoOp>(
        elementwise.getLoc(), elementwise.getKindAttr(),
        elementwise.getInputs(), copy.getDest(),
        elementwise.getIndexingMapsAttr());
    rewriter.eraseOp(copy);
    rewriter.eraseOp(elementwise);
  });
}

// Reuse a private, last-use allocation only in the same dynamic scope. An
// operand defined outside a loop is deliberately excluded: its single SSA
// use may execute repeatedly and does not make the old contents dead.
static void reuseElementwiseInputs(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations) {
  llvm::SmallVector<ComputeElementwiseOp> operations;
  module.walk([&](ComputeElementwiseOp op) { operations.push_back(op); });
  // Start with the last consumer. Rewriting its producer first would add a
  // destination write use to the allocation and hide the original last-use
  // relation of a whole private expression chain.
  for (auto op : llvm::reverse(operations)) {
    if (pipelineOperations.contains(op))
      continue;
    for (auto [index, input] : llvm::enumerate(op.getInputs())) {
      if (op.getKind() == ComputeElementwiseKind::Select && index != 2)
        continue;
      auto result = mlir::dyn_cast<mlir::OpResult>(input);
      auto producer = result ? result.getOwner() : nullptr;
      if (!producer || producer->getBlock() != op->getBlock() ||
          !input.hasOneUse() || input.getType() != op.getResult().getType() ||
          pipelineOperations.contains(producer))
        continue;
      auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(producer);
      if (!effects)
        continue;
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
      effects.getEffectsOnValue(input, instances);
      if (!llvm::any_of(instances, [](const auto &effect) {
            return mlir::isa<mlir::MemoryEffects::Allocate>(effect.getEffect());
          }))
        continue;
      if (auto maps = op.getIndexingMapsAttr())
        if (!mlir::cast<mlir::AffineMapAttr>(maps[index])
                 .getValue()
                 .isIdentity())
          continue;
      bool safe = true;
      {
        mlir::AliasAnalysis aliases(module);
        for (auto [otherIndex, other] : llvm::enumerate(op.getInputs()))
          if (otherIndex != index &&
              mlir::isa<mlir::MemRefType>(other.getType()) &&
              !aliases.alias(input, other).isNo())
            safe = false;
      }
      if (!safe)
        continue;
      rewriter.setInsertionPoint(op);
      rewriter.create<ComputeElementwiseIntoOp>(op.getLoc(), op.getKindAttr(),
                                                op.getInputs(), input,
                                                op.getIndexingMapsAttr());
      rewriter.replaceOp(op, input);
      break;
    }
  }
}

static mlir::LogicalResult
materializeLoopCarriedDestinations(mlir::ModuleOp module,
                                   mlir::IRRewriter &rewriter) {
  struct Rewrite {
    mlir::Operation *operation = nullptr;
    mlir::Value destination;
  };
  llvm::SmallVector<Rewrite, 8> rewrites;
  module.walk([&](mlir::Operation *operation) {
    mlir::Value result;
    if (auto elementwise = mlir::dyn_cast<ComputeElementwiseOp>(operation)) {
      if (!hasMapFreeEquivalent(elementwise))
        return;
      result = elementwise.getResult();
    } else if (auto materialize =
                   mlir::dyn_cast<LayoutMaterializeOp>(operation)) {
      result = materialize.getResult();
    } else if (auto copy = mlir::dyn_cast<MoveCopyOp>(operation)) {
      if (copy.getDdrResourceAttr())
        return;
      result = copy.getResult();
    } else {
      return;
    }
    const bool allowCurrentOperationRead =
        mlir::isa<ComputeElementwiseOp>(operation);
    if (mlir::BlockArgument destination = getDeadLoopCarriedDestination(
            operation, result, allowCurrentOperationRead))
      rewrites.push_back({operation, destination});
  });

  for (const Rewrite &rewrite : rewrites) {
    rewriter.setInsertionPoint(rewrite.operation);
    if (auto elementwise =
            mlir::dyn_cast<ComputeElementwiseOp>(rewrite.operation)) {
      rewriter.create<ComputeElementwiseIntoOp>(
          elementwise.getLoc(), elementwise.getKindAttr(),
          elementwise.getInputs(), rewrite.destination, mlir::ArrayAttr{});
      rewriter.replaceOp(elementwise, rewrite.destination);
      continue;
    }
    mlir::Value source;
    if (auto materialize =
            mlir::dyn_cast<LayoutMaterializeOp>(rewrite.operation))
      source = materialize.getSource();
    else
      source = mlir::cast<MoveCopyOp>(rewrite.operation).getSource();
    rewriter.create<MoveCopyIntoOp>(rewrite.operation->getLoc(), source,
                                    rewrite.destination);
    rewriter.replaceOp(rewrite.operation, rewrite.destination);
  }
  return mlir::verify(module);
}

} // namespace

void preservePrivateScalarBroadcasts(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter,
    const llvm::DenseSet<mlir::Operation *> &pipelineOperations) {
  module.walk([&](mlir::memref::LoadOp load) {
    auto type = load.getMemRefType();
    auto allocation = load.getMemRef().getDefiningOp<mlir::memref::AllocOp>();
    if (!allocation || type.getRank() != 0 ||
        !mlir::isa<mlir::FloatType>(type.getElementType()) ||
        !isWaferSPMMemRefType(type) || load.getResult().use_empty() ||
        allocation->getBlock() != load->getBlock() ||
        pipelineOperations.contains(load))
      return;
    // A private allocation with exactly this read and one initializing write
    // has no escaping alias or later mutation. Moving its read to CT consumers
    // therefore preserves the loaded value, including inside nested loops.
    StorageLoadOp initialization;
    for (auto *user : allocation.getResult().getUsers()) {
      if (user == load)
        continue;
      auto write = mlir::dyn_cast<StorageLoadOp>(user);
      if (!write || initialization || write.getDest() != allocation ||
          write->getBlock() != load->getBlock() ||
          !write->isBeforeInBlock(load) || pipelineOperations.contains(write))
        return;
      initialization = write;
    }
    if (!initialization)
      return;
    llvm::SmallVector<mlir::OpOperand *> uses;
    for (auto &use : load.getResult().getUses()) {
      auto *consumer = use.getOwner();
      if (!mlir::isa<ComputeElementwiseOp, ComputeElementwiseIntoOp>(
              consumer) ||
          use.getOperandNumber() != 1 || pipelineOperations.contains(consumer))
        return;
      auto *owner = load->getBlock()->findAncestorOpInBlock(*consumer);
      if (!owner || !load->isBeforeInBlock(owner))
        return;
      uses.push_back(&use);
    }
    for (auto *use : uses) {
      auto *consumer = use->getOwner();
      auto update = [&](auto op, mlir::Type output) {
        rewriter.modifyOpInPlace(op, [&] {
          if (!op.getIndexingMapsAttr()) {
            int64_t rank = mlir::cast<mlir::MemRefType>(output).getRank();
            auto identity = rewriter.getMultiDimIdentityMap(rank);
            auto scalar =
                mlir::AffineMap::get(rank, 0, {}, module.getContext());
            op.setIndexingMapsAttr(
                rewriter.getAffineMapArrayAttr({identity, scalar, identity}));
          }
          use->set(allocation.getResult());
        });
      };
      if (auto op = mlir::dyn_cast<ComputeElementwiseOp>(consumer))
        update(op, op.getResult().getType());
      else {
        auto into = mlir::cast<ComputeElementwiseIntoOp>(consumer);
        update(into, into.getDest().getType());
      }
    }
    rewriter.eraseOp(load);
  });
}

mlir::LogicalResult
optimizeStorage(mlir::ModuleOp module, mlir::IRRewriter &rewriter,
                const llvm::DenseSet<mlir::Operation *> &pipelineOperations) {
  eliminateGemmWritebacks(module, rewriter, pipelineOperations);
  eliminatePrivatePointwisePublications(module, rewriter, pipelineOperations);
  eliminateElementwiseWritebacks(module, rewriter, pipelineOperations);
  reuseElementwiseInputs(module, rewriter, pipelineOperations);
  eliminateUnusedStorageInitialization(module, rewriter, pipelineOperations);
  return materializeLoopCarriedDestinations(module, rewriter);
}

} // namespace wafer::compiler::detail
