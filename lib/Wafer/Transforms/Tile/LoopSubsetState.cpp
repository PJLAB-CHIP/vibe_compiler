//===- LoopSubsetState.cpp - Local tensor recurrence normalization --------===//

#include "LoopSubsetState.h"
#include "TensorPreparation.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace wafer::compiler::detail {
namespace {

// Publish a conditional subset on its incoming edges. Joining a compact
// updated slice with an unchanged strided slice before insertion loses the
// exact copy geometry during bufferization. Tensor insertion is pure; this
// rewrite is legal only when its other operands already dominate the branch
// and the original conditional slice has no independent observer.
void distributeConditionalInsertions(mlir::IRRewriter &rewriter,
                                     mlir::ModuleOp module) {
  llvm::SmallVector<mlir::tensor::InsertSliceOp> pending;
  module.walk([&](TileRegionOp region) {
    region.walk([&](mlir::tensor::InsertSliceOp insertion) {
      pending.push_back(insertion);
    });
  });
  for (size_t index = 0; index < pending.size(); ++index) {
    auto insertion = pending[index];
    auto source = mlir::dyn_cast<mlir::OpResult>(insertion.getSource());
    auto branch = source ? mlir::dyn_cast<mlir::scf::IfOp>(source.getOwner())
                         : mlir::scf::IfOp{};
    if (!branch || !source.hasOneUse() || branch.getElseRegion().empty())
      continue;
    mlir::DominanceInfo dominance(branch->getParentOp());
    if (!llvm::all_of(insertion->getOperands().drop_front(),
                      [&](mlir::Value operand) {
                        return dominance.properlyDominates(operand, branch);
                      }))
      continue;
    for (mlir::scf::YieldOp yield : {branch.thenYield(), branch.elseYield()}) {
      rewriter.setInsertionPoint(yield);
      mlir::IRMapping mapping;
      mapping.map(source, yield.getOperand(source.getResultNumber()));
      auto edge = mlir::cast<mlir::tensor::InsertSliceOp>(
          rewriter.clone(*insertion, mapping));
      rewriter.modifyOpInPlace(yield, [&] {
        yield->setOperand(source.getResultNumber(), edge.getResult());
      });
      pending.push_back(edge);
    }
    rewriter.modifyOpInPlace(
        branch, [&] { source.setType(insertion.getResult().getType()); });
    rewriter.replaceOp(insertion, source);
  }
}

// The pinned subset helper does not correctly handle collection through nested
// loop state or forks (upstream llvm-project#188761). Prove a direct single
// subset chain before calling it; matching, disjointness and invariant indices
// remain the standard helper's responsibility.
bool hasDirectSubsetChain(mlir::scf::ForOp loop, mlir::BlockArgument argument) {
  mlir::Value value = argument;
  auto *yield = loop.getBody()->getTerminator();
  unsigned index = argument.getArgNumber() - 1;
  while (true) {
    mlir::Value next;
    bool yielded = false;
    for (mlir::OpOperand &use : value.getUses()) {
      auto *owner = use.getOwner();
      if (owner->getBlock() != loop.getBody())
        return false;
      if (owner == yield) {
        if (yielded || next || use.getOperandNumber() != index)
          return false;
        yielded = true;
      } else if (auto extraction =
                     mlir::dyn_cast<mlir::SubsetExtractionOpInterface>(owner)) {
        if (&use != &extraction.getSourceOperand())
          return false;
      } else if (auto insertion =
                     mlir::dyn_cast<mlir::SubsetInsertionOpInterface>(owner)) {
        if (&use != &insertion.getDestinationOperand() || next || yielded)
          return false;
        next = insertion.getUpdatedDestination();
      } else {
        return false;
      }
    }
    if (yielded)
      return true;
    if (!next)
      return false;
    value = next;
  }
}

bool canPromoteSubsets(mlir::scf::ForOp loop) {
  // Extracting before a possibly empty loop could execute an originally
  // unobserved, out-of-bounds subset. Do not speculate that access.
  auto lower = mlir::getConstantIntValue(loop.getLowerBound());
  auto upper = mlir::getConstantIntValue(loop.getUpperBound());
  auto step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower)
    return false;
  for (mlir::BlockArgument argument : loop.getRegionIterArgs())
    if (mlir::isa<mlir::TensorType>(argument.getType()) &&
        !hasDirectSubsetChain(loop, argument))
      return false;
  return true;
}

} // namespace

mlir::LogicalResult
normalizeLoopSubsetState(mlir::ModuleOp module,
                         StructuredMaterializationRelations &relations,
                         mlir::RewriterBase::Listener *externalListener) {
  support::ScopedCompileTimingSpan timing(
      "transform", "layout-and-bufferization", "loop-subset-state");
  StructuredBufferReplacementListener listener(relations, externalListener);
  mlir::IRRewriter rewriter(module.getContext(), &listener);
  distributeConditionalInsertions(rewriter, module);
  if (!listener.finalizeAfterRewrite() || mlir::failed(mlir::verify(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations)))
    return mlir::failure();
  mlir::RewritePatternSet patterns(module.getContext());
  mlir::scf::ForOp::getCanonicalizationPatterns(patterns, module.getContext());
  mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  preserveTensorIterationScopes(patterns);
  mlir::GreedyRewriteConfig config;
  config.maxIterations = 10;
  config.listener = &listener;
  mlir::FrozenRewritePatternSet frozen(std::move(patterns));
  uint64_t examined = 0, promoted = 0;
  bool valid = true;
  while (valid) {
    uint64_t roundPromoted = 0;
    module.walk([&](TileRegionOp region) {
      region.walk([&](mlir::scf::ForOp loop) {
        ++examined;
        if (!canPromoteSubsets(loop))
          return;
        unsigned before = loop.getNumRegionIterArgs();
        auto result = mlir::hoistLoopInvariantSubsets(
            rewriter,
            mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
        roundPromoted += result.getRegionIterArgs().size() - before;
      });
    });
    if (!roundPromoted)
      break;
    promoted += roundPromoted;
    // Each promotion moves a subset through one enclosing loop. Removing the
    // old identity carrier exposes proofs in outer loops on the next round;
    // the nested-state guard must inspect that fresh IR, not the old carrier.
    auto walked = module.walk([&](TileRegionOp region) {
      config.scope = &region.getBody();
      return mlir::failed(mlir::applyPatternsAndFoldGreedily(region.getBody(),
                                                             frozen, config))
                 ? mlir::WalkResult::interrupt()
                 : mlir::WalkResult::advance();
    });
    valid = !walked.wasInterrupted() && listener.finalizeAfterRewrite() &&
            mlir::succeeded(mlir::verify(module)) &&
            mlir::succeeded(
                checkStructuredBufferRelationsCurrent(module, relations));
  }
  support::addCompileCounter("loop-subsets", "examined-loops", examined);
  support::addCompileCounter("loop-subsets", "promoted-subsets", promoted);
  return mlir::success(valid);
}

} // namespace wafer::compiler::detail
