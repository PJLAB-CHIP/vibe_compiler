//===- LoadPipelining.cpp - Tile LoadPipelining -------------------===//

#include "Wafer/Transforms/Tile/LoadPipelining.h"
#include "LoopPipeliningInternal.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Tile/RotatingBuffers.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <utility>

namespace wafer::compiler::detail {
namespace {
struct LoadPipeline {
  mlir::scf::ForOp loop;
  llvm::SmallVector<StorageLoadOp, 4> loads;
  llvm::SmallVector<mlir::memref::AllocOp, 4> allocations;
};

bool collectLoadDependencies(mlir::Value value, mlir::scf::ForOp loop,
                             llvm::DenseSet<mlir::Operation *> &early) {
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    return argument.getOwner() != loop.getBody() ||
           argument == loop.getInductionVar();
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !loop->isAncestor(definition))
    return true;
  if (definition->getNumRegions() || !mlir::isMemoryEffectFree(definition))
    return false;
  if (!early.insert(definition).second)
    return true;
  return llvm::all_of(definition->getOperands(), [&](mlir::Value operand) {
    return collectLoadDependencies(operand, loop, early);
  });
}

bool hasIterationLocalUses(mlir::Value value, StorageLoadOp load,
                           mlir::scf::ForOp loop,
                           llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *user = use.getOwner();
    if (user == load && use.getOperandNumber() == 1)
      continue;
    if (!loop->isAncestor(user))
      return false;
    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user)) {
      if (view.getViewSource() != value || user->getNumResults() != 1 ||
          !hasIterationLocalUses(user->getResult(0), load, loop, visited))
        return false;
      continue;
    }
    mlir::DominanceInfo dominance(loop);
    if (!dominance.properlyDominates(load, user))
      return false;
    auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user);
    if (!interface)
      return false;
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> effects;
    interface.getEffectsOnValue(value, effects);
    if (effects.empty() || llvm::any_of(effects, [](const auto &effect) {
          return !mlir::isa<mlir::MemoryEffects::Read,
                            mlir::MemoryEffects::Write>(effect.getEffect());
        }))
      return false;
  }
  return true;
}

bool collectConditionalDependencies(
    StorageLoadOp load, mlir::scf::ForOp loop,
    llvm::DenseSet<mlir::Operation *> &dependencies) {
  for (auto *parent = load->getParentOp(); parent != loop;
       parent = parent->getParentOp()) {
    auto conditional = mlir::dyn_cast<mlir::scf::IfOp>(parent);
    if (!conditional || !collectLoadDependencies(conditional.getCondition(),
                                                 loop, dependencies))
      return false;
  }
  return true;
}

// Recreate only the existing conditional scopes and pure address dependencies.
// Computation stays in its original branch; moving the original load preserves
// the actual buffer-owner relation. The mapping lives only during this rewrite.
mlir::scf::IfOp extractPrefetchScope(LoadPipeline &pipeline,
                                     mlir::IRRewriter &rewriter,
                                     llvm::DenseSet<mlir::Operation *> &early) {
  early.clear();
  for (auto load : pipeline.loads)
    if (!collectLoadDependencies(load.getDest(), pipeline.loop, early))
      return {};
  llvm::SmallVector<mlir::Operation *> address;
  for (auto &operation : pipeline.loop.getBody()->without_terminator())
    if (early.contains(&operation))
      address.push_back(&operation);
  for (auto *operation : llvm::reverse(address))
    rewriter.moveOpBefore(operation, pipeline.loop.getBody(),
                          pipeline.loop.getBody()->begin());
  rewriter.setInsertionPoint(pipeline.loop);
  auto always =
      rewriter.create<mlir::arith::ConstantIntOp>(pipeline.loop.getLoc(), 1, 1);
  if (address.empty())
    rewriter.setInsertionPointToStart(pipeline.loop.getBody());
  else
    rewriter.setInsertionPointAfter(address.back());
  auto prefetch = rewriter.create<mlir::scf::IfOp>(
      pipeline.loop.getLoc(), always, /*withElseRegion=*/false);
  mlir::IRMapping mapping;
  for (auto load : pipeline.loads)
    mapping.map(load.getDest(), load.getDest());
  llvm::DenseMap<mlir::Block *, mlir::Block *> blocks;
  blocks[pipeline.loop.getBody()] = &prefetch.getThenRegion().front();
  llvm::SmallVector<mlir::Operation *> copied;
  std::function<mlir::Value(mlir::Value)> cloneValue;
  std::function<mlir::Block *(mlir::Block *)> cloneBlock;
  cloneBlock = [&](mlir::Block *block) -> mlir::Block * {
    auto known = blocks.find(block);
    if (known != blocks.end())
      return known->second;
    auto original = mlir::cast<mlir::scf::IfOp>(block->getParentOp());
    auto *parent = cloneBlock(original->getBlock());
    auto condition = cloneValue(original.getCondition());
    rewriter.setInsertionPoint(parent->getTerminator());
    auto scope = rewriter.create<mlir::scf::IfOp>(
        original.getLoc(), condition, !original.getElseRegion().empty());
    blocks[&original.getThenRegion().front()] = &scope.getThenRegion().front();
    if (!original.getElseRegion().empty())
      blocks[&original.getElseRegion().front()] =
          &scope.getElseRegion().front();
    return blocks.lookup(block);
  };
  cloneValue = [&](mlir::Value value) -> mlir::Value {
    if (auto known = mapping.lookupOrNull(value))
      return known;
    auto *definition = value.getDefiningOp();
    if (!definition || !pipeline.loop->isAncestor(definition))
      return value;
    for (auto operand : definition->getOperands())
      mapping.map(operand, cloneValue(operand));
    auto *block = cloneBlock(definition->getBlock());
    auto scopes = block->getOps<mlir::scf::IfOp>();
    if (scopes.empty())
      rewriter.setInsertionPoint(block->getTerminator());
    else
      rewriter.setInsertionPoint(*scopes.begin());
    rewriter.clone(*definition, mapping);
    copied.push_back(definition);
    return mapping.lookup(value);
  };
  for (auto load : pipeline.loads) {
    auto source = cloneValue(load.getSource());
    auto *block = cloneBlock(load->getBlock());
    rewriter.modifyOpInPlace(load,
                             [&] { load.getSourceMutable().set(source); });
    rewriter.moveOpBefore(load, block->getTerminator());
  }
  for (auto *operation : llvm::reverse(copied))
    if (operation->use_empty() && mlir::isMemoryEffectFree(operation))
      rewriter.eraseOp(operation);
  early.insert(prefetch);
  return prefetch;
}

llvm::SmallVector<LoadPipeline, 4> findLoadPipelines(mlir::ModuleOp module) {
  support::ScopedCompileTimingSpan timing("analysis", "load-pipelining",
                                          "current-loops");
  llvm::SmallVector<LoadPipeline, 4> pipelines;
  module.walk([&](mlir::scf::ForOp loop) {
    support::addCompileCounter("load-pipelining", "queried-loops", 1);
    auto tripCount = getStaticTripCount(loop);
    if (!loop->getParentOfType<TileRegionOp>() ||
        (tripCount && *tripCount == 0) || checkLoopPipeliningDomain(loop, 2))
      return;
    // This construction pipelines one local loop. Peeling a cross-Tile
    // protocol requires a joint transformation of every participant; cloning
    // its messages locally cannot preserve the existing completion identity.
    for (auto &operation : loop.getBody()->without_terminator()) {
      auto effects = mlir::getEffectsRecursively(&operation);
      if (!effects)
        return;
      if (llvm::any_of(*effects, [](const auto &effect) {
            return effect.getResource() == WaferCommunicationResource::get() ||
                   effect.getResource() == WaferSyncResource::get();
          }))
        return;
    }
    // The post-order walk discovers inner candidates first. Do not bind an
    // overlapping outer transform to handles that the inner rewrite replaces.
    if (llvm::any_of(pipelines, [&](const LoadPipeline &selected) {
          return loop->isAncestor(selected.loop);
        }))
      return;
    LoadPipeline pipeline{loop};
    llvm::SmallVector<StorageLoadOp> loads;
    loop.walk([&](StorageLoadOp load) { loads.push_back(load); });
    for (StorageLoadOp load : loads) {
      auto allocation = load.getDest().getDefiningOp<mlir::memref::AllocOp>();
      if (!allocation || allocation->getBlock() != load->getBlock() ||
          allocation.getResult().hasOneUse() ||
          !allocation.getDynamicSizes().empty() ||
          !allocation.getSymbolOperands().empty())
        continue;
      llvm::DenseSet<mlir::Value> visited;
      llvm::DenseSet<mlir::Operation *> dependencies;
      if (!hasIterationLocalUses(allocation, load, loop, visited) ||
          !collectLoadDependencies(load.getSource(), loop, dependencies) ||
          !collectConditionalDependencies(load, loop, dependencies))
        continue;
      pipeline.loads.push_back(load);
      pipeline.allocations.push_back(allocation);
    }
    if (!pipeline.loads.empty())
      pipelines.push_back(std::move(pipeline));
  });
  return pipelines;
}

} // namespace

LoadPipelineQueryResult queryDistanceOneLoadPipelines(mlir::ModuleOp module) {
  LoadPipelineQueryResult result;
  if (!module) {
    result.failure = LoopPipeliningFailure{
        LoopPipeliningFailureKind::BrokenContract,
        {},
        "load pipeline query requires current physical IR"};
    return result;
  }
  for (auto &pipeline : findLoadPipelines(module))
    result.loops.push_back(pipeline.loop);
  if (result.loops.empty())
    result.failure =
        LoopPipeliningFailure{LoopPipeliningFailureKind::Unsupported,
                              {},
                              "no load has a bounded domain, movable address "
                              "and complete local lifetime"};
  return result;
}

PipelinedModuleResult materializeDistanceOneLoadPipelines(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    StructuredMaterializationRelations &relations) {
  support::ScopedCompileTimingSpan timing("transformation", "load-pipelining",
                                          "materialize");
  if (!module || mlir::failed(mlir::verify(*module)))
    return materializationFailure(LoopPipeliningFailureKind::BrokenContract,
                                  "load pipeline requires verified current IR");
  // Bufferization can leave unchanged memrefs as loop-carried arguments. The
  // pinned SCF pipeliner does not accept a yield of a BlockArgument. Apply the
  // standard ForOp folds to the selected loops, then rediscover all handles.
  llvm::SmallVector<mlir::Operation *, 4> loops;
  for (const LoadPipeline &pipeline : findLoadPipelines(*module)) {
    for (mlir::memref::AllocOp allocation : pipeline.allocations)
      if (!llvm::any_of(relations.buffers, [&](const auto &relation) {
            return relation.buffer == allocation.getResult();
          }))
        return materializationFailure(
            LoopPipeliningFailureKind::BrokenContract,
            "load pipeline allocation has no current owner");
    loops.push_back(pipeline.loop);
  }
  mlir::RewritePatternSet patterns(module->getContext());
  mlir::scf::ForOp::getCanonicalizationPatterns(patterns, module->getContext());
  mlir::GreedyRewriteConfig config;
  config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
  config.maxNumRewrites = loops.size() + 1;
  if (mlir::failed(
          mlir::applyOpPatternsAndFold(loops, std::move(patterns), config)))
    return materializationFailure(
        LoopPipeliningFailureKind::Indeterminate,
        "load pipeline loop normalization did not converge");
  rebuildCurrentBufferOwnerRelations(module->getOperation(), relations);
  auto pipelines = findLoadPipelines(*module);
  if (pipelines.empty())
    return materializationFailure(
        LoopPipeliningFailureKind::Unsupported,
        "no current load has a complete rotating lifetime");
  llvm::SmallVector<RotatingAllocationBinding, 4> rotations;
  for (const LoadPipeline &pipeline : pipelines)
    for (mlir::memref::AllocOp allocation : pipeline.allocations) {
      if (!llvm::any_of(relations.buffers, [&](const auto &relation) {
            return relation.buffer == allocation.getResult();
          }))
        return materializationFailure(
            LoopPipeliningFailureKind::BrokenContract,
            "load pipeline allocation has no current owner");
      rotations.push_back({allocation, pipeline.loop, 2});
    }
  mlir::IRRewriter rewriter(module->getContext());
  for (const auto &rotation : rotations)
    rewriter.moveOpBefore(rotation.allocation, rotation.loop);
  auto rotated =
      materializeRotatingAllocations(std::move(module), rotations, relations);
  if (!rotated.succeeded())
    return {{}, std::move(rotated.failure)};
  module = std::move(rotated.materialized->module);
  llvm::SmallVector<TilePipelineChoice, 4> choices;
  for (LoadPipeline &pipeline : pipelines) {
    llvm::DenseSet<mlir::Operation *> early;
    for (StorageLoadOp load : pipeline.loads) {
      early.insert(load);
      for (mlir::Value operand : load->getOperands())
        if (!collectLoadDependencies(operand, pipeline.loop, early))
          return materializationFailure(
              LoopPipeliningFailureKind::CompilerBug,
              "load pipeline rotation changed an address dependency");
    }
    auto tripCount = getStaticTripCount(pipeline.loop);
    if (!tripCount || *tripCount < 2 ||
        llvm::any_of(
            pipeline.loads,
            [&](StorageLoadOp load) {
              return load->getBlock() != pipeline.loop.getBody();
            })) {
      if (!extractPrefetchScope(pipeline, rewriter, early))
        return materializationFailure(LoopPipeliningFailureKind::CompilerBug,
                                      "prefetch scope lost a prepared address");
    }
    TilePipelineChoice choice;
    choice.loop = pipeline.loop;
    bool hasConsumerStage = false;
    for (mlir::Operation &operation :
         pipeline.loop.getBody()->without_terminator()) {
      const uint32_t stage = early.contains(&operation) ? 0 : 1;
      hasConsumerStage |= stage == 1;
      choice.operations.push_back({&operation, stage});
    }
    if (!hasConsumerStage)
      return materializationFailure(LoopPipeliningFailureKind::Unsupported,
                                    "load pipeline has no consumer stage");
    choices.push_back(std::move(choice));
  }
  auto prepared = prepareLoopPipelines(*module, choices);
  if (!prepared.succeeded())
    return {{}, std::move(prepared.failure)};
  auto result = pipelineLoops(std::move(module), std::move(*prepared.prepared));
  if (result.succeeded())
    rebuildCurrentBufferOwnerRelations(
        result.materialized->module->getOperation(), relations);
  return result;
}

} // namespace wafer::compiler::detail
