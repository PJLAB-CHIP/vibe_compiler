//===- LoadPipelining.cpp - Tile LoadPipelining -------------------===//

#include "Wafer/Transforms/Tile/LoadPipelining.h"
#include "LoopPipeliningInternal.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Tile/RotatingBuffers.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

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
  if (definition->getBlock() != loop.getBody() ||
      !mlir::isMemoryEffectFree(definition))
    return false;
  if (!early.insert(definition).second)
    return true;
  return llvm::all_of(definition->getOperands(), [&](mlir::Value operand) {
    return collectLoadDependencies(operand, loop, early);
  });
}

bool hasOnlyPostLoadReads(mlir::Value value, StorageLoadOp load,
                          mlir::scf::ForOp loop,
                          llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *user = use.getOwner();
    if (user == load && use.getOperandNumber() == 1)
      continue;
    if (user->getBlock() != loop.getBody())
      return false;
    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(user)) {
      if (view.getViewSource() != value || user->getNumResults() != 1 ||
          !hasOnlyPostLoadReads(user->getResult(0), load, loop, visited))
        return false;
      continue;
    }
    if (!load->isBeforeInBlock(user))
      return false;
    auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user);
    if (!interface)
      return false;
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> effects;
    interface.getEffectsOnValue(value, effects);
    if (effects.empty() || llvm::any_of(effects, [](const auto &effect) {
          return !mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
        }))
      return false;
  }
  return true;
}

llvm::SmallVector<LoadPipeline, 4> findLoadPipelines(mlir::ModuleOp module) {
  llvm::SmallVector<LoadPipeline, 4> pipelines;
  module.walk([&](mlir::scf::ForOp loop) {
    auto tripCount = getStaticTripCount(loop);
    if (!loop->getParentOfType<TileRegionOp>() || !tripCount ||
        *tripCount < 2 ||
        llvm::any_of(
            loop.getBody()->without_terminator(),
            [](auto &operation) { return operation.getNumRegions() != 0; }))
      return;
    // This construction pipelines one local loop. Peeling a cross-Tile
    // protocol requires a joint transformation of every participant; cloning
    // its messages locally cannot preserve the existing completion identity.
    for (auto &operation : loop.getBody()->without_terminator()) {
      auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
      if (!effects)
        return;
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      if (llvm::any_of(instances, [](const auto &effect) {
            return effect.getResource() == WaferCommunicationResource::get() ||
                   effect.getResource() == WaferSyncResource::get();
          }))
        return;
    }
    LoadPipeline pipeline{loop};
    for (StorageLoadOp load : loop.getBody()->getOps<StorageLoadOp>()) {
      auto allocation = load.getDest().getDefiningOp<mlir::memref::AllocOp>();
      if (!allocation || allocation->getBlock() != loop.getBody() ||
          allocation.getResult().hasOneUse() ||
          !allocation.getDynamicSizes().empty() ||
          !allocation.getSymbolOperands().empty())
        continue;
      llvm::DenseSet<mlir::Value> visited;
      llvm::DenseSet<mlir::Operation *> dependencies;
      if (!hasOnlyPostLoadReads(allocation, load, loop, visited) ||
          !collectLoadDependencies(load.getSource(), loop, dependencies))
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

bool hasDistanceOneLoadPipeline(mlir::ModuleOp module) {
  return module && !findLoadPipelines(module).empty();
}

llvm::SmallVector<mlir::scf::ForOp, 4>
getDistanceOneLoadPipelineLoops(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::scf::ForOp, 4> loops;
  if (module)
    for (auto &pipeline : findLoadPipelines(module))
      loops.push_back(pipeline.loop);
  return loops;
}

PipelinedModuleResult materializeDistanceOneLoadPipelines(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    StructuredMaterializationRelations &relations) {
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
