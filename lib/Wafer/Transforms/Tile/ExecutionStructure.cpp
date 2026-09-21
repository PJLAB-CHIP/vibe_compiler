//===- ExecutionStructure.cpp - Current Tile execution rewrite --------===//

#include "Wafer/Transforms/Tile/ExecutionStructure.h"

#include "StorageInitialization.h"
#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace wafer::compiler::detail {
namespace {

constexpr llvm::StringLiteral kOperationIndexAttr =
    "wafer.internal.execution_operation_index";
constexpr llvm::StringLiteral kPhaseAttr = "wafer.internal.execution_phase";
constexpr llvm::StringLiteral kPhaseIterationAttr =
    "wafer.internal.execution_phase_iteration";
constexpr llvm::StringLiteral kKernelIterationAttr =
    "wafer.internal.execution_kernel_iteration";

PreparedExecutionStructureResult
prepareFailure(ExecutionStructureFailureKind kind, llvm::StringRef detail,
               std::optional<uint32_t> pipeline = std::nullopt) {
  return {{}, ExecutionStructureFailure{kind, pipeline, detail.str()}};
}

MaterializedExecutionStructureResult
materializationFailure(ExecutionStructureFailureKind kind,
                       llvm::StringRef detail,
                       std::optional<uint32_t> pipeline = std::nullopt) {
  return {{}, ExecutionStructureFailure{kind, pipeline, detail.str()}};
}

RotatingAllocationMaterializationResult
rotationFailure(ExecutionStructureFailureKind kind, llvm::StringRef detail) {
  return {{}, ExecutionStructureFailure{kind, {}, detail.str()}};
}

std::optional<uint64_t> getStaticTripCount(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower)
    return std::nullopt;
  const __int128 difference =
      static_cast<__int128>(*upper) - static_cast<__int128>(*lower);
  if (difference <= 0 || difference > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return 1 +
         (static_cast<uint64_t>(difference) - 1) / static_cast<uint64_t>(*step);
}

uint32_t getStageCount(llvm::ArrayRef<TilePipelineOperation> operations) {
  uint32_t count = 0;
  for (const TilePipelineOperation &operation : operations) {
    if (operation.stage == std::numeric_limits<uint32_t>::max())
      return 0;
    count = std::max(count, operation.stage + 1);
  }
  return count;
}

mlir::Operation *getTopLevelOwner(mlir::Operation *operation,
                                  mlir::Block *body) {
  while (operation && operation->getBlock() != body)
    operation = operation->getParentOp();
  return operation;
}

llvm::SmallVector<mlir::Value, 8>
getNestedOperands(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::DenseSet<mlir::Value> seen;
  operation->walk([&](mlir::Operation *nested) {
    for (mlir::Value operand : nested->getOperands())
      if (seen.insert(operand).second)
        operands.push_back(operand);
  });
  return operands;
}

struct ActualDependence {
  mlir::Operation *source = nullptr;
  uint32_t distance = 0;
};

std::optional<ActualDependence> getActualDependence(mlir::Value operand,
                                                    mlir::scf::ForOp loop) {
  mlir::Operation *source = operand.getDefiningOp();
  if (source && loop->isAncestor(source))
    return ActualDependence{getTopLevelOwner(source, loop.getBody()), 0};
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(operand);
  if (!argument || argument.getOwner() != loop.getBody() ||
      argument.getArgNumber() == 0)
    return std::nullopt;
  const unsigned iterationArgument = argument.getArgNumber() - 1;
  auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (iterationArgument >= yield.getNumOperands())
    return std::nullopt;
  mlir::Operation *yieldSource =
      yield.getOperand(iterationArgument).getDefiningOp();
  if (!yieldSource || !loop->isAncestor(yieldSource))
    return std::nullopt;
  return ActualDependence{getTopLevelOwner(yieldSource, loop.getBody()), 1};
}

void collectStorageRoots(mlir::Value value, llvm::DenseSet<mlir::Value> &roots,
                         llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return;
  mlir::Operation *definition = value.getDefiningOp();
  if (auto view =
          mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(definition)) {
    collectStorageRoots(view.getViewSource(), roots, visited);
    return;
  }
  if (auto select =
          mlir::dyn_cast_or_null<mlir::SelectLikeOpInterface>(definition)) {
    collectStorageRoots(select.getTrueValue(), roots, visited);
    collectStorageRoots(select.getFalseValue(), roots, visited);
    return;
  }
  roots.insert(value);
}

llvm::DenseSet<mlir::Value> getStorageRoots(mlir::Value value) {
  llvm::DenseSet<mlir::Value> roots;
  llvm::DenseSet<mlir::Value> visited;
  collectStorageRoots(value, roots, visited);
  return roots;
}

bool isExternalToLoop(mlir::Value root, mlir::scf::ForOp loop) {
  mlir::Operation *definition = root.getDefiningOp();
  return !definition || !loop->isAncestor(definition);
}

bool isDistanceOneRotatingView(mlir::Value value, mlir::scf::ForOp loop) {
  while (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
             value.getDefiningOp()))
    value = view.getViewSource();
  auto select = value.getDefiningOp<mlir::arith::SelectOp>();
  if (!select)
    return false;
  auto first = select.getTrueValue().getDefiningOp<mlir::memref::AllocOp>();
  auto second = select.getFalseValue().getDefiningOp<mlir::memref::AllocOp>();
  if (!first || !second || first == second ||
      first->getBlock() != loop->getBlock() ||
      second->getBlock() != loop->getBlock() || !first->isBeforeInBlock(loop) ||
      !second->isBeforeInBlock(loop))
    return false;
  auto condition = select.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
  if (!condition ||
      condition.getPredicate() != mlir::arith::CmpIPredicate::eq ||
      mlir::getConstantIntValue(condition.getRhs()) != 1)
    return false;
  auto remainder = condition.getLhs().getDefiningOp<mlir::arith::RemUIOp>();
  if (!remainder || mlir::getConstantIntValue(remainder.getRhs()) != 2)
    return false;
  auto iteration = remainder.getLhs().getDefiningOp<mlir::arith::DivUIOp>();
  if (!iteration || iteration.getRhs() != loop.getStep())
    return false;
  auto delta = iteration.getLhs().getDefiningOp<mlir::arith::SubIOp>();
  return delta && delta.getLhs() == loop.getInductionVar() &&
         delta.getRhs() == loop.getLowerBound();
}

std::optional<std::string>
checkExternalEffects(mlir::scf::ForOp loop,
                     const std::map<mlir::Operation *, uint32_t> &stages) {
  struct Summary {
    bool write = false;
    bool distanceOneRotation = true;
    std::set<uint32_t> stages;
  };
  using EffectList = llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4>;
  std::function<std::optional<EffectList>(mlir::Operation *)> getEffects =
      [&](mlir::Operation *operation) -> std::optional<EffectList> {
    if (auto effects = mlir::getEffectsRecursively(operation))
      return EffectList(effects->begin(), effects->end());
    if (mlir::isa<mlir::async::AwaitOp, mlir::async::YieldOp>(operation))
      return EffectList{};
    auto execute = mlir::dyn_cast<mlir::async::ExecuteOp>(operation);
    if (!execute)
      return std::nullopt;
    EffectList combined;
    for (mlir::Operation &nested : execute.getBody()->getOperations()) {
      std::optional<EffectList> effects = getEffects(&nested);
      if (!effects)
        return std::nullopt;
      combined.append(*effects);
    }
    return combined;
  };

  llvm::DenseMap<mlir::Value, Summary> summaries;
  for (const auto &[operation, stage] : stages) {
    std::optional<EffectList> effects = getEffects(operation);
    if (!effects)
      return "pipelined operation has no typed memory-effect contract";
    for (const auto &effect : *effects) {
      if (!effect.getValue())
        continue;
      for (mlir::Value root : getStorageRoots(effect.getValue())) {
        if (!isExternalToLoop(root, loop))
          continue;
        Summary &summary = summaries[root];
        summary.distanceOneRotation &=
            isDistanceOneRotatingView(effect.getValue(), loop);
        summary.write |=
            !mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
        summary.stages.insert(stage);
      }
    }
  }
  for (const auto &[root, summary] : summaries) {
    (void)root;
    if (summary.write && summary.stages.size() > 1 &&
        !(summary.distanceOneRotation &&
          *summary.stages.rbegin() - *summary.stages.begin() < 2))
      return "cross-stage external write has no current rotating root";
  }
  return std::nullopt;
}

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
        llvm::any_of(loop.getBody()->without_terminator(), [](auto &operation) {
          return operation.getNumRegions() != 0;
        }))
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

bool hasTemporaryAttributes(mlir::Operation *root) {
  bool found = false;
  root->walk([&](mlir::Operation *operation) {
    found |= operation->hasAttr(kOperationIndexAttr) ||
             operation->hasAttr(kPhaseAttr) ||
             operation->hasAttr(kPhaseIterationAttr) ||
             operation->hasAttr(kKernelIterationAttr);
  });
  return found;
}

void clearTemporaryAttributes(mlir::Operation *root,
                              mlir::RewriterBase &rewriter) {
  root->walk([&](mlir::Operation *operation) {
    if (!operation->hasAttr(kOperationIndexAttr) &&
        !operation->hasAttr(kPhaseAttr) &&
        !operation->hasAttr(kPhaseIterationAttr) &&
        !operation->hasAttr(kKernelIterationAttr))
      return;
    rewriter.modifyOpInPlace(operation, [&] {
      operation->removeAttr(kOperationIndexAttr);
      operation->removeAttr(kPhaseAttr);
      operation->removeAttr(kPhaseIterationAttr);
      operation->removeAttr(kKernelIterationAttr);
    });
  });
}

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
    mlir::Operation *anchor = getTopLevelOwner(user, loop.getBody());
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

static void preservePrivateScalarBroadcasts(
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
      if (!mlir::isa<ComputeElementwiseOp, ComputeElementwiseIntoOp>(consumer) ||
          use.getOperandNumber() != 1 || pipelineOperations.contains(consumer))
        return;
      auto *owner = getTopLevelOwner(consumer, load->getBlock());
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
            auto scalar = mlir::AffineMap::get(rank, 0, {}, module.getContext());
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

mlir::LogicalResult verifyMaterializedExecutionStructure(
    const MaterializedExecutionStructure &materialized,
    std::string *failureReason);

} // namespace

PreparedExecutionStructureResult
prepareTileExecutionStructure(mlir::ModuleOp module,
                              llvm::ArrayRef<TilePipelineChoice> choices,
                              const ExecutionStructureLimits &limits) {
  if (!module || limits.maxFiniteUnrolledOperations == 0)
    return prepareFailure(ExecutionStructureFailureKind::BrokenContract,
                          "execution structure requires valid current IR and "
                          "a positive work limit");
  if (hasTemporaryAttributes(module))
    return prepareFailure(ExecutionStructureFailureKind::BrokenContract,
                          "execution structure input contains private attrs");

  PreparedExecutionStructure prepared;
  prepared.module = module;
  prepared.limits = limits;
  llvm::DenseSet<mlir::Operation *> loops;
  for (auto [pipelineIndex, choice] : llvm::enumerate(choices)) {
    const uint32_t pipeline = static_cast<uint32_t>(pipelineIndex);
    mlir::scf::ForOp loop = choice.loop;
    if (!loop || !module->isAncestor(loop) ||
        !loops.insert(loop.getOperation()).second)
      return prepareFailure(ExecutionStructureFailureKind::BrokenContract,
                            "pipeline loop is stale or duplicated", pipeline);
    if (!loop.getRegion().hasOneBlock())
      return prepareFailure(ExecutionStructureFailureKind::Unsupported,
                            "pipeline loop is not single-block", pipeline);
    std::optional<uint64_t> tripCount = getStaticTripCount(loop);
    const uint32_t stageCount = getStageCount(choice.operations);
    if (!tripCount || stageCount < 2 || *tripCount < stageCount ||
        choice.operations.empty())
      return prepareFailure(ExecutionStructureFailureKind::BrokenContract,
                            "pipeline stages or static trip count are invalid",
                            pipeline);

    std::vector<bool> usedStages(stageCount, false);
    llvm::DenseSet<mlir::Operation *> boundOperations;
    std::map<mlir::Operation *, uint32_t> stages;
    for (const TilePipelineOperation &binding : choice.operations) {
      if (!binding.operation || binding.stage >= stageCount ||
          binding.operation->getBlock() != loop.getBody() ||
          binding.operation->hasTrait<mlir::OpTrait::IsTerminator>() ||
          !boundOperations.insert(binding.operation).second)
        return prepareFailure(ExecutionStructureFailureKind::BrokenContract,
                              "pipeline operation is stale or duplicated",
                              pipeline);
      usedStages[binding.stage] = true;
      stages.emplace(binding.operation, binding.stage);
    }
    if (llvm::is_contained(usedStages, false))
      return prepareFailure(ExecutionStructureFailureKind::BrokenContract,
                            "pipeline stage assignment has an empty stage",
                            pipeline);
    for (mlir::Operation &operation : loop.getBody()->without_terminator())
      if (!boundOperations.count(&operation))
        return prepareFailure(ExecutionStructureFailureKind::BrokenContract,
                              "pipeline does not cover every loop operation",
                              pipeline);

    for (const TilePipelineOperation &destination : choice.operations)
      for (mlir::Value operand : getNestedOperands(destination.operation)) {
        std::optional<ActualDependence> dependence =
            getActualDependence(operand, loop);
        if (!dependence || dependence->source == destination.operation)
          continue;
        auto source = stages.find(dependence->source);
        if (source == stages.end())
          return prepareFailure(ExecutionStructureFailureKind::BrokenContract,
                                "pipeline SSA dependence has no bound source",
                                pipeline);
        if (dependence->distance == 0 && source->second > destination.stage)
          return prepareFailure(
              ExecutionStructureFailureKind::BrokenContract,
              "pipeline reverses a same-iteration SSA dependence", pipeline);
      }

    if (std::optional<std::string> effectFailure =
            checkExternalEffects(loop, stages))
      return prepareFailure(ExecutionStructureFailureKind::Unsupported,
                            *effectFailure, pipeline);

    for (const TilePipelineOperation &binding : choice.operations)
      for (mlir::Value result : binding.operation->getResults()) {
        if (!mlir::isa<mlir::async::TokenType>(result.getType()))
          continue;
        for (mlir::Operation *user : result.getUsers()) {
          mlir::Operation *top = getTopLevelOwner(user, loop.getBody());
          auto userStage = stages.find(top);
          if (userStage != stages.end() && userStage->second != binding.stage &&
              choice.lowering != TilePipelineLowering::FiniteUnrolled)
            return prepareFailure(
                ExecutionStructureFailureKind::Unsupported,
                "cross-stage async token requires finite unrolling", pipeline);
        }
      }

    if (choice.lowering == TilePipelineLowering::FiniteUnrolled &&
        (*tripCount >
         limits.maxFiniteUnrolledOperations / choice.operations.size()))
      return prepareFailure(ExecutionStructureFailureKind::Indeterminate,
                            "finite pipeline exceeds its code-size bound",
                            pipeline);

    prepared.pipelines.push_back(PreparedTilePipeline{
        loop, choice.operations, choice.lowering, *tripCount, stageCount});
  }
  return {std::move(prepared), {}};
}

MaterializedExecutionStructureResult
materializeExecutionStructure(mlir::OwningOpRef<mlir::ModuleOp> module,
                              PreparedExecutionStructure prepared) {
  if (!module || prepared.module != *module ||
      hasTemporaryAttributes(module->getOperation()))
    return materializationFailure(
        ExecutionStructureFailureKind::BrokenContract,
        "prepared execution structure is stale for the owned module");

  mlir::IRRewriter rewriter(module->getContext());
  MaterializedExecutionStructure result;
  uint64_t nextOperationIndex = 0;
  struct OperationBinding {
    size_t pipeline = 0;
    uint32_t stage = 0;
  };
  std::map<uint64_t, OperationBinding> bindings;

  for (auto [pipelineIndex, pipeline] : llvm::enumerate(prepared.pipelines)) {
    MaterializedExecutionPipeline materialized;
    materialized.stageCount = pipeline.stageCount;
    materialized.originalOperationCount = pipeline.operations.size();
    const uint64_t maximumStage = pipeline.stageCount - 1;
    materialized.kernelDynamicTripCount = pipeline.tripCount - maximumStage;
    result.pipelines.push_back(std::move(materialized));

    for (const TilePipelineOperation &operation : pipeline.operations) {
      if (nextOperationIndex >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return materializationFailure(
            ExecutionStructureFailureKind::Indeterminate,
            "pipeline operation mapping exceeds its work domain",
            static_cast<uint32_t>(pipelineIndex));
      const uint64_t index = nextOperationIndex++;
      bindings.emplace(index, OperationBinding{pipelineIndex, operation.stage});
      rewriter.modifyOpInPlace(operation.operation, [&] {
        operation.operation->setAttr(
            kOperationIndexAttr,
            rewriter.getI64IntegerAttr(static_cast<int64_t>(index)));
      });
    }

    mlir::scf::PipeliningOption options;
    options.getScheduleFn =
        [operations = pipeline.operations](
            mlir::scf::ForOp,
            std::vector<std::pair<mlir::Operation *, unsigned>> &schedule) {
          for (const TilePipelineOperation &operation : operations)
            schedule.emplace_back(operation.operation, operation.stage);
        };
    options.annotateFn = [&](mlir::Operation *operation,
                             mlir::scf::PipeliningOption::PipelinerPart part,
                             unsigned iteration) {
      MaterializedExecutionPhase phase = MaterializedExecutionPhase::Kernel;
      if (part == mlir::scf::PipeliningOption::PipelinerPart::Prologue)
        phase = MaterializedExecutionPhase::Prologue;
      else if (part == mlir::scf::PipeliningOption::PipelinerPart::Epilogue)
        phase = MaterializedExecutionPhase::Epilogue;
      operation->setAttr(
          kPhaseAttr, rewriter.getI32IntegerAttr(static_cast<uint32_t>(phase)));
      operation->setAttr(kPhaseIterationAttr,
                         rewriter.getI64IntegerAttr(iteration));
      if (phase == MaterializedExecutionPhase::Kernel)
        operation->setAttr(kKernelIterationAttr, rewriter.getI64IntegerAttr(0));
    };
    bool modified = false;
    rewriter.setInsertionPoint(pipeline.loop);
    auto kernel =
        mlir::scf::pipelineForLoop(rewriter, pipeline.loop, options, &modified);
    if (mlir::failed(kernel))
      return materializationFailure(
          modified ? ExecutionStructureFailureKind::CompilerBug
                   : ExecutionStructureFailureKind::Unsupported,
          "pinned SCF pipelining rejected a prepared current-IR schedule",
          static_cast<uint32_t>(pipelineIndex));

    // Publish the exact static kernel domain. The pinned helper leaves
    // `ub - stage * step` as unfurled arith operations; current completion and
    // lifetime consumers need the proven nonempty loop bounds in the IR.
    const uint64_t factor = pipeline.tripCount - maximumStage;
    std::optional<int64_t> lower =
        mlir::getConstantIntValue(kernel->getLowerBound());
    std::optional<int64_t> step = mlir::getConstantIntValue(kernel->getStep());
    const __int128 upper =
        lower && step
            ? static_cast<__int128>(*lower) +
                  static_cast<__int128>(factor) * *step
            : static_cast<__int128>(std::numeric_limits<int64_t>::max()) + 1;
    if (factor == 0 || !lower || !step || *step <= 0 ||
        upper > std::numeric_limits<int64_t>::max() ||
        upper < std::numeric_limits<int64_t>::min())
      return materializationFailure(
          ExecutionStructureFailureKind::CompilerBug,
          "static pipeline has non-constant or overflowing bounds",
          static_cast<uint32_t>(pipelineIndex));
    rewriter.setInsertionPoint(*kernel);
    auto constant = rewriter.create<mlir::arith::ConstantOp>(
        kernel->getLoc(),
        rewriter.getIntegerAttr(kernel->getUpperBound().getType(),
                                static_cast<int64_t>(upper)));
    rewriter.modifyOpInPlace(*kernel, [&] { kernel->setUpperBound(constant); });
    if (pipeline.lowering == TilePipelineLowering::FiniteUnrolled) {
      if (mlir::failed(mlir::loopUnrollByFactor(
              *kernel, factor,
              [&](unsigned iteration, mlir::Operation *operation,
                  mlir::OpBuilder builder) {
                operation->setAttr(kKernelIterationAttr,
                                   builder.getI64IntegerAttr(iteration));
              })))
        return materializationFailure(
            ExecutionStructureFailureKind::CompilerBug,
            "finite pipeline failed mechanical unrolling",
            static_cast<uint32_t>(pipelineIndex));
    }
  }

  module->walk([&](mlir::Operation *operation) {
    auto index =
        operation->getAttrOfType<mlir::IntegerAttr>(kOperationIndexAttr);
    auto phase = operation->getAttrOfType<mlir::IntegerAttr>(kPhaseAttr);
    auto phaseIteration =
        operation->getAttrOfType<mlir::IntegerAttr>(kPhaseIterationAttr);
    if (!index || !phase || !phaseIteration)
      return;
    auto binding = bindings.find(index.getValue().getZExtValue());
    if (binding == bindings.end() ||
        binding->second.pipeline >= result.pipelines.size())
      return;
    MaterializedExecutionPhase materializedPhase =
        static_cast<MaterializedExecutionPhase>(
            phase.getValue().getZExtValue());
    uint64_t iteration = phaseIteration.getValue().getZExtValue();
    if (materializedPhase == MaterializedExecutionPhase::Kernel)
      if (auto kernelIteration =
              operation->getAttrOfType<mlir::IntegerAttr>(kKernelIterationAttr))
        iteration = kernelIteration.getValue().getZExtValue();
    result.pipelines[binding->second.pipeline].operations.push_back(
        {operation, binding->second.stage, materializedPhase, iteration});
  });
  clearTemporaryAttributes(module->getOperation(), rewriter);
  if (hasTemporaryAttributes(module->getOperation()))
    return materializationFailure(
        ExecutionStructureFailureKind::CompilerBug,
        "materialized execution structure retained private attributes");
  llvm::DenseSet<mlir::Operation *> pipelineOperations;
  for (const MaterializedExecutionPipeline &pipeline : result.pipelines)
    for (const MaterializedExecutionOperation &operation : pipeline.operations)
      pipelineOperations.insert(operation.operation);
  preservePrivateScalarBroadcasts(*module, rewriter, pipelineOperations);
  eliminateGemmWritebacks(*module, rewriter, pipelineOperations);
  eliminatePrivatePointwisePublications(*module, rewriter, pipelineOperations);
  eliminateElementwiseWritebacks(*module, rewriter, pipelineOperations);
  reuseElementwiseInputs(*module, rewriter, pipelineOperations);
  eliminateUnusedStorageInitialization(module->getOperation(), rewriter,
                                       pipelineOperations);
  if (mlir::failed(materializeLoopCarriedDestinations(*module, rewriter)))
    return materializationFailure(
        ExecutionStructureFailureKind::CompilerBug,
        "execution structure produced an invalid explicit loop-carried "
        "destination");
  result.module = std::move(module);
  std::string failureReason;
  if (mlir::failed(
          verifyMaterializedExecutionStructure(result, &failureReason)))
    return materializationFailure(ExecutionStructureFailureKind::CompilerBug,
                                  failureReason);
  return {std::move(result), {}};
}

namespace {

mlir::LogicalResult verifyMaterializedExecutionStructure(
    const MaterializedExecutionStructure &materialized,
    std::string *failureReason) {
  auto fail = [&](llvm::StringRef detail) {
    if (failureReason)
      *failureReason = detail.str();
    return mlir::failure();
  };
  if (failureReason)
    failureReason->clear();
  mlir::ModuleOp module = materialized.module.get();
  if (!module || mlir::failed(mlir::verify(module)) ||
      hasTemporaryAttributes(module))
    return fail("execution-structure verifier requires clean valid IR");
  llvm::DenseSet<mlir::Operation *> operations;
  for (const MaterializedExecutionPipeline &pipeline : materialized.pipelines) {
    if (pipeline.stageCount < 2 || pipeline.originalOperationCount == 0 ||
        pipeline.operations.empty())
      return fail("materialized pipeline is empty or malformed");
    for (const MaterializedExecutionOperation &operation : pipeline.operations)
      if (!operation.operation || !module->isAncestor(operation.operation) ||
          operation.stage >= pipeline.stageCount ||
          !operations.insert(operation.operation).second)
        return fail("materialized pipeline relation is stale");
  }
  return mlir::success();
}

} // namespace

RotatingAllocationMaterializationResult materializeRotatingAllocations(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    llvm::ArrayRef<RotatingAllocationBinding> bindings,
    StructuredMaterializationRelations &relations) {
  if (!module)
    return rotationFailure(ExecutionStructureFailureKind::BrokenContract,
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
          ExecutionStructureFailureKind::BrokenContract,
          "rotating allocation requires a static pre-loop TileRegion root");

    const bool hasRelation =
        llvm::any_of(relations.buffers, [&](const auto &entry) {
          return entry.buffer == allocation.getResult();
        });
    if (!hasRelation)
      return rotationFailure(
          ExecutionStructureFailureKind::BrokenContract,
          "rotating allocation has no current typed owner relation");

    PreparedRotation rotation{binding, *tripCount};
    for (mlir::Operation *user : allocation.getResult().getUsers()) {
      if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(user)) {
        if (rotation.deallocation)
          return rotationFailure(
              ExecutionStructureFailureKind::BrokenContract,
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
            ExecutionStructureFailureKind::Unsupported,
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
    return rotationFailure(ExecutionStructureFailureKind::CompilerBug,
                           "rotating allocation produced verifier-invalid IR");
  return {std::move(result), {}};
}

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

MaterializedExecutionStructureResult materializeDistanceOneLoadPipelines(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    StructuredMaterializationRelations &relations) {
  if (!module || mlir::failed(mlir::verify(*module)))
    return materializationFailure(ExecutionStructureFailureKind::BrokenContract,
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
            ExecutionStructureFailureKind::BrokenContract,
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
        ExecutionStructureFailureKind::Indeterminate,
        "load pipeline loop normalization did not converge");
  rebuildCurrentBufferOwnerRelations(module->getOperation(), relations);
  auto pipelines = findLoadPipelines(*module);
  if (pipelines.empty())
    return materializationFailure(
        ExecutionStructureFailureKind::Unsupported,
        "no current load has a complete rotating lifetime");
  llvm::SmallVector<RotatingAllocationBinding, 4> rotations;
  for (const LoadPipeline &pipeline : pipelines)
    for (mlir::memref::AllocOp allocation : pipeline.allocations) {
      if (!llvm::any_of(relations.buffers, [&](const auto &relation) {
            return relation.buffer == allocation.getResult();
          }))
        return materializationFailure(
            ExecutionStructureFailureKind::BrokenContract,
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
              ExecutionStructureFailureKind::CompilerBug,
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
      return materializationFailure(ExecutionStructureFailureKind::Unsupported,
                                    "load pipeline has no consumer stage");
    choices.push_back(std::move(choice));
  }
  auto prepared = prepareTileExecutionStructure(*module, choices);
  if (!prepared.succeeded())
    return {{}, std::move(prepared.failure)};
  auto result = materializeExecutionStructure(std::move(module),
                                              std::move(*prepared.prepared));
  if (result.succeeded())
    rebuildCurrentBufferOwnerRelations(
        result.materialized->module->getOperation(), relations);
  return result;
}

} // namespace wafer::compiler::detail
