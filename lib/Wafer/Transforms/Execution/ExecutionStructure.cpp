//===- ExecutionStructure.cpp - Current Tile execution rewrite --------===//

#include "Wafer/Transforms/ExecutionStructure.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

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

std::optional<std::string>
checkExternalEffects(mlir::scf::ForOp loop,
                     const std::map<mlir::Operation *, uint32_t> &stages) {
  struct Summary {
    bool write = false;
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
        summary.write |=
            !mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
        summary.stages.insert(stage);
      }
    }
  }
  for (const auto &[root, summary] : summaries) {
    (void)root;
    if (summary.write && summary.stages.size() > 1)
      return "cross-stage external write has no current rotating root";
  }
  return std::nullopt;
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

    if (pipeline.lowering == TilePipelineLowering::FiniteUnrolled) {
      const uint64_t factor = pipeline.tripCount - maximumStage;
      std::optional<int64_t> lower =
          mlir::getConstantIntValue(kernel->getLowerBound());
      std::optional<int64_t> step =
          mlir::getConstantIntValue(kernel->getStep());
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
            "finite pipeline has non-constant or overflowing bounds",
            static_cast<uint32_t>(pipelineIndex));
      rewriter.setInsertionPoint(*kernel);
      auto constant = rewriter.create<mlir::arith::ConstantOp>(
          kernel->getLoc(),
          rewriter.getIntegerAttr(kernel->getUpperBound().getType(),
                                  static_cast<int64_t>(upper)));
      rewriter.modifyOpInPlace(*kernel,
                               [&] { kernel->setUpperBound(constant); });
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

    bool hasRelation = false;
    auto findRelation = [&](const auto &entries) {
      hasRelation |= llvm::any_of(entries, [&](const auto &entry) {
        return entry.buffer == allocation.getResult();
      });
    };
    findRelation(relations.operationResultBuffers);
    findRelation(relations.operandBuffers);
    findRelation(relations.scratchBuffers);
    findRelation(relations.outputBuffers);
    findRelation(relations.ddrBuffers);
    findRelation(relations.partialReductionContributions);
    findRelation(relations.partialReductionMergeInputs);
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

    auto expandRelations = [&](auto &entries) {
      const size_t originalSize = entries.size();
      for (size_t index = 0; index < originalSize; ++index) {
        if (entries[index].buffer != allocation.getResult())
          continue;
        auto original = entries[index];
        for (mlir::Value slot :
             llvm::ArrayRef<mlir::Value>(slots).drop_front()) {
          auto copy = original;
          copy.buffer = slot;
          entries.push_back(std::move(copy));
        }
      }
    };
    expandRelations(relations.operationResultBuffers);
    expandRelations(relations.operandBuffers);
    expandRelations(relations.scratchBuffers);
    expandRelations(relations.outputBuffers);
    expandRelations(relations.ddrBuffers);
    expandRelations(relations.partialReductionContributions);
    expandRelations(relations.partialReductionMergeInputs);
    result.slots.append(slots.begin(), slots.end());
  }
  if (mlir::failed(mlir::verify(*result.module)))
    return rotationFailure(ExecutionStructureFailureKind::CompilerBug,
                           "rotating allocation produced verifier-invalid IR");
  return {std::move(result), {}};
}

} // namespace wafer::compiler::detail
