//===- LoopPipelining.cpp - Tile LoopPipelining -------------------===//

#include "Wafer/Transforms/Tile/LoopPipelining.h"
#include "LoopPipeliningInternal.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
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
PipelinedModuleResult materializationFailure(LoopPipeliningFailureKind kind,
                                             llvm::StringRef detail,
                                             std::optional<uint32_t> pipeline) {
  return {{}, LoopPipeliningFailure{kind, pipeline, detail.str()}};
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

namespace {
constexpr llvm::StringLiteral kOperationIndexAttr =
    "wafer.internal.execution_operation_index";
constexpr llvm::StringLiteral kPhaseAttr = "wafer.internal.execution_phase";
constexpr llvm::StringLiteral kPhaseIterationAttr =
    "wafer.internal.execution_phase_iteration";
constexpr llvm::StringLiteral kKernelIterationAttr =
    "wafer.internal.execution_kernel_iteration";

PreparedLoopPipelinesResult
prepareFailure(LoopPipeliningFailureKind kind, llvm::StringRef detail,
               std::optional<uint32_t> pipeline = std::nullopt) {
  return {{}, LoopPipeliningFailure{kind, pipeline, detail.str()}};
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

} // namespace

PreparedLoopPipelinesResult
prepareLoopPipelines(mlir::ModuleOp module,
                     llvm::ArrayRef<TilePipelineChoice> choices,
                     const LoopPipeliningLimits &limits) {
  if (!module || limits.maxFiniteUnrolledOperations == 0)
    return prepareFailure(LoopPipeliningFailureKind::BrokenContract,
                          "loop pipelining requires valid current IR and "
                          "a positive work limit");
  if (hasTemporaryAttributes(module))
    return prepareFailure(LoopPipeliningFailureKind::BrokenContract,
                          "loop pipelining input contains private attrs");

  PreparedLoopPipelines prepared;
  prepared.module = module;
  prepared.limits = limits;
  llvm::DenseSet<mlir::Operation *> loops;
  for (auto [pipelineIndex, choice] : llvm::enumerate(choices)) {
    const uint32_t pipeline = static_cast<uint32_t>(pipelineIndex);
    mlir::scf::ForOp loop = choice.loop;
    if (!loop || !module->isAncestor(loop) ||
        !loops.insert(loop.getOperation()).second)
      return prepareFailure(LoopPipeliningFailureKind::BrokenContract,
                            "pipeline loop is stale or duplicated", pipeline);
    if (!loop.getRegion().hasOneBlock())
      return prepareFailure(LoopPipeliningFailureKind::Unsupported,
                            "pipeline loop is not single-block", pipeline);
    std::optional<uint64_t> tripCount = getStaticTripCount(loop);
    const uint32_t stageCount = getStageCount(choice.operations);
    if (!tripCount || stageCount < 2 || *tripCount < stageCount ||
        choice.operations.empty())
      return prepareFailure(LoopPipeliningFailureKind::BrokenContract,
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
        return prepareFailure(LoopPipeliningFailureKind::BrokenContract,
                              "pipeline operation is stale or duplicated",
                              pipeline);
      usedStages[binding.stage] = true;
      stages.emplace(binding.operation, binding.stage);
    }
    if (llvm::is_contained(usedStages, false))
      return prepareFailure(LoopPipeliningFailureKind::BrokenContract,
                            "pipeline stage assignment has an empty stage",
                            pipeline);
    for (mlir::Operation &operation : loop.getBody()->without_terminator())
      if (!boundOperations.count(&operation))
        return prepareFailure(LoopPipeliningFailureKind::BrokenContract,
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
          return prepareFailure(LoopPipeliningFailureKind::BrokenContract,
                                "pipeline SSA dependence has no bound source",
                                pipeline);
        if (dependence->distance == 0 && source->second > destination.stage)
          return prepareFailure(
              LoopPipeliningFailureKind::BrokenContract,
              "pipeline reverses a same-iteration SSA dependence", pipeline);
      }

    if (std::optional<std::string> effectFailure =
            checkExternalEffects(loop, stages))
      return prepareFailure(LoopPipeliningFailureKind::Unsupported,
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
                LoopPipeliningFailureKind::Unsupported,
                "cross-stage async token requires finite unrolling", pipeline);
        }
      }

    if (choice.lowering == TilePipelineLowering::FiniteUnrolled &&
        (*tripCount >
         limits.maxFiniteUnrolledOperations / choice.operations.size()))
      return prepareFailure(LoopPipeliningFailureKind::Indeterminate,
                            "finite pipeline exceeds its code-size bound",
                            pipeline);

    prepared.pipelines.push_back(PreparedTilePipeline{
        loop, choice.operations, choice.lowering, *tripCount, stageCount});
  }
  return {std::move(prepared), {}};
}

PipelinedModuleResult pipelineLoops(mlir::OwningOpRef<mlir::ModuleOp> module,
                                    PreparedLoopPipelines prepared) {
  if (!module || prepared.module != *module ||
      hasTemporaryAttributes(module->getOperation()))
    return materializationFailure(
        LoopPipeliningFailureKind::BrokenContract,
        "prepared loop pipelining is stale for the owned module");

  mlir::IRRewriter rewriter(module->getContext());
  PipelinedModule result;
  uint64_t nextOperationIndex = 0;
  struct OperationBinding {
    size_t pipeline = 0;
    uint32_t stage = 0;
  };
  std::map<uint64_t, OperationBinding> bindings;

  for (auto [pipelineIndex, pipeline] : llvm::enumerate(prepared.pipelines)) {
    PipelinedLoop materialized;
    materialized.stageCount = pipeline.stageCount;
    materialized.originalOperationCount = pipeline.operations.size();
    const uint64_t maximumStage = pipeline.stageCount - 1;
    materialized.kernelTripCount = pipeline.tripCount - maximumStage;
    result.pipelines.push_back(std::move(materialized));

    for (const TilePipelineOperation &operation : pipeline.operations) {
      if (nextOperationIndex >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return materializationFailure(
            LoopPipeliningFailureKind::Indeterminate,
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
      PipelinePhase phase = PipelinePhase::Kernel;
      if (part == mlir::scf::PipeliningOption::PipelinerPart::Prologue)
        phase = PipelinePhase::Prologue;
      else if (part == mlir::scf::PipeliningOption::PipelinerPart::Epilogue)
        phase = PipelinePhase::Epilogue;
      operation->setAttr(
          kPhaseAttr, rewriter.getI32IntegerAttr(static_cast<uint32_t>(phase)));
      operation->setAttr(kPhaseIterationAttr,
                         rewriter.getI64IntegerAttr(iteration));
      if (phase == PipelinePhase::Kernel)
        operation->setAttr(kKernelIterationAttr, rewriter.getI64IntegerAttr(0));
    };
    bool modified = false;
    rewriter.setInsertionPoint(pipeline.loop);
    auto kernel =
        mlir::scf::pipelineForLoop(rewriter, pipeline.loop, options, &modified);
    if (mlir::failed(kernel))
      return materializationFailure(
          modified ? LoopPipeliningFailureKind::CompilerBug
                   : LoopPipeliningFailureKind::Unsupported,
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
          LoopPipeliningFailureKind::CompilerBug,
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
            LoopPipeliningFailureKind::CompilerBug,
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
    PipelinePhase materializedPhase =
        static_cast<PipelinePhase>(phase.getValue().getZExtValue());
    uint64_t iteration = phaseIteration.getValue().getZExtValue();
    if (materializedPhase == PipelinePhase::Kernel)
      if (auto kernelIteration =
              operation->getAttrOfType<mlir::IntegerAttr>(kKernelIterationAttr))
        iteration = kernelIteration.getValue().getZExtValue();
    result.pipelines[binding->second.pipeline].operations.push_back(
        {operation, binding->second.stage, materializedPhase, iteration});
  });
  clearTemporaryAttributes(module->getOperation(), rewriter);
  if (hasTemporaryAttributes(module->getOperation()))
    return materializationFailure(
        LoopPipeliningFailureKind::CompilerBug,
        "materialized loop pipelining retained private attributes");
  result.module = std::move(module);
  std::string failureReason;
  if (mlir::failed(verifyPipelinedModule(result, &failureReason)))
    return materializationFailure(LoopPipeliningFailureKind::CompilerBug,
                                  failureReason);
  return {std::move(result), {}};
}

mlir::LogicalResult verifyPipelinedModule(const PipelinedModule &materialized,
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
    return fail("loop-pipelining verifier requires clean valid IR");
  llvm::DenseSet<mlir::Operation *> operations;
  for (const PipelinedLoop &pipeline : materialized.pipelines) {
    if (pipeline.stageCount < 2 || pipeline.originalOperationCount == 0 ||
        pipeline.operations.empty())
      return fail("materialized pipeline is empty or malformed");
    for (const PipelinedOperation &operation : pipeline.operations)
      if (!operation.operation || !module->isAncestor(operation.operation) ||
          operation.stage >= pipeline.stageCount ||
          !operations.insert(operation.operation).second)
        return fail("materialized pipeline relation is stale");
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
