//===- LoopPipelining.cpp - Tile LoopPipelining -------------------===//

#include "Wafer/Transforms/Tile/LoopPipelining.h"
#include "LoopPipeliningInternal.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/Analysis/Instr/StaticIndexRange.h"
#include "Wafer/IR/WaferDialect.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

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
  if (!loop)
    return std::nullopt;
  if (loop.getLowerBound() == loop.getUpperBound())
    return 0;
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0)
    return std::nullopt;
  if (*upper <= *lower)
    return 0;
  const __int128 difference =
      static_cast<__int128>(*upper) - static_cast<__int128>(*lower);
  if (difference <= 0 || difference > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return 1 +
         (static_cast<uint64_t>(difference) - 1) / static_cast<uint64_t>(*step);
}

std::optional<LoopPipeliningFailure>
checkLoopPipeliningDomain(mlir::scf::ForOp loop, uint32_t stageCount) {
  using Bounds = mlir::ValueBoundsConstraintSet;
  using mlir::presburger::BoundType;
  auto bound = [](mlir::Value value, BoundType kind) -> std::optional<int64_t> {
    if (auto constant = mlir::getConstantIntValue(value))
      return constant;
    auto interval =
        memory_planning::detail::evaluateNonNegativeStaticIndexRange(value);
    if (interval.succeeded() && !interval.range.empty)
      return kind == BoundType::LB ? interval.range.min : interval.range.max;
    auto result = Bounds::computeConstantBound(kind, Bounds::Variable(value),
                                               {}, /*closedUB=*/true);
    if (mlir::failed(result))
      return std::nullopt;
    return *result;
  };
  auto stepMin = bound(loop.getStep(), BoundType::LB);
  auto stepMax = bound(loop.getStep(), BoundType::UB);
  auto lowerMin = bound(loop.getLowerBound(), BoundType::LB);
  auto lowerMax = bound(loop.getLowerBound(), BoundType::UB);
  auto upperMin = bound(loop.getUpperBound(), BoundType::LB);
  auto upperMax = bound(loop.getUpperBound(), BoundType::UB);
  if (!stepMin || !stepMax || *stepMin <= 0 || !lowerMin || !lowerMax ||
      !upperMin || !upperMax)
    return LoopPipeliningFailure{
        LoopPipeliningFailureKind::Unsupported,
        {},
        "pipeline domain lacks bounded indices or a proven positive step"};
  const __int128 shift = static_cast<__int128>(stageCount - 1) * *stepMax;
  unsigned width = 64;
  if (auto integer =
          mlir::dyn_cast<mlir::IntegerType>(loop.getInductionVar().getType()))
    width = integer.getWidth();
  if (width > 64)
    return LoopPipeliningFailure{
        LoopPipeliningFailureKind::Unsupported,
        {},
        "pipeline index width exceeds the supported range proof"};
  const __int128 min = llvm::APInt::getSignedMinValue(width).getSExtValue();
  const __int128 max = llvm::APInt::getSignedMaxValue(width).getSExtValue();
  if (static_cast<__int128>(*upperMax) - *lowerMin > max ||
      static_cast<__int128>(*upperMin) - *lowerMax < min ||
      static_cast<__int128>(std::max(*lowerMax, *upperMax)) + shift > max ||
      static_cast<__int128>(*upperMin) - shift < min)
    return LoopPipeliningFailure{
        LoopPipeliningFailureKind::Unsupported,
        {},
        "pipeline stage offsets may overflow the loop index domain"};
  return std::nullopt;
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

bool isSafePipelineMetadata(mlir::Operation *operation,
                            mlir::Value positiveStep) {
  return mlir::isMemoryEffectFree(operation) &&
         (mlir::isSpeculatable(operation) ||
          (mlir::isa<mlir::arith::DivUIOp>(operation) &&
           operation->getOperand(1) == positiveStep));
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
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    if (auto entry = analysis::getSingleExecutionRegionEntryOperand(argument)) {
      collectStorageRoots(entry, roots, visited);
      return;
    }
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value))
    if (auto exit = analysis::getSingleExecutionRegionExitOperand(result)) {
      collectStorageRoots(exit, roots, visited);
      return;
    }
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
                     const std::map<mlir::Operation *, uint32_t> &stages,
                     llvm::ArrayRef<TilePipelineOperation> schedule) {
  using EffectList = llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4>;
  std::function<std::optional<EffectList>(mlir::Operation *)> getEffects =
      [&](mlir::Operation *operation) -> std::optional<EffectList> {
    if (operation->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>() ||
        mlir::isa<mlir::async::ExecuteOp>(operation)) {
      EffectList combined;
      for (auto &region : operation->getRegions())
        for (auto &block : region)
          for (auto &nested : block) {
            auto effects = getEffects(&nested);
            if (!effects)
              return std::nullopt;
            combined.append(*effects);
          }
      return combined;
    }
    if (mlir::isa<mlir::async::AwaitOp, mlir::async::YieldOp>(operation))
      return EffectList{};
    auto interface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (!interface)
      return std::nullopt;
    EffectList effects, bounded;
    interface.getEffects(effects);
    for (auto effect : effects) {
      if (effect.getValue()) {
        bounded.push_back(effect);
        continue;
      }
      if (!mlir::isa<WaferTileDataflowOpInterface, WaferInstructionOpInterface>(
              operation))
        return std::nullopt;
      auto *resource = effect.getResource();
      if (resource == WaferComputeResource::get() ||
          resource == WaferMovementResource::get())
        continue;
      std::optional<MemorySpace> space;
      if (resource == WaferSPMResource::get())
        space = MemorySpace::SPM;
      else if (resource == WaferDDRResource::get())
        space = MemorySpace::DDR;
      if (!space || !llvm::any_of(effects, [&](const auto &actual) {
            auto value = actual.getValue();
            auto type = value
                            ? mlir::dyn_cast<mlir::MemRefType>(value.getType())
                            : mlir::MemRefType{};
            auto memory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
            return memory && memory.getSpace() == *space &&
                   actual.getEffect() == effect.getEffect();
          }))
        return std::nullopt;
    }
    return bounded;
  };

  struct Access {
    mlir::Value value;
    bool write;
    uint32_t stage;
    mlir::Operation *operation;
    bool iterationPrivate;
  };
  llvm::DenseMap<mlir::Operation *, size_t> scheduleOrder;
  for (auto [index, binding] : llvm::enumerate(schedule))
    scheduleOrder[binding.operation] = index;
  llvm::SmallVector<Access> accesses;
  mlir::AliasAnalysis aliases(loop->getParentOfType<mlir::ModuleOp>());
  auto differentAddressSpaces = [](mlir::Value lhs, mlir::Value rhs) {
    auto a = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
    auto b = mlir::dyn_cast<mlir::MemRefType>(rhs.getType());
    auto first = a ? getWaferMemoryAttr(a) : MemoryAttr{};
    auto second = b ? getWaferMemoryAttr(b) : MemoryAttr{};
    return first && second && first.getSpace() != second.getSpace();
  };
  auto unview = [](mlir::Value value) {
    while (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
               value.getDefiningOp()))
      value = view.getViewSource();
    return value;
  };
  auto disjointInvariantSlices = [&](mlir::Value lhs, mlir::Value rhs) {
    auto a = lhs.getDefiningOp<mlir::memref::SubViewOp>();
    auto b = rhs.getDefiningOp<mlir::memref::SubViewOp>();
    if (!a || !b || a.getSource() != b.getSource())
      return false;
    // Invariant slices are disjoint for all iteration pairs, not just for
    // equal IVs. Dynamic per-iteration slices require a separate distance
    // proof.
    for (auto slice : {a, b})
      for (mlir::Value operand : slice->getOperands().drop_front()) {
        auto roots = getStorageRoots(operand);
        if (llvm::any_of(roots, [&](mlir::Value root) {
              auto arg = mlir::dyn_cast<mlir::BlockArgument>(root);
              return (arg && arg.getOwner() == loop.getBody()) ||
                     !isExternalToLoop(root, loop);
            }))
          return false;
      }
    auto overlap = mlir::ValueBoundsConstraintSet::areOverlappingSlices(
        loop.getContext(), mlir::HyperrectangularSlice(a),
        mlir::HyperrectangularSlice(b));
    return mlir::succeeded(overlap) && !*overlap;
  };
  // Iterate in IR order so rejection diagnostics never depend on pointers.
  for (auto &operation : loop.getBody()->without_terminator()) {
    uint32_t stage = stages.at(&operation);
    auto effects = getEffects(&operation);
    if (!effects)
      return "pipelined operation has no typed memory-effect contract";
    for (const auto &effect : *effects) {
      if (mlir::isa<mlir::MemoryEffects::Allocate>(effect.getEffect()))
        continue;
      auto roots = getStorageRoots(effect.getValue());
      bool iterationPrivate =
          !roots.empty() && llvm::all_of(roots, [&](mlir::Value root) {
            auto *definition = root.getDefiningOp();
            auto allocation =
                mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(
                    definition);
            return allocation && loop->isAncestor(definition) &&
                   allocation.getEffectOnValue<mlir::MemoryEffects::Allocate>(
                       root);
          });
      accesses.push_back(
          {effect.getValue(),
           !mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()), stage,
           &operation, iterationPrivate});
    }
  }
  for (auto [i, lhs] : llvm::enumerate(accesses))
    for (auto rhs : llvm::ArrayRef<Access>(accesses).drop_front(i + 1)) {
      const bool reversed =
          lhs.stage > rhs.stage ||
          (lhs.stage == rhs.stage && scheduleOrder.lookup(lhs.operation) >
                                         scheduleOrder.lookup(rhs.operation));
      if (lhs.operation == rhs.operation || (!lhs.write && !rhs.write) ||
          (!reversed && (lhs.stage == rhs.stage ||
                         (lhs.iterationPrivate && rhs.iterationPrivate))) ||
          differentAddressSpaces(lhs.value, rhs.value) ||
          aliases.alias(lhs.value, rhs.value).isNo() ||
          disjointInvariantSlices(lhs.value, rhs.value))
        continue;
      auto lhsRoots = getStorageRoots(lhs.value);
      auto rhsRoots = getStorageRoots(rhs.value);
      if (llvm::all_of(lhsRoots, [&](mlir::Value a) {
            return llvm::all_of(rhsRoots, [&](mlir::Value b) {
              return aliases.alias(a, b).isNo();
            });
          }))
        continue;
      auto first = unview(lhs.value), second = unview(rhs.value);
      if (!reversed && lhs.stage != rhs.stage && first == second &&
          isDistanceOneRotatingView(first, loop) &&
          std::max(lhs.stage, rhs.stage) - std::min(lhs.stage, rhs.stage) < 2)
        continue;
      std::string detail;
      llvm::raw_string_ostream stream(detail);
      stream << "cross-stage memory dependence has no proven disjoint range or "
                "rotating lifetime: "
             << lhs.operation->getName() << " stage=" << lhs.stage
             << (lhs.write ? " write " : " read ") << lhs.value << "; "
             << rhs.operation->getName() << " stage=" << rhs.stage
             << (rhs.write ? " write " : " read ") << rhs.value;
      return detail;
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
        llvm::any_of(loops,
                     [&](mlir::Operation *other) {
                       return other->isAncestor(loop) ||
                              loop->isAncestor(other);
                     }) ||
        !loops.insert(loop.getOperation()).second)
      return prepareFailure(LoopPipeliningFailureKind::BrokenContract,
                            "pipeline loop is stale, duplicated or overlaps "
                            "another selected scope",
                            pipeline);
    if (!loop.getRegion().hasOneBlock())
      return prepareFailure(LoopPipeliningFailureKind::Unsupported,
                            "pipeline loop is not single-block", pipeline);
    std::optional<uint64_t> tripCount = getStaticTripCount(loop);
    const uint32_t stageCount = getStageCount(choice.operations);
    if (stageCount < 2 || choice.operations.empty())
      return prepareFailure(LoopPipeliningFailureKind::BrokenContract,
                            "pipeline stages are invalid", pipeline);

    if (auto failure = checkLoopPipeliningDomain(loop, stageCount)) {
      failure->pipeline = pipeline;
      return {{}, std::move(failure)};
    }
    const bool guarded = !tripCount || *tripCount < stageCount;
    if (choice.lowering == TilePipelineLowering::FiniteUnrolled && guarded)
      return prepareFailure(
          LoopPipeliningFailureKind::Unsupported,
          "finite unrolling requires a nonempty static kernel", pipeline);

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
      if (guarded && binding.stage + 1 < stageCount &&
          binding.operation->getNumResults() != 0 &&
          !isSafePipelineMetadata(binding.operation, loop.getStep()))
        return prepareFailure(
            LoopPipeliningFailureKind::Unsupported,
            "early stage result has no safe inactive-path value", pipeline);
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
            checkExternalEffects(loop, stages, choice.operations))
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
        loop, choice.operations, choice.lowering, tripCount, stageCount});
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
    const bool guarded =
        !pipeline.tripCount || *pipeline.tripCount < pipeline.stageCount;
    if (pipeline.tripCount)
      materialized.kernelTripCount =
          guarded ? *pipeline.tripCount : *pipeline.tripCount - maximumStage;
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
    options.supportDynamicLoops = guarded;
    options.peelEpilogue = !guarded;
    if (guarded) {
      // The pinned helper classifies a constant short loop before reading the
      // schedule. Keep its exact bound in SSA so prologue guards are emitted
      // for every stage, including static zero/one-iteration domains.
      if (pipeline.tripCount) {
        rewriter.setInsertionPoint(pipeline.loop);
        auto zero = rewriter.create<mlir::arith::ConstantOp>(
            pipeline.loop.getLoc(),
            rewriter.getIntegerAttr(pipeline.loop.getUpperBound().getType(),
                                    0));
        auto bound = rewriter.create<mlir::arith::AddIOp>(
            pipeline.loop.getLoc(), pipeline.loop.getUpperBound(), zero);
        rewriter.modifyOpInPlace(pipeline.loop,
                                 [&] { pipeline.loop.setUpperBound(bound); });
      }
      options.predicateFn = [step = pipeline.loop.getStep()](
                                mlir::RewriterBase &rewriter,
                                mlir::Operation *operation,
                                mlir::Value predicate) -> mlir::Operation * {
        if (isSafePipelineMetadata(operation, step))
          return operation;
        if (operation->getNumResults())
          return nullptr;
        rewriter.setInsertionPoint(operation);
        if (auto scope = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
          if (scope.getElseRegion().empty()) {
            mlir::Value condition = predicate;
            if (mlir::getConstantIntValue(scope.getCondition()) != 1)
              condition = rewriter.create<mlir::arith::AndIOp>(
                  operation->getLoc(), predicate, scope.getCondition());
            rewriter.modifyOpInPlace(
                scope, [&] { scope.getConditionMutable().set(condition); });
            return scope;
          }
        }
        auto guarded = rewriter.create<mlir::scf::IfOp>(
            operation->getLoc(), predicate, /*withElseRegion=*/false);
        // Move the actual operation; the helper tracks this wrapper as the
        // replacement operation, with no artificial result on the false path.
        if (auto index = operation->getAttr(kOperationIndexAttr)) {
          guarded->setAttr(kOperationIndexAttr, index);
          operation->removeAttr(kOperationIndexAttr);
        }
        rewriter.moveOpBefore(operation, guarded.thenYield());
        return guarded;
      };
    }
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

    if (guarded)
      continue;

    // Publish the exact static kernel domain. The pinned helper leaves
    // `ub - stage * step` as unfurled arith operations; current completion and
    // lifetime consumers need the proven nonempty loop bounds in the IR.
    const uint64_t factor = *pipeline.tripCount - maximumStage;
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
