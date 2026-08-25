//===- ExecutionStructureMaterialization.cpp - Selected SCF phases ---===//

#include "Wafer/Planning/PhysicalDataflow/ExecutionStructureMaterialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
#include <tuple>
#include <utility>

namespace wafer::compiler::detail {
namespace {

constexpr llvm::StringLiteral kEventIndexAttr =
    "wafer.internal.execution_event_index";
constexpr llvm::StringLiteral kPhaseAttr = "wafer.internal.execution_phase";
constexpr llvm::StringLiteral kPhaseIterationAttr =
    "wafer.internal.execution_phase_iteration";
constexpr llvm::StringLiteral kKernelIterationAttr =
    "wafer.internal.execution_kernel_iteration";

PreparedExecutionStructureResult
prepareFailure(ExecutionStructureMaterializationFailureKind kind,
               llvm::StringRef detail,
               std::optional<PipelineScopeId> scope = std::nullopt) {
  return {{},
          ExecutionStructureMaterializationFailure{kind, std::move(scope),
                                                   detail.str()}};
}

MaterializedExecutionStructureResult
materializationFailure(ExecutionStructureMaterializationFailureKind kind,
                       llvm::StringRef detail,
                       std::optional<PipelineScopeId> scope = std::nullopt) {
  return {{},
          ExecutionStructureMaterializationFailure{kind, std::move(scope),
                                                   detail.str()}};
}

std::optional<uint64_t> getStaticTripCount(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower)
    return std::nullopt;
  const __int128 wideDifference =
      static_cast<__int128>(*upper) - static_cast<__int128>(*lower);
  if (wideDifference <= 0 ||
      wideDifference > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  const uint64_t difference = static_cast<uint64_t>(wideDifference);
  return 1 + (difference - 1) / static_cast<uint64_t>(*step);
}

uint32_t getStageCount(const PipelinedExecutionStructure &plan) {
  uint32_t count = 0;
  for (const EventStageAssignment &assignment : plan.eventStages) {
    if (assignment.stage.getValue() == std::numeric_limits<uint32_t>::max())
      return 0;
    count = std::max(count, assignment.stage.getValue() + 1);
  }
  return count;
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
  mlir::Operation *destination = nullptr;
  uint32_t distance = 0;
};

std::optional<ActualDependence>
getActualDependence(mlir::Value operand, mlir::Operation *destination,
                    mlir::scf::ForOp loop) {
  mlir::Operation *source = operand.getDefiningOp();
  if (source && loop->isAncestor(source))
    return ActualDependence{getTopLevelOwner(source, loop.getBody()),
                            destination, 0};
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
  return ActualDependence{getTopLevelOwner(yieldSource, loop.getBody()),
                          destination, 1};
}

bool hasPlanDependence(const PipelinedExecutionStructure &plan,
                       const EventId &source, const EventId &destination,
                       uint32_t distance) {
  if (source == destination && distance == 0)
    return true;
  return llvm::any_of(plan.dependences, [&](const auto &dependence) {
    return dependence.source == source &&
           dependence.destination == destination &&
           dependence.iterationDistance == distance;
  });
}

bool isExternalToLoop(mlir::Value root, mlir::scf::ForOp loop) {
  mlir::Operation *definition = root.getDefiningOp();
  return !definition || !loop->isAncestor(definition);
}

bool hasExternalStorageProof(mlir::Value root,
                             llvm::ArrayRef<ExternalStorageStageProof> proofs) {
  return llvm::any_of(proofs,
                      [&](const auto &proof) { return proof.root == root; });
}

std::optional<std::string>
checkExternalEffects(const ExecutionStructureLoopBinding &binding,
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
      mlir::Value value = effect.getValue();
      if (!value)
        continue;
      for (mlir::Value root : getStorageRoots(value)) {
        if (!isExternalToLoop(root, binding.steadyLoop))
          continue;
        Summary &summary = summaries[root];
        summary.write |=
            !llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
        summary.stages.insert(stage);
      }
    }
  }
  for (const auto &[root, summary] : summaries)
    if (summary.write && summary.stages.size() > 1 &&
        !hasExternalStorageProof(root, binding.externalStorageProofs))
      return "cross-stage external write has no exact rotating-storage proof";
  return std::nullopt;
}

bool hasTemporaryAttributes(mlir::Operation *root) {
  bool found = false;
  root->walk([&](mlir::Operation *operation) {
    found |= operation->hasAttr(kEventIndexAttr) ||
             operation->hasAttr(kPhaseAttr) ||
             operation->hasAttr(kPhaseIterationAttr) ||
             operation->hasAttr(kKernelIterationAttr);
  });
  return found;
}

void clearTemporaryAttributes(mlir::Operation *root,
                              mlir::RewriterBase &rewriter) {
  root->walk([&](mlir::Operation *operation) {
    if (!operation->hasAttr(kEventIndexAttr) &&
        !operation->hasAttr(kPhaseAttr) &&
        !operation->hasAttr(kPhaseIterationAttr) &&
        !operation->hasAttr(kKernelIterationAttr))
      return;
    rewriter.modifyOpInPlace(operation, [&] {
      operation->removeAttr(kEventIndexAttr);
      operation->removeAttr(kPhaseAttr);
      operation->removeAttr(kPhaseIterationAttr);
      operation->removeAttr(kKernelIterationAttr);
    });
  });
}

} // namespace

PreparedExecutionStructureResult prepareExecutionStructureMaterialization(
    mlir::ModuleOp module, const ExecutionStructurePlan &plan,
    const BufferPlan &buffers,
    llvm::ArrayRef<ExecutionStructureLoopBinding> bindings,
    const ExecutionStructureMaterializationLimits &limits) {
  if (!module || limits.maxFiniteUnrolledOperations == 0)
    return prepareFailure(
        ExecutionStructureMaterializationFailureKind::BrokenContract,
        "execution-structure materialization requires a module and positive "
        "work limit");
  if (hasTemporaryAttributes(module))
    return prepareFailure(
        ExecutionStructureMaterializationFailureKind::BrokenContract,
        "execution-structure input contains private construction attributes");

  std::set<StorageObjectId> selectedObjects;
  for (const StorageObjectPlan &object : buffers.storageObjects)
    if (!selectedObjects.insert(object.id).second)
      return prepareFailure(
          ExecutionStructureMaterializationFailureKind::BrokenContract,
          "execution-structure storage input has duplicate objects");
  std::set<StorageObjectId> familyObjects;
  for (const SlotFamilyPlan &family : buffers.slotFamilies)
    if (family.id.objects.empty() || family.multiplicity < 2 ||
        llvm::any_of(
            family.id.objects,
            [&](const StorageObjectId &object) {
              return !selectedObjects.count(object) ||
                     !familyObjects.insert(object).second;
            }))
      return prepareFailure(
          ExecutionStructureMaterializationFailureKind::BrokenContract,
          "execution-structure storage input has a malformed slot family");

  std::map<PipelineScopeId, const ExecutionStructureLoopBinding *> byScope;
  for (const ExecutionStructureLoopBinding &binding : bindings)
    if (binding.scope.events.empty() || !binding.steadyLoop ||
        !module->isAncestor(binding.steadyLoop) ||
        !byScope.try_emplace(binding.scope, &binding).second)
      return prepareFailure(
          ExecutionStructureMaterializationFailureKind::BrokenContract,
          "execution-structure loop bindings are malformed or duplicated");

  PreparedExecutionStructure prepared;
  prepared.module = module;
  prepared.limits = limits;
  std::set<PipelineScopeId> consumed;
  for (const ExecutionStructureChoice &choice : plan.scopes) {
    if (std::holds_alternative<SerializedExecutionStructure>(choice))
      continue;
    const auto &pipeline = std::get<PipelinedExecutionStructure>(choice);
    auto binding = byScope.find(pipeline.scope);
    if (binding == byScope.end())
      return prepareFailure(
          ExecutionStructureMaterializationFailureKind::BrokenContract,
          "pipelined structure has no exact actual loop binding",
          pipeline.scope);
    consumed.insert(pipeline.scope);
    mlir::scf::ForOp loop = binding->second->steadyLoop;
    if (!loop.getRegion().hasOneBlock())
      return prepareFailure(
          ExecutionStructureMaterializationFailureKind::Unsupported,
          "pipelined steady loop is not single-block", pipeline.scope);
    std::optional<uint64_t> tripCount = getStaticTripCount(loop);
    if (!tripCount || *tripCount != pipeline.iteration.steadyTripCount)
      return prepareFailure(
          ExecutionStructureMaterializationFailureKind::BrokenContract,
          "actual steady-loop trip count differs from the selected iteration "
          "class",
          pipeline.scope);
    const uint32_t stageCount = getStageCount(pipeline);
    if (stageCount < 2 || pipeline.launchDistance != 1 ||
        pipeline.iteration.prefixCount != 1 ||
        pipeline.iteration.tailCount > 1 ||
        pipeline.iteration.steadyTripCount < stageCount)
      return prepareFailure(
          ExecutionStructureMaterializationFailureKind::BrokenContract,
          "selected pipeline has an invalid stage or iteration class",
          pipeline.scope);
    if (pipeline.lowering == ExecutionStructureLowering::FiniteUnrolled) {
      uint64_t actualOperations = 0;
      for (const ExecutionEventOperationGroup &group :
           binding->second->eventOrder) {
        if (actualOperations >
            std::numeric_limits<uint64_t>::max() - group.operations.size())
          return prepareFailure(
              ExecutionStructureMaterializationFailureKind::Indeterminate,
              "finite-unrolled operation count overflows", pipeline.scope);
        actualOperations += group.operations.size();
      }
      if (actualOperations == 0 ||
          pipeline.iteration.steadyTripCount >
              limits.maxFiniteUnrolledOperations / actualOperations)
        return prepareFailure(
            ExecutionStructureMaterializationFailureKind::Indeterminate,
            "finite-unrolled execution structure exceeds its code-size bound",
            pipeline.scope);
    }

    std::map<EventId, StageId> eventStages;
    for (const EventStageAssignment &assignment : pipeline.eventStages)
      if (!eventStages.try_emplace(assignment.event, assignment.stage).second)
        return prepareFailure(
            ExecutionStructureMaterializationFailureKind::BrokenContract,
            "selected pipeline has duplicate event stages", pipeline.scope);
    std::set<EventId> boundEvents;
    llvm::DenseSet<mlir::Operation *> boundOperations;
    std::map<mlir::Operation *, EventId> operationEvents;
    std::map<mlir::Operation *, uint32_t> operationStages;
    PreparedExecutionStructureScope preparedScope;
    preparedScope.plan = pipeline;
    preparedScope.steadyLoop = loop;
    for (const ExternalStorageStageProof &proof :
         binding->second->externalStorageProofs) {
      llvm::DenseSet<mlir::Value> proofRoots = getStorageRoots(proof.root);
      if (!proof.root || proofRoots.size() != 1 ||
          !proofRoots.count(proof.root) || !selectedObjects.count(proof.object))
        return prepareFailure(
            ExecutionStructureMaterializationFailureKind::BrokenContract,
            "external storage proof has an unknown root or object",
            pipeline.scope);
      auto family = llvm::find_if(buffers.slotFamilies, [&](const auto &entry) {
        return llvm::is_contained(entry.id.objects, proof.object);
      });
      if (family == buffers.slotFamilies.end() ||
          !(family->occurrence == pipeline.recurrence) ||
          !llvm::is_contained(family->rotationIterators,
                              pipeline.iteration.recurrenceAxis))
        return prepareFailure(
            ExecutionStructureMaterializationFailureKind::BrokenContract,
            "external storage proof has no matching rotating slot family",
            pipeline.scope);
    }
    for (const ExecutionEventOperationGroup &group :
         binding->second->eventOrder) {
      auto stage = eventStages.find(group.event);
      if (stage == eventStages.end() || !boundEvents.insert(group.event).second)
        return prepareFailure(
            ExecutionStructureMaterializationFailureKind::BrokenContract,
            "actual event order is outside or duplicates the selected stage "
            "map",
            pipeline.scope);
      for (mlir::Operation *operation : group.operations) {
        if (!operation || operation->getBlock() != loop.getBody() ||
            operation->hasTrait<mlir::OpTrait::IsTerminator>() ||
            !boundOperations.insert(operation).second)
          return prepareFailure(
              ExecutionStructureMaterializationFailureKind::BrokenContract,
              "actual event operation is stale, duplicated, nested, or a "
              "terminator",
              pipeline.scope);
        operationEvents.emplace(operation, group.event);
        operationStages.emplace(operation, stage->second.getValue());
        preparedScope.operations.push_back(
            {group.event, operation, stage->second});
      }
    }
    if (boundEvents.size() != eventStages.size())
      return prepareFailure(
          ExecutionStructureMaterializationFailureKind::BrokenContract,
          "actual event order does not cover every selected kernel event",
          pipeline.scope);
    for (mlir::Operation &operation : loop.getBody()->without_terminator())
      if (!boundOperations.count(&operation))
        return prepareFailure(
            ExecutionStructureMaterializationFailureKind::BrokenContract,
            "actual event mapping does not cover every steady-loop operation",
            pipeline.scope);
    if (preparedScope.operations.empty())
      return prepareFailure(
          ExecutionStructureMaterializationFailureKind::Unsupported,
          "pipelined steady loop has no executable operation", pipeline.scope);

    std::map<mlir::Operation *, size_t> operationOrder;
    for (auto [index, operation] : llvm::enumerate(preparedScope.operations))
      operationOrder.emplace(operation.operation, index);
    std::map<EventId, std::vector<mlir::Operation *>> operationsByEvent;
    for (const PreparedExecutionStructureOperation &entry :
         preparedScope.operations)
      operationsByEvent[entry.event].push_back(entry.operation);
    std::set<CompletionObligation> completionObligations;
    for (const CompletionObligation &obligation :
         pipeline.completionObligations) {
      if (!eventStages.count(obligation.issue) ||
          !eventStages.count(obligation.completion) ||
          !completionObligations.insert(obligation).second)
        return prepareFailure(
            ExecutionStructureMaterializationFailureKind::BrokenContract,
            "selected pipeline has a duplicate or unbound completion",
            pipeline.scope);
      if (obligation.protocol == CompletionProtocol::Unknown ||
          obligation.protocol == CompletionProtocol::NCCSynchronousWriteback)
        return prepareFailure(
            ExecutionStructureMaterializationFailureKind::Unsupported,
            "selected pipeline contains an unsupported completion protocol",
            pipeline.scope);
      if (obligation.protocol != CompletionProtocol::DirectDTE)
        continue;
      const std::vector<mlir::Operation *> &issues =
          operationsByEvent[obligation.issue];
      const std::vector<mlir::Operation *> &completions =
          operationsByEvent[obligation.completion];
      llvm::DenseSet<mlir::Value> tokens;
      for (mlir::Operation *issue : issues)
        for (mlir::Value value : issue->getResults())
          if (mlir::isa<mlir::async::TokenType>(value.getType()))
            tokens.insert(value);
      if (tokens.empty() || completions.empty())
        return prepareFailure(
            ExecutionStructureMaterializationFailureKind::BrokenContract,
            "Direct-DTE completion has no actual token or completion owner",
            pipeline.scope);
      for (mlir::Value token : tokens) {
        llvm::DenseSet<mlir::Operation *> users;
        for (mlir::Operation *user : token.getUsers()) {
          mlir::Operation *top = getTopLevelOwner(user, loop.getBody());
          if (!top || !llvm::is_contained(completions, top))
            return prepareFailure(
                ExecutionStructureMaterializationFailureKind::BrokenContract,
                "Direct-DTE token escapes its selected completion event",
                pipeline.scope);
          users.insert(top);
        }
        if (users.size() != 1)
          return prepareFailure(
              ExecutionStructureMaterializationFailureKind::BrokenContract,
              "Direct-DTE token does not have one exact completion consumer",
              pipeline.scope);
      }
      for (mlir::Operation *completion : completions)
        for (mlir::Value operand : getNestedOperands(completion))
          if (mlir::isa<mlir::async::TokenType>(operand.getType()) &&
              !tokens.count(operand))
            return prepareFailure(
                ExecutionStructureMaterializationFailureKind::BrokenContract,
                "Direct-DTE completion consumes an unplanned token",
                pipeline.scope);
    }
    const int64_t operationsPerIteration =
        static_cast<int64_t>(preparedScope.operations.size());
    for (const PreparedExecutionStructureOperation &entry :
         preparedScope.operations) {
      const int64_t destinationCycle =
          static_cast<int64_t>(operationOrder.at(entry.operation)) +
          static_cast<int64_t>(entry.stage.getValue()) * operationsPerIteration;
      for (mlir::Value operand : getNestedOperands(entry.operation)) {
        std::optional<ActualDependence> dependence =
            getActualDependence(operand, entry.operation, loop);
        if (!dependence || !dependence->source ||
            (dependence->source == dependence->destination &&
             dependence->distance == 0))
          continue;
        auto sourceOrder = operationOrder.find(dependence->source);
        auto sourceStage = operationStages.find(dependence->source);
        auto sourceEvent = operationEvents.find(dependence->source);
        if (sourceOrder == operationOrder.end() ||
            sourceStage == operationStages.end() ||
            sourceEvent == operationEvents.end())
          return prepareFailure(
              ExecutionStructureMaterializationFailureKind::BrokenContract,
              "actual SSA dependence has no selected event owner",
              pipeline.scope);
        const int64_t sourceCycle =
            static_cast<int64_t>(sourceOrder->second) +
            static_cast<int64_t>(sourceStage->second) * operationsPerIteration;
        const bool orderedWithinEvent =
            sourceEvent->second == entry.event && dependence->distance == 0;
        if (destinationCycle <
                sourceCycle - operationsPerIteration * dependence->distance ||
            (!orderedWithinEvent &&
             !hasPlanDependence(pipeline, sourceEvent->second, entry.event,
                                dependence->distance))) {
          std::string detail;
          llvm::raw_string_ostream stream(detail);
          stream << "actual SSA dependence contradicts the selected "
                    "distance/stage schedule: "
                 << dependence->source->getName() << " stage "
                 << sourceStage->second << " -> " << entry.operation->getName()
                 << " stage " << entry.stage.getValue() << " at distance "
                 << dependence->distance;
          return prepareFailure(
              ExecutionStructureMaterializationFailureKind::BrokenContract,
              stream.str(), pipeline.scope);
        }
      }
    }
    if (std::optional<std::string> effectFailure =
            checkExternalEffects(*binding->second, operationStages))
      return prepareFailure(
          ExecutionStructureMaterializationFailureKind::Unsupported,
          *effectFailure, pipeline.scope);

    for (const PreparedExecutionStructureOperation &entry :
         preparedScope.operations)
      for (mlir::Value result : entry.operation->getResults()) {
        if (!mlir::isa<mlir::async::TokenType>(result.getType()))
          continue;
        for (mlir::Operation *user : result.getUsers()) {
          mlir::Operation *top = getTopLevelOwner(user, loop.getBody());
          auto userStage = operationStages.find(top);
          if (userStage != operationStages.end() &&
              userStage->second != entry.stage.getValue() &&
              pipeline.lowering != ExecutionStructureLowering::FiniteUnrolled)
            return prepareFailure(
                ExecutionStructureMaterializationFailureKind::Unsupported,
                "cross-stage async token requires finite-unrolled lowering",
                pipeline.scope);
        }
      }
    prepared.scopes.push_back(std::move(preparedScope));
  }
  if (consumed.size() != byScope.size())
    return prepareFailure(
        ExecutionStructureMaterializationFailureKind::BrokenContract,
        "actual loop bindings include a scope not selected as Pipelined");
  llvm::sort(prepared.scopes, [](const auto &lhs, const auto &rhs) {
    return lhs.plan.scope < rhs.plan.scope;
  });
  return {std::move(prepared), {}};
}

MaterializedExecutionStructureResult
materializeExecutionStructure(mlir::OwningOpRef<mlir::ModuleOp> module,
                              PreparedExecutionStructure prepared) {
  if (!module || prepared.module != *module ||
      hasTemporaryAttributes(module->getOperation()))
    return materializationFailure(
        ExecutionStructureMaterializationFailureKind::BrokenContract,
        "prepared execution structure is stale for the owned module");

  mlir::IRRewriter rewriter(module->getContext());
  MaterializedExecutionStructure result;
  uint64_t nextEventIndex = 0;
  std::map<uint64_t, std::pair<size_t, EventId>> eventsByIndex;
  for (auto [scopeIndex, scope] : llvm::enumerate(prepared.scopes)) {
    MaterializedExecutionStructureScope materializedScope;
    materializedScope.plan = scope.plan;
    materializedScope.stageCount = getStageCount(scope.plan);
    const uint64_t maximumStage = materializedScope.stageCount - 1;
    materializedScope.kernelDynamicTripCount =
        scope.plan.iteration.steadyTripCount - maximumStage;

    std::map<EventId, uint32_t> operationCounts;
    for (const PreparedExecutionStructureOperation &entry : scope.operations) {
      if (nextEventIndex >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return materializationFailure(
            ExecutionStructureMaterializationFailureKind::Indeterminate,
            "execution event mapping exceeds its representable work domain",
            scope.plan.scope);
      uint64_t eventIndex = nextEventIndex++;
      eventsByIndex.emplace(eventIndex,
                            std::make_pair(scopeIndex, entry.event));
      ++operationCounts[entry.event];
      rewriter.modifyOpInPlace(entry.operation, [&] {
        entry.operation->setAttr(
            kEventIndexAttr,
            rewriter.getI64IntegerAttr(static_cast<int64_t>(eventIndex)));
      });
    }
    materializedScope.originalOperationCounts.assign(operationCounts.begin(),
                                                     operationCounts.end());
    result.scopes.push_back(std::move(materializedScope));

    mlir::scf::PipeliningOption options;
    options.getScheduleFn =
        [operations = scope.operations](
            mlir::scf::ForOp loop,
            std::vector<std::pair<mlir::Operation *, unsigned>> &schedule) {
          (void)loop;
          for (const PreparedExecutionStructureOperation &entry : operations)
            schedule.emplace_back(entry.operation, entry.stage.getValue());
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
    rewriter.setInsertionPoint(scope.steadyLoop);
    mlir::FailureOr<mlir::scf::ForOp> kernel = mlir::scf::pipelineForLoop(
        rewriter, scope.steadyLoop, options, &modified);
    if (mlir::failed(kernel))
      return materializationFailure(
          modified ? ExecutionStructureMaterializationFailureKind::CompilerBug
                   : ExecutionStructureMaterializationFailureKind::Unsupported,
          "pinned SCF pipelining rejected a fully prepared schedule",
          scope.plan.scope);
    if (scope.plan.lowering == ExecutionStructureLowering::FiniteUnrolled) {
      const uint64_t factor =
          scope.plan.iteration.steadyTripCount - maximumStage;
      std::optional<int64_t> lower =
          mlir::getConstantIntValue(kernel->getLowerBound());
      std::optional<int64_t> step =
          mlir::getConstantIntValue(kernel->getStep());
      const __int128 finiteUpper =
          lower && step
              ? static_cast<__int128>(*lower) +
                    static_cast<__int128>(factor) * *step
              : static_cast<__int128>(std::numeric_limits<int64_t>::max()) + 1;
      if (factor == 0 || !lower || !step || *step <= 0 ||
          finiteUpper > std::numeric_limits<int64_t>::max() ||
          finiteUpper < std::numeric_limits<int64_t>::min())
        return materializationFailure(
            ExecutionStructureMaterializationFailureKind::CompilerBug,
            "finite-unrolled kernel has non-constant or overflowing bounds",
            scope.plan.scope);
      rewriter.setInsertionPoint(*kernel);
      auto upper = rewriter.create<mlir::arith::ConstantOp>(
          kernel->getLoc(),
          rewriter.getIntegerAttr(kernel->getUpperBound().getType(),
                                  static_cast<int64_t>(finiteUpper)));
      rewriter.modifyOpInPlace(*kernel, [&] { kernel->setUpperBound(upper); });
      if (mlir::failed(mlir::loopUnrollByFactor(
              *kernel, factor,
              [&](unsigned iteration, mlir::Operation *operation,
                  mlir::OpBuilder builder) {
                operation->setAttr(kKernelIterationAttr,
                                   builder.getI64IntegerAttr(iteration));
              })))
        return materializationFailure(
            ExecutionStructureMaterializationFailureKind::CompilerBug,
            "finite-unrolled execution structure failed mechanical unroll",
            scope.plan.scope);
    }
  }

  module->walk([&](mlir::Operation *operation) {
    auto eventIndex =
        operation->getAttrOfType<mlir::IntegerAttr>(kEventIndexAttr);
    auto phase = operation->getAttrOfType<mlir::IntegerAttr>(kPhaseAttr);
    auto phaseIteration =
        operation->getAttrOfType<mlir::IntegerAttr>(kPhaseIterationAttr);
    if (!eventIndex || !phase || !phaseIteration)
      return;
    const uint64_t index = eventIndex.getValue().getZExtValue();
    auto selected = eventsByIndex.find(index);
    if (selected == eventsByIndex.end() ||
        selected->second.first >= result.scopes.size())
      return;
    const MaterializedExecutionPhase materializedPhase =
        static_cast<MaterializedExecutionPhase>(
            phase.getValue().getZExtValue());
    uint64_t iteration = phaseIteration.getValue().getZExtValue();
    if (materializedPhase == MaterializedExecutionPhase::Kernel)
      if (auto kernelIteration =
              operation->getAttrOfType<mlir::IntegerAttr>(kKernelIterationAttr))
        iteration = kernelIteration.getValue().getZExtValue();
    const auto &plan = result.scopes[selected->second.first].plan;
    auto stage = llvm::find_if(plan.eventStages, [&](const auto &assignment) {
      return assignment.event == selected->second.second;
    });
    if (stage == plan.eventStages.end())
      return;
    result.scopes[selected->second.first].events.push_back(
        {selected->second.second, operation, stage->stage, materializedPhase,
         iteration});
  });
  clearTemporaryAttributes(module->getOperation(), rewriter);
  if (hasTemporaryAttributes(module->getOperation()))
    return materializationFailure(
        ExecutionStructureMaterializationFailureKind::CompilerBug,
        "materialized execution structure retained temporary attributes");
  result.module = std::move(module);
  std::string failureReason;
  if (mlir::failed(
          verifyMaterializedExecutionStructure(result, &failureReason)))
    return materializationFailure(
        ExecutionStructureMaterializationFailureKind::CompilerBug,
        failureReason);
  return {std::move(result), {}};
}

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
      hasTemporaryAttributes(module.getOperation()))
    return fail("execution-structure verifier requires clean valid IR");

  std::set<PipelineScopeId> scopes;
  llvm::DenseSet<mlir::Operation *> relatedOperations;
  for (const MaterializedExecutionStructureScope &scope : materialized.scopes) {
    if (!scopes.insert(scope.plan.scope).second || scope.stageCount < 2)
      return fail("materialized execution scope is missing or duplicated");
    std::map<EventId, StageId> selectedStages;
    for (const EventStageAssignment &assignment : scope.plan.eventStages)
      if (!selectedStages.try_emplace(assignment.event, assignment.stage)
               .second)
        return fail("materialized execution scope has duplicate event stages");
    for (const MaterializedExecutionEvent &event : scope.events) {
      auto selected = selectedStages.find(event.event);
      if (!event.operation || !module->isAncestor(event.operation) ||
          selected == selectedStages.end() || selected->second != event.stage ||
          !relatedOperations.insert(event.operation).second)
        return fail("materialized execution relation is stale");
    }
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
