//===- ScheduleMaterialization.cpp - Selected event schedule IR ------===//

#include "Wafer/Planning/PhysicalDataflow/ScheduleMaterialization.h"

#include "Wafer/Analysis/Scheduling/NCCCompletionAnalysis.h"
#include "Wafer/IR/NCCCompletion.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <set>
#include <utility>

namespace wafer::compiler::detail {
namespace {

PreparedScheduleMaterializationResult
prepareFailure(ScheduleMaterializationFailureKind kind, llvm::StringRef detail,
               std::optional<EventId> event = std::nullopt) {
  return {{},
          ScheduleMaterializationFailure{kind, std::move(event), detail.str()}};
}

MaterializedScheduleResult
materializationFailure(ScheduleMaterializationFailureKind kind,
                       llvm::StringRef detail,
                       std::optional<EventId> event = std::nullopt) {
  return {{},
          ScheduleMaterializationFailure{kind, std::move(event), detail.str()}};
}

bool moduleOwns(mlir::ModuleOp module, mlir::Operation *operation) {
  return module && operation && module->isAncestor(operation);
}

bool blockBelongsToModule(mlir::ModuleOp module, mlir::Block *block) {
  return module && block && block->getParentOp() &&
         module->isAncestor(block->getParentOp());
}

std::optional<TileId> getScopeTile(const ControlScopeId &scope) {
  if (const auto *tile = std::get_if<TileControlScope>(&scope))
    return tile->tile;
  return std::nullopt;
}

bool isDTEIssue(mlir::Operation *operation) {
  return mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation);
}

std::vector<mlir::Value>
getDTEIssueTokens(const ScheduleEventIRBinding &binding) {
  std::vector<mlir::Value> tokens;
  for (mlir::Operation *operation : binding.operations)
    if (isDTEIssue(operation))
      for (mlir::Value result : operation->getResults())
        if (mlir::isa<mlir::async::TokenType>(result.getType()))
          tokens.push_back(result);
  return tokens;
}

llvm::SmallVector<mlir::Block *, 4>
getBindingBlocks(const ScheduleEventIRBinding &binding) {
  llvm::SmallVector<mlir::Block *, 4> blocks;
  if (binding.blocks.empty()) {
    if (binding.block)
      blocks.push_back(binding.block);
    return blocks;
  }
  blocks.append(binding.blocks.begin(), binding.blocks.end());
  return blocks;
}

bool bindingContainsBlock(const ScheduleEventIRBinding &binding,
                          mlir::Block *block) {
  llvm::SmallVector<mlir::Block *, 4> blocks = getBindingBlocks(binding);
  return llvm::is_contained(blocks, block);
}

bool haveSameBindingBlocks(const ScheduleEventIRBinding &lhs,
                           const ScheduleEventIRBinding &rhs) {
  llvm::SmallVector<mlir::Block *, 4> lhsBlocks = getBindingBlocks(lhs);
  llvm::SmallVector<mlir::Block *, 4> rhsBlocks = getBindingBlocks(rhs);
  return lhsBlocks.size() == rhsBlocks.size() &&
         llvm::all_of(lhsBlocks, [&](mlir::Block *block) {
           return llvm::is_contained(rhsBlocks, block);
         });
}

std::vector<mlir::Value>
getDTEIssueTokens(const ScheduleEventIRBinding &binding, mlir::Block *block) {
  ScheduleEventIRBinding local = binding;
  llvm::erase_if(local.operations, [&](mlir::Operation *operation) {
    return operation->getBlock() != block;
  });
  return getDTEIssueTokens(local);
}

} // namespace

PreparedScheduleMaterializationResult prepareScheduleMaterialization(
    const ScheduleDomain &domain, const ClosedSchedulePlan &plan,
    llvm::ArrayRef<ScheduleIRModule> modules,
    llvm::ArrayRef<ScheduleEventIRBinding> bindings) {
  if (!domain.contains(plan) || modules.empty())
    return prepareFailure(ScheduleMaterializationFailureKind::BrokenContract,
                          "schedule materialization received a stale plan or "
                          "empty module set");
  std::map<int64_t, mlir::ModuleOp> modulesByTile;
  mlir::MLIRContext *context = nullptr;
  for (const ScheduleIRModule &module : modules) {
    mlir::ModuleOp current = module.module;
    if (!current || (context && current.getContext() != context) ||
        !modulesByTile.try_emplace(module.tile.getValue(), current).second)
      return prepareFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "schedule materialization has a missing or duplicate Tile module");
    context = current.getContext();
  }
  for (const ScheduleIRModule &module : modules) {
    bool hasDerivedCompletion = false;
    mlir::ModuleOp current = module.module;
    current.walk([&](mlir::Operation *operation) {
      hasDerivedCompletion |=
          mlir::isa<SyncNCCJoinOp, InstrDTEWaitOp>(operation);
    });
    if (hasDerivedCompletion)
      return prepareFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "schedule materialization input contains an unowned derived "
          "completion");
  }

  std::map<EventId, const ScheduleEventIRBinding *> byEvent;
  llvm::DenseSet<mlir::Operation *> boundOperations;
  for (const ScheduleEventIRBinding &binding : bindings) {
    auto module = modulesByTile.find(binding.tile.getValue());
    llvm::SmallVector<mlir::Block *, 4> blocks = getBindingBlocks(binding);
    llvm::SmallDenseSet<mlir::Block *, 4> uniqueBlocks;
    if (module == modulesByTile.end() || !binding.block || blocks.empty() ||
        llvm::any_of(blocks,
                     [&](mlir::Block *block) {
                       return !blockBelongsToModule(module->second, block) ||
                              !uniqueBlocks.insert(block).second;
                     }) ||
        !llvm::is_contained(blocks, binding.block) ||
        !byEvent.try_emplace(binding.event, &binding).second)
      return prepareFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "schedule event binding is stale, duplicated, or on an unknown Tile",
          binding.event);
    for (mlir::Operation *operation : binding.operations)
      if (!operation || !bindingContainsBlock(binding, operation->getBlock()) ||
          operation->hasTrait<mlir::OpTrait::IsTerminator>() ||
          !moduleOwns(module->second, operation) ||
          !boundOperations.insert(operation).second ||
          mlir::isa<SyncNCCJoinOp, InstrDTEWaitOp>(operation))
        return prepareFailure(
            ScheduleMaterializationFailureKind::BrokenContract,
            "schedule event operation is stale, duplicated, a terminator, or "
            "a pre-existing derived completion",
            binding.event);
  }
  for (const ScheduleIRModule &module : modules) {
    bool missingExecutableEvent = false;
    mlir::ModuleOp current = module.module;
    current.walk([&](mlir::Operation *operation) {
      if ((mlir::isa<WaferNCCIssueOpInterface>(operation) ||
           mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation)) &&
          !boundOperations.count(operation))
        missingExecutableEvent = true;
    });
    if (missingExecutableEvent)
      return prepareFailure(ScheduleMaterializationFailureKind::BrokenContract,
                            "actual executable issue has no EventId binding");
  }

  PreparedScheduleMaterialization prepared;
  prepared.plan = plan;
  prepared.modules.assign(modules.begin(), modules.end());
  std::set<EventId> scheduledEvents;
  for (const ControlOrder &order : plan.controlOrders) {
    if (order.events.empty())
      return prepareFailure(ScheduleMaterializationFailureKind::BrokenContract,
                            "selected control order is empty");
    PreparedScheduleScope scope;
    scope.order = order;
    std::optional<TileId> expectedTile = getScopeTile(order.scope);
    for (const EventId &event : order.events) {
      auto binding = byEvent.find(event);
      if (binding == byEvent.end() || !scheduledEvents.insert(event).second ||
          (expectedTile && binding->second->tile != *expectedTile))
        return prepareFailure(
            ScheduleMaterializationFailureKind::BrokenContract,
            "selected control order has a missing, duplicate, or wrong-Tile "
            "event binding",
            event);
      scope.events.push_back(*binding->second);
    }
    prepared.scopes.push_back(std::move(scope));
  }
  if (scheduledEvents.size() != bindings.size())
    return prepareFailure(
        ScheduleMaterializationFailureKind::BrokenContract,
        "actual event bindings are not all-and-only the selected control "
        "orders");

  std::map<mlir::Block *, std::vector<const ScheduleEventIRBinding *>>
      eventsByBlock;
  for (const PreparedScheduleScope &scope : prepared.scopes)
    for (const ScheduleEventIRBinding &binding : scope.events)
      for (mlir::Block *block : getBindingBlocks(binding))
        eventsByBlock[block].push_back(&binding);
  for (const auto &[block, blockEvents] : eventsByBlock) {
    (void)block;
    std::map<mlir::Operation *, size_t> operationPositions;
    size_t nextPosition = 0;
    for (const ScheduleEventIRBinding *binding : blockEvents)
      for (mlir::Operation *operation : binding->operations)
        if (operation->getBlock() == block)
          operationPositions.emplace(operation, nextPosition++);
    for (const ScheduleEventIRBinding *binding : blockEvents)
      for (mlir::Operation *operation : binding->operations) {
        if (operation->getBlock() != block)
          continue;
        const size_t destination = operationPositions.at(operation);
        for (mlir::Value operand : operation->getOperands()) {
          auto source = operationPositions.find(operand.getDefiningOp());
          if (source != operationPositions.end() &&
              source->second >= destination)
            return prepareFailure(
                ScheduleMaterializationFailureKind::BrokenContract,
                "selected actual order violates an SSA dependence",
                binding->event);
        }
      }
  }

  std::map<EventId, NCCWorker> workers;
  for (const EventWorkerBinding &binding : plan.workerBindings)
    if (!workers.try_emplace(binding.event, binding.worker).second)
      return prepareFailure(ScheduleMaterializationFailureKind::BrokenContract,
                            "selected schedule has duplicate worker bindings",
                            binding.event);
  for (const auto &[event, worker] : workers) {
    auto binding = byEvent.find(event);
    unsigned issueCount = 0;
    if (binding != byEvent.end())
      for (mlir::Operation *operation : binding->second->operations)
        issueCount += mlir::isa<WaferNCCIssueOpInterface>(operation);
    if (issueCount == 0 || static_cast<uint32_t>(worker) >= kNCCWorkerCount)
      return prepareFailure(ScheduleMaterializationFailureKind::BrokenContract,
                            "worker binding has no actual typed NCC issue",
                            event);
  }

  std::map<EventId, const CompletionPlacement *> completions;
  for (const CompletionPlacement &placement : plan.completionPlacements)
    if (!completions.try_emplace(placement.issue, &placement).second)
      return prepareFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "selected schedule has multiple completion placements for one "
          "issue",
          placement.issue);
  for (const auto &[event, binding] : byEvent) {
    bool hasNCCIssue = false;
    bool hasDTEIssue = false;
    for (mlir::Operation *operation : binding->operations) {
      hasNCCIssue |= mlir::isa<WaferNCCIssueOpInterface>(operation);
      hasDTEIssue |= isDTEIssue(operation);
    }
    auto completion = completions.find(event);
    const bool hasMatchingNCCCompletion =
        completion != completions.end() &&
        (completion->second->protocol == CompletionProtocol::NCCParticipant ||
         completion->second->protocol ==
             CompletionProtocol::NCCSynchronousWriteback);
    if (hasNCCIssue &&
        (workers.find(event) == workers.end() || !hasMatchingNCCCompletion))
      return prepareFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "actual NCC issue lacks its selected worker or NCC completion",
          event);
    if (hasDTEIssue &&
        (completion == completions.end() ||
         completion->second->protocol != CompletionProtocol::DirectDTE))
      return prepareFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "actual Direct-DTE issue lacks its selected completion", event);
  }

  for (const CompletionPlacement &placement : plan.completionPlacements) {
    if (!byEvent.count(placement.issue) ||
        !byEvent.count(placement.completion) ||
        !byEvent.count(placement.boundary.after))
      return prepareFailure(ScheduleMaterializationFailureKind::BrokenContract,
                            "completion placement has an unbound event",
                            placement.issue);
    if (placement.protocol != CompletionProtocol::NoAsynchronousCompletion &&
        !haveSameBindingBlocks(*byEvent.at(placement.issue),
                               *byEvent.at(placement.completion)))
      return prepareFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "asynchronous issue and completion have different occurrences",
          placement.issue);
    if (placement.protocol == CompletionProtocol::NCCParticipant ||
        placement.protocol == CompletionProtocol::NCCSynchronousWriteback) {
      const NCCCompletionKind expected =
          placement.protocol == CompletionProtocol::NCCParticipant
              ? NCCCompletionKind::OrderedAsynchronousIssue
              : NCCCompletionKind::SynchronousWriteback;
      if (!llvm::all_of(
              byEvent.at(placement.issue)->operations,
              [&](mlir::Operation *operation) {
                return !mlir::isa<WaferNCCIssueOpInterface>(operation) ||
                       getNCCOperationCompletion(operation).kind == expected;
              }) ||
          !llvm::any_of(byEvent.at(placement.issue)->operations,
                        [](mlir::Operation *operation) {
                          return mlir::isa<WaferNCCIssueOpInterface>(operation);
                        }))
        return prepareFailure(
            ScheduleMaterializationFailureKind::BrokenContract,
            "NCC completion protocol differs from its actual typed issue",
            placement.issue);
    }
    if (placement.protocol != CompletionProtocol::DirectDTE)
      continue;
    std::vector<mlir::Value> tokens =
        getDTEIssueTokens(*byEvent.at(placement.issue));
    if (tokens.empty())
      return prepareFailure(ScheduleMaterializationFailureKind::BrokenContract,
                            "Direct-DTE placement has no actual issue token",
                            placement.issue);
    if (llvm::any_of(tokens,
                     [](mlir::Value token) { return !token.use_empty(); }))
      return prepareFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "Direct-DTE issue token already has an unplanned consumer",
          placement.issue);
  }
  return {std::move(prepared), {}};
}

MaterializedScheduleResult
materializeSchedule(std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules,
                    PreparedScheduleMaterialization prepared) {
  if (modules.empty() || modules.size() != prepared.modules.size())
    return materializationFailure(
        ScheduleMaterializationFailureKind::BrokenContract,
        "prepared schedule has a different owned module set");
  for (auto [module, expected] : llvm::zip_equal(modules, prepared.modules))
    if (!module || *module != expected.module)
      return materializationFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "prepared schedule is stale for the owned module epoch");

  mlir::IRRewriter rewriter(modules.front()->getContext());
  std::map<mlir::Block *, std::vector<const ScheduleEventIRBinding *>>
      eventsByBlock;
  for (const PreparedScheduleScope &scope : prepared.scopes)
    for (const ScheduleEventIRBinding &binding : scope.events)
      for (mlir::Block *block : getBindingBlocks(binding))
        eventsByBlock[block].push_back(&binding);
  for (const auto &[block, blockEvents] : eventsByBlock) {
    mlir::Operation *terminator = block->getTerminator();
    if (!terminator)
      return materializationFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "schedule control block has no terminator");
    for (const ScheduleEventIRBinding *binding : blockEvents)
      for (mlir::Operation *operation : binding->operations)
        if (operation->getBlock() == block)
          rewriter.moveOpBefore(operation, terminator);
  }

  std::map<EventId, const ScheduleEventIRBinding *> byEvent;
  std::map<EventId, const PreparedScheduleScope *> scopeByEvent;
  for (const PreparedScheduleScope &scope : prepared.scopes)
    for (const ScheduleEventIRBinding &binding : scope.events) {
      byEvent.emplace(binding.event, &binding);
      scopeByEvent.emplace(binding.event, &scope);
    }
  for (const EventWorkerBinding &binding : prepared.plan.workerBindings)
    for (mlir::Operation *operation : byEvent.at(binding.event)->operations)
      if (mlir::isa<WaferNCCIssueOpInterface>(operation) &&
          mlir::failed(setNCCIssueWorker(operation, binding.worker)))
        return materializationFailure(
            ScheduleMaterializationFailureKind::CompilerBug,
            "selected worker could not be applied to an actual NCC issue",
            binding.event);

  std::map<EventBoundaryId, std::vector<CompletionPlacement>> byBoundary;
  for (const CompletionPlacement &placement :
       prepared.plan.completionPlacements)
    byBoundary[placement.boundary].push_back(placement);
  std::vector<MaterializedCompletionGroup> completionGroups;
  for (auto &[boundary, placements] : byBoundary) {
    auto boundaryEvent = byEvent.find(boundary.after);
    if (boundaryEvent == byEvent.end())
      return materializationFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "selected completion boundary is absent from actual control");
    const PreparedScheduleScope *scope = scopeByEvent.at(boundary.after);
    auto logicalBoundary = llvm::find(scope->order.events, boundary.after);
    if (logicalBoundary == scope->order.events.end())
      return materializationFailure(
          ScheduleMaterializationFailureKind::BrokenContract,
          "selected completion boundary is absent from its control order");
    const size_t logicalBoundaryIndex = static_cast<size_t>(
        std::distance(scope->order.events.begin(), logicalBoundary));
    llvm::sort(placements);
    llvm::SmallVector<mlir::Block *, 4> occurrenceBlocks;
    for (const CompletionPlacement &placement : placements) {
      if (scopeByEvent.at(placement.issue) != scope ||
          scopeByEvent.at(placement.completion) != scope)
        return materializationFailure(
            ScheduleMaterializationFailureKind::BrokenContract,
            "selected completion crosses an actual control scope");
      const ScheduleEventIRBinding &occurrenceOwner =
          placement.protocol == CompletionProtocol::NoAsynchronousCompletion
              ? *boundaryEvent->second
              : *byEvent.at(placement.issue);
      for (mlir::Block *block : getBindingBlocks(occurrenceOwner))
        if (!llvm::is_contained(occurrenceBlocks, block))
          occurrenceBlocks.push_back(block);
    }
    for (mlir::Block *block : occurrenceBlocks) {
      mlir::Operation *insertionAnchor = block->getTerminator();
      for (size_t index = logicalBoundaryIndex + 1;
           index < scope->order.events.size(); ++index) {
        const ScheduleEventIRBinding *event =
            byEvent.at(scope->order.events[index]);
        auto operation =
            llvm::find_if(event->operations, [&](mlir::Operation *candidate) {
              return candidate->getBlock() == block;
            });
        if (operation != event->operations.end()) {
          insertionAnchor = *operation;
          break;
        }
      }
      rewriter.setInsertionPoint(insertionAnchor);
      uint32_t participantMask = 0;
      std::vector<mlir::Value> dteTokens;
      std::vector<CompletionPlacement> blockPlacements;
      for (const CompletionPlacement &placement : placements) {
        const bool applies =
            placement.protocol == CompletionProtocol::NoAsynchronousCompletion
                ? bindingContainsBlock(*boundaryEvent->second, block)
                : bindingContainsBlock(*byEvent.at(placement.issue), block);
        if (!applies)
          continue;
        blockPlacements.push_back(placement);
        if (placement.protocol == CompletionProtocol::NCCParticipant)
          participantMask |= placement.participantMask;
        if (placement.protocol == CompletionProtocol::DirectDTE) {
          std::vector<mlir::Value> tokens =
              getDTEIssueTokens(*byEvent.at(placement.issue), block);
          dteTokens.insert(dteTokens.end(), tokens.begin(), tokens.end());
        }
      }
      MaterializedCompletionGroup group;
      group.boundary = boundary;
      group.block = block;
      group.placements = std::move(blockPlacements);
      if (participantMask != 0) {
        llvm::SmallVector<int64_t, 3> participants;
        for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker)
          if ((participantMask & (uint32_t{1} << worker)) != 0)
            participants.push_back(worker);
        auto join = rewriter.create<SyncNCCJoinOp>(
            insertionAnchor->getLoc(), participants, rewriter.getUnitAttr());
        group.operations.push_back(join);
      }
      if (!dteTokens.empty()) {
        auto wait = rewriter.create<InstrDTEWaitOp>(insertionAnchor->getLoc(),
                                                    dteTokens);
        group.operations.push_back(wait);
      }
      completionGroups.push_back(std::move(group));
    }
  }

  for (mlir::OwningOpRef<mlir::ModuleOp> &module : modules)
    if (mlir::failed(mlir::verify(*module)))
      return materializationFailure(
          ScheduleMaterializationFailureKind::CompilerBug,
          "selected schedule produced verifier-invalid Instr IR");
  MaterializedSchedule result;
  result.modules = std::move(modules);
  for (const ScheduleIRModule &module : prepared.modules)
    result.moduleTiles.push_back(module.tile);
  result.plan = std::move(prepared.plan);
  for (const PreparedScheduleScope &scope : prepared.scopes)
    result.eventBindings.insert(result.eventBindings.end(),
                                scope.events.begin(), scope.events.end());
  result.completionGroups = std::move(completionGroups);
  std::string failureReason;
  if (mlir::failed(verifyMaterializedSchedule(result, &failureReason)))
    return materializationFailure(
        ScheduleMaterializationFailureKind::CompilerBug, failureReason);
  return {std::move(result), {}};
}

mlir::LogicalResult
verifyMaterializedSchedule(const MaterializedSchedule &materialized,
                           std::string *failureReason) {
  auto fail = [&](llvm::StringRef detail) {
    if (failureReason)
      *failureReason = detail.str();
    return mlir::failure();
  };
  if (failureReason)
    failureReason->clear();
  if (materialized.modules.empty() ||
      materialized.moduleTiles.size() != materialized.modules.size())
    return fail("schedule verifier requires owned modules");
  std::map<int64_t, mlir::ModuleOp> modulesByTile;
  for (auto [owned, tile] :
       llvm::zip_equal(materialized.modules, materialized.moduleTiles))
    if (!owned ||
        !modulesByTile.try_emplace(tile.getValue(), owned.get()).second)
      return fail("schedule verifier has a missing or duplicate Tile module");
  std::map<EventId, const ScheduleEventIRBinding *> byEvent;
  llvm::DenseSet<mlir::Operation *> boundOperations;
  for (const ScheduleEventIRBinding &binding : materialized.eventBindings) {
    auto module = modulesByTile.find(binding.tile.getValue());
    llvm::SmallVector<mlir::Block *, 4> blocks = getBindingBlocks(binding);
    llvm::SmallDenseSet<mlir::Block *, 4> uniqueBlocks;
    if (module == modulesByTile.end() || !binding.block || blocks.empty() ||
        llvm::any_of(blocks,
                     [&](mlir::Block *block) {
                       return !blockBelongsToModule(module->second, block) ||
                              !uniqueBlocks.insert(block).second;
                     }) ||
        !llvm::is_contained(blocks, binding.block) ||
        !byEvent.try_emplace(binding.event, &binding).second)
      return fail("schedule verifier has duplicate or stale event bindings");
    for (mlir::Operation *operation : binding.operations)
      if (!operation || !bindingContainsBlock(binding, operation->getBlock()) ||
          !moduleOwns(module->second, operation) ||
          !boundOperations.insert(operation).second)
        return fail("schedule verifier has stale or duplicate event ops");
  }
  std::set<EventId> scheduledEvents;
  std::map<mlir::Block *, std::vector<mlir::Operation *>> operationsByBlock;
  for (const ControlOrder &order : materialized.plan.controlOrders) {
    for (const EventId &event : order.events) {
      auto binding = byEvent.find(event);
      if (binding == byEvent.end() || !scheduledEvents.insert(event).second)
        return fail("schedule verifier is missing a control event");
      for (mlir::Operation *operation : binding->second->operations)
        operationsByBlock[operation->getBlock()].push_back(operation);
    }
  }
  for (const auto &[block, operations] : operationsByBlock) {
    (void)block;
    for (size_t index = 1; index < operations.size(); ++index)
      if (!operations[index - 1]->isBeforeInBlock(operations[index]))
        return fail("actual operation order differs from ClosedSchedulePlan");
  }
  if (scheduledEvents.size() != byEvent.size())
    return fail("schedule verifier control does not cover every event");
  for (const EventWorkerBinding &binding : materialized.plan.workerBindings) {
    auto event = byEvent.find(binding.event);
    if (event == byEvent.end())
      return fail("schedule verifier is missing a worker event");
    bool found = false;
    for (mlir::Operation *operation : event->second->operations)
      if (auto issue = mlir::dyn_cast<WaferNCCIssueOpInterface>(operation)) {
        found = true;
        if (issue.getIssueWorker() != binding.worker)
          return fail("actual NCC worker differs from ClosedSchedulePlan");
      }
    if (!found)
      return fail("schedule worker binding has no actual NCC issue");
  }

  std::set<CompletionPlacement> actualPlacements;
  std::set<std::pair<EventBoundaryId, mlir::Block *>> actualBoundaries;
  llvm::DenseSet<mlir::Operation *> groupedCompletionOperations;
  for (const MaterializedCompletionGroup &group :
       materialized.completionGroups) {
    if (group.placements.empty() || !group.block ||
        !actualBoundaries.insert({group.boundary, group.block}).second)
      return fail("schedule completion group is empty or duplicated");
    auto boundaryEvent = byEvent.find(group.boundary.after);
    if (boundaryEvent == byEvent.end())
      return fail("schedule completion group has an unknown boundary");
    const ControlOrder *control = nullptr;
    size_t boundaryIndex = 0;
    for (const ControlOrder &candidate : materialized.plan.controlOrders) {
      auto position = llvm::find(candidate.events, group.boundary.after);
      if (position == candidate.events.end())
        continue;
      if (control)
        return fail("schedule completion boundary appears in multiple scopes");
      control = &candidate;
      boundaryIndex = static_cast<size_t>(
          std::distance(candidate.events.begin(), position));
    }
    if (!control)
      return fail("schedule completion boundary has no control scope");

    InstrDTEWaitOp actualWait;
    SyncNCCJoinOp actualJoin;
    uint32_t actualMask = 0;
    for (mlir::Operation *operation : group.operations) {
      if (!operation || operation->getBlock() != group.block ||
          !groupedCompletionOperations.insert(operation).second)
        return fail("schedule completion group has a stale or duplicate op");
      if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation)) {
        if (actualWait)
          return fail("schedule completion group has multiple DTE waits");
        actualWait = wait;
      }
      if (auto join = mlir::dyn_cast<SyncNCCJoinOp>(operation)) {
        if (actualJoin)
          return fail("schedule completion group has multiple NCC joins");
        actualJoin = join;
        actualMask |= getNCCOperationCompletion(join).participantMask;
      }
      if (!mlir::isa<InstrDTEWaitOp, SyncNCCJoinOp>(operation))
        return fail("schedule completion group contains an unexpected op");
      for (size_t index = 0; index <= boundaryIndex; ++index) {
        const auto *event = byEvent.at(control->events[index]);
        if (!bindingContainsBlock(*event, group.block))
          continue;
        for (mlir::Operation *scheduled : event->operations)
          if (scheduled->getBlock() != group.block)
            continue;
          else if (!scheduled->isBeforeInBlock(operation))
            return fail("actual completion precedes its selected boundary");
      }
      for (size_t index = boundaryIndex + 1; index < control->events.size();
           ++index) {
        const auto *event = byEvent.at(control->events[index]);
        if (!bindingContainsBlock(*event, group.block))
          continue;
        for (mlir::Operation *scheduled : event->operations)
          if (scheduled->getBlock() != group.block)
            continue;
          else if (!operation->isBeforeInBlock(scheduled))
            return fail("actual completion exceeds its selected boundary");
      }
    }
    uint32_t expectedMask = 0;
    llvm::DenseSet<mlir::Value> expectedTokens;
    for (const CompletionPlacement &placement : group.placements) {
      if (!(placement.boundary == group.boundary))
        return fail("schedule completion is stored under the wrong boundary");
      auto issueEvent = byEvent.find(placement.issue);
      if (issueEvent == byEvent.end())
        return fail("schedule completion has an unknown issue event");
      actualPlacements.insert(placement);
      const ScheduleEventIRBinding &occurrenceOwner =
          placement.protocol == CompletionProtocol::NoAsynchronousCompletion
              ? *boundaryEvent->second
              : *issueEvent->second;
      if (!bindingContainsBlock(occurrenceOwner, group.block))
        return fail("completion is bound to the wrong occurrence block");
      if (placement.protocol == CompletionProtocol::NCCParticipant)
        expectedMask |= placement.participantMask;
      if (placement.protocol == CompletionProtocol::DirectDTE)
        for (mlir::Value token :
             getDTEIssueTokens(*issueEvent->second, group.block))
          if (!expectedTokens.insert(token).second)
            return fail("one Direct-DTE token has multiple placements");
    }
    if (actualMask != expectedMask ||
        static_cast<bool>(actualJoin) != (expectedMask != 0) ||
        static_cast<bool>(actualWait) != !expectedTokens.empty())
      return fail("actual completion group differs from ClosedSchedulePlan");
    if (actualWait) {
      llvm::DenseSet<mlir::Value> actualTokens;
      for (mlir::Value token : actualWait.getTokens())
        if (!actualTokens.insert(token).second)
          return fail("actual DTE wait contains a duplicate token");
      if (actualTokens.size() != expectedTokens.size() ||
          llvm::any_of(expectedTokens, [&](mlir::Value token) {
            return !actualTokens.count(token);
          }))
        return fail("actual DTE wait tokens differ from ClosedSchedulePlan");
    }
  }
  const std::set<CompletionPlacement> expectedPlacements(
      materialized.plan.completionPlacements.begin(),
      materialized.plan.completionPlacements.end());
  if (actualPlacements != expectedPlacements ||
      expectedPlacements.size() !=
          materialized.plan.completionPlacements.size())
    return fail("schedule verifier has incomplete completion placement");
  std::set<std::pair<EventBoundaryId, mlir::Block *>> expectedBoundaries;
  for (const CompletionPlacement &placement :
       materialized.plan.completionPlacements) {
    auto boundary = byEvent.find(placement.boundary.after);
    auto issue = byEvent.find(placement.issue);
    if (boundary == byEvent.end())
      return fail("schedule verifier has an unknown completion boundary");
    if (issue == byEvent.end())
      return fail("schedule verifier has an unknown completion issue");
    const ScheduleEventIRBinding &occurrenceOwner =
        placement.protocol == CompletionProtocol::NoAsynchronousCompletion
            ? *boundary->second
            : *issue->second;
    for (mlir::Block *block : getBindingBlocks(occurrenceOwner))
      expectedBoundaries.insert({placement.boundary, block});
  }
  if (actualBoundaries != expectedBoundaries)
    return fail("schedule verifier has incomplete completion occurrences");

  for (const mlir::OwningOpRef<mlir::ModuleOp> &owned : materialized.modules) {
    mlir::ModuleOp module = owned.get();
    if (!module || mlir::failed(mlir::verify(module)))
      return fail("schedule verifier received invalid Instr IR");
    auto completion =
        analysis::NCCCompletionAnalysis::create(module, failureReason);
    if (mlir::failed(completion))
      return mlir::failure();
    bool pending = false;
    module.walk([&](mlir::func::ReturnOp operation) {
      auto state = completion->getOperationState(operation);
      pending |= state && state->pendingBefore != 0;
    });
    if (pending)
      return fail("actual schedule reaches a return with pending NCC work");
    llvm::DenseSet<mlir::Value> issues;
    llvm::DenseSet<mlir::Value> waits;
    module.walk([&](mlir::Operation *operation) {
      if (mlir::isa<SyncNCCJoinOp, InstrDTEWaitOp>(operation) &&
          !groupedCompletionOperations.count(operation))
        pending = true;
      if (mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation))
        for (mlir::Value result : operation->getResults())
          if (mlir::isa<mlir::async::TokenType>(result.getType()))
            issues.insert(result);
      if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation))
        for (mlir::Value token : wait.getTokens())
          if (!waits.insert(token).second)
            pending = true;
    });
    if (pending || issues.size() != waits.size() ||
        llvm::any_of(issues,
                     [&](mlir::Value token) { return !waits.count(token); }))
      return fail("actual schedule has unmatched Direct-DTE completion");
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
