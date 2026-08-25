//===- ScheduleDomain.h - Fixed-generation event schedule ---*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SCHEDULEDOMAIN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SCHEDULEDOMAIN_H

#include "Wafer/Planning/PhysicalDataflow/SchedulePlan.h"
#include "Wafer/Planning/PhysicalDataflow/StructureSpecificStorageDomain.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

struct ScheduleDomainInput {
  ExecutionStructurePlan structure;
  BufferPlan buffers;
  std::vector<PlannedEvent> events;
  std::vector<EventDependency> hardDependencies;
  std::vector<DisjunctiveResourceOrder> orderChoices;
  std::vector<CompletionObligation> completionObligations;
  std::vector<PlannedResourceUse> resourceUses;
  std::vector<EventComponent> components;
  std::vector<SlotLifetimeRequirement> slotLifetimes;
};

struct ScheduleDomainLimits {
  uint64_t maxSuccessorSteps = 1000000;
};

enum class ScheduleDomainFailureKind : uint8_t {
  ExactRejection,
  Indeterminate,
  BrokenContract,
};

struct ScheduleDomainFailure {
  ScheduleDomainFailureKind kind = ScheduleDomainFailureKind::BrokenContract;
  std::string detail;
};

enum class ScheduleSuccessorKind : uint8_t {
  Plan,
  End,
  Indeterminate,
  CompilerBug,
};

class ScheduleCursor {
private:
  std::vector<uint32_t> workerIndices;
  std::vector<std::vector<EventId>> resourceOrders;
  std::vector<std::vector<EventId>> controlOrders;
  bool proposal = false;

  friend class ScheduleDomain;
};

class ScheduleSuccessor {
public:
  ScheduleSuccessorKind getKind() const { return kind; }
  const ClosedSchedulePlan *getPlan() const { return plan ? &*plan : nullptr; }
  const ScheduleCursor *getCursor() const {
    return cursor ? &*cursor : nullptr;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  ScheduleSuccessor(ScheduleSuccessorKind kind,
                    std::optional<ClosedSchedulePlan> plan = {},
                    std::optional<ScheduleCursor> cursor = {},
                    std::string detail = {})
      : kind(kind), plan(std::move(plan)), cursor(std::move(cursor)),
        detail(std::move(detail)) {}

  ScheduleSuccessorKind kind;
  std::optional<ClosedSchedulePlan> plan;
  std::optional<ScheduleCursor> cursor;
  std::string detail;

  friend class ScheduleDomain;
};

struct ScheduleDomainResult;

class ScheduleDomain {
public:
  ScheduleSuccessor getFirstPlan() const;
  ScheduleSuccessor getNextPlan(const ScheduleCursor &cursor) const;
  bool contains(const ClosedSchedulePlan &plan) const;
  bool isForGeneration(const ExecutionStructurePlan &structure,
                       const BufferPlan &buffers) const {
    return input.structure == structure && input.buffers == buffers;
  }

private:
  struct WorkerDomain {
    EventId event;
    std::vector<NCCWorker> workers;
  };
  struct ResourceDomain {
    ResourceKey resource;
    std::vector<EventId> events;
  };
  struct ControlDomain {
    ControlScopeId scope;
    std::vector<EventId> events;
  };

  enum class AdvanceKind : uint8_t { Advanced, End, Indeterminate };
  struct AdvanceResult {
    AdvanceKind kind = AdvanceKind::End;
    std::string detail;
  };

  ScheduleDomain(ScheduleDomainInput input, std::vector<WorkerDomain> workers,
                 std::vector<ResourceDomain> resources,
                 std::vector<ControlDomain> controls,
                 std::vector<EventResourceBinding> fixedResourceBindings,
                 ScheduleDomainLimits limits)
      : input(std::move(input)), workers(std::move(workers)),
        resources(std::move(resources)), controls(std::move(controls)),
        fixedResourceBindings(std::move(fixedResourceBindings)),
        limits(limits) {}

  std::optional<ScheduleCursor> getInitialCursor() const;
  std::optional<ScheduleCursor> getReceiverFeasibleProposalCursor() const;
  bool
  addSelectedResourceEdges(const ScheduleCursor &cursor,
                           std::set<std::pair<EventId, EventId>> &edges) const;
  AdvanceResult advance(ScheduleCursor &cursor) const;
  ClosedSchedulePlan buildPlan(const ScheduleCursor &cursor) const;
  std::optional<ScheduleCursor> getCursor(const ClosedSchedulePlan &plan) const;

  ScheduleDomainInput input;
  std::vector<WorkerDomain> workers;
  std::vector<ResourceDomain> resources;
  std::vector<ControlDomain> controls;
  std::vector<EventResourceBinding> fixedResourceBindings;
  ScheduleDomainLimits limits;
  std::optional<std::pair<ClosedSchedulePlan, ScheduleCursor>> firstPlan;

  friend ScheduleDomainResult buildScheduleDomain(ScheduleDomainInput,
                                                  const ScheduleDomainLimits &);
};

struct ScheduleDomainResult {
  std::optional<ScheduleDomain> domain;
  std::optional<ScheduleDomainFailure> failure;

  bool succeeded() const { return domain.has_value(); }
};

ScheduleDomainResult buildScheduleDomain(
    ScheduleDomainInput input,
    const ScheduleDomainLimits &limits = ScheduleDomainLimits());

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SCHEDULEDOMAIN_H
