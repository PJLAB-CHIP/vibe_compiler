//===- ExecutionStructureDomain.h - Finite structure domain -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_EXECUTIONSTRUCTUREDOMAIN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_EXECUTIONSTRUCTUREDOMAIN_H

#include "Wafer/Planning/PhysicalDataflow/ExecutionStructurePlan.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

struct ExecutionStructureLimits {
  uint32_t maxStages = 0;
  uint64_t maxSuccessorSteps = 1000000;
};

/// Policy-free input to the finite successor kernel. Production derives these
/// descriptions from one EventGraph and selected temporal facts; bounded
/// reference tests can exercise the same kernel without constructing IR.
struct ExecutionStructureScopeDescription {
  PipelineScopeId id;
  uint64_t tripCount = 1;
  std::vector<EventId> stageableEvents;
  std::vector<EventDependency> dependencies;
  bool pipelinedEligible = false;
};

enum class ExecutionStructureDomainFailureKind : uint8_t {
  UnsupportedSemantics,
  Indeterminate,
  BrokenContract,
};

struct ExecutionStructureDomainFailure {
  ExecutionStructureDomainFailureKind kind =
      ExecutionStructureDomainFailureKind::BrokenContract;
  std::string detail;
};

enum class ExecutionStructureSuccessorKind : uint8_t {
  Plan,
  End,
  Indeterminate,
  CompilerBug,
};

class ExecutionStructureCursor {
private:
  struct ScopeCursor {
    bool serialized = true;
    uint32_t stageCount = 0;
    std::vector<uint32_t> eventStages;
    uint64_t launchDistance = 0;
  };

  std::vector<ScopeCursor> scopes;

  friend class ExecutionStructureDomain;
};

class ExecutionStructureSuccessor {
public:
  ExecutionStructureSuccessorKind getKind() const { return kind; }
  const ExecutionStructurePlan *getPlan() const {
    return plan ? &*plan : nullptr;
  }
  const ExecutionStructureCursor *getCursor() const {
    return cursor ? &*cursor : nullptr;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  ExecutionStructureSuccessor(
      ExecutionStructureSuccessorKind kind,
      std::optional<ExecutionStructurePlan> plan = {},
      std::optional<ExecutionStructureCursor> cursor = {},
      std::string detail = {})
      : kind(kind), plan(std::move(plan)), cursor(std::move(cursor)),
        detail(std::move(detail)) {}

  ExecutionStructureSuccessorKind kind;
  std::optional<ExecutionStructurePlan> plan;
  std::optional<ExecutionStructureCursor> cursor;
  std::string detail;

  friend class ExecutionStructureDomain;
};

struct ExecutionStructureDomainResult;

class ExecutionStructureDomain {
public:
  ExecutionStructureSuccessor getFirstPlan() const;
  ExecutionStructureSuccessor
  getNextPlan(const ExecutionStructureCursor &cursor) const;
  bool contains(const ExecutionStructurePlan &plan) const;

private:
  using ScopeDomain = ExecutionStructureScopeDescription;

  enum class ScopeAdvanceKind : uint8_t { Choice, End, Indeterminate };
  struct ScopeAdvance {
    ScopeAdvanceKind kind = ScopeAdvanceKind::End;
    std::string detail;
  };

  ExecutionStructureDomain(std::vector<ScopeDomain> scopes,
                           ExecutionStructureLimits limits)
      : scopes(std::move(scopes)), limits(limits) {}

  ScopeAdvance
  advanceScope(size_t index,
               ExecutionStructureCursor::ScopeCursor &cursor) const;
  bool containsScope(size_t index,
                     const ExecutionStructureChoice &choice) const;
  ExecutionStructurePlan
  buildPlan(const ExecutionStructureCursor &cursor) const;

  std::vector<ScopeDomain> scopes;
  ExecutionStructureLimits limits;

  friend ExecutionStructureDomainResult buildExecutionStructureDomain(
      const EventGraph &, llvm::ArrayRef<TemporalScopeDescriptor>,
      const TemporalPlan &, const ExecutionStructureLimits &);
  friend ExecutionStructureDomainResult buildExecutionStructureDomain(
      llvm::ArrayRef<ExecutionStructureScopeDescription>,
      const ExecutionStructureLimits &);
};

struct ExecutionStructureDomainResult {
  std::optional<ExecutionStructureDomain> domain;
  std::optional<ExecutionStructureDomainFailure> failure;

  bool succeeded() const { return domain.has_value(); }
};

ExecutionStructureDomainResult buildExecutionStructureDomain(
    const EventGraph &graph,
    llvm::ArrayRef<TemporalScopeDescriptor> temporalScopes,
    const TemporalPlan &temporal,
    const ExecutionStructureLimits &limits = ExecutionStructureLimits());

ExecutionStructureDomainResult buildExecutionStructureDomain(
    llvm::ArrayRef<ExecutionStructureScopeDescription> scopes,
    const ExecutionStructureLimits &limits = ExecutionStructureLimits());

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_EXECUTIONSTRUCTUREDOMAIN_H
