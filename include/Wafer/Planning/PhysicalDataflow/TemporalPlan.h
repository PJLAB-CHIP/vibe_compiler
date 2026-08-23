//===- TemporalPlan.h - Per-execution temporal plan schema ----*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALPLAN_H

#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

struct TopLevelWorkPieceId {
  uint32_t piece = 0;

  friend bool operator==(const TopLevelWorkPieceId &lhs,
                         const TopLevelWorkPieceId &rhs) {
    return lhs.piece == rhs.piece;
  }
  friend bool operator<(const TopLevelWorkPieceId &lhs,
                        const TopLevelWorkPieceId &rhs) {
    return lhs.piece < rhs.piece;
  }
};

struct NestedUseClassId {
  DemandFragmentId relation;
  llvm::SmallVector<int64_t, 4> requestedOffsets;
  llvm::SmallVector<int64_t, 4> requestedExtents;

  friend bool operator==(const NestedUseClassId &lhs,
                         const NestedUseClassId &rhs) {
    return lhs.relation == rhs.relation &&
           lhs.requestedOffsets == rhs.requestedOffsets &&
           lhs.requestedExtents == rhs.requestedExtents;
  }
  friend bool operator<(const NestedUseClassId &lhs,
                        const NestedUseClassId &rhs) {
    return std::tie(lhs.relation, lhs.requestedOffsets, lhs.requestedExtents) <
           std::tie(rhs.relation, rhs.requestedOffsets, rhs.requestedExtents);
  }
};

/// Stable identity of one exact nested invocation class. The rectangular
/// coordinates are the class itself, not cached wave work: equal producer
/// classes merge all selected region-use relations and share one temporal
/// choice, while different main/tail/halo classes cannot be collapsed.
struct NestedInvocationClassId {
  RegionExecutionId parent;
  std::vector<NestedUseClassId> uses;
  llvm::SmallVector<int64_t, 4> producerOffsets;
  llvm::SmallVector<int64_t, 4> producerExtents;

  friend bool operator==(const NestedInvocationClassId &lhs,
                         const NestedInvocationClassId &rhs) {
    return lhs.parent == rhs.parent && lhs.uses == rhs.uses &&
           lhs.producerOffsets == rhs.producerOffsets &&
           lhs.producerExtents == rhs.producerExtents;
  }
  friend bool operator<(const NestedInvocationClassId &lhs,
                        const NestedInvocationClassId &rhs) {
    return std::tie(lhs.parent, lhs.uses, lhs.producerOffsets,
                    lhs.producerExtents) < std::tie(rhs.parent, rhs.uses,
                                                    rhs.producerOffsets,
                                                    rhs.producerExtents);
  }
};

using TraversalInvocationId =
    std::variant<TopLevelWorkPieceId, NestedInvocationClassId>;

struct TraversalScopeId {
  RegionExecutionId execution;
  TraversalInvocationId invocation = TopLevelWorkPieceId{};

  friend bool operator==(const TraversalScopeId &lhs,
                         const TraversalScopeId &rhs) {
    return lhs.execution == rhs.execution && lhs.invocation == rhs.invocation;
  }
  friend bool operator<(const TraversalScopeId &lhs,
                        const TraversalScopeId &rhs) {
    if (!(lhs.execution == rhs.execution))
      return lhs.execution < rhs.execution;
    return lhs.invocation < rhs.invocation;
  }
};

struct TemporalScopePlan {
  TraversalScopeId id;
  llvm::SmallVector<int64_t, 4> iteratorTileSizes;
  llvm::SmallVector<uint32_t, 4> waveLoopOrder;

  friend bool operator==(const TemporalScopePlan &lhs,
                         const TemporalScopePlan &rhs) {
    return lhs.id == rhs.id && lhs.iteratorTileSizes == rhs.iteratorTileSizes &&
           lhs.waveLoopOrder == rhs.waveLoopOrder;
  }
  friend bool operator<(const TemporalScopePlan &lhs,
                        const TemporalScopePlan &rhs) {
    return std::tie(lhs.id, lhs.iteratorTileSizes, lhs.waveLoopOrder) <
           std::tie(rhs.id, rhs.iteratorTileSizes, rhs.waveLoopOrder);
  }
};

struct TemporalPlan {
  std::vector<TemporalScopePlan> scopes;

  friend bool operator==(const TemporalPlan &lhs, const TemporalPlan &rhs) {
    return lhs.scopes == rhs.scopes;
  }
  friend bool operator<(const TemporalPlan &lhs, const TemporalPlan &rhs) {
    return lhs.scopes < rhs.scopes;
  }
};

enum class BrokenTemporalPlanReason : uint8_t {
  DuplicateRootWork,
  RegionWorkMismatch,
  MissingExecution,
  DuplicateScope,
  InvalidLocalExtent,
};

struct BrokenTemporalPlan {
  BrokenTemporalPlanReason reason =
      BrokenTemporalPlanReason::RegionWorkMismatch;
  std::optional<analysis::RootRegionWorkId> work;
  std::string detail;
};

using CanonicalTemporalPlanOutcome =
    std::variant<TemporalPlan, BrokenTemporalPlan>;

const TemporalPlan *
getTemporalPlan(const CanonicalTemporalPlanOutcome &outcome);

/// Returns the required execution named by a scope, or null for an explicit
/// replica. This is a semantic query used by canonical-only consumers; it does
/// not recover identity from position or spelling.
const ExecutionInstanceId *getRequiredExecution(const TraversalScopeId &scope);

bool isTopLevelScope(const TraversalScopeId &scope);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALPLAN_H
