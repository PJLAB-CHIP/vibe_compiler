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

struct TemporalScopeId {
  RegionExecutionId execution;

  friend bool operator==(const TemporalScopeId &lhs,
                         const TemporalScopeId &rhs) {
    return lhs.execution == rhs.execution;
  }
  friend bool operator<(const TemporalScopeId &lhs,
                        const TemporalScopeId &rhs) {
    return lhs.execution < rhs.execution;
  }
};

struct TemporalScopePlan {
  TemporalScopeId id;
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

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALPLAN_H
