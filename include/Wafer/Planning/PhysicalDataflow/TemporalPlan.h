//===- TemporalPlan.h - Per-execution temporal plan schema ----*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALPLAN_H

#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

struct TemporalScopePlan {
  ExecutionInstanceId execution;
  llvm::SmallVector<int64_t, 4> iteratorTileSizes;
  llvm::SmallVector<uint32_t, 4> waveLoopOrder;
};

struct TemporalPlan {
  std::vector<TemporalScopePlan> scopes;
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

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALPLAN_H
