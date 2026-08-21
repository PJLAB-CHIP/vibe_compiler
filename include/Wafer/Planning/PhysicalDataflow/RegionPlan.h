//===- RegionPlan.h - Region and execution planning schema ----*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_REGIONPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_REGIONPLAN_H

#include "Wafer/Analysis/PhysicalDataflow/RootRegionWork.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

struct RequiredRootExecution {
  analysis::RootRegionWorkId work;
  LogicalShardId shard;

  friend bool operator==(const RequiredRootExecution &lhs,
                         const RequiredRootExecution &rhs) {
    return lhs.work == rhs.work && lhs.shard == rhs.shard;
  }
  friend bool operator<(const RequiredRootExecution &lhs,
                        const RequiredRootExecution &rhs) {
    if (lhs.work < rhs.work)
      return true;
    if (rhs.work < lhs.work)
      return false;
    return lhs.shard < rhs.shard;
  }
};

struct RequiredMergeExecution {
  analysis::RootRegionWorkId work;
  ReductionGroupId group;

  friend bool operator==(const RequiredMergeExecution &lhs,
                         const RequiredMergeExecution &rhs) {
    return lhs.work == rhs.work && lhs.group == rhs.group;
  }
  friend bool operator<(const RequiredMergeExecution &lhs,
                        const RequiredMergeExecution &rhs) {
    if (lhs.work < rhs.work)
      return true;
    if (rhs.work < lhs.work)
      return false;
    return lhs.group < rhs.group;
  }
};

using RequiredExecutionSource =
    std::variant<RequiredRootExecution, RequiredMergeExecution>;

struct ExecutionInstanceId {
  RequiredExecutionSource source;

  friend bool operator==(const ExecutionInstanceId &lhs,
                         const ExecutionInstanceId &rhs) {
    return lhs.source == rhs.source;
  }
  friend bool operator<(const ExecutionInstanceId &lhs,
                        const ExecutionInstanceId &rhs) {
    if (lhs.source.index() != rhs.source.index())
      return lhs.source.index() < rhs.source.index();
    if (const auto *lhsRoot = std::get_if<RequiredRootExecution>(&lhs.source))
      return *lhsRoot < std::get<RequiredRootExecution>(rhs.source);
    return std::get<RequiredMergeExecution>(lhs.source) <
           std::get<RequiredMergeExecution>(rhs.source);
  }
};

struct ExecutionInstancePlan {
  ExecutionInstanceId id;
};

struct DemandFragmentId {
  analysis::RootBoundaryId source;
  analysis::RootUseId use;
  std::optional<LogicalShardId> ownerShard;
  std::optional<ReductionGroupId> reductionGroup;
  std::optional<TileId> ownerTile;

  friend bool operator==(const DemandFragmentId &lhs,
                         const DemandFragmentId &rhs) {
    return lhs.source == rhs.source && lhs.use == rhs.use &&
           lhs.ownerShard == rhs.ownerShard &&
           lhs.reductionGroup == rhs.reductionGroup &&
           lhs.ownerTile == rhs.ownerTile;
  }
  friend bool operator<(const DemandFragmentId &lhs,
                        const DemandFragmentId &rhs) {
    if (lhs.source != rhs.source)
      return lhs.source < rhs.source;
    if (!(lhs.use == rhs.use))
      return lhs.use < rhs.use;
    if (lhs.reductionGroup != rhs.reductionGroup)
      return lhs.reductionGroup < rhs.reductionGroup;
    if (lhs.ownerShard != rhs.ownerShard)
      return lhs.ownerShard < rhs.ownerShard;
    if (lhs.ownerTile.has_value() != rhs.ownerTile.has_value())
      return lhs.ownerTile.has_value() < rhs.ownerTile.has_value();
    return lhs.ownerTile &&
           lhs.ownerTile->getValue() < rhs.ownerTile->getValue();
  }
};

struct ExternalUseBinding {
  DemandFragmentId fragment;
};

struct RegionGroupPlan {
  TileId tile{0};
  llvm::SmallVector<analysis::RootRegionWorkId, 2> mandatoryRoots;
  std::vector<ExecutionInstancePlan> executions;
  std::vector<ExternalUseBinding> externalBindings;
};

struct RegionPlan {
  std::vector<RegionGroupPlan> groups;
};

enum class BrokenRegionPlanReason : uint8_t {
  DuplicateRootWork,
  WorkStructureMismatch,
  MissingBoundaryFragment,
  DuplicateExecution,
  DuplicateFragment,
};

struct BrokenRegionPlan {
  BrokenRegionPlanReason reason = BrokenRegionPlanReason::WorkStructureMismatch;
  std::optional<analysis::RootRegionWorkId> work;
  std::string detail;
};

using CanonicalRegionPlanOutcome = std::variant<RegionPlan, BrokenRegionPlan>;

const RegionPlan *getRegionPlan(const CanonicalRegionPlanOutcome &outcome);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_REGIONPLAN_H
