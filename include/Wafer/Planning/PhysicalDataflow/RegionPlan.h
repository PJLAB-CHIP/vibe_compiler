//===- RegionPlan.h - Region and execution planning schema ----*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_REGIONPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_REGIONPLAN_H

#include "Wafer/Analysis/PhysicalDataflow/RootRegionWork.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
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
  struct TopLevel {
    friend bool operator==(TopLevel, TopLevel) { return true; }
    friend bool operator<(TopLevel, TopLevel) { return false; }
  };
  struct NestedUnder {
    RequiredExecutionSource consumer;

    friend bool operator==(const NestedUnder &lhs, const NestedUnder &rhs) {
      return lhs.consumer == rhs.consumer;
    }
    friend bool operator<(const NestedUnder &lhs, const NestedUnder &rhs) {
      return lhs.consumer < rhs.consumer;
    }
  };
  using Placement = std::variant<TopLevel, NestedUnder>;
  Placement placement = TopLevel{};

  friend bool operator==(const ExecutionInstancePlan &lhs,
                         const ExecutionInstancePlan &rhs) {
    return lhs.id == rhs.id && lhs.placement == rhs.placement;
  }
  friend bool operator<(const ExecutionInstancePlan &lhs,
                        const ExecutionInstancePlan &rhs) {
    if (lhs.id < rhs.id)
      return true;
    if (rhs.id < lhs.id)
      return false;
    return lhs.placement < rhs.placement;
  }
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

  friend bool operator==(const ExternalUseBinding &lhs,
                         const ExternalUseBinding &rhs) {
    return lhs.fragment == rhs.fragment;
  }
  friend bool operator<(const ExternalUseBinding &lhs,
                        const ExternalUseBinding &rhs) {
    return lhs.fragment < rhs.fragment;
  }
};

struct ReplicaExecutionId {
  RequiredRootExecution producer;
  DemandFragmentId fragment;

  friend bool operator==(const ReplicaExecutionId &lhs,
                         const ReplicaExecutionId &rhs) {
    return lhs.producer == rhs.producer && lhs.fragment == rhs.fragment;
  }
  friend bool operator<(const ReplicaExecutionId &lhs,
                        const ReplicaExecutionId &rhs) {
    if (lhs.producer < rhs.producer)
      return true;
    if (rhs.producer < lhs.producer)
      return false;
    return lhs.fragment < rhs.fragment;
  }
};

struct ReplicaExecutionPlan {
  ReplicaExecutionId id;
  ExecutionInstancePlan::Placement placement =
      ExecutionInstancePlan::TopLevel{};

  friend bool operator==(const ReplicaExecutionPlan &lhs,
                         const ReplicaExecutionPlan &rhs) {
    return lhs.id == rhs.id && lhs.placement == rhs.placement;
  }
  friend bool operator<(const ReplicaExecutionPlan &lhs,
                        const ReplicaExecutionPlan &rhs) {
    if (lhs.id < rhs.id)
      return true;
    if (rhs.id < lhs.id)
      return false;
    return lhs.placement < rhs.placement;
  }
};

using RegionExecutionId = std::variant<ExecutionInstanceId, ReplicaExecutionId>;

enum class LocalUseDelivery : uint8_t {
  StoredRegionValue,
  DirectNestedValue,
};

struct LocalUseBinding {
  DemandFragmentId fragment;
  RegionExecutionId producer;
  LocalUseDelivery delivery = LocalUseDelivery::StoredRegionValue;

  friend bool operator==(const LocalUseBinding &lhs,
                         const LocalUseBinding &rhs) {
    return lhs.fragment == rhs.fragment && lhs.producer == rhs.producer &&
           lhs.delivery == rhs.delivery;
  }
  friend bool operator<(const LocalUseBinding &lhs,
                        const LocalUseBinding &rhs) {
    if (!(lhs.fragment == rhs.fragment))
      return lhs.fragment < rhs.fragment;
    if (!(lhs.producer == rhs.producer))
      return lhs.producer < rhs.producer;
    return lhs.delivery < rhs.delivery;
  }
};

struct RegionGroupPlan {
  TileId tile{0};
  llvm::SmallVector<analysis::RootRegionWorkId, 2> mandatoryRoots;
  std::vector<ExecutionInstancePlan> executions;
  std::vector<ReplicaExecutionPlan> replicas;
  std::vector<LocalUseBinding> localBindings;
  std::vector<ExternalUseBinding> externalBindings;

  friend bool operator==(const RegionGroupPlan &lhs,
                         const RegionGroupPlan &rhs) {
    return lhs.tile == rhs.tile && lhs.mandatoryRoots == rhs.mandatoryRoots &&
           lhs.executions == rhs.executions && lhs.replicas == rhs.replicas &&
           lhs.localBindings == rhs.localBindings &&
           lhs.externalBindings == rhs.externalBindings;
  }
  friend bool operator<(const RegionGroupPlan &lhs,
                        const RegionGroupPlan &rhs) {
    if (lhs.tile != rhs.tile)
      return lhs.tile.getValue() < rhs.tile.getValue();
    return std::tie(lhs.mandatoryRoots, lhs.executions, lhs.replicas,
                    lhs.localBindings, lhs.externalBindings) <
           std::tie(rhs.mandatoryRoots, rhs.executions, rhs.replicas,
                    rhs.localBindings, rhs.externalBindings);
  }
};

struct RegionPlan {
  std::vector<RegionGroupPlan> groups;

  friend bool operator==(const RegionPlan &lhs, const RegionPlan &rhs) {
    return lhs.groups == rhs.groups;
  }
  friend bool operator<(const RegionPlan &lhs, const RegionPlan &rhs) {
    return lhs.groups < rhs.groups;
  }
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
