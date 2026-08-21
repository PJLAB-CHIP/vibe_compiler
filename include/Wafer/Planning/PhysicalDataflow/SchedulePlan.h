//===- SchedulePlan.h - Canonical typed order and worker plan -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_SCHEDULEPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_SCHEDULEPLAN_H

#include "Wafer/IR/NCCCompletion.h"
#include "Wafer/Planning/PhysicalDataflow/BufferPlan.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

using ScheduleNodeId = StorageAccessSite;

struct ScheduleWorkerBinding {
  /// Canonical default for worker-capable issues emitted by this high-level
  /// node. The node itself is not asserted to be an NCC operation.
  ScheduleNodeId node;
  NCCWorker worker = NCCWorker::Worker0;

  friend bool operator==(const ScheduleWorkerBinding &lhs,
                         const ScheduleWorkerBinding &rhs) {
    return lhs.node == rhs.node && lhs.worker == rhs.worker;
  }
};

struct ClosedSchedulePlan {
  std::vector<ScheduleNodeId> order;
  std::vector<ScheduleWorkerBinding> workerBindings;

  friend bool operator==(const ClosedSchedulePlan &lhs,
                         const ClosedSchedulePlan &rhs) {
    return lhs.order == rhs.order && lhs.workerBindings == rhs.workerBindings;
  }
};

struct ScheduleDependency {
  ScheduleNodeId predecessor;
  ScheduleNodeId successor;
  StorageObjectId object;

  friend bool operator==(const ScheduleDependency &lhs,
                         const ScheduleDependency &rhs) {
    return lhs.predecessor == rhs.predecessor &&
           lhs.successor == rhs.successor && lhs.object == rhs.object;
  }
  friend bool operator<(const ScheduleDependency &lhs,
                        const ScheduleDependency &rhs) {
    if (!(lhs.predecessor == rhs.predecessor))
      return lhs.predecessor < rhs.predecessor;
    if (!(lhs.successor == rhs.successor))
      return lhs.successor < rhs.successor;
    return lhs.object < rhs.object;
  }
};

struct CanonicalScheduleCoordinate {
  ClosedSchedulePlan plan;
  std::vector<ScheduleDependency> dependencies;
};

enum class BrokenSchedulePlanReason : uint8_t {
  EmptyScheduleInput,
  DuplicateSerializedExecution,
  DuplicateStorageObject,
  DuplicateStorageResource,
  DuplicateLifetime,
  MissingStorageResource,
  MissingLifetime,
  UnknownStorageObject,
  UnknownExecutionSite,
  EmptyUseSet,
  CyclicDependency,
};

struct BrokenSchedulePlan {
  BrokenSchedulePlanReason reason =
      BrokenSchedulePlanReason::EmptyScheduleInput;
  std::optional<ScheduleNodeId> node;
  std::string detail;
};

using CanonicalSchedulePlanOutcome =
    std::variant<CanonicalScheduleCoordinate, BrokenSchedulePlan>;

const CanonicalScheduleCoordinate *
getCanonicalScheduleCoordinate(const CanonicalSchedulePlanOutcome &outcome);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_SCHEDULEPLAN_H
