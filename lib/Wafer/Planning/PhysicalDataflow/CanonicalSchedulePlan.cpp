//===- CanonicalSchedulePlan.cpp - Stable worker0 order ---------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalSchedulePlan.h"

#include "llvm/ADT/StringRef.h"

#include <map>
#include <set>
#include <utility>

namespace wafer::compiler::detail {
namespace {

CanonicalSchedulePlanOutcome
broken(BrokenSchedulePlanReason reason, llvm::StringRef detail,
       std::optional<ScheduleNodeId> node = std::nullopt) {
  return BrokenSchedulePlan{reason, std::move(node), detail.str()};
}

} // namespace

CanonicalSchedulePlanOutcome
buildCanonicalSchedulePlan(const CanonicalStorageCoordinate &storage,
                           const SerializedExecutionPlan &serialized) {
  if (storage.plan.storageObjects.empty() || serialized.executions.empty())
    return broken(BrokenSchedulePlanReason::EmptyScheduleInput,
                  "canonical schedule requires storage and executions");

  std::set<ExecutionInstanceId> executions;
  std::set<ScheduleNodeId> nodes;
  for (const ExecutionInstanceId &execution : serialized.executions) {
    if (!executions.insert(execution).second)
      return broken(BrokenSchedulePlanReason::DuplicateSerializedExecution,
                    "schedule input has duplicate serialized execution",
                    ScheduleNodeId{execution});
    nodes.insert(ScheduleNodeId{execution});
  }

  std::set<StorageObjectId> objects;
  for (const StorageObjectPlan &object : storage.plan.storageObjects)
    if (!objects.insert(object.id).second)
      return broken(BrokenSchedulePlanReason::DuplicateStorageObject,
                    "schedule input has duplicate storage object");

  std::set<StorageObjectId> resourceObjects;
  for (const StorageResourceDescription &resource : storage.resources)
    if (!resourceObjects.insert(resource.object).second)
      return broken(BrokenSchedulePlanReason::DuplicateStorageResource,
                    "schedule input has duplicate storage resource");
  if (resourceObjects != objects)
    return broken(BrokenSchedulePlanReason::MissingStorageResource,
                  "storage objects and resources differ");

  for (const PhysicalVersionStorageBinding &binding :
       storage.plan.versionBindings)
    if (!objects.count(binding.object))
      return broken(BrokenSchedulePlanReason::UnknownStorageObject,
                    "version binding references an unknown storage object");
  for (const ReductionGatherStorageBinding &binding :
       storage.plan.gatherStagingBindings)
    if (!objects.count(binding.stagingObject))
      return broken(BrokenSchedulePlanReason::UnknownStorageObject,
                    "gather binding references an unknown storage object");

  std::set<StorageObjectId> lifetimeObjects;
  std::set<ScheduleDependency> dependencies;
  std::set<std::pair<ScheduleNodeId, ScheduleNodeId>> graphEdges;
  auto addSite = [&](const StorageAccessSite &site)
      -> std::optional<CanonicalSchedulePlanOutcome> {
    if (const auto *execution = std::get_if<ExecutionInstanceId>(&site))
      if (!executions.count(*execution))
        return broken(BrokenSchedulePlanReason::UnknownExecutionSite,
                      "storage lifetime references an unknown execution",
                      ScheduleNodeId{*execution});
    nodes.insert(site);
    return std::nullopt;
  };

  for (const StorageLifetimeDescription &lifetime : storage.lifetimes) {
    if (!objects.count(lifetime.object))
      return broken(BrokenSchedulePlanReason::UnknownStorageObject,
                    "lifetime references an unknown storage object");
    if (!lifetimeObjects.insert(lifetime.object).second)
      return broken(BrokenSchedulePlanReason::DuplicateLifetime,
                    "storage object has duplicate lifetimes");
    if (lifetime.uses.empty())
      return broken(BrokenSchedulePlanReason::EmptyUseSet,
                    "storage lifetime has no use", lifetime.definition);
    if (auto failure = addSite(lifetime.definition))
      return std::move(*failure);
    for (const StorageAccessSite &use : lifetime.uses) {
      if (auto failure = addSite(use))
        return std::move(*failure);
      if (lifetime.definition == use)
        continue;
      dependencies.insert({lifetime.definition, use, lifetime.object});
      graphEdges.insert({lifetime.definition, use});
    }
  }
  if (lifetimeObjects != objects)
    return broken(BrokenSchedulePlanReason::MissingLifetime,
                  "storage object and lifetime coverage differ");

  std::map<ScheduleNodeId, std::set<ScheduleNodeId>> successors;
  std::map<ScheduleNodeId, size_t> indegree;
  for (const ScheduleNodeId &node : nodes)
    indegree.try_emplace(node, 0);
  for (const auto &[predecessor, successor] : graphEdges) {
    successors[predecessor].insert(successor);
    ++indegree[successor];
  }

  std::set<ScheduleNodeId> ready;
  for (const auto &[node, degree] : indegree)
    if (degree == 0)
      ready.insert(node);

  CanonicalScheduleCoordinate result;
  while (!ready.empty()) {
    ScheduleNodeId node = *ready.begin();
    ready.erase(ready.begin());
    result.plan.order.push_back(node);
    auto next = successors.find(node);
    if (next == successors.end())
      continue;
    for (const ScheduleNodeId &successor : next->second) {
      auto degree = indegree.find(successor);
      if (degree == indegree.end() || degree->second == 0)
        return broken(BrokenSchedulePlanReason::CyclicDependency,
                      "schedule dependency accounting is inconsistent",
                      successor);
      if (--degree->second == 0)
        ready.insert(successor);
    }
  }
  if (result.plan.order.size() != nodes.size())
    return broken(BrokenSchedulePlanReason::CyclicDependency,
                  "storage dependency graph is cyclic");

  for (const ScheduleNodeId &node : result.plan.order)
    result.plan.workerBindings.push_back({node, NCCWorker::Worker0});
  result.dependencies.assign(dependencies.begin(), dependencies.end());
  return result;
}

} // namespace wafer::compiler::detail
