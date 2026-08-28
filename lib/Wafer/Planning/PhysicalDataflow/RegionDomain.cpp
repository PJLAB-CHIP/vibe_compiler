//===- RegionDomain.cpp - Region execution and use domain --------------===//

#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

bool groupOrder(const RegionGroupPlan &lhs, const RegionGroupPlan &rhs) {
  if (lhs.tile != rhs.tile)
    return lhs.tile.getValue() < rhs.tile.getValue();
  return std::lexicographical_compare(
      lhs.mandatoryRoots.begin(), lhs.mandatoryRoots.end(),
      rhs.mandatoryRoots.begin(), rhs.mandatoryRoots.end());
}

bool isCanonicalRestrictedGrowth(llvm::ArrayRef<uint32_t> labels) {
  if (labels.empty() || labels.front() != 0)
    return false;
  uint32_t maximum = 0;
  for (uint32_t label : llvm::drop_begin(labels)) {
    if (label > maximum + 1)
      return false;
    maximum = std::max(maximum, label);
  }
  return true;
}

using ConnectedGroup = llvm::SmallVector<uint32_t, 8>;
using ConnectedPartition = llvm::SmallVector<ConnectedGroup, 8>;

template <typename ComponentT>
bool isConnectedGroup(const ComponentT &component,
                      llvm::ArrayRef<uint32_t> group) {
  if (group.empty() || !llvm::is_sorted(group) ||
      std::adjacent_find(group.begin(), group.end()) != group.end() ||
      llvm::any_of(group, [&](uint32_t vertex) {
        return vertex >= component.works.size();
      }))
    return false;
  if (group.size() == 1)
    return true;
  const size_t count = component.works.size();
  if (component.potentialEdges.size() != count * count)
    return false;
  std::set<uint32_t> reached{group.front()};
  llvm::SmallVector<uint32_t, 8> worklist{group.front()};
  while (!worklist.empty()) {
    const uint32_t current = worklist.pop_back_val();
    for (uint32_t candidate : group) {
      if (reached.count(candidate) ||
          (!component.potentialEdges[current * count + candidate] &&
           !component.potentialEdges[candidate * count + current]))
        continue;
      reached.insert(candidate);
      worklist.push_back(candidate);
    }
  }
  return reached.size() == group.size();
}

template <typename ComponentT>
std::optional<ConnectedGroup>
getConnectedParent(const ComponentT &component,
                   llvm::ArrayRef<uint32_t> group) {
  if (group.size() <= 1 || !isConnectedGroup(component, group))
    return std::nullopt;
  const uint32_t anchor = group.front();
  for (uint32_t removable : llvm::reverse(group)) {
    if (removable == anchor)
      continue;
    ConnectedGroup parent;
    for (uint32_t vertex : group)
      if (vertex != removable)
        parent.push_back(vertex);
    if (isConnectedGroup(component, parent))
      return parent;
  }
  return std::nullopt;
}

template <typename ComponentT>
llvm::SmallVector<ConnectedGroup, 8>
getConnectedChildren(const ComponentT &component,
                     llvm::ArrayRef<uint32_t> group,
                     llvm::ArrayRef<uint32_t> allowed) {
  llvm::SmallVector<ConnectedGroup, 8> children;
  const size_t count = component.works.size();
  for (uint32_t candidate : allowed) {
    if (llvm::is_contained(group, candidate))
      continue;
    bool adjacent = llvm::any_of(group, [&](uint32_t member) {
      return component.potentialEdges[member * count + candidate] ||
             component.potentialEdges[candidate * count + member];
    });
    if (!adjacent)
      continue;
    ConnectedGroup child(group.begin(), group.end());
    child.push_back(candidate);
    llvm::sort(child);
    std::optional<ConnectedGroup> parent = getConnectedParent(component, child);
    if (parent && *parent == group)
      children.push_back(std::move(child));
  }
  llvm::sort(children);
  return children;
}

template <typename ComponentT>
std::optional<ConnectedGroup>
getNextConnectedGroup(const ComponentT &component,
                      llvm::ArrayRef<uint32_t> current,
                      llvm::ArrayRef<uint32_t> allowed) {
  if (current.empty() || allowed.empty() ||
      current.front() != allowed.front() ||
      !isConnectedGroup(component, current))
    return std::nullopt;
  ConnectedGroup node(current.begin(), current.end());
  llvm::SmallVector<ConnectedGroup, 8> children =
      getConnectedChildren(component, node, allowed);
  if (!children.empty())
    return children.front();
  while (node.size() > 1) {
    std::optional<ConnectedGroup> parent = getConnectedParent(component, node);
    if (!parent)
      return std::nullopt;
    llvm::SmallVector<ConnectedGroup, 8> siblings =
        getConnectedChildren(component, *parent, allowed);
    auto currentSibling = llvm::find(siblings, node);
    if (currentSibling == siblings.end())
      return std::nullopt;
    if (++currentSibling != siblings.end())
      return *currentSibling;
    node = std::move(*parent);
  }
  return std::nullopt;
}

ConnectedPartition
getFirstConnectedPartition(llvm::ArrayRef<uint32_t> allowed) {
  ConnectedPartition partition;
  for (uint32_t vertex : allowed)
    partition.push_back(ConnectedGroup{vertex});
  return partition;
}

llvm::SmallVector<uint32_t, 8> subtractGroup(llvm::ArrayRef<uint32_t> allowed,
                                             llvm::ArrayRef<uint32_t> group) {
  llvm::SmallVector<uint32_t, 8> remaining;
  for (uint32_t vertex : allowed)
    if (!llvm::is_contained(group, vertex))
      remaining.push_back(vertex);
  return remaining;
}

template <typename ComponentT>
std::optional<ConnectedPartition>
advanceConnectedPartition(const ComponentT &component,
                          llvm::ArrayRef<ConnectedGroup> current,
                          llvm::ArrayRef<uint32_t> allowed) {
  if (current.empty() || allowed.empty() ||
      current.front().front() != allowed.front() ||
      !isConnectedGroup(component, current.front()))
    return std::nullopt;
  llvm::SmallVector<uint32_t, 8> remaining =
      subtractGroup(allowed, current.front());
  if (!remaining.empty()) {
    if (current.size() < 2)
      return std::nullopt;
    std::optional<ConnectedPartition> suffix =
        advanceConnectedPartition(component, current.drop_front(), remaining);
    if (suffix) {
      ConnectedPartition result{current.front()};
      result.append(suffix->begin(), suffix->end());
      return result;
    }
  } else if (current.size() != 1) {
    return std::nullopt;
  }

  std::optional<ConnectedGroup> next =
      getNextConnectedGroup(component, current.front(), allowed);
  if (!next)
    return std::nullopt;
  ConnectedPartition result{*next};
  remaining = subtractGroup(allowed, *next);
  ConnectedPartition suffix = getFirstConnectedPartition(remaining);
  result.append(suffix.begin(), suffix.end());
  return result;
}

ConnectedPartition labelsToPartition(llvm::ArrayRef<uint32_t> labels) {
  ConnectedPartition result;
  if (labels.empty())
    return result;
  const uint32_t count = *llvm::max_element(labels) + 1;
  result.resize(count);
  for (auto [vertex, label] : llvm::enumerate(labels)) {
    if (label >= result.size())
      return {};
    result[label].push_back(static_cast<uint32_t>(vertex));
  }
  return result;
}

llvm::SmallVector<uint32_t, 8>
partitionToLabels(size_t vertexCount,
                  llvm::ArrayRef<ConnectedGroup> partition) {
  llvm::SmallVector<uint32_t, 8> labels(vertexCount,
                                        std::numeric_limits<uint32_t>::max());
  for (auto [label, group] : llvm::enumerate(partition))
    for (uint32_t vertex : group) {
      if (vertex >= labels.size() ||
          labels[vertex] != std::numeric_limits<uint32_t>::max())
        return {};
      labels[vertex] = static_cast<uint32_t>(label);
    }
  if (llvm::is_contained(labels, std::numeric_limits<uint32_t>::max()))
    return {};
  return labels;
}

template <typename ComponentT>
bool isLegalPartition(const ComponentT &component,
                      llvm::ArrayRef<uint32_t> labels) {
  const size_t count = component.works.size();
  if (labels.size() != count || !isCanonicalRestrictedGrowth(labels) ||
      component.potentialEdges.size() != count * count)
    return false;
  const uint32_t groupCount = *llvm::max_element(labels) + 1;
  for (uint32_t group = 0; group < groupCount; ++group) {
    llvm::SmallVector<size_t, 8> members;
    for (auto [index, label] : llvm::enumerate(labels))
      if (label == group)
        members.push_back(index);
    if (members.empty())
      return false;
    if (members.size() == 1)
      continue;
    std::set<size_t> reached{members.front()};
    llvm::SmallVector<size_t, 8> worklist{members.front()};
    while (!worklist.empty()) {
      size_t current = worklist.pop_back_val();
      for (size_t candidate : members)
        if (!reached.count(candidate) &&
            (component.potentialEdges[current * count + candidate] ||
             component.potentialEdges[candidate * count + current])) {
          reached.insert(candidate);
          worklist.push_back(candidate);
        }
    }
    if (reached.size() != members.size())
      return false;
  }
  return true;
}

template <typename ComponentT>
llvm::SmallVector<uint32_t, 8>
getInitialComponentLabels(const ComponentT &component) {
  llvm::SmallVector<uint32_t, 8> labels;
  for (size_t index = 0; index < component.works.size(); ++index)
    labels.push_back(static_cast<uint32_t>(index));
  return labels;
}

template <typename ComponentT>
std::optional<llvm::SmallVector<uint32_t, 8>>
getNextLabels(const ComponentT &component, llvm::ArrayRef<uint32_t> current) {
  if (!isLegalPartition(component, current))
    return std::nullopt;
  ConnectedPartition partition = labelsToPartition(current);
  llvm::SmallVector<uint32_t, 8> allowed;
  for (size_t vertex = 0; vertex < component.works.size(); ++vertex)
    allowed.push_back(static_cast<uint32_t>(vertex));
  std::optional<ConnectedPartition> next =
      advanceConnectedPartition(component, partition, allowed);
  if (!next)
    return std::nullopt;
  llvm::SmallVector<uint32_t, 8> labels =
      partitionToLabels(component.works.size(), *next);
  if (!isLegalPartition(component, labels))
    return std::nullopt;
  return labels;
}

const RegionGroupPlan *findBaseGroup(llvm::ArrayRef<RegionGroupPlan> groups,
                                     const analysis::RootRegionWorkId &work) {
  auto group = llvm::find_if(groups, [&](const RegionGroupPlan &candidate) {
    return candidate.mandatoryRoots.size() == 1 &&
           candidate.mandatoryRoots.front() == work;
  });
  return group == groups.end() ? nullptr : &*group;
}

bool isRootExecution(const ExecutionInstanceId &id) {
  return std::holds_alternative<RequiredRootExecution>(id.source);
}

} // namespace

mlir::FailureOr<RegionDomain>
RegionDomain::create(llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
                     std::string *failureReason) {
  CanonicalRegionPlanOutcome canonical = buildCanonicalRegionPlan(rootWorks);
  const RegionPlan *base = getRegionPlan(canonical);
  if (!base) {
    if (failureReason)
      *failureReason = std::get<BrokenRegionPlan>(canonical).detail;
    return mlir::failure();
  }

  std::map<LogicalShardId,
           std::pair<analysis::RootRegionWorkId, ExecutionInstanceId>>
      shardOwners;
  std::map<ReductionGroupId,
           std::pair<analysis::RootRegionWorkId, ExecutionInstanceId>>
      mergeOwners;
  std::map<LogicalShardId,
           std::pair<analysis::RootRegionWorkId, ExecutionInstanceId>>
      consumers;
  std::map<analysis::RootRegionWorkId, bool> replicaAllowedByWork;
  std::map<analysis::RootRegionWorkId, const analysis::RootRegionWork *> works;
  for (const analysis::RootRegionWork &work : rootWorks) {
    works.emplace(work.id, &work);
    replicaAllowedByWork[work.id] =
        work.rootOperation && mlir::isMemoryEffectFree(work.rootOperation);
  }
  for (const RegionGroupPlan &group : base->groups)
    for (const ExecutionInstancePlan &execution : group.executions) {
      if (const auto *root =
              std::get_if<RequiredRootExecution>(&execution.id.source)) {
        if (!shardOwners.try_emplace(root->shard, root->work, execution.id)
                 .second ||
            !consumers.try_emplace(root->shard, root->work, execution.id)
                 .second) {
          if (failureReason)
            *failureReason = "region domain has duplicate root execution";
          return mlir::failure();
        }
      } else {
        const auto &merge =
            std::get<RequiredMergeExecution>(execution.id.source);
        if (!mergeOwners.try_emplace(merge.group, merge.work, execution.id)
                 .second) {
          if (failureReason)
            *failureReason = "region domain has duplicate merge execution";
          return mlir::failure();
        }
      }
    }

  std::vector<LocalFragment> fragments;
  for (const RegionGroupPlan &consumerGroup : base->groups) {
    for (const ExternalUseBinding &external : consumerGroup.externalBindings) {
      if (external.fragment.source.kind !=
          analysis::RootBoundaryKind::StructuredResult)
        continue;
      auto consumer = consumers.find(external.fragment.use.destinationShard);
      if (consumer == consumers.end())
        continue;
      std::optional<std::pair<analysis::RootRegionWorkId, ExecutionInstanceId>>
          producer;
      if (external.fragment.ownerShard) {
        auto owner = shardOwners.find(*external.fragment.ownerShard);
        if (owner != shardOwners.end())
          producer = owner->second;
      } else if (external.fragment.reductionGroup) {
        auto owner = mergeOwners.find(*external.fragment.reductionGroup);
        if (owner != mergeOwners.end())
          producer = owner->second;
      }
      if (!producer)
        continue;
      const bool allowsRequiredLocal =
          producer->first.tile == consumer->second.first.tile;
      const analysis::RootRegionWork *producerWork = works[producer->first];
      const analysis::RootRegionWork *consumerWork =
          works[consumer->second.first];
      const bool directSSA =
          producerWork && consumerWork && producerWork->rootOperation &&
          consumerWork->rootOperation &&
          external.fragment.source.index <
              producerWork->rootOperation->getNumResults() &&
          external.fragment.use.operand <
              consumerWork->rootOperation->getNumOperands() &&
          consumerWork->rootOperation->getOperand(
              external.fragment.use.operand) ==
              producerWork->rootOperation->getResult(
                  external.fragment.source.index);
      const bool allowsReplica = isRootExecution(producer->second) &&
                                 directSSA &&
                                 replicaAllowedByWork[producer->first];
      if (!allowsRequiredLocal && !allowsReplica)
        continue;
      fragments.push_back({external.fragment, producer->first,
                           consumer->second.first, producer->second,
                           consumer->second.second, allowsRequiredLocal,
                           allowsReplica});
    }
  }
  llvm::sort(fragments, [](const LocalFragment &lhs, const LocalFragment &rhs) {
    return std::tie(lhs.fragment, lhs.producerWork, lhs.consumerWork,
                    lhs.producer) < std::tie(rhs.fragment, rhs.producerWork,
                                             rhs.consumerWork, rhs.producer);
  });
  fragments.erase(
      std::unique(fragments.begin(), fragments.end(),
                  [](const LocalFragment &lhs, const LocalFragment &rhs) {
                    return lhs.fragment == rhs.fragment &&
                           lhs.producer == rhs.producer;
                  }),
      fragments.end());

  std::map<int64_t, llvm::SmallVector<analysis::RootRegionWorkId, 8>> byTile;
  for (const RegionGroupPlan &group : base->groups)
    byTile[group.tile.getValue()].push_back(group.mandatoryRoots.front());
  llvm::SmallVector<mlir::Operation *, 128> orderedRoots;
  for (const analysis::RootRegionWork &work : rootWorks)
    if (work.rootOperation &&
        !llvm::is_contained(orderedRoots, work.rootOperation))
      orderedRoots.push_back(work.rootOperation);
  llvm::sort(orderedRoots, [](mlir::Operation *lhs, mlir::Operation *rhs) {
    return lhs != rhs && lhs->isBeforeInBlock(rhs);
  });
  llvm::DenseMap<mlir::Operation *, uint32_t> rootOrdinals;
  for (auto [ordinal, operation] : llvm::enumerate(orderedRoots))
    rootOrdinals.try_emplace(operation, static_cast<uint32_t>(ordinal));
  std::map<analysis::RootRegionWorkId, uint32_t> workOrdinals;
  for (const analysis::RootRegionWork &work : rootWorks) {
    auto ordinal = rootOrdinals.find(work.rootOperation);
    if (ordinal == rootOrdinals.end())
      return mlir::failure();
    workOrdinals.emplace(work.id, ordinal->second);
  }
  llvm::SmallVector<Component, 16> components;
  for (auto &[tileValue, works] : byTile) {
    llvm::sort(works);
    const size_t count = works.size();
    llvm::SmallVector<uint8_t, 64> edges(count * count, 0);
    for (const LocalFragment &fragment : fragments) {
      if (!fragment.allowsRequiredLocal)
        continue;
      auto producer = llvm::lower_bound(works, fragment.producerWork);
      auto consumer = llvm::lower_bound(works, fragment.consumerWork);
      if (producer == works.end() || *producer != fragment.producerWork ||
          consumer == works.end() || *consumer != fragment.consumerWork ||
          producer == consumer)
        continue;
      const size_t lhs = std::distance(works.begin(), producer);
      const size_t rhs = std::distance(works.begin(), consumer);
      edges[lhs * count + rhs] = 1;
      edges[rhs * count + lhs] = 1;
    }
    llvm::SmallVector<uint8_t, 8> assigned(count, 0);
    for (size_t start = 0; start < count; ++start) {
      if (assigned[start])
        continue;
      llvm::SmallVector<size_t, 8> indices{start};
      llvm::SmallVector<size_t, 8> worklist{start};
      assigned[start] = 1;
      while (!worklist.empty()) {
        size_t current = worklist.pop_back_val();
        for (size_t candidate = 0; candidate < count; ++candidate)
          if (!assigned[candidate] && edges[current * count + candidate]) {
            assigned[candidate] = 1;
            indices.push_back(candidate);
            worklist.push_back(candidate);
          }
      }
      llvm::sort(indices, [&](size_t lhs, size_t rhs) {
        return std::tie(workOrdinals.at(works[lhs]), works[lhs]) <
               std::tie(workOrdinals.at(works[rhs]), works[rhs]);
      });
      Component component;
      component.tile = TileId(tileValue);
      for (size_t index : indices)
        component.works.push_back(works[index]);
      const size_t componentSize = indices.size();
      component.potentialEdges.assign(componentSize * componentSize, 0);
      for (size_t lhs = 0; lhs < componentSize; ++lhs)
        for (size_t rhs = 0; rhs < componentSize; ++rhs)
          component.potentialEdges[lhs * componentSize + rhs] =
              edges[indices[lhs] * count + indices[rhs]];
      components.push_back(std::move(component));
    }
  }
  llvm::sort(components, [](const Component &lhs, const Component &rhs) {
    return std::tuple(lhs.tile.getValue(), lhs.works.front()) <
           std::tuple(rhs.tile.getValue(), rhs.works.front());
  });
  return RegionDomain(base->groups, std::move(components),
                      std::move(fragments));
}

llvm::SmallVector<llvm::SmallVector<uint32_t, 8>, 8>
RegionDomain::getFirstLabels() const {
  llvm::SmallVector<llvm::SmallVector<uint32_t, 8>, 8> labels;
  for (const Component &component : components)
    labels.push_back(getInitialComponentLabels(component));
  return labels;
}

bool RegionDomain::advanceLabels(
    llvm::SmallVectorImpl<llvm::SmallVector<uint32_t, 8>> &labels) const {
  if (labels.size() != components.size())
    return false;
  for (size_t reverse = 0; reverse < components.size(); ++reverse) {
    const size_t index = components.size() - reverse - 1;
    auto next = getNextLabels(components[index], labels[index]);
    if (!next)
      continue;
    labels[index] = std::move(*next);
    for (size_t reset = index + 1; reset < components.size(); ++reset)
      labels[reset] = getInitialComponentLabels(components[reset]);
    return true;
  }
  return false;
}

std::vector<const RegionDomain::LocalFragment *>
RegionDomain::getChoiceFragments(
    llvm::ArrayRef<llvm::SmallVector<uint32_t, 8>> labels) const {
  std::map<analysis::RootRegionWorkId, std::pair<size_t, uint32_t>> groups;
  for (auto [componentIndex, values] :
       llvm::enumerate(llvm::zip_equal(components, labels))) {
    const auto &[component, componentLabels] = values;
    for (auto [work, label] : llvm::zip_equal(component.works, componentLabels))
      groups[work] = {componentIndex, label};
  }
  std::vector<const LocalFragment *> result;
  for (const LocalFragment &fragment : localFragments) {
    auto producer = groups.find(fragment.producerWork);
    auto consumer = groups.find(fragment.consumerWork);
    const bool sameGroup = producer != groups.end() &&
                           consumer != groups.end() &&
                           producer->second == consumer->second;
    if ((sameGroup && fragment.allowsRequiredLocal) || fragment.allowsReplica)
      result.push_back(&fragment);
  }
  return result;
}

bool RegionDomain::advanceChoices(
    llvm::ArrayRef<llvm::SmallVector<uint32_t, 8>> labels,
    llvm::ArrayRef<const LocalFragment *> fragments,
    llvm::SmallVectorImpl<uint8_t> &choices) const {
  if (choices.size() != fragments.size())
    return false;
  std::map<analysis::RootRegionWorkId, std::pair<size_t, uint32_t>> groups;
  for (auto [componentIndex, values] :
       llvm::enumerate(llvm::zip_equal(components, labels))) {
    const auto &[component, componentLabels] = values;
    for (auto [work, label] : llvm::zip_equal(component.works, componentLabels))
      groups[work] = {componentIndex, label};
  }
  for (size_t reverse = 0; reverse < choices.size(); ++reverse) {
    const size_t index = choices.size() - reverse - 1;
    const LocalFragment &fragment = *fragments[index];
    auto producer = groups.find(fragment.producerWork);
    auto consumer = groups.find(fragment.consumerWork);
    const bool sameGroup = producer != groups.end() &&
                           consumer != groups.end() &&
                           producer->second == consumer->second;
    llvm::SmallVector<uint8_t, 3> allowed{0};
    if (sameGroup && fragment.allowsRequiredLocal)
      allowed.push_back(1);
    if (fragment.allowsReplica)
      allowed.push_back(2);
    auto current = llvm::find(allowed, choices[index]);
    if (current == allowed.end())
      return false;
    if (++current != allowed.end()) {
      choices[index] = *current;
      return true;
    }
    choices[index] = 0;
  }
  return false;
}

std::optional<RegionPlan>
RegionDomain::buildPlan(llvm::ArrayRef<llvm::SmallVector<uint32_t, 8>> labels,
                        llvm::ArrayRef<uint8_t> choices) const {
  if (labels.size() != components.size())
    return std::nullopt;
  std::vector<const LocalFragment *> fragments = getChoiceFragments(labels);
  if (fragments.size() != choices.size())
    return std::nullopt;

  RegionPlan plan;
  std::map<analysis::RootRegionWorkId, size_t> groupByWork;
  for (auto [component, componentLabels] :
       llvm::zip_equal(components, labels)) {
    if (!isLegalPartition(component, componentLabels))
      return std::nullopt;
    const uint32_t groupCount = *llvm::max_element(componentLabels) + 1;
    for (uint32_t label = 0; label < groupCount; ++label) {
      RegionGroupPlan group;
      group.tile = component.tile;
      for (auto [work, selected] :
           llvm::zip_equal(component.works, componentLabels)) {
        if (selected != label)
          continue;
        const RegionGroupPlan *base = findBaseGroup(baseGroups, work);
        if (!base)
          return std::nullopt;
        group.mandatoryRoots.append(base->mandatoryRoots.begin(),
                                    base->mandatoryRoots.end());
        group.executions.insert(group.executions.end(),
                                base->executions.begin(),
                                base->executions.end());
        group.externalBindings.insert(group.externalBindings.end(),
                                      base->externalBindings.begin(),
                                      base->externalBindings.end());
      }
      llvm::sort(group.mandatoryRoots);
      llvm::sort(group.executions);
      llvm::sort(group.externalBindings);
      plan.groups.push_back(std::move(group));
    }
  }
  llvm::sort(plan.groups, groupOrder);
  for (auto [groupIndex, group] : llvm::enumerate(plan.groups))
    for (const analysis::RootRegionWorkId &work : group.mandatoryRoots)
      if (!groupByWork.try_emplace(work, groupIndex).second)
        return std::nullopt;

  for (auto [fragment, choice] : llvm::zip_equal(fragments, choices)) {
    if (choice == 0)
      continue;
    auto producerGroup = groupByWork.find(fragment->producerWork);
    auto consumerGroup = groupByWork.find(fragment->consumerWork);
    if (producerGroup == groupByWork.end() ||
        consumerGroup == groupByWork.end())
      return std::nullopt;
    const bool sameGroup = producerGroup->second == consumerGroup->second;
    if (choice == 1 && (!sameGroup || !fragment->allowsRequiredLocal))
      return std::nullopt;
    if (choice == 2 && !fragment->allowsReplica)
      return std::nullopt;
    RegionGroupPlan &group = plan.groups[consumerGroup->second];
    auto external = llvm::find_if(
        group.externalBindings, [&](const ExternalUseBinding &binding) {
          return binding.fragment == fragment->fragment;
        });
    if (external == group.externalBindings.end())
      return std::nullopt;
    group.externalBindings.erase(external);

    RegionExecutionId producerId;
    if (choice == 1) {
      auto execution = llvm::find_if(
          group.executions, [&](const ExecutionInstancePlan &candidate) {
            return candidate.id == fragment->producer;
          });
      if (execution == group.executions.end())
        return std::nullopt;
      producerId = fragment->producer;
    } else {
      const auto *required =
          std::get_if<RequiredRootExecution>(&fragment->producer.source);
      if (!required || !fragment->allowsReplica)
        return std::nullopt;
      ReplicaExecutionPlan replica;
      replica.id.producer = *required;
      replica.id.fragment = fragment->fragment;
      if (llvm::is_contained(group.replicas, replica))
        return std::nullopt;
      group.replicas.push_back(replica);
      producerId = replica.id;
    }
    group.localBindings.push_back({fragment->fragment, std::move(producerId)});
  }

  for (RegionGroupPlan &group : plan.groups) {
    for (const ExternalUseBinding &binding : group.externalBindings)
      if (binding.fragment.source.kind ==
              analysis::RootBoundaryKind::StructuredResult &&
          llvm::any_of(group.mandatoryRoots,
                       [&](const analysis::RootRegionWorkId &work) {
                         return work.root == binding.fragment.source.semantic;
                       }))
        return std::nullopt;

    for (const LocalUseBinding &binding : group.localBindings)
      if (const auto *required =
              std::get_if<ExecutionInstanceId>(&binding.producer)) {
        auto execution = llvm::find_if(
            group.executions, [&](const ExecutionInstancePlan &candidate) {
              return candidate.id == *required;
            });
        if (execution == group.executions.end())
          return std::nullopt;
      }

    if (group.mandatoryRoots.size() > 1) {
      std::map<analysis::RootRegionWorkId, size_t> index;
      for (auto [position, work] : llvm::enumerate(group.mandatoryRoots))
        index[work] = position;
      std::vector<std::vector<size_t>> edges(group.mandatoryRoots.size());
      for (auto [fragment, choice] : llvm::zip_equal(fragments, choices)) {
        if (choice == 0)
          continue;
        auto lhs = index.find(fragment->producerWork);
        auto rhs = index.find(fragment->consumerWork);
        if (lhs == index.end() || rhs == index.end() || lhs == rhs)
          continue;
        edges[lhs->second].push_back(rhs->second);
        edges[rhs->second].push_back(lhs->second);
      }
      std::set<size_t> reached{0};
      llvm::SmallVector<size_t, 8> worklist{0};
      while (!worklist.empty()) {
        size_t current = worklist.pop_back_val();
        for (size_t next : edges[current])
          if (reached.insert(next).second)
            worklist.push_back(next);
      }
      if (reached.size() != group.mandatoryRoots.size())
        return std::nullopt;
    }
    llvm::sort(group.executions);
    llvm::sort(group.replicas);
    llvm::sort(group.localBindings);
    llvm::sort(group.externalBindings);
  }

  std::vector<std::set<size_t>> successors(plan.groups.size());
  std::vector<size_t> indegree(plan.groups.size(), 0);
  for (auto [consumerIndex, group] : llvm::enumerate(plan.groups)) {
    for (const ExternalUseBinding &binding : group.externalBindings) {
      if (binding.fragment.source.kind !=
          analysis::RootBoundaryKind::StructuredResult)
        continue;
      std::set<size_t> producers;
      for (const LocalFragment &fragment : localFragments) {
        if (!(fragment.fragment == binding.fragment))
          continue;
        auto producer = groupByWork.find(fragment.producerWork);
        if (producer != groupByWork.end())
          producers.insert(producer->second);
      }
      if (producers.empty())
        for (auto [producerIndex, candidate] : llvm::enumerate(plan.groups)) {
          if (binding.fragment.ownerTile &&
              candidate.tile != *binding.fragment.ownerTile)
            continue;
          if (llvm::any_of(candidate.mandatoryRoots,
                           [&](const analysis::RootRegionWorkId &work) {
                             return work.root ==
                                    binding.fragment.source.semantic;
                           }))
            producers.insert(producerIndex);
        }
      for (size_t producerIndex : producers) {
        if (producerIndex == consumerIndex)
          return std::nullopt;
        if (successors[producerIndex].insert(consumerIndex).second)
          ++indegree[consumerIndex];
      }
    }
  }
  std::set<size_t> ready;
  for (auto [index, degree] : llvm::enumerate(indegree))
    if (degree == 0)
      ready.insert(index);
  size_t visited = 0;
  while (!ready.empty()) {
    size_t current = *ready.begin();
    ready.erase(ready.begin());
    ++visited;
    for (size_t successor : successors[current])
      if (--indegree[successor] == 0)
        ready.insert(successor);
  }
  if (visited != plan.groups.size())
    return std::nullopt;
  return plan;
}

RegionSuccessor RegionDomain::findPlan(RegionCursor cursor,
                                       bool advanceCurrent) const {
  while (true) {
    std::vector<const LocalFragment *> fragments =
        getChoiceFragments(cursor.labels);
    if (cursor.fragmentChoices.size() != fragments.size())
      cursor.fragmentChoices.assign(fragments.size(), 0);
    if (advanceCurrent) {
      if (!advanceChoices(cursor.labels, fragments, cursor.fragmentChoices)) {
        if (!advanceLabels(cursor.labels))
          return {RegionSuccessorKind::End};
        fragments = getChoiceFragments(cursor.labels);
        cursor.fragmentChoices.assign(fragments.size(), 0);
      }
    }
    if (std::optional<RegionPlan> plan =
            buildPlan(cursor.labels, cursor.fragmentChoices))
      return {RegionSuccessorKind::Plan, std::move(plan), std::move(cursor)};
    advanceCurrent = true;
  }
}

RegionSuccessor RegionDomain::getFirstPlan() const {
  RegionCursor cursor;
  cursor.labels = getFirstLabels();
  cursor.fragmentChoices.assign(getChoiceFragments(cursor.labels).size(), 0);
  return findPlan(std::move(cursor), /*advanceCurrent=*/false);
}

RegionSuccessor RegionDomain::getNextPlan(const RegionCursor &cursor) const {
  return findPlan(cursor, /*advanceCurrent=*/true);
}

std::optional<RegionCursor>
RegionDomain::getCursor(const RegionPlan &plan) const {
  if (!llvm::is_sorted(plan.groups, groupOrder))
    return std::nullopt;
  RegionCursor cursor;
  cursor.labels.resize(components.size());
  for (auto [componentIndex, component] : llvm::enumerate(components)) {
    std::map<size_t, uint32_t> labelsByGroup;
    for (const analysis::RootRegionWorkId &work : component.works) {
      auto group =
          llvm::find_if(plan.groups, [&](const RegionGroupPlan &entry) {
            return llvm::is_contained(entry.mandatoryRoots, work);
          });
      if (group == plan.groups.end() || group->tile != component.tile)
        return std::nullopt;
      size_t groupIndex = std::distance(plan.groups.begin(), group);
      auto [label, inserted] = labelsByGroup.try_emplace(
          groupIndex, static_cast<uint32_t>(labelsByGroup.size()));
      (void)inserted;
      cursor.labels[componentIndex].push_back(label->second);
    }
    if (!isLegalPartition(component, cursor.labels[componentIndex]))
      return std::nullopt;
  }
  std::vector<const LocalFragment *> fragments =
      getChoiceFragments(cursor.labels);
  for (const LocalFragment *fragment : fragments) {
    auto group = llvm::find_if(plan.groups, [&](const RegionGroupPlan &entry) {
      return llvm::is_contained(entry.mandatoryRoots, fragment->consumerWork);
    });
    if (group == plan.groups.end())
      return std::nullopt;
    unsigned externalCount = llvm::count_if(
        group->externalBindings, [&](const ExternalUseBinding &binding) {
          return binding.fragment == fragment->fragment;
        });
    auto local = llvm::find_if(group->localBindings,
                               [&](const LocalUseBinding &binding) {
                                 return binding.fragment == fragment->fragment;
                               });
    if (externalCount == 1 && local == group->localBindings.end()) {
      cursor.fragmentChoices.push_back(0);
      continue;
    }
    if (externalCount != 0 || local == group->localBindings.end())
      return std::nullopt;
    if (const auto *required =
            std::get_if<ExecutionInstanceId>(&local->producer)) {
      if (!(*required == fragment->producer))
        return std::nullopt;
      cursor.fragmentChoices.push_back(1);
    } else {
      const auto &replica = std::get<ReplicaExecutionId>(local->producer);
      const auto *producerRoot =
          std::get_if<RequiredRootExecution>(&fragment->producer.source);
      if (!producerRoot || !(replica.producer == *producerRoot) ||
          !(replica.fragment == fragment->fragment))
        return std::nullopt;
      cursor.fragmentChoices.push_back(2);
    }
  }
  std::optional<RegionPlan> rebuilt =
      buildPlan(cursor.labels, cursor.fragmentChoices);
  if (!rebuilt || !(*rebuilt == plan))
    return std::nullopt;
  return cursor;
}

bool RegionDomain::contains(const RegionPlan &plan) const {
  return getCursor(plan).has_value();
}

std::vector<RegionPlan> RegionDomain::getProposals() const {
  std::vector<RegionPlan> proposals;
  std::set<RegionPlan> seen;
  auto append = [&](std::optional<RegionPlan> plan) {
    if (!plan || !contains(*plan) || !seen.insert(*plan).second)
      return;
    proposals.push_back(std::move(*plan));
  };

  llvm::SmallVector<llvm::SmallVector<uint32_t, 8>, 8> singleton =
      getFirstLabels();

  llvm::SmallVector<llvm::SmallVector<uint32_t, 8>, 8> maximal;
  for (const Component &component : components)
    maximal.push_back(
        llvm::SmallVector<uint32_t, 8>(component.works.size(), 0));

  auto makeChoices = [&](llvm::ArrayRef<llvm::SmallVector<uint32_t, 8>> labels,
                         uint8_t requested) {
    std::map<analysis::RootRegionWorkId, std::pair<size_t, uint32_t>> groups;
    for (auto [componentIndex, values] :
         llvm::enumerate(llvm::zip_equal(components, labels))) {
      const auto &[component, componentLabels] = values;
      for (auto [work, label] :
           llvm::zip_equal(component.works, componentLabels))
        groups[work] = {componentIndex, label};
    }
    std::vector<const LocalFragment *> fragments = getChoiceFragments(labels);
    llvm::SmallVector<uint8_t, 16> choices;
    choices.reserve(fragments.size());
    for (const LocalFragment *fragment : fragments) {
      auto producer = groups.find(fragment->producerWork);
      auto consumer = groups.find(fragment->consumerWork);
      const bool sameGroup = producer != groups.end() &&
                             consumer != groups.end() &&
                             producer->second == consumer->second;
      uint8_t choice = 0;
      if (requested == 1 && sameGroup && fragment->allowsRequiredLocal)
        choice = 1;
      else if (requested == 2 && fragment->allowsReplica)
        choice = 2;
      choices.push_back(choice);
    }
    return choices;
  };

  // Grow connected groups from the singleton point. Each tentative union is
  // accepted only when it does not place an unlocalizable structured boundary
  // inside the group. Connectivity is preserved because unions follow a
  // potential local edge; the complete RegionPlan is constructed and checked
  // once after the incremental partition is closed.
  llvm::SmallVector<llvm::SmallVector<uint32_t, 8>, 8> coherent = singleton;
  for (auto [componentIndex, component] : llvm::enumerate(components)) {
    const size_t count = component.works.size();
    std::map<DemandFragmentId, size_t> localRealizations;
    for (const LocalFragment &fragment : localFragments)
      if (fragment.allowsRequiredLocal)
        ++localRealizations[fragment.fragment];
    llvm::SmallVector<std::pair<size_t, size_t>, 8> cannotLink;
    llvm::SmallVector<std::pair<size_t, size_t>, 32> directedEdges;
    auto findWorkIndex =
        [&](const analysis::RootRegionWorkId &work) -> std::optional<size_t> {
      auto found = llvm::find(component.works, work);
      return found == component.works.end()
                 ? std::nullopt
                 : std::optional<size_t>(
                       std::distance(component.works.begin(), found));
    };
    for (const LocalFragment &fragment : localFragments) {
      std::optional<size_t> producer = findWorkIndex(fragment.producerWork);
      std::optional<size_t> consumer = findWorkIndex(fragment.consumerWork);
      if (producer && consumer && *producer != *consumer &&
          !llvm::is_contained(directedEdges,
                              std::make_pair(*producer, *consumer)))
        directedEdges.emplace_back(*producer, *consumer);
    }
    for (const RegionGroupPlan &base : baseGroups) {
      if (base.tile != component.tile || base.mandatoryRoots.size() != 1)
        continue;
      auto consumer = llvm::find(component.works, base.mandatoryRoots.front());
      if (consumer == component.works.end())
        continue;
      for (const ExternalUseBinding &binding : base.externalBindings) {
        if (binding.fragment.source.kind !=
                analysis::RootBoundaryKind::StructuredResult ||
            localRealizations[binding.fragment] == 1)
          continue;
        auto producer = llvm::find_if(
            component.works, [&](const analysis::RootRegionWorkId &work) {
              return work.root == binding.fragment.source.semantic;
            });
        if (producer == component.works.end() || producer == consumer)
          continue;
        size_t lhs = std::distance(component.works.begin(), producer);
        size_t rhs = std::distance(component.works.begin(), consumer);
        if (rhs < lhs)
          std::swap(lhs, rhs);
        if (!llvm::is_contained(cannotLink, std::make_pair(lhs, rhs)))
          cannotLink.emplace_back(lhs, rhs);
        size_t producerIndex = std::distance(component.works.begin(), producer);
        size_t consumerIndex = std::distance(component.works.begin(), consumer);
        if (!llvm::is_contained(directedEdges,
                                std::make_pair(producerIndex, consumerIndex)))
          directedEdges.emplace_back(producerIndex, consumerIndex);
      }
    }
    auto isAcyclic = [&](llvm::ArrayRef<uint32_t> labels) {
      const uint32_t groupCount = *llvm::max_element(labels) + 1;
      std::vector<std::set<uint32_t>> successors(groupCount);
      std::vector<uint32_t> indegree(groupCount, 0);
      for (const auto &[producer, consumer] : directedEdges) {
        uint32_t source = labels[producer];
        uint32_t destination = labels[consumer];
        if (source != destination &&
            successors[source].insert(destination).second)
          ++indegree[destination];
      }
      std::set<uint32_t> ready;
      for (auto [group, degree] : llvm::enumerate(indegree))
        if (degree == 0)
          ready.insert(static_cast<uint32_t>(group));
      uint32_t visited = 0;
      while (!ready.empty()) {
        uint32_t current = *ready.begin();
        ready.erase(ready.begin());
        ++visited;
        for (uint32_t successor : successors[current])
          if (--indegree[successor] == 0)
            ready.insert(successor);
      }
      return visited == groupCount;
    };
    for (size_t lhs = 0; lhs < count; ++lhs) {
      for (size_t rhs = lhs + 1; rhs < count; ++rhs) {
        if (!component.potentialEdges[lhs * count + rhs] ||
            coherent[componentIndex][lhs] == coherent[componentIndex][rhs])
          continue;
        auto trial = coherent;
        const uint32_t kept = trial[componentIndex][lhs];
        const uint32_t removed = trial[componentIndex][rhs];
        for (uint32_t &label : trial[componentIndex])
          if (label == removed)
            label = kept;
        std::map<uint32_t, uint32_t> normalized;
        for (uint32_t &label : trial[componentIndex]) {
          auto [entry, inserted] = normalized.try_emplace(
              label, static_cast<uint32_t>(normalized.size()));
          (void)inserted;
          label = entry->second;
        }
        if (llvm::none_of(cannotLink,
                          [&](const auto &edge) {
                            return trial[componentIndex][edge.first] ==
                                   trial[componentIndex][edge.second];
                          }) &&
            isAcyclic(trial[componentIndex]))
          coherent = std::move(trial);
      }
    }
  }

  append(buildPlan(coherent, makeChoices(coherent, /*local once=*/1)));

  append(buildPlan(maximal, makeChoices(maximal, /*local once=*/1)));
  append(buildPlan(singleton, llvm::SmallVector<uint8_t, 16>(
                                  getChoiceFragments(singleton).size(), 0)));
  append(buildPlan(singleton, makeChoices(singleton, /*explicit replicas=*/2)));
  return proposals;
}

} // namespace wafer::compiler::detail
