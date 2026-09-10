//===- RegionDomain.cpp - Region execution and use domain --------------===//

#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"

#include "mlir/IR/BuiltinTypes.h"
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

std::optional<uint64_t> getLogicalElementByteWidth(mlir::Type type) {
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    type = shaped.getElementType();
  unsigned bitWidth = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    bitWidth = integer.getWidth();
  else if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    bitWidth = floating.getWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;
  return bitWidth / 8;
}

std::optional<uint64_t>
getExactLogicalBytes(const analysis::RootRegionWork &consumerWork,
                     const DemandFragmentId &fragment) {
  auto boundary = llvm::find_if(consumerWork.boundaries,
                                [&](const analysis::RootBoundaryWork &work) {
                                  return work.id == fragment.source;
                                });
  if (boundary == consumerWork.boundaries.end() || !boundary->sourceValue)
    return std::nullopt;
  auto use = llvm::find_if(boundary->consumerUses,
                           [&](const analysis::RootBoundaryUseWork &work) {
                             return work.id == fragment.use;
                           });
  if (use == boundary->consumerUses.end() || !use->requiredDomain)
    return std::nullopt;
  const analysis::ExactIndexSet &domain = *use->requiredDomain;
  if (domain.isEmpty())
    return uint64_t{0};
  if (domain.getForm() != analysis::ExactIndexSetForm::BoxUnion ||
      domain.getBoxes().empty())
    return std::nullopt;
  std::optional<uint64_t> elementBytes =
      getLogicalElementByteWidth(boundary->sourceValue.getType());
  if (!elementBytes)
    return std::nullopt;
  uint64_t elements = 0;
  for (const analysis::StaticRectangularIndexSet &box : domain.getBoxes()) {
    uint64_t volume = 1;
    for (int64_t size : box.sizes) {
      if (size < 0 ||
          (size != 0 && volume > std::numeric_limits<uint64_t>::max() /
                                     static_cast<uint64_t>(size)))
        return std::nullopt;
      volume *= static_cast<uint64_t>(size);
    }
    if (volume > std::numeric_limits<uint64_t>::max() - elements)
      return std::nullopt;
    elements += volume;
  }
  if (*elementBytes != 0 &&
      elements > std::numeric_limits<uint64_t>::max() / *elementBytes)
    return std::nullopt;
  return elements * *elementBytes;
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
  std::map<analysis::DemandDestination,
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
        consumers.emplace(merge.group,
                          std::make_pair(merge.work, execution.id));
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
      auto consumer = consumers.find(external.fragment.use.destination);
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
      std::optional<uint64_t> exactLogicalBytes;
      if (allowsRequiredLocal && consumerWork)
        exactLogicalBytes =
            getExactLogicalBytes(*consumerWork, external.fragment);
      fragments.push_back({external.fragment, producer->first,
                           consumer->second.first, producer->second,
                           consumer->second.second, allowsRequiredLocal,
                           allowsReplica, exactLogicalBytes});
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

RegionProposalMetrics
RegionDomain::getProposalMetrics(const RegionPlan &plan) const {
  RegionProposalMetrics metrics;
  metrics.regions = plan.groups.size();
  for (const RegionGroupPlan &group : plan.groups) {
    metrics.maximumRootsPerRegion = std::max<uint64_t>(
        metrics.maximumRootsPerRegion, group.mandatoryRoots.size());
    metrics.fusionMerges += group.mandatoryRoots.size() - 1;
    metrics.localBindings += group.localBindings.size();
    metrics.externalBindings += group.externalBindings.size();
    for (const LocalUseBinding &binding : group.localBindings) {
      const LocalFragment *fragment = nullptr;
      if (const auto *required =
              std::get_if<ExecutionInstanceId>(&binding.producer)) {
        auto found = llvm::find_if(localFragments, [&](const LocalFragment &f) {
          return f.fragment == binding.fragment && f.producer == *required;
        });
        if (found != localFragments.end())
          fragment = &*found;
      }
      if (!fragment || !fragment->exactLogicalBytes ||
          *fragment->exactLogicalBytes > std::numeric_limits<uint64_t>::max() -
                                             metrics.exactLogicalBytes) {
        metrics.exactLogicalBytesKnown = false;
        metrics.exactLogicalBytes = 0;
        continue;
      }
      if (metrics.exactLogicalBytesKnown)
        metrics.exactLogicalBytes += *fragment->exactLogicalBytes;
    }
  }
  return metrics;
}

std::vector<RegionPlan>
RegionDomain::getProposals(uint64_t maximumPlans) const {
  return buildRefinedProposals(maximumPlans, nullptr);
}

std::vector<RegionPlan>
RegionDomain::getRefinementProposals(const RegionPlan &plan,
                                     uint64_t maximumPlans) const {
  if (!contains(plan))
    return {};
  return buildRefinedProposals(maximumPlans, &plan);
}

std::vector<RegionPlan> RegionDomain::buildRefinedProposals(
    uint64_t maximumPlans, const RegionPlan *neighborhoodCenter) const {
  std::vector<RegionPlan> proposals;
  if (maximumPlans == 0)
    return proposals;
  std::set<RegionPlan> seen;
  auto append = [&](std::optional<RegionPlan> plan) {
    if (!plan || !contains(*plan) || !seen.insert(*plan).second)
      return;
    proposals.push_back(std::move(*plan));
  };
  auto retainStructuralPareto = [&](std::vector<RegionPlan> plans) {
    if (plans.size() <= 2)
      return plans;
    std::vector<RegionProposalMetrics> metrics;
    metrics.reserve(plans.size());
    for (const RegionPlan &plan : plans)
      metrics.push_back(getProposalMetrics(plan));
    std::vector<RegionPlan> retained;
    retained.reserve(plans.size());
    for (size_t candidate = 0; candidate < plans.size(); ++candidate) {
      const bool anchor = candidate == 0 || candidate + 1 == plans.size();
      bool dominated = false;
      if (!anchor)
        for (size_t other = 0; other < plans.size(); ++other) {
          if (other == candidate ||
              metrics[other].regions != metrics[candidate].regions ||
              metrics[other].exactLogicalBytesKnown !=
                  metrics[candidate].exactLogicalBytesKnown)
            continue;
          const bool noWorse = metrics[other].localBindings >=
                                   metrics[candidate].localBindings &&
                               metrics[other].externalBindings <=
                                   metrics[candidate].externalBindings &&
                               (!metrics[candidate].exactLogicalBytesKnown ||
                                metrics[other].exactLogicalBytes >=
                                    metrics[candidate].exactLogicalBytes);
          const bool strict =
              metrics[other].localBindings > metrics[candidate].localBindings ||
              metrics[other].externalBindings <
                  metrics[candidate].externalBindings ||
              (metrics[candidate].exactLogicalBytesKnown &&
               metrics[other].exactLogicalBytes >
                   metrics[candidate].exactLogicalBytes);
          if (noWorse && strict) {
            dominated = true;
            break;
          }
        }
      if (!dominated)
        retained.push_back(std::move(plans[candidate]));
    }
    return retained;
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

  if (!neighborhoodCenter) {
    append(buildPlan(singleton, llvm::SmallVector<uint8_t, 16>(
                                    getChoiceFragments(singleton).size(), 0)));
    if (proposals.size() >= maximumPlans)
      return proposals;
  }

  // Build one deterministic maximum-gain sequence from singleton to a
  // graph-coherent fixed point. A group participates at most once in each
  // greedy matching round, so shallow prefixes cover independent edges before
  // later rounds grow those groups again. This is proposal ordering, not a
  // group-size or legality rule. Only a bounded number of prefixes become
  // RegionPlans; the union history is query-local work destroyed before any
  // candidate IR exists. The canonical raw successor remains complete.
  struct ProgressiveComponentState {
    struct FragmentEdge {
      size_t producer = 0;
      size_t consumer = 0;
      const LocalFragment *fragment = nullptr;
    };
    llvm::SmallVector<std::pair<size_t, size_t>, 8> cannotLink;
    llvm::SmallVector<std::pair<size_t, size_t>, 32> directedEdges;
    llvm::SmallVector<FragmentEdge, 32> fragmentEdges;
  };
  std::map<DemandFragmentId, size_t> localRealizations;
  for (const LocalFragment &fragment : localFragments)
    if (fragment.allowsRequiredLocal)
      ++localRealizations[fragment.fragment];
  std::vector<ProgressiveComponentState> progressive(components.size());
  for (auto [componentIndex, component] : llvm::enumerate(components)) {
    ProgressiveComponentState &state = progressive[componentIndex];
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
      if (!producer || !consumer || *producer == *consumer)
        continue;
      if (!llvm::is_contained(state.directedEdges,
                              std::make_pair(*producer, *consumer)))
        state.directedEdges.emplace_back(*producer, *consumer);
      if (fragment.allowsRequiredLocal)
        state.fragmentEdges.push_back({*producer, *consumer, &fragment});
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
        if (!llvm::is_contained(state.cannotLink, std::make_pair(lhs, rhs)))
          state.cannotLink.emplace_back(lhs, rhs);
        const size_t producerIndex =
            std::distance(component.works.begin(), producer);
        const size_t consumerIndex =
            std::distance(component.works.begin(), consumer);
        if (!llvm::is_contained(state.directedEdges,
                                std::make_pair(producerIndex, consumerIndex)))
          state.directedEdges.emplace_back(producerIndex, consumerIndex);
      }
    }
  }

  auto isAcyclic = [&](const ProgressiveComponentState &state,
                       llvm::ArrayRef<uint32_t> labels) {
    const uint32_t groupCount = *llvm::max_element(labels) + 1;
    std::vector<std::set<uint32_t>> successors(groupCount);
    std::vector<uint32_t> indegree(groupCount, 0);
    for (const auto &[producer, consumer] : state.directedEdges) {
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

  struct MergeStep {
    size_t component = 0;
    size_t lhs = 0;
    size_t rhs = 0;
    bool exactBytesKnown = false;
    uint64_t exactBytes = 0;
    uint64_t bindingCount = 0;
  };
  using ProposalLabels = llvm::SmallVector<llvm::SmallVector<uint32_t, 8>, 8>;
  auto normalizeLabels = [](llvm::SmallVectorImpl<uint32_t> &labels) {
    std::map<uint32_t, uint32_t> normalized;
    for (uint32_t &label : labels) {
      auto [entry, inserted] = normalized.try_emplace(
          label, static_cast<uint32_t>(normalized.size()));
      (void)inserted;
      label = entry->second;
    }
  };
  auto mergeLabels = [](llvm::SmallVectorImpl<uint32_t> &labels, size_t lhs,
                        size_t rhs) {
    const uint32_t kept = labels[lhs];
    const uint32_t removed = labels[rhs];
    for (uint32_t &label : labels)
      if (label == removed)
        label = kept;
    std::map<uint32_t, uint32_t> normalized;
    for (uint32_t &label : labels) {
      auto [entry, inserted] = normalized.try_emplace(
          label, static_cast<uint32_t>(normalized.size()));
      (void)inserted;
      label = entry->second;
    }
  };

  struct PartitionScore {
    unsigned __int128 knownExactBytes = 0;
    uint64_t localBindings = 0;
    uint64_t unknownBindings = 0;
  };
  auto getComponentScore = [&](size_t componentIndex,
                               llvm::ArrayRef<uint32_t> labels) {
    PartitionScore score;
    std::map<DemandFragmentId, std::optional<uint64_t>> localized;
    for (const ProgressiveComponentState::FragmentEdge &edge :
         progressive[componentIndex].fragmentEdges)
      if (labels[edge.producer] == labels[edge.consumer])
        localized.try_emplace(edge.fragment->fragment,
                              edge.fragment->exactLogicalBytes);
    score.localBindings = localized.size();
    for (const auto &[fragment, bytes] : localized) {
      (void)fragment;
      if (bytes)
        score.knownExactBytes += *bytes;
      else
        ++score.unknownBindings;
    }
    return score;
  };
  auto isBetterScore = [](const PartitionScore &lhs,
                          const PartitionScore &rhs) {
    if (lhs.knownExactBytes != rhs.knownExactBytes)
      return lhs.knownExactBytes > rhs.knownExactBytes;
    if (lhs.localBindings != rhs.localBindings)
      return lhs.localBindings > rhs.localBindings;
    return lhs.unknownBindings > rhs.unknownBindings;
  };
  auto refineComponent = [&](size_t componentIndex,
                             llvm::ArrayRef<uint32_t> seed) {
    const Component &component = components[componentIndex];
    const ProgressiveComponentState &state = progressive[componentIndex];
    llvm::SmallVector<uint32_t, 8> current(seed.begin(), seed.end());
    llvm::SmallVector<uint32_t, 8> best = current;
    PartitionScore bestScore = getComponentScore(componentIndex, best);
    llvm::SmallVector<uint8_t, 8> locked(component.works.size(), 0);
    for (size_t step = 0; step < component.works.size(); ++step) {
      struct MoveCandidate {
        size_t vertex = 0;
        analysis::RootRegionWorkId destinationRepresentative;
        llvm::SmallVector<uint32_t, 8> labels;
        PartitionScore score;
      };
      std::optional<MoveCandidate> selected;
      const size_t count = component.works.size();
      for (size_t vertex = 0; vertex < count; ++vertex) {
        if (locked[vertex])
          continue;
        const uint32_t sourceLabel = current[vertex];
        if (std::count(current.begin(), current.end(), sourceLabel) <= 1)
          continue;
        std::set<uint32_t> destinationLabels;
        for (size_t adjacent = 0; adjacent < count; ++adjacent)
          if (current[adjacent] != sourceLabel &&
              (component.potentialEdges[vertex * count + adjacent] ||
               component.potentialEdges[adjacent * count + vertex]))
            destinationLabels.insert(current[adjacent]);
        for (uint32_t destinationLabel : destinationLabels) {
          llvm::SmallVector<uint32_t, 8> trial = current;
          trial[vertex] = destinationLabel;
          normalizeLabels(trial);
          if (!isLegalPartition(component, trial) ||
              llvm::any_of(state.cannotLink,
                           [&](const auto &edge) {
                             return trial[edge.first] == trial[edge.second];
                           }) ||
              !isAcyclic(state, trial))
            continue;
          auto representative = llvm::find(current, destinationLabel);
          assert(representative != current.end());
          PartitionScore score = getComponentScore(componentIndex, trial);
          MoveCandidate candidate{
              vertex,
              component.works[std::distance(current.begin(), representative)],
              std::move(trial), score};
          bool replace =
              !selected || isBetterScore(candidate.score, selected->score);
          if (selected && !isBetterScore(candidate.score, selected->score) &&
              !isBetterScore(selected->score, candidate.score))
            replace = std::tie(component.works[candidate.vertex],
                               candidate.destinationRepresentative) <
                      std::tie(component.works[selected->vertex],
                               selected->destinationRepresentative);
          if (replace)
            selected = std::move(candidate);
        }
      }
      if (!selected)
        break;
      current = std::move(selected->labels);
      locked[selected->vertex] = 1;
      if (isBetterScore(selected->score, bestScore)) {
        best = current;
        bestScore = selected->score;
      }
    }
    return best;
  };
  auto refineLabels = [&](ProposalLabels labels) {
    for (size_t componentIndex = 0; componentIndex < labels.size();
         ++componentIndex)
      labels[componentIndex] =
          refineComponent(componentIndex, labels[componentIndex]);
    return labels;
  };
  auto getGlobalScore = [&](const ProposalLabels &labels) {
    PartitionScore score;
    for (size_t componentIndex = 0; componentIndex < labels.size();
         ++componentIndex) {
      PartitionScore componentScore =
          getComponentScore(componentIndex, labels[componentIndex]);
      score.knownExactBytes += componentScore.knownExactBytes;
      score.localBindings += componentScore.localBindings;
      score.unknownBindings += componentScore.unknownBindings;
    }
    return score;
  };
  auto isLegalComponentLabels = [&](size_t componentIndex,
                                    llvm::ArrayRef<uint32_t> labels) {
    const ProgressiveComponentState &state = progressive[componentIndex];
    return isLegalPartition(components[componentIndex], labels) &&
           llvm::none_of(state.cannotLink,
                         [&](const auto &edge) {
                           return labels[edge.first] == labels[edge.second];
                         }) &&
           isAcyclic(state, labels);
  };

  if (neighborhoodCenter) {
    std::optional<RegionCursor> centerCursor = getCursor(*neighborhoodCenter);
    if (!centerCursor)
      return proposals;
    const ProposalLabels center = centerCursor->labels;
    struct NeighborCandidate {
      ProposalLabels labels;
      PartitionScore score;
      uint64_t regions = 0;
    };
    std::vector<NeighborCandidate> candidates;
    auto isBetterNeighbor = [&](const NeighborCandidate &lhs,
                                const NeighborCandidate &rhs) {
      if (isBetterScore(lhs.score, rhs.score))
        return true;
      if (isBetterScore(rhs.score, lhs.score))
        return false;
      if (lhs.regions != rhs.regions)
        return lhs.regions < rhs.regions;
      return lhs.labels < rhs.labels;
    };
    auto addCandidate = [&](ProposalLabels labels) {
      if (labels == center ||
          llvm::any_of(candidates, [&](const NeighborCandidate &candidate) {
            return candidate.labels == labels;
          }))
        return;
      PartitionScore score = getGlobalScore(labels);
      uint64_t regions = 0;
      for (const auto &componentLabels : labels)
        regions += *llvm::max_element(componentLabels) + 1;
      candidates.push_back({std::move(labels), score, regions});
      llvm::sort(candidates, isBetterNeighbor);
      if (candidates.size() > maximumPlans)
        candidates.resize(maximumPlans);
    };

    for (size_t componentIndex = 0; componentIndex < components.size();
         ++componentIndex) {
      const Component &component = components[componentIndex];
      const ProgressiveComponentState &state = progressive[componentIndex];
      const llvm::SmallVector<uint32_t, 8> &current = center[componentIndex];
      const size_t count = component.works.size();
      for (size_t vertex = 0; vertex < count; ++vertex) {
        const uint32_t sourceLabel = current[vertex];
        if (std::count(current.begin(), current.end(), sourceLabel) <= 1)
          continue;
        std::set<uint32_t> destinations;
        for (size_t adjacent = 0; adjacent < count; ++adjacent)
          if (current[adjacent] != sourceLabel &&
              (component.potentialEdges[vertex * count + adjacent] ||
               component.potentialEdges[adjacent * count + vertex]))
            destinations.insert(current[adjacent]);
        for (uint32_t destination : destinations) {
          ProposalLabels trial = center;
          trial[componentIndex][vertex] = destination;
          normalizeLabels(trial[componentIndex]);
          if (isLegalComponentLabels(componentIndex, trial[componentIndex]))
            addCandidate(std::move(trial));
        }
      }

      std::set<std::pair<uint32_t, uint32_t>> groupPairs;
      for (const ProgressiveComponentState::FragmentEdge &edge :
           state.fragmentEdges) {
        uint32_t lhs = current[edge.producer];
        uint32_t rhs = current[edge.consumer];
        if (lhs == rhs)
          continue;
        if (rhs < lhs)
          std::swap(lhs, rhs);
        groupPairs.emplace(lhs, rhs);
      }
      for (const auto &[lhsLabel, rhsLabel] : groupPairs) {
        auto lhs = llvm::find(current, lhsLabel);
        auto rhs = llvm::find(current, rhsLabel);
        assert(lhs != current.end() && rhs != current.end());
        ProposalLabels trial = center;
        mergeLabels(trial[componentIndex], std::distance(current.begin(), lhs),
                    std::distance(current.begin(), rhs));
        if (!isLegalComponentLabels(componentIndex, trial[componentIndex]))
          continue;
        addCandidate(std::move(trial));
      }
    }
    for (const NeighborCandidate &candidate : candidates) {
      append(buildPlan(candidate.labels,
                       makeChoices(candidate.labels, /*local once=*/1)));
      if (proposals.size() >= maximumPlans)
        break;
    }
    return proposals;
  }

  struct MergeCandidate {
    MergeStep step;
    analysis::RootRegionWorkId lhsRepresentative;
    analysis::RootRegionWorkId rhsRepresentative;
  };
  auto isBetterCandidate = [](const MergeCandidate &lhs,
                              const MergeCandidate &rhs) {
    if (lhs.step.exactBytesKnown != rhs.step.exactBytesKnown)
      return lhs.step.exactBytesKnown;
    if (lhs.step.exactBytesKnown && lhs.step.exactBytes != rhs.step.exactBytes)
      return lhs.step.exactBytes > rhs.step.exactBytes;
    if (lhs.step.bindingCount != rhs.step.bindingCount)
      return lhs.step.bindingCount > rhs.step.bindingCount;
    return std::tie(lhs.lhsRepresentative.root, lhs.rhsRepresentative.root,
                    lhs.step.component, lhs.step.lhs, lhs.step.rhs) <
           std::tie(rhs.lhsRepresentative.root, rhs.rhsRepresentative.root,
                    rhs.step.component, rhs.step.lhs, rhs.step.rhs);
  };

  ProposalLabels coherent = singleton;
  std::vector<MergeStep> mergeHistory;
  while (true) {
    std::vector<llvm::SmallVector<uint8_t, 8>> matched;
    matched.reserve(components.size());
    for (const Component &component : components)
      matched.emplace_back(component.works.size(), 0);
    bool roundChanged = false;
    while (true) {
      std::optional<MergeCandidate> best;
      for (auto [componentIndex, state] : llvm::enumerate(progressive)) {
        using LabelPair = std::pair<uint32_t, uint32_t>;
        struct GainAccumulator {
          std::map<DemandFragmentId, std::optional<uint64_t>> fragments;
          size_t lhs = 0;
          size_t rhs = 0;
        };
        std::map<LabelPair, GainAccumulator> candidates;
        llvm::ArrayRef<uint32_t> labels = coherent[componentIndex];
        auto groupMatched = [&](uint32_t label) {
          for (auto [vertex, current] : llvm::enumerate(labels))
            if (current == label && matched[componentIndex][vertex])
              return true;
          return false;
        };
        for (const ProgressiveComponentState::FragmentEdge &edge :
             state.fragmentEdges) {
          uint32_t lhsLabel = labels[edge.producer];
          uint32_t rhsLabel = labels[edge.consumer];
          if (lhsLabel == rhsLabel || groupMatched(lhsLabel) ||
              groupMatched(rhsLabel))
            continue;
          if (rhsLabel < lhsLabel)
            std::swap(lhsLabel, rhsLabel);
          GainAccumulator &candidate = candidates[{lhsLabel, rhsLabel}];
          if (candidate.fragments.empty()) {
            candidate.lhs = edge.producer;
            candidate.rhs = edge.consumer;
          }
          candidate.fragments.try_emplace(edge.fragment->fragment,
                                          edge.fragment->exactLogicalBytes);
        }

        for (const auto &[groupPair, gain] : candidates) {
          (void)groupPair;
          llvm::SmallVector<uint32_t, 8> trial = coherent[componentIndex];
          mergeLabels(trial, gain.lhs, gain.rhs);
          if (llvm::any_of(state.cannotLink,
                           [&](const auto &edge) {
                             return trial[edge.first] == trial[edge.second];
                           }) ||
              !isAcyclic(state, trial))
            continue;

          uint64_t bytes = 0;
          bool bytesKnown = true;
          for (const auto &[fragment, fragmentBytes] : gain.fragments) {
            (void)fragment;
            if (!fragmentBytes ||
                *fragmentBytes > std::numeric_limits<uint64_t>::max() - bytes) {
              bytesKnown = false;
              bytes = 0;
              break;
            }
            bytes += *fragmentBytes;
          }
          auto findRepresentative = [&](uint32_t label) {
            auto found = llvm::find(coherent[componentIndex], label);
            assert(found != coherent[componentIndex].end());
            return std::distance(coherent[componentIndex].begin(), found);
          };
          size_t lhs = findRepresentative(labels[gain.lhs]);
          size_t rhs = findRepresentative(labels[gain.rhs]);
          if (components[componentIndex].works[rhs] <
              components[componentIndex].works[lhs])
            std::swap(lhs, rhs);
          MergeCandidate candidate{
              {componentIndex, lhs, rhs, bytesKnown, bytes,
               static_cast<uint64_t>(gain.fragments.size())},
              components[componentIndex].works[lhs],
              components[componentIndex].works[rhs]};
          if (!best || isBetterCandidate(candidate, *best))
            best = std::move(candidate);
        }
      }
      if (!best)
        break;
      llvm::ArrayRef<uint32_t> labels = coherent[best->step.component];
      const uint32_t lhsLabel = labels[best->step.lhs];
      const uint32_t rhsLabel = labels[best->step.rhs];
      for (auto [vertex, label] : llvm::enumerate(labels))
        if (label == lhsLabel || label == rhsLabel)
          matched[best->step.component][vertex] = 1;
      mergeLabels(coherent[best->step.component], best->step.lhs,
                  best->step.rhs);
      mergeHistory.push_back(best->step);
      roundChanged = true;
    }
    if (!roundChanged)
      break;
  }

  const uint64_t snapshotCount =
      std::min<uint64_t>(maximumPlans, mergeHistory.size() + 1);
  if (snapshotCount > 1) {
    ProposalLabels snapshot = singleton;
    size_t applied = 0;
    uint64_t previousTarget = 0;
    for (uint64_t index = 1; index < snapshotCount; ++index) {
      const unsigned __int128 numerator =
          static_cast<unsigned __int128>(index) * mergeHistory.size();
      const uint64_t target = static_cast<uint64_t>(
          (numerator + snapshotCount - 2) / (snapshotCount - 1));
      if (target == previousTarget)
        continue;
      while (applied < target) {
        const MergeStep &step = mergeHistory[applied++];
        mergeLabels(snapshot[step.component], step.lhs, step.rhs);
      }
      ProposalLabels refined = refineLabels(snapshot);
      std::optional<RegionPlan> refinedPlan =
          buildPlan(refined, makeChoices(refined, /*local once=*/1));
      if (!refinedPlan)
        refinedPlan =
            buildPlan(snapshot, makeChoices(snapshot, /*local once=*/1));
      append(std::move(refinedPlan));
      previousTarget = target;
    }
  }

  if (proposals.size() >= maximumPlans)
    return retainStructuralPareto(std::move(proposals));
  append(buildPlan(maximal, makeChoices(maximal, /*local once=*/1)));
  if (proposals.size() >= maximumPlans)
    return retainStructuralPareto(std::move(proposals));
  append(buildPlan(singleton, makeChoices(singleton, /*explicit replicas=*/2)));
  return retainStructuralPareto(std::move(proposals));
}

} // namespace wafer::compiler::detail
