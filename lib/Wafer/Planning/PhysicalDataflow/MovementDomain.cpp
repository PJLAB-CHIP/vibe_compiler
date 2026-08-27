//===- MovementDomain.cpp - Exact payload transfer domain ------------===//

#include "Wafer/Planning/PhysicalDataflow/MovementDomain.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <set>
#include <type_traits>

namespace wafer::compiler::detail {
namespace {

struct TileIdLess {
  bool operator()(TileId lhs, TileId rhs) const {
    return lhs.getValue() < rhs.getValue();
  }
};

MovementDomainResult failed(MovementDomainFailureKind kind,
                            llvm::StringRef detail) {
  return {{}, MovementDomainFailure{kind, detail.str()}};
}

bool exactDomainsEqual(const analysis::ExactIndexSet &lhs,
                       const analysis::ExactIndexSet &rhs) {
  return lhs.getRank() == rhs.getRank() &&
         lhs.getPresburgerSet().isObviouslyEqual(rhs.getPresburgerSet());
}

bool hasCapability(const MovementTransportFacts &transport, TileId source,
                   TileId destination) {
  return llvm::binary_search(transport.endpointTransfers,
                             EndpointTransferCapability{source, destination});
}

bool hasEndpointPath(const MovementTransportFacts &transport, TileId source,
                     TileId destination) {
  if (source == destination)
    return true;
  std::set<TileId, TileIdLess> reached{source};
  bool changed = true;
  while (changed) {
    changed = false;
    for (const EndpointTransferCapability &edge : transport.endpointTransfers) {
      if (!reached.count(edge.source) || reached.count(edge.destination))
        continue;
      if (edge.destination == destination)
        return true;
      reached.insert(edge.destination);
      changed = true;
    }
  }
  return false;
}

bool incrementDigits(std::vector<uint32_t> &digits, uint32_t maximum) {
  for (size_t reverse = 0; reverse < digits.size(); ++reverse) {
    const size_t index = digits.size() - reverse - 1;
    if (digits[index] == maximum)
      continue;
    ++digits[index];
    std::fill(digits.begin() + index + 1, digits.end(), 0);
    return true;
  }
  return false;
}

bool hasRestrictedGrowthForm(llvm::ArrayRef<uint32_t> membership) {
  uint32_t maximum = 0;
  for (uint32_t label : membership) {
    if (label == 0)
      continue;
    if (label > maximum + 1)
      return false;
    maximum = std::max(maximum, label);
  }
  return true;
}

uint32_t getGroupCount(llvm::ArrayRef<uint32_t> membership) {
  return membership.empty()
             ? 0
             : *std::max_element(membership.begin(), membership.end());
}

struct TreeDomain {
  TileId root{0};
  std::vector<TileId> terminals;
  std::vector<TileId> relayCandidates;
};

TreeDomain buildTreeDomain(const MovementTransportFacts &transport, TileId root,
                           llvm::ArrayRef<TileId> terminals) {
  TreeDomain domain;
  domain.root = root;
  domain.terminals.assign(terminals.begin(), terminals.end());
  llvm::sort(domain.terminals, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  domain.terminals.erase(
      std::unique(domain.terminals.begin(), domain.terminals.end()),
      domain.terminals.end());
  for (TileId tile : transport.availableTiles)
    if (tile != root && !llvm::is_contained(domain.terminals, tile))
      domain.relayCandidates.push_back(tile);
  return domain;
}

std::vector<TileId> getActiveNodes(const TreeDomain &domain,
                                   const MovementCursor::TreeChoice &choice) {
  std::vector<TileId> active{domain.root};
  active.insert(active.end(), domain.terminals.begin(), domain.terminals.end());
  for (auto [index, relay] : llvm::enumerate(domain.relayCandidates))
    if (choice.relayMask & (uint64_t{1} << index))
      active.push_back(relay);
  llvm::sort(active, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  active.erase(std::unique(active.begin(), active.end()), active.end());
  return active;
}

std::vector<std::vector<TileId>>
getParentCandidates(const MovementTransportFacts &transport,
                    const TreeDomain &domain,
                    const MovementCursor::TreeChoice &choice,
                    std::vector<TileId> *children = nullptr) {
  std::vector<TileId> active = getActiveNodes(domain, choice);
  std::vector<TileId> localChildren;
  for (TileId node : active)
    if (node != domain.root)
      localChildren.push_back(node);
  std::vector<std::vector<TileId>> candidates;
  for (TileId child : localChildren) {
    std::vector<TileId> parents;
    for (TileId parent : active)
      if (parent != child && hasCapability(transport, parent, child))
        parents.push_back(parent);
    candidates.push_back(std::move(parents));
  }
  if (children)
    *children = std::move(localChildren);
  return candidates;
}

bool incrementParents(std::vector<size_t> &choices,
                      llvm::ArrayRef<std::vector<TileId>> candidates) {
  for (size_t reverse = 0; reverse < choices.size(); ++reverse) {
    const size_t index = choices.size() - reverse - 1;
    if (choices[index] + 1 >= candidates[index].size())
      continue;
    ++choices[index];
    std::fill(choices.begin() + index + 1, choices.end(), 0);
    return true;
  }
  return false;
}

bool isValidTree(const MovementTransportFacts &transport,
                 const TreeDomain &domain,
                 const MovementCursor::TreeChoice &choice) {
  if (domain.relayCandidates.size() >= 64 ||
      (choice.relayMask >> domain.relayCandidates.size()) != 0)
    return false;
  std::vector<TileId> children;
  std::vector<std::vector<TileId>> candidates =
      getParentCandidates(transport, domain, choice, &children);
  if (choice.parentChoices.size() != children.size() ||
      llvm::any_of(llvm::enumerate(choice.parentChoices), [&](auto indexed) {
        return indexed.value() >= candidates[indexed.index()].size();
      }))
    return false;
  std::map<TileId, TileId, TileIdLess> parent;
  for (auto [child, options, selected] :
       llvm::zip_equal(children, candidates, choice.parentChoices))
    parent.emplace(child, options[selected]);

  const size_t activeCount = children.size() + 1;
  for (TileId child : children) {
    TileId current = child;
    size_t steps = 0;
    while (current != domain.root) {
      auto edge = parent.find(current);
      if (edge == parent.end() || ++steps >= activeCount)
        return false;
      current = edge->second;
    }
  }
  for (auto [index, relay] : llvm::enumerate(domain.relayCandidates)) {
    if (!(choice.relayMask & (uint64_t{1} << index)))
      continue;
    bool contributes = false;
    for (TileId terminal : domain.terminals) {
      TileId current = terminal;
      while (current != domain.root) {
        if (current == relay) {
          contributes = true;
          break;
        }
        current = parent.at(current);
      }
      if (contributes)
        break;
    }
    if (!contributes)
      return false;
  }
  return true;
}

bool setFirstTree(const MovementTransportFacts &transport,
                  const TreeDomain &domain, MovementCursor::TreeChoice &choice,
                  uint64_t firstMask = 0) {
  if (domain.relayCandidates.size() >= 64)
    return false;
  const uint64_t maskLimit = uint64_t{1} << domain.relayCandidates.size();
  for (uint64_t mask = firstMask; mask < maskLimit; ++mask) {
    choice.relayMask = mask;
    std::vector<std::vector<TileId>> candidates =
        getParentCandidates(transport, domain, choice);
    if (llvm::any_of(candidates,
                     [](const auto &parents) { return parents.empty(); }))
      continue;
    choice.parentChoices.assign(candidates.size(), 0);
    do {
      if (isValidTree(transport, domain, choice))
        return true;
    } while (incrementParents(choice.parentChoices, candidates));
  }
  return false;
}

bool advanceTree(const MovementTransportFacts &transport,
                 const TreeDomain &domain, MovementCursor::TreeChoice &choice) {
  if (!isValidTree(transport, domain, choice))
    return false;
  std::vector<std::vector<TileId>> candidates =
      getParentCandidates(transport, domain, choice);
  while (incrementParents(choice.parentChoices, candidates))
    if (isValidTree(transport, domain, choice))
      return true;
  return setFirstTree(transport, domain, choice, choice.relayMask + 1);
}

std::vector<MovementHop> getTreeHops(const MovementTransportFacts &transport,
                                     const TreeDomain &domain,
                                     const MovementCursor::TreeChoice &choice) {
  std::vector<TileId> children;
  std::vector<std::vector<TileId>> candidates =
      getParentCandidates(transport, domain, choice, &children);
  std::vector<MovementHop> hops;
  for (auto [child, options, selected] :
       llvm::zip_equal(children, candidates, choice.parentChoices))
    hops.push_back({options[selected], child});
  return hops;
}

std::optional<MovementCursor::TreeChoice>
getTreeCursor(const MovementTransportFacts &transport, const TreeDomain &domain,
              llvm::ArrayRef<MovementHop> hops) {
  MovementCursor::TreeChoice choice;
  std::set<TileId, TileIdLess> active;
  active.insert(domain.root);
  active.insert(domain.terminals.begin(), domain.terminals.end());
  for (const MovementHop &hop : hops) {
    active.insert(hop.source);
    active.insert(hop.destination);
  }
  for (TileId tile : active) {
    if (tile == domain.root || llvm::is_contained(domain.terminals, tile))
      continue;
    auto relay = llvm::find(domain.relayCandidates, tile);
    if (relay == domain.relayCandidates.end())
      return std::nullopt;
    choice.relayMask |= uint64_t{1} << static_cast<size_t>(std::distance(
                            domain.relayCandidates.begin(), relay));
  }
  std::vector<TileId> children;
  std::vector<std::vector<TileId>> candidates =
      getParentCandidates(transport, domain, choice, &children);
  std::map<TileId, TileId, TileIdLess> parents;
  for (const MovementHop &hop : hops)
    if (!parents.try_emplace(hop.destination, hop.source).second)
      return std::nullopt;
  for (auto [index, child] : llvm::enumerate(children)) {
    auto parent = parents.find(child);
    if (parent == parents.end())
      return std::nullopt;
    auto selected = llvm::find(candidates[index], parent->second);
    if (selected == candidates[index].end())
      return std::nullopt;
    choice.parentChoices.push_back(static_cast<size_t>(
        std::distance(candidates[index].begin(), selected)));
  }
  if (parents.size() != children.size() ||
      !isValidTree(transport, domain, choice) ||
      !llvm::equal(getTreeHops(transport, domain, choice), hops))
    return std::nullopt;
  return choice;
}

} // namespace

bool MovementDomain::advanceClassChoice(
    size_t classIndex, MovementCursor::ClassChoice &choice) const {
  const ReuseClass &domain = classes[classIndex];
  if (choice.membership.size() != domain.actions.size())
    return false;

  auto getGroups = [&](llvm::ArrayRef<uint32_t> membership) {
    std::vector<std::vector<size_t>> groups(getGroupCount(membership));
    for (auto [index, label] : llvm::enumerate(membership))
      if (label != 0)
        groups[label - 1].push_back(index);
    return groups;
  };
  auto validMembership = [&](llvm::ArrayRef<uint32_t> membership) {
    if (membership.size() != domain.actions.size() ||
        !hasRestrictedGrowthForm(membership))
      return false;
    std::vector<std::vector<size_t>> groups = getGroups(membership);
    for (const auto &group : groups) {
      if (group.empty() || (domain.actions[group.front()].kind ==
                                MovementPlanActionKind::ExternalLoad &&
                            group.size() < 2))
        return false;
      std::set<TileId, TileIdLess> destinations;
      for (size_t index : group)
        if (!domain.actions[index].peerCapable ||
            !destinations.insert(domain.actions[index].destination).second)
          return false;
    }
    return true;
  };
  auto getTree = [&](llvm::ArrayRef<size_t> group,
                     size_t rootChoice) -> std::optional<TreeDomain> {
    if (group.empty())
      return std::nullopt;
    std::vector<TileId> terminals;
    for (size_t index : group)
      terminals.push_back(domain.actions[index].destination);
    const bool external = domain.actions[group.front()].kind ==
                          MovementPlanActionKind::ExternalLoad;
    if ((!external && rootChoice != 0) ||
        (external && rootChoice >= group.size()))
      return std::nullopt;
    TileId root = external ? domain.actions[group[rootChoice]].destination
                           : domain.actions[group.front()].source;
    return buildTreeDomain(transport, root, terminals);
  };
  auto setFirstGroupTree = [&](llvm::ArrayRef<size_t> group,
                               MovementCursor::TreeChoice &selected,
                               size_t firstRoot = 0) {
    const bool external =
        !group.empty() && domain.actions[group.front()].kind ==
                              MovementPlanActionKind::ExternalLoad;
    const size_t rootCount = external ? group.size() : 1;
    for (size_t root = firstRoot; root < rootCount; ++root) {
      std::optional<TreeDomain> tree = getTree(group, root);
      selected = {};
      selected.rootChoice = root;
      if (tree && setFirstTree(transport, *tree, selected))
        return true;
    }
    return false;
  };
  auto advanceGroupTree = [&](llvm::ArrayRef<size_t> group,
                              MovementCursor::TreeChoice &selected) {
    std::optional<TreeDomain> tree = getTree(group, selected.rootChoice);
    if (tree && advanceTree(transport, *tree, selected))
      return true;
    return setFirstGroupTree(group, selected, selected.rootChoice + 1);
  };
  auto firstTrees = [&](llvm::ArrayRef<uint32_t> membership,
                        std::vector<MovementCursor::TreeChoice> &trees) {
    trees.clear();
    for (const auto &group : getGroups(membership)) {
      MovementCursor::TreeChoice selected;
      if (!setFirstGroupTree(group, selected))
        return false;
      trees.push_back(std::move(selected));
    }
    return true;
  };

  if (validMembership(choice.membership)) {
    std::vector<std::vector<size_t>> groups = getGroups(choice.membership);
    if (choice.trees.size() == groups.size()) {
      for (size_t reverse = 0; reverse < groups.size(); ++reverse) {
        const size_t index = groups.size() - reverse - 1;
        if (!advanceGroupTree(groups[index], choice.trees[index]))
          continue;
        for (size_t reset = index + 1; reset < groups.size(); ++reset) {
          if (!setFirstGroupTree(groups[reset], choice.trees[reset]))
            return false;
        }
        return true;
      }
    }
  }

  std::vector<uint32_t> next = choice.membership;
  while (incrementDigits(next, static_cast<uint32_t>(domain.actions.size()))) {
    std::vector<MovementCursor::TreeChoice> trees;
    if (!validMembership(next) || !firstTrees(next, trees))
      continue;
    choice.membership = std::move(next);
    choice.trees = std::move(trees);
    return true;
  }
  return false;
}

std::optional<MovementPlan>
MovementDomain::buildPlan(const MovementCursor &cursor) const {
  if (cursor.classes.size() != classes.size())
    return std::nullopt;
  MovementPlan plan = base;
  plan.peerGraphs.clear();
  for (auto [classIndex, zipped] :
       llvm::enumerate(llvm::zip_equal(classes, cursor.classes))) {
    const ReuseClass &domain = std::get<0>(zipped);
    const MovementCursor::ClassChoice &choice = std::get<1>(zipped);
    if (choice.membership.size() != domain.actions.size() ||
        !hasRestrictedGrowthForm(choice.membership))
      return std::nullopt;
    const uint32_t groupCount = getGroupCount(choice.membership);
    if (choice.trees.size() != groupCount)
      return std::nullopt;
    for (uint32_t label = 1; label <= groupCount; ++label) {
      std::vector<size_t> members;
      std::vector<TileId> terminals;
      std::vector<MovementActionId> actions;
      for (auto [index, selected] : llvm::enumerate(choice.membership))
        if (selected == label) {
          members.push_back(index);
          terminals.push_back(domain.actions[index].destination);
          actions.push_back(domain.actions[index].action);
        }
      if (members.empty())
        return std::nullopt;
      llvm::sort(actions);
      const bool external = domain.actions[members.front()].kind ==
                            MovementPlanActionKind::ExternalLoad;
      const size_t rootChoice = choice.trees[label - 1].rootChoice;
      if ((external && rootChoice >= members.size()) ||
          (!external && rootChoice != 0) || (external && members.size() < 2))
        return std::nullopt;
      TileId root = external ? domain.actions[members[rootChoice]].destination
                             : domain.actions[members.front()].source;
      TreeDomain tree = buildTreeDomain(transport, root, terminals);
      if (!isValidTree(transport, tree, choice.trees[label - 1]))
        return std::nullopt;
      std::vector<MovementHop> hops =
          getTreeHops(transport, tree, choice.trees[label - 1]);
      PeerTransferGraphKind kind =
          external ? PeerTransferGraphKind::ExternalLoadFanout
                   : PeerTransferGraphKind::SoftwareFanout;
      if (!external && members.size() == 1)
        kind = hops.size() == 1 &&
                       hops.front().source ==
                           domain.actions[members.front()].source &&
                       hops.front().destination ==
                           domain.actions[members.front()].destination
                   ? PeerTransferGraphKind::TargetRoutedPeer
                   : PeerTransferGraphKind::SoftwareRelay;
      PeerTransferGraphPlan realization{kind, std::move(hops),
                                        std::move(actions)};
      if (external)
        realization.ddrRoot = std::get<ExternalLoadId>(
            domain.actions[members[rootChoice]].action);
      plan.peerGraphs.push_back(std::move(realization));
    }
    (void)classIndex;
  }
  llvm::sort(plan.peerGraphs, [](const PeerTransferGraphPlan &lhs,
                                 const PeerTransferGraphPlan &rhs) {
    return lhs.actions < rhs.actions;
  });
  return plan;
}

MovementSuccessor MovementDomain::getFirstPlan() const {
  MovementCursor cursor;
  for (const ReuseClass &domain : classes) {
    MovementCursor::ClassChoice choice;
    choice.membership.assign(domain.actions.size(), 0);
    cursor.classes.push_back(std::move(choice));
  }
  std::optional<MovementPlan> plan = buildPlan(cursor);
  if (!plan)
    return {MovementSuccessorKind::CompilerBug,
            {},
            {},
            "movement domain has no valid first plan"};
  return {MovementSuccessorKind::Plan, std::move(plan), std::move(cursor)};
}

std::optional<MovementCursor>
MovementDomain::getCursor(const MovementPlan &plan) const {
  MovementPlan normalized = plan;
  normalized.peerGraphs.clear();
  MovementPlan normalizedBase = base;
  normalizedBase.peerGraphs.clear();
  if (!(normalized == normalizedBase))
    return std::nullopt;

  if (!llvm::is_sorted(plan.peerGraphs, [](const PeerTransferGraphPlan &lhs,
                                           const PeerTransferGraphPlan &rhs) {
        return lhs.actions < rhs.actions;
      }))
    return std::nullopt;
  std::map<MovementActionId, const PeerTransferGraphPlan *> graphsByAction;
  for (const PeerTransferGraphPlan &graph : plan.peerGraphs) {
    if (graph.actions.empty() || !llvm::is_sorted(graph.actions) ||
        std::adjacent_find(graph.actions.begin(), graph.actions.end()) !=
            graph.actions.end())
      return std::nullopt;
    for (const MovementActionId &action : graph.actions)
      if (!graphsByAction.try_emplace(action, &graph).second)
        return std::nullopt;
  }

  MovementCursor cursor;
  std::set<const PeerTransferGraphPlan *> consumedGraphs;
  for (const ReuseClass &domain : classes) {
    MovementCursor::ClassChoice choice;
    choice.membership.assign(domain.actions.size(), 0);
    std::vector<const PeerTransferGraphPlan *> groups;
    for (auto [index, action] : llvm::enumerate(domain.actions)) {
      auto graph = graphsByAction.find(action.action);
      if (graph == graphsByAction.end())
        continue;
      if (!action.peerCapable)
        return std::nullopt;
      auto found = llvm::find(groups, graph->second);
      size_t group = 0;
      if (found == groups.end()) {
        groups.push_back(graph->second);
        consumedGraphs.insert(graph->second);
        group = groups.size();
      } else {
        group = static_cast<size_t>(std::distance(groups.begin(), found)) + 1;
      }
      choice.membership[index] = static_cast<uint32_t>(group);
    }
    if (!hasRestrictedGrowthForm(choice.membership))
      return std::nullopt;
    for (auto [groupIndex, graph] : llvm::enumerate(groups)) {
      std::vector<size_t> members;
      std::vector<TileId> terminals;
      std::vector<MovementActionId> expectedActions;
      for (auto [index, label] : llvm::enumerate(choice.membership))
        if (label == groupIndex + 1) {
          members.push_back(index);
          terminals.push_back(domain.actions[index].destination);
          expectedActions.push_back(domain.actions[index].action);
        }
      llvm::sort(expectedActions);
      if (members.empty() || expectedActions != graph->actions)
        return std::nullopt;
      const bool external = domain.actions[members.front()].kind ==
                            MovementPlanActionKind::ExternalLoad;
      size_t rootChoice = 0;
      if (external) {
        if (members.size() < 2 ||
            graph->kind != PeerTransferGraphKind::ExternalLoadFanout ||
            !graph->ddrRoot)
          return std::nullopt;
        auto root = llvm::find_if(members, [&](size_t member) {
          return domain.actions[member].action ==
                 MovementActionId(*graph->ddrRoot);
        });
        if (root == members.end())
          return std::nullopt;
        rootChoice = static_cast<size_t>(std::distance(members.begin(), root));
      } else if (graph->ddrRoot) {
        return std::nullopt;
      }
      TileId root = external ? domain.actions[members[rootChoice]].destination
                             : domain.actions[members.front()].source;
      TreeDomain tree = buildTreeDomain(transport, root, terminals);
      std::optional<MovementCursor::TreeChoice> treeCursor =
          getTreeCursor(transport, tree, graph->hops);
      if (!treeCursor)
        return std::nullopt;
      treeCursor->rootChoice = rootChoice;
      PeerTransferGraphKind expectedKind =
          external ? PeerTransferGraphKind::ExternalLoadFanout
                   : PeerTransferGraphKind::SoftwareFanout;
      if (!external && members.size() == 1)
        expectedKind = graph->hops.size() == 1 &&
                               graph->hops.front().source ==
                                   domain.actions[members.front()].source &&
                               graph->hops.front().destination ==
                                   domain.actions[members.front()].destination
                           ? PeerTransferGraphKind::TargetRoutedPeer
                           : PeerTransferGraphKind::SoftwareRelay;
      if (graph->kind != expectedKind)
        return std::nullopt;
      choice.trees.push_back(std::move(*treeCursor));
    }
    cursor.classes.push_back(std::move(choice));
  }
  if (consumedGraphs.size() != plan.peerGraphs.size())
    return std::nullopt;
  std::optional<MovementPlan> rebuilt = buildPlan(cursor);
  if (!rebuilt || !(*rebuilt == plan))
    return std::nullopt;
  return cursor;
}

bool MovementDomain::contains(const MovementPlan &plan) const {
  return getCursor(plan).has_value();
}

std::vector<MovementPlan> MovementDomain::getProposals() const {
  auto buildProposal = [&](bool maximal) -> std::optional<MovementPlan> {
    MovementCursor cursor;
    for (const ReuseClass &domain : classes) {
      MovementCursor::ClassChoice choice;
      choice.membership.assign(domain.actions.size(), 0);
      std::vector<std::set<int64_t>> destinations;
      for (auto [index, action] : llvm::enumerate(domain.actions)) {
        const bool external =
            action.kind == MovementPlanActionKind::ExternalLoad;
        if (!action.peerCapable || (external && !maximal))
          continue;
        size_t group = destinations.size();
        if (maximal)
          for (size_t candidate = 0; candidate < destinations.size();
               ++candidate)
            if (!destinations[candidate].count(action.destination.getValue())) {
              group = candidate;
              break;
            }
        if (group == destinations.size())
          destinations.emplace_back();
        destinations[group].insert(action.destination.getValue());
        choice.membership[index] = static_cast<uint32_t>(group + 1);
      }
      if (!domain.actions.empty() &&
          domain.actions.front().kind == MovementPlanActionKind::ExternalLoad) {
        std::vector<size_t> counts(getGroupCount(choice.membership) + 1, 0);
        for (uint32_t label : choice.membership)
          ++counts[label];
        for (uint32_t &label : choice.membership)
          if (label != 0 && counts[label] < 2)
            label = 0;
        std::map<uint32_t, uint32_t> canonical;
        for (uint32_t &label : choice.membership)
          if (label != 0) {
            auto [entry, inserted] = canonical.try_emplace(
                label, static_cast<uint32_t>(canonical.size() + 1));
            (void)inserted;
            label = entry->second;
          }
      }
      for (size_t group = 0; group < getGroupCount(choice.membership);
           ++group) {
        std::vector<TileId> terminals;
        TileId root{0};
        bool foundRoot = false;
        for (auto [index, label] : llvm::enumerate(choice.membership))
          if (label == group + 1) {
            terminals.push_back(domain.actions[index].destination);
            root = domain.actions[index].source;
            foundRoot = true;
          }
        if (!foundRoot)
          return std::nullopt;
        const bool external =
            domain.actions.front().kind == MovementPlanActionKind::ExternalLoad;
        if (external)
          root = terminals.front();
        TreeDomain tree = buildTreeDomain(transport, root, terminals);
        MovementCursor::TreeChoice selected;
        selected.rootChoice = 0;
        if (!setFirstTree(transport, tree, selected))
          return std::nullopt;
        choice.trees.push_back(std::move(selected));
      }
      cursor.classes.push_back(std::move(choice));
    }
    return buildPlan(cursor);
  };

  std::vector<MovementPlan> proposals;
  for (bool maximal : {true, false}) {
    std::optional<MovementPlan> proposal = buildProposal(maximal);
    if (proposal && contains(*proposal) &&
        !llvm::is_contained(proposals, *proposal))
      proposals.push_back(std::move(*proposal));
  }
  return proposals;
}

MovementSuccessor
MovementDomain::getNextPlan(const MovementCursor &cursor) const {
  std::optional<MovementPlan> current = buildPlan(cursor);
  if (!current || !contains(*current))
    return {MovementSuccessorKind::CompilerBug,
            {},
            {},
            "movement cursor is outside the current domain"};
  MovementCursor next = cursor;
  for (size_t reverse = 0; reverse < classes.size(); ++reverse) {
    const size_t index = classes.size() - reverse - 1;
    if (!advanceClassChoice(index, next.classes[index]))
      continue;
    for (size_t reset = index + 1; reset < classes.size(); ++reset) {
      next.classes[reset].membership.assign(classes[reset].actions.size(), 0);
      next.classes[reset].trees.clear();
    }
    std::optional<MovementPlan> plan = buildPlan(next);
    return plan ? MovementSuccessor(MovementSuccessorKind::Plan,
                                    std::move(plan), std::move(next))
                : MovementSuccessor(MovementSuccessorKind::CompilerBug, {}, {},
                                    "movement successor is malformed");
  }
  return {MovementSuccessorKind::End};
}

MovementTransportFacts
buildOpaqueEndpointTransportFacts(llvm::ArrayRef<TileId> inputTiles) {
  MovementTransportFacts facts;
  facts.availableTiles.assign(inputTiles.begin(), inputTiles.end());
  llvm::sort(facts.availableTiles, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  for (TileId source : facts.availableTiles)
    for (TileId destination : facts.availableTiles)
      if (source != destination)
        facts.endpointTransfers.push_back({source, destination});
  return facts;
}

MovementDomainResult
buildMovementDomain(const CanonicalMovementCoordinate &canonical,
                    const RepresentationPlan &representations,
                    llvm::ArrayRef<TileId> availableTiles) {
  return buildMovementDomain(canonical, representations,
                             buildOpaqueEndpointTransportFacts(availableTiles));
}

MovementDomainResult
buildMovementDomain(const CanonicalMovementCoordinate &canonical,
                    const RepresentationPlan &representations,
                    const MovementTransportFacts &inputTransport) {
  MovementTransportFacts transport = inputTransport;
  llvm::sort(transport.availableTiles, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  llvm::sort(transport.endpointTransfers);
  if (transport.availableTiles.empty() ||
      std::adjacent_find(transport.availableTiles.begin(),
                         transport.availableTiles.end()) !=
          transport.availableTiles.end() ||
      std::adjacent_find(transport.endpointTransfers.begin(),
                         transport.endpointTransfers.end()) !=
          transport.endpointTransfers.end() ||
      llvm::any_of(transport.endpointTransfers, [&](const auto &edge) {
        return edge.source == edge.destination ||
               !llvm::is_contained(transport.availableTiles, edge.source) ||
               !llvm::is_contained(transport.availableTiles, edge.destination);
      }))
    return failed(MovementDomainFailureKind::BrokenContract,
                  "movement domain has invalid endpoint facts");
  if (transport.availableTiles.size() > 63)
    return failed(MovementDomainFailureKind::UnsupportedSemantics,
                  "movement relay subsets exceed the current typed Tile "
                  "identity width");
  if (!transport.ddrStagesSupported &&
      (!canonical.plan.externalLoads.empty() ||
       !canonical.plan.ddrTransfers.empty() ||
       !canonical.plan.reductionGathers.empty() ||
       !canonical.plan.publications.empty()))
    return failed(MovementDomainFailureKind::UnsupportedSemantics,
                  "canonical movement requires an unavailable DDR stage");
  if (!canonical.plan.peerGraphs.empty())
    return failed(MovementDomainFailureKind::BrokenContract,
                  "canonical movement unexpectedly contains a peer graph");

  std::map<RegionValueVersionId, PhysicalVersionId> primaryVersions;
  for (const LogicalRepresentationPlan &logical : representations.logicalValues)
    if (!primaryVersions.try_emplace(logical.value, logical.primary).second)
      return failed(MovementDomainFailureKind::BrokenContract,
                    "movement representation has duplicate logical values");

  std::map<BoundaryRegionValueId, PhysicalVersionId> boundaryVersions;
  for (const PhysicalUseBinding &binding : representations.uses)
    if (const auto *boundary =
            std::get_if<BoundaryRepresentationUseId>(&binding.use))
      if (!boundaryVersions.try_emplace(boundary->value, binding.version)
               .second)
        return failed(MovementDomainFailureKind::BrokenContract,
                      "movement representation has duplicate boundary uses");

  std::map<PhysicalVersionId, MemLayout> versionEncodings;
  for (const PhysicalVersionPlan &version : representations.physicalVersions)
    if (!versionEncodings.try_emplace(version.id, version.encoding).second)
      return failed(MovementDomainFailureKind::BrokenContract,
                    "movement representation has duplicate physical versions");

  std::map<MovementActionId, const MovementResourceDescription *> resources;
  for (const MovementResourceDescription &resource : canonical.resources)
    if (!resources.try_emplace(resource.action, &resource).second)
      return failed(MovementDomainFailureKind::BrokenContract,
                    "movement coordinate has duplicate resource actions");

  auto findPrimary =
      [&](const RegionValueVersionId &logical) -> const PhysicalVersionId * {
    auto found = primaryVersions.find(logical);
    return found == primaryVersions.end() ? nullptr : &found->second;
  };
  auto findBoundary =
      [&](const BoundaryRegionValueId &value) -> const PhysicalVersionId * {
    auto found = boundaryVersions.find(value);
    return found == boundaryVersions.end() ? nullptr : &found->second;
  };
  auto findEncoding =
      [&](const PhysicalVersionId &version) -> std::optional<MemLayout> {
    auto found = versionEncodings.find(version);
    return found == versionEncodings.end()
               ? std::nullopt
               : std::optional<MemLayout>(found->second);
  };
  auto findResource = [&](const MovementActionId &action)
      -> const MovementResourceDescription * {
    auto found = resources.find(action);
    return found == resources.end() ? nullptr : found->second;
  };

  MovementPlan base = canonical.plan;
  for (ExternalLoadPlan &load : base.externalLoads) {
    const PhysicalVersionId *destination = findBoundary(load.id.destination);
    if (!destination || !versionEncodings.count(*destination))
      return failed(MovementDomainFailureKind::BrokenContract,
                    "external load has no selected destination version");
    load.destination = *destination;
  }
  for (DDRBoundaryTransferPlan &transfer : base.ddrTransfers) {
    const PhysicalVersionId *source = findPrimary(transfer.source.logicalValue);
    const PhysicalVersionId *destination =
        findBoundary(transfer.id.destination);
    if (!source || !destination || !versionEncodings.count(*source) ||
        !versionEncodings.count(*destination))
      return failed(MovementDomainFailureKind::BrokenContract,
                    "boundary transfer has missing selected versions");
    transfer.source = *source;
    transfer.destination = *destination;
  }
  for (ReductionGatherPlan &gather : base.reductionGathers) {
    const PhysicalVersionId *source = findPrimary(gather.source.logicalValue);
    if (!source || !versionEncodings.count(*source))
      return failed(MovementDomainFailureKind::BrokenContract,
                    "reduction gather has no selected source version");
    gather.source = *source;
  }
  for (ResultPublicationPlan &publication : base.publications) {
    const PhysicalVersionId *source =
        findPrimary(publication.source.logicalValue);
    if (!source)
      return failed(MovementDomainFailureKind::BrokenContract,
                    "publication has no selected source version");
    publication.source = *source;
  }
  for (ResultDiscardPlan &discard : base.discards) {
    const PhysicalVersionId *source = findPrimary(discard.source.logicalValue);
    if (!source)
      return failed(MovementDomainFailureKind::BrokenContract,
                    "discard has no selected source version");
    discard.source = *source;
  }

  std::vector<MovementDomain::ReuseClass> classes;
  std::map<analysis::RootBoundaryId, std::vector<size_t>> externalClassBuckets;
  using BoundaryClassKey = std::tuple<PhysicalVersionId, int64_t, MemLayout>;
  std::map<BoundaryClassKey, std::vector<size_t>> boundaryClassBuckets;
  auto addVariable = [&](MovementPlanActionKind kind, size_t planIndex,
                         const MovementActionId &action,
                         const PhysicalVersionId *sourceVersion,
                         const PhysicalVersionId *destinationVersion) -> bool {
    const MovementResourceDescription *resource = findResource(action);
    const bool external = kind == MovementPlanActionKind::ExternalLoad;
    TileId semanticDestination = std::visit(
        [&](const auto &id) -> TileId {
          using T = std::decay_t<decltype(id)>;
          if constexpr (std::is_same_v<T, ExternalLoadId>)
            return id.destination.work.tile;
          if constexpr (std::is_same_v<T, DDRBoundaryTransferId>)
            return id.destination.work.tile;
          if constexpr (std::is_same_v<T, ReductionGatherId>) {
            const ReductionGatherPlan &gather =
                base.reductionGathers[planIndex];
            return std::visit(
                [](const auto &source) { return source.work.tile; },
                gather.mergeExecution.source);
          }
          return TileId(0);
        },
        action);
    if (!resource || !resource->destinationTile ||
        *resource->destinationTile != semanticDestination ||
        (external ? resource->sourceTile.has_value()
                  : !resource->sourceTile.has_value()) ||
        (!external && !llvm::is_contained(transport.availableTiles,
                                          *resource->sourceTile)) ||
        !llvm::is_contained(transport.availableTiles,
                            *resource->destinationTile))
      return false;
    std::optional<MemLayout> sourceEncoding =
        sourceVersion ? findEncoding(*sourceVersion) : std::nullopt;
    std::optional<MemLayout> destinationEncoding =
        destinationVersion ? findEncoding(*destinationVersion) : sourceEncoding;
    MovementDomain::ActionVariable variable;
    variable.kind = kind;
    variable.planIndex = planIndex;
    variable.action = action;
    if (resource->sourceTile)
      variable.source = *resource->sourceTile;
    variable.destination = *resource->destinationTile;
    variable.peerCapable = external
                               ? destinationEncoding.has_value()
                               : variable.source != variable.destination &&
                                     sourceEncoding && destinationEncoding &&
                                     *sourceEncoding == *destinationEncoding &&
                                     hasEndpointPath(transport, variable.source,
                                                     variable.destination);

    std::vector<size_t> *candidateClasses = nullptr;
    if (variable.peerCapable && external) {
      const ExternalLoadPlan &load = base.externalLoads[planIndex];
      candidateClasses =
          &externalClassBuckets[load.id.destination.fragment.source];
    } else if (variable.peerCapable &&
               kind == MovementPlanActionKind::BoundaryTransfer &&
               sourceVersion && destinationEncoding) {
      candidateClasses = &boundaryClassBuckets[BoundaryClassKey{
          *sourceVersion, variable.source.getValue(), *destinationEncoding}];
    }
    if (candidateClasses) {
      for (size_t classIndex : *candidateClasses) {
        MovementDomain::ReuseClass &candidate = classes[classIndex];
        if (candidate.actions.empty() ||
            candidate.actions.front().kind != kind ||
            !candidate.actions.front().peerCapable)
          continue;
        const auto &first = candidate.actions.front();
        const MovementResourceDescription *firstResource =
            findResource(first.action);
        bool compatible = firstResource &&
                          firstResource->elementType == resource->elementType &&
                          exactDomainsEqual(firstResource->exactDomain,
                                            resource->exactDomain);
        if (external) {
          const ExternalLoadPlan &firstLoad =
              base.externalLoads[first.planIndex];
          const ExternalLoadPlan &load = base.externalLoads[planIndex];
          compatible =
              compatible &&
              firstLoad.id.destination.fragment.source ==
                  load.id.destination.fragment.source &&
              findEncoding(firstLoad.destination) == destinationEncoding;
        } else {
          const auto &firstTransfer = base.ddrTransfers[first.planIndex];
          compatible =
              compatible && sourceVersion &&
              firstTransfer.source == *sourceVersion &&
              first.source == variable.source &&
              findEncoding(firstTransfer.destination) == destinationEncoding;
        }
        if (compatible) {
          candidate.actions.push_back(std::move(variable));
          return true;
        }
      }
    }
    const size_t classIndex = classes.size();
    classes.push_back({{std::move(variable)}});
    if (candidateClasses)
      candidateClasses->push_back(classIndex);
    return true;
  };
  for (auto [index, load] : llvm::enumerate(base.externalLoads))
    if (!addVariable(MovementPlanActionKind::ExternalLoad, index,
                     MovementActionId(load.id), nullptr, &load.destination))
      return failed(MovementDomainFailureKind::BrokenContract,
                    "external load has no exact destination endpoint");
  for (auto [index, transfer] : llvm::enumerate(base.ddrTransfers))
    if (!addVariable(MovementPlanActionKind::BoundaryTransfer, index,
                     MovementActionId(transfer.id), &transfer.source,
                     &transfer.destination))
      return failed(MovementDomainFailureKind::BrokenContract,
                    "boundary transfer has no exact endpoints");
  for (auto [index, gather] : llvm::enumerate(base.reductionGathers))
    if (!addVariable(MovementPlanActionKind::Gather, index,
                     MovementActionId(gather.id), &gather.source, nullptr))
      return failed(MovementDomainFailureKind::BrokenContract,
                    "reduction gather has no exact endpoints");
  for (MovementDomain::ReuseClass &reuse : classes)
    llvm::sort(reuse.actions, [](const auto &lhs, const auto &rhs) {
      return lhs.action < rhs.action;
    });
  llvm::sort(classes, [](const auto &lhs, const auto &rhs) {
    return lhs.actions.front().action < rhs.actions.front().action;
  });
  return {MovementDomain(std::move(base), canonical.resources,
                         std::move(transport), std::move(classes)),
          {}};
}

} // namespace wafer::compiler::detail
