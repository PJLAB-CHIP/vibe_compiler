//===- AccessReuse.cpp - Bounded profitable reuse proposals --------------===//
#include "Wafer/Planning/PhysicalDataflow/AccessReuse.h"
#include "Wafer/Support/CompileTiming.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

std::optional<uint64_t> executions(const analysis::ReadAccess &read) {
  uint64_t count = 1;
  for (const auto &loop : read.loops) {
    uint64_t trips = (loop.upper - loop.lower - 1) / loop.step + 1;
    if (count > std::numeric_limits<uint64_t>::max() / trips)
      return std::nullopt;
    count *= trips;
  }
  return count;
}

struct TrafficEstimate {
  long double savedDDRBytes = 0;
  long double hopMessages = 0;
  std::map<std::pair<int64_t, int64_t>, long double> sends, messages, copies;
  std::map<std::tuple<int64_t, int64_t, int64_t>, long double> links;
};

bool addPeer(TrafficEstimate &traffic, TileModuleOp donor, int64_t destination,
             long double payload, long double count) {
  auto module = donor->getParentOfType<mlir::ModuleOp>();
  TargetTopologyOp topology;
  for (auto op : module.getOps<TargetTopologyOp>())
    topology = op;
  if (!topology || topology.getTileGrid().size() != 2)
    return false;
  const int64_t rows = topology.getTileGrid()[0],
                columns = topology.getTileGrid()[1];
  const int64_t card = donor.getCardId(), sender = donor.getTileId();
  if (rows <= 0 || columns <= 0 || rows > 64 || columns > 64 || sender < 0 ||
      destination < 0 || sender / columns >= rows ||
      destination / columns >= rows)
    return false;
  traffic.sends[{card, sender}] += payload;
  traffic.messages[{card, sender}] += count;
  // Dimension-order is the existing cost prior, not a claim about actual
  // routes.
  int64_t current = sender;
  while (current % columns != destination % columns) {
    int64_t next =
        current + (current % columns < destination % columns ? 1 : -1);
    traffic.links[{card, current, next}] += payload;
    traffic.hopMessages += count;
    current = next;
  }
  while (current != destination) {
    int64_t next = current + (current < destination ? columns : -columns);
    traffic.links[{card, current, next}] += payload;
    traffic.hopMessages += count;
    current = next;
  }
  return true;
}

const analysis::ScopedReadAccess *
findWindow(const analysis::AccessReuseAnalysis &facts,
           const AccessReuseAction &action, mlir::scf::ForOp scope) {
  for (const auto &window : facts.scopes)
    if (window.scope == scope && !action.reads.empty() &&
        llvm::is_contained(window.reads, action.reads.front()))
      return &window;
  return nullptr;
}

bool equivalentWindows(const analysis::ScopedReadAccess &a,
                       const analysis::ScopedReadAccess &b) {
  auto aTile = a.tile, bTile = b.tile;
  return a.argument == b.argument && aTile.getCardId() == bTile.getCardId() &&
         a.sizes == b.sizes && a.sourceStrides == b.sourceStrides &&
         mlir::cast<mlir::MemRefType>(a.source.getType()).getElementType() ==
             mlir::cast<mlir::MemRefType>(b.source.getType())
                 .getElementType() &&
         a.linearOffset == b.linearOffset &&
         a.outerLoops.size() == b.outerLoops.size() &&
         llvm::all_of(
             llvm::zip_equal(a.outerLoops, b.outerLoops), [](const auto &pair) {
               return std::get<0>(pair).bounds() == std::get<1>(pair).bounds();
             });
}

std::optional<long double>
estimateBenefit(const AccessReuseChoice &choice,
                const analysis::AccessReuseAnalysis &facts,
                const analysis::SearchCostPolicy &policy) {
  TrafficEstimate traffic;
  llvm::SmallVector<const analysis::ScopedReadAccess *, 16> residents;
  for (const auto &action : choice.actions) {
    if (action.reads.empty())
      return std::nullopt;
    auto read = llvm::find_if(facts.reads, [&](const auto &r) {
      return r.load == action.reads.front();
    });
    if (read == facts.reads.end())
      return std::nullopt;
    auto tile = read->tile;
    const auto tileKey =
        std::make_pair<int64_t, int64_t>(tile.getCardId(), tile.getTileId());
    if (action.kind == AccessReuseKind::Peer) {
      auto count = executions(*read);
      if (!count || !read->sourceType.getElementType().isIntOrFloat())
        return std::nullopt;
      long double payload =
          static_cast<long double>(read->physicalBytes) * *count;
      long double logicalBytes =
          static_cast<long double>(read->sourceType.getNumElements()) *
          read->sourceType.getElementType().getIntOrFloatBitWidth() / 8 *
          *count;
      auto edges = buildPeerReuseEdges(action.reads, action.peerTopology);
      if (mlir::failed(edges))
        return std::nullopt;
      for (auto edge : *edges) {
        traffic.savedDDRBytes += logicalBytes;
        if (!addPeer(traffic,
                     action.reads[edge.source]->getParentOfType<TileModuleOp>(),
                     action.reads[edge.destination]
                         ->getParentOfType<TileModuleOp>()
                         .getTileId(),
                     payload, *count))
          return std::nullopt;
      }
      continue;
    }
    auto window = findWindow(facts, action, action.scope);
    if (!window)
      return std::nullopt;
    const long double outerExecutions = window->scopeExecutions;
    traffic.savedDDRBytes +=
        (window->readBytes - window->windowBytes) * outerExecutions;
    if (action.kind == AccessReuseKind::Sliding) {
      traffic.copies[tileKey] += window->payloadBytes * outerExecutions;
    } else {
      for (auto load : action.reads) {
        auto access = llvm::find_if(
            facts.reads, [&](const auto &r) { return r.load == load; });
        if (access == facts.reads.end())
          return std::nullopt;
        if (access->acceptsSourceView)
          continue;
        auto count = executions(*access);
        if (!count)
          return std::nullopt;
        traffic.copies[tileKey] +=
            static_cast<long double>(access->physicalBytes) * *count;
      }
    }
    if (action.kind == AccessReuseKind::Sliding) {
      auto slide = llvm::find_if(facts.sliding, [&](const auto &s) {
        return s.access.load == read->load;
      });
      if (slide == facts.sliding.end())
        return std::nullopt;
      auto count = executions(*read);
      if (!count || !outerExecutions)
        return std::nullopt;
      long double trips = *count / outerExecutions;
      long double added = static_cast<long double>(read->physicalBytes) *
                          slide->shift /
                          read->sourceType.getDimSize(slide->axis);
      traffic.copies[tileKey] +=
          2 * (read->physicalBytes - added) * (trips - 1) * outerExecutions;
    } else {
      if (action.shareWindow)
        residents.push_back(window);
      if (action.kind == AccessReuseKind::TwoLevel) {
        const analysis::ScopedReadAccess *inner = nullptr;
        for (const auto &candidate : facts.scopes)
          if (candidate.source == window->source &&
              candidate.scope == action.innerScope)
            inner = &candidate;
        if (!inner)
          return std::nullopt;
        traffic.copies[tileKey] +=
            static_cast<long double>(inner->windowBytes) *
            inner->scopeExecutions;
      }
    }
  }
  if (!residents.empty()) {
    llvm::SmallVector<llvm::SmallVector<const analysis::ScopedReadAccess *, 4>,
                      4>
        groups;
    for (auto window : residents) {
      auto group = llvm::find_if(groups, [&](const auto &group) {
        return equivalentWindows(*group.front(), *window) &&
               !llvm::any_of(group, [&](auto other) {
                 return other->tile == window->tile;
               });
      });
      if (group == groups.end()) {
        groups.emplace_back();
        group = std::prev(groups.end());
      }
      group->push_back(window);
    }
    bool shared = false;
    for (const auto &group : groups)
      for (auto receiver : llvm::ArrayRef(group).drop_front()) {
        auto donor = group.front();
        long double bytes = static_cast<long double>(donor->windowBytes) *
                            donor->scopeExecutions;
        traffic.savedDDRBytes += bytes;
        auto destination = receiver->tile;
        if (!addPeer(traffic, donor->tile, destination.getTileId(), bytes,
                     donor->scopeExecutions))
          return std::nullopt;
        shared = true;
      }
    if (!shared)
      return std::nullopt;
  }
  if (!policy.ddrNominalBytesPerSecond ||
      !policy.directionalNoCBytesPerSecond ||
      !policy.dteEndpointBytesPerSecondEstimate ||
      !policy.spmExplicitMovementBytesPerSecondPerTileEstimate)
    return std::nullopt;
  long double endpoint = 0, startup = 0, link = 0, copy = 0;
  for (const auto &[tile, bytes] : traffic.sends) {
    endpoint = std::max(endpoint, bytes * 1e12L /
                                      policy.dteEndpointBytesPerSecondEstimate);
    startup =
        std::max(startup, policy.dteFirstMessagePicosecondsEstimate +
                              (traffic.messages.at(tile) - 1) *
                                  policy.dteMessageStartupPicosecondsEstimate);
  }
  for (const auto &[edge, bytes] : traffic.links)
    link = std::max(link, bytes * 1e12L / policy.directionalNoCBytesPerSecond);
  for (const auto &[tile, bytes] : traffic.copies)
    copy = std::max(
        copy, bytes * 1e12L /
                  policy.spmExplicitMovementBytesPerSecondPerTileEstimate);
  return traffic.savedDDRBytes * 1e12L / policy.ddrNominalBytesPerSecond -
         endpoint - startup - link - copy -
         traffic.hopMessages * policy.noCHopPicosecondsEstimate;
}

} // namespace

mlir::FailureOr<llvm::SmallVector<BroadcastTreeEdge, 16>>
buildPeerReuseEdges(llvm::ArrayRef<StorageLoadOp> reads,
                    PeerReuseTopology topology) {
  if (reads.size() < 2)
    return mlir::failure();
  llvm::SmallVector<BroadcastTreeEdge, 16> edges;
  if (topology == PeerReuseTopology::Direct) {
    for (size_t i = 1; i < reads.size(); ++i)
      edges.push_back({0, i});
    return edges;
  }
  auto module = reads.front()->getParentOfType<mlir::ModuleOp>();
  auto topologies = module.getOps<TargetTopologyOp>();
  if (topologies.empty())
    return mlir::failure();
  auto grid = (*topologies.begin()).getTileGrid();
  if (grid.size() != 2 || grid[0] <= 0 || grid[1] <= 0)
    return mlir::failure();
  llvm::SmallVector<uint64_t, 16> participants;
  for (size_t i = 0; i < reads.size(); ++i)
    participants.push_back(i);
  return buildMinimumHopBroadcastTree(
      participants, [&](uint64_t a, uint64_t b) -> std::optional<uint64_t> {
        int64_t x = reads[a]->getParentOfType<TileModuleOp>().getTileId();
        int64_t y = reads[b]->getParentOfType<TileModuleOp>().getTileId();
        if (x < 0 || y < 0 || x / grid[1] >= grid[0] || y / grid[1] >= grid[0])
          return std::nullopt;
        return std::abs(x / grid[1] - y / grid[1]) +
               std::abs(x % grid[1] - y % grid[1]);
      });
}

AccessReuseChoice
selectPeerAccessReuse(const analysis::AccessReuseAnalysis &facts) {
  AccessReuseChoice choice;
  for (const auto &group : facts.peers) {
    AccessReuseAction action;
    for (const auto &read : group.reads)
      action.reads.push_back(read.load);
    choice.actions.push_back(std::move(action));
  }
  return choice;
}

AccessReuseProposals
proposeAccessReuse(const analysis::AccessReuseAnalysis &facts,
                   const analysis::SearchCostPolicy &policy) {
  support::ScopedCompileTimingSpan timing("query", "access-reuse",
                                          "profitable-choices");
  AccessReuseProposals result;
  using Source = std::pair<int64_t, int64_t>;
  struct RankedChoice {
    long double benefit, priority;
    Source source;
    unsigned depth;
    AccessReuseChoice choice;
  };
  auto priority = [&](const AccessReuseChoice &choice, long double gain) {
    // Scheduling only: compare saved time per byte of selected working set.
    // There is no capacity threshold and every profitable choice is retained.
    std::map<std::pair<int64_t, int64_t>, long double> windows;
    for (const auto &action : choice.actions)
      for (auto load : action.reads) {
        auto read = llvm::find_if(
            facts.reads, [&](const auto &r) { return r.load == load; });
        if (read == facts.reads.end())
          continue;
        auto tile = read->tile;
        auto key = std::make_pair<int64_t, int64_t>(tile.getCardId(),
                                                    tile.getTileId());
        if (action.kind == AccessReuseKind::Peer) {
          windows[key] += read->physicalBytes;
        } else {
          auto window = findWindow(facts, action, action.scope);
          windows[key] +=
              action.kind == AccessReuseKind::Sliding
                  ? 2 * static_cast<long double>(read->physicalBytes)
                  : window->windowBytes;
          if (action.kind == AccessReuseKind::TwoLevel)
            for (const auto &inner : facts.scopes)
              if (inner.scope == action.innerScope &&
                  inner.source == window->source)
                windows[key] += inner.windowBytes;
          break;
        }
      }
    long double maximum = 1;
    for (const auto &[tile, bytes] : windows)
      maximum = std::max(maximum, bytes);
    return gain / maximum;
  };
  llvm::SmallVector<RankedChoice, 4> ranked;
  auto consider = [&](Source source, AccessReuseChoice choice, unsigned depth) {
    ++result.opportunities;
    auto gain = estimateBenefit(choice, facts, policy);
    if (!gain || !std::isfinite(*gain)) {
      ++result.unknownBenefit;
      return;
    }
    if (*gain <= policy.instructionFixedPicosecondsEstimate) {
      ++result.lowBenefit;
      return;
    }
    ranked.push_back(
        {*gain, priority(choice, *gain), source, depth, std::move(choice)});
  };
  std::map<Source, AccessReuseChoice> peerSources;
  for (const auto &group : facts.peers) {
    auto first = group.reads.front();
    auto &choice = peerSources[{first.tile.getCardId(), first.argument}];
    AccessReuseAction action;
    for (const auto &read : group.reads)
      action.reads.push_back(read.load);
    choice.actions.push_back(std::move(action));
  }
  for (auto &[source, choice] : peerSources) {
    auto tree = choice;
    for (auto &action : tree.actions)
      action.peerTopology = PeerReuseTopology::SpanningTree;
    consider(source, std::move(choice), 0);
    consider(source, std::move(tree), 0);
  }

  // Keep the existing no-residency lifetime alternative alongside promotion.
  // A large aggregate gain from promotion must not starve plain peer reuse.
  AccessReuseChoice peerChoice;
  std::map<Source, const RankedChoice *> bestPeers;
  for (const auto &entry : ranked) {
    auto &best = bestPeers[entry.source];
    if (!best || entry.benefit > best->benefit)
      best = &entry;
  }
  for (const auto &[source, best] : bestPeers)
    llvm::append_range(peerChoice.actions, best->choice.actions);

  using Domain = std::vector<std::tuple<int64_t, int64_t, int64_t>>;
  using ScopeKey = std::tuple<Source, Domain, AccessReuseKind>;
  std::map<ScopeKey, AccessReuseChoice> scoped;
  auto domainKey = [](mlir::scf::ForOp scope) {
    Domain result;
    if (auto loops = analysis::getEnclosingStaticLoopDomains(scope))
      for (const auto &loop : *loops)
        result.push_back(loop.bounds());
    auto own = analysis::getStaticLoopDomain(scope);
    if (own)
      result.push_back(own->bounds());
    return result;
  };
  for (const auto &window : facts.scopes) {
    auto tile = window.tile;
    Source source{tile.getCardId(), window.argument};
    auto &choice =
        scoped[{source, domainKey(window.scope), AccessReuseKind::Resident}];
    choice.actions.push_back(
        {AccessReuseKind::Resident, window.reads, window.scope, {}});
    const analysis::ScopedReadAccess *inner = nullptr;
    for (const auto &candidate : facts.scopes) {
      auto scope = window.scope;
      if (candidate.source != window.source ||
          !scope->isAncestor(candidate.scope) ||
          candidate.windowBytes >= window.windowBytes)
        continue;
      auto innerScope = inner ? inner->scope : mlir::scf::ForOp{};
      if (!inner || candidate.scope->isAncestor(innerScope))
        inner = &candidate;
    }
    if (inner) {
      auto &two =
          scoped[{source, domainKey(window.scope), AccessReuseKind::TwoLevel}];
      two.actions.push_back({AccessReuseKind::TwoLevel, window.reads,
                             window.scope, inner->scope});
    }
  }
  for (const auto &window : facts.sliding) {
    auto tile = window.access.tile;
    auto &choice = scoped[{{tile.getCardId(), window.access.argument},
                           domainKey(window.scope),
                           AccessReuseKind::Sliding}];
    choice.actions.push_back(
        {AccessReuseKind::Sliding, {window.access.load}, window.scope, {}});
  }
  for (auto &[key, choice] : scoped) {
    auto source = std::get<0>(key);
    unsigned depth = std::get<1>(key).size();
    if (std::get<2>(key) != AccessReuseKind::Sliding) {
      auto shared = choice;
      for (auto &action : shared.actions)
        action.shareWindow = true;
      consider(source, std::move(shared), depth);
    }
    consider(source, std::move(choice), depth);
  }
  llvm::stable_sort(ranked, [](const auto &a, const auto &b) {
    if (a.priority != b.priority)
      return a.priority > b.priority;
    if (a.benefit != b.benefit)
      return a.benefit > b.benefit;
    return a.depth > b.depth;
  });
  // One joint proposal takes the best worthwhile choice for each source.
  // There is no input power set, and no requirement to beat the global winner
  // before related sources can be combined.
  AccessReuseChoice combined;
  std::set<Source> sources;
  llvm::DenseSet<mlir::Operation *> combinedReads;
  for (const auto &entry : ranked) {
    if (sources.count(entry.source))
      continue;
    bool conflict = llvm::any_of(entry.choice.actions, [&](const auto &a) {
      return llvm::any_of(
          a.reads, [&](auto read) { return combinedReads.contains(read); });
    });
    if (conflict)
      continue;
    sources.insert(entry.source);
    llvm::append_range(combined.actions, entry.choice.actions);
    for (const auto &action : entry.choice.actions)
      for (auto read : action.reads)
        combinedReads.insert(read);
  }
  if (!peerChoice.actions.empty())
    result.choices.push_back(peerChoice);
  // A temporal choice must be evaluated together with independently useful
  // peer supply for the other resources. This is linear in opportunities,
  // not a power set, and does not require any single-input global win.
  for (const auto &entry : ranked) {
    if (llvm::all_of(entry.choice.actions, [](const auto &action) {
          return action.kind == AccessReuseKind::Peer;
        }))
      continue;
    auto joint = entry.choice;
    for (const auto &action : peerChoice.actions) {
      auto first = action.reads.front();
      auto read = llvm::find_if(facts.reads,
                                [&](const auto &r) { return r.load == first; });
      auto tile = read->tile;
      if (Source{tile.getCardId(), read->argument} != entry.source)
        joint.actions.push_back(action);
    }
    auto gain = estimateBenefit(joint, facts, policy);
    if (gain && *gain > policy.instructionFixedPicosecondsEstimate)
      result.choices.push_back(std::move(joint));
  }
  if (sources.size() > 1 &&
      !llvm::all_of(combined.actions, [](const auto &action) {
        return action.kind == AccessReuseKind::Peer;
      })) {
    auto gain = estimateBenefit(combined, facts, policy);
    if (gain && *gain > policy.instructionFixedPicosecondsEstimate)
      result.choices.push_back(std::move(combined));
  }
  for (auto &entry : ranked) {
    bool duplicate = llvm::any_of(result.choices, [&](const auto &other) {
      return other.actions.size() == entry.choice.actions.size() &&
             llvm::all_of(llvm::zip_equal(other.actions, entry.choice.actions),
                          [](const auto &pair) {
                            const auto &[a, b] = pair;
                            return a.kind == b.kind && a.reads == b.reads &&
                                   a.scope == b.scope &&
                                   a.innerScope == b.innerScope &&
                                   a.shareWindow == b.shareWindow &&
                                   a.peerTopology == b.peerTopology;
                          });
    });
    if (!duplicate)
      result.choices.push_back(std::move(entry.choice));
  }
  return result;
}

mlir::FailureOr<AccessReuseChoice>
mapAccessReuseChoice(const AccessReuseChoice &choice,
                     const mlir::IRMapping &mapping) {
  AccessReuseChoice result;
  for (auto action : choice.actions) {
    for (auto &read : action.reads) {
      auto mapped = mapping.lookupOrNull(read.getOperation());
      read = mlir::dyn_cast_or_null<StorageLoadOp>(mapped);
      if (!read)
        return mlir::failure();
    }
    for (auto *scope : {&action.scope, &action.innerScope}) {
      if (!*scope)
        continue;
      *scope = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
          mapping.lookupOrNull(scope->getOperation()));
      if (!*scope)
        return mlir::failure();
    }
    result.actions.push_back(std::move(action));
  }
  return result;
}

} // namespace wafer::compiler::detail
