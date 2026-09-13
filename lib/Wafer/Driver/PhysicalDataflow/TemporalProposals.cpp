//===- TemporalProposals.cpp - Feedback ordered integer choices --------===//

#include "TemporalProposals.h"

#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Target/PhysicalTensor/PhysicalLayout.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace wafer::compiler::detail {
namespace {
size_t hashChoices(const std::vector<TemporalChoice> &choices) {
  auto hash = llvm::hash_combine(choices.size());
  for (const auto &choice : choices) {
    hash = llvm::hash_combine(hash, static_cast<unsigned>(choice.kind),
                              choice.scopes.size());
    for (const auto &scope : choice.scopes)
      hash = llvm::hash_combine(
          hash,
          llvm::hash_combine_range(scope.iteratorTileSizes.begin(),
                                   scope.iteratorTileSizes.end()),
          llvm::hash_combine_range(scope.loopOrder.begin(),
                                   scope.loopOrder.end()));
  }
  // Hashes only locate buckets. Full equality below still checks the live
  // operation handles; neither hash order nor addresses order proposals.
  return static_cast<size_t>(hash);
}
} // namespace

std::optional<size_t>
TemporalProposals::find(const std::vector<TemporalChoice> &choices) const {
  auto found = entryIndex.find(hashChoices(choices));
  if (found == entryIndex.end())
    return std::nullopt;
  for (size_t index : found->second) {
    support::addCompileCounter("search", "temporal-point-equality-checks", 1);
    if (entries[index].choices == choices)
      return index;
  }
  return std::nullopt;
}

bool TemporalProposals::complete(std::vector<TemporalChoice> &choices) const {
  if (choices.size() != domains.size())
    return false;
  for (auto [domain, choice] : llvm::zip_equal(domains, choices)) {
    auto descriptors = domain->getScopeDescriptors(choice.kind);
    if (descriptors.size() != choice.scopes.size())
      return false;
    for (auto [descriptor, scope] :
         llvm::zip_equal(descriptors, choice.scopes)) {
      auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                               scope.iteratorTileSizes,
                                               descriptor.precedence);
      if (mlir::failed(order))
        return false;
      scope.loopOrder = std::move(*order);
    }
    if (!domain->contains(choice))
      return false;
  }
  return true;
}

bool TemporalProposals::append(std::vector<TemporalChoice> choices,
                               TemporalProposalKind kind,
                               std::optional<Probe> probe, bool preserveOrder) {
  const bool valid =
      preserveOrder ? choices.size() == domains.size() &&
                          llvm::all_of(llvm::zip_equal(domains, choices),
                                       [](auto pair) {
                                         return std::get<0>(pair)->contains(
                                             std::get<1>(pair));
                                       })
                    : complete(choices);
  if (!valid || find(choices))
    return false;
  queues[static_cast<unsigned>(kind)].push_back(entries.size());
  entryIndex[hashChoices(choices)].push_back(entries.size());
  entries.push_back({std::move(choices), std::move(probe), {}, {}});
  return true;
}

bool TemporalProposals::prepareNext(TemporalProposalKind kind) {
  if (!queues[static_cast<unsigned>(kind)].empty())
    return true;
  if (kind == TemporalProposalKind::Repair)
    return appendCapacityDirection();
  if (kind == TemporalProposalKind::Explore && appendSeedPoint())
    return true;
  auto &probes = probeQueues[static_cast<unsigned>(kind)];
  while (true) {
    while (!probes.empty()) {
      auto probe = std::move(probes.front());
      probes.pop_front();
      if (materializeProbe(std::move(probe), kind))
        return true;
    }
    if (kind != TemporalProposalKind::Improve || !advanceImprovementPoll())
      return false;
    if (!queues[static_cast<unsigned>(kind)].empty())
      return true;
  }
}

std::vector<TemporalChoice> TemporalProposals::take(TemporalProposalKind kind) {
  const bool available = prepareNext(kind);
  assert(available && "taking a temporal proposal requires a prepared point");
  (void)available;
  auto &queue = queues[static_cast<unsigned>(kind)];
  size_t index = queue.front();
  queue.pop_front();
  entries[index].taken = true;
  firstCapacityPending.erase(index);
  return entries[index].choices;
}

bool TemporalProposals::visitRaw(const std::vector<TemporalChoice> &choices) {
  if (find(choices))
    return false;
  entryIndex[hashChoices(choices)].push_back(entries.size());
  entries.push_back({choices, {}, {}, {}, true});
  return true;
}

std::vector<TemporalProposals::Coordinate> TemporalProposals::coordinates(
    const std::vector<TemporalChoice> &choices) const {
  std::vector<Coordinate> result;
  for (auto [domainIndex, choice] : llvm::enumerate(choices)) {
    auto descriptors = domains[domainIndex]->getScopeDescriptors(choice.kind);
    for (auto [scopeIndex, descriptor] : llvm::enumerate(descriptors))
      for (auto [iterator, capability] :
           llvm::enumerate(descriptor.iteratorCapabilities))
        if (capability == IteratorTilingCapability::Tileable &&
            descriptor.iterationExtents[iterator] > 1)
          result.push_back({domainIndex, scopeIndex, iterator});
  }
  return result;
}

TemporalSizeInterval
TemporalProposals::bounds(const std::vector<TemporalChoice> &choices,
                          Coordinate coordinate) const {
  const auto &descriptor = domains[coordinate.domain]->getScopeDescriptors(
      choices[coordinate.domain].kind)[coordinate.scope];
  int64_t extent = descriptor.iterationExtents[coordinate.iterator];
  // This is the existing exact reshape domain, not a memory estimate.
  int64_t lower =
      llvm::is_contained(descriptor.exactReshapeDimensions, coordinate.iterator)
          ? extent / 2 + 1
          : 1;
  return {lower, extent};
}

int64_t &TemporalProposals::value(std::vector<TemporalChoice> &choices,
                                  Coordinate coordinate) {
  return choices[coordinate.domain]
      .scopes[coordinate.scope]
      .iteratorTileSizes[coordinate.iterator];
}

int64_t TemporalProposals::expandDistance(int64_t distance) {
  return distance > std::numeric_limits<int64_t>::max() / 2
             ? std::numeric_limits<int64_t>::max()
             : 2 * distance;
}

void TemporalProposals::appendProbe(Probe probe, TemporalProposalKind kind) {
  probeQueues[static_cast<unsigned>(kind)].push_back(std::move(probe));
}

bool TemporalProposals::materializeProbe(Probe probe,
                                         TemporalProposalKind kind) {
  auto next = entries[probe.anchor].choices;
  support::addCompileCounter("search", "temporal-neighbor-materializations", 1);
  bool changed = false;
  for (auto coordinate : probe.coordinates) {
    auto interval = bounds(next, coordinate);
    int64_t &size = value(next, coordinate);
    int64_t room =
        probe.direction < 0 ? size - interval.lower : interval.upper - size;
    int64_t step = std::min(probe.distance, room);
    size += probe.direction * step;
    changed |= step != 0;
  }
  return changed && append(std::move(next), kind, std::move(probe));
}

void TemporalProposals::seed(const std::vector<TemporalChoice> &initial,
                             size_t stratum) {
  SeedFamily family{initial, coordinates(initial), {}};
  size_t levels = 0;
  for (Coordinate coordinate : family.coordinates) {
    if (family.largest.empty() ||
        family.largest.back().domain != coordinate.domain ||
        family.largest.back().scope != coordinate.scope)
      family.largest.push_back(coordinate);
    else if (bounds(initial, coordinate).upper >
             bounds(initial, family.largest.back()).upper)
      family.largest.back() = coordinate;
    const auto interval = bounds(initial, coordinate);
    int64_t size = initial[coordinate.domain]
                       .scopes[coordinate.scope]
                       .iteratorTileSizes[coordinate.iterator];
    size_t depth = 0;
    while (size > interval.lower) {
      size = interval.lower + (size - interval.lower) / 2;
      ++depth;
    }
    levels = std::max(levels, depth);
  }
  const size_t index = seedFamilies.size();
  seedFamilies.push_back(std::move(family));
  const bool independent = llvm::all_of(initial, [](const auto &choice) {
    return choice.kind == TemporalTraversalKind::Independent;
  });
  std::deque<SeedPoint> fresh{{index, 0, SeedVariant::Full}};
  if (!independent)
    fresh.push_back({index, 0, SeedVariant::Kernel});
  const std::array variants{SeedVariant::Largest, SeedVariant::Coupled,
                            SeedVariant::All};
  for (size_t offset = 0; offset < levels; ++offset) {
    const size_t level = (stratum % levels + offset) % levels + 1;
    for (size_t variant = 0; variant < variants.size(); ++variant)
      fresh.push_back(
          {index, level,
           variants[(stratum % variants.size() + variant) % variants.size()]});
  }
  if (independent && fresh.size() > 1) {
    fresh.push_back(fresh.front());
    fresh.pop_front();
  }
  if (independent)
    fresh.push_back({index, 0, SeedVariant::Kernel});
  // Interleave the families without constructing any unvisited full tuple.
  std::deque<SeedPoint> prior;
  prior.swap(seedPoints);
  while (!prior.empty() || !fresh.empty()) {
    if (!prior.empty()) {
      seedPoints.push_back(prior.front());
      prior.pop_front();
    }
    if (!fresh.empty()) {
      seedPoints.push_back(fresh.front());
      fresh.pop_front();
    }
  }
}

bool TemporalProposals::appendSeedPoint() {
  while (!seedPoints.empty()) {
    const auto point = seedPoints.front();
    seedPoints.pop_front();
    const auto &family = seedFamilies[point.family];
    auto next = family.initial;
    support::addCompileCounter("search", "temporal-seed-materializations", 1);
    if (point.variant == SeedVariant::Kernel) {
      bool changed = false;
      for (auto [domainIndex, choice] : llvm::enumerate(next)) {
        auto descriptors =
            domains[domainIndex]->getScopeDescriptors(choice.kind);
        for (auto [scopeIndex, descriptor] : llvm::enumerate(descriptors)) {
          int64_t extent = 1;
          for (auto [size, capability] :
               llvm::zip_equal(descriptor.iterationExtents,
                               descriptor.iteratorCapabilities))
            if (capability == IteratorTilingCapability::FullExtentOnly)
              extent = std::max(extent, size);
          if (extent == 1)
            continue;
          for (auto [iterator, capability] :
               llvm::enumerate(descriptor.iteratorCapabilities)) {
            if (capability != IteratorTilingCapability::Tileable)
              continue;
            Coordinate coordinate{domainIndex, scopeIndex, iterator};
            int64_t &size = value(next, coordinate);
            const int64_t proposed = std::max(bounds(next, coordinate).lower,
                                              std::min(size, extent));
            changed |= size != proposed;
            size = proposed;
          }
        }
      }
      if (!changed)
        continue;
    } else if (point.variant != SeedVariant::Full) {
      const auto &selected = point.variant == SeedVariant::Largest
                                 ? family.largest
                                 : family.coordinates;
      for (Coordinate coordinate : selected) {
        const auto interval = bounds(next, coordinate);
        int64_t &size = value(next, coordinate);
        for (size_t level = 0; level < point.level; ++level)
          size = interval.lower + (size - interval.lower) / 2;
      }
    }
    if (point.variant == SeedVariant::Coupled ||
        point.variant == SeedVariant::Kernel) {
      if (!complete(next))
        continue;
      for (auto [domain, choice] : llvm::zip_equal(domains, next))
        if (auto coupled = domain->getCoupledStateProposal(choice))
          choice = std::move(*coupled);
    }
    if (append(std::move(next), TemporalProposalKind::Explore))
      return true;
  }
  return false;
}

void TemporalProposals::observeAccepted(
    const std::vector<TemporalChoice> &choices, uint64_t duration) {
  auto index = find(choices);
  if (!index)
    return;
  auto &entry = entries[*index];
  if (entry.bestDuration && duration >= *entry.bestDuration)
    return;
  const bool first = !entry.bestDuration;
  entry.bestDuration = duration;
  if (entry.probe) {
    Probe probe = *entry.probe; // Appending can relocate entries.
    if (probe.referenceDuration && duration < *probe.referenceDuration) {
      probe.anchor = *index;
      probe.distance = expandDistance(probe.distance);
      probe.referenceDuration = duration;
      appendProbe(std::move(probe), TemporalProposalKind::Improve);
    } else if (probe.distance > 1) {
      probe.distance /= 2;
      appendProbe(std::move(probe), TemporalProposalKind::Improve);
    }
  }
  if (!first)
    return;
  auto all = coordinates(choices);
  ImprovementPoll poll{*index, all, {}};
  for (const auto coordinate : all) {
    if (poll.groups.empty() ||
        poll.groups.back().front().domain != coordinate.domain ||
        poll.groups.back().front().scope != coordinate.scope)
      poll.groups.emplace_back();
    poll.groups.back().push_back(coordinate);
  }
  improvementPolls.push_back(std::move(poll));
  // A combination may improve even when every one-coordinate move worsens.
  // Its admission is independent of the single-coordinate observations.
  for (int direction : {-1, 1})
    appendProbe({*index, all, direction, 1, duration},
                TemporalProposalKind::Explore);
}

bool TemporalProposals::advanceImprovementPoll() {
  while (!improvementPolls.empty()) {
    auto poll = std::move(improvementPolls.front());
    improvementPolls.pop_front();
    const auto &anchor = entries[poll.anchor].choices;
    size_t maximumAxes = 0;
    for (const auto &group : poll.groups)
      maximumAxes = std::max(maximumAxes, group.size());
    std::vector<std::pair<Coordinate, int>> changes;
    bool preserveOrder = false;
    std::optional<std::vector<TemporalChoice>> next;
    if (poll.batch < 2 * (maximumAxes + 1)) {
      const int direction = poll.batch % 2 == 0 ? -1 : 1;
      const size_t axis = poll.batch++ / 2;
      if (!axis)
        for (auto coordinate : poll.coordinates)
          changes.emplace_back(coordinate, direction);
      else
        for (const auto &group : poll.groups)
          changes.emplace_back(group[(axis - 1) % group.size()], direction);
    } else {
      bool selected = false;
      for (size_t offset = 0; offset < 4 && !selected; ++offset) {
        const size_t phase = poll.phase;
        poll.phase = (poll.phase + 1) % 4;
        if (phase == 0 && poll.single < 2 * poll.coordinates.size()) {
          changes.emplace_back(poll.coordinates[poll.single / 2],
                               poll.single % 2 == 0 ? -1 : 1);
          ++poll.single;
          selected = true;
        } else if (phase == 1 && poll.fine < poll.coordinates.size()) {
          auto coordinate = poll.coordinates[poll.fine++];
          const uint64_t duration = *entries[poll.anchor].bestDuration;
          for (int direction : {-1, 1})
            appendProbe({poll.anchor, {coordinate}, direction, 1, duration},
                        TemporalProposalKind::Improve);
          appendLayoutBoundaries(poll.anchor, coordinate, duration);
          selected = true;
        } else if (phase == 2 && poll.pairSecond < poll.coordinates.size()) {
          const int direction = poll.pairDirection++ == 0 ? -1 : 1;
          changes.emplace_back(poll.coordinates[poll.pairFirst], direction);
          changes.emplace_back(poll.coordinates[poll.pairSecond], -direction);
          if (poll.pairDirection == 2) {
            poll.pairDirection = 0;
            if (++poll.pairSecond == poll.coordinates.size())
              poll.pairSecond = ++poll.pairFirst + 1;
          }
          selected = true;
        } else if (phase == 3) {
          while (poll.orderDomain < anchor.size()) {
            const auto &scopes = anchor[poll.orderDomain].scopes;
            if (poll.orderScope == scopes.size()) {
              ++poll.orderDomain;
              poll.orderScope = poll.orderPosition = 0;
              continue;
            }
            const auto &order = scopes[poll.orderScope].loopOrder;
            if (poll.orderPosition + 1 >= order.size()) {
              ++poll.orderScope;
              poll.orderPosition = 0;
              continue;
            }
            next = anchor;
            auto &changedOrder =
                (*next)[poll.orderDomain].scopes[poll.orderScope].loopOrder;
            std::swap(changedOrder[poll.orderPosition],
                      changedOrder[poll.orderPosition + 1]);
            ++poll.orderPosition;
            preserveOrder = selected = true;
            break;
          }
        }
      }
      if (!selected)
        continue;
    }
    if (!changes.empty()) {
      next = anchor;
      for (auto [coordinate, direction] : changes) {
        int64_t &size = value(*next, coordinate);
        auto interval = bounds(*next, coordinate);
        const int64_t room =
            direction < 0 ? size - interval.lower : interval.upper - size;
        size += direction * std::min(room, std::max<int64_t>(1, size / 2));
      }
    }
    improvementPolls.push_back(std::move(poll));
    if (next) {
      support::addCompileCounter("search", "temporal-neighbor-materializations",
                                 1);
      append(std::move(*next), TemporalProposalKind::Improve, {},
             preserveOrder);
    }
    // A lazy poll step either queued one complete point or a few local fine
    // directions. Duplicates do not terminate the remaining poll.
    return true;
  }
  return false;
}

void TemporalProposals::appendLayoutBoundaries(size_t anchor,
                                               Coordinate coordinate,
                                               uint64_t duration) {
  const auto &choices = entries[anchor].choices;
  const auto &scope = choices[coordinate.domain].scopes[coordinate.scope];
  auto operation = mlir::dyn_cast<mlir::linalg::LinalgOp>(scope.operation);
  if (!operation)
    return;
  const int64_t current = scope.iteratorTileSizes[coordinate.iterator];
  const auto interval = bounds(choices, coordinate);
  // The blocked-layout alternative's C axis must be an actual projected
  // iterator. Affine/nonprojected accesses do not acquire an invented axis.
  // This target geometry orders proposals; it does not select that layout.
  for (mlir::OpOperand &operand : operation->getOpOperands()) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(operand.get().getType());
    if (!type || !type.hasStaticShape() || type.getRank() == 0 ||
        !type.getElementType().isIntOrFloat())
      continue;
    auto map = operation.getMatchingIndexingMap(&operand);
    if (!map || map.getNumResults() != type.getRank())
      continue;
    auto last = mlir::dyn_cast<mlir::AffineDimExpr>(
        map.getResult(map.getNumResults() - 1));
    if (!last || last.getPosition() != coordinate.iterator)
      continue;
    auto geometry = computePhysicalTensorGeometry(
        type.getShape(), type.getElementTypeBitWidth(),
        type.getElementType().isInteger(8), PhysicalTensorLayout::Cx, {});
    if (!geometry || geometry->cBlock <= 1)
      continue;
    int64_t lower = current - current % geometry->cBlock;
    for (int64_t base : {lower, lower <= interval.upper - geometry->cBlock
                                    ? lower + geometry->cBlock
                                    : lower})
      for (int64_t offset : {-1, 0, 1}) {
        if ((offset < 0 && base == 0) ||
            (offset > 0 && base == std::numeric_limits<int64_t>::max()))
          continue;
        int64_t point = base + offset;
        if (point < interval.lower || point > interval.upper ||
            point == current)
          continue;
        appendProbe({anchor,
                     {coordinate},
                     point < current ? -1 : 1,
                     point < current ? current - point : point - current,
                     duration},
                    TemporalProposalKind::Improve);
      }
  }
}

bool TemporalProposals::observeCapacity(
    const std::vector<TemporalChoice> &choices,
    const std::set<TemporalCoordinate> &affectedCoordinates) {
  auto index = find(choices);
  if (!index)
    return false;
  bool fresh = false;
  const auto available = coordinates(choices);
  for (Coordinate coordinate : available)
    if (affectedCoordinates.count(coordinate) &&
        entries[*index].capacityObserved.insert(coordinate).second)
      fresh = true;
  if (!fresh)
    return false;
  CapacityPoll poll{*index, {}, {}};
  std::optional<std::pair<size_t, size_t>> previousScope;
  for (Coordinate coordinate : entries[*index].capacityObserved) {
    const int64_t current = choices[coordinate.domain]
                                .scopes[coordinate.scope]
                                .iteratorTileSizes[coordinate.iterator];
    if (current <= bounds(choices, coordinate).lower)
      continue;
    auto scope = std::make_pair(coordinate.domain, coordinate.scope);
    if (previousScope != scope) {
      poll.groups.emplace_back();
      previousScope = scope;
    }
    poll.groups.back().push_back(coordinate);
    poll.coordinates.push_back(coordinate);
  }
  if (poll.coordinates.empty())
    return false;
  if (!firstCapacityAnchor) {
    firstCapacityAnchor = *index;
    poll.initialRound = true;
  }
  capacityPolls.push_back(std::move(poll));
  return true;
}

bool TemporalProposals::appendCapacityDirection() {
  while (!capacityPolls.empty()) {
    auto poll = std::move(capacityPolls.front());
    capacityPolls.pop_front();
    size_t maximumAxes = 0;
    for (const auto &group : poll.groups)
      maximumAxes = std::max(maximumAxes, group.size());
    std::vector<Coordinate> selected;
    const bool initialDirection =
        poll.initialRound && poll.batch <= maximumAxes;
    if (poll.batch == 0) {
      // One proposal can repair all independently certified Tile scopes.
      selected = poll.coordinates;
      ++poll.batch;
    } else if (poll.batch <= maximumAxes) {
      for (const auto &group : poll.groups)
        selected.push_back(group[(poll.batch - 1) % group.size()]);
      ++poll.batch;
    } else if (poll.single < poll.coordinates.size()) {
      selected.push_back(poll.coordinates[poll.single++]);
    } else if (poll.pairSecond < poll.coordinates.size()) {
      selected.push_back(poll.coordinates[poll.pairFirst]);
      selected.push_back(poll.coordinates[poll.pairSecond]);
      if (++poll.pairSecond == poll.coordinates.size())
        poll.pairSecond = ++poll.pairFirst + 1;
    } else {
      continue;
    }
    if (poll.initialRound && poll.batch > maximumAxes)
      firstCapacityRoundComplete = true;
    // Every direction starts at its own immutable parent point. Selecting
    // another axis must not inherit the earlier sibling's smaller dimensions.
    auto next = entries[poll.anchor].choices;
    support::addCompileCounter("search", "temporal-neighbor-materializations",
                               1);
    for (Coordinate coordinate : selected) {
      int64_t &size = value(next, coordinate);
      size = std::max(bounds(next, coordinate).lower, size / 2);
    }
    // Appending new feedback never displaces the unvisited sibling directions.
    capacityPolls.push_back(std::move(poll));
    auto previous = find(next);
    const bool queued = append(std::move(next), TemporalProposalKind::Repair);
    if (initialDirection) {
      const auto index =
          queued ? std::optional<size_t>(entries.size() - 1) : previous;
      if (index && !entries[*index].taken)
        firstCapacityPending.insert(*index);
    }
    if (queued)
      return true;
  }
  return false;
}

} // namespace wafer::compiler::detail
