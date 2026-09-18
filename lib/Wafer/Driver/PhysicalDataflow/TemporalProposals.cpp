//===- TemporalProposals.cpp - Feedback ordered integer choices --------===//

#include "TemporalProposals.h"

#include "Wafer/Planning/PhysicalDataflow/OperandReuse.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Target/PhysicalTensor/PhysicalLayout.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <numeric>

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

bool prepareScopePair(llvm::ArrayRef<TemporalCoordinate> coordinates,
                      size_t &first, size_t &second) {
  // Coordinates are contiguous by domain/scope. Keep local axis exchanges
  // and use the separate grouped directions for cross-scope coordination.
  // Enumerating every pair of unrelated scope instances grows quadratically
  // with the number of spatial pieces without defining another local move.
  while (first + 1 < coordinates.size()) {
    if (second < coordinates.size() &&
        coordinates[first].domain == coordinates[second].domain &&
        coordinates[first].scope == coordinates[second].scope)
      return true;
    second = ++first + 1;
  }
  return false;
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
                               TemporalProposalKind kind, bool preserveOrder) {
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
  entries.push_back({std::move(choices), {}, {}});
  return true;
}

bool TemporalProposals::prepareNext(TemporalProposalKind kind) {
  if (!queues[static_cast<unsigned>(kind)].empty())
    return true;
  if (kind == TemporalProposalKind::Repair)
    return appendCapacityDirection();
  if (kind == TemporalProposalKind::Explore && appendSeedPoint())
    return true;
  if (kind != TemporalProposalKind::Improve)
    return false;
  while (advanceImprovementPoll())
    if (!queues[static_cast<unsigned>(kind)].empty())
      return true;
  return false;
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
  entries.push_back({choices, {}, {}, true});
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
  auto descriptors = domains[coordinate.domain]->getScopeDescriptors(
      choices[coordinate.domain].kind);
  const auto &descriptor = descriptors[coordinate.scope];
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

void TemporalProposals::rankGroups(const std::vector<TemporalChoice> &choices,
                                   std::vector<std::vector<Coordinate>> &groups,
                                   bool shrinking) const {
  for (auto &group : groups) {
    const auto first = group.front();
    const auto &scope = choices[first.domain].scopes[first.scope];
    auto descriptors =
        domains[first.domain]->getScopeDescriptors(choices[first.domain].kind);
    auto operands = getReadOperandProjections(
        scope.operation, descriptors[first.scope].iterationExtents);
    llvm::SmallVector<uint64_t> reuse(scope.iteratorTileSizes.size(), 0);
    if (operands)
      for (const auto &operand : *operands)
        for (size_t axis = 0; axis < reuse.size(); ++axis)
          if (axis < operand.iterators.size() && !operand.iterators.test(axis))
            reuse[axis] = llvm::SaturatingAdd(reuse[axis], operand.bytes);
    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(scope.operation);
    auto roles = tiling ? tiling.getLoopIteratorTypes()
                        : llvm::SmallVector<mlir::utils::IteratorType>{};
    auto roleRank = [&](Coordinate c) {
      if (c.iterator >= roles.size())
        return 2;
      return roles[c.iterator] == mlir::utils::IteratorType::parallel ? 0 : 1;
    };
    // Missing evidence leaves the ordinary size/axis order intact. These
    // logical bytes never describe an allocation or authorize a rewrite.
    llvm::sort(group, [&](Coordinate a, Coordinate b) {
      if (roleRank(a) != roleRank(b))
        return roleRank(a) < roleRank(b);
      if (reuse[a.iterator] != reuse[b.iterator])
        return shrinking ? reuse[a.iterator] < reuse[b.iterator]
                         : reuse[a.iterator] > reuse[b.iterator];
      auto sizeA = scope.iteratorTileSizes[a.iterator];
      auto sizeB = scope.iteratorTileSizes[b.iterator];
      if (sizeA != sizeB)
        return sizeA > sizeB;
      return a < b;
    });
  }
}

void TemporalProposals::seed(const std::vector<TemporalChoice> &initial) {
  SeedFamily family{initial, coordinates(initial), {}};
  size_t levels = 0;
  for (Coordinate coordinate : family.coordinates) {
    if (family.groups.empty() ||
        family.groups.back().front().domain != coordinate.domain ||
        family.groups.back().front().scope != coordinate.scope)
      family.groups.emplace_back();
    family.groups.back().push_back(coordinate);
    const auto interval = bounds(initial, coordinate);
    int64_t size = initial[coordinate.domain]
                       .scopes[coordinate.scope]
                       .iteratorTileSizes[coordinate.iterator];
    size_t depth = 0;
    while (size > interval.lower) {
      size = std::max(interval.lower, size / 2);
      ++depth;
    }
    levels = std::max(levels, depth);
  }
  rankGroups(initial, family.groups, true);
  size_t maximumAxes = 0;
  for (const auto &group : family.groups)
    maximumAxes = std::max(maximumAxes, group.size());
  const size_t index = seedFamilies.size();
  seedFamilies.push_back(std::move(family));
  const bool independent = llvm::all_of(initial, [](const auto &choice) {
    return choice.kind == TemporalTraversalKind::Independent;
  });
  std::deque<SeedPoint> fresh{{index, 0, SeedVariant::Full}};
  if (!independent)
    fresh.push_back({index, 0, SeedVariant::Kernel});
  // Every structural family starts at the same coarse scale. The source
  // access relation orders axes; a structural ordinal cannot shrink them.
  for (size_t level = 1; level <= levels; ++level) {
    for (size_t axis = 0; axis < maximumAxes; ++axis)
      fresh.push_back({index, level, SeedVariant::Axis, axis});
    fresh.push_back({index, level, SeedVariant::Coupled});
    fresh.push_back({index, level, SeedVariant::All});
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
      std::vector<Coordinate> selected;
      if (point.variant == SeedVariant::Axis) {
        for (const auto &group : family.groups)
          selected.push_back(group[point.axis % group.size()]);
      } else {
        selected = family.coordinates;
      }
      for (Coordinate coordinate : selected) {
        const auto interval = bounds(next, coordinate);
        int64_t &size = value(next, coordinate);
        for (size_t level = 0; level < point.level; ++level)
          size = std::max(interval.lower, size / 2);
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
    const std::vector<TemporalChoice> &choices,
    const analysis::KnownSearchObjective &objective) {
  auto index = find(choices);
  if (!index)
    return;
  auto &entry = entries[*index];
  if (entry.bestObjective &&
      analysis::compareSearchObjectives(objective, *entry.bestObjective) !=
          analysis::SearchObjectiveComparison::Better)
    return;
  entry.bestObjective = objective;
  if (bestObjective &&
      analysis::compareSearchObjectives(objective, *bestObjective) !=
          analysis::SearchObjectiveComparison::Better)
    return;
  bestObjective = objective;
  bestEntry = *index;
  // Feedback selects the next round's anchor. It never restarts the current
  // round or a completed scale, including after a late seed/capacity result.
  if (!performanceStarted) {
    performanceStarted = true;
    initializeScales(choices);
  }
}

void TemporalProposals::initializeScales(
    const std::vector<TemporalChoice> &choices) {
  for (auto [domainIndex, domain] : llvm::enumerate(domains))
    for (auto kind :
         {TemporalTraversalKind::Joint, TemporalTraversalKind::Independent}) {
      auto descriptors = domain->getScopeDescriptors(kind);
      auto probe = choices;
      probe[domainIndex].kind = kind;
      probe[domainIndex].scopes.clear();
      for (const auto &descriptor : descriptors)
        probe[domainIndex].scopes.push_back(
            {descriptor.operation, descriptor.iterationExtents, {}});
      for (auto [scopeIndex, descriptor] : llvm::enumerate(descriptors)) {
        auto &axes = scales[descriptor.operation];
        axes.resize(std::max(axes.size(), descriptor.iterationExtents.size()));
        for (auto [axis, capability] :
             llvm::enumerate(descriptor.iteratorCapabilities)) {
          if (capability != IteratorTilingCapability::Tileable ||
              descriptor.iterationExtents[axis] <= 1 || axes[axis].step)
            continue;
          int64_t base = descriptor.iterationExtents[axis];
          for (const auto &scope : choices[domainIndex].scopes)
            if (scope.operation == descriptor.operation)
              base = scope.iteratorTileSizes[axis];
          auto &scale = axes[axis];
          scale.alignment =
              getAlignment(probe, {domainIndex, scopeIndex, axis});
          const int64_t grain = scale.alignment.value_or(1);
          scale.step = std::max(grain, base / 2);
          if (scale.alignment)
            scale.step = grain * llvm::bit_floor(uint64_t(scale.step / grain));
          for (int64_t step = scale.step; step >= grain; step /= 2)
            ++scale.rounds;
          performanceRounds = std::max(performanceRounds, scale.rounds);
        }
      }
    }
}

void TemporalProposals::beginImprovementPoll() {
  const auto &anchor = entries[*bestEntry].choices;
  ImprovementPoll poll{*bestEntry, {}};
  auto all = coordinates(anchor);
  std::vector<std::vector<Coordinate>> local;
  for (auto coordinate : all) {
    if (local.empty() || local.back().front().domain != coordinate.domain ||
        local.back().front().scope != coordinate.scope)
      local.emplace_back();
    local.back().push_back(coordinate);
  }
  rankGroups(anchor, local, false);
  // Only an exact shared program-input window supports a cross-scope group.
  // An invariant coordinate does not change this read window. Shape alone is
  // never enough; each coordinate belongs to at most one such group.
  using GroupKey =
      std::tuple<mlir::Operation *, uint64_t, uint64_t, std::vector<int64_t>,
                 std::vector<int64_t>, unsigned, int64_t>;
  std::map<GroupKey, size_t> groupIndex;
  std::vector<std::vector<Coordinate>> shared;
  size_t orderDirections = 0;
  for (const auto &group : local) {
    const auto first = group.front();
    const auto &scope = anchor[first.domain].scopes[first.scope];
    auto descriptor = domains[first.domain]->getScopeDescriptors(
        anchor[first.domain].kind)[first.scope];
    auto reads =
        getReadOperandProjections(scope.operation, descriptor.iterationExtents);
    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(scope.operation);
    auto tile = scope.operation->getParentOfType<TileModuleOp>();
    auto roles = tiling ? tiling.getLoopIteratorTypes()
                        : llvm::SmallVector<mlir::utils::IteratorType>{};
    orderDirections += scope.loopOrder.empty() ? 0 : scope.loopOrder.size() - 1;
    for (auto coordinate : group) {
      poll.directions.push_back({{coordinate, 1}});
      poll.directions.push_back({{coordinate, -1}});
      if (!reads || !tile || coordinate.iterator >= roles.size())
        continue;
      for (const auto &read : *reads) {
        if (!read.programArgument ||
            coordinate.iterator >= read.iterators.size() ||
            read.iterators.test(coordinate.iterator))
          continue;
        GroupKey key{scope.operation->getParentOfType<mlir::ModuleOp>(),
                     tile.getCardId(),
                     *read.programArgument,
                     {read.offsets.begin(), read.offsets.end()},
                     {read.sizes.begin(), read.sizes.end()},
                     static_cast<unsigned>(roles[coordinate.iterator]),
                     descriptor.iterationExtents[coordinate.iterator]};
        auto [it, inserted] = groupIndex.emplace(std::move(key), shared.size());
        if (inserted)
          shared.emplace_back();
        shared[it->second].push_back(coordinate);
        break;
      }
    }
    // Adjacent ranked axes sharing an actual read projection provide a
    // linear set of exchanges, rather than every pair of local axes.
    if (reads)
      for (size_t axis = 1; axis < group.size(); ++axis) {
        auto a = group[axis - 1], b = group[axis];
        if (!llvm::any_of(*reads, [&](const auto &read) {
              return a.iterator < read.iterators.size() &&
                     b.iterator < read.iterators.size() &&
                     (read.iterators.test(a.iterator) ||
                      read.iterators.test(b.iterator));
            }))
          continue;
        poll.directions.push_back({{a, -1}, {b, 1}});
        poll.directions.push_back({{a, 1}, {b, -1}});
      }
  }
  size_t groups = 0;
  for (const auto &group : shared) {
    if (!llvm::any_of(group, [&](Coordinate c) {
          return c.domain != group.front().domain ||
                 c.scope != group.front().scope;
        }))
      continue;
    ++groups;
    for (int direction : {-1, 1}) {
      std::vector<std::pair<Coordinate, int>> changes;
      for (auto coordinate : group)
        changes.emplace_back(coordinate, direction);
      poll.directions.push_back(std::move(changes));
    }
  }
  support::addCompileCounter("search", "performance-rounds", 1);
  support::addCompileCounter("search", "performance-candidate-bound",
                             8 * all.size() + 4 * groups + 2 * orderDirections);
  improvementPoll = std::move(poll);
}

bool TemporalProposals::changeSize(std::vector<TemporalChoice> &choices,
                                   Coordinate coordinate, int direction) const {
  auto &scope = choices[coordinate.domain].scopes[coordinate.scope];
  const auto &scale = scales.find(scope.operation)->second[coordinate.iterator];
  if (performanceRound >= scale.rounds)
    return false;
  const int64_t step = scale.step >> performanceRound;
  const auto interval = bounds(choices, coordinate);
  int64_t &size = value(choices, coordinate);
  const bool wasFull = size == interval.upper;
  if (scale.alignment) {
    const int64_t grain = *scale.alignment;
    const int64_t remainder = size % grain;
    const int64_t nearest =
        direction < 0 ? (remainder ? remainder : grain) : grain - remainder;
    const int64_t distance = nearest + step - grain;
    const int64_t room =
        direction < 0 ? size - interval.lower : interval.upper - size;
    if (distance > room)
      return false;
    size += direction * distance;
  } else {
    const int64_t room =
        direction < 0 ? size - interval.lower : interval.upper - size;
    const int64_t target = size + direction * std::min(room, step);
    std::optional<int64_t> selected;
    for (int64_t candidate = interval.upper;;
         candidate = std::max(interval.lower, candidate / 2)) {
      if ((direction < 0 ? candidate < size : candidate > size) &&
          (!selected ||
           std::abs(candidate - target) < std::abs(*selected - target)))
        selected = candidate;
      if (candidate == interval.lower)
        break;
    }
    if (!selected)
      return false;
    size = *selected;
  }
  if (wasFull != (size == interval.upper)) {
    auto descriptor = domains[coordinate.domain]->getScopeDescriptors(
        choices[coordinate.domain].kind)[coordinate.scope];
    auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                             scope.iteratorTileSizes,
                                             descriptor.precedence);
    if (mlir::failed(order))
      return false;
    scope.loopOrder = std::move(*order);
  }
  return true;
}

bool TemporalProposals::advanceImprovementPoll() {
  while (performanceRound < performanceRounds) {
    if (!improvementPoll)
      beginImprovementPoll();
    auto &poll = *improvementPoll;
    const auto anchor = entries[poll.anchor].choices;
    auto next = anchor;
    if (poll.position < poll.directions.size()) {
      bool changed = false;
      for (auto [coordinate, direction] : poll.directions[poll.position++])
        changed |= changeSize(next, coordinate, direction);
      if (!changed)
        continue;
    } else {
      bool selected = false;
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
        auto &changedOrder =
            next[poll.orderDomain].scopes[poll.orderScope].loopOrder;
        std::swap(changedOrder[poll.orderPosition],
                  changedOrder[poll.orderPosition + 1]);
        ++poll.orderPosition;
        selected = true;
        break;
      }
      if (!selected) {
        improvementPoll.reset();
        ++performanceRound;
        continue;
      }
    }
    support::addCompileCounter("search", "temporal-neighbor-materializations",
                               1);
    // Preserve the chosen loop order in untouched scopes. The exact domain
    // may append one relation-coordinated alternative, not a second search.
    auto coordinated = next;
    bool coupled = false;
    for (auto [index, domain] : llvm::enumerate(domains))
      if (auto choice = domain->getCoupledStateProposal(coordinated[index],
                                                        &anchor[index])) {
        coordinated[index] = std::move(*choice);
        coupled = true;
      }
    bool added = append(std::move(next), TemporalProposalKind::Improve, true);
    if (coupled)
      added |=
          append(std::move(coordinated), TemporalProposalKind::Improve, true);
    if (added)
      return true;
  }
  return false;
}

std::optional<int64_t>
TemporalProposals::getAlignment(const std::vector<TemporalChoice> &choices,
                                Coordinate coordinate) const {
  const auto &scope = choices[coordinate.domain].scopes[coordinate.scope];
  auto operation = mlir::dyn_cast<mlir::linalg::LinalgOp>(scope.operation);
  if (!operation)
    return std::nullopt;
  int64_t alignment = 1;
  // The interface's actual projected channel axes supply target geometry.
  // This is a proposal granularity, not a selected layout or legality proof.
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
    int64_t factor = geometry->cBlock / std::gcd(alignment, geometry->cBlock);
    if (alignment > std::numeric_limits<int64_t>::max() / factor)
      return std::nullopt;
    alignment *= factor;
  }
  return alignment > 1 ? std::optional<int64_t>(alignment) : std::nullopt;
}

bool TemporalProposals::observeCapacity(
    const std::vector<TemporalChoice> &choices,
    const std::set<TemporalCoordinate> &affectedCoordinates, bool prioritize) {
  auto index = find(choices);
  if (!index)
    return false;
  bool fresh = false;
  const auto available = coordinates(choices);
  for (Coordinate coordinate : available)
    if (affectedCoordinates.count(coordinate) &&
        entries[*index].capacityObserved.insert(coordinate).second)
      fresh = true;
  if (!fresh) {
    if (!prioritize || !llvm::any_of(capacityPolls, [&](const auto &poll) {
          return poll.anchor == *index;
        }))
      return false;
    priorityCapacityAnchor = *index;
    return appendCapacityDirection();
  }
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
  rankGroups(choices, poll.groups, true);
  if (!firstCapacityAnchor) {
    firstCapacityAnchor = *index;
    poll.initialRound = true;
  }
  capacityPolls.push_back(std::move(poll));
  if (prioritize) {
    priorityCapacityAnchor = *index;
    return appendCapacityDirection();
  }
  return true;
}

bool TemporalProposals::appendCapacityDirection() {
  while (!capacityPolls.empty()) {
    auto selectedPoll = capacityPolls.begin();
    if (priorityCapacityAnchor) {
      auto found = llvm::find_if(capacityPolls, [&](const auto &poll) {
        return poll.anchor == *priorityCapacityAnchor;
      });
      if (found != capacityPolls.end())
        selectedPoll = found;
      else
        priorityCapacityAnchor.reset();
    }
    if (!priorityCapacityAnchor && preferFreshCapacity) {
      // Advance the deepest observed repair chain alongside older siblings.
      // Arrival order alone lets a later ordinary/shallow failure repeatedly
      // displace the next step of an already advancing capacity repair.
      for (auto candidate = capacityPolls.begin();
           candidate != capacityPolls.end(); ++candidate)
        if (candidate->batch == 0 &&
            (selectedPoll->batch != 0 ||
             entries[candidate->anchor].repairDepth >=
                 entries[selectedPoll->anchor].repairDepth))
          selectedPoll = candidate;
    }
    auto poll = std::move(*selectedPoll);
    capacityPolls.erase(selectedPoll);
    size_t maximumAxes = 0;
    for (const auto &group : poll.groups)
      maximumAxes = std::max(maximumAxes, group.size());
    std::vector<Coordinate> selected;
    const bool initialDirection =
        poll.initialRound && poll.batch <= maximumAxes;
    bool jointDirection = poll.batch == 0;
    if (jointDirection) {
      // One proposal can repair all independently certified Tile scopes.
      selected = poll.coordinates;
      // Keep a uniquely most valuable reuse axis while jointly shrinking
      // the other certified dimensions. Uniformly halving every dimension
      // repeats the largest invariant input and loses the reuse that the
      // repaired physical candidate is meant to realize. This orders one
      // explicit choice only; the ordinary single/pair directions remain.
      for (const auto &group : poll.groups) {
        if (group.size() < 2)
          continue;
        const auto first = group.front();
        const auto &scope =
            entries[poll.anchor].choices[first.domain].scopes[first.scope];
        auto descriptors = domains[first.domain]->getScopeDescriptors(
            entries[poll.anchor].choices[first.domain].kind);
        auto operands = getReadOperandProjections(
            scope.operation, descriptors[first.scope].iterationExtents);
        if (!operands)
          continue;
        uint64_t maximum = 0;
        std::optional<Coordinate> preferred;
        bool tied = false;
        for (auto coordinate : group) {
          uint64_t bytes = 0;
          for (const auto &operand : *operands)
            if (coordinate.iterator < operand.iterators.size() &&
                !operand.iterators.test(coordinate.iterator))
              bytes = llvm::SaturatingAdd(bytes, operand.bytes);
          if (bytes > maximum) {
            maximum = bytes;
            preferred = coordinate;
            tied = false;
          } else if (bytes == maximum)
            tied = true;
        }
        if (preferred && !tied)
          llvm::erase_if(selected, [&](auto coordinate) {
            return !(coordinate < *preferred) && !(*preferred < coordinate);
          });
      }
      // Keep the all-coordinate alternative after the original individual
      // directions, so it cannot displace the progressing repair chain.
      poll.remainingWholeDirection = selected.size() != poll.coordinates.size();
      ++poll.batch;
    } else if (poll.batch <= maximumAxes) {
      for (const auto &group : poll.groups)
        selected.push_back(group[(poll.batch - 1) % group.size()]);
      ++poll.batch;
    } else if (poll.single < poll.coordinates.size()) {
      selected.push_back(poll.coordinates[poll.single++]);
    } else if (prepareScopePair(poll.coordinates, poll.pairFirst,
                                poll.pairSecond)) {
      selected.push_back(poll.coordinates[poll.pairFirst]);
      selected.push_back(poll.coordinates[poll.pairSecond]);
      if (++poll.pairSecond == poll.coordinates.size())
        poll.pairSecond = ++poll.pairFirst + 1;
    } else if (poll.remainingWholeDirection) {
      selected = poll.coordinates;
      poll.remainingWholeDirection = false;
      jointDirection = true;
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
    if (jointDirection) {
      // One coupled choice preserves equal loop domains across equivalent
      // current scopes. Refining only a donor destroys matching peer windows
      // even when recipients did not themselves produce a capacity witness.
      // This is a choice coupling, never a capacity claim for those scopes.
      auto certified = selected;
      auto available = coordinates(next);
      using Projections =
          std::optional<llvm::SmallVector<OperandProjection, 4>>;
      std::map<mlir::Operation *, Projections> readCache;
      auto readSources = [&](const auto &descriptor) -> const Projections & {
        auto found = readCache.find(descriptor.operation);
        if (found == readCache.end())
          found = readCache
                      .emplace(descriptor.operation,
                               getReadOperandProjections(
                                   descriptor.operation,
                                   descriptor.iterationExtents))
                      .first;
        return found->second;
      };
      for (auto reference : certified) {
        auto aDescriptors = domains[reference.domain]->getScopeDescriptors(
            next[reference.domain].kind);
        const auto &a = aDescriptors[reference.scope];
        auto aOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(a.operation);
        if (!aOp)
          continue;
        for (auto other : available) {
          if (other.iterator != reference.iterator)
            continue;
          auto bDescriptors = domains[other.domain]->getScopeDescriptors(
              next[other.domain].kind);
          const auto &b = bDescriptors[other.scope];
          auto bOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(b.operation);
          auto aTile = a.operation->getParentOfType<TileModuleOp>();
          auto bTile = b.operation->getParentOfType<TileModuleOp>();
          if (!aTile || !bTile || aTile.getCardId() != bTile.getCardId() ||
              a.operation->getParentOfType<mlir::ModuleOp>() !=
                  b.operation->getParentOfType<mlir::ModuleOp>())
            continue;
          const auto &aReads = readSources(a);
          const auto &bReads = readSources(b);
          bool shared =
              aReads && bReads && llvm::any_of(*aReads, [&](const auto &x) {
                return x.programArgument &&
                       llvm::any_of(*bReads, [&](const auto &y) {
                         if (x.programArgument != y.programArgument ||
                             x.offsets != y.offsets || x.sizes != y.sizes ||
                             x.iterators != y.iterators ||
                             a.iterationExtents.size() !=
                                 b.iterationExtents.size())
                           return false;
                         const auto &aChoice =
                             next[reference.domain].scopes[reference.scope];
                         const auto &bChoice =
                             next[other.domain].scopes[other.scope];
                         for (unsigned axis = 0;
                              axis < a.iterationExtents.size(); ++axis) {
                           if (a.iterationExtents[axis] ==
                               b.iterationExtents[axis])
                             continue;
                           // Uneven spatial pieces can still share this exact
                           // read when the differing axis is invariant and
                           // both choices consume their full local extent.
                           if (x.iterators.test(axis) ||
                               aChoice.iteratorTileSizes[axis] !=
                                   a.iterationExtents[axis] ||
                               bChoice.iteratorTileSizes[axis] !=
                                   b.iterationExtents[axis])
                             return false;
                         }
                         return true;
                       });
              });
          if (!shared)
            continue;
          if (!bOp || a.iteratorCapabilities != b.iteratorCapabilities ||
              aOp.getIndexingMapsArray() != bOp.getIndexingMapsArray() ||
              aOp.getIteratorTypesArray() != bOp.getIteratorTypesArray() ||
              value(next, reference) != value(next, other))
            continue;
          if (!llvm::any_of(selected, [&](auto point) {
                return !(point < other) && !(other < point);
              }))
            selected.push_back(other);
        }
      }
    }
    for (Coordinate coordinate : selected) {
      int64_t &size = value(next, coordinate);
      size = std::max(bounds(next, coordinate).lower, size / 2);
    }
    // Appending new feedback never displaces the unvisited sibling directions.
    const size_t anchor = poll.anchor;
    const uint64_t depth =
        llvm::SaturatingAdd(entries[anchor].repairDepth, uint64_t(1));
    capacityPolls.push_back(std::move(poll));
    auto &repairQueue =
        queues[static_cast<unsigned>(TemporalProposalKind::Repair)];
    const size_t queueStart = repairQueue.size();
    auto enqueue = [&](std::vector<TemporalChoice> point) {
      auto previous = find(point);
      const bool queued =
          append(std::move(point), TemporalProposalKind::Repair);
      if (initialDirection) {
        const auto index =
            queued ? std::optional<size_t>(entries.size() - 1) : previous;
        if (index && !entries[*index].taken)
          firstCapacityPending.insert(*index);
      }
      if (queued)
        entries.back().repairDepth = depth;
      return queued;
    };
    bool coupledQueued = false;
    if (complete(next)) {
      // A producer-only repair can otherwise leave its state consumer at full
      // extent, preventing the existing common traversal from materializing.
      // Use the same current-map query as the seed generator. This is another
      // explicit choice, not a new buffer owner or a capacity legality claim.
      auto coupled = next;
      bool changed = false;
      for (auto [index, domain] : llvm::enumerate(domains))
        if (auto coordinated = domain->getCoupledStateProposal(
                coupled[index], &entries[anchor].choices[index])) {
          coupled[index] = std::move(*coordinated);
          changed = true;
        }
      if (changed)
        coupledQueued = enqueue(std::move(coupled));
    }
    const bool originalQueued = enqueue(std::move(next));
    if (coupledQueued || originalQueued) {
      if (priorityCapacityAnchor && *priorityCapacityAnchor == anchor) {
        auto index = repairQueue[queueStart];
        repairQueue.erase(repairQueue.begin() + queueStart);
        repairQueue.push_front(index);
        priorityCapacityAnchor.reset();
      }
      preferFreshCapacity = !preferFreshCapacity;
      return true;
    }
  }
  return false;
}

} // namespace wafer::compiler::detail
