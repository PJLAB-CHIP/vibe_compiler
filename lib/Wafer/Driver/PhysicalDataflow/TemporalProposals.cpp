//===- TemporalProposals.cpp - Feedback ordered integer choices --------===//

#include "TemporalProposals.h"

#include "Wafer/Target/PhysicalTensor/PhysicalLayout.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

namespace wafer::compiler::detail {

std::optional<size_t>
TemporalProposals::find(const std::vector<TemporalChoice> &choices) const {
  for (auto [index, entry] : llvm::enumerate(entries))
    if (entry.choices == choices)
      return index;
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
                               std::optional<Probe> probe) {
  if (!complete(choices) || find(choices))
    return false;
  queues[static_cast<unsigned>(kind)].push_back(entries.size());
  entries.push_back({std::move(choices), std::move(probe), {}, {}});
  return true;
}

bool TemporalProposals::empty(TemporalProposalKind kind) const {
  return queues[static_cast<unsigned>(kind)].empty();
}

std::vector<TemporalChoice> TemporalProposals::take(TemporalProposalKind kind) {
  auto &queue = queues[static_cast<unsigned>(kind)];
  size_t index = queue.front();
  queue.pop_front();
  return entries[index].choices;
}

bool TemporalProposals::visitRaw(const std::vector<TemporalChoice> &choices) {
  if (find(choices))
    return false;
  entries.push_back({choices, {}, {}, {}});
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

bool TemporalProposals::appendProbe(Probe probe, TemporalProposalKind kind) {
  auto next = probe.anchor;
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
  auto &explore = queues[static_cast<unsigned>(TemporalProposalKind::Explore)];
  std::deque<size_t> prior;
  prior.swap(explore);
  append(initial, TemporalProposalKind::Explore);
  const bool independent = llvm::all_of(initial, [](const auto &choice) {
    return choice.kind == TemporalTraversalKind::Independent;
  });
  std::optional<std::vector<TemporalChoice>> independentKernel;
  const auto all = coordinates(initial);
  // A full-only iterator supplies an existing kernel extent worth sampling
  // on the other iterators of that scope. This is a distant shape-derived
  // proposal (for any operation exposing that capability), not an alignment
  // restriction or a prediction that the resulting tile fits SPM.
  auto kernelExtents = initial;
  bool hasKernelExtent = false;
  for (auto [domainIndex, choice] : llvm::enumerate(kernelExtents)) {
    auto descriptors = domains[domainIndex]->getScopeDescriptors(choice.kind);
    for (auto [scopeIndex, descriptor] : llvm::enumerate(descriptors)) {
      std::optional<int64_t> extent;
      for (auto [size, capability] : llvm::zip_equal(
               descriptor.iterationExtents, descriptor.iteratorCapabilities))
        if (capability == IteratorTilingCapability::FullExtentOnly && size > 1)
          extent = extent ? std::max(*extent, size) : size;
      if (!extent)
        continue;
      for (auto coordinate : all)
        if (coordinate.domain == domainIndex &&
            coordinate.scope == scopeIndex) {
          int64_t &size = value(kernelExtents, coordinate);
          int64_t next = std::max(bounds(kernelExtents, coordinate).lower,
                                  std::min(size, *extent));
          hasKernelExtent |= next != size;
          size = next;
        }
    }
  }
  if (hasKernelExtent && complete(kernelExtents)) {
    for (auto [domain, choice] : llvm::zip_equal(domains, kernelExtents))
      if (auto coupled = domain->getCoupledStateProposal(choice))
        choice = std::move(*coupled);
    if (independent)
      independentKernel = std::move(kernelExtents);
    else
      append(std::move(kernelExtents), TemporalProposalKind::Explore);
  }
  // Independent interval exploration samples successive lower midpoints. These
  // are distant proposals, not nearest neighbors or deductions from capacity.
  // The complete raw cursor remains responsible for all unvisited points and
  // loop orders.
  auto low = initial;
  for (auto coordinate : all)
    value(low, coordinate) = bounds(initial, coordinate).lower;
  std::vector<std::vector<std::vector<TemporalChoice>>> scales;
  auto sample = initial;
  while (true) {
    bool changed = false;
    for (auto coordinate : all) {
      int64_t &size = value(sample, coordinate);
      auto interval = bounds(sample, coordinate);
      int64_t next = interval.lower + (size - interval.lower) / 2;
      changed |= size != next;
      size = next;
    }
    if (!changed)
      break;
    auto single = initial;
    for (size_t domain = 0; domain < domains.size(); ++domain) {
      auto descriptors =
          domains[domain]->getScopeDescriptors(initial[domain].kind);
      for (size_t scope = 0; scope < descriptors.size(); ++scope) {
        std::optional<Coordinate> largest;
        for (auto coordinate : all)
          if (coordinate.domain == domain && coordinate.scope == scope &&
              (!largest || bounds(initial, coordinate).upper >
                               bounds(initial, *largest).upper))
            largest = coordinate;
        if (largest)
          value(single, *largest) = value(sample, *largest);
      }
    }
    std::vector<std::vector<TemporalChoice>> scale;
    scale.push_back(std::move(single));
    auto coordinated = sample;
    // A domain query requires a complete typed choice, including loop order.
    // Completing only when appending would silently suppress every coupled
    // proposal whose sizes activated a loop absent from the initial choice.
    if (complete(coordinated)) {
      for (auto [domain, choice] : llvm::zip_equal(domains, coordinated))
        if (auto coupled = domain->getCoupledStateProposal(choice))
          choice = std::move(*coupled);
      scale.push_back(std::move(coordinated));
    }
    scale.push_back(sample);
    scales.push_back(std::move(scale));
  }
  // Different structural sessions start in different scale strata instead
  // of all spending a short global budget at the same numeric scale. This
  // changes only visitation order: no stratum or combination is removed.
  for (size_t offset = 0; offset < scales.size(); ++offset) {
    auto &scale = scales[(stratum % scales.size() + offset) % scales.size()];
    for (size_t variant = 0; variant < scale.size(); ++variant)
      append(
          std::move(scale[(stratum % scale.size() + variant) % scale.size()]),
          TemporalProposalKind::Explore);
  }
  append(std::move(low), TemporalProposalKind::Explore);
  // Joint supplies every structure's full-extent anchor. The independently
  // traversed family starts with its numeric stratum, so alternating families
  // also explores a complete numeric tuple before repeating another full-size
  // experiment. Its full extent and kernel-only change remain reachable.
  if (independent) {
    if (explore.size() > 1) {
      explore.push_back(explore.front());
      explore.pop_front();
    }
    if (independentKernel)
      append(std::move(*independentKernel), TemporalProposalKind::Explore);
  }
  // Joint and Independent seed families receive alternating opportunities.
  std::deque<size_t> fresh;
  fresh.swap(explore);
  while (!prior.empty() || !fresh.empty()) {
    if (!prior.empty()) {
      explore.push_back(prior.front());
      prior.pop_front();
    }
    if (!fresh.empty()) {
      explore.push_back(fresh.front());
      fresh.pop_front();
    }
  }
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
      probe.anchor = choices;
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
  for (auto coordinate : all) {
    for (int direction : {-1, 1})
      appendProbe({choices, {coordinate}, direction, 1, duration},
                  TemporalProposalKind::Improve);
    appendLayoutBoundaries(choices, coordinate, duration);
  }
  // A combination may improve even when every one-coordinate move worsens.
  // Its admission is independent of the single-coordinate observations.
  for (int direction : {-1, 1})
    appendProbe({choices, all, direction, 1, duration},
                TemporalProposalKind::Explore);
}

void TemporalProposals::appendLayoutBoundaries(
    const std::vector<TemporalChoice> &choices, Coordinate coordinate,
    uint64_t duration) {
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
        appendProbe({choices,
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
    const std::set<size_t> &affectedDomains) {
  auto index = find(choices);
  if (!index)
    return false;
  std::set<size_t> freshDomains;
  for (size_t domain : affectedDomains)
    if (entries[*index].capacityObserved.insert(domain).second)
      freshDomains.insert(domain);
  if (freshDomains.empty())
    return false;
  const auto &previous = entries[*index].probe;
  Probe probe{choices, {}, -1, 1, {}};
  if (previous && previous->direction < 0) {
    for (auto coordinate : previous->coordinates)
      if (freshDomains.count(coordinate.domain))
        probe.coordinates.push_back(coordinate);
    if (!probe.coordinates.empty())
      probe.distance = expandDistance(previous->distance);
  }
  if (probe.coordinates.empty()) {
    auto all = coordinates(choices);
    for (size_t domain : freshDomains) {
      // A Region certificate does not distinguish independent roots or a
      // fused traversal from its reduction. Never invent that attribution.
      if (domains[domain]->getScopeDescriptors(choices[domain].kind).size() !=
          1)
        continue;
      std::optional<Coordinate> largest;
      auto current = choices;
      for (auto coordinate : all)
        if (coordinate.domain == domain &&
            value(current, coordinate) > bounds(choices, coordinate).lower &&
            (!largest || value(current, coordinate) > value(current, *largest)))
          largest = coordinate;
      if (largest)
        probe.coordinates.push_back(*largest);
    }
  }
  return appendProbe(std::move(probe), TemporalProposalKind::Repair);
}

} // namespace wafer::compiler::detail
