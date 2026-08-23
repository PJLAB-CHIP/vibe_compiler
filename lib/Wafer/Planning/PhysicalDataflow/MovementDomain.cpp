//===- MovementDomain.cpp - Explicit transfer realization domain -----===//

#include "Wafer/Planning/PhysicalDataflow/MovementDomain.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

MovementDomainResult failed(MovementDomainFailureKind kind,
                            llvm::StringRef detail) {
  return {{}, MovementDomainFailure{kind, detail.str()}};
}

const PhysicalVersionId *findPrimary(const RepresentationPlan &plan,
                                     const RegionValueVersionId &logical) {
  auto found = llvm::find_if(plan.logicalValues,
                             [&](const LogicalRepresentationPlan &candidate) {
                               return candidate.value == logical;
                             });
  return found == plan.logicalValues.end() ? nullptr : &found->primary;
}

const PhysicalVersionId *findBoundaryUse(const RepresentationPlan &plan,
                                         const BoundaryRegionValueId &value) {
  RepresentationUseId use = BoundaryRepresentationUseId{value};
  auto found = llvm::find_if(plan.uses, [&](const PhysicalUseBinding &binding) {
    return binding.use == use;
  });
  return found == plan.uses.end() ? nullptr : &found->version;
}

bool hasVersion(const RepresentationPlan &plan, const PhysicalVersionId &id) {
  return llvm::any_of(
      plan.physicalVersions,
      [&](const PhysicalVersionPlan &version) { return version.id == id; });
}

std::optional<MemLayout> findEncoding(const RepresentationPlan &plan,
                                      const PhysicalVersionId &id) {
  auto found = llvm::find_if(
      plan.physicalVersions,
      [&](const PhysicalVersionPlan &version) { return version.id == id; });
  return found == plan.physicalVersions.end()
             ? std::nullopt
             : std::optional<MemLayout>(found->encoding);
}

const MovementResourceDescription *
findResource(llvm::ArrayRef<MovementResourceDescription> resources,
             const MovementActionId &action) {
  auto found = llvm::find_if(resources, [&](const auto &resource) {
    return resource.action == action;
  });
  return found == resources.end() ? nullptr : &*found;
}

std::vector<MovementHop> buildHops(TileId source, TileId destination,
                                   llvm::ArrayRef<TileId> relays) {
  std::vector<MovementHop> hops;
  TileId current = source;
  for (TileId relay : relays) {
    hops.push_back({current, relay});
    current = relay;
  }
  hops.push_back({current, destination});
  return hops;
}

bool advancePermutation(std::vector<TileId> &relays,
                        llvm::ArrayRef<TileId> available) {
  for (size_t reverse = 0; reverse < relays.size(); ++reverse) {
    const size_t index = relays.size() - reverse - 1;
    std::set<int64_t> prefix;
    for (TileId tile : llvm::ArrayRef<TileId>(relays).take_front(index))
      prefix.insert(tile.getValue());
    for (TileId candidate : available) {
      if (candidate.getValue() <= relays[index].getValue() ||
          prefix.count(candidate.getValue()))
        continue;
      relays[index] = candidate;
      prefix.insert(candidate.getValue());
      size_t suffix = index + 1;
      for (TileId fill : available) {
        if (suffix == relays.size())
          break;
        if (prefix.insert(fill.getValue()).second)
          relays[suffix++] = fill;
      }
      if (suffix == relays.size())
        return true;
    }
  }
  return false;
}

} // namespace

bool MovementDomain::advanceChoice(size_t variable,
                                   MovementCursor::Choice &choice) const {
  const Variable &domain = variables[variable];
  if (domain.source == domain.destination || !domain.peerCapable)
    return false;
  if (choice.kind == MovementRealizationKind::DDRStage) {
    choice.kind = MovementRealizationKind::TargetRoutedPeer;
    return true;
  }
  if (choice.kind == MovementRealizationKind::TargetRoutedPeer) {
    if (domain.availableRelays.empty())
      return false;
    choice.kind = MovementRealizationKind::SoftwareRelay;
    choice.relays = {domain.availableRelays.front()};
    return true;
  }
  if (advancePermutation(choice.relays, domain.availableRelays))
    return true;
  if (choice.relays.size() == domain.availableRelays.size())
    return false;
  choice.relays.assign(domain.availableRelays.begin(),
                       domain.availableRelays.begin() + choice.relays.size() +
                           1);
  return true;
}

std::optional<MovementPlan>
MovementDomain::buildPlan(const MovementCursor &cursor) const {
  if (cursor.choices.size() != variables.size())
    return std::nullopt;
  MovementPlan plan = base;
  for (auto [variable, choice] : llvm::zip_equal(variables, cursor.choices)) {
    MovementRealization realization;
    realization.kind = choice.kind;
    if (choice.kind == MovementRealizationKind::TargetRoutedPeer)
      realization.hops = {{variable.source, variable.destination}};
    else if (choice.kind == MovementRealizationKind::SoftwareRelay)
      realization.hops =
          buildHops(variable.source, variable.destination, choice.relays);
    if (variable.gather)
      plan.reductionGathers[variable.planIndex].realization =
          std::move(realization);
    else
      plan.ddrTransfers[variable.planIndex].realization =
          std::move(realization);
  }
  return plan;
}

MovementSuccessor MovementDomain::getFirstPlan() const {
  MovementCursor cursor;
  cursor.choices.resize(variables.size());
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
  if (plan.externalLoads != base.externalLoads ||
      plan.publications != base.publications ||
      plan.discards != base.discards ||
      plan.ddrTransfers.size() != base.ddrTransfers.size() ||
      plan.reductionGathers.size() != base.reductionGathers.size())
    return std::nullopt;
  MovementCursor cursor;
  cursor.choices.resize(variables.size());
  for (auto [index, variable] : llvm::enumerate(variables)) {
    const MovementRealization &realization =
        variable.gather ? plan.reductionGathers[variable.planIndex].realization
                        : plan.ddrTransfers[variable.planIndex].realization;
    MovementCursor::Choice &choice = cursor.choices[index];
    choice.kind = realization.kind;
    if (choice.kind == MovementRealizationKind::DDRStage) {
      if (!realization.hops.empty())
        return std::nullopt;
      continue;
    }
    if (realization.hops.empty() ||
        realization.hops.front().source != variable.source ||
        realization.hops.back().destination != variable.destination)
      return std::nullopt;
    TileId current = variable.source;
    std::set<int64_t> seen{current.getValue()};
    for (auto [hopIndex, hop] : llvm::enumerate(realization.hops)) {
      if (hop.source != current ||
          !llvm::is_contained(availableTiles, hop.destination))
        return std::nullopt;
      current = hop.destination;
      if (hopIndex + 1 != realization.hops.size()) {
        if (!seen.insert(current.getValue()).second ||
            current == variable.destination)
          return std::nullopt;
        choice.relays.push_back(current);
      }
    }
    if (choice.kind == MovementRealizationKind::TargetRoutedPeer &&
        (realization.hops.size() != 1 || !choice.relays.empty()))
      return std::nullopt;
    if (choice.kind == MovementRealizationKind::SoftwareRelay &&
        choice.relays.empty())
      return std::nullopt;
  }
  std::optional<MovementPlan> rebuilt = buildPlan(cursor);
  if (!rebuilt || !(*rebuilt == plan))
    return std::nullopt;
  return cursor;
}

bool MovementDomain::contains(const MovementPlan &plan) const {
  return getCursor(plan).has_value();
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
  for (size_t reverse = 0; reverse < variables.size(); ++reverse) {
    const size_t index = variables.size() - reverse - 1;
    if (!advanceChoice(index, next.choices[index]))
      continue;
    for (size_t reset = index + 1; reset < variables.size(); ++reset)
      next.choices[reset] = MovementCursor::Choice{};
    std::optional<MovementPlan> plan = buildPlan(next);
    return plan ? MovementSuccessor(MovementSuccessorKind::Plan,
                                    std::move(plan), std::move(next))
                : MovementSuccessor(MovementSuccessorKind::CompilerBug, {}, {},
                                    "movement successor is malformed");
  }
  return {MovementSuccessorKind::End};
}

MovementDomainResult
buildMovementDomain(const CanonicalMovementCoordinate &canonical,
                    const RepresentationPlan &representations,
                    llvm::ArrayRef<TileId> inputTiles) {
  std::vector<TileId> availableTiles(inputTiles.begin(), inputTiles.end());
  llvm::sort(availableTiles, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  if (availableTiles.empty() ||
      std::adjacent_find(availableTiles.begin(), availableTiles.end()) !=
          availableTiles.end())
    return failed(MovementDomainFailureKind::BrokenContract,
                  "movement domain has an invalid endpoint set");

  MovementPlan base = canonical.plan;
  for (ExternalLoadPlan &load : base.externalLoads) {
    const PhysicalVersionId *destination =
        findBoundaryUse(representations, load.id.destination);
    if (!destination || !hasVersion(representations, *destination))
      return failed(MovementDomainFailureKind::BrokenContract,
                    "external load has no selected destination version");
    load.destination = *destination;
  }
  for (DDRBoundaryTransferPlan &transfer : base.ddrTransfers) {
    const PhysicalVersionId *source =
        findPrimary(representations, transfer.source.logicalValue);
    const PhysicalVersionId *destination =
        findBoundaryUse(representations, transfer.id.destination);
    if (!source || !destination || !hasVersion(representations, *source) ||
        !hasVersion(representations, *destination))
      return failed(MovementDomainFailureKind::BrokenContract,
                    "boundary transfer has missing selected versions");
    transfer.source = *source;
    transfer.destination = *destination;
  }
  for (ReductionGatherPlan &gather : base.reductionGathers) {
    const PhysicalVersionId *source =
        findPrimary(representations, gather.source.logicalValue);
    if (!source || !hasVersion(representations, *source))
      return failed(MovementDomainFailureKind::BrokenContract,
                    "reduction gather has no selected source version");
    gather.source = *source;
  }
  for (ResultPublicationPlan &publication : base.publications) {
    const PhysicalVersionId *source =
        findPrimary(representations, publication.source.logicalValue);
    if (!source)
      return failed(MovementDomainFailureKind::BrokenContract,
                    "publication has no selected source version");
    publication.source = *source;
  }
  for (ResultDiscardPlan &discard : base.discards) {
    const PhysicalVersionId *source =
        findPrimary(representations, discard.source.logicalValue);
    if (!source)
      return failed(MovementDomainFailureKind::BrokenContract,
                    "discard has no selected source version");
    discard.source = *source;
  }

  std::vector<MovementDomain::Variable> variables;
  auto addVariable = [&](bool gather, size_t planIndex,
                         const MovementActionId &action,
                         const PhysicalVersionId &sourceVersion,
                         const PhysicalVersionId *destinationVersion) -> bool {
    const MovementResourceDescription *resource =
        findResource(canonical.resources, action);
    if (!resource || !resource->sourceTile || !resource->destinationTile ||
        !llvm::is_contained(availableTiles, *resource->sourceTile) ||
        !llvm::is_contained(availableTiles, *resource->destinationTile))
      return false;
    MovementDomain::Variable variable;
    variable.gather = gather;
    variable.planIndex = planIndex;
    variable.source = *resource->sourceTile;
    variable.destination = *resource->destinationTile;
    std::optional<MemLayout> sourceEncoding =
        findEncoding(representations, sourceVersion);
    std::optional<MemLayout> destinationEncoding =
        destinationVersion ? findEncoding(representations, *destinationVersion)
                           : sourceEncoding;
    variable.peerCapable = sourceEncoding && destinationEncoding &&
                           *sourceEncoding == *destinationEncoding;
    for (TileId tile : availableTiles)
      if (tile != variable.source && tile != variable.destination)
        variable.availableRelays.push_back(tile);
    variables.push_back(std::move(variable));
    return true;
  };
  for (auto [index, transfer] : llvm::enumerate(base.ddrTransfers))
    if (!addVariable(false, index, MovementActionId(transfer.id),
                     transfer.source, &transfer.destination))
      return failed(MovementDomainFailureKind::BrokenContract,
                    "boundary transfer has no exact endpoints");
  for (auto [index, gather] : llvm::enumerate(base.reductionGathers))
    if (!addVariable(true, index, MovementActionId(gather.id), gather.source,
                     nullptr))
      return failed(MovementDomainFailureKind::BrokenContract,
                    "reduction gather has no exact endpoints");
  return {MovementDomain(std::move(base), canonical.resources,
                         std::move(availableTiles), std::move(variables)),
          {}};
}

} // namespace wafer::compiler::detail
