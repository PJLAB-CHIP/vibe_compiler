//===- StorageRequirements.cpp - Production storage facts -----------===//

#include "Wafer/Planning/PhysicalDataflow/StorageRequirements.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <type_traits>

namespace wafer::compiler::detail {
namespace {

StorageRequirementDerivationResult failed(StorageRequirementFailureKind kind,
                                          llvm::StringRef detail) {
  return {{}, StorageRequirementFailure{kind, detail.str()}};
}

bool sameSet(const analysis::ExactIndexSet &lhs,
             const analysis::ExactIndexSet &rhs) {
  return lhs.getRank() == rhs.getRank() &&
         lhs.getPresburgerSet().isObviouslyEqual(rhs.getPresburgerSet());
}

bool sameResource(const StorageResourceDescription &lhs,
                  const StorageResourceDescription &rhs) {
  return lhs.elementType == rhs.elementType && lhs.encoding == rhs.encoding &&
         sameSet(lhs.exactDomain, rhs.exactDomain) &&
         sameSet(lhs.residentDomain, rhs.residentDomain);
}

size_t hashSetStructure(const analysis::ExactIndexSet &set) {
  const mlir::presburger::PresburgerSet &presburger = set.getPresburgerSet();
  llvm::hash_code hash =
      llvm::hash_combine(set.getRank(), presburger.getNumDisjuncts());
  for (unsigned disjunctIndex = 0; disjunctIndex < presburger.getNumDisjuncts();
       ++disjunctIndex) {
    const mlir::presburger::IntegerRelation &disjunct =
        presburger.getDisjunct(disjunctIndex);
    hash = llvm::hash_combine(
        hash, disjunct.getNumDomainVars(), disjunct.getNumRangeVars(),
        disjunct.getNumSymbolVars(), disjunct.getNumLocalVars(),
        disjunct.getNumEqualities(), disjunct.getNumInequalities(),
        disjunct.getNumCols());
    for (unsigned row = 0; row < disjunct.getNumEqualities(); ++row)
      for (unsigned column = 0; column < disjunct.getNumCols(); ++column)
        hash = llvm::hash_combine(hash, disjunct.atEq(row, column));
    for (unsigned row = 0; row < disjunct.getNumInequalities(); ++row)
      for (unsigned column = 0; column < disjunct.getNumCols(); ++column)
        hash = llvm::hash_combine(hash, disjunct.atIneq(row, column));
  }
  return static_cast<size_t>(hash);
}

std::optional<RegionExecutionId>
getVersionExecution(const PhysicalVersionId &version) {
  return std::visit(
      [](const auto &logical) -> std::optional<RegionExecutionId> {
        using T = std::decay_t<decltype(logical)>;
        if constexpr (std::is_same_v<T, BoundaryRegionValueId>) {
          return RegionExecutionId{ExecutionInstanceId{RequiredRootExecution{
              logical.work, logical.fragment.use.destinationShard}}};
        } else if constexpr (std::is_same_v<T, SupportRegionValueId>) {
          return std::nullopt;
        } else if constexpr (std::is_same_v<T, ExecutionResultValueId>) {
          return logical.execution;
        } else {
          return RegionExecutionId{logical.execution};
        }
      },
      version.logicalValue);
}

const ReductionGatherPlan *findGather(const MovementPlan &movement,
                                      const ReductionGatherId &id) {
  auto found = llvm::find_if(
      movement.reductionGathers,
      [&](const ReductionGatherPlan &candidate) { return candidate.id == id; });
  return found == movement.reductionGathers.end() ? nullptr : &*found;
}

std::optional<RegionExecutionId>
getActionExecution(const MovementActionId &action,
                   const MovementPlan &movement) {
  if (const auto *load = std::get_if<ExternalLoadId>(&action))
    return RegionExecutionId{ExecutionInstanceId{RequiredRootExecution{
        load->destination.work,
        load->destination.fragment.use.destinationShard}}};
  if (const auto *transfer = std::get_if<DDRBoundaryTransferId>(&action))
    return RegionExecutionId{ExecutionInstanceId{RequiredRootExecution{
        transfer->destination.work,
        transfer->destination.fragment.use.destinationShard}}};
  if (const auto *gatherId = std::get_if<ReductionGatherId>(&action)) {
    const ReductionGatherPlan *gather = findGather(movement, *gatherId);
    if (!gather)
      return std::nullopt;
    return getVersionExecution(gather->source);
  }
  return std::nullopt;
}

std::optional<RegionExecutionId>
getObjectExecution(const StorageObjectId &object,
                   const MovementPlan &movement) {
  if (const auto *version = std::get_if<PhysicalVersionId>(&object.origin))
    return getVersionExecution(*version);
  if (const auto *gather =
          std::get_if<ReductionGatherStagingId>(&object.origin)) {
    const ReductionGatherPlan *plan = findGather(movement, gather->gather);
    return plan ? getVersionExecution(plan->source) : std::nullopt;
  }
  const auto &relay = std::get<PeerRelayStorageId>(object.origin);
  if (relay.graphActions.empty())
    return std::nullopt;
  std::optional<RegionExecutionId> execution =
      getActionExecution(relay.graphActions.front(), movement);
  for (const MovementActionId &action :
       llvm::ArrayRef<MovementActionId>(relay.graphActions).drop_front()) {
    std::optional<RegionExecutionId> candidate =
        getActionExecution(action, movement);
    if (candidate.has_value() != execution.has_value() ||
        (candidate && !(*candidate == *execution)))
      return std::nullopt;
  }
  return execution;
}

std::optional<OccurrenceRelationId>
getOccurrence(const StorageObjectId &object, const MovementPlan &movement,
              const TemporalPlan &temporal,
              llvm::ArrayRef<TemporalScopeDescriptor> descriptors) {
  std::optional<RegionExecutionId> execution =
      getObjectExecution(object, movement);
  if (!execution)
    return std::nullopt;
  llvm::SmallVector<const TemporalScopePlan *, 2> selected;
  for (const TemporalScopePlan &scope : temporal.scopes)
    if (scope.id.execution == *execution)
      selected.push_back(&scope);
  if (selected.size() != 1)
    return std::nullopt;
  auto descriptor =
      llvm::find_if(descriptors, [&](const TemporalScopeDescriptor &candidate) {
        return candidate.id == selected.front()->id;
      });
  if (descriptor == descriptors.end() ||
      descriptor->iterationExtents.size() !=
          selected.front()->iteratorTileSizes.size())
    return std::nullopt;
  OccurrenceRelationId occurrence;
  occurrence.scope = selected.front()->id;
  for (auto [extent, tile] : llvm::zip_equal(
           descriptor->iterationExtents, selected.front()->iteratorTileSizes)) {
    if (extent <= 0 || tile <= 0)
      return std::nullopt;
    occurrence.axisOccurrences.push_back(
        (static_cast<uint64_t>(extent) + static_cast<uint64_t>(tile) - 1) /
        static_cast<uint64_t>(tile));
  }
  return occurrence;
}

std::optional<uint32_t>
getOccurrenceCount(const OccurrenceRelationId &occurrence) {
  uint64_t count = 1;
  for (uint64_t axis : occurrence.axisOccurrences) {
    if (axis == 0 || count > std::numeric_limits<uint64_t>::max() / axis)
      return std::nullopt;
    count *= axis;
  }
  return static_cast<uint32_t>(
      std::min<uint64_t>(count, std::numeric_limits<uint32_t>::max()));
}

} // namespace

StorageRequirementDerivationResult deriveStorageRequirements(
    const CanonicalStorageCoordinate &canonical,
    const RepresentationPlan &representations, const MovementPlan &movement,
    const TemporalPlan &temporal,
    llvm::ArrayRef<TemporalScopeDescriptor> scopeDescriptors,
    const StorageRequirementLimits &limits) {
  if (canonical.plan.storageObjects.empty() ||
      canonical.plan.versionBindings.empty() ||
      representations.physicalVersions.empty())
    return failed(StorageRequirementFailureKind::BrokenContract,
                  "storage requirements need closed G/H/I inventories");
  if (limits.maxRotationOptions == 0)
    return failed(StorageRequirementFailureKind::BrokenContract,
                  "storage rotation work limit must be positive");

  std::map<StorageObjectId, const StorageObjectPlan *> objects;
  std::map<StorageObjectId, const StorageResourceDescription *> resources;
  std::map<StorageObjectId, const StorageLifetimeDescription *> lifetimes;
  for (const StorageObjectPlan &object : canonical.plan.storageObjects)
    if (!objects.try_emplace(object.id, &object).second)
      return failed(StorageRequirementFailureKind::BrokenContract,
                    "storage requirements found duplicate objects");
  for (const StorageResourceDescription &resource : canonical.resources)
    if (!resources.try_emplace(resource.object, &resource).second)
      return failed(StorageRequirementFailureKind::BrokenContract,
                    "storage requirements found duplicate resources");
  for (const StorageLifetimeDescription &lifetime : canonical.lifetimes)
    if (!lifetimes.try_emplace(lifetime.object, &lifetime).second)
      return failed(StorageRequirementFailureKind::BrokenContract,
                    "storage requirements found duplicate lifetimes");
  if (objects.size() != resources.size() || objects.size() != lifetimes.size())
    return failed(StorageRequirementFailureKind::BrokenContract,
                  "storage object/resource/lifetime inventories differ");
  std::set<PhysicalVersionId> representedVersions;
  std::set<PhysicalVersionId> identityAliasVersions;
  for (const PhysicalVersionPlan &version : representations.physicalVersions)
    if (!representedVersions.insert(version.id).second)
      return failed(StorageRequirementFailureKind::BrokenContract,
                    "storage requirements found duplicate physical versions");
    else if (!version.id.derivation.empty() &&
             version.id.derivation.back().kind ==
                 PhysicalVersionDerivationKind::AliasView)
      identityAliasVersions.insert(version.id);
  std::set<PhysicalVersionId> boundVersions;
  for (const PhysicalVersionStorageBinding &binding :
       canonical.plan.versionBindings)
    if (!representedVersions.count(binding.version) ||
        !boundVersions.insert(binding.version).second)
      return failed(StorageRequirementFailureKind::BrokenContract,
                    "storage requirements found an unknown version binding");
  if (boundVersions != representedVersions)
    return failed(StorageRequirementFailureKind::BrokenContract,
                  "storage requirements do not cover every physical version");

  struct ResourceBucketKey {
    int64_t tile = 0;
    const void *elementType = nullptr;
    MemLayout encoding = MemLayout::Tensor;
    unsigned exactRank = 0;
    unsigned residentRank = 0;
    size_t exactStructure = 0;
    size_t residentStructure = 0;
  };
  struct ResourceBucketKeyLess {
    bool operator()(const ResourceBucketKey &lhs,
                    const ResourceBucketKey &rhs) const {
      if (lhs.tile != rhs.tile)
        return lhs.tile < rhs.tile;
      if (lhs.elementType != rhs.elementType)
        return std::less<const void *>{}(lhs.elementType, rhs.elementType);
      return std::tie(lhs.encoding, lhs.exactRank, lhs.residentRank,
                      lhs.exactStructure, lhs.residentStructure) <
             std::tie(rhs.encoding, rhs.exactRank, rhs.residentRank,
                      rhs.exactStructure, rhs.residentStructure);
    }
  };
  auto getBucketKey = [](const StorageObjectPlan &object,
                         const StorageResourceDescription &resource) {
    return ResourceBucketKey{object.tile.getValue(),
                             resource.elementType.getAsOpaquePointer(),
                             resource.encoding,
                             resource.exactDomain.getRank(),
                             resource.residentDomain.getRank(),
                             hashSetStructure(resource.exactDomain),
                             hashSetStructure(resource.residentDomain)};
  };

  std::map<ResourceBucketKey,
           std::vector<const PhysicalVersionStorageBinding *>,
           ResourceBucketKeyLess>
      reuseCandidates;
  for (const PhysicalVersionStorageBinding &candidate :
       canonical.plan.versionBindings) {
    const auto *candidateVersion =
        std::get_if<PhysicalVersionId>(&candidate.object.origin);
    if (!candidateVersion || identityAliasVersions.count(*candidateVersion))
      continue;
    auto object = objects.find(candidate.object);
    auto resource = resources.find(candidate.object);
    if (object == objects.end() || resource == resources.end())
      return failed(StorageRequirementFailureKind::BrokenContract,
                    "reuse candidate has no closed storage facts");
    reuseCandidates[getBucketKey(*object->second, *resource->second)].push_back(
        &candidate);
  }

  std::map<StorageObjectId, std::set<StorageAccessSite>> accessSites;
  for (const auto &[object, lifetime] : lifetimes) {
    std::set<StorageAccessSite> sites(lifetime->uses.begin(),
                                      lifetime->uses.end());
    sites.insert(lifetime->definition);
    accessSites.emplace(object, std::move(sites));
  }

  DerivedStorageRequirements result;
  for (const PhysicalVersionStorageBinding &target :
       canonical.plan.versionBindings) {
    const bool identityAlias = !target.version.derivation.empty() &&
                               target.version.derivation.back().kind ==
                                   PhysicalVersionDerivationKind::AliasView;
    if (target.kind == StorageBindingKind::IdentityAlias || identityAlias)
      continue;
    auto targetObject = objects.find(target.object);
    auto targetResource = resources.find(target.object);
    auto targetLifetime = lifetimes.find(target.object);
    if (targetObject == objects.end() || targetResource == resources.end() ||
        targetLifetime == lifetimes.end())
      return failed(StorageRequirementFailureKind::BrokenContract,
                    "version binding has no closed storage facts");
    auto bucket = reuseCandidates.find(
        getBucketKey(*targetObject->second, *targetResource->second));
    if (bucket == reuseCandidates.end())
      continue;
    for (const PhysicalVersionStorageBinding *candidatePointer :
         bucket->second) {
      const PhysicalVersionStorageBinding &candidate = *candidatePointer;
      const auto *candidateVersion =
          std::get_if<PhysicalVersionId>(&candidate.object.origin);
      if (!candidateVersion || candidate.version == target.version ||
          candidate.object == target.object ||
          identityAliasVersions.count(*candidateVersion))
        continue;
      auto candidateObject = objects.find(candidate.object);
      auto candidateResource = resources.find(candidate.object);
      auto candidateLifetime = lifetimes.find(candidate.object);
      if (candidateObject == objects.end() ||
          candidateResource == resources.end() ||
          candidateLifetime == lifetimes.end() ||
          !sameResource(*candidateResource->second, *targetResource->second))
        continue;
      const std::set<StorageAccessSite> &candidateSites =
          accessSites.at(candidate.object);
      const std::set<StorageAccessSite> &targetSites =
          accessSites.at(target.object);
      const bool forcedOverlap =
          llvm::any_of(targetSites, [&](const StorageAccessSite &site) {
            return candidateSites.count(site) != 0;
          });
      if (forcedOverlap)
        continue;
      result.reuse.push_back(
          {target.version, candidate.object, StorageReuseProof::RequiresOrder});
    }
  }
  llvm::sort(result.reuse, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.version, lhs.object, lhs.proof) <
           std::tie(rhs.version, rhs.object, rhs.proof);
  });
  result.reuse.erase(std::unique(result.reuse.begin(), result.reuse.end(),
                                 [](const auto &lhs, const auto &rhs) {
                                   return lhs.version == rhs.version &&
                                          lhs.object == rhs.object &&
                                          lhs.proof == rhs.proof;
                                 }),
                     result.reuse.end());

  std::map<OccurrenceRelationId, std::vector<StorageObjectId>> byOccurrence;
  for (const StorageObjectPlan &object : canonical.plan.storageObjects) {
    if (const auto *version = std::get_if<PhysicalVersionId>(&object.id.origin);
        version && identityAliasVersions.count(*version))
      continue;
    std::optional<OccurrenceRelationId> occurrence =
        getOccurrence(object.id, movement, temporal, scopeDescriptors);
    std::optional<uint32_t> count =
        occurrence ? getOccurrenceCount(*occurrence) : std::nullopt;
    if (!occurrence || !count || *count <= 1)
      continue;
    byOccurrence[*occurrence].push_back(object.id);
  }
  for (auto &[occurrence, familyObjects] : byOccurrence) {
    llvm::sort(familyObjects);
    familyObjects.erase(std::unique(familyObjects.begin(), familyObjects.end()),
                        familyObjects.end());
    std::optional<uint32_t> upper = getOccurrenceCount(occurrence);
    if (!upper || *upper <= 1)
      continue;
    llvm::SmallVector<uint32_t, 4> activeAxes;
    for (auto [axis, count] : llvm::enumerate(occurrence.axisOccurrences))
      if (count > 1)
        activeAxes.push_back(static_cast<uint32_t>(axis));
    if (activeAxes.empty())
      continue;
    SlotFamilyRequirement family;
    family.id.objects = std::move(familyObjects);
    family.occurrence = occurrence;
    family.upperBound = *upper;
    do {
      if (family.rotationOptions.size() >= limits.maxRotationOptions)
        return failed(StorageRequirementFailureKind::Indeterminate,
                      "storage rotation enumeration exceeded its work limit");
      family.rotationOptions.push_back(activeAxes);
    } while (std::next_permutation(activeAxes.begin(), activeAxes.end()));
    result.slotFamilies.push_back(std::move(family));
  }
  llvm::sort(result.slotFamilies,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  return {std::move(result), {}};
}

} // namespace wafer::compiler::detail
