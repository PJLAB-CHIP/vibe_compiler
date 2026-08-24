//===- StorageDomain.cpp - Storage binding and slot domain ------------===//

#include "Wafer/Planning/PhysicalDataflow/StorageDomain.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

StorageDomainResult failed(StorageDomainFailureKind kind,
                           llvm::StringRef detail) {
  return {{}, StorageDomainFailure{kind, detail.str()}};
}

bool sameResource(const StorageResourceDescription &lhs,
                  const StorageResourceDescription &rhs) {
  if (lhs.elementType != rhs.elementType || lhs.encoding != rhs.encoding ||
      lhs.exactDomain.getRank() != rhs.exactDomain.getRank() ||
      lhs.exactDomain.getBoxes().size() != rhs.exactDomain.getBoxes().size() ||
      lhs.residentDomain.getRank() != rhs.residentDomain.getRank() ||
      lhs.residentDomain.getBoxes().size() !=
          rhs.residentDomain.getBoxes().size())
    return false;
  for (auto [lhsBox, rhsBox] :
       llvm::zip_equal(lhs.exactDomain.getBoxes(), rhs.exactDomain.getBoxes()))
    if (lhsBox.offsets != rhsBox.offsets || lhsBox.sizes != rhsBox.sizes)
      return false;
  for (auto [lhsBox, rhsBox] : llvm::zip_equal(lhs.residentDomain.getBoxes(),
                                               rhs.residentDomain.getBoxes()))
    if (lhsBox.offsets != rhsBox.offsets || lhsBox.sizes != rhsBox.sizes)
      return false;
  return true;
}

std::optional<PhysicalVersionId> aliasSource(const PhysicalVersionId &version) {
  if (version.derivation.empty())
    return std::nullopt;
  const PhysicalVersionDerivationStep &step = version.derivation.back();
  if (step.kind != PhysicalVersionDerivationKind::AliasView ||
      !step.sourceLogicalValue)
    return std::nullopt;
  return PhysicalVersionId{*step.sourceLogicalValue};
}

} // namespace

std::optional<BufferPlan>
StorageDomain::buildPlan(const StorageCursor &cursor) const {
  if (cursor.bindingOptionIndices.size() != bindings.size() ||
      cursor.familyOptionIndices.size() != families.size())
    return std::nullopt;
  BufferPlan plan = base.plan;
  plan.slotFamilies.clear();
  plan.orderRequirements.clear();
  std::set<StorageObjectId> familyObjects;
  for (auto [index, domain] : llvm::enumerate(bindings)) {
    if (cursor.bindingOptionIndices[index] >= domain.options.size())
      return std::nullopt;
    const BindingOption &option =
        domain.options[cursor.bindingOptionIndices[index]];
    auto binding =
        llvm::find_if(plan.versionBindings, [&](const auto &candidate) {
          return candidate.version == domain.version;
        });
    if (binding == plan.versionBindings.end())
      return std::nullopt;
    binding->object = option.object;
    binding->kind = option.kind;
    if (option.order)
      plan.orderRequirements.push_back(*option.order);
  }
  for (auto [index, domain] : llvm::enumerate(families)) {
    if (cursor.familyOptionIndices[index] >= domain.options.size())
      return std::nullopt;
    const FamilyOption &option =
        domain.options[cursor.familyOptionIndices[index]];
    SlotFamilyId selectedId = domain.id;
    for (StorageObjectId &object : selectedId.objects)
      if (const auto *version =
              std::get_if<PhysicalVersionId>(&object.origin)) {
        auto binding =
            llvm::find_if(plan.versionBindings, [&](const auto &candidate) {
              return candidate.version == *version;
            });
        if (binding == plan.versionBindings.end())
          return std::nullopt;
        object = binding->object;
      }
    llvm::sort(selectedId.objects);
    selectedId.objects.erase(
        std::unique(selectedId.objects.begin(), selectedId.objects.end()),
        selectedId.objects.end());
    if (selectedId.objects.empty())
      return std::nullopt;
    for (const StorageObjectId &object : selectedId.objects)
      if (!familyObjects.insert(object).second)
        return std::nullopt;
    plan.slotFamilies.push_back({std::move(selectedId), domain.occurrence,
                                 option.multiplicity, option.rotation});
  }

  std::set<StorageObjectId> referenced;
  for (const PhysicalVersionStorageBinding &binding : plan.versionBindings)
    referenced.insert(binding.object);
  for (const ReductionGatherStorageBinding &binding :
       plan.gatherStagingBindings)
    referenced.insert(binding.stagingObject);
  for (const SlotFamilyPlan &family : plan.slotFamilies)
    referenced.insert(family.id.objects.begin(), family.id.objects.end());
  for (const StorageObjectPlan &object : plan.storageObjects)
    if (std::holds_alternative<PeerRelayStorageId>(object.id.origin))
      referenced.insert(object.id);
  llvm::erase_if(plan.storageObjects, [&](const StorageObjectPlan &object) {
    return !referenced.count(object.id);
  });
  llvm::sort(plan.storageObjects);
  llvm::sort(plan.versionBindings);
  llvm::sort(plan.slotFamilies);
  llvm::sort(plan.orderRequirements);
  if (std::adjacent_find(plan.orderRequirements.begin(),
                         plan.orderRequirements.end()) !=
      plan.orderRequirements.end())
    return std::nullopt;
  std::map<PhysicalVersionId, uint32_t> indegree;
  std::map<PhysicalVersionId, std::vector<PhysicalVersionId>> successors;
  for (const BufferOrderRequirement &order : plan.orderRequirements) {
    if (order.earlier == order.later)
      return std::nullopt;
    indegree.try_emplace(order.earlier, 0);
    ++indegree[order.later];
    successors[order.earlier].push_back(order.later);
  }
  std::set<PhysicalVersionId> ready;
  for (const auto &[version, degree] : indegree)
    if (degree == 0)
      ready.insert(version);
  size_t visited = 0;
  while (!ready.empty()) {
    PhysicalVersionId version = *ready.begin();
    ready.erase(ready.begin());
    ++visited;
    for (const PhysicalVersionId &successor : successors[version])
      if (--indegree[successor] == 0)
        ready.insert(successor);
  }
  if (visited != indegree.size())
    return std::nullopt;
  return plan;
}

StorageSuccessor StorageDomain::getFirstPlan() const {
  StorageCursor cursor;
  cursor.bindingOptionIndices.assign(bindings.size(), 0);
  cursor.familyOptionIndices.assign(families.size(), 0);
  std::optional<BufferPlan> plan = buildPlan(cursor);
  if (!plan)
    return {StorageSuccessorKind::CompilerBug,
            {},
            {},
            "storage domain has no valid first plan"};
  return {StorageSuccessorKind::Plan, std::move(plan), std::move(cursor)};
}

std::optional<StorageCursor>
StorageDomain::getCursor(const BufferPlan &plan) const {
  StorageCursor cursor;
  cursor.bindingOptionIndices.resize(bindings.size());
  cursor.familyOptionIndices.resize(families.size());
  for (auto [index, domain] : llvm::enumerate(bindings)) {
    auto binding =
        llvm::find_if(plan.versionBindings, [&](const auto &candidate) {
          return candidate.version == domain.version;
        });
    if (binding == plan.versionBindings.end())
      return std::nullopt;
    auto option =
        llvm::find_if(domain.options, [&](const BindingOption &entry) {
          return entry.object == binding->object && entry.kind == binding->kind;
        });
    if (option == domain.options.end())
      return std::nullopt;
    cursor.bindingOptionIndices[index] =
        static_cast<uint32_t>(std::distance(domain.options.begin(), option));
  }
  for (auto [index, domain] : llvm::enumerate(families)) {
    SlotFamilyId selectedId = domain.id;
    for (StorageObjectId &object : selectedId.objects)
      if (const auto *version =
              std::get_if<PhysicalVersionId>(&object.origin)) {
        auto binding =
            llvm::find_if(plan.versionBindings, [&](const auto &candidate) {
              return candidate.version == *version;
            });
        if (binding == plan.versionBindings.end())
          return std::nullopt;
        object = binding->object;
      }
    llvm::sort(selectedId.objects);
    selectedId.objects.erase(
        std::unique(selectedId.objects.begin(), selectedId.objects.end()),
        selectedId.objects.end());
    auto family =
        llvm::find_if(plan.slotFamilies, [&](const SlotFamilyPlan &candidate) {
          return candidate.id == selectedId &&
                 candidate.occurrence == domain.occurrence;
        });
    if (family == plan.slotFamilies.end())
      return std::nullopt;
    auto option = llvm::find_if(domain.options, [&](const FamilyOption &entry) {
      return entry.multiplicity == family->multiplicity &&
             entry.rotation == family->rotationIterators;
    });
    if (option == domain.options.end())
      return std::nullopt;
    cursor.familyOptionIndices[index] =
        static_cast<uint32_t>(std::distance(domain.options.begin(), option));
  }
  std::optional<BufferPlan> rebuilt = buildPlan(cursor);
  if (!rebuilt || !(*rebuilt == plan))
    return std::nullopt;
  return cursor;
}

bool StorageDomain::contains(const BufferPlan &plan) const {
  return getCursor(plan).has_value();
}

const StorageResourceDescription *
StorageDomain::findResource(const StorageObjectId &object) const {
  auto found = llvm::find_if(base.resources, [&](const auto &resource) {
    return resource.object == object;
  });
  return found == base.resources.end() ? nullptr : &*found;
}

bool StorageDomain::advanceCursor(StorageCursor &next) const {
  for (size_t reverse = 0; reverse < families.size(); ++reverse) {
    const size_t index = families.size() - reverse - 1;
    if (++next.familyOptionIndices[index] < families[index].options.size()) {
      for (size_t reset = index + 1; reset < families.size(); ++reset)
        next.familyOptionIndices[reset] = 0;
      return true;
    }
    next.familyOptionIndices[index] = 0;
  }
  for (size_t reverse = 0; reverse < bindings.size(); ++reverse) {
    const size_t index = bindings.size() - reverse - 1;
    if (++next.bindingOptionIndices[index] < bindings[index].options.size()) {
      for (size_t reset = index + 1; reset < bindings.size(); ++reset)
        next.bindingOptionIndices[reset] = 0;
      std::fill(next.familyOptionIndices.begin(),
                next.familyOptionIndices.end(), 0);
      return true;
    }
    next.bindingOptionIndices[index] = 0;
  }
  return false;
}

StorageSuccessor StorageDomain::getNextPlan(const StorageCursor &cursor) const {
  std::optional<BufferPlan> current = buildPlan(cursor);
  if (!current || !contains(*current))
    return {StorageSuccessorKind::CompilerBug,
            {},
            {},
            "storage cursor is outside the current domain"};
  StorageCursor next = cursor;
  while (advanceCursor(next))
    if (std::optional<BufferPlan> plan = buildPlan(next))
      return {StorageSuccessorKind::Plan, std::move(plan), std::move(next)};
  return {StorageSuccessorKind::End};
}

StorageDomainResult
buildStorageDomain(const CanonicalStorageCoordinate &canonical,
                   llvm::ArrayRef<StorageReuseRequirement> reuse,
                   llvm::ArrayRef<SlotFamilyRequirement> familyRequirements) {
  std::map<PhysicalVersionId, PhysicalVersionStorageBinding> baseBindings;
  std::map<StorageObjectId, const StorageObjectPlan *> objects;
  std::map<StorageObjectId, const StorageResourceDescription *> resources;
  for (const StorageObjectPlan &object : canonical.plan.storageObjects)
    if (!objects.try_emplace(object.id, &object).second)
      return failed(StorageDomainFailureKind::BrokenContract,
                    "storage domain has duplicate objects");
  for (const StorageResourceDescription &resource : canonical.resources)
    if (!resources.try_emplace(resource.object, &resource).second)
      return failed(StorageDomainFailureKind::BrokenContract,
                    "storage domain has duplicate resources");
  for (const PhysicalVersionStorageBinding &binding :
       canonical.plan.versionBindings)
    if (!baseBindings.try_emplace(binding.version, binding).second)
      return failed(StorageDomainFailureKind::BrokenContract,
                    "storage domain has duplicate version bindings");
  if (objects.size() != canonical.resources.size() || baseBindings.empty())
    return failed(StorageDomainFailureKind::BrokenContract,
                  "storage domain inventories disagree");

  std::map<PhysicalVersionId, std::vector<StorageReuseRequirement>>
      reuseByValue;
  std::set<std::pair<PhysicalVersionId, StorageObjectId>> reuseKeys;
  for (const StorageReuseRequirement &requirement : reuse) {
    if (!baseBindings.count(requirement.version) ||
        !objects.count(requirement.object) ||
        baseBindings.at(requirement.version).object == requirement.object ||
        !reuseKeys.emplace(requirement.version, requirement.object).second)
      return failed(StorageDomainFailureKind::BrokenContract,
                    "reuse requirements are unknown or duplicated");
    reuseByValue[requirement.version].push_back(requirement);
  }

  std::vector<StorageDomain::BindingDomain> bindings;
  for (const auto &[version, base] : baseBindings) {
    StorageDomain::BindingDomain domain;
    domain.version = version;
    std::optional<PhysicalVersionId> source = aliasSource(version);
    if (source) {
      auto sourceBinding = baseBindings.find(*source);
      if (sourceBinding == baseBindings.end())
        return failed(StorageDomainFailureKind::BrokenContract,
                      "identity alias has no source storage binding");
      domain.options.push_back({sourceBinding->second.object,
                                StorageBindingKind::IdentityAlias,
                                {}});
    } else {
      domain.options.push_back({base.object, StorageBindingKind::Fresh, {}});
    }
    if (!source)
      for (const StorageReuseRequirement &requirement : reuseByValue[version]) {
        auto targetResource = resources.find(base.object);
        auto reuseResource = resources.find(requirement.object);
        auto reuseObject = objects.find(requirement.object);
        if (targetResource == resources.end() ||
            reuseResource == resources.end() || reuseObject == objects.end())
          return failed(StorageDomainFailureKind::BrokenContract,
                        "reuse requirement references a missing object");
        if (reuseObject->second->tile != objects.at(base.object)->tile ||
            !sameResource(*targetResource->second, *reuseResource->second))
          return failed(StorageDomainFailureKind::UnsupportedSemantics,
                        "reuse requirement has incompatible storage resources");
        StorageDomain::BindingOption option;
        option.object = requirement.object;
        option.kind = StorageBindingKind::Reuse;
        if (requirement.proof == StorageReuseProof::RequiresOrder)
          if (const auto *earlier =
                  std::get_if<PhysicalVersionId>(&requirement.object.origin))
            option.order = BufferOrderRequirement{
                *earlier, version, BufferOrderKind::ReuseAfterCompletion};
          else
            return failed(StorageDomainFailureKind::UnsupportedSemantics,
                          "ordered reuse source is not a physical version");
        domain.options.push_back(std::move(option));
      }
    llvm::sort(domain.options, [](const auto &lhs, const auto &rhs) {
      return std::tie(lhs.kind, lhs.object) < std::tie(rhs.kind, rhs.object);
    });
    bindings.push_back(std::move(domain));
  }
  llvm::sort(bindings, [](const auto &lhs, const auto &rhs) {
    return lhs.version < rhs.version;
  });

  std::vector<StorageDomain::FamilyDomain> families;
  std::set<SlotFamilyId> familyIds;
  for (const SlotFamilyRequirement &requirement : familyRequirements) {
    SlotFamilyId id = requirement.id;
    llvm::sort(id.objects);
    if (id.objects.empty() ||
        std::adjacent_find(id.objects.begin(), id.objects.end()) !=
            id.objects.end() ||
        !familyIds.insert(id).second || requirement.lowerBound == 0 ||
        requirement.lowerBound > requirement.upperBound)
      return failed(StorageDomainFailureKind::BrokenContract,
                    "slot family identity or multiplicity bounds are invalid");
    for (const StorageObjectId &object : id.objects)
      if (!objects.count(object))
        return failed(StorageDomainFailureKind::BrokenContract,
                      "slot family references a missing storage object");
    if (requirement.occurrence.axisOccurrences.empty() ||
        llvm::is_contained(requirement.occurrence.axisOccurrences, uint64_t{0}))
      return failed(StorageDomainFailureKind::BrokenContract,
                    "slot family requires a finite exact recurrence");
    uint64_t tripCount = 1;
    for (uint64_t count : requirement.occurrence.axisOccurrences) {
      if (tripCount > std::numeric_limits<uint64_t>::max() / count)
        return failed(StorageDomainFailureKind::Indeterminate,
                      "slot occurrence product exceeds its typed range");
      tripCount *= count;
    }
    const uint32_t representableTrip = static_cast<uint32_t>(
        std::min<uint64_t>(tripCount, std::numeric_limits<uint32_t>::max()));
    if (requirement.upperBound > representableTrip)
      return failed(StorageDomainFailureKind::BrokenContract,
                    "slot multiplicity exceeds the exact occurrence count");
    llvm::SmallVector<uint32_t, 4> activeAxes;
    for (auto [axis, count] :
         llvm::enumerate(requirement.occurrence.axisOccurrences))
      if (count > 1)
        activeAxes.push_back(static_cast<uint32_t>(axis));
    if (requirement.upperBound > 1 &&
        (activeAxes.empty() || requirement.rotationOptions.empty()))
      return failed(
          StorageDomainFailureKind::BrokenContract,
          "multi-slot family requires an exact recurrence and rotation");
    for (const auto &rotation : requirement.rotationOptions) {
      llvm::SmallVector<uint32_t, 4> covered(rotation.begin(), rotation.end());
      llvm::sort(covered);
      if (rotation.empty() || covered != activeAxes ||
          std::adjacent_find(covered.begin(), covered.end()) != covered.end())
        return failed(
            StorageDomainFailureKind::BrokenContract,
            "slot rotation must cover every active recurrence axis once");
    }
    StorageDomain::FamilyDomain family;
    family.id = std::move(id);
    family.occurrence = requirement.occurrence;
    for (uint32_t multiplicity = requirement.lowerBound;
         multiplicity <= requirement.upperBound; ++multiplicity) {
      if (multiplicity == 1) {
        family.options.push_back({1, {}});
        continue;
      }
      for (const auto &rotation : requirement.rotationOptions)
        family.options.push_back({multiplicity, rotation});
      if (multiplicity == std::numeric_limits<uint32_t>::max())
        break;
    }
    families.push_back(std::move(family));
  }
  llvm::sort(families,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  return {StorageDomain(canonical, std::move(bindings), std::move(families)),
          {}};
}

} // namespace wafer::compiler::detail
