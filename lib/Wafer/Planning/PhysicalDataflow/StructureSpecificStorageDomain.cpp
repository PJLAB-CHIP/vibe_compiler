//===- StructureSpecificStorageDomain.cpp - Fixed-structure slots -----===//

#include "Wafer/Planning/PhysicalDataflow/StructureSpecificStorageDomain.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace wafer::compiler::detail {
namespace {

StructureSpecificStorageDomainResult
failed(StructureSpecificStorageFailureKind kind, llvm::StringRef detail) {
  return {{}, StructureSpecificStorageFailure{kind, detail.str()}};
}

std::optional<uint32_t> getMinimumMultiplicity(uint32_t liveStageDistance,
                                               uint64_t launchDistance) {
  if (launchDistance == 0)
    return std::nullopt;
  const uint64_t overlap =
      liveStageDistance == 0
          ? 0
          : 1 + (uint64_t(liveStageDistance) - 1) / launchDistance;
  if (overlap >= std::numeric_limits<uint32_t>::max())
    return std::nullopt;
  return static_cast<uint32_t>(1 + overlap);
}

} // namespace

BufferPlan StructureSpecificStorageDomain::buildPlan(
    const StructureSpecificStorageCursor &cursor) const {
  BufferPlan plan = base;
  plan.slotFamilies.clear();
  if (cursor.families.size() != families.size())
    return plan;
  for (auto [family, choice] : llvm::zip_equal(families, cursor.families)) {
    llvm::SmallVector<uint32_t, 4> rotation;
    if (choice.multiplicity > 1) {
      if (choice.rotationIndex >= family.rotations.size())
        return base;
      rotation = family.rotations[choice.rotationIndex];
    }
    plan.slotFamilies.push_back(
        {family.id, family.occurrence, choice.multiplicity, rotation});
  }
  llvm::sort(plan.slotFamilies);
  return plan;
}

StructureSpecificStorageSuccessor
StructureSpecificStorageDomain::getFirstPlan() const {
  StructureSpecificStorageCursor cursor;
  for (const FamilyDomain &family : families)
    cursor.families.push_back({family.lowerBound, 0});
  BufferPlan plan = buildPlan(cursor);
  if (!contains(plan))
    return {StructureSpecificStorageSuccessorKind::CompilerBug,
            {},
            {},
            "structure-specific storage has no valid first plan"};
  return {StructureSpecificStorageSuccessorKind::Plan, std::move(plan),
          std::move(cursor)};
}

bool StructureSpecificStorageDomain::advance(
    StructureSpecificStorageCursor &cursor) const {
  for (size_t reverse = 0; reverse < families.size(); ++reverse) {
    const size_t index = families.size() - reverse - 1;
    const FamilyDomain &family = families[index];
    auto &choice = cursor.families[index];
    if (choice.multiplicity > 1 &&
        choice.rotationIndex + 1 < family.rotations.size()) {
      ++choice.rotationIndex;
      return true;
    }
    choice.rotationIndex = 0;
    if (choice.multiplicity < family.upperBound) {
      ++choice.multiplicity;
      return true;
    }
    choice.multiplicity = family.lowerBound;
  }
  return false;
}

StructureSpecificStorageSuccessor StructureSpecificStorageDomain::getNextPlan(
    const StructureSpecificStorageCursor &cursor) const {
  if (!getCursor(buildPlan(cursor)))
    return {StructureSpecificStorageSuccessorKind::CompilerBug,
            {},
            {},
            "structure-specific storage cursor is outside its domain"};
  StructureSpecificStorageCursor next = cursor;
  if (!advance(next))
    return {StructureSpecificStorageSuccessorKind::End};
  BufferPlan plan = buildPlan(next);
  if (!contains(plan))
    return {StructureSpecificStorageSuccessorKind::CompilerBug,
            {},
            {},
            "structure-specific storage successor produced an invalid plan"};
  return {StructureSpecificStorageSuccessorKind::Plan, std::move(plan),
          std::move(next)};
}

std::optional<StructureSpecificStorageCursor>
StructureSpecificStorageDomain::getCursor(const BufferPlan &plan) const {
  BufferPlan withoutFamilies = plan;
  withoutFamilies.slotFamilies.clear();
  if (!(withoutFamilies == base) || plan.slotFamilies.size() != families.size())
    return std::nullopt;
  StructureSpecificStorageCursor cursor;
  for (const FamilyDomain &family : families) {
    auto selected =
        llvm::find_if(plan.slotFamilies, [&](const auto &candidate) {
          return candidate.id == family.id &&
                 candidate.occurrence == family.occurrence;
        });
    if (selected == plan.slotFamilies.end() ||
        selected->multiplicity < family.lowerBound ||
        selected->multiplicity > family.upperBound)
      return std::nullopt;
    StructureSpecificStorageCursor::FamilyCursor choice;
    choice.multiplicity = selected->multiplicity;
    if (choice.multiplicity == 1) {
      if (!selected->rotationIterators.empty())
        return std::nullopt;
    } else {
      auto rotation = llvm::find(family.rotations, selected->rotationIterators);
      if (rotation == family.rotations.end())
        return std::nullopt;
      choice.rotationIndex = static_cast<uint32_t>(
          std::distance(family.rotations.begin(), rotation));
    }
    cursor.families.push_back(choice);
  }
  if (!(buildPlan(cursor) == plan))
    return std::nullopt;
  return cursor;
}

bool StructureSpecificStorageDomain::contains(const BufferPlan &plan) const {
  return getCursor(plan).has_value();
}

StructureSpecificStorageDomainResult buildStructureSpecificStorageDomain(
    const ExecutionStructurePlan &structure, const BufferPlan &initial,
    llvm::ArrayRef<PlannedEvent> foundationEvents) {
  if (structure.scopes.empty() || initial.storageObjects.empty() ||
      foundationEvents.empty())
    return failed(
        StructureSpecificStorageFailureKind::BrokenContract,
        "structure-specific storage requires K, I, and J foundation facts");

  std::set<StorageObjectId> selectedObjects;
  for (const StorageObjectPlan &object : initial.storageObjects)
    if (!selectedObjects.insert(object.id).second)
      return failed(StructureSpecificStorageFailureKind::BrokenContract,
                    "initial storage has duplicate objects");
  for (const PhysicalVersionStorageBinding &binding : initial.versionBindings)
    if (!selectedObjects.count(binding.object))
      return failed(StructureSpecificStorageFailureKind::BrokenContract,
                    "initial storage binding references an unknown object");
  for (const ReductionGatherStorageBinding &binding :
       initial.gatherStagingBindings)
    if (!selectedObjects.count(binding.stagingObject))
      return failed(StructureSpecificStorageFailureKind::BrokenContract,
                    "initial gather binding references an unknown object");

  std::set<EventId> graphEvents;
  for (const PlannedEvent &event : foundationEvents)
    graphEvents.insert(event.id);
  std::set<EventId> structureEvents;
  std::set<StorageObjectId> assignedFamilies;
  std::vector<StructureSpecificStorageDomain::FamilyDomain> families;
  std::vector<SlotLifetimeRequirement> lifetimes;
  for (const ExecutionStructureChoice &choice : structure.scopes) {
    const PipelineScopeId &scope = getPipelineScope(choice);
    if (scope.events.empty())
      return failed(StructureSpecificStorageFailureKind::BrokenContract,
                    "execution structure has an empty scope");
    for (const EventId &event : scope.events)
      if (!graphEvents.count(event) || !structureEvents.insert(event).second)
        return failed(StructureSpecificStorageFailureKind::BrokenContract,
                      "execution structure has stale or overlapping events");
    const auto *pipelined = std::get_if<PipelinedExecutionStructure>(&choice);
    if (!pipelined)
      continue;
    const uint32_t recurrenceAxis = pipelined->iteration.recurrenceAxis;
    const bool occurrenceSumOverflows =
        pipelined->iteration.prefixCount >
            std::numeric_limits<uint64_t>::max() -
                pipelined->iteration.tailCount ||
        pipelined->iteration.steadyTripCount >
            std::numeric_limits<uint64_t>::max() -
                pipelined->iteration.prefixCount -
                pipelined->iteration.tailCount;
    if (pipelined->launchDistance != 1 ||
        recurrenceAxis >= pipelined->recurrence.axisOccurrences.size() ||
        occurrenceSumOverflows || pipelined->iteration.prefixCount != 1 ||
        pipelined->iteration.tailCount > 1 ||
        pipelined->recurrence.axisOccurrences[recurrenceAxis] !=
            pipelined->iteration.prefixCount +
                pipelined->iteration.steadyTripCount +
                pipelined->iteration.tailCount ||
        pipelined->iteration.steadyTripCount < 2)
      return failed(StructureSpecificStorageFailureKind::BrokenContract,
                    "pipelined structure has an invalid iteration class");
    std::map<EventId, uint32_t> stages;
    for (const EventStageAssignment &assignment : pipelined->eventStages)
      if (!stages.try_emplace(assignment.event, assignment.stage.getValue())
               .second)
        return failed(StructureSpecificStorageFailureKind::BrokenContract,
                      "pipelined structure has duplicate stage assignments");

    struct ObjectLifetime {
      std::vector<EventId> ready;
      std::vector<EventId> release;
    };
    std::map<StorageObjectId, std::map<StorageObjectId, ObjectLifetime>>
        byObject;
    for (const EventId &event : scope.events) {
      const auto *buffer = std::get_if<BufferEventAction>(&event.action);
      if (!buffer)
        continue;
      if (!selectedObjects.count(buffer->storageObject))
        return failed(StructureSpecificStorageFailureKind::BrokenContract,
                      "pipeline event references an unknown selected object");
      if (event.kind == PlannedEventKind::BufferReady)
        byObject[buffer->storageObject][buffer->semanticObject].ready.push_back(
            event);
      else if (event.kind == PlannedEventKind::BufferRelease)
        byObject[buffer->storageObject][buffer->semanticObject]
            .release.push_back(event);
    }
    for (auto &[object, semanticLifetimes] : byObject) {
      if (semanticLifetimes.empty() || !assignedFamilies.insert(object).second)
        return failed(StructureSpecificStorageFailureKind::BrokenContract,
                      "pipeline object lacks one closed lifetime family");
      uint32_t liveDistance = 0;
      std::vector<EventId> readyEvents;
      std::vector<EventId> releaseEvents;
      for (auto &[semanticObject, lifetime] : semanticLifetimes) {
        (void)semanticObject;
        if (lifetime.ready.empty() || lifetime.release.empty())
          return failed(StructureSpecificStorageFailureKind::BrokenContract,
                        "semantic object lacks ready or release events");
        uint32_t firstReady = std::numeric_limits<uint32_t>::max();
        uint32_t lastRelease = 0;
        for (const EventId &event : lifetime.ready) {
          auto stage = stages.find(event);
          if (stage == stages.end())
            return failed(StructureSpecificStorageFailureKind::BrokenContract,
                          "buffer-ready event has no selected stage");
          firstReady = std::min(firstReady, stage->second);
          readyEvents.push_back(event);
        }
        for (const EventId &event : lifetime.release) {
          auto stage = stages.find(event);
          if (stage == stages.end())
            return failed(StructureSpecificStorageFailureKind::BrokenContract,
                          "buffer-release event has no selected stage");
          lastRelease = std::max(lastRelease, stage->second);
          releaseEvents.push_back(event);
        }
        if (lastRelease < firstReady)
          return failed(StructureSpecificStorageFailureKind::ExactRejection,
                        "buffer release precedes its ready stage");
        liveDistance = std::max(liveDistance, lastRelease - firstReady);
      }
      std::optional<uint32_t> lower =
          getMinimumMultiplicity(liveDistance, pipelined->launchDistance);
      const uint32_t upper = static_cast<uint32_t>(std::min<uint64_t>(
          pipelined->recurrence.axisOccurrences[recurrenceAxis],
          std::numeric_limits<uint32_t>::max()));
      if (!lower || *lower > upper)
        return failed(StructureSpecificStorageFailureKind::ExactRejection,
                      "pipeline live distance has no representable slot count");
      std::vector<llvm::SmallVector<uint32_t, 4>> rotations{
          llvm::SmallVector<uint32_t, 4>{recurrenceAxis}};
      if (upper > 1 &&
          pipelined->recurrence.axisOccurrences[recurrenceAxis] <= 1)
        return failed(StructureSpecificStorageFailureKind::BrokenContract,
                      "multi-slot family has no active recurrence axis");
      SlotFamilyId id{{object}};
      families.push_back(
          {id, pipelined->recurrence, *lower, upper, std::move(rotations)});
      llvm::sort(readyEvents);
      llvm::sort(releaseEvents);
      lifetimes.push_back({id, pipelined->recurrence, pipelined->iteration,
                           std::move(readyEvents), std::move(releaseEvents),
                           liveDistance, *lower, upper});
    }
  }
  if (structureEvents != graphEvents)
    return failed(StructureSpecificStorageFailureKind::BrokenContract,
                  "execution structure does not cover the foundation graph");
  llvm::sort(families,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(lifetimes);
  BufferPlan base = initial;
  base.slotFamilies.clear();
  return {StructureSpecificStorageDomain(structure, std::move(base),
                                         std::move(families),
                                         std::move(lifetimes)),
          {}};
}

StructureSpecificStorageDomainResult
buildStructureSpecificStorageDomain(const ExecutionStructurePlan &structure,
                                    const BufferPlan &initial,
                                    const EventGraph &foundation) {
  return buildStructureSpecificStorageDomain(structure, initial,
                                             foundation.getEvents());
}

} // namespace wafer::compiler::detail
