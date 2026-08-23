//===- StructureSpecificStorageDomain.h - Fixed-structure slots -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STRUCTURESPECIFICSTORAGEDOMAIN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STRUCTURESPECIFICSTORAGEDOMAIN_H

#include "Wafer/Planning/PhysicalDataflow/ExecutionStructurePlan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

struct StructureSpecificStorageLimits {
  uint64_t maxRotationPlans = 100000;
};

struct SlotLifetimeRequirement {
  SlotFamilyId family;
  OccurrenceRelationId occurrence;
  std::vector<EventId> readyEvents;
  std::vector<EventId> releaseEvents;
  uint32_t liveStageDistance = 0;
  uint32_t minimumMultiplicity = 1;
  uint32_t maximumMultiplicity = 1;

  friend bool operator==(const SlotLifetimeRequirement &lhs,
                         const SlotLifetimeRequirement &rhs) {
    return lhs.family == rhs.family && lhs.occurrence == rhs.occurrence &&
           lhs.readyEvents == rhs.readyEvents &&
           lhs.releaseEvents == rhs.releaseEvents &&
           lhs.liveStageDistance == rhs.liveStageDistance &&
           lhs.minimumMultiplicity == rhs.minimumMultiplicity &&
           lhs.maximumMultiplicity == rhs.maximumMultiplicity;
  }
  friend bool operator<(const SlotLifetimeRequirement &lhs,
                        const SlotLifetimeRequirement &rhs) {
    return std::tie(lhs.family, lhs.occurrence, lhs.readyEvents,
                    lhs.releaseEvents, lhs.liveStageDistance,
                    lhs.minimumMultiplicity, lhs.maximumMultiplicity) <
           std::tie(rhs.family, rhs.occurrence, rhs.readyEvents,
                    rhs.releaseEvents, rhs.liveStageDistance,
                    rhs.minimumMultiplicity, rhs.maximumMultiplicity);
  }
};

enum class StructureSpecificStorageFailureKind : uint8_t {
  ExactRejection,
  Indeterminate,
  BrokenContract,
};

struct StructureSpecificStorageFailure {
  StructureSpecificStorageFailureKind kind =
      StructureSpecificStorageFailureKind::BrokenContract;
  std::string detail;
};

enum class StructureSpecificStorageSuccessorKind : uint8_t {
  Plan,
  End,
  CompilerBug,
};

class StructureSpecificStorageCursor {
private:
  struct FamilyCursor {
    uint32_t multiplicity = 1;
    uint32_t rotationIndex = 0;
  };
  std::vector<FamilyCursor> families;

  friend class StructureSpecificStorageDomain;
};

class StructureSpecificStorageSuccessor {
public:
  StructureSpecificStorageSuccessorKind getKind() const { return kind; }
  const BufferPlan *getPlan() const { return plan ? &*plan : nullptr; }
  const StructureSpecificStorageCursor *getCursor() const {
    return cursor ? &*cursor : nullptr;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  StructureSpecificStorageSuccessor(
      StructureSpecificStorageSuccessorKind kind,
      std::optional<BufferPlan> plan = {},
      std::optional<StructureSpecificStorageCursor> cursor = {},
      std::string detail = {})
      : kind(kind), plan(std::move(plan)), cursor(std::move(cursor)),
        detail(std::move(detail)) {}

  StructureSpecificStorageSuccessorKind kind;
  std::optional<BufferPlan> plan;
  std::optional<StructureSpecificStorageCursor> cursor;
  std::string detail;

  friend class StructureSpecificStorageDomain;
};

struct StructureSpecificStorageDomainResult;

class StructureSpecificStorageDomain {
public:
  StructureSpecificStorageSuccessor getFirstPlan() const;
  StructureSpecificStorageSuccessor
  getNextPlan(const StructureSpecificStorageCursor &cursor) const;
  bool contains(const BufferPlan &plan) const;
  bool isForStructure(const ExecutionStructurePlan &plan) const {
    return structure == plan;
  }
  llvm::ArrayRef<SlotLifetimeRequirement> getLifetimeRequirements() const {
    return lifetimeRequirements;
  }

private:
  struct FamilyDomain {
    SlotFamilyId id;
    OccurrenceRelationId occurrence;
    uint32_t lowerBound = 1;
    uint32_t upperBound = 1;
    std::vector<llvm::SmallVector<uint32_t, 4>> rotations;
  };

  StructureSpecificStorageDomain(
      ExecutionStructurePlan structure, BufferPlan base,
      std::vector<FamilyDomain> families,
      std::vector<SlotLifetimeRequirement> lifetimeRequirements)
      : structure(std::move(structure)), base(std::move(base)),
        families(std::move(families)),
        lifetimeRequirements(std::move(lifetimeRequirements)) {}

  BufferPlan buildPlan(const StructureSpecificStorageCursor &cursor) const;
  std::optional<StructureSpecificStorageCursor>
  getCursor(const BufferPlan &plan) const;
  bool advance(StructureSpecificStorageCursor &cursor) const;

  ExecutionStructurePlan structure;
  BufferPlan base;
  std::vector<FamilyDomain> families;
  std::vector<SlotLifetimeRequirement> lifetimeRequirements;

  friend StructureSpecificStorageDomainResult
  buildStructureSpecificStorageDomain(const ExecutionStructurePlan &,
                                      const BufferPlan &,
                                      llvm::ArrayRef<PlannedEvent>,
                                      const StructureSpecificStorageLimits &);
};

struct StructureSpecificStorageDomainResult {
  std::optional<StructureSpecificStorageDomain> domain;
  std::optional<StructureSpecificStorageFailure> failure;

  bool succeeded() const { return domain.has_value(); }
};

StructureSpecificStorageDomainResult buildStructureSpecificStorageDomain(
    const ExecutionStructurePlan &structure, const BufferPlan &initial,
    const EventGraph &foundation,
    const StructureSpecificStorageLimits &limits =
        StructureSpecificStorageLimits());

StructureSpecificStorageDomainResult buildStructureSpecificStorageDomain(
    const ExecutionStructurePlan &structure, const BufferPlan &initial,
    llvm::ArrayRef<PlannedEvent> foundationEvents,
    const StructureSpecificStorageLimits &limits =
        StructureSpecificStorageLimits());

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STRUCTURESPECIFICSTORAGEDOMAIN_H
