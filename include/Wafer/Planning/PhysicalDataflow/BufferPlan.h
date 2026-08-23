//===- BufferPlan.h - Physical storage ownership and lifetime -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_BUFFERPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_BUFFERPLAN_H

#include "Wafer/Planning/PhysicalDataflow/ExecutionOccurrence.h"
#include "Wafer/Planning/PhysicalDataflow/MovementPlan.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

struct ReductionGatherStagingId {
  ReductionGatherId gather;

  friend bool operator==(const ReductionGatherStagingId &lhs,
                         const ReductionGatherStagingId &rhs) {
    return lhs.gather == rhs.gather;
  }
  friend bool operator<(const ReductionGatherStagingId &lhs,
                        const ReductionGatherStagingId &rhs) {
    return lhs.gather < rhs.gather;
  }
};

using StorageObjectOrigin =
    std::variant<PhysicalVersionId, ReductionGatherStagingId>;

struct StorageObjectId {
  StorageObjectOrigin origin;

  friend bool operator==(const StorageObjectId &lhs,
                         const StorageObjectId &rhs) {
    return lhs.origin == rhs.origin;
  }
  friend bool operator!=(const StorageObjectId &lhs,
                         const StorageObjectId &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const StorageObjectId &lhs,
                        const StorageObjectId &rhs) {
    return lhs.origin < rhs.origin;
  }
};

/// One canonical fresh storage object. A plan entry is one independent slot;
/// slot families and multiplicity are introduced only by the full domain.
struct StorageObjectPlan {
  StorageObjectId id;
  TileId tile{0};

  friend bool operator==(const StorageObjectPlan &lhs,
                         const StorageObjectPlan &rhs) {
    return lhs.id == rhs.id && lhs.tile == rhs.tile;
  }
  friend bool operator<(const StorageObjectPlan &lhs,
                        const StorageObjectPlan &rhs) {
    if (lhs.id != rhs.id)
      return lhs.id < rhs.id;
    return lhs.tile.getValue() < rhs.tile.getValue();
  }
};

enum class StorageBindingKind : uint8_t {
  Fresh,
  IdentityAlias,
  Reuse,
};

struct PhysicalVersionStorageBinding {
  PhysicalVersionId version;
  StorageObjectId object;
  StorageBindingKind kind = StorageBindingKind::Fresh;

  friend bool operator==(const PhysicalVersionStorageBinding &lhs,
                         const PhysicalVersionStorageBinding &rhs) {
    return lhs.version == rhs.version && lhs.object == rhs.object &&
           lhs.kind == rhs.kind;
  }
  friend bool operator<(const PhysicalVersionStorageBinding &lhs,
                        const PhysicalVersionStorageBinding &rhs) {
    return std::tie(lhs.version, lhs.object, lhs.kind) <
           std::tie(rhs.version, rhs.object, rhs.kind);
  }
};

struct ReductionGatherStorageBinding {
  ReductionGatherId gather;
  StorageObjectId stagingObject;

  friend bool operator==(const ReductionGatherStorageBinding &lhs,
                         const ReductionGatherStorageBinding &rhs) {
    return lhs.gather == rhs.gather && lhs.stagingObject == rhs.stagingObject;
  }
  friend bool operator<(const ReductionGatherStorageBinding &lhs,
                        const ReductionGatherStorageBinding &rhs) {
    return std::tie(lhs.gather, lhs.stagingObject) <
           std::tie(rhs.gather, rhs.stagingObject);
  }
};

struct SlotFamilyId {
  std::vector<StorageObjectId> objects;

  friend bool operator==(const SlotFamilyId &lhs, const SlotFamilyId &rhs) {
    return lhs.objects == rhs.objects;
  }
  friend bool operator<(const SlotFamilyId &lhs, const SlotFamilyId &rhs) {
    return lhs.objects < rhs.objects;
  }
};

struct SlotFamilyPlan {
  SlotFamilyId id;
  OccurrenceRelationId occurrence;
  uint32_t multiplicity = 1;
  llvm::SmallVector<uint32_t, 4> rotationIterators;

  friend bool operator==(const SlotFamilyPlan &lhs, const SlotFamilyPlan &rhs) {
    return lhs.id == rhs.id && lhs.occurrence == rhs.occurrence &&
           lhs.multiplicity == rhs.multiplicity &&
           lhs.rotationIterators == rhs.rotationIterators;
  }
  friend bool operator<(const SlotFamilyPlan &lhs, const SlotFamilyPlan &rhs) {
    return std::tie(lhs.id, lhs.occurrence, lhs.multiplicity,
                    lhs.rotationIterators) < std::tie(rhs.id, rhs.occurrence,
                                                      rhs.multiplicity,
                                                      rhs.rotationIterators);
  }
};

enum class BufferOrderKind : uint8_t { ReuseAfterCompletion };

struct BufferOrderRequirement {
  PhysicalVersionId earlier;
  PhysicalVersionId later;
  BufferOrderKind kind = BufferOrderKind::ReuseAfterCompletion;

  friend bool operator==(const BufferOrderRequirement &lhs,
                         const BufferOrderRequirement &rhs) {
    return lhs.earlier == rhs.earlier && lhs.later == rhs.later &&
           lhs.kind == rhs.kind;
  }
  friend bool operator<(const BufferOrderRequirement &lhs,
                        const BufferOrderRequirement &rhs) {
    return std::tie(lhs.earlier, lhs.later, lhs.kind) <
           std::tie(rhs.earlier, rhs.later, rhs.kind);
  }
};

struct BufferPlan {
  std::vector<StorageObjectPlan> storageObjects;
  std::vector<PhysicalVersionStorageBinding> versionBindings;
  std::vector<ReductionGatherStorageBinding> gatherStagingBindings;
  std::vector<SlotFamilyPlan> slotFamilies;
  std::vector<BufferOrderRequirement> orderRequirements;

  friend bool operator==(const BufferPlan &lhs, const BufferPlan &rhs) {
    return lhs.storageObjects == rhs.storageObjects &&
           lhs.versionBindings == rhs.versionBindings &&
           lhs.gatherStagingBindings == rhs.gatherStagingBindings &&
           lhs.slotFamilies == rhs.slotFamilies &&
           lhs.orderRequirements == rhs.orderRequirements;
  }
  friend bool operator<(const BufferPlan &lhs, const BufferPlan &rhs) {
    return std::tie(lhs.storageObjects, lhs.versionBindings,
                    lhs.gatherStagingBindings, lhs.slotFamilies,
                    lhs.orderRequirements) <
           std::tie(rhs.storageObjects, rhs.versionBindings,
                    rhs.gatherStagingBindings, rhs.slotFamilies,
                    rhs.orderRequirements);
  }
};

using StorageAccessSite = std::variant<ExecutionInstanceId, MovementActionId>;

struct StorageResourceDescription {
  StorageObjectId object;
  analysis::ExactIndexSet exactDomain;
  analysis::ExactIndexSet residentDomain;
  mlir::Type elementType;
  MemLayout encoding = MemLayout::Tensor;

  StorageResourceDescription(StorageObjectId object,
                             analysis::ExactIndexSet exactDomain,
                             mlir::Type elementType, MemLayout encoding)
      : object(std::move(object)), exactDomain(exactDomain),
        residentDomain(std::move(exactDomain)), elementType(elementType),
        encoding(encoding) {}

  StorageResourceDescription(StorageObjectId object,
                             analysis::ExactIndexSet exactDomain,
                             analysis::ExactIndexSet residentDomain,
                             mlir::Type elementType, MemLayout encoding)
      : object(std::move(object)), exactDomain(std::move(exactDomain)),
        residentDomain(std::move(residentDomain)), elementType(elementType),
        encoding(encoding) {}
};

struct StorageLifetimeDescription {
  StorageObjectId object;
  StorageAccessSite definition;
  std::vector<StorageAccessSite> uses;
};

struct CanonicalStorageCoordinate {
  BufferPlan plan;
  std::vector<StorageResourceDescription> resources;
  std::vector<StorageLifetimeDescription> lifetimes;
};

enum class BrokenStoragePlanReason : uint8_t {
  EmptyStorageInput,
  DuplicatePhysicalVersion,
  MissingPhysicalVersion,
  DuplicateMovementAction,
  MissingMovementResource,
  MissingSerializedExecution,
  PlanBindingMismatch,
  DuplicateStorageObject,
  DuplicateDefinition,
  DuplicateResultDiscard,
  ResourceMismatch,
  MissingDefinition,
  MissingUse,
};

struct BrokenStoragePlan {
  BrokenStoragePlanReason reason = BrokenStoragePlanReason::EmptyStorageInput;
  std::optional<StorageObjectId> object;
  std::string detail;
};

using CanonicalStoragePlanOutcome =
    std::variant<CanonicalStorageCoordinate, BrokenStoragePlan>;

const CanonicalStorageCoordinate *
getCanonicalStorageCoordinate(const CanonicalStoragePlanOutcome &outcome);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_BUFFERPLAN_H
