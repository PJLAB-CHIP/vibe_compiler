//===- BufferPlan.h - Physical storage ownership and lifetime -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_BUFFERPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_BUFFERPLAN_H

#include "Wafer/Planning/PhysicalDataflow/MovementPlan.h"

#include <cstdint>
#include <optional>
#include <string>
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
};

struct PhysicalVersionStorageBinding {
  PhysicalVersionId version;
  StorageObjectId object;
};

struct ReductionGatherStorageBinding {
  ReductionGatherId gather;
  StorageObjectId stagingObject;
};

struct BufferPlan {
  std::vector<StorageObjectPlan> storageObjects;
  std::vector<PhysicalVersionStorageBinding> versionBindings;
  std::vector<ReductionGatherStorageBinding> gatherStagingBindings;
};

using StorageAccessSite = std::variant<ExecutionInstanceId, MovementActionId>;

struct StorageResourceDescription {
  StorageObjectId object;
  analysis::ExactIndexSet exactDomain;
  mlir::Type elementType;
  MemLayout encoding = MemLayout::Tensor;
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
