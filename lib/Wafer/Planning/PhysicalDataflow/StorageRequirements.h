//===- StorageRequirements.h - Production storage facts ----*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEREQUIREMENTS_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEREQUIREMENTS_H

#include "Wafer/Planning/PhysicalDataflow/StorageDomain.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

enum class StorageRequirementFailureKind : uint8_t {
  UnsupportedSemantics,
  Indeterminate,
  BrokenContract,
};

struct StorageRequirementFailure {
  StorageRequirementFailureKind kind =
      StorageRequirementFailureKind::BrokenContract;
  std::string detail;
};

struct DerivedStorageRequirements {
  std::vector<StorageReuseRequirement> reuse;
  std::vector<SlotFamilyRequirement> slotFamilies;
};

struct StorageRequirementLimits {
  uint64_t maxRotationOptions = 100000;
};

struct StorageRequirementDerivationResult {
  std::optional<DerivedStorageRequirements> requirements;
  std::optional<StorageRequirementFailure> failure;

  bool succeeded() const { return requirements.has_value(); }
};

/// Derives storage alternatives from closed G/H/E facts. The query reads no
/// target capacity, byte footprint, actual offsets, loop IR or schedule.
StorageRequirementDerivationResult deriveStorageRequirements(
    const CanonicalStorageCoordinate &canonical,
    const RepresentationPlan &representations, const MovementPlan &movement,
    const TemporalPlan &temporal,
    llvm::ArrayRef<TemporalScopeDescriptor> scopeDescriptors,
    const StorageRequirementLimits &limits = StorageRequirementLimits());

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEREQUIREMENTS_H
