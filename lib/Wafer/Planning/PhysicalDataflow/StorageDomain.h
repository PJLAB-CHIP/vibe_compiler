//===- StorageDomain.h - Storage binding and slot domain -----*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEDOMAIN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEDOMAIN_H

#include "Wafer/Planning/PhysicalDataflow/CanonicalStoragePlan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

enum class StorageReuseProof : uint8_t {
  ProvenDisjoint,
  RequiresOrder,
};

struct StorageReuseRequirement {
  PhysicalVersionId version;
  StorageObjectId object;
  StorageReuseProof proof = StorageReuseProof::ProvenDisjoint;

  friend bool operator==(const StorageReuseRequirement &lhs,
                         const StorageReuseRequirement &rhs) {
    return lhs.version == rhs.version && lhs.object == rhs.object &&
           lhs.proof == rhs.proof;
  }
};

struct SlotFamilyRequirement {
  SlotFamilyId id;
  OccurrenceRelationId occurrence;
  uint32_t lowerBound = 1;
  uint32_t upperBound = 1;
  std::vector<llvm::SmallVector<uint32_t, 4>> rotationOptions;

  friend bool operator==(const SlotFamilyRequirement &lhs,
                         const SlotFamilyRequirement &rhs) {
    return lhs.id == rhs.id && lhs.occurrence == rhs.occurrence &&
           lhs.lowerBound == rhs.lowerBound &&
           lhs.upperBound == rhs.upperBound &&
           lhs.rotationOptions == rhs.rotationOptions;
  }
};

enum class StorageDomainFailureKind : uint8_t {
  UnsupportedSemantics,
  Indeterminate,
  BrokenContract,
};

struct StorageDomainFailure {
  StorageDomainFailureKind kind = StorageDomainFailureKind::BrokenContract;
  std::string detail;
};

enum class StorageSuccessorKind : uint8_t { Plan, End, CompilerBug };

class StorageCursor {
private:
  std::vector<uint32_t> bindingOptionIndices;
  std::vector<uint32_t> familyOptionIndices;

  friend class StorageDomain;
};

class StorageSuccessor {
public:
  StorageSuccessorKind getKind() const { return kind; }
  const BufferPlan *getPlan() const { return plan ? &*plan : nullptr; }
  const StorageCursor *getCursor() const { return cursor ? &*cursor : nullptr; }
  llvm::StringRef getDetail() const { return detail; }

private:
  StorageSuccessor(StorageSuccessorKind kind,
                   std::optional<BufferPlan> plan = {},
                   std::optional<StorageCursor> cursor = {},
                   std::string detail = {})
      : kind(kind), plan(std::move(plan)), cursor(std::move(cursor)),
        detail(std::move(detail)) {}

  StorageSuccessorKind kind;
  std::optional<BufferPlan> plan;
  std::optional<StorageCursor> cursor;
  std::string detail;

  friend class StorageDomain;
};

struct StorageDomainResult;

class StorageDomain {
public:
  StorageSuccessor getFirstPlan() const;
  StorageSuccessor getNextPlan(const StorageCursor &cursor) const;
  bool contains(const BufferPlan &plan) const;
  const StorageResourceDescription *
  findResource(const StorageObjectId &object) const;
  const CanonicalStorageCoordinate &getCanonicalCoordinate() const {
    return base;
  }

private:
  struct BindingOption {
    StorageObjectId object;
    StorageBindingKind kind = StorageBindingKind::Fresh;
    std::optional<BufferOrderRequirement> order;
  };

  struct BindingDomain {
    PhysicalVersionId version;
    std::vector<BindingOption> options;
  };

  struct FamilyOption {
    uint32_t multiplicity = 1;
    llvm::SmallVector<uint32_t, 4> rotation;
  };

  struct FamilyDomain {
    SlotFamilyId id;
    OccurrenceRelationId occurrence;
    std::vector<FamilyOption> options;
  };

  StorageDomain(CanonicalStorageCoordinate base,
                std::vector<BindingDomain> bindings,
                std::vector<FamilyDomain> families)
      : base(std::move(base)), bindings(std::move(bindings)),
        families(std::move(families)) {}

  std::optional<BufferPlan> buildPlan(const StorageCursor &cursor) const;
  std::optional<StorageCursor> getCursor(const BufferPlan &plan) const;
  bool advanceCursor(StorageCursor &cursor) const;

  CanonicalStorageCoordinate base;
  std::vector<BindingDomain> bindings;
  std::vector<FamilyDomain> families;

  friend StorageDomainResult
  buildStorageDomain(const CanonicalStorageCoordinate &,
                     llvm::ArrayRef<StorageReuseRequirement>,
                     llvm::ArrayRef<SlotFamilyRequirement>);
};

struct StorageDomainResult {
  std::optional<StorageDomain> domain;
  std::optional<StorageDomainFailure> failure;

  bool succeeded() const { return domain.has_value(); }
};

StorageDomainResult
buildStorageDomain(const CanonicalStorageCoordinate &canonical,
                   llvm::ArrayRef<StorageReuseRequirement> reuse = {},
                   llvm::ArrayRef<SlotFamilyRequirement> families = {});

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STORAGEDOMAIN_H
