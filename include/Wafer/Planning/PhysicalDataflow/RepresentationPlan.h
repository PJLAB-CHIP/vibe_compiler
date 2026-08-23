//===- RepresentationPlan.h - Logical and physical versions --*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_REPRESENTATIONPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_REPRESENTATIONPLAN_H

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

#include "mlir/IR/Types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

struct BoundaryRegionValueId {
  analysis::RootRegionWorkId work;
  DemandFragmentId fragment;

  friend bool operator==(const BoundaryRegionValueId &lhs,
                         const BoundaryRegionValueId &rhs) {
    return lhs.work == rhs.work && lhs.fragment == rhs.fragment;
  }
  friend bool operator<(const BoundaryRegionValueId &lhs,
                        const BoundaryRegionValueId &rhs) {
    if (lhs.work != rhs.work)
      return lhs.work < rhs.work;
    return lhs.fragment < rhs.fragment;
  }
};

struct SupportRegionValueId {
  analysis::RootRegionWorkId work;
  analysis::SupportValueId support;

  friend bool operator==(const SupportRegionValueId &lhs,
                         const SupportRegionValueId &rhs) {
    return lhs.work == rhs.work && lhs.support == rhs.support;
  }
  friend bool operator<(const SupportRegionValueId &lhs,
                        const SupportRegionValueId &rhs) {
    if (lhs.work != rhs.work)
      return lhs.work < rhs.work;
    return lhs.support < rhs.support;
  }
};

struct ExecutionResultValueId {
  ExecutionInstanceId execution;
  uint32_t result = 0;

  friend bool operator==(const ExecutionResultValueId &lhs,
                         const ExecutionResultValueId &rhs) {
    return lhs.execution == rhs.execution && lhs.result == rhs.result;
  }
  friend bool operator<(const ExecutionResultValueId &lhs,
                        const ExecutionResultValueId &rhs) {
    if (!(lhs.execution == rhs.execution))
      return lhs.execution < rhs.execution;
    return lhs.result < rhs.result;
  }
};

struct ReductionPartialValueId {
  ExecutionInstanceId execution;
  ReductionGroupId group;
  uint32_t result = 0;

  friend bool operator==(const ReductionPartialValueId &lhs,
                         const ReductionPartialValueId &rhs) {
    return lhs.execution == rhs.execution && lhs.group == rhs.group &&
           lhs.result == rhs.result;
  }
  friend bool operator<(const ReductionPartialValueId &lhs,
                        const ReductionPartialValueId &rhs) {
    if (!(lhs.execution == rhs.execution))
      return lhs.execution < rhs.execution;
    if (lhs.group != rhs.group)
      return lhs.group < rhs.group;
    return lhs.result < rhs.result;
  }
};

struct CoupledComponentValueId {
  ExecutionInstanceId execution;
  ReductionGroupId group;
  wafer::CoupledReductionComponentKind component =
      wafer::CoupledReductionComponentKind::Maximum;

  friend bool operator==(const CoupledComponentValueId &lhs,
                         const CoupledComponentValueId &rhs) {
    return lhs.execution == rhs.execution && lhs.group == rhs.group &&
           lhs.component == rhs.component;
  }
  friend bool operator<(const CoupledComponentValueId &lhs,
                        const CoupledComponentValueId &rhs) {
    if (!(lhs.execution == rhs.execution))
      return lhs.execution < rhs.execution;
    if (lhs.group != rhs.group)
      return lhs.group < rhs.group;
    return lhs.component < rhs.component;
  }
};

using RegionValueVersionId =
    std::variant<BoundaryRegionValueId, SupportRegionValueId,
                 ExecutionResultValueId, ReductionPartialValueId,
                 CoupledComponentValueId>;

struct BoundaryRepresentationUseId {
  BoundaryRegionValueId value;

  friend bool operator==(const BoundaryRepresentationUseId &lhs,
                         const BoundaryRepresentationUseId &rhs) {
    return lhs.value == rhs.value;
  }
  friend bool operator<(const BoundaryRepresentationUseId &lhs,
                        const BoundaryRepresentationUseId &rhs) {
    return lhs.value < rhs.value;
  }
};

struct LocalRepresentationUseId {
  analysis::RootRegionWorkId consumerWork;
  DemandFragmentId fragment;

  friend bool operator==(const LocalRepresentationUseId &lhs,
                         const LocalRepresentationUseId &rhs) {
    return lhs.consumerWork == rhs.consumerWork && lhs.fragment == rhs.fragment;
  }
  friend bool operator<(const LocalRepresentationUseId &lhs,
                        const LocalRepresentationUseId &rhs) {
    if (lhs.consumerWork != rhs.consumerWork)
      return lhs.consumerWork < rhs.consumerWork;
    return lhs.fragment < rhs.fragment;
  }
};

using RepresentationUseId =
    std::variant<BoundaryRepresentationUseId, LocalRepresentationUseId>;

struct SharedRepresentationAnchor {
  friend bool operator==(SharedRepresentationAnchor,
                         SharedRepresentationAnchor) {
    return true;
  }
  friend bool operator<(SharedRepresentationAnchor,
                        SharedRepresentationAnchor) {
    return false;
  }
};

struct UseRepresentationAnchor {
  RepresentationUseId use;

  friend bool operator==(const UseRepresentationAnchor &lhs,
                         const UseRepresentationAnchor &rhs) {
    return lhs.use == rhs.use;
  }
  friend bool operator<(const UseRepresentationAnchor &lhs,
                        const UseRepresentationAnchor &rhs) {
    return lhs.use < rhs.use;
  }
};

using RepresentationAnchorId =
    std::variant<SharedRepresentationAnchor, UseRepresentationAnchor>;

enum class PhysicalVersionDerivationKind : uint8_t {
  LayoutConversion,
  AliasView,
};

struct PhysicalVersionDerivationStep {
  PhysicalVersionDerivationKind kind =
      PhysicalVersionDerivationKind::LayoutConversion;
  MemLayout sourceEncoding = MemLayout::Tensor;
  MemLayout encoding = MemLayout::Tensor;
  RepresentationAnchorId anchor = SharedRepresentationAnchor{};
  std::optional<RegionValueVersionId> sourceLogicalValue;

  friend bool operator==(const PhysicalVersionDerivationStep &lhs,
                         const PhysicalVersionDerivationStep &rhs) {
    return lhs.kind == rhs.kind && lhs.sourceEncoding == rhs.sourceEncoding &&
           lhs.encoding == rhs.encoding && lhs.anchor == rhs.anchor &&
           lhs.sourceLogicalValue == rhs.sourceLogicalValue;
  }
  friend bool operator<(const PhysicalVersionDerivationStep &lhs,
                        const PhysicalVersionDerivationStep &rhs) {
    return std::tie(lhs.kind, lhs.sourceEncoding, lhs.encoding, lhs.anchor,
                    lhs.sourceLogicalValue) <
           std::tie(rhs.kind, rhs.sourceEncoding, rhs.encoding, rhs.anchor,
                    rhs.sourceLogicalValue);
  }
};

struct PhysicalVersionId {
  RegionValueVersionId logicalValue;
  std::vector<PhysicalVersionDerivationStep> derivation;

  friend bool operator==(const PhysicalVersionId &lhs,
                         const PhysicalVersionId &rhs) {
    return lhs.logicalValue == rhs.logicalValue &&
           lhs.derivation == rhs.derivation;
  }
  friend bool operator!=(const PhysicalVersionId &lhs,
                         const PhysicalVersionId &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const PhysicalVersionId &lhs,
                        const PhysicalVersionId &rhs) {
    if (lhs.logicalValue.index() != rhs.logicalValue.index())
      return lhs.logicalValue.index() < rhs.logicalValue.index();
    const bool logicalLess = std::visit(
        [&](const auto &lhsValue) {
          using T = std::decay_t<decltype(lhsValue)>;
          return lhsValue < std::get<T>(rhs.logicalValue);
        },
        lhs.logicalValue);
    if (logicalLess)
      return true;
    const bool logicalGreater = std::visit(
        [&](const auto &rhsValue) {
          using T = std::decay_t<decltype(rhsValue)>;
          return rhsValue < std::get<T>(lhs.logicalValue);
        },
        rhs.logicalValue);
    return !logicalGreater && lhs.derivation < rhs.derivation;
  }
};

struct PhysicalVersionPlan {
  PhysicalVersionId id;
  MemLayout encoding = MemLayout::Tensor;

  friend bool operator==(const PhysicalVersionPlan &lhs,
                         const PhysicalVersionPlan &rhs) {
    return lhs.id == rhs.id && lhs.encoding == rhs.encoding;
  }
  friend bool operator<(const PhysicalVersionPlan &lhs,
                        const PhysicalVersionPlan &rhs) {
    if (lhs.id != rhs.id)
      return lhs.id < rhs.id;
    return lhs.encoding < rhs.encoding;
  }
};

struct LogicalRepresentationPlan {
  RegionValueVersionId value;
  PhysicalVersionId primary;

  friend bool operator==(const LogicalRepresentationPlan &lhs,
                         const LogicalRepresentationPlan &rhs) {
    return lhs.value == rhs.value && lhs.primary == rhs.primary;
  }
  friend bool operator<(const LogicalRepresentationPlan &lhs,
                        const LogicalRepresentationPlan &rhs) {
    if (!(lhs.value == rhs.value))
      return PhysicalVersionId{lhs.value} < PhysicalVersionId{rhs.value};
    return lhs.primary < rhs.primary;
  }
};

struct PhysicalUseBinding {
  RepresentationUseId use;
  PhysicalVersionId version;

  friend bool operator==(const PhysicalUseBinding &lhs,
                         const PhysicalUseBinding &rhs) {
    return lhs.use == rhs.use && lhs.version == rhs.version;
  }
  friend bool operator<(const PhysicalUseBinding &lhs,
                        const PhysicalUseBinding &rhs) {
    if (!(lhs.use == rhs.use))
      return lhs.use < rhs.use;
    return lhs.version < rhs.version;
  }
};

struct RepresentationPlan {
  std::vector<LogicalRepresentationPlan> logicalValues;
  std::vector<PhysicalVersionPlan> physicalVersions;
  std::vector<PhysicalUseBinding> uses;

  friend bool operator==(const RepresentationPlan &lhs,
                         const RepresentationPlan &rhs) {
    return lhs.logicalValues == rhs.logicalValues &&
           lhs.physicalVersions == rhs.physicalVersions && lhs.uses == rhs.uses;
  }
  friend bool operator<(const RepresentationPlan &lhs,
                        const RepresentationPlan &rhs) {
    return std::tie(lhs.logicalValues, lhs.physicalVersions, lhs.uses) <
           std::tie(rhs.logicalValues, rhs.physicalVersions, rhs.uses);
  }
};

struct RepresentationResourceDescription {
  PhysicalVersionId version;
  analysis::ExactIndexSet exactDomain;
  mlir::Type elementType;
  MemLayout encoding = MemLayout::Tensor;
};

struct CanonicalRepresentationCoordinate {
  RepresentationPlan plan;
  std::vector<RepresentationResourceDescription> resources;
};

enum class BrokenRepresentationPlanReason : uint8_t {
  DuplicateRootWork,
  PlanWorkMismatch,
  MissingValueDescriptor,
  DuplicateLogicalValue,
};

struct BrokenRepresentationPlan {
  BrokenRepresentationPlanReason reason =
      BrokenRepresentationPlanReason::PlanWorkMismatch;
  std::optional<analysis::RootRegionWorkId> work;
  std::string detail;
};

enum class UnsupportedRepresentationFeature : uint8_t {
  ElementType,
  TensorEncoding,
};

struct UnsupportedRepresentationPlan {
  UnsupportedRepresentationFeature feature =
      UnsupportedRepresentationFeature::ElementType;
  std::optional<analysis::RootRegionWorkId> work;
  std::string detail;
};

using CanonicalRepresentationPlanOutcome =
    std::variant<CanonicalRepresentationCoordinate,
                 UnsupportedRepresentationPlan, BrokenRepresentationPlan>;

const CanonicalRepresentationCoordinate *getCanonicalRepresentationCoordinate(
    const CanonicalRepresentationPlanOutcome &outcome);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_REPRESENTATIONPLAN_H
