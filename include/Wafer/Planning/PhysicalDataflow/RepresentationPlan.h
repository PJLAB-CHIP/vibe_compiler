//===- RepresentationPlan.h - Logical and physical versions --*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_REPRESENTATIONPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_REPRESENTATIONPLAN_H

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

#include "mlir/IR/Types.h"

#include <cstdint>
#include <optional>
#include <string>
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

struct PhysicalVersionId {
  RegionValueVersionId logicalValue;

  friend bool operator==(const PhysicalVersionId &lhs,
                         const PhysicalVersionId &rhs) {
    return lhs.logicalValue == rhs.logicalValue;
  }
  friend bool operator<(const PhysicalVersionId &lhs,
                        const PhysicalVersionId &rhs) {
    if (lhs.logicalValue.index() != rhs.logicalValue.index())
      return lhs.logicalValue.index() < rhs.logicalValue.index();
    return std::visit(
        [&](const auto &lhsValue) {
          using T = std::decay_t<decltype(lhsValue)>;
          return lhsValue < std::get<T>(rhs.logicalValue);
        },
        lhs.logicalValue);
  }
};

struct PhysicalVersionPlan {
  PhysicalVersionId id;
  MemLayout encoding = MemLayout::Tensor;
};

struct RepresentationPlan {
  std::vector<PhysicalVersionPlan> primaryVersions;
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
