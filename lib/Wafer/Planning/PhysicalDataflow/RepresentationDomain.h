//===- RepresentationDomain.h - Physical version layout domain -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_REPRESENTATIONDOMAIN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_REPRESENTATIONDOMAIN_H

#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"
#include "Wafer/Planning/PhysicalDataflow/RepresentationPBQPSolver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

enum class RepresentationDomainFailureKind : uint8_t {
  UnsupportedSemantics,
  BrokenContract,
};

struct RepresentationDomainFailure {
  RepresentationDomainFailureKind kind =
      RepresentationDomainFailureKind::BrokenContract;
  std::optional<RegionValueVersionId> value;
  std::string detail;
};

enum class RepresentationSuccessorKind : uint8_t {
  Plan,
  End,
  CompilerBug,
};

class RepresentationCursor {
private:
  std::vector<uint32_t> primaryLayoutIndices;
  std::vector<uint32_t> useOptionIndices;

  friend class RepresentationDomain;
};

class RepresentationSuccessor {
public:
  RepresentationSuccessorKind getKind() const { return kind; }
  const RepresentationPlan *getPlan() const { return plan ? &*plan : nullptr; }
  const RepresentationCursor *getCursor() const {
    return cursor ? &*cursor : nullptr;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  RepresentationSuccessor(RepresentationSuccessorKind kind,
                          std::optional<RepresentationPlan> plan = {},
                          std::optional<RepresentationCursor> cursor = {},
                          std::string detail = {})
      : kind(kind), plan(std::move(plan)), cursor(std::move(cursor)),
        detail(std::move(detail)) {}

  RepresentationSuccessorKind kind;
  std::optional<RepresentationPlan> plan;
  std::optional<RepresentationCursor> cursor;
  std::string detail;

  friend class RepresentationDomain;
};

struct RepresentationDomainResult;

struct IdentityAliasRequirement {
  RepresentationUseId use;
  RegionValueVersionId source;
};

struct RepresentationLayoutTupleConstraint {
  std::vector<RegionValueVersionId> values;
  std::vector<std::vector<MemLayout>> legalTuples;
};

struct RepresentationProposalResult {
  RepresentationPBQPStatus status = RepresentationPBQPStatus::BrokenContract;
  std::optional<RepresentationPlan> plan;
  std::optional<RepresentationPBQPCost> cost;
  uint64_t work = 0;
};

/// Complete lazy domain of legal primary layouts and explicit use bindings.
/// A use may bind its primary or one typed conversion at a shared/value or
/// per-use anchor. Domain membership is determined only by exact logical
/// resources and the current physical encoding verifier.
class RepresentationDomain {
public:
  RepresentationSuccessor getFirstPlan() const;
  RepresentationSuccessor getNextPlan(const RepresentationCursor &cursor) const;
  bool contains(const RepresentationPlan &plan) const;
  RepresentationProposalResult getPBQPProposal(uint64_t workLimit) const;

  const RepresentationResourceDescription *
  findResource(const RegionValueVersionId &value) const;

private:
  struct ValueDomain {
    RegionValueVersionId value;
    PhysicalVersionId primary;
    RepresentationResourceDescription resource;
    std::vector<MemLayout> layouts;
  };

  struct UseDomain {
    RepresentationUseId use;
    size_t valueIndex = 0;
    std::vector<MemLayout> layouts;
    std::optional<size_t> aliasSourceIndex;
  };

  struct UseOption {
    PhysicalVersionId version;
    MemLayout encoding = MemLayout::Tensor;
  };

  struct TupleConstraint {
    std::vector<size_t> valueIndices;
    std::vector<std::vector<MemLayout>> legalTuples;
  };

  RepresentationDomain(std::vector<ValueDomain> values,
                       std::vector<UseDomain> uses,
                       std::vector<TupleConstraint> constraints)
      : values(std::move(values)), uses(std::move(uses)),
        constraints(std::move(constraints)) {}

  std::vector<UseOption>
  getUseOptions(size_t index,
                llvm::ArrayRef<uint32_t> primaryLayoutIndices) const;
  std::optional<RepresentationPlan>
  buildPlan(const RepresentationCursor &cursor) const;
  std::optional<RepresentationCursor>
  getCursor(const RepresentationPlan &plan) const;
  bool isPrimaryAssignmentLegal(const RepresentationCursor &cursor) const;
  bool advanceCursor(RepresentationCursor &cursor) const;

  std::vector<ValueDomain> values;
  std::vector<UseDomain> uses;
  std::vector<TupleConstraint> constraints;

  friend RepresentationDomainResult buildRepresentationDomain(
      const CanonicalRepresentationCoordinate &,
      llvm::ArrayRef<IdentityAliasRequirement>,
      llvm::ArrayRef<RepresentationLayoutTupleConstraint>);
};

struct RepresentationDomainResult {
  std::optional<RepresentationDomain> domain;
  std::optional<RepresentationDomainFailure> failure;

  bool succeeded() const { return domain.has_value(); }
};

RepresentationDomainResult buildRepresentationDomain(
    const CanonicalRepresentationCoordinate &canonical,
    llvm::ArrayRef<IdentityAliasRequirement> aliases = {},
    llvm::ArrayRef<RepresentationLayoutTupleConstraint> constraints = {});

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_REPRESENTATIONDOMAIN_H
