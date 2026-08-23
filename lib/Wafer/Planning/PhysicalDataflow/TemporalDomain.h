//===- TemporalDomain.h - Complete per-scope temporal domain -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_TEMPORALDOMAIN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_TEMPORALDOMAIN_H

#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

enum class IteratorTilingCapability : uint8_t {
  Tileable,
  FullExtentOnly,
};

struct TemporalPrecedenceEdge {
  uint32_t before = 0;
  uint32_t after = 0;

  friend bool operator==(const TemporalPrecedenceEdge &lhs,
                         const TemporalPrecedenceEdge &rhs) {
    return lhs.before == rhs.before && lhs.after == rhs.after;
  }
  friend bool operator<(const TemporalPrecedenceEdge &lhs,
                        const TemporalPrecedenceEdge &rhs) {
    return std::tie(lhs.before, lhs.after) < std::tie(rhs.before, rhs.after);
  }
};

struct TemporalSizeInterval {
  int64_t lower = 1;
  int64_t upper = 1;

  friend bool operator==(const TemporalSizeInterval &lhs,
                         const TemporalSizeInterval &rhs) {
    return lhs.lower == rhs.lower && lhs.upper == rhs.upper;
  }
};

/// Disjoint exact partition of one unresolved integer interval. The singleton
/// is visited first; `above` and `below` remain ordinary reachable siblings.
struct TemporalIntervalChildren {
  TemporalSizeInterval singleton;
  std::optional<TemporalSizeInterval> above;
  std::optional<TemporalSizeInterval> below;
};

mlir::FailureOr<TemporalIntervalChildren>
splitTemporalSizeInterval(TemporalSizeInterval interval,
                          std::optional<int64_t> proposal = std::nullopt,
                          std::string *failureReason = nullptr);

/// Returns the stable first linear extension for the dimensions made active
/// by one concrete size vector.
mlir::FailureOr<llvm::SmallVector<uint32_t, 4>> buildFirstTemporalWaveLoopOrder(
    llvm::ArrayRef<int64_t> iteratorExtents,
    llvm::ArrayRef<int64_t> iteratorTileSizes,
    llvm::ArrayRef<TemporalPrecedenceEdge> precedence = {},
    std::string *failureReason = nullptr);

/// Immutable descriptor of one traversal variable. Offsets, extents,
/// capability and precedence are derived facts and do not enter TemporalPlan.
struct TemporalScopeDescriptor {
  TraversalScopeId id;
  llvm::SmallVector<int64_t, 4> iterationOffsets;
  llvm::SmallVector<int64_t, 4> iterationExtents;
  llvm::SmallVector<IteratorTilingCapability, 4> iteratorCapabilities;
  llvm::SmallVector<TemporalPrecedenceEdge, 4> precedence;
  std::optional<TraversalScopeId> parentScope;

  friend bool operator==(const TemporalScopeDescriptor &lhs,
                         const TemporalScopeDescriptor &rhs) {
    return lhs.id == rhs.id && lhs.iterationOffsets == rhs.iterationOffsets &&
           lhs.iterationExtents == rhs.iterationExtents &&
           lhs.iteratorCapabilities == rhs.iteratorCapabilities &&
           lhs.precedence == rhs.precedence &&
           lhs.parentScope == rhs.parentScope;
  }
};

enum class TemporalDomainFailureKind : uint8_t {
  UnsupportedSemantics,
  Indeterminate,
  BrokenContract,
};

struct TemporalDomainFailure {
  TemporalDomainFailureKind kind = TemporalDomainFailureKind::BrokenContract;
  std::optional<TraversalScopeId> scope;
  std::string detail;
};

enum class TemporalSuccessorKind : uint8_t {
  Plan,
  End,
  Unsupported,
  Indeterminate,
  CompilerBug,
};

struct TemporalDomainResult;

class TemporalCursor {
private:
  TemporalPlan plan;

  friend class TemporalDomain;
};

class TemporalSuccessor {
public:
  TemporalSuccessorKind getKind() const { return kind; }
  const TemporalPlan *getPlan() const { return plan ? &*plan : nullptr; }
  const TemporalCursor *getCursor() const {
    return cursor ? &*cursor : nullptr;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  TemporalSuccessor(TemporalSuccessorKind kind,
                    std::optional<TemporalPlan> plan = {},
                    std::optional<TemporalCursor> cursor = {},
                    std::string detail = {})
      : kind(kind), plan(std::move(plan)), cursor(std::move(cursor)),
        detail(std::move(detail)) {}

  TemporalSuccessorKind kind;
  std::optional<TemporalPlan> plan;
  std::optional<TemporalCursor> cursor;
  std::string detail;

  friend class TemporalDomain;
};

/// Complete lazy Cartesian domain of per-scope positive sizes and every
/// active-iterator linear extension. No point vector, wave list, IR, target
/// preference or resource estimate is retained by the domain.
class TemporalDomain {
public:
  TemporalSuccessor getFirstPlan() const;
  TemporalSuccessor getNextPlan(const TemporalCursor &cursor) const;
  /// Closes a caller-selected valid prefix with the first choices for all
  /// remaining and newly-derived child scopes. This is the search-controller
  /// transition for an interval/order branch; it does not rank, materialize
  /// or mutate IR.
  TemporalSuccessor completePrefix(const TemporalPlan &prefix) const;
  bool contains(const TemporalPlan &plan) const;

private:
  struct NestedTemporalFacts;
  struct Completion;

  TemporalDomain(std::vector<TemporalScopeDescriptor> scopes,
                 std::shared_ptr<const NestedTemporalFacts> nestedFacts = {})
      : scopes(std::move(scopes)), nestedFacts(std::move(nestedFacts)) {}

  static TemporalScopePlan
  getFirstScopePlan(const TemporalScopeDescriptor &scope);
  static bool advanceScopePlan(const TemporalScopeDescriptor &scope,
                               TemporalScopePlan &plan);
  Completion completePlan(llvm::ArrayRef<TemporalScopePlan> prefix) const;

  std::vector<TemporalScopeDescriptor> scopes;
  std::shared_ptr<const NestedTemporalFacts> nestedFacts;

  friend struct TemporalDomainResult;
  friend TemporalDomainResult
      buildTemporalDomain(llvm::ArrayRef<TemporalScopeDescriptor>);
  friend TemporalDomainResult
  buildTemporalDomain(const RegionPlan &,
                      llvm::ArrayRef<analysis::RootRegionWork>);
};

struct TemporalDomainResult {
  std::optional<TemporalDomain> domain;
  std::optional<TemporalDomainFailure> failure;

  bool succeeded() const { return domain.has_value(); }
};

/// Validates already-derived scopes. This is the policy-free entry used by
/// tests and by parent-dependent nested-scope derivation.
TemporalDomainResult
buildTemporalDomain(llvm::ArrayRef<TemporalScopeDescriptor> scopes);

/// Derives top-level required/replica scopes and parent-dependent nested
/// classes from the selected region plan and root work. A reconstruction that
/// has no composed exact relation is typed unsupported; it is never replaced
/// by full-producer or bounding-box work.
TemporalDomainResult
buildTemporalDomain(const RegionPlan &regions,
                    llvm::ArrayRef<analysis::RootRegionWork> rootWorks);

/// Exact one-dimensional wave partition used by both domain witnesses and the
/// eventual loop builder. The result is ordered, disjoint and preserves a
/// nonzero source offset.
mlir::FailureOr<llvm::SmallVector<IteratorInterval, 8>>
buildTemporalAxisWaves(IteratorInterval interval, int64_t tileSize,
                       std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_TEMPORALDOMAIN_H
