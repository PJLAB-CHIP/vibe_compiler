//===- TemporalDomain.h - Live-operation temporal choices -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALDOMAIN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALDOMAIN_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

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

mlir::FailureOr<llvm::SmallVector<uint32_t, 4>> buildFirstTemporalLoopOrder(
    llvm::ArrayRef<int64_t> iteratorExtents,
    llvm::ArrayRef<int64_t> iteratorTileSizes,
    llvm::ArrayRef<TemporalPrecedenceEdge> precedence = {},
    std::string *failureReason = nullptr);

/// A synchronous descriptor derived from one live current operation. The
/// operation handle is valid only while the owning TileRegion is unchanged.
struct TemporalScopeDescriptor {
  mlir::Operation *operation = nullptr;
  llvm::SmallVector<int64_t, 4> iterationExtents;
  llvm::SmallVector<IteratorTilingCapability, 4> iteratorCapabilities;
  llvm::SmallVector<TemporalPrecedenceEdge, 4> precedence;

  friend bool operator==(const TemporalScopeDescriptor &lhs,
                         const TemporalScopeDescriptor &rhs) {
    return lhs.operation == rhs.operation &&
           lhs.iterationExtents == rhs.iterationExtents &&
           lhs.iteratorCapabilities == rhs.iteratorCapabilities &&
           lhs.precedence == rhs.precedence;
  }
};

/// Explicit parameters for one live traversal. This object is consumed by the
/// immediate apply call and is not a cross-stage operation identity.
struct TemporalScopeChoice {
  mlir::Operation *operation = nullptr;
  llvm::SmallVector<int64_t, 4> iteratorTileSizes;
  llvm::SmallVector<uint32_t, 4> loopOrder;

  friend bool operator==(const TemporalScopeChoice &lhs,
                         const TemporalScopeChoice &rhs) {
    return lhs.operation == rhs.operation &&
           lhs.iteratorTileSizes == rhs.iteratorTileSizes &&
           lhs.loopOrder == rhs.loopOrder;
  }
};

struct TemporalChoice {
  std::vector<TemporalScopeChoice> scopes;

  friend bool operator==(const TemporalChoice &lhs, const TemporalChoice &rhs) {
    return lhs.scopes == rhs.scopes;
  }
};

enum class TemporalFusionQueryKind : uint8_t {
  ExactDerived,
  NonUnique,
  Unsupported,
  Indeterminate,
  BrokenContract,
};

struct TemporalFusionQueryResult {
  TemporalFusionQueryKind kind = TemporalFusionQueryKind::BrokenContract;
  std::string detail;
};

/// Determines whether one current producer result is uniquely determined by
/// its only current consumer traversal for every legal tile of that traversal.
TemporalFusionQueryResult
queryTemporalProducerFusion(mlir::OpResult producer,
                            mlir::OpOperand &consumerOperand);

enum class TemporalDomainFailureKind : uint8_t {
  BrokenContract,
};

struct TemporalDomainFailure {
  TemporalDomainFailureKind kind = TemporalDomainFailureKind::BrokenContract;
  std::string detail;
};

enum class TemporalSuccessorKind : uint8_t {
  Choice,
  End,
  CompilerBug,
};

struct TemporalDomainResult;

class TemporalCursor {
private:
  TemporalChoice choice;

  friend class TemporalDomain;
};

class TemporalSuccessor {
public:
  TemporalSuccessorKind getKind() const { return kind; }
  const TemporalChoice *getChoice() const {
    return choice ? &*choice : nullptr;
  }
  const TemporalCursor *getCursor() const {
    return cursor ? &*cursor : nullptr;
  }
  llvm::StringRef getDetail() const { return detail; }

private:
  TemporalSuccessor(TemporalSuccessorKind kind,
                    std::optional<TemporalChoice> choice = {},
                    std::optional<TemporalCursor> cursor = {},
                    std::string detail = {})
      : kind(kind), choice(std::move(choice)), cursor(std::move(cursor)),
        detail(std::move(detail)) {}

  TemporalSuccessorKind kind;
  std::optional<TemporalChoice> choice;
  std::optional<TemporalCursor> cursor;
  std::string detail;

  friend class TemporalDomain;
};

/// Complete lazy domain for traversal roots in one unchanged TileRegion.
/// Descriptors and cursors borrow live operation handles and must be discarded
/// before applying a choice or otherwise mutating the region.
class TemporalDomain {
public:
  TemporalSuccessor getFirstChoice() const;
  TemporalSuccessor getNextChoice(const TemporalCursor &cursor) const;
  TemporalSuccessor completePrefix(const TemporalChoice &prefix) const;
  bool contains(const TemporalChoice &choice) const;

  TileRegionOp getRegion() const { return region; }
  llvm::ArrayRef<TemporalScopeDescriptor> getScopeDescriptors() const {
    return scopes;
  }

private:
  struct Completion;

  TemporalDomain(TileRegionOp region,
                 std::vector<TemporalScopeDescriptor> scopes)
      : region(region), scopes(std::move(scopes)) {}

  static TemporalScopeChoice
  getFirstScopeChoice(const TemporalScopeDescriptor &scope);
  static bool advanceScopeChoice(const TemporalScopeDescriptor &scope,
                                 TemporalScopeChoice &choice);
  Completion completeChoice(llvm::ArrayRef<TemporalScopeChoice> prefix) const;

  TileRegionOp region;
  std::vector<TemporalScopeDescriptor> scopes;

  friend struct TemporalDomainResult;
  friend TemporalDomainResult buildTemporalDomain(TileRegionOp);
};

struct TemporalDomainResult {
  std::optional<TemporalDomain> domain;
  std::optional<TemporalDomainFailure> failure;

  bool succeeded() const { return domain.has_value(); }
};

/// Derives traversal roots, exact-derived producer edges, iterator extents and
/// capabilities directly from one verifier-valid structural TileRegion.
TemporalDomainResult buildTemporalDomain(TileRegionOp region);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALDOMAIN_H
