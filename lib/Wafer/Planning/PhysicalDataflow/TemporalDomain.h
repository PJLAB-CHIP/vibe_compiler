//===- TemporalDomain.h - Live-operation temporal choices -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALDOMAIN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALDOMAIN_H

#include "Wafer/Analysis/Linalg/IndexRelation.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
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

enum class TemporalScopeRole : uint8_t {
  Traversal,
  FusedReduction,
};

/// A synchronous descriptor derived from one live current operation. The
/// operation handle is valid only while the owning TileRegion is unchanged.
struct TemporalScopeDescriptor {
  mlir::Operation *operation = nullptr;
  llvm::SmallVector<int64_t, 4> iterationExtents;
  llvm::SmallVector<IteratorTilingCapability, 4> iteratorCapabilities;
  llvm::SmallVector<TemporalPrecedenceEdge, 4> precedence;
  llvm::SmallVector<uint32_t, 2> exactReshapeDimensions;
  TemporalScopeRole role = TemporalScopeRole::Traversal;

  friend bool operator==(const TemporalScopeDescriptor &lhs,
                         const TemporalScopeDescriptor &rhs) {
    return lhs.operation == rhs.operation &&
           lhs.iterationExtents == rhs.iterationExtents &&
           lhs.iteratorCapabilities == rhs.iteratorCapabilities &&
           lhs.precedence == rhs.precedence &&
           lhs.exactReshapeDimensions == rhs.exactReshapeDimensions &&
           lhs.role == rhs.role;
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

enum class TemporalTraversalKind : uint8_t {
  Joint,
  Independent,
};

struct TemporalChoice {
  TemporalTraversalKind kind = TemporalTraversalKind::Joint;
  std::vector<TemporalScopeChoice> scopes;

  friend bool operator==(const TemporalChoice &lhs, const TemporalChoice &rhs) {
    return lhs.kind == rhs.kind && lhs.scopes == rhs.scopes;
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

struct TemporalViewDimensionMapping {
  int32_t viewDimension = -1;
  int64_t offset = 0;
};

/// One current exact producer-to-consumer path. `viewTransparent` distinguishes
/// a direct SSA edge from a path through one or more current tensor support
/// operations. All handles are borrowed from one unchanged TileRegion.
struct TemporalFusionPathResult {
  TemporalFusionQueryKind kind = TemporalFusionQueryKind::BrokenContract;
  mlir::OpResult producer;
  mlir::OpOperand *consumerOperand = nullptr;
  bool viewTransparent = false;
  llvm::SmallVector<TemporalViewDimensionMapping, 4> producerDimensions;
  std::optional<analysis::IndexRelation> consumerViewToProducer;
  llvm::SmallVector<uint32_t, 2> forcedFullExtentConsumerDimensions;
  llvm::SmallVector<uint32_t, 2> generalReshapeConsumerDimensions;
  std::string detail;

  bool isExact() const {
    return kind == TemporalFusionQueryKind::ExactDerived && producer &&
           consumerOperand;
  }
};

/// One all-use producer group that may be materialized once in a common joint
/// traversal. `consumerValue` is either the producer itself or the final value
/// of one exact transparent view chain. All IR handles are borrowed from the
/// same unchanged TileRegion.
struct TemporalJointProducerGroup {
  mlir::OpResult producer;
  mlir::Value consumerValue;
  llvm::SmallVector<mlir::OpOperand *, 4> consumerOperands;
  llvm::SmallVector<TemporalViewDimensionMapping, 4> producerDimensions;
  std::optional<analysis::IndexRelation> consumerViewToProducer;
  llvm::SmallVector<uint32_t, 2> generalReshapeConsumerDimensions;

  bool isViewTransparent() const { return consumerValue != producer; }
};

/// One current operand demand, possibly through a transparent view chain.
/// Window access and invariance are properties of the same relation. Its
/// rectangular representation and disjointness are queried for each choice.
struct TemporalOperandFusion {
  mlir::OpResult producer;
  mlir::Value consumerValue;
  mlir::OpOperand *consumerOperand;
  analysis::IndexRelation iterationToProducer;
  llvm::SmallVector<int64_t, 4> iterationShape;
};

analysis::RectangularTileImageResult
queryTemporalOperandTile(const TemporalOperandFusion &fusion,
                         const TemporalScopeChoice &choice);

enum class TemporalConcatQueryKind : uint8_t {
  Exact,
  NotConcat,
  NonUnique,
  Unsupported,
  ResourceExhausted,
  BrokenContract,
};

struct TemporalConcatSegment {
  mlir::Value source;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  std::optional<mlir::OpResult> derivedProducer;
};

/// One current all-and-only static insert_slice assembly consumed by a single
/// temporal traversal. The query result borrows all handles from the current
/// unchanged TileRegion and is consumed by the immediate apply call.
struct TemporalConcatQueryResult {
  TemporalConcatQueryKind kind = TemporalConcatQueryKind::NotConcat;
  mlir::Value assembledValue;
  mlir::OpOperand *consumerOperand = nullptr;
  llvm::SmallVector<TemporalConcatSegment, 4> segments;
  std::string detail;

  bool isExact() const {
    return kind == TemporalConcatQueryKind::Exact && assembledValue &&
           consumerOperand && !segments.empty();
  }
};

/// Determines whether one current producer result is uniquely determined by
/// its only current consumer traversal for every legal tile of that traversal.
TemporalFusionQueryResult
queryTemporalProducerFusion(mlir::OpResult producer,
                            mlir::OpOperand &consumerOperand);

/// Follows a pure all-result-single-use static tensor support chain and
/// returns the final current consumer when the complete edge is an exact
/// derived temporal producer relation.
TemporalFusionPathResult
queryTemporalProducerFusionPath(mlir::OpResult producer);

/// Recognizes a static, unit-stride, nonoverlapping insert_slice chain rooted
/// in tensor.empty whose source rectangles exactly cover the assembled value.
TemporalConcatQueryResult
queryTemporalConcatAssembly(mlir::OpOperand &consumerOperand);

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
  TemporalSuccessor getFirstIndependentChoice() const;
  TemporalSuccessor getNextChoice(const TemporalCursor &cursor) const;
  TemporalSuccessor completePrefix(const TemporalChoice &prefix) const;
  bool contains(const TemporalChoice &choice) const;

  TileRegionOp getRegion() const { return region; }
  llvm::ArrayRef<TemporalScopeDescriptor> getScopeDescriptors(
      TemporalTraversalKind kind = TemporalTraversalKind::Joint) const {
    return kind == TemporalTraversalKind::Joint ? jointScopes
                                                : independentScopes;
  }
  llvm::ArrayRef<TemporalJointProducerGroup> getJointProducerGroups() const {
    return jointProducerGroups;
  }
  llvm::ArrayRef<TemporalOperandFusion> getOperandFusions() const {
    return operandFusions;
  }

private:
  struct Completion;

  TemporalDomain(TileRegionOp region,
                 std::vector<TemporalScopeDescriptor> jointScopes,
                 std::vector<TemporalScopeDescriptor> independentScopes,
                 std::vector<TemporalJointProducerGroup> jointProducerGroups,
                 std::vector<TemporalOperandFusion> operandFusions)
      : region(region), jointScopes(std::move(jointScopes)),
        independentScopes(std::move(independentScopes)),
        jointProducerGroups(std::move(jointProducerGroups)),
        operandFusions(std::move(operandFusions)) {}

  static TemporalScopeChoice
  getFirstScopeChoice(const TemporalScopeDescriptor &scope);
  static bool advanceScopeChoice(const TemporalScopeDescriptor &scope,
                                 TemporalScopeChoice &choice);
  Completion completeChoice(TemporalTraversalKind kind,
                            llvm::ArrayRef<TemporalScopeChoice> prefix) const;
  bool isJointChoiceCompatible(const TemporalChoice &choice) const;

  TileRegionOp region;
  std::vector<TemporalScopeDescriptor> jointScopes;
  std::vector<TemporalScopeDescriptor> independentScopes;
  std::vector<TemporalJointProducerGroup> jointProducerGroups;
  std::vector<TemporalOperandFusion> operandFusions;

  friend struct TemporalDomainResult;
  friend TemporalDomainResult buildTemporalDomain(TileRegionOp);
  friend mlir::FailureOr<TemporalDomain>
  remapTemporalDomain(const TemporalDomain &, TileRegionOp,
                      const mlir::IRMapping &, std::string *);
};

struct TemporalDomainResult {
  std::optional<TemporalDomain> domain;
  std::optional<TemporalDomainFailure> failure;

  bool succeeded() const { return domain.has_value(); }
};

/// Derives traversal roots, exact-derived producer edges, iterator extents and
/// capabilities directly from one verifier-valid structural TileRegion.
TemporalDomainResult buildTemporalDomain(TileRegionOp region);

/// Read-only iteration-to-result map of a current structured tensor result.
/// Dialect adapters expose semantics; transformation uses the same map
/// contract.

/// Remaps one immutable structural TemporalDomain onto a fresh clone of the
/// same current TileRegion. This copies only query metadata and typed handles;
/// it does not create IR or survive the candidate transaction. The caller must
/// immediately apply the remapped choice before mutating the clone.
mlir::FailureOr<TemporalDomain>
remapTemporalDomain(const TemporalDomain &source, TileRegionOp mappedRegion,
                    const mlir::IRMapping &mapping,
                    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_TEMPORALDOMAIN_H
