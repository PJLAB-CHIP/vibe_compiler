//===- SpatialDomain.h - Complete spatial plan domain --------*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_SPATIALDOMAIN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_SPATIALDOMAIN_H

#include "Wafer/Analysis/Linalg/SemanticRootAnalysis.h"
#include "Wafer/Analysis/Linalg/StructuredDAGAnalysis.h"
#include "Wafer/IR/Topology/TargetTopology.h"
#include "Wafer/Planning/PhysicalDataflow/AttentionSpatialConstraints.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

enum class SpatialIteratorKind : uint8_t { Parallel, Reduction };

/// Immutable current-IR facts needed by both the canonical constructor and
/// the complete spatial domain. They are derived for one unchanged
/// StructuredDAGAnalysis epoch and contain no operation identity.
struct SpatialRootDomainFacts {
  StructuredDAGNodeID node = 0;
  SemanticRootKey root;
  llvm::SmallVector<int64_t, 8> iteratorExtents;
  llvm::SmallVector<SpatialIteratorKind, 8> iteratorKinds;
  llvm::SmallBitVector partitionableParallelIterators;
  llvm::SmallVector<llvm::SmallBitVector, 4> resultParallelIteratorsByGroup;
  llvm::SmallBitVector partitionableReductionIterators;
  uint32_t reductionResultGroupCount = 0;
  std::optional<AttentionSpatialConstraints> attention;
};

enum class SpatialDomainFailureKind : uint8_t {
  UnsupportedSemantics,
  BrokenContract,
};

struct SpatialDomainFailure {
  SpatialDomainFailureKind kind = SpatialDomainFailureKind::BrokenContract;
  std::optional<SemanticRootKey> root;
  std::string detail;
};

struct SpatialDomainProblemResult;

/// One immutable problem shared by every successor/evaluation in a search
/// session. The structural problem is the sole authority for roots, iterator
/// extents and available Tiles; root facts add only typed source semantics.
class SpatialDomainProblem {
public:
  const SemanticRootAnalysis &getSemanticRoots() const { return semanticRoots; }
  const SpatialPlanningProblem &getStructuralProblem() const {
    return structuralProblem;
  }
  llvm::ArrayRef<SpatialRootDomainFacts> getRoots() const { return roots; }
  llvm::ArrayRef<llvm::SmallVector<uint32_t, 4>> getComponents() const {
    return components;
  }
  const SpatialRootDomainFacts *findRoot(const SemanticRootKey &root) const;

private:
  SpatialDomainProblem(
      SemanticRootAnalysis semanticRoots,
      SpatialPlanningProblem structuralProblem,
      llvm::SmallVector<SpatialRootDomainFacts, 16> roots,
      llvm::SmallVector<llvm::SmallVector<uint32_t, 4>, 4> components)
      : semanticRoots(std::move(semanticRoots)),
        structuralProblem(std::move(structuralProblem)),
        roots(std::move(roots)), components(std::move(components)) {}

  SemanticRootAnalysis semanticRoots;
  SpatialPlanningProblem structuralProblem;
  llvm::SmallVector<SpatialRootDomainFacts, 16> roots;
  llvm::SmallVector<llvm::SmallVector<uint32_t, 4>, 4> components;

  friend struct SpatialDomainProblemResult;
  friend SpatialDomainProblemResult
  buildSpatialDomainProblem(const StructuredDAGAnalysis &,
                            llvm::ArrayRef<TileId>);
};

struct SpatialDomainProblemResult {
  std::optional<SpatialDomainProblem> problem;
  std::optional<SpatialDomainFailure> failure;

  bool succeeded() const { return problem.has_value(); }
};

SpatialDomainProblemResult
buildSpatialDomainProblem(const StructuredDAGAnalysis &dag,
                          llvm::ArrayRef<TileId> availableTiles);

/// Reassigns only the existing logical shards to available Tiles so exact
/// producer-owner intersections and their destination shards are co-located
/// whenever one injective embedding can do so. This is a deterministic
/// proposal transformation: it does not change axes, merge choices, the raw
/// spatial domain, or any legality result.
mlir::FailureOr<SpatialPlan>
buildGraphCoherentSpatialProposal(const SpatialDomainProblem &problem,
                                  const SpatialPlan &plan,
                                  const SpatialAssignment &assignment,
                                  const analysis::ExactDemandProof &demand,
                                  std::string *failureReason = nullptr);

enum class SpatialPlanSuccessorKind : uint8_t { Successor, End, Failure };

enum class SpatialSuccessorDomain : uint8_t {
  AllPlacements,
  /// One canonical embedding/merge witness per axis tuple. Interleaved with
  /// AllPlacements by the driver; does not remove any raw-domain point.
  AxisSchemes,
};

struct SpatialPlanSuccessor {
  SpatialPlanSuccessorKind kind = SpatialPlanSuccessorKind::Failure;
  std::optional<SpatialPlan> plan;
  std::optional<SpatialDomainFailure> failure;
};

struct SpatialDomainEvaluation {
  std::optional<SpatialAssignment> assignment;
  std::optional<analysis::ExactDemandOutcome> demand;
  std::optional<SpatialDomainFailure> failure;

  bool isSatisfied() const {
    return assignment && demand &&
           analysis::classifyExactDemandOutcome(*demand) ==
               analysis::ExactDemandOutcomeCategory::Satisfied;
  }
};

/// Numeric visitation cursors on one unchanged source domain. No allocation,
/// placement effect, capacity or downstream IR is represented here.
class SpatialDirectionCursor {
private:
  struct RootCursor {
    size_t root;
    llvm::SmallVector<size_t, 8> eligible;
    llvm::SmallVector<size_t, 4> support;
    size_t supportSize = 1;
    size_t ratio = 0;
    size_t maximumRatios = 0;
    bool started = false;
    bool exhausted = false;
  };
  std::vector<RootCursor> roots;
  size_t nextRoot = 0;
  bool initialized = false;
  friend class SpatialPlanDomain;
};

struct SpatialPlanDomainResult;

/// Complete lazy Cartesian domain of compact SpatialPlan values. Partition
/// schemes, injective embeddings and per-group merge placements are advanced
/// without materializing a point vector. Proposals are ordinary members of
/// this same domain and never remove raw successors.
class SpatialPlanDomain {
public:
  SpatialPlan getFirstPlan() const;
  SpatialPlanSuccessor
  getNextPlan(const SpatialPlan &plan,
              SpatialSuccessorDomain successorDomain =
                  SpatialSuccessorDomain::AllPlacements) const;
  /// One root direction at a time, with roots round-robin by source work.
  /// Visits all axis supports before further ratios within the same support.
  SpatialPlanSuccessor
  getNextDirectionPlan(const SpatialPlan &anchor,
                       SpatialDirectionCursor &cursor) const;
  bool contains(const SpatialPlan &plan) const;

  SpatialDomainEvaluation evaluate(const StructuredDAGAnalysis &dag,
                                   const SpatialPlan &plan,
                                   const analysis::IndexRelationLimits &limits =
                                       analysis::IndexRelationLimits()) const;
  mlir::FailureOr<SpatialAssignment>
  close(const SpatialPlan &plan, std::string *failureReason = nullptr) const;

  /// Deterministic topology-shaped seeds. Every returned plan is validated by
  /// `contains`; this list is visitation priority, not a legality shortlist.
  llvm::SmallVector<SpatialPlan, 4> getProposals() const;

  /// Places relation-coordinated variants before their raw topology-shaped
  /// seeds. Both forms remain ordinary domain members; demand evaluation and
  /// the complete raw successor traversal are unchanged.
  mlir::FailureOr<llvm::SmallVector<SpatialPlan, 8>>
  getGraphCoherentProposals(const StructuredDAGAnalysis &dag,
                            const analysis::IndexRelationLimits &limits =
                                analysis::IndexRelationLimits()) const;

  const SpatialDomainProblem &getProblem() const { return problem; }
  const TargetTopology &getTopology() const { return topology; }
  CardId getCardId() const { return cardId; }

private:
  SpatialPlanDomain(SpatialDomainProblem problem, TargetTopology topology,
                    CardId cardId)
      : problem(std::move(problem)), topology(std::move(topology)),
        cardId(cardId) {}

  SpatialDomainProblem problem;
  TargetTopology topology;
  CardId cardId{0};

  friend struct SpatialPlanDomainResult;
  friend SpatialPlanDomainResult
  buildSpatialPlanDomain(const StructuredDAGAnalysis &, const TargetTopology &,
                         CardId);
};

struct SpatialPlanDomainResult {
  std::optional<SpatialPlanDomain> domain;
  std::optional<SpatialDomainFailure> failure;

  bool succeeded() const { return domain.has_value(); }
};

SpatialPlanDomainResult buildSpatialPlanDomain(const StructuredDAGAnalysis &dag,
                                               const TargetTopology &topology,
                                               CardId cardId);

/// Source-semantic reduction groups for one closed partition combination.
/// The result is sorted, complete and independent of physical embedding.
mlir::FailureOr<llvm::SmallVector<ReductionGroupId, 8>>
deriveSpatialReductionGroups(const SpatialRootDomainFacts &root,
                             llvm::ArrayRef<IteratorPartition> partitions,
                             std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_SPATIALDOMAIN_H
