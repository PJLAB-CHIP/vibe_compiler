//===- SpatialDomain.h - Complete spatial plan domain --------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SPATIALDOMAIN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SPATIALDOMAIN_H

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"
#include "Wafer/IR/Target/TargetTopology.h"
#include "Wafer/Planning/PhysicalDataflow/AttentionSpatialConstraints.h"
#include "Wafer/Planning/PhysicalDataflow/SemanticRootAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

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

enum class SpatialPlanSuccessorKind : uint8_t { Successor, End, Failure };

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

struct SpatialPlanDomainResult;

/// Complete lazy Cartesian domain of compact SpatialPlan values. Partition
/// schemes, injective embeddings and per-group merge placements are advanced
/// without materializing a point vector. Proposals are ordinary members of
/// this same domain and never remove raw successors.
class SpatialPlanDomain {
public:
  SpatialPlan getFirstPlan() const;
  SpatialPlanSuccessor getNextPlan(const SpatialPlan &plan) const;
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

  llvm::SmallVector<StructuredDAGNodePlacement, 16>
  getNodePlacements(const SpatialPlan &plan) const;

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

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SPATIALDOMAIN_H
