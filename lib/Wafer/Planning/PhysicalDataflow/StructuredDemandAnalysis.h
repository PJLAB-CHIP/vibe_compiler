//===- StructuredDemandAnalysis.h - Exact structured demand ----*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STRUCTUREDDEMANDANALYSIS_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STRUCTUREDDEMANDANALYSIS_H

#include "Wafer/Planning/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Analysis/Linalg/StructuredDAGAnalysis.h"
#include "Wafer/Analysis/Linalg/SemanticRootAnalysis.h"

#include "mlir/Support/LogicalResult.h"

#include <memory>
#include <string>

namespace wafer::compiler::detail {

/// Immutable function-local SSA/indexing facts. The object is valid only while
/// the source function remains unchanged and contains no assignment, target,
/// candidate, cost, or materialized-IR state.
class StructuredRelationFacts {
public:
  static mlir::FailureOr<StructuredRelationFacts>
  create(const StructuredDAGAnalysis &dag,
         const analysis::IndexRelationLimits &limits =
             analysis::IndexRelationLimits(),
         std::string *failureReason = nullptr);

  ~StructuredRelationFacts();
  StructuredRelationFacts(StructuredRelationFacts &&) noexcept;
  StructuredRelationFacts &operator=(StructuredRelationFacts &&) noexcept;
  StructuredRelationFacts(const StructuredRelationFacts &) = delete;
  StructuredRelationFacts &operator=(const StructuredRelationFacts &) = delete;

  const StructuredDAGAnalysis &getDAG() const;
  const SemanticRootAnalysis &getSemanticRoots() const;

private:
  class Impl;
  explicit StructuredRelationFacts(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;

  friend analysis::ExactDemandOutcome
  deriveExactDemand(const StructuredRelationFacts &, const SpatialAssignment &,
                    const analysis::IndexRelationLimits &);
};

/// Thin MLIR analysis wrapper over the same policy-free facts builder. It is
/// anchored at one func.func and relies exclusively on AnalysisManager
/// preservation/invalidation.
class StructuredRelationAnalysis {
public:
  explicit StructuredRelationAnalysis(mlir::Operation *operation);

  bool isValid() const { return facts.has_value(); }
  const StructuredRelationFacts *getFacts() const {
    return facts ? &*facts : nullptr;
  }
  llvm::StringRef getFailureReason() const { return failureReason; }

private:
  std::optional<StructuredRelationFacts> facts;
  std::string failureReason;
};

analysis::ExactDemandOutcome
deriveExactDemand(const StructuredRelationFacts &facts,
                  const SpatialAssignment &assignment,
                  const analysis::IndexRelationLimits &limits =
                      analysis::IndexRelationLimits());

/// Driver-owned immutable planning session. It owns facts and query-local
/// memoization, never an AnalysisManager handle. `close` is mandatory before
/// the first IR mutation; querying a closed session returns a typed compiler
/// contract error.
class DemandPlanningSession {
public:
  static mlir::FailureOr<DemandPlanningSession>
  create(const StructuredDAGAnalysis &dag,
         const analysis::IndexRelationLimits &limits =
             analysis::IndexRelationLimits(),
         std::string *failureReason = nullptr);

  DemandPlanningSession(DemandPlanningSession &&) noexcept;
  DemandPlanningSession &operator=(DemandPlanningSession &&) noexcept;
  ~DemandPlanningSession();
  DemandPlanningSession(const DemandPlanningSession &) = delete;
  DemandPlanningSession &operator=(const DemandPlanningSession &) = delete;

  analysis::ExactDemandOutcome query(const SpatialAssignment &assignment);
  void close();
  bool isClosed() const { return closed; }
  const StructuredRelationFacts &getFacts() const { return facts; }

private:
  class Cache;
  DemandPlanningSession(StructuredRelationFacts facts,
                        analysis::IndexRelationLimits limits);

  StructuredRelationFacts facts;
  analysis::IndexRelationLimits limits;
  std::unique_ptr<Cache> cache;
  bool closed = false;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_STRUCTUREDDEMANDANALYSIS_H
