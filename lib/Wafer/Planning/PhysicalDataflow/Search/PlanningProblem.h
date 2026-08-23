//===- PlanningProblem.h - Immutable physical planning input -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGPROBLEM_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGPROBLEM_H

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialDomain.h"

#include "mlir/Support/LogicalResult.h"

#include <string>
#include <utility>

namespace wafer::compiler::detail {

/// Immutable, policy-free facts for one search session. `program` is a
/// non-owning borrow whose outer CardProgramAnalysis must outlive the problem
/// and every session created from it. Its address never participates in state
/// identity.
class PhysicalDataflowPlanningProblem {
public:
  static mlir::FailureOr<PhysicalDataflowPlanningProblem>
  create(const CardProgramAnalysis &program, CardId cardId,
         const analysis::IndexRelationLimits &relationLimits =
             analysis::IndexRelationLimits(),
         std::string *failureReason = nullptr);

  const CardProgramAnalysis &getProgram() const { return program; }
  CardId getCardId() const { return cardId; }
  const SpatialPlanDomain &getSpatialDomain() const { return spatialDomain; }
  const analysis::IndexRelationLimits &getRelationLimits() const {
    return relationLimits;
  }

private:
  PhysicalDataflowPlanningProblem(const CardProgramAnalysis &program,
                                  CardId cardId,
                                  SpatialPlanDomain spatialDomain,
                                  analysis::IndexRelationLimits relationLimits)
      : program(program), cardId(cardId),
        spatialDomain(std::move(spatialDomain)),
        relationLimits(relationLimits) {}

  const CardProgramAnalysis &program;
  CardId cardId{0};
  SpatialPlanDomain spatialDomain;
  analysis::IndexRelationLimits relationLimits;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_PLANNINGPROBLEM_H
