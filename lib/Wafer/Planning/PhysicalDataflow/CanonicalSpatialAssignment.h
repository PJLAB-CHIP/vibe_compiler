//===- CanonicalSpatialAssignment.h - Direct spatial coordinate -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_CANONICALSPATIALASSIGNMENT_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_CANONICALSPATIALASSIGNMENT_H

#include "Wafer/Analysis/Linalg/SemanticRootAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

#include <string>

namespace wafer::compiler::detail {

/// One query-local deterministic spatial coordinate. `plan` is the compact
/// construction witness; `assignment` is its exact structural closure. The
/// semantic-root analysis keeps only current-borrow operation lookup handles.
struct CanonicalSpatialCoordinate {
  SemanticRootAnalysis semanticRoots;
  SpatialPlanningProblem problem;
  SpatialPlan plan;
  SpatialAssignment assignment;
};

mlir::FailureOr<CanonicalSpatialCoordinate>
buildCanonicalSpatialAssignment(const StructuredDAGAnalysis &dag,
                                llvm::ArrayRef<TileId> availableTiles,
                                std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_CANONICALSPATIALASSIGNMENT_H
