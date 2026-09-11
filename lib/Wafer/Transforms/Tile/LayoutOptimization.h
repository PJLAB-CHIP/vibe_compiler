//===- LayoutOptimization.h - Current layout/bufferization -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_LAYOUTOPTIMIZATION_H
#define WAFER_TRANSFORMS_TILE_LAYOUTOPTIMIZATION_H

#include "Wafer/Planning/PhysicalDataflow/ExactPBQPSolver.h"
#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <string>

namespace wafer::compiler::detail {

struct LayoutOptimizationStatistics {
  uint64_t invocations = 0;
  uint64_t functionBoundaryQueries = 0;
  uint64_t solverWork = 0;
  uint64_t pbqpVariables = 0;
  uint64_t pbqpFactors = 0;
  uint64_t valueGroups = 0;
  uint64_t dominatedLayoutStatesPruned = 0;
  uint64_t useBindings = 0;
  uint64_t tupleVariables = 0;
  uint64_t conversionActivations = 0;
  uint64_t canonicalAssignmentsBuilt = 0;
  uint64_t canonicalAssignmentFallbacks = 0;
  uint64_t selectedMaterializations = 0;
  uint64_t layoutMaterializationsBefore = 0;
  uint64_t layoutMaterializationsAfter = 0;
  uint64_t unusedMaterializationsErased = 0;
  uint64_t sharedMaterializationsReused = 0;
  uint64_t bufferizationInvocations = 0;
  uint64_t outputDestinations = 0;
  uint64_t outputSubviews = 0;
  uint64_t boundarySourceViewsElided = 0;
  uint64_t necessaryCopies = 0;
  uint64_t redundantPublicationCopies = 0;
};

struct LayoutOptimizationResult {
  ExactPBQPStatus status = ExactPBQPStatus::BrokenContract;
  LayoutOptimizationStatistics statistics;
  std::string detail;

  bool succeeded() const {
    return status == ExactPBQPStatus::Optimal ||
           status == ExactPBQPStatus::Feasible;
  }
};

/// Resolves layouts directly on one candidate's current SSA/use graph, binds
/// observable output pieces to DDR subviews and runs function-boundary plus
/// region-local One-Shot Bufferization exactly once.  The supplied relations
/// are retargeted in the same transaction and never refer to source graph IDs.
///
/// A canonical factor-valid assignment guarantees a complete legal result;
/// PBQP attempts to minimize the exact number of unique actual layout
/// materializations and returns Feasible rather than claiming optimality when
/// its work budget is exhausted. Target descriptor, engine and execution costs
/// belong to downstream actual IR analysis and never participate in layout
/// assignment.
LayoutOptimizationResult
resolveCurrentLayoutsAndBufferize(mlir::ModuleOp module,
                                  StructuredMaterializationRelations &relations,
                                  uint64_t workLimit = UINT64_C(1048576));

/// Checks the stable output boundary of the transformation.  Function tensor
/// boundaries and executable tensor semantics must be gone; tensor values may
/// remain only on TileRegion boundaries and their standard bufferization
/// bridges.
mlir::LogicalResult verifyLayoutResolvedTileRegions(mlir::ModuleOp module);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_LAYOUTOPTIMIZATION_H
