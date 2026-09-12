//===- LayoutOptimization.h - Current layout/bufferization -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_LAYOUTOPTIMIZATION_H
#define WAFER_TRANSFORMS_TILE_LAYOUTOPTIMIZATION_H

#include "Wafer/Planning/PhysicalDataflow/ExactPBQPSolver.h"
#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "Wafer/IR/WaferDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

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
  uint64_t loopInvariantMaterializations = 0;
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

/// Layout parameters on current SSA values or exact current operands.
struct LayoutValueConstraint {
  mlir::Value value;
  MemLayout layout = MemLayout::Tensor;
};
struct LayoutUseConstraint {
  mlir::Operation *owner = nullptr;
  unsigned operandNumber = 0;
  MemLayout layout = MemLayout::Tensor;
};
using LayoutConstraint =
    std::variant<LayoutValueConstraint, LayoutUseConstraint>;

enum class LayoutDomain : uint8_t {
  AllLegal,
  /// Restricts a single local optimum query to its relevant publication/use
  /// states. This local objective reduction must not prune physical search.
  Relevant,
};

enum class LayoutMaterializationPlacement : uint8_t {
  FirstUse,
  LoopInvariant,
};

/// Apply the same placement choice to physical copies created by structured
/// lowering. The return value counts actually moved operations. No memory or
/// completion admission is performed here; both are rebuilt downstream.
mlir::FailureOr<uint64_t>
optimizePhysicalMovementPlacement(mlir::Operation *root,
                                  StructuredMaterializationRelations &relations,
                                  LayoutMaterializationPlacement placement);

class LayoutAssignmentQuery;
struct LayoutQueryResult {
  LayoutOptimizationResult outcome;
  std::unique_ptr<LayoutAssignmentQuery> query;
};

/// Read-only PBQP query on one unchanged, prepared actual module. It owns no
/// future allocations or instructions. Apply consumes an assignment on that
/// module or its exact IRMapping clone, then the mutated owner's query expires.
class LayoutAssignmentQuery {
public:
  ~LayoutAssignmentQuery();
  ExactPBQPResult solve(uint64_t workLimit,
                        std::optional<LayoutConstraint> constraint = {}) const;
  std::vector<LayoutConstraint>
  alternatives(const ExactPBQPResult &center) const;
  bool hasLoopInvariantPlacement(const ExactPBQPResult &assignment) const;
  LayoutOptimizationResult
  apply(mlir::ModuleOp target, StructuredMaterializationRelations &relations,
        const ExactPBQPResult &assignment,
        const mlir::IRMapping *mapping = nullptr,
        LayoutMaterializationPlacement placement =
            LayoutMaterializationPlacement::FirstUse) const;

private:
  struct Impl;
  explicit LayoutAssignmentQuery(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
  friend LayoutQueryResult queryCurrentLayoutAssignment(mlir::ModuleOp,
                                                        LayoutDomain);
};

LayoutOptimizationResult
prepareCurrentLayoutInput(mlir::ModuleOp module,
                          StructuredMaterializationRelations &relations);
LayoutQueryResult
queryCurrentLayoutAssignment(mlir::ModuleOp module,
                             LayoutDomain domain = LayoutDomain::AllLegal);

/// Resolves layouts directly on one candidate's current SSA/use graph, binds
/// observable output pieces to DDR subviews and runs function-boundary plus
/// region-local One-Shot Bufferization exactly once.  The supplied relations
/// are retargeted in the same transaction and never refer to source graph IDs.
///
/// A canonical factor-valid assignment guarantees a complete legal result;
/// PBQP minimizes physical bytes plus one unit per shared materialization,
/// and returns Feasible rather than claiming optimality when
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
