//===- SpatialPartitionPropagation.h ----------------------------*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_SPATIALPARTITIONPROPAGATION_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_SPATIALPARTITIONPROPAGATION_H

#include "Wafer/Planning/PhysicalDataflow/SpatialDomain.h"

namespace wafer::compiler::detail {

enum class SpatialPropagationOrder { ProducersFirst, ConsumersFirst };

/// Bounded proposal construction from current SSA/index relations. Unknown or
/// unrepresentable edge images retain the seed; no source IR is modified.
mlir::FailureOr<SpatialPlan> propagateSpatialPartitions(
    const SpatialPlanDomain &domain, const StructuredDAGAnalysis &dag,
    const SpatialPlan &seed, const analysis::IndexRelationLimits &limits,
    SpatialPropagationOrder order = SpatialPropagationOrder::ProducersFirst);

} // namespace wafer::compiler::detail

#endif
