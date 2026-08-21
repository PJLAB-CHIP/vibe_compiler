//===- CanonicalRepresentationPlan.h - Tensor primary versions -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALREPRESENTATIONPLAN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALREPRESENTATIONPLAN_H

#include "Wafer/Planning/PhysicalDataflow/RepresentationPlan.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Builds one Tensor-encoded primary physical version for every nonempty
/// shaped logical value in the canonical B--E coordinate. The query emits no
/// IR and creates no derived conversion, alias, movement, or buffer choice.
CanonicalRepresentationPlanOutcome buildCanonicalRepresentationPlan(
    const RegionPlan &regions, const TemporalPlan &temporal,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALREPRESENTATIONPLAN_H
