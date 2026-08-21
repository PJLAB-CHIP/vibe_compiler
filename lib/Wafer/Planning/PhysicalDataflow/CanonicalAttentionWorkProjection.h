//===- CanonicalAttentionWorkProjection.h - Project attention -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALATTENTIONWORKPROJECTION_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALATTENTIONWORKPROJECTION_H

#include "Wafer/Planning/PhysicalDataflow/AttentionWorkDescription.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Projects fixed attention semantics onto the already closed canonical C--K
/// coordinate. The query emits no IR and creates no planning choice.
CanonicalAttentionWorkProjectionOutcome buildCanonicalAttentionWorkProjection(
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    const CanonicalRepresentationCoordinate &representations,
    const CanonicalMovementCoordinate &movements,
    const CanonicalStorageCoordinate &storage,
    const CanonicalScheduleCoordinate &schedule);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALATTENTIONWORKPROJECTION_H
