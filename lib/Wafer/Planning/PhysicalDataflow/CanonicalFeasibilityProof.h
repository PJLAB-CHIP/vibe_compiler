//===- CanonicalFeasibilityProof.h - Close canonical resources -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALFEASIBILITYPROOF_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALFEASIBILITYPROOF_H

#include "Wafer/Planning/PhysicalDataflow/FeasibilityProof.h"
#include "Wafer/Target/Core/TargetMemory.h"

#include <optional>

namespace wafer::compiler::detail {

struct CanonicalFeasibilityOptions {
  /// Internal/test-only exact packing work control. Normal callers omit it.
  std::optional<uint64_t> packingSearchNodeBudget;
};

CanonicalFeasibilityOutcome buildCanonicalFeasibilityProof(
    const CanonicalStorageCoordinate &storage,
    const CanonicalMovementCoordinate &movements,
    const CanonicalScheduleCoordinate &schedule,
    const CanonicalAttentionWorkCoordinate &attention,
    const TargetMemoryPolicy &memory,
    const CanonicalFeasibilityOptions &options = {});

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALFEASIBILITYPROOF_H
