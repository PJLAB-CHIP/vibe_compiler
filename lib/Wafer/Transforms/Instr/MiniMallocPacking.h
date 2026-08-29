//===- MiniMallocPacking.h - MiniMalloc adapter boundary --------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_INSTR_MINIMALLOCPACKING_H
#define WAFER_TRANSFORMS_INSTR_MINIMALLOCPACKING_H

#include "Wafer/Transforms/Instr/StaticMemoryPacking.h"

#include <cstdint>

namespace wafer::memory_planning::detail {

/// Converts a Wafer StaticPackingProblem into the private third-party
/// MiniMalloc representation, validates the boundary, and maps its typed
/// result back. No third-party type crosses this owner-private boundary.
PackingResult solveWithMiniMalloc(const StaticPackingProblem &problem,
                                  uint64_t searchNodeBudget);

} // namespace wafer::memory_planning::detail

#endif // WAFER_TRANSFORMS_INSTR_MINIMALLOCPACKING_H
