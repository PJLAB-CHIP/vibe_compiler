//===- CollectiveAlgorithms.h - Generic collective algorithms -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_COLLECTIVEALGORITHMS_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_COLLECTIVEALGORITHMS_H

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace wafer::compiler::detail {

/// Builds a deterministic minimum-cost Hamiltonian ring over participant IDs.
/// Topology ownership stays with the caller through the distance oracle.
mlir::FailureOr<llvm::SmallVector<uint64_t, 16>>
buildMinimumHopRing(
    llvm::ArrayRef<uint64_t> participants, uint64_t maximumParticipants,
    llvm::function_ref<std::optional<uint64_t>(uint64_t, uint64_t)>
        distanceOracle);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_COLLECTIVEALGORITHMS_H
