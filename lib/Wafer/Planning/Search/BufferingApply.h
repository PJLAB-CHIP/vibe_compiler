//===- BufferingApply.h - Lower selected buffer scopes -------*- C++ -*-===//

#pragma once

#include "Wafer/Planning/Search/Buffering.h"
#include "Wafer/Transforms/Bufferization/SelectedBufferMaterialization.h"

#include <vector>

namespace wafer::compiler::detail {

/// Converts a current buffering assignment into exact per-Tile materializer
/// scopes. Serialized choices produce no request and therefore no compiler
/// work. The result follows `expectedTileIds` order exactly.
mlir::FailureOr<std::vector<llvm::SmallVector<SelectedBufferingScope, 4>>>
buildSelectedBufferingScopes(
    const CardProgramAnalysis &program, const CardBufferingDomain &domain,
    const CardBufferingAssignment &assignment,
    const CardDataMovementDomain &movementDomain,
    const CardDataMovementAssignment &movementAssignment,
    llvm::ArrayRef<TileId> expectedTileIds,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
