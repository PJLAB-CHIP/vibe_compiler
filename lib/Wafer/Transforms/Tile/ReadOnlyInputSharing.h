//===- ReadOnlyInputSharing.h - Actual external input reuse -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_READONLYINPUTSHARING_H
#define WAFER_TRANSFORMS_TILE_READONLYINPUTSHARING_H

#include "BoundaryMovement.h"

namespace wafer::compiler::detail {

/// Query existing loads with proved equal windows in matching static loop
/// domains.
bool hasReadOnlyInputSharing(mlir::ModuleOp module);

/// Replace duplicate DDR loads with actual peer messages in this transaction.
/// Completion and SPM placement remain downstream responsibilities.
BoundaryMovementResult
materializeReadOnlyInputSharing(mlir::ModuleOp module,
                                StructuredMaterializationRelations &relations);

} // namespace wafer::compiler::detail

#endif
