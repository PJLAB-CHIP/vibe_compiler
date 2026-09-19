//===- ScalarExecution.h - Materialize scalar execution choices -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_SCALAREXECUTION_H
#define WAFER_TRANSFORMS_TILE_SCALAREXECUTION_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer::compiler::detail {

/// Query already decomposed pointwise SSA. This is target applicability, not
/// a performance or memory admission decision.
bool hasCPUScalarAlternative(mlir::ModuleOp module);

/// Materialize the selected register-only alternative in its original dynamic
/// scope. The driver compares this actual owner through the ordinary leaf.
mlir::FailureOr<unsigned>
materializeCPUScalarAlternative(mlir::ModuleOp module,
                                StructuredMaterializationRelations &relations);

} // namespace wafer::compiler::detail

#endif
