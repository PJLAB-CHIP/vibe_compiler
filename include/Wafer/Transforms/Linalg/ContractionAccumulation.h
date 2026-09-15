//===- ContractionAccumulation.h - Explicit wide accumulation -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_CONTRACTIONACCUMULATION_H
#define WAFER_TRANSFORMS_LINALG_CONTRACTIONACCUMULATION_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::func {
class FuncOp;
}

namespace wafer {

/// Keeps a constant DPS initializer as an immutable tensor value rather than
/// an independent compute root. Nonconstant initializers remain unchanged.
mlir::LogicalResult foldContractionInitializers(mlir::func::FuncOp function);

/// Recognizes the existing low-precision multiply/add contraction contract.
bool requiresWideContractionState(mlir::Operation *operation);

/// Builds the actual F32 initializer and contraction in the caller's candidate.
/// The caller owns replacement of the original op and any final output cast.
mlir::FailureOr<mlir::linalg::GenericOp>
materializeContractionState(mlir::linalg::LinalgOp operation,
                            mlir::OpBuilder &builder);

/// Converts an actual tensor state at its final logical output boundary.
mlir::Value convertContractionState(mlir::Value state, mlir::Type element,
                                    mlir::OpBuilder &builder);

/// Materializes wide state inside the selected local function. This must not
/// run before structural partitioning in the production driver. Existing mixed
/// precision and noncanonical scalar payloads are unchanged.
mlir::LogicalResult promoteContractionAccumulation(mlir::func::FuncOp function);

} // namespace wafer

#endif
