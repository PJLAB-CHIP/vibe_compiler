//===- StructuredIterationTile.h - Exact iterator tile mechanics -*- C++
//-*-===//

#pragma once

#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/SmallVector.h"

#include <string>

namespace wafer::tensor_program_to_tile_region {

/// Actual implementation returned by one exact TilingInterface iterator
/// request. The result positions are expressed in each structured result's
/// domain and are derived from the same interface invocation.
struct StructuredIterationTile {
  llvm::SmallVector<mlir::Operation *, 2> operations;
  llvm::SmallVector<mlir::Value, 2> values;
  llvm::SmallVector<mlir::Operation *, 4> generatedSlices;
  llvm::SmallVector<llvm::SmallVector<mlir::OpFoldResult>, 2> resultOffsets;
  llvm::SmallVector<llvm::SmallVector<mlir::OpFoldResult>, 2> resultSizes;
};

/// Validates and materializes one nonempty exact iterator rectangle. This is
/// the common policy-free primitive used by deterministic baseline traversal
/// and selected single-root region construction.
mlir::FailureOr<StructuredIterationTile> materializeStructuredIterationTile(
    mlir::Operation *operation, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, std::string *failureReason);

} // namespace wafer::tensor_program_to_tile_region
