//===- ConfiguredReductionTraversal.h ------------------------*- C++ -*-===//

#pragma once

#include "TensorProgramScope.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace wafer::tensor_program_to_tile_region {

mlir::FailureOr<mlir::Value> materializeConfiguredComputeTile(
    mlir::OpBuilder &builder, TensorProgramScope scope,
    mlir::linalg::LinalgOp sourceCompute,
    llvm::ArrayRef<mlir::OpFoldResult> outputOffsets,
    llvm::ArrayRef<int64_t> outputSizes,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes);

} // namespace wafer::tensor_program_to_tile_region
