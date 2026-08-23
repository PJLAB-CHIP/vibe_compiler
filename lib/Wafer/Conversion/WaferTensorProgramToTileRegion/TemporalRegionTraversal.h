//===- TemporalRegionTraversal.h - Iterator wave loops -*- C++ -*-===//

#pragma once

#include "ProducerTileFusionInternal.h"
#include "TensorProgramScope.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"

namespace wafer::tensor_program_to_tile_region {

mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
materializeTemporalRegionTraversal(
    mlir::Operation *root, TensorProgramScope scope,
    llvm::ArrayRef<int64_t> spatialOffsets,
    llvm::ArrayRef<int64_t> spatialSizes,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles,
    mlir::ValueRange outputDestinations,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> &operationNodes,
    std::string *failureReason,
    llvm::SmallVectorImpl<MaterializedCoupledProducerTile>
        *sharedProducerTiles = nullptr);

} // namespace wafer::tensor_program_to_tile_region
