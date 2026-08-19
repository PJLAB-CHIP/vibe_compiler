//===- ProducerTileFusionInternal.h - Private producer fusion support -----===//

#ifndef WAFER_CONVERSION_TENSORPROGRAMTOTILEREGION_PRODUCERTILEFUSIONINTERNAL_H
#define WAFER_CONVERSION_TENSORPROGRAMTOTILEREGION_PRODUCERTILEFUSIONINTERNAL_H

#include "Internal.h"

#include "llvm/ADT/STLFunctionalExtras.h"

namespace wafer::tensor_program_to_tile_region {

struct MaterializedCoupledProducerTile {
  mlir::OpResult producerResult;
  mlir::Block *block = nullptr;
  mlir::Type tileType;
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  mlir::Value tiledValue;
};

using EnqueueProducerSlices = llvm::function_ref<void(
    llvm::ArrayRef<mlir::Operation *>, mlir::Operation *,
    llvm::ArrayRef<mlir::Operation *>)>;

/// Rewrites one extract window through the exact semantics of
/// tensor.insert_slice. Unsupported dynamic geometry remains unchanged; every
/// materialized input window is returned to the caller's producer-fusion
/// worklist.
void materializeWindowedInsertSlice(
    mlir::IRRewriter &rewriter, mlir::tensor::ExtractSliceOp slice,
    mlir::tensor::InsertSliceOp insert, mlir::OpResult producerResult,
    EnqueueProducerSlices enqueueSlices,
    llvm::SmallVectorImpl<MaterializedCoupledProducerTile>
        &materializedCoupledTiles);

} // namespace wafer::tensor_program_to_tile_region

#endif // WAFER_CONVERSION_TENSORPROGRAMTOTILEREGION_PRODUCERTILEFUSIONINTERNAL_H
