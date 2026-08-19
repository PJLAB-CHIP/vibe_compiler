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

/// Producer fusion with a caller-owned function-local cache shared by several
/// sink tiles of one selected coupled group. The cache never crosses a block,
/// function, mutation epoch, or actual CardModule owner.
mlir::LogicalResult fuseCandidateProducerSlicesWithCache(
    mlir::Operation *tiledConsumer, mlir::Operation *sourceConsumer,
    TensorProgramScope scope,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    mlir::OpBuilder::Listener *insertionListener, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes,
    llvm::SmallVectorImpl<MaterializedCoupledProducerTile>
        &materializedProducerTiles);

/// Reuses exact producer tiles already materialized for an earlier sink of the
/// same coupled function before recursive fusion sees the new sink slices.
void reuseMaterializedProducerTiles(
    llvm::ArrayRef<mlir::Operation *> generatedSlices,
    llvm::ArrayRef<MaterializedCoupledProducerTile> materializedProducerTiles);

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
