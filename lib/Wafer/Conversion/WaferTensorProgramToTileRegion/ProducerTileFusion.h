//===- ProducerTileFusion.h - Structured producer tile fusion -*- C++ -*-===//
#pragma once

#include "TensorProgramScope.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace wafer::tensor_program_to_tile_region {

/// Recursively materializes the producers of slices created for one current
/// structured consumer tile. It follows SSA and TilingInterface relations;
/// no operation name or persisted correspondence participates.
mlir::LogicalResult fuseCandidateProducerSlices(
    mlir::Operation *tiledConsumer, mlir::Operation *sourceConsumer,
    TensorProgramScope scope,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    mlir::OpBuilder::Listener *insertionListener, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes =
        nullptr);

/// Same producer-fusion relation for a functional TensorProgram that has no
/// scheduling destination arguments or temporal assignment yet.
mlir::LogicalResult fuseTensorProgramProducerSlices(
    mlir::Operation *tiledConsumer, mlir::Operation *sourceConsumer,
    mlir::func::FuncOp function,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason);

/// Erases the dead pure tensor producer/view closure left after producer
/// tiling. Operations with observable or unknown effects remain explicit.
void eraseDeadCandidateSupportClosure(
    TensorProgramScope scope,
    llvm::ArrayRef<mlir::Operation *> preservedOperations = {});

void eraseDeadTensorProgramClosure(mlir::func::FuncOp function);

} // namespace wafer::tensor_program_to_tile_region
