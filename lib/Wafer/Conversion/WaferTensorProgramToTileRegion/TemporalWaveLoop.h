//===- TemporalWaveLoop.h - Compact iterator wave loops -*- C++ -*-===//

#pragma once

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace wafer::tensor_program_to_tile_region {

using TemporalWaveLeafBuilder =
    llvm::function_ref<mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>(
        mlir::OpBuilder &, llvm::ArrayRef<mlir::OpFoldResult>,
        llvm::ArrayRef<int64_t>, mlir::ValueRange,
        llvm::MutableArrayRef<mlir::LoopLikeOpInterface>)>;

/// Builds the compact prologue/steady/tail Cartesian traversal for one
/// already-selected iterator tile vector and loop order. The callback owns
/// leaf semantics; this function is the only owner of wave-loop construction.
mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
materializeTemporalWaveLoopNest(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<int64_t> iterationSizes,
    llvm::ArrayRef<int64_t> iteratorTileSizes,
    llvm::ArrayRef<uint32_t> waveLoopOrder, mlir::ValueRange initialValues,
    TemporalWaveLeafBuilder buildLeaf, std::string *failureReason);

mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
materializeTemporalWaveLoopNest(mlir::OpBuilder &builder, mlir::Location loc,
                                llvm::ArrayRef<int64_t> iterationOffsets,
                                llvm::ArrayRef<int64_t> iterationSizes,
                                llvm::ArrayRef<int64_t> iteratorTileSizes,
                                llvm::ArrayRef<uint32_t> waveLoopOrder,
                                mlir::ValueRange initialValues,
                                TemporalWaveLeafBuilder buildLeaf,
                                std::string *failureReason);

} // namespace wafer::tensor_program_to_tile_region
