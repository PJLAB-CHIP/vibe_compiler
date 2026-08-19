//===- EdgeFragmentPlanning.h - Peer fragment geometry -*- C++ -*-===//
#pragma once

#include "Internal.h"

namespace wafer::tensor_program_to_tile_region {

bool isContained(llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
                 llvm::ArrayRef<int64_t> containerOffsets,
                 llvm::ArrayRef<int64_t> containerSizes);

bool overlaps(llvm::ArrayRef<int64_t> lhsOffsets,
              llvm::ArrayRef<int64_t> lhsSizes,
              llvm::ArrayRef<int64_t> rhsOffsets,
              llvm::ArrayRef<int64_t> rhsSizes);

bool isFullStaticResultDomain(mlir::Operation *operation, unsigned resultNumber,
                              llvm::ArrayRef<int64_t> offsets,
                              llvm::ArrayRef<int64_t> sizes);

bool isOneFullTemporalWave(
    mlir::Operation *operation,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles);

bool isOneFullTemporalWaveClosure(
    mlir::Operation *operation,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles);

bool isOneFullTemporalWaveForResultDemand(
    mlir::Operation *operation, unsigned resultNumber,
    llvm::ArrayRef<int64_t> resultSizes,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles);

mlir::LogicalResult splitIndependentPeerFragmentsAtTemporalWaves(
    llvm::MutableArrayRef<SpatialEdgeStrategy> edgeStrategies,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason);

} // namespace wafer::tensor_program_to_tile_region
