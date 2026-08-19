//===- TemporalTileShape.h ----------------------------------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_TEMPORALTILESHAPE_H
#define WAFER_COMPILER_PLANNING_TEMPORALTILESHAPE_H

#include "Wafer/Support/TargetPolicy.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace wafer::compiler::detail {

uint64_t estimateAlignedTileResidencyBytes(
    llvm::ArrayRef<int64_t> tileShape, uint64_t elementBytes,
    uint64_t tensorMultiplicity, const TargetMemoryPolicy &memory);

int64_t getNextLowerTemporalWaveTileSize(int64_t fullExtent,
                                         int64_t currentTileSize);

int64_t getNextLowerDivisibleTemporalTileSize(int64_t fullExtent,
                                              int64_t currentTileSize);

std::optional<unsigned> selectTemporalTileRefinementAxis(
    llvm::ArrayRef<int64_t> fullShape, llvm::ArrayRef<int64_t> currentShape,
    uint64_t knownBytesPerIterationPoint);

llvm::SmallVector<int64_t, 4> deriveCapacityTemporalTileShape(
    llvm::ArrayRef<int64_t> maximumShardShape, uint64_t elementBytes,
    uint64_t tensorMultiplicity, const TargetMemoryPolicy &memory,
    unsigned additionalWaveRefinements = 0);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_TEMPORALTILESHAPE_H
