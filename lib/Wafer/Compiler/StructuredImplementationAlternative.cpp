//===- StructuredImplementationAlternative.cpp -------------------------===//

#include "StructuredImplementationAlternative.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/Internal.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <utility>

namespace wafer::compiler::detail {
namespace {

static mlir::FailureOr<uint64_t> checkedProduct(llvm::ArrayRef<int64_t> values,
                                                llvm::StringRef label,
                                                std::string *failureReason) {
  uint64_t product = 1;
  for (int64_t value : values) {
    if (value <= 0) {
      if (failureReason)
        *failureReason = (label + " must be statically positive").str();
      return mlir::failure();
    }
    uint64_t factor = static_cast<uint64_t>(value);
    if (factor != 0 &&
        product > std::numeric_limits<uint64_t>::max() / factor) {
      if (failureReason)
        *failureReason = (label + " element product overflows").str();
      return mlir::failure();
    }
    product *= factor;
  }
  return product;
}

static mlir::FailureOr<uint64_t>
checkedTileCount(llvm::ArrayRef<int64_t> shape,
                 llvm::ArrayRef<int64_t> tileSizes, llvm::StringRef label,
                 std::string *failureReason) {
  if (shape.size() != tileSizes.size()) {
    if (failureReason)
      *failureReason = (label + " shape and tile ranks differ").str();
    return mlir::failure();
  }
  uint64_t count = 1;
  for (auto [extent, tile] : llvm::zip_equal(shape, tileSizes)) {
    if (extent <= 0 || tile <= 0 || tile > extent) {
      if (failureReason)
        *failureReason =
            (label + " tiles must be positive and within the domain").str();
      return mlir::failure();
    }
    uint64_t trips = static_cast<uint64_t>(extent / tile) +
                     static_cast<uint64_t>(extent % tile != 0);
    if (trips != 0 && count > std::numeric_limits<uint64_t>::max() / trips) {
      if (failureReason)
        *failureReason = (label + " tile count overflows").str();
      return mlir::failure();
    }
    count *= trips;
  }
  return count;
}

} // namespace

StructuredImplementationAlternativePoint::
    StructuredImplementationAlternativePoint(
        StructuredAlternativeKey key, StructuredAlternativeDomain domain,
        StructuredAlternativeParameters parameters,
        StructuredAlternativeStructuralEstimates estimates)
    : key(std::move(key)), domain(std::move(domain)),
      parameters(std::move(parameters)), estimates(std::move(estimates)) {}

mlir::FailureOr<StructuredAlternativeStructuralEstimates>
buildStructuredAlternativeStructuralEstimates(
    const StructuredAlternativeDomain &domain,
    const StructuredAlternativeParameters &parameters,
    std::string *failureReason) {
  if (domain.outputShape.empty()) {
    if (failureReason)
      *failureReason = "structured alternative requires an output domain";
    return mlir::failure();
  }
  if (parameters.parallelPartitionCount <= 0) {
    if (failureReason)
      *failureReason =
          "structured alternative parallel partition count must be positive";
    return mlir::failure();
  }
  if (domain.reductionShape.empty() != parameters.reductionTileSizes.empty()) {
    if (failureReason)
      *failureReason =
          "structured alternative reduction domain and tile rank differ";
    return mlir::failure();
  }

  mlir::FailureOr<uint64_t> outputElements =
      checkedProduct(domain.outputShape, "output domain", failureReason);
  mlir::FailureOr<uint64_t> reductionElements =
      checkedProduct(domain.reductionShape, "reduction domain", failureReason);
  mlir::FailureOr<uint64_t> outputTiles = checkedTileCount(
      domain.outputShape, parameters.outputTileSizes, "output", failureReason);
  mlir::FailureOr<uint64_t> reductionTiles =
      checkedTileCount(domain.reductionShape, parameters.reductionTileSizes,
                       "reduction", failureReason);
  if (mlir::failed(outputElements) || mlir::failed(reductionElements) ||
      mlir::failed(outputTiles) || mlir::failed(reductionTiles))
    return mlir::failure();

  const uint64_t partitions =
      static_cast<uint64_t>(parameters.parallelPartitionCount);
  if (*reductionTiles >
      std::numeric_limits<uint64_t>::max() - (partitions - 1)) {
    if (failureReason)
      *failureReason = "structured alternative work-unit bound overflows";
    return mlir::failure();
  }
  const uint64_t reductionUnits = *reductionTiles + partitions - 1;
  if (reductionUnits != 0 &&
      *outputTiles > std::numeric_limits<uint64_t>::max() / reductionUnits) {
    if (failureReason)
      *failureReason = "structured alternative work-unit bound overflows";
    return mlir::failure();
  }

  StructuredAlternativeStructuralEstimates result;
  result.logicalOutputElements = *outputElements;
  result.logicalReductionElements = *reductionElements;
  result.outputTileCount = *outputTiles;
  result.reductionTileCount = *reductionTiles;
  result.parallelPartitionCount = partitions;
  result.structuredWorkUnitUpperBound = *outputTiles * reductionUnits;
  return result;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeStructuredImplementationAlternativeToTileRegion(
    mlir::ModuleOp sourceModule,
    const StructuredImplementationAlternativePoint &point, int64_t logicalRank,
    std::string *failureReason) {
  wafer::tensor_program_to_tile_region::PreparedCompleteRankTensorProgram
      prepared;
  if (mlir::failed(wafer::tensor_program_to_tile_region::
                       prepareCompleteRankTensorProgram(sourceModule, prepared,
                                                        failureReason)))
    return mlir::failure();
  StructuredImplementationAlternativeMaterialization materialization;
  if (mlir::failed(
          point.materialize(*prepared.module, failureReason, &materialization)))
    return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> lowered;
  if (mlir::failed(
          wafer::tensor_program_to_tile_region::
              lowerPreparedCompleteRankTensorProgramToTileRegion(
                  std::move(prepared), lowered, failureReason, logicalRank,
                  materialization.coveredTopLevelOperations)))
    return mlir::failure();
  return lowered;
}

} // namespace wafer::compiler::detail
