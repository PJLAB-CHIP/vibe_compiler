//===- CandidateMaterialization.cpp - Candidate conversion APIs -----===//

#include "Internal.h"

using namespace wafer;
using namespace wafer::tensor_program_to_tile_region;

mlir::LogicalResult wafer::lowerSpatialOutputShardsToTileRegionModule(
    mlir::ModuleOp sourceModule, unsigned functionalArgumentCount,
    llvm::ArrayRef<SpatialOutputShard> outputShards,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalPartition,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<CardProgramSourceOperationLineage> sourceLineage,
    llvm::ArrayRef<StructuredOperandDemandLineage> operandDemandLineage) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModule) {
    setFailureReason(failureReason,
                     "spatial output lowering requires a source module");
    return mlir::failure();
  }
  if (currentLogicalPartition < 0) {
    setFailureReason(
        failureReason,
        "spatial output lowering requires a non-negative logical partition");
    return mlir::failure();
  }

  // The complete module is the atomic transformation boundary. Keep every
  // direct symbol and target fact alive while the ordinary candidate
  // materializer resolves symbol references and rewrites the current SSA.
  mlir::IRMapping cloneMapping;
  mlir::OwningOpRef<mlir::ModuleOp> candidate =
      mlir::cast<mlir::ModuleOp>(sourceModule->clone(cloneMapping));
  mlir::func::FuncOp function = findSingleStandaloneTensorProgram(*candidate);
  if (!function) {
    setFailureReason(
        failureReason,
        "spatial output module must contain exactly one defined tensor "
        "program function");
    return mlir::failure();
  }
  if (mlir::failed(verifyTensorProgramScope(function, functionalArgumentCount,
                                            failureReason)))
    return mlir::failure();

  llvm::DenseSet<mlir::Operation *> seenTemporalOperations;
  llvm::SmallVector<StructuredOpTemporalTile, 16> mappedTemporalTiles;
  mappedTemporalTiles.reserve(operationTemporalTiles.size());
  for (const StructuredOpTemporalTile &tile : operationTemporalTiles) {
    if (!tile.operation ||
        !seenTemporalOperations.insert(tile.operation).second) {
      setFailureReason(failureReason,
                       "structured temporal mapping contains a null or "
                       "duplicate operation");
      return mlir::failure();
    }
    mlir::Operation *mapped = cloneMapping.lookupOrNull(tile.operation);
    if (!mapped) {
      setFailureReason(
          failureReason,
          "structured temporal mapping operation is outside source module");
      return mlir::failure();
    }
    mappedTemporalTiles.push_back(
        StructuredOpTemporalTile{mapped, tile.iteratorTileSizes});
  }

  TensorProgramScope scope(function, functionalArgumentCount);
  if (mlir::failed(materializeCandidateOutputTileSlices(
          scope, outputShards, mappedTemporalTiles, failureReason)))
    return mlir::failure();
  if (mlir::failed(convertTensorProgramToTileRegionModuleInPlace(
          *candidate, sourceModule.getContext(), functionalArgumentCount,
          currentLogicalPartition, failureReason,
          /*suppressDiagnostics=*/true,
          /*verifyResult=*/true,
          /*populateFallbackFailureReason=*/true,
          /*peerEndpoints=*/{}, /*selectedDDRStages=*/{},
          /*emissionRelations=*/nullptr, sourceLineage, operandDemandLineage)))
    return mlir::failure();

  module = std::move(candidate);
  return mlir::success();
}
