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
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    StructuredMaterializationRelations *materializationRelations,
    bool requireOneStructuredRootPerRegion) {
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
  if (mlir::failed(appendTileOutputDestinations(function, failureReason)))
    return mlir::failure();
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
    mappedTemporalTiles.push_back(StructuredOpTemporalTile{
        mapped, tile.iteratorTileSizes, tile.waveLoopOrder});
  }

  llvm::DenseSet<mlir::Operation *> seenNodeOperations;
  llvm::DenseSet<uint32_t> seenNodeIds;
  llvm::SmallVector<StructuredOperationNodeMapping, 16> mappedOperationNodes;
  mappedOperationNodes.reserve(operationNodes.size());
  for (const StructuredOperationNodeMapping &node : operationNodes) {
    if (!node.operation || !seenNodeOperations.insert(node.operation).second ||
        !seenNodeIds.insert(node.structuredNodeId).second) {
      setFailureReason(failureReason,
                       "structured operation-node mapping contains a null or "
                       "duplicate entry");
      return mlir::failure();
    }
    mlir::Operation *mapped = cloneMapping.lookupOrNull(node.operation);
    if (!mapped) {
      setFailureReason(failureReason,
                       "structured operation-node mapping is outside source "
                       "module");
      return mlir::failure();
    }
    mappedOperationNodes.push_back(
        {mapped, node.structuredNodeId, node.coupledComponentIndices});
  }

  TensorProgramScope scope(function, functionalArgumentCount);
  if (mlir::failed(materializeCandidateOutputTileSlices(
          scope, outputShards, mappedTemporalTiles, failureReason,
          &mappedOperationNodes)))
    return mlir::failure();
  TileRegionEmissionRelations emissionRelations;
  if (mlir::failed(convertTensorProgramToTileRegionModuleInPlace(
          *candidate, sourceModule.getContext(), functionalArgumentCount,
          currentLogicalPartition, failureReason,
          /*suppressDiagnostics=*/true,
          /*verifyResult=*/true,
          /*populateFallbackFailureReason=*/true,
          /*peerEndpoints=*/{}, /*selectedDDRStages=*/{}, &emissionRelations,
          mappedOperationNodes, requireOneStructuredRootPerRegion,
          /*representations=*/{}, /*implementations=*/{}, outputShards)))
    return mlir::failure();

  if (materializationRelations)
    *materializationRelations =
        std::move(emissionRelations.materializedBuffers);
  module = std::move(candidate);
  return mlir::success();
}
