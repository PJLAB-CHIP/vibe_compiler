//===- CandidateMaterialization.cpp - Candidate conversion APIs -------===//

#include "Internal.h"

using namespace wafer;
using namespace wafer::group_to_tile_region;

mlir::LogicalResult wafer::lowerCandidateGroupToTileRegionModule(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank) {
  if (failureReason)
    failureReason->clear();

  if (!candidateReductionTileSizes.empty()) {
    setFailureReason(
        failureReason,
        "candidate reduction split is disabled because it cannot preserve "
        "source reduction order");
    return mlir::failure();
  }

  module = detail::cloneGroupToStandaloneModule(group);
  GroupOp clonedGroup = findSingleStandaloneGroup(*module);
  if (!clonedGroup) {
    setFailureReason(failureReason, "standalone module has no wafer.group");
    return mlir::failure();
  }

  if (mlir::failed(materializeCandidateTileSlices(
          clonedGroup, candidateTileOffsets, candidateTileSizes,
          candidateReductionTileSizes, failureReason)))
    return mlir::failure();

  return convertGroupToTileRegionModuleInPlace(
      *module, group.getContext(), currentLogicalRank, failureReason);
}

mlir::LogicalResult wafer::lowerCompleteCandidateGroupToTileRegionModule(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank) {
  if (failureReason)
    failureReason->clear();

  if (!candidateReductionTileSizes.empty()) {
    setFailureReason(
        failureReason,
        "candidate reduction split is disabled because it cannot preserve "
        "source reduction order");
    return mlir::failure();
  }

  mlir::OwningOpRef<mlir::ModuleOp> candidateModule =
      detail::cloneGroupToStandaloneModule(group);
  GroupOp clonedGroup = findSingleStandaloneGroup(*candidateModule);
  if (!clonedGroup) {
    setFailureReason(failureReason, "standalone module has no wafer.group");
    return mlir::failure();
  }

  if (mlir::failed(materializeCompleteCandidateTraversal(
          clonedGroup, candidateTileSizes, candidateReductionTileSizes,
          failureReason)))
    return mlir::failure();

  if (mlir::failed(convertGroupToTileRegionModuleInPlace(
          *candidateModule, group.getContext(), currentLogicalRank,
          failureReason)))
    return mlir::failure();

  module = std::move(candidateModule);
  return mlir::success();
}
