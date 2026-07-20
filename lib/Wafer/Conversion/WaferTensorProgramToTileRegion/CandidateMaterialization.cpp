//===- CandidateMaterialization.cpp - Candidate conversion APIs -----===//

#include "Internal.h"

using namespace wafer;
using namespace wafer::tensor_program_to_tile_region;

namespace {

static mlir::FailureOr<mlir::func::FuncOp>
cloneVerifiedTensorProgram(mlir::func::FuncOp function,
                           mlir::OwningOpRef<mlir::ModuleOp> &module,
                           std::string *failureReason) {
  if (mlir::failed(verifyTensorProgramScope(function, failureReason)))
    return mlir::failure();
  module = wafer::detail::cloneTensorProgramToStandaloneModule(function);
  mlir::func::FuncOp cloned = findSingleStandaloneTensorProgram(*module);
  if (!cloned) {
    setFailureReason(
        failureReason,
        "standalone module must contain exactly one tensor program");
    return mlir::failure();
  }
  return cloned;
}

} // namespace

mlir::LogicalResult wafer::lowerCandidateTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative,
    bool useDirectMappedBoundaryTransfer) {
  if (failureReason)
    failureReason->clear();

  mlir::FailureOr<mlir::func::FuncOp> cloned =
      cloneVerifiedTensorProgram(function, module, failureReason);
  if (mlir::failed(cloned))
    return mlir::failure();
  TensorProgramScope scope(*cloned);
  if (mlir::failed(materializeCandidateTileSlices(
          scope, candidateTileOffsets, candidateTileSizes,
          candidateReductionTileSizes, failureReason)))
    return mlir::failure();
  return convertTensorProgramToTileRegionModuleInPlace(
      *module, function.getContext(), currentLogicalRank, failureReason,
      /*suppressDiagnostics=*/true, /*verifyResult=*/true,
      /*populateFallbackFailureReason=*/true, selectedAlternative,
      useDirectMappedBoundaryTransfer);
}

mlir::LogicalResult
wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative,
    bool useDirectMappedBoundaryTransfer) {
  if (failureReason)
    failureReason->clear();

  mlir::OwningOpRef<mlir::ModuleOp> candidateModule;
  mlir::FailureOr<mlir::func::FuncOp> cloned =
      cloneVerifiedTensorProgram(function, candidateModule, failureReason);
  if (mlir::failed(cloned))
    return mlir::failure();
  TensorProgramScope scope(*cloned);
  if (mlir::failed(materializeCompleteCandidateTraversal(
          scope, candidateTileSizes, candidateReductionTileSizes,
          failureReason)))
    return mlir::failure();
  if (mlir::failed(convertTensorProgramToTileRegionModuleInPlace(
          *candidateModule, function.getContext(), currentLogicalRank,
          failureReason, /*suppressDiagnostics=*/true, /*verifyResult=*/true,
          /*populateFallbackFailureReason=*/true, selectedAlternative,
          useDirectMappedBoundaryTransfer)))
    return mlir::failure();
  module = std::move(candidateModule);
  return mlir::success();
}
