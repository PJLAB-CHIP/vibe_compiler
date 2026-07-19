//===- CandidateMaterialization.cpp - Candidate conversion APIs -----===//

#include "Internal.h"

#include <limits>

using namespace wafer;
using namespace wafer::tensor_program_to_tile_region;

namespace {

static mlir::FailureOr<CandidateInvocationOrdinals>
reserveCandidateInvocationOrdinals(int64_t currentLogicalRank,
                                   uint64_t invocationOrdinalBase,
                                   std::string *failureReason) {
  constexpr uint64_t invocationRangeSize = uint64_t{1} << 12;
  if (currentLogicalRank < 0 || currentLogicalRank > 0xffff) {
    setFailureReason(failureReason,
                     "candidate invocation ordinal rank is outside u16");
    return mlir::failure();
  }
  if (invocationOrdinalBase >
      std::numeric_limits<uint64_t>::max() - invocationRangeSize) {
    setFailureReason(failureReason,
                     "candidate invocation ordinal range is out of bounds");
    return mlir::failure();
  }
  return CandidateInvocationOrdinals{invocationOrdinalBase,
                                     invocationOrdinalBase,
                                     invocationOrdinalBase,
                                     invocationOrdinalBase +
                                         invocationRangeSize};
}

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
    int64_t currentLogicalRank, uint64_t invocationOrdinalBase) {
  if (failureReason)
    failureReason->clear();

  mlir::FailureOr<mlir::func::FuncOp> cloned =
      cloneVerifiedTensorProgram(function, module, failureReason);
  if (mlir::failed(cloned))
    return mlir::failure();
  TensorProgramScope scope(*cloned);
  mlir::FailureOr<CandidateInvocationOrdinals> invocationOrdinals =
      reserveCandidateInvocationOrdinals(currentLogicalRank,
                                         invocationOrdinalBase,
                                         failureReason);
  if (mlir::failed(invocationOrdinals))
    return mlir::failure();
  if (mlir::failed(materializeCandidateTileSlices(
          scope, candidateTileOffsets, candidateTileSizes,
          candidateReductionTileSizes, *invocationOrdinals, failureReason)))
    return mlir::failure();
  return convertTensorProgramToTileRegionModuleInPlace(
      *module, function.getContext(), currentLogicalRank, failureReason);
}

mlir::LogicalResult
wafer::lowerCompleteCandidateTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank, uint64_t invocationOrdinalBase) {
  if (failureReason)
    failureReason->clear();

  mlir::OwningOpRef<mlir::ModuleOp> candidateModule;
  mlir::FailureOr<mlir::func::FuncOp> cloned =
      cloneVerifiedTensorProgram(function, candidateModule, failureReason);
  if (mlir::failed(cloned))
    return mlir::failure();
  TensorProgramScope scope(*cloned);
  mlir::FailureOr<CandidateInvocationOrdinals> invocationOrdinals =
      reserveCandidateInvocationOrdinals(currentLogicalRank,
                                         invocationOrdinalBase,
                                         failureReason);
  if (mlir::failed(invocationOrdinals))
    return mlir::failure();
  if (mlir::failed(materializeCompleteCandidateTraversal(
          scope, candidateTileSizes, candidateReductionTileSizes,
          *invocationOrdinals, failureReason)))
    return mlir::failure();
  if (mlir::failed(convertTensorProgramToTileRegionModuleInPlace(
          *candidateModule, function.getContext(), currentLogicalRank,
          failureReason)))
    return mlir::failure();
  module = std::move(candidateModule);
  return mlir::success();
}
