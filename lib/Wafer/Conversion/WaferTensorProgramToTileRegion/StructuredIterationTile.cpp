//===- StructuredIterationTile.cpp - Exact iterator tile mechanics ------===//

#include "StructuredIterationTile.h"

#include "Internal.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace wafer::tensor_program_to_tile_region {

mlir::FailureOr<StructuredIterationTile> materializeStructuredIterationTile(
    mlir::Operation *operation, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, std::string *failureReason) {
  auto fail =
      [&](llvm::StringRef message) -> mlir::FailureOr<StructuredIterationTile> {
    setFailureReason(failureReason, message);
    return mlir::failure();
  };
  if (!operation)
    return fail("structured iterator tile requires an operation");
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(operation);
  if (!tiling)
    return fail("structured iterator tile requires TilingInterface");
  if (operation->getNumResults() == 0 || offsets.size() != sizes.size() ||
      offsets.size() != tiling.getLoopIteratorTypes().size())
    return fail("structured iterator tile rank does not match the operation");

  for (auto [requestedOffset, requestedSize] :
       llvm::zip_equal(offsets, sizes)) {
    std::optional<int64_t> offset = mlir::getConstantIntValue(requestedOffset);
    std::optional<int64_t> size = mlir::getConstantIntValue(requestedSize);
    if ((offset && *offset < 0) || (size && *size <= 0))
      return fail("structured iterator tile has a negative or empty bound");
  }

  StructuredIterationTile result;
  result.resultOffsets.resize(operation->getNumResults());
  result.resultSizes.resize(operation->getNumResults());
  for (unsigned resultNumber = 0; resultNumber < operation->getNumResults();
       ++resultNumber) {
    if (mlir::failed(
            tiling.getResultTilePosition(builder, resultNumber, offsets, sizes,
                                         result.resultOffsets[resultNumber],
                                         result.resultSizes[resultNumber])))
      return fail("TilingInterface cannot map one structured result tile");
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        operation->getResult(resultNumber).getType());
    if (!resultType ||
        result.resultOffsets[resultNumber].size() !=
            static_cast<size_t>(resultType.getRank()) ||
        result.resultSizes[resultNumber].size() !=
            static_cast<size_t>(resultType.getRank()))
      return fail("structured result tile rank is inconsistent");
  }

  mlir::FailureOr<mlir::TilingResult> tiled =
      tiling.getTiledImplementation(builder, offsets, sizes);
  if (mlir::failed(tiled))
    return fail("TilingInterface rejected the exact iterator tile");
  if (tiled->tiledOps.empty() ||
      tiled->tiledValues.size() != operation->getNumResults())
    return fail("TilingInterface returned an incomplete structured tile");
  result.operations.assign(tiled->tiledOps.begin(), tiled->tiledOps.end());
  result.values.assign(tiled->tiledValues.begin(), tiled->tiledValues.end());
  result.generatedSlices.assign(tiled->generatedSlices.begin(),
                                tiled->generatedSlices.end());
  return result;
}

} // namespace wafer::tensor_program_to_tile_region
