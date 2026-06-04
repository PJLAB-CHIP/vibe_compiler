//===- TileRegionOps.cpp - Wafer TileRegion verifier implementation
//----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult TileRegionOp::verify() {
  for (mlir::Type resultType : getResultTypes()) {
    if (isSPMTileBuffer(resultType))
      return emitOpError(
          "SPM tile buffers cannot cross wafer.tile_region boundaries");
  }
  for (auto input : getInputs()) {
    if (isSPMTileBuffer(input.getType()))
      return emitOpError(
          "SPM tile buffers cannot cross wafer.tile_region boundaries");
  }

  return mlir::success();
}

mlir::LogicalResult TileRegionOp::verifyRegions() {
  if (getBody().empty())
    return emitOpError("expected non-empty body region");

  mlir::Block &block = getBody().front();
  if (block.getNumArguments() != getInputs().size())
    return emitOpError("expected ")
           << getInputs().size()
           << " body block arguments matching tile_region inputs, got "
           << block.getNumArguments();

  for (auto [index, inputAndArg] :
       llvm::enumerate(llvm::zip(getInputs(), block.getArguments()))) {
    mlir::Type inputType = std::get<0>(inputAndArg).getType();
    mlir::Type blockArgType = std::get<1>(inputAndArg).getType();
    if (blockArgType != inputType)
      return emitOpError("body block argument type ")
             << blockArgType << " does not match input type " << inputType
             << " at index " << index;
    if (isSPMTileBuffer(blockArgType))
      return emitOpError(
          "SPM tile buffers cannot cross wafer.tile_region boundaries");
  }

  auto yield = mlir::dyn_cast<TileYieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("expected wafer.tile_yield terminator");

  if (yield.getValues().size() != getNumResults())
    return emitOpError(
               "expected tile_yield value count to match result count, got ")
           << yield.getValues().size() << " values and " << getNumResults()
           << " results";

  for (auto [index, yieldedAndResult] :
       llvm::enumerate(llvm::zip(yield.getValues(), getResults()))) {
    mlir::Type yieldedType = std::get<0>(yieldedAndResult).getType();
    mlir::Type resultType = std::get<1>(yieldedAndResult).getType();
    if (isSPMTileBuffer(yieldedType))
      return emitOpError(
          "SPM tile buffers cannot cross wafer.tile_region boundaries");
    if (yieldedType != resultType)
      return emitOpError("tile_yield type ")
             << yieldedType << " does not match tile_region result type "
             << resultType << " at index " << index;
  }

  for (mlir::NamedAttribute attr : getOperation()->getAttrs())
    return emitOpError("does not accept semantic attributes");

  return mlir::success();
}
