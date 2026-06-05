//===- TileRegionOps.cpp - Wafer TileRegion verifier implementation
//----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult TileRegionOp::verify() {
  for (mlir::Type resultType : getResultTypes()) {
    if (isSPMStorage(resultType))
      return emitOpError(
          "SPM storage values cannot cross wafer.tile.region boundaries");
  }
  for (auto input : getInputs()) {
    if (isSPMStorage(input.getType()))
      return emitOpError(
          "SPM storage values cannot cross wafer.tile.region boundaries");
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
           << " body block arguments matching wafer.tile.region inputs, got "
           << block.getNumArguments();

  for (auto [index, inputAndArg] :
       llvm::enumerate(llvm::zip(getInputs(), block.getArguments()))) {
    mlir::Type inputType = std::get<0>(inputAndArg).getType();
    mlir::Type blockArgType = std::get<1>(inputAndArg).getType();
    if (blockArgType != inputType)
      return emitOpError("body block argument type ")
             << blockArgType << " does not match input type " << inputType
             << " at index " << index;
    if (isSPMStorage(blockArgType))
      return emitOpError(
          "SPM storage values cannot cross wafer.tile.region boundaries");
  }

  auto yield = mlir::dyn_cast<TileYieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("expected wafer.tile.yield terminator");

  if (yield.getValues().size() != getNumResults())
    return emitOpError(
               "expected tile.yield value count to match result count, got ")
           << yield.getValues().size() << " values and " << getNumResults()
           << " results";

  for (auto [index, yieldedAndResult] :
       llvm::enumerate(llvm::zip(yield.getValues(), getResults()))) {
    mlir::Type yieldedType = std::get<0>(yieldedAndResult).getType();
    mlir::Type resultType = std::get<1>(yieldedAndResult).getType();
    if (isSPMStorage(yieldedType))
      return emitOpError(
          "SPM storage values cannot cross wafer.tile.region boundaries");
    if (yieldedType != resultType)
      return emitOpError("tile.yield type ")
             << yieldedType << " does not match wafer.tile.region result type "
             << resultType << " at index " << index;
  }

  for (mlir::NamedAttribute attr : getOperation()->getAttrs())
    return emitOpError("does not accept semantic attributes");

  return mlir::success();
}
