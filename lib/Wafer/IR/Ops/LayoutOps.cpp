//===- LayoutOps.cpp - Wafer Layout verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult LayoutMaterializeOp::verify() {
  auto sourceType = mlir::dyn_cast<TileBufferType>(getSource().getType());
  auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitOpError("expects tile_buffer source and result types");

  if (sourceType.getTensorType() != resultType.getTensorType())
    return emitOpError("layout materialize must preserve logical tensor type");

  if (getTileBufferMemorySpace(sourceType).getValue() !=
      getTileBufferMemorySpace(resultType).getValue())
    return emitOpError("layout materialize must preserve memory space");

  if (getTileBufferLayout(sourceType).getValue() ==
      getTileBufferLayout(resultType).getValue())
    return emitOpError("layout materialize must change mem_layout");

  return mlir::success();
}
