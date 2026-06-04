//===- ViewOps.cpp - Wafer view verifier implementation ----------------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

using namespace wafer;
using namespace wafer::detail;

static std::optional<int64_t>
getStaticElementCount(mlir::RankedTensorType tensorType) {
  if (!tensorType.hasStaticShape())
    return std::nullopt;
  int64_t count = 1;
  for (int64_t dim : tensorType.getShape()) {
    int64_t next = 0;
    if (!checkedMul(count, dim, next))
      return std::nullopt;
    count = next;
  }
  return count;
}

mlir::LogicalResult ViewReshapeOp::verify() {
  auto sourceType = mlir::dyn_cast<TileBufferType>(getSource().getType());
  auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitOpError("expects tile_buffer source and result");
  if (!hasTileBufferMemorySpace(sourceType, MemorySpace::SPM) ||
      !hasTileBufferMemorySpace(resultType, MemorySpace::SPM))
    return emitOpError("reshape source/result must use SPM memory space");
  if (getTileBufferLayout(sourceType).getValue() !=
      getTileBufferLayout(resultType).getValue())
    return emitOpError("reshape must preserve mem_layout");
  if (getTileBufferMemorySpace(sourceType).getValue() !=
      getTileBufferMemorySpace(resultType).getValue())
    return emitOpError("reshape must preserve memory_space");

  mlir::RankedTensorType sourceTensor = getTileBufferTensorType(sourceType);
  mlir::RankedTensorType resultTensor = getTileBufferTensorType(resultType);
  if (sourceTensor.getElementType() != resultTensor.getElementType())
    return emitOpError("reshape element types must match");

  std::optional<int64_t> sourceElements = getStaticElementCount(sourceTensor);
  std::optional<int64_t> resultElements = getStaticElementCount(resultTensor);
  if (!sourceElements || !resultElements)
    return emitOpError("reshape requires static tensor shapes");
  if (*sourceElements != *resultElements)
    return emitOpError("reshape must preserve static element count");
  return mlir::success();
}

void ViewReshapeOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  if (auto sourceType = mlir::dyn_cast<TileBufferType>(getSource().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                            sourceType);
  if (auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType()))
    appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                            resultType);
}

mlir::LogicalResult ViewReshapeOp::verifyWaferLayoutContract() {
  llvm::SmallVector<WaferLayoutRequirement, 2> requirements;
  collectWaferLayoutRequirements(requirements);
  return verifyLayoutRequirements(getOperation(), requirements);
}

mlir::OpFoldResult ViewReshapeOp::fold(FoldAdaptor adaptor) {
  if (getSource().getType() == getResult().getType())
    return getSource();
  return {};
}
