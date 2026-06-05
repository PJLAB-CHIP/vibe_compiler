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
  std::optional<mlir::RankedTensorType> sourceTensor =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(getResult().getType());
  if (!sourceTensor || !resultTensor)
    return emitOpError("expects Wafer buffer source and result");
  if (!hasWaferMemorySpace(getSource().getType(), MemorySpace::SPM) ||
      !hasWaferMemorySpace(getResult().getType(), MemorySpace::SPM))
    return emitOpError("reshape source/result must use SPM memory space");
  if (getWaferLayout(getSource().getType()) !=
      getWaferLayout(getResult().getType()))
    return emitOpError("reshape must preserve layout");
  if (getWaferMemorySpace(getSource().getType()) !=
      getWaferMemorySpace(getResult().getType()))
    return emitOpError("reshape must preserve memory space");

  if (sourceTensor->getElementType() != resultTensor->getElementType())
    return emitOpError("reshape element types must match");

  std::optional<int64_t> sourceElements = getStaticElementCount(*sourceTensor);
  std::optional<int64_t> resultElements = getStaticElementCount(*resultTensor);
  if (!sourceElements || !resultElements)
    return emitOpError("reshape requires static tensor shapes");
  if (*sourceElements != *resultElements)
    return emitOpError("reshape must preserve static element count");
  return mlir::success();
}

void ViewReshapeOp::collectWaferLayoutRequirements(
    llvm::SmallVectorImpl<WaferLayoutRequirement> &requirements) {
  appendLayoutRequirement(requirements, WaferValueRole::Operand, 0,
                          getSource().getType());
  appendLayoutRequirement(requirements, WaferValueRole::Result, 0,
                          getResult().getType());
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
