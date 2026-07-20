//===- LayoutOps.cpp - Wafer Layout verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult LayoutMaterializeOp::verify() {
  std::optional<mlir::RankedTensorType> sourceTensor =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> resultTensor =
      getLogicalTensorType(getResult().getType());
  if (!sourceTensor || !resultTensor)
    return emitOpError("expects Wafer buffer source and result types");

  if (*sourceTensor != *resultTensor)
    return emitOpError("layout materialize must preserve logical tensor type");

  if (getWaferMemorySpace(getSource().getType()) !=
      getWaferMemorySpace(getResult().getType()))
    return emitOpError("layout materialize must preserve memory space");

  if (getWaferLayout(getSource().getType()) ==
      getWaferLayout(getResult().getType()))
    return emitOpError("layout materialize must change layout");

  return mlir::success();
}
