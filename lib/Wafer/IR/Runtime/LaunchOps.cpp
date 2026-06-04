//===- LaunchOps.cpp - Wafer Launch verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult LaunchOp::verify() {
  if (getPackageRefAttr().getValue().empty())
    return emitOpError("launch package_ref must be non-empty");
  if (getSpmBytesAttr().getInt() < 0 || getDdrBytesAttr().getInt() < 0)
    return emitOpError("launch resource byte summaries must be non-negative");

  if (getNumResults() != getOutputs().size())
    return emitOpError("launch result count must match output count");

  for (mlir::Value input : getInputs()) {
    if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
      return emitOpError("launch boundary values must be ranked tensors");
  }
  for (mlir::Value output : getOutputs()) {
    if (!mlir::isa<mlir::RankedTensorType>(output.getType()))
      return emitOpError("launch boundary values must be ranked tensors");
  }

  for (auto [index, resultAndOutput] :
       llvm::enumerate(llvm::zip(getResults(), getOutputs()))) {
    mlir::Type resultType = std::get<0>(resultAndOutput).getType();
    mlir::Type outputType = std::get<1>(resultAndOutput).getType();
    if (!mlir::isa<mlir::RankedTensorType>(resultType))
      return emitOpError("launch boundary values must be ranked tensors");
    if (resultType != outputType)
      return emitOpError("launch result type must match output type at index ")
             << index;
  }

  return mlir::success();
}
