//===- DDROps.cpp - Wafer DDR verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult DdrExternalBindingOp::verify() {
  auto tensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(getValue().getType());
  if (!tensorType || !tensorType.hasStaticShape())
    return emitOpError(
        "DDR external binding value must be a static ranked tensor");

  int64_t bytes = getBytesAttr().getInt();
  if (bytes <= 0)
    return emitOpError("DDR external binding bytes must be positive");

  std::optional<int64_t> expectedBytes = getCompactTensorByteSize(tensorType);
  if (!expectedBytes)
    return emitOpError("DDR external binding compact tensor storage size is "
                       "not representable");
  if (bytes != *expectedBytes)
    return emitOpError(
               "DDR external binding bytes must match compact tensor storage "
               "size, got ")
           << bytes << " and expected " << *expectedBytes;

  if (getAlignmentAttr().getInt() <= 0)
    return emitOpError("DDR external binding alignment must be positive");

  bool readOnly = getReadOnlyAttr().getValue();
  if (getKindAttr().getValue() == DdrBindingKind::Input && !readOnly)
    return emitOpError("DDR external input binding must be read-only");
  if (getKindAttr().getValue() == DdrBindingKind::Output && readOnly)
    return emitOpError("DDR external output binding must be writable");

  return mlir::success();
}
