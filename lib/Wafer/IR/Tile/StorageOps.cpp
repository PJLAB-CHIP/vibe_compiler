//===- StorageOps.cpp - Wafer SPM verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "WaferIRVerification.h"

#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

mlir::LogicalResult StorageLoadOp::verify() {
  std::optional<mlir::RankedTensorType> sourceType =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (!sourceType || !destTensor)
    return emitOpError("expects DDR memref source and SPM memref destination");

  if (*destTensor != sourceType)
    return emitOpError(
        "tile.load destination tensor type must match source tensor type");
  if (!hasWaferMemorySpace(getSource().getType(), MemorySpace::DDR))
    return emitOpError("tile.load source must use DDR memory space");
  if (!hasWaferLayout(getSource().getType(), MemLayout::Tensor))
    return emitOpError("tile.load source must use tensor layout");
  if (!hasWaferMemorySpace(getDest().getType(), MemorySpace::SPM))
    return emitOpError("tile.load destination must use SPM memory space");
  std::optional<MemLayout> destLayout = getWaferLayout(getDest().getType());
  if (!destLayout ||
      (*destLayout != MemLayout::Tensor && *destLayout != MemLayout::NTensor &&
       *destLayout != MemLayout::Cx && *destLayout != MemLayout::NCx))
    return emitOpError(
        "tile.load destination must use tensor/ntensor/cx/ncx layout");

  return mlir::success();
}

mlir::LogicalResult StorageStoreOp::verify() {
  std::optional<mlir::RankedTensorType> sourceTensor =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> destType =
      getLogicalTensorType(getDest().getType());
  if (!sourceTensor || !destType)
    return emitOpError("expects SPM memref source and DDR memref dest");

  if (*sourceTensor != destType)
    return emitOpError(
        "tile.store source tensor type must match dest tensor type");
  if (!hasWaferMemorySpace(getSource().getType(), MemorySpace::SPM))
    return emitOpError("tile.store source must use SPM memory space");
  std::optional<MemLayout> sourceLayout = getWaferLayout(getSource().getType());
  if (!sourceLayout ||
      (*sourceLayout != MemLayout::Tensor &&
       *sourceLayout != MemLayout::NTensor && *sourceLayout != MemLayout::Cx &&
       *sourceLayout != MemLayout::NCx))
    return emitOpError(
        "tile.store source must use tensor/ntensor/cx/ncx layout for external "
        "writeback");
  if (!hasWaferMemorySpace(getDest().getType(), MemorySpace::DDR))
    return emitOpError("tile.store dest must use DDR memory space");
  if (!hasWaferLayout(getDest().getType(), MemLayout::Tensor))
    return emitOpError("tile.store dest must use tensor layout");

  return mlir::success();
}
