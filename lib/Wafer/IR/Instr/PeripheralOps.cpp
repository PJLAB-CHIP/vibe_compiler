//===- PeripheralOps.cpp - Wafer peripheral instruction operations -------===//

#include "Wafer/IR/WaferDialect.h"

#include "InstructionVerifierUtils.h"
#include "OpVerifierUtils.h"

#include <initializer_list>
#include <optional>
#include <utility>

using namespace wafer;
using namespace wafer::detail;
using namespace wafer::instr_detail;

namespace {

static mlir::LogicalResult
verifyStaticElementCountMatches(mlir::Operation *op, mlir::Type type,
                                int64_t expected, llvm::StringRef role) {
  std::optional<mlir::RankedTensorType> tensor = getLogicalTensorType(type);
  if (!tensor || !tensor->hasStaticShape())
    return op->emitOpError() << "target_geometry_mismatch: " << role
                             << " element count must be statically known";
  if (expected < 0 || tensor->getNumElements() != expected)
    return op->emitOpError() << "target_geometry_mismatch: " << role
                             << " element count must equal elem_count";
  return mlir::success();
}

static std::pair<size_t, size_t> getPeripheralArity(InstrPeripheralKind kind) {
  switch (kind) {
  case InstrPeripheralKind::Count:
    return {1, 0};
  case InstrPeripheralKind::ArgMax:
  case InstrPeripheralKind::ArgMin:
    return {1, 2};
  case InstrPeripheralKind::Factorize:
    return {1, 3};
  case InstrPeripheralKind::Bilinear:
    return {1, 1};
  case InstrPeripheralKind::Lut16:
  case InstrPeripheralKind::Lut32:
    return {2, 1};
  case InstrPeripheralKind::RandGen:
    return {2, 3};
  case InstrPeripheralKind::ElemMask:
    return {1, 1};
  }
  llvm_unreachable("unknown peripheral kind");
}

static mlir::LogicalResult verifyPeripheralShapeAttr(mlir::Operation *op,
                                                     mlir::Type bufferType,
                                                     llvm::StringRef name) {
  auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(name);
  if (!attr)
    return op->emitOpError() << "peripheral kind requires " << name << " attr";
  if (mlir::failed(verifyI64Array(op, attr, name, 4, /*positive=*/true)))
    return mlir::failure();
  return verifyDataShapeAttrMatchesBuffer(op, bufferType, attr, name);
}

static mlir::LogicalResult verifyRequiredUInt32Attr(mlir::Operation *op,
                                                    llvm::StringRef name) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (!attr)
    return op->emitOpError() << "peripheral kind requires " << name << " attr";
  return verifyOptionalUInt32Attr(op, attr, name);
}

static mlir::LogicalResult
verifyForbiddenPeripheralAttrs(mlir::Operation *op,
                               std::initializer_list<llvm::StringRef> names) {
  for (llvm::StringRef name : names) {
    if (op->hasAttr(name))
      return op->emitOpError()
             << "peripheral kind must not have " << name << " attr";
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult InstrPeripheralOp::verify() {
  if (mlir::failed(verifyPositiveI64Attr(getOperation(), getElemCountAttr(),
                                         "elem_count")) ||
      mlir::failed(verifyUInt32Value(
          getOperation(), getElemCountAttr().getInt(), "elem_count")))
    return mlir::failure();

  auto [expectedInputs, expectedDests] =
      getPeripheralArity(getKindAttr().getValue());
  if (getInputs().size() != expectedInputs)
    return emitOpError() << "peripheral kind expects " << expectedInputs
                         << " input operand(s)";
  if (getDests().size() != expectedDests)
    return emitOpError() << "peripheral kind expects " << expectedDests
                         << " dest operand(s)";

  for (mlir::Value input : getInputs()) {
    if (mlir::failed(verifySPMMemRef(getOperation(), input.getType(), "input")))
      return mlir::failure();
  }
  for (mlir::Value dest : getDests()) {
    if (mlir::failed(verifySPMMemRef(getOperation(), dest.getType(), "dest")))
      return mlir::failure();
  }

  int64_t elemCount = getElemCountAttr().getInt();
  if (mlir::failed(verifyStaticElementCountMatches(
          getOperation(), getInputs().front().getType(), elemCount,
          "peripheral primary input")))
    return mlir::failure();

  if (getKindAttr().getValue() == InstrPeripheralKind::ArgMax ||
      getKindAttr().getValue() == InstrPeripheralKind::ArgMin) {
    std::optional<mlir::RankedTensorType> inputTensor =
        getLogicalTensorType(getInputs().front().getType());
    std::optional<mlir::RankedTensorType> valueTensor =
        getLogicalTensorType(getDests().front().getType());
    std::optional<mlir::RankedTensorType> indexTensor =
        getLogicalTensorType(getDests()[1].getType());
    if (inputTensor->getElementType() != valueTensor->getElementType())
      return emitOpError(
          "arg peripheral value dest element type must match input");
    if (!indexTensor->getElementType().isInteger(32))
      return emitOpError("arg peripheral index dest element type must be i32");
    if (mlir::failed(verifyStaticElementCountMatches(
            getOperation(), getDests().front().getType(), 1,
            "arg peripheral value dest")) ||
        mlir::failed(verifyStaticElementCountMatches(
            getOperation(), getDests()[1].getType(), 1,
            "arg peripheral index dest")))
      return mlir::failure();
  }
  switch (getKindAttr().getValue()) {
  case InstrPeripheralKind::Count:
    return emitOpError(
        "count peripheral writeback is not represented in instruction IR");
  case InstrPeripheralKind::ArgMax:
  case InstrPeripheralKind::ArgMin:
    return verifyForbiddenPeripheralAttrs(
        getOperation(), {"source_shape", "dest_shape", "lut_elem_count",
                         "scale", "probability", "rounding_mode"});
  case InstrPeripheralKind::Factorize:
    for (mlir::Value dest : getDests())
      if (mlir::failed(verifyStaticElementCountMatches(
              getOperation(), dest.getType(), elemCount, "factorize dest")))
        return mlir::failure();
    return verifyForbiddenPeripheralAttrs(
        getOperation(), {"source_shape", "dest_shape", "lut_elem_count",
                         "scale", "probability", "rounding_mode"});
  case InstrPeripheralKind::RandGen:
    if (mlir::failed(verifyStaticElementCountMatches(
            getOperation(), getInputs()[1].getType(), elemCount,
            "rand_gen secondary input")))
      return mlir::failure();
    for (mlir::Value dest : getDests())
      if (mlir::failed(verifyStaticElementCountMatches(
              getOperation(), dest.getType(), elemCount, "rand_gen dest")))
        return mlir::failure();
    return verifyForbiddenPeripheralAttrs(
        getOperation(), {"source_shape", "dest_shape", "lut_elem_count",
                         "scale", "probability", "rounding_mode"});
  case InstrPeripheralKind::Bilinear:
    if (mlir::failed(verifyPeripheralShapeAttr(
            getOperation(), getInputs().front().getType(), "source_shape")) ||
        mlir::failed(verifyPeripheralShapeAttr(
            getOperation(), getDests().front().getType(), "dest_shape")))
      return mlir::failure();
    return verifyForbiddenPeripheralAttrs(
        getOperation(),
        {"lut_elem_count", "scale", "probability", "rounding_mode"});
  case InstrPeripheralKind::Lut16:
  case InstrPeripheralKind::Lut32:
    if (mlir::failed(
            verifyRequiredUInt32Attr(getOperation(), "lut_elem_count")))
      return mlir::failure();
    if (mlir::failed(verifyStaticElementCountMatches(
            getOperation(), getDests().front().getType(), elemCount,
            "LUT dest")) ||
        mlir::failed(verifyStaticElementCountMatches(
            getOperation(), getInputs()[1].getType(),
            getLutElemCountAttr().getInt(), "LUT table")))
      return mlir::failure();
    return verifyForbiddenPeripheralAttrs(
        getOperation(), {"source_shape", "dest_shape", "scale", "probability",
                         "rounding_mode"});
  case InstrPeripheralKind::ElemMask:
    if (mlir::failed(verifyRequiredUInt32Attr(getOperation(), "scale")) ||
        mlir::failed(verifyRequiredUInt32Attr(getOperation(), "probability")) ||
        mlir::failed(verifyOptionalRoundingMode(
            getOperation(), getRoundingModeAttr(), "rounding_mode")))
      return mlir::failure();
    if (!getRoundingModeAttr())
      return emitOpError("peripheral kind requires rounding_mode attr");
    if (mlir::failed(verifyStaticElementCountMatches(
            getOperation(), getDests().front().getType(), elemCount,
            "elem_mask dest")))
      return mlir::failure();
    return verifyForbiddenPeripheralAttrs(
        getOperation(), {"source_shape", "dest_shape", "lut_elem_count"});
  }
  return mlir::success();
}

InstrFamily InstrPeripheralOp::getInstructionFamily() {
  return InstrFamily::CT;
}

NCCWorker InstrPeripheralOp::getIssueWorker() { return getWorker(); }
