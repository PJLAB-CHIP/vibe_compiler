//===- WaferDialect.cpp - Wafer dialect implementation -------------------===//

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace wafer;

#include "Wafer/Dialect/Wafer/IR/WaferEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Wafer/Dialect/Wafer/IR/WaferAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Wafer/Dialect/Wafer/IR/WaferTypes.cpp.inc"

#include "Wafer/Dialect/Wafer/IR/WaferOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Wafer/Dialect/Wafer/IR/WaferOps.cpp.inc"

mlir::LogicalResult TileBufferType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type tensorType, mlir::Attribute layout,
    mlir::Attribute memorySpace) {
  if (!mlir::isa<mlir::RankedTensorType>(tensorType))
    return emitError() << "tile_buffer logical type must be a ranked tensor";
  if (!mlir::isa<MemLayoutAttr>(layout))
    return emitError() << "tile_buffer layout must be a wafer mem_layout attr";
  if (!mlir::isa<MemorySpaceAttr>(memorySpace))
    return emitError()
           << "tile_buffer memory space must be a wafer memory_space attr";
  return mlir::success();
}

static bool isSPMMemRef(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return false;
  auto memorySpace =
      mlir::dyn_cast_or_null<wafer::MemorySpaceAttr>(memrefType.getMemorySpace());
  return memorySpace && memorySpace.getValue() == wafer::MemorySpace::SPM;
}

static bool isSPMTileBuffer(mlir::Type type) {
  auto tileBufferType = mlir::dyn_cast<wafer::TileBufferType>(type);
  if (!tileBufferType)
    return false;
  auto memorySpace =
      mlir::cast<wafer::MemorySpaceAttr>(tileBufferType.getMemorySpace());
  return memorySpace.getValue() ==
         wafer::MemorySpace::SPM;
}

mlir::LogicalResult GroupOp::verify() {
  if (getNumResults() != getOuts().size())
    return emitOpError("expected result count to match outs count, got ")
           << getNumResults() << " results and " << getOuts().size()
           << " outs";

  for (auto [index, resultAndOut] :
       llvm::enumerate(llvm::zip(getResults(), getOuts()))) {
    mlir::Type resultType = std::get<0>(resultAndOut).getType();
    mlir::Type outType = std::get<1>(resultAndOut).getType();
    if (resultType != outType)
      return emitOpError("result type ")
             << resultType << " does not match outs type " << outType
             << " at index " << index;
  }

  for (auto input : getInputs()) {
    if (isSPMMemRef(input.getType()))
      return emitOpError("does not accept SPM memref inputs");
  }

  if (getBody().empty())
    return emitOpError("expected non-empty body region");

  mlir::Block &block = getBody().front();
  size_t expectedBlockArgs = getInputs().size() + getOuts().size();
  if (block.getNumArguments() != expectedBlockArgs)
    return emitOpError("expected ")
           << expectedBlockArgs
           << " body block arguments matching group ins plus outs, got "
           << block.getNumArguments();

  unsigned blockArgIndex = 0;
  for (auto input : getInputs()) {
    mlir::Type blockArgType = block.getArgument(blockArgIndex).getType();
    if (blockArgType != input.getType())
      return emitOpError("body block argument type ")
             << blockArgType << " does not match input type " << input.getType()
             << " at index " << blockArgIndex;
    if (isSPMMemRef(blockArgType))
      return emitOpError("does not accept SPM memref body arguments");
    ++blockArgIndex;
  }
  for (auto out : getOuts()) {
    mlir::Type blockArgType = block.getArgument(blockArgIndex).getType();
    if (blockArgType != out.getType())
      return emitOpError("body block argument type ")
             << blockArgType << " does not match outs type " << out.getType()
             << " at index " << blockArgIndex;
    ++blockArgIndex;
  }

  auto yield = mlir::dyn_cast<GroupYieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("expected wafer.group_yield terminator");

  if (yield.getValues().size() != getNumResults())
    return emitOpError("expected group_yield value count to match result count, got ")
           << yield.getValues().size() << " values and " << getNumResults()
           << " results";

  for (auto [index, yieldedAndResult] :
       llvm::enumerate(llvm::zip(yield.getValues(), getResults()))) {
    mlir::Type yieldedType = std::get<0>(yieldedAndResult).getType();
    mlir::Type resultType = std::get<1>(yieldedAndResult).getType();
    if (yieldedType != resultType)
      return emitOpError("group yield type ")
             << yieldedType << " does not match result type " << resultType
             << " at index " << index;
  }

  for (mlir::NamedAttribute attr : getOperation()->getAttrs()) {
    if (attr.getName() != getOperandSegmentSizesAttrName())
      return emitOpError("does not accept semantic attributes");
  }

  return mlir::success();
}

mlir::LogicalResult TileRegionOp::verify() {
  if (getBody().empty())
    return emitOpError("expected non-empty body region");

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

void WaferDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Wafer/Dialect/Wafer/IR/WaferAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Wafer/Dialect/Wafer/IR/WaferTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Wafer/Dialect/Wafer/IR/WaferOps.cpp.inc"
      >();
}
