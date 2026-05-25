//===- WaferDialect.cpp - Wafer dialect implementation -------------------===//

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
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

static wafer::MemLayoutAttr getTileBufferLayout(wafer::TileBufferType type) {
  return mlir::cast<wafer::MemLayoutAttr>(type.getLayout());
}

static wafer::MemorySpaceAttr
getTileBufferMemorySpace(wafer::TileBufferType type) {
  return mlir::cast<wafer::MemorySpaceAttr>(type.getMemorySpace());
}

static mlir::RankedTensorType
getTileBufferTensorType(wafer::TileBufferType type) {
  return mlir::cast<mlir::RankedTensorType>(type.getTensorType());
}

static bool hasTileBufferLayout(wafer::TileBufferType type,
                                wafer::MemLayout layout) {
  return getTileBufferLayout(type).getValue() == layout;
}

static bool hasTileBufferMemorySpace(wafer::TileBufferType type,
                                     wafer::MemorySpace memorySpace) {
  return getTileBufferMemorySpace(type).getValue() == memorySpace;
}

static bool hasStaticMismatch(int64_t lhs, int64_t rhs) {
  return lhs != mlir::ShapedType::kDynamic && rhs != mlir::ShapedType::kDynamic &&
         lhs != rhs;
}

static mlir::LogicalResult verifyCommP2P(mlir::Operation *op,
                                         mlir::Value buffer,
                                         mlir::IntegerAttr peer,
                                         mlir::IntegerAttr bytes,
                                         mlir::Type tokenType) {
  auto tileBufferType = mlir::dyn_cast<TileBufferType>(buffer.getType());
  if (!tileBufferType)
    return op->emitOpError("comm p2p buffer must be a tile_buffer");
  if (!hasTileBufferMemorySpace(tileBufferType, MemorySpace::SPM))
    return op->emitOpError("comm p2p buffer must use SPM memory space");
  if (!mlir::isa<mlir::async::TokenType>(tokenType))
    return op->emitOpError("comm p2p result must be an async token");
  if (peer.getInt() < 0)
    return op->emitOpError("comm peer must be non-negative");
  if (bytes.getInt() <= 0)
    return op->emitOpError("comm byte count must be positive");
  return mlir::success();
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

mlir::LogicalResult CommRecvOp::verify() {
  return verifyCommP2P(getOperation(), getBuffer(), getPeerAttr(),
                       getBytesAttr(), getToken().getType());
}

mlir::LogicalResult CommSendOp::verify() {
  return verifyCommP2P(getOperation(), getBuffer(), getPeerAttr(),
                       getBytesAttr(), getToken().getType());
}

mlir::LogicalResult CommWaitOp::verify() {
  for (mlir::Value token : getTokens()) {
    if (!mlir::isa<mlir::async::TokenType>(token.getType()))
      return emitOpError("comm wait operands must be async tokens");
  }
  return mlir::success();
}

mlir::LogicalResult ComputeGemmOp::verify() {
  auto lhsType = mlir::dyn_cast<TileBufferType>(getLhs().getType());
  auto rhsType = mlir::dyn_cast<TileBufferType>(getRhs().getType());
  auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType());
  if (!lhsType || !rhsType || !resultType)
    return emitOpError("expects tile_buffer operands and result");

  for (TileBufferType type : {lhsType, rhsType, resultType}) {
    if (!hasTileBufferMemorySpace(type, MemorySpace::SPM))
      return emitOpError("gemm tile buffers must use SPM memory space");
    if (!hasTileBufferLayout(type, MemLayout::Cx))
      return emitOpError("gemm tile buffers must use cx mem_layout");
  }

  mlir::RankedTensorType lhsTensor = getTileBufferTensorType(lhsType);
  mlir::RankedTensorType rhsTensor = getTileBufferTensorType(rhsType);
  mlir::RankedTensorType resultTensor = getTileBufferTensorType(resultType);
  if (lhsTensor.getRank() != 2 || rhsTensor.getRank() != 2 ||
      resultTensor.getRank() != 2)
    return emitOpError("gemm expects rank-2 tile buffer tensor types");

  if (lhsTensor.getElementType() != rhsTensor.getElementType() ||
      lhsTensor.getElementType() != resultTensor.getElementType())
    return emitOpError("gemm operand and result element types must match");

  if (hasStaticMismatch(lhsTensor.getDimSize(1), rhsTensor.getDimSize(0)))
    return emitOpError("gemm lhs K dimension must match rhs K dimension");
  if (hasStaticMismatch(lhsTensor.getDimSize(0), resultTensor.getDimSize(0)) ||
      hasStaticMismatch(rhsTensor.getDimSize(1), resultTensor.getDimSize(1)))
    return emitOpError("gemm result shape must be lhs M by rhs N");

  return mlir::success();
}

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

mlir::LogicalResult LoadTileOp::verify() {
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(getSource().getType());
  auto resultType = mlir::dyn_cast<TileBufferType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitOpError("expects ranked tensor source and tile_buffer result");

  if (resultType.getTensorType() != sourceType)
    return emitOpError("load_tile result tensor type must match source tensor type");
  if (!hasTileBufferMemorySpace(resultType, MemorySpace::SPM))
    return emitOpError("load_tile result must use SPM memory space");
  if (!hasTileBufferLayout(resultType, MemLayout::Tensor))
    return emitOpError("load_tile result must use tensor mem_layout");

  return mlir::success();
}

mlir::LogicalResult StoreTileOp::verify() {
  auto sourceType = mlir::dyn_cast<TileBufferType>(getSource().getType());
  auto destType = mlir::dyn_cast<mlir::RankedTensorType>(getDest().getType());
  if (!sourceType || !destType)
    return emitOpError("expects tile_buffer source and ranked tensor dest");

  if (sourceType.getTensorType() != destType)
    return emitOpError("store_tile source tensor type must match dest tensor type");
  if (!hasTileBufferMemorySpace(sourceType, MemorySpace::SPM))
    return emitOpError("store_tile source must use SPM memory space");
  if (!hasTileBufferLayout(sourceType, MemLayout::Tensor))
    return emitOpError(
        "store_tile source must use tensor mem_layout for external writeback");

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
