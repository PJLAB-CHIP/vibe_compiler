//===- ComputeOps.cpp - Wafer compute instruction operations -------------===//

#include "Wafer/IR/WaferDialect.h"

#include "InstructionVerification.h"
#include "WaferIRVerification.h"

#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <optional>
#include <utility>

using namespace wafer;
using namespace wafer::detail;
using namespace wafer::instr_detail;

namespace {

static mlir::LogicalResult
verifyStaticElementCountEqual(mlir::Operation *op, mlir::Type lhsType,
                              mlir::Type rhsType, llvm::StringRef message) {
  std::optional<mlir::RankedTensorType> lhs = getLogicalTensorType(lhsType);
  std::optional<mlir::RankedTensorType> rhs = getLogicalTensorType(rhsType);
  if (!lhs || !rhs || !lhs->hasStaticShape() || !rhs->hasStaticShape())
    return op->emitOpError(
        "target_geometry_mismatch: instruction element counts must be "
        "statically known");
  if (lhs->getNumElements() != rhs->getNumElements())
    return op->emitOpError() << "target_geometry_mismatch: " << message;
  return mlir::success();
}

static mlir::LogicalResult
verifyStaticElementCountFitsUInt32(mlir::Operation *op, mlir::Type type,
                                   llvm::StringRef role) {
  std::optional<mlir::RankedTensorType> tensor = getLogicalTensorType(type);
  if (!tensor || !tensor->hasStaticShape())
    return op->emitOpError() << "target_geometry_mismatch: " << role
                             << " element count must be statically known";

  uint64_t elements = 1;
  for (int64_t dim : tensor->getShape()) {
    if (elements != 0 && static_cast<uint64_t>(dim) >
                             std::numeric_limits<uint32_t>::max() / elements)
      return op->emitOpError() << "target_abi_narrowing: " << role
                               << " element count must fit uint32_t";
    elements *= static_cast<uint64_t>(dim);
  }
  return mlir::success();
}

static mlir::LogicalResult
verifyStaticPhysicalElementCountFitsUInt32(mlir::Operation *op, mlir::Type type,
                                           llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  std::optional<WaferPhysicalTensorInfo> info =
      memrefType ? computeWaferPhysicalTensorInfo(memrefType) : std::nullopt;
  if (!info || info->physicalElements <= 0)
    return op->emitOpError()
           << "target_geometry_mismatch: " << role
           << " physical traversal count must be static and positive";
  if (static_cast<uint64_t>(info->physicalElements) >
      std::numeric_limits<uint32_t>::max())
    return op->emitOpError() << "target_abi_narrowing: " << role
                             << " physical traversal count must fit uint32_t";
  return mlir::success();
}

static mlir::LogicalResult verifyStaticDataShapeBounds(mlir::Operation *op,
                                                       mlir::Type type,
                                                       llvm::StringRef role) {
  std::optional<mlir::RankedTensorType> tensor = getLogicalTensorType(type);
  if (!tensor || !tensor->hasStaticShape() || tensor->getRank() > 4)
    return op->emitOpError() << "target_geometry_mismatch: " << role
                             << " shape must be static with rank at most 4";
  return verifyDataShapeBounds(op, tensor->getShape(), role);
}

static mlir::LogicalResult verifyConvShapeRelation(
    mlir::Operation *op, InstrConvKind kind, mlir::DenseI64ArrayAttr inputShape,
    mlir::DenseI64ArrayAttr weightShape, mlir::DenseI64ArrayAttr outputShape,
    mlir::DenseI64ArrayAttr pads, mlir::DenseI64ArrayAttr unpads,
    mlir::DenseI64ArrayAttr kernelStrides, mlir::DenseI64ArrayAttr dilations) {
  if (kind != InstrConvKind::Conv)
    return op->emitOpError(
        "unsupported_target_geometry: only the ordinary convolution output "
        "relation is defined by the current instruction contract");

  llvm::ArrayRef<int64_t> input = inputShape.asArrayRef();
  llvm::ArrayRef<int64_t> weight = weightShape.asArrayRef();
  llvm::ArrayRef<int64_t> output = outputShape.asArrayRef();
  llvm::ArrayRef<int64_t> pad = pads.asArrayRef();
  llvm::ArrayRef<int64_t> unpad = unpads.asArrayRef();
  llvm::ArrayRef<int64_t> kernelStride = kernelStrides.asArrayRef();
  llvm::ArrayRef<int64_t> dilation = dilations.asArrayRef();
  int64_t kernelX = kernelStride[0];
  int64_t kernelY = kernelStride[1];
  int64_t strideX = kernelStride[2];
  int64_t strideY = kernelStride[3];
  int64_t dilationX = dilation[0];
  int64_t dilationY = dilation[1];

  if (kernelX != weight[1] || kernelY != weight[0])
    return op->emitOpError(
        "target_geometry_mismatch: convolution kernel dimensions must match "
        "the weight shape");
  if (output[0] != input[0])
    return op->emitOpError(
        "target_geometry_mismatch: convolution batch dimensions must match");
  if (input[3] != weight[3])
    return op->emitOpError(
        "target_geometry_mismatch: convolution input channels must match the "
        "weight input channels");

  int64_t expectedChannels = weight[2];
  if (output[3] != expectedChannels)
    return op->emitOpError(
        "target_geometry_mismatch: convolution output channels do not match "
        "the weight relation");

  mlir::FailureOr<int64_t> expectedH = computeWindowedOutputDim(
      op, input[1], kernelY, strideY, dilationY, pad[0], pad[1], unpad[0],
      unpad[1], "convolution height");
  mlir::FailureOr<int64_t> expectedW = computeWindowedOutputDim(
      op, input[2], kernelX, strideX, dilationX, pad[2], pad[3], unpad[2],
      unpad[3], "convolution width");
  if (mlir::failed(expectedH) || mlir::failed(expectedW))
    return mlir::failure();
  if (output[1] != *expectedH || output[2] != *expectedW)
    return op->emitOpError(
        "target_geometry_mismatch: convolution output spatial shape does not "
        "match input/kernel/stride/dilation/pad/unpad");
  return mlir::success();
}

static mlir::LogicalResult verifyPoolShapeRelation(
    mlir::Operation *op, mlir::DenseI64ArrayAttr sourceShape,
    mlir::DenseI64ArrayAttr destShape, mlir::DenseI64ArrayAttr pads,
    mlir::DenseI64ArrayAttr kernelStrides) {
  llvm::ArrayRef<int64_t> source = sourceShape.asArrayRef();
  llvm::ArrayRef<int64_t> dest = destShape.asArrayRef();
  llvm::ArrayRef<int64_t> pad = pads.asArrayRef();
  llvm::ArrayRef<int64_t> kernelStride = kernelStrides.asArrayRef();
  if (dest[0] != source[0] || dest[3] != source[3])
    return op->emitOpError(
        "target_geometry_mismatch: pool batch and channel dimensions must "
        "match");
  mlir::FailureOr<int64_t> expectedH = computeWindowedOutputDim(
      op, source[1], kernelStride[1], kernelStride[3], /*dilation=*/1, pad[0],
      pad[1], /*unpadBefore=*/0, /*unpadAfter=*/0, "pool height");
  mlir::FailureOr<int64_t> expectedW = computeWindowedOutputDim(
      op, source[2], kernelStride[0], kernelStride[2], /*dilation=*/1, pad[2],
      pad[3], /*unpadBefore=*/0, /*unpadAfter=*/0, "pool width");
  if (mlir::failed(expectedH) || mlir::failed(expectedW))
    return mlir::failure();
  if (dest[1] != *expectedH || dest[2] != *expectedW)
    return op->emitOpError(
        "target_geometry_mismatch: pool destination spatial shape does not "
        "match source/kernel/stride/pad");
  return mlir::success();
}

static mlir::LogicalResult verifyUnpoolShapeRelation(
    mlir::Operation *op, mlir::DenseI64ArrayAttr sourceShape,
    mlir::DenseI64ArrayAttr destShape, mlir::DenseI64ArrayAttr kernelStrides) {
  llvm::ArrayRef<int64_t> source = sourceShape.asArrayRef();
  llvm::ArrayRef<int64_t> dest = destShape.asArrayRef();
  llvm::ArrayRef<int64_t> kernelStride = kernelStrides.asArrayRef();
  if (dest[0] != source[0] || dest[3] != source[3])
    return op->emitOpError(
        "target_geometry_mismatch: unpool batch and channel dimensions must "
        "match");
  int64_t expectedH = 0;
  int64_t expectedW = 0;
  int64_t scaledH = 0;
  int64_t scaledW = 0;
  if (!checkedMul(source[1] - 1, kernelStride[3], scaledH) ||
      !checkedAdd(scaledH, kernelStride[1], expectedH) ||
      !checkedMul(source[2] - 1, kernelStride[2], scaledW) ||
      !checkedAdd(scaledW, kernelStride[0], expectedW))
    return op->emitOpError(
        "target_range_overflow: unpool spatial geometry overflows int64");
  if (dest[1] != expectedH || dest[2] != expectedW)
    return op->emitOpError(
        "target_geometry_mismatch: unpool destination spatial shape does not "
        "match source/kernel/stride");
  return mlir::success();
}

static mlir::LogicalResult verifyAlignedSPMGemmMemRef(mlir::Operation *op,
                                                      mlir::Type type,
                                                      llvm::StringRef role) {
  if (mlir::failed(verifySPMMemRef(op, type, role)))
    return mlir::failure();
  std::optional<mlir::RankedTensorType> tensor = getLogicalTensorType(type);
  if (!tensor)
    return op->emitOpError() << role << " must be a ranked Wafer memref";
  const MemLayout expectedLayout =
      tensor->getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  std::optional<MemLayout> layout = getWaferLayout(type);
  if (!layout || *layout != expectedLayout)
    return op->emitOpError()
           << role
           << (tensor->getRank() > 2 ? " rank > 2 must use ncx layout"
                                     : " rank <= 2 must use cx layout");
  return mlir::success();
}

static bool isAlignedGemmLayout(MemLayout layout) {
  return layout == MemLayout::Cx || layout == MemLayout::NCx;
}

static mlir::LogicalResult verifyAlignedSPMMemRef(mlir::Operation *op,
                                                  mlir::Type type,
                                                  llvm::StringRef role) {
  if (mlir::failed(verifySPMMemRef(op, type, role)))
    return mlir::failure();
  std::optional<MemLayout> layout = getWaferLayout(type);
  if (!layout || !isAlignedGemmLayout(*layout))
    return op->emitOpError() << role << " must use cx/ncx SPM layout";
  return mlir::success();
}

static mlir::LogicalResult
verifyNCxSPMMemRef(mlir::Operation *op, mlir::Type type, llvm::StringRef role) {
  if (mlir::failed(verifySPMMemRef(op, type, role)))
    return mlir::failure();
  if (!hasWaferLayout(type, MemLayout::NCx))
    return op->emitOpError()
           << role
           << " must use ncx SPM layout; native pool/unpool has no layout "
              "operand and cx requires explicit materialization";
  return mlir::success();
}

static mlir::LogicalResult verifySameShape(mlir::Operation *op,
                                           mlir::RankedTensorType lhs,
                                           mlir::RankedTensorType rhs,
                                           llvm::StringRef message) {
  if (lhs.getRank() != rhs.getRank())
    return op->emitOpError() << message;
  for (auto [lhsDim, rhsDim] : llvm::zip(lhs.getShape(), rhs.getShape())) {
    if (hasStaticMismatch(lhsDim, rhsDim))
      return op->emitOpError() << message;
  }
  return mlir::success();
}

static mlir::LogicalResult
verifyInstructionReduceContract(mlir::Operation *op, mlir::Value input,
                                mlir::Value dest, mlir::IntegerAttr dimAttr) {
  std::optional<mlir::RankedTensorType> inputTensor =
      getLogicalTensorType(input.getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(dest.getType());
  if (!inputTensor)
    return op->emitOpError("expects Wafer buffer input");
  if (!destTensor)
    return op->emitOpError("expects Wafer buffer dest");

  if (!hasWaferMemorySpace(input.getType(), MemorySpace::SPM) ||
      !hasWaferMemorySpace(dest.getType(), MemorySpace::SPM))
    return op->emitOpError("reduce operands/dest must use SPM memory space");

  if (inputTensor->getRank() != 4 || destTensor->getRank() != 4)
    return op->emitOpError("reduce operands/dest must use rank4 NHWC geometry");
  if (!hasWaferLayout(input.getType(), MemLayout::NCx) ||
      !hasWaferLayout(dest.getType(), MemLayout::NCx))
    return op->emitOpError("reduce operands/dest must use ncx layout");

  if (inputTensor->getElementType() != destTensor->getElementType())
    return op->emitOpError(
        "reduce input element type must match dest element type");

  if (!dimAttr)
    return op->emitOpError("requires target reduce dim attr");
  int64_t targetDim = dimAttr.getInt();
  if (targetDim < 0 || targetDim > 5)
    return op->emitOpError("reduce dim must be a target code in [0, 5]");

  int64_t rank = inputTensor->getRank();
  llvm::SmallVector<int64_t, 3> dims =
      getInstrReduceLogicalDims(targetDim, rank);

  if (dims.empty())
    return op->emitOpError("reduce dim is not valid for input rank");

  llvm::DenseSet<int64_t> reducedDims;
  for (int64_t dim : dims) {
    if (!reducedDims.insert(dim).second)
      return op->emitOpError("reduce dim maps to duplicate logical dims");
  }

  if (targetDim == 3 || targetDim == 5)
    return op->emitOpError(
        "reduce N/HWC axes have no supported target contract");
  if (destTensor->getRank() != inputTensor->getRank())
    return op->emitOpError("reduce dest must retain input rank");
  for (int64_t inputDim = 0; inputDim < rank; ++inputDim) {
    int64_t expected =
        reducedDims.contains(inputDim) ? 1 : inputTensor->getDimSize(inputDim);
    if (hasStaticMismatch(expected, destTensor->getDimSize(inputDim)))
      return op->emitOpError(
          "reduce dest must retain non-reduced dimensions and set reduced "
          "dimensions to one");
  }

  return mlir::success();
}

static mlir::Type getMemRefElementType(mlir::Type type) {
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type))
    return memrefType.getElementType();
  return {};
}

static std::optional<ComputeElementwiseKind>
toComputeElementwiseKind(InstrElementwiseKind kind) {
  switch (kind) {
  case InstrElementwiseKind::Add:
    return ComputeElementwiseKind::Add;
  case InstrElementwiseKind::Sub:
    return ComputeElementwiseKind::Sub;
  case InstrElementwiseKind::Mul:
    return ComputeElementwiseKind::Mul;
  case InstrElementwiseKind::Max:
    return ComputeElementwiseKind::Max;
  case InstrElementwiseKind::Min:
    return ComputeElementwiseKind::Min;
  case InstrElementwiseKind::Neg:
    return ComputeElementwiseKind::Neg;
  case InstrElementwiseKind::Recip:
    return ComputeElementwiseKind::Recip;
  case InstrElementwiseKind::Square:
    return ComputeElementwiseKind::Square;
  case InstrElementwiseKind::Sqrt:
    return ComputeElementwiseKind::Sqrt;
  case InstrElementwiseKind::Rsqrt:
    return ComputeElementwiseKind::Rsqrt;
  case InstrElementwiseKind::Exp:
    return ComputeElementwiseKind::Exp;
  case InstrElementwiseKind::Ln:
    return ComputeElementwiseKind::Ln;
  case InstrElementwiseKind::Tanh:
    return ComputeElementwiseKind::Tanh;
  case InstrElementwiseKind::Sin:
    return ComputeElementwiseKind::Sin;
  case InstrElementwiseKind::Cos:
    return ComputeElementwiseKind::Cos;
  case InstrElementwiseKind::Eq:
    return ComputeElementwiseKind::Eq;
  case InstrElementwiseKind::Ne:
    return ComputeElementwiseKind::Ne;
  case InstrElementwiseKind::Lt:
    return ComputeElementwiseKind::Lt;
  case InstrElementwiseKind::Le:
    return ComputeElementwiseKind::Le;
  case InstrElementwiseKind::Gt:
    return ComputeElementwiseKind::Gt;
  case InstrElementwiseKind::Ge:
    return ComputeElementwiseKind::Ge;
  case InstrElementwiseKind::Abs:
  case InstrElementwiseKind::Log2:
  case InstrElementwiseKind::Pow2:
  case InstrElementwiseKind::ExpLp:
  case InstrElementwiseKind::Sigmoid:
  case InstrElementwiseKind::Relu:
  case InstrElementwiseKind::SatRelu:
  case InstrElementwiseKind::LeakyRelu:
  case InstrElementwiseKind::Softplus:
  case InstrElementwiseKind::LogicNot:
  case InstrElementwiseKind::LogicAnd:
  case InstrElementwiseKind::LogicOr:
  case InstrElementwiseKind::LogicXor:
    return std::nullopt;
  }
  llvm_unreachable("unknown instruction elementwise kind");
}

static bool isInstrRelationKind(InstrElementwiseKind kind) {
  switch (kind) {
  case InstrElementwiseKind::Eq:
  case InstrElementwiseKind::Ne:
  case InstrElementwiseKind::Lt:
  case InstrElementwiseKind::Le:
  case InstrElementwiseKind::Gt:
  case InstrElementwiseKind::Ge:
    return true;
  default:
    return false;
  }
}

static unsigned getInstrElementwiseArity(InstrElementwiseKind kind) {
  switch (kind) {
  case InstrElementwiseKind::Abs:
  case InstrElementwiseKind::Recip:
  case InstrElementwiseKind::Square:
  case InstrElementwiseKind::Sqrt:
  case InstrElementwiseKind::Rsqrt:
  case InstrElementwiseKind::Neg:
  case InstrElementwiseKind::LogicNot:
  case InstrElementwiseKind::Log2:
  case InstrElementwiseKind::Ln:
  case InstrElementwiseKind::Pow2:
  case InstrElementwiseKind::Exp:
  case InstrElementwiseKind::ExpLp:
  case InstrElementwiseKind::Sin:
  case InstrElementwiseKind::Cos:
  case InstrElementwiseKind::Tanh:
  case InstrElementwiseKind::Sigmoid:
  case InstrElementwiseKind::Relu:
  case InstrElementwiseKind::SatRelu:
  case InstrElementwiseKind::LeakyRelu:
  case InstrElementwiseKind::Softplus:
    return 1;
  case InstrElementwiseKind::Max:
  case InstrElementwiseKind::Min:
  case InstrElementwiseKind::Add:
  case InstrElementwiseKind::Sub:
  case InstrElementwiseKind::Mul:
  case InstrElementwiseKind::Eq:
  case InstrElementwiseKind::Ne:
  case InstrElementwiseKind::Ge:
  case InstrElementwiseKind::Gt:
  case InstrElementwiseKind::Le:
  case InstrElementwiseKind::Lt:
  case InstrElementwiseKind::LogicAnd:
  case InstrElementwiseKind::LogicOr:
  case InstrElementwiseKind::LogicXor:
    return 2;
  }
  llvm_unreachable("unknown instruction elementwise kind");
}

static mlir::LogicalResult verifySimpleInstrElementwiseContract(
    mlir::Operation *op, InstrElementwiseKind kind, mlir::ValueRange inputs,
    mlir::Type destType) {
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(destType);
  if (!destTensor)
    return op->emitOpError("expects Wafer buffer dest");
  std::optional<MemLayout> destLayout = getWaferLayout(destType);
  if (!destLayout)
    return op->emitOpError("elementwise dest must carry Wafer layout");

  unsigned expectedArity = getInstrElementwiseArity(kind);
  if (inputs.size() != expectedArity)
    return op->emitOpError("elementwise kind expects ")
           << expectedArity << " operand(s), got " << inputs.size();

  std::optional<mlir::RankedTensorType> firstInputTensor;
  for (mlir::Value input : inputs) {
    std::optional<mlir::RankedTensorType> inputTensor =
        getLogicalTensorType(input.getType());
    if (!inputTensor)
      return op->emitOpError("expects Wafer buffer operands");
    if (getWaferLayout(input.getType()) != destLayout)
      return op->emitOpError(
          "elementwise operands must use the dest layout family");
    if (!firstInputTensor)
      firstInputTensor = inputTensor;
    if (mlir::failed(verifySameShape(
            op, *inputTensor, *destTensor,
            "elementwise operand shapes must match dest shape")))
      return mlir::failure();
    if (isInstrRelationKind(kind)) {
      if (!destTensor->getElementType().isInteger(1))
        return op->emitOpError("relation dest element type must be i1");
      if (inputTensor->getElementType() != firstInputTensor->getElementType())
        return op->emitOpError(
            "relation operand element types must match each other");
      continue;
    }
    if (mlir::failed(verifySameElementType(
            op, *inputTensor, *destTensor,
            "elementwise operand element types must match dest element type")))
      return mlir::failure();
  }
  return mlir::success();
}

enum class ConvertTypeTag { Int8, Int16, Int32, Bf16, Fp16, Fp32, Tf32 };

static mlir::Type getConvertType(mlir::MLIRContext *context,
                                 ConvertTypeTag tag) {
  switch (tag) {
  case ConvertTypeTag::Int8:
    return mlir::IntegerType::get(context, 8);
  case ConvertTypeTag::Int16:
    return mlir::IntegerType::get(context, 16);
  case ConvertTypeTag::Int32:
    return mlir::IntegerType::get(context, 32);
  case ConvertTypeTag::Bf16:
    return mlir::BFloat16Type::get(context);
  case ConvertTypeTag::Fp16:
    return mlir::Float16Type::get(context);
  case ConvertTypeTag::Fp32:
    return mlir::Float32Type::get(context);
  case ConvertTypeTag::Tf32:
    return mlir::FloatTF32Type::get(context);
  }
  llvm_unreachable("unknown convert type tag");
}

static std::pair<mlir::Type, mlir::Type>
getInstrConvertTypePairImpl(mlir::MLIRContext *context, InstrConvertKind kind) {
  auto i8 = getConvertType(context, ConvertTypeTag::Int8);
  auto i16 = getConvertType(context, ConvertTypeTag::Int16);
  auto i32 = getConvertType(context, ConvertTypeTag::Int32);
  auto bf16 = getConvertType(context, ConvertTypeTag::Bf16);
  auto fp16 = getConvertType(context, ConvertTypeTag::Fp16);
  auto fp32 = getConvertType(context, ConvertTypeTag::Fp32);
  auto tf32 = getConvertType(context, ConvertTypeTag::Tf32);

  switch (kind) {
  case InstrConvertKind::Int8Fp16:
    return {i8, fp16};
  case InstrConvertKind::Int8Bf16:
    return {i8, bf16};
  case InstrConvertKind::Int8Fp32:
    return {i8, fp32};
  case InstrConvertKind::Int8Tf32:
    return {i8, tf32};
  case InstrConvertKind::Int16Fp16:
    return {i16, fp16};
  case InstrConvertKind::Int16Bf16:
    return {i16, bf16};
  case InstrConvertKind::Int16Fp32:
    return {i16, fp32};
  case InstrConvertKind::Int16Tf32:
    return {i16, tf32};
  case InstrConvertKind::Int32Fp16:
    return {i32, fp16};
  case InstrConvertKind::Int32Bf16:
    return {i32, bf16};
  case InstrConvertKind::Int32Fp32:
    return {i32, fp32};
  case InstrConvertKind::Int32Tf32:
    return {i32, tf32};
  case InstrConvertKind::Bf16Int8:
    return {bf16, i8};
  case InstrConvertKind::Bf16Int16:
    return {bf16, i16};
  case InstrConvertKind::Bf16Int32:
    return {bf16, i32};
  case InstrConvertKind::Bf16Fp16:
    return {bf16, fp16};
  case InstrConvertKind::Bf16Fp32:
    return {bf16, fp32};
  case InstrConvertKind::Bf16Tf32:
    return {bf16, tf32};
  case InstrConvertKind::Fp16Int8:
    return {fp16, i8};
  case InstrConvertKind::Fp16Int16:
    return {fp16, i16};
  case InstrConvertKind::Fp16Int32:
    return {fp16, i32};
  case InstrConvertKind::Fp16Bf16:
    return {fp16, bf16};
  case InstrConvertKind::Fp16Fp32:
    return {fp16, fp32};
  case InstrConvertKind::Fp16Tf32:
    return {fp16, tf32};
  case InstrConvertKind::Fp32Int8:
    return {fp32, i8};
  case InstrConvertKind::Fp32Int16:
    return {fp32, i16};
  case InstrConvertKind::Fp32Int32:
    return {fp32, i32};
  case InstrConvertKind::Fp32Fp16:
    return {fp32, fp16};
  case InstrConvertKind::Fp32Bf16:
    return {fp32, bf16};
  case InstrConvertKind::Fp32Tf32:
    return {fp32, tf32};
  case InstrConvertKind::Tf32Int8:
    return {tf32, i8};
  case InstrConvertKind::Tf32Int16:
    return {tf32, i16};
  case InstrConvertKind::Tf32Int32:
    return {tf32, i32};
  case InstrConvertKind::Tf32Fp16:
    return {tf32, fp16};
  case InstrConvertKind::Tf32Bf16:
    return {tf32, bf16};
  case InstrConvertKind::Tf32Fp32:
    return {tf32, fp32};
  }
  llvm_unreachable("unknown instruction convert kind");
}

static mlir::LogicalResult verifyNoEmptyVariadicInputs(mlir::Operation *op,
                                                       mlir::ValueRange inputs,
                                                       llvm::StringRef name) {
  if (inputs.empty())
    return op->emitOpError() << name << " requires at least one input";
  return mlir::success();
}

} // namespace

llvm::SmallVector<int64_t, 3>
wafer::getInstrReduceLogicalDims(int64_t targetDim, int64_t inputRank) {
  auto fromTrailingDim = [&](int64_t trailingIndex) -> std::optional<int64_t> {
    if (trailingIndex < 0 || trailingIndex >= inputRank)
      return std::nullopt;
    return inputRank - 1 - trailingIndex;
  };

  llvm::SmallVector<int64_t, 3> dims;
  switch (targetDim) {
  case 0:
    if (auto dim = fromTrailingDim(0))
      dims.push_back(*dim);
    break;
  case 1:
    if (auto dim = fromTrailingDim(1))
      dims.push_back(*dim);
    break;
  case 2:
    if (auto dim = fromTrailingDim(2))
      dims.push_back(*dim);
    break;
  case 3:
    if (auto dim = fromTrailingDim(3))
      dims.push_back(*dim);
    break;
  case 4: {
    auto h = fromTrailingDim(2);
    auto w = fromTrailingDim(1);
    if (h && w)
      dims.append({*h, *w});
    break;
  }
  case 5: {
    auto h = fromTrailingDim(2);
    auto w = fromTrailingDim(1);
    auto c = fromTrailingDim(0);
    if (h && w && c)
      dims.append({*h, *w, *c});
    break;
  }
  default:
    break;
  }
  return dims;
}

std::pair<mlir::Type, mlir::Type>
wafer::getInstrConvertTypePair(mlir::MLIRContext *context,
                               InstrConvertKind kind) {
  return getInstrConvertTypePairImpl(context, kind);
}

InstrConvertParameterKind
wafer::getInstrConvertParameterKind(InstrConvertKind kind) {
  switch (kind) {
  case InstrConvertKind::Int8Fp16:
  case InstrConvertKind::Int8Bf16:
  case InstrConvertKind::Int8Fp32:
  case InstrConvertKind::Int8Tf32:
    return InstrConvertParameterKind::ZeroPoint;
  case InstrConvertKind::Int16Bf16:
  case InstrConvertKind::Int16Fp32:
  case InstrConvertKind::Int16Tf32:
  case InstrConvertKind::Int32Fp16:
  case InstrConvertKind::Int32Bf16:
  case InstrConvertKind::Int32Fp32:
  case InstrConvertKind::Int32Tf32:
  case InstrConvertKind::Bf16Int16:
  case InstrConvertKind::Bf16Int32:
  case InstrConvertKind::Fp16Int8:
  case InstrConvertKind::Fp16Int16:
  case InstrConvertKind::Fp16Int32:
  case InstrConvertKind::Fp16Bf16:
  case InstrConvertKind::Fp32Int8:
  case InstrConvertKind::Fp32Int16:
  case InstrConvertKind::Fp32Int32:
  case InstrConvertKind::Fp32Fp16:
  case InstrConvertKind::Fp32Bf16:
  case InstrConvertKind::Fp32Tf32:
  case InstrConvertKind::Tf32Int8:
  case InstrConvertKind::Tf32Int16:
  case InstrConvertKind::Tf32Int32:
  case InstrConvertKind::Tf32Bf16:
    return InstrConvertParameterKind::RoundingMode;
  case InstrConvertKind::Int16Fp16:
  case InstrConvertKind::Bf16Int8:
  case InstrConvertKind::Bf16Fp16:
  case InstrConvertKind::Bf16Fp32:
  case InstrConvertKind::Bf16Tf32:
  case InstrConvertKind::Fp16Fp32:
  case InstrConvertKind::Fp16Tf32:
  case InstrConvertKind::Tf32Fp16:
  case InstrConvertKind::Tf32Fp32:
    return InstrConvertParameterKind::None;
  }
  llvm_unreachable("unknown instruction convert kind");
}

mlir::LogicalResult InstrFillOp::verify() {
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  if (getValue().getType() != destTensor->getElementType())
    return emitOpError("fill value type must match dest element type");
  unsigned scalarWidth = 0;
  if (auto integerType =
          mlir::dyn_cast<mlir::IntegerType>(destTensor->getElementType()))
    scalarWidth = integerType.getWidth();
  else if (auto floatType =
               mlir::dyn_cast<mlir::FloatType>(destTensor->getElementType()))
    scalarWidth = floatType.getWidth();
  else
    return emitOpError(
        "target_abi_narrowing: fill value type must be a target-encodable "
        "integer or float");
  if (scalarWidth > 32)
    return emitOpError(
        "target_abi_narrowing: fill value type exceeds the uint32_t target "
        "scalar ABI");
  FillDomain domain = getFillDomain().value_or(FillDomain::LogicalValid);
  if (domain == FillDomain::LogicalValid) {
    if (!hasWaferLayout(getDest().getType(), MemLayout::Tensor))
      return emitOpError(
          "logical_valid fill destination must use tensor layout");
    if (!mlir::memref::isStaticShapeAndContiguousRowMajor(
            mlir::cast<mlir::MemRefType>(getDest().getType())))
      return emitOpError(
          "logical_valid fill requires a static contiguous row-major view; "
          "strided fill must be decomposed before Instr");
    return verifyStaticElementCountFitsUInt32(getOperation(),
                                              getDest().getType(), "fill dest");
  }

  auto destType = mlir::cast<mlir::MemRefType>(getDest().getType());
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(destType);
  if (!info || info->physicalBytes <= 0 || info->physicalElements <= 0)
    return emitOpError(
        "physical_footprint fill requires a static positive physical "
        "destination footprint");
  uint64_t elements = static_cast<uint64_t>(info->physicalElements);
  if (info->bitPackedElement) {
    if (static_cast<uint64_t>(info->physicalBytes) >
        std::numeric_limits<uint32_t>::max() / UINT64_C(8))
      return emitOpError(
          "target_abi_narrowing: bitpacked physical fill element count must "
          "fit uint32_t");
    elements = static_cast<uint64_t>(info->physicalBytes) * UINT64_C(8);
  }
  if (elements > std::numeric_limits<uint32_t>::max())
    return emitOpError(
        "target_abi_narrowing: physical fill element count must fit uint32_t");
  return mlir::success();
}

InstrFamily InstrFillOp::getInstructionFamily() { return InstrFamily::TDMA; }

mlir::LogicalResult InstrElementwiseOp::verify() {
  if (mlir::failed(verifyNoEmptyVariadicInputs(getOperation(), getInputs(),
                                               "elementwise")))
    return mlir::failure();
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    if (mlir::failed(verifySPMMemRef(getOperation(), input.getType(), "input")))
      return mlir::failure();
  }
  std::optional<ComputeElementwiseKind> computeKind =
      toComputeElementwiseKind(getKindAttr().getValue());
  mlir::LogicalResult contract =
      computeKind
          ? verifyElementwiseTileContract(getOperation(), *computeKind,
                                          getInputs(), getDest().getType())
          : verifySimpleInstrElementwiseContract(
                getOperation(), getKindAttr().getValue(), getInputs(),
                getDest().getType());
  if (mlir::failed(contract))
    return mlir::failure();
  return verifyStaticPhysicalElementCountFitsUInt32(
      getOperation(), getDest().getType(), "elementwise dest");
}

InstrFamily InstrElementwiseOp::getInstructionFamily() {
  return InstrFamily::CT;
}

mlir::LogicalResult InstrBit2FpOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();

  std::optional<mlir::RankedTensorType> sourceTensor =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (!sourceTensor || !destTensor)
    return emitOpError("bit2fp expects ranked Wafer memrefs");
  if (!sourceTensor->getElementType().isInteger(1))
    return emitOpError("bit2fp source element type must be i1");
  if (!mlir::isa<mlir::FloatType>(destTensor->getElementType()))
    return emitOpError("bit2fp dest element type must be floating point");
  if (getWaferLayout(getSource().getType()) !=
      getWaferLayout(getDest().getType()))
    return emitOpError("bit2fp source and dest layout families must match");
  if (mlir::failed(verifySameShape(getOperation(), *sourceTensor, *destTensor,
                                   "bit2fp source and dest shapes must match")))
    return mlir::failure();
  return verifyStaticPhysicalElementCountFitsUInt32(
      getOperation(), getDest().getType(), "bit2fp dest");
}

InstrFamily InstrBit2FpOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrMaskMoveOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getMask().getType(), "mask")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();

  std::optional<mlir::RankedTensorType> sourceTensor =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> maskTensor =
      getLogicalTensorType(getMask().getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (!sourceTensor || !maskTensor || !destTensor)
    return emitOpError("mask_move expects ranked Wafer memrefs");
  std::optional<MemLayout> destLayout = getWaferLayout(getDest().getType());
  if (!destLayout || getWaferLayout(getSource().getType()) != destLayout ||
      getWaferLayout(getMask().getType()) != destLayout)
    return emitOpError(
        "mask_move source, mask and dest layout families must match");
  if (mlir::failed(verifySameShape(getOperation(), *sourceTensor, *destTensor,
                                   "mask_move source and dest shapes must "
                                   "match")) ||
      mlir::failed(verifySameElementType(
          getOperation(), *sourceTensor, *destTensor,
          "mask_move source and dest element types must match")) ||
      mlir::failed(
          verifySameShape(getOperation(), *maskTensor, *destTensor,
                          "mask_move mask and dest shapes must match")))
    return mlir::failure();
  if (maskTensor->getElementType() != sourceTensor->getElementType())
    return emitOpError(
        "mask_move mask element type must match source element type after "
        "bit2fp conversion");
  if (!mlir::isa<mlir::FloatType>(maskTensor->getElementType()))
    return emitOpError("mask_move mask element type must be floating point");
  return verifyStaticPhysicalElementCountFitsUInt32(
      getOperation(), getDest().getType(), "mask_move dest");
}

InstrFamily InstrMaskMoveOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrReduceOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getInput().getType(), "input")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  if (mlir::failed(verifyInstructionReduceContract(getOperation(), getInput(),
                                                   getDest(), getDimAttr())))
    return mlir::failure();
  return verifyStaticDataShapeBounds(getOperation(), getInput().getType(),
                                     "reduce input shape");
}

InstrFamily InstrReduceOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrConvertOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  if (getWaferLayout(getSource().getType()) !=
      getWaferLayout(getDest().getType()))
    return emitOpError("convert source and dest layout families must match");
  mlir::Type srcElement = getMemRefElementType(getSource().getType());
  mlir::Type dstElement = getMemRefElementType(getDest().getType());
  auto [expectedSrc, expectedDst] =
      getInstrConvertTypePair(getContext(), getKindAttr().getValue());
  if (srcElement != expectedSrc)
    return emitOpError("convert kind source type does not match source "
                       "element type");
  if (dstElement != expectedDst)
    return emitOpError(
        "convert kind destination type does not match dest element type");
  InstrConvertKind kind = getKindAttr().getValue();
  mlir::IntegerAttr zeroPoint = getZeroPointAttr();
  mlir::IntegerAttr roundingMode = getRoundingModeAttr();
  switch (getInstrConvertParameterKind(kind)) {
  case InstrConvertParameterKind::ZeroPoint:
    if (!zeroPoint)
      return emitOpError("convert kind requires zero_point attr");
    if (roundingMode)
      return emitOpError("zero-point convert must not have rounding_mode attr");
    break;
  case InstrConvertParameterKind::RoundingMode:
    if (zeroPoint)
      return emitOpError("convert kind must not have zero_point attr");
    if (!roundingMode)
      return emitOpError("convert kind requires rounding_mode attr");
    break;
  case InstrConvertParameterKind::None:
    if (zeroPoint)
      return emitOpError("convert kind must not have zero_point attr");
    if (roundingMode)
      return emitOpError("convert kind must not have rounding_mode attr");
    break;
  }
  if (mlir::failed(verifyStaticElementCountEqual(
          getOperation(), getSource().getType(), getDest().getType(),
          "convert source and destination element counts must match")) ||
      mlir::failed(
          verifyOptionalUInt32Attr(getOperation(), zeroPoint, "zero_point")) ||
      mlir::failed(verifyOptionalRoundingMode(getOperation(), roundingMode,
                                              "rounding_mode")) ||
      mlir::failed(verifyStaticPhysicalElementCountFitsUInt32(
          getOperation(), getDest().getType(), "convert dest")))
    return mlir::failure();
  return mlir::success();
}

InstrFamily InstrConvertOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrGemmOp::verify() {
  if (auto partial = getPsum()) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(partial.getType());
    auto output = mlir::dyn_cast<mlir::MemRefType>(getDest().getType());
    if (!type || !output || !type.getElementType().isF32() ||
        type.getShape() != output.getShape() ||
        type.getLayout() != output.getLayout() ||
        type.getMemorySpace() != output.getMemorySpace())
      return emitOpError("psum must be F32 with destination shape and layout");
  }
  if (mlir::failed(verifyAlignedSPMGemmMemRef(getOperation(),
                                              getLhs().getType(), "lhs")) ||
      mlir::failed(verifyAlignedSPMGemmMemRef(getOperation(),
                                              getRhs().getType(), "rhs")) ||
      mlir::failed(verifyAlignedSPMGemmMemRef(getOperation(),
                                              getDest().getType(), "dest")))
    return mlir::failure();

  std::optional<mlir::RankedTensorType> lhsTensor =
      getLogicalTensorType(getLhs().getType());
  std::optional<mlir::RankedTensorType> rhsTensor =
      getLogicalTensorType(getRhs().getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (mlir::failed(
          verifySameElementType(getOperation(), *lhsTensor, *rhsTensor,
                                "gemm lhs and rhs element types must match")))
    return mlir::failure();
  if (lhsTensor->getElementType() != destTensor->getElementType() &&
      !((lhsTensor->getElementType().isF16() ||
         lhsTensor->getElementType().isBF16()) &&
        destTensor->getElementType().isF32()))
    return emitOpError("gemm requires equal input types and the same output "
                       "type or f16/bf16 inputs with f32 output");

  if (static_cast<bool>(getLhsOrientationAttr()) !=
      static_cast<bool>(getRhsOrientationAttr()))
    return emitOpError(
        "lhs_orientation and rhs_orientation must either both be present for "
        "oriented GEMM or both be absent for normal/normal GEMM");
  GemmOrientation lhsOrientation =
      getLhsOrientation().value_or(GemmOrientation::Normal);
  GemmOrientation rhsOrientation =
      getRhsOrientation().value_or(GemmOrientation::Normal);

  int64_t m = getMAttr().getInt();
  int64_t k = getKAttr().getInt();
  int64_t n = getNAttr().getInt();
  if (m <= 0 || k <= 0 || n <= 0)
    return emitOpError("m, k and n must be positive");
  if (mlir::failed(verifyUInt16Value(getOperation(), m, "m")) ||
      mlir::failed(verifyGemmKBounds(getOperation(), k)) ||
      mlir::failed(verifyUInt16Value(getOperation(), n, "n")))
    return mlir::failure();

  if (lhsTensor->getRank() == 2 && rhsTensor->getRank() == 2 &&
      destTensor->getRank() == 2) {
    if (hasAnyBatchedGemmAttrs(getOperation()))
      return emitOpError("rank-2 gemm must not carry batched GEMM attrs");
    int64_t lhsMDim = lhsOrientation == GemmOrientation::Normal ? 0 : 1;
    int64_t lhsKDim = lhsOrientation == GemmOrientation::Normal ? 1 : 0;
    int64_t rhsKDim = rhsOrientation == GemmOrientation::Normal ? 0 : 1;
    int64_t rhsNDim = rhsOrientation == GemmOrientation::Normal ? 1 : 0;
    if (hasStaticMismatch(lhsTensor->getDimSize(lhsMDim), m) ||
        hasStaticMismatch(lhsTensor->getDimSize(lhsKDim), k) ||
        hasStaticMismatch(rhsTensor->getDimSize(rhsKDim), k) ||
        hasStaticMismatch(rhsTensor->getDimSize(rhsNDim), n) ||
        hasStaticMismatch(destTensor->getDimSize(0), m) ||
        hasStaticMismatch(destTensor->getDimSize(1), n))
      return emitOpError("m/k/n attrs must match GEMM operand shapes");
    return mlir::success();
  }

  BatchedGemmDimAttrs attrs;
  if (mlir::failed(verifyBatchedGemmTileContract(
          getOperation(), *lhsTensor, *rhsTensor, *destTensor, lhsOrientation,
          rhsOrientation, attrs)))
    return mlir::failure();
  if (hasStaticMismatch(attrs.lhsMDim >= 0
                            ? lhsTensor->getDimSize(attrs.lhsMDim)
                            : mlir::ShapedType::kDynamic,
                        m) ||
      hasStaticMismatch(attrs.lhsContractingDim >= 0
                            ? lhsTensor->getDimSize(attrs.lhsContractingDim)
                            : mlir::ShapedType::kDynamic,
                        k) ||
      hasStaticMismatch(attrs.rhsNDim >= 0
                            ? rhsTensor->getDimSize(attrs.rhsNDim)
                            : mlir::ShapedType::kDynamic,
                        n))
    return emitOpError("m/k/n attrs must match batched GEMM dimensions");
  if (mlir::failed(verifyGemmBatchBounds(getOperation(), attrs.batchCount)))
    return mlir::failure();
  for (size_t index = 0; index < attrs.lhsBatchDims.size(); ++index) {
    int64_t expected = static_cast<int64_t>(index);
    if (attrs.lhsBatchDims[index] != expected ||
        attrs.rhsBatchDims[index] != expected ||
        attrs.resultBatchDims[index] != expected)
      return emitOpError(
          "target_geometry_mismatch: target GEMM ABI requires leading "
          "canonical batch dimensions");
  }
  int64_t matrixRankBase = lhsTensor->getRank() - 2;
  int64_t expectedLhsMDim = lhsOrientation == GemmOrientation::Normal
                                ? matrixRankBase
                                : matrixRankBase + 1;
  int64_t expectedLhsKDim = lhsOrientation == GemmOrientation::Normal
                                ? matrixRankBase + 1
                                : matrixRankBase;
  int64_t expectedRhsKDim = rhsOrientation == GemmOrientation::Normal
                                ? matrixRankBase
                                : matrixRankBase + 1;
  int64_t expectedRhsNDim = rhsOrientation == GemmOrientation::Normal
                                ? matrixRankBase + 1
                                : matrixRankBase;
  if (attrs.lhsMDim != expectedLhsMDim ||
      attrs.lhsContractingDim != expectedLhsKDim ||
      attrs.rhsContractingDim != expectedRhsKDim ||
      attrs.rhsNDim != expectedRhsNDim || attrs.resultMDim != matrixRankBase ||
      attrs.resultNDim != matrixRankBase + 1)
    return emitOpError(
        "target_geometry_mismatch: target GEMM ABI requires canonical trailing "
        "M/K/N dimensions");
  return mlir::success();
}

InstrFamily InstrGemmOp::getInstructionFamily() { return InstrFamily::NE; }

mlir::LogicalResult InstrConvOp::verify() {
  if (mlir::failed(verifyAlignedSPMMemRef(getOperation(), getInput().getType(),
                                          "input")) ||
      mlir::failed(verifyAlignedSPMMemRef(getOperation(), getWeight().getType(),
                                          "weight")) ||
      mlir::failed(
          verifyAlignedSPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();

  if (!hasWaferLayout(getInput().getType(), MemLayout::NCx) ||
      !hasWaferLayout(getDest().getType(), MemLayout::NCx) ||
      !hasWaferLayout(getWeight().getType(), MemLayout::Cx))
    return emitOpError(
        "convolution input/dest must use ncx and weight must use cx layout");

  std::optional<mlir::RankedTensorType> inputTensor =
      getLogicalTensorType(getInput().getType());
  std::optional<mlir::RankedTensorType> weightTensor =
      getLogicalTensorType(getWeight().getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (inputTensor->getElementType() != destTensor->getElementType() &&
      !((inputTensor->getElementType().isF16() ||
         inputTensor->getElementType().isBF16()) &&
        destTensor->getElementType().isF32()))
    return emitOpError(
        "conv dest element type must match input or widen f16/bf16 to f32");
  if (mlir::failed(verifySameElementType(
          getOperation(), *inputTensor, *weightTensor,
          "conv input and weight element types must match")) ||
      mlir::failed(verifyI64Array(getOperation(), getInputShapeAttr(),
                                  "input_shape", 4, /*positive=*/true)) ||
      mlir::failed(verifyI64Array(getOperation(), getWeightShapeAttr(),
                                  "weight_shape", 4, /*positive=*/true)) ||
      mlir::failed(verifyI64Array(getOperation(), getOutputShapeAttr(),
                                  "output_shape", 4, /*positive=*/true)) ||
      mlir::failed(verifyI64Array(getOperation(), getPadsAttr(), "pads", 4,
                                  /*positive=*/false)) ||
      mlir::failed(verifyI64Array(getOperation(), getUnpadsAttr(), "unpads", 4,
                                  /*positive=*/false)) ||
      mlir::failed(verifyI64Array(getOperation(), getKernelStridesAttr(),
                                  "kernel_strides", 4,
                                  /*positive=*/true)) ||
      mlir::failed(verifyI64Array(getOperation(), getDilationsAttr(),
                                  "dilations", 2, /*positive=*/true)))
    return mlir::failure();
  if (mlir::failed(verifyDataShapeAttrMatchesBuffer(
          getOperation(), getInput().getType(), getInputShapeAttr(),
          "input_shape")) ||
      mlir::failed(
          verifyShapeAttrMatchesBuffer(getOperation(), getWeight().getType(),
                                       getWeightShapeAttr(), "weight_shape")) ||
      mlir::failed(verifyDataShapeAttrMatchesBuffer(
          getOperation(), getDest().getType(), getOutputShapeAttr(),
          "output_shape")) ||
      mlir::failed(
          verifyPaddingBounds(getOperation(), getPadsAttr(), "pads")) ||
      mlir::failed(
          verifyPaddingBounds(getOperation(), getUnpadsAttr(), "unpads")) ||
      mlir::failed(verifyKernelStrideBounds(
          getOperation(), getKernelStridesAttr(), "kernel_strides")) ||
      mlir::failed(verifyDilationBounds(getOperation(), getDilationsAttr(),
                                        "dilations")) ||
      mlir::failed(verifyConvShapeRelation(
          getOperation(), getKindAttr().getValue(), getInputShapeAttr(),
          getWeightShapeAttr(), getOutputShapeAttr(), getPadsAttr(),
          getUnpadsAttr(), getKernelStridesAttr(), getDilationsAttr())))
    return mlir::failure();
  return mlir::success();
}

InstrFamily InstrConvOp::getInstructionFamily() { return InstrFamily::NE; }

static bool isIndexedPoolKind(InstrPoolKind kind) {
  return kind == InstrPoolKind::IndexedMax || kind == InstrPoolKind::IndexedMin;
}

mlir::LogicalResult InstrPoolOp::verify() {
  if (mlir::failed(
          verifyNCxSPMMemRef(getOperation(), getInput().getType(), "input")) ||
      mlir::failed(verifyI64Array(getOperation(), getSourceShapeAttr(),
                                  "source_shape", 4, /*positive=*/true)) ||
      mlir::failed(verifyI64Array(getOperation(), getDestShapeAttr(),
                                  "dest_shape", 4, /*positive=*/true)) ||
      mlir::failed(verifyI64Array(getOperation(), getPadsAttr(), "pads", 4,
                                  /*positive=*/false)) ||
      mlir::failed(verifyI64Array(getOperation(), getKernelStridesAttr(),
                                  "kernel_strides", 4,
                                  /*positive=*/true)))
    return mlir::failure();

  bool indexed = isIndexedPoolKind(getKindAttr().getValue());
  size_t expectedDests = indexed ? 2 : 1;
  if (getDests().size() != expectedDests)
    return emitOpError() << "pool kind expects " << expectedDests
                         << " dest operand(s)";

  std::optional<mlir::RankedTensorType> inputTensor =
      getLogicalTensorType(getInput().getType());
  if (mlir::failed(verifyDataShapeAttrMatchesBuffer(
          getOperation(), getInput().getType(), getSourceShapeAttr(),
          "source_shape")) ||
      mlir::failed(
          verifyPaddingBounds(getOperation(), getPadsAttr(), "pads")) ||
      mlir::failed(verifyKernelStrideBounds(
          getOperation(), getKernelStridesAttr(), "kernel_strides")) ||
      mlir::failed(verifyPoolShapeRelation(getOperation(), getSourceShapeAttr(),
                                           getDestShapeAttr(), getPadsAttr(),
                                           getKernelStridesAttr())))
    return mlir::failure();
  for (auto [index, dest] : llvm::enumerate(getDests())) {
    if (mlir::failed(
            verifyNCxSPMMemRef(getOperation(), dest.getType(), "dest")))
      return mlir::failure();
    std::optional<mlir::RankedTensorType> destTensor =
        getLogicalTensorType(dest.getType());
    if (mlir::failed(verifyDataShapeAttrMatchesBuffer(
            getOperation(), dest.getType(), getDestShapeAttr(), "dest_shape")))
      return mlir::failure();
    if (index == 0) {
      if (mlir::failed(verifySameElementType(
              getOperation(), *inputTensor, *destTensor,
              "pool value dest element type must match input element type")))
        return mlir::failure();
      continue;
    }
    if (!destTensor->getElementType().isInteger(16))
      return emitOpError("indexed pool index dest element type must be i16");
  }
  return mlir::success();
}

InstrFamily InstrPoolOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrUnpoolOp::verify() {
  if (mlir::failed(
          verifyNCxSPMMemRef(getOperation(), getInput().getType(), "input")) ||
      mlir::failed(
          verifyNCxSPMMemRef(getOperation(), getDest().getType(), "dest")) ||
      mlir::failed(verifyI64Array(getOperation(), getSourceShapeAttr(),
                                  "source_shape", 4, /*positive=*/true)) ||
      mlir::failed(verifyI64Array(getOperation(), getDestShapeAttr(),
                                  "dest_shape", 4, /*positive=*/true)) ||
      mlir::failed(verifyI64Array(getOperation(), getKernelStridesAttr(),
                                  "kernel_strides", 4,
                                  /*positive=*/true)))
    return mlir::failure();

  std::optional<mlir::RankedTensorType> inputTensor =
      getLogicalTensorType(getInput().getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (mlir::failed(verifySameElementType(
          getOperation(), *inputTensor, *destTensor,
          "unpool input and dest element types must match")))
    return mlir::failure();
  if (mlir::failed(verifyDataShapeAttrMatchesBuffer(
          getOperation(), getInput().getType(), getSourceShapeAttr(),
          "source_shape")) ||
      mlir::failed(
          verifyDataShapeAttrMatchesBuffer(getOperation(), getDest().getType(),
                                           getDestShapeAttr(), "dest_shape")) ||
      mlir::failed(verifyKernelStrideBounds(
          getOperation(), getKernelStridesAttr(), "kernel_strides")) ||
      mlir::failed(verifyUnpoolShapeRelation(
          getOperation(), getSourceShapeAttr(), getDestShapeAttr(),
          getKernelStridesAttr())))
    return mlir::failure();
  mlir::Value index = getIndex();
  if (getKindAttr().getValue() == InstrUnpoolKind::Avg) {
    if (index)
      return emitOpError("avg unpool must not have index operand");
  } else {
    if (!index)
      return emitOpError("unpool kind requires index operand");
    if (mlir::failed(
            verifyNCxSPMMemRef(getOperation(), index.getType(), "index")))
      return mlir::failure();
    auto indexType = mlir::cast<mlir::MemRefType>(index.getType());
    if (!indexType.getElementType().isInteger(16))
      return emitOpError("index element type must be i16");
    if (mlir::failed(verifyDataShapeAttrMatchesBuffer(getOperation(), indexType,
                                                      getSourceShapeAttr(),
                                                      "index source_shape")))
      return mlir::failure();

    std::optional<WaferPhysicalTensorInfo> indexInfo =
        computeWaferPhysicalTensorInfo(indexType);
    if (!indexInfo || indexInfo->physicalElements <= 0 ||
        indexInfo->physicalElements < inputTensor->getNumElements())
      return emitOpError(
          "target_geometry_mismatch: index physical capacity must cover one "
          "i16 entry per source element");
  }
  return mlir::success();
}

InstrFamily InstrUnpoolOp::getInstructionFamily() { return InstrFamily::CT; }

#define WAFER_DEFINE_NCC_ISSUE_WORKER(OP)                                      \
  NCCWorker OP::getIssueWorker() { return getWorker(); }

WAFER_DEFINE_NCC_ISSUE_WORKER(InstrFillOp)
WAFER_DEFINE_NCC_ISSUE_WORKER(InstrElementwiseOp)
WAFER_DEFINE_NCC_ISSUE_WORKER(InstrBit2FpOp)
WAFER_DEFINE_NCC_ISSUE_WORKER(InstrMaskMoveOp)
WAFER_DEFINE_NCC_ISSUE_WORKER(InstrReduceOp)
WAFER_DEFINE_NCC_ISSUE_WORKER(InstrConvertOp)
WAFER_DEFINE_NCC_ISSUE_WORKER(InstrGemmOp)
WAFER_DEFINE_NCC_ISSUE_WORKER(InstrConvOp)
WAFER_DEFINE_NCC_ISSUE_WORKER(InstrPoolOp)
WAFER_DEFINE_NCC_ISSUE_WORKER(InstrUnpoolOp)

#undef WAFER_DEFINE_NCC_ISSUE_WORKER
