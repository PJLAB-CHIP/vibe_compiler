//===- InstructionOps.cpp - Wafer instruction verifier implementation ----===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <initializer_list>
#include <limits>
#include <optional>
#include <utility>

using namespace wafer;
using namespace wafer::detail;

namespace {

static mlir::LogicalResult verifyWaferMemRef(mlir::Operation *op,
                                             mlir::Type type,
                                             MemorySpace memorySpace,
                                             llvm::StringRef role) {
  if (!getLogicalTensorType(type))
    return op->emitOpError() << role << " must be a Wafer memref";
  if (!hasWaferMemorySpace(type, memorySpace)) {
    return op->emitOpError()
           << role << " must be a "
           << (memorySpace == MemorySpace::SPM ? "SPM" : "DDR")
           << " Wafer memref";
  }
  return mlir::success();
}

static mlir::LogicalResult verifySPMMemRef(mlir::Operation *op, mlir::Type type,
                                           llvm::StringRef role) {
  return verifyWaferMemRef(op, type, MemorySpace::SPM, role);
}

static mlir::LogicalResult verifyDDRMemRef(mlir::Operation *op, mlir::Type type,
                                           llvm::StringRef role) {
  return verifyWaferMemRef(op, type, MemorySpace::DDR, role);
}

static mlir::LogicalResult verifyPositiveI64Attr(mlir::Operation *op,
                                                 mlir::IntegerAttr attr,
                                                 llvm::StringRef name) {
  if (attr.getInt() <= 0)
    return op->emitOpError() << name << " must be positive";
  return mlir::success();
}

static mlir::LogicalResult
verifyNonNegativeOptionalI64Attr(mlir::Operation *op, mlir::IntegerAttr attr,
                                 llvm::StringRef name) {
  if (attr && attr.getInt() < 0)
    return op->emitOpError() << name << " must be non-negative";
  return mlir::success();
}

static mlir::LogicalResult verifyDescriptorArray(mlir::Operation *op,
                                                 mlir::DenseI64ArrayAttr attr,
                                                 llvm::StringRef name,
                                                 bool positive) {
  if (attr.size() != 3)
    return op->emitOpError() << name << " must contain exactly 3 entries";
  for (int64_t value : attr.asArrayRef()) {
    if (positive && value <= 0)
      return op->emitOpError() << name << " entries must be positive";
    if (!positive && value < 0)
      return op->emitOpError() << name << " entries must be non-negative";
  }
  return mlir::success();
}

static mlir::LogicalResult verifyI64Array(mlir::Operation *op,
                                          mlir::DenseI64ArrayAttr attr,
                                          llvm::StringRef name,
                                          int64_t expectedSize, bool positive) {
  if (attr.size() != expectedSize)
    return op->emitOpError()
           << name << " must contain exactly " << expectedSize << " entries";
  for (int64_t value : attr.asArrayRef()) {
    if (positive && value <= 0)
      return op->emitOpError() << name << " entries must be positive";
    if (!positive && value < 0)
      return op->emitOpError() << name << " entries must be non-negative";
  }
  return mlir::success();
}

static mlir::LogicalResult
verifyPermutationI64Array(mlir::Operation *op, mlir::DenseI64ArrayAttr attr,
                          llvm::StringRef name) {
  if (mlir::failed(verifyI64Array(op, attr, name, 4, /*positive=*/false)))
    return mlir::failure();
  bool seen[4] = {false, false, false, false};
  for (int64_t value : attr.asArrayRef()) {
    if (value >= 4)
      return op->emitOpError()
             << name << " entries must be in the range [0, 3]";
    if (seen[value])
      return op->emitOpError() << name << " entries must form a permutation";
    seen[value] = true;
  }
  return mlir::success();
}

static int64_t getOptionalI64AttrValue(mlir::IntegerAttr attr) {
  return attr ? attr.getInt() : 0;
}

static mlir::FailureOr<int64_t>
getMovementDescriptorEnd(mlir::Operation *op, int64_t offset,
                         int64_t innerBytes, mlir::DenseI64ArrayAttr strides,
                         mlir::DenseI64ArrayAttr iterations,
                         llvm::StringRef role) {
  int64_t end = 0;
  if (!checkedAdd(offset, innerBytes, end))
    return op->emitOpError()
           << role << " descriptor byte range overflows int64";

  if (!strides || !iterations)
    return end;

  for (auto [stride, iteration] :
       llvm::zip(strides.asArrayRef(), iterations.asArrayRef())) {
    int64_t span = 0;
    if (!checkedMul(stride, iteration - 1, span) || !checkedAdd(end, span, end))
      return op->emitOpError()
             << role << " descriptor byte range overflows int64";
  }
  return end;
}

static mlir::LogicalResult verifyDescriptorWithinPhysicalRange(
    mlir::Operation *op, mlir::Type type, mlir::IntegerAttr offsetAttr,
    mlir::IntegerAttr innerBytesAttr, mlir::DenseI64ArrayAttr strides,
    mlir::DenseI64ArrayAttr iterations, llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return mlir::success();

  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes < 0)
    return mlir::success();

  mlir::FailureOr<int64_t> end = getMovementDescriptorEnd(
      op, getOptionalI64AttrValue(offsetAttr), innerBytesAttr.getInt(), strides,
      iterations, role);
  if (mlir::failed(end))
    return mlir::failure();
  if (*end > info->physicalBytes)
    return op->emitOpError()
           << role << " descriptor byte range exceeds physical byte size";
  return mlir::success();
}

static mlir::LogicalResult verifyMovementDescriptor(
    mlir::Operation *op, mlir::IntegerAttr byteCount,
    mlir::IntegerAttr innerBytes, mlir::DenseI64ArrayAttr srcStrides,
    mlir::DenseI64ArrayAttr srcIterations, mlir::DenseI64ArrayAttr dstStrides,
    mlir::DenseI64ArrayAttr dstIterations) {
  if (mlir::failed(verifyPositiveI64Attr(op, byteCount, "byte_count")) ||
      mlir::failed(verifyPositiveI64Attr(op, innerBytes, "inner_bytes")))
    return mlir::failure();
  if (innerBytes.getInt() > byteCount.getInt())
    return op->emitOpError("inner_bytes must not exceed byte_count");
  if (srcStrides &&
      mlir::failed(verifyDescriptorArray(op, srcStrides, "src_strides",
                                         /*positive=*/false)))
    return mlir::failure();
  if (srcIterations &&
      mlir::failed(verifyDescriptorArray(op, srcIterations, "src_iterations",
                                         /*positive=*/true)))
    return mlir::failure();
  if (dstStrides &&
      mlir::failed(verifyDescriptorArray(op, dstStrides, "dst_strides",
                                         /*positive=*/false)))
    return mlir::failure();
  if (dstIterations &&
      mlir::failed(verifyDescriptorArray(op, dstIterations, "dst_iterations",
                                         /*positive=*/true)))
    return mlir::failure();
  return mlir::success();
}

static bool isAlignedGemmLayout(MemLayout layout) {
  return layout == MemLayout::Cx || layout == MemLayout::NCx;
}

static mlir::LogicalResult verifyAlignedSPMGemmMemRef(mlir::Operation *op,
                                                      mlir::Type type,
                                                      llvm::StringRef role) {
  if (mlir::failed(verifySPMMemRef(op, type, role)))
    return mlir::failure();
  std::optional<MemLayout> layout = getWaferLayout(type);
  if (!layout || !isAlignedGemmLayout(*layout))
    return op->emitOpError("lhs, rhs and dest must use aligned SPM layouts");
  return mlir::success();
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

static mlir::LogicalResult verifySameElementType(mlir::Operation *op,
                                                 mlir::RankedTensorType lhs,
                                                 mlir::RankedTensorType rhs,
                                                 llvm::StringRef message) {
  if (lhs.getElementType() != rhs.getElementType())
    return op->emitOpError() << message;
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
                                mlir::Value dest, mlir::Value init,
                                mlir::IntegerAttr dimAttr) {
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

  MemLayout expectedInputLayout =
      inputTensor->getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  MemLayout expectedDestLayout =
      destTensor->getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  if (!hasWaferLayout(input.getType(), expectedInputLayout) ||
      !hasWaferLayout(dest.getType(), expectedDestLayout)) {
    if (inputTensor->getRank() > 2 || destTensor->getRank() > 2)
      return op->emitOpError(
          "reduce rank > 2 operands/dest must use ncx layout");
    return op->emitOpError("reduce rank <= 2 operands/dest must use cx layout");
  }

  if (inputTensor->getElementType() != destTensor->getElementType())
    return op->emitOpError(
        "reduce input element type must match dest element type");

  if (!dimAttr)
    return op->emitOpError("requires target reduce dim attr");
  int64_t targetDim = dimAttr.getInt();
  if (targetDim < 0 || targetDim > 5)
    return op->emitOpError("reduce dim must be a target code in [0, 5]");

  int64_t rank = inputTensor->getRank();
  if (rank <= 0 || rank > 4)
    return op->emitOpError("reduce input rank must be in [1, 4]");

  auto fromTrailingDim = [&](int64_t trailingIndex) -> std::optional<int64_t> {
    if (trailingIndex >= rank)
      return std::nullopt;
    return rank - 1 - trailingIndex;
  };

  llvm::SmallVector<int64_t, 3> dims;
  switch (targetDim) {
  case 0: { // C
    if (auto dim = fromTrailingDim(0))
      dims.push_back(*dim);
    break;
  }
  case 1: { // W
    if (auto dim = fromTrailingDim(1))
      dims.push_back(*dim);
    break;
  }
  case 2: { // H
    if (auto dim = fromTrailingDim(2))
      dims.push_back(*dim);
    break;
  }
  case 3: { // N
    if (auto dim = fromTrailingDim(3))
      dims.push_back(*dim);
    break;
  }
  case 4: { // HW
    auto h = fromTrailingDim(2);
    auto w = fromTrailingDim(1);
    if (h && w)
      dims.append({*h, *w});
    break;
  }
  case 5: { // HWC
    auto h = fromTrailingDim(2);
    auto w = fromTrailingDim(1);
    auto c = fromTrailingDim(0);
    if (h && w && c)
      dims.append({*h, *w, *c});
    break;
  }
  }

  if (dims.empty())
    return op->emitOpError("reduce dim is not valid for input rank");

  llvm::DenseSet<int64_t> reducedDims;
  for (int64_t dim : dims) {
    if (!reducedDims.insert(dim).second)
      return op->emitOpError("reduce dim maps to duplicate logical dims");
  }

  if (destTensor->getRank() !=
      inputTensor->getRank() - static_cast<int64_t>(dims.size()))
    return op->emitOpError(
        "reduce dest rank must match input rank minus reduce dimensions");

  int64_t destDim = 0;
  for (int64_t inputDim = 0; inputDim < inputTensor->getRank(); ++inputDim) {
    if (reducedDims.contains(inputDim))
      continue;
    if (hasStaticMismatch(inputTensor->getDimSize(inputDim),
                          destTensor->getDimSize(destDim)))
      return op->emitOpError(
          "reduce dest shape must match non-reduced input dimensions");
    ++destDim;
  }

  if (init && init.getType() != inputTensor->getElementType())
    return op->emitOpError(
        "reduce init operand type must match input element type");

  mlir::Attribute initValue = op->getAttr("init_value");
  if (initValue) {
    auto typedInit = mlir::dyn_cast<mlir::TypedAttr>(initValue);
    if (!typedInit || typedInit.getType() != inputTensor->getElementType())
      return op->emitOpError(
          "reduce init_value type must match input element type");
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
  case InstrElementwiseKind::Div:
    return ComputeElementwiseKind::Div;
  case InstrElementwiseKind::Max:
    return ComputeElementwiseKind::Max;
  case InstrElementwiseKind::Min:
    return ComputeElementwiseKind::Min;
  case InstrElementwiseKind::Neg:
    return ComputeElementwiseKind::Neg;
  case InstrElementwiseKind::Recip:
    return ComputeElementwiseKind::Recip;
  case InstrElementwiseKind::Sqrt:
    return ComputeElementwiseKind::Sqrt;
  case InstrElementwiseKind::Rsqrt:
    return ComputeElementwiseKind::Rsqrt;
  case InstrElementwiseKind::Exp:
    return ComputeElementwiseKind::Exp;
  case InstrElementwiseKind::Tanh:
    return ComputeElementwiseKind::Tanh;
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
  case InstrElementwiseKind::Square:
  case InstrElementwiseKind::Log2:
  case InstrElementwiseKind::Ln:
  case InstrElementwiseKind::Pow2:
  case InstrElementwiseKind::ExpLp:
  case InstrElementwiseKind::Sin:
  case InstrElementwiseKind::Cos:
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
  case InstrElementwiseKind::Div:
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
getInstrConvertTypePair(mlir::MLIRContext *context, InstrConvertKind kind) {
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

static bool requiresZeroPoint(InstrConvertKind kind) {
  switch (kind) {
  case InstrConvertKind::Int8Fp16:
  case InstrConvertKind::Int8Bf16:
  case InstrConvertKind::Int8Fp32:
  case InstrConvertKind::Int8Tf32:
    return true;
  default:
    return false;
  }
}

static bool requiresRoundingMode(InstrConvertKind kind) {
  switch (kind) {
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
    return true;
  default:
    return false;
  }
}

static mlir::LogicalResult verifyOptionalUInt32Attr(mlir::Operation *op,
                                                    mlir::IntegerAttr attr,
                                                    llvm::StringRef name) {
  if (!attr)
    return mlir::success();
  int64_t value = attr.getInt();
  if (value < 0 || value > std::numeric_limits<uint32_t>::max())
    return op->emitOpError() << name << " must fit uint32_t";
  return mlir::success();
}

static mlir::LogicalResult verifyOptionalRoundingMode(mlir::Operation *op,
                                                      mlir::IntegerAttr attr,
                                                      llvm::StringRef name) {
  if (!attr)
    return mlir::success();
  int64_t value = attr.getInt();
  if (value < 0 || value > 4)
    return op->emitOpError() << name << " must be a RND_MODE value in [0, 4]";
  return mlir::success();
}

static void appendInstructionIssueEffect(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects,
    WaferResourceKind resource, int64_t bytes) {
  appendResourceEffect(effects, resource, WaferResourceAccess::Issue,
                       WaferValueRole::None, 0, bytes);
}

static mlir::LogicalResult verifyNoEmptyVariadicInputs(mlir::Operation *op,
                                                       mlir::ValueRange inputs,
                                                       llvm::StringRef name) {
  if (inputs.empty())
    return op->emitOpError() << name << " requires at least one input";
  return mlir::success();
}

} // namespace

mlir::LogicalResult InstrRDMAOp::verify() {
  if (mlir::failed(
          verifyDDRMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  return verifyMovementDescriptor(getOperation(), getByteCountAttr(),
                                  getInnerBytesAttr(), getSrcStridesAttr(),
                                  getSrcIterationsAttr(), {}, {});
}

InstrFamily InstrRDMAOp::getInstructionFamily() { return InstrFamily::RDMA; }

mlir::LogicalResult InstrRDMAOp::verifyInstructionContract() {
  return verify();
}

void InstrRDMAOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::DDR,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getByteCountAttr().getInt());
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getByteCountAttr().getInt());
  appendInstructionIssueEffect(effects, WaferResourceKind::Movement,
                               getByteCountAttr().getInt());
}

mlir::LogicalResult InstrRDMAOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 3> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrWDMAOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifyDDRMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  return verifyMovementDescriptor(getOperation(), getByteCountAttr(),
                                  getInnerBytesAttr(), {}, {},
                                  getDstStridesAttr(), getDstIterationsAttr());
}

InstrFamily InstrWDMAOp::getInstructionFamily() { return InstrFamily::WDMA; }

mlir::LogicalResult InstrWDMAOp::verifyInstructionContract() {
  return verify();
}

void InstrWDMAOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getByteCountAttr().getInt());
  appendResourceEffect(effects, WaferResourceKind::DDR,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getByteCountAttr().getInt());
  appendInstructionIssueEffect(effects, WaferResourceKind::Movement,
                               getByteCountAttr().getInt());
}

mlir::LogicalResult InstrWDMAOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 3> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrGatherScatterOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  if (mlir::failed(verifyNonNegativeOptionalI64Attr(
          getOperation(), getSrcOffsetAttr(), "src_offset")) ||
      mlir::failed(verifyNonNegativeOptionalI64Attr(
          getOperation(), getDstOffsetAttr(), "dst_offset")) ||
      mlir::failed(verifyMovementDescriptor(
          getOperation(), getByteCountAttr(), getInnerBytesAttr(),
          getSrcStridesAttr(), getSrcIterationsAttr(), getDstStridesAttr(),
          getDstIterationsAttr())) ||
      mlir::failed(verifyDescriptorWithinPhysicalRange(
          getOperation(), getSource().getType(), getSrcOffsetAttr(),
          getInnerBytesAttr(), getSrcStridesAttr(), getSrcIterationsAttr(),
          "source")) ||
      mlir::failed(verifyDescriptorWithinPhysicalRange(
          getOperation(), getDest().getType(), getDstOffsetAttr(),
          getInnerBytesAttr(), getDstStridesAttr(), getDstIterationsAttr(),
          "dest")))
    return mlir::failure();
  return mlir::success();
}

InstrFamily InstrGatherScatterOp::getInstructionFamily() {
  return InstrFamily::TDMA;
}

mlir::LogicalResult InstrGatherScatterOp::verifyInstructionContract() {
  return verify();
}

void InstrGatherScatterOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getByteCountAttr().getInt());
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getByteCountAttr().getInt());
  appendInstructionIssueEffect(effects, WaferResourceKind::Movement,
                               getByteCountAttr().getInt());
}

mlir::LogicalResult InstrGatherScatterOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 3> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrFillOp::verify() {
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  if (getValue().getType() != destTensor->getElementType())
    return emitOpError("fill value type must match dest element type");
  return mlir::success();
}

InstrFamily InstrFillOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrFillOp::verifyInstructionContract() {
  return verify();
}

void InstrFillOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrFillOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 2> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

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
  if (computeKind) {
    return verifyElementwiseTileContract(getOperation(), *computeKind,
                                         getInputs(), getDest().getType());
  }
  return verifySimpleInstrElementwiseContract(getOperation(),
                                              getKindAttr().getValue(),
                                              getInputs(), getDest().getType());
}

InstrFamily InstrElementwiseOp::getInstructionFamily() {
  return InstrFamily::CT;
}

mlir::LogicalResult InstrElementwiseOp::verifyInstructionContract() {
  return verify();
}

void InstrElementwiseOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    appendResourceEffect(effects, WaferResourceKind::SPM,
                         WaferResourceAccess::Read, WaferValueRole::Operand,
                         index, getCompactByteSizeOrUnknown(input.getType()));
  }
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand,
                       getInputs().size(),
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrElementwiseOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
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
  return verifySameShape(getOperation(), *sourceTensor, *destTensor,
                         "bit2fp source and dest shapes must match");
}

InstrFamily InstrBit2FpOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrBit2FpOp::verifyInstructionContract() {
  return verify();
}

void InstrBit2FpOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getSource().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrBit2FpOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 3> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

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
  return mlir::success();
}

InstrFamily InstrMaskMoveOp::getInstructionFamily() {
  return InstrFamily::TDMA;
}

mlir::LogicalResult InstrMaskMoveOp::verifyInstructionContract() {
  return verify();
}

void InstrMaskMoveOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getSource().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getMask().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 2,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Movement,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrMaskMoveOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrReduceOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getInput().getType(), "input")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
  return verifyInstructionReduceContract(getOperation(), getInput(), getDest(),
                                         getInit(), getDimAttr());
}

InstrFamily InstrReduceOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrReduceOp::verifyInstructionContract() {
  return verify();
}

void InstrReduceOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getInput().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrReduceOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrConvertOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();
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
  mlir::IntegerAttr zeroPoint =
      getOperation()->getAttrOfType<mlir::IntegerAttr>("zero_point");
  mlir::IntegerAttr roundingMode =
      getOperation()->getAttrOfType<mlir::IntegerAttr>("rounding_mode");
  if (requiresZeroPoint(kind)) {
    if (!zeroPoint)
      return emitOpError("convert kind requires zero_point attr");
    if (roundingMode)
      return emitOpError("zero-point convert must not have rounding_mode attr");
  } else if (zeroPoint) {
    return emitOpError("convert kind must not have zero_point attr");
  }
  if (requiresRoundingMode(kind)) {
    if (!roundingMode)
      return emitOpError("convert kind requires rounding_mode attr");
  } else if (roundingMode) {
    return emitOpError("convert kind must not have rounding_mode attr");
  }
  if (mlir::failed(
          verifyOptionalUInt32Attr(getOperation(), zeroPoint, "zero_point")) ||
      mlir::failed(verifyOptionalRoundingMode(getOperation(), roundingMode,
                                              "rounding_mode")))
    return mlir::failure();
  return mlir::success();
}

InstrFamily InstrConvertOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrConvertOp::verifyInstructionContract() {
  return verify();
}

void InstrConvertOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getSource().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrConvertOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrGemmOp::verify() {
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
                                "gemm lhs and rhs element types must match")) ||
      mlir::failed(verifySameElementType(
          getOperation(), *lhsTensor, *destTensor,
          "gemm operand and dest element types must match")))
    return mlir::failure();

  int64_t m = getMAttr().getInt();
  int64_t k = getKAttr().getInt();
  int64_t n = getNAttr().getInt();
  if (m <= 0 || k <= 0 || n <= 0)
    return emitOpError("m, k and n must be positive");

  if (lhsTensor->getRank() == 2 && rhsTensor->getRank() == 2 &&
      destTensor->getRank() == 2) {
    if (hasAnyBatchedGemmAttrs(getOperation()))
      return emitOpError("rank-2 gemm must not carry batched GEMM attrs");
    if (hasStaticMismatch(lhsTensor->getDimSize(0), m) ||
        hasStaticMismatch(lhsTensor->getDimSize(1), k) ||
        hasStaticMismatch(rhsTensor->getDimSize(0), k) ||
        hasStaticMismatch(rhsTensor->getDimSize(1), n) ||
        hasStaticMismatch(destTensor->getDimSize(0), m) ||
        hasStaticMismatch(destTensor->getDimSize(1), n))
      return emitOpError("m/k/n attrs must match GEMM operand shapes");
    return mlir::success();
  }

  BatchedGemmDimAttrs attrs;
  if (mlir::failed(verifyBatchedGemmTileContract(
          getOperation(), *lhsTensor, *rhsTensor, *destTensor, attrs)))
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
  return mlir::success();
}

InstrFamily InstrGemmOp::getInstructionFamily() { return InstrFamily::NE; }

mlir::LogicalResult InstrGemmOp::verifyInstructionContract() {
  return verify();
}

void InstrGemmOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getLhs().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getRhs().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 2,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrGemmOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrConvOp::verify() {
  if (mlir::failed(verifyAlignedSPMMemRef(getOperation(), getInput().getType(),
                                          "input")) ||
      mlir::failed(verifyAlignedSPMMemRef(getOperation(), getWeight().getType(),
                                          "weight")) ||
      mlir::failed(
          verifyAlignedSPMMemRef(getOperation(), getDest().getType(), "dest")))
    return mlir::failure();

  std::optional<mlir::RankedTensorType> inputTensor =
      getLogicalTensorType(getInput().getType());
  std::optional<mlir::RankedTensorType> weightTensor =
      getLogicalTensorType(getWeight().getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  if (mlir::failed(verifySameElementType(
          getOperation(), *inputTensor, *weightTensor,
          "conv input and weight element types must match")) ||
      mlir::failed(verifySameElementType(
          getOperation(), *inputTensor, *destTensor,
          "conv input and dest element types must match")) ||
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
  return mlir::success();
}

InstrFamily InstrConvOp::getInstructionFamily() { return InstrFamily::NE; }

mlir::LogicalResult InstrConvOp::verifyInstructionContract() {
  return verify();
}

void InstrConvOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getInput().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getWeight().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 2,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrConvOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

static bool isIndexedPoolKind(InstrPoolKind kind) {
  return kind == InstrPoolKind::IndexedMax || kind == InstrPoolKind::IndexedMin;
}

mlir::LogicalResult InstrPoolOp::verify() {
  if (mlir::failed(verifyAlignedSPMMemRef(getOperation(), getInput().getType(),
                                          "input")) ||
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
  for (auto [index, dest] : llvm::enumerate(getDests())) {
    if (mlir::failed(
            verifyAlignedSPMMemRef(getOperation(), dest.getType(), "dest")))
      return mlir::failure();
    std::optional<mlir::RankedTensorType> destTensor =
        getLogicalTensorType(dest.getType());
    if (index == 0) {
      if (mlir::failed(verifySameElementType(
              getOperation(), *inputTensor, *destTensor,
              "pool value dest element type must match input element type")))
        return mlir::failure();
      continue;
    }
    if (!destTensor->getElementType().isInteger(32))
      return emitOpError("indexed pool index dest element type must be i32");
  }
  return mlir::success();
}

InstrFamily InstrPoolOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrPoolOp::verifyInstructionContract() {
  return verify();
}

void InstrPoolOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getInput().getType()));
  for (auto [index, dest] : llvm::enumerate(getDests())) {
    appendResourceEffect(effects, WaferResourceKind::SPM,
                         WaferResourceAccess::Write, WaferValueRole::Operand,
                         index + 1,
                         getCompactByteSizeOrUnknown(dest.getType()));
  }
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDests().front().getType()));
}

mlir::LogicalResult InstrPoolOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrUnpoolOp::verify() {
  if (mlir::failed(verifyAlignedSPMMemRef(getOperation(), getInput().getType(),
                                          "input")) ||
      mlir::failed(verifyAlignedSPMMemRef(getOperation(), getDest().getType(),
                                          "dest")) ||
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
  mlir::IntegerAttr index =
      getOperation()->getAttrOfType<mlir::IntegerAttr>("index");
  if (getKindAttr().getValue() == InstrUnpoolKind::Avg) {
    if (index)
      return emitOpError("avg unpool must not have index attr");
  } else {
    if (!index)
      return emitOpError("unpool kind requires scalar index attr");
    if (mlir::failed(verifyOptionalUInt32Attr(getOperation(), index, "index")))
      return mlir::failure();
  }
  return mlir::success();
}

InstrFamily InstrUnpoolOp::getInstructionFamily() { return InstrFamily::CT; }

mlir::LogicalResult InstrUnpoolOp::verifyInstructionContract() {
  return verify();
}

void InstrUnpoolOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getInput().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Compute,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrUnpoolOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}

mlir::LogicalResult InstrTDMADataMoveOp::verify() {
  if (mlir::failed(
          verifySPMMemRef(getOperation(), getSource().getType(), "source")) ||
      mlir::failed(
          verifySPMMemRef(getOperation(), getDest().getType(), "dest")) ||
      mlir::failed(verifyI64Array(getOperation(), getSourceShapeAttr(),
                                  "source_shape", 4, /*positive=*/true)) ||
      mlir::failed(verifyI64Array(getOperation(), getDestShapeAttr(),
                                  "dest_shape", 4, /*positive=*/true)))
    return mlir::failure();
  if (getPermutationAttr() &&
      mlir::failed(verifyPermutationI64Array(
          getOperation(), getPermutationAttr(), "permutation")))
    return mlir::failure();
  if (getPadsAttr() &&
      mlir::failed(verifyI64Array(getOperation(), getPadsAttr(), "pads", 4,
                                  /*positive=*/false)))
    return mlir::failure();
  if (getKernelStridesAttr() &&
      mlir::failed(verifyI64Array(getOperation(), getKernelStridesAttr(),
                                  "kernel_strides", 4,
                                  /*positive=*/true)))
    return mlir::failure();

  switch (getKindAttr().getValue()) {
  case InstrDataMoveKind::Mirror:
  case InstrDataMoveKind::Rotate90:
  case InstrDataMoveKind::Rotate180:
  case InstrDataMoveKind::Rotate270:
  case InstrDataMoveKind::Nchw2Nhwc:
  case InstrDataMoveKind::Nhwc2Nchw:
  case InstrDataMoveKind::TensorNom:
    if (getPermutationAttr())
      return emitOpError(
          "non-parameterized data_move must not have permutation attr");
    if (getPadsAttr())
      return emitOpError("non-parameterized data_move must not have pads attr");
    if (getKernelStridesAttr())
      return emitOpError(
          "non-parameterized data_move must not have kernel_strides attr");
    break;
  case InstrDataMoveKind::Transpose:
    if (!getPermutationAttr())
      return emitOpError("transpose data_move requires permutation attr");
    if (getPadsAttr())
      return emitOpError("transpose data_move must not have pads attr");
    if (getKernelStridesAttr())
      return emitOpError(
          "transpose data_move must not have kernel_strides attr");
    break;
  case InstrDataMoveKind::Pad:
    if (!getPadsAttr())
      return emitOpError("pad data_move requires pads attr");
    if (getPermutationAttr())
      return emitOpError("pad data_move must not have permutation attr");
    if (getKernelStridesAttr())
      return emitOpError("pad data_move must not have kernel_strides attr");
    break;
  case InstrDataMoveKind::Img2Col:
    if (!getPadsAttr())
      return emitOpError("img2col data_move requires pads attr");
    if (!getKernelStridesAttr())
      return emitOpError("img2col data_move requires kernel_strides attr");
    if (getPermutationAttr())
      return emitOpError("img2col data_move must not have permutation attr");
    break;
  }

  std::optional<mlir::RankedTensorType> sourceTensor =
      getLogicalTensorType(getSource().getType());
  std::optional<mlir::RankedTensorType> destTensor =
      getLogicalTensorType(getDest().getType());
  return verifySameElementType(getOperation(), *sourceTensor, *destTensor,
                               "tdma_data_move source and dest element types "
                               "must match");
}

InstrFamily InstrTDMADataMoveOp::getInstructionFamily() {
  return InstrFamily::TDMA;
}

mlir::LogicalResult InstrTDMADataMoveOp::verifyInstructionContract() {
  return verify();
}

void InstrTDMADataMoveOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Read, WaferValueRole::Operand, 0,
                       getCompactByteSizeOrUnknown(getSource().getType()));
  appendResourceEffect(effects, WaferResourceKind::SPM,
                       WaferResourceAccess::Write, WaferValueRole::Operand, 1,
                       getCompactByteSizeOrUnknown(getDest().getType()));
  appendInstructionIssueEffect(
      effects, WaferResourceKind::Movement,
      getCompactByteSizeOrUnknown(getDest().getType()));
}

mlir::LogicalResult InstrTDMADataMoveOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 4> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
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
                                                     llvm::StringRef name) {
  auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(name);
  if (!attr)
    return op->emitOpError() << "peripheral kind requires " << name << " attr";
  return verifyI64Array(op, attr, name, 4, /*positive=*/true);
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

mlir::LogicalResult InstrPeripheralOp::verify() {
  if (mlir::failed(verifyPositiveI64Attr(getOperation(), getElemCountAttr(),
                                         "elem_count")))
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
  }
  switch (getKindAttr().getValue()) {
  case InstrPeripheralKind::Count:
    return emitOpError(
        "count peripheral writeback is not represented in instruction IR");
  case InstrPeripheralKind::ArgMax:
  case InstrPeripheralKind::ArgMin:
  case InstrPeripheralKind::Factorize:
  case InstrPeripheralKind::RandGen:
    return verifyForbiddenPeripheralAttrs(
        getOperation(), {"source_shape", "dest_shape", "lut_elem_count",
                         "scale", "probability", "rounding_mode"});
  case InstrPeripheralKind::Bilinear:
    if (mlir::failed(
            verifyPeripheralShapeAttr(getOperation(), "source_shape")) ||
        mlir::failed(verifyPeripheralShapeAttr(getOperation(), "dest_shape")))
      return mlir::failure();
    return verifyForbiddenPeripheralAttrs(
        getOperation(),
        {"lut_elem_count", "scale", "probability", "rounding_mode"});
  case InstrPeripheralKind::Lut16:
  case InstrPeripheralKind::Lut32:
    if (mlir::failed(
            verifyRequiredUInt32Attr(getOperation(), "lut_elem_count")))
      return mlir::failure();
    return verifyForbiddenPeripheralAttrs(
        getOperation(), {"source_shape", "dest_shape", "scale", "probability",
                         "rounding_mode"});
  case InstrPeripheralKind::ElemMask:
    if (mlir::failed(verifyRequiredUInt32Attr(getOperation(), "scale")) ||
        mlir::failed(verifyRequiredUInt32Attr(getOperation(), "probability")) ||
        mlir::failed(verifyOptionalRoundingMode(
            getOperation(),
            getOperation()->getAttrOfType<mlir::IntegerAttr>("rounding_mode"),
            "rounding_mode")))
      return mlir::failure();
    if (!getOperation()->hasAttr("rounding_mode"))
      return emitOpError("peripheral kind requires rounding_mode attr");
    return verifyForbiddenPeripheralAttrs(
        getOperation(), {"source_shape", "dest_shape", "lut_elem_count"});
  }
  return mlir::success();
}

InstrFamily InstrPeripheralOp::getInstructionFamily() {
  return InstrFamily::CT;
}

mlir::LogicalResult InstrPeripheralOp::verifyInstructionContract() {
  return verify();
}

void InstrPeripheralOp::collectWaferResourceEffects(
    llvm::SmallVectorImpl<WaferResourceEffect> &effects) {
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    appendResourceEffect(effects, WaferResourceKind::SPM,
                         WaferResourceAccess::Read, WaferValueRole::Operand,
                         index, getCompactByteSizeOrUnknown(input.getType()));
  }
  for (auto [index, dest] : llvm::enumerate(getDests())) {
    appendResourceEffect(effects, WaferResourceKind::SPM,
                         WaferResourceAccess::Write, WaferValueRole::Operand,
                         getInputs().size() + index,
                         getCompactByteSizeOrUnknown(dest.getType()));
  }
  appendInstructionIssueEffect(effects, WaferResourceKind::Compute,
                               getElemCountAttr().getInt());
}

mlir::LogicalResult InstrPeripheralOp::verifyWaferResourceEffectContract() {
  llvm::SmallVector<WaferResourceEffect, 8> effects;
  collectWaferResourceEffects(effects);
  return verifyResourceEffects(getOperation(), effects);
}
