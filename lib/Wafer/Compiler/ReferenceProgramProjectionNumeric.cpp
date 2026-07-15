//===- ReferenceProgramProjectionNumeric.cpp - Numeric projection -------===//

#include "ReferenceProgramProjectionInternal.h"

#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <optional>

namespace wafer::compiler::reference_detail {
namespace {

bool isSupportedElementwiseKind(wafer::InstrElementwiseKind kind,
                                size_t &arity) {
  switch (kind) {
  case wafer::InstrElementwiseKind::Abs:
  case wafer::InstrElementwiseKind::Recip:
  case wafer::InstrElementwiseKind::Square:
  case wafer::InstrElementwiseKind::Sqrt:
  case wafer::InstrElementwiseKind::Rsqrt:
  case wafer::InstrElementwiseKind::Neg:
  case wafer::InstrElementwiseKind::Log2:
  case wafer::InstrElementwiseKind::Ln:
  case wafer::InstrElementwiseKind::Pow2:
  case wafer::InstrElementwiseKind::Exp:
  case wafer::InstrElementwiseKind::ExpLp:
  case wafer::InstrElementwiseKind::Sin:
  case wafer::InstrElementwiseKind::Cos:
  case wafer::InstrElementwiseKind::Tanh:
  case wafer::InstrElementwiseKind::Sigmoid:
  case wafer::InstrElementwiseKind::Relu:
  case wafer::InstrElementwiseKind::SatRelu:
  case wafer::InstrElementwiseKind::Softplus:
    arity = 1;
    return true;
  case wafer::InstrElementwiseKind::Max:
  case wafer::InstrElementwiseKind::Min:
  case wafer::InstrElementwiseKind::Add:
  case wafer::InstrElementwiseKind::Sub:
  case wafer::InstrElementwiseKind::Mul:
  case wafer::InstrElementwiseKind::Div:
    arity = 2;
    return true;
  default:
    return false;
  }
}

struct ConvertSpec {
  NumericFormat source;
  NumericFormat dest;
  wafer::InstrConvertParameterKind parameter;
};

std::optional<ConvertSpec> getConvertSpec(mlir::MLIRContext *context,
                                          wafer::InstrConvertKind kind) {
  auto [sourceType, destType] = wafer::getInstrConvertTypePair(context, kind);
  std::optional<NumericFormat> source = getNumericFormat(sourceType);
  std::optional<NumericFormat> dest = getNumericFormat(destType);
  if (!source || !dest)
    return std::nullopt;
  return ConvertSpec{*source, *dest, wafer::getInstrConvertParameterKind(kind)};
}

llvm::Expected<llvm::APFloat::roundingMode>
projectRoundingMode(uint64_t value) {
  switch (value) {
  case 0:
    return llvm::APFloat::rmNearestTiesToEven;
  case 1:
    return llvm::APFloat::rmTowardZero;
  case 2:
    return llvm::APFloat::rmTowardPositive;
  case 3:
    return llvm::APFloat::rmTowardNegative;
  case 4:
    return llvm::APFloat::rmNearestTiesToEven;
  default:
    return unsupported("convert rounding mode is outside RND_MODE");
  }
}

} // namespace

llvm::Expected<bool>
ProgramProjector::projectNumeric(mlir::Operation &operation, Command &command) {
  if (auto convert = mlir::dyn_cast<wafer::InstrConvertOp>(operation)) {
    std::optional<ConvertSpec> spec =
        getConvertSpec(convert.getContext(), convert.getKind());
    if (!spec)
      return unsupported("unknown instruction convert kind");
    auto sourceType = convert.getSource().getType();
    auto destType = convert.getDest().getType();
    if (!sourceType.hasStaticShape() || !destType.hasStaticShape() ||
        !wafer::isWaferSPMMemRefType(sourceType) ||
        !wafer::isWaferSPMMemRefType(destType))
      return unsupported("convert requires static Wafer SPM memrefs");
    if (!matchesNumericFormat(sourceType.getElementType(), spec->source) ||
        !matchesNumericFormat(destType.getElementType(), spec->dest))
      return unsupported("convert kind disagrees with memref element type");
    if (sourceType.getNumElements() != destType.getNumElements())
      return unsupported(
          "convert source and destination element counts disagree");
    if (!wafer::computeWaferPhysicalTensorInfo(sourceType) ||
        !wafer::computeWaferPhysicalTensorInfo(destType))
      return unsupported("convert has no accepted physical layout");

    std::optional<uint64_t> zeroPoint = convert.getZeroPoint();
    std::optional<uint64_t> roundingMode = convert.getRoundingMode();
    switch (spec->parameter) {
    case wafer::InstrConvertParameterKind::None:
      if (zeroPoint || roundingMode)
        return unsupported("plain convert has unexpected parameters");
      break;
    case wafer::InstrConvertParameterKind::RoundingMode: {
      if (zeroPoint || !roundingMode)
        return unsupported(
            "rounding convert has invalid parameter combination");
      auto projectedRounding = projectRoundingMode(*roundingMode);
      if (!projectedRounding)
        return projectedRounding.takeError();
      command.roundingMode = *projectedRounding;
      if (*roundingMode == 4) {
        command.stochasticRounding = true;
        program.usesStochasticRounding = true;
      }
      break;
    }
    case wafer::InstrConvertParameterKind::ZeroPoint:
      if (!zeroPoint || roundingMode)
        return unsupported(
            "zero-point convert has invalid parameter combination");
      return unsupported("INT8 zero-point convert formula is not evidenced");
    }

    auto source = use(convert.getSource());
    auto dest = use(convert.getDest());
    if (!source)
      return source.takeError();
    if (!dest)
      return dest.takeError();
    command.kind = CommandKind::Convert;
    command.source = *source;
    command.dest = *dest;
    command.sourceFormat = spec->source;
    command.destFormat = spec->dest;
  } else if (auto reduce = mlir::dyn_cast<wafer::InstrReduceOp>(operation)) {
    (void)reduce;
    return unsupported(
        "target-native reduce has no compiler-owned source equivalence "
        "proof");
  } else if (auto gemm = mlir::dyn_cast<wafer::InstrGemmOp>(operation)) {
    auto lhsType = requireF32Buffer(gemm.getLhs(), "GEMM lhs");
    if (!lhsType)
      return lhsType.takeError();
    auto rhsType = requireF32Buffer(gemm.getRhs(), "GEMM rhs");
    if (!rhsType)
      return rhsType.takeError();
    auto destType = requireF32Buffer(gemm.getDest(), "GEMM dest");
    if (!destType)
      return destType.takeError();
    int64_t m = static_cast<int64_t>(gemm.getM());
    int64_t n = static_cast<int64_t>(gemm.getN());
    int64_t k = static_cast<int64_t>(gemm.getK());
    if (lhsType->getRank() != rhsType->getRank() ||
        lhsType->getRank() != destType->getRank() || lhsType->getRank() < 2)
      return invalid("GEMM accepted buffer ranks disagree");
    int64_t rank = lhsType->getRank();
    if (lhsType->getDimSize(rank - 2) != m ||
        lhsType->getDimSize(rank - 1) != k ||
        rhsType->getDimSize(rank - 2) != k ||
        rhsType->getDimSize(rank - 1) != n ||
        destType->getDimSize(rank - 2) != m ||
        destType->getDimSize(rank - 1) != n)
      return invalid("GEMM dimensions disagree with accepted buffers");
    command.batchShape.assign(destType->getShape().begin(),
                              destType->getShape().end() - 2);
    if (lhsType->getShape().drop_back(2) !=
            llvm::ArrayRef<int64_t>(command.batchShape) ||
        rhsType->getShape().drop_back(2) !=
            llvm::ArrayRef<int64_t>(command.batchShape))
      return invalid("GEMM accepted batch shapes disagree");
    if (rank > 2) {
      llvm::SmallVector<int64_t> expectedBatchDims;
      for (int64_t dimension = 0; dimension < rank - 2; ++dimension)
        expectedBatchDims.push_back(dimension);
      auto hasBatchDims = [&](llvm::StringRef name) {
        auto attr = gemm->getAttrOfType<mlir::DenseI64ArrayAttr>(name);
        return attr &&
               attr.asArrayRef() == llvm::ArrayRef<int64_t>(expectedBatchDims);
      };
      auto hasDim = [&](llvm::StringRef name, int64_t expected) {
        auto attr = gemm->getAttrOfType<mlir::IntegerAttr>(name);
        return attr && attr.getInt() == expected;
      };
      int64_t batchCount = 1;
      for (int64_t size : command.batchShape) {
        if (size <= 0 ||
            batchCount > std::numeric_limits<int64_t>::max() / size)
          return invalid("GEMM accepted batch count overflows");
        batchCount *= size;
      }
      if (!hasBatchDims("lhs_batch_dims") || !hasBatchDims("rhs_batch_dims") ||
          !hasBatchDims("result_batch_dims") ||
          !hasDim("lhs_m_dim", rank - 2) ||
          !hasDim("lhs_contracting_dim", rank - 1) ||
          !hasDim("rhs_contracting_dim", rank - 2) ||
          !hasDim("rhs_n_dim", rank - 1) || !hasDim("result_m_dim", rank - 2) ||
          !hasDim("result_n_dim", rank - 1) ||
          !hasDim("batch_count", batchCount))
        return invalid("GEMM batched dimension attrs are not canonical");
    } else if (gemm.getBatchCount()) {
      return invalid("rank-2 GEMM unexpectedly carries batch attrs");
    }
    command.kind = CommandKind::Gemm;
    auto lhs = use(gemm.getLhs());
    auto rhs = use(gemm.getRhs());
    auto dest = use(gemm.getDest());
    if (!lhs)
      return lhs.takeError();
    if (!rhs)
      return rhs.takeError();
    if (!dest)
      return dest.takeError();
    command.lhs = *lhs;
    command.rhs = *rhs;
    command.dest = *dest;
    command.m = m;
    command.n = n;
    command.k = k;
  } else if (auto elementwise =
                 mlir::dyn_cast<wafer::InstrElementwiseOp>(operation)) {
    size_t arity = 0;
    if (!isSupportedElementwiseKind(elementwise.getKind(), arity) ||
        elementwise.getInputs().size() != arity)
      return unsupported("elementwise kind or arity");
    auto destType = requireF32Buffer(elementwise.getDest(), "elementwise dest");
    if (!destType)
      return destType.takeError();
    command.kind = CommandKind::Elementwise;
    command.elementwiseKind = elementwise.getKind();
    auto dest = use(elementwise.getDest());
    if (!dest)
      return dest.takeError();
    command.dest = *dest;
    for (mlir::Value value : elementwise.getInputs()) {
      auto inputType = requireF32Buffer(value, "elementwise input");
      if (!inputType)
        return inputType.takeError();
      if (inputType->getShape() != destType->getShape())
        return invalid("elementwise input shape disagrees with destination");
      auto input = use(value);
      if (!input)
        return input.takeError();
      command.inputs.push_back(*input);
    }
  } else if (auto bit2fp = mlir::dyn_cast<wafer::InstrBit2FpOp>(operation)) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(bit2fp.getSource().getType());
    auto destType = requireF32Buffer(bit2fp.getDest(), "bit2fp dest");
    auto sourceInfo = sourceType
                          ? wafer::computeWaferPhysicalTensorInfo(sourceType)
                          : std::nullopt;
    if (!sourceType || !sourceType.hasStaticShape() ||
        !sourceType.getElementType().isInteger(1) ||
        !wafer::isWaferSPMMemRefType(sourceType) || !sourceInfo ||
        sourceInfo->layout != wafer::MemLayout::Tensor || !destType)
      return unsupported(
          "bit2fp requires static tensor-layout i1 and f32 SPM memrefs");
    if (sourceType.getShape() != destType->getShape())
      return invalid("bit2fp source and destination shapes disagree");
    auto source = use(bit2fp.getSource());
    auto dest = use(bit2fp.getDest());
    if (!source)
      return source.takeError();
    if (!dest)
      return dest.takeError();
    command.kind = CommandKind::Bit2Fp;
    command.source = *source;
    command.dest = *dest;
  } else if (auto maskMove =
                 mlir::dyn_cast<wafer::InstrMaskMoveOp>(operation)) {
    auto sourceType =
        requireF32Buffer(maskMove.getSource(), "mask_move source");
    auto maskType = requireF32Buffer(maskMove.getMask(), "mask_move mask");
    auto destType = requireF32Buffer(maskMove.getDest(), "mask_move dest");
    if (!sourceType)
      return sourceType.takeError();
    if (!maskType)
      return maskType.takeError();
    if (!destType)
      return destType.takeError();
    if (sourceType->getShape() != destType->getShape() ||
        maskType->getShape() != destType->getShape())
      return invalid("mask_move buffer shapes disagree");
    auto source = use(maskMove.getSource());
    auto mask = use(maskMove.getMask());
    auto dest = use(maskMove.getDest());
    if (!source)
      return source.takeError();
    if (!mask)
      return mask.takeError();
    if (!dest)
      return dest.takeError();
    command.kind = CommandKind::MaskMove;
    command.source = *source;
    command.mask = *mask;
    command.dest = *dest;
  } else if (auto fill = mlir::dyn_cast<wafer::InstrFillOp>(operation)) {
    auto destType = mlir::dyn_cast<mlir::MemRefType>(fill.getDest().getType());
    auto destInfo = destType ? wafer::computeWaferPhysicalTensorInfo(destType)
                             : std::nullopt;
    if (!destType || !destType.hasStaticShape() ||
        !wafer::isWaferSPMMemRefType(destType) || !destInfo ||
        (!destType.getElementType().isF32() &&
         !destType.getElementType().isInteger(1)))
      return unsupported("fill requires a static f32 or i1 SPM memref");
    if (destType.getElementType().isInteger(1) &&
        destInfo->layout != wafer::MemLayout::Tensor)
      return unsupported("i1 fill requires tensor-layout storage");
    auto dest = use(fill.getDest());
    auto scalar = use(fill.getValue());
    if (!dest)
      return dest.takeError();
    if (!scalar)
      return scalar.takeError();
    command.kind = CommandKind::Fill;
    command.dest = *dest;
    command.scalar = *scalar;

  } else {
    return false;
  }
  return true;
}

llvm::Expected<mlir::MemRefType>
ProgramProjector::requireF32Buffer(mlir::Value value, llvm::StringRef purpose) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !type.getElementType().isF32() || !type.hasStaticShape())
    return unsupported((purpose + " requires a static f32 memref").str());
  if (!wafer::computeWaferPhysicalTensorInfo(type))
    return unsupported((purpose + " has no accepted physical layout").str());
  return type;
}

} // namespace wafer::compiler::reference_detail
