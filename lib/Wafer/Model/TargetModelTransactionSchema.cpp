//===- TargetModelTransactionSchema.cpp - Transaction validation -----===//

#include "TargetModelKernelInternal.h"

#include "Wafer/Target/TargetFormat.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <limits>
#include <optional>
#include <type_traits>
#include <variant>

namespace wafer::model::kernel_detail {

llvm::Error kernelError(TargetModelKernelErrorCode code,
                        const llvm::Twine &detail) {
  return llvm::make_error<TargetModelKernelError>(code, detail.str());
}

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

llvm::Expected<uint64_t>
getDescriptorSegmentCount(const std::array<uint32_t, 3> &iterations) {
  uint64_t count = 1;
  for (uint32_t iteration : iterations) {
    if (iteration == 0)
      return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                         "movement iteration counts must be positive");
    if (!checkedMultiply(count, iteration, count))
      return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                         "movement segment count overflows");
  }
  return count;
}

namespace {

llvm::Error requireFormat(LogicalFormat format, llvm::StringRef role) {
  if (!findLogicalFormatDescriptor(format))
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       llvm::Twine(role) + " has an unknown logical format");
  return llvm::Error::success();
}

llvm::Error requireEngineFormat(LogicalFormat format, TargetFormatEngine engine,
                                llvm::StringRef role) {
  if (llvm::Error error = requireFormat(format, role))
    return error;
  const TargetFormatEncodingRecord *record = findTargetFormatEncoding(
      TargetProfileId::waferTx81SingleCardKernelV1(), engine, format);
  if (!record || !record->isSupported() || !record->dataFormatCode)
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       llvm::Twine(role) +
                           " format is unsupported by its target engine");
  return llvm::Error::success();
}

llvm::Error requirePositive(llvm::ArrayRef<uint32_t> values,
                            llvm::StringRef role) {
  if (llvm::is_contained(values, UINT32_C(0)))
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       llvm::Twine(role) + " must be strictly positive");
  return llvm::Error::success();
}

llvm::Error validateDescriptor(uint32_t byteCount, uint32_t innerBytes,
                               const std::array<uint32_t, 3> &iterations,
                               std::optional<LogicalFormat> format) {
  if (byteCount == 0 || innerBytes == 0)
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "movement byte_count and inner_bytes must be positive");
  llvm::Expected<uint64_t> segments = getDescriptorSegmentCount(iterations);
  if (!segments)
    return segments.takeError();
  uint64_t payload = 0;
  if (!checkedMultiply(innerBytes, *segments, payload) || payload != byteCount)
    return kernelError(
        TargetModelKernelErrorCode::InvalidTransactionField,
        "movement byte_count must equal inner_bytes times all iterations");
  if (!format)
    return llvm::Error::success();
  if (llvm::Error error = requireFormat(*format, "movement"))
    return error;
  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(*format);
  if (!descriptor->bitpacked) {
    const uint64_t bytes = descriptor->storageBits / 8;
    if (bytes == 0 || innerBytes % bytes != 0)
      return kernelError(
          TargetModelKernelErrorCode::InvalidTransactionField,
          "movement inner_bytes is not a whole number of logical elements");
  }
  return llvm::Error::success();
}

template <typename Enum>
std::optional<Enum> symbolizeKnownEnum(uint32_t value) {
  if constexpr (std::is_same_v<Enum, InstrElementwiseKind>)
    return static_cast<std::optional<InstrElementwiseKind> (*)(uint32_t)>(
        symbolizeInstrElementwiseKind)(value);
  if constexpr (std::is_same_v<Enum, InstrReduceKind>)
    return static_cast<std::optional<InstrReduceKind> (*)(uint32_t)>(
        symbolizeInstrReduceKind)(value);
  if constexpr (std::is_same_v<Enum, InstrConvKind>)
    return static_cast<std::optional<InstrConvKind> (*)(uint32_t)>(
        symbolizeInstrConvKind)(value);
  if constexpr (std::is_same_v<Enum, InstrPoolKind>)
    return static_cast<std::optional<InstrPoolKind> (*)(uint32_t)>(
        symbolizeInstrPoolKind)(value);
  if constexpr (std::is_same_v<Enum, InstrUnpoolKind>)
    return static_cast<std::optional<InstrUnpoolKind> (*)(uint32_t)>(
        symbolizeInstrUnpoolKind)(value);
  llvm_unreachable("unsupported model enum symbolizer");
}

template <typename Enum>
llvm::Error requireKnownEnum(Enum value, llvm::StringRef role) {
  if (!symbolizeKnownEnum<Enum>(static_cast<uint32_t>(value)))
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       llvm::Twine(role) + " has an unknown enum value");
  return llvm::Error::success();
}

llvm::Error validateConvert(const compiler::TargetConvertTransaction &value) {
  if (value.elementCount == 0)
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "convert element_count must be positive");
  const uint32_t opcode = static_cast<uint32_t>(value.kind);
  if (!symbolizeInstrConvertKind(opcode))
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "convert has an unknown kind");
  const TargetConvertRoute *route =
      findTargetConvertRoute(TargetProfileId::waferTx81SingleCardKernelV1(),
                             static_cast<uint16_t>(opcode));
  if (!route)
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "convert kind has no exact target route");
  switch (route->parameterKind) {
  case TargetConvertParameterKind::None:
    if (value.zeroPoint || value.roundingMode)
      return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                         "parameterless convert carries an optional field");
    break;
  case TargetConvertParameterKind::RoundingMode:
    if (value.zeroPoint || !value.roundingMode ||
        *value.roundingMode > std::numeric_limits<uint8_t>::max())
      return kernelError(
          TargetModelKernelErrorCode::InvalidTransactionField,
          "rounding convert requires exactly one known rounding mode");
    if (llvm::Expected<NumericRoundingMode> mode =
            parseNumericRoundingMode(static_cast<uint8_t>(*value.roundingMode));
        !mode) {
      llvm::consumeError(mode.takeError());
      return kernelError(
          TargetModelKernelErrorCode::InvalidTransactionField,
          "rounding convert requires exactly one known rounding mode");
    }
    break;
  case TargetConvertParameterKind::ZeroPoint:
    if (!value.zeroPoint || value.roundingMode)
      return kernelError(
          TargetModelKernelErrorCode::InvalidTransactionField,
          "zero-point convert requires exactly one uint32 zero point");
    break;
  }
  return llvm::Error::success();
}

llvm::Error
validateElementwise(const compiler::TargetElementwiseTransaction &value) {
  if (value.elementCount == 0)
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "elementwise element_count must be positive");
  if (llvm::Error error = requireKnownEnum(value.kind, "elementwise kind"))
    return error;
  if (llvm::Error error = requireEngineFormat(
          value.format, TargetFormatEngine::CT, "elementwise"))
    return error;
  bool binary = false;
  switch (value.kind) {
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
    binary = true;
    break;
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
    break;
  }
  if (value.rhs.has_value() != binary)
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "elementwise rhs presence differs from operation arity");
  const bool logic = value.kind == InstrElementwiseKind::LogicNot ||
                     value.kind == InstrElementwiseKind::LogicAnd ||
                     value.kind == InstrElementwiseKind::LogicOr ||
                     value.kind == InstrElementwiseKind::LogicXor;
  const bool relation = value.kind == InstrElementwiseKind::Eq ||
                        value.kind == InstrElementwiseKind::Ne ||
                        value.kind == InstrElementwiseKind::Ge ||
                        value.kind == InstrElementwiseKind::Gt ||
                        value.kind == InstrElementwiseKind::Le ||
                        value.kind == InstrElementwiseKind::Lt;
  if (logic != (value.format == LogicalFormat::Bool) ||
      (relation && value.format == LogicalFormat::Bool))
    return kernelError(
        TargetModelKernelErrorCode::InvalidTransactionField,
        "elementwise format differs from logic/relation operation class");
  return llvm::Error::success();
}

llvm::Error validateShapeProduct(llvm::ArrayRef<uint32_t> shape,
                                 llvm::StringRef role) {
  if (llvm::Error error = requirePositive(shape, role))
    return error;
  uint64_t product = 1;
  for (uint32_t dimension : shape)
    if (!checkedMultiply(product, dimension, product))
      return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                         llvm::Twine(role) + " element product overflows");
  return llvm::Error::success();
}

llvm::Error validatePayload(const compiler::TargetTransactionPayload &payload) {
  return std::visit(
      [](const auto &value) -> llvm::Error {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T,
                                     compiler::TargetStridedDMATransaction>) {
          if (value.direction != compiler::TargetDMADirection::Read &&
              value.direction != compiler::TargetDMADirection::Write)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "DMA has an unknown direction");
          if (llvm::Error error = requireEngineFormat(
                  value.format,
                  value.direction == compiler::TargetDMADirection::Read
                      ? TargetFormatEngine::RDMA
                      : TargetFormatEngine::WDMA,
                  "DMA"))
            return error;
          return validateDescriptor(value.byteCount, value.innerBytes,
                                    value.iterations, std::nullopt);
        } else if constexpr (std::is_same_v<
                                 T, compiler::TargetGatherScatterTransaction>) {
          if (llvm::Error error =
                  validateDescriptor(value.byteCount, value.innerBytes,
                                     value.sourceIterations, std::nullopt))
            return error;
          return validateDescriptor(value.byteCount, value.innerBytes,
                                    value.destinationIterations, std::nullopt);
        } else if constexpr (std::is_same_v<
                                 T, compiler::TargetMemsetTransaction>) {
          if (value.elementCount == 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "memset element_count must be positive");
          if (llvm::Error error = requireEngineFormat(
                  value.format, TargetFormatEngine::TDMA, "memset"))
            return error;
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<
                                 T, compiler::TargetBit2FPTransaction> ||
                             std::is_same_v<
                                 T, compiler::TargetMaskMoveTransaction>) {
          if (value.elementCount == 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "CT movement element_count must be positive");
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "CT movement");
        } else if constexpr (std::is_same_v<T,
                                            compiler::TargetGemmTransaction>) {
          if (value.m == 0 || value.k == 0 || value.n == 0 ||
              value.batchCount == 0 ||
              value.m > std::numeric_limits<uint16_t>::max() ||
              value.k > std::numeric_limits<uint16_t>::max() ||
              value.n > std::numeric_limits<uint16_t>::max() ||
              value.batchCount > std::numeric_limits<uint16_t>::max())
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "GEMM dimensions and batch count must be positive uint16");
          return requireEngineFormat(value.format, TargetFormatEngine::NE,
                                     "GEMM");
        } else if constexpr (std::is_same_v<
                                 T, compiler::TargetElementwiseTransaction>) {
          return validateElementwise(value);
        } else if constexpr (std::is_same_v<
                                 T, compiler::TargetReduceTransaction>) {
          if (llvm::Error error = requireKnownEnum(value.kind, "reduce kind"))
            return error;
          if (value.dimension >
              static_cast<uint32_t>(NativeCTReduceDimension::Trailing2And1And0))
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "reduce dimension has an unknown selector");
          if (llvm::Error error =
                  validateShapeProduct(value.nhwc, "reduce shape"))
            return error;
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "reduce");
        } else if constexpr (std::is_same_v<
                                 T, compiler::TargetConvertTransaction>) {
          return validateConvert(value);
        } else if constexpr (std::is_same_v<T,
                                            compiler::TargetConvTransaction>) {
          if (llvm::Error error =
                  requireKnownEnum(value.kind, "convolution kind"))
            return error;
          for (auto [shape, role] :
               {std::pair<llvm::ArrayRef<uint32_t>, llvm::StringRef>(
                    value.inputShape, "convolution input shape"),
                {value.weightShape, "convolution weight shape"},
                {value.outputShape, "convolution output shape"},
                {value.kernelStrides, "convolution strides"},
                {value.dilations, "convolution dilations"}})
            if (llvm::Error error = validateShapeProduct(shape, role))
              return error;
          return requireEngineFormat(value.format, TargetFormatEngine::NE,
                                     "convolution");
        } else if constexpr (std::is_same_v<T,
                                            compiler::TargetPoolTransaction>) {
          if (llvm::Error error = requireKnownEnum(value.kind, "pool kind"))
            return error;
          const bool indexed = value.kind == InstrPoolKind::IndexedMax ||
                               value.kind == InstrPoolKind::IndexedMin;
          if (value.indexDestination.has_value() != indexed)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "pool index destination presence differs from pool kind");
          for (auto [shape, role] :
               {std::pair<llvm::ArrayRef<uint32_t>, llvm::StringRef>(
                    value.sourceShape, "pool source shape"),
                {value.destinationShape, "pool destination shape"},
                {value.kernelStrides, "pool strides"}})
            if (llvm::Error error = validateShapeProduct(shape, role))
              return error;
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "pool");
        } else if constexpr (std::is_same_v<
                                 T, compiler::TargetUnpoolTransaction>) {
          if (llvm::Error error = requireKnownEnum(value.kind, "unpool kind"))
            return error;
          if (value.index.has_value() == (value.kind == InstrUnpoolKind::Avg))
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "unpool index presence differs from unpool kind");
          for (auto [shape, role] :
               {std::pair<llvm::ArrayRef<uint32_t>, llvm::StringRef>(
                    value.sourceShape, "unpool source shape"),
                {value.destinationShape, "unpool destination shape"},
                {value.kernelStrides, "unpool strides"}})
            if (llvm::Error error = validateShapeProduct(shape, role))
              return error;
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "unpool");
        } else if constexpr (std::is_same_v<
                                 T, compiler::TargetTDMATransformTransaction>) {
          if (value.kind != compiler::TargetTDMATransformKind::Pad &&
              value.kind != compiler::TargetTDMATransformKind::ImageToColumn)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "TDMA transform has an unknown kind");
          if (value.kernelStrides.has_value() !=
              (value.kind == compiler::TargetTDMATransformKind::ImageToColumn))
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "TDMA kernel strides presence differs from transform kind");
          if (llvm::Error error =
                  validateShapeProduct(value.sourceShape, "TDMA source shape"))
            return error;
          if (llvm::Error error = validateShapeProduct(
                  value.destinationShape, "TDMA destination shape"))
            return error;
          if (value.kernelStrides)
            if (llvm::Error error =
                    validateShapeProduct(*value.kernelStrides, "TDMA strides"))
              return error;
          return requireEngineFormat(value.format, TargetFormatEngine::TDMA,
                                     "TDMA transform");
        } else if constexpr (
            std::is_same_v<T,
                           compiler::TargetPeripheralArgExtremaTransaction>) {
          if (value.kind != InstrPeripheralKind::ArgMax &&
              value.kind != InstrPeripheralKind::ArgMin)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "arg-extrema payload carries a different peripheral kind");
          if (value.elementCount == 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "arg-extrema element_count must be positive");
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "arg-extrema");
        } else if constexpr (
            std::is_same_v<T, compiler::TargetPeripheralBilinearTransaction>) {
          if (value.elementCount == 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "bilinear element_count must be positive");
          if (llvm::Error error = validateShapeProduct(value.sourceShape,
                                                       "bilinear source shape"))
            return error;
          if (llvm::Error error = validateShapeProduct(
                  value.destinationShape, "bilinear destination shape"))
            return error;
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "bilinear");
        } else if constexpr (std::is_same_v<
                                 T, compiler::TargetPeripheralLUTTransaction>) {
          if (value.kind != InstrPeripheralKind::Lut16 &&
              value.kind != InstrPeripheralKind::Lut32)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "LUT payload carries a different peripheral kind");
          if (value.elementCount == 0 || value.tableElementCount == 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "LUT element counts must be positive");
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "LUT");
        } else if constexpr (
            std::is_same_v<T, compiler::TargetPeripheralRandomTransaction> ||
            std::is_same_v<T,
                           compiler::TargetPeripheralElementMaskTransaction>) {
          if (value.elementCount == 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "peripheral element_count must be positive");
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "peripheral");
        } else if constexpr (std::is_same_v<
                                 T,
                                 compiler::TargetDirectDTEBeginTransaction>) {
          if (value.rankCount == 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "Direct DTE begin rank_count must be positive");
          return llvm::Error::success();
        } else if constexpr (
            std::is_same_v<T, compiler::TargetDirectDTESendTransaction> ||
            std::is_same_v<T, compiler::TargetDirectDTEReceiveTransaction>) {
          if (value.byteCount == 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "Direct DTE byte_count must be positive");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<
                                 T, compiler::TargetDirectDTEWaitTransaction>) {
          if (value.event == 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "Direct DTE wait event must be nonzero");
          return llvm::Error::success();
        } else {
          return llvm::Error::success();
        }
      },
      payload);
}

} // namespace
} // namespace wafer::model::kernel_detail

namespace wafer::model {
using namespace kernel_detail;

llvm::StringRef
stringifyTargetModelKernelErrorCode(TargetModelKernelErrorCode code) {
  switch (code) {
  case TargetModelKernelErrorCode::InvalidTransactionField:
    return "invalid-transaction-field";
  case TargetModelKernelErrorCode::WorkBudgetExceeded:
    return "work-budget-exceeded";
  case TargetModelKernelErrorCode::UnsupportedTransaction:
    return "unsupported-transaction";
  case TargetModelKernelErrorCode::MemoryReadFailure:
    return "memory-read-failure";
  case TargetModelKernelErrorCode::NumericResolutionFailure:
    return "numeric-resolution-failure";
  case TargetModelKernelErrorCode::PhysicalCodecFailure:
    return "physical-codec-failure";
  case TargetModelKernelErrorCode::ManagedReferenceBackendUnavailable:
    return "managed-reference-backend-unavailable";
  case TargetModelKernelErrorCode::ManagedReferenceBackendFailure:
    return "managed-reference-backend-failure";
  case TargetModelKernelErrorCode::BulkBackendUnavailable:
    return "bulk-backend-unavailable";
  case TargetModelKernelErrorCode::BulkBackendFailure:
    return "bulk-backend-failure";
  }
  llvm_unreachable("unknown target model kernel error code");
}

char TargetModelKernelError::ID;

void TargetModelKernelError::log(llvm::raw_ostream &stream) const {
  stream << "target model kernel " << stringifyTargetModelKernelErrorCode(code)
         << ": " << detail;
}

std::error_code TargetModelKernelError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Error validateTargetModelTransactionFields(
    const compiler::TargetTransaction &transaction) {
  if (transaction.logicalRank < 0)
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "transaction logical rank must be non-negative");
  return validatePayload(transaction.payload);
}

} // namespace wafer::model
