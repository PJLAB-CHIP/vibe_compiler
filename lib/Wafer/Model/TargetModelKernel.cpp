//===- TargetModelKernel.cpp - Plain target transaction kernels ---------===//

#include "Wafer/Model/TargetModelKernel.h"

#include "Wafer/Target/PhysicalTensorCodec.h"
#include "Wafer/Target/TargetFormat.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <type_traits>
#include <variant>

namespace wafer::model {
namespace {

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

llvm::Expected<std::vector<uint64_t>> getSegmentAddresses(
    uint64_t base, uint32_t innerBytes, const std::array<uint32_t, 3> &strides,
    const std::array<uint32_t, 3> &iterations, TargetModelKernelBudget budget) {
  llvm::Expected<uint64_t> count = getDescriptorSegmentCount(iterations);
  if (!count)
    return count.takeError();
  if (*count > budget.getMaximumMovementSegments())
    return kernelError(TargetModelKernelErrorCode::WorkBudgetExceeded,
                       "movement segment count exceeds the explicit budget");
  std::vector<uint64_t> addresses;
  addresses.reserve(static_cast<size_t>(*count));
  for (uint64_t outer = 0; outer < iterations[2]; ++outer)
    for (uint64_t middle = 0; middle < iterations[1]; ++middle)
      for (uint64_t inner = 0; inner < iterations[0]; ++inner) {
        uint64_t address = base;
        for (auto [index, coordinate] :
             llvm::enumerate(std::array<uint64_t, 3>{inner, middle, outer})) {
          uint64_t delta = 0;
          if (!checkedMultiply(coordinate, strides[index], delta) ||
              !checkedAdd(address, delta, address) ||
              !checkedAdd(address, innerBytes, delta))
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "movement segment address overflows");
        }
        addresses.push_back(address);
      }
  return addresses;
}

llvm::Expected<std::vector<uint8_t>>
readSnapshot(const InvocationMemoryRegistry &memory, int64_t rank,
             TargetModelAddressSpace space, uint64_t address, uint64_t bytes,
             uint64_t alignment = 1) {
  llvm::Expected<std::vector<uint8_t>> result =
      memory.readSnapshot(rank, space, address, bytes, alignment);
  if (!result)
    return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                       llvm::toString(result.takeError()));
  return result;
}

llvm::Expected<TargetModelCommandEffect>
executeMovement(const compiler::TargetTransaction &transaction,
                const compiler::TargetStridedDMATransaction &value,
                const InvocationMemoryRegistry &memory,
                TargetModelKernelBudget budget) {
  if (value.byteCount > budget.getMaximumMovementBytes())
    return kernelError(TargetModelKernelErrorCode::WorkBudgetExceeded,
                       "DMA byte_count exceeds the explicit budget");
  llvm::Expected<std::vector<uint64_t>> addresses = getSegmentAddresses(
      value.direction == compiler::TargetDMADirection::Read ? value.source
                                                            : value.destination,
      value.innerBytes, value.strides, value.iterations, budget);
  if (!addresses)
    return addresses.takeError();

  if (value.direction == compiler::TargetDMADirection::Read) {
    std::vector<uint8_t> payload;
    payload.reserve(value.byteCount);
    for (uint64_t address : *addresses) {
      llvm::Expected<std::vector<uint8_t>> segment = readSnapshot(
          memory, transaction.logicalRank, TargetModelAddressSpace::CardDDR,
          address, value.innerBytes);
      if (!segment)
        return segment.takeError();
      payload.insert(payload.end(), segment->begin(), segment->end());
    }
    return TargetModelCommandEffect{
        {TargetModelByteWrite{transaction.logicalRank,
                              TargetModelAddressSpace::RankSPM,
                              value.destination, 1, std::move(payload)}},
        {},
        TargetModelControlAction::None};
  }

  llvm::Expected<std::vector<uint8_t>> payload = readSnapshot(
      memory, transaction.logicalRank, TargetModelAddressSpace::RankSPM,
      value.source, value.byteCount);
  if (!payload)
    return payload.takeError();
  std::vector<TargetModelByteWrite> writes;
  writes.reserve(addresses->size());
  for (auto [index, address] : llvm::enumerate(*addresses)) {
    const size_t begin = index * value.innerBytes;
    writes.push_back(TargetModelByteWrite{
        transaction.logicalRank, TargetModelAddressSpace::CardDDR, address, 1,
        std::vector<uint8_t>(payload->begin() + begin,
                             payload->begin() + begin + value.innerBytes)});
  }
  return TargetModelCommandEffect{
      std::move(writes), {}, TargetModelControlAction::None};
}

llvm::Expected<TargetModelCommandEffect>
executeGatherScatter(const compiler::TargetTransaction &transaction,
                     const compiler::TargetGatherScatterTransaction &value,
                     const InvocationMemoryRegistry &memory,
                     TargetModelKernelBudget budget) {
  if (value.byteCount > budget.getMaximumMovementBytes())
    return kernelError(TargetModelKernelErrorCode::WorkBudgetExceeded,
                       "gather/scatter byte_count exceeds explicit budget");
  llvm::Expected<std::vector<uint64_t>> sourceAddresses =
      getSegmentAddresses(value.source, value.innerBytes, value.sourceStrides,
                          value.sourceIterations, budget);
  if (!sourceAddresses)
    return sourceAddresses.takeError();
  llvm::Expected<std::vector<uint64_t>> destinationAddresses =
      getSegmentAddresses(value.destination, value.innerBytes,
                          value.destinationStrides, value.destinationIterations,
                          budget);
  if (!destinationAddresses)
    return destinationAddresses.takeError();
  if (sourceAddresses->size() != destinationAddresses->size())
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "gather/scatter source and destination segment counts "
                       "differ");
  std::vector<std::vector<uint8_t>> snapshots;
  snapshots.reserve(sourceAddresses->size());
  for (uint64_t address : *sourceAddresses) {
    llvm::Expected<std::vector<uint8_t>> segment = readSnapshot(
        memory, transaction.logicalRank, TargetModelAddressSpace::RankSPM,
        address, value.innerBytes);
    if (!segment)
      return segment.takeError();
    snapshots.push_back(std::move(*segment));
  }
  std::vector<TargetModelByteWrite> writes;
  writes.reserve(destinationAddresses->size());
  for (auto [index, address] : llvm::enumerate(*destinationAddresses))
    writes.push_back({transaction.logicalRank, TargetModelAddressSpace::RankSPM,
                      address, 1, std::move(snapshots[index])});
  return TargetModelCommandEffect{
      std::move(writes), {}, TargetModelControlAction::None};
}

llvm::Expected<NumericElementwiseOperation>
getNumericOperation(InstrElementwiseKind kind) {
#define WAFER_ELEMENTWISE_CASE(NAME)                                           \
  case InstrElementwiseKind::NAME:                                             \
    return NumericElementwiseOperation::NAME
  switch (kind) {
    WAFER_ELEMENTWISE_CASE(Abs);
    WAFER_ELEMENTWISE_CASE(Recip);
    WAFER_ELEMENTWISE_CASE(Square);
    WAFER_ELEMENTWISE_CASE(Sqrt);
    WAFER_ELEMENTWISE_CASE(Rsqrt);
    WAFER_ELEMENTWISE_CASE(Neg);
    WAFER_ELEMENTWISE_CASE(Max);
    WAFER_ELEMENTWISE_CASE(Min);
    WAFER_ELEMENTWISE_CASE(Add);
    WAFER_ELEMENTWISE_CASE(Sub);
    WAFER_ELEMENTWISE_CASE(Mul);
    WAFER_ELEMENTWISE_CASE(Div);
    WAFER_ELEMENTWISE_CASE(Eq);
    WAFER_ELEMENTWISE_CASE(Ne);
    WAFER_ELEMENTWISE_CASE(Ge);
    WAFER_ELEMENTWISE_CASE(Gt);
    WAFER_ELEMENTWISE_CASE(Le);
    WAFER_ELEMENTWISE_CASE(Lt);
    WAFER_ELEMENTWISE_CASE(LogicNot);
    WAFER_ELEMENTWISE_CASE(LogicAnd);
    WAFER_ELEMENTWISE_CASE(LogicOr);
    WAFER_ELEMENTWISE_CASE(LogicXor);
    WAFER_ELEMENTWISE_CASE(Log2);
    WAFER_ELEMENTWISE_CASE(Ln);
    WAFER_ELEMENTWISE_CASE(Pow2);
    WAFER_ELEMENTWISE_CASE(Exp);
    WAFER_ELEMENTWISE_CASE(ExpLp);
    WAFER_ELEMENTWISE_CASE(Sin);
    WAFER_ELEMENTWISE_CASE(Cos);
    WAFER_ELEMENTWISE_CASE(Tanh);
    WAFER_ELEMENTWISE_CASE(Sigmoid);
    WAFER_ELEMENTWISE_CASE(Relu);
    WAFER_ELEMENTWISE_CASE(SatRelu);
    WAFER_ELEMENTWISE_CASE(LeakyRelu);
    WAFER_ELEMENTWISE_CASE(Softplus);
  }
#undef WAFER_ELEMENTWISE_CASE
  return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                     "elementwise kind has no numeric operation mapping");
}

NumericTensorLayout getCountLayout(LogicalFormat format) {
  return format == LogicalFormat::Bool ? NumericTensorLayout::Tensor
                                       : NumericTensorLayout::Cx;
}

llvm::Expected<NumericTensorKey> makeTensor(LogicalFormat format,
                                            std::vector<uint64_t> shape) {
  llvm::Expected<NumericTensorKey> key = NumericTensorKey::create(
      format, getCountLayout(format), std::move(shape));
  if (!key)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(key.takeError()));
  return key;
}

llvm::Expected<std::vector<RawLogicalValue>>
readTensor(const InvocationMemoryRegistry &memory, int64_t rank,
           uint64_t address, const NumericTensorKey &key) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(bytes.takeError()));
  llvm::Expected<std::vector<uint8_t>> storage = readSnapshot(
      memory, rank, TargetModelAddressSpace::RankSPM, address, *bytes);
  if (!storage)
    return storage.takeError();
  llvm::Expected<std::vector<RawLogicalValue>> values =
      unpackPhysicalTensorLogicalValues(key, *storage);
  if (!values)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(values.takeError()));
  return values;
}

llvm::Expected<std::vector<uint8_t>>
packTensor(const InvocationMemoryRegistry &memory, int64_t rank,
           uint64_t address, const NumericTensorKey &key,
           llvm::ArrayRef<RawLogicalValue> values) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(bytes.takeError()));
  llvm::Expected<std::vector<uint8_t>> storage = readSnapshot(
      memory, rank, TargetModelAddressSpace::RankSPM, address, *bytes);
  if (!storage)
    return storage.takeError();
  llvm::Expected<std::vector<uint8_t>> packed =
      packPhysicalTensorLogicalValues(key, values, *storage);
  if (!packed)
    return kernelError(TargetModelKernelErrorCode::PhysicalCodecFailure,
                       llvm::toString(packed.takeError()));
  return packed;
}

llvm::Expected<ResolvedNumericCommand>
resolveNumeric(NumericCommandKey command) {
  llvm::Expected<ResolvedNumericCommand> resolved = resolveNumericCommand(
      ModelProfileId::formalDeterministicV1(), std::move(command));
  if (!resolved)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(resolved.takeError()));
  return resolved;
}

llvm::Expected<TargetModelCommandEffect>
executeElementwise(const compiler::TargetTransaction &transaction,
                   const compiler::TargetElementwiseTransaction &value,
                   const InvocationMemoryRegistry &memory,
                   TargetModelKernelBudget budget) {
  llvm::Expected<NumericElementwiseOperation> operation =
      getNumericOperation(value.kind);
  if (!operation)
    return operation.takeError();
  const LogicalFormat destinationFormat =
      isNumericElementwiseRelation(*operation) ? LogicalFormat::Bool
                                               : value.format;
  llvm::Expected<NumericTensorKey> inputKey =
      makeTensor(value.format, {value.elementCount});
  llvm::Expected<NumericTensorKey> destinationKey =
      makeTensor(destinationFormat, {value.elementCount});
  if (!inputKey)
    return inputKey.takeError();
  if (!destinationKey)
    return destinationKey.takeError();
  std::vector<NumericTensorKey> inputKeys;
  inputKeys.push_back(*inputKey);
  if (value.rhs)
    inputKeys.push_back(*inputKey);
  llvm::Expected<NumericCommandKey> command =
      NumericCommandKey::createCTElementwise(
          memory.getAddressPlan().getTargetProfile(), *operation,
          std::move(inputKeys), *destinationKey);
  if (!command)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(command.takeError()));
  llvm::Expected<ResolvedNumericCommand> resolved =
      resolveNumeric(std::move(*command));
  if (!resolved)
    return resolved.takeError();

  std::vector<std::vector<RawLogicalValue>> inputs;
  llvm::Expected<std::vector<RawLogicalValue>> lhs =
      readTensor(memory, transaction.logicalRank, value.lhs, *inputKey);
  if (!lhs)
    return lhs.takeError();
  inputs.push_back(std::move(*lhs));
  if (value.rhs) {
    llvm::Expected<std::vector<RawLogicalValue>> rhs =
        readTensor(memory, transaction.logicalRank, *value.rhs, *inputKey);
    if (!rhs)
      return rhs.takeError();
    inputs.push_back(std::move(*rhs));
  }
  std::vector<llvm::ArrayRef<RawLogicalValue>> views;
  for (const std::vector<RawLogicalValue> &input : inputs)
    views.push_back(input);
  FormalNumericExecutionContext localContext;
  llvm::Expected<FormalTensorNumericResult> result = executeFormalTensorNumeric(
      localContext, *resolved, views, budget.getNumericBudget());
  if (!result)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(result.takeError()));
  llvm::Expected<std::vector<uint8_t>> packed =
      packTensor(memory, transaction.logicalRank, value.destination,
                 *destinationKey, result->values);
  if (!packed)
    return packed.takeError();
  return TargetModelCommandEffect{
      {TargetModelByteWrite{transaction.logicalRank,
                            TargetModelAddressSpace::RankSPM, value.destination,
                            1, std::move(*packed)}},
      result->flags,
      TargetModelControlAction::None};
}

llvm::Expected<TargetModelCommandEffect>
executeConvert(const compiler::TargetTransaction &transaction,
               const compiler::TargetConvertTransaction &value,
               const InvocationMemoryRegistry &memory,
               TargetModelKernelBudget budget) {
  const uint16_t opcode = static_cast<uint16_t>(value.kind);
  const TargetConvertRoute *route = findTargetConvertRoute(
      memory.getAddressPlan().getTargetProfile(), opcode);
  if (!route)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       "convert route is not registered for target profile");
  llvm::Expected<NumericTensorKey> sourceKey =
      makeTensor(route->source, {value.elementCount});
  llvm::Expected<NumericTensorKey> destinationKey =
      makeTensor(route->destination, {value.elementCount});
  if (!sourceKey)
    return sourceKey.takeError();
  if (!destinationKey)
    return destinationKey.takeError();
  std::optional<NumericConvertParameter> parameter;
  if (value.zeroPoint)
    parameter = NumericConvertParameter::zeroPoint(*value.zeroPoint);
  if (value.roundingMode) {
    llvm::Expected<NumericRoundingMode> mode =
        parseNumericRoundingMode(static_cast<uint8_t>(*value.roundingMode));
    if (!mode)
      return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                         llvm::toString(mode.takeError()));
    parameter = NumericConvertParameter::roundingMode(*mode);
  }
  llvm::Expected<NumericCommandKey> command =
      NumericCommandKey::createCTConvert(
          memory.getAddressPlan().getTargetProfile(), opcode, *sourceKey,
          *destinationKey, parameter);
  if (!command)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(command.takeError()));
  llvm::Expected<ResolvedNumericCommand> resolved =
      resolveNumeric(std::move(*command));
  if (!resolved)
    return resolved.takeError();
  llvm::Expected<std::vector<RawLogicalValue>> source =
      readTensor(memory, transaction.logicalRank, value.source, *sourceKey);
  if (!source)
    return source.takeError();
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*source};
  FormalNumericExecutionContext localContext;
  llvm::Expected<FormalTensorNumericResult> result = executeFormalTensorNumeric(
      localContext, *resolved, views, budget.getNumericBudget());
  if (!result)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(result.takeError()));
  llvm::Expected<std::vector<uint8_t>> packed =
      packTensor(memory, transaction.logicalRank, value.destination,
                 *destinationKey, result->values);
  if (!packed)
    return packed.takeError();
  return TargetModelCommandEffect{
      {TargetModelByteWrite{transaction.logicalRank,
                            TargetModelAddressSpace::RankSPM, value.destination,
                            1, std::move(*packed)}},
      result->flags,
      TargetModelControlAction::None};
}

llvm::Expected<TargetModelCommandEffect>
executeGemm(const compiler::TargetTransaction &transaction,
            const compiler::TargetGemmTransaction &value,
            const InvocationMemoryRegistry &memory,
            TargetModelKernelBudget budget) {
  const bool batched = value.batchCount > 1;
  const NumericTensorLayout layout =
      batched ? NumericTensorLayout::NCx : NumericTensorLayout::Cx;
  std::vector<uint64_t> lhsShape =
      batched ? std::vector<uint64_t>{value.batchCount, value.m, value.k}
              : std::vector<uint64_t>{value.m, value.k};
  std::vector<uint64_t> rhsShape =
      batched ? std::vector<uint64_t>{value.batchCount, value.k, value.n}
              : std::vector<uint64_t>{value.k, value.n};
  std::vector<uint64_t> destinationShape =
      batched ? std::vector<uint64_t>{value.batchCount, value.m, value.n}
              : std::vector<uint64_t>{value.m, value.n};
  llvm::Expected<NumericTensorKey> lhsKey =
      NumericTensorKey::create(value.format, layout, std::move(lhsShape));
  llvm::Expected<NumericTensorKey> rhsKey =
      NumericTensorKey::create(value.format, layout, std::move(rhsShape));
  llvm::Expected<NumericTensorKey> destinationKey = NumericTensorKey::create(
      value.format, layout, std::move(destinationShape));
  if (!lhsKey || !rhsKey || !destinationKey) {
    llvm::Error errors = llvm::Error::success();
    if (!lhsKey)
      errors = llvm::joinErrors(std::move(errors), lhsKey.takeError());
    if (!rhsKey)
      errors = llvm::joinErrors(std::move(errors), rhsKey.takeError());
    if (!destinationKey)
      errors = llvm::joinErrors(std::move(errors), destinationKey.takeError());
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(std::move(errors)));
  }
  const uint64_t rank = lhsKey->getShape().size();
  llvm::Expected<NumericGemmAxes> axes = getCanonicalNumericGemmAxes(rank);
  if (!axes)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(axes.takeError()));
  llvm::Expected<NumericCommandKey> command = NumericCommandKey::createNEGemm(
      memory.getAddressPlan().getTargetProfile(), *lhsKey, *rhsKey,
      *destinationKey, value.m, value.k, value.n, value.batchCount, *axes);
  if (!command)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(command.takeError()));
  llvm::Expected<ResolvedNumericCommand> resolved =
      resolveNumeric(std::move(*command));
  if (!resolved)
    return resolved.takeError();
  llvm::Expected<std::vector<RawLogicalValue>> lhs =
      readTensor(memory, transaction.logicalRank, value.lhs, *lhsKey);
  llvm::Expected<std::vector<RawLogicalValue>> rhs =
      readTensor(memory, transaction.logicalRank, value.rhs, *rhsKey);
  if (!lhs)
    return lhs.takeError();
  if (!rhs)
    return rhs.takeError();
  std::vector<llvm::ArrayRef<RawLogicalValue>> views{*lhs, *rhs};
  FormalNumericExecutionContext localContext;
  llvm::Expected<FormalTensorNumericResult> result = executeFormalTensorNumeric(
      localContext, *resolved, views, budget.getNumericBudget());
  if (!result)
    return kernelError(TargetModelKernelErrorCode::NumericResolutionFailure,
                       llvm::toString(result.takeError()));
  llvm::Expected<std::vector<uint8_t>> packed =
      packTensor(memory, transaction.logicalRank, value.destination,
                 *destinationKey, result->values);
  if (!packed)
    return packed.takeError();
  return TargetModelCommandEffect{
      {TargetModelByteWrite{transaction.logicalRank,
                            TargetModelAddressSpace::RankSPM, value.destination,
                            1, std::move(*packed)}},
      result->flags,
      TargetModelControlAction::None};
}

llvm::Expected<TargetModelCommandEffect>
executeMemset(const compiler::TargetTransaction &transaction,
              const compiler::TargetMemsetTransaction &value,
              const InvocationMemoryRegistry &memory) {
  llvm::Expected<NumericTensorKey> key =
      makeTensor(value.format, {value.elementCount});
  if (!key)
    return key.takeError();
  const LogicalScalarCodecPolicy policy =
      getModelProfileRecord(ModelProfileId::formalDeterministicV1())
          .numericEncodePolicy;
  llvm::Expected<RawLogicalValue> scalar = makeRawLogicalValue(
      value.format, value.value, policy.nonCanonicalEncoding);
  if (!scalar)
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       llvm::toString(scalar.takeError()));
  std::vector<RawLogicalValue> values(value.elementCount, *scalar);
  llvm::Expected<std::vector<uint8_t>> packed = packTensor(
      memory, transaction.logicalRank, value.destination, *key, values);
  if (!packed)
    return packed.takeError();
  return TargetModelCommandEffect{
      {TargetModelByteWrite{transaction.logicalRank,
                            TargetModelAddressSpace::RankSPM, value.destination,
                            1, std::move(*packed)}},
      {},
      TargetModelControlAction::None};
}

TargetModelControlAction
getControlAction(const compiler::TargetTransactionPayload &payload) {
  if (std::holds_alternative<compiler::TargetLocalFenceTransaction>(payload))
    return TargetModelControlAction::LocalFence;
  if (std::holds_alternative<compiler::TargetDirectDTEBeginTransaction>(
          payload))
    return TargetModelControlAction::DirectDTEBegin;
  if (std::holds_alternative<compiler::TargetDirectDTESendTransaction>(payload))
    return TargetModelControlAction::DirectDTESend;
  if (std::holds_alternative<compiler::TargetDirectDTEReceiveTransaction>(
          payload))
    return TargetModelControlAction::DirectDTEReceive;
  if (std::holds_alternative<compiler::TargetDirectDTEWaitTransaction>(payload))
    return TargetModelControlAction::DirectDTEWait;
  if (std::holds_alternative<compiler::TargetDirectDTEFinishTransaction>(
          payload))
    return TargetModelControlAction::DirectDTEFinish;
  return TargetModelControlAction::None;
}

llvm::Error
validateControlAddresses(const compiler::TargetTransaction &transaction,
                         const InvocationAddressPlan &plan) {
  if (const auto *begin =
          std::get_if<compiler::TargetDirectDTEBeginTransaction>(
              &transaction.payload)) {
    if (begin->rankCount != plan.getLogicalRanks().size())
      return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                         "Direct DTE rank_count differs from invocation");
    llvm::Expected<TargetModelResolvedRange> status =
        plan.resolve(transaction.logicalRank, TargetModelAddressSpace::CardDDR,
                     TargetModelAccess::ReadWrite, begin->statusAddress, 4, 4);
    if (!status)
      return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                         llvm::toString(status.takeError()));
  } else if (const auto *send =
                 std::get_if<compiler::TargetDirectDTESendTransaction>(
                     &transaction.payload)) {
    if (send->localTile > std::numeric_limits<uint16_t>::max() ||
        send->remoteTile > std::numeric_limits<uint16_t>::max() ||
        send->remoteFSM >= 4)
      return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                         "Direct DTE send tile or receiver FSM is outside "
                         "the accepted target ABI domain");
    if (send->highPerformance)
      return kernelError(
          TargetModelKernelErrorCode::UnsupportedTransaction,
          "Direct DTE high-performance allocation is outside the accepted "
          "normal sender profile");
    llvm::Expected<TargetModelResolvedRange> source =
        plan.resolve(transaction.logicalRank, TargetModelAddressSpace::RankSPM,
                     TargetModelAccess::Read, send->source, send->byteCount, 1);
    llvm::Expected<TargetModelResolvedRange> destination = plan.resolve(
        transaction.logicalRank, TargetModelAddressSpace::RankSPM,
        TargetModelAccess::Write, send->remoteDestination, send->byteCount, 1);
    if (!source || !destination) {
      llvm::Error errors = llvm::Error::success();
      if (!source)
        errors = llvm::joinErrors(std::move(errors), source.takeError());
      if (!destination)
        errors = llvm::joinErrors(std::move(errors), destination.takeError());
      return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                         llvm::toString(std::move(errors)));
    }
  } else if (const auto *receive =
                 std::get_if<compiler::TargetDirectDTEReceiveTransaction>(
                     &transaction.payload)) {
    if (receive->localTile > std::numeric_limits<uint16_t>::max() ||
        receive->remoteTile > std::numeric_limits<uint16_t>::max() ||
        receive->localFSM >= 4)
      return kernelError(
          TargetModelKernelErrorCode::InvalidTransactionField,
          "Direct DTE receive tile or receiver FSM is outside the accepted "
          "target ABI domain");
    llvm::Expected<TargetModelResolvedRange> destination = plan.resolve(
        transaction.logicalRank, TargetModelAddressSpace::RankSPM,
        TargetModelAccess::Write, receive->destination, receive->byteCount, 1);
    if (!destination)
      return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                         llvm::toString(destination.takeError()));
  }
  return llvm::Error::success();
}

} // namespace

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

llvm::Expected<TargetModelCommandEffect>
executeTargetModelCommand(const compiler::TargetTransaction &transaction,
                          const InvocationMemoryRegistry &memory,
                          TargetModelKernelBudget budget) {
  if (llvm::Error error = validateTargetModelTransactionFields(transaction))
    return std::move(error);
  if (!llvm::is_contained(memory.getAddressPlan().getLogicalRanks(),
                          transaction.logicalRank))
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "transaction logical rank is outside invocation");

  if (const auto *value = std::get_if<compiler::TargetStridedDMATransaction>(
          &transaction.payload))
    return executeMovement(transaction, *value, memory, budget);
  if (const auto *value = std::get_if<compiler::TargetGatherScatterTransaction>(
          &transaction.payload))
    return executeGatherScatter(transaction, *value, memory, budget);
  if (const auto *value =
          std::get_if<compiler::TargetMemsetTransaction>(&transaction.payload))
    return executeMemset(transaction, *value, memory);
  if (const auto *value = std::get_if<compiler::TargetElementwiseTransaction>(
          &transaction.payload))
    return executeElementwise(transaction, *value, memory, budget);
  if (const auto *value =
          std::get_if<compiler::TargetConvertTransaction>(&transaction.payload))
    return executeConvert(transaction, *value, memory, budget);
  if (const auto *value =
          std::get_if<compiler::TargetGemmTransaction>(&transaction.payload))
    return executeGemm(transaction, *value, memory, budget);

  TargetModelControlAction control = getControlAction(transaction.payload);
  if (control != TargetModelControlAction::None) {
    if (llvm::Error error =
            validateControlAddresses(transaction, memory.getAddressPlan()))
      return std::move(error);
    return TargetModelCommandEffect{{}, {}, control};
  }
  return kernelError(
      TargetModelKernelErrorCode::UnsupportedTransaction,
      "typed transaction is field-valid but has no evidence-backed plain "
      "functional kernel");
}

llvm::Error
commitTargetModelCommandEffect(InvocationMemoryRegistry &memory,
                               FormalNumericExecutionContext &context,
                               TargetModelCommandEffect effect) {
  if (effect.controlAction != TargetModelControlAction::None &&
      !effect.pendingWrites.empty())
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "control effect cannot carry immediate byte writes");
  if (llvm::Error error = memory.applyAtomically(effect.pendingWrites))
    return error;
  context.recordCommittedFlags(effect.numericFlags);
  return llvm::Error::success();
}

} // namespace wafer::model
