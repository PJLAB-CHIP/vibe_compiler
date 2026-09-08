//===- TargetModelCommandValidation.cpp - Command validation -----===//

#include "TargetModelKernelInternal.h"

#include "Wafer/Target/TargetFormat.h"
#include "Wafer/Target/Tx81InstructionLimits.h"

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
      return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                         "movement iteration counts must be positive");
    if (!checkedMultiply(count, iteration, count))
      return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                         "movement segment count overflows");
  }
  return count;
}

namespace {

llvm::Error requireFormat(LogicalFormat format, llvm::StringRef role) {
  if (!findLogicalFormatDescriptor(format))
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       llvm::Twine(role) + " has an unknown logical format");
  return llvm::Error::success();
}

llvm::Error requireEngineFormat(LogicalFormat format, TargetFormatEngine engine,
                                llvm::StringRef role) {
  if (llvm::Error error = requireFormat(format, role))
    return error;
  const TargetFormatEncodingRecord *record =
      findTargetFormatEncoding(engine, format);
  if (!record)
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       llvm::Twine(role) +
                           " format is unsupported by its target engine");
  return llvm::Error::success();
}

llvm::Error requirePositive(llvm::ArrayRef<uint32_t> values,
                            llvm::StringRef role) {
  if (llvm::is_contained(values, UINT32_C(0)))
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       llvm::Twine(role) + " must be strictly positive");
  return llvm::Error::success();
}

llvm::Error validateDescriptor(uint32_t byteCount, uint32_t innerBytes,
                               const std::array<uint32_t, 3> &iterations,
                               std::optional<LogicalFormat> format) {
  if (byteCount == 0 || innerBytes == 0)
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       "movement byte_count and inner_bytes must be positive");
  llvm::Expected<uint64_t> segments = getDescriptorSegmentCount(iterations);
  if (!segments)
    return segments.takeError();
  uint64_t payload = 0;
  if (!checkedMultiply(innerBytes, *segments, payload) || payload != byteCount)
    return kernelError(
        TargetModelKernelErrorCode::InvalidCommandField,
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
          TargetModelKernelErrorCode::InvalidCommandField,
          "movement inner_bytes is not a whole number of logical elements");
  }
  return llvm::Error::success();
}

template <typename Enum>
llvm::Error requireRegisteredOperation(Enum value, llvm::ArrayRef<Enum> values,
                                       llvm::StringRef role) {
  if (!llvm::is_contained(values, value))
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       llvm::Twine(role) + " is not registered");
  return llvm::Error::success();
}

llvm::Error validateConvert(const target::TargetConvertCommand &value) {
  if (value.elementCount == 0)
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       "convert element_count must be positive");
  const uint16_t opcode = value.operation.getOpcode();
  const TargetConvertRoute *route = findTargetConvertRoute(opcode);
  if (!route)
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       "convert kind has no exact target route");
  switch (route->parameterKind) {
  case TargetConvertParameterKind::None:
    if (value.parameter)
      return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                         "parameterless convert carries an optional field");
    break;
  case TargetConvertParameterKind::RoundingMode:
    if (!value.parameter || !value.parameter->getRoundingMode())
      return kernelError(
          TargetModelKernelErrorCode::InvalidCommandField,
          "rounding convert requires exactly one known rounding mode");
    break;
  case TargetConvertParameterKind::ZeroPoint:
    if (!value.parameter || !value.parameter->getZeroPoint())
      return kernelError(
          TargetModelKernelErrorCode::InvalidCommandField,
          "zero-point convert requires exactly one uint32 zero point");
    break;
  }
  return llvm::Error::success();
}

llvm::Error validateElementwise(const target::TargetElementwiseCommand &value) {
  if (value.elementCount == 0)
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       "elementwise element_count must be positive");
  if (llvm::Error error = requireRegisteredOperation(
          value.operation, getTargetElementwiseOperations(),
          "elementwise operation"))
    return error;
  if (llvm::Error error = requireEngineFormat(
          value.format, TargetFormatEngine::CT, "elementwise"))
    return error;
  const bool binary = getTargetElementwiseArity(value.operation) == 2;
  if (value.rhs.has_value() != binary)
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       "elementwise rhs presence differs from operation arity");
  const bool logic = isTargetElementwiseLogic(value.operation);
  const bool relation = isTargetElementwiseRelation(value.operation);
  if (logic != (value.format == LogicalFormat::Bool) ||
      (relation && value.format == LogicalFormat::Bool))
    return kernelError(
        TargetModelKernelErrorCode::InvalidCommandField,
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
      return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                         llvm::Twine(role) + " element product overflows");
  return llvm::Error::success();
}

llvm::Error validateInclusiveRange(llvm::ArrayRef<uint32_t> values,
                                   llvm::StringRef role, uint32_t minimum,
                                   uint32_t maximum) {
  for (uint32_t value : values)
    if (value < minimum || value > maximum)
      return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                         llvm::Twine(role) + " must be in [" +
                             llvm::Twine(minimum) + ", " +
                             llvm::Twine(maximum) + "]");
  return llvm::Error::success();
}

llvm::Error validateDataShape(llvm::ArrayRef<uint32_t> shape,
                              llvm::StringRef role) {
  if (shape.size() != 4)
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       llvm::Twine(role) + " must contain exactly 4 entries");
  if (llvm::Error error = validateShapeProduct(shape, role))
    return error;

  constexpr std::array<llvm::StringLiteral, 4> dimensionNames = {"N", "H", "W",
                                                                 "C"};
  constexpr std::array<uint32_t, 4> dimensionMaximums = {
      Tx81InstructionLimits::dataShapeOuterMax,
      Tx81InstructionLimits::dataShapeOuterMax,
      Tx81InstructionLimits::dataShapeOuterMax,
      Tx81InstructionLimits::dataShapeChannelMax};
  for (auto [index, value] : llvm::enumerate(shape))
    if (value > dimensionMaximums[index])
      return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                         llvm::Twine(role) + " " + dimensionNames[index] +
                             " dimension must be in [1, " +
                             llvm::Twine(dimensionMaximums[index]) + "]");
  return llvm::Error::success();
}

llvm::Error validateWeightShape(llvm::ArrayRef<uint32_t> shape,
                                llvm::StringRef role) {
  if (llvm::Error error = validateShapeProduct(shape, role))
    return error;
  return validateInclusiveRange(
      shape, role, /*minimum=*/1,
      static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()));
}

llvm::Error validatePadding(llvm::ArrayRef<uint32_t> values,
                            llvm::StringRef role) {
  return validateInclusiveRange(values, role, /*minimum=*/0,
                                Tx81InstructionLimits::paddingMax);
}

llvm::Error validateKernelStrides(llvm::ArrayRef<uint32_t> values,
                                  llvm::StringRef role) {
  if (values.size() != 4)
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       llvm::Twine(role) + " must contain exactly 4 entries");
  if (llvm::Error error =
          validateInclusiveRange(values.take_front(2), role, /*minimum=*/1,
                                 Tx81InstructionLimits::kernelMax))
    return error;
  return validateInclusiveRange(values.drop_front(2), role, /*minimum=*/1,
                                Tx81InstructionLimits::strideMax);
}

llvm::Error validateDilations(llvm::ArrayRef<uint32_t> values,
                              llvm::StringRef role) {
  return validateInclusiveRange(values, role, /*minimum=*/1,
                                Tx81InstructionLimits::dilationMax);
}

llvm::Error validatePayload(const target::TargetCommandPayload &payload) {
  return std::visit(
      [](const auto &value) -> llvm::Error {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, target::TargetStridedDMACommand>) {
          if (value.direction != target::TargetDMADirection::Read &&
              value.direction != target::TargetDMADirection::Write)
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "DMA has an unknown direction");
          if (llvm::Error error = requireEngineFormat(
                  value.format,
                  value.direction == target::TargetDMADirection::Read
                      ? TargetFormatEngine::RDMA
                      : TargetFormatEngine::WDMA,
                  "DMA"))
            return error;
          return validateDescriptor(value.byteCount, value.innerBytes,
                                    value.iterations, std::nullopt);
        } else if constexpr (std::is_same_v<
                                 T, target::TargetGatherScatterCommand>) {
          if (llvm::Error error =
                  validateDescriptor(value.byteCount, value.innerBytes,
                                     value.sourceIterations, std::nullopt))
            return error;
          return validateDescriptor(value.byteCount, value.innerBytes,
                                    value.destinationIterations, std::nullopt);
        } else if constexpr (std::is_same_v<T, target::TargetMemsetCommand>) {
          if (value.elementCount == 0)
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "memset element_count must be positive");
          if (llvm::Error error = requireEngineFormat(
                  value.format, TargetFormatEngine::TDMA, "memset"))
            return error;
          if (value.format == LogicalFormat::Bool &&
              value.elementCount % UINT32_C(8) != 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "memset BOOL physical-footprint element_count must be a "
                "multiple of 8 for byte granularity");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<T, target::TargetBit2FPCommand> ||
                             std::is_same_v<T, target::TargetMaskMoveCommand>) {
          if (value.elementCount == 0)
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "CT movement element_count must be positive");
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "CT movement");
        } else if constexpr (std::is_same_v<T, target::TargetGemmCommand>) {
          if (value.m == 0 || value.m > std::numeric_limits<uint16_t>::max() ||
              value.n == 0 || value.n > std::numeric_limits<uint16_t>::max())
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "GEMM m and n must be positive uint16");
          if (value.k == 0 || value.k > Tx81InstructionLimits::gemmKMax)
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                llvm::Twine("GEMM k must be in [1, ") +
                    llvm::Twine(Tx81InstructionLimits::gemmKMax) + "]");
          if (value.batchCount == 0 ||
              value.batchCount > Tx81InstructionLimits::gemmBatchMax)
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                llvm::Twine("GEMM batch_count must be in [1, ") +
                    llvm::Twine(Tx81InstructionLimits::gemmBatchMax) + "]");
          if ((value.lhsOrientation != TargetGemmOrientation::Normal &&
               value.lhsOrientation != TargetGemmOrientation::Transpose) ||
              (value.rhsOrientation != TargetGemmOrientation::Normal &&
               value.rhsOrientation != TargetGemmOrientation::Transpose))
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "GEMM orientation is not a closed enum value");
          return requireEngineFormat(value.format, TargetFormatEngine::NE,
                                     "GEMM");
        } else if constexpr (std::is_same_v<T,
                                            target::TargetElementwiseCommand>) {
          return validateElementwise(value);
        } else if constexpr (std::is_same_v<T, target::TargetReduceCommand>) {
          if (llvm::Error error = requireRegisteredOperation(
                  value.operation, getTargetReduceOperations(),
                  "reduce operation"))
            return error;
          if (value.dimension >
              static_cast<uint32_t>(TargetReduceDimension::Trailing2And1And0))
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "reduce dimension has an unknown selector");
          if (value.dimension ==
                  static_cast<uint32_t>(TargetReduceDimension::Trailing3) ||
              value.dimension == static_cast<uint32_t>(
                                     TargetReduceDimension::Trailing2And1And0))
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "reduce N/HWC axes have no supported target contract");
          if (llvm::Error error = validateDataShape(value.nhwc, "reduce shape"))
            return error;
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "reduce");
        } else if constexpr (std::is_same_v<T, target::TargetConvertCommand>) {
          return validateConvert(value);
        } else if constexpr (std::is_same_v<T, target::TargetConvCommand>) {
          if (llvm::Error error = requireRegisteredOperation(
                  value.operation, getTargetConvolutionOperations(),
                  "convolution operation"))
            return error;
          if (llvm::Error error = validateDataShape(value.inputShape,
                                                    "convolution input shape"))
            return error;
          if (llvm::Error error = validateWeightShape(
                  value.weightShape, "convolution weight shape"))
            return error;
          if (llvm::Error error = validateDataShape(value.outputShape,
                                                    "convolution output shape"))
            return error;
          if (llvm::Error error =
                  validatePadding(value.pads, "convolution pads"))
            return error;
          if (llvm::Error error =
                  validatePadding(value.unpads, "convolution unpads"))
            return error;
          if (llvm::Error error = validateKernelStrides(
                  value.kernelStrides, "convolution kernel_strides"))
            return error;
          if (llvm::Error error =
                  validateDilations(value.dilations, "convolution dilations"))
            return error;
          if (llvm::Error error =
                  requireEngineFormat(value.inputFormat, TargetFormatEngine::NE,
                                      "convolution input"))
            return error;
          return requireEngineFormat(value.outputFormat, TargetFormatEngine::NE,
                                     "convolution output");
        } else if constexpr (std::is_same_v<T, target::TargetPoolCommand>) {
          if (llvm::Error error = requireRegisteredOperation(
                  value.operation, getTargetPoolingOperations(),
                  "pool operation"))
            return error;
          const bool indexed =
              value.operation == TargetPoolingOperation::IndexedMaximum ||
              value.operation == TargetPoolingOperation::IndexedMinimum;
          if (value.indexDestination.has_value() != indexed)
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "pool index destination presence differs from pool kind");
          if (llvm::Error error =
                  validateDataShape(value.sourceShape, "pool source shape"))
            return error;
          if (llvm::Error error = validateDataShape(value.destinationShape,
                                                    "pool destination shape"))
            return error;
          if (llvm::Error error = validatePadding(value.pads, "pool pads"))
            return error;
          if (llvm::Error error = validateKernelStrides(value.kernelStrides,
                                                        "pool kernel_strides"))
            return error;
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "pool");
        } else if constexpr (std::is_same_v<T, target::TargetUnpoolCommand>) {
          if (llvm::Error error = requireRegisteredOperation(
                  value.operation, getTargetUnpoolingOperations(),
                  "unpool operation"))
            return error;
          if (value.indexAddress.has_value() ==
              (value.operation == TargetUnpoolingOperation::Average))
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "unpool index presence differs from unpool kind");
          if (llvm::Error error =
                  validateDataShape(value.sourceShape, "unpool source shape"))
            return error;
          if (llvm::Error error = validateDataShape(value.destinationShape,
                                                    "unpool destination shape"))
            return error;
          if (llvm::Error error = validateKernelStrides(
                  value.kernelStrides, "unpool kernel_strides"))
            return error;
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "unpool");
        } else if constexpr (std::is_same_v<
                                 T, target::TargetTDMATransformCommand>) {
          if (value.kind != target::TargetTDMATransformKind::Pad &&
              value.kind != target::TargetTDMATransformKind::ImageToColumn)
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "TDMA transform has an unknown kind");
          if (value.kernelStrides.has_value() !=
              (value.kind == target::TargetTDMATransformKind::ImageToColumn))
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "TDMA kernel strides presence differs from transform kind");
          if (llvm::Error error =
                  validateDataShape(value.sourceShape, "TDMA source shape"))
            return error;
          if (llvm::Error error = validateDataShape(value.destinationShape,
                                                    "TDMA destination shape"))
            return error;
          if (llvm::Error error = validatePadding(value.pads, "TDMA pads"))
            return error;
          if (value.kernelStrides)
            if (llvm::Error error = validateKernelStrides(
                    *value.kernelStrides, "TDMA kernel_strides"))
              return error;
          return requireEngineFormat(value.format, TargetFormatEngine::TDMA,
                                     "TDMA transform");
        } else if constexpr (std::is_same_v<
                                 T,
                                 target::TargetPeripheralArgExtremaCommand>) {
          if (value.operation != TargetPeripheralOperation::ArgMaximum &&
              value.operation != TargetPeripheralOperation::ArgMinimum)
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "arg-extrema payload carries a different peripheral kind");
          if (value.elementCount == 0)
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "arg-extrema element_count must be positive");
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "arg-extrema");
        } else if constexpr (std::is_same_v<
                                 T, target::TargetPeripheralBilinearCommand>) {
          if (value.elementCount == 0)
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "bilinear element_count must be positive");
          if (llvm::Error error =
                  validateDataShape(value.sourceShape, "bilinear source shape"))
            return error;
          if (llvm::Error error = validateDataShape(
                  value.destinationShape, "bilinear destination shape"))
            return error;
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "bilinear");
        } else if constexpr (std::is_same_v<
                                 T, target::TargetPeripheralLUTCommand>) {
          if (value.operation != TargetPeripheralOperation::LookupTable16 &&
              value.operation != TargetPeripheralOperation::LookupTable32)
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "LUT payload carries a different peripheral kind");
          if (value.elementCount == 0 || value.tableElementCount == 0)
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "LUT element counts must be positive");
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "LUT");
        } else if constexpr (
            std::is_same_v<T, target::TargetPeripheralRandomCommand> ||
            std::is_same_v<T, target::TargetPeripheralElementMaskCommand>) {
          if (value.elementCount == 0)
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "peripheral element_count must be positive");
          return requireEngineFormat(value.format, TargetFormatEngine::CT,
                                     "peripheral");
        } else if constexpr (std::is_same_v<T,
                                            target::TargetDDRPublishCommand> ||
                             std::is_same_v<T,
                                            target::TargetDDRAcquireCommand>) {
          if (value.readyAddress % 64 || !value.byteCount)
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "DDR publication requires cache-line aligned storage");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<T, target::TargetNCCJoinCommand>) {
          if (value.participantMask == 0 ||
              (value.participantMask & ~kAllTargetNCCWorkersMask) != 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "NCC join participant mask is empty or outside the target "
                "worker domain");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<
                                 T, target::TargetDirectDTEBeginCommand>) {
          if (value.participantCount == 0)
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "Direct DTE begin participant_count must be positive");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<
                                 T, target::TargetDirectDTESendCommand> ||
                             std::is_same_v<
                                 T, target::TargetDirectDTEReceiveCommand>) {
          if (value.byteCount == 0)
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "Direct DTE byte_count must be positive");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<
                                 T, target::TargetDirectDTEMultiSendCommand>) {
          if (value.bytesPerDestination != 256 ||
              (value.destinationCount != 2 && value.destinationCount != 4 &&
               value.destinationCount != 8 && value.destinationCount != 15))
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "Direct DTE multi-send command is outside the qualified "
                "count/byte domain");
          return llvm::Error::success();
        } else if constexpr (
            std::is_same_v<
                T, target::TargetDirectDTEMultiSendDestinationCommand>) {
          if (value.event == 0 || value.remoteFSM >= 4)
            return kernelError(
                TargetModelKernelErrorCode::InvalidCommandField,
                "Direct DTE multi-send destination is malformed");
          return llvm::Error::success();
        } else if constexpr (std::is_same_v<
                                 T, target::TargetDirectDTESendIssueCommand> ||
                             std::is_same_v<
                                 T, target::TargetDirectDTEWaitCommand>) {
          if (value.event == 0)
            return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                               "Direct DTE issue/wait event must be nonzero");
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
  case TargetModelKernelErrorCode::InvalidCommandField:
    return "invalid-command-field";
  case TargetModelKernelErrorCode::WorkBudgetExceeded:
    return "work-budget-exceeded";
  case TargetModelKernelErrorCode::UnsupportedCommand:
    return "unsupported-command";
  case TargetModelKernelErrorCode::MemoryReadFailure:
    return "memory-read-failure";
  case TargetModelKernelErrorCode::FormalNumericFailure:
    return "formal-numeric-failure";
  case TargetModelKernelErrorCode::PhysicalCodecFailure:
    return "physical-codec-failure";
  case TargetModelKernelErrorCode::ManagedReferenceBackendUnavailable:
    return "managed-reference-backend-unavailable";
  case TargetModelKernelErrorCode::ManagedReferenceBackendFailure:
    return "managed-reference-backend-failure";
  case TargetModelKernelErrorCode::OneDNNBackendUnavailable:
    return "onednn-backend-unavailable";
  case TargetModelKernelErrorCode::OneDNNBackendFailure:
    return "onednn-backend-failure";
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

llvm::Error
validateTargetModelCommandFields(const compiler::TargetCommand &command) {
  if (command.cardId.getValue() < 0 || command.tileId.getValue() < 0 ||
      command.launchSlotId.getValue() < 0)
    return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                       "command physical identity and launch slot must be "
                       "non-negative");
  if (command.nccIssueDomain) {
    uint32_t worker = static_cast<uint32_t>(command.nccIssueDomain->worker);
    if (worker >= kTargetNCCWorkerCount ||
        command.nccIssueDomain->engine == TargetCallTSMEngine::DirectDTE)
      return kernelError(
          TargetModelKernelErrorCode::InvalidCommandField,
          "NCC issue domain is outside the target worker/engine domain");
    if (command.nccIssueDomain->completionBehavior !=
            TargetNCCCompletionBehavior::OrderedAsynchronousIssue &&
        command.nccIssueDomain->completionBehavior !=
            TargetNCCCompletionBehavior::SynchronousWriteback)
      return kernelError(TargetModelKernelErrorCode::InvalidCommandField,
                         "NCC issue domain has an invalid completion behavior");
  }
  return validatePayload(command.payload);
}

} // namespace wafer::model
