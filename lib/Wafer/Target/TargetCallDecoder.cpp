//===- TargetCallDecoder.cpp - Typed target call payload decoder --------===//

#include "Wafer/Target/TargetCall.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>
#include <optional>

namespace wafer {
namespace {

using namespace target;

static uint32_t argument32(llvm::ArrayRef<uint64_t> arguments, size_t index) {
  assert(arguments[index] <= std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(arguments[index]);
}

template <size_t N>
static std::array<uint32_t, N>
argumentArray32(llvm::ArrayRef<uint64_t> arguments, size_t start) {
  std::array<uint32_t, N> result{};
  for (size_t index = 0; index < N; ++index)
    result[index] = argument32(arguments, start + index);
  return result;
}

static llvm::Error verifyStaticKind(uint32_t actual, uint32_t expected,
                                    llvm::StringRef family) {
  if (actual == expected)
    return llvm::Error::success();
  return llvm::createStringError(
      "%s target call carries a kind field inconsistent with its typed "
      "symbol",
      family.str().c_str());
}

static llvm::Expected<LogicalFormat> decodeFormat(TargetFormatEngine engine,
                                                  uint64_t code) {
  if (code > std::numeric_limits<uint32_t>::max())
    return llvm::createStringError("target format field does not fit uint32");
  return decodeTargetFormat(engine, static_cast<uint32_t>(code));
}

static llvm::Expected<TargetCommandPayload>
buildBuiltinCommand(const TargetCallDecodeConfig &config,
                    TargetCallBuiltin builtin,
                    llvm::ArrayRef<uint64_t> arguments) {
  switch (builtin) {
  case TargetCallBuiltin::RDMA:
  case TargetCallBuiltin::WDMA: {
    TargetFormatEngine engine = builtin == TargetCallBuiltin::RDMA
                                    ? TargetFormatEngine::RDMA
                                    : TargetFormatEngine::WDMA;
    llvm::Expected<LogicalFormat> format = decodeFormat(engine, arguments[10]);
    if (!format)
      return format.takeError();
    return TargetCommandPayload{TargetStridedDMACommand{
        builtin == TargetCallBuiltin::RDMA ? TargetDMADirection::Read
                                           : TargetDMADirection::Write,
        arguments[0], arguments[1], argument32(arguments, 2),
        argument32(arguments, 3), argumentArray32<3>(arguments, 4),
        argumentArray32<3>(arguments, 7), *format}};
  }
  case TargetCallBuiltin::GatherScatter:
    return TargetCommandPayload{TargetGatherScatterCommand{
        arguments[0], arguments[1], argument32(arguments, 2),
        argument32(arguments, 3), argumentArray32<3>(arguments, 4),
        argumentArray32<3>(arguments, 7), argumentArray32<3>(arguments, 10),
        argumentArray32<3>(arguments, 13)}};
  case TargetCallBuiltin::Memset: {
    llvm::Expected<LogicalFormat> format =
        decodeFormat(TargetFormatEngine::TDMA, arguments[3]);
    if (!format)
      return format.takeError();
    return TargetCommandPayload{
        TargetMemsetCommand{arguments[0], argument32(arguments, 1),
                            argument32(arguments, 2), *format}};
  }
  case TargetCallBuiltin::Bit2FP: {
    llvm::Expected<LogicalFormat> format =
        decodeFormat(TargetFormatEngine::CT, arguments[3]);
    if (!format)
      return format.takeError();
    return TargetCommandPayload{TargetBit2FPCommand{
        arguments[0], arguments[1], argument32(arguments, 2), *format}};
  }
  case TargetCallBuiltin::MaskMove: {
    llvm::Expected<LogicalFormat> format =
        decodeFormat(TargetFormatEngine::CT, arguments[4]);
    if (!format)
      return format.takeError();
    return TargetCommandPayload{
        TargetMaskMoveCommand{arguments[0], argument32(arguments, 1),
                              arguments[2], argument32(arguments, 3), *format}};
  }
  case TargetCallBuiltin::Gemm:
  case TargetCallBuiltin::GemmOriented: {
    auto input = decodeFormat(TargetFormatEngine::NE, arguments[8]);
    if (!input)
      return input.takeError();
    auto output = decodeFormat(TargetFormatEngine::NE, arguments[9]);
    if (!output)
      return output.takeError();
    std::optional<TargetGemmPartial> psum;
    if (arguments[10] != kDisabledGemmPartialFormat) {
      auto format = decodeFormat(TargetFormatEngine::NE, arguments[10]);
      if (!format)
        return format.takeError();
      psum = TargetGemmPartial{arguments[3], *format};
    } else if (arguments[3] != 0) {
      return llvm::createStringError(
          "disabled GEMM psum requires zero address");
    }
    uint32_t lhsOrientation = 0, rhsOrientation = 0;
    if (builtin == TargetCallBuiltin::GemmOriented) {
      lhsOrientation = argument32(arguments, 11);
      rhsOrientation = argument32(arguments, 12);
      if (lhsOrientation >
              static_cast<uint32_t>(TargetGemmOrientation::Transpose) ||
          rhsOrientation >
              static_cast<uint32_t>(TargetGemmOrientation::Transpose))
        return llvm::createStringError(
            "oriented GEMM orientation field is not a closed enum value");
    }
    return TargetCommandPayload{TargetGemmCommand{
        arguments[0], arguments[1], arguments[2], argument32(arguments, 4),
        argument32(arguments, 5), argument32(arguments, 6),
        argument32(arguments, 7), *input, *output,
        static_cast<TargetGemmOrientation>(lhsOrientation),
        static_cast<TargetGemmOrientation>(rhsOrientation), psum}};
  }
  case TargetCallBuiltin::TDMAPad:
  case TargetCallBuiltin::TDMAImg2Col: {
    bool imageToColumn = builtin == TargetCallBuiltin::TDMAImg2Col;
    size_t formatIndex = imageToColumn ? 18 : 14;
    llvm::Expected<LogicalFormat> format =
        decodeFormat(TargetFormatEngine::TDMA, arguments[formatIndex]);
    if (!format)
      return format.takeError();
    std::optional<std::array<uint32_t, 4>> kernelStrides;
    if (imageToColumn)
      kernelStrides = argumentArray32<4>(arguments, 14);
    return TargetCommandPayload{TargetTDMATransformCommand{
        imageToColumn ? TargetTDMATransformKind::ImageToColumn
                      : TargetTDMATransformKind::Pad,
        arguments[0], arguments[1], argumentArray32<4>(arguments, 2),
        argumentArray32<4>(arguments, 6), argumentArray32<4>(arguments, 10),
        kernelStrides, *format}};
  }
  case TargetCallBuiltin::DDRPublish:
    return TargetCommandPayload{
        TargetDDRPublishCommand{arguments[0], arguments[1], arguments[2]}};
  case TargetCallBuiltin::DDRAcquire:
    return TargetCommandPayload{
        TargetDDRAcquireCommand{arguments[0], arguments[1], arguments[2]}};
  case TargetCallBuiltin::DDRReadMapping: {
    uint32_t size = argument32(arguments, 1);
    if (size == 0 || size > uint32_t(std::numeric_limits<int32_t>::max()))
      return llvm::createStringError("DDR read mapping requires a positive int32 byte range");
    return TargetCommandPayload{TargetMemoryMappingCommand{
        TargetScalarMemorySpace::DDR, arguments[0], size}};
  }
  case TargetCallBuiltin::SPMMapping:
    return TargetCommandPayload{TargetMemoryMappingCommand{
        TargetScalarMemorySpace::SPM, arguments[0], 1}};
  case TargetCallBuiltin::NCCJoin: {
    uint32_t participants = argument32(arguments, 0);
    if (participants == 0 || (participants & ~kAllTargetNCCWorkersMask) != 0)
      return llvm::createStringError(
          "NCC join participant mask is empty or outside the target worker "
          "domain");
    return TargetCommandPayload{TargetNCCJoinCommand{participants}};
  }
  case TargetCallBuiltin::DirectDTEBegin:
  case TargetCallBuiltin::DirectDTEBeginAfterPrepare:
    if (argument32(arguments, 1) != config.tileCount)
      return llvm::createStringError(
          "Direct-DTE begin participant count does not match the physical "
          "Tile invocation domain");
    return TargetCommandPayload{
        TargetDirectDTEBeginCommand{arguments[0], argument32(arguments, 1)}};
  case TargetCallBuiltin::DirectDTESendPrepare:
    if (argument32(arguments, 6) > 1)
      return llvm::createStringError(
          "Direct-DTE high-performance flag is not boolean");
    return TargetCommandPayload{TargetDirectDTESendCommand{
        arguments[0], arguments[1], argument32(arguments, 2),
        argument32(arguments, 3), argument32(arguments, 4),
        argument32(arguments, 5), argument32(arguments, 6) != 0}};
  case TargetCallBuiltin::DirectDTEMultiSendPrepare: {
    uint32_t bytes = argument32(arguments, 1);
    uint32_t destinationCount = argument32(arguments, 3);
    uint32_t kind = argument32(arguments, 4);
    uint32_t highPerformance = argument32(arguments, 5);
    if (bytes != 256 ||
        (destinationCount != 2 && destinationCount != 4 &&
         destinationCount != 8 && destinationCount != 15) ||
        (kind !=
             static_cast<uint32_t>(TargetDirectDTEMultiSendKind::Broadcast) &&
         kind !=
             static_cast<uint32_t>(TargetDirectDTEMultiSendKind::Scatter)) ||
        highPerformance > 1)
      return llvm::createStringError(
          "Direct-DTE multi-send prepare carries an unsupported kind, "
          "destination count, byte count, or high-performance flag");
    return TargetCommandPayload{TargetDirectDTEMultiSendCommand{
        static_cast<TargetDirectDTEMultiSendKind>(kind), arguments[0], bytes,
        argument32(arguments, 2), destinationCount, highPerformance != 0}};
  }
  case TargetCallBuiltin::DirectDTEMultiSendAddDestination:
    return TargetCommandPayload{TargetDirectDTEMultiSendDestinationCommand{
        arguments[0], arguments[1], argument32(arguments, 2),
        argument32(arguments, 3)}};
  case TargetCallBuiltin::DirectDTESendIssue:
    return TargetCommandPayload{TargetDirectDTESendIssueCommand{arguments[0]}};
  case TargetCallBuiltin::DirectDTERecvPrepare:
    return TargetCommandPayload{TargetDirectDTEReceiveCommand{
        arguments[0], argument32(arguments, 1), argument32(arguments, 2),
        argument32(arguments, 3), argument32(arguments, 4)}};
  case TargetCallBuiltin::DirectDTEWait:
    return TargetCommandPayload{TargetDirectDTEWaitCommand{arguments[0]}};
  case TargetCallBuiltin::DirectDTEFinish:
    return TargetCommandPayload{TargetDirectDTEFinishCommand{}};
  }
  llvm_unreachable("unknown target-call builtin");
}

static llvm::Expected<TargetCommandPayload>
buildElementwiseCommand(const TargetCallDecodeConfig &config,
                        TargetElementwiseOperation kind,
                        llvm::ArrayRef<uint64_t> arguments) {
  bool unary = arguments.size() == 4;
  size_t destinationIndex = unary ? 1 : 2;
  size_t countIndex = unary ? 2 : 3;
  size_t formatIndex = unary ? 3 : 4;
  llvm::Expected<LogicalFormat> format =
      decodeFormat(TargetFormatEngine::CT, arguments[formatIndex]);
  if (!format)
    return format.takeError();
  return TargetCommandPayload{TargetElementwiseCommand{
      kind, arguments[0],
      unary ? std::nullopt : std::optional<uint64_t>(arguments[1]),
      arguments[destinationIndex], argument32(arguments, countIndex), *format}};
}

static llvm::Expected<TargetCommandPayload>
buildReduceCommand(const TargetCallDecodeConfig &config,
                   TargetReduceOperation kind,
                   llvm::ArrayRef<uint64_t> arguments) {
  llvm::Expected<LogicalFormat> format =
      decodeFormat(TargetFormatEngine::CT, arguments[7]);
  if (!format)
    return format.takeError();
  return TargetCommandPayload{TargetReduceCommand{
      kind, arguments[0], arguments[1], argument32(arguments, 2),
      argumentArray32<4>(arguments, 3), *format}};
}

static llvm::Expected<TargetCommandPayload>
buildConvertCommand(TargetConvertOperation kind,
                    llvm::ArrayRef<uint64_t> arguments) {
  std::optional<TargetConvertParameter> parameter;
  const TargetConvertRoute *route = findTargetConvertRoute(kind.getOpcode());
  assert(route && "target-call descriptor must carry a registered convert");
  switch (route->parameterKind) {
  case TargetConvertParameterKind::ZeroPoint:
    parameter = TargetConvertParameter::zeroPoint(argument32(arguments, 3));
    break;
  case TargetConvertParameterKind::RoundingMode: {
    uint32_t raw = argument32(arguments, 4);
    if (raw > std::numeric_limits<uint8_t>::max())
      return llvm::createStringError("convert rounding mode is out of range");
    auto mode = parseTargetRoundingMode(static_cast<uint8_t>(raw));
    if (!mode)
      return mode.takeError();
    parameter = TargetConvertParameter::roundingMode(*mode);
    break;
  }
  case TargetConvertParameterKind::None:
    break;
  }
  return TargetCommandPayload{TargetConvertCommand{
      kind, arguments[0], arguments[1], argument32(arguments, 2), parameter}};
}

static llvm::Expected<TargetCommandPayload>
buildConvCommand(const TargetCallDecodeConfig &config,
                 TargetConvolutionOperation kind,
                 llvm::ArrayRef<uint64_t> arguments) {
  if (llvm::Error error = verifyStaticKind(
          argument32(arguments, 3), static_cast<uint32_t>(kind), "convolution"))
    return std::move(error);
  llvm::Expected<LogicalFormat> inputFormat =
      decodeFormat(TargetFormatEngine::NE, arguments[30]);
  if (!inputFormat)
    return inputFormat.takeError();
  llvm::Expected<LogicalFormat> outputFormat =
      decodeFormat(TargetFormatEngine::NE, arguments[31]);
  if (!outputFormat)
    return outputFormat.takeError();
  return TargetCommandPayload{TargetConvCommand{
      kind, arguments[0], arguments[1], arguments[2],
      argumentArray32<4>(arguments, 4), argumentArray32<4>(arguments, 8),
      argumentArray32<4>(arguments, 12), argumentArray32<4>(arguments, 16),
      argumentArray32<4>(arguments, 20), argumentArray32<4>(arguments, 24),
      argumentArray32<2>(arguments, 28), *inputFormat, *outputFormat}};
}

static llvm::Expected<TargetCommandPayload>
buildPoolCommand(const TargetCallDecodeConfig &config,
                 TargetPoolingOperation kind,
                 llvm::ArrayRef<uint64_t> arguments) {
  bool indexed = kind == TargetPoolingOperation::IndexedMaximum ||
                 kind == TargetPoolingOperation::IndexedMinimum;
  size_t firstField = indexed ? 3 : 2;
  if (llvm::Error error = verifyStaticKind(argument32(arguments, firstField),
                                           static_cast<uint32_t>(kind), "pool"))
    return std::move(error);
  llvm::Expected<LogicalFormat> format =
      decodeFormat(TargetFormatEngine::CT, arguments[firstField + 17]);
  if (!format)
    return format.takeError();
  return TargetCommandPayload{TargetPoolCommand{
      kind, arguments[0], arguments[1],
      indexed ? std::optional<uint64_t>(arguments[2]) : std::nullopt,
      argumentArray32<4>(arguments, firstField + 1),
      argumentArray32<4>(arguments, firstField + 5),
      argumentArray32<4>(arguments, firstField + 9),
      argumentArray32<4>(arguments, firstField + 13), *format}};
}

static llvm::Expected<TargetCommandPayload>
buildUnpoolCommand(const TargetCallDecodeConfig &config,
                   TargetUnpoolingOperation kind,
                   llvm::ArrayRef<uint64_t> arguments) {
  if (llvm::Error error = verifyStaticKind(
          argument32(arguments, 2), static_cast<uint32_t>(kind), "unpool"))
    return std::move(error);
  llvm::Expected<LogicalFormat> format =
      decodeFormat(TargetFormatEngine::CT, arguments[16]);
  if (!format)
    return format.takeError();
  return TargetCommandPayload{TargetUnpoolCommand{
      kind, arguments[0], arguments[1],
      kind == TargetUnpoolingOperation::Average
          ? std::nullopt
          : std::optional<uint32_t>(argument32(arguments, 3)),
      argumentArray32<4>(arguments, 4), argumentArray32<4>(arguments, 8),
      argumentArray32<4>(arguments, 12), *format}};
}

static llvm::Expected<TargetCommandPayload>
buildPeripheralCommand(const TargetCallDecodeConfig &config,
                       TargetPeripheralOperation kind,
                       llvm::ArrayRef<uint64_t> arguments) {
  size_t addressCount = 0;
  switch (kind) {
  case TargetPeripheralOperation::ArgMaximum:
  case TargetPeripheralOperation::ArgMinimum:
  case TargetPeripheralOperation::LookupTable16:
  case TargetPeripheralOperation::LookupTable32:
    addressCount = 3;
    break;
  case TargetPeripheralOperation::Bilinear:
  case TargetPeripheralOperation::ElementMask:
    addressCount = 2;
    break;
  case TargetPeripheralOperation::Random:
    addressCount = 5;
    break;
  case TargetPeripheralOperation::Count:
  case TargetPeripheralOperation::Factorize:
    return llvm::createStringError(
        "peripheral kind has no registered target-call ABI");
  }
  if (llvm::Error error =
          verifyStaticKind(argument32(arguments, addressCount),
                           static_cast<uint32_t>(kind), "peripheral"))
    return std::move(error);
  size_t elementIndex = addressCount + 1;
  size_t formatIndex = addressCount + 2;
  llvm::Expected<LogicalFormat> format =
      decodeFormat(TargetFormatEngine::CT, arguments[formatIndex]);
  if (!format)
    return format.takeError();

  switch (kind) {
  case TargetPeripheralOperation::ArgMaximum:
  case TargetPeripheralOperation::ArgMinimum:
    return TargetCommandPayload{TargetPeripheralArgExtremaCommand{
        kind, arguments[0], arguments[1], arguments[2],
        argument32(arguments, elementIndex), *format}};
  case TargetPeripheralOperation::Bilinear:
    return TargetCommandPayload{TargetPeripheralBilinearCommand{
        arguments[0], arguments[1], argument32(arguments, elementIndex),
        *format, argumentArray32<4>(arguments, addressCount + 3),
        argumentArray32<4>(arguments, addressCount + 7)}};
  case TargetPeripheralOperation::LookupTable16:
  case TargetPeripheralOperation::LookupTable32:
    return TargetCommandPayload{TargetPeripheralLUTCommand{
        kind, arguments[0], arguments[1], arguments[2],
        argument32(arguments, elementIndex), *format,
        argument32(arguments, addressCount + 3)}};
  case TargetPeripheralOperation::Random:
    return TargetCommandPayload{TargetPeripheralRandomCommand{
        {arguments[0], arguments[1]},
        {arguments[2], arguments[3], arguments[4]},
        argument32(arguments, elementIndex),
        *format}};
  case TargetPeripheralOperation::ElementMask:
    return TargetCommandPayload{TargetPeripheralElementMaskCommand{
        arguments[0], arguments[1], argument32(arguments, elementIndex),
        *format, argument32(arguments, addressCount + 4),
        argument32(arguments, addressCount + 5),
        argument32(arguments, addressCount + 6)}};
  case TargetPeripheralOperation::Count:
  case TargetPeripheralOperation::Factorize:
    break;
  }
  llvm_unreachable("unknown peripheral kind");
}

} // namespace

llvm::Expected<target::TargetCommandPayload>
decodeTargetCallPayload(const TargetCallDescriptor &descriptor,
                        const TargetCallDecodeConfig &config,
                        llvm::ArrayRef<uint64_t> arguments) {
  if (arguments.size() != descriptor.arguments.size())
    return llvm::createStringError(
        "target-call payload argument count does not match its descriptor");
  for (auto [index, scalar] : llvm::enumerate(descriptor.arguments))
    if (scalar == TargetCallScalarType::I32 &&
        arguments[index] > std::numeric_limits<uint32_t>::max())
      return llvm::createStringError(
          "target-call i32 payload argument does not fit uint32");

  llvm::Expected<std::optional<TargetNCCWorker>> worker =
      decodeTargetCallNCCWorker(descriptor, arguments);
  if (!worker)
    return worker.takeError();
  llvm::ArrayRef<uint64_t> payloadArguments = arguments;
  if (descriptor.issueDomain &&
      descriptor.issueDomain->nccWorkerArgument.has_value())
    payloadArguments = arguments.drop_back();

  if (const auto *builtin =
          std::get_if<TargetCallBuiltin>(&descriptor.semantic))
    return buildBuiltinCommand(config, *builtin, payloadArguments);
  if (const auto *kind =
          std::get_if<TargetElementwiseOperation>(&descriptor.semantic))
    return buildElementwiseCommand(config, *kind, payloadArguments);
  if (const auto *kind =
          std::get_if<TargetReduceOperation>(&descriptor.semantic))
    return buildReduceCommand(config, *kind, payloadArguments);
  if (const auto *kind =
          std::get_if<TargetConvertOperation>(&descriptor.semantic))
    return buildConvertCommand(*kind, payloadArguments);
  if (const auto *kind =
          std::get_if<TargetConvolutionOperation>(&descriptor.semantic))
    return buildConvCommand(config, *kind, payloadArguments);
  if (const auto *kind =
          std::get_if<TargetPoolingOperation>(&descriptor.semantic))
    return buildPoolCommand(config, *kind, payloadArguments);
  if (const auto *kind =
          std::get_if<TargetUnpoolingOperation>(&descriptor.semantic))
    return buildUnpoolCommand(config, *kind, payloadArguments);
  if (const auto *kind =
          std::get_if<TargetPeripheralOperation>(&descriptor.semantic))
    return buildPeripheralCommand(config, *kind, payloadArguments);
  llvm_unreachable("unknown target-call semantic");
}

llvm::Expected<std::optional<TargetNCCWorker>>
decodeTargetCallNCCWorker(const TargetCallDescriptor &descriptor,
                          llvm::ArrayRef<uint64_t> arguments) {
  if (arguments.size() != descriptor.arguments.size())
    return llvm::createStringError(
        "target-call worker argument count does not match its descriptor");
  if (!descriptor.issueDomain)
    return std::optional<TargetNCCWorker>{};
  if (!descriptor.issueDomain->nccWorkerArgument)
    return std::optional<TargetNCCWorker>{};
  size_t index = *descriptor.issueDomain->nccWorkerArgument;
  if (index >= arguments.size() || index + 1 != arguments.size() ||
      descriptor.arguments[index] != TargetCallScalarType::I32)
    return llvm::createStringError(
        "target-call registry has an invalid trailing NCC worker argument");
  uint64_t worker = arguments[index];
  if (worker >= kTargetNCCWorkerCount)
    return llvm::createStringError(
        "target-call NCC worker is outside the target worker domain");
  return std::optional<TargetNCCWorker>{static_cast<TargetNCCWorker>(worker)};
}

} // namespace wafer
