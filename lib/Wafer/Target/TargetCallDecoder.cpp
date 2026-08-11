//===- TargetCallDecoder.cpp - Typed target call payload decoder --------===//

#include "Wafer/Target/TargetCall.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <limits>
#include <optional>

namespace wafer {
namespace {

using namespace compiler;

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

static llvm::Expected<LogicalFormat>
decodeFormat(TargetFormatEngine engine, uint64_t code) {
  if (code > std::numeric_limits<uint32_t>::max())
    return llvm::createStringError("target format field does not fit uint32");
  return decodeTargetFormat(engine,
                            static_cast<uint32_t>(code));
}

static llvm::Expected<TargetTransactionPayload>
buildBuiltinTransaction(const TargetCallDecodeContext &context,
                        TargetCallBuiltin builtin,
                        llvm::ArrayRef<uint64_t> arguments) {
  switch (builtin) {
  case TargetCallBuiltin::RDMA:
  case TargetCallBuiltin::WDMA: {
    TargetFormatEngine engine = builtin == TargetCallBuiltin::RDMA
                                    ? TargetFormatEngine::RDMA
                                    : TargetFormatEngine::WDMA;
    llvm::Expected<LogicalFormat> format =
        decodeFormat(engine, arguments[10]);
    if (!format)
      return format.takeError();
    return TargetTransactionPayload{TargetStridedDMATransaction{
        builtin == TargetCallBuiltin::RDMA ? TargetDMADirection::Read
                                           : TargetDMADirection::Write,
        arguments[0], arguments[1], argument32(arguments, 2),
        argument32(arguments, 3), argumentArray32<3>(arguments, 4),
        argumentArray32<3>(arguments, 7), *format}};
  }
  case TargetCallBuiltin::GatherScatter:
    return TargetTransactionPayload{TargetGatherScatterTransaction{
        arguments[0], arguments[1], argument32(arguments, 2),
        argument32(arguments, 3), argumentArray32<3>(arguments, 4),
        argumentArray32<3>(arguments, 7), argumentArray32<3>(arguments, 10),
        argumentArray32<3>(arguments, 13)}};
  case TargetCallBuiltin::Memset: {
    llvm::Expected<LogicalFormat> format =
        decodeFormat(TargetFormatEngine::TDMA, arguments[3]);
    if (!format)
      return format.takeError();
    return TargetTransactionPayload{
        TargetMemsetTransaction{arguments[0], argument32(arguments, 1),
                                argument32(arguments, 2), *format}};
  }
  case TargetCallBuiltin::Bit2FP: {
    llvm::Expected<LogicalFormat> format =
        decodeFormat(TargetFormatEngine::CT, arguments[3]);
    if (!format)
      return format.takeError();
    return TargetTransactionPayload{TargetBit2FPTransaction{
        arguments[0], arguments[1], argument32(arguments, 2), *format}};
  }
  case TargetCallBuiltin::MaskMove: {
    llvm::Expected<LogicalFormat> format =
        decodeFormat(TargetFormatEngine::CT, arguments[4]);
    if (!format)
      return format.takeError();
    return TargetTransactionPayload{TargetMaskMoveTransaction{
        arguments[0], argument32(arguments, 1), arguments[2],
        argument32(arguments, 3), *format}};
  }
  case TargetCallBuiltin::Gemm: {
    llvm::Expected<LogicalFormat> format =
        decodeFormat(TargetFormatEngine::NE, arguments[7]);
    if (!format)
      return format.takeError();
    return TargetTransactionPayload{TargetGemmTransaction{
        arguments[0], arguments[1], arguments[2], argument32(arguments, 3),
        argument32(arguments, 4), argument32(arguments, 5),
        argument32(arguments, 6), *format}};
  }
  case TargetCallBuiltin::GemmOriented: {
    llvm::Expected<LogicalFormat> format =
        decodeFormat(TargetFormatEngine::NE, arguments[7]);
    if (!format)
      return format.takeError();
    uint32_t lhsOrientation = argument32(arguments, 8);
    uint32_t rhsOrientation = argument32(arguments, 9);
    if (lhsOrientation > static_cast<uint32_t>(GemmOrientation::Transpose) ||
        rhsOrientation > static_cast<uint32_t>(GemmOrientation::Transpose))
      return llvm::createStringError(
          "oriented GEMM orientation field is not a closed enum value");
    return TargetTransactionPayload{TargetGemmTransaction{
        arguments[0], arguments[1], arguments[2], argument32(arguments, 3),
        argument32(arguments, 4), argument32(arguments, 5),
        argument32(arguments, 6), *format,
        static_cast<GemmOrientation>(lhsOrientation),
        static_cast<GemmOrientation>(rhsOrientation)}};
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
    return TargetTransactionPayload{TargetTDMATransformTransaction{
        imageToColumn ? TargetTDMATransformKind::ImageToColumn
                      : TargetTDMATransformKind::Pad,
        arguments[0], arguments[1], argumentArray32<4>(arguments, 2),
        argumentArray32<4>(arguments, 6), argumentArray32<4>(arguments, 10),
        kernelStrides, *format}};
  }
  case TargetCallBuiltin::NCCJoin: {
    uint32_t participants = argument32(arguments, 0);
    if (participants == 0 || (participants & ~kAllNCCWorkersMask) != 0)
      return llvm::createStringError(
          "NCC join participant mask is empty or outside the target worker "
          "domain");
    return TargetTransactionPayload{TargetNCCJoinTransaction{participants}};
  }
  case TargetCallBuiltin::DirectDTEBegin:
  case TargetCallBuiltin::DirectDTEBeginAfterPrepare:
    if (argument32(arguments, 1) != context.physicalTileCount)
      return llvm::createStringError(
          "Direct-DTE begin participant count does not match the physical "
          "Tile invocation domain");
    return TargetTransactionPayload{TargetDirectDTEBeginTransaction{
        arguments[0], argument32(arguments, 1)}};
  case TargetCallBuiltin::DirectDTESendPrepare:
    if (argument32(arguments, 6) > 1)
      return llvm::createStringError(
          "Direct-DTE high-performance flag is not boolean");
    return TargetTransactionPayload{TargetDirectDTESendTransaction{
        arguments[0], arguments[1], argument32(arguments, 2),
        argument32(arguments, 3), argument32(arguments, 4),
        argument32(arguments, 5), argument32(arguments, 6) != 0}};
  case TargetCallBuiltin::DirectDTESendIssue:
    return TargetTransactionPayload{
        TargetDirectDTESendIssueTransaction{arguments[0]}};
  case TargetCallBuiltin::DirectDTERecvPrepare:
    return TargetTransactionPayload{TargetDirectDTEReceiveTransaction{
        arguments[0], argument32(arguments, 1), argument32(arguments, 2),
        argument32(arguments, 3), argument32(arguments, 4)}};
  case TargetCallBuiltin::DirectDTEWait:
    return TargetTransactionPayload{
        TargetDirectDTEWaitTransaction{arguments[0]}};
  case TargetCallBuiltin::DirectDTEFinish:
    return TargetTransactionPayload{TargetDirectDTEFinishTransaction{}};
  }
  llvm_unreachable("unknown target-call builtin");
}

static llvm::Expected<TargetTransactionPayload>
buildElementwiseTransaction(const TargetCallDecodeContext &context,
                            InstrElementwiseKind kind,
                            llvm::ArrayRef<uint64_t> arguments) {
  bool unary = arguments.size() == 4;
  size_t destinationIndex = unary ? 1 : 2;
  size_t countIndex = unary ? 2 : 3;
  size_t formatIndex = unary ? 3 : 4;
  llvm::Expected<LogicalFormat> format =
      decodeFormat(TargetFormatEngine::CT, arguments[formatIndex]);
  if (!format)
    return format.takeError();
  return TargetTransactionPayload{TargetElementwiseTransaction{
      kind, arguments[0],
      unary ? std::nullopt : std::optional<uint64_t>(arguments[1]),
      arguments[destinationIndex], argument32(arguments, countIndex), *format}};
}

static llvm::Expected<TargetTransactionPayload>
buildReduceTransaction(const TargetCallDecodeContext &context,
                       InstrReduceKind kind,
                       llvm::ArrayRef<uint64_t> arguments) {
  llvm::Expected<LogicalFormat> format =
      decodeFormat(TargetFormatEngine::CT, arguments[7]);
  if (!format)
    return format.takeError();
  return TargetTransactionPayload{TargetReduceTransaction{
      kind, arguments[0], arguments[1], argument32(arguments, 2),
      argumentArray32<4>(arguments, 3), *format}};
}

static llvm::Expected<TargetTransactionPayload>
buildConvertTransaction(InstrConvertKind kind,
                        llvm::ArrayRef<uint64_t> arguments) {
  std::optional<uint32_t> zeroPoint;
  std::optional<uint32_t> roundingMode;
  switch (getInstrConvertParameterKind(kind)) {
  case InstrConvertParameterKind::ZeroPoint:
    zeroPoint = argument32(arguments, 3);
    break;
  case InstrConvertParameterKind::RoundingMode:
    roundingMode = argument32(arguments, 4);
    break;
  case InstrConvertParameterKind::None:
    break;
  }
  return TargetTransactionPayload{TargetConvertTransaction{
      kind, arguments[0], arguments[1], argument32(arguments, 2), zeroPoint,
      roundingMode}};
}

static llvm::Expected<TargetTransactionPayload>
buildConvTransaction(const TargetCallDecodeContext &context, InstrConvKind kind,
                     llvm::ArrayRef<uint64_t> arguments) {
  if (llvm::Error error = verifyStaticKind(
          argument32(arguments, 3), static_cast<uint32_t>(kind), "convolution"))
    return std::move(error);
  llvm::Expected<LogicalFormat> format =
      decodeFormat(TargetFormatEngine::NE, arguments[30]);
  if (!format)
    return format.takeError();
  return TargetTransactionPayload{TargetConvTransaction{
      kind, arguments[0], arguments[1], arguments[2],
      argumentArray32<4>(arguments, 4), argumentArray32<4>(arguments, 8),
      argumentArray32<4>(arguments, 12), argumentArray32<4>(arguments, 16),
      argumentArray32<4>(arguments, 20), argumentArray32<4>(arguments, 24),
      argumentArray32<2>(arguments, 28), *format}};
}

static llvm::Expected<TargetTransactionPayload>
buildPoolTransaction(const TargetCallDecodeContext &context, InstrPoolKind kind,
                     llvm::ArrayRef<uint64_t> arguments) {
  bool indexed =
      kind == InstrPoolKind::IndexedMax || kind == InstrPoolKind::IndexedMin;
  size_t firstField = indexed ? 3 : 2;
  if (llvm::Error error = verifyStaticKind(argument32(arguments, firstField),
                                           static_cast<uint32_t>(kind), "pool"))
    return std::move(error);
  llvm::Expected<LogicalFormat> format =
      decodeFormat(TargetFormatEngine::CT, arguments[firstField + 17]);
  if (!format)
    return format.takeError();
  return TargetTransactionPayload{TargetPoolTransaction{
      kind, arguments[0], arguments[1],
      indexed ? std::optional<uint64_t>(arguments[2]) : std::nullopt,
      argumentArray32<4>(arguments, firstField + 1),
      argumentArray32<4>(arguments, firstField + 5),
      argumentArray32<4>(arguments, firstField + 9),
      argumentArray32<4>(arguments, firstField + 13), *format}};
}

static llvm::Expected<TargetTransactionPayload>
buildUnpoolTransaction(const TargetCallDecodeContext &context,
                       InstrUnpoolKind kind,
                       llvm::ArrayRef<uint64_t> arguments) {
  if (llvm::Error error = verifyStaticKind(
          argument32(arguments, 2), static_cast<uint32_t>(kind), "unpool"))
    return std::move(error);
  llvm::Expected<LogicalFormat> format =
      decodeFormat(TargetFormatEngine::CT, arguments[16]);
  if (!format)
    return format.takeError();
  return TargetTransactionPayload{TargetUnpoolTransaction{
      kind, arguments[0], arguments[1],
      kind == InstrUnpoolKind::Avg
          ? std::nullopt
          : std::optional<uint32_t>(argument32(arguments, 3)),
      argumentArray32<4>(arguments, 4), argumentArray32<4>(arguments, 8),
      argumentArray32<4>(arguments, 12), *format}};
}

static llvm::Expected<TargetTransactionPayload>
buildPeripheralTransaction(const TargetCallDecodeContext &context,
                           InstrPeripheralKind kind,
                           llvm::ArrayRef<uint64_t> arguments) {
  size_t addressCount = 0;
  switch (kind) {
  case InstrPeripheralKind::ArgMax:
  case InstrPeripheralKind::ArgMin:
  case InstrPeripheralKind::Lut16:
  case InstrPeripheralKind::Lut32:
    addressCount = 3;
    break;
  case InstrPeripheralKind::Bilinear:
  case InstrPeripheralKind::ElemMask:
    addressCount = 2;
    break;
  case InstrPeripheralKind::RandGen:
    addressCount = 5;
    break;
  case InstrPeripheralKind::Count:
  case InstrPeripheralKind::Factorize:
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
  case InstrPeripheralKind::ArgMax:
  case InstrPeripheralKind::ArgMin:
    return TargetTransactionPayload{TargetPeripheralArgExtremaTransaction{
        kind, arguments[0], arguments[1], arguments[2],
        argument32(arguments, elementIndex), *format}};
  case InstrPeripheralKind::Bilinear:
    return TargetTransactionPayload{TargetPeripheralBilinearTransaction{
        arguments[0], arguments[1], argument32(arguments, elementIndex),
        *format, argumentArray32<4>(arguments, addressCount + 3),
        argumentArray32<4>(arguments, addressCount + 7)}};
  case InstrPeripheralKind::Lut16:
  case InstrPeripheralKind::Lut32:
    return TargetTransactionPayload{TargetPeripheralLUTTransaction{
        kind, arguments[0], arguments[1], arguments[2],
        argument32(arguments, elementIndex), *format,
        argument32(arguments, addressCount + 3)}};
  case InstrPeripheralKind::RandGen:
    return TargetTransactionPayload{TargetPeripheralRandomTransaction{
        {arguments[0], arguments[1]},
        {arguments[2], arguments[3], arguments[4]},
        argument32(arguments, elementIndex),
        *format}};
  case InstrPeripheralKind::ElemMask:
    return TargetTransactionPayload{TargetPeripheralElementMaskTransaction{
        arguments[0], arguments[1], argument32(arguments, elementIndex),
        *format, argument32(arguments, addressCount + 4),
        argument32(arguments, addressCount + 5),
        argument32(arguments, addressCount + 6)}};
  case InstrPeripheralKind::Count:
  case InstrPeripheralKind::Factorize:
    break;
  }
  llvm_unreachable("unknown peripheral kind");
}

} // namespace

llvm::Expected<compiler::TargetTransactionPayload>
decodeTargetCallPayload(const TargetCallDescriptor &descriptor,
                        const TargetCallDecodeContext &context,
                        llvm::ArrayRef<uint64_t> arguments) {
  if (arguments.size() != descriptor.arguments.size())
    return llvm::createStringError(
        "target-call payload argument count does not match its descriptor");
  for (auto [index, scalar] : llvm::enumerate(descriptor.arguments))
    if (scalar == TargetCallScalarType::I32 &&
        arguments[index] > std::numeric_limits<uint32_t>::max())
      return llvm::createStringError(
          "target-call i32 payload argument does not fit uint32");

  llvm::Expected<std::optional<NCCWorker>> worker =
      decodeTargetCallNCCWorker(descriptor, arguments);
  if (!worker)
    return worker.takeError();
  llvm::ArrayRef<uint64_t> payloadArguments = arguments;
  if (descriptor.issueDomain &&
      descriptor.issueDomain->nccWorkerArgument.has_value())
    payloadArguments = arguments.drop_back();

  if (const auto *builtin =
          std::get_if<TargetCallBuiltin>(&descriptor.semantic))
    return buildBuiltinTransaction(context, *builtin, payloadArguments);
  if (const auto *kind =
          std::get_if<InstrElementwiseKind>(&descriptor.semantic))
    return buildElementwiseTransaction(context, *kind, payloadArguments);
  if (const auto *kind = std::get_if<InstrReduceKind>(&descriptor.semantic))
    return buildReduceTransaction(context, *kind, payloadArguments);
  if (const auto *kind = std::get_if<InstrConvertKind>(&descriptor.semantic))
    return buildConvertTransaction(*kind, payloadArguments);
  if (const auto *kind = std::get_if<InstrConvKind>(&descriptor.semantic))
    return buildConvTransaction(context, *kind, payloadArguments);
  if (const auto *kind = std::get_if<InstrPoolKind>(&descriptor.semantic))
    return buildPoolTransaction(context, *kind, payloadArguments);
  if (const auto *kind = std::get_if<InstrUnpoolKind>(&descriptor.semantic))
    return buildUnpoolTransaction(context, *kind, payloadArguments);
  if (const auto *kind = std::get_if<InstrPeripheralKind>(&descriptor.semantic))
    return buildPeripheralTransaction(context, *kind, payloadArguments);
  llvm_unreachable("unknown target-call semantic");
}

llvm::Expected<std::optional<NCCWorker>>
decodeTargetCallNCCWorker(const TargetCallDescriptor &descriptor,
                          llvm::ArrayRef<uint64_t> arguments) {
  if (arguments.size() != descriptor.arguments.size())
    return llvm::createStringError(
        "target-call worker argument count does not match its descriptor");
  if (!descriptor.issueDomain)
    return std::optional<NCCWorker>{};
  if (!descriptor.issueDomain->nccWorkerArgument)
    return std::optional<NCCWorker>{};
  size_t index = *descriptor.issueDomain->nccWorkerArgument;
  if (index >= arguments.size() || index + 1 != arguments.size() ||
      descriptor.arguments[index] != TargetCallScalarType::I32)
    return llvm::createStringError(
        "target-call registry has an invalid trailing NCC worker argument");
  uint64_t worker = arguments[index];
  if (worker >= kNCCWorkerCount)
    return llvm::createStringError(
        "target-call NCC worker is outside the target worker domain");
  return std::optional<NCCWorker>{static_cast<NCCWorker>(worker)};
}

} // namespace wafer
