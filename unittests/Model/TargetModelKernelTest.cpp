//===- TargetModelKernelTest.cpp - Plain target kernel tests -------------===//

#include "Wafer/Model/TargetModelKernel.h"

#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
#include "Wafer/Target/PhysicalTensorCodec.h"
#include "Wafer/Target/TargetCall.h"
#include "Wafer/Target/TargetFormat.h"

#include "../../lib/Wafer/Model/TargetModelTileCommandTracker.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;
using namespace wafer::target;

constexpr uint64_t kDDRBase = UINT64_C(0x100000);

TargetModelKernelBudget makeBudget() {
  return TargetModelKernelBudget::create(
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/4096,
                                      /*maximumFusedMultiplyAdds=*/4096),
      /*maximumMovementBytes=*/4096,
      /*maximumMovementSegments=*/256);
}

template <typename T> std::string expectError(llvm::Expected<T> value) {
  if (value) {
    ADD_FAILURE() << "expected an LLVM Error";
    return {};
  }
  return llvm::toString(value.takeError());
}

std::string expectError(llvm::Error error) {
  if (!error) {
    ADD_FAILURE() << "expected an LLVM Error";
    return {};
  }
  return llvm::toString(std::move(error));
}

TargetCallInvocationDescriptor makeInvocation(size_t tileCount = 1) {
  std::vector<TargetCallTileDescriptor> tiles;
  for (size_t launchSlot = 0; launchSlot < tileCount; ++launchSlot) {
    const uint64_t status =
        kDDRBase + UINT64_C(0x2000) + launchSlot * UINT64_C(0x1000);
    tiles.push_back(TargetCallTileDescriptor{
        CardId(0),
        TileId(static_cast<int64_t>(launchSlot)),
        LaunchSlotId(static_cast<int64_t>(launchSlot)),
        {{0,
          KernelABISlotRole::UserInput,
          0,
          "input-name-is-not-semantic",
          "u8",
          MemLayout::Tensor,
          {256},
          256,
          256},
         {1,
          KernelABISlotRole::Output,
          0,
          "output-name-is-not-semantic",
          "u8",
          MemLayout::Tensor,
          {256},
          256,
          256},
         {2,
          KernelABISlotRole::TransportStatus,
          0,
          "status-name-is-not-semantic",
          "u32",
          MemLayout::Tensor,
          {1},
          WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_BYTES,
          WAFER_TX81_DIRECT_DTE_STATUS_STORAGE_ALIGNMENT}},
        {kDDRBase, kDDRBase + UINT64_C(0x1000), status},
        TargetIdentityId::waferTx81SingleCard(),
        KernelRuntimeABIId::waferTx81Kernel()});
  }
  return {TargetIdentityId::waferTx81SingleCard(), std::move(tiles)};
}

InvocationMemoryRegistry makeRegistry(size_t tileCount = 1) {
  std::vector<TargetModelInputBinding> inputs;
  std::vector<uint8_t> bytes(256);
  for (size_t index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<uint8_t>(index);
  inputs.push_back(
      {getTargetModelResourceId(CardId(0), TileId(0),
                                KernelABISlotRole::UserInput,
                                /*resourceIndex=*/0),
       std::move(bytes)});
  return llvm::cantFail(InvocationMemoryRegistry::create(llvm::cantFail(
      InvocationAddressPlan::create(makeInvocation(tileCount), inputs))));
}

NumericTensorKey makeTensor(LogicalFormat format, PhysicalTensorLayout layout,
                            std::vector<uint64_t> shape) {
  return llvm::cantFail(
      NumericTensorKey::create(format, layout, std::move(shape)));
}

void writeTensor(InvocationMemoryRegistry &memory, int64_t launchSlot,
                 uint64_t address, const NumericTensorKey &key,
                 llvm::ArrayRef<RawLogicalValue> values) {
  std::vector<uint8_t> bytes =
      llvm::cantFail(packPhysicalTensorLogicalValues(key, values, UINT8_C(0)));
  llvm::cantFail(memory.applyAtomically(
      {TargetModelByteWrite{launchSlot, TargetModelAddressSpace::TileSPM,
                            address, 1, std::move(bytes)}}));
}

std::vector<RawLogicalValue> readTensor(const InvocationMemoryRegistry &memory,
                                        int64_t launchSlot, uint64_t address,
                                        const NumericTensorKey &key) {
  uint64_t bytes = llvm::cantFail(getPhysicalTensorStorageBytes(key));
  return llvm::cantFail(unpackPhysicalTensorLogicalValues(
      key,
      llvm::cantFail(memory.readSnapshot(
          launchSlot, TargetModelAddressSpace::TileSPM, address, bytes, 1))));
}

uint64_t supportedFormatCode(TargetFormatEngine engine, LogicalFormat format) {
  const TargetFormatEncodingRecord *record =
      findTargetFormatEncoding(engine, format);
  assert(record);
  return record->dataFormatCode;
}

uint64_t supportedF32Code(TargetFormatEngine engine) {
  return supportedFormatCode(engine, LogicalFormat::F32);
}

class FakeBulkBackend final : public TargetModelBulkBackend {
public:
  explicit FakeBulkBackend(bool hasMatchingExecution)
      : hasMatchingExecution(hasMatchingExecution) {}

  llvm::Expected<std::optional<TargetModelBulkResult>>
  tryExecute(const TargetModelNumericRequest &request) const override {
    ++invocations;
    if (!hasMatchingExecution)
      return std::optional<TargetModelBulkResult>();
    TargetModelBulkResult result{
        request.destinationTemplate,
        {},
        {1, 1, 0, TargetModelBulkEvidenceKind::ExactQualificationRecord,
         "sha256:fake-qualification-record", "fake-bulk"}};
    return std::optional<TargetModelBulkResult>(std::move(result));
  }

  mutable uint64_t invocations = 0;

private:
  bool hasMatchingExecution;
};

std::vector<uint64_t>
makeFieldValidArguments(const TargetCallDescriptor &descriptor) {
  std::vector<uint64_t> arguments;
  arguments.reserve(descriptor.arguments.size());
  for (auto [index, scalar] : llvm::enumerate(descriptor.arguments))
    arguments.push_back(scalar == TargetCallScalarType::I64
                            ? kDDRBase + index * UINT64_C(0x1000)
                            : UINT64_C(1));
  if (descriptor.issueDomain &&
      descriptor.issueDomain->nccWorkerArgument.has_value())
    arguments[*descriptor.issueDomain->nccWorkerArgument] = 0;

  if (const auto *builtin =
          std::get_if<TargetCallBuiltin>(&descriptor.semantic)) {
    switch (*builtin) {
    case TargetCallBuiltin::RDMA:
    case TargetCallBuiltin::WDMA:
      arguments[2] = 4;
      arguments[3] = 4;
      arguments[7] = arguments[8] = arguments[9] = 1;
      arguments[10] = supportedF32Code(*builtin == TargetCallBuiltin::RDMA
                                           ? TargetFormatEngine::RDMA
                                           : TargetFormatEngine::WDMA);
      break;
    case TargetCallBuiltin::GatherScatter:
      arguments[2] = 1;
      arguments[3] = 1;
      arguments[7] = arguments[8] = arguments[9] = 1;
      arguments[13] = arguments[14] = arguments[15] = 1;
      break;
    case TargetCallBuiltin::Memset:
      arguments[3] = supportedF32Code(TargetFormatEngine::TDMA);
      break;
    case TargetCallBuiltin::Bit2FP:
      arguments[3] = supportedF32Code(TargetFormatEngine::CT);
      break;
    case TargetCallBuiltin::MaskMove:
      arguments[4] = supportedF32Code(TargetFormatEngine::CT);
      break;
    case TargetCallBuiltin::Gemm:
      arguments[3] = arguments[4] = arguments[5] = arguments[6] = 1;
      arguments[7] = supportedF32Code(TargetFormatEngine::NE);
      break;
    case TargetCallBuiltin::GemmOriented:
      arguments[3] = arguments[4] = arguments[5] = arguments[6] = 1;
      arguments[7] = supportedF32Code(TargetFormatEngine::NE);
      arguments[8] = 1;
      arguments[9] = 0;
      break;
    case TargetCallBuiltin::TDMAPad:
      arguments[14] = supportedF32Code(TargetFormatEngine::TDMA);
      break;
    case TargetCallBuiltin::TDMAImg2Col:
      arguments[18] = supportedF32Code(TargetFormatEngine::TDMA);
      break;
    case TargetCallBuiltin::DirectDTEBegin:
    case TargetCallBuiltin::DirectDTEBeginAfterPrepare:
      arguments[1] = 16;
      break;
    case TargetCallBuiltin::DirectDTESendPrepare:
      arguments[6] = 1;
      break;
    case TargetCallBuiltin::NCCJoin:
    case TargetCallBuiltin::DirectDTESendIssue:
    case TargetCallBuiltin::DirectDTERecvPrepare:
    case TargetCallBuiltin::DirectDTEWait:
    case TargetCallBuiltin::DirectDTEFinish:
      break;
    }
    return arguments;
  }
  if (const auto *operation =
          std::get_if<NumericElementwiseOperation>(&descriptor.semantic)) {
    const bool logic = isNumericElementwiseLogic(*operation);
    const size_t formatArgument =
        arguments.size() - 1 -
        static_cast<size_t>(descriptor.issueDomain &&
                            descriptor.issueDomain->nccWorkerArgument);
    arguments[formatArgument] =
        supportedFormatCode(TargetFormatEngine::CT,
                            logic ? LogicalFormat::Bool : LogicalFormat::F32);
    return arguments;
  }
  if (std::holds_alternative<NumericReduceOperation>(descriptor.semantic)) {
    arguments[2] = 0;
    arguments[7] = supportedF32Code(TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *operation =
          std::get_if<TargetConvolutionOperation>(&descriptor.semantic)) {
    arguments[3] = static_cast<uint32_t>(*operation);
    arguments[30] = supportedF32Code(TargetFormatEngine::NE);
    return arguments;
  }
  if (const auto *operation =
          std::get_if<TargetPoolingOperation>(&descriptor.semantic)) {
    const bool indexed = *operation == TargetPoolingOperation::IndexedMaximum ||
                         *operation == TargetPoolingOperation::IndexedMinimum;
    const size_t firstField = indexed ? 3 : 2;
    arguments[firstField] = static_cast<uint32_t>(*operation);
    arguments[firstField + 17] = supportedF32Code(TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *operation =
          std::get_if<TargetUnpoolingOperation>(&descriptor.semantic)) {
    arguments[2] = static_cast<uint32_t>(*operation);
    arguments[16] = supportedF32Code(TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *operation =
          std::get_if<TargetPeripheralOperation>(&descriptor.semantic)) {
    size_t addressCount = 0;
    switch (*operation) {
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
      llvm_unreachable("unregistered peripheral kind");
    }
    arguments[addressCount] = static_cast<uint32_t>(*operation);
    arguments[addressCount + 2] = supportedF32Code(TargetFormatEngine::CT);
  }
  return arguments;
}

TEST(TargetModelKernelTest, EveryTypedCallPayloadHasClosedFieldValidation) {
  size_t validated = 0;
  for (const TargetCallDescriptor &descriptor : getTargetCallDescriptors()) {
    TargetCallDecodeConfig config{16};
    llvm::Expected<TargetCommandPayload> payload = decodeTargetCallPayload(
        descriptor, config, makeFieldValidArguments(descriptor));
    ASSERT_TRUE(static_cast<bool>(payload))
        << descriptor.symbol << ": " << llvm::toString(payload.takeError());
    llvm::Error error = validateTargetModelCommandFields(
        TargetCommand{CardId(0), TileId(0), LaunchSlotId(0),
                      validated, std::move(*payload)});
    ASSERT_FALSE(static_cast<bool>(error))
        << descriptor.symbol << ": " << llvm::toString(std::move(error));
    ++validated;
  }
  EXPECT_EQ(validated, 112u);
}

TEST(TargetModelTileCommandTrackerTest,
     NCCJoinDoesNotCompleteAnEarlierDirectDTEOrdinal) {
  wafer::model::detail::TargetModelTileCommandTracker state;
  ASSERT_TRUE(state.beginIssue(0)); // Direct-DTE send/event.
  ASSERT_TRUE(state.beginIssue(1)); // Worker-0 NCC issue.
  ASSERT_TRUE(state.addNCCPending(TargetNCCWorker::Worker0, 1));
  ASSERT_TRUE(state.beginIssue(2)); // Worker-0 participant join.

  llvm::SmallVector<uint64_t, 8> joined = state.takeNCCParticipantPending(
      uint32_t{1} << static_cast<uint32_t>(TargetNCCWorker::Worker0));
  ASSERT_EQ(joined.size(), 1u);
  EXPECT_EQ(joined.front(), 1u);
  ASSERT_TRUE(state.markComplete(joined.front()));
  ASSERT_TRUE(state.markComplete(2));
  EXPECT_EQ(state.getNextCompletedOrdinal(), 0u);

  ASSERT_TRUE(state.beginIssue(3));   // Direct-DTE wait.
  ASSERT_TRUE(state.markComplete(0)); // Matching DTE event, not the NCC join.
  EXPECT_EQ(state.getNextCompletedOrdinal(), 3u);
  ASSERT_TRUE(state.markComplete(3));
  EXPECT_EQ(state.getNextCompletedOrdinal(), 4u);
}

TEST(TargetModelTileCommandTrackerTest,
     NCCJoinCompletesOnlyParticipantWorkers) {
  wafer::model::detail::TargetModelTileCommandTracker state;
  ASSERT_TRUE(state.beginIssue(0));
  ASSERT_TRUE(state.addNCCPending(TargetNCCWorker::Worker0, 0));
  ASSERT_TRUE(state.beginIssue(1));
  ASSERT_TRUE(state.addNCCPending(TargetNCCWorker::Worker1, 1));
  ASSERT_TRUE(state.beginIssue(2));

  llvm::SmallVector<uint64_t, 8> worker0 = state.takeNCCParticipantPending(
      uint32_t{1} << static_cast<uint32_t>(TargetNCCWorker::Worker0));
  ASSERT_EQ(worker0.size(), 1u);
  EXPECT_EQ(worker0.front(), 0u);
  ASSERT_TRUE(state.markComplete(worker0.front()));
  ASSERT_TRUE(state.markComplete(2));
  EXPECT_EQ(state.getNextCompletedOrdinal(), 1u);
  EXPECT_EQ(state.getPendingNCCCount(TargetNCCWorker::Worker0), 0u);
  EXPECT_EQ(state.getPendingNCCCount(TargetNCCWorker::Worker1), 1u);

  ASSERT_TRUE(state.beginIssue(3));
  llvm::SmallVector<uint64_t, 8> worker1 = state.takeNCCParticipantPending(
      uint32_t{1} << static_cast<uint32_t>(TargetNCCWorker::Worker1));
  ASSERT_EQ(worker1.size(), 1u);
  EXPECT_EQ(worker1.front(), 1u);
  ASSERT_TRUE(state.markComplete(worker1.front()));
  ASSERT_TRUE(state.markComplete(3));
  EXPECT_EQ(state.getNextCompletedOrdinal(), 4u);
}

TEST(TargetModelTileCommandTrackerTest,
     SynchronousWritebackCompletesParticipantEpochWithoutParkingItself) {
  wafer::model::detail::TargetModelTileCommandTracker state;
  ASSERT_TRUE(state.beginIssue(0));
  ASSERT_TRUE(state.addNCCPending(TargetNCCWorker::Worker0, 0));
  ASSERT_TRUE(state.hasNCCPending());
  ASSERT_TRUE(state.beginIssue(1)); // Synchronous worker-0 writeback.

  llvm::SmallVector<uint64_t, 8> completed = state.takeNCCParticipantPending(
      uint32_t{1} << static_cast<uint32_t>(TargetNCCWorker::Worker0));
  ASSERT_EQ(completed.size(), 1u);
  ASSERT_TRUE(state.markComplete(completed.front()));
  ASSERT_TRUE(state.markComplete(1));

  EXPECT_FALSE(state.hasNCCPending());
  EXPECT_EQ(state.getNextCompletedOrdinal(), 2u);
  EXPECT_EQ(state.getNextIssuedOrdinal(), 2u);
}

TEST(TargetModelKernelTest, TargetRegisterBoundsFailClosedAtModelEntry) {
  auto validate = [](auto payload) {
    return validateTargetModelCommandFields(
        TargetCommand{CardId(0), TileId(0), LaunchSlotId(0), 0,
                      TargetCommandPayload{std::move(payload)}});
  };
  auto expectInvalid = [&](auto payload, llvm::StringRef expected) {
    std::string error = expectError(validate(std::move(payload)));
    EXPECT_NE(error.find(expected.str()), std::string::npos) << error;
  };
  auto expectValid = [&](auto payload) {
    llvm::Error error = validate(std::move(payload));
    EXPECT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  };

  expectValid(
      TargetGemmCommand{0, 0, 0, 1, 16384, 1, 4096, LogicalFormat::F16});
  expectInvalid(TargetGemmCommand{0, 0, 0, 1, 16385, 1, 1, LogicalFormat::F16},
                "GEMM k must be in [1, 16384]");
  expectInvalid(TargetGemmCommand{0, 0, 0, 1, 1, 1, 4097, LogicalFormat::F16},
                "GEMM batch_count must be in [1, 4096]");

  expectValid(TargetReduceCommand{
      NumericReduceOperation::Sum,
      0,
      0,
      static_cast<uint32_t>(NativeCTReduceDimension::Trailing0),
      {4096, 4096, 4096, 16384},
      LogicalFormat::F32});
  expectInvalid(TargetReduceCommand{NumericReduceOperation::Sum,
                                    0,
                                    0,
                                    static_cast<uint32_t>(
                                        NativeCTReduceDimension::Trailing0),
                                    {4097, 1, 1, 1},
                                    LogicalFormat::F32},
                "reduce shape N dimension must be in [1, 4096]");
  expectInvalid(TargetReduceCommand{NumericReduceOperation::Sum,
                                    0,
                                    0,
                                    static_cast<uint32_t>(
                                        NativeCTReduceDimension::Trailing0),
                                    {1, 1, 1, 16385},
                                    LogicalFormat::F32},
                "reduce shape C dimension must be in [1, 16384]");

  auto makeConv = [] {
    return TargetConvCommand{TargetConvolutionOperation::Convolution,
                             0,
                             0,
                             0,
                             {1, 1, 1, 1},
                             {1, 1, 1, 1},
                             {1, 1, 1, 1},
                             {0, 0, 0, 0},
                             {0, 0, 0, 0},
                             {1, 1, 1, 1},
                             {1, 1},
                             LogicalFormat::F16};
  };
  TargetConvCommand conv = makeConv();
  conv.pads[0] = 1024;
  expectInvalid(std::move(conv), "convolution pads must be in [0, 1023]");
  conv = makeConv();
  conv.kernelStrides[0] = 256;
  expectInvalid(std::move(conv),
                "convolution kernel_strides must be in [1, 255]");
  conv = makeConv();
  conv.kernelStrides[2] = 1024;
  expectInvalid(std::move(conv),
                "convolution kernel_strides must be in [1, 1023]");
  conv = makeConv();
  conv.dilations[0] = 1024;
  expectInvalid(std::move(conv), "convolution dilations must be in [1, 1023]");
}

TEST(TargetModelKernelTest, ConvolutionWeightShapeIsNotADataShape) {
  TargetConvCommand conv{TargetConvolutionOperation::Convolution,
                         0,
                         0,
                         0,
                         {1, 1, 1, 4097},
                         {1, 1, 4097, 1},
                         {1, 1, 1, 1},
                         {0, 0, 0, 0},
                         {0, 0, 0, 0},
                         {1, 1, 1, 1},
                         {1, 1},
                         LogicalFormat::F16};
  llvm::Error error = validateTargetModelCommandFields(
      TargetCommand{CardId(0), TileId(0), LaunchSlotId(0), 0,
                    TargetCommandPayload{conv}});
  EXPECT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  conv.weightShape[2] =
      static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()) + 1;
  std::string failure = expectError(validateTargetModelCommandFields(
      TargetCommand{CardId(0), TileId(0), LaunchSlotId(0), 0,
                    TargetCommandPayload{std::move(conv)}}));
  EXPECT_NE(failure.find("convolution weight shape must be in [1, 65535]"),
            std::string::npos)
      << failure;
}

TEST(TargetModelKernelTest,
     BoolMemsetFieldValidationRequiresPhysicalFootprintByteGranularity) {
  auto validate = [](uint32_t elementCount) {
    return validateTargetModelCommandFields(
        TargetCommand{CardId(0), TileId(0), LaunchSlotId(0), 0,
                      TargetMemsetCommand{0, UINT32_C(1), elementCount,
                                          LogicalFormat::Bool}});
  };

  llvm::Error valid = validate(16);
  EXPECT_FALSE(static_cast<bool>(valid)) << llvm::toString(std::move(valid));
  std::string failure = expectError(validate(9));
  EXPECT_NE(failure.find("physical-footprint element_count must be a multiple "
                         "of 8 for byte granularity"),
            std::string::npos)
      << failure;
}

TEST(TargetModelKernelTest, StridedRDMAAndWDMAApplyOnlyCompleteEffects) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  TargetCommand rdma{CardId(0), TileId(0), LaunchSlotId(0), 0,
                     TargetStridedDMACommand{TargetDMADirection::Read,
                                             kDDRBase,
                                             spm,
                                             8,
                                             4,
                                             {8, 0, 0},
                                             {2, 1, 1},
                                             LogicalFormat::F32}};
  TargetModelCommandEffect readEffect =
      llvm::cantFail(executeTargetModelCommand(rdma, memory, makeBudget()));
  ASSERT_EQ(readEffect.pendingWrites.size(), 1u);
  EXPECT_FALSE(readEffect.pendingWrites.front().stridedLayout.has_value());
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, spm, 8, 1)),
            std::vector<uint8_t>(8, 0));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(readEffect)));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, spm, 8, 1)),
            (std::vector<uint8_t>{0, 1, 2, 3, 8, 9, 10, 11}));

  TargetCommand wdma{CardId(0), TileId(0), LaunchSlotId(0), 1,
                     TargetStridedDMACommand{TargetDMADirection::Write,
                                             spm,
                                             kDDRBase + UINT64_C(0x1000),
                                             8,
                                             4,
                                             {8, 0, 0},
                                             {2, 1, 1},
                                             LogicalFormat::F32}};
  TargetModelCommandEffect writeEffect =
      llvm::cantFail(executeTargetModelCommand(wdma, memory, makeBudget()));
  ASSERT_EQ(writeEffect.pendingWrites.size(), 1u);
  ASSERT_TRUE(writeEffect.pendingWrites.front().stridedLayout.has_value());
  EXPECT_EQ(writeEffect.pendingWrites.front().stridedLayout->iterations,
            (std::array<uint32_t, 3>{2, 1, 1}));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(writeEffect)));
  std::vector<uint8_t> output = llvm::cantFail(memory.readSlotSnapshot(0, 1));
  EXPECT_EQ(std::vector<uint8_t>(output.begin(), output.begin() + 4),
            (std::vector<uint8_t>{0, 1, 2, 3}));
  EXPECT_EQ(std::vector<uint8_t>(output.begin() + 8, output.begin() + 12),
            (std::vector<uint8_t>{8, 9, 10, 11}));
}

TEST(TargetModelKernelTest,
     GatherScatterSnapshotsOverlappingSourceBeforeCompactWrite) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  llvm::cantFail(memory.applyAtomically(
      {TargetModelByteWrite{0,
                            TargetModelAddressSpace::TileSPM,
                            spm,
                            1,
                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}}));
  TargetCommand gather{
      CardId(0), TileId(0), LaunchSlotId(0), 0,
      TargetGatherScatterCommand{
          spm, spm + 2, 8, 2, {2, 0, 0}, {4, 1, 1}, {2, 0, 0}, {4, 1, 1}}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(gather, memory, makeBudget()));
  ASSERT_EQ(effect.pendingWrites.size(), 1u);
  ASSERT_TRUE(effect.pendingWrites.front().stridedLayout.has_value());
  EXPECT_EQ(effect.pendingWrites.front().bytes,
            (std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, spm, 10, 1)),
            (std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(effect)));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, spm, 10, 1)),
            (std::vector<uint8_t>{0, 1, 0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(TargetModelKernelTest, GatherScatterAllowsRepeatedSourceSegments) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  llvm::cantFail(memory.applyAtomically({TargetModelByteWrite{
      0, TargetModelAddressSpace::TileSPM, spm, 1, {7, 9}}}));
  const uint64_t destination = spm + UINT64_C(0x100);
  TargetCommand gather{
      CardId(0), TileId(0), LaunchSlotId(0), 0,
      TargetGatherScatterCommand{
          spm, destination, 4, 2, {0, 0, 0}, {2, 1, 1}, {2, 0, 0}, {2, 1, 1}}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(gather, memory, makeBudget()));
  ASSERT_EQ(effect.pendingWrites.size(), 1u);
  EXPECT_EQ(effect.pendingWrites.front().bytes,
            (std::vector<uint8_t>{7, 9, 7, 9}));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(effect)));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, destination, 4, 1)),
            (std::vector<uint8_t>{7, 9, 7, 9}));
}

TEST(TargetModelKernelTest,
     OverlappingGatherDestinationFailsWithoutPartialWrites) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  llvm::cantFail(memory.applyAtomically({TargetModelByteWrite{
      0, TargetModelAddressSpace::TileSPM, spm, 1, {1, 2, 3, 4}}}));
  const uint64_t destination = spm + UINT64_C(0x100);
  TargetCommand gather{
      CardId(0), TileId(0), LaunchSlotId(0), 0,
      TargetGatherScatterCommand{
          spm, destination, 4, 2, {2, 0, 0}, {2, 1, 1}, {0, 0, 0}, {2, 1, 1}}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(gather, memory, makeBudget()));
  ASSERT_EQ(effect.pendingWrites.size(), 1u);
  std::string error = expectError(
      applyTargetModelCommandEffect(memory, config, std::move(effect)));
  EXPECT_NE(error.find("pending writes overlap"), std::string::npos);
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, destination, 4, 1)),
            std::vector<uint8_t>(4, 0));
  EXPECT_FALSE(config.getAggregateFlags().any());
}

TEST(TargetModelKernelTest,
     NonCanonicalMovementUsesLinearFallbackWithoutChangingDescriptorOrder) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  TargetCommand rdma{CardId(0), TileId(0), LaunchSlotId(0), 0,
                     TargetStridedDMACommand{TargetDMADirection::Read,
                                             kDDRBase,
                                             spm,
                                             12,
                                             2,
                                             {4, 6, 0},
                                             {3, 2, 1},
                                             LogicalFormat::F16}};
  TargetModelCommandEffect readEffect =
      llvm::cantFail(executeTargetModelCommand(rdma, memory, makeBudget()));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(readEffect)));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, spm, 12, 1)),
            (std::vector<uint8_t>{0, 1, 4, 5, 8, 9, 6, 7, 10, 11, 14, 15}));

  TargetCommand wdma{CardId(0), TileId(0), LaunchSlotId(0), 1,
                     TargetStridedDMACommand{TargetDMADirection::Write,
                                             spm,
                                             kDDRBase + UINT64_C(0x1000),
                                             12,
                                             2,
                                             {4, 6, 0},
                                             {3, 2, 1},
                                             LogicalFormat::F16}};
  TargetModelCommandEffect writeEffect =
      llvm::cantFail(executeTargetModelCommand(wdma, memory, makeBudget()));
  ASSERT_EQ(writeEffect.pendingWrites.size(), 1u);
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(writeEffect)));
  std::vector<uint8_t> output = llvm::cantFail(memory.readSlotSnapshot(0, 1));
  EXPECT_EQ(std::vector<uint8_t>(output.begin(), output.begin() + 16),
            (std::vector<uint8_t>{0, 1, 0, 0, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0,
                                  14, 15}));
}

TEST(TargetModelKernelTest,
     LargeTransposeMovementKeepsOneEffectFor352256Segments) {
  constexpr uint32_t kColumns = 688;
  constexpr uint32_t kRows = 512;
  constexpr uint32_t kSegmentCount = kColumns * kRows;
  constexpr uint32_t kInnerBytes = 2;
  constexpr uint32_t kPayloadBytes = kSegmentCount * kInnerBytes;
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  const uint64_t transposed = spm + UINT64_C(0x100000);
  const uint64_t roundTrip = spm + UINT64_C(0x200000);
  std::vector<uint8_t> payload(kPayloadBytes);
  for (uint64_t segment = 0; segment < kSegmentCount; ++segment) {
    payload[segment * 2] = static_cast<uint8_t>(segment);
    payload[segment * 2 + 1] = static_cast<uint8_t>(segment >> 8);
  }
  llvm::cantFail(memory.applyAtomically({TargetModelByteWrite{
      0, TargetModelAddressSpace::TileSPM, spm, 1, payload}}));

  TargetModelKernelBudget largeBudget = TargetModelKernelBudget::create(
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1,
                                      /*maximumFusedMultiplyAdds=*/1),
      kPayloadBytes, kSegmentCount);
  TargetCommand transpose{
      CardId(0), TileId(0), LaunchSlotId(0), 0,
      TargetGatherScatterCommand{spm,
                                 transposed,
                                 kPayloadBytes,
                                 kInnerBytes,
                                 {kInnerBytes, kColumns * kInnerBytes, 0},
                                 {kColumns, kRows, 1},
                                 {kRows * kInnerBytes, kInnerBytes, 0},
                                 {kColumns, kRows, 1}}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(transpose, memory, largeBudget));
  ASSERT_EQ(effect.pendingWrites.size(), 1u);
  ASSERT_TRUE(effect.pendingWrites.front().stridedLayout.has_value());
  EXPECT_EQ(effect.pendingWrites.front().bytes.size(), kPayloadBytes);
  EXPECT_EQ(effect.pendingWrites.front().stridedLayout->strides,
            (std::array<uint32_t, 3>{kRows * kInnerBytes, kInnerBytes, 0}));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(effect)));
  std::vector<uint8_t> output = llvm::cantFail(memory.readSnapshot(
      0, TargetModelAddressSpace::TileSPM, transposed, kPayloadBytes, 1));
  auto expectTransposedSegment = [&](uint32_t column, uint32_t row) {
    const uint64_t sourceSegment = row * kColumns + column;
    const uint64_t destinationOffset = (column * kRows + row) * kInnerBytes;
    EXPECT_EQ(output[destinationOffset], payload[sourceSegment * 2]);
    EXPECT_EQ(output[destinationOffset + 1], payload[sourceSegment * 2 + 1]);
  };
  expectTransposedSegment(0, 0);
  expectTransposedSegment(1, 0);
  expectTransposedSegment(0, 1);
  expectTransposedSegment(kColumns / 2, kRows / 2);
  expectTransposedSegment(kColumns - 1, kRows - 1);

  TargetCommand reverse{
      CardId(0), TileId(0), LaunchSlotId(0), 1,
      TargetGatherScatterCommand{transposed,
                                 roundTrip,
                                 kPayloadBytes,
                                 kInnerBytes,
                                 {kRows * kInnerBytes, kInnerBytes, 0},
                                 {kColumns, kRows, 1},
                                 {kInnerBytes, kColumns * kInnerBytes, 0},
                                 {kColumns, kRows, 1}}};
  TargetModelCommandEffect reverseEffect =
      llvm::cantFail(executeTargetModelCommand(reverse, memory, largeBudget));
  ASSERT_EQ(reverseEffect.pendingWrites.size(), 1u);
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(reverseEffect)));
  EXPECT_EQ(
      llvm::cantFail(memory.readSnapshot(0, TargetModelAddressSpace::TileSPM,
                                         roundTrip, kPayloadBytes, 1)),
      payload);
}

TEST(TargetModelKernelTest, ElementwiseUsesPhysicalCodecAndFormalNumeric) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  NumericTensorKey key =
      makeTensor(LogicalFormat::F32, PhysicalTensorLayout::Cx, {4});
  writeTensor(memory, 0, spm, key,
              {{LogicalFormat::F32, UINT64_C(0x3f800000)},
               {LogicalFormat::F32, UINT64_C(0x40000000)},
               {LogicalFormat::F32, UINT64_C(0x40400000)},
               {LogicalFormat::F32, UINT64_C(0x40800000)}});
  writeTensor(memory, 0, spm + UINT64_C(0x1000), key,
              {{LogicalFormat::F32, UINT64_C(0x40800000)},
               {LogicalFormat::F32, UINT64_C(0x40400000)},
               {LogicalFormat::F32, UINT64_C(0x40000000)},
               {LogicalFormat::F32, UINT64_C(0x3f800000)}});
  TargetCommand add{CardId(0), TileId(0), LaunchSlotId(0), 0,
                    TargetElementwiseCommand{NumericElementwiseOperation::Add,
                                             spm, spm + UINT64_C(0x1000),
                                             spm + UINT64_C(0x2000), 4,
                                             LogicalFormat::F32}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(add, memory, makeBudget()));
  std::vector<RawLogicalValue> before =
      readTensor(memory, 0, spm + UINT64_C(0x2000), key);
  for (const RawLogicalValue &value : before)
    EXPECT_EQ(value.bits, UINT64_C(0));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(effect)));
  std::vector<RawLogicalValue> result =
      readTensor(memory, 0, spm + UINT64_C(0x2000), key);
  ASSERT_EQ(result.size(), 4u);
  for (const RawLogicalValue &value : result)
    EXPECT_EQ(value.bits, UINT64_C(0x40a00000));
  EXPECT_FALSE(config.getAggregateFlags().any());
}

TEST(TargetModelKernelTest, NativeF32SumUsesFixedShapeABIAndFormalNumeric) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  NumericTensorKey input =
      makeTensor(LogicalFormat::F32, PhysicalTensorLayout::NCx, {1, 1, 2, 2});
  NumericTensorKey destination =
      makeTensor(LogicalFormat::F32, PhysicalTensorLayout::NCx, {1, 1, 2});
  writeTensor(memory, 0, spm, input,
              {{LogicalFormat::F32, UINT64_C(0x3f800000)},
               {LogicalFormat::F32, UINT64_C(0x40000000)},
               {LogicalFormat::F32, UINT64_C(0x40400000)},
               {LogicalFormat::F32, UINT64_C(0x40800000)}});
  TargetCommand reduce{
      CardId(0), TileId(0), LaunchSlotId(0), 0,
      TargetReduceCommand{
          NumericReduceOperation::Sum,
          spm,
          spm + UINT64_C(0x1000),
          static_cast<uint32_t>(NativeCTReduceDimension::Trailing0),
          {1, 1, 2, 2},
          LogicalFormat::F32}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(reduce, memory, makeBudget()));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(effect)));
  std::vector<RawLogicalValue> result =
      readTensor(memory, 0, spm + UINT64_C(0x1000), destination);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0].bits, UINT64_C(0x40400000));
  EXPECT_EQ(result[1].bits, UINT64_C(0x40e00000));
  EXPECT_FALSE(config.getAggregateFlags().any());
}

TEST(TargetModelKernelTest, ConvertAndGemmUseResolvedFormalCommands) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  NumericTensorKey f32 =
      makeTensor(LogicalFormat::F32, PhysicalTensorLayout::Cx, {2});
  NumericTensorKey f16 =
      makeTensor(LogicalFormat::F16, PhysicalTensorLayout::Cx, {2});
  writeTensor(memory, 0, spm, f32,
              {{LogicalFormat::F32, UINT64_C(0x3f800000)},
               {LogicalFormat::F32, UINT64_C(0x40000000)}});
  writeTensor(memory, 0, spm + UINT64_C(0x1000), f16,
              {{LogicalFormat::F16, 0}, {LogicalFormat::F16, 0}});
  TargetCommand convert{
      CardId(0), TileId(0), LaunchSlotId(0), 0,
      TargetConvertCommand{
          llvm::cantFail(TargetConvertOperation::create(
              findTargetConvertRoute(LogicalFormat::F32, LogicalFormat::F16)
                  ->opcode)),
          spm, spm + UINT64_C(0x1000), 2, std::nullopt, 0}};
  TargetModelCommandEffect convertEffect =
      llvm::cantFail(executeTargetModelCommand(convert, memory, makeBudget()));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(convertEffect)));
  std::vector<RawLogicalValue> converted =
      readTensor(memory, 0, spm + UINT64_C(0x1000), f16);
  EXPECT_EQ(converted[0].bits, UINT64_C(0x3c00));
  EXPECT_EQ(converted[1].bits, UINT64_C(0x4000));

  NumericTensorKey matrix =
      makeTensor(LogicalFormat::F16, PhysicalTensorLayout::Cx, {2, 2});
  writeTensor(memory, 0, spm + UINT64_C(0x3000), matrix,
              {{LogicalFormat::F16, UINT64_C(0x3c00)},
               {LogicalFormat::F16, UINT64_C(0)},
               {LogicalFormat::F16, UINT64_C(0)},
               {LogicalFormat::F16, UINT64_C(0x3c00)}});
  writeTensor(memory, 0, spm + UINT64_C(0x4000), matrix,
              {{LogicalFormat::F16, UINT64_C(0x4000)},
               {LogicalFormat::F16, UINT64_C(0x4200)},
               {LogicalFormat::F16, UINT64_C(0x4400)},
               {LogicalFormat::F16, UINT64_C(0x4500)}});
  TargetCommand gemm{CardId(0), TileId(0), LaunchSlotId(0), 1,
                     TargetGemmCommand{spm + UINT64_C(0x3000),
                                       spm + UINT64_C(0x4000),
                                       spm + UINT64_C(0x5000), 2, 2, 2, 1,
                                       LogicalFormat::F16}};
  TargetModelCommandEffect gemmEffect =
      llvm::cantFail(executeTargetModelCommand(gemm, memory, makeBudget()));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(gemmEffect)));
  std::vector<RawLogicalValue> product =
      readTensor(memory, 0, spm + UINT64_C(0x5000), matrix);
  ASSERT_EQ(product.size(), 4u);
  EXPECT_EQ(product[0].bits, UINT64_C(0x4000));
  EXPECT_EQ(product[1].bits, UINT64_C(0x4200));
  EXPECT_EQ(product[2].bits, UINT64_C(0x4400));
  EXPECT_EQ(product[3].bits, UINT64_C(0x4500));
}

TEST(TargetModelKernelTest,
     PhysicalFootprintMemsetOverwritesAlignedTailAndUnusedBoolBits) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  const uint64_t cxDestination = spm + UINT64_C(0x3000);
  const uint64_t boolDestination = spm + UINT64_C(0x4000);
  llvm::cantFail(memory.applyAtomically(
      {TargetModelByteWrite{0, TargetModelAddressSpace::TileSPM, cxDestination,
                            1, std::vector<uint8_t>(256, UINT8_C(0xa5))},
       TargetModelByteWrite{0, TargetModelAddressSpace::TileSPM,
                            boolDestination, 1,
                            std::vector<uint8_t>(2, UINT8_C(0xa5))}}));

  TargetCommand cxFill{CardId(0), TileId(0), LaunchSlotId(0), 0,
                       TargetMemsetCommand{cxDestination, UINT32_C(0x3555),
                                           /*elementCount=*/128,
                                           LogicalFormat::F16}};
  TargetModelCommandEffect cxEffect =
      llvm::cantFail(executeTargetModelCommand(cxFill, memory, makeBudget()));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(cxEffect)));
  std::vector<uint8_t> cxBytes = llvm::cantFail(memory.readSnapshot(
      0, TargetModelAddressSpace::TileSPM, cxDestination, 256, 1));
  for (size_t index = 0; index < cxBytes.size(); index += 2) {
    EXPECT_EQ(cxBytes[index], UINT8_C(0x55));
    EXPECT_EQ(cxBytes[index + 1], UINT8_C(0x35));
  }

  TargetCommand boolFill{
      CardId(0), TileId(0), LaunchSlotId(0), 1,
      TargetMemsetCommand{boolDestination, UINT32_C(1),
                          /*elementCount=*/16, LogicalFormat::Bool}};
  TargetModelCommandEffect boolEffect =
      llvm::cantFail(executeTargetModelCommand(boolFill, memory, makeBudget()));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(boolEffect)));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::TileSPM, boolDestination, 2, 1)),
            (std::vector<uint8_t>{UINT8_C(0xff), UINT8_C(0xff)}));
}

TEST(TargetModelKernelTest, BatchedGemmUsesImplicitNCxStorageContract) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  NumericTensorKey batchMatrices =
      makeTensor(LogicalFormat::F32, PhysicalTensorLayout::NCx, {2, 2, 2});
  writeTensor(memory, 0, spm, batchMatrices,
              {{LogicalFormat::F32, UINT64_C(0x3f800000)},
               {LogicalFormat::F32, UINT64_C(0x40000000)},
               {LogicalFormat::F32, UINT64_C(0x40400000)},
               {LogicalFormat::F32, UINT64_C(0x40800000)},
               {LogicalFormat::F32, UINT64_C(0x40a00000)},
               {LogicalFormat::F32, UINT64_C(0x40c00000)},
               {LogicalFormat::F32, UINT64_C(0x40e00000)},
               {LogicalFormat::F32, UINT64_C(0x41000000)}});
  writeTensor(memory, 0, spm + UINT64_C(0x1000), batchMatrices,
              {{LogicalFormat::F32, UINT64_C(0x3f800000)},
               {LogicalFormat::F32, UINT64_C(0)},
               {LogicalFormat::F32, UINT64_C(0)},
               {LogicalFormat::F32, UINT64_C(0x3f800000)},
               {LogicalFormat::F32, UINT64_C(0x40000000)},
               {LogicalFormat::F32, UINT64_C(0x3f800000)},
               {LogicalFormat::F32, UINT64_C(0x3f800000)},
               {LogicalFormat::F32, UINT64_C(0x40000000)}});
  TargetCommand gemm{CardId(0), TileId(0), LaunchSlotId(0), 0,
                     TargetGemmCommand{spm, spm + UINT64_C(0x1000),
                                       spm + UINT64_C(0x2000), 2, 2, 2, 2,
                                       LogicalFormat::F32}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(gemm, memory, makeBudget()));
  llvm::cantFail(
      applyTargetModelCommandEffect(memory, config, std::move(effect)));

  std::vector<RawLogicalValue> product =
      readTensor(memory, 0, spm + UINT64_C(0x2000), batchMatrices);
  ASSERT_EQ(product.size(), 8u);
  const std::vector<uint64_t> expected{
      UINT64_C(0x3f800000), UINT64_C(0x40000000), UINT64_C(0x40400000),
      UINT64_C(0x40800000), UINT64_C(0x41800000), UINT64_C(0x41880000),
      UINT64_C(0x41b00000), UINT64_C(0x41b80000)};
  for (auto [value, expectedBits] : llvm::zip_equal(product, expected))
    EXPECT_EQ(value.bits, expectedBits);
}

TEST(TargetModelKernelTest,
     BulkThenFormalUsesMatchingBackendOrFailsBeyondFormalBudget) {
  InvocationMemoryRegistry memory = makeRegistry();
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  NumericTensorKey matrix =
      makeTensor(LogicalFormat::F32, PhysicalTensorLayout::Cx, {4, 4});
  std::vector<RawLogicalValue> values(
      16, {LogicalFormat::F32, UINT64_C(0x3f800000)});
  writeTensor(memory, 0, spm, matrix, values);
  writeTensor(memory, 0, spm + UINT64_C(0x1000), matrix, values);
  TargetCommand gemm{CardId(0), TileId(0), LaunchSlotId(0), 0,
                     TargetGemmCommand{spm, spm + UINT64_C(0x1000),
                                       spm + UINT64_C(0x2000), 4, 4, 4, 1,
                                       LogicalFormat::F32}};
  TargetModelKernelBudget smallBudget = TargetModelKernelBudget::create(
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1,
                                      /*maximumFusedMultiplyAdds=*/1),
      /*maximumMovementBytes=*/4096,
      /*maximumMovementSegments=*/256);

  FakeBulkBackend missing(/*hasMatchingExecution=*/false);
  std::string error = expectError(executeTargetModelCommand(
      gemm, memory, smallBudget,
      TargetModelExecutionPolicy::bulkThenFormal(missing)));
  EXPECT_NE(error.find("bulk-backend-unavailable"), std::string::npos);
  EXPECT_EQ(missing.invocations, 1u);

  FakeBulkBackend matched(/*hasMatchingExecution=*/true);
  TargetModelCommandEffect effect = llvm::cantFail(executeTargetModelCommand(
      gemm, memory, smallBudget,
      TargetModelExecutionPolicy::bulkThenFormal(matched)));
  EXPECT_EQ(effect.numericBackend, TargetModelNumericBackend::Bulk);
  EXPECT_EQ(effect.bulkEvidence.matmulInvocations, 1u);
  EXPECT_EQ(effect.bulkEvidence.formalFusedMultiplyAdds, 0u);
  EXPECT_EQ(effect.bulkEvidence.evidenceKind,
            TargetModelBulkEvidenceKind::ExactQualificationRecord);
  EXPECT_EQ(effect.bulkEvidence.evidenceDigest,
            "sha256:fake-qualification-record");
  EXPECT_EQ(matched.invocations, 1u);
}

TEST(TargetModelKernelTest, FailedEffectDoesNotModifyBytesOrNumericFlags) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext config;
  FormalNumericExceptionFlags flags;
  flags.inexact = true;
  TargetModelCommandEffect invalid{
      {TargetModelByteWrite{
          0, TargetModelAddressSpace::CardDDR, kDDRBase, 1, {9}}},
      flags,
      TargetModelControlAction::None};
  std::string error = expectError(
      applyTargetModelCommandEffect(memory, config, std::move(invalid)));
  EXPECT_NE(error.find("access-denied"), std::string::npos);
  EXPECT_FALSE(config.getAggregateFlags().any());
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::CardDDR, kDDRBase, 1, 1)),
            (std::vector<uint8_t>{0}));
}

TEST(TargetModelKernelTest, ControlCommandsValidateTypedEndpoints) {
  InvocationMemoryRegistry memory = makeRegistry(2);
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  TargetCommand begin{
      CardId(0), TileId(0), LaunchSlotId(0), 0,
      TargetDirectDTEBeginCommand{kDDRBase + UINT64_C(0x2000), 2}};
  EXPECT_EQ(
      llvm::cantFail(executeTargetModelCommand(begin, memory, makeBudget()))
          .controlAction,
      TargetModelControlAction::DirectDTEBegin);
  TargetCommand send{CardId(0), TileId(0), LaunchSlotId(0), 1,
                     TargetDirectDTESendCommand{spm, spm, 16, 0, 1, 0, false}};
  EXPECT_EQ(
      llvm::cantFail(executeTargetModelCommand(send, memory, makeBudget()))
          .controlAction,
      TargetModelControlAction::DirectDTESendPrepare);
  TargetCommand sendIssue{CardId(0), TileId(0), LaunchSlotId(0),
                          2, TargetDirectDTESendIssueCommand{UINT64_C(0x100)}};
  EXPECT_EQ(
      llvm::cantFail(executeTargetModelCommand(sendIssue, memory, makeBudget()))
          .controlAction,
      TargetModelControlAction::DirectDTESendIssue);
  std::get<TargetDirectDTESendCommand>(send.payload).localTile = 65536;
  std::string error =
      expectError(executeTargetModelCommand(send, memory, makeBudget()));
  EXPECT_NE(error.find("outside the accepted target ABI domain"),
            std::string::npos);

  auto &sendPayload = std::get<TargetDirectDTESendCommand>(send.payload);
  sendPayload.localTile = 0;
  sendPayload.remoteFSM = 4;
  error = expectError(executeTargetModelCommand(send, memory, makeBudget()));
  EXPECT_NE(error.find("outside the accepted target ABI domain"),
            std::string::npos);

  sendPayload.remoteFSM = 0;
  sendPayload.highPerformance = true;
  error = expectError(executeTargetModelCommand(send, memory, makeBudget()));
  EXPECT_NE(error.find("unsupported-command"), std::string::npos);
}

TEST(TargetModelKernelTest, UnsupportedFamilyIsNotSilentlyApproximated) {
  InvocationMemoryRegistry memory = makeRegistry();
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  TargetCommand bit2fp{
      CardId(0), TileId(0), LaunchSlotId(0), 0,
      TargetBit2FPCommand{spm, spm + UINT64_C(0x1000), 4, LogicalFormat::F32}};
  std::string error =
      expectError(executeTargetModelCommand(bit2fp, memory, makeBudget()));
  EXPECT_NE(error.find("unsupported-command"), std::string::npos);
}

} // namespace
