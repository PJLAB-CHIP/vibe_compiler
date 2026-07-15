//===- TargetModelKernelTest.cpp - Plain target kernel tests -------------===//

#include "Wafer/Model/TargetModelKernel.h"

#include "Wafer/Target/PhysicalTensorCodec.h"
#include "Wafer/Target/TargetCall.h"
#include "Wafer/Target/TargetFormat.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

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

TargetCallInvocationDescriptor makeInvocation(size_t rankCount = 1) {
  std::vector<TargetCallRankDescriptor> ranks;
  for (size_t rank = 0; rank < rankCount; ++rank) {
    const uint64_t rankBase = kDDRBase + rank * UINT64_C(0x10000);
    ranks.push_back(TargetCallRankDescriptor{
        static_cast<int64_t>(rank),
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
          4,
          4}},
        {rankBase, rankBase + UINT64_C(0x1000), rankBase + UINT64_C(0x2000)},
        TargetIdentityId::waferTx81SingleCard(),
        KernelRuntimeABIId::waferTx81KernelV1()});
  }
  return {TargetProfileId::waferTx81SingleCardKernelV1(), std::move(ranks)};
}

InvocationMemoryRegistry makeRegistry(size_t rankCount = 1) {
  std::vector<TargetModelInputBinding> inputs;
  for (size_t rank = 0; rank < rankCount; ++rank) {
    std::vector<uint8_t> bytes(256);
    for (size_t index = 0; index < bytes.size(); ++index)
      bytes[index] = static_cast<uint8_t>(index);
    inputs.push_back({static_cast<int64_t>(rank), 0, std::move(bytes)});
  }
  return llvm::cantFail(InvocationMemoryRegistry::create(llvm::cantFail(
      InvocationAddressPlan::create(makeInvocation(rankCount), inputs))));
}

NumericTensorKey makeTensor(LogicalFormat format, NumericTensorLayout layout,
                            std::vector<uint64_t> shape) {
  return llvm::cantFail(
      NumericTensorKey::create(format, layout, std::move(shape)));
}

void writeTensor(InvocationMemoryRegistry &memory, int64_t rank,
                 uint64_t address, const NumericTensorKey &key,
                 llvm::ArrayRef<RawLogicalValue> values) {
  std::vector<uint8_t> bytes =
      llvm::cantFail(packPhysicalTensorLogicalValues(key, values, UINT8_C(0)));
  llvm::cantFail(memory.applyAtomically({TargetModelByteWrite{
      rank, TargetModelAddressSpace::RankSPM, address, 1, std::move(bytes)}}));
}

std::vector<RawLogicalValue> readTensor(const InvocationMemoryRegistry &memory,
                                        int64_t rank, uint64_t address,
                                        const NumericTensorKey &key) {
  uint64_t bytes = llvm::cantFail(getPhysicalTensorStorageBytes(key));
  return llvm::cantFail(unpackPhysicalTensorLogicalValues(
      key, llvm::cantFail(memory.readSnapshot(
               rank, TargetModelAddressSpace::RankSPM, address, bytes, 1))));
}

uint64_t supportedFormatCode(TargetFormatEngine engine, LogicalFormat format) {
  const TargetFormatEncodingRecord *record = findTargetFormatEncoding(
      TargetProfileId::waferTx81SingleCardKernelV1(), engine, format);
  assert(record && record->isSupported() && record->dataFormatCode);
  return *record->dataFormatCode;
}

uint64_t supportedF32Code(TargetFormatEngine engine) {
  return supportedFormatCode(engine, LogicalFormat::F32);
}

class FakeBulkBackend final : public TargetModelBulkBackend {
public:
  explicit FakeBulkBackend(bool admit) : admit(admit) {}

  llvm::Expected<std::optional<TargetModelBulkResult>>
  tryExecute(const TargetModelBulkRequest &request) const override {
    ++invocations;
    if (!admit)
      return std::optional<TargetModelBulkResult>();
    TargetModelBulkResult result{
        request.destinationTemplate,
        {},
        {1, 1, 0, "sha256:fake-admission", "fake-bulk"}};
    return std::optional<TargetModelBulkResult>(std::move(result));
  }

  mutable uint64_t invocations = 0;

private:
  bool admit;
};

std::vector<uint64_t>
makeFieldValidArguments(const TargetCallDescriptor &descriptor) {
  std::vector<uint64_t> arguments;
  arguments.reserve(descriptor.arguments.size());
  for (auto [index, scalar] : llvm::enumerate(descriptor.arguments))
    arguments.push_back(scalar == TargetCallScalarType::I64
                            ? kDDRBase + index * UINT64_C(0x1000)
                            : UINT64_C(1));

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
    case TargetCallBuiltin::TDMAPad:
      arguments[14] = supportedF32Code(TargetFormatEngine::TDMA);
      break;
    case TargetCallBuiltin::TDMAImg2Col:
      arguments[18] = supportedF32Code(TargetFormatEngine::TDMA);
      break;
    case TargetCallBuiltin::DirectDTEBegin:
      arguments[1] = 16;
      break;
    case TargetCallBuiltin::DirectDTESendPrepare:
      arguments[6] = 1;
      break;
    case TargetCallBuiltin::LocalFence:
    case TargetCallBuiltin::DirectDTERecvPrepare:
    case TargetCallBuiltin::DirectDTEWait:
    case TargetCallBuiltin::DirectDTEFinish:
      break;
    }
    return arguments;
  }
  if (const auto *kind =
          std::get_if<InstrElementwiseKind>(&descriptor.semantic)) {
    const bool logic = *kind == InstrElementwiseKind::LogicNot ||
                       *kind == InstrElementwiseKind::LogicAnd ||
                       *kind == InstrElementwiseKind::LogicOr ||
                       *kind == InstrElementwiseKind::LogicXor;
    arguments.back() =
        supportedFormatCode(TargetFormatEngine::CT,
                            logic ? LogicalFormat::Bool : LogicalFormat::F32);
    return arguments;
  }
  if (std::holds_alternative<InstrReduceKind>(descriptor.semantic)) {
    arguments[2] = 0;
    arguments[7] = supportedF32Code(TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *kind = std::get_if<InstrConvKind>(&descriptor.semantic)) {
    arguments[3] = static_cast<uint32_t>(*kind);
    arguments[30] = supportedF32Code(TargetFormatEngine::NE);
    return arguments;
  }
  if (const auto *kind = std::get_if<InstrPoolKind>(&descriptor.semantic)) {
    const bool indexed = *kind == InstrPoolKind::IndexedMax ||
                         *kind == InstrPoolKind::IndexedMin;
    const size_t firstField = indexed ? 3 : 2;
    arguments[firstField] = static_cast<uint32_t>(*kind);
    arguments[firstField + 17] = supportedF32Code(TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *kind = std::get_if<InstrUnpoolKind>(&descriptor.semantic)) {
    arguments[2] = static_cast<uint32_t>(*kind);
    arguments[16] = supportedF32Code(TargetFormatEngine::CT);
    return arguments;
  }
  if (const auto *kind =
          std::get_if<InstrPeripheralKind>(&descriptor.semantic)) {
    size_t addressCount = 0;
    switch (*kind) {
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
      llvm_unreachable("unregistered peripheral kind");
    }
    arguments[addressCount] = static_cast<uint32_t>(*kind);
    arguments[addressCount + 2] = supportedF32Code(TargetFormatEngine::CT);
  }
  return arguments;
}

TEST(TargetModelKernelTest, EveryTypedCallPayloadHasClosedFieldValidation) {
  TargetCallDecodeContext context{
      TargetProfileId::waferTx81SingleCardKernelV1(), 16};
  size_t validated = 0;
  for (const TargetCallDescriptor &descriptor : getTargetCallDescriptors()) {
    llvm::Expected<TargetTransactionPayload> payload = decodeTargetCallPayload(
        descriptor, context, makeFieldValidArguments(descriptor));
    ASSERT_TRUE(static_cast<bool>(payload))
        << descriptor.symbol << ": " << llvm::toString(payload.takeError());
    llvm::Error error = validateTargetModelTransactionFields(
        TargetTransaction{0, validated, std::move(*payload)});
    ASSERT_FALSE(static_cast<bool>(error))
        << descriptor.symbol << ": " << llvm::toString(std::move(error));
    ++validated;
  }
  EXPECT_EQ(validated, 109u);
}

TEST(TargetModelKernelTest, StridedRDMAAndWDMACommitOnlyCompleteEffects) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext context;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  TargetTransaction rdma{0, 0,
                         TargetStridedDMATransaction{TargetDMADirection::Read,
                                                     kDDRBase,
                                                     spm,
                                                     8,
                                                     4,
                                                     {8, 0, 0},
                                                     {2, 1, 1},
                                                     LogicalFormat::F32}};
  TargetModelCommandEffect readEffect =
      llvm::cantFail(executeTargetModelCommand(rdma, memory, makeBudget()));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::RankSPM, spm, 8, 1)),
            std::vector<uint8_t>(8, 0));
  llvm::cantFail(
      commitTargetModelCommandEffect(memory, context, std::move(readEffect)));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::RankSPM, spm, 8, 1)),
            (std::vector<uint8_t>{0, 1, 2, 3, 8, 9, 10, 11}));

  TargetTransaction wdma{
      0, 1,
      TargetStridedDMATransaction{TargetDMADirection::Write,
                                  spm,
                                  kDDRBase + UINT64_C(0x1000),
                                  8,
                                  4,
                                  {8, 0, 0},
                                  {2, 1, 1},
                                  LogicalFormat::F32}};
  TargetModelCommandEffect writeEffect =
      llvm::cantFail(executeTargetModelCommand(wdma, memory, makeBudget()));
  llvm::cantFail(
      commitTargetModelCommandEffect(memory, context, std::move(writeEffect)));
  std::vector<uint8_t> output = llvm::cantFail(memory.readSlotSnapshot(0, 1));
  EXPECT_EQ(std::vector<uint8_t>(output.begin(), output.begin() + 4),
            (std::vector<uint8_t>{0, 1, 2, 3}));
  EXPECT_EQ(std::vector<uint8_t>(output.begin() + 8, output.begin() + 12),
            (std::vector<uint8_t>{8, 9, 10, 11}));
}

TEST(TargetModelKernelTest, ElementwiseUsesPhysicalCodecAndFormalNumeric) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext context;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  NumericTensorKey key =
      makeTensor(LogicalFormat::F32, NumericTensorLayout::Cx, {4});
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
  TargetTransaction add{0, 0,
                        TargetElementwiseTransaction{
                            InstrElementwiseKind::Add, spm,
                            spm + UINT64_C(0x1000), spm + UINT64_C(0x2000), 4,
                            LogicalFormat::F32}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(add, memory, makeBudget()));
  std::vector<RawLogicalValue> before =
      readTensor(memory, 0, spm + UINT64_C(0x2000), key);
  for (const RawLogicalValue &value : before)
    EXPECT_EQ(value.bits, UINT64_C(0));
  llvm::cantFail(
      commitTargetModelCommandEffect(memory, context, std::move(effect)));
  std::vector<RawLogicalValue> result =
      readTensor(memory, 0, spm + UINT64_C(0x2000), key);
  ASSERT_EQ(result.size(), 4u);
  for (const RawLogicalValue &value : result)
    EXPECT_EQ(value.bits, UINT64_C(0x40a00000));
  EXPECT_FALSE(context.getAggregateFlags().any());
}

TEST(TargetModelKernelTest, ConvertAndGemmUseResolvedFormalCommands) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext context;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  NumericTensorKey f32 =
      makeTensor(LogicalFormat::F32, NumericTensorLayout::Cx, {2});
  NumericTensorKey f16 =
      makeTensor(LogicalFormat::F16, NumericTensorLayout::Cx, {2});
  writeTensor(memory, 0, spm, f32,
              {{LogicalFormat::F32, UINT64_C(0x3f800000)},
               {LogicalFormat::F32, UINT64_C(0x40000000)}});
  writeTensor(memory, 0, spm + UINT64_C(0x1000), f16,
              {{LogicalFormat::F16, 0}, {LogicalFormat::F16, 0}});
  TargetTransaction convert{
      0, 0,
      TargetConvertTransaction{InstrConvertKind::Fp32Fp16, spm,
                               spm + UINT64_C(0x1000), 2, std::nullopt, 0}};
  TargetModelCommandEffect convertEffect =
      llvm::cantFail(executeTargetModelCommand(convert, memory, makeBudget()));
  llvm::cantFail(commitTargetModelCommandEffect(memory, context,
                                                std::move(convertEffect)));
  std::vector<RawLogicalValue> converted =
      readTensor(memory, 0, spm + UINT64_C(0x1000), f16);
  EXPECT_EQ(converted[0].bits, UINT64_C(0x3c00));
  EXPECT_EQ(converted[1].bits, UINT64_C(0x4000));

  NumericTensorKey matrix =
      makeTensor(LogicalFormat::F16, NumericTensorLayout::Cx, {2, 2});
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
  TargetTransaction gemm{0, 1,
                         TargetGemmTransaction{spm + UINT64_C(0x3000),
                                               spm + UINT64_C(0x4000),
                                               spm + UINT64_C(0x5000), 2, 2, 2,
                                               1, LogicalFormat::F16}};
  TargetModelCommandEffect gemmEffect =
      llvm::cantFail(executeTargetModelCommand(gemm, memory, makeBudget()));
  llvm::cantFail(
      commitTargetModelCommandEffect(memory, context, std::move(gemmEffect)));
  std::vector<RawLogicalValue> product =
      readTensor(memory, 0, spm + UINT64_C(0x5000), matrix);
  ASSERT_EQ(product.size(), 4u);
  EXPECT_EQ(product[0].bits, UINT64_C(0x4000));
  EXPECT_EQ(product[1].bits, UINT64_C(0x4200));
  EXPECT_EQ(product[2].bits, UINT64_C(0x4400));
  EXPECT_EQ(product[3].bits, UINT64_C(0x4500));
}

TEST(TargetModelKernelTest,
     PreferAdmittedGemmUsesExactBackendOrFailsBeyondFormalBudget) {
  InvocationMemoryRegistry memory = makeRegistry();
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  NumericTensorKey matrix =
      makeTensor(LogicalFormat::F32, NumericTensorLayout::Cx, {4, 4});
  std::vector<RawLogicalValue> values(
      16, {LogicalFormat::F32, UINT64_C(0x3f800000)});
  writeTensor(memory, 0, spm, matrix, values);
  writeTensor(memory, 0, spm + UINT64_C(0x1000), matrix, values);
  TargetTransaction gemm{0, 0,
                         TargetGemmTransaction{spm, spm + UINT64_C(0x1000),
                                               spm + UINT64_C(0x2000), 4, 4, 4,
                                               1, LogicalFormat::F32}};
  TargetModelKernelBudget smallBudget = TargetModelKernelBudget::create(
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1,
                                      /*maximumFusedMultiplyAdds=*/1),
      /*maximumMovementBytes=*/4096,
      /*maximumMovementSegments=*/256);

  FakeBulkBackend missing(/*admit=*/false);
  std::string error = expectError(executeTargetModelCommand(
      gemm, memory, smallBudget,
      TargetModelExecutionPolicy::preferAdmitted(missing)));
  EXPECT_NE(error.find("bulk-backend-unavailable"), std::string::npos);
  EXPECT_EQ(missing.invocations, 1u);

  FakeBulkBackend admitted(/*admit=*/true);
  TargetModelCommandEffect effect = llvm::cantFail(executeTargetModelCommand(
      gemm, memory, smallBudget,
      TargetModelExecutionPolicy::preferAdmitted(admitted)));
  EXPECT_EQ(effect.numericBackend, TargetModelNumericBackend::Bulk);
  EXPECT_EQ(effect.bulkEvidence.matmulInvocations, 1u);
  EXPECT_EQ(effect.bulkEvidence.formalFusedMultiplyAdds, 0u);
  EXPECT_EQ(effect.bulkEvidence.admissionRecordDigest, "sha256:fake-admission");
  EXPECT_EQ(admitted.invocations, 1u);
}

TEST(TargetModelKernelTest, FailedEffectDoesNotPublishBytesOrNumericFlags) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext context;
  FormalNumericExceptionFlags flags;
  flags.inexact = true;
  TargetModelCommandEffect invalid{
      {TargetModelByteWrite{
          0, TargetModelAddressSpace::CardDDR, kDDRBase, 1, {9}}},
      flags,
      TargetModelControlAction::None};
  std::string error = expectError(
      commitTargetModelCommandEffect(memory, context, std::move(invalid)));
  EXPECT_NE(error.find("access-denied"), std::string::npos);
  EXPECT_FALSE(context.getAggregateFlags().any());
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::CardDDR, kDDRBase, 1, 1)),
            (std::vector<uint8_t>{0}));
}

TEST(TargetModelKernelTest, ControlTransactionsPreflightTypedEndpoints) {
  InvocationMemoryRegistry memory = makeRegistry(2);
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  TargetTransaction begin{
      0, 0, TargetDirectDTEBeginTransaction{kDDRBase + UINT64_C(0x2000), 2}};
  EXPECT_EQ(
      llvm::cantFail(executeTargetModelCommand(begin, memory, makeBudget()))
          .controlAction,
      TargetModelControlAction::DirectDTEBegin);
  TargetTransaction send{
      0, 1, TargetDirectDTESendTransaction{spm, spm, 16, 0, 1, 0, false}};
  EXPECT_EQ(
      llvm::cantFail(executeTargetModelCommand(send, memory, makeBudget()))
          .controlAction,
      TargetModelControlAction::DirectDTESend);
  std::get<TargetDirectDTESendTransaction>(send.payload).localTile = 65536;
  std::string error =
      expectError(executeTargetModelCommand(send, memory, makeBudget()));
  EXPECT_NE(error.find("outside the accepted target ABI domain"),
            std::string::npos);

  auto &sendPayload = std::get<TargetDirectDTESendTransaction>(send.payload);
  sendPayload.localTile = 0;
  sendPayload.remoteFSM = 4;
  error = expectError(executeTargetModelCommand(send, memory, makeBudget()));
  EXPECT_NE(error.find("outside the accepted target ABI domain"),
            std::string::npos);

  sendPayload.remoteFSM = 0;
  sendPayload.highPerformance = true;
  error = expectError(executeTargetModelCommand(send, memory, makeBudget()));
  EXPECT_NE(error.find("unsupported-transaction"), std::string::npos);
}

TEST(TargetModelKernelTest, UnsupportedFamilyIsNotSilentlyApproximated) {
  InvocationMemoryRegistry memory = makeRegistry();
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  TargetTransaction bit2fp{0, 0,
                           TargetBit2FPTransaction{spm, spm + UINT64_C(0x1000),
                                                   4, LogicalFormat::F32}};
  std::string error =
      expectError(executeTargetModelCommand(bit2fp, memory, makeBudget()));
  EXPECT_NE(error.find("unsupported-transaction"), std::string::npos);
}

} // namespace
