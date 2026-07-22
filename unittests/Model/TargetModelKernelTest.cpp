//===- TargetModelKernelTest.cpp - Plain target kernel tests -------------===//

#include "Wafer/Model/TargetModelKernel.h"

#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
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
          WAFER_TX81_DIRECT_DTE_STATUS_V2_STORAGE_BYTES,
          WAFER_TX81_DIRECT_DTE_STATUS_V2_STORAGE_ALIGNMENT}},
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
  tryExecute(const TargetModelNumericRequest &request) const override {
    ++invocations;
    if (!admit)
      return std::optional<TargetModelBulkResult>();
    TargetModelBulkResult result{
        request.destinationTemplate,
        {},
        {1, 1, 0, TargetModelBulkProvenanceKind::ExactQualificationRecord,
         "sha256:fake-admission", "fake-bulk"}};
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
    case TargetCallBuiltin::GemmOrientedV2:
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
  size_t validated = 0;
  for (const TargetCallDescriptor &descriptor : getTargetCallDescriptors()) {
    const bool oriented = descriptor.semantic ==
                          TargetCallSemantic(TargetCallBuiltin::GemmOrientedV2);
    TargetCallDecodeContext context{
        oriented ? TargetProfileId::waferTx81SingleCardKernelV2()
                 : TargetProfileId::waferTx81SingleCardKernelV1(),
        16};
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
  EXPECT_EQ(validated, 111u);
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
  ASSERT_EQ(readEffect.pendingWrites.size(), 1u);
  EXPECT_FALSE(readEffect.pendingWrites.front().stridedLayout.has_value());
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
  ASSERT_EQ(writeEffect.pendingWrites.size(), 1u);
  ASSERT_TRUE(writeEffect.pendingWrites.front().stridedLayout.has_value());
  EXPECT_EQ(writeEffect.pendingWrites.front().stridedLayout->iterations,
            (std::array<uint32_t, 3>{2, 1, 1}));
  llvm::cantFail(
      commitTargetModelCommandEffect(memory, context, std::move(writeEffect)));
  std::vector<uint8_t> output = llvm::cantFail(memory.readSlotSnapshot(0, 1));
  EXPECT_EQ(std::vector<uint8_t>(output.begin(), output.begin() + 4),
            (std::vector<uint8_t>{0, 1, 2, 3}));
  EXPECT_EQ(std::vector<uint8_t>(output.begin() + 8, output.begin() + 12),
            (std::vector<uint8_t>{8, 9, 10, 11}));
}

TEST(TargetModelKernelTest,
     GatherScatterSnapshotsOverlappingSourceBeforeCompactWrite) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext context;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  llvm::cantFail(memory.applyAtomically(
      {TargetModelByteWrite{0,
                            TargetModelAddressSpace::RankSPM,
                            spm,
                            1,
                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}}));
  TargetTransaction gather{
      0, 0,
      TargetGatherScatterTransaction{
          spm, spm + 2, 8, 2, {2, 0, 0}, {4, 1, 1}, {2, 0, 0}, {4, 1, 1}}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(gather, memory, makeBudget()));
  ASSERT_EQ(effect.pendingWrites.size(), 1u);
  ASSERT_TRUE(effect.pendingWrites.front().stridedLayout.has_value());
  EXPECT_EQ(effect.pendingWrites.front().bytes,
            (std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::RankSPM, spm, 10, 1)),
            (std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  llvm::cantFail(
      commitTargetModelCommandEffect(memory, context, std::move(effect)));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::RankSPM, spm, 10, 1)),
            (std::vector<uint8_t>{0, 1, 0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(TargetModelKernelTest, GatherScatterAllowsRepeatedSourceSegments) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext context;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  llvm::cantFail(memory.applyAtomically({TargetModelByteWrite{
      0, TargetModelAddressSpace::RankSPM, spm, 1, {7, 9}}}));
  const uint64_t destination = spm + UINT64_C(0x100);
  TargetTransaction gather{
      0, 0,
      TargetGatherScatterTransaction{
          spm, destination, 4, 2, {0, 0, 0}, {2, 1, 1}, {2, 0, 0}, {2, 1, 1}}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(gather, memory, makeBudget()));
  ASSERT_EQ(effect.pendingWrites.size(), 1u);
  EXPECT_EQ(effect.pendingWrites.front().bytes,
            (std::vector<uint8_t>{7, 9, 7, 9}));
  llvm::cantFail(
      commitTargetModelCommandEffect(memory, context, std::move(effect)));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::RankSPM, destination, 4, 1)),
            (std::vector<uint8_t>{7, 9, 7, 9}));
}

TEST(TargetModelKernelTest,
     OverlappingGatherDestinationFailsWithoutPartialPublication) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext context;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  llvm::cantFail(memory.applyAtomically({TargetModelByteWrite{
      0, TargetModelAddressSpace::RankSPM, spm, 1, {1, 2, 3, 4}}}));
  const uint64_t destination = spm + UINT64_C(0x100);
  TargetTransaction gather{
      0, 0,
      TargetGatherScatterTransaction{
          spm, destination, 4, 2, {2, 0, 0}, {2, 1, 1}, {0, 0, 0}, {2, 1, 1}}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(gather, memory, makeBudget()));
  ASSERT_EQ(effect.pendingWrites.size(), 1u);
  std::string error = expectError(
      commitTargetModelCommandEffect(memory, context, std::move(effect)));
  EXPECT_NE(error.find("pending writes overlap"), std::string::npos);
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::RankSPM, destination, 4, 1)),
            std::vector<uint8_t>(4, 0));
  EXPECT_FALSE(context.getAggregateFlags().any());
}

TEST(TargetModelKernelTest,
     NonCanonicalMovementUsesLinearFallbackWithoutChangingDescriptorOrder) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext context;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  TargetTransaction rdma{0, 0,
                         TargetStridedDMATransaction{TargetDMADirection::Read,
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
      commitTargetModelCommandEffect(memory, context, std::move(readEffect)));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::RankSPM, spm, 12, 1)),
            (std::vector<uint8_t>{0, 1, 4, 5, 8, 9, 6, 7, 10, 11, 14, 15}));

  TargetTransaction wdma{
      0, 1,
      TargetStridedDMATransaction{TargetDMADirection::Write,
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
      commitTargetModelCommandEffect(memory, context, std::move(writeEffect)));
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
  FormalNumericExecutionContext context;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  const uint64_t transposed = spm + UINT64_C(0x100000);
  const uint64_t roundTrip = spm + UINT64_C(0x200000);
  std::vector<uint8_t> payload(kPayloadBytes);
  for (uint64_t segment = 0; segment < kSegmentCount; ++segment) {
    payload[segment * 2] = static_cast<uint8_t>(segment);
    payload[segment * 2 + 1] = static_cast<uint8_t>(segment >> 8);
  }
  llvm::cantFail(memory.applyAtomically({TargetModelByteWrite{
      0, TargetModelAddressSpace::RankSPM, spm, 1, payload}}));

  TargetModelKernelBudget largeBudget = TargetModelKernelBudget::create(
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1,
                                      /*maximumFusedMultiplyAdds=*/1),
      kPayloadBytes, kSegmentCount);
  TargetTransaction transpose{
      0, 0,
      TargetGatherScatterTransaction{spm,
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
      commitTargetModelCommandEffect(memory, context, std::move(effect)));
  std::vector<uint8_t> output = llvm::cantFail(memory.readSnapshot(
      0, TargetModelAddressSpace::RankSPM, transposed, kPayloadBytes, 1));
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

  TargetTransaction reverse{
      0, 1,
      TargetGatherScatterTransaction{transposed,
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
  llvm::cantFail(commitTargetModelCommandEffect(memory, context,
                                                std::move(reverseEffect)));
  EXPECT_EQ(
      llvm::cantFail(memory.readSnapshot(0, TargetModelAddressSpace::RankSPM,
                                         roundTrip, kPayloadBytes, 1)),
      payload);
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

TEST(TargetModelKernelTest, NativeF32SumUsesFixedShapeABIAndFormalNumeric) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext context;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  NumericTensorKey input =
      makeTensor(LogicalFormat::F32, NumericTensorLayout::NCx, {1, 1, 2, 2});
  NumericTensorKey destination =
      makeTensor(LogicalFormat::F32, NumericTensorLayout::NCx, {1, 1, 2});
  writeTensor(memory, 0, spm, input,
              {{LogicalFormat::F32, UINT64_C(0x3f800000)},
               {LogicalFormat::F32, UINT64_C(0x40000000)},
               {LogicalFormat::F32, UINT64_C(0x40400000)},
               {LogicalFormat::F32, UINT64_C(0x40800000)}});
  TargetTransaction reduce{
      0, 0,
      TargetReduceTransaction{
          InstrReduceKind::Sum,
          spm,
          spm + UINT64_C(0x1000),
          static_cast<uint32_t>(NativeCTReduceDimension::Trailing0),
          {1, 1, 2, 2},
          LogicalFormat::F32}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(reduce, memory, makeBudget()));
  llvm::cantFail(
      commitTargetModelCommandEffect(memory, context, std::move(effect)));
  std::vector<RawLogicalValue> result =
      readTensor(memory, 0, spm + UINT64_C(0x1000), destination);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0].bits, UINT64_C(0x40400000));
  EXPECT_EQ(result[1].bits, UINT64_C(0x40e00000));
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
     PhysicalFootprintMemsetOverwritesAlignedTailAndUnusedBoolBits) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext context;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  const uint64_t cxDestination = spm + UINT64_C(0x3000);
  const uint64_t boolDestination = spm + UINT64_C(0x4000);
  llvm::cantFail(memory.applyAtomically(
      {TargetModelByteWrite{0, TargetModelAddressSpace::RankSPM, cxDestination,
                            1, std::vector<uint8_t>(256, UINT8_C(0xa5))},
       TargetModelByteWrite{0, TargetModelAddressSpace::RankSPM,
                            boolDestination, 1,
                            std::vector<uint8_t>(2, UINT8_C(0xa5))}}));

  TargetTransaction cxFill{
      0, 0,
      TargetMemsetTransaction{cxDestination, UINT32_C(0x3555),
                              /*elementCount=*/128, LogicalFormat::F16}};
  TargetModelCommandEffect cxEffect =
      llvm::cantFail(executeTargetModelCommand(cxFill, memory, makeBudget()));
  llvm::cantFail(
      commitTargetModelCommandEffect(memory, context, std::move(cxEffect)));
  std::vector<uint8_t> cxBytes = llvm::cantFail(memory.readSnapshot(
      0, TargetModelAddressSpace::RankSPM, cxDestination, 256, 1));
  for (size_t index = 0; index < cxBytes.size(); index += 2) {
    EXPECT_EQ(cxBytes[index], UINT8_C(0x55));
    EXPECT_EQ(cxBytes[index + 1], UINT8_C(0x35));
  }

  TargetTransaction boolFill{
      0, 1,
      TargetMemsetTransaction{boolDestination, UINT32_C(1),
                              /*elementCount=*/16, LogicalFormat::Bool}};
  TargetModelCommandEffect boolEffect =
      llvm::cantFail(executeTargetModelCommand(boolFill, memory, makeBudget()));
  llvm::cantFail(
      commitTargetModelCommandEffect(memory, context, std::move(boolEffect)));
  EXPECT_EQ(llvm::cantFail(memory.readSnapshot(
                0, TargetModelAddressSpace::RankSPM, boolDestination, 2, 1)),
            (std::vector<uint8_t>{UINT8_C(0xff), UINT8_C(0xff)}));
}

TEST(TargetModelKernelTest, BatchedGemmUsesImplicitNCxStorageContract) {
  InvocationMemoryRegistry memory = makeRegistry();
  FormalNumericExecutionContext context;
  const uint64_t spm = memory.getAddressPlan().getSPMBase();
  NumericTensorKey batchMatrices =
      makeTensor(LogicalFormat::F32, NumericTensorLayout::NCx, {2, 2, 2});
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
  TargetTransaction gemm{0, 0,
                         TargetGemmTransaction{spm, spm + UINT64_C(0x1000),
                                               spm + UINT64_C(0x2000), 2, 2, 2,
                                               2, LogicalFormat::F32}};
  TargetModelCommandEffect effect =
      llvm::cantFail(executeTargetModelCommand(gemm, memory, makeBudget()));
  llvm::cantFail(
      commitTargetModelCommandEffect(memory, context, std::move(effect)));

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
  EXPECT_EQ(effect.bulkEvidence.provenanceKind,
            TargetModelBulkProvenanceKind::ExactQualificationRecord);
  EXPECT_EQ(effect.bulkEvidence.provenanceDigest, "sha256:fake-admission");
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
