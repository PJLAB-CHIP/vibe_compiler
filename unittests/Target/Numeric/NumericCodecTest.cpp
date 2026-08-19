//===- NumericCodecTest.cpp - Logical scalar codec conformance tests -----===//

#include "Wafer/Target/Numeric/NumericCodec.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Target/Numeric/NumericSemantics.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace {

using wafer::LogicalBitOrder;
using wafer::LogicalByteOrder;
using wafer::LogicalFormat;
using wafer::LogicalScalarCodecPolicy;
using wafer::LogicalValueClass;
using wafer::NonCanonicalEncodingPolicy;
using wafer::RawLogicalValue;

constexpr NonCanonicalEncodingPolicy kReject =
    NonCanonicalEncodingPolicy::Reject;
constexpr NonCanonicalEncodingPolicy kClear =
    NonCanonicalEncodingPolicy::ClearUnusedBits;
constexpr LogicalScalarCodecPolicy kLittleLSBReject = {
    LogicalByteOrder::LittleEndian,
    LogicalBitOrder::LeastSignificantBitFirstWithinByte, kReject};
constexpr LogicalScalarCodecPolicy kLittleLSBClear = {
    LogicalByteOrder::LittleEndian,
    LogicalBitOrder::LeastSignificantBitFirstWithinByte, kClear};
constexpr LogicalScalarCodecPolicy kBigMSBReject = {
    LogicalByteOrder::BigEndian,
    LogicalBitOrder::MostSignificantBitFirstWithinByte, kReject};

RawLogicalValue expectMake(LogicalFormat format, uint64_t bits,
                           NonCanonicalEncodingPolicy policy = kReject) {
  llvm::Expected<RawLogicalValue> value =
      wafer::makeRawLogicalValue(format, bits, policy);
  EXPECT_TRUE(static_cast<bool>(value))
      << (value ? std::string() : llvm::toString(value.takeError()));
  return value ? *value : RawLogicalValue{format, 0};
}

struct IndependentFloatEncoding {
  LogicalFormat format;
  uint8_t storageBits;
  uint8_t semanticBits;
  uint8_t exponentBits;
  uint8_t fractionBits;
};

wafer::LogicalValueClassification
classifyIndependent(IndependentFloatEncoding encoding, uint64_t bits) {
  const uint8_t semanticShift = encoding.storageBits - encoding.semanticBits;
  const uint64_t semantic = bits >> semanticShift;
  const uint64_t fractionMask = (UINT64_C(1) << encoding.fractionBits) - 1;
  const uint64_t exponentMask = (UINT64_C(1) << encoding.exponentBits) - 1;
  const uint64_t fraction = semantic & fractionMask;
  const uint64_t exponent = (semantic >> encoding.fractionBits) & exponentMask;
  const bool negative =
      ((semantic >> (encoding.fractionBits + encoding.exponentBits)) & 1) != 0;

  if (exponent == 0)
    return {fraction == 0 ? LogicalValueClass::Zero
                          : LogicalValueClass::Subnormal,
            negative};
  if (exponent != exponentMask)
    return {LogicalValueClass::Normal, negative};
  if (fraction == 0)
    return {LogicalValueClass::Infinity, negative};
  const uint64_t quietBit = UINT64_C(1) << (encoding.fractionBits - 1);
  return {(fraction & quietBit) != 0 ? LogicalValueClass::QuietNaN
                                     : LogicalValueClass::SignalingNaN,
          negative};
}

void expectIndependentClassification(IndependentFloatEncoding encoding,
                                     uint64_t bits) {
  llvm::Expected<wafer::LogicalValueClassification> actual =
      wafer::classifyRawLogicalValue({encoding.format, bits}, kReject);
  ASSERT_TRUE(static_cast<bool>(actual))
      << (actual ? std::string() : llvm::toString(actual.takeError()));
  const wafer::LogicalValueClassification expected =
      classifyIndependent(encoding, bits);
  EXPECT_EQ(actual->valueClass, expected.valueClass);
  EXPECT_EQ(actual->negative, expected.negative);
}

void expectRoundTrip(LogicalFormat format, uint64_t bits) {
  const wafer::LogicalFormatDescriptor *descriptor =
      wafer::findLogicalFormatDescriptor(format);
  ASSERT_NE(descriptor, nullptr);
  std::array<uint8_t, 8> bytes{};
  ASSERT_FALSE(static_cast<bool>(
      wafer::writeRawLogicalValue({format, bits}, bytes, 0, kLittleLSBReject)));
  const size_t byteCount = descriptor->storageBits / 8;
  for (size_t index = 0; index < bytes.size(); ++index) {
    const uint8_t expected =
        index < byteCount
            ? static_cast<uint8_t>(bits >> static_cast<unsigned>(index * 8))
            : UINT8_C(0);
    EXPECT_EQ(bytes[index], expected);
  }
  llvm::Expected<RawLogicalValue> decoded =
      wafer::readRawLogicalValue(format, bytes, 0, kLittleLSBReject);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << (decoded ? std::string() : llvm::toString(decoded.takeError()));
  EXPECT_EQ(decoded->format, format);
  EXPECT_EQ(decoded->bits, bits);
}

TEST(NumericCodecTest, ExhaustivelyRoundTripsEightAndSixteenBitDomains) {
  for (uint64_t bits = 0; bits < 256; ++bits) {
    expectRoundTrip(LogicalFormat::I8, bits);
    expectRoundTrip(LogicalFormat::U8, bits);
  }
  for (uint64_t bits = 0; bits < 65536; ++bits) {
    expectRoundTrip(LogicalFormat::F16, bits);
    expectRoundTrip(LogicalFormat::BF16, bits);
  }
}

TEST(NumericCodecTest, ExhaustivelyRoundTripsCanonicalTF32Domain) {
  for (uint64_t semantic = 0; semantic < (UINT64_C(1) << 19); ++semantic)
    expectRoundTrip(LogicalFormat::TF32, semantic << 13);
}

TEST(NumericCodecTest, StratifiesWideDomainsAgainstIndependentRawBytes) {
  struct Domain {
    LogicalFormat format;
    unsigned storageBits;
  };
  constexpr Domain domains[] = {
      {LogicalFormat::I16, 16}, {LogicalFormat::I32, 32},
      {LogicalFormat::I64, 64}, {LogicalFormat::U16, 16},
      {LogicalFormat::U32, 32}, {LogicalFormat::U64, 64},
      {LogicalFormat::F32, 32},
  };

  for (const Domain &domain : domains) {
    const uint64_t mask = domain.storageBits == 64
                              ? UINT64_MAX
                              : (UINT64_C(1) << domain.storageBits) - 1;
    const uint64_t highBit = UINT64_C(1) << (domain.storageBits - 1);
    const uint64_t boundaries[] = {0,           1,        highBit - 1, highBit,
                                   highBit + 1, mask - 1, mask};
    for (uint64_t bits : boundaries)
      expectRoundTrip(domain.format, bits & mask);

    uint64_t state = UINT64_C(0x243f6a8885a308d3) ^ domain.storageBits;
    for (unsigned sample = 0; sample < 4096; ++sample) {
      state =
          state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
      expectRoundTrip(domain.format, state & mask);
    }
  }
}

TEST(NumericCodecTest, BooleanIsBitAddressableAndPreservesNeighbors) {
  for (uint64_t bitOffset = 0; bitOffset < 16; ++bitOffset) {
    std::array<uint8_t, 2> bytes{UINT8_C(0xa5), UINT8_C(0x5a)};
    const std::array<uint8_t, 2> original = bytes;
    ASSERT_FALSE(static_cast<bool>(wafer::writeRawLogicalValue(
        {LogicalFormat::Bool, 1}, bytes, bitOffset, kLittleLSBReject)));
    llvm::Expected<RawLogicalValue> one = wafer::readRawLogicalValue(
        LogicalFormat::Bool, bytes, bitOffset, kLittleLSBReject);
    ASSERT_TRUE(static_cast<bool>(one));
    EXPECT_EQ(one->bits, 1u);

    const uint8_t mask = static_cast<uint8_t>(UINT8_C(1) << (bitOffset % 8));
    const size_t byteIndex = static_cast<size_t>(bitOffset / 8);
    for (size_t index = 0; index < bytes.size(); ++index)
      EXPECT_EQ(static_cast<uint8_t>(bytes[index] &
                                     (index == byteIndex ? ~mask : UINT8_MAX)),
                static_cast<uint8_t>(original[index] &
                                     (index == byteIndex ? ~mask : UINT8_MAX)));

    ASSERT_FALSE(static_cast<bool>(wafer::writeRawLogicalValue(
        {LogicalFormat::Bool, 0}, bytes, bitOffset, kLittleLSBReject)));
    llvm::Expected<RawLogicalValue> zero = wafer::readRawLogicalValue(
        LogicalFormat::Bool, bytes, bitOffset, kLittleLSBReject);
    ASSERT_TRUE(static_cast<bool>(zero));
    EXPECT_EQ(zero->bits, 0u);
  }
}

TEST(NumericCodecTest, LayoutBitOffsetsDriveBooleanCodecWithoutGeometryCopy) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  auto memory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                       wafer::MemLayout::Tensor);
  auto type = mlir::MemRefType::get({2, 9}, mlir::IntegerType::get(&context, 1),
                                    mlir::MemRefLayoutAttrInterface{}, memory);
  const wafer::ModelProfileRecord &profile = wafer::getModelProfileRecord(
      wafer::ModelProfileId::formalDeterministic());

  std::array<uint8_t, 3> storage{};
  std::array<uint8_t, 3> expected{};
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t column = 0; column < 9; ++column) {
      const int64_t linear = row * 9 + column;
      std::optional<int64_t> bitOffset =
          wafer::computeWaferPhysicalElementBitOffset(type, {row, column});
      ASSERT_TRUE(bitOffset);
      ASSERT_EQ(*bitOffset, linear);
      const bool value = (linear % 3) == 1;
      ASSERT_FALSE(static_cast<bool>(wafer::writeRawLogicalValue(
          {LogicalFormat::Bool, value ? UINT64_C(1) : UINT64_C(0)}, storage,
          static_cast<uint64_t>(*bitOffset), profile.numericEncodePolicy)));
      if (value)
        expected[linear / 8] |=
            static_cast<uint8_t>(UINT8_C(1) << (linear % 8));
    }
  }
  EXPECT_EQ(storage, expected);

  for (int64_t linear = 0; linear < 18; ++linear) {
    llvm::Expected<RawLogicalValue> decoded = wafer::readRawLogicalValue(
        LogicalFormat::Bool, storage, static_cast<uint64_t>(linear),
        profile.numericDecodePolicy);
    ASSERT_TRUE(static_cast<bool>(decoded))
        << (decoded ? std::string() : llvm::toString(decoded.takeError()));
    EXPECT_EQ(decoded->bits, (linear % 3) == 1 ? UINT64_C(1) : UINT64_C(0));
  }
}

TEST(NumericCodecTest, TF32PolicyRejectsOrClearsLowThirteenBits) {
  llvm::Expected<RawLogicalValue> rejected = wafer::makeRawLogicalValue(
      LogicalFormat::TF32, UINT64_C(0x3f800001), kReject);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("noncanonical"),
            std::string::npos);

  RawLogicalValue normalized =
      expectMake(LogicalFormat::TF32, UINT64_C(0x3f801fff), kClear);
  EXPECT_EQ(normalized.bits, UINT64_C(0x3f800000));

  std::array<uint8_t, 4> bytes{1, 0, 128, 63};
  llvm::Expected<RawLogicalValue> decoded = wafer::readRawLogicalValue(
      LogicalFormat::TF32, bytes, 0, kLittleLSBReject);
  EXPECT_FALSE(static_cast<bool>(decoded));
  EXPECT_FALSE(llvm::toString(decoded.takeError()).empty());
  decoded = wafer::readRawLogicalValue(LogicalFormat::TF32, bytes, 0,
                                       kLittleLSBClear);
  ASSERT_TRUE(static_cast<bool>(decoded));
  EXPECT_EQ(decoded->bits, UINT64_C(0x3f800000));
}

TEST(NumericCodecTest, StratifiesNoncanonicalTF32Encodings) {
  uint64_t state = UINT64_C(0x13198a2e03707344);
  for (unsigned sample = 0; sample < 4096; ++sample) {
    state = state * UINT64_C(2862933555777941757) + UINT64_C(3037000493);
    const uint64_t semantic = (state >> 13) & ((UINT64_C(1) << 19) - 1);
    const uint64_t padding = (state & UINT64_C(0x1fff)) | UINT64_C(1);
    const uint64_t bits = (semantic << 13) | padding;

    llvm::Expected<RawLogicalValue> rejected =
        wafer::makeRawLogicalValue(LogicalFormat::TF32, bits, kReject);
    ASSERT_FALSE(static_cast<bool>(rejected));
    EXPECT_NE(llvm::toString(rejected.takeError()).find("noncanonical"),
              std::string::npos);

    RawLogicalValue cleared = expectMake(LogicalFormat::TF32, bits, kClear);
    EXPECT_EQ(cleared.bits, semantic << 13);
  }
}

TEST(NumericCodecTest, ClassifiesFloatingDomainsAgainstIndependentBitOracle) {
  constexpr IndependentFloatEncoding exhaustive[] = {
      {LogicalFormat::F16, 16, 16, 5, 10},
      {LogicalFormat::BF16, 16, 16, 8, 7},
      {LogicalFormat::TF32, 32, 19, 8, 10},
  };
  for (const IndependentFloatEncoding &encoding : exhaustive) {
    const uint64_t patterns = UINT64_C(1) << encoding.semanticBits;
    const uint8_t shift = encoding.storageBits - encoding.semanticBits;
    for (uint64_t semantic = 0; semantic < patterns; ++semantic)
      expectIndependentClassification(encoding, semantic << shift);
  }

  constexpr IndependentFloatEncoding f32 = {LogicalFormat::F32, 32, 32, 8, 23};
  constexpr uint64_t boundaries[] = {
      UINT64_C(0x00000000), UINT64_C(0x80000000), UINT64_C(0x00000001),
      UINT64_C(0x007fffff), UINT64_C(0x00800000), UINT64_C(0x7f7fffff),
      UINT64_C(0x7f800000), UINT64_C(0xff800000), UINT64_C(0x7f800001),
      UINT64_C(0x7fc00000), UINT64_C(0xff800001), UINT64_C(0xffc00000),
  };
  for (uint64_t bits : boundaries)
    expectIndependentClassification(f32, bits);

  uint64_t state = UINT64_C(0xa4093822299f31d0);
  for (unsigned sample = 0; sample < 65536; ++sample) {
    state = state * UINT64_C(3202034522624059733) + UINT64_C(1);
    expectIndependentClassification(f32, state & UINT64_C(0xffffffff));
  }
}

TEST(NumericCodecTest, UsesLittleEndianAndRejectsInvalidRanges) {
  std::array<uint8_t, 8> bytes{};
  ASSERT_FALSE(static_cast<bool>(wafer::writeRawLogicalValue(
      {LogicalFormat::U32, UINT64_C(0x78563412)}, bytes, 8, kLittleLSBReject)));
  EXPECT_EQ(bytes[1], UINT8_C(0x12));
  EXPECT_EQ(bytes[2], UINT8_C(0x34));
  EXPECT_EQ(bytes[3], UINT8_C(0x56));
  EXPECT_EQ(bytes[4], UINT8_C(0x78));

  llvm::Expected<RawLogicalValue> unaligned = wafer::readRawLogicalValue(
      LogicalFormat::F16, bytes, 1, kLittleLSBReject);
  ASSERT_FALSE(static_cast<bool>(unaligned));
  EXPECT_NE(llvm::toString(unaligned.takeError()).find("byte-aligned"),
            std::string::npos);

  llvm::Expected<RawLogicalValue> outOfBounds = wafer::readRawLogicalValue(
      LogicalFormat::U64, bytes, 8, kLittleLSBReject);
  ASSERT_FALSE(static_cast<bool>(outOfBounds));
  EXPECT_NE(llvm::toString(outOfBounds.takeError()).find("exceeds"),
            std::string::npos);

  llvm::Expected<RawLogicalValue> tooWide =
      wafer::makeRawLogicalValue(LogicalFormat::U8, UINT64_C(0x100), kReject);
  ASSERT_FALSE(static_cast<bool>(tooWide));
  EXPECT_NE(llvm::toString(tooWide.takeError()).find("storage width"),
            std::string::npos);
}

TEST(NumericCodecTest, CodecPolicyMakesByteAndBitOrderExplicit) {
  std::array<uint8_t, 4> bytes{};
  ASSERT_FALSE(static_cast<bool>(wafer::writeRawLogicalValue(
      {LogicalFormat::U32, UINT64_C(0x12345678)}, bytes, 0, kBigMSBReject)));
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0x12, 0x34, 0x56, 0x78}));
  llvm::Expected<RawLogicalValue> decoded =
      wafer::readRawLogicalValue(LogicalFormat::U32, bytes, 0, kBigMSBReject);
  ASSERT_TRUE(static_cast<bool>(decoded));
  EXPECT_EQ(decoded->bits, UINT64_C(0x12345678));

  std::array<uint8_t, 1> packed{};
  ASSERT_FALSE(static_cast<bool>(wafer::writeRawLogicalValue(
      {LogicalFormat::Bool, 1}, packed, 0, kBigMSBReject)));
  EXPECT_EQ(packed[0], UINT8_C(0x80));
  decoded =
      wafer::readRawLogicalValue(LogicalFormat::Bool, packed, 0, kBigMSBReject);
  ASSERT_TRUE(static_cast<bool>(decoded));
  EXPECT_EQ(decoded->bits, 1u);
}

TEST(NumericCodecTest, ClassifiesFloatingSpecialValuesAndSignedZero) {
  struct Case {
    LogicalFormat format;
    uint64_t bits;
    LogicalValueClass expectedClass;
    bool negative;
  };
  constexpr Case cases[] = {
      {LogicalFormat::F16, 0x0000, LogicalValueClass::Zero, false},
      {LogicalFormat::F16, 0x8000, LogicalValueClass::Zero, true},
      {LogicalFormat::F16, 0x0001, LogicalValueClass::Subnormal, false},
      {LogicalFormat::F16, 0x3c00, LogicalValueClass::Normal, false},
      {LogicalFormat::F16, 0x7c00, LogicalValueClass::Infinity, false},
      {LogicalFormat::F16, 0xfc00, LogicalValueClass::Infinity, true},
      {LogicalFormat::F16, 0x7e00, LogicalValueClass::QuietNaN, false},
      {LogicalFormat::F16, 0x7d00, LogicalValueClass::SignalingNaN, false},
      {LogicalFormat::BF16, 0x7f80, LogicalValueClass::Infinity, false},
      {LogicalFormat::BF16, 0x7fc0, LogicalValueClass::QuietNaN, false},
      {LogicalFormat::F32, 0x80000000, LogicalValueClass::Zero, true},
      {LogicalFormat::F32, 0x7f800001, LogicalValueClass::SignalingNaN, false},
      {LogicalFormat::TF32, 0x7fc00000, LogicalValueClass::QuietNaN, false},
      {LogicalFormat::TF32, 0x00002000, LogicalValueClass::Subnormal, false},
  };
  for (const Case &testCase : cases) {
    llvm::Expected<wafer::LogicalValueClassification> classified =
        wafer::classifyRawLogicalValue({testCase.format, testCase.bits},
                                       kReject);
    ASSERT_TRUE(static_cast<bool>(classified))
        << (classified ? std::string()
                       : llvm::toString(classified.takeError()));
    EXPECT_EQ(classified->valueClass, testCase.expectedClass);
    EXPECT_EQ(classified->negative, testCase.negative);
  }
}

TEST(NumericCodecTest, ClassifiesIntegerSignWithoutHostCasts) {
  llvm::Expected<wafer::LogicalValueClassification> signedNegative =
      wafer::classifyRawLogicalValue({LogicalFormat::I8, 0x80}, kReject);
  ASSERT_TRUE(static_cast<bool>(signedNegative));
  EXPECT_EQ(signedNegative->valueClass, LogicalValueClass::SignedInteger);
  EXPECT_TRUE(signedNegative->negative);

  llvm::Expected<wafer::LogicalValueClassification> unsignedHigh =
      wafer::classifyRawLogicalValue({LogicalFormat::U64, UINT64_MAX}, kReject);
  ASSERT_TRUE(static_cast<bool>(unsignedHigh));
  EXPECT_EQ(unsignedHigh->valueClass, LogicalValueClass::UnsignedInteger);
  EXPECT_FALSE(unsignedHigh->negative);
}

} // namespace
