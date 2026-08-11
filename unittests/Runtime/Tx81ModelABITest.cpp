//===- Tx81ModelABITest.cpp - TX81 model wire ABI tests -----------------===//

#include "Wafer/Runtime/Tx81ModelABI.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

uint32_t readU32(const std::vector<uint8_t> &bytes, size_t offset) {
  uint32_t value = 0;
  for (size_t index = 0; index < 4; ++index)
    value |= static_cast<uint32_t>(bytes[offset + index]) << (index * 8);
  return value;
}

uint64_t readU64(const std::vector<uint8_t> &bytes, size_t offset) {
  uint64_t value = 0;
  for (size_t index = 0; index < 8; ++index)
    value |= static_cast<uint64_t>(bytes[offset + index]) << (index * 8);
  return value;
}

wafer::runtime::Tx81ModelTensorDescriptor
tensor(wafer::runtime::Tx81ModelTensorClass tensorClass, int64_t tile,
       uint64_t slot, uint64_t address, int64_t launchSlot = -1) {
  if (launchSlot < 0)
    launchSlot = tile;
  return {tensorClass,
          wafer::PhysicalCardId(0),
          wafer::PhysicalTileId(tile),
          wafer::LaunchSlotId(launchSlot),
          slot,
          address,
          64,
          "f32",
          {16}};
}

TEST(Tx81ModelABITest, AcceptsExplicitNonIdentityPhysicalTileBinding) {
  using namespace wafer::runtime;
  llvm::Expected<Tx81ModelBootParamImage> image = buildTx81ModelBootParam(
      {tensor(Tx81ModelTensorClass::Input, /*tile=*/1, /*slot=*/0,
              /*address=*/0x1000, /*launchSlot=*/0),
       tensor(Tx81ModelTensorClass::Output, /*tile=*/1, /*slot=*/1,
              /*address=*/0x2000, /*launchSlot=*/0)},
      0x3000);
  ASSERT_TRUE(static_cast<bool>(image)) << llvm::toString(image.takeError());

  llvm::Expected<Tx81ModelBootParamImage> conflicting = buildTx81ModelBootParam(
      {tensor(Tx81ModelTensorClass::Input, /*tile=*/1, /*slot=*/0,
              /*address=*/0x1000, /*launchSlot=*/0),
       tensor(Tx81ModelTensorClass::Output, /*tile=*/0, /*slot=*/1,
              /*address=*/0x2000, /*launchSlot=*/0)},
      0x3000);
  ASSERT_FALSE(static_cast<bool>(conflicting));
  EXPECT_NE(llvm::toString(conflicting.takeError()).find("not one-to-one"),
            std::string::npos);
}

TEST(Tx81ModelABITest, BuildsCanonicalParameterFreeBootParam) {
  using namespace wafer::runtime;
  std::vector<Tx81ModelTensorDescriptor> tensors = {
      tensor(Tx81ModelTensorClass::Output, 1, 2, 0x5000),
      tensor(Tx81ModelTensorClass::Input, 1, 1, 0x3000),
      tensor(Tx81ModelTensorClass::Input, 0, 4, 0x2000),
      tensor(Tx81ModelTensorClass::Output, 0, 9, 0x4000),
  };
  llvm::Expected<Tx81ModelBootParamImage> image =
      buildTx81ModelBootParam(tensors, 0x12345000);
  ASSERT_TRUE(static_cast<bool>(image)) << llvm::toString(image.takeError());
  // 56-byte head + four tensors + one exact-build trailing record.
  ASSERT_EQ(image->bytes.size(), 56u + 5u * 72u);
  EXPECT_EQ(readU32(image->bytes, 0), image->bytes.size());
  EXPECT_EQ(readU32(image->bytes, 4), 0x00200000u);
  EXPECT_EQ(readU32(image->bytes, 8), 2u);
  EXPECT_EQ(readU32(image->bytes, 12), 2u);
  EXPECT_EQ(readU32(image->bytes, 16), 0u);
  EXPECT_EQ(readU32(image->bytes, 40), 16u);
  EXPECT_EQ(readU32(image->bytes, 44), 0xffffffffu);
  EXPECT_EQ(readU64(image->bytes, 48), 0x12345000u);
  EXPECT_EQ(readU64(image->bytes, 56), 0x2000u);
  EXPECT_EQ(readU64(image->bytes, 56 + 72), 0x3000u);
  EXPECT_EQ(readU64(image->bytes, 56 + 2 * 72), 0x4000u);
  EXPECT_EQ(readU64(image->bytes, 56 + 3 * 72), 0x5000u);
  EXPECT_EQ(readU32(image->bytes, 56 + 16), 5u);
  EXPECT_EQ(readU32(image->bytes, 56 + 20), 1u);
  EXPECT_EQ(readU64(image->bytes, 56 + 24), 16u);
  EXPECT_EQ(readU64(image->bytes, image->bytes.size() - 72), 0u);
}

TEST(Tx81ModelABITest, RejectsUnqualifiedTensorContracts) {
  using namespace wafer::runtime;
  auto parameter = tensor(Tx81ModelTensorClass::Parameter, 0, 0, 0x1000);
  auto qualifiedInput = tensor(Tx81ModelTensorClass::Input, 0, 1, 0x1800);
  auto output = tensor(Tx81ModelTensorClass::Output, 0, 1, 0x2000);
  llvm::Expected<Tx81ModelBootParamImage> rejected =
      buildTx81ModelBootParam({parameter, qualifiedInput, output}, 0x3000);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("parameter layout"),
            std::string::npos);

  auto input = tensor(Tx81ModelTensorClass::Input, 0, 0, 0x1000);
  input.dtype = "f16";
  rejected = buildTx81ModelBootParam({input, output}, 0x3000);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("only f32"),
            std::string::npos);

  input = tensor(Tx81ModelTensorClass::Input, 0, 0, 0x1001);
  rejected = buildTx81ModelBootParam({input, output}, 0x3000);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("misaligned"),
            std::string::npos);

  input = tensor(Tx81ModelTensorClass::Input, 0, 0, 0x1000);
  rejected = buildTx81ModelBootParam({input, output}, 0x3001);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("misaligned"),
            std::string::npos);
}

TEST(Tx81ModelABITest, BuildsType7NestedPayloads) {
  using namespace wafer::runtime;
  llvm::Expected<std::vector<uint8_t>> modules =
      buildTx81DynlibRunModules("123456789");
  ASSERT_TRUE(static_cast<bool>(modules))
      << llvm::toString(modules.takeError());
  ASSERT_EQ(modules->size(), 456u);
  EXPECT_EQ((*modules)[0], 1u);
  EXPECT_EQ(std::string(modules->begin() + 8, modules->begin() + 17),
            "123456789");

  llvm::Expected<std::vector<uint8_t>> tlv =
      buildTx81DynlibRunTLV(0x8899aabbccd8ULL);
  ASSERT_TRUE(static_cast<bool>(tlv)) << llvm::toString(tlv.takeError());
  ASSERT_EQ(tlv->size(), 16u);
  EXPECT_EQ(readU32(*tlv, 0), 7u);
  EXPECT_EQ(readU32(*tlv, 4), 8u);
  EXPECT_EQ(readU64(*tlv, 8), 0x8899aabbccd8ULL);
}

} // namespace
