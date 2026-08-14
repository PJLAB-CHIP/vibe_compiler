//===- PhysicalTensorCodecTest.cpp - Physical tensor codec tests --------===//

#include "Wafer/Target/PhysicalTensorCodec.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace wafer;

template <typename T> std::string expectError(llvm::Expected<T> value) {
  if (value) {
    ADD_FAILURE() << "expected an LLVM Error";
    return {};
  }
  return llvm::toString(value.takeError());
}

NumericTensorKey makeTensor(LogicalFormat format, PhysicalTensorLayout layout,
                            std::vector<uint64_t> shape) {
  return llvm::cantFail(
      NumericTensorKey::create(format, layout, std::move(shape)));
}

TEST(PhysicalTensorCodecTest, CxRoundTripPreservesTemplatePadding) {
  NumericTensorKey key =
      makeTensor(LogicalFormat::F16, PhysicalTensorLayout::Cx, {2, 3});
  std::vector<RawLogicalValue> values{
      {LogicalFormat::F16, UINT64_C(0x3c00)},
      {LogicalFormat::F16, UINT64_C(0x4000)},
      {LogicalFormat::F16, UINT64_C(0x4200)},
      {LogicalFormat::F16, UINT64_C(0x4400)},
      {LogicalFormat::F16, UINT64_C(0x4500)},
      {LogicalFormat::F16, UINT64_C(0x4600)},
  };
  llvm::Expected<std::vector<uint8_t>> packed =
      packPhysicalTensorLogicalValues(key, values, UINT8_C(0xa5));
  ASSERT_TRUE(static_cast<bool>(packed)) << llvm::toString(packed.takeError());
  ASSERT_EQ(packed->size(), llvm::cantFail(getPhysicalTensorStorageBytes(key)));

  llvm::Expected<std::vector<RawLogicalValue>> unpacked =
      unpackPhysicalTensorLogicalValues(key, *packed);
  ASSERT_TRUE(static_cast<bool>(unpacked))
      << llvm::toString(unpacked.takeError());
  ASSERT_EQ(unpacked->size(), values.size());
  for (size_t index = 0; index < values.size(); ++index) {
    EXPECT_EQ((*unpacked)[index].format, values[index].format);
    EXPECT_EQ((*unpacked)[index].bits, values[index].bits);
  }

  std::vector<RawLogicalValue> changed = values;
  changed.front().bits = UINT64_C(0x3800);
  llvm::Expected<std::vector<uint8_t>> repacked =
      packPhysicalTensorLogicalValues(key, changed, *packed);
  ASSERT_TRUE(static_cast<bool>(repacked))
      << llvm::toString(repacked.takeError());
  size_t changedBytes = 0;
  for (size_t index = 0; index < packed->size(); ++index)
    changedBytes += (*packed)[index] != (*repacked)[index];
  EXPECT_GT(changedBytes, 0u);
  EXPECT_LE(changedBytes, 2u);
}

TEST(PhysicalTensorCodecTest, BitpackedBoolUsesSharedPhysicalGeometry) {
  NumericTensorKey key =
      makeTensor(LogicalFormat::Bool, PhysicalTensorLayout::Tensor, {2, 5});
  std::vector<RawLogicalValue> values;
  for (uint64_t index = 0; index < key.getElementCount(); ++index)
    values.push_back({LogicalFormat::Bool, index % 3 == 0});
  llvm::Expected<std::vector<uint8_t>> packed =
      packPhysicalTensorLogicalValues(key, values, UINT8_C(0x5a));
  ASSERT_TRUE(static_cast<bool>(packed)) << llvm::toString(packed.takeError());
  llvm::Expected<std::vector<RawLogicalValue>> unpacked =
      unpackPhysicalTensorLogicalValues(key, *packed);
  ASSERT_TRUE(static_cast<bool>(unpacked))
      << llvm::toString(unpacked.takeError());
  ASSERT_EQ(unpacked->size(), values.size());
  for (size_t index = 0; index < values.size(); ++index) {
    EXPECT_EQ((*unpacked)[index].format, values[index].format);
    EXPECT_EQ((*unpacked)[index].bits, values[index].bits);
  }

  for (NumericTensorKey blocked :
       {makeTensor(LogicalFormat::Bool, PhysicalTensorLayout::Cx, {2, 5}),
        makeTensor(LogicalFormat::Bool, PhysicalTensorLayout::NCx,
                   {1, 2, 5})}) {
    llvm::Expected<std::vector<uint8_t>> blockedPacked =
        packPhysicalTensorLogicalValues(blocked, values, UINT8_C(0xa5));
    ASSERT_TRUE(static_cast<bool>(blockedPacked))
        << llvm::toString(blockedPacked.takeError());
    EXPECT_EQ(blockedPacked->size(),
              llvm::cantFail(getPhysicalTensorStorageBytes(blocked)));
    llvm::Expected<std::vector<RawLogicalValue>> blockedUnpacked =
        unpackPhysicalTensorLogicalValues(blocked, *blockedPacked);
    ASSERT_TRUE(static_cast<bool>(blockedUnpacked))
        << llvm::toString(blockedUnpacked.takeError());
    ASSERT_EQ(blockedUnpacked->size(), values.size());
    for (size_t index = 0; index < values.size(); ++index) {
      EXPECT_EQ((*blockedUnpacked)[index].format, values[index].format);
      EXPECT_EQ((*blockedUnpacked)[index].bits, values[index].bits);
    }
  }
}

TEST(PhysicalTensorCodecTest, RejectsSizeCountAndEncodingMismatch) {
  NumericTensorKey key =
      makeTensor(LogicalFormat::F32, PhysicalTensorLayout::NCx, {1, 2, 3});
  uint64_t bytes = llvm::cantFail(getPhysicalTensorStorageBytes(key));
  ASSERT_GT(bytes, 0u);
  std::string error = expectError(unpackPhysicalTensorLogicalValues(
      key, std::vector<uint8_t>(static_cast<size_t>(bytes - 1), 0)));
  EXPECT_NE(error.find("invalid-storage-size"), std::string::npos);

  error = expectError(packPhysicalTensorLogicalValues(
      key, std::vector<RawLogicalValue>{}, UINT8_C(0)));
  EXPECT_NE(error.find("invalid-logical-value-count"), std::string::npos);

  std::vector<RawLogicalValue> values(
      static_cast<size_t>(key.getElementCount()),
      RawLogicalValue{LogicalFormat::F32, UINT64_C(0)});
  values.back().format = LogicalFormat::I32;
  error = expectError(packPhysicalTensorLogicalValues(key, values, UINT8_C(0)));
  EXPECT_NE(error.find("invalid-logical-encoding"), std::string::npos);
}

} // namespace
