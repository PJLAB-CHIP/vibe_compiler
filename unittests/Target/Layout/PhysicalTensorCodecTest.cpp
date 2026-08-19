//===- PhysicalTensorCodecTest.cpp - Physical tensor codec tests --------===//

#include "Wafer/Target/Layout/PhysicalTensorCodec.h"

#include "gtest/gtest.h"

#include "llvm/Support/Errc.h"
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

llvm::Expected<std::vector<uint8_t>>
packWithWindows(const NumericTensorKey &key,
                llvm::ArrayRef<RawLogicalValue> values, uint64_t maxBytes,
                uint64_t maxValues, uint8_t paddingFill) {
  llvm::Expected<PhysicalTensorWindowPlan> plan =
      PhysicalTensorWindowPlan::create(key);
  if (!plan)
    return plan.takeError();
  const LogicalScalarCodecPolicy policy =
      getModelProfileRecord(ModelProfileId::formalDeterministic())
          .numericEncodePolicy;
  std::vector<uint8_t> result;
  while (!plan->done()) {
    llvm::Expected<PhysicalTensorWindowPlan::WriteWindow> window =
        plan->takeNext(maxBytes, maxValues, paddingFill);
    if (!window)
      return window.takeError();
    if (window->physicalOffset != result.size())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "physical window plan produced a non-contiguous span");
    if (window->bytes.size() > maxBytes)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "physical window exceeded byte budget");
    for (const auto &element : window->elements) {
      if (element.logicalIndex >= values.size())
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "physical window referenced an unknown logical value");
      if (llvm::Error error =
              writeRawLogicalValue(values[element.logicalIndex], window->bytes,
                                   element.windowBitOffset, policy))
        return std::move(error);
    }
    result.insert(result.end(), window->bytes.begin(), window->bytes.end());
  }
  if (plan->getPlannedValueCount() != values.size())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "physical window plan did not cover every logical value");
  return result;
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

TEST(PhysicalTensorCodecTest,
     BoundedPhysicalOrderWindowsMatchFullCodecForBlockedAndBoolLayouts) {
  struct Case {
    LogicalFormat format;
    PhysicalTensorLayout layout;
    std::vector<uint64_t> shape;
    uint64_t maxBytes;
    uint64_t maxValues;
  };
  const std::vector<Case> cases = {
      {LogicalFormat::F32, PhysicalTensorLayout::Cx, {2, 128}, 64, 16},
      {LogicalFormat::F32, PhysicalTensorLayout::NCx, {2, 3, 128}, 68, 17},
      {LogicalFormat::F16, PhysicalTensorLayout::Cx, {2, 65}, 30, 15},
      {LogicalFormat::Bool, PhysicalTensorLayout::Tensor, {2, 5}, 1, 8},
      {LogicalFormat::Bool, PhysicalTensorLayout::Cx, {2, 5}, 7, 56},
      {LogicalFormat::Bool, PhysicalTensorLayout::NCx, {2, 3, 5}, 9, 72},
  };
  for (const Case &testCase : cases) {
    NumericTensorKey key =
        makeTensor(testCase.format, testCase.layout, testCase.shape);
    std::vector<RawLogicalValue> values;
    values.reserve(static_cast<size_t>(key.getElementCount()));
    for (uint64_t index = 0; index < key.getElementCount(); ++index) {
      const uint64_t bits = testCase.format == LogicalFormat::Bool
                                ? index % 3 == 0
                                : (testCase.format == LogicalFormat::F16
                                       ? UINT64_C(0x3c00) + index
                                       : UINT64_C(0x3f800000) + index);
      values.push_back({testCase.format, bits});
    }
    llvm::Expected<std::vector<uint8_t>> full =
        packPhysicalTensorLogicalValues(key, values, UINT8_C(0xa5));
    ASSERT_TRUE(static_cast<bool>(full)) << llvm::toString(full.takeError());
    llvm::Expected<std::vector<uint8_t>> bounded = packWithWindows(
        key, values, testCase.maxBytes, testCase.maxValues, UINT8_C(0xa5));
    ASSERT_TRUE(static_cast<bool>(bounded))
        << llvm::toString(bounded.takeError());
    EXPECT_EQ(*bounded, *full)
        << "layout=" << stringifyPhysicalTensorLayout(testCase.layout).str()
        << " format=" << stringifyLogicalFormat(testCase.format).str();
  }
}

} // namespace
