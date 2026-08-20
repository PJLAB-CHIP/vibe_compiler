//===- NpyPayloadTest.cpp - Frontend NPY payload tests ------------------===//

#include "Wafer/Frontend/Program/Program.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

class NpyPayloadTest : public testing::Test {
protected:
  void SetUp() override {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "wafer-npy-payload-test", temporaryDirectory);
    ASSERT_FALSE(error) << error.message();
  }

  void TearDown() override {
    if (!temporaryDirectory.empty())
      EXPECT_FALSE(llvm::sys::fs::remove_directories(temporaryDirectory));
  }

  llvm::SmallString<256> path(llvm::StringRef filename) const {
    llvm::SmallString<256> result(temporaryDirectory);
    llvm::sys::path::append(result, filename);
    return result;
  }

  void writeNpyV1(llvm::StringRef destination, llvm::StringRef descr,
                  llvm::ArrayRef<uint8_t> payload) {
    std::string header = (llvm::Twine("{'descr': '") + descr +
                          "', 'fortran_order': False, 'shape': (2,), }")
                             .str();
    constexpr size_t prefixSize = 10;
    constexpr size_t alignment = 64;
    size_t padding =
        (alignment - ((prefixSize + header.size() + 1) % alignment)) %
        alignment;
    header.append(padding, ' ');
    header.push_back('\n');
    ASSERT_LE(header.size(), 0xffffu);

    std::string file("\x93NUMPY", 6);
    file.push_back('\x01');
    file.push_back('\x00');
    uint16_t headerSize = static_cast<uint16_t>(header.size());
    file.push_back(static_cast<char>(headerSize & 0xff));
    file.push_back(static_cast<char>((headerSize >> 8) & 0xff));
    file.append(header);
    file.append(reinterpret_cast<const char *>(payload.data()), payload.size());

    std::error_code error;
    llvm::raw_fd_ostream output(destination, error, llvm::sys::fs::OF_None);
    ASSERT_FALSE(error) << error.message();
    output.write(file.data(), file.size());
    output.close();
    ASSERT_FALSE(output.has_error());
  }

  llvm::SmallString<256> temporaryDirectory;
};

TEST_F(NpyPayloadTest, LoadsLittleEndianF16WithCanonicalBytes) {
  llvm::SmallString<256> payloadPath = path("little-endian-f16.npy");
  const std::vector<uint8_t> bytes = {0x00, 0x3c, 0x00, 0xc0};
  writeNpyV1(payloadPath, "<f2", bytes);

  llvm::Expected<wafer::frontend::NpyTensorPayload> payload =
      wafer::frontend::loadNpyTensorPayload(payloadPath);
  ASSERT_TRUE(static_cast<bool>(payload))
      << llvm::toString(payload.takeError());
  EXPECT_EQ(payload->dtype, wafer::ProgramElementType::F16);
  EXPECT_EQ(payload->shape, std::vector<int64_t>({2}));
  EXPECT_EQ(payload->bytes, bytes);
}

TEST_F(NpyPayloadTest, LoadsOpaqueTwoByteElementsAsBf16WithoutChangingBytes) {
  llvm::SmallString<256> payloadPath = path("bf16.npy");
  const std::vector<uint8_t> bytes = {0x80, 0x3f, 0x00, 0xc0};
  writeNpyV1(payloadPath, "|V2", bytes);

  llvm::Expected<wafer::frontend::NpyTensorPayload> payload =
      wafer::frontend::loadNpyTensorPayload(payloadPath);
  ASSERT_TRUE(static_cast<bool>(payload))
      << llvm::toString(payload.takeError());
  EXPECT_EQ(payload->dtype, wafer::ProgramElementType::BF16);
  EXPECT_EQ(payload->shape, std::vector<int64_t>({2}));
  EXPECT_EQ(payload->bytes, bytes);
}

TEST_F(NpyPayloadTest, RejectsBigEndianF16) {
  llvm::SmallString<256> payloadPath = path("big-endian-f16.npy");
  writeNpyV1(payloadPath, ">f2", {0x3c, 0x00, 0xc0, 0x00});

  llvm::Expected<wafer::frontend::NpyTensorPayload> payload =
      wafer::frontend::loadNpyTensorPayload(payloadPath);
  ASSERT_FALSE(static_cast<bool>(payload));
  EXPECT_NE(llvm::toString(payload.takeError()).find("unsupported dtype"),
            std::string::npos);
}

TEST_F(NpyPayloadTest, NativeEndianF16FollowsHostEndianness) {
  llvm::SmallString<256> payloadPath = path("native-endian-f16.npy");
  const std::vector<uint8_t> bytes = {0x00, 0x3c, 0x00, 0xc0};
  writeNpyV1(payloadPath, "=f2", bytes);

  llvm::Expected<wafer::frontend::NpyTensorPayload> payload =
      wafer::frontend::loadNpyTensorPayload(payloadPath);
  if (llvm::endianness::native == llvm::endianness::little) {
    ASSERT_TRUE(static_cast<bool>(payload))
        << llvm::toString(payload.takeError());
    EXPECT_EQ(payload->dtype, wafer::ProgramElementType::F16);
    EXPECT_EQ(payload->bytes, bytes);
  } else {
    ASSERT_FALSE(static_cast<bool>(payload));
    EXPECT_NE(llvm::toString(payload.takeError()).find("unsupported dtype"),
              std::string::npos);
  }
}

} // namespace
