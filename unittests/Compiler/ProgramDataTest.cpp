//===- ProgramDataTest.cpp - Transaction-owned program data tests --------===//

#include "Wafer/Compiler/ProgramData.h"
#include "Wafer/Compiler/ProgramInvocation.h"

#include "Wafer/Frontend/Program.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace wafer::compiler;

class ProgramDataTest : public testing::Test {
protected:
  void SetUp() override {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "wafer-program-data-test", temporaryDirectory);
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

  /// Writes a canonical v1 row-major NPY payload.
  void writeNpy(llvm::StringRef filename, llvm::StringRef descr,
                llvm::ArrayRef<int64_t> shape,
                llvm::ArrayRef<uint8_t> payload) {
    std::string header = "{'descr': '" + descr.str() +
                         "', 'fortran_order': False, 'shape': (";
    for (auto [index, dim] : llvm::enumerate(shape)) {
      if (index)
        header += ", ";
      header += std::to_string(dim);
    }
    header += "), }";
    constexpr size_t kPreambleSize = 10;
    size_t padding = (16 - ((kPreambleSize + header.size() + 1) % 16)) % 16;
    header.append(padding, ' ');
    header.push_back('\n');
    ASSERT_LT(header.size(), 65536u);

    std::vector<uint8_t> bytes;
    const char magic[] = "\x93NUMPY";
    bytes.insert(bytes.end(), magic, magic + 6);
    bytes.push_back(1);
    bytes.push_back(0);
    bytes.push_back(static_cast<uint8_t>(header.size() & 0xff));
    bytes.push_back(static_cast<uint8_t>((header.size() >> 8) & 0xff));
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), payload.begin(), payload.end());

    llvm::SmallString<256> target = path(filename);
    std::error_code error;
    llvm::raw_fd_ostream output(target, error);
    ASSERT_FALSE(error) << error.message();
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    output.close();
    ASSERT_FALSE(output.has_error());
  }

  std::vector<uint8_t> f32Bytes(llvm::ArrayRef<float> values) {
    std::vector<uint8_t> bytes(values.size() * 4);
    std::memcpy(bytes.data(), values.data(), values.size() * 4);
    return bytes;
  }

  llvm::SmallString<256> temporaryDirectory;
};

TEST_F(ProgramDataTest, EstablishmentVerifiesHeaderExtentAndDigest) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("valid.npy", "<f4", {2, 2}, payload);

  auto source = ProgramDataSource::establish(path("valid.npy"), "data/weight", nullptr);
  ASSERT_TRUE(static_cast<bool>(source));
  EXPECT_EQ(source->getDType(), "f32");
  EXPECT_TRUE(source->getShape() == llvm::ArrayRef<int64_t>({2, 2}));
  EXPECT_EQ(source->getSize(), static_cast<uint64_t>(source->getPayloadOffset()) +
                                  payload.size());
  EXPECT_EQ(source->getLocator(), "data/weight");
  EXPECT_FALSE(source->getContentDigest().empty());

  // Pinned content: reads observe the established bytes.
  std::vector<uint8_t> readback(payload.size());
  ASSERT_FALSE(
      static_cast<bool>(source->readRange(source->getPayloadOffset(),
                                          readback)));
  EXPECT_EQ(readback, payload);

  // Re-establishment of unchanged bytes produces the same digest.
  auto repeated = ProgramDataSource::establish(path("valid.npy"), "data/weight", nullptr);
  ASSERT_TRUE(static_cast<bool>(repeated));
  EXPECT_EQ(repeated->getContentDigest(), source->getContentDigest());
}

TEST_F(ProgramDataTest, EstablishmentClassifiesPayloadFailures) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("truncated.npy", "<f4", {2, 2}, {payload[0], payload[1]});
  writeNpy("trailing.npy", "<f4", {2, 2}, payload);
  {
    llvm::SmallString<256> target = path("trailing.npy");
    std::error_code error;
    llvm::raw_fd_ostream output(target, error, llvm::sys::fs::OF_Append);
    ASSERT_FALSE(error);
    output << "extra";
    output.close();
  }
  writeNpy("fortran.npy", "<f4", {2, 2}, payload);
  {
    // Rewrite with Fortran order marker.
    std::vector<uint8_t> bytes;
    const char magic[] = "\x93NUMPY";
    bytes.insert(bytes.end(), magic, magic + 6);
    bytes.push_back(1);
    bytes.push_back(0);
    std::string header = "{'descr': '<f4', 'fortran_order': True, 'shape': "
                         "(2, 2), }";
    bytes.push_back(static_cast<uint8_t>(header.size() & 0xff));
    bytes.push_back(static_cast<uint8_t>((header.size() >> 8) & 0xff));
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    llvm::SmallString<256> target = path("fortran.npy");
    std::error_code error;
    llvm::raw_fd_ostream output(target, error);
    ASSERT_FALSE(error);
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    output.close();
  }
  writeNpy("unsupported.npy", "<c8", {2, 2}, payload);

  auto classify = [&](llvm::StringRef filename,
                      ProgramDataFailureKind expected) {
    ProgramDataFailure failure;
    auto source = ProgramDataSource::establish(path(filename), "data/x", &failure);
    ASSERT_FALSE(static_cast<bool>(source)) << filename.str();
    llvm::consumeError(source.takeError());
    EXPECT_EQ(failure.kind, expected) << filename.str();
    EXPECT_EQ(failure.locator, "data/x");
  };
  classify("truncated.npy", ProgramDataFailureKind::TruncatedPayload);
  classify("trailing.npy", ProgramDataFailureKind::TrailingPayload);
  classify("fortran.npy", ProgramDataFailureKind::UnsupportedEncoding);
  classify("unsupported.npy", ProgramDataFailureKind::UnsupportedEncoding);
  classify("missing.npy", ProgramDataFailureKind::MissingPayload);

  {
    // Invalid magic.
    llvm::SmallString<256> target = path("bad-magic.npy");
    std::error_code error;
    llvm::raw_fd_ostream output(target, error);
    ASSERT_FALSE(error);
    output << "not a numpy file at all, long enough";
    output.close();
  }
  classify("bad-magic.npy", ProgramDataFailureKind::HeaderInvalid);
}

TEST_F(ProgramDataTest, EstablishmentPinsContentAgainstPathMutation) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("stable.npy", "<f4", {2, 2}, payload);

  auto source = ProgramDataSource::establish(path("stable.npy"), "data/weight", nullptr);
  ASSERT_TRUE(static_cast<bool>(source));
  const std::string digest = source->getContentDigest().str();

  // Replace the path content after establishment; the owned bytes and digest
  // must not change.
  std::vector<uint8_t> replacement = f32Bytes({9.0f, 9.0f, 9.0f, 9.0f});
  writeNpy("stable.npy", "<f4", {2, 2}, replacement);
  std::vector<uint8_t> readback(payload.size());
  ASSERT_FALSE(static_cast<bool>(
      source->readRange(source->getPayloadOffset(), readback)));
  EXPECT_EQ(readback, payload);
  EXPECT_EQ(source->getContentDigest(), digest);
}

TEST_F(ProgramDataTest, RangeCreationChecksGeometryAndMaterializes) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f});
  writeNpy("range.npy", "<f4", {4, 2}, payload);

  auto source = ProgramDataSource::establish(path("range.npy"), "data/weight", nullptr);
  ASSERT_TRUE(static_cast<bool>(source));
  const ProgramTensorId tensorId{ProgramResourceRole::Parameter, 0};

  auto classify = [&](llvm::ArrayRef<int64_t> offsets,
                      llvm::ArrayRef<int64_t> sizes,
                      llvm::ArrayRef<int64_t> strides,
                      ProgramDataFailureKind expected) {
    ProgramDataFailure failure;
    auto range = ProgramDataRange::create(
        tensorId, "f32", {4, 2}, sizes, wafer::frontend::ProgramDistributionKind::Partitioned,
        offsets, sizes, strides, SourceDataId{0}, *source, &failure);
    ASSERT_FALSE(static_cast<bool>(range));
    llvm::consumeError(range.takeError());
    EXPECT_EQ(failure.kind, expected);
  };

  // Full tensor: contiguous, materializes the whole payload.
  {
    ProgramDataFailure failure;
    auto range = ProgramDataRange::create(
        tensorId, "f32", {4, 2}, {4, 2},
        wafer::frontend::ProgramDistributionKind::Replicated,
        std::vector<int64_t>{0, 0}, std::vector<int64_t>{4, 2},
        std::vector<int64_t>{1, 1}, SourceDataId{0}, *source, &failure);
    ASSERT_TRUE(static_cast<bool>(range));
    EXPECT_EQ(range->getRegionLength(), payload.size());
    std::vector<uint8_t> out(payload.size());
    ASSERT_FALSE(static_cast<bool>(range->materialize(*source, out)));
    EXPECT_EQ(out, payload);
  }

  // Row-contiguous slice (rows 1..3): offsets [1,0], sizes [2,2].
  {
    ProgramDataFailure failure;
    auto range = ProgramDataRange::create(
        tensorId, "f32", {4, 2}, {2, 2},
        wafer::frontend::ProgramDistributionKind::Partitioned,
        std::vector<int64_t>{1, 0}, std::vector<int64_t>{2, 2},
        std::vector<int64_t>{1, 1}, SourceDataId{0}, *source, &failure);
    ASSERT_TRUE(static_cast<bool>(range));
    std::vector<uint8_t> out(range->getRegionLength());
    ASSERT_FALSE(static_cast<bool>(range->materialize(*source, out)));
    std::vector<uint8_t> expected = f32Bytes({3.0f, 4.0f, 5.0f, 6.0f});
    EXPECT_EQ(out, expected);
  }

  // Strided column slice (second column): offsets [0,1], sizes [4,1].
  {
    ProgramDataFailure failure;
    auto range = ProgramDataRange::create(
        tensorId, "f32", {4, 2}, {4, 1},
        wafer::frontend::ProgramDistributionKind::Partitioned,
        std::vector<int64_t>{0, 1}, std::vector<int64_t>{4, 1},
        std::vector<int64_t>{1, 1}, SourceDataId{0}, *source, &failure);
    ASSERT_TRUE(static_cast<bool>(range));
    std::vector<uint8_t> out(range->getRegionLength());
    ASSERT_FALSE(static_cast<bool>(range->materialize(*source, out)));
    std::vector<uint8_t> expected = f32Bytes({2.0f, 4.0f, 6.0f, 8.0f});
    EXPECT_EQ(out, expected);
  }

  // Failure classes.
  classify({4, 0}, {1, 2}, {1, 1}, ProgramDataFailureKind::ShapeMismatch);
  classify({0, 0}, {2, 2}, {1, 2}, ProgramDataFailureKind::UnsupportedEncoding);
  {
    // dtype disagreement with the payload source.
    ProgramDataFailure failure;
    auto range = ProgramDataRange::create(
        tensorId, "f16", {4, 2}, {4, 2},
        wafer::frontend::ProgramDistributionKind::Replicated,
        std::vector<int64_t>{0, 0}, std::vector<int64_t>{4, 2},
        std::vector<int64_t>{1, 1}, SourceDataId{0}, *source, &failure);
    ASSERT_FALSE(static_cast<bool>(range));
    llvm::consumeError(range.takeError());
    EXPECT_EQ(failure.kind, ProgramDataFailureKind::DTypeMismatch);
  }
  {
    // Range over a slice that would exceed the payload.
    writeNpy("small.npy", "<f4", {2, 2}, f32Bytes({1.0f, 2.0f, 3.0f, 4.0f}));
    auto small = ProgramDataSource::establish(path("small.npy"), "data/small", nullptr);
    ASSERT_TRUE(static_cast<bool>(small));
    ProgramDataFailure failure;
    auto range = ProgramDataRange::create(
        tensorId, "f32", {4, 2}, {4, 2},
        wafer::frontend::ProgramDistributionKind::Replicated,
        std::vector<int64_t>{0, 0}, std::vector<int64_t>{4, 2},
        std::vector<int64_t>{1, 1}, SourceDataId{0}, *small, &failure);
    ASSERT_FALSE(static_cast<bool>(range));
    llvm::consumeError(range.takeError());
    EXPECT_EQ(failure.kind, ProgramDataFailureKind::TruncatedPayload);
  }
}

TEST_F(ProgramDataTest, HandoffOwnsSourcesAndResolvesRanges) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("handoff.npy", "<f4", {2, 2}, payload);

  ProgramDataHandoff handoff;
  ProgramDataFailure failure;
  auto id = handoff.establishSource(path("handoff.npy"), "data/weight",
                                    &failure);
  ASSERT_TRUE(static_cast<bool>(id));
  EXPECT_EQ(id->value, 0);
  EXPECT_EQ(handoff.getSourceCount(), 1u);

  ProgramDataFailure rangeFailure;
  auto range = ProgramDataRange::create(
      {ProgramResourceRole::Parameter, 0}, "f32", {2, 2}, {2, 2},
      wafer::frontend::ProgramDistributionKind::Replicated,
      std::vector<int64_t>{0, 0}, std::vector<int64_t>{2, 2},
      std::vector<int64_t>{1, 1}, *id, handoff.getSource(*id), &rangeFailure);
  ASSERT_TRUE(static_cast<bool>(range));
  ASSERT_FALSE(static_cast<bool>(handoff.addRange(std::move(*range))));

  const ProgramDataRange *found = handoff.findRange(
      {ProgramResourceRole::Parameter, 0});
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->getSourceId(), *id);
  EXPECT_EQ(handoff.findRange({ProgramResourceRole::Parameter, 1}), nullptr);
  EXPECT_EQ(handoff.findRange({ProgramResourceRole::Constant, 0}), nullptr);

  // Duplicate tensor identity is rejected.
  auto duplicate = ProgramDataRange::create(
      {ProgramResourceRole::Parameter, 0}, "f32", {2, 2}, {2, 2},
      wafer::frontend::ProgramDistributionKind::Replicated,
      std::vector<int64_t>{0, 0}, std::vector<int64_t>{2, 2},
      std::vector<int64_t>{1, 1}, *id, handoff.getSource(*id), nullptr);
  ASSERT_TRUE(static_cast<bool>(duplicate));
  llvm::Error duplicateError = handoff.addRange(std::move(*duplicate));
  EXPECT_TRUE(static_cast<bool>(duplicateError));
  llvm::consumeError(std::move(duplicateError));

  // Materialization through the handoff accounts the read.
  std::vector<uint8_t> out(payload.size());
  ASSERT_FALSE(static_cast<bool>(handoff.materializeRange(*found, out)));
  EXPECT_EQ(out, payload);
  EXPECT_EQ(handoff.getIOStatistics().rangeMaterializations, 1u);
  EXPECT_EQ(handoff.getIOStatistics().sourceOpens, 1u);
  EXPECT_EQ(handoff.getIOStatistics().digestPasses, 1u);
}

TEST_F(ProgramDataTest, RegionDigestIdentifiesByteIdenticalShards) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("original.npy", "<f4", {2, 2}, payload);
  // A replicated shard stores the same payload under a different header.
  writeNpy("shard.npy", "<f4", {2, 2}, payload);

  auto original = ProgramDataSource::establish(path("original.npy"), "data/weight", nullptr);
  auto shard = ProgramDataSource::establish(path("shard.npy"),
                         "parameter_shards/weight/partition_00000.npy",
                         nullptr);
  ASSERT_TRUE(static_cast<bool>(original));
  ASSERT_TRUE(static_cast<bool>(shard));

  auto originalRegion = computePayloadRegionDigest(
      *original, std::vector<int64_t>{0, 0}, std::vector<int64_t>{2, 2});
  auto shardRegion = computePayloadRegionDigest(
      *shard, std::vector<int64_t>{0, 0}, std::vector<int64_t>{2, 2});
  ASSERT_TRUE(static_cast<bool>(originalRegion));
  ASSERT_TRUE(static_cast<bool>(shardRegion));
  EXPECT_EQ(*originalRegion, *shardRegion);

  // A genuinely different partition must not deduplicate.
  std::vector<uint8_t> other = f32Bytes({5.0f, 6.0f, 7.0f, 8.0f});
  writeNpy("other-shard.npy", "<f4", {2, 2}, other);
  auto otherShard = ProgramDataSource::establish(
      path("other-shard.npy"), "parameter_shards/weight/partition_00001.npy",
      nullptr);
  ASSERT_TRUE(static_cast<bool>(otherShard));
  auto otherRegion = computePayloadRegionDigest(
      *otherShard, std::vector<int64_t>{0, 0}, std::vector<int64_t>{2, 2});
  ASSERT_TRUE(static_cast<bool>(otherRegion));
  EXPECT_NE(*originalRegion, *otherRegion);
}

TEST_F(ProgramDataTest, MaterializeSourceToFileVerifiesDigest) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("copy.npy", "<f4", {2, 2}, payload);

  ProgramDataHandoff handoff;
  ProgramDataFailure failure;
  auto id = handoff.establishSource(path("copy.npy"), "data/weight",
                                    &failure);
  ASSERT_TRUE(static_cast<bool>(id));

  llvm::SmallString<256> destination = path("materialized.npy");
  ASSERT_FALSE(static_cast<bool>(
      handoff.materializeSourceToFile(*id, destination)));
  auto digest = ProgramDataHandoff::computeFileDigest(destination);
  ASSERT_TRUE(static_cast<bool>(digest));
  EXPECT_EQ(*digest, handoff.getSource(*id).getContentDigest());
  EXPECT_EQ(handoff.getIOStatistics().materializedFileWrites, 1u);
  EXPECT_EQ(handoff.getIOStatistics().materializedWriteBytes,
            handoff.getSource(*id).getSize());

  // Corrupting the written file must change its digest: this is exactly the
  // readback check materializeSourceToFile performs on the helper boundary.
  {
    std::error_code error;
    llvm::raw_fd_ostream output(destination, error, llvm::sys::fs::OF_Append);
    ASSERT_FALSE(error);
    output << "corruption";
    output.close();
  }
  auto corrupted = ProgramDataHandoff::computeFileDigest(destination);
  ASSERT_TRUE(static_cast<bool>(corrupted));
  EXPECT_NE(*corrupted, handoff.getSource(*id).getContentDigest());
}

TEST_F(ProgramDataTest, SharedProgramTensorViewsNeverDuplicatePayload) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  auto storage =
      std::make_shared<const std::vector<uint8_t>>(payload);
  auto tensor = ProgramTensor::share("f32", {2, 2}, storage, 0);
  ASSERT_TRUE(static_cast<bool>(tensor));
  EXPECT_EQ(tensor->getBytes().size(), payload.size());
  EXPECT_EQ(tensor->getBytes().data(), storage->data());
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(tensor->getBytes()), llvm::ArrayRef<uint8_t>(payload));

  // Out-of-bounds views are rejected.
  auto invalid = ProgramTensor::share("f32", {2, 2}, storage, 2);
  EXPECT_FALSE(static_cast<bool>(invalid));
  llvm::consumeError(invalid.takeError());
}

TEST_F(ProgramDataTest, DTypeWidthTableIsTheSingleSource) {
  EXPECT_EQ(getProgramDTypeElementBytes("f16"), 2);
  EXPECT_EQ(getProgramDTypeElementBytes("bf16"), 2);
  EXPECT_EQ(getProgramDTypeElementBytes("f32"), 4);
  EXPECT_EQ(getProgramDTypeElementBytes("i64"), 8);
  EXPECT_EQ(getProgramDTypeElementBytes("i1"), 1);
  EXPECT_EQ(getProgramDTypeElementBytes("i8"), 1);
  EXPECT_FALSE(getProgramDTypeElementBytes("f128").has_value());
}

} // namespace
