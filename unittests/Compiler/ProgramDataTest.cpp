//===- ProgramDataTest.cpp - Transaction-owned program data tests --------===//

#include "Wafer/Compiler/ProgramData.h"
#include "Wafer/Compiler/ProgramInvocation.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "Wafer/Frontend/Program.h"

#include "../../lib/Wafer/Compiler/CardExecutableInternal.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
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
    std::string header =
        "{'descr': '" + descr.str() + "', 'fortran_order': False, 'shape': (";
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

  llvm::SmallString<256> ownedPath(llvm::StringRef name) const {
    return path(("owned-" + name).str());
  }

  llvm::SmallString<256> temporaryDirectory;
};

TEST_F(ProgramDataTest, EstablishmentVerifiesHeaderExtentAndDigest) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("valid.npy", "<f4", {2, 2}, payload);

  auto source = ProgramDataSource::establish(
      path("valid.npy"), ownedPath("valid"), "data/weight", nullptr);
  ASSERT_TRUE(static_cast<bool>(source));
  EXPECT_EQ(source->getDType(), "f32");
  EXPECT_TRUE(source->getShape() == llvm::ArrayRef<int64_t>({2, 2}));
  EXPECT_EQ(source->getSize(),
            static_cast<uint64_t>(source->getPayloadOffset()) + payload.size());
  EXPECT_EQ(source->getLocator(), "data/weight");
  EXPECT_FALSE(source->getContentDigest().empty());
  auto ownedDigest =
      ProgramDataHandoff::computeFileDigest(source->getOwnedFilePath());
  ASSERT_TRUE(static_cast<bool>(ownedDigest));
  EXPECT_EQ(*ownedDigest, source->getContentDigest());

  // Owned content: reads observe the established bytes.
  std::vector<uint8_t> readback(payload.size());
  ASSERT_FALSE(static_cast<bool>(
      source->readRange(source->getPayloadOffset(), readback)));
  EXPECT_EQ(readback, payload);

  // Re-establishment of unchanged bytes produces the same digest.
  auto repeated = ProgramDataSource::establish(
      path("valid.npy"), ownedPath("valid-2"), "data/weight", nullptr);
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
    auto source = ProgramDataSource::establish(
        path(filename), ownedPath(filename), "data/x", &failure);
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

TEST_F(ProgramDataTest, EstablishmentRejectsNonAdmittedProgramDtype) {
  // Bitpacked boolean decodes at the NPY source layer as i1, but the program
  // boundary admits no boolean target representation yet: establishment must
  // fail closed with a typed classification.
  std::vector<uint8_t> payload = {1, 0, 1, 0};
  writeNpy("bool.npy", "|b1", {2, 2}, payload);

  ProgramDataFailure failure;
  auto source = ProgramDataSource::establish(
      path("bool.npy"), ownedPath("bool"), "data/bool", &failure);
  ASSERT_FALSE(static_cast<bool>(source));
  llvm::consumeError(source.takeError());
  EXPECT_EQ(failure.kind, ProgramDataFailureKind::UnsupportedEncoding);
}

TEST_F(ProgramDataTest, EstablishmentOwnsContentAgainstInPlaceMutation) {
  // Payload well above the old mmap threshold: the owned copy, not the user
  // file mapping, must answer reads after an in-place rewrite of the same
  // inode.
  std::vector<float> values;
  for (int index = 0; index < 8192; ++index)
    values.push_back(static_cast<float>(index));
  std::vector<uint8_t> payload = f32Bytes(values);
  writeNpy("large.npy", "<f4", {8192}, payload);

  auto source = ProgramDataSource::establish(
      path("large.npy"), ownedPath("large"), "data/large", nullptr);
  ASSERT_TRUE(static_cast<bool>(source));
  EXPECT_GT(source->getSize(), 16384u)
      << "fixture must exceed the pinned LLVM mmap threshold";
  const std::string digest = source->getContentDigest().str();

  // In-place rewrite of the same inode: the NPY header stays intact, the
  // payload bytes change to a different pattern of the same size.
  {
    std::vector<uint8_t> prefix(
        static_cast<size_t>(source->getPayloadOffset()));
    ASSERT_FALSE(static_cast<bool>(source->readRange(0, prefix)));
    std::vector<uint8_t> replacement(payload.size(), 0xAA);
    llvm::SmallString<256> target = path("large.npy");
    std::error_code error;
    llvm::raw_fd_ostream output(target, error);
    ASSERT_FALSE(error);
    output.write(reinterpret_cast<const char *>(prefix.data()), prefix.size());
    output.write(reinterpret_cast<const char *>(replacement.data()),
                 replacement.size());
    output.close();
    ASSERT_FALSE(output.has_error());
  }

  std::vector<uint8_t> readback(payload.size());
  ASSERT_FALSE(static_cast<bool>(
      source->readRange(source->getPayloadOffset(), readback)));
  EXPECT_EQ(readback, payload);
  EXPECT_EQ(source->getContentDigest(), digest);

  // A fresh establishment observes the mutated bytes: the mutation above was
  // real and the owned content really diverged.
  auto reestablished = ProgramDataSource::establish(
      path("large.npy"), ownedPath("large-2"), "data/large", nullptr);
  ASSERT_TRUE(static_cast<bool>(reestablished));
  EXPECT_NE(reestablished->getContentDigest(), digest);
}

TEST_F(ProgramDataTest, RangeCreationChecksGeometryAndMaterializes) {
  std::vector<uint8_t> payload =
      f32Bytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  writeNpy("range.npy", "<f4", {4, 2}, payload);

  auto source = ProgramDataSource::establish(
      path("range.npy"), ownedPath("range"), "data/weight", nullptr);
  ASSERT_TRUE(static_cast<bool>(source));
  const ProgramTensorId tensorId{ProgramResourceRole::Parameter, 0};

  auto classify =
      [&](llvm::ArrayRef<int64_t> offsets, llvm::ArrayRef<int64_t> sizes,
          llvm::ArrayRef<int64_t> strides, ProgramDataFailureKind expected) {
        ProgramDataFailure failure;
        auto range = ProgramDataRange::create(
            tensorId, "f32", {4, 2}, sizes,
            wafer::frontend::ProgramDistributionKind::Partitioned,
            ProgramDataRangeOrigin::OriginalSource, offsets, sizes, strides,
            SourceDataId{0}, *source, &failure);
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
        ProgramDataRangeOrigin::OriginalSource, std::vector<int64_t>{0, 0},
        std::vector<int64_t>{4, 2}, std::vector<int64_t>{1, 1}, SourceDataId{0},
        *source, &failure);
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
        ProgramDataRangeOrigin::OriginalSource, std::vector<int64_t>{1, 0},
        std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1}, SourceDataId{0},
        *source, &failure);
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
        ProgramDataRangeOrigin::OriginalSource, std::vector<int64_t>{0, 1},
        std::vector<int64_t>{4, 1}, std::vector<int64_t>{1, 1}, SourceDataId{0},
        *source, &failure);
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
        ProgramDataRangeOrigin::OriginalSource, std::vector<int64_t>{0, 0},
        std::vector<int64_t>{4, 2}, std::vector<int64_t>{1, 1}, SourceDataId{0},
        *source, &failure);
    ASSERT_FALSE(static_cast<bool>(range));
    llvm::consumeError(range.takeError());
    EXPECT_EQ(failure.kind, ProgramDataFailureKind::DTypeMismatch);
  }
  {
    // A source that is smaller than the global tensor is rejected by the
    // origin shape proof before any region arithmetic.
    writeNpy("small.npy", "<f4", {2, 2}, f32Bytes({1.0f, 2.0f, 3.0f, 4.0f}));
    auto small = ProgramDataSource::establish(
        path("small.npy"), ownedPath("small"), "data/small", nullptr);
    ASSERT_TRUE(static_cast<bool>(small));
    ProgramDataFailure failure;
    auto range = ProgramDataRange::create(
        tensorId, "f32", {4, 2}, {4, 2},
        wafer::frontend::ProgramDistributionKind::Replicated,
        ProgramDataRangeOrigin::OriginalSource, std::vector<int64_t>{0, 0},
        std::vector<int64_t>{4, 2}, std::vector<int64_t>{1, 1}, SourceDataId{0},
        *small, &failure);
    ASSERT_FALSE(static_cast<bool>(range));
    llvm::consumeError(range.takeError());
    EXPECT_EQ(failure.kind, ProgramDataFailureKind::ShapeMismatch);
  }
}

TEST_F(ProgramDataTest, RangeCreationProvesSourceShapeByOrigin) {
  std::vector<uint8_t> payload =
      f32Bytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  writeNpy("shape.npy", "<f4", {4, 2}, payload);
  auto source = ProgramDataSource::establish(
      path("shape.npy"), ownedPath("shape"), "data/weight", nullptr);
  ASSERT_TRUE(static_cast<bool>(source));
  const ProgramTensorId tensorId{ProgramResourceRole::Parameter, 0};

  auto create =
      [&](llvm::ArrayRef<int64_t> global, llvm::ArrayRef<int64_t> local,
          ProgramDataRangeOrigin origin, llvm::ArrayRef<int64_t> offsets,
          llvm::ArrayRef<int64_t> sizes, ProgramDataFailure *failure) {
        return ProgramDataRange::create(
            tensorId, "f32", global, local,
            wafer::frontend::ProgramDistributionKind::Partitioned, origin,
            offsets, sizes, std::vector<int64_t>(offsets.size(), 1),
            SourceDataId{0}, *source, failure);
      };

  // Same rank and byte count, different shape: an original-source range must
  // prove source shape == global shape.
  {
    ProgramDataFailure failure;
    auto range = create({2, 4}, {2, 4}, ProgramDataRangeOrigin::OriginalSource,
                        {0, 0}, {2, 4}, &failure);
    ASSERT_FALSE(static_cast<bool>(range));
    llvm::consumeError(range.takeError());
    EXPECT_EQ(failure.kind, ProgramDataFailureKind::ShapeMismatch);
  }

  // Original source with the matching global shape is admitted.
  {
    ProgramDataFailure failure;
    auto range = create({4, 2}, {2, 2}, ProgramDataRangeOrigin::OriginalSource,
                        {2, 0}, {2, 2}, &failure);
    ASSERT_TRUE(static_cast<bool>(range));
  }

  // A materialized shard source must carry exactly the local shape.
  {
    ProgramDataFailure failure;
    auto range =
        create({4, 2}, {2, 2}, ProgramDataRangeOrigin::MaterializedShard,
               {0, 0}, {2, 2}, &failure);
    ASSERT_FALSE(static_cast<bool>(range));
    llvm::consumeError(range.takeError());
    EXPECT_EQ(failure.kind, ProgramDataFailureKind::ShapeMismatch);
  }
  {
    // Non-zero slice offsets are not a materialized shard.
    ProgramDataFailure failure;
    auto range =
        create({4, 2}, {2, 2}, ProgramDataRangeOrigin::MaterializedShard,
               {2, 0}, {2, 2}, &failure);
    ASSERT_FALSE(static_cast<bool>(range));
    llvm::consumeError(range.takeError());
    EXPECT_EQ(failure.kind, ProgramDataFailureKind::ShapeMismatch);
  }
  {
    // Partial coverage is not a materialized shard.
    ProgramDataFailure failure;
    auto range =
        create({4, 2}, {2, 2}, ProgramDataRangeOrigin::MaterializedShard,
               {0, 0}, {2, 1}, &failure);
    ASSERT_FALSE(static_cast<bool>(range));
    llvm::consumeError(range.takeError());
    EXPECT_EQ(failure.kind, ProgramDataFailureKind::ShapeMismatch);
  }
  {
    // A real materialized shard (local shape stored in its own source file)
    // passes the proof against a shard-shaped source.
    writeNpy("shard.npy", "<f4", {2, 2}, f32Bytes({5.0f, 6.0f, 7.0f, 8.0f}));
    auto shard = ProgramDataSource::establish(
        path("shard.npy"), ownedPath("shard"),
        "parameter_shards/weight/partition_00000.npy", nullptr);
    ASSERT_TRUE(static_cast<bool>(shard));
    ProgramDataFailure failure;
    auto range = ProgramDataRange::create(
        tensorId, "f32", {4, 2}, {2, 2},
        wafer::frontend::ProgramDistributionKind::Partitioned,
        ProgramDataRangeOrigin::MaterializedShard, std::vector<int64_t>{0, 0},
        std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1}, SourceDataId{0},
        *shard, &failure);
    ASSERT_TRUE(static_cast<bool>(range));
    std::vector<uint8_t> out(range->getRegionLength());
    ASSERT_FALSE(static_cast<bool>(range->materialize(*shard, out)));
    EXPECT_EQ(out, f32Bytes({5.0f, 6.0f, 7.0f, 8.0f}));
  }
}

TEST_F(ProgramDataTest, HandoffOwnsSourcesAndResolvesRanges) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("handoff.npy", "<f4", {2, 2}, payload);

  ProgramDataHandoff handoff(temporaryDirectory.str().str());
  ProgramDataFailure failure;
  auto id =
      handoff.establishSource(path("handoff.npy"), "data/weight", &failure);
  ASSERT_TRUE(static_cast<bool>(id));
  EXPECT_EQ(id->value, 0);
  EXPECT_EQ(handoff.getSourceCount(), 1u);

  ProgramDataFailure rangeFailure;
  auto range = ProgramDataRange::create(
      {ProgramResourceRole::Parameter, 0}, "f32", {2, 2}, {2, 2},
      wafer::frontend::ProgramDistributionKind::Replicated,
      ProgramDataRangeOrigin::OriginalSource, std::vector<int64_t>{0, 0},
      std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1}, *id,
      handoff.getSource(*id), &rangeFailure);
  ASSERT_TRUE(static_cast<bool>(range));
  ASSERT_FALSE(static_cast<bool>(handoff.addRange(std::move(*range))));

  const ProgramDataRange *found =
      handoff.findRange({ProgramResourceRole::Parameter, 0});
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->getSourceId(), *id);
  EXPECT_EQ(handoff.findRange({ProgramResourceRole::Parameter, 1}), nullptr);
  EXPECT_EQ(handoff.findRange({ProgramResourceRole::Constant, 0}), nullptr);

  // Duplicate tensor identity is rejected.
  auto duplicate = ProgramDataRange::create(
      {ProgramResourceRole::Parameter, 0}, "f32", {2, 2}, {2, 2},
      wafer::frontend::ProgramDistributionKind::Replicated,
      ProgramDataRangeOrigin::OriginalSource, std::vector<int64_t>{0, 0},
      std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 1}, *id,
      handoff.getSource(*id), nullptr);
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
  EXPECT_EQ(handoff.getIOStatistics().fileOpens, 3u);
  EXPECT_EQ(handoff.getIOStatistics().readWindows, 4u);
  EXPECT_EQ(handoff.getIOStatistics().headerReads, 2u);
  EXPECT_EQ(handoff.getIOStatistics().digestPasses, 2u);
}

TEST_F(ProgramDataTest, HandoffStorageSurvivesMoveAndStagingCleanup) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("retained.npy", "<f4", {4}, payload);

  llvm::SmallString<256> staging = path("compile-staging");
  ASSERT_FALSE(llvm::sys::fs::create_directory(staging));
  std::unique_ptr<ProgramDataHandoff> retained;
  SourceDataId sourceId;
  std::string ownedPath;
  {
    // Production uses the stable output parent, not `staging`, as the
    // handoff storage parent.
    ProgramDataHandoff handoff(temporaryDirectory.str().str());
    ProgramDataFailure failure;
    auto established = handoff.establishSource(path("retained.npy"),
                                               "data/retained", &failure);
    ASSERT_TRUE(static_cast<bool>(established));
    sourceId = *established;
    ownedPath = handoff.getSource(sourceId).getOwnedFilePath().str();
    retained = std::make_unique<ProgramDataHandoff>(std::move(handoff));
  }

  ASSERT_FALSE(llvm::sys::fs::remove_directories(staging));
  EXPECT_TRUE(llvm::sys::fs::exists(ownedPath));
  std::vector<uint8_t> readback(payload.size());
  ASSERT_FALSE(static_cast<bool>(retained->getSource(sourceId).readRange(
      retained->getSource(sourceId).getPayloadOffset(), readback)));
  EXPECT_EQ(readback, payload);

  retained.reset();
  EXPECT_FALSE(llvm::sys::fs::exists(ownedPath));
}

TEST_F(ProgramDataTest, LargeRangeUsesBoundedAccountedReadWindows) {
  constexpr size_t kElementCount = 700000;
  std::vector<uint8_t> payload(kElementCount * sizeof(float));
  for (size_t index = 0; index < payload.size(); ++index)
    payload[index] = static_cast<uint8_t>(index);
  writeNpy("large-range.npy", "<f4", {static_cast<int64_t>(kElementCount)},
           payload);

  ProgramDataHandoff handoff(temporaryDirectory.str().str());
  ProgramDataFailure failure;
  auto sourceId = handoff.establishSource(path("large-range.npy"),
                                          "data/large-range", &failure);
  ASSERT_TRUE(static_cast<bool>(sourceId));
  auto range = ProgramDataRange::create(
      {ProgramResourceRole::Parameter, 0}, "f32",
      {static_cast<int64_t>(kElementCount)},
      {static_cast<int64_t>(kElementCount)},
      wafer::frontend::ProgramDistributionKind::Replicated,
      ProgramDataRangeOrigin::OriginalSource, std::vector<int64_t>{0},
      std::vector<int64_t>{static_cast<int64_t>(kElementCount)},
      std::vector<int64_t>{1}, *sourceId, handoff.getSource(*sourceId),
      &failure);
  ASSERT_TRUE(static_cast<bool>(range));
  ASSERT_FALSE(static_cast<bool>(handoff.addRange(std::move(*range))));
  const ProgramDataRange *establishedRange =
      handoff.findRange({ProgramResourceRole::Parameter, 0});
  ASSERT_NE(establishedRange, nullptr);

  const uint64_t windowsBefore = handoff.getIOStatistics().readWindows;
  const uint64_t bytesBefore = handoff.getIOStatistics().readBytes;
  const uint64_t opensBefore = handoff.getIOStatistics().fileOpens;
  std::vector<uint8_t> materialized(payload.size());
  ASSERT_FALSE(static_cast<bool>(
      handoff.materializeRange(*establishedRange, materialized)));
  EXPECT_EQ(materialized, payload);
  const uint64_t expectedWindows =
      (payload.size() + kProgramDataReadWindowBytes - 1) /
      kProgramDataReadWindowBytes;
  EXPECT_EQ(handoff.getIOStatistics().readWindows - windowsBefore,
            expectedWindows);
  EXPECT_EQ(handoff.getIOStatistics().readBytes - bytesBefore, payload.size());
  EXPECT_EQ(handoff.getIOStatistics().fileOpens, opensBefore)
      << "range reads must reuse the source-owned handle";
  EXPECT_LE(handoff.getIOStatistics().maximumReadWindowBytes,
            kProgramDataReadWindowBytes);
  EXPECT_EQ(handoff.getIOStatistics().rangeMaterializations, 1u);
}

TEST_F(ProgramDataTest, VerifyShardAgainstSourceDeduplicatesAndAdopts) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("original.npy", "<f4", {2, 2}, payload);
  // A replicated shard stores the same payload under a different file.
  writeNpy("replica.npy", "<f4", {2, 2}, payload);
  // A genuinely different partition must not deduplicate.
  std::vector<uint8_t> other = f32Bytes({5.0f, 6.0f, 7.0f, 8.0f});
  writeNpy("other-shard.npy", "<f4", {2, 2}, other);

  ProgramDataHandoff handoff(temporaryDirectory.str().str());
  ProgramDataFailure failure;
  auto originalId =
      handoff.establishSource(path("original.npy"), "data/weight", &failure);
  ASSERT_TRUE(static_cast<bool>(originalId));

  auto replica = handoff.establishHelperOutput(
      path("replica.npy"), "parameter_shards/weight/partition_00000.npy",
      &failure);
  ASSERT_TRUE(static_cast<bool>(replica));
  auto realPartition = handoff.establishHelperOutput(
      path("other-shard.npy"), "parameter_shards/weight/partition_00001.npy",
      &failure);
  ASSERT_TRUE(static_cast<bool>(realPartition));

  // The replicated shard is byte-identical to the original region.
  ProgramDataFailure verifyFailure;
  llvm::Expected<bool> identical = handoff.verifyShardAgainstSource(
      **replica, *originalId, std::vector<int64_t>{0, 0},
      std::vector<int64_t>{2, 2}, &verifyFailure);
  ASSERT_TRUE(static_cast<bool>(identical));
  EXPECT_TRUE(*identical);
  EXPECT_EQ(handoff.getSourceCount(), 1u);

  // The real partition is adopted as a new owned source.
  llvm::Expected<bool> partitioned = handoff.verifyShardAgainstSource(
      **realPartition, *originalId, std::vector<int64_t>{0, 0},
      std::vector<int64_t>{2, 2}, &verifyFailure);
  ASSERT_TRUE(static_cast<bool>(partitioned));
  EXPECT_FALSE(*partitioned);
  SourceDataId shardId = handoff.adoptCandidate(*realPartition);
  EXPECT_EQ(handoff.getSourceCount(), 2u);
  EXPECT_EQ(handoff.getCandidateCount(), 1u);
  EXPECT_EQ(handoff.getSource(shardId).getContentDigest(),
            (*realPartition)->getContentDigest());

  // Region digest failures are typed, not raw I/O errors.
  ProgramDataFailure rankFailure;
  llvm::Expected<bool> badRank = handoff.verifyShardAgainstSource(
      **replica, *originalId, std::vector<int64_t>{0}, std::vector<int64_t>{2},
      &rankFailure);
  ASSERT_FALSE(static_cast<bool>(badRank));
  llvm::consumeError(badRank.takeError());
  EXPECT_EQ(rankFailure.kind, ProgramDataFailureKind::ShapeMismatch);

  const std::string unusedCandidatePath = (*replica)->getOwnedFilePath().str();
  EXPECT_TRUE(llvm::sys::fs::exists(unusedCandidatePath));
  handoff.discardUnadoptedCandidates();
  EXPECT_EQ(handoff.getCandidateCount(), 0u);
  EXPECT_FALSE(llvm::sys::fs::exists(unusedCandidatePath));

  // The I/O ledger covers every establishment and digest pass: two helper
  // outputs, one canonical source, source-copy and owned-content digests for
  // all three, two shard region-digest pairs, plus the failed-rank attempt is
  // classified before any I/O.
  EXPECT_EQ(handoff.getIOStatistics().sourceOpens, 3u);
  EXPECT_EQ(handoff.getIOStatistics().fileOpens, 9u);
  EXPECT_EQ(handoff.getIOStatistics().readWindows, 13u);
  EXPECT_EQ(handoff.getIOStatistics().headerReads, 6u);
  EXPECT_EQ(handoff.getIOStatistics().helperOutputReadbacks, 2u);
  EXPECT_EQ(handoff.getIOStatistics().digestPasses, 10u);
}

TEST_F(ProgramDataTest, MaterializeSourceToFileVerifiesDigest) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("copy.npy", "<f4", {2, 2}, payload);

  ProgramDataHandoff handoff(temporaryDirectory.str().str());
  ProgramDataFailure failure;
  auto id = handoff.establishSource(path("copy.npy"), "data/weight", &failure);
  ASSERT_TRUE(static_cast<bool>(id));

  llvm::SmallString<256> destination = path("materialized.npy");
  ASSERT_FALSE(
      static_cast<bool>(handoff.materializeSourceToFile(*id, destination)));
  auto digest = ProgramDataHandoff::computeFileDigest(destination);
  ASSERT_TRUE(static_cast<bool>(digest));
  EXPECT_EQ(*digest, handoff.getSource(*id).getContentDigest());
  EXPECT_EQ(handoff.getIOStatistics().materializedFileWrites, 1u);
  EXPECT_EQ(handoff.getIOStatistics().fileOpens, 5u);
  EXPECT_EQ(handoff.getIOStatistics().readWindows, 5u);
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

TEST_F(ProgramDataTest, MaterializationFailuresCarryTypedClassification) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("copy.npy", "<f4", {2, 2}, payload);

  ProgramDataHandoff handoff(temporaryDirectory.str().str());
  ProgramDataFailure failure;
  auto id = handoff.establishSource(path("copy.npy"), "data/weight", &failure);
  ASSERT_TRUE(static_cast<bool>(id));

  // A destination whose parent is a regular file cannot be created: the
  // failure must be typed MaterializationIO with the source locator.
  {
    llvm::SmallString<256> blocker = path("blocker");
    std::error_code error;
    llvm::raw_fd_ostream output(blocker, error);
    ASSERT_FALSE(error);
    output << "file";
    output.close();
  }
  llvm::SmallString<256> impossible = path("blocker");
  llvm::sys::path::append(impossible, "sub");
  llvm::sys::path::append(impossible, "materialized.npy");
  ProgramDataFailure materializationFailure;
  llvm::Error error =
      handoff.materializeSourceToFile(*id, impossible, &materializationFailure);
  EXPECT_TRUE(static_cast<bool>(error));
  llvm::consumeError(std::move(error));
  EXPECT_EQ(materializationFailure.kind,
            ProgramDataFailureKind::MaterializationIO);
  EXPECT_EQ(materializationFailure.locator, "data/weight");
  EXPECT_FALSE(materializationFailure.detail.empty());

  // A handoff without a scratch directory cannot establish: typed failure,
  // not a raw environment error.
  ProgramDataHandoff unrooted;
  ProgramDataFailure unrootedFailure;
  auto unrootedId = unrooted.establishSource(path("copy.npy"), "data/weight",
                                             &unrootedFailure);
  ASSERT_FALSE(static_cast<bool>(unrootedId));
  llvm::consumeError(unrootedId.takeError());
  EXPECT_EQ(unrootedFailure.kind, ProgramDataFailureKind::MaterializationIO);
}

TEST_F(ProgramDataTest, SharedProgramTensorViewsNeverDuplicatePayload) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  auto storage = std::make_shared<const std::vector<uint8_t>>(payload);
  auto tensor = ProgramTensor::share("f32", {2, 2}, storage, 0);
  ASSERT_TRUE(static_cast<bool>(tensor));
  EXPECT_EQ(tensor->getBytes().size(), payload.size());
  EXPECT_EQ(tensor->getBytes().data(), storage->data());
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(tensor->getBytes()),
            llvm::ArrayRef<uint8_t>(payload));

  // Out-of-bounds views are rejected.
  auto invalid = ProgramTensor::share("f32", {2, 2}, storage, 2);
  EXPECT_FALSE(static_cast<bool>(invalid));
  llvm::consumeError(invalid.takeError());
}

TEST_F(ProgramDataTest, DTypeWidthTableAdmitsProgramBoundaryDtypes) {
  EXPECT_EQ(getProgramDTypeElementBytes("f16"), 2);
  EXPECT_EQ(getProgramDTypeElementBytes("bf16"), 2);
  EXPECT_EQ(getProgramDTypeElementBytes("f32"), 4);
  EXPECT_EQ(getProgramDTypeElementBytes("i64"), 8);
  EXPECT_EQ(getProgramDTypeElementBytes("i8"), 1);
  // Boolean has no target representation yet: not admitted at the program
  // boundary even though the NPY source layer can decode it.
  EXPECT_FALSE(getProgramDTypeElementBytes("i1").has_value());
  EXPECT_FALSE(getProgramDTypeElementBytes("f128").has_value());
}

namespace {

wafer::frontend::ProgramPartitionSlice
singleCardPartitionSlice(llvm::ArrayRef<int64_t> shape) {
  wafer::frontend::ProgramPartitionSlice slice;
  slice.partitionId = 0;
  slice.replicaId = 0;
  slice.offsets.assign(shape.size(), 0);
  slice.sizes.assign(shape.begin(), shape.end());
  slice.strides.assign(shape.size(), 1);
  return slice;
}

wafer::frontend::ProgramBoundaryBinding
shapedBoundary(int64_t index, llvm::ArrayRef<int64_t> shape) {
  wafer::frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  binding.globalShape.assign(shape.begin(), shape.end());
  binding.localShape.assign(shape.begin(), shape.end());
  binding.dtype = "f32";
  binding.partitionSlices.push_back(singleCardPartitionSlice(shape));
  return binding;
}

} // namespace

TEST_F(ProgramDataTest, PrepareProgramInvocationsMaterializesOnceAcrossTiles) {
  std::vector<uint8_t> payload = f32Bytes({1.0f, 2.0f, 3.0f, 4.0f});
  writeNpy("weight.npy", "<f4", {4}, payload);
  writeNpy("replicated-shard.npy", "<f4", {4}, payload);

  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<4xf32>, %weight: tensor<4xf32>)
      -> tensor<4xf32> {
    %tmp = tensor.empty() : tensor<4xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%lhs, %weight : tensor<4xf32>, tensor<4xf32>)
        outs(%tmp : tensor<4xf32>) {
      ^bb0(%a: f32, %b: f32, %old: f32):
        %value = arith.addf %a, %b : f32
        linalg.yield %value : f32
    } -> tensor<4xf32>
    return %sum : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(tensorProgram);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.programParameterCount = 1;
  program.distributedInputs = {shapedBoundary(0, {4})};
  program.distributedOutputs = {shapedBoundary(0, {4})};
  wafer::frontend::ProgramParameterBinding parameter;
  parameter.argumentIndex = 1;
  parameter.name = "weight";
  parameter.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  parameter.globalShape = {4};
  parameter.localShape = {4};
  parameter.dtype = "f32";
  parameter.partitionSlices.push_back(singleCardPartitionSlice({4}));
  program.parameters.push_back(std::move(parameter));

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(config));

  ProgramDataHandoff handoff(temporaryDirectory.str().str());
  ProgramDataFailure failure;
  auto sourceId =
      handoff.establishSource(path("weight.npy"), "data/weight", &failure);
  ASSERT_TRUE(static_cast<bool>(sourceId));
  auto unusedCandidate = handoff.establishHelperOutput(
      path("replicated-shard.npy"),
      "parameter_shards/weight/partition_00000.npy", &failure);
  ASSERT_TRUE(static_cast<bool>(unusedCandidate));
  EXPECT_EQ(handoff.getCandidateCount(), 1u);
  ProgramDataFailure rangeFailure;
  auto range = ProgramDataRange::create(
      {ProgramResourceRole::Parameter, 1}, "f32", {4}, {4},
      wafer::frontend::ProgramDistributionKind::Replicated,
      ProgramDataRangeOrigin::OriginalSource, std::vector<int64_t>{0},
      std::vector<int64_t>{4}, std::vector<int64_t>{1}, *sourceId,
      handoff.getSource(*sourceId), &rangeFailure);
  ASSERT_TRUE(static_cast<bool>(range));
  ASSERT_FALSE(static_cast<bool>(handoff.addRange(std::move(*range))));

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  auto cardExecutable = wafer::compiler::detail::buildCardExecutable(
      context, *tensorProgram, std::move(program), *config,
      wafer::OptimizationConfig::none(), diagnostics, std::nullopt, handoff);
  if (!cardExecutable)
    FAIL() << diagnosticsText << llvm::toString(cardExecutable.takeError());
  ASSERT_EQ(cardExecutable->getTileExecutables().size(), 16u);
  EXPECT_EQ(cardExecutable->getProgramDataHandoff().getCandidateCount(), 0u)
      << "CardExecutable must retain only adopted live sources";

  auto inputTensor =
      ProgramTensor::create("f32", {4}, f32Bytes({9.0f, 9.0f, 9.0f, 9.0f}));
  ASSERT_TRUE(static_cast<bool>(inputTensor));
  ProgramGlobalInputBinding input{0, std::move(*inputTensor)};
  llvm::Expected<std::vector<ProgramTileInvocation>> invocations =
      prepareProgramInvocations(*cardExecutable, {input});
  ASSERT_TRUE(static_cast<bool>(invocations));
  EXPECT_EQ(invocations->size(), 16u);

  // One range materialization for the whole card, shared across all 16 Tiles.
  EXPECT_EQ(cardExecutable->getProgramDataHandoff()
                .getIOStatistics()
                .rangeMaterializations,
            1u);
  const uint8_t *sharedStorage = nullptr;
  for (const ProgramTileInvocation &invocation : *invocations) {
    const ProgramTensor *weight = nullptr;
    for (const ProgramInputBinding &binding : invocation.inputs) {
      if (binding.role != ProgramResourceRole::Parameter)
        continue;
      EXPECT_EQ(binding.tensor.getDType(), "f32");
      EXPECT_EQ(binding.tensor.getShape(), llvm::ArrayRef<int64_t>({4}));
      EXPECT_EQ(binding.tensor.getBytes(), llvm::ArrayRef<uint8_t>(payload));
      weight = &binding.tensor;
      break;
    }
    ASSERT_NE(weight, nullptr) << "every Tile must bind the parameter";
    if (!sharedStorage)
      sharedStorage = weight->getBytes().data();
    EXPECT_EQ(weight->getBytes().data(), sharedStorage)
        << "Tile parameter views must share one materialization";
  }
}

} // namespace
