//===- WaferRunBoardIOTest.cpp - wafer-run board file tests -------------===//

#include "WaferRunBoardIO.h"

#include "Wafer/Target/TargetFormat.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace {

using wafer::runtime::BoardRuntimeBinding;
using wafer::runtime::BoardRuntimeInvocationRequest;
using wafer::runtime::BoardRuntimeOutput;
using wafer::runtime::PackageAccessMode;
using wafer::runtime::PackageManifest;
using wafer::runtime::PackageResourceRecord;
using wafer::runtime::PackageResourceRole;
using wafer::runtime::ResourceId;
using wafer::runtime::cli::BoardInvocationFilePlan;
using wafer::runtime::cli::ResourceFile;

class WaferRunBoardIOTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(
        llvm::sys::fs::createUniqueDirectory("wafer-run-board-io-test", root));
  }

  void TearDown() override { llvm::sys::fs::remove_directories(root); }

  std::string path(llvm::StringRef name) const {
    llvm::SmallString<256> result(root);
    llvm::sys::path::append(result, name);
    return result.str().str();
  }

  void writeBytes(llvm::StringRef file, llvm::ArrayRef<uint8_t> bytes) {
    std::error_code error;
    llvm::raw_fd_ostream output(file, error, llvm::sys::fs::OF_None);
    ASSERT_FALSE(error);
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    output.close();
    ASSERT_FALSE(output.has_error());
  }

  std::vector<uint8_t> readBytes(llvm::StringRef file) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(file, /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false);
    EXPECT_TRUE(static_cast<bool>(buffer));
    if (!buffer)
      return {};
    llvm::StringRef bytes = (*buffer)->getBuffer();
    return {reinterpret_cast<const uint8_t *>(bytes.data()),
            reinterpret_cast<const uint8_t *>(bytes.data()) + bytes.size()};
  }

  PackageResourceRecord resource(uint64_t id, PackageResourceRole role,
                                 PackageAccessMode access) const {
    return {ResourceId(id), 0, role, 0,      "resource",
            {"u8", {4}},    4, 1,    access, true};
  }

  PackageManifest manifest(std::vector<PackageResourceRecord> resources) const {
    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    PackageManifest manifest(
        target.id, target.targetIdentity, target.kernelRuntimeABI,
        wafer::TargetLaunchABIId::perRankPointerBlockV1(), target.moduleFormat);
    manifest.rankCount = 1;
    manifest.resources = std::move(resources);
    return manifest;
  }

  llvm::Expected<BoardInvocationFilePlan>
  prepare(const PackageManifest &manifest,
          llvm::ArrayRef<ResourceFile> resources,
          llvm::ArrayRef<ResourceFile> expected,
          llvm::ArrayRef<ResourceFile> outputs) const {
    return wafer::runtime::cli::prepareBoardInvocationFiles(
        manifest, BoardRuntimeInvocationRequest{}, resources, expected,
        outputs);
  }

  const BoardRuntimeBinding &binding(const BoardInvocationFilePlan &plan,
                                     uint64_t resourceId) const {
    auto found = llvm::find_if(
        plan.request.bindings, [&](const BoardRuntimeBinding &candidate) {
          return candidate.resource == ResourceId(resourceId);
        });
    EXPECT_NE(found, plan.request.bindings.end());
    return *found;
  }

  llvm::SmallString<256> root;
};

TEST_F(WaferRunBoardIOTest, CaptureOnlyPublishesExactRawBytes) {
  PackageManifest package = manifest(
      {resource(0, PackageResourceRole::UserInput, PackageAccessMode::ReadOnly),
       resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  const std::vector<uint8_t> input = {1, 2, 3, 4};
  const std::vector<uint8_t> actual = {9, 8, 7, 6};
  const std::string inputPath = path("input.raw");
  const std::string outputPath = path("output.raw");
  writeBytes(inputPath, input);

  llvm::Expected<BoardInvocationFilePlan> plan =
      prepare(package, {{0, inputPath}}, {}, {{1, outputPath}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  EXPECT_EQ(binding(*plan, 0).bytes, input);
  EXPECT_EQ(binding(*plan, 1).bytes,
            (std::vector<uint8_t>{0xa5, 0xa5, 0xa5, 0xa5}));

  ASSERT_FALSE(wafer::runtime::cli::validateAndPublishBoardOutputs(
      {{ResourceId(1), actual}}, *plan));
  EXPECT_EQ(readBytes(outputPath), actual);
}

TEST_F(WaferRunBoardIOTest,
       SemanticRemapReusesPreparedBytesAcrossDifferentResourceIds) {
  PackageResourceRecord sourceInput =
      resource(10, PackageResourceRole::UserInput, PackageAccessMode::ReadOnly);
  sourceInput.name = "input";
  PackageResourceRecord sourceOutput =
      resource(11, PackageResourceRole::Output, PackageAccessMode::WriteOnly);
  sourceOutput.name = "output";
  PackageManifest source = manifest({sourceInput, sourceOutput});

  PackageResourceRecord internal = resource(20, PackageResourceRole::Workspace,
                                            PackageAccessMode::ReadWrite);
  internal.hostVisible = false;
  internal.name = "internal";
  PackageResourceRecord targetInput = sourceInput;
  targetInput.id = ResourceId(21);
  targetInput.name = "renamed_input";
  PackageResourceRecord targetOutput = sourceOutput;
  targetOutput.id = ResourceId(22);
  targetOutput.name = "renamed_output";
  PackageManifest target = manifest({internal, targetInput, targetOutput});

  const std::vector<uint8_t> input = {1, 2, 3, 4};
  const std::vector<uint8_t> expected = {9, 8, 7, 6};
  const std::string inputPath = path("input.raw");
  const std::string expectedPath = path("expected.raw");
  const std::string outputPath = path("output.raw");
  writeBytes(inputPath, input);
  writeBytes(expectedPath, expected);
  llvm::Expected<BoardInvocationFilePlan> sourcePlan = prepare(
      source, {{10, inputPath}}, {{11, expectedPath}}, {{11, outputPath}});
  ASSERT_TRUE(static_cast<bool>(sourcePlan))
      << llvm::toString(sourcePlan.takeError());

  llvm::Expected<BoardInvocationFilePlan> targetPlan =
      wafer::runtime::cli::remapBoardInvocationFilePlan(*sourcePlan, source,
                                                        target);
  ASSERT_TRUE(static_cast<bool>(targetPlan))
      << llvm::toString(targetPlan.takeError());
  EXPECT_EQ(binding(*targetPlan, 21).bytes, input);
  EXPECT_EQ(binding(*targetPlan, 22).bytes,
            (std::vector<uint8_t>{0xf6, 0xf7, 0xf8, 0xf9}));
  EXPECT_TRUE(targetPlan->expectedBytes.contains(22));
  EXPECT_TRUE(targetPlan->outputPaths.contains(22));
  EXPECT_TRUE(targetPlan->writableResourceBytes.contains(22));
  EXPECT_FALSE(targetPlan->expectedBytes.contains(11));
  EXPECT_FALSE(targetPlan->outputPaths.contains(11));
  EXPECT_FALSE(targetPlan->writableResourceBytes.contains(11));

  ASSERT_FALSE(wafer::runtime::cli::validateBoardOutputs(
      {{ResourceId(22), expected}}, *targetPlan));
  EXPECT_FALSE(llvm::sys::fs::exists(outputPath));
  ASSERT_FALSE(wafer::runtime::cli::validateAndPublishBoardOutputs(
      {{ResourceId(22), expected}}, *targetPlan));
  EXPECT_EQ(readBytes(outputPath), expected);
}

TEST_F(WaferRunBoardIOTest, SemanticRemapRejectsContractDrift) {
  PackageResourceRecord sourceInput =
      resource(10, PackageResourceRole::UserInput, PackageAccessMode::ReadOnly);
  sourceInput.name = "input";
  PackageManifest source = manifest({sourceInput});
  PackageResourceRecord targetInput = sourceInput;
  targetInput.id = ResourceId(20);
  targetInput.alignment = 2;
  PackageManifest target = manifest({targetInput});

  const std::string inputPath = path("input.raw");
  writeBytes(inputPath, {1, 2, 3, 4});
  llvm::Expected<BoardInvocationFilePlan> sourcePlan =
      prepare(source, {{10, inputPath}}, {}, {});
  ASSERT_TRUE(static_cast<bool>(sourcePlan))
      << llvm::toString(sourcePlan.takeError());
  llvm::Expected<BoardInvocationFilePlan> targetPlan =
      wafer::runtime::cli::remapBoardInvocationFilePlan(*sourcePlan, source,
                                                        target);
  ASSERT_FALSE(static_cast<bool>(targetPlan));
  EXPECT_NE(llvm::toString(targetPlan.takeError())
                .find("host-visible resource contract differs"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, DuplicateOutputResourceIsRejected) {
  PackageManifest package = manifest(
      {resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  llvm::Expected<BoardInvocationFilePlan> plan = prepare(
      package, {}, {}, {{1, path("first.raw")}, {1, path("second.raw")}});
  ASSERT_FALSE(static_cast<bool>(plan));
  EXPECT_NE(llvm::toString(plan.takeError())
                .find("duplicate ResourceId for --output"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, MultipleResourcesCannotShareAnOutputPath) {
  PackageManifest package = manifest(
      {resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly),
       resource(2, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  const std::string outputPath = path("output.raw");
  llvm::Expected<BoardInvocationFilePlan> plan =
      prepare(package, {}, {}, {{1, outputPath}, {2, outputPath}});
  ASSERT_FALSE(static_cast<bool>(plan));
  EXPECT_NE(llvm::toString(plan.takeError()).find("use the same raw file path"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, LexicalOutputAliasesAreRejected) {
  PackageManifest package = manifest(
      {resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly),
       resource(2, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  const std::string outputPath = path("output.raw");
  const std::string aliasedOutputPath = root.str().str() + "/./output.raw";
  llvm::Expected<BoardInvocationFilePlan> plan =
      prepare(package, {}, {}, {{1, outputPath}, {2, aliasedOutputPath}});
  ASSERT_FALSE(static_cast<bool>(plan));
  EXPECT_NE(llvm::toString(plan.takeError()).find("use the same raw file path"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, ParentSymlinkOutputAliasesAreRejected) {
  PackageManifest package = manifest(
      {resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly),
       resource(2, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  const std::string realParent = path("real-parent");
  const std::string aliasParent = path("alias-parent");
  ASSERT_FALSE(llvm::sys::fs::create_directory(realParent));
  ASSERT_EQ(::symlink(realParent.c_str(), aliasParent.c_str()), 0);

  llvm::SmallString<256> realOutput(realParent);
  llvm::sys::path::append(realOutput, "output.raw");
  llvm::SmallString<256> aliasedOutput(aliasParent);
  llvm::sys::path::append(aliasedOutput, "output.raw");
  llvm::Expected<BoardInvocationFilePlan> plan =
      prepare(package, {}, {},
              {{1, realOutput.str().str()}, {2, aliasedOutput.str().str()}});
  ASSERT_FALSE(static_cast<bool>(plan));
  EXPECT_NE(llvm::toString(plan.takeError()).find("use the same raw file path"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, DotDotAfterParentSymlinkUsesFilesystemResolution) {
  PackageManifest package = manifest(
      {resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly),
       resource(2, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  const std::string realParent = path("real-parent");
  const std::string nestedParent = path("real-parent/nested");
  const std::string aliasParent = path("alias-parent");
  ASSERT_FALSE(llvm::sys::fs::create_directories(nestedParent));
  ASSERT_EQ(::symlink(nestedParent.c_str(), aliasParent.c_str()), 0);

  llvm::SmallString<256> directOutput(realParent);
  llvm::sys::path::append(directOutput, "output.raw");
  llvm::SmallString<256> aliasedOutput(aliasParent);
  llvm::sys::path::append(aliasedOutput, "..", "output.raw");
  llvm::Expected<BoardInvocationFilePlan> plan =
      prepare(package, {}, {},
              {{1, directOutput.str().str()}, {2, aliasedOutput.str().str()}});
  ASSERT_FALSE(static_cast<bool>(plan));
  EXPECT_NE(llvm::toString(plan.takeError()).find("use the same raw file path"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, ExistingDirectoryOutputIsRejectedBeforeExecution) {
  PackageManifest package = manifest(
      {resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  const std::string directoryPath = path("output-directory");
  ASSERT_FALSE(llvm::sys::fs::create_directory(directoryPath));

  llvm::Expected<BoardInvocationFilePlan> plan =
      prepare(package, {}, {}, {{1, directoryPath}});
  ASSERT_FALSE(static_cast<bool>(plan));
  EXPECT_NE(llvm::toString(plan.takeError())
                .find("raw output path names an existing directory"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, UnknownAndReadOnlyOutputResourcesAreRejected) {
  PackageManifest package = manifest(
      {resource(0, PackageResourceRole::UserInput, PackageAccessMode::ReadOnly),
       resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  const std::string inputPath = path("input.raw");
  writeBytes(inputPath, {1, 2, 3, 4});

  for (uint64_t invalidId : {UINT64_C(0), UINT64_C(99)}) {
    SCOPED_TRACE(invalidId);
    llvm::Expected<BoardInvocationFilePlan> plan =
        prepare(package, {{0, inputPath}}, {},
                {{1, path("output.raw")}, {invalidId, path("invalid.raw")}});
    ASSERT_FALSE(static_cast<bool>(plan));
    EXPECT_NE(llvm::toString(plan.takeError())
                  .find("ResourceIds outside the package"),
              std::string::npos);
  }
}

TEST_F(WaferRunBoardIOTest, ReadableAndWriteOnlyBindingsStayDistinct) {
  PackageManifest package = manifest(
      {resource(0, PackageResourceRole::UserInput, PackageAccessMode::ReadOnly),
       resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly),
       resource(2, PackageResourceRole::Workspace,
                PackageAccessMode::ReadWrite)});
  const std::string inputPath = path("input.raw");
  const std::string readWritePath = path("read-write.raw");
  writeBytes(inputPath, {1, 2, 3, 4});
  writeBytes(readWritePath, {5, 6, 7, 8});

  llvm::Expected<BoardInvocationFilePlan> plan = prepare(
      package, {{0, inputPath}, {2, readWritePath}}, {},
      {{1, path("write-only-output.raw")}, {2, path("read-write-output.raw")}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  EXPECT_EQ(binding(*plan, 0).bytes, (std::vector<uint8_t>{1, 2, 3, 4}));
  EXPECT_EQ(binding(*plan, 1).bytes,
            (std::vector<uint8_t>{0xa5, 0xa5, 0xa5, 0xa5}));
  EXPECT_EQ(binding(*plan, 2).bytes, (std::vector<uint8_t>{5, 6, 7, 8}));

  llvm::Expected<BoardInvocationFilePlan> initializedWriteOnly = prepare(
      package, {{0, inputPath}, {1, inputPath}, {2, readWritePath}}, {},
      {{1, path("write-only-output.raw")}, {2, path("read-write-output.raw")}});
  ASSERT_FALSE(static_cast<bool>(initializedWriteOnly));
  EXPECT_NE(llvm::toString(initializedWriteOnly.takeError())
                .find("must not initialize write-only ResourceId 1"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, ExpectedAndOutputCompareBeforeCapture) {
  PackageManifest package = manifest(
      {resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  const std::vector<uint8_t> expected = {2, 4, 6, 8};
  const std::string expectedPath = path("expected.raw");
  const std::string outputPath = path("output.raw");
  writeBytes(expectedPath, expected);
  writeBytes(outputPath, {7, 7, 7, 7});

  llvm::Expected<BoardInvocationFilePlan> plan =
      prepare(package, {}, {{1, expectedPath}}, {{1, outputPath}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  EXPECT_EQ(binding(*plan, 1).bytes,
            (std::vector<uint8_t>{
                static_cast<uint8_t>(~2), static_cast<uint8_t>(~4),
                static_cast<uint8_t>(~6), static_cast<uint8_t>(~8)}));

  llvm::Error mismatch = wafer::runtime::cli::validateAndPublishBoardOutputs(
      {{ResourceId(1), {2, 4, 0, 8}}}, *plan);
  ASSERT_TRUE(static_cast<bool>(mismatch));
  EXPECT_NE(llvm::toString(std::move(mismatch)).find("at byte 2"),
            std::string::npos);
  EXPECT_EQ(readBytes(outputPath), (std::vector<uint8_t>{7, 7, 7, 7}));

  ASSERT_FALSE(wafer::runtime::cli::validateAndPublishBoardOutputs(
      {{ResourceId(1), expected}}, *plan));
  EXPECT_EQ(readBytes(outputPath), expected);

  llvm::Expected<BoardInvocationFilePlan> compareOnly =
      prepare(package, {}, {{1, expectedPath}}, {});
  ASSERT_TRUE(static_cast<bool>(compareOnly))
      << llvm::toString(compareOnly.takeError());
  ASSERT_FALSE(wafer::runtime::cli::validateAndPublishBoardOutputs(
      {{ResourceId(1), expected}}, *compareOnly));
}

TEST_F(WaferRunBoardIOTest, DuplicateAndUnexpectedProviderOutputsAreRejected) {
  PackageManifest package = manifest(
      {resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  const std::string outputPath = path("output.raw");
  llvm::Expected<BoardInvocationFilePlan> plan =
      prepare(package, {}, {}, {{1, outputPath}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());

  for (std::vector<BoardRuntimeOutput> outputs :
       {std::vector<BoardRuntimeOutput>{{ResourceId(1), {1, 2, 3, 4}},
                                        {ResourceId(1), {1, 2, 3, 4}}},
        std::vector<BoardRuntimeOutput>{{ResourceId(99), {1, 2, 3, 4}}}}) {
    llvm::Error error =
        wafer::runtime::cli::validateAndPublishBoardOutputs(outputs, *plan);
    ASSERT_TRUE(static_cast<bool>(error));
    EXPECT_NE(llvm::toString(std::move(error))
                  .find("unexpected or duplicate writable ResourceId"),
              std::string::npos);
    EXPECT_FALSE(llvm::sys::fs::exists(outputPath));
  }
}

TEST_F(WaferRunBoardIOTest, MissingAndWrongSizedProviderOutputsAreRejected) {
  PackageManifest package = manifest(
      {resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  const std::string outputPath = path("output.raw");
  llvm::Expected<BoardInvocationFilePlan> plan =
      prepare(package, {}, {}, {{1, outputPath}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());

  llvm::Error missing =
      wafer::runtime::cli::validateAndPublishBoardOutputs({}, *plan);
  ASSERT_TRUE(static_cast<bool>(missing));
  EXPECT_NE(llvm::toString(std::move(missing))
                .find("did not return all-and-only writable resources"),
            std::string::npos);

  llvm::Error wrongSize = wafer::runtime::cli::validateAndPublishBoardOutputs(
      {{ResourceId(1), {1, 2, 3}}}, *plan);
  ASSERT_TRUE(static_cast<bool>(wrongSize));
  EXPECT_NE(llvm::toString(std::move(wrongSize))
                .find("wrong-sized writable ResourceId 1"),
            std::string::npos);
  EXPECT_FALSE(llvm::sys::fs::exists(outputPath));
}

TEST_F(WaferRunBoardIOTest, StagingFailurePreservesEveryDestination) {
  PackageManifest package = manifest(
      {resource(1, PackageResourceRole::Output, PackageAccessMode::WriteOnly),
       resource(2, PackageResourceRole::Output, PackageAccessMode::WriteOnly)});
  const std::string firstPath = path("first.raw");
  const std::string invalidPath = path(std::string(300, 'x'));
  const std::vector<uint8_t> original = {4, 3, 2, 1};
  writeBytes(firstPath, original);

  llvm::Expected<BoardInvocationFilePlan> plan =
      prepare(package, {}, {}, {{1, firstPath}, {2, invalidPath}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  llvm::Error error = wafer::runtime::cli::validateAndPublishBoardOutputs(
      {{ResourceId(1), {1, 1, 1, 1}}, {ResourceId(2), {2, 2, 2, 2}}}, *plan);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(
      llvm::toString(std::move(error)).find("failed to stage raw output file"),
      std::string::npos);
  EXPECT_EQ(readBytes(firstPath), original);

  std::error_code iterationError;
  for (llvm::sys::fs::directory_iterator iterator(root, iterationError), end;
       iterator != end && !iterationError; iterator.increment(iterationError))
    EXPECT_EQ(llvm::sys::path::filename(iterator->path()), "first.raw");
  EXPECT_FALSE(iterationError);
}

} // namespace
