//===- WaferRunBoardIOTest.cpp - Board invocation file IO ----------------===//

#include "WaferRunBoardIO.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Package/Manifest/PackageManifest.h"
#include "Wafer/Runtime/Board/BoardRuntime.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::runtime;
using namespace wafer::runtime::cli;
using wafer::runtime::LaunchSlotId;

/// Canonical 16-Tile grid package with the given external input/output ports
/// and no program tensors.
class WaferRunBoardIOTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(
        llvm::sys::fs::createUniqueDirectory("wafer-run-board-io-test", root));
  }

  void TearDown() override { llvm::sys::fs::remove_directories(root); }

  static ExternalPortRecord port(uint64_t id, int64_t roleIndex,
                                 llvm::StringRef dtype = "u8",
                                 std::vector<int64_t> shape = {4},
                                 uint64_t bytes = 4, uint64_t alignment = 1) {
    auto programType = llvm::cantFail(wafer::parseProgramElementType(dtype));
    auto targetType = llvm::cantFail(wafer::parseLogicalFormat(dtype));
    return ExternalPortRecord{PortId(id), roleIndex,  programType,
                              shape,      targetType, PackageMemLayout::Tensor,
                              shape,      bytes,      alignment};
  }

  PackageManifest manifest(std::vector<ExternalPortRecord> inputs,
                           std::vector<ExternalPortRecord> outputs) const {
    PackageManifest manifest(wafer::kCurrentTargetIdentity,
                             wafer::kCurrentKernelRuntimeABI, makeGridLaunch(),
                             wafer::kCurrentTargetModuleFormat);
    manifest.program = ProgramId(0);
    manifest.cardCount = 1;
    manifest.tileCount = 16;
    manifest.programData = {"data/program-data.bin", 0, 1,
                            emptyProgramDataDigest()};
    manifest.inputs = std::move(inputs);
    manifest.outputs = std::move(outputs);
    manifest.modules = {
        {ModuleId(0),
         "modules/tile_00000.so",
         moduleDigest(),
         wafer::kCurrentTargetModuleFormat.str(),
         {{PackageModuleExportRole::Main, "main"}}},
    };
    for (int64_t launchSlot = 0; launchSlot < 16; ++launchSlot) {
      std::vector<TileEntryArgumentRecord> arguments;
      for (const ExternalPortRecord &input : manifest.inputs)
        arguments.push_back({static_cast<uint64_t>(arguments.size()),
                             ExternalInputArgument{input.id},
                             PackageAccessMode::ReadOnly});
      for (const ExternalPortRecord &output : manifest.outputs)
        arguments.push_back({static_cast<uint64_t>(arguments.size()),
                             ExternalOutputArgument{output.id},
                             PackageAccessMode::WriteOnly});
      manifest.entries.push_back(
          {EntryId(launchSlot), wafer::CardId(0), wafer::TileId(launchSlot),
           LaunchSlotId(launchSlot), ModuleId(0), std::move(arguments),
           PackageEntryCompletionKind::ReturnAfterLocalDrain,
           NoTransportRequirements{}});
    }
    return manifest;
  }

  llvm::SmallString<256> writeFile(llvm::StringRef relativePath,
                                   llvm::ArrayRef<uint8_t> bytes) const {
    llvm::SmallString<256> path(root);
    llvm::sys::path::append(path, relativePath);
    std::error_code error;
    llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
    EXPECT_FALSE(error);
    if (error)
      return path;
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    output.close();
    EXPECT_FALSE(output.has_error());
    return path;
  }

  llvm::Expected<std::vector<uint8_t>>
  readFileBytes(llvm::StringRef path) const {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false);
    if (!buffer)
      return llvm::createStringError(buffer.getError(),
                                     "failed to read file: " + path);
    return std::vector<uint8_t>(
        reinterpret_cast<const uint8_t *>((*buffer)->getBuffer().data()),
        reinterpret_cast<const uint8_t *>((*buffer)->getBuffer().data()) +
            (*buffer)->getBuffer().size());
  }

  static std::vector<uint8_t> bytes(uint8_t value, size_t count = 4) {
    return std::vector<uint8_t>(count, value);
  }

  static wafer::RuntimeLaunchContract makeGridLaunch() {
    return llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
        wafer::KernelLaunchForm::Grid,
        wafer::KernelEntryABI::TileMajorPointerTable,
        {wafer::RuntimeLaunchPhaseRole::Main}));
  }

  static std::string moduleDigest() {
    llvm::SHA256 hasher;
    hasher.update(llvm::StringRef("\x7f"
                                  "ELFtyped-package-test"));
    return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
  }

  static std::string emptyProgramDataDigest() {
    return "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b78"
           "52b855";
  }

  llvm::SmallString<256> root;
};

TEST_F(WaferRunBoardIOTest, CaptureOnlyWritesExactRawBytes) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  llvm::SmallString<256> capturePath = writeFile("capture.raw", {});
  llvm::Expected<BoardInvocationFilePlan> plan = prepareBoardInvocationFiles(
      package, BoardRuntimeInvocationRequest{},
      /*inputFiles=*/{{0, input.str().str()}},
      /*expectedFiles=*/{},
      /*outputFiles=*/{{1, capturePath.str().str()}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  EXPECT_EQ(plan->request.bindings.size(), 1u);
  EXPECT_EQ(plan->request.bindings.front().port, PortId(0));
  EXPECT_EQ(plan->request.bindings.front().bytes, bytes(0xAB));
  EXPECT_EQ(plan->outputBytes.lookup(1), 4u);
  EXPECT_TRUE(plan->expectedBytes.empty());

  std::vector<BoardRuntimeOutput> outputs = {
      {PortId(1), bytes(0x5A)},
  };
  EXPECT_FALSE(validateBoardOutputs(outputs, *plan));
  llvm::Error error = validateAndWriteBoardOutputs(outputs, *plan);
  EXPECT_FALSE(error) << llvm::toString(std::move(error));
  llvm::Expected<std::vector<uint8_t>> written =
      readFileBytes(capturePath.str());
  ASSERT_TRUE(static_cast<bool>(written))
      << llvm::toString(written.takeError());
  EXPECT_EQ(*written, bytes(0x5A));
}

TEST_F(WaferRunBoardIOTest,
       SemanticRemapReusesPreparedBytesAcrossDifferentPortIds) {
  // Source ports use dense ids 0/1; the target package re-ids them 20/21.
  // The stable identity is the role index, so the prepared bytes carry over.
  // Plan keys index the source manifest's output vector, so the source
  // output domain must be dense 0..n-1.
  PackageManifest source = manifest({port(0, 0)}, {port(0, 0), port(1, 1)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0x11));
  llvm::SmallString<256> expected = writeFile("expected.raw", bytes(0x22));
  llvm::SmallString<256> captureZero = writeFile("capture-0.raw", {});
  llvm::SmallString<256> captureOne = writeFile("capture-1.raw", {});
  llvm::Expected<BoardInvocationFilePlan> sourcePlan =
      prepareBoardInvocationFiles(
          source, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}},
          /*expectedFiles=*/{{1, expected.str().str()}},
          /*outputFiles=*/
          {{0, captureZero.str().str()}, {1, captureOne.str().str()}});
  ASSERT_TRUE(static_cast<bool>(sourcePlan))
      << llvm::toString(sourcePlan.takeError());

  PackageManifest target = manifest({port(20, 0)}, {port(20, 0), port(21, 1)});
  llvm::Expected<BoardInvocationFilePlan> targetPlan =
      remapBoardInvocationFilePlan(*sourcePlan, source, target);
  ASSERT_TRUE(static_cast<bool>(targetPlan))
      << llvm::toString(targetPlan.takeError());
  ASSERT_EQ(targetPlan->request.bindings.size(), 1u);
  EXPECT_EQ(targetPlan->request.bindings.front().port, PortId(20));
  EXPECT_EQ(targetPlan->request.bindings.front().bytes, bytes(0x11));
  EXPECT_EQ(targetPlan->expectedBytes.lookup(21), bytes(0x22));
  EXPECT_EQ(targetPlan->expectedComparisons.lookup(21),
            BoardOutputComparisonKind::Exact);
  EXPECT_EQ(targetPlan->outputPaths.lookup(20), captureZero.str().str());
  EXPECT_EQ(targetPlan->outputPaths.lookup(21), captureOne.str().str());
  EXPECT_EQ(targetPlan->outputBytes.lookup(20), 4u);
  EXPECT_EQ(targetPlan->outputBytes.lookup(21), 4u);

  // The remapped plan validates and writes against the target ports.
  std::vector<BoardRuntimeOutput> outputs = {
      {PortId(20), bytes(0x22)},
      {PortId(21), bytes(0x22)},
  };
  llvm::Error error = validateAndWriteBoardOutputs(outputs, *targetPlan);
  EXPECT_FALSE(error) << llvm::toString(std::move(error));
  llvm::Expected<std::vector<uint8_t>> written =
      readFileBytes(captureOne.str());
  ASSERT_TRUE(static_cast<bool>(written))
      << llvm::toString(written.takeError());
  EXPECT_EQ(*written, bytes(0x22));
}

TEST_F(WaferRunBoardIOTest, SemanticRemapRejectsPortDomainDrift) {
  PackageManifest source = manifest({port(0, 0)}, {port(0, 0), port(1, 1)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0x11));
  llvm::SmallString<256> expected = writeFile("expected.raw", bytes(0x22));
  llvm::Expected<BoardInvocationFilePlan> sourcePlan =
      prepareBoardInvocationFiles(
          source, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}},
          /*expectedFiles=*/{{1, expected.str().str()}},
          /*outputFiles=*/{{0, "capture-0.raw"}, {1, "capture-1.raw"}});
  ASSERT_TRUE(static_cast<bool>(sourcePlan))
      << llvm::toString(sourcePlan.takeError());

  PackageManifest driftedInput =
      manifest({port(10, 1)}, {port(11, 0), port(12, 1)});
  llvm::Expected<BoardInvocationFilePlan> rejected =
      remapBoardInvocationFilePlan(*sourcePlan, source, driftedInput);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("input port domain differs"),
      std::string::npos);

  PackageManifest driftedOutput =
      manifest({port(10, 0)}, {port(11, 0), port(12, 2)});
  rejected = remapBoardInvocationFilePlan(*sourcePlan, source, driftedOutput);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("output port domain differs"),
      std::string::npos);

  PackageManifest extraOutput = manifest({port(10, 0)}, {port(11, 0)});
  rejected = remapBoardInvocationFilePlan(*sourcePlan, source, extraOutput);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("external port domain differs"),
      std::string::npos);

  PackageManifest emptyBytes =
      manifest({port(10, 0, "u8", {0}, 0, 1)}, {port(11, 0), port(12, 1)});
  rejected = remapBoardInvocationFilePlan(*sourcePlan, source, emptyBytes);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("input port domain differs"),
      std::string::npos);
}

TEST_F(WaferRunBoardIOTest, SemanticRemapRejectsExistingProfilerRecords) {
  PackageManifest source = manifest({port(0, 0)}, {port(1, 0)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0x11));
  llvm::SmallString<256> expected = writeFile("expected.raw", bytes(0x22));
  llvm::Expected<BoardInvocationFilePlan> sourcePlan =
      prepareBoardInvocationFiles(source, BoardRuntimeInvocationRequest{},
                                  /*inputFiles=*/{{0, input.str().str()}},
                                  /*expectedFiles=*/{{1, expected.str().str()}},
                                  /*outputFiles=*/{});
  ASSERT_TRUE(static_cast<bool>(sourcePlan))
      << llvm::toString(sourcePlan.takeError());
  sourcePlan->request.profilerRecordBytes =
      std::vector<std::vector<uint8_t>>(16, std::vector<uint8_t>(64));

  PackageManifest target = manifest({port(20, 0)}, {port(21, 0)});
  llvm::Expected<BoardInvocationFilePlan> rejected =
      remapBoardInvocationFilePlan(*sourcePlan, source, target);
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("already has profiler records"),
      std::string::npos);
}

TEST_F(WaferRunBoardIOTest, DuplicateOutputPortIsRejected) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  llvm::Expected<BoardInvocationFilePlan> rejected =
      prepareBoardInvocationFiles(
          package, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}},
          /*expectedFiles=*/{},
          /*outputFiles=*/{{1, "first.raw"}, {1, "second.raw"}});
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError()).find("duplicate port id"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, MultiplePortsCannotShareAnOutputPath) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0), port(2, 1)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  llvm::Expected<BoardInvocationFilePlan> rejected =
      prepareBoardInvocationFiles(
          package, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}},
          /*expectedFiles=*/{},
          /*outputFiles=*/{{1, "output.raw"}, {2, "output.raw"}});
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("use the same raw file path"),
      std::string::npos);
}

TEST_F(WaferRunBoardIOTest, LexicalOutputAliasesAreRejected) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0), port(2, 1)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  std::string lexicallyAliased = (root.str() + "/./output.raw").str();
  std::string direct = (root.str() + "/output.raw").str();
  llvm::Expected<BoardInvocationFilePlan> rejected =
      prepareBoardInvocationFiles(
          package, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}},
          /*expectedFiles=*/{},
          /*outputFiles=*/{{1, lexicallyAliased}, {2, direct}});
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("use the same raw file path"),
      std::string::npos);
}

TEST_F(WaferRunBoardIOTest, ParentSymlinkOutputAliasesAreRejected) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0), port(2, 1)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  ASSERT_FALSE(llvm::sys::fs::create_directory(root + "/real"));
  ASSERT_FALSE(llvm::sys::fs::create_link(root + "/real", root + "/link"));
  std::string aliased = (root.str() + "/link/output.raw").str();
  std::string direct = (root.str() + "/real/output.raw").str();
  llvm::Expected<BoardInvocationFilePlan> rejected =
      prepareBoardInvocationFiles(package, BoardRuntimeInvocationRequest{},
                                  /*inputFiles=*/{{0, input.str().str()}},
                                  /*expectedFiles=*/{},
                                  /*outputFiles=*/{{1, aliased}, {2, direct}});
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("use the same raw file path"),
      std::string::npos);
}

TEST_F(WaferRunBoardIOTest, DotDotAfterParentSymlinkUsesFilesystemResolution) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0), port(2, 1)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  ASSERT_FALSE(llvm::sys::fs::create_directory(root + "/real"));
  ASSERT_FALSE(llvm::sys::fs::create_link(root + "/real", root + "/link"));
  std::string throughDotDot = (root.str() + "/link/../output.raw").str();
  std::string direct = (root.str() + "/output.raw").str();
  llvm::Expected<BoardInvocationFilePlan> rejected =
      prepareBoardInvocationFiles(
          package, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}},
          /*expectedFiles=*/{},
          /*outputFiles=*/{{1, throughDotDot}, {2, direct}});
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("use the same raw file path"),
      std::string::npos);
}

TEST_F(WaferRunBoardIOTest, ExistingDirectoryOutputIsRejectedBeforeExecution) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  ASSERT_FALSE(llvm::sys::fs::create_directory(root + "/output.raw"));
  llvm::Expected<BoardInvocationFilePlan> rejected =
      prepareBoardInvocationFiles(
          package, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}},
          /*expectedFiles=*/{},
          /*outputFiles=*/{{1, (root.str() + "/output.raw").str()}});
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError())
                .find("names an existing "
                      "directory"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, UnknownAndReadOnlyOutputPortsAreRejected) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  llvm::Expected<BoardInvocationFilePlan> rejected =
      prepareBoardInvocationFiles(
          package, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}},
          /*expectedFiles=*/{},
          /*outputFiles=*/
          {{1, "output.raw"}, {0, "input.raw"}, {99, "other.raw"}});
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("port ids outside the package"),
      std::string::npos);
}

TEST_F(WaferRunBoardIOTest, ReadableInputsAndWriteOnlyOutputsStayDistinct) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  // A raw file supplied for the write-only output port is not a readable
  // input and cannot be bound: every declared input is still consumed and
  // the leftover file is rejected.
  llvm::Expected<BoardInvocationFilePlan> rejected =
      prepareBoardInvocationFiles(
          package, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}, {1, "output.raw"}},
          /*expectedFiles=*/{},
          /*outputFiles=*/{{1, "provider.raw"}});
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("port ids outside the package"),
      std::string::npos);
}

TEST_F(WaferRunBoardIOTest, ExpectedAndOutputCompareBeforeCapture) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  std::vector<uint8_t> expectedBytes = {0, 1, 2, 3};
  llvm::SmallString<256> expectedFile =
      writeFile("expected.raw", expectedBytes);
  llvm::SmallString<256> captureFile = writeFile("capture.raw", {});
  llvm::Expected<BoardInvocationFilePlan> plan = prepareBoardInvocationFiles(
      package, BoardRuntimeInvocationRequest{},
      /*inputFiles=*/{{0, input.str().str()}},
      /*expectedFiles=*/{{1, expectedFile.str().str()}},
      /*outputFiles=*/{{1, captureFile.str().str()}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());

  std::vector<BoardRuntimeOutput> mismatching = {
      {PortId(1), {0, 1, 0, 3}},
  };
  llvm::Error error = validateAndWriteBoardOutputs(mismatching, *plan);
  ASSERT_TRUE(static_cast<bool>(error));
  std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find("differs for port 1 at byte 2"), std::string::npos);
  EXPECT_NE(message.find("expected=0x2"), std::string::npos);
  EXPECT_NE(message.find("actual=0x0"), std::string::npos);
  // The mismatching result never reaches the capture file.
  llvm::Expected<std::vector<uint8_t>> untouched =
      readFileBytes(captureFile.str());
  ASSERT_TRUE(static_cast<bool>(untouched))
      << llvm::toString(untouched.takeError());
  EXPECT_TRUE(untouched->empty());

  // The compare-only variant passes without writing anything.
  llvm::Expected<BoardInvocationFilePlan> compareOnly =
      prepareBoardInvocationFiles(
          package, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}},
          /*expectedFiles=*/{{1, expectedFile.str().str()}},
          /*outputFiles=*/{});
  ASSERT_TRUE(static_cast<bool>(compareOnly))
      << llvm::toString(compareOnly.takeError());
  std::vector<BoardRuntimeOutput> matching = {
      {PortId(1), expectedBytes},
  };
  error = validateAndWriteBoardOutputs(matching, *compareOnly);
  EXPECT_FALSE(error) << llvm::toString(std::move(error));
}

TEST_F(WaferRunBoardIOTest,
       RelaxedF16ComparisonAcceptsSignedZeroAndOneUlpOnly) {
  PackageManifest package =
      manifest({port(0, 0)}, {port(1, 0, "f16", {4}, 8, 2)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  // 1.0f16 = 0x3C00; +1 ULP = 0x3C01; signed zero is 0x8000.
  std::vector<uint8_t> expectedBytes = {0x00, 0x3C, 0x00, 0x3C,
                                        0x00, 0x80, 0x00, 0x3C};
  llvm::SmallString<256> expectedFile =
      writeFile("expected.raw", expectedBytes);
  llvm::Expected<BoardInvocationFilePlan> plan = prepareBoardInvocationFiles(
      package, BoardRuntimeInvocationRequest{},
      /*inputFiles=*/{{0, input.str().str()}},
      /*expectedFiles=*/{},
      /*outputFiles=*/{},
      /*relaxedF16ExpectedFiles=*/{{1, expectedFile.str().str()}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());
  EXPECT_EQ(plan->expectedComparisons.lookup(1),
            BoardOutputComparisonKind::RelaxedF16);

  // Signed zero matches its positive counterpart, and one ULP is accepted.
  std::vector<uint8_t> oneUlp = {0x01, 0x3C, 0x00, 0x3C,
                                 0x00, 0x00, 0x00, 0x3C};
  llvm::Error error = validateBoardOutputs({{PortId(1), oneUlp}}, *plan);
  EXPECT_FALSE(error) << llvm::toString(std::move(error));

  // Two ULPs exceed the maximum and are rejected.
  std::vector<uint8_t> tooFar = {0x02, 0x3C, 0x00, 0x3C,
                                 0x00, 0x00, 0x00, 0x3C};
  error = validateBoardOutputs({{PortId(1), tooFar}}, *plan);
  ASSERT_TRUE(static_cast<bool>(error));
  std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find("differs for port 1 at element 0"), std::string::npos);
  EXPECT_NE(message.find("ulp=2"), std::string::npos);
}

TEST_F(WaferRunBoardIOTest,
       RelaxedF16ComparisonRejectsNonfiniteAndNonF16Ports) {
  PackageManifest package =
      manifest({port(0, 0)}, {port(1, 0, "f16", {4}, 8, 2)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  std::vector<uint8_t> finite = {0x00, 0x3C, 0x00, 0x3C,
                                 0x00, 0x3C, 0x00, 0x3C};
  llvm::SmallString<256> expectedFile = writeFile("expected.raw", finite);
  llvm::Expected<BoardInvocationFilePlan> plan = prepareBoardInvocationFiles(
      package, BoardRuntimeInvocationRequest{},
      /*inputFiles=*/{{0, input.str().str()}},
      /*expectedFiles=*/{},
      /*outputFiles=*/{},
      /*relaxedF16ExpectedFiles=*/{{1, expectedFile.str().str()}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());

  // +Infinity = 0x7C00 in the actual result is not a finite f16.
  std::vector<uint8_t> infinite = {0x00, 0x7C, 0x00, 0x3C,
                                   0x00, 0x3C, 0x00, 0x3C};
  llvm::Error error = validateBoardOutputs({{PortId(1), infinite}}, *plan);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error))
                .find("contains NaN or infinity for "
                      "port 1 at element 0"),
            std::string::npos);

  // The relaxed policy is only valid for f16 output ports.
  PackageManifest u8Package = manifest({port(0, 0)}, {port(1, 0)});
  llvm::Expected<BoardInvocationFilePlan> rejected =
      prepareBoardInvocationFiles(
          u8Package, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}},
          /*expectedFiles=*/{},
          /*outputFiles=*/{},
          /*relaxedF16ExpectedFiles=*/{{1, expectedFile.str().str()}});
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(
      llvm::toString(rejected.takeError()).find("requires an f16 output port"),
      std::string::npos);
}

TEST_F(WaferRunBoardIOTest,
       ExpectedPortCannotSelectExactAndRelaxedPoliciesTogether) {
  PackageManifest package =
      manifest({port(0, 0)}, {port(1, 0, "f16", {4}, 8, 2)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  std::vector<uint8_t> finite = {0x00, 0x3C, 0x00, 0x3C,
                                 0x00, 0x3C, 0x00, 0x3C};
  llvm::SmallString<256> exactFile = writeFile("exact.raw", finite);
  llvm::SmallString<256> relaxedFile = writeFile("relaxed.raw", finite);
  llvm::Expected<BoardInvocationFilePlan> rejected =
      prepareBoardInvocationFiles(
          package, BoardRuntimeInvocationRequest{},
          /*inputFiles=*/{{0, input.str().str()}},
          /*expectedFiles=*/{{1, exactFile.str().str()}},
          /*outputFiles=*/{},
          /*relaxedF16ExpectedFiles=*/{{1, relaxedFile.str().str()}});
  ASSERT_FALSE(static_cast<bool>(rejected));
  EXPECT_NE(llvm::toString(rejected.takeError())
                .find("duplicate port id across --expected and "
                      "--expected-f16-relaxed"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, DuplicateAndUnexpectedProviderOutputsAreRejected) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  llvm::Expected<BoardInvocationFilePlan> plan =
      prepareBoardInvocationFiles(package, BoardRuntimeInvocationRequest{},
                                  /*inputFiles=*/{{0, input.str().str()}},
                                  /*expectedFiles=*/{},
                                  /*outputFiles=*/{{1, "provider.raw"}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());

  llvm::Error error = validateBoardOutputs(
      {{PortId(1), bytes(0x00)}, {PortId(9), bytes(0x00)}}, *plan);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error))
                .find("unexpected or duplicate output port 9"),
            std::string::npos);

  error = validateBoardOutputs(
      {{PortId(1), bytes(0x00)}, {PortId(1), bytes(0x00)}}, *plan);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error))
                .find("unexpected or duplicate output port 1"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, MissingAndWrongSizedProviderOutputsAreRejected) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0), port(2, 1)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  llvm::Expected<BoardInvocationFilePlan> plan = prepareBoardInvocationFiles(
      package, BoardRuntimeInvocationRequest{},
      /*inputFiles=*/{{0, input.str().str()}},
      /*expectedFiles=*/{},
      /*outputFiles=*/{{1, "provider-1.raw"}, {2, "provider-2.raw"}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());

  llvm::Error error = validateBoardOutputs({{PortId(1), bytes(0x00)}}, *plan);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("all-and-only output ports"),
            std::string::npos);

  error = validateBoardOutputs(
      {{PortId(1), bytes(0x00)}, {PortId(2), bytes(0x00, 3)}}, *plan);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("wrong-sized output port 2"),
            std::string::npos);
}

TEST_F(WaferRunBoardIOTest, StagingFailurePreservesEveryDestination) {
  PackageManifest package = manifest({port(0, 0)}, {port(1, 0), port(2, 1)});
  llvm::SmallString<256> input = writeFile("input.raw", bytes(0xAB));
  ASSERT_FALSE(llvm::sys::fs::create_directory(root + "/out"));
  std::string shortPath = (root.str() + "/out/short.raw").str();
  std::string longPath = (root.str() + "/out/" + std::string(300, 'x')).str();
  llvm::Expected<BoardInvocationFilePlan> plan = prepareBoardInvocationFiles(
      package, BoardRuntimeInvocationRequest{},
      /*inputFiles=*/{{0, input.str().str()}},
      /*expectedFiles=*/{},
      /*outputFiles=*/{{1, shortPath}, {2, longPath}});
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());

  std::vector<BoardRuntimeOutput> outputs = {
      {PortId(1), bytes(0x11)},
      {PortId(2), bytes(0x22)},
  };
  llvm::Error error = validateAndWriteBoardOutputs(outputs, *plan);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error))
                .find("failed to stage raw output "
                      "file"),
            std::string::npos);
  // Neither destination was replaced: no raw file exists in the output dir.
  std::error_code directoryError;
  for (llvm::sys::fs::directory_iterator
           iterator(root + "/out", directoryError),
       end;
       iterator != end && !directoryError; iterator.increment(directoryError))
    ADD_FAILURE() << "unexpected file created: " << iterator->path();
  ASSERT_FALSE(directoryError);
}

} // namespace
