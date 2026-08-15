#include "WaferProfileCollection.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Target/TargetFormat.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using wafer::runtime::BoardRuntimeOutput;
using wafer::runtime::CardResourceScope;
using wafer::runtime::PackageAccessMode;
using wafer::runtime::PackageManifest;
using wafer::runtime::PackageResourceRecord;
using wafer::runtime::PackageResourceRole;
using wafer::runtime::ResourceId;
using wafer::runtime::cli::BoardInvocationFilePlan;
using wafer::runtime::cli::BoardOutputComparisonKind;
using wafer::runtime::cli::BoardProfileOutputValidationMode;
using wafer::runtime::cli::BoardProfileOutputValidator;
using wafer::runtime::cli::BoardProfileProtocolLaunch;
using wafer::runtime::cli::BoardProfileProtocolObservation;
using wafer::runtime::cli::BoardProfileProtocolStep;

constexpr uint32_t kTraceFlags = WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED |
                                 WAFER_TX81_PROFILER_RECORD_ENTRY_BEGUN |
                                 WAFER_TX81_PROFILER_RECORD_ENTRY_ENDED |
                                 WAFER_TX81_PROFILER_RECORD_COMPLETE;

BoardProfileProtocolObservation
validObservation(const BoardProfileProtocolStep &step) {
  BoardProfileProtocolObservation observation;
  if (step.launch == BoardProfileProtocolLaunch::Primary) {
    observation.deviceExecutionNanoseconds = 800;
    observation.hostSubmitNanoseconds = 200;
    observation.launchToCompletionNanoseconds = 1000;
    observation.completionObservationResolutionNanoseconds = 2;
  }
  if (step.launch == BoardProfileProtocolLaunch::Count)
    observation.countSequences.fill(3);
  if (step.launch == BoardProfileProtocolLaunch::Trace)
    for (auto &tile : observation.trace)
      tile = {/*countedEventCount=*/3,
              /*nextSequence=*/3,
              /*storedEventCount=*/3,
              /*droppedEventCount=*/0,
              /*recordFlags=*/kTraceFlags,
              /*traceState=*/WAFER_TX81_PROFILER_TRACE_COMPLETE};
  return observation;
}

class WaferProfileOutputValidationTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
        "wafer-profile-output-validation-test", root));
  }

  void TearDown() override { llvm::sys::fs::remove_directories(root); }

  PackageResourceRecord outputResource(uint64_t id, int64_t roleIndex,
                                       llvm::StringRef name) const {
    return {ResourceId(id),
            CardResourceScope{wafer::CardId(0)},
            PackageResourceRole::Output,
            roleIndex,
            name.str(),
            {"f16", {2}},
            4,
            4,
            PackageAccessMode::WriteOnly,
            true};
  }

  PackageManifest manifest(uint64_t firstId, uint64_t secondId,
                           llvm::StringRef namePrefix) const {
    PackageManifest package(
        wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
        llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
            wafer::KernelLaunchForm::Grid,
            wafer::KernelEntryABI::TileMajorPointerTable,
            {wafer::RuntimeLaunchPhaseRole::Main})),
        wafer::kCurrentTargetModuleFormat);
    package.cardCount = 1;
    package.tileCount = 16;
    // Deliberately non-canonical manifest order.
    package.resources = {
        outputResource(firstId, /*roleIndex=*/7,
                       (namePrefix + "-output7").str()),
        outputResource(secondId, /*roleIndex=*/3,
                       (namePrefix + "-output3").str()),
    };
    return package;
  }

  std::vector<uint8_t> bytesFor(const PackageResourceRecord &resource) const {
    return {static_cast<uint8_t>(resource.roleIndex + 1),
            static_cast<uint8_t>(resource.roleIndex),
            static_cast<uint8_t>(resource.roleIndex * 2), UINT8_C(0x5a)};
  }

  BoardInvocationFilePlan
  plan(const PackageManifest &package,
       const std::set<int64_t> &externallyExpectedRoleIndices,
       llvm::StringRef unusedPathPrefix,
       const std::set<int64_t> &relaxedF16RoleIndices = {}) const {
    BoardInvocationFilePlan result;
    for (const PackageResourceRecord &resource : package.resources) {
      const uint64_t id = resource.id.getValue();
      result.writableResourceBytes[id] = resource.bytes;
      if (externallyExpectedRoleIndices.count(resource.roleIndex) != 0) {
        result.expectedBytes[id] = bytesFor(resource);
        result.expectedComparisons[id] =
            relaxedF16RoleIndices.count(resource.roleIndex) != 0
                ? BoardOutputComparisonKind::RelaxedF16
                : BoardOutputComparisonKind::Exact;
      } else {
        result.outputPaths[id] =
            (unusedPathPrefix + llvm::Twine("-") + llvm::Twine(id)).str();
      }
    }
    return result;
  }

  std::vector<BoardRuntimeOutput>
  outputs(const PackageManifest &package) const {
    std::vector<BoardRuntimeOutput> result;
    for (const PackageResourceRecord &resource : package.resources)
      result.push_back({resource.id, bytesFor(resource)});
    return result;
  }

  llvm::SmallString<256> root;
};

llvm::Error writeProfileReportMember(llvm::StringRef directory,
                                     llvm::StringRef name,
                                     llvm::StringRef contents) {
  llvm::SmallString<256> path(directory);
  llvm::sys::path::append(path, name);
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "failed to create test report file");
  output << contents;
  output.close();
  if (output.has_error())
    return llvm::createStringError(output.error(),
                                   "failed to write test report file");
  return llvm::Error::success();
}

class WaferProfileReportDirectoryTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
        "wafer-profile-report-directory-test", root));
  }

  void TearDown() override { llvm::sys::fs::remove_directories(root); }

  llvm::Error stageValidReport(llvm::StringRef runId,
                               llvm::StringRef stagingDirectory) const {
    std::string evidence = "{\"run_id\":\"" + runId.str() + "\"}\n";
    if (llvm::Error error = writeProfileReportMember(stagingDirectory,
                                                     "evidence.json", evidence))
      return error;
    if (llvm::Error error =
            writeProfileReportMember(stagingDirectory, "analysis.json", "{}\n"))
      return error;
    return writeProfileReportMember(stagingDirectory, "index.html",
                                    "<!doctype html>\n");
  }

  static std::error_code removeManagedRun(llvm::StringRef runDirectory) {
    return llvm::sys::fs::remove_directories(runDirectory,
                                             /*IgnoreErrors=*/false);
  }

  void expectMode0777(llvm::StringRef path) const {
    llvm::sys::fs::file_status status;
    ASSERT_FALSE(llvm::sys::fs::status(path, status));
    EXPECT_EQ(status.permissions() & llvm::sys::fs::all_all,
              llvm::sys::fs::all_all);
  }

  llvm::SmallString<256> root;
};

TEST_F(WaferProfileReportDirectoryTest,
       ReplacesCurrentWithExactlyThreeWorldAccessibleFiles) {
  auto stage = [&](llvm::StringRef runId, llvm::StringRef stagingDirectory) {
    return stageValidReport(runId, stagingDirectory);
  };
  auto first = wafer::runtime::cli::testing::writeProfileReportForTesting(
      root, stage, removeManagedRun);
  ASSERT_TRUE(static_cast<bool>(first)) << llvm::toString(first.takeError());
  const std::string firstRunDirectory = first->runDirectory;

  auto second = wafer::runtime::cli::testing::writeProfileReportForTesting(
      root, stage, removeManagedRun);
  ASSERT_TRUE(static_cast<bool>(second)) << llvm::toString(second.takeError());
  EXPECT_FALSE(llvm::sys::fs::exists(firstRunDirectory));

  llvm::SmallString<256> runs(root);
  llvm::sys::path::append(runs, "runs");
  expectMode0777(runs);
  expectMode0777(second->runDirectory);

  llvm::SmallString<256> current(runs);
  llvm::sys::path::append(current, "current");
  EXPECT_EQ(llvm::sys::fs::get_file_type(current, /*Follow=*/false),
            llvm::sys::fs::file_type::symlink_file);
  llvm::SmallString<256> resolved;
  ASSERT_FALSE(llvm::sys::fs::real_path(current, resolved));
  EXPECT_EQ(resolved, second->runDirectory);

  std::set<std::string> members;
  std::error_code walkError;
  for (llvm::sys::fs::directory_iterator
           iterator(second->runDirectory, walkError,
                    /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(walkError)) {
    ASSERT_FALSE(walkError);
    members.insert(llvm::sys::path::filename(iterator->path()).str());
    expectMode0777(iterator->path());
  }
  ASSERT_FALSE(walkError);
  EXPECT_EQ(members, (std::set<std::string>{"analysis.json", "evidence.json",
                                            "index.html"}));
}

TEST_F(WaferProfileReportDirectoryTest,
       OldRunRemovalFailureIsReturnedAfterCurrentEntryReplacement) {
  auto first = wafer::runtime::cli::testing::writeProfileReportForTesting(
      root,
      [&](llvm::StringRef runId, llvm::StringRef stagingDirectory) {
        return stageValidReport(runId, stagingDirectory);
      },
      removeManagedRun);
  ASSERT_TRUE(static_cast<bool>(first)) << llvm::toString(first.takeError());

  size_t removalCalls = 0;
  std::string removedDirectory;
  std::string secondRunId;
  auto second = wafer::runtime::cli::testing::writeProfileReportForTesting(
      root,
      [&](llvm::StringRef runId, llvm::StringRef stagingDirectory) {
        secondRunId = runId.str();
        return stageValidReport(runId, stagingDirectory);
      },
      [&](llvm::StringRef runDirectory) {
        ++removalCalls;
        removedDirectory = runDirectory.str();
        return std::make_error_code(std::errc::permission_denied);
      });
  ASSERT_FALSE(static_cast<bool>(second));
  std::string message = llvm::toString(second.takeError());
  EXPECT_NE(message.find("previous run could not be removed"),
            std::string::npos);
  EXPECT_EQ(removalCalls, 1u);
  EXPECT_EQ(removedDirectory, first->runDirectory);
  EXPECT_TRUE(llvm::sys::fs::exists(first->runDirectory));

  llvm::SmallString<256> current(root);
  llvm::sys::path::append(current, "runs", "current");
  llvm::SmallString<256> resolved;
  ASSERT_FALSE(llvm::sys::fs::real_path(current, resolved));
  EXPECT_EQ(llvm::sys::path::filename(resolved), secondRunId);
}

TEST_F(WaferProfileReportDirectoryTest,
       MismatchedEvidenceIdentityRejectsCurrentBeforeStagingOrDeletion) {
  llvm::SmallString<256> runs(root);
  llvm::sys::path::append(runs, "runs");
  ASSERT_FALSE(llvm::sys::fs::create_directory(runs));
  llvm::SmallString<256> forged(runs);
  llvm::sys::path::append(forged, "run-forged");
  ASSERT_FALSE(llvm::sys::fs::create_directory(forged));
  ASSERT_FALSE(static_cast<bool>(writeProfileReportMember(
      forged, "evidence.json", "{\"run_id\":\"run-other\"}\n")));
  ASSERT_FALSE(static_cast<bool>(
      writeProfileReportMember(forged, "analysis.json", "{}\n")));
  ASSERT_FALSE(static_cast<bool>(
      writeProfileReportMember(forged, "index.html", "<!doctype html>\n")));
  llvm::SmallString<256> current(runs);
  llvm::sys::path::append(current, "current");
  ASSERT_FALSE(llvm::sys::fs::create_link(forged, current));

  bool staged = false;
  bool removed = false;
  auto result = wafer::runtime::cli::testing::writeProfileReportForTesting(
      root,
      [&](llvm::StringRef, llvm::StringRef) {
        staged = true;
        return llvm::Error::success();
      },
      [&](llvm::StringRef) {
        removed = true;
        return std::error_code();
      });
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("run_id"),
            std::string::npos);
  EXPECT_FALSE(staged);
  EXPECT_FALSE(removed);
  EXPECT_TRUE(llvm::sys::fs::exists(forged));
}

TEST_F(WaferProfileReportDirectoryTest,
       UnexpectedCurrentMemberRejectsRecursiveDeletion) {
  llvm::SmallString<256> runs(root);
  llvm::sys::path::append(runs, "runs");
  ASSERT_FALSE(llvm::sys::fs::create_directory(runs));
  llvm::SmallString<256> prior(runs);
  llvm::sys::path::append(prior, "run-prior");
  ASSERT_FALSE(llvm::sys::fs::create_directory(prior));
  ASSERT_FALSE(static_cast<bool>(writeProfileReportMember(
      prior, "evidence.json", "{\"run_id\":\"run-prior\"}\n")));
  ASSERT_FALSE(static_cast<bool>(
      writeProfileReportMember(prior, "analysis.json", "{}\n")));
  ASSERT_FALSE(static_cast<bool>(
      writeProfileReportMember(prior, "index.html", "<!doctype html>\n")));
  ASSERT_FALSE(static_cast<bool>(
      writeProfileReportMember(prior, "unexpected.txt", "do not delete\n")));
  llvm::SmallString<256> current(runs);
  llvm::sys::path::append(current, "current");
  ASSERT_FALSE(llvm::sys::fs::create_link(prior, current));

  bool staged = false;
  bool removed = false;
  auto result = wafer::runtime::cli::testing::writeProfileReportForTesting(
      root,
      [&](llvm::StringRef, llvm::StringRef) {
        staged = true;
        return llvm::Error::success();
      },
      [&](llvm::StringRef) {
        removed = true;
        return std::error_code();
      });
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("unexpected file"),
            std::string::npos);
  EXPECT_FALSE(staged);
  EXPECT_FALSE(removed);
  EXPECT_TRUE(llvm::sys::fs::exists(prior));
}

TEST(WaferProfileCollectionTest, FixedOrderRunsOnePrimaryThenCountAndTrace) {
  std::vector<BoardProfileProtocolStep> steps;
  bool measurementsConsumed = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      /*traceCapacity=*/8,
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        steps.push_back(step);
        return validObservation(step);
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>
              samples) -> llvm::Error {
        measurementsConsumed = true;
        EXPECT_EQ(samples.size(), 1u);
        if (samples.size() != 1)
          return llvm::createStringError(llvm::errc::invalid_argument,
                                         "unexpected sample domain");
        EXPECT_EQ(samples[0].id, "primary");
        EXPECT_EQ(samples[0].sampleIndex, 0u);
        EXPECT_EQ(samples[0].deviceElapsedNanoseconds, 800u);
        EXPECT_EQ(samples[0].deviceTimerKind, "tx-stream-events");
        EXPECT_EQ(samples[0].hostSubmitNanoseconds, 200u);
        EXPECT_EQ(samples[0].hostLaunchToCompletionNanoseconds, 1000u);
        EXPECT_EQ(samples[0].completionObservationResolutionNanoseconds, 2u);
        return llvm::Error::success();
      });
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(steps.size(), 3u);
  EXPECT_EQ(steps[0].launch, BoardProfileProtocolLaunch::Primary);
  EXPECT_EQ(steps[1].launch, BoardProfileProtocolLaunch::Count);
  EXPECT_EQ(steps[2].launch, BoardProfileProtocolLaunch::Trace);
  EXPECT_TRUE(measurementsConsumed);
  ASSERT_EQ(result->samples.size(), 1u);
  EXPECT_EQ(result->samples[0].id, "primary");
  EXPECT_EQ(result->primarySampleId, "primary");
}

TEST_F(WaferProfileOutputValidationTest,
       AllExpectedPreservesComparisonPolicyAndCanonicalSemanticKeys) {
  PackageManifest primary = manifest(101, 102, "primary");
  BoardInvocationFilePlan primaryPlan =
      plan(primary, {/*roleIndex=*/3, /*roleIndex=*/7}, "/unused-output",
           {/*relaxedF16RoleIndex=*/7});
  BoardProfileOutputValidator validation(root.str().str());
  ASSERT_FALSE(
      validation.recordPrimaryOutputs(primary, primaryPlan, outputs(primary)));
  EXPECT_EQ(validation.getMode(),
            BoardProfileOutputValidationMode::ExternalExpected);
  ASSERT_EQ(validation.getResources().size(), 2u);
  EXPECT_EQ(
      std::get<CardResourceScope>(validation.getResources()[0].scope).cardId,
      wafer::CardId(0));
  EXPECT_EQ(validation.getResources()[0].roleIndex, 3);
  EXPECT_EQ(
      std::get<CardResourceScope>(validation.getResources()[1].scope).cardId,
      wafer::CardId(0));
  EXPECT_EQ(validation.getResources()[1].roleIndex, 7);
  ASSERT_TRUE(
      validation.getResources()[0].externalExpectedComparison.has_value());
  EXPECT_EQ(*validation.getResources()[0].externalExpectedComparison,
            BoardOutputComparisonKind::Exact);
  ASSERT_TRUE(
      validation.getResources()[1].externalExpectedComparison.has_value());
  EXPECT_EQ(*validation.getResources()[1].externalExpectedComparison,
            BoardOutputComparisonKind::RelaxedF16);
  for (const auto &resource : validation.getResources()) {
    EXPECT_TRUE(
        llvm::StringRef(resource.referenceSha256).starts_with("sha256:"));
    EXPECT_EQ(resource.referenceSha256.size(), 71u);
    EXPECT_TRUE(resource.primaryOutputValidated);
    EXPECT_FALSE(resource.diagnosticCapturesMatchPrimary);
  }

  PackageManifest capture = manifest(901, 902, "capture");
  BoardInvocationFilePlan capturePlan =
      plan(capture, {/*roleIndex=*/3, /*roleIndex=*/7}, "/unused-capture",
           {/*relaxedF16RoleIndex=*/7});
  ASSERT_FALSE(validation.validateDiagnosticOutputs(capture, capturePlan,
                                                    outputs(capture)));
  for (const auto &resource : validation.getResources())
    EXPECT_FALSE(resource.diagnosticCapturesMatchPrimary);
  ASSERT_FALSE(validation.validateDiagnosticOutputs(capture, capturePlan,
                                                    outputs(capture)));
  ASSERT_FALSE(validation.removeReferenceFiles());
  for (const auto &resource : validation.getResources()) {
    EXPECT_TRUE(resource.primaryOutputValidated);
    EXPECT_TRUE(resource.diagnosticCapturesMatchPrimary);
  }
}

TEST_F(WaferProfileOutputValidationTest,
       ReportWriteRequiresBothDiagnosticCaptures) {
  PackageManifest primary = manifest(121, 122, "primary");
  BoardInvocationFilePlan primaryPlan = plan(primary, {}, "/unused-output");
  BoardProfileOutputValidator validation(root.str().str());
  ASSERT_FALSE(
      validation.recordPrimaryOutputs(primary, primaryPlan, outputs(primary)));

  PackageManifest capture = manifest(221, 222, "capture");
  BoardInvocationFilePlan capturePlan = plan(capture, {}, "/unused-capture");
  ASSERT_FALSE(validation.validateDiagnosticOutputs(capture, capturePlan,
                                                    outputs(capture)));
  llvm::Error error = validation.removeReferenceFiles();
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("count and trace"),
            std::string::npos);
}

TEST_F(WaferProfileOutputValidationTest,
       ComparisonPolicyChangeIsRejectedEvenWhenCoverageIsUnchanged) {
  PackageManifest primary = manifest(111, 112, "primary");
  BoardInvocationFilePlan primaryPlan =
      plan(primary, {/*roleIndex=*/3}, "/unused-output");
  BoardProfileOutputValidator validation(root.str().str());
  ASSERT_FALSE(
      validation.recordPrimaryOutputs(primary, primaryPlan, outputs(primary)));

  PackageManifest capture = manifest(211, 212, "capture");
  BoardInvocationFilePlan capturePlan =
      plan(capture, {/*roleIndex=*/3}, "/unused-capture",
           {/*relaxedF16RoleIndex=*/3});
  llvm::Error error = validation.validateDiagnosticOutputs(capture, capturePlan,
                                                           outputs(capture));
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("policy changed"),
            std::string::npos);
}

TEST_F(WaferProfileOutputValidationTest,
       OutputOnlyUsesSameSessionReferenceAcrossIdsNamesAndPaths) {
  PackageManifest primary = manifest(11, 12, "primary");
  BoardInvocationFilePlan primaryPlan =
      plan(primary, {}, "/path-that-must-not-be-read-or-written");
  BoardProfileOutputValidator validation(root.str().str());
  ASSERT_FALSE(
      validation.recordPrimaryOutputs(primary, primaryPlan, outputs(primary)));
  EXPECT_EQ(validation.getMode(),
            BoardProfileOutputValidationMode::SameSessionPrimary);
  for (const auto &resource : validation.getResources())
    EXPECT_FALSE(resource.externalExpectedComparison.has_value());

  PackageManifest capture = manifest(501, 502, "different-names");
  BoardInvocationFilePlan capturePlan =
      plan(capture, {}, "/completely-different-unused-path");
  ASSERT_FALSE(validation.validateDiagnosticOutputs(capture, capturePlan,
                                                    outputs(capture)));
  ASSERT_FALSE(validation.validateDiagnosticOutputs(capture, capturePlan,
                                                    outputs(capture)));
  ASSERT_FALSE(validation.removeReferenceFiles());

  for (const auto &entry : primaryPlan.outputPaths)
    EXPECT_FALSE(llvm::sys::fs::exists(entry.second));
  for (const auto &entry : capturePlan.outputPaths)
    EXPECT_FALSE(llvm::sys::fs::exists(entry.second));
}

TEST_F(WaferProfileOutputValidationTest,
       PartialExpectedUsesMixedPerResourceCoverage) {
  PackageManifest primary = manifest(21, 22, "primary");
  BoardInvocationFilePlan primaryPlan =
      plan(primary, {/*roleIndex=*/7}, "/unused-output");
  BoardProfileOutputValidator validation(root.str().str());
  ASSERT_FALSE(
      validation.recordPrimaryOutputs(primary, primaryPlan, outputs(primary)));
  EXPECT_EQ(validation.getMode(), BoardProfileOutputValidationMode::Mixed);
  ASSERT_EQ(validation.getResources().size(), 2u);
  EXPECT_FALSE(
      validation.getResources()[0].externalExpectedComparison.has_value());
  ASSERT_TRUE(
      validation.getResources()[1].externalExpectedComparison.has_value());
  EXPECT_EQ(*validation.getResources()[1].externalExpectedComparison,
            BoardOutputComparisonKind::Exact);

  PackageManifest capture = manifest(71, 72, "capture");
  BoardInvocationFilePlan capturePlan =
      plan(capture, {/*roleIndex=*/7}, "/unused-capture");
  ASSERT_FALSE(validation.validateDiagnosticOutputs(capture, capturePlan,
                                                    outputs(capture)));
  ASSERT_FALSE(validation.validateDiagnosticOutputs(capture, capturePlan,
                                                    outputs(capture)));
  ASSERT_FALSE(validation.removeReferenceFiles());
}

TEST_F(WaferProfileOutputValidationTest,
       ExternalExpectedMismatchPrecedesSameSessionReference) {
  PackageManifest primary = manifest(31, 32, "primary");
  BoardInvocationFilePlan primaryPlan =
      plan(primary, {/*roleIndex=*/3, /*roleIndex=*/7}, "/unused-output");
  std::vector<BoardRuntimeOutput> actual = outputs(primary);
  actual.front().bytes[1] ^= UINT8_C(0xff);
  BoardProfileOutputValidator validation(root.str().str());
  llvm::Error error =
      validation.recordPrimaryOutputs(primary, primaryPlan, actual);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error))
                .find("complete board output differs for ResourceId"),
            std::string::npos);
}

TEST_F(WaferProfileOutputValidationTest,
       OutputMismatchStopsAtTheFirstAffectedLaunch) {
  enum class FailureKind {
    Primary,
    CountCapture,
    TraceCapture,
  };
  for (auto [failure, expectedCalls] :
       {std::pair(FailureKind::Primary, size_t(1)),
        std::pair(FailureKind::CountCapture, size_t(2)),
        std::pair(FailureKind::TraceCapture, size_t(3))}) {
    SCOPED_TRACE(static_cast<int>(failure));
    PackageManifest primary = manifest(41, 42, "primary");
    PackageManifest capture = manifest(81, 82, "capture");
    BoardInvocationFilePlan primaryPlan =
        plan(primary, {/*roleIndex=*/3, /*roleIndex=*/7}, "/unused-output");
    BoardInvocationFilePlan capturePlan =
        plan(capture, {/*roleIndex=*/3, /*roleIndex=*/7}, "/unused-capture");
    BoardProfileOutputValidator validation(root.str().str());
    size_t calls = 0;
    bool measurementsConsumed = false;
    auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
        /*traceCapacity=*/8,
        [&](const BoardProfileProtocolStep &step)
            -> llvm::Expected<BoardProfileProtocolObservation> {
          ++calls;
          const bool primaryLaunch =
              step.launch == BoardProfileProtocolLaunch::Primary;
          const PackageManifest &package = primaryLaunch ? primary : capture;
          const BoardInvocationFilePlan &invocation =
              primaryLaunch ? primaryPlan : capturePlan;
          std::vector<BoardRuntimeOutput> actual = outputs(package);
          const bool corrupt =
              (failure == FailureKind::Primary &&
               step.launch == BoardProfileProtocolLaunch::Primary) ||
              (failure == FailureKind::CountCapture &&
               step.launch == BoardProfileProtocolLaunch::Count) ||
              (failure == FailureKind::TraceCapture &&
               step.launch == BoardProfileProtocolLaunch::Trace);
          if (corrupt)
            actual.front().bytes[2] ^= UINT8_C(0xff);

          llvm::Error outputError = [&]() -> llvm::Error {
            if (step.launch == BoardProfileProtocolLaunch::Primary)
              return validation.recordPrimaryOutputs(package, invocation,
                                                     actual);
            return validation.validateDiagnosticOutputs(package, invocation,
                                                        actual);
          }();
          if (outputError)
            return std::move(outputError);
          return validObservation(step);
        },
        [&](llvm::ArrayRef<
            wafer::runtime::cli::BoardProfileMeasurementSample>) {
          measurementsConsumed = true;
          return llvm::Error::success();
        });
    ASSERT_FALSE(static_cast<bool>(result));
    EXPECT_EQ(calls, expectedCalls);
    EXPECT_FALSE(measurementsConsumed);
    EXPECT_NE(llvm::toString(result.takeError()).find("differs"),
              std::string::npos);
  }
}

TEST(WaferProfileCollectionTest, FirstLaunchErrorStopsEveryLaterCall) {
  for (size_t failureIndex : {size_t(0), size_t(1), size_t(2)}) {
    size_t calls = 0;
    bool measurementsConsumed = false;
    auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
        /*traceCapacity=*/8,
        [&](const BoardProfileProtocolStep &step)
            -> llvm::Expected<BoardProfileProtocolObservation> {
          const size_t index = calls++;
          if (index == failureIndex)
            return llvm::createStringError(llvm::errc::io_error,
                                           "injected launch failure");
          return validObservation(step);
        },
        [&](llvm::ArrayRef<
            wafer::runtime::cli::BoardProfileMeasurementSample>) {
          measurementsConsumed = true;
          return llvm::Error::success();
        });
    ASSERT_FALSE(static_cast<bool>(result));
    EXPECT_EQ(calls, failureIndex + 1);
    EXPECT_FALSE(measurementsConsumed);
    llvm::consumeError(result.takeError());
  }
}

TEST(WaferProfileCollectionTest,
     PrimaryHostGateFailureDestroysLocalSessionAndMakesNoLaterLaunch) {
  struct LocalSession {
    explicit LocalSession(bool &destroyed) : destroyed(destroyed) {}
    ~LocalSession() { destroyed = true; }
    bool &destroyed;
  };

  size_t providerLaunches = 0;
  bool sessionDestroyed = false;
  bool measurementsConsumed = false;
  auto runCollection =
      [&]() -> llvm::Expected<wafer::runtime::cli::BoardProfileProtocolResult> {
    std::optional<LocalSession> session;
    return wafer::runtime::cli::runFixedBoardProfileProtocol(
        /*traceCapacity=*/8,
        [&](const BoardProfileProtocolStep &step)
            -> llvm::Expected<BoardProfileProtocolObservation> {
          ++providerLaunches;
          EXPECT_EQ(step.launch, BoardProfileProtocolLaunch::Primary);
          session.emplace(sessionDestroyed);
          // Models the host output/topology gate after the first successful
          // provider invocation has created the collection-local session.
          return llvm::createStringError(llvm::errc::invalid_argument,
                                         "primary host output gate failed");
        },
        [&](llvm::ArrayRef<
            wafer::runtime::cli::BoardProfileMeasurementSample>) {
          measurementsConsumed = true;
          return llvm::Error::success();
        });
  };

  auto result = runCollection();
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_EQ(providerLaunches, 1u);
  EXPECT_TRUE(sessionDestroyed);
  EXPECT_FALSE(measurementsConsumed);
  EXPECT_NE(llvm::toString(result.takeError()).find("primary host output"),
            std::string::npos);
}

TEST(WaferProfileCollectionTest,
     QuantizedZeroPrimaryObservationResolutionIsRetained) {
  size_t calls = 0;
  bool measurementsConsumed = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      /*traceCapacity=*/8,
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        ++calls;
        BoardProfileProtocolObservation observation = validObservation(step);
        if (step.launch == BoardProfileProtocolLaunch::Primary)
          observation.completionObservationResolutionNanoseconds = 0;
        return observation;
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>
              samples) {
        measurementsConsumed = true;
        EXPECT_EQ(samples.front().completionObservationResolutionNanoseconds,
                  0u);
        return llvm::Error::success();
      });
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(calls, 3u);
  EXPECT_TRUE(measurementsConsumed);
}

TEST(WaferProfileCollectionTest,
     MissingPrimaryDeviceTimingStopsBeforeLaterLaunches) {
  size_t calls = 0;
  bool measurementsConsumed = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      /*traceCapacity=*/8,
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        ++calls;
        BoardProfileProtocolObservation observation = validObservation(step);
        if (step.launch == BoardProfileProtocolLaunch::Primary)
          observation.deviceExecutionNanoseconds.reset();
        return observation;
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>) {
        measurementsConsumed = true;
        return llvm::Error::success();
      });
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_EQ(calls, 1u);
  EXPECT_FALSE(measurementsConsumed);
  EXPECT_NE(llvm::toString(result.takeError()).find("device execution timing"),
            std::string::npos);
}

TEST(WaferProfileCollectionTest, QuantizedZeroPrimaryDeviceTimeIsAccepted) {
  bool measurementsConsumed = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      /*traceCapacity=*/8,
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        BoardProfileProtocolObservation observation = validObservation(step);
        if (step.launch == BoardProfileProtocolLaunch::Primary)
          observation.deviceExecutionNanoseconds = 0;
        return observation;
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>
              samples) {
        measurementsConsumed = true;
        EXPECT_EQ(samples.front().deviceElapsedNanoseconds, 0u);
        return llvm::Error::success();
      });
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(measurementsConsumed);
}

TEST(WaferProfileCollectionTest,
     HostSubmitOutsidePrimaryEnvelopeStopsBeforeLaterLaunches) {
  size_t calls = 0;
  bool measurementsConsumed = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      /*traceCapacity=*/8,
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        ++calls;
        BoardProfileProtocolObservation observation = validObservation(step);
        if (step.launch == BoardProfileProtocolLaunch::Primary)
          observation.hostSubmitNanoseconds =
              observation.launchToCompletionNanoseconds + 1;
        return observation;
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>) {
        measurementsConsumed = true;
        return llvm::Error::success();
      });
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_EQ(calls, 1u);
  EXPECT_FALSE(measurementsConsumed);
  EXPECT_NE(llvm::toString(result.takeError()).find("exceeds"),
            std::string::npos);
}

TEST(WaferProfileCollectionTest, QuantizedZeroPrimaryHostEnvelopeIsRetained) {
  size_t calls = 0;
  bool measurementsConsumed = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      /*traceCapacity=*/8,
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        ++calls;
        BoardProfileProtocolObservation observation = validObservation(step);
        if (step.launch == BoardProfileProtocolLaunch::Primary) {
          observation.hostSubmitNanoseconds = 0;
          observation.launchToCompletionNanoseconds = 0;
        }
        return observation;
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>
              samples) {
        measurementsConsumed = true;
        EXPECT_EQ(samples.front().hostSubmitNanoseconds, 0u);
        EXPECT_EQ(samples.front().hostLaunchToCompletionNanoseconds, 0u);
        return llvm::Error::success();
      });
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(calls, 3u);
  EXPECT_TRUE(measurementsConsumed);
}

TEST(WaferProfileCollectionTest, CountCapacityFailurePreventsTraceLaunch) {
  size_t calls = 0;
  size_t traceCalls = 0;
  bool measurementsConsumed = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      /*traceCapacity=*/2,
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        ++calls;
        if (step.launch == BoardProfileProtocolLaunch::Trace)
          ++traceCalls;
        return validObservation(step);
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>) {
        measurementsConsumed = true;
        return llvm::Error::success();
      });
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_EQ(calls, 2u);
  EXPECT_EQ(traceCalls, 0u);
  EXPECT_FALSE(measurementsConsumed);
  EXPECT_NE(llvm::toString(result.takeError()).find("exceeds"),
            std::string::npos);
}

TEST(WaferProfileCollectionTest, TraceMismatchStopsBeforeReportWrite) {
  size_t calls = 0;
  size_t traceCalls = 0;
  bool measurementsConsumed = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      /*traceCapacity=*/8,
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        ++calls;
        BoardProfileProtocolObservation observation = validObservation(step);
        if (step.launch == BoardProfileProtocolLaunch::Trace) {
          ++traceCalls;
          observation.trace[7].nextSequence = 4;
        }
        return observation;
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>) {
        measurementsConsumed = true;
        return llvm::Error::success();
      });
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_EQ(calls, 3u);
  EXPECT_EQ(traceCalls, 1u);
  EXPECT_FALSE(measurementsConsumed);
  EXPECT_NE(llvm::toString(result.takeError()).find("audit"),
            std::string::npos);
}

} // namespace
