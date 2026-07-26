#include "WaferProfileCampaign.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Target/TargetFormat.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using wafer::runtime::BoardRuntimeOutput;
using wafer::runtime::PackageAccessMode;
using wafer::runtime::PackageManifest;
using wafer::runtime::PackageResourceRecord;
using wafer::runtime::PackageResourceRole;
using wafer::runtime::ResourceId;
using wafer::runtime::cli::BoardInvocationFilePlan;
using wafer::runtime::cli::BoardProfileOutputValidationMode;
using wafer::runtime::cli::BoardProfileOutputValidationState;
using wafer::runtime::cli::BoardProfileProtocolCandidate;
using wafer::runtime::cli::BoardProfileProtocolLaunch;
using wafer::runtime::cli::BoardProfileProtocolObservation;
using wafer::runtime::cli::BoardProfileProtocolStep;

constexpr uint32_t kTraceFlags = WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED |
                                 WAFER_TX81_PROFILER_RECORD_ENTRY_BEGUN |
                                 WAFER_TX81_PROFILER_RECORD_ENTRY_ENDED |
                                 WAFER_TX81_PROFILER_RECORD_PUBLISHED;

BoardProfileProtocolObservation
validObservation(const BoardProfileProtocolStep &step) {
  BoardProfileProtocolObservation observation;
  if (step.launch == BoardProfileProtocolLaunch::Measurement) {
    observation.launchToCompletionNanoseconds = 1000;
    observation.completionObservationResolutionNanoseconds = 2;
  }
  if (step.launch == BoardProfileProtocolLaunch::Count)
    observation.countSequences.fill(3);
  if (step.launch == BoardProfileProtocolLaunch::Trace)
    for (auto &tile : observation.trace)
      tile = {/*preflightCount=*/3,
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

  PackageResourceRecord outputResource(uint64_t id, int64_t logicalRank,
                                       int64_t roleIndex,
                                       llvm::StringRef name) const {
    return {ResourceId(id),
            logicalRank,
            PackageResourceRole::Output,
            roleIndex,
            name.str(),
            {"u8", {4}},
            4,
            4,
            PackageAccessMode::WriteOnly,
            true};
  }

  PackageManifest manifest(uint64_t firstId, uint64_t secondId,
                           llvm::StringRef namePrefix) const {
    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    PackageManifest package(
        target.id, target.targetIdentity, target.kernelRuntimeABI,
        wafer::TargetLaunchABIId::perRankPointerBlockV1(), target.moduleFormat);
    package.rankCount = 2;
    // Deliberately non-canonical manifest order.
    package.resources = {
        outputResource(firstId, /*logicalRank=*/1, /*roleIndex=*/7,
                       (namePrefix + "-rank1").str()),
        outputResource(secondId, /*logicalRank=*/0, /*roleIndex=*/3,
                       (namePrefix + "-rank0").str()),
    };
    return package;
  }

  std::vector<uint8_t> bytesFor(const PackageResourceRecord &resource) const {
    return {static_cast<uint8_t>(resource.logicalRank + 1),
            static_cast<uint8_t>(resource.roleIndex),
            static_cast<uint8_t>(resource.logicalRank + resource.roleIndex),
            UINT8_C(0x5a)};
  }

  BoardInvocationFilePlan
  plan(const PackageManifest &package,
       const std::set<int64_t> &externallyExpectedRoleIndices,
       llvm::StringRef unusedPathPrefix) const {
    BoardInvocationFilePlan result;
    for (const PackageResourceRecord &resource : package.resources) {
      const uint64_t id = resource.id.getValue();
      result.writableResourceBytes[id] = resource.bytes;
      if (externallyExpectedRoleIndices.count(resource.roleIndex) != 0)
        result.expectedBytes[id] = bytesFor(resource);
      else
        result.outputPaths[id] =
            (unusedPathPrefix + llvm::Twine("-") + llvm::Twine(id)).str();
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

TEST(WaferProfileCampaignTest, FixedOrderUsesOneSerialProtocolAndFinalWinner) {
  std::vector<BoardProfileProtocolStep> steps;
  bool finalized = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      {/*baseline=*/8, /*winner=*/8},
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        steps.push_back(step);
        return validObservation(step);
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>
              samples) {
        finalized = true;
        EXPECT_EQ(samples.size(), 20u);
        for (const auto &sample : samples)
          EXPECT_EQ(sample.completionObservationResolutionNanoseconds, 2u);
        return llvm::Error::success();
      });
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(steps.size(), 28u);
  EXPECT_EQ(steps[0].candidate, BoardProfileProtocolCandidate::Winner);
  EXPECT_EQ(steps[0].launch, BoardProfileProtocolLaunch::Warmup);
  EXPECT_EQ(steps[1].candidate, BoardProfileProtocolCandidate::Baseline);
  EXPECT_EQ(steps[1].launch, BoardProfileProtocolLaunch::Warmup);
  const char *orders[] = {"ABBA", "BAAB", "ABBA", "BAAB", "ABBA"};
  for (size_t block = 0; block < 5; ++block)
    for (size_t position = 0; position < 4; ++position) {
      const BoardProfileProtocolStep &step = steps[2 + block * 4 + position];
      EXPECT_EQ(step.launch, BoardProfileProtocolLaunch::Measurement);
      EXPECT_EQ(step.block, static_cast<int32_t>(block));
      EXPECT_EQ(step.position, static_cast<int32_t>(position));
      EXPECT_EQ(step.candidate, orders[block][position] == 'A'
                                    ? BoardProfileProtocolCandidate::Baseline
                                    : BoardProfileProtocolCandidate::Winner);
    }
  EXPECT_EQ(steps[22].launch, BoardProfileProtocolLaunch::Summary);
  EXPECT_EQ(steps[23].launch, BoardProfileProtocolLaunch::Summary);
  EXPECT_EQ(steps[24].launch, BoardProfileProtocolLaunch::Count);
  EXPECT_EQ(steps[25].launch, BoardProfileProtocolLaunch::Count);
  EXPECT_EQ(steps[26].launch, BoardProfileProtocolLaunch::Trace);
  EXPECT_EQ(steps[27].launch, BoardProfileProtocolLaunch::Trace);
  EXPECT_TRUE(finalized);
  EXPECT_EQ(result->finalWinnerSampleId, "winner-b4-p2");
}

TEST_F(WaferProfileOutputValidationTest,
       AllExpectedUsesExternalExactAndCanonicalSemanticKeys) {
  PackageManifest winner = manifest(101, 102, "winner");
  BoardInvocationFilePlan winnerPlan =
      plan(winner, {/*roleIndex=*/3, /*roleIndex=*/7}, "/unused-winner-output");
  BoardProfileOutputValidationState validation(root.str().str());
  ASSERT_FALSE(validation.establishProductionWinnerReference(winner, winnerPlan,
                                                             outputs(winner)));
  EXPECT_EQ(validation.getMode(),
            BoardProfileOutputValidationMode::ExternalExact);
  ASSERT_EQ(validation.getResources().size(), 2u);
  EXPECT_EQ(validation.getResources()[0].logicalRank, 0);
  EXPECT_EQ(validation.getResources()[0].roleIndex, 3);
  EXPECT_EQ(validation.getResources()[1].logicalRank, 1);
  EXPECT_EQ(validation.getResources()[1].roleIndex, 7);
  for (const auto &resource : validation.getResources()) {
    ASSERT_TRUE(resource.externalExpectedExact.has_value());
    EXPECT_TRUE(*resource.externalExpectedExact);
    EXPECT_TRUE(
        llvm::StringRef(resource.referenceSha256).starts_with("sha256:"));
    EXPECT_EQ(resource.referenceSha256.size(), 71u);
  }

  ASSERT_FALSE(validation.validateProductionWinnerRepeat(winner, winnerPlan,
                                                         outputs(winner)));
  PackageManifest baseline = manifest(901, 902, "renamed-baseline");
  BoardInvocationFilePlan baselinePlan =
      plan(baseline, {/*roleIndex=*/3, /*roleIndex=*/7},
           "/different-unused-baseline-output");
  ASSERT_FALSE(validation.validateCandidateEquivalent(baseline, baselinePlan,
                                                      outputs(baseline)));
  ASSERT_FALSE(validation.finalizeAndRemoveReferences());
  for (const auto &resource : validation.getResources()) {
    EXPECT_TRUE(resource.productionWinnerRepeatExact);
    EXPECT_TRUE(resource.candidateEquivalentExact);
  }
}

TEST_F(WaferProfileOutputValidationTest,
       OutputOnlyUsesSameSessionReferenceAcrossIdsNamesAndPaths) {
  PackageManifest winner = manifest(11, 12, "winner");
  BoardInvocationFilePlan winnerPlan =
      plan(winner, {}, "/path-that-must-not-be-read-or-written");
  BoardProfileOutputValidationState validation(root.str().str());
  ASSERT_FALSE(validation.establishProductionWinnerReference(winner, winnerPlan,
                                                             outputs(winner)));
  EXPECT_EQ(validation.getMode(),
            BoardProfileOutputValidationMode::SameSessionProductionWinner);
  for (const auto &resource : validation.getResources())
    EXPECT_FALSE(resource.externalExpectedExact.has_value());

  ASSERT_FALSE(validation.validateProductionWinnerRepeat(winner, winnerPlan,
                                                         outputs(winner)));
  PackageManifest capture = manifest(501, 502, "different-names");
  BoardInvocationFilePlan capturePlan =
      plan(capture, {}, "/completely-different-unused-path");
  ASSERT_FALSE(validation.validateCandidateEquivalent(capture, capturePlan,
                                                      outputs(capture)));
  ASSERT_FALSE(validation.finalizeAndRemoveReferences());

  for (const auto &entry : winnerPlan.outputPaths)
    EXPECT_FALSE(llvm::sys::fs::exists(entry.second));
  for (const auto &entry : capturePlan.outputPaths)
    EXPECT_FALSE(llvm::sys::fs::exists(entry.second));
}

TEST_F(WaferProfileOutputValidationTest,
       PartialExpectedUsesMixedPerResourceCoverage) {
  PackageManifest winner = manifest(21, 22, "winner");
  BoardInvocationFilePlan winnerPlan =
      plan(winner, {/*roleIndex=*/7}, "/unused-winner-output");
  BoardProfileOutputValidationState validation(root.str().str());
  ASSERT_FALSE(validation.establishProductionWinnerReference(winner, winnerPlan,
                                                             outputs(winner)));
  EXPECT_EQ(validation.getMode(), BoardProfileOutputValidationMode::Mixed);
  ASSERT_EQ(validation.getResources().size(), 2u);
  EXPECT_FALSE(validation.getResources()[0].externalExpectedExact.has_value());
  ASSERT_TRUE(validation.getResources()[1].externalExpectedExact.has_value());
  EXPECT_TRUE(*validation.getResources()[1].externalExpectedExact);

  ASSERT_FALSE(validation.validateProductionWinnerRepeat(winner, winnerPlan,
                                                         outputs(winner)));
  PackageManifest baseline = manifest(71, 72, "baseline");
  BoardInvocationFilePlan baselinePlan =
      plan(baseline, {/*roleIndex=*/7}, "/unused-baseline-output");
  ASSERT_FALSE(validation.validateCandidateEquivalent(baseline, baselinePlan,
                                                      outputs(baseline)));
  ASSERT_FALSE(validation.finalizeAndRemoveReferences());
}

TEST_F(WaferProfileOutputValidationTest,
       ExternalExpectedMismatchPrecedesSameSessionReference) {
  PackageManifest winner = manifest(31, 32, "winner");
  BoardInvocationFilePlan winnerPlan =
      plan(winner, {/*roleIndex=*/3, /*roleIndex=*/7}, "/unused-output");
  std::vector<BoardRuntimeOutput> actual = outputs(winner);
  actual.front().bytes[1] ^= UINT8_C(0xff);
  BoardProfileOutputValidationState validation(root.str().str());
  llvm::Error error =
      validation.establishProductionWinnerReference(winner, winnerPlan, actual);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error))
                .find("complete board output differs for ResourceId"),
            std::string::npos);
}

TEST_F(WaferProfileOutputValidationTest,
       WinnerBaselineAndCaptureMismatchStopAtFirstLaunch) {
  enum class FailureKind {
    BaselineWarmup,
    WinnerRepeat,
    SummaryCapture,
    CountCapture,
    TraceCapture,
  };
  for (auto [failure, expectedCalls] :
       {std::pair(FailureKind::BaselineWarmup, size_t(2)),
        std::pair(FailureKind::WinnerRepeat, size_t(4)),
        std::pair(FailureKind::SummaryCapture, size_t(23)),
        std::pair(FailureKind::CountCapture, size_t(25)),
        std::pair(FailureKind::TraceCapture, size_t(27))}) {
    SCOPED_TRACE(static_cast<int>(failure));
    PackageManifest winner = manifest(41, 42, "winner");
    PackageManifest baseline = manifest(81, 82, "baseline");
    BoardInvocationFilePlan winnerPlan =
        plan(winner, {}, "/unused-winner-output");
    BoardInvocationFilePlan baselinePlan =
        plan(baseline, {}, "/unused-baseline-output");
    BoardProfileOutputValidationState validation(root.str().str());
    size_t calls = 0;
    bool finalized = false;
    auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
        {8, 8},
        [&](const BoardProfileProtocolStep &step)
            -> llvm::Expected<BoardProfileProtocolObservation> {
          ++calls;
          const bool useWinner =
              step.candidate == BoardProfileProtocolCandidate::Winner;
          const PackageManifest &package = useWinner ? winner : baseline;
          const BoardInvocationFilePlan &invocation =
              useWinner ? winnerPlan : baselinePlan;
          std::vector<BoardRuntimeOutput> actual = outputs(package);
          const bool corrupt =
              (failure == FailureKind::BaselineWarmup && !useWinner &&
               step.launch == BoardProfileProtocolLaunch::Warmup) ||
              (failure == FailureKind::WinnerRepeat && useWinner &&
               step.launch == BoardProfileProtocolLaunch::Measurement) ||
              (failure == FailureKind::SummaryCapture &&
               step.launch == BoardProfileProtocolLaunch::Summary) ||
              (failure == FailureKind::CountCapture &&
               step.launch == BoardProfileProtocolLaunch::Count) ||
              (failure == FailureKind::TraceCapture &&
               step.launch == BoardProfileProtocolLaunch::Trace);
          if (corrupt)
            actual.front().bytes[2] ^= UINT8_C(0xff);

          llvm::Error outputError = [&]() -> llvm::Error {
            if (useWinner && step.launch == BoardProfileProtocolLaunch::Warmup)
              return validation.establishProductionWinnerReference(
                  package, invocation, actual);
            if (useWinner &&
                step.launch == BoardProfileProtocolLaunch::Measurement)
              return validation.validateProductionWinnerRepeat(
                  package, invocation, actual);
            return validation.validateCandidateEquivalent(package, invocation,
                                                          actual);
          }();
          if (outputError)
            return std::move(outputError);
          return validObservation(step);
        },
        [&](llvm::ArrayRef<
            wafer::runtime::cli::BoardProfileMeasurementSample>) {
          finalized = true;
          return llvm::Error::success();
        });
    ASSERT_FALSE(static_cast<bool>(result));
    EXPECT_EQ(calls, expectedCalls);
    EXPECT_FALSE(finalized);
    EXPECT_NE(llvm::toString(result.takeError()).find("reference"),
              std::string::npos);
  }
}

TEST(WaferProfileCampaignTest, FirstLaunchErrorStopsEveryLaterCall) {
  for (size_t failureIndex :
       {size_t(0), size_t(7), size_t(22), size_t(24), size_t(26)}) {
    size_t calls = 0;
    bool finalized = false;
    auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
        {8, 8},
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
          finalized = true;
          return llvm::Error::success();
        });
    ASSERT_FALSE(static_cast<bool>(result));
    EXPECT_EQ(calls, failureIndex + 1);
    EXPECT_FALSE(finalized);
    llvm::consumeError(result.takeError());
  }
}

TEST(WaferProfileCampaignTest,
     ZeroMeasurementObservationResolutionStopsBeforeLaterLaunches) {
  size_t calls = 0;
  bool finalized = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      {8, 8},
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        ++calls;
        BoardProfileProtocolObservation observation = validObservation(step);
        if (step.launch == BoardProfileProtocolLaunch::Measurement)
          observation.completionObservationResolutionNanoseconds = 0;
        return observation;
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>) {
        finalized = true;
        return llvm::Error::success();
      });
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_EQ(calls, 3u);
  EXPECT_FALSE(finalized);
  EXPECT_NE(llvm::toString(result.takeError()).find("resolution"),
            std::string::npos);
}

TEST(WaferProfileCampaignTest, CountCapacityFailurePreventsEveryTraceLaunch) {
  size_t calls = 0;
  size_t traceCalls = 0;
  bool finalized = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      {/*baseline=*/2, /*winner=*/8},
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        ++calls;
        if (step.launch == BoardProfileProtocolLaunch::Trace)
          ++traceCalls;
        return validObservation(step);
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>) {
        finalized = true;
        return llvm::Error::success();
      });
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_EQ(calls, 26u);
  EXPECT_EQ(traceCalls, 0u);
  EXPECT_FALSE(finalized);
  EXPECT_NE(llvm::toString(result.takeError()).find("exceeds"),
            std::string::npos);
}

TEST(WaferProfileCampaignTest,
     TraceMismatchStopsBeforeWinnerTraceAndReportPublication) {
  size_t calls = 0;
  size_t traceCalls = 0;
  bool finalized = false;
  auto result = wafer::runtime::cli::runFixedBoardProfileProtocol(
      {8, 8},
      [&](const BoardProfileProtocolStep &step)
          -> llvm::Expected<BoardProfileProtocolObservation> {
        ++calls;
        BoardProfileProtocolObservation observation = validObservation(step);
        if (step.launch == BoardProfileProtocolLaunch::Trace) {
          ++traceCalls;
          if (step.candidate == BoardProfileProtocolCandidate::Baseline)
            observation.trace[7].nextSequence = 4;
        }
        return observation;
      },
      [&](llvm::ArrayRef<wafer::runtime::cli::BoardProfileMeasurementSample>) {
        finalized = true;
        return llvm::Error::success();
      });
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_EQ(calls, 27u);
  EXPECT_EQ(traceCalls, 1u);
  EXPECT_FALSE(finalized);
  EXPECT_NE(llvm::toString(result.takeError()).find("audit"),
            std::string::npos);
}

} // namespace
