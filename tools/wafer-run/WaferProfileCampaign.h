//===- WaferProfileCampaign.h - Automatic board profile run ----*- C++ -*-===//

#ifndef WAFER_TOOLS_WAFER_RUN_WAFERPROFILECAMPAIGN_H
#define WAFER_TOOLS_WAFER_RUN_WAFERPROFILECAMPAIGN_H

#include "Wafer/Runtime/ProfileCompanion.h"
#include "WaferRunBoardIO.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace wafer::runtime::cli {

enum class BoardProfileProtocolLaunch {
  Warmup,
  Measurement,
  Summary,
  Count,
  Trace,
};

struct BoardProfileProtocolStep {
  BoardProfileProtocolLaunch launch = BoardProfileProtocolLaunch::Warmup;
  uint32_t sampleIndex = 0;
};

struct BoardProfileTraceTileAudit {
  uint64_t preflightCount = 0;
  uint64_t nextSequence = 0;
  uint64_t storedEventCount = 0;
  uint32_t droppedEventCount = 0;
  uint32_t recordFlags = 0;
  uint32_t traceState = 0;
};

struct BoardProfileProtocolObservation {
  uint64_t launchToCompletionNanoseconds = 0;
  uint64_t completionObservationResolutionNanoseconds = 0;
  std::array<uint64_t, 16> countSequences{};
  std::array<BoardProfileTraceTileAudit, 16> trace{};
};

struct BoardProfileMeasurementSample {
  std::string id;
  uint32_t sampleIndex = 0;
  uint64_t elapsedNanoseconds = 0;
  uint64_t completionObservationResolutionNanoseconds = 0;
};

struct BoardProfileProtocolResult {
  std::vector<BoardProfileMeasurementSample> samples;
  std::string finalSampleId;
};

enum class BoardProfileOutputValidationMode {
  ExternalExpected,
  Mixed,
  SameSessionProduction,
};

struct BoardProfileOutputValidationResource {
  int64_t logicalRank = -1;
  PackageResourceRole role = PackageResourceRole::Output;
  int64_t roleIndex = -1;
  uint64_t bytes = 0;
  std::string referenceSha256;
  std::optional<BoardOutputComparisonKind> externalExpectedComparison;
  bool productionRepeatsExact = false;
  bool diagnosticCapturesExact = false;
};

llvm::StringRef stringifyBoardProfileOutputValidationMode(
    BoardProfileOutputValidationMode mode);

/// Semantic-keyed output oracle for one board profile campaign. The first
/// uninstrumented production-artifact warmup is validated against every
/// supplied external expected tensor with its exact or relaxed-f16 policy,
/// then staged as the same-session byte reference. Every later result is first
/// checked against its own external expected tensor, then compared with the
/// staged reference by stable `(logical_rank, role, role_index)` and exact
/// typed resource contract.
///
/// Reference files are private to the already-created report staging
/// directory. They are not keyed by ResourceId, resource name, or user path,
/// and must be removed before report publication.
class BoardProfileOutputValidationState {
public:
  explicit BoardProfileOutputValidationState(std::string stagingDirectory);
  ~BoardProfileOutputValidationState();
  BoardProfileOutputValidationState(BoardProfileOutputValidationState &&);
  BoardProfileOutputValidationState &
  operator=(BoardProfileOutputValidationState &&);
  BoardProfileOutputValidationState(const BoardProfileOutputValidationState &) =
      delete;
  BoardProfileOutputValidationState &
  operator=(const BoardProfileOutputValidationState &) = delete;

  llvm::Error establishProductionReference(
      const PackageManifest &manifest, const BoardInvocationFilePlan &plan,
      llvm::ArrayRef<BoardRuntimeOutput> outputs);
  llvm::Error
  validateProductionRepeat(const PackageManifest &manifest,
                           const BoardInvocationFilePlan &plan,
                           llvm::ArrayRef<BoardRuntimeOutput> outputs);
  llvm::Error
  validateDiagnosticCapture(const PackageManifest &manifest,
                            const BoardInvocationFilePlan &plan,
                            llvm::ArrayRef<BoardRuntimeOutput> outputs);
  llvm::Error finalizeAndRemoveReferences();

  BoardProfileOutputValidationMode getMode() const;
  llvm::ArrayRef<BoardProfileOutputValidationResource> getResources() const;

private:
  llvm::Error validateAgainstReference(
      const PackageManifest &manifest, const BoardInvocationFilePlan &plan,
      llvm::ArrayRef<BoardRuntimeOutput> outputs, bool productionRepeat);

  struct Impl;
  std::unique_ptr<Impl> impl;
};

/// Tool-internal deterministic protocol seam. It owns the fixed launch order,
/// count-before-trace admission, per-trace audit checks, and finalization
/// boundary; the callback owns one qualified session and exact output/site
/// validation for each synchronous launch.
llvm::Expected<BoardProfileProtocolResult> runFixedBoardProfileProtocol(
    uint64_t traceCapacity,
    llvm::function_ref<llvm::Expected<BoardProfileProtocolObservation>(
        const BoardProfileProtocolStep &)>
        execute,
    llvm::function_ref<
        llvm::Error(llvm::ArrayRef<BoardProfileMeasurementSample>)>
        finalize);

#if defined(WAFER_PROFILE_CAMPAIGN_TESTING)
namespace testing {

struct ProfileReportPublicationResult {
  std::string runId;
  std::string runDirectory;
  std::string currentEntry;
};

/// Unit-test seam for the filesystem publication transaction. The stage
/// callback must create exactly evidence.json, analysis.json, and index.html;
/// the removal callback permits deterministic old-run deletion failures.
llvm::Expected<ProfileReportPublicationResult>
publishProfileReportForTesting(
    llvm::StringRef companionRoot,
    llvm::function_ref<llvm::Error(llvm::StringRef runId,
                                  llvm::StringRef stagingDirectory)>
        stage,
    llvm::function_ref<std::error_code(llvm::StringRef)> removeManagedRun);

} // namespace testing
#endif

struct BoardProfileCampaignResult {
  BoardRuntimeInvocationResult finalResult;
  BoardInvocationFilePlan finalPlan;
  std::string runDirectory;
};

/// Executes the compiler-owned, fixed profiler campaign in one qualified
/// board session. The caller supplies only the ordinary invocation file plan;
/// all diagnostic capture packages and profiler buffers come from the
/// verified sibling companion.
llvm::Expected<BoardProfileCampaignResult>
runBoardProfileCampaign(const VerifiedProfileCompanion &companion,
                        const PackageManifest &productionManifest,
                        const BoardInvocationFilePlan &productionPlan,
                        BoardRuntimeDriver &driver);

} // namespace wafer::runtime::cli

#endif // WAFER_TOOLS_WAFER_RUN_WAFERPROFILECAMPAIGN_H
