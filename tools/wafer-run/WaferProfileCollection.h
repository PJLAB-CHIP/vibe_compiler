//===- WaferProfileCollection.h - Board profile collection -----*- C++ -*-===//

#ifndef WAFER_TOOLS_WAFER_RUN_WAFERPROFILECOLLECTION_H
#define WAFER_TOOLS_WAFER_RUN_WAFERPROFILECOLLECTION_H

#include "Wafer/Runtime/ProfileInstrumentation.h"
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
  Primary,
  Count,
  Trace,
};

struct BoardProfileProtocolStep {
  BoardProfileProtocolLaunch launch = BoardProfileProtocolLaunch::Primary;
};

struct BoardProfileTraceTileAudit {
  uint64_t countedEventCount = 0;
  uint64_t nextSequence = 0;
  uint64_t storedEventCount = 0;
  uint32_t droppedEventCount = 0;
  uint32_t recordFlags = 0;
  uint32_t traceState = 0;
};

struct BoardProfileProtocolObservation {
  std::optional<uint64_t> deviceExecutionNanoseconds;
  uint64_t hostSubmitNanoseconds = 0;
  uint64_t launchToCompletionNanoseconds = 0;
  uint64_t completionObservationResolutionNanoseconds = 0;
  std::array<uint64_t, 16> countSequences{};
  std::array<BoardProfileTraceTileAudit, 16> trace{};
};

struct BoardProfileMeasurementSample {
  std::string id;
  uint32_t sampleIndex = 0;
  uint64_t deviceElapsedNanoseconds = 0;
  std::string deviceTimerKind;
  uint64_t hostSubmitNanoseconds = 0;
  uint64_t hostLaunchToCompletionNanoseconds = 0;
  uint64_t completionObservationResolutionNanoseconds = 0;
};

struct BoardProfileProtocolResult {
  std::vector<BoardProfileMeasurementSample> samples;
  std::string primarySampleId;
};

enum class BoardProfileOutputValidationMode {
  ExternalExpected,
  Mixed,
  SameSessionPrimary,
};

struct BoardProfileOutputValidationResource {
  PackageResourceScope scope;
  PackageResourceRole role = PackageResourceRole::Output;
  int64_t roleIndex = -1;
  uint64_t bytes = 0;
  std::string referenceSha256;
  std::optional<BoardOutputComparisonKind> externalExpectedComparison;
  bool primaryOutputValidated = false;
  bool diagnosticCapturesMatchPrimary = false;
};

llvm::StringRef stringifyBoardProfileOutputValidationMode(
    BoardProfileOutputValidationMode mode);

/// Semantic-keyed output validator for one board profile collection. The single
/// uninstrumented profiled-package execution is validated against every
/// supplied external expected tensor with its exact or relaxed-f16 policy,
/// then stored as the same-session byte reference. Count and trace results are
/// first checked against their own external expected tensors, then compared
/// with the staged primary reference by stable
/// `(scope, role, role_index)` and exact typed resource contract.
///
/// Reference files are private to the report's temporary directory. They are
/// not keyed by ResourceId, resource name, or user path, and must be removed
/// before that directory receives its final name.
class BoardProfileOutputValidator {
public:
  explicit BoardProfileOutputValidator(std::string stagingDirectory);
  ~BoardProfileOutputValidator();
  BoardProfileOutputValidator(BoardProfileOutputValidator &&);
  BoardProfileOutputValidator &operator=(BoardProfileOutputValidator &&);
  BoardProfileOutputValidator(const BoardProfileOutputValidator &) = delete;
  BoardProfileOutputValidator &
  operator=(const BoardProfileOutputValidator &) = delete;

  llvm::Error recordPrimaryOutputs(const PackageManifest &manifest,
                                   const BoardInvocationFilePlan &plan,
                                   llvm::ArrayRef<BoardRuntimeOutput> outputs);
  llvm::Error
  validateDiagnosticOutputs(const PackageManifest &manifest,
                            const BoardInvocationFilePlan &plan,
                            llvm::ArrayRef<BoardRuntimeOutput> outputs);
  llvm::Error removeReferenceFiles();

  BoardProfileOutputValidationMode getMode() const;
  llvm::ArrayRef<BoardProfileOutputValidationResource> getResources() const;

private:
  llvm::Error
  compareWithPrimaryOutputs(const PackageManifest &manifest,
                            const BoardInvocationFilePlan &plan,
                            llvm::ArrayRef<BoardRuntimeOutput> outputs);

  struct Impl;
  std::unique_ptr<Impl> impl;
};

/// Runs the fixed primary/count/trace sequence, checks the count before the
/// trace launch, and verifies every trace record. The callbacks execute one
/// qualified board session and consume the verified measurements.
llvm::Expected<BoardProfileProtocolResult> runFixedBoardProfileProtocol(
    uint64_t traceCapacity,
    llvm::function_ref<llvm::Expected<BoardProfileProtocolObservation>(
        const BoardProfileProtocolStep &)>
        execute,
    llvm::function_ref<
        llvm::Error(llvm::ArrayRef<BoardProfileMeasurementSample>)>
        consumeMeasurements);

#if defined(WAFER_PROFILE_COLLECTION_TESTING)
namespace testing {

struct ProfileReportWriteResult {
  std::string runId;
  std::string runDirectory;
  std::string currentEntry;
};

/// Unit-test seam for the staged report-directory update. The stage
/// callback must create exactly evidence.json, analysis.json, and index.html;
/// the removal callback permits deterministic old-run deletion failures.
llvm::Expected<ProfileReportWriteResult> writeProfileReportForTesting(
    llvm::StringRef instrumentationRoot,
    llvm::function_ref<llvm::Error(llvm::StringRef runId,
                                   llvm::StringRef stagingDirectory)>
        stage,
    llvm::function_ref<std::error_code(llvm::StringRef)> removeManagedRun);

} // namespace testing
#endif

struct BoardProfileCollectionResult {
  BoardRuntimeInvocationResult finalResult;
  BoardInvocationFilePlan finalPlan;
  std::string runDirectory;
};

/// Executes the compiler-owned, fixed profiler collection in one qualified
/// board session. The caller supplies only the ordinary invocation file plan;
/// all diagnostic capture packages and profiler buffers come from the
/// verified sibling instrumentation.
llvm::Expected<BoardProfileCollectionResult>
runBoardProfileCollection(const VerifiedProfileInstrumentation &instrumentation,
                          const PackageManifest &primaryManifest,
                          const BoardInvocationFilePlan &primaryPlan,
                          BoardRuntimeDriver &driver);

} // namespace wafer::runtime::cli

#endif // WAFER_TOOLS_WAFER_RUN_WAFERPROFILECOLLECTION_H
