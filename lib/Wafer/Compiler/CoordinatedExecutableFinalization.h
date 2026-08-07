//===- CoordinatedExecutableFinalization.h - Exact all-rank gate -*- C++
//-*-===//

#ifndef WAFER_COMPILER_COORDINATEDEXECUTABLEFINALIZATION_H
#define WAFER_COMPILER_COORDINATEDEXECUTABLEFINALIZATION_H

#include "CoordinatedDataflowSearch.h"
#include "CoordinatedExecutableAdmission.h"
#include "WholeVariantSelection.h"

#include "Wafer/Frontend/Program.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace wafer::compiler::detail {

/// Invocation-local identity of a schedule action's ordering rewrite. This is
/// not persisted in accepted IR or package metadata. It lets qualification
/// distinguish a sibling that was actually derived by serializing a witnessed
/// Direct-DTE/compute window from an unrelated naturally sequential action.
/// Compiler-private selector for focused qualification of one scheduling
/// mechanism. It is not a public optimization policy or composable bitset.
/// Production owns the complete candidate domain; the serialized Direct-DTE
/// seam extends that same domain with its typed qualification siblings.
enum class CoordinatedScheduleActionFamily : uint8_t {
  Production,
  ProductionWithSerializedDirectDTECompute,
  ReadyOrderQualification,
  StaticFixedSlotQualification,
  DisjointWorkerPlacementQualification,
};

enum class CoordinatedExecutableAdmissionFailureKind : uint8_t {
  None,
  RankDomain,
  RankFinalization,
  SPMAllocation,
  WholeVariantExactGate,
  WorkLedger,
};

/// Typed feedback returned to the still-live structured coordinator. It names
/// the failed semantic gate and optional logical rank, but carries no partial
/// placement and cannot mutate or repair the Tile parent.
struct CoordinatedExecutableAdmissionFailure {
  CoordinatedExecutableAdmissionFailureKind kind =
      CoordinatedExecutableAdmissionFailureKind::None;
  int64_t logicalRank = -1;
  std::string gate;
};

struct AdmittedCoordinatedExecutable {
  AdmittedCoordinatedExecutable(
      int64_t stableSemanticOrdinal, bool reservedBaseline,
      AcceptedWholeVariant variant, uint32_t scheduleActionOrdinal = 0,
      CoordinatedScheduleActionIdentity actionIdentity = {},
      bool implementationAlternativeOrigin = false)
      : stableSemanticOrdinal(stableSemanticOrdinal),
        scheduleActionOrdinal(scheduleActionOrdinal),
        actionIdentity(actionIdentity), reservedBaseline(reservedBaseline),
        implementationAlternativeOrigin(implementationAlternativeOrigin),
        variant(std::move(variant)) {}

  AdmittedCoordinatedExecutable(AdmittedCoordinatedExecutable &&) noexcept =
      default;
  AdmittedCoordinatedExecutable &
  operator=(AdmittedCoordinatedExecutable &&) noexcept = default;
  AdmittedCoordinatedExecutable(const AdmittedCoordinatedExecutable &) = delete;
  AdmittedCoordinatedExecutable &
  operator=(const AdmittedCoordinatedExecutable &) = delete;

  int64_t stableSemanticOrdinal = 0;
  uint32_t scheduleActionOrdinal = 0;
  CoordinatedScheduleActionIdentity actionIdentity;
  bool reservedBaseline = false;
  bool implementationAlternativeOrigin = false;
  AcceptedWholeVariant variant;
};

/// Stable digest of the actual admitted Instr modules and their coordinated
/// identities. This is a test/diagnostic projection only; it is never an
/// artifact key or a substitute for exact admission.
std::string computeAdmittedCoordinatedExecutableFrontierDigest(
    llvm::ArrayRef<AdmittedCoordinatedExecutable> frontier);

/// One finite all-rank scheduling action materialized from canonical unplaced
/// Instr parents. The action owns exactly one sibling for every logical rank;
/// no rank-local frontier or cross-rank product is exposed.
struct CoordinatedScheduleAction {
  uint32_t stableOrdinal = 0;
  CoordinatedReadyOrderKind readyOrderKind =
      CoordinatedReadyOrderKind::Canonical;
  CoordinatedBufferingKind bufferingKind = CoordinatedBufferingKind::Single;
  uint32_t bufferingPlanOrdinal = 0;
  CoordinatedWorkerPlacementKind workerPlacementKind =
      CoordinatedWorkerPlacementKind::Unplaced;
  uint32_t workerPlacementPlanOrdinal = 0;
  CoordinatedScheduleSerializationKind serializationKind =
      CoordinatedScheduleSerializationKind::Unchanged;
  std::optional<CoordinatedCommunicationActionPointIdentity>
      communicationPointIdentity;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> rankModules;
};

/// Stable identity of the cheap pre-allocation action model. The model keeps
/// separate structural values instead of fabricating a time or hardware cost.
/// It is valid only for coverage pruning; final Instr recost remains the
/// selection owner.
inline constexpr llvm::StringLiteral
    kCoordinatedScheduleActionEstimateModelIdentity =
        "executable-schedule-recipe-structural-coverage-v2";

enum class CoordinatedScheduleActionEstimateDimension : uint8_t {
  QualifiedOverlapWindows,
  DirectDTEOverlapWindows,
  SteadyStateParticipantWaits,
  InstructionSites,
  EventSites,
  MaximumDependencyDepth,
  ReadyOrderInversions,
  SPMMovementBytes,
  Count,
};

inline constexpr size_t kCoordinatedScheduleActionEstimateDimensionCount =
    static_cast<size_t>(CoordinatedScheduleActionEstimateDimension::Count);

/// One disposable action's current-IR-derived estimate. Missing values stay
/// explicit; a known value on another dimension may still make the action the
/// representative of its typed scheduling shape.
struct CoordinatedScheduleActionEstimate {
  uint32_t stableOrdinal = 0;
  CoordinatedReadyOrderKind readyOrderKind =
      CoordinatedReadyOrderKind::Canonical;
  CoordinatedBufferingKind bufferingKind = CoordinatedBufferingKind::Single;
  CoordinatedWorkerPlacementKind workerPlacementKind =
      CoordinatedWorkerPlacementKind::Unplaced;
  CoordinatedScheduleSerializationKind serializationKind =
      CoordinatedScheduleSerializationKind::Unchanged;
  std::optional<CoordinatedCommunicationActionPointIdentity>
      communicationPointIdentity;
  std::string fixedSlotCoverageIdentity;
  std::array<std::optional<uint64_t>,
             kCoordinatedScheduleActionEstimateDimensionCount>
      values;
};

/// Selects a deterministic bounded coverage beam. The mandatory ordinal-zero
/// baseline and one best model estimate per typed action shape are retained;
/// spare capacity is filled by the remaining model order. Input order is not a
/// tie-break.
mlir::FailureOr<llvm::SmallVector<uint32_t, 8>>
selectCoordinatedScheduleActionEstimateBeam(
    llvm::ArrayRef<CoordinatedScheduleActionEstimate> estimates);

/// Invocation-local accounting for the recipe-first schedule search. Read-only
/// enumeration owns no IR. Every attempted complete-rank recipe charges one
/// rank clone per parent even when materialization or the downstream exact
/// consumer rejects it; successfulActionClones counts only accepted actions.
struct CoordinatedScheduleRecipeStatistics {
  uint64_t enumeratedRecipes = 0;
  uint64_t coverageRetainedRecipes = 0;
  uint64_t materializationAttempts = 0;
  uint64_t materializationFailures = 0;
  uint64_t materializationBackfills = 0;
  uint64_t successfulActionClones = 0;
  uint64_t actualRankClones = 0;
};

enum class CoordinatedExecutableFinalizationStepKind : uint8_t {
  Accepted,
  RecoverableRejected,
  Exhausted,
};

enum class CoordinatedExecutableFinalizationLane : uint8_t {
  Seed,
  Expansion,
};

/// Pure invocation-local policy for the expensive part of executable
/// finalization. The mandatory baseline is recorded outside lane rotation;
/// every later materialized action, including an exact rejection, advances
/// the stable A/B turn. Availability may temporarily force one lane without
/// losing the pending preference for the other lane.
class CoordinatedExecutableFinalizationLaneCoordinator {
public:
  mlir::LogicalResult recordMandatoryBaselineAccepted();

  std::optional<CoordinatedExecutableFinalizationLane>
  chooseNextLane(bool seedAvailable, bool expansionAvailable) const;

  mlir::LogicalResult
  recordAttempt(CoordinatedExecutableFinalizationLane lane,
                bool exactAccepted);

  bool canAttempt() const;
  uint32_t getAttemptCount() const { return attemptCount; }
  uint32_t getExactAcceptedCount() const { return exactAcceptedCount; }

private:
  bool baselineAccepted = false;
  bool preferSeed = true;
  uint32_t attemptCount = 0;
  uint32_t exactAcceptedCount = 0;
};

struct CoordinatedExecutableFinalizationStep {
  CoordinatedExecutableFinalizationStepKind kind =
      CoordinatedExecutableFinalizationStepKind::Exhausted;
  std::optional<AdmittedCoordinatedExecutable> admitted;
  CoordinatedExecutableAdmissionFailure failure;
};

/// Setup rejection is recoverable only when it describes this Tile candidate,
/// not a rank-domain or work-ledger protocol violation.
bool isRecoverableCoordinatedExecutableSetupFailure(
    const CoordinatedExecutableAdmissionFailure &failure);

/// Invocation-local resumable owner for one exact-seeded Tile variant. It owns
/// canonical Instr parents and a read-only ordered recipe cursor, never a
/// persistent action clone. Borrowed program/config/provider inputs must
/// outlive it. Production interleaves these cursors so one Tile variant cannot
/// consume the invocation-wide schedule-attempt budget.
class CoordinatedExecutableFinalizationCursor {
public:
  ~CoordinatedExecutableFinalizationCursor();
  CoordinatedExecutableFinalizationCursor(
      CoordinatedExecutableFinalizationCursor &&) noexcept;
  CoordinatedExecutableFinalizationCursor &
  operator=(CoordinatedExecutableFinalizationCursor &&) noexcept;
  CoordinatedExecutableFinalizationCursor(
      const CoordinatedExecutableFinalizationCursor &) = delete;
  CoordinatedExecutableFinalizationCursor &
  operator=(const CoordinatedExecutableFinalizationCursor &) = delete;

  int64_t getStableSemanticOrdinal() const;
  bool isReservedBaseline() const;
  size_t getCanonicalParentCount() const;
  bool seedAttempted() const;
  bool exactSeedAccepted() const;
  bool exhausted() const;

private:
  struct Impl;
  explicit CoordinatedExecutableFinalizationCursor(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;

  friend mlir::FailureOr<
      std::unique_ptr<CoordinatedExecutableFinalizationCursor>>
  beginCoordinatedExecutableFinalization(
      const CoordinatedTileVariant &,
      const frontend::FrontendProgramVerificationResult &,
      const ExecutionConfig &, const OptimizationConfig &,
      CoordinatedWorkLedger &, llvm::raw_ostream &,
      CoordinatedExecutableAdmissionFailure &,
      WholeVariantSelectionStatistics *, WholeVariantSelectionMode,
      unsigned,
      llvm::ArrayRef<const CoordinatedCommunicationActionProvider *>);
  friend mlir::FailureOr<CoordinatedExecutableFinalizationStep>
  advanceCoordinatedExecutableFinalization(
      CoordinatedExecutableFinalizationCursor &,
      ExecutableFinalizationReservation);
};

mlir::FailureOr<std::unique_ptr<CoordinatedExecutableFinalizationCursor>>
beginCoordinatedExecutableFinalization(
    const CoordinatedTileVariant &tileVariant,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig,
    const OptimizationConfig &optimizations, CoordinatedWorkLedger &ledger,
    llvm::raw_ostream &diagnostics,
    CoordinatedExecutableAdmissionFailure &failure,
    WholeVariantSelectionStatistics *statistics = nullptr,
    WholeVariantSelectionMode selectionMode =
        WholeVariantSelectionMode::Production,
    unsigned rankPipelineParallelism = 0,
    llvm::ArrayRef<const CoordinatedCommunicationActionProvider *>
        communicationProviders = {});

/// Advances exactly one recipe. The caller supplies either the Tile seed's
/// reservation for the first canonical action or one invocation-global
/// expansion reservation returned by the shared work ledger.
mlir::FailureOr<CoordinatedExecutableFinalizationStep>
advanceCoordinatedExecutableFinalization(
    CoordinatedExecutableFinalizationCursor &cursor,
    ExecutableFinalizationReservation reservation);

/// Result of the downstream exact consumer for one materialized complete-rank
/// action. A recoverable rejection keeps the recipe attempt charged and asks
/// the ordered unmaterialized domain to backfill; failure is reserved for a
/// fatal invariant or mandatory-baseline rejection.
enum class CoordinatedScheduleActionConsumption : uint8_t {
  Accepted,
  RecoverableRejection,
};

/// Enumerates a typed read-only recipe domain, retains a structural coverage
/// beam, and materializes only retained recipes plus bounded deterministic
/// backfill. The consumer may reject a non-baseline exact action without
/// aborting the domain walk; the mandatory ordinal-zero baseline fails closed.
mlir::LogicalResult walkCoordinatedScheduleActions(
    llvm::ArrayRef<mlir::ModuleOp> canonicalInstrParents,
    const OptimizationConfig &optimizations,
    llvm::function_ref<mlir::FailureOr<CoordinatedScheduleActionConsumption>(
        CoordinatedScheduleAction &&)>
        consume,
    uint64_t *materializationWork = nullptr, bool baselineOnly = false,
    CoordinatedScheduleActionFamily family =
        CoordinatedScheduleActionFamily::Production,
    CoordinatedScheduleRecipeStatistics *statistics = nullptr,
    const frontend::FrontendProgramVerificationResult *program = nullptr,
    llvm::ArrayRef<const CoordinatedCommunicationActionProvider *>
        communicationProviders = {});

/// Derives the bounded complete-rank ready-order, fixed-slot, and worker
/// siblings from canonical unplaced Instr parents. A requested action is
/// published only when the same typed action succeeds for all ranks.
mlir::FailureOr<std::vector<CoordinatedScheduleAction>>
deriveCoordinatedScheduleActions(
    llvm::ArrayRef<mlir::ModuleOp> canonicalInstrParents,
    const OptimizationConfig &optimizations,
    uint64_t *materializationWork = nullptr,
    CoordinatedScheduleActionFamily family =
        CoordinatedScheduleActionFamily::Production,
    CoordinatedScheduleRecipeStatistics *statistics = nullptr,
    const frontend::FrontendProgramVerificationResult *program = nullptr,
    llvm::ArrayRef<const CoordinatedCommunicationActionProvider *>
        communicationProviders = {});

/// Compatibility helper for focused tests and legacy compiler-private callers.
/// Production uses the resumable cursor plus the invocation lane coordinator
/// above. This helper evaluates one complete all-rank Tile candidate on
/// disposable clones. Tile is
/// lowered once per rank. A read-only recipe domain is structurally pruned
/// before any schedule clone; retained and bounded backfill actions then pass
/// the same all-rank finalization, DDR, transport, resource, ABI, and final
/// recost gates. Success consumes the candidate's coordinator reservation
/// atomically; failure also closes the attempted reservation with actual work
/// while leaving the original Tile parent available for a newly reserved
/// coordinator-owned repair sibling.
mlir::FailureOr<std::vector<AdmittedCoordinatedExecutable>>
finalizeCoordinatedTileVariant(
    const CoordinatedTileVariant &tileVariant,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig,
    const OptimizationConfig &optimizations, CoordinatedWorkLedger &ledger,
    llvm::raw_ostream &diagnostics,
    CoordinatedExecutableAdmissionFailure &failure,
    WholeVariantSelectionStatistics *statistics = nullptr,
    WholeVariantSelectionMode selectionMode =
        WholeVariantSelectionMode::Production,
    const analysis::WholeCardInstructionProgramCost *productionBaselineCost =
        nullptr,
    unsigned rankPipelineParallelism = 0,
    llvm::ArrayRef<const CoordinatedCommunicationActionProvider *>
        communicationProviders = {});

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_COORDINATEDEXECUTABLEFINALIZATION_H
