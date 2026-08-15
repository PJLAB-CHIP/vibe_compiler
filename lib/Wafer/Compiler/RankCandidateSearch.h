//===- RankCandidateSearch.h - Rank candidate search -*- C++ -*-===//

#ifndef WAFER_COMPILER_RANKCANDIDATESEARCH_H
#define WAFER_COMPILER_RANKCANDIDATESEARCH_H

#include "Wafer/Support/OptimizationConfig.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

class StructuredImplementationAlternativeProvider;

/// The schedule action family is finite: original/ready order, the canonical
/// bounded fixed-slot loop identities, original/disjoint worker placement, and
/// qualification-only serialized Direct-DTE/compute siblings. Production does
/// not enumerate the serialized family. Qualification adds those recipes to
/// the same read-only domain and remains subject to the shared attempt and
/// exact-success caps below.
inline constexpr uint32_t kMaximumCoordinatedFixedSlotActions = 8;
inline constexpr uint32_t kMaximumCoordinatedSerializedScheduleActions =
    2 * kMaximumCoordinatedFixedSlotActions;
inline constexpr uint32_t kMaximumCoordinatedScheduleActions =
    2 * (1 + kMaximumCoordinatedFixedSlotActions) * 2 +
    kMaximumCoordinatedSerializedScheduleActions;
/// Recipe selection retains at most eight exact successes. Materialization and
/// downstream exact rejection may consume a deterministic second beam of
/// ordered backfill recipes, but never traverse the full finite recipe domain.
/// The production lane coordinator applies these invocation-wide limits across
/// every Tile variant; an individual executable-evaluation cursor never
/// restarts them. Standalone compatibility walkers use the same numeric bounds
/// for their local walk but are not the production invocation owner.
inline constexpr uint32_t kMaximumCoordinatedExactScheduleActions = 8;
inline constexpr uint32_t
    kMaximumCoordinatedScheduleRecipeMaterializationAttempts = 16;
/// A persistent cursor owns only canonical Instr parents and a read-only recipe
/// cursor. Every live cursor has already consumed one exact-success slot, so
/// this explicit bound is also implied by the invocation-wide success limit.
inline constexpr uint32_t kMaximumLiveCandidateEvaluations = 8;
inline constexpr uint32_t kMaximumCoordinatedConnectionDPStates = 8;
inline constexpr uint32_t kMaximumCoordinatedConnectionDAGBeamStates = 8;
inline constexpr uint32_t kMaximumCoordinatedConnectionExpansionsPerStep = 8;
/// The pre-materialization state is bounded independently from live actual IR.
/// Production pulls at most one actual Tile parent at a time, while the
/// compatibility wrapper may retain this many successfully materialized
/// representatives for legacy unit/integration consumers.
inline constexpr uint32_t kMaximumStructuredCandidateStates = 64;
inline constexpr uint32_t kMaximumCoordinatedSuccessfulActualCandidates = 16;
inline constexpr char kCoordinatedConnectionProposalEstimateModel[] =
    "connection-choice-byte-residency-ddr-fragmentation";

/// Stable semantic work classes charged to the one invocation-local search
/// ledger. They are compiler scheduling policy, not IR or output descriptors.
enum class CoordinatedWorkKind : uint8_t {
  StructuredExpansion,
  ActualTileClone,
  TileToInstrLowering,
  ExecutableScheduleAction,
  SPMAllocationProblem,
  DDRAllocationDomain,
  TransportValidation,
  ABIValidation,
  RepairExpansion,
  Count,
};

inline constexpr size_t kCoordinatedWorkKindCount =
    static_cast<size_t>(CoordinatedWorkKind::Count);

struct CoordinatedWorkEstimate {
  std::array<uint64_t, kCoordinatedWorkKindCount> counts = {};

  uint64_t get(CoordinatedWorkKind kind) const {
    return counts[static_cast<size_t>(kind)];
  }
  void set(CoordinatedWorkKind kind, uint64_t value) {
    counts[static_cast<size_t>(kind)] = value;
  }
  std::optional<uint64_t> getTotal() const;
};

struct CandidateEvaluationReservation {
  uint64_t id = 0;
};

struct CoordinatedWorkLedgerSnapshot {
  uint64_t capacity = 0;
  uint64_t consumed = 0;
  uint64_t mandatoryGenerationReserved = 0;
  uint64_t evaluationWorkReserved = 0;
  uint64_t repairReserved = 0;
  uint64_t unreserved = 0;
  uint64_t scheduleAttemptCapacity =
      kMaximumCoordinatedScheduleRecipeMaterializationAttempts;
  uint64_t scheduleAttemptsReserved = 0;
  uint64_t scheduleAttemptsConsumed = 0;
  std::array<uint64_t, kCoordinatedWorkKindCount> consumedByKind = {};
};

/// One deterministic invocation-level work ledger shared by structured
/// generation, executable evaluation, and repair. Mandatory baseline
/// generation and executable evaluation plus a finite repair allowance are
/// reserved in the constructor, before any candidate is generated. Optional
/// generation cannot borrow those credits.
class CoordinatedWorkLedger {
public:
  static constexpr uint64_t kDefaultCapacity = 262144;
  static constexpr uint64_t kDefaultRepairReserve = 16384;

  static mlir::FailureOr<CoordinatedWorkLedger>
  create(int64_t rankCount, uint64_t capacity = kDefaultCapacity,
         uint64_t repairReserve = kDefaultRepairReserve);

  int64_t getRankCount() const { return rankCount; }
  CandidateEvaluationReservation getMandatoryBaselineReservation() const {
    return mandatoryBaselineReservation;
  }

  /// Closes the pre-reserved baseline generation allowance with actual work.
  /// Every actual count must be no greater than the deterministic upper bound
  /// reserved at construction.
  mlir::LogicalResult
  completeMandatoryBaselineGeneration(const CoordinatedWorkEstimate &actual);

  /// Consumes unreserved optional generation credits. This never consumes a
  /// evaluation or repair reservation.
  bool tryConsumeGeneration(const CoordinatedWorkEstimate &actual);
  bool tryConsumeGeneration(CoordinatedWorkKind kind, uint64_t credits = 1);

  /// Reserves one complete all-rank schedule action before admitting its
  /// actual Tile clone to the coordinated candidates.
  std::optional<CandidateEvaluationReservation>
  tryReserveCandidateEvaluation(const CoordinatedWorkEstimate &upperBound);

  /// Reserves one expansion action from the invocation-wide schedule-attempt
  /// budget. Unlike a Tile seed reservation, this contains no Tile-to-Instr
  /// setup work.
  std::optional<CandidateEvaluationReservation>
  tryReserveExecutableScheduleAttempt();

  /// Closes one attempted schedule action after exact evaluation accepts or
  /// rejects it. Unused upper-bound credits are released; an
  /// actual count beyond any reserved class fails.
  mlir::LogicalResult
  completeCandidateEvaluation(CandidateEvaluationReservation reservation,
                              const CoordinatedWorkEstimate &actual);

  /// Releases a rejected non-mandatory candidate before executable
  /// evaluation. The mandatory baseline reservation cannot be released.
  mlir::LogicalResult
  releaseCandidateEvaluation(CandidateEvaluationReservation reservation);

  /// Consumes the pre-reserved repair allowance. It cannot borrow mandatory
  /// baseline or admitted executable-evaluation credits.
  bool tryConsumeRepair(uint64_t credits = 1);
  bool tryConsumeRepair(const CoordinatedWorkEstimate &actual);

  CoordinatedWorkLedgerSnapshot getSnapshot() const;

  static mlir::FailureOr<CoordinatedWorkEstimate>
  getCandidateEvaluationUpperBound(int64_t rankCount);
  static mlir::FailureOr<CoordinatedWorkEstimate>
  getExecutableScheduleAttemptUpperBound(int64_t rankCount);

private:
  struct CandidateEvaluationReservationRecord {
    uint64_t id = 0;
    bool mandatoryBaseline = false;
    uint32_t scheduleAttemptSlots = 0;
    CoordinatedWorkEstimate upperBound;
  };

  CoordinatedWorkLedger(int64_t rankCount, uint64_t capacity,
                        uint64_t repairReserve,
                        CoordinatedWorkEstimate generationUpperBound,
                        CoordinatedWorkEstimate evaluationUpperBound);

  static mlir::FailureOr<CoordinatedWorkEstimate>
  getMandatoryGenerationUpperBound(int64_t rankCount);
  CandidateEvaluationReservationRecord *findReservation(uint64_t id);
  const CandidateEvaluationReservationRecord *
  findReservation(uint64_t id) const;
  bool canReserve(uint64_t credits) const;
  void addConsumed(const CoordinatedWorkEstimate &actual);

  int64_t rankCount = 0;
  uint64_t capacity = 0;
  uint64_t consumed = 0;
  uint64_t mandatoryGenerationReserved = 0;
  uint64_t repairReserved = 0;
  uint64_t scheduleAttemptsReserved = 0;
  uint64_t scheduleAttemptsConsumed = 0;
  uint64_t nextReservationId = 1;
  CoordinatedWorkEstimate mandatoryGenerationUpperBound;
  std::array<uint64_t, kCoordinatedWorkKindCount> consumedByKind = {};
  std::vector<CandidateEvaluationReservationRecord>
      candidateEvaluationReservations;
  CandidateEvaluationReservation mandatoryBaselineReservation;
};

/// One real rank entry in an all-rank Tile candidate. The module is the only
/// semantic representation. selectedTileIR is optional immutable debug text for
/// externally constructed fixtures; production candidates construction leaves
/// it empty and the executable finalizer captures the current actual clone only
/// when an exact action needs to build an executable.
struct CoordinatedRankTileModule {
  CoordinatedRankTileModule(int64_t logicalRank,
                            mlir::OwningOpRef<mlir::ModuleOp> module,
                            std::shared_ptr<const std::string> selectedTileIR)
      : logicalRank(logicalRank), module(std::move(module)),
        selectedTileIR(std::move(selectedTileIR)) {}

  CoordinatedRankTileModule(CoordinatedRankTileModule &&) = default;
  CoordinatedRankTileModule &operator=(CoordinatedRankTileModule &&) = default;
  CoordinatedRankTileModule(const CoordinatedRankTileModule &) = delete;
  CoordinatedRankTileModule &
  operator=(const CoordinatedRankTileModule &) = delete;

  int64_t logicalRank = 0;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::shared_ptr<const std::string> selectedTileIR;
};

struct CoordinatedTileVariant {
  int64_t stableSemanticOrdinal = 0;
  bool reservedBaseline = false;
  /// Query-local provenance used only by final winner diagnostics. It records
  /// that the actual Tile clone originated from a typed implementation
  /// alternative provider; provider identity and algorithm details are not
  /// carried through the common coordinator.
  bool implementationAlternativeOrigin = false;
  uint8_t repairDepth = 0;
  CandidateEvaluationReservation evaluationReservation;
  std::vector<CoordinatedRankTileModule> ranks;
  std::string candidateSetDigest;
};

using RankCandidateSet = std::vector<CoordinatedTileVariant>;

/// Knowledge carried by one pre-Instr selection dimension. Estimated values
/// carry an explicit stable model identity and are comparable only under the
/// same model. Unknown facts are comparable only when their current-IR-derived
/// disposition is identical; they are never interpreted as zero.
enum class StructuredCandidateMetricKnowledge : uint8_t {
  Known,
  Estimated,
  Unknown,
  Unsupported,
  Overflow,
};

struct StructuredCandidateMetric {
  uint64_t value = 0;
  StructuredCandidateMetricKnowledge knowledge =
      StructuredCandidateMetricKnowledge::Known;
  /// Stable model identity for Estimated values, or current-IR-derived
  /// identity for another non-Known disposition. It is transient search
  /// state, not an IR attribute or serialized program field.
  std::string disposition;
};

/// Full structured-stage projection of the selection-sensitive dimensions in
/// tasks/06. SPM high-water is deliberately not an execution dimension: the
/// SPM allocator remains the hard capacity authority.
enum class StructuredCandidateCostDimension : uint8_t {
  DDRAggregateReadBytes,
  DDRAggregateWriteBytes,
  DDRMaximumRankIssueWork,
  LocalAggregateMovementBytes,
  LocalMaximumRankMovementBytes,
  CompletionMaximumRankWaitWork,
  CriticalPathLowerBound,
  NoCAggregatePayloadBytes,
  NoCLinkWork,
  NoCEndpointWork,
  SPMAggregateMovementBytes,
  SPMMaximumRankMovementBytes,
  ComputeAggregateWork,
  ComputeMaximumRankWork,
  RecomputeAggregateWork,
  TileUnderutilization,
  InstrAggregateWork,
  DescriptorPressure,
  ResourcePressure,
  AllRankCouplingWork,
  Count,
};

inline constexpr size_t kStructuredCandidateCostDimensionCount =
    static_cast<size_t>(StructuredCandidateCostDimension::Count);

struct StructuredCandidateCost {
  std::array<StructuredCandidateMetric, kStructuredCandidateCostDimensionCount>
      selection;
  /// Facts that must remain identical before DP/beam states can be compared.
  /// These digests are derived from types, SSA/control structure and typed op
  /// parameters; source symbol/value names and candidate ordinals are absent.
  std::string futureLiveInterface;
  std::string physicalVersions;
  std::string loopReuseAndEffects;
  std::string collectiveAndPeerInterface;
};

enum class StructuredCandidateDominance : uint8_t {
  LeftDominates,
  RightDominates,
  Equivalent,
  Incomparable,
};

struct StructuredCandidateCostView {
  int64_t stableSemanticOrdinal = 0;
  bool reservedBaseline = false;
  const StructuredCandidateCost *facts = nullptr;
};

struct StructuredParetoInsertion {
  bool retainCandidate = false;
  llvm::SmallVector<size_t, 8> eraseIndices;
};

/// Re-derives all transient structured candidates facts from the actual
/// complete-rank Tile clone. No proposal object, name convention, evaluation
/// offset or historical schedule participates.
mlir::FailureOr<StructuredCandidateCost>
deriveStructuredCandidateCost(const CoordinatedTileVariant &variant);

/// Verifies shared collective parameters and all-and-only peer message
/// matching directly across the actual rank Tile modules. This is a C2
/// tuple-legality check; physical routes, offsets, FSMs and waits remain C3.
mlir::LogicalResult
verifyCoordinatedStructuredCommunication(const CoordinatedTileVariant &variant,
                                         int64_t expectedRankCount);

StructuredCandidateDominance
compareStructuredCandidateCost(const StructuredCandidateCost &left,
                               const StructuredCandidateCost &right);

/// Returns false only when current Tile IR already proves a regression in a
/// production promotion dimension whose work is exact and invariant under
/// schedule-action. Unknown facts remain viable; this is not a winner or
/// a replacement for final admitted-executable hardware selection.
bool canStillSatisfyCoordinatedProductionPromotion(
    const StructuredCandidateCost &candidate,
    const StructuredCandidateCost &baseline);

/// Pure deterministic insertion for complete-rank structured states. The
/// reserved baseline is never erased. Exact equivalent states use semantic
/// ordinal only as the final tie-break. Future incremental DP/beam solvers may
/// reuse this primitive, but this operation does not itself implement either
/// solver.
mlir::FailureOr<StructuredParetoInsertion> planStructuredParetoInsertion(
    StructuredCandidateCostView candidate,
    llvm::ArrayRef<StructuredCandidateCostView> existing);

struct StructuredCandidateSearchStatistics {
  uint64_t derivedCandidates = 0;
  uint64_t materializedCandidates = 0;
  uint64_t candidateMaterializationFailures = 0;
  uint64_t promotionIneligibleCandidatesRejected = 0;
  uint64_t equivalentCandidatesRejected = 0;
  uint64_t dominatedCandidatesRejected = 0;
  uint64_t dominatedStatesErased = 0;
  uint64_t evaluationReservationDenied = 0;
  uint64_t generationReservationDenied = 0;
  uint64_t structuredConnections = 0;
  uint64_t connectionProposalExpansions = 0;
  uint64_t connectionActionMaterializations = 0;
  uint64_t connectionDPStatesMerged = 0;
  uint64_t connectionDominatedStatesPruned = 0;
  uint64_t connectionBeamStatesPruned = 0;
  uint64_t connectionLocalChoicesPruned = 0;
  uint64_t connectionIncompatibleSharedVersionsPruned = 0;
  uint64_t maximumConnectionLocalChoices = 0;
  uint64_t maximumConnectionExpansionsPerParent = 0;
  uint64_t maximumConnectionStates = 0;
  uint64_t retainedConnectionCandidates = 0;
  uint64_t connectionWorkerCount = 1;
  bool usedGeneralDAGBeam = false;
  uint64_t retainedStates = 0;
  uint64_t structuralProposalsDerived = 0;
  uint64_t structuralEquivalentProposalsMerged = 0;
  uint64_t structuralDominatedProposalsPruned = 0;
  uint64_t structuralCoverageBeamPruned = 0;
  uint64_t structuralPendingPeak = 0;
  uint64_t structuralPendingAtStop = 0;
  uint64_t implementationAlternativeQueries = 0;
  uint64_t implementationAlternativeProposals = 0;
  uint64_t implementationAlternativeMaterializations = 0;
  uint64_t retainedImplementationAlternativeCandidates = 0;
  uint64_t retainedActualCandidates = 0;
  uint64_t actualRankClones = 0;
  uint64_t actualCandidateAttempts = 0;
  uint64_t successfulActualCandidates = 0;
  uint64_t exactFailureBackfills = 0;
  uint64_t materializationFailureBackfills = 0;
  uint64_t peakLiveActualCandidates = 0;
  bool structuralBudgetExhausted = false;
  std::string structuredCandidateSetDigest;
  std::string actualCandidateSetDigest;
};

enum class CoordinatedTileRepairAction : uint8_t {
  SelectiveSpill,
  SplitAtExplicitDDRBoundary,
};

inline constexpr uint8_t kMaximumCoordinatedRepairDepth = 1;

struct RankCandidateSearchConfig {
  int64_t rankCount = 0;
  int64_t candidateParallelism = 1;
  OptimizationConfig optimizations = OptimizationConfig::production();
  bool reservedBaselineOnly = false;
  uint32_t maximumSuccessfulActualCandidates =
      kMaximumCoordinatedSuccessfulActualCandidates;
  /// Borrowed, compiler-private capability providers. Their lifetime must
  /// cover the search session. Common search treats every returned point as an
  /// opaque typed recipe and never recovers semantics from provider keys.
  std::vector<const StructuredImplementationAlternativeProvider *>
      implementationAlternativeProviders;
};

/// How the sole live actual Tile parent left the coordinator. Exact
/// dispositions are reported after executable evaluation closes its work
/// reservation. The compatibility disposition transfers the still-live
/// reservation to the legacy returned candidates.
enum class CoordinatedActualCandidateDisposition : uint8_t {
  ExactAccepted,
  ExactRejected,
  RetainedForCompatibility,
};

/// Invocation-local search over structural candidates and actual clones, with
/// deterministic work accounting. The source module is borrowed
/// and must remain immutable and alive until this session is destroyed. At
/// most one actual candidate may be outstanding; callers must report its
/// disposition before requesting the next representative.
class RankCandidateSearchSession {
public:
  static mlir::FailureOr<std::unique_ptr<RankCandidateSearchSession>>
  create(mlir::ModuleOp sourceModule, const RankCandidateSearchConfig &config,
         CoordinatedWorkLedger &ledger,
         StructuredCandidateSearchStatistics *statistics = nullptr);

  ~RankCandidateSearchSession();
  RankCandidateSearchSession(RankCandidateSearchSession &&) noexcept;
  RankCandidateSearchSession &operator=(RankCandidateSearchSession &&) noexcept;
  RankCandidateSearchSession(const RankCandidateSearchSession &) = delete;
  RankCandidateSearchSession &
  operator=(const RankCandidateSearchSession &) = delete;

  /// Returns nullptr on deterministic exhaustion or policy stop. Materialized
  /// and actual-candidates-rejected proposals are skipped internally and
  /// trigger stable-order backfill from the still-live structural candidates.
  mlir::FailureOr<std::unique_ptr<CoordinatedTileVariant>>
  admitNextActualCandidate();

  mlir::LogicalResult
  completeActiveCandidate(int64_t stableSemanticOrdinal,
                          CoordinatedActualCandidateDisposition disposition,
                          llvm::StringRef failureGate = {});

  bool exhausted() const;
  llvm::StringRef getStructuredCandidateSetDigest() const;
  llvm::StringRef getActualCandidateSetDigest() const;

private:
  struct Impl;
  explicit RankCandidateSearchSession(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};

/// Builds a coordinated candidates of actual complete-rank Tile clones. The
/// current production seam materializes the mandatory conservative variant;
/// every returned member nevertheless already owns all-and-only rank entries
/// under one executable-evaluation reservation. No Tile-to-Instr, completion,
/// SPM, DDR, transport, ABI, cost selection, or final IR replacement occurs
/// here.
mlir::FailureOr<RankCandidateSet> buildRankCandidateSet(
    mlir::ModuleOp sourceModule, const RankCandidateSearchConfig &config,
    CoordinatedWorkLedger &ledger,
    StructuredCandidateSearchStatistics *structuredStatistics = nullptr);

mlir::LogicalResult
verifyCoordinatedTileVariant(const CoordinatedTileVariant &variant,
                             int64_t expectedRankCount);

std::string computeRankCandidateSetDigest(const RankCandidateSet &candidates);

std::string computeCoordinatedTileVariantContentDigest(
    const CoordinatedTileVariant &variant);

/// Materializes one all-rank repair sibling directly from a still-live,
/// unplaced actual Tile parent. Executable-evaluation reservation is admitted
/// before repair work is charged; no source replay or failed Instr/offset state
/// is accepted.
mlir::FailureOr<CoordinatedTileVariant> materializeCoordinatedTileRepair(
    const CoordinatedTileVariant &parent, CoordinatedTileRepairAction action,
    int64_t stableSemanticOrdinal, CoordinatedWorkLedger &ledger,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_RANKCANDIDATESEARCH_H
