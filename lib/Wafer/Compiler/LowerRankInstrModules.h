//===- LowerRankInstrModules.h - Rank Instr module lowering -*- C++ -*-===//

#ifndef WAFER_COMPILER_LOWERRANKINSTRMODULES_H
#define WAFER_COMPILER_LOWERRANKINSTRMODULES_H

#include "CoordinatedCommunicationAction.h"

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Frontend/Program.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

enum class CoordinatedReadyOrderKind : uint8_t {
  Canonical,
  ReadyOrder,
};

enum class CoordinatedBufferingKind : uint8_t {
  Single,
  StaticFixedSlot,
};

enum class CoordinatedWorkerPlacementKind : uint8_t {
  Unplaced,
  DisjointComponents,
};

enum class CoordinatedScheduleSerializationKind : uint8_t {
  Unchanged,
  DirectDTEComputeWindows,
};

/// Query-local key for one complete-rank scheduling action. It is never
/// reconstructed from rank descriptors or persisted in IR/packages.
struct RankScheduleActionKey {
  CoordinatedReadyOrderKind readyOrderKind =
      CoordinatedReadyOrderKind::Canonical;
  CoordinatedBufferingKind bufferingKind = CoordinatedBufferingKind::Single;
  uint32_t bufferingPlanOrdinal = 0;
  CoordinatedWorkerPlacementKind workerPlacementKind =
      CoordinatedWorkerPlacementKind::Unplaced;
  uint32_t workerPlacementPlanOrdinal = 0;
  CoordinatedScheduleSerializationKind serializationKind =
      CoordinatedScheduleSerializationKind::Unchanged;
  std::optional<CommunicationActionKey> communicationActionKey;
};

struct RankExecutableSet {
  RankExecutableSet(
      std::vector<RankExecutable> ranks,
      RuntimeLaunchContract runtimeLaunchContract,
      analysis::CardInstructionProgramCost resourceCost)
      : ranks(std::move(ranks)),
        runtimeLaunchContract(std::move(runtimeLaunchContract)),
        resourceCost(std::move(resourceCost)) {}

  std::vector<RankExecutable> ranks;
  RuntimeLaunchContract runtimeLaunchContract;
  analysis::CardInstructionProgramCost resourceCost;
};

/// Invocation-local search instrumentation. It is never stored in selected
/// IR, Tile executables, or package files.
struct RankCandidateSearchStatistics {
  uint64_t tileModuleLoweringAttempts = 0;
  uint64_t tileModuleLoweringSuccesses = 0;
  uint64_t targetLoweringVerificationInvocations = 0;
  uint64_t targetRankVerificationInvocations = 0;
  uint64_t cardExecutablesProduced = 0;
  uint64_t paretoRetainedVariants = 0;
  uint64_t scheduleEstimatedActions = 0;
  uint64_t scheduleCoverageRetainedActions = 0;
  uint64_t scheduleMaterializationAttempts = 0;
  uint64_t scheduleMaterializationFailures = 0;
  uint64_t scheduleMaterializationBackfills = 0;
  uint64_t scheduleSuccessfulActionClones = 0;
  uint64_t scheduleActualRankClones = 0;
  uint64_t scheduleExactActions = 0;
  uint64_t scheduleSeedAttempts = 0;
  uint64_t scheduleExpansionAttempts = 0;
  uint64_t peakLiveEvaluationCursors = 0;
  uint64_t peakLiveCanonicalInstrParents = 0;
  uint64_t peakLiveScheduleActionClones = 0;
  uint64_t maximumRankPipelineWorkers = 1;
  uint64_t noCProfitabilityEvaluations = 0;
  uint64_t noCProfitabilityRejected = 0;
  uint64_t noCProfitabilityIndeterminate = 0;
  uint64_t noCProfitabilityEstimated = 0;
  uint64_t noCProfitabilityProven = 0;
};

/// Consumes exactly one owned, already rank-finalized Instr module per logical
/// rank. The disposable tuple receives DDR placement, index lowering, exact
/// Direct-DTE binding, card resource validation, RankExecutable
/// construction, and target ABI/LLVM verification. `selectedTileIR` is
/// query-local diagnostic text and must match the rank domain.
mlir::FailureOr<RankExecutableSet> lowerRankInstrModules(
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> rankModules,
    llvm::ArrayRef<std::shared_ptr<const std::string>> selectedTileIR,
    const RankScheduleActionKey &actionKey,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    std::string *failedCheck = nullptr,
    RankCandidateSearchStatistics *statistics = nullptr,
    unsigned rankPipelineParallelism = 0);

/// Prove from one resulting rank's current IR that every DDR movement is
/// attached to a function input or returned output root.
bool hasBoundaryOnlyDDRMovementEvidence(const RankExecutable &rank);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_LOWERRANKINSTRMODULES_H
