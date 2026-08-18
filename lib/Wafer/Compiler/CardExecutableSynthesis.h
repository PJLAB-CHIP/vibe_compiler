//===- CardExecutableSynthesis.h - Card executable synthesis -*- C++ -*-===//

#ifndef WAFER_COMPILER_CARDEXECUTABLESYNTHESIS_H
#define WAFER_COMPILER_CARDEXECUTABLESYNTHESIS_H

#include "CardExecutableLowering.h"
#include "StructuredDAGCandidateSchedule.h"

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"
#include "Wafer/Analysis/TheoreticalScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Frontend/Program.h"
#include "Wafer/Support/TargetPolicy.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::compiler::detail {

/// Query-local instrumentation for the card candidate search. These
/// counters describe compiler work only; none is persisted in IR or package
/// files.
struct CardExecutableSynthesisStatistics {
  uint64_t structuredNodeCount = 0;
  uint64_t structuredEdgeCount = 0;
  uint64_t dependencyComponentCount = 0;
  bool independentComponentPlacementProven = false;
  uint64_t candidateProposals = 0;
  uint64_t multiReductionAxisCandidateProposals = 0;
  uint64_t nodePlacementGroups = 0;
  uint64_t nodePlacementStatesExpanded = 0;
  uint64_t nodePlacementTransitionsRejected = 0;
  uint64_t nodePlacementResourceEquivalentStates = 0;
  uint64_t nodePlacementEnumerationFailures = 0;
  uint64_t nodePlacementCandidateProposals = 0;
  uint64_t multiStagePlacementCandidateProposals = 0;
  uint64_t independentComponentCandidateProposals = 0;
  uint64_t alternativeEdgeActionCandidateProposals = 0;
  uint64_t layoutAssignedCandidateProposals = 0;
  uint64_t layoutConversionCandidateProposals = 0;
  uint64_t layoutBufferedCandidateProposals = 0;
  uint64_t bufferedCandidateProposals = 0;
  uint64_t applicableFusionLogicalEdges = 0;
  uint64_t fusedEdgeCandidateProposals = 0;
  uint64_t resourceScheduleRejections = 0;
  uint64_t resourceScheduleMemoHits = 0;
  uint64_t resourceScheduleMemoMisses = 0;
  uint64_t cheapPrunedCandidates = 0;
  uint64_t strictDominatedCandidates = 0;
  uint64_t incumbentClosedFeedbackCandidates = 0;
  uint64_t preBufferEquivalentRejections = 0;
  uint64_t bufferStructureEquivalentRejections = 0;
  uint64_t feedbackBeamDeferredCandidates = 0;
  uint64_t feedbackRootBudgetClosures = 0;
  uint64_t shortlistedCandidates = 0;
  uint64_t materializedCandidates = 0;
  uint64_t indeterminateCompilationFailures = 0;
  /// Exact-demand instrumentation: only proven logical contradictions delete
  /// placement trials; unsupported semantics and indeterminate failures stop
  /// the owning legalization path as typed failures.
  uint64_t exactDemandSatisfiedEdges = 0;
  uint64_t baselineExactDemandPairQueries = 0;
  uint64_t provenLogicalInfeasibleTrials = 0;
  uint64_t unsupportedSemanticRelationTrials = 0;
  uint64_t indeterminateDemandQueries = 0;
  uint64_t edgeCarrierMaterializationRejections = 0;
  /// Typed abort state of the owning legalization path, when it stopped for
  /// unsupported semantics or an indeterminate failure. Diagnostic only.
  analysis::ExactDemandStatus demandAbortStatus =
      analysis::ExactDemandStatus::Satisfied;
  std::string demandAbortDetail;
  uint64_t multiReductionAxisCandidateMaterializations = 0;
  uint64_t nodePlacementCandidateMaterializations = 0;
  uint64_t multiStagePlacementCandidateMaterializations = 0;
  uint64_t independentComponentCandidateMaterializations = 0;
  uint64_t alternativeEdgeActionCandidateMaterializations = 0;
  uint64_t layoutAssignedCandidateMaterializations = 0;
  uint64_t layoutConversionCandidateMaterializations = 0;
  uint64_t layoutBufferedCandidateMaterializations = 0;
  uint64_t bufferedCandidateMaterializations = 0;
  uint64_t rotatingSlotAllocationsMaterialized = 0;
  uint64_t allocationFeedbackTransitions = 0;
  uint64_t allocationFeedbackCandidates = 0;
  uint64_t allocationFeedbackProgressivePromotions = 0;
  uint64_t allocationFeedbackPrioritySelections = 0;
  uint64_t allocationFeedbackLookaheadCandidates = 0;
  uint64_t allocationFeedbackEndpointCandidates = 0;
  uint64_t allocationFeedbackLookaheadBoundaryClosures = 0;
  uint64_t spmFailureProbeAttempts = 0;
  uint64_t spmFailureProbeEarlyRejections = 0;
  uint64_t spmFailureProbeTilesSkipped = 0;
  uint64_t bufferFeedbackTransitions = 0;
  uint64_t bufferFeedbackCandidates = 0;
  uint64_t candidatePriorityReorders = 0;
  uint64_t materializationRejections = 0;
  uint64_t acceptedCandidates = 0;
  uint64_t plannedCandidates = 0;
  uint64_t schedulePlanRejections = 0;
  uint64_t nodePlacementCandidateAcceptances = 0;
  uint64_t multiStagePlacementCandidateAcceptances = 0;
  uint64_t independentComponentCandidateAcceptances = 0;
  uint64_t alternativeEdgeActionCandidateAcceptances = 0;
  uint64_t layoutAssignedCandidateAcceptances = 0;
  uint64_t layoutConversionCandidateAcceptances = 0;
  uint64_t layoutBufferedCandidateAcceptances = 0;
  uint64_t bufferedCandidateAcceptances = 0;
  uint64_t actualFusionCandidateAcceptances = 0;
  uint64_t selectedActualFusedLogicalEdges = 0;
  uint64_t selectedStableOrdinal = 0;
  uint64_t selectedOutputMappingCount = 0;
  uint64_t selectedUniqueActiveTileCount = 0;
  uint64_t selectedParallelComponentCount = 0;
  uint64_t selectedTemporalWaveLowerBound = 0;
  uint64_t selectedInstructionExecutionLowerBound = 0;
  uint64_t selectedPeakOutputTileFootprintEstimate = 0;
  uint64_t selectedPeakAlignedResidencyEstimate = 0;
  uint64_t selectedPeerTransferCount = 0;
  uint64_t selectedPeerBytes = 0;
  uint64_t selectedBufferCount = 1;
  uint64_t selectedSPMMovementWork = 0;
  uint64_t selectedDDRMovementWork = 0;
  uint64_t selectedMakespanPicoseconds = 0;
  analysis::StaticDurationTermMask enabledDurationTerms = 0;
  CardExecutableLoweringStatistics exactGates;
  uint64_t selectedExecutableRematerializations = 0;
  CardExecutableLoweringStatistics selectedExecutableRematerializationGates;
};

/// Query result at the TensorProgram-to-executable boundary. The executable IR
/// is the sole semantic result. The printed Tile dataflow snapshots are
/// same-invocation diagnostic trace and are never admitted into the executable
/// Tile modules, package files, or runtime contract.
struct CardExecutableSynthesisResult {
  CardExecutableSynthesisResult(CardExecutableLoweringResult executable,
                             std::vector<std::string> tileDataflowIRTrace)
      : executable(std::move(executable)),
        tileDataflowIRTrace(std::move(tileDataflowIRTrace)) {}

  CardExecutableLoweringResult executable;
  std::vector<std::string> tileDataflowIRTrace;
};

/// Runs the legacy candidate search controller. This boundary deliberately
/// owns only search statistics; it does not accept or construct a baseline
/// ledger. Q51.Core replaces this implementation while preserving the common
/// exact CardModule-to-executable gates.
mlir::FailureOr<CardExecutableSynthesisResult> searchCardExecutable(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics,
    ProgramDataHandoff &programData,
    CardExecutableSynthesisStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0, bool requestTileIRTrace = false);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_CARDEXECUTABLESYNTHESIS_H
