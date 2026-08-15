//===- WholeCardExecutableSynthesis.h - Card-program exact seam -*- C++ -*-===//

#ifndef WAFER_COMPILER_WHOLECARDEXECUTABLESYNTHESIS_H
#define WAFER_COMPILER_WHOLECARDEXECUTABLESYNTHESIS_H

#include "WholeCardExecutableLowering.h"

#include "Wafer/Analysis/TheoreticalScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Frontend/Program.h"
#include "Wafer/Support/OptimizationConfig.h"
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

/// Query-local instrumentation for the whole-card candidate search. These
/// counters describe compiler work only; none is persisted in IR or package
/// files.
struct WholeCardExecutableSynthesisStatistics {
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
  uint64_t baselineCardProgramMaterializations = 0;
  uint64_t baselineScopedCardProgramMaterializations = 0;
  uint64_t baselineRegionSPMCapacityChecks = 0;
  uint64_t baselineRegionSPMCapacityOverflowProofs = 0;
  uint64_t baselineRegionSPMChecksRequiringFunctionScope = 0;
  uint64_t baselineRegionSPMCapacityAnalysisFailures = 0;
  uint64_t baselineMaximumRegionSPMQueryWorkers = 1;
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
  WholeCardSynthesisStatistics exactGates;
  uint64_t selectedExecutableRematerializations = 0;
  WholeCardSynthesisStatistics selectedExecutableRematerializationGates;
};

/// Query result at the TensorProgram-to-executable boundary. The executable IR
/// is the sole semantic result. The printed Tile dataflow snapshots are
/// same-invocation diagnostic trace and are never admitted into the executable
/// physical Tile modules, package files, or runtime contract.
struct WholeCardCompilationResult {
  WholeCardCompilationResult(WholeCardExecutable executable,
                             std::vector<std::string> tileDataflowIRTrace)
      : executable(std::move(executable)),
        tileDataflowIRTrace(std::move(tileDataflowIRTrace)) {}

  WholeCardExecutable executable;
  std::vector<std::string> tileDataflowIRTrace;
};

/// Derives the capacity-respecting temporal tile at canonical ceilDiv wave
/// breakpoints. Each capacity/refinement step moves to a strictly smaller
/// class, so the algorithm is bounded by the finite static shape domain.
llvm::SmallVector<int64_t, 4>
deriveCapacityTemporalShape(llvm::ArrayRef<int64_t> maximumShardShape,
                            uint64_t elementBytes, uint64_t tensorMultiplicity,
                            const TargetMemoryPolicy &memory,
                            unsigned additionalWaveRefinements);

/// Searches, materializes and admits one whole-card physical executable.
///
/// Both optimization policies follow the same IR sequence:
///
///   TensorProgram -> CardProgram -> physical Tile modules -> Instr modules
///   -> required NCC join placement/SPM planning -> whole-card executable
///   lowering.
///
/// `none` materializes only the deterministic maximum-participation,
/// capacity-respecting joint baseline; disabling selection never permits a
/// hard-resource-illegal executable. `search` derives a query-local joint
/// spatial/temporal candidates from the current structured DAG, each static
/// output/iteration domain, finite ceilDiv wave breakpoints, known tensor byte
/// footprint and available physical Tiles. When SSA/effect analysis proves
/// independent observable components, one joint candidate can bind those
/// components to disjoint Tile groups and CardProgram materialization emits
/// only each Tile's selected root/producer closures. The factorized finite
/// search has no candidate-count, depth, beam-width, or elapsed-time cap. It
/// only removes proven-infeasible lower bounds and exact-equivalent work before
/// cloning, then selects the admitted candidate with minimum theoretical
/// makespan under one cohort-wide enabled-term set.
///
/// The source module is borrowed and remains unchanged. Every per-Tile module
/// module crosses TileRegion -> Instr, physical-Tile memory planning, and the
/// same whole-card verification path. Physical identities come from
/// verified topology and CardProgram structure, never from vector position
/// or logical partition identity. Optional inspection output is returned as a
/// separate same-invocation trace rather than stored in the executable.
mlir::FailureOr<WholeCardCompilationResult> synthesizeWholeCardExecutable(
    mlir::ModuleOp tensorProgram,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics,
    WholeCardExecutableSynthesisStatistics *statistics = nullptr,
    unsigned tilePipelineParallelism = 0);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_WHOLECARDEXECUTABLESYNTHESIS_H
