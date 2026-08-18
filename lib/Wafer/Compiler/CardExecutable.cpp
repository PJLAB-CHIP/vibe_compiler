//===- CardExecutable.cpp - Card executable ownership ------------------===//

#include "CardExecutableInternal.h"

#include "CardExecutableSynthesis.h"
#include "CompilationStatistics.h"
#include "DeterministicCardExecutableSynthesis.h"

#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

CardExecutable::CardExecutable(CardExecutable &&) = default;
CardExecutable &CardExecutable::operator=(CardExecutable &&) = default;
CardExecutable::~CardExecutable() = default;

CardExecutable::CardExecutable(
    ExecutionConfig executionConfig,
    RuntimeLaunchContract runtimeLaunchContract,
    std::shared_ptr<mlir::MLIRContext> context,
    std::vector<TileExecutable> tiles,
    std::unique_ptr<ProgramDataHandoff> programData)
    : executionConfig(executionConfig),
      runtimeLaunchContract(std::move(runtimeLaunchContract)),
      context(std::move(context)), tiles(std::move(tiles)),
      programData(std::move(programData)) {}

const ProgramDataHandoff &CardExecutable::getProgramDataHandoff() const {
  return *programData;
}

static void
printInstructionWorkMetric(llvm::raw_ostream &os,
                           const analysis::ScheduleCostMetric &metric) {
  os << analysis::stringifyScheduleCostKnowledge(metric.knowledge) << ':';
  if (metric.isKnown())
    os << metric.value;
  else
    os << analysis::stringifyScheduleCostReason(metric.reason);
}

static void
printInstructionExecutionCount(llvm::raw_ostream &os, llvm::StringRef name,
                               const analysis::InstructionExecutionCount &count,
                               bool alwaysPrintBounds = false) {
  os << ' ' << name << "_sites=";
  printInstructionWorkMetric(os, count.staticSites);
  os << ' ' << name << "_exact=";
  printInstructionWorkMetric(os, count.exactExecutions);
  auto equal = [](const analysis::ScheduleCostMetric &left,
                  const analysis::ScheduleCostMetric &right) {
    return left.value == right.value && left.knowledge == right.knowledge &&
           left.reason == right.reason;
  };
  if (!alwaysPrintBounds && equal(count.exactExecutions, count.lowerBound) &&
      equal(count.exactExecutions, count.upperBound))
    return;
  os << ' ' << name << "_lower=";
  printInstructionWorkMetric(os, count.lowerBound);
  os << ' ' << name << "_upper=";
  printInstructionWorkMetric(os, count.upperBound);
}

static void
printInstructionProgramWork(llvm::raw_ostream &os,
                            const analysis::InstructionProgramWork &work) {
  printInstructionExecutionCount(os, "instructions", work.instructions,
                                 /*alwaysPrintBounds=*/true);
  printInstructionExecutionCount(os, "events", work.asynchronousEvents);
  printInstructionExecutionCount(os, "rdma", work.rdmaIssues);
  printInstructionExecutionCount(os, "wdma", work.wdmaIssues);
  printInstructionExecutionCount(os, "tdma", work.tdmaIssues);
  printInstructionExecutionCount(os, "ct", work.ctIssues);
  printInstructionExecutionCount(os, "ne", work.neIssues);
  printInstructionExecutionCount(os, "dte", work.dteOperations);
  printInstructionExecutionCount(os, "gather_scatter",
                                 work.gatherScatterOperations);
  printInstructionExecutionCount(os, "dte_send", work.dteSendOperations);
  printInstructionExecutionCount(os, "dte_receive", work.dteReceiveOperations);
  printInstructionExecutionCount(os, "dte_wait", work.dteWaitOperations);
  printInstructionExecutionCount(os, "ncc_join", work.nccJoins,
                                 /*alwaysPrintBounds=*/true);
  printInstructionExecutionCount(os, "ncc_join_steady",
                                 work.steadyStateNCCJoins);
  printInstructionExecutionCount(os, "ncc_join_nonterminal",
                                 work.nonTerminalNCCJoins);
  printInstructionExecutionCount(os, "ncc_participant_wait",
                                 work.nccParticipantWaits);
  printInstructionExecutionCount(os, "ncc_participant_wait_steady",
                                 work.steadyStateNCCParticipantWaits);
  printInstructionExecutionCount(os, "ncc_participant_wait_nonterminal",
                                 work.nonTerminalNCCParticipantWaits);
  printInstructionExecutionCount(os, "intrinsic_ncc_drain",
                                 work.intrinsicNCCDrains);
}

static void
printInstructionProgramResources(llvm::raw_ostream &os,
                                 const analysis::InstructionProgramCost &cost) {
  auto print = [&](llvm::StringRef name,
                   const analysis::ScheduleCostMetric &metric) {
    os << ' ' << name << '=';
    printInstructionWorkMetric(os, metric);
  };
  print("ddr_read_bytes", cost.ddrReadBytes);
  print("ddr_write_bytes", cost.ddrWriteBytes);
  print("spm_movement_bytes", cost.spmMovementBytes);
  print("gather_scatter_bytes", cost.gatherScatterBytes);
  print("compute_npu_f16_bf16_ops", cost.compute.npuF16Bf16LogicalOps);
  print("compute_npu_other_ops", cost.compute.npuOtherLogicalOps);
  print("compute_vector_f16_bf16_ops", cost.compute.vectorF16Bf16LogicalOps);
  print("compute_vector_f32_ops", cost.compute.vectorF32LogicalOps);
  print("compute_vector_other_ops", cost.compute.vectorOtherLogicalOps);
  print("noc_transmit_bytes", cost.noc.aggregateTransmitBytes);
  print("noc_receive_bytes", cost.noc.aggregateReceiveBytes);
  print("spm_high_water_bytes", cost.spmHighWaterBytes);
  print("ddr_high_water_bytes", cost.ddrHighWaterBytes);
  print("spm_buffers", cost.compilerOwnedSPMBufferCount);
  print("ddr_buffers", cost.compilerOwnedDDRBufferCount);
}

static void printAcceptedInstructionWork(
    llvm::raw_ostream &diagnostics,
    const detail::CardExecutableLoweringResult &accepted) {
  const analysis::CardInstructionProgramCost &cost = accepted.resourceCost;
  for (auto [tileIndex, tileCost] : llvm::enumerate(cost.tileCosts)) {
    const TileExecutable &tile = accepted.tiles[tileIndex];
    diagnostics << "wafer-compile: instruction-work scope=tile card_id="
                << tile.getCardId().getValue()
                << " tile_id=" << tile.getTileId().getValue();
    printInstructionProgramWork(diagnostics, tileCost.work);
    printInstructionProgramResources(diagnostics, tileCost);
    diagnostics << '\n';
  }
  diagnostics << "wafer-compile: instruction-work scope=card";
  printInstructionProgramWork(diagnostics, cost.aggregateWork);
  auto printAggregate = [&](llvm::StringRef name,
                            const analysis::ScheduleCostMetric &metric) {
    diagnostics << ' ' << name << '=';
    printInstructionWorkMetric(diagnostics, metric);
  };
  printAggregate("ddr_read_bytes", cost.aggregateDDRReadBytes);
  printAggregate("ddr_write_bytes", cost.aggregateDDRWriteBytes);
  printAggregate("spm_movement_bytes", cost.aggregateSPMMovementBytes);
  printAggregate("gather_scatter_bytes", cost.aggregateGatherScatterBytes);
  printAggregate("compute_npu_f16_bf16_ops",
                 cost.aggregateCompute.npuF16Bf16LogicalOps);
  printAggregate("compute_npu_other_ops",
                 cost.aggregateCompute.npuOtherLogicalOps);
  printAggregate("compute_vector_f16_bf16_ops",
                 cost.aggregateCompute.vectorF16Bf16LogicalOps);
  printAggregate("compute_vector_f32_ops",
                 cost.aggregateCompute.vectorF32LogicalOps);
  printAggregate("compute_vector_other_ops",
                 cost.aggregateCompute.vectorOtherLogicalOps);
  printAggregate("noc_transmit_bytes",
                 cost.aggregateNoC.aggregateTransmitBytes);
  printAggregate("noc_receive_bytes", cost.aggregateNoC.aggregateReceiveBytes);
  printAggregate("spm_high_water_bytes", cost.summedTileSPMHighWaterBytes);
  printAggregate("ddr_high_water_bytes", cost.summedTileDDRHighWaterBytes);
  printAggregate("spm_buffers", cost.aggregateCompilerOwnedSPMBufferCount);
  printAggregate("ddr_buffers", cost.aggregateCompilerOwnedDDRBufferCount);
  diagnostics << '\n';

  diagnostics << "wafer-compile: instruction-work scope=tile-maxima";
  printInstructionProgramWork(diagnostics, cost.maximumTileWork);
  printAggregate("spm_high_water_bytes", cost.maximumTileSPMHighWaterBytes);
  printAggregate("ddr_high_water_bytes", cost.maximumTileDDRHighWaterBytes);
  printAggregate("spm_buffers", cost.maximumTileCompilerOwnedSPMBufferCount);
  printAggregate("ddr_buffers", cost.maximumTileCompilerOwnedDDRBufferCount);
  diagnostics << '\n';
}

static llvm::Expected<CardExecutable>
compileTensorProgramModuleToCardExecutable(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    const detail::CompileClock::time_point &totalStart,
    ProgramDataHandoff &programData, CompilationIRTrace &irTrace,
    bool requestTileIRTrace) {
  auto fail = [&](llvm::StringRef message) -> llvm::Error {
    diagnostics << "wafer-compile: " << message << "\n";
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   message.str().c_str());
  };

  if (failAfterLaunchSlot && *failAfterLaunchSlot >= 0 &&
      *failAfterLaunchSlot < executionConfig.getTileCount())
    return fail("test-only injected failure after Tile launch slot " +
                std::to_string(*failAfterLaunchSlot));

  const detail::CompileClock::time_point synthesisStart =
      detail::CompileClock::now();
  mlir::FailureOr<detail::CardExecutableSynthesisResult> compiled =
      [&]() -> mlir::FailureOr<detail::CardExecutableSynthesisResult> {
    if (optimizations.isNone()) {
      detail::DeterministicBaselineLedger ledger;
      std::vector<std::string> tileDataflowIRTrace;
      mlir::FailureOr<detail::DeterministicCardExecutableSynthesisResult>
          result =
          detail::synthesizeDeterministicCardExecutable(
              tensorModule, program, executionConfig, diagnostics, programData,
              &ledger, /*tilePipelineParallelism=*/0,
              requestTileIRTrace ? &tileDataflowIRTrace : nullptr);
      if (mlir::failed(result))
        return mlir::failure();
      const detail::CardExecutableLoweringStatistics &exactGates =
          ledger.exactGates;
      diagnostics
          << "wafer-compile: compile-stats "
             "stage=deterministic-card-executable-synthesis"
          << " wall_ms=" << detail::elapsedCompileMilliseconds(synthesisStart)
          << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
          << " tile_count=" << result->executable.tiles.size()
          << " source_preparations=" << ledger.baselineSourcePreparations
          << " materialization_preparations="
          << ledger.baselineMaterializationPreparations
          << " card_program_materializations="
          << ledger.baselineCardModuleMaterializations
          << " root_shard_materializations="
          << ledger.baselineRootShardMaterializations
          << " tile_entry_materializations="
          << ledger.baselineTileEntryMaterializations
          << " spatial_legalization_transitions="
          << ledger.spatialLegalizationTransitions
          << " spatial_coordinate_queries="
          << ledger.spatialCoordinateQueries
          << " exact_demand_edges=" << ledger.exactDemandSatisfiedEdges
          << " region_spm_capacity_checks="
          << ledger.baselineRegionSPMCapacityChecks
          << " region_spm_capacity_overflow_proofs="
          << ledger.baselineRegionSPMCapacityOverflowProofs
          << " function_spm_capacity_overflow_proofs="
          << ledger.baselineFunctionSPMCapacityOverflowProofs
          << " region_spm_capacity_analysis_failures="
          << ledger.baselineRegionSPMCapacityAnalysisFailures
          << " indeterminate_compilation_failures="
          << ledger.indeterminateCompilationFailures
          << " materialization_rejections="
          << ledger.materializationRejections
          << " card_executable_compilations="
          << exactGates.cardModuleCompilationInvocations
          << " tile_pipeline_workers="
          << exactGates.maximumTilePipelineWorkers
          << " region_spm_query_workers="
          << ledger.baselineMaximumRegionSPMQueryWorkers
          << " function_spm_capacity_checks="
          << ledger.baselineFunctionScopedSPMCapacityChecks
          << " function_spm_query_workers="
          << ledger.baselineMaximumFunctionSPMQueryWorkers
          << " tile_ir_prints=" << ledger.baselineTileIRPrints << '\n';
      return detail::CardExecutableSynthesisResult(
          std::move(result->executable), std::move(tileDataflowIRTrace));
    }

    assert(optimizations.isSearch() &&
           "card executable synthesis requires a typed optimization policy");
    detail::CardExecutableSynthesisStatistics statistics;
    mlir::FailureOr<detail::CardExecutableSynthesisResult> result =
        detail::searchCardExecutable(
            tensorModule, program, executionConfig, diagnostics, programData,
            &statistics, /*tilePipelineParallelism=*/0, requestTileIRTrace);
    if (mlir::failed(result))
      return mlir::failure();
    const detail::CardExecutableLoweringStatistics &exactGates =
        statistics.exactGates;
    diagnostics
        << "wafer-compile: compile-stats stage=card-executable-search"
        << " wall_ms=" << detail::elapsedCompileMilliseconds(synthesisStart)
        << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
        << " tile_count=" << result->executable.tiles.size()
        << " candidate_proposals=" << statistics.candidateProposals
        << " cheap_pruned_candidates=" << statistics.cheapPrunedCandidates
        << " strict_dominated_candidates="
        << statistics.strictDominatedCandidates
        << " pre_buffer_equivalent_rejections="
        << statistics.preBufferEquivalentRejections
        << " buffer_structure_equivalent_rejections="
        << statistics.bufferStructureEquivalentRejections
        << " shortlisted_candidates=" << statistics.shortlistedCandidates
        << " materialized_candidates=" << statistics.materializedCandidates
        << " indeterminate_compilation_failures="
        << statistics.indeterminateCompilationFailures
        << " materialization_rejections="
        << statistics.materializationRejections
        << " accepted_candidates=" << statistics.acceptedCandidates
        << " selected_stable_ordinal=" << statistics.selectedStableOrdinal
        << " selected_output_mapping_count="
        << statistics.selectedOutputMappingCount
        << " selected_unique_active_tile_count="
        << statistics.selectedUniqueActiveTileCount
        << " selected_parallel_component_count="
        << statistics.selectedParallelComponentCount
        << " selected_makespan_ps=" << statistics.selectedMakespanPicoseconds
        << " enabled_duration_terms=" << statistics.enabledDurationTerms
        << " card_executable_compilations="
        << exactGates.cardModuleCompilationInvocations
        << " selected_executable_rematerializations="
        << statistics.selectedExecutableRematerializations
        << " selected_rematerialization_card_executable_compilations="
        << statistics.selectedExecutableRematerializationGates
               .cardModuleCompilationInvocations
        << " tile_pipeline_workers="
        << exactGates.maximumTilePipelineWorkers << '\n';
    return std::move(*result);
  }();
  if (mlir::failed(compiled))
    return fail("card executable synthesis failed");
  detail::CardExecutableLoweringResult &accepted = compiled->executable;
  printAcceptedInstructionWork(diagnostics, accepted);

  if (requestTileIRTrace &&
      accepted.tiles.size() != compiled->tileDataflowIRTrace.size())
    return fail("compiler IR trace does not cover the accepted Tile "
                "domain");
  if (requestTileIRTrace) {
    irTrace.tiles.clear();
    irTrace.tiles.reserve(accepted.tiles.size());
    for (auto [index, tile] : llvm::enumerate(accepted.tiles))
      irTrace.tiles.push_back({tile.getCardId(), tile.getTileId(),
                               tile.getLaunchSlotId(),
                               std::move(compiled->tileDataflowIRTrace[index])});
  }

  std::vector<TileExecutable> tiles = std::move(accepted.tiles);
  if (tiles.size() != static_cast<size_t>(executionConfig.getTileCount()))
    return fail("Tile executable domain is incomplete");

  diagnostics << "wafer-compile: compile-stats stage=card-executable"
              << " wall_ms=" << detail::elapsedCompileMilliseconds(totalStart)
              << " peak_rss_kib=" << detail::getCompilePeakRSSKiB() << "\n";
  // All tensor-program verification is complete. Real partitions have been
  // adopted into the source identity space; every remaining helper candidate
  // is byte-identical dead data and must not escape in the executable.
  programData.discardUnadoptedCandidates();
  return CardExecutableBuilder::makeCardExecutable(
      executionConfig, std::move(accepted.runtimeLaunchContract), context,
      std::move(tiles),
      std::make_unique<ProgramDataHandoff>(std::move(programData)));
}

static llvm::Expected<CardExecutable> buildCardExecutableImpl(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData, CompilationIRTrace &irTrace,
    bool requestTileIRTrace) {
  const detail::CompileClock::time_point totalStart =
      detail::CompileClock::now();
  wafer::support::ScopedCompileTimingSpan cardExecutableTiming(
      "stage", "tensor-program-to-executable", "card-executable");
  return compileTensorProgramModuleToCardExecutable(
      context, tensorModule, program, executionConfig, optimizations,
      diagnostics, failAfterLaunchSlot, totalStart, programData, irTrace,
      requestTileIRTrace);
}

llvm::Expected<CardExecutable> detail::buildCardExecutable(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData) {
  CompilationIRTrace discardedTrace;
  return buildCardExecutableImpl(context, tensorModule, std::move(program),
                                 executionConfig, optimizations, diagnostics,
                                 failAfterLaunchSlot, programData,
                                 discardedTrace, /*requestTileIRTrace=*/false);
}

llvm::Expected<CardExecutable> detail::buildCardExecutableWithIRTrace(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    ProgramDataHandoff &programData, CompilationIRTrace &irTrace) {
  return buildCardExecutableImpl(context, tensorModule, std::move(program),
                                 executionConfig, optimizations, diagnostics,
                                 failAfterLaunchSlot, programData, irTrace,
                                 /*requestTileIRTrace=*/true);
}

} // namespace wafer::compiler
