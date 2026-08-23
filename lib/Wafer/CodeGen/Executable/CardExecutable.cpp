//===- CardExecutable.cpp - Card executable ownership ------------------===//

#include "Wafer/CodeGen/Executable/CardExecutableInternal.h"

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Driver/CompilationStatistics.h"
#include "Wafer/Planning/Baseline/CardBaselineCompilation.h"
#include "Wafer/Planning/PhysicalDataflow/Search/PlanningSession.h"

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

CardExecutable::CardExecutable(ExecutionConfig executionConfig,
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

  const detail::CompileClock::time_point compilationStart =
      detail::CompileClock::now();
  const bool reportDetailedStatistics =
      static_cast<bool>(wafer::support::getActiveCompileTimingSession());
  struct PolicyCompilationResult {
    detail::CardExecutableLoweringResult executable;
    std::vector<std::string> tileDataflowIRTrace;
  };
  std::optional<PolicyCompilationResult> selected;

  if (optimizations.isNone()) {
    std::optional<detail::BaselineStatistics> statistics;
    if (reportDetailedStatistics)
      statistics.emplace();
    mlir::FailureOr<detail::CardBaselineCompilationResult> baseline =
        detail::compileCardBaseline(
            tensorModule, program, executionConfig, diagnostics, programData,
            statistics ? &*statistics : nullptr,
            /*tilePipelineParallelism=*/0, requestTileIRTrace);
    if (mlir::failed(baseline))
      return fail("card executable baseline failed");

    if (statistics) {
      const detail::CardExecutableLoweringStatistics &exactGates =
          statistics->exactGates;
      diagnostics
          << "wafer-compile: compile-stats "
             "stage=deterministic-card-executable-baseline"
          << " wall_ms=" << detail::elapsedCompileMilliseconds(compilationStart)
          << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
          << " tile_count=" << baseline->executable.tiles.size()
          << " source_preparations=" << statistics->baselineSourcePreparations
          << " materialization_preparations="
          << statistics->baselineMaterializationPreparations
          << " card_program_materializations="
          << statistics->baselineCardModuleMaterializations
          << " tile_entry_materializations="
          << statistics->baselineTileEntryMaterializations
          << " tile_materialization_workers="
          << statistics->baselineMaximumTileMaterializationWorkers
          << " actual_spm_capacity_rejections="
          << statistics->actualSPMCapacityRejections
          << " actual_temporal_refinements="
          << statistics->actualTemporalRefinements
          << " spatial_coordinate_queries="
          << statistics->spatialCoordinateQueries
          << " exact_demand_edges=" << statistics->exactDemandSatisfiedEdges
          << " indeterminate_compilation_failures="
          << statistics->indeterminateCompilationFailures
          << " materialization_rejections="
          << statistics->materializationRejections
          << " card_executable_compilations="
          << exactGates.cardModuleCompilationInvocations
          << " tile_pipeline_workers=" << exactGates.maximumTilePipelineWorkers
          << " tile_ir_prints=" << statistics->baselineTileIRPrints << '\n';
    }
    selected.emplace(
        PolicyCompilationResult{std::move(baseline->executable),
                                std::move(baseline->tileDataflowIRTrace)});
  } else if (optimizations.isSearch()) {
    mlir::FailureOr<std::unique_ptr<detail::CardProgramAnalysis>>
        programAnalysis = [&]() {
          wafer::support::ScopedCompileTimingSpan timing(
              "query", "physical-search", "analyze-card-program");
          return detail::analyzeCardProgram(tensorModule, program,
                                            executionConfig, diagnostics);
        }();
    if (mlir::failed(programAnalysis))
      return fail("physical search analysis failed");
    std::string failureReason;
    mlir::FailureOr<detail::PhysicalDataflowPlanningProblem> problem =
        detail::PhysicalDataflowPlanningProblem::create(
            **programAnalysis, CardId(0), analysis::IndexRelationLimits(),
            &failureReason);
    if (mlir::failed(problem))
      return fail("physical search problem failed: " + failureReason);
    detail::PhysicalDataflowPlanningSession session(*problem);
    mlir::FailureOr<detail::IncompletePlanningDomain> incomplete = [&]() {
      wafer::support::ScopedCompileTimingSpan timing(
          "planning", "physical-search", "planning-frontier");
      return session.getFirstIncompleteState(&failureReason);
    }();
    if (mlir::failed(incomplete))
      return fail("physical search foundation failed: " + failureReason);
    const detail::PlanningWorkCounts &work = incomplete->getWork();
    diagnostics << "wafer-compile: physical-search incomplete"
                << " required_coordinate="
                << detail::stringifyRequiredPlanningCoordinate(
                       incomplete->getRequiredCoordinate())
                << " spatial_successor_steps=" << work.spatialSuccessorSteps
                << " spatial_demand_queries=" << work.spatialDemandQueries
                << " spatial_states=" << work.spatialStatesQueued
                << " root_work_steps=" << work.rootWorkSuccessorSteps
                << " root_works=" << work.rootWorksValidated
                << " region_steps=" << work.regionSuccessorSteps
                << " region_states=" << work.regionStatesQueued
                << " temporal_steps=" << work.temporalSuccessorSteps
                << " temporal_states=" << work.temporalStatesQueued
                << " temporal_unsupported=" << work.unsupportedTemporalChoices
                << " temporal_indeterminate="
                << work.indeterminateTemporalChoices
                << " structural_readiness_queries="
                << work.structuralReadinessQueries
                << " representation_steps=" << work.representationSuccessorSteps
                << " representation_states=" << work.representationStatesQueued
                << " representation_unsupported="
                << work.unsupportedRepresentationChoices
                << " movement_steps=" << work.movementSuccessorSteps
                << " movement_states=" << work.movementStatesQueued
                << " movement_unsupported=" << work.unsupportedMovementChoices
                << " storage_steps=" << work.storageSuccessorSteps
                << " storage_states=" << work.storageStatesQueued
                << " storage_unsupported=" << work.unsupportedStorageChoices
                << " candidate_actualizations=0\n";
    return fail("card executable search has an incomplete planning domain");
  } else {
    return fail("card executable compilation requires an optimization policy");
  }

  detail::CardExecutableLoweringResult accepted =
      std::move(selected->executable);
  std::vector<std::string> tileDataflowIRTrace =
      std::move(selected->tileDataflowIRTrace);
  assert((optimizations.isNone() || optimizations.isSearch()) &&
         "card executable compilation requires a typed optimization policy");

  if (reportDetailedStatistics)
    printAcceptedInstructionWork(diagnostics, accepted);

  if (requestTileIRTrace && accepted.tiles.size() != tileDataflowIRTrace.size())
    return fail("compiler IR trace does not cover the accepted Tile "
                "domain");
  if (requestTileIRTrace) {
    irTrace.tiles.clear();
    irTrace.tiles.reserve(accepted.tiles.size());
    for (auto [index, tile] : llvm::enumerate(accepted.tiles))
      irTrace.tiles.push_back({tile.getCardId(), tile.getTileId(),
                               tile.getLaunchSlotId(),
                               std::move(tileDataflowIRTrace[index])});
  }

  std::vector<TileExecutable> tiles = std::move(accepted.tiles);
  if (tiles.size() != static_cast<size_t>(executionConfig.getTileCount()))
    return fail("Tile executable domain is incomplete");

  if (reportDetailedStatistics)
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
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
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
