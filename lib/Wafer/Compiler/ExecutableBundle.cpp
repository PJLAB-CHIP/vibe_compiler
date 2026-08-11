//===- ExecutableBundle.cpp - Physical Tile executable bundle -----------===//

#include "ExecutableBundleInternal.h"

#include "CompilationStatistics.h"
#include "WholeCardExecutableSynthesis.h"

#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

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
    const detail::AcceptedWholeCardExecutable &accepted) {
  const analysis::WholeCardInstructionProgramCost &cost = accepted.resourceCost;
  for (auto [tileIndex, tileCost] : llvm::enumerate(cost.tileCosts)) {
    const PhysicalTileExecutable &tile = accepted.tiles[tileIndex];
    diagnostics << "wafer-compile: instruction-work scope=tile card_id="
                << tile.getPhysicalCardId().getValue()
                << " tile_id=" << tile.getPhysicalTileId().getValue();
    printInstructionProgramWork(diagnostics, tileCost.work);
    printInstructionProgramResources(diagnostics, tileCost);
    diagnostics << '\n';
  }
  diagnostics << "wafer-compile: instruction-work scope=whole-card";
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

static llvm::Expected<ExecutableBundle> buildWholeCardBundle(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics, std::optional<int64_t> failAfterLaunchSlot,
    const detail::CompileClock::time_point &totalStart) {
  auto fail = [&](llvm::StringRef message) -> llvm::Error {
    diagnostics << "wafer-compile: " << message << "\n";
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   message.str().c_str());
  };

  if (failAfterLaunchSlot && *failAfterLaunchSlot >= 0 &&
      *failAfterLaunchSlot < executionConfig.getPhysicalTileCount())
    return fail("test-only injected failure after physical Tile launch slot " +
                std::to_string(*failAfterLaunchSlot));

  detail::WholeCardExecutableSynthesisStatistics statistics;
  const detail::CompileClock::time_point synthesisStart =
      detail::CompileClock::now();
  mlir::FailureOr<detail::AcceptedWholeCardExecutable> accepted =
      detail::synthesizeWholeCardExecutable(tensorModule, program,
                                            executionConfig, optimizations,
                                            diagnostics, &statistics);
  if (mlir::failed(accepted))
    return fail("whole-card executable synthesis failed");

  diagnostics
      << "wafer-compile: compile-stats "
         "stage=whole-card-executable-synthesis"
      << " wall_ms=" << detail::elapsedCompileMilliseconds(synthesisStart)
      << " peak_rss_kib=" << detail::getCompilePeakRSSKiB()
      << " physical_tile_count=" << accepted->tiles.size()
      << " candidate_proposals=" << statistics.candidateProposals
      << " cheap_pruned_candidates=" << statistics.cheapPrunedCandidates
      << " strict_dominated_candidates=" << statistics.strictDominatedCandidates
      << " pre_buffer_equivalent_rejections="
      << statistics.preBufferEquivalentRejections
      << " buffer_structure_equivalent_rejections="
      << statistics.bufferStructureEquivalentRejections
      << " shortlisted_candidates=" << statistics.shortlistedCandidates
      << " materialized_candidates=" << statistics.materializedCandidates
      << " materialization_rejections=" << statistics.materializationRejections
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
      << " target_gate_invocations="
      << statistics.exactGates.targetGateInvocations
      << " target_tile_gate_invocations="
      << statistics.exactGates.targetTileGateInvocations
      << " selected_executable_rematerializations="
      << statistics.selectedExecutableRematerializations
      << " selected_rematerialization_target_gate_invocations="
      << statistics.selectedExecutableRematerializationGates
             .targetGateInvocations
      << " tile_pipeline_workers="
      << statistics.exactGates.maximumTilePipelineWorkers << '\n';
  printAcceptedInstructionWork(diagnostics, *accepted);

  std::vector<PhysicalTileExecutable> tiles = std::move(accepted->tiles);
  if (tiles.size() !=
      static_cast<size_t>(executionConfig.getPhysicalTileCount()))
    return fail("executable bundle physical Tile domain is incomplete");

  diagnostics << "wafer-compile: compile-stats stage=executable-bundle"
              << " wall_ms=" << detail::elapsedCompileMilliseconds(totalStart)
              << " peak_rss_kib=" << detail::getCompilePeakRSSKiB() << "\n";
  return ExecutableBundleBuilder::makeBundle(
      executionConfig, std::move(accepted->runtimeLaunchContract), context,
      std::move(tiles));
}

static llvm::Expected<ExecutableBundle> buildExecutableBundleImpl(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot) {
  const detail::CompileClock::time_point totalStart =
      detail::CompileClock::now();
  wafer::support::ScopedCompileTimingSpan executableBundleTiming(
      "stage", "tensor-program-to-executable", "executable-bundle");
  return buildWholeCardBundle(context, tensorModule, program, executionConfig,
                              optimizations, diagnostics, failAfterLaunchSlot,
                              totalStart);
}

llvm::Expected<ExecutableBundle> detail::buildExecutableBundle(
    std::shared_ptr<mlir::MLIRContext> &context, mlir::ModuleOp tensorModule,
    frontend::FrontendProgramVerificationResult program,
    ExecutionConfig executionConfig, OptimizationConfig optimizations,
    llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLaunchSlot) {
  return buildExecutableBundleImpl(context, tensorModule, std::move(program),
                                   executionConfig, optimizations, diagnostics,
                                   failAfterLaunchSlot);
}

} // namespace wafer::compiler
