//===- ScheduledRankFinalization.cpp - Final rank candidates -------------===//

#include "ScheduledRankFinalization.h"

#include "Wafer/Support/CompileTiming.h"

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Support/CompileWorkStatistics.h"
#include "Wafer/Target/TargetIdentity.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace wafer::compiler::detail {
namespace {

static bool hasWholeVariantFacts(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *operation) {
    if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation))
      found |= static_cast<bool>(
          allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName));
    if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
      found |= send.getBinding().has_value();
    if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation))
      found |= recv.getBinding().has_value();
  });
  return found;
}

static std::optional<std::string>
getExactCostClosureFailure(const analysis::InstructionProgramCost &cost) {
  struct NamedMetric {
    llvm::StringLiteral name;
    const analysis::ScheduleCostMetric *metric;
  };
  const NamedMetric required[] = {
      {"NPU f16/bf16 logical ops", &cost.compute.npuF16Bf16LogicalOps},
      {"NPU other logical ops", &cost.compute.npuOtherLogicalOps},
      {"vector f16/bf16 logical ops", &cost.compute.vectorF16Bf16LogicalOps},
      {"vector f32 logical ops", &cost.compute.vectorF32LogicalOps},
      {"vector other logical ops", &cost.compute.vectorOtherLogicalOps},
      {"DDR read bytes", &cost.ddrReadBytes},
      {"DDR write bytes", &cost.ddrWriteBytes},
      {"SPM movement bytes", &cost.spmMovementBytes},
      {"gather/scatter bytes", &cost.gatherScatterBytes},
      {"NoC transmit bytes", &cost.noc.aggregateTransmitBytes},
      {"NoC receive bytes", &cost.noc.aggregateReceiveBytes},
      {"instruction count", &cost.instructionCount},
      {"event count", &cost.eventCount},
      {"NCC join count", &cost.nccJoinCount},
      {"steady-state NCC join count", &cost.steadyStateNCCJoinCount},
      {"non-terminal NCC join count", &cost.nonTerminalNCCJoinCount},
      {"NCC participant wait count", &cost.nccParticipantWaitCount},
      {"steady-state NCC participant wait count",
       &cost.steadyStateNCCParticipantWaitCount},
      {"non-terminal NCC participant wait count",
       &cost.nonTerminalNCCParticipantWaitCount},
      {"intrinsic NCC drain count", &cost.intrinsicNCCDrainCount},
  };
  for (const NamedMetric &entry : required) {
    if (entry.metric->isKnown())
      continue;
    std::string failure;
    llvm::raw_string_ostream os(failure);
    os << entry.name << " is "
       << analysis::stringifyScheduleCostKnowledge(entry.metric->knowledge)
       << " (" << analysis::stringifyScheduleCostReason(entry.metric->reason)
       << ")";
    return failure;
  }
  return std::nullopt;
}

} // namespace

mlir::FailureOr<std::vector<FinalizedRankCandidate>>
finalizeScheduledRankCandidateFrontier(
    std::vector<wafer::ScheduledRankCandidate> frontier,
    bool requireReservedBaseline, RankCompletionPolicy completionPolicy,
    RankFinalizationFailure *failure) {
  if (failure)
    *failure = {};
  if (frontier.empty()) {
    if (requireReservedBaseline) {
      if (failure)
        failure->kind = RankFinalizationFailureKind::Contract;
      return mlir::failure();
    }
    return std::vector<FinalizedRankCandidate>{};
  }
  unsigned baselineCount = llvm::count_if(
      frontier, [](const wafer::ScheduledRankCandidate &candidate) {
        return candidate.reservedBaseline;
      });
  if ((requireReservedBaseline && baselineCount != 1) || baselineCount > 1) {
    if (failure) {
      failure->kind = RankFinalizationFailureKind::Contract;
      failure->stableOrdinal = frontier.front().stableOrdinal;
    }
    frontier.front().module->emitError()
        << "rank_frontier_baseline_contract: expected "
        << (requireReservedBaseline ? "exactly one" : "at most one")
        << " reserved baseline but found " << baselineCount;
    return mlir::failure();
  }

  std::vector<FinalizedRankCandidate> finalized;
  finalized.reserve(frontier.size());
  for (wafer::ScheduledRankCandidate &candidate : frontier) {
    auto recordFailure = [&](RankFinalizationFailureKind kind) {
      if (failure)
        *failure = {kind, candidate.stableOrdinal};
    };
    if (hasWholeVariantFacts(*candidate.module)) {
      recordFailure(RankFinalizationFailureKind::WholeVariantFacts);
      candidate.module->emitError()
          << "rank_frontier_contains_whole_variant_facts: DDR placement and "
             "Direct DTE bindings must be recomputed by the all-rank "
             "coordinator";
      if (candidate.reservedBaseline)
        return mlir::failure();
      continue;
    }
    wafer::support::recordCompileWork(
        wafer::support::CompileWorkKind::TerminalCandidateClone);
    bool loweredTileDataflow =
        containsTileDataflowOperations(*candidate.module);
    std::string loweringFailure;
    if (mlir::failed(convertTileRegionToInstrModule(*candidate.module,
                                                    &loweringFailure))) {
      recordFailure(RankFinalizationFailureKind::TileToInstr);
      candidate.module->emitError()
          << "terminal complete-rank Tile-to-Instr conversion failed"
          << (loweringFailure.empty() ? "" : ": ") << loweringFailure;
      if (candidate.reservedBaseline)
        return mlir::failure();
      continue;
    }
    if (!loweredTileDataflow &&
        mlir::failed(normalizeMinimumNCCJoins(*candidate.module))) {
      recordFailure(RankFinalizationFailureKind::Completion);
      candidate.module->emitError(
          "terminal complete-rank completion normalization failed");
      if (candidate.reservedBaseline)
        return mlir::failure();
      continue;
    }
    clearRankCandidatePhysicalFacts(*candidate.module);
    if (mlir::failed(mlir::verify(*candidate.module))) {
      recordFailure(RankFinalizationFailureKind::Verification);
      candidate.module->emitError(
          "terminal complete-rank Instr parent failed verification");
      if (candidate.reservedBaseline)
        return mlir::failure();
      continue;
    }
    mlir::LogicalResult finalizationResult = mlir::success();
    RankFinalizationFailureKind pipelineFailure =
        RankFinalizationFailureKind::Contract;
    if (completionPolicy == RankCompletionPolicy::RebuildFromCurrentEffects) {
      mlir::PassManager preparation(candidate.module->getContext());
      wafer::support::attachCompileTiming(preparation,
                                          "scheduled-rank-preparation");
      wafer::buildPrepareScheduledRankCandidatePipeline(preparation);
      if (mlir::failed(preparation.run(*candidate.module))) {
        finalizationResult = mlir::failure();
        pipelineFailure =
            RankFinalizationFailureKind::FunctionBoundaryBufferization;
      } else if (mlir::failed(rebuildMinimumNCCJoins(*candidate.module))) {
        finalizationResult = mlir::failure();
        pipelineFailure = RankFinalizationFailureKind::Completion;
      } else if (mlir::failed(mlir::verify(*candidate.module))) {
        finalizationResult = mlir::failure();
        pipelineFailure = RankFinalizationFailureKind::Verification;
      }
      mlir::PassManager spmPlanning(candidate.module->getContext());
      wafer::support::attachCompileTiming(spmPlanning,
                                          "scheduled-rank-spm-planning");
      wafer::buildPlanSPMMemoryPipeline(spmPlanning);
      spmPlanning.addPass(mlir::createCanonicalizerPass());
      if (mlir::succeeded(finalizationResult)) {
        finalizationResult = spmPlanning.run(*candidate.module);
        if (mlir::failed(finalizationResult))
          pipelineFailure = RankFinalizationFailureKind::SPMAllocation;
      }
    } else {
      mlir::PassManager manager(candidate.module->getContext());
      wafer::support::attachCompileTiming(manager,
                                          "scheduled-rank-finalization");
      wafer::buildFinalizeScheduledRankCandidatePipeline(manager);
      finalizationResult = manager.run(*candidate.module);
    }
    if (mlir::failed(finalizationResult)) {
      recordFailure(pipelineFailure);
      if (candidate.reservedBaseline)
        return mlir::failure();
      continue;
    }
    if (hasWholeVariantFacts(*candidate.module)) {
      recordFailure(RankFinalizationFailureKind::WholeVariantFacts);
      candidate.module->emitError()
          << "rank_finalization_created_whole_variant_facts";
      if (candidate.reservedBaseline)
        return mlir::failure();
      continue;
    }

    analysis::InstructionProgramCost exactCost =
        analysis::analyzeInstructionProgramCost(
            candidate.module->getOperation(),
            analysis::getTargetScheduleCostPolicy());
    if (std::optional<std::string> failure =
            getExactCostClosureFailure(exactCost)) {
      recordFailure(RankFinalizationFailureKind::ExactCost);
      candidate.module->emitError()
          << "rank_finalization_exact_cost: " << *failure;
      if (candidate.reservedBaseline)
        return mlir::failure();
      continue;
    }
    finalized.emplace_back(
        std::move(candidate.module), candidate.stableOrdinal,
        candidate.artifactKind, candidate.reservedBaseline,
        candidate.bufferingKind, candidate.bufferingPlanOrdinal,
        candidate.workerPlacementKind, candidate.workerPlacementPlanOrdinal,
        candidate.frontierOrderOrdinal);
    finalized.back().selectedTileIR = std::move(candidate.selectedTileIR);
  }

  if (finalized.empty() && requireReservedBaseline)
    return mlir::failure();
  if (!finalized.empty() && failure)
    *failure = {};
  return finalized;
}

} // namespace wafer::compiler::detail
