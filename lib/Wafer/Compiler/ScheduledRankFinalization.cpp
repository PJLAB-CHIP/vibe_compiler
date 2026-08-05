//===- ScheduledRankFinalization.cpp - Final rank candidates -------------===//

#include "ScheduledRankFinalization.h"

#include "Wafer/Support/CompileTiming.h"

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Target/TargetIdentity.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
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
    bool requireReservedBaseline) {
  if (frontier.empty()) {
    if (requireReservedBaseline)
      return mlir::failure();
    return std::vector<FinalizedRankCandidate>{};
  }
  unsigned baselineCount = llvm::count_if(
      frontier, [](const wafer::ScheduledRankCandidate &candidate) {
        return candidate.reservedBaseline;
      });
  if ((requireReservedBaseline && baselineCount != 1) || baselineCount > 1) {
    frontier.front().module->emitError()
        << "rank_frontier_baseline_contract: expected "
        << (requireReservedBaseline ? "exactly one" : "at most one")
        << " reserved baseline but found " << baselineCount;
    return mlir::failure();
  }

  std::vector<FinalizedRankCandidate> finalized;
  finalized.reserve(frontier.size());
  for (wafer::ScheduledRankCandidate &candidate : frontier) {
    if (hasWholeVariantFacts(*candidate.module)) {
      candidate.module->emitError()
          << "rank_frontier_contains_whole_variant_facts: DDR placement and "
             "Direct DTE bindings must be recomputed by the all-rank "
             "coordinator";
      if (candidate.reservedBaseline)
        return mlir::failure();
      continue;
    }
    mlir::PassManager manager(candidate.module->getContext());
    wafer::support::attachCompileTiming(manager, "scheduled-rank-finalization");
    wafer::buildFinalizeScheduledRankCandidatePipeline(manager);
    if (mlir::failed(manager.run(*candidate.module))) {
      if (candidate.reservedBaseline)
        return mlir::failure();
      continue;
    }
    if (hasWholeVariantFacts(*candidate.module)) {
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
  return finalized;
}

} // namespace wafer::compiler::detail
