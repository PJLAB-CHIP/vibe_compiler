//===- ScheduledRankFinalization.cpp - Exact rank finalization -----------===//

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

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
finalizeCoordinatedRankModule(mlir::OwningOpRef<mlir::ModuleOp> module,
                              RankFinalizationFailure *failure) {
  if (failure)
    *failure = {};
  if (!module) {
    if (failure)
      failure->kind = RankFinalizationFailureKind::Contract;
    return mlir::failure();
  }

  auto recordFailure = [&](RankFinalizationFailureKind kind) {
    if (failure)
      failure->kind = kind;
  };
  if (containsTileDataflowOperations(module->getOperation())) {
    recordFailure(RankFinalizationFailureKind::Contract);
    module->emitError()
        << "rank_finalization_requires_canonical_instr_action: Tile "
           "dataflow must be lowered exactly once while constructing the "
           "canonical Instr parent";
    return mlir::failure();
  }
  if (hasWholeVariantFacts(*module)) {
    recordFailure(RankFinalizationFailureKind::WholeVariantFacts);
    module->emitError()
        << "rank_finalization_contains_whole_variant_facts: DDR placement and "
           "Direct DTE bindings must be recomputed by coordinated admission";
    return mlir::failure();
  }

  wafer::support::recordCompileWork(
      wafer::support::CompileWorkKind::FinalizationCandidateClone);
  clearRankCandidatePhysicalFacts(*module);
  if (mlir::failed(mlir::verify(*module))) {
    recordFailure(RankFinalizationFailureKind::Verification);
    module->emitError(
        "executable-finalization Instr parent failed verification");
    return mlir::failure();
  }

  mlir::PassManager preparation(module->getContext());
  wafer::support::attachCompileTiming(preparation,
                                      "coordinated-rank-preparation");
  wafer::buildPrepareScheduledRankCandidatePipeline(preparation);
  if (mlir::failed(preparation.run(*module))) {
    recordFailure(RankFinalizationFailureKind::FunctionBoundaryBufferization);
    return mlir::failure();
  }
  if (mlir::failed(rebuildMinimumNCCJoins(*module))) {
    recordFailure(RankFinalizationFailureKind::Completion);
    return mlir::failure();
  }
  if (mlir::failed(mlir::verify(*module))) {
    recordFailure(RankFinalizationFailureKind::Verification);
    return mlir::failure();
  }

  mlir::PassManager spmPlanning(module->getContext());
  wafer::support::attachCompileTiming(spmPlanning,
                                      "coordinated-rank-spm-planning");
  wafer::buildPlanSPMMemoryPipeline(spmPlanning);
  spmPlanning.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(spmPlanning.run(*module))) {
    recordFailure(RankFinalizationFailureKind::SPMAllocation);
    return mlir::failure();
  }
  if (hasWholeVariantFacts(*module)) {
    recordFailure(RankFinalizationFailureKind::WholeVariantFacts);
    module->emitError("rank_finalization_created_whole_variant_facts");
    return mlir::failure();
  }

  analysis::InstructionProgramCost exactCost =
      analysis::analyzeInstructionProgramCost(
          module->getOperation(), analysis::getTargetScheduleCostPolicy());
  if (std::optional<std::string> exactFailure =
          getExactCostClosureFailure(exactCost)) {
    recordFailure(RankFinalizationFailureKind::ExactCost);
    module->emitError() << "rank_finalization_exact_cost: " << *exactFailure;
    return mlir::failure();
  }

  if (failure)
    *failure = {};
  return std::move(module);
}

} // namespace wafer::compiler::detail
