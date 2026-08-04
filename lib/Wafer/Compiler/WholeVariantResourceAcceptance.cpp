//===- WholeVariantResourceAcceptance.cpp - All-rank resource gate ------===//

#include "WholeVariantResourceAcceptance.h"

#include "AcceptedCallClosure.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace wafer::compiler::detail {
namespace {

static mlir::LogicalResult
requireKnown(mlir::ModuleOp anchor, llvm::StringRef name,
             const analysis::ScheduleCostMetric &metric) {
  if (metric.isKnown())
    return mlir::success();
  return anchor.emitOpError()
         << "whole_variant_resource_acceptance: " << name << " is "
         << analysis::stringifyScheduleCostKnowledge(metric.knowledge) << " ("
         << analysis::stringifyScheduleCostReason(metric.reason) << ")";
}

} // namespace

mlir::FailureOr<analysis::WholeCardInstructionProgramCost>
acceptWholeVariantResources(llvm::ArrayRef<mlir::ModuleOp> rankModules,
                            const ExecutionConfig &executionConfig) {
  if (rankModules.empty())
    return mlir::failure();
  if (rankModules.size() != static_cast<size_t>(executionConfig.getRankCount()))
    return mlir::ModuleOp(rankModules.front()).emitOpError()
           << "whole_variant_resource_acceptance: rank domain has "
           << rankModules.size() << " modules but ExecutionConfig requires "
           << executionConfig.getRankCount();

  const analysis::TargetScheduleCostPolicy policy =
      analysis::getTargetScheduleCostPolicy();
  mlir::ModuleOp diagnosticAnchor = rankModules.front();
  if (policy.spmAddressLimit < policy.spmAddressBase)
    return diagnosticAnchor.emitOpError(
        "whole_variant_resource_acceptance: target SPM range is invalid");

  llvm::SmallVector<mlir::Operation *, 16> rankRoots;
  rankRoots.reserve(rankModules.size());
  for (mlir::ModuleOp module : rankModules) {
    llvm::Expected<AcceptedCallClosure> closure =
        analyzeAcceptedCallClosure(module);
    if (!closure)
      return module.emitOpError()
             << "whole_variant_resource_acceptance: accepted call closure "
                "is invalid: "
             << llvm::toString(closure.takeError());
    rankRoots.push_back(closure->entry.getOperation());
  }
  analysis::WholeCardInstructionProgramCost cost =
      analysis::analyzeWholeCardInstructionProgramCost(rankRoots, policy);

  const uint64_t perRankSPMCapacity =
      policy.spmAddressLimit - policy.spmAddressBase;
  for (auto &&[rank, rankCost] : llvm::enumerate(cost.rankCosts)) {
    mlir::ModuleOp rankModule = rankModules[rank];
    if (mlir::failed(requireKnown(rankModule, "SPM high-water",
                                  rankCost.spmHighWaterBytes)))
      return mlir::failure();
    if (rankCost.spmHighWaterBytes.value > perRankSPMCapacity)
      return rankModule.emitOpError()
             << "whole_variant_resource_acceptance: rank " << rank
             << " SPM high-water " << rankCost.spmHighWaterBytes.value
             << " exceeds per-rank capacity " << perRankSPMCapacity;
  }

  auto requireAggregate = [&](llvm::StringRef name,
                              const analysis::ScheduleCostMetric &metric) {
    return requireKnown(diagnosticAnchor, name, metric);
  };
  if (mlir::failed(requireAggregate("aggregate DDR read bytes",
                                    cost.aggregateDDRReadBytes)) ||
      mlir::failed(requireAggregate("aggregate DDR write bytes",
                                    cost.aggregateDDRWriteBytes)) ||
      mlir::failed(requireAggregate("aggregate SPM movement bytes",
                                    cost.aggregateSPMMovementBytes)) ||
      mlir::failed(
          requireAggregate("aggregate NoC transmit bytes",
                           cost.aggregateNoC.aggregateTransmitBytes)) ||
      mlir::failed(requireAggregate("aggregate NoC receive bytes",
                                    cost.aggregateNoC.aggregateReceiveBytes)) ||
      mlir::failed(requireAggregate("minimum-hop link-byte demand",
                                    cost.minimumHopLinkByteDemand)) ||
      mlir::failed(requireAggregate("aggregate instruction count",
                                    cost.aggregateInstructionCount)) ||
      mlir::failed(requireAggregate("aggregate event count",
                                    cost.aggregateEventCount)) ||
      mlir::failed(requireAggregate("aggregate NCC join count",
                                    cost.aggregateNCCJoinCount)) ||
      mlir::failed(requireAggregate("aggregate steady-state NCC join count",
                                    cost.aggregateSteadyStateNCCJoinCount)) ||
      mlir::failed(requireAggregate("aggregate non-terminal NCC join count",
                                    cost.aggregateNonTerminalNCCJoinCount)) ||
      mlir::failed(requireAggregate("aggregate NCC participant wait count",
                                    cost.aggregateNCCParticipantWaitCount)) ||
      mlir::failed(
          requireAggregate("aggregate steady-state NCC participant wait count",
                           cost.aggregateSteadyStateNCCParticipantWaitCount)) ||
      mlir::failed(
          requireAggregate("aggregate non-terminal NCC participant wait count",
                           cost.aggregateNonTerminalNCCParticipantWaitCount)) ||
      mlir::failed(requireAggregate("aggregate intrinsic NCC drain count",
                                    cost.aggregateIntrinsicNCCDrainCount)) ||
      mlir::failed(requireAggregate("summed rank SPM high-water",
                                    cost.summedRankSPMHighWaterBytes)))
    return mlir::failure();
  for (const analysis::ScheduleCostMetric &collective :
       cost.aggregateNoC.collectiveTransmitBytes)
    if (mlir::failed(requireAggregate("aggregate collective transmit bytes",
                                      collective)))
      return mlir::failure();

  if (cost.aggregateNoC.aggregateTransmitBytes.value !=
      cost.aggregateNoC.aggregateReceiveBytes.value)
    return diagnosticAnchor.emitOpError()
           << "whole_variant_resource_acceptance: matched all-rank NoC "
              "transmit and receive bytes differ ("
           << cost.aggregateNoC.aggregateTransmitBytes.value << " vs "
           << cost.aggregateNoC.aggregateReceiveBytes.value << ")";

  return cost;
}

} // namespace wafer::compiler::detail

namespace wafer::compiler::testing {

mlir::FailureOr<analysis::WholeCardInstructionProgramCost>
acceptWholeVariantResources(llvm::ArrayRef<mlir::ModuleOp> rankModules,
                            const ExecutionConfig &executionConfig) {
  return detail::acceptWholeVariantResources(rankModules, executionConfig);
}

} // namespace wafer::compiler::testing
