//===- ScheduledRankFinalization.cpp - Final rank candidates -------------===//

#include "ScheduledRankFinalization.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Pipelines/Pipelines.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"

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

} // namespace

mlir::FailureOr<std::vector<FinalizedRankCandidate>>
finalizeScheduledRankCandidateFrontier(
    std::vector<wafer::ScheduledRankCandidate> frontier) {
  if (frontier.empty())
    return mlir::failure();
  unsigned baselineCount = llvm::count_if(
      frontier, [](const wafer::ScheduledRankCandidate &candidate) {
        return candidate.reservedBaseline;
      });
  if (baselineCount != 1) {
    frontier.front().module->emitError()
        << "rank_frontier_baseline_contract: expected exactly one reserved "
           "baseline but found "
        << baselineCount;
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

    mlir::FailureOr<int64_t> estimatedTimePs =
        wafer::estimateScheduledRankProgramTimePs(*candidate.module);
    if (mlir::failed(estimatedTimePs)) {
      if (candidate.reservedBaseline)
        return mlir::failure();
      continue;
    }
    finalized.emplace_back(std::move(candidate.module), *estimatedTimePs,
                           candidate.discoveryOrder,
                           candidate.reservedBaseline);
  }

  if (finalized.empty())
    return mlir::failure();
  return finalized;
}

} // namespace wafer::compiler::detail
