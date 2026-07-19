//===- ScheduledRankFinalization.cpp - Final rank candidates -------------===//

#include "ScheduledRankFinalization.h"

#include "Wafer/Pipelines/Pipelines.h"

#include "mlir/Pass/PassManager.h"

namespace wafer::compiler::detail {

mlir::FailureOr<std::vector<FinalizedRankCandidate>>
finalizeScheduledRankCandidateFrontier(
    std::vector<wafer::ScheduledRankCandidate> frontier) {
  std::vector<FinalizedRankCandidate> finalized;
  finalized.reserve(frontier.size());
  for (wafer::ScheduledRankCandidate &candidate : frontier) {
    mlir::PassManager manager(candidate.module->getContext());
    wafer::buildFinalizeScheduledTensorProgramPipeline(manager);
    if (mlir::failed(manager.run(*candidate.module)))
      continue;

    mlir::FailureOr<int64_t> estimatedTimePs =
        wafer::estimateScheduledRankProgramTimePs(*candidate.module);
    if (mlir::failed(estimatedTimePs))
      continue;
    finalized.emplace_back(std::move(candidate.module), *estimatedTimePs,
                           candidate.discoveryOrder);
  }

  if (finalized.empty())
    return mlir::failure();
  return finalized;
}

} // namespace wafer::compiler::detail
