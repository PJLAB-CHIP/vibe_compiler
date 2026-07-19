//===- ScheduledRankFinalization.cpp - Final rank candidates -------------===//

#include "ScheduledRankFinalization.h"

#include "../Pipelines/QualificationInternal.h"
#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Support/OptimizationQualification.h"

#include "mlir/Pass/PassManager.h"

namespace wafer::compiler::detail {

mlir::FailureOr<std::vector<FinalizedRankCandidate>>
finalizeScheduledRankCandidateFrontier(
    std::vector<wafer::ScheduledRankCandidate> frontier, int64_t logicalRank,
    const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &optimizationConfiguration) {
  if (logicalRank < 0 || logicalRank >= 16)
    return mlir::failure();
  std::vector<FinalizedRankCandidate> finalized;
  finalized.reserve(frontier.size());
  for (wafer::ScheduledRankCandidate &candidate : frontier) {
    if (candidate.discoveryOrder < 0 || candidate.discoveryOrder >= 6)
      continue;
    uint64_t invocationOrdinal =
        (static_cast<uint64_t>(logicalRank) * 6 +
         static_cast<uint64_t>(candidate.discoveryOrder)) *
            2 +
        1;
    mlir::PassManager manager(candidate.module->getContext());
    if (!wafer::qualification_internal::
            buildFinalizeScheduledTensorProgramPipeline(
                manager, proposal, optimizationConfiguration,
                /*diagnostic=*/nullptr, invocationOrdinal))
      continue;
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
