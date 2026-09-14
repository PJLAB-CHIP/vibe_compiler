//===- CommunicationProposals.h - Construct current communication choices ===//
#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_COMMUNICATIONPROPOSALS_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_COMMUNICATIONPROPOSALS_H

#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/Transforms/Instr/SharedDDRCompletion.h"

namespace wafer::compiler::detail {

struct CommunicationProposalStatistics {
  uint64_t localInputReads = 0;
  uint64_t ddrPackets = 0;
  uint64_t replacedMessages = 0;
  uint64_t issuePlacements = 0;
};

struct CommunicationProposalResult {
  SharedDDRCompletionResult outcome;
  CommunicationProposalStatistics statistics;
};

/// Completes a search communication proposal on its actual Instr owner.
/// Fixed baseline/qualification choices do not call this constructor.
/// A failed owner is discarded by the caller. No memory placement is run here.
CommunicationProposalResult constructCommunicationProposal(
    llvm::MutableArrayRef<StandaloneTileModule> tiles);

} // namespace wafer::compiler::detail
#endif
