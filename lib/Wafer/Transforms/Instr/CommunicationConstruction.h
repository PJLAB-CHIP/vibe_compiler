//===- CommunicationConstruction.h - Construct legal current communication ===//
#ifndef WAFER_TRANSFORMS_INSTR_COMMUNICATIONCONSTRUCTION_H
#define WAFER_TRANSFORMS_INSTR_COMMUNICATIONCONSTRUCTION_H

#include "Wafer/Transforms/Instr/SharedDDRCompletion.h"
#include <cstdint>

namespace wafer::compiler::detail {

struct CommunicationConstructionStatistics {
  uint64_t localInputReads = 0;
  uint64_t ddrPackets = 0;
  uint64_t replacedMessages = 0;
  uint64_t issuePlacements = 0;
};

struct CommunicationConstructionResult {
  SharedDDRCompletionResult outcome;
  CommunicationConstructionStatistics statistics;
};

/// Completes a search communication proposal on its actual Instr owner.
/// Success includes rebuilt Direct-DTE waits and joint-order verification;
/// these remain valid until the caller changes the corresponding IR facts.
/// Fixed baseline/qualification choices do not call this constructor.
/// A failed owner is discarded by the caller. No memory placement is run here.
CommunicationConstructionResult
constructCommunication(llvm::ArrayRef<mlir::ModuleOp> modules,
                       llvm::ArrayRef<TileId> tileIds);

} // namespace wafer::compiler::detail
#endif
