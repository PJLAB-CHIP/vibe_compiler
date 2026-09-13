//===- CapacityFeedback.h - Actual input access attribution -*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_CAPACITYFEEDBACK_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_CAPACITYFEEDBACK_H

#include "TemporalProposals.h"
#include "Wafer/Target/TopologyIds.h"
#include "Wafer/Transforms/Instr/MemoryPlanning.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <set>
#include <string>

namespace wafer::compiler::detail {

enum class CapacityFeedbackStatus : uint8_t { Valid, BrokenContract };

struct InputCapacityFeedback {
  CapacityFeedbackStatus status = CapacityFeedbackStatus::Valid;
  std::set<TemporalCoordinate> coordinates;
  uint64_t unavailableDemands = 0;
  uint64_t ambiguousInputs = 0;
  uint64_t sharedInputDemands = 0;
  // A broken current tensor indexing contract is not an unavailable source.
  std::string detail;
};

/// Reads the actual failed allocation and its writers while they are alive.
/// Associates the complete structural reader set through validated topology
/// and the existing program argument ABI. No failed IR handle escapes this
/// query.
InputCapacityFeedback
deriveInputCapacityFeedback(CardId card, TileId tile,
                            const SPMMemoryPlanningFailure &failure,
                            llvm::ArrayRef<const TemporalDomain *> domains,
                            llvm::ArrayRef<TemporalChoice> choices);

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_CAPACITYFEEDBACK_H
