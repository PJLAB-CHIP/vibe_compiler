//===- AttentionWorkDescription.cpp - Attention work schema -----------===//

#include "Wafer/Planning/PhysicalDataflow/AttentionWorkDescription.h"

namespace wafer::compiler::detail {

const CanonicalAttentionWorkCoordinate *getCanonicalAttentionWorkCoordinate(
    const CanonicalAttentionWorkProjectionOutcome &outcome) {
  return std::get_if<CanonicalAttentionWorkCoordinate>(&outcome);
}

} // namespace wafer::compiler::detail
