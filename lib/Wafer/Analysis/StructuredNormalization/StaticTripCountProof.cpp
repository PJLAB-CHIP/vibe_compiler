//===- StaticTripCountProof.cpp - Recomputable loop trip proof -----------===//

#include "Wafer/Analysis/IdentityViewProof.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace wafer {

StaticTripCountProof proveStaticTripCount(mlir::scf::ForOp loop) {
  std::optional<int64_t> tripCount = mlir::constantTripCount(
      loop.getLowerBound(), loop.getUpperBound(), loop.getStep());
  if (!tripCount || *tripCount < 0)
    return {IdentityProofStatus::Unsupported, std::nullopt};
  return {IdentityProofStatus::ProvenIdentity, tripCount};
}

} // namespace wafer
