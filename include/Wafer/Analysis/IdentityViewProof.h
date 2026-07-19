//===- IdentityViewProof.h - Recomputable identity/trip proofs -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_IDENTITYVIEWPROOF_H
#define WAFER_ANALYSIS_IDENTITYVIEWPROOF_H

#include <cstdint>
#include <optional>

namespace mlir {
namespace memref {
class SubViewOp;
} // namespace memref
namespace scf {
class ForOp;
} // namespace scf
namespace tensor {
class ExtractSliceOp;
} // namespace tensor
} // namespace mlir

namespace wafer {

enum class IdentityProofStatus : uint8_t {
  ProvenIdentity,
  ProvenNonIdentity,
  Unsupported,
  ResourceExhausted,
};

struct IdentityViewProof {
  IdentityProofStatus status = IdentityProofStatus::Unsupported;
  uint64_t inspectedDimensions = 0;
};

struct StaticTripCountProof {
  IdentityProofStatus status = IdentityProofStatus::Unsupported;
  std::optional<int64_t> tripCount;
};

IdentityViewProof proveIdentityView(mlir::tensor::ExtractSliceOp slice);
IdentityViewProof proveIdentityView(mlir::memref::SubViewOp subview);
StaticTripCountProof proveStaticTripCount(mlir::scf::ForOp loop);

} // namespace wafer

#endif // WAFER_ANALYSIS_IDENTITYVIEWPROOF_H
