//===- OnlineAttentionDecomposition.h - Lower current online state -*- C++
//-*-===//

#ifndef WAFER_TRANSFORMS_LINALG_ONLINEATTENTIONDECOMPOSITION_H
#define WAFER_TRANSFORMS_LINALG_ONLINEATTENTIONDECOMPOSITION_H

#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <string>

namespace wafer {

enum class OnlineAttentionDecompositionFailureKind : uint8_t {
  None,
  UnsupportedSemantics,
  BrokenContract,
  CompilerFailure,
};

struct OnlineAttentionDecompositionFailure {
  OnlineAttentionDecompositionFailureKind kind =
      OnlineAttentionDecompositionFailureKind::None;
  std::string detail;
};

struct OnlineAttentionDecompositionStatistics {
  uint64_t decomposedOperations = 0;
  uint64_t qkContractions = 0;
  uint64_t pvContractions = 0;
  uint64_t scoreApplications = 0;
  uint64_t rowReductions = 0;
  uint64_t normalizationFactors = 0;
  uint64_t probabilityUpdates = 0;
  uint64_t stateScales = 0;
  uint64_t scoreScratchTensors = 0;
  uint64_t maximumScoreElements = 0;
};

/// Replaces every current online-attention operation in `module` with actual
/// Linalg/Tensor/arith/math operations. Existing SCF loops, TileRegions,
/// spatial state merge/finalize and endpoint relations are preserved. A
/// post-mutation failure requires the caller to discard the candidate owner.
mlir::FailureOr<OnlineAttentionDecompositionStatistics>
decomposeOnlineAttention(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    OnlineAttentionDecompositionFailure *failure = nullptr);

/// Checks the stable handoff condition required by layout/bufferization.
mlir::LogicalResult
verifyOnlineAttentionDecompositionComplete(mlir::ModuleOp module);

} // namespace wafer

#endif // WAFER_TRANSFORMS_LINALG_ONLINEATTENTIONDECOMPOSITION_H
