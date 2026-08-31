//===- TemporalTiling.h - Apply live-operation temporal choices -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_TEMPORALTILING_H
#define WAFER_TRANSFORMS_LINALG_TEMPORALTILING_H

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"
#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <string>

namespace wafer {

enum class TemporalTilingFailureKind : uint8_t {
  None,
  BrokenContract,
  CompilerFailure,
};

struct TemporalTilingFailure {
  TemporalTilingFailureKind kind = TemporalTilingFailureKind::None;
  std::string detail;
};

struct TemporalTilingStatistics {
  uint64_t tiledTraversals = 0;
  uint64_t loops = 0;
  uint64_t specializedTails = 0;
  uint64_t specializedConcatBoundaries = 0;
  uint64_t fusedProducers = 0;
  uint64_t viewTransparentProducers = 0;
  uint64_t tileLocalAssemblies = 0;
  uint64_t assembledSegments = 0;
  uint64_t decomposedPads = 0;
  uint64_t decomposedConstantGenerates = 0;
};

/// Immediately applies one query-local temporal choice to the same unchanged
/// TileRegion from which `domain` was built. The domain and choice are invalid
/// after this call, regardless of success. On post-mutation failure the caller
/// must discard the complete candidate owner.
mlir::FailureOr<TemporalTilingStatistics>
applyTemporalTiling(const compiler::detail::TemporalDomain &domain,
                    const compiler::detail::TemporalChoice &choice,
                    StructuredMaterializationRelations &relations,
                    TemporalTilingFailure *failure = nullptr);

} // namespace wafer

#endif // WAFER_TRANSFORMS_LINALG_TEMPORALTILING_H
