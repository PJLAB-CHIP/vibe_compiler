//===- CommunicationRegionClosure.h - Close communication scopes -*- C++
//-*-===//

#ifndef WAFER_TRANSFORMS_LINALG_COMMUNICATIONREGIONCLOSURE_H
#define WAFER_TRANSFORMS_LINALG_COMMUNICATIONREGIONCLOSURE_H

#include "Wafer/Transforms/Linalg/SpatialRegionMaterialization.h"
#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace wafer {

struct CommunicationRegionClosureStatistics {
  uint64_t closedExchangeComponents = 0;
  uint64_t mergedTileScopes = 0;
  uint64_t mergedRegions = 0;
};

/// Closes only complete cross-Tile exchanges for which current structural IR
/// proves one producer-before-consumer cut on every participating Tile. The
/// transformation merges the Regions belonging to that exchange and retargets
/// the live endpoint relations. It does not create messages, buffers, rounds,
/// completion facts, or a persistent communication plan.
mlir::LogicalResult closeCrossTileCommunicationRegions(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations,
    CommunicationRegionClosureStatistics *statistics = nullptr,
    SpatialRegionMaterializationFailure *failure = nullptr);

} // namespace wafer

#endif // WAFER_TRANSFORMS_LINALG_COMMUNICATIONREGIONCLOSURE_H
