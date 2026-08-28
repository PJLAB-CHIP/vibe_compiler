//===- StandaloneTileModules.h - Create standalone Tile modules -*- C++ -*-===//

#ifndef WAFER_CONVERSION_STANDALONETILEMODULES_H
#define WAFER_CONVERSION_STANDALONETILEMODULES_H

#include "Wafer/Analysis/Structured/StructuredMaterializationRelations.h"
#include "Wafer/Target/Core/TopologyIds.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace wafer {

/// One owning, standalone module for a selected Tile.
struct StandaloneTileModule {
  CardId cardId;
  TileId tileId;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations materializationRelations;
};

/// Consumes the verified top-level wafer.tile.module operations in
/// `sourceModule` and moves each Tile body into one standalone module. Results
/// are ordered by `(card_id, tile_id)` and retain source module attributes,
/// target topology, logical card-partition mesh, shared declarations, and only
/// the matching Tile body.
///
/// This changes module multiplicity, so it is a compiler API rather than an
/// MLIR pass. It does not make scheduling decisions.
/// Consuming the source makes the large executable bodies an ownership
/// transfer rather than a speculative clone. Only the small declarations that
/// every standalone module must own are copied. Search state, other Tile
/// bodies, Target execution bindings and launch slots are never inferred or
/// copied from the logical mesh.
mlir::FailureOr<llvm::SmallVector<StandaloneTileModule, 16>>
createStandaloneTileModules(mlir::OwningOpRef<mlir::ModuleOp> sourceModule,
                            std::string *failureReason = nullptr,
                            const StructuredMaterializationRelations
                                *materializationRelations = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_STANDALONETILEMODULES_H
