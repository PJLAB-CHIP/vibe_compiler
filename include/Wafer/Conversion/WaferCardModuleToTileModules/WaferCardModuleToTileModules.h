//===- WaferCardModuleToTileModules.h - Per-Tile modules ------*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERCARDMODULETOTILEMODULES_H
#define WAFER_CONVERSION_WAFERCARDMODULETOTILEMODULES_H

#include "Wafer/Analysis/Structured/StructuredMaterializationRelations.h"
#include "Wafer/Target/Core/TopologyIds.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace wafer {

/// One owning, standalone module for a selected Tile.
struct TileModule {
  CardId cardId;
  TileId tileId;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations materializationRelations;
};

/// Consumes the unique verified wafer.card.module in `sourceModule` and moves
/// its Tile bodies into one standalone module per Tile. Results are ordered by
/// increasing tile_id and retain the source module attributes, target
/// topology, logical card-partition mesh, card-wide shared symbol declarations,
/// and only the matching tile.module body.
///
/// This changes module multiplicity, so it is a compiler API rather than an
/// MLIR pass. It does not make scheduling decisions.
/// Consuming the source makes the large executable bodies an ownership
/// transfer rather than a speculative clone. Only the small declarations that
/// every standalone module must own are copied. Search state, other Tile
/// bodies, Target execution bindings and launch slots are never inferred or
/// copied from the logical mesh.
mlir::FailureOr<llvm::SmallVector<TileModule, 16>>
splitCardModuleIntoTileModules(mlir::OwningOpRef<mlir::ModuleOp> sourceModule,
                               std::string *failureReason = nullptr,
                               const StructuredMaterializationRelations
                                   *materializationRelations = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERCARDMODULETOTILEMODULES_H
