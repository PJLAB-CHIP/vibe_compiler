//===- WaferCardProgramToTileModules.h - Per-Tile modules ------*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERCARDPROGRAMTOTILEMODULES_H
#define WAFER_CONVERSION_WAFERCARDPROGRAMTOTILEMODULES_H

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/Target/PhysicalIds.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace wafer {

/// One owning, standalone module for a selected physical Tile.
struct PhysicalTileModule {
  PhysicalCardId cardId;
  PhysicalTileId tileId;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations materializationRelations;
};

/// Splits the unique verified wafer.card.program in `sourceModule` into one
/// standalone module per physical Tile. Results are ordered by increasing
/// tile_id and retain the source module attributes, target topology, logical
/// card-partition mesh, card-wide shared symbol declarations, and only the
/// matching tile.program body.
///
/// This changes module multiplicity, so it is a compiler API rather than an
/// MLIR pass. It does not make scheduling decisions.
/// The source module is not mutated. Search state, other Tile bodies,
/// Physical execution bindings and target launch slots are never inferred or
/// copied from the logical mesh.
mlir::FailureOr<llvm::SmallVector<PhysicalTileModule, 16>>
splitCardProgramIntoPhysicalTileModules(
    mlir::ModuleOp sourceModule, std::string *failureReason = nullptr,
    const StructuredMaterializationRelations *materializationRelations =
        nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERCARDPROGRAMTOTILEMODULES_H
