//===- WaferCardProgramToTileModules.h - Whole-card projection -*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERCARDPROGRAMTOTILEMODULES_H
#define WAFER_CONVERSION_WAFERCARDPROGRAMTOTILEMODULES_H

#include "Wafer/Target/PhysicalIds.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace wafer {

/// One owning, standalone projection of a selected Tile program.
struct ProjectedPhysicalTileModule {
  PhysicalCardId cardId;
  PhysicalTileId tileId;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

/// Projects the unique verified wafer.card.program in `sourceModule` into one
/// standalone module per physical Tile. Results are ordered by increasing
/// tile_id and retain the source module attributes, target topology, logical
/// card-partition mesh, card-wide shared symbol declarations, and only the
/// matching tile.program body.
///
/// This is an artifact boundary, not a pass and not a scheduling decision.
/// The source module is not mutated. Search state, other Tile bodies,
/// Physical execution bindings and target launch slots are never inferred or
/// copied from the logical mesh.
mlir::FailureOr<llvm::SmallVector<ProjectedPhysicalTileModule, 16>>
projectCardProgramToPhysicalTileModules(mlir::ModuleOp sourceModule,
                                        std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERCARDPROGRAMTOTILEMODULES_H
