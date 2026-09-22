//===- LoadPipelining.h - Tile LoadPipelining -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_LOADPIPELINING_H
#define WAFER_TRANSFORMS_TILE_LOADPIPELINING_H

#include "Wafer/Transforms/Tile/LoopPipelining.h"
#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer::compiler::detail {
struct LoadPipelineQueryResult {
  llvm::SmallVector<mlir::scf::ForOp, 4> loops;
  std::optional<LoopPipeliningFailure> failure;
  bool succeeded() const { return !failure.has_value(); }
};

/// Query verified current physical load/consumer IR. Eligibility requires a
/// complete iteration-local definition and no escaping alias. Later reads and
/// writes within that lifetime are allowed. No capacity prediction is used.
LoadPipelineQueryResult queryDistanceOneLoadPipelines(mlir::ModuleOp module);

/// Materialize two actual slots and the existing SCF pipeline in the owned
/// transaction. The ordinary Instr/completion/memory path remains the consumer.
PipelinedModuleResult materializeDistanceOneLoadPipelines(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    StructuredMaterializationRelations &relations);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_LOADPIPELINING_H
