//===- LoadPipelining.h - Tile LoadPipelining -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_LOADPIPELINING_H
#define WAFER_TRANSFORMS_TILE_LOADPIPELINING_H

#include "Wafer/Transforms/Tile/LoopPipelining.h"
#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer::compiler::detail {
/// Query the current physical load/consumer graph. No capacity prediction is
/// used; eligibility requires a complete per-iteration definition and no
/// escaping or mutated alias of the selected load destination.
bool hasDistanceOneLoadPipeline(mlir::ModuleOp module);
llvm::SmallVector<mlir::scf::ForOp, 4>
getDistanceOneLoadPipelineLoops(mlir::ModuleOp module);

/// Materialize two actual slots and the existing SCF pipeline in the owned
/// transaction. The ordinary Instr/completion/memory path remains the consumer.
PipelinedModuleResult materializeDistanceOneLoadPipelines(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    StructuredMaterializationRelations &relations);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_LOADPIPELINING_H
