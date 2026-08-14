//===- Pipelines.h - Wafer named pipeline registration ---------*- C++ -*-===//

#ifndef WAFER_PIPELINES_PIPELINES_H
#define WAFER_PIPELINES_PIPELINES_H

#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/TargetConversion.h"

#include <cstdint>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace wafer {

/// Normalize imported StableHLO without crossing the StableHLO legality
/// boundary. The result may still contain StableHLO operations.
void buildNormalizeImportedStablehloPipeline(mlir::OpPassManager &pm);

/// Eliminate the supported StableHLO operation set into structured tensor IR.
void buildLegalizeStablehloToStructuredTensorPipeline(
    mlir::OpPassManager &pm);

/// Apply bounded, Wafer-owned simplification to structured tensor IR.
void buildSimplifyStructuredTensorPipeline(mlir::OpPassManager &pm);

/// Compose import normalization, StableHLO legalization, bounded structured
/// simplification and final best-effort canonicalization.
void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm);

/// Add the atomic pass implementing Wafer's function-boundary bufferization
/// contract.
void addInstrFunctionBoundaryBufferizationPass(mlir::OpPassManager &pm);

/// Canonicalize and bufferize an Instr module without assigning SPM or DDR
/// offsets. This composite exposes its canonicalization role explicitly while
/// keeping function-boundary bufferization available as an independent leaf.
void buildBufferizeInstrFunctionsPipeline(mlir::OpPassManager &pm);

/// Produce memory-planning-ready Instr IR by applying the fixed
/// function-boundary bufferization contract and then recomputing the joins
/// required by the resulting function-local outstanding NCC accesses. Both
/// leaf stages remain visible at their real Module/Func anchors.
void buildPrepareInstrForMemoryPlanningPipeline(mlir::OpPassManager &pm);

/// Add target-topology materialization with the pass's typed options.
void addMaterializeTargetTopologyPass(
    mlir::OpPassManager &pm,
    const MaterializeTargetTopologyPassOptions &options = {});

/// Add execution-mesh materialization with the pass's typed options.
void addMaterializeExecutionMeshPass(
    mlir::OpPassManager &pm,
    const MaterializeExecutionMeshPassOptions &options = {});

/// Add the TileRegion-anchored conversion leaf to a TileRegion pass manager.
void addTileRegionToInstrConversionPass(mlir::OpPassManager &pm);

/// Add the function-anchored required-NCC-join placement leaf.
void addRequiredNCCJoinPlacementPass(mlir::OpPassManager &pm);

/// Add the function-anchored leaf that discards derived NCC joins and
/// recomputes the required set from the current instruction IR.
void addRecomputeRequiredNCCJoinPlacementPass(mlir::OpPassManager &pm);

/// Compose TileRegion conversion and function synchronization at their real
/// nested anchors.
void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm);

/// Add atomic, Module-anchored SPM offset assignment. The optional failure
/// result is a compiler adapter over the same pass implementation used by
/// textual pass replay.
void addAssignSPMOffsetsPass(mlir::OpPassManager &pm,
                             const PlanSPMMemoryPassOptions &options,
                             SPMMemoryPlanningFailure *failure = nullptr);

/// Add atomic, Module-anchored DDR offset assignment.
void addAssignDDROffsetsPass(mlir::OpPassManager &pm,
                             const PlanDDRMemoryPassOptions &options);

/// Lower affine control and indexing after a physical Tile has passed memory
/// planning. The upstream affine pass remains the atomic implementation.
void addLowerAffineControlAndIndexingPass(mlir::OpPassManager &pm);

/// Add the closed Instr-to-target-LLVM conversion for one typed target
/// request. The conversion itself remains one atomic Module transaction until
/// verifier-legal intermediate contracts are defined.
void addLowerInstrToTargetLLVMPass(
    mlir::OpPassManager &pm, const TargetConversionRequest &request);

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm);
#endif

void registerWaferPipelines();

} // namespace wafer

#endif // WAFER_PIPELINES_PIPELINES_H
