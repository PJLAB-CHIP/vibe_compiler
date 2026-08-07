//===- CompleteRankMaterializationInternal.h ------------------*- C++ -*-===//

#ifndef WAFER_LIB_TRANSFORMS_PHYSICALDATAFLOW_COMPLETERANKMATERIALIZATIONINTERNAL_H
#define WAFER_LIB_TRANSFORMS_PHYSICALDATAFLOW_COMPLETERANKMATERIALIZATIONINTERNAL_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace wafer::complete_rank_materialization {

mlir::memref::GlobalOp findOrCreateDenseConstantGlobal(
    mlir::ModuleOp module, mlir::arith::ConstantOp constant,
    mlir::MemRefType memrefType, mlir::ElementsAttr elements);

void outlineRankDenseTensorConstants(mlir::ModuleOp module);

} // namespace wafer::complete_rank_materialization

namespace wafer::tensor_program_scheduling {

/// Temporary internal entry retained for legacy mechanism tests. Current
/// coordinated consumers use materializeConservativeCompleteRankBaseline.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
materializeCompleteRankTileProgram(mlir::ModuleOp sourceModule,
                                   int64_t logicalRank,
                                   unsigned *materializedRegionCount = nullptr);

struct SelectiveSpillMaterialization {
  mlir::Value ddrBuffer;
  mlir::Value reloadedValue;
  wafer::StorageStoreOp store;
  wafer::StorageLoadOp load;
  mlir::Operation *reloadAllocation = nullptr;
};

bool canMaterializeSelectiveTileSpill(wafer::TileRegionOp region,
                                      mlir::Value root,
                                      mlir::Operation *storeAfter,
                                      mlir::Operation *reloadBefore);

mlir::FailureOr<SelectiveSpillMaterialization>
materializeSelectiveTileSpill(wafer::TileRegionOp region, mlir::Value root,
                              mlir::Operation *storeAfter,
                              mlir::Operation *reloadBefore);

struct TileRegionPartition {
  wafer::TileRegionOp head;
  wafer::TileRegionOp tail;
};

mlir::FailureOr<TileRegionPartition>
partitionTileRegionAtDDRBoundary(wafer::TileRegionOp region,
                                 mlir::Operation *tailBegin);

} // namespace wafer::tensor_program_scheduling

#endif // WAFER_LIB_TRANSFORMS_PHYSICALDATAFLOW_COMPLETERANKMATERIALIZATIONINTERNAL_H
