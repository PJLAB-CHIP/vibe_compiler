//===- RotatingBuffers.h - Tile RotatingBuffers -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TILE_ROTATINGBUFFERS_H
#define WAFER_TRANSFORMS_TILE_ROTATINGBUFFERS_H

#include "Wafer/Transforms/Tile/LoopPipelining.h"
#include "Wafer/Transforms/Tile/StructuredMaterializationRelations.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"

namespace wafer::compiler::detail {
struct RotatingAllocationBinding {
  mlir::memref::AllocOp allocation;
  mlir::scf::ForOp loop;
  uint32_t multiplicity = 0;
};

struct RotatingAllocationMaterialization {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SmallVector<mlir::Value, 4> slots;
};

struct RotatingAllocationMaterializationResult {
  std::optional<RotatingAllocationMaterialization> materialized;
  std::optional<LoopPipeliningFailure> failure;

  bool succeeded() const { return materialized.has_value(); }
};

RotatingAllocationMaterializationResult materializeRotatingAllocations(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    llvm::ArrayRef<RotatingAllocationBinding> bindings,
    StructuredMaterializationRelations &relations);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_TILE_ROTATINGBUFFERS_H
