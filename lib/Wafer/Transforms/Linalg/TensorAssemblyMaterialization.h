//===- TensorAssemblyMaterialization.h - Selected tensor demand ----------===//

#ifndef WAFER_TRANSFORMS_LINALG_TENSORASSEMBLYMATERIALIZATION_H
#define WAFER_TRANSFORMS_LINALG_TENSORASSEMBLYMATERIALIZATION_H

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

std::optional<int64_t> resolveStaticIndex(mlir::OpFoldResult value);

struct TensorAssemblyTilePiece {
  mlir::Value value;
  analysis::StaticRectangularIndexSet resultWindow;
};

struct MaterializedTensorAssemblyRead {
  mlir::Value value;
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> sourceReads;
};

analysis::TensorAssemblyReadResult
queryTensorAssemblySlice(mlir::tensor::ExtractSliceOp read,
                         const analysis::IndexRelationLimits &limits =
                             analysis::IndexRelationLimits());

// Consume one already-partitioned current read. The caller owns placement,
// loop mutation, replacement, and any explicitly selected computation fusion.
mlir::FailureOr<MaterializedTensorAssemblyRead>
materializeTensorAssemblyRead(mlir::OpBuilder &builder, mlir::Location location,
                              mlir::RankedTensorType resultType,
                              llvm::ArrayRef<int64_t> sizes,
                              const analysis::TensorAssemblyReadResult &read);

// The caller resolves each source against the current IR and supplies its
// compact tensor. Build the selected demand in result coordinates; no source
// identity, fusion, loop placement or allocation is inferred here.
mlir::FailureOr<mlir::Value> materializeTensorAssemblyTile(
    mlir::OpBuilder &builder, mlir::Location location,
    mlir::RankedTensorType resultType,
    const analysis::StaticRectangularIndexSet &requested,
    llvm::ArrayRef<TensorAssemblyTilePiece> pieces);

} // namespace wafer::compiler::detail

#endif // WAFER_TRANSFORMS_LINALG_TENSORASSEMBLYMATERIALIZATION_H
