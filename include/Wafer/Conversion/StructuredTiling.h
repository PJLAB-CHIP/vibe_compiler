//===- StructuredTiling.h - Interface-driven structured tiling -*- C++ -*-===//

#ifndef WAFER_CONVERSION_STRUCTUREDTILING_H
#define WAFER_CONVERSION_STRUCTUREDTILING_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace wafer {

struct OperandTileIterationDomain {
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
};

struct OperandTileMaterialization {
  OperandTileIterationDomain iterationDomain;
  llvm::SmallVector<mlir::Operation *, 2> tiledOperations;
  llvm::SmallVector<mlir::Value, 2> tiledValues;
  llvm::SmallVector<mlir::Operation *, 4> generatedSlices;
};

struct PartialReductionTileMaterialization {
  llvm::SmallVector<int, 2> reductionDimensions;
  llvm::SmallVector<mlir::Value, 2> initialValues;
  llvm::SmallVector<mlir::Operation *, 2> partialOperations;
  llvm::SmallVector<mlir::Value, 2> partialValues;
  llvm::SmallVector<mlir::Operation *, 4> generatedSlices;
  llvm::SmallVector<mlir::Operation *, 2> mergeOperations;
  llvm::SmallVector<mlir::Value, 2> mergedValues;
};

mlir::FailureOr<OperandTileMaterialization> materializeConsumerFromOperandTile(
    mlir::Operation *consumer, mlir::OpBuilder &builder, unsigned operandNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes,
    std::string *failureReason = nullptr);

mlir::FailureOr<PartialReductionTileMaterialization>
materializePartialReductionTile(
    mlir::Operation *reduction, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> iterationSizes,
    std::string *failureReason = nullptr);

mlir::FailureOr<PartialReductionTileMaterialization>
materializePartialReductionTile(
    mlir::Operation *reduction, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> iterationSizes,
    mlir::ValueRange resultTileDestinations,
    std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_STRUCTUREDTILING_H
