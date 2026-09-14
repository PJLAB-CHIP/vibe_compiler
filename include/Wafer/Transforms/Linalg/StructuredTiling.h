//===- StructuredTiling.h - Interface-driven structured tiling -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_STRUCTUREDTILING_H
#define WAFER_TRANSFORMS_LINALG_STRUCTUREDTILING_H

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

/// Combines one already reduced output tile with its current destination,
/// cloning the original Linalg scalar combiner and preserving its dtype.
mlir::FailureOr<mlir::Value>
combineReductionPartial(mlir::Operation *reduction, unsigned resultNumber,
                        mlir::Value partial, mlir::Value destination,
                        mlir::OpBuilder &builder,
                        std::string *failureReason = nullptr);

/// One actual fixed iteration-domain tile. This is spatial materialization,
/// not temporal loop generation: callers provide a single exact offset/size
/// vector and own all returned operations in their current IR transaction.
struct IterationTileMaterialization {
  llvm::SmallVector<mlir::Operation *, 2> tiledOperations;
  llvm::SmallVector<mlir::Value, 2> tiledValues;
  llvm::SmallVector<mlir::Operation *, 4> generatedSlices;
  llvm::SmallVector<llvm::SmallVector<mlir::OpFoldResult, 4>, 2> resultOffsets;
  llvm::SmallVector<llvm::SmallVector<mlir::OpFoldResult, 4>, 2> resultSizes;
};

mlir::FailureOr<IterationTileMaterialization>
materializeOperationFromIterationTile(
    mlir::Operation *operation, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> iterationSizes,
    std::string *failureReason = nullptr);

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

#endif // WAFER_TRANSFORMS_LINALG_STRUCTUREDTILING_H
