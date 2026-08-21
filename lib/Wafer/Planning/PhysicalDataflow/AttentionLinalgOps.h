//===- AttentionLinalgOps.h - Attention Linalg construction -*- C++ -*-===//
#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

namespace wafer::compiler::detail::attention_linalg {

enum class PointwiseKind {
  Identity,
  Maximum,
  Add,
  Subtract,
  Multiply,
  Exp,
  Log,
  ScaleAdd,
  WeightedAdd,
  Divide,
};

mlir::AffineMap mapForDims(mlir::MLIRContext *context, unsigned loopRank,
                           llvm::ArrayRef<unsigned> dims);
llvm::SmallVector<unsigned, 6> sequence(unsigned count);
mlir::Value createEmpty(mlir::OpBuilder &builder, mlir::Location loc,
                        llvm::ArrayRef<int64_t> shape, mlir::Type elementType);
mlir::FailureOr<mlir::Value>
createExactStaticReshape(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value source, mlir::RankedTensorType targetType);
mlir::Value createFill(mlir::OpBuilder &builder, mlir::Location loc,
                       llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                       mlir::Value scalar);
mlir::Value cloneLinalg(mlir::OpBuilder &builder, mlir::linalg::LinalgOp source,
                        mlir::ValueRange inputs, mlir::Value init);
mlir::FailureOr<mlir::Value>
createFloatConvert(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value input, mlir::FloatType targetElementType);
mlir::FailureOr<mlir::Value>
createLogicalValueContraction(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value probability, mlir::Value values,
                              mlir::Value init);
mlir::Value createPointwise(mlir::OpBuilder &builder, mlir::Location loc,
                            llvm::ArrayRef<mlir::Value> inputs,
                            llvm::ArrayRef<mlir::AffineMap> maps,
                            llvm::ArrayRef<int64_t> outputShape,
                            mlir::Type elementType, PointwiseKind kind);
mlir::Value createSlice(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value source,
                        llvm::ArrayRef<mlir::OpFoldResult> offsets,
                        llvm::ArrayRef<int64_t> sizes);

} // namespace wafer::compiler::detail::attention_linalg
