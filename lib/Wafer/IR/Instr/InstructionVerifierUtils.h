//===- InstructionVerifierUtils.h - Instruction helpers -------*- C++ -*-===//
//
// Private helpers shared by instruction operation-family implementations.
//
//===----------------------------------------------------------------------===//

#ifndef WAFER_LIB_IR_INSTR_INSTRUCTIONVERIFIERUTILS_H
#define WAFER_LIB_IR_INSTR_INSTRUCTIONVERIFIERUTILS_H

#include "Wafer/IR/WaferDialect.h"

#include "llvm/ADT/SmallVector.h"

namespace wafer::instr_detail {

mlir::LogicalResult verifySPMMemRef(mlir::Operation *op, mlir::Type type,
                                    llvm::StringRef role);
mlir::LogicalResult verifyDDRMemRef(mlir::Operation *op, mlir::Type type,
                                    llvm::StringRef role);
mlir::LogicalResult verifyPositiveI64Attr(mlir::Operation *op,
                                          mlir::IntegerAttr attr,
                                          llvm::StringRef name);
mlir::LogicalResult verifyI64Array(mlir::Operation *op,
                                   mlir::DenseI64ArrayAttr attr,
                                   llvm::StringRef name, int64_t expectedSize,
                                   bool positive);
mlir::LogicalResult verifyUInt32Value(mlir::Operation *op, int64_t value,
                                      llvm::StringRef name);
mlir::LogicalResult verifyUInt16Value(mlir::Operation *op, int64_t value,
                                      llvm::StringRef name);
mlir::LogicalResult verifyUInt32Array(mlir::Operation *op,
                                      mlir::DenseI64ArrayAttr values,
                                      llvm::StringRef name);
mlir::LogicalResult verifyUInt16Array(mlir::Operation *op,
                                      mlir::DenseI64ArrayAttr values,
                                      llvm::StringRef name);
mlir::LogicalResult verifyShapeAttrMatchesBuffer(mlir::Operation *op,
                                                 mlir::Type type,
                                                 mlir::DenseI64ArrayAttr shape,
                                                 llvm::StringRef name);
mlir::FailureOr<int64_t>
computeWindowedOutputDim(mlir::Operation *op, int64_t input, int64_t kernel,
                         int64_t stride, int64_t dilation, int64_t padBefore,
                         int64_t padAfter, int64_t unpadBefore,
                         int64_t unpadAfter, llvm::StringRef role);
mlir::LogicalResult verifySameElementType(mlir::Operation *op,
                                          mlir::RankedTensorType lhs,
                                          mlir::RankedTensorType rhs,
                                          llvm::StringRef message);
mlir::LogicalResult verifyOptionalUInt32Attr(mlir::Operation *op,
                                             mlir::IntegerAttr attr,
                                             llvm::StringRef name);
mlir::LogicalResult verifyOptionalRoundingMode(mlir::Operation *op,
                                               mlir::IntegerAttr attr,
                                               llvm::StringRef name);
} // namespace wafer::instr_detail

#endif // WAFER_LIB_IR_INSTR_INSTRUCTIONVERIFIERUTILS_H
