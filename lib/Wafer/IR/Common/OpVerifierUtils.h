//===- OpVerifierUtils.h - Wafer op verifier helpers ----------*- C++ -*-===//

#ifndef WAFER_IR_OPS_OPVERIFIERUTILS_H
#define WAFER_IR_OPS_OPVERIFIERUTILS_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace wafer::detail {

bool isSPMMemRef(mlir::Type type);
bool isSPMStorage(mlir::Type type);
wafer::MemLayoutAttr getStorageLayout(wafer::StorageType type);
wafer::MemorySpaceAttr getStorageMemorySpace(wafer::StorageType type);
mlir::RankedTensorType getStorageTensorType(wafer::StorageType type);
bool hasStorageLayout(wafer::StorageType type, wafer::MemLayout layout);
bool hasStorageMemorySpace(wafer::StorageType type,
                           wafer::MemorySpace memorySpace);
bool hasStaticMismatch(int64_t lhs, int64_t rhs);
bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result);
bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result);
std::optional<int64_t> getPhysicalTileId(int64_t cardY, int64_t cardX,
                                         int64_t tileY, int64_t tileX,
                                         int64_t cardXCount, int64_t tileYCount,
                                         int64_t tileXCount);
std::optional<int64_t>
getCompactTensorByteSize(mlir::RankedTensorType tensorType);
std::optional<int64_t> getCompactByteSize(mlir::Type type);
int64_t getCompactByteSizeOrUnknown(mlir::Type type);
void appendLayoutRequirement(
    llvm::SmallVectorImpl<wafer::WaferLayoutRequirement> &requirements,
    wafer::WaferValueRole role, unsigned index, wafer::StorageType type);
void appendResourceEffect(
    llvm::SmallVectorImpl<wafer::WaferResourceEffect> &effects,
    wafer::WaferResourceKind resource, wafer::WaferResourceAccess access,
    wafer::WaferValueRole role, unsigned index, int64_t bytes);
mlir::LogicalResult verifyLayoutRequirements(
    mlir::Operation *op,
    llvm::ArrayRef<wafer::WaferLayoutRequirement> requirements);
mlir::LogicalResult
verifyResourceEffects(mlir::Operation *op,
                      llvm::ArrayRef<wafer::WaferResourceEffect> effects);

mlir::LogicalResult verifyCommP2P(mlir::Operation *op, mlir::Value buffer,
                                  mlir::IntegerAttr peer,
                                  mlir::IntegerAttr bytes,
                                  mlir::Type tokenType);
mlir::LogicalResult verifyCommWaitTokens(mlir::Operation *op,
                                         mlir::OperandRange tokens);

struct BatchedGemmDimAttrs {
  llvm::SmallVector<int64_t, 2> lhsBatchDims;
  llvm::SmallVector<int64_t, 2> rhsBatchDims;
  llvm::SmallVector<int64_t, 2> resultBatchDims;
  int64_t batchCount = 0;
  int64_t lhsMDim = -1;
  int64_t lhsContractingDim = -1;
  int64_t rhsContractingDim = -1;
  int64_t rhsNDim = -1;
  int64_t resultMDim = -1;
  int64_t resultNDim = -1;
};

bool hasAnyBatchedGemmAttrs(mlir::Operation *op);
mlir::LogicalResult verifyBatchedGemmTileContract(
    mlir::Operation *op, mlir::RankedTensorType lhsTensor,
    mlir::RankedTensorType rhsTensor, mlir::RankedTensorType resultTensor,
    BatchedGemmDimAttrs &attrs);
mlir::LogicalResult
verifyElementwiseTileContract(mlir::Operation *op,
                              wafer::ComputeElementwiseKind kind,
                              mlir::ValueRange inputs, mlir::Type resultType);
mlir::LogicalResult verifyReduceTileContract(mlir::Operation *op,
                                             mlir::Value input,
                                             mlir::Type resultType);

} // namespace wafer::detail

#endif // WAFER_IR_OPS_OPVERIFIERUTILS_H
