//===- TensorResultIndexing.h - Static tensor support relations -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_LINALG_TENSORRESULTINDEXING_H
#define WAFER_ANALYSIS_LINALG_TENSORRESULTINDEXING_H

#include "Wafer/Analysis/Linalg/IndexRelation.h"
#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer::analysis {

enum class TensorResultIndexingStatus : uint8_t {
  Exact,
  Unsupported,
  ResourceExhausted,
  BrokenContract,
};

struct TensorOperandIndexing {
  uint32_t operand = 0;
  TensorIndexingOperandRole role = TensorIndexingOperandRole::Source;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> strides;
  IndexRelation resultToOperand;
};

/// Exact static indexing facts for one current pure tensor support result.
/// Operation and value handles are borrowed from the current IR epoch and may
/// only be used synchronously while that IR is unchanged.
struct TensorResultIndexing {
  mlir::OpResult result;
  TensorIndexingTransformKind kind = TensorIndexingTransformKind::Cast;
  llvm::SmallVector<TensorOperandIndexing, 2> operands;
};

struct TensorResultIndexingResult {
  TensorResultIndexingStatus status =
      TensorResultIndexingStatus::BrokenContract;
  std::optional<TensorResultIndexing> indexing;
  std::string detail;

  bool isExact() const {
    return status == TensorResultIndexingStatus::Exact && indexing.has_value();
  }
};

/// Derives result-to-operand relations from the current operation's typed
/// WaferTensorIndexingOpInterface. This query never mutates IR and does not
/// materialize or predict tiles, buffers, movement, or storage.
TensorResultIndexingResult deriveTensorResultIndexing(
    mlir::OpResult result,
    const IndexRelationLimits &limits = IndexRelationLimits());

/// Current structured-compute access maps. These adapters expose dialect
/// semantics only; no fusion, tiling, or placement decisions belong here.
mlir::FailureOr<mlir::AffineMap>
getStructuredOperandMap(mlir::OpOperand &operand);
mlir::FailureOr<mlir::AffineMap> getStructuredResultMap(mlir::OpResult result);

IndexRelationResult deriveIterationOperandRelation(
    mlir::OpOperand &operand, llvm::ArrayRef<int64_t> iterationShape,
    const IndexRelationLimits &limits = IndexRelationLimits());

/// Follow an actual unary transparent support chain from an operand to the
/// specified producer result. Multi-source updates are not transparent views.
IndexRelationResult deriveIterationProducerRelation(
    mlir::OpOperand &operand, llvm::ArrayRef<int64_t> iterationShape,
    mlir::OpResult producer,
    const IndexRelationLimits &limits = IndexRelationLimits());

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_LINALG_TENSORRESULTINDEXING_H
