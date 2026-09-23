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
  llvm::SmallVector<int64_t, 4> sizes;
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

  /// A complete pure read from one source, proved by the indexing relation.
  /// Partial definitions and destination updates remain materialization leaves.
  const TensorOperandIndexing *getTransparentSource() const {
    if (operands.size() != 1 ||
        operands.front().role != TensorIndexingOperandRole::Source ||
        !operands.front()
             .resultToOperand.hasTotalBoundedAffineMapConstruction())
      return nullptr;
    return &operands.front();
  }
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

/// One current transparent, total, single-source indexing chain. The source
/// handle is borrowed for this IR epoch; the relation contains no allocation
/// facts.
struct TensorViewIndexing {
  mlir::Value source;
  IndexRelation resultToSource;
};

struct TensorViewIndexingResult {
  TensorResultIndexingStatus status = TensorResultIndexingStatus::Unsupported;
  std::optional<TensorViewIndexing> indexing;
  std::string detail;

  bool isExact() const {
    return status == TensorResultIndexingStatus::Exact && indexing.has_value();
  }
};

/// Compose current interface relations before querying a selected tile. Stop
/// at opaque, partial, multi-source or block boundaries; never select a path by
/// operation type. Local order is proved separately for the selected demand.
TensorViewIndexingResult deriveTensorViewIndexing(
    mlir::Value value,
    const IndexRelationLimits &limits = IndexRelationLimits());

/// Prove both the exact dense source window and its local row-major order.
/// Equal element counts alone do not authorize reshaping a selected tile.
StaticRectangularIndexSetResult getTensorViewTileSource(
    const TensorViewIndexing &indexing, llvm::ArrayRef<int64_t> resultShape,
    const StaticRectangularIndexSet &requested,
    const IndexRelationLimits &limits = IndexRelationLimits());

/// Map a finite result demand to one operand of an exact current support
/// result. Insert destinations exclude the overwritten source window; empty
/// images have no rectangles. The query preserves exactness and bounds work.
StaticRectangularIndexSetPiecesResult getTensorOperandDemand(
    const TensorResultIndexing &indexing, const TensorOperandIndexing &operand,
    llvm::ArrayRef<StaticRectangularIndexSet> demand,
    const IndexRelationLimits &limits = IndexRelationLimits());

enum class TensorAssemblyStatus : uint8_t {
  Exact,
  NotAssembly,
  Unsupported,
  ResourceExhausted,
  BrokenContract,
};

struct TensorAssemblySegment {
  mlir::Value source;
  mlir::OpOperand *sourceOperand = nullptr;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
};

struct TensorAssemblyResult {
  TensorAssemblyStatus status = TensorAssemblyStatus::NotAssembly;
  llvm::SmallVector<TensorAssemblySegment, 4> segments;
  std::string detail;

  bool isExact() const {
    return status == TensorAssemblyStatus::Exact && !segments.empty();
  }
};

/// Prove that the current insert chain defines every element exactly once.
/// Segments are returned newest insertion first, matching SSA traversal.
/// This is a source/coverage query only: other uses of the chain and
/// producer fusion eligibility are decisions for its immediate consumer.
TensorAssemblyResult queryTensorAssembly(
    mlir::Value value,
    const IndexRelationLimits &limits = IndexRelationLimits());

struct TensorAssemblyDemandPiece {
  mlir::Value source;
  mlir::OpOperand *sourceOperand = nullptr;
  StaticRectangularIndexSet resultWindow;
  StaticRectangularIndexSet sourceWindow;
};

struct TensorAssemblyDemandResult {
  TensorAssemblyStatus status = TensorAssemblyStatus::NotAssembly;
  llvm::SmallVector<TensorAssemblyDemandPiece, 4> pieces;
  std::string detail;

  bool isExact() const { return status == TensorAssemblyStatus::Exact; }
};

/// Resolve one static demand against current insert_slice SSA updates.
/// The newest write wins; uncovered points read the actual old destination.
/// A demanded tensor.empty point is unsupported because its value is undefined.
/// Windows and borrowed handles are valid only in the current IR epoch.
TensorAssemblyDemandResult queryTensorAssemblyDemand(
    mlir::Value value, const StaticRectangularIndexSet &requested,
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
