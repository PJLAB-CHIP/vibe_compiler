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
TensorResultIndexingResult deriveTensorResultIndexing(mlir::OpResult result,
                                                      IndexRelationWork &work);

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

/// A current bounded scf.for coordinate, optionally translated by one exact
/// positive affine scale and static origin. The grid describes actual loop
/// instances; it does not select placement, storage, or future iterations.
struct TensorLoopGrid {
  mlir::Value offset;
  mlir::Value induction;
  int64_t base = 0;
  int64_t scale = 1;
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 0;
};

struct TensorLoopGridResult {
  TensorResultIndexingStatus status = TensorResultIndexingStatus::Unsupported;
  std::optional<TensorLoopGrid> grid;
  std::string detail;
};

struct TensorIndexParameter {
  mlir::Value induction;
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 1;
};

/// Standard expressions with the actual bounded SSA parameter bindings.
/// Parameters retain loop steps; their rectangular bounds are not their domain.
struct TensorIndexExpressions {
  mlir::AffineMap map;
  llvm::SmallVector<TensorIndexParameter, 4> parameters;
};

struct TensorIndexExpressionsResult {
  TensorResultIndexingStatus status = TensorResultIndexingStatus::Unsupported;
  std::optional<TensorIndexExpressions> expressions;
  std::string detail;
  bool isExact() const {
    return status == TensorResultIndexingStatus::Exact &&
           expressions.has_value();
  }
};

TensorIndexExpressionsResult
queryTensorIndexExpressions(mlir::MLIRContext *context,
                            llvm::ArrayRef<mlir::OpFoldResult> values,
                            IndexRelationWork &work);

TensorLoopGridResult
queryTensorLoopGrid(mlir::OpFoldResult offset,
                    const IndexRelationLimits &limits = IndexRelationLimits());
std::optional<int64_t> getTensorLoopGridFloor(const TensorLoopGrid &grid,
                                              int64_t coordinate);
std::optional<int64_t> getTensorLoopGridCeil(const TensorLoopGrid &grid,
                                             int64_t coordinate);
std::optional<int64_t> getTensorLoopIndex(const TensorLoopGrid &grid,
                                          int64_t coordinate);

/// Prove both the exact dense source window and its local row-major order.
/// Equal element counts alone do not authorize reshaping a selected tile.
StaticRectangularIndexSetResult getTensorViewTileSource(
    const TensorViewIndexing &indexing, llvm::ArrayRef<int64_t> resultShape,
    const StaticRectangularIndexSet &requested,
    const IndexRelationLimits &limits = IndexRelationLimits());

struct TensorViewTilePiece {
  StaticRectangularIndexSet result;
  StaticRectangularIndexSet source;
};

struct TensorViewTilePieceResult {
  IndexRelationStatus status = IndexRelationStatus::Unsupported;
  /// Exact with no piece means the source domain does not meet this request.
  std::optional<TensorViewTilePiece> piece;
  std::string reason;

  bool isExact() const { return status == IndexRelationStatus::Exact; }
};

/// Intersect one actual source rectangle with the selected view demand and
/// prove the resulting pair of rectangles and their local element order.
TensorViewTilePieceResult getTensorViewTilePiece(
    const TensorViewIndexing &indexing, llvm::ArrayRef<int64_t> resultShape,
    const StaticRectangularIndexSet &requested,
    const StaticRectangularIndexSet &sourceWindow,
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

/// An actual loop's finite partition. Split points are induction coordinates,
/// not tensor coordinates. The final bound need not be a reached iteration.
struct TensorAssemblyReadLoop {
  mlir::Value induction;
  int64_t step = 0;
  llvm::SmallVector<int64_t, 4> boundaries;
};

struct TensorAssemblyReadPiece {
  mlir::Value source;
  mlir::AffineMap sourceOffsets;
  llvm::SmallVector<int64_t, 4> sourceSizes;
  StaticRectangularIndexSet resultWindow;
};

struct TensorAssemblyReadCase {
  llvm::SmallVector<int64_t, 4> lowerBounds;
  llvm::SmallVector<int64_t, 4> upperBounds;
  llvm::SmallVector<TensorAssemblyReadPiece, 4> pieces;
};

struct TensorAssemblyReadResult {
  TensorAssemblyStatus status = TensorAssemblyStatus::NotAssembly;
  llvm::SmallVector<TensorAssemblyReadLoop, 4> loops;
  llvm::SmallVector<TensorAssemblyReadCase, 4> cases;
  std::string detail;
  /// Compact piece coordinates. A proved transparent reshape may make this
  /// differ from the consumer shape while preserving row-major element order.
  llvm::SmallVector<int64_t, 4> shape;

  bool isExact() const { return status == TensorAssemblyStatus::Exact; }
};

/// Resolve the actual bounded read family, preserving holes between reads.
/// Each case has static local pieces; sourceOffsets takes the live induction
/// values in loops order. This is current-index analysis, not a future IR or
/// storage plan. Mutation invalidates all returned handles and proofs.
TensorAssemblyReadResult queryTensorAssemblyRead(
    mlir::Value value, llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<int64_t> sizes,
    const IndexRelationLimits &limits = IndexRelationLimits());

/// One last-writer path through the current structural SSA DAG. Map/domain
/// dimensions are live parameters followed by consumer-local coordinates.
struct TensorSubsetSource {
  mlir::Value source;
  mlir::AffineMap coordinates;
  llvm::SmallVector<IndexDomainCondition, 4> conditions;
};

struct TensorSubsetDemand {
  llvm::SmallVector<TensorIndexParameter, 4> parameters;
  llvm::SmallVector<int64_t, 4> shape;
  mlir::IntegerSet domain;
  llvm::SmallVector<IndexDomainCondition, 4> scopeConditions;
  llvm::SmallVector<TensorSubsetSource, 4> sources;
  /// The visited current DAG contains an insertion with an old destination.
  /// Discovery uses this fact to distinguish assembly choices from pure views.
  bool hasDestinationUpdates = false;
};

struct TensorSubsetDemandResult {
  TensorAssemblyStatus status = TensorAssemblyStatus::Unsupported;
  std::optional<TensorSubsetDemand> demand;
  std::string detail;
  bool isExact() const {
    return status == TensorAssemblyStatus::Exact && demand.has_value();
  }
};

/// Follow both insertion operands and transparent views without enumerating
/// dynamic iterations or converting paths to rectangle products. SSA handles,
/// parameters and proofs are borrowed only for this mutation-free invocation.
TensorSubsetDemandResult
queryTensorSubsetDemand(mlir::Value value,
                        llvm::ArrayRef<mlir::OpFoldResult> offsets,
                        llvm::ArrayRef<int64_t> sizes, mlir::Operation *scope,
                        IndexRelationWork &work);

enum class TensorSubsetBlockStatus : uint8_t {
  Copy,
  Subdivide,
  Unsupported,
  ResourceExhausted,
  BrokenContract,
};

/// A sufficient whole-block proof, derived from one current source relation.
/// Dimensions bind parameters followed by the block's consumer-local origin.
/// Guards imply a dense source rectangle with exactly the consumer's local
/// row-major order. This contains mathematical facts, not an operation plan.
struct TensorSubsetBlock {
  llvm::SmallVector<IndexDomainCondition, 4> guards;
  mlir::AffineMap sourceOffsets;
  llvm::SmallVector<int64_t, 4> sourceSizes;
};

struct TensorSubsetBlockResult {
  TensorSubsetBlockStatus status = TensorSubsetBlockStatus::Unsupported;
  std::optional<TensorSubsetBlock> block;
  /// Deterministic blocking axes, with outer axes first to preserve contiguous
  /// inner copies. An empty list leaves the caller's stable nonunit-axis rule.
  llvm::SmallVector<unsigned, 4> blockingAxes;
  std::string detail;
};

TensorSubsetBlockResult queryTensorSubsetBlock(const TensorSubsetDemand &demand,
                                               const TensorSubsetSource &source,
                                               llvm::ArrayRef<int64_t> shape,
                                               IndexRelationWork &work);

struct StaticTensorSubsetBlock {
  unsigned source;
  StaticRectangularIndexSet destination;
  TensorSubsetBlock copy;
};

struct StaticTensorSubsetBlocksResult {
  IndexRelationStatus status = IndexRelationStatus::Unsupported;
  llvm::SmallVector<StaticTensorSubsetBlock, 4> blocks;
  std::string detail;
};

/// Recover the existing static rectangular fast path from the same DAG
/// conditions and ordered-copy proof. Source offsets may vary with actual
/// parameters; destination rectangles must be fixed over their whole domain.
/// Nonrectangular or varying partitions remain eligible for subdivision.
StaticTensorSubsetBlocksResult
queryStaticTensorSubsetBlocks(const TensorSubsetDemand &demand,
                              IndexRelationWork &work);

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
