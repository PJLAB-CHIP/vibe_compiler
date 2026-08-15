//===- Internal.h - Tensor program to tile-region internals -*- C++ -*-===//
#pragma once

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/TopologyIds.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <optional>
#include <string>
#include <utility>

namespace wafer {
struct SpatialEdgeFragment;
struct SpatialEdgeStrategy;
} // namespace wafer

namespace wafer::tensor_program_to_tile_region {

/// Structural capability of one output-defining structured operation. This is
/// derived from the current interfaces and never persisted in IR.
enum class StructuredRootCapability {
  Tiled,
  FullTraversalOnly,
  Unsupported,
};

StructuredRootCapability classifyStructuredRoot(mlir::Operation *operation);

/// A verified private tensor-program scheduling scope. Its entry arguments are
/// the unchanged source inputs followed by compiler-created scheduling
/// destinations; func.return yields one root per destination. The appended
/// destinations never cross the CardModule conversion boundary.
class TensorProgramScope {
public:
  TensorProgramScope(mlir::func::FuncOp function,
                     unsigned functionalArgumentCount)
      : function(function), functionalArgumentCount(functionalArgumentCount) {}

  mlir::func::FuncOp getFunction() { return function; }
  mlir::Block &getBody() { return function.getBody().front(); }
  mlir::func::ReturnOp getReturn() {
    return mlir::cast<mlir::func::ReturnOp>(getBody().getTerminator());
  }
  unsigned getOutputCount() { return function.getNumResults(); }
  unsigned getInputCount() { return functionalArgumentCount; }
  mlir::ValueRange getInputs() {
    return mlir::ValueRange(function.getArguments())
        .take_front(getInputCount());
  }
  mlir::ValueRange getOutputs() {
    return mlir::ValueRange(function.getArguments())
        .take_back(getOutputCount());
  }
  mlir::TypeRange getResultTypes() { return function.getResultTypes(); }
  mlir::Location getLoc() { return function.getLoc(); }
  mlir::MLIRContext *getContext() { return function.getContext(); }

private:
  mlir::func::FuncOp function;
  unsigned functionalArgumentCount;
};

struct BufferVersions {
  mlir::Value tensor;
  mlir::Value nTensor;
  mlir::Value cx;
  mlir::Value nCx;
};

struct BatchedGemmAttrs {
  int64_t batchCount = 0;
  llvm::SmallVector<int64_t, 4> lhsBatchDims;
  llvm::SmallVector<int64_t, 4> rhsBatchDims;
  llvm::SmallVector<int64_t, 4> resultBatchDims;
  int64_t lhsMDim = -1;
  int64_t lhsContractingDim = -1;
  int64_t rhsContractingDim = -1;
  int64_t rhsNDim = -1;
  int64_t resultMDim = -1;
  int64_t resultNDim = -1;
};

/// Canonical ordinary 2-D convolution geometry recovered solely from the
/// current Linalg op's iterator kinds and affine indexing maps. Permutations
/// are result-dimension-to-source-dimension mappings accepted by
/// wafer.tile.transpose.
struct OrdinaryConv2DGeometry {
  llvm::SmallVector<int64_t, 4> inputToNHWC;
  llvm::SmallVector<int64_t, 4> weightToXYOI;
  llvm::SmallVector<int64_t, 4> outputToNHWC;
  llvm::SmallVector<int64_t, 4> outputFromNHWC;
  llvm::SmallVector<int64_t, 4> pads;
  llvm::SmallVector<int64_t, 4> unpads;
  llvm::SmallVector<int64_t, 2> stridesHW;
  llvm::SmallVector<int64_t, 2> dilationsHW;
};

mlir::FailureOr<OrdinaryConv2DGeometry>
inferOrdinaryConv2DGeometry(mlir::linalg::LinalgOp op);

struct ElementwiseExprValue {
  mlir::Value buffer;
  mlir::AffineMap indexingMap;
};

struct StateSnapshot {
  llvm::DenseMap<mlir::Value, BufferVersions> buffers;
  llvm::DenseMap<mlir::Value, mlir::Value> scalarValues;
  llvm::DenseMap<mlir::Value, mlir::Attribute> scalarAttrs;
  llvm::DenseMap<mlir::Value, mlir::Attribute> tensorAttrs;
  llvm::DenseMap<mlir::Value, mlir::Value> compilerOwnedBuffers;
  llvm::DenseMap<mlir::Value, mlir::Value> externalBuffers;
  llvm::DenseSet<mlir::Value> writableExternalBuffers;
  llvm::DenseSet<mlir::Value> selectedDDRStageExternalBuffers;
  llvm::DenseMap<mlir::Value, unsigned> externalOutputIndices;
  llvm::DenseMap<mlir::Value, mlir::Value> directYieldBuffers;
  llvm::DenseMap<mlir::Value, mlir::Value> fillInitScalars;
  llvm::DenseMap<mlir::Value, mlir::Attribute> fillInitAttrs;
};

enum class CandidatePeerEndpointKind : uint8_t { Send, Receive };

/// Query-local endpoint attached to one tensor value in a private candidate
/// clone.  TileRegionBodyEmitter consumes it into explicit peer IR and never
/// serializes this record.
struct CandidatePeerEndpoint {
  mlir::Value value;
  /// Exact compiler-owned buffer backing a receive value.  The value may be
  /// rebuilt by tensor tiling, while this SSA allocation remains the stable
  /// relation used to find the surviving ToTensor view.
  mlir::Value carrierBuffer;
  CandidatePeerEndpointKind kind = CandidatePeerEndpointKind::Send;
  TileId peer{0};
  uint64_t bytes = 0;
  int64_t communicationId = 0;
  int64_t payloadSlice = 0;
  /// Position of the selected edge's consumer in the pristine tensor-program
  /// body, plus its operand number.  These fields impose one query-local
  /// receiver-safe order across every Tile; they are deliberately
  /// separate from the logical Direct-DTE message identity.
  uint64_t consumerScheduleOrdinal = 0;
  unsigned consumerOperand = 0;
  /// Query-local identity of the selected exact fragment.  It is used only in
  /// C++ planning/diagnostics; the relation is never encoded in IR.
  const SpatialEdgeFragment *selectedFragment = nullptr;
};

/// Query-local relation between one selected DDR stage and the tensor-level
/// compiler-owned buffer created for it.  This relation is passed directly to
/// TileRegion lowering; it is never encoded in Location or serialized in IR.
struct CandidateSelectedDDRStage {
  mlir::Value buffer;
  const SpatialEdgeStrategy *strategy = nullptr;
};

/// Concrete TileRegion allocation produced for one selected DDR stage.
struct MaterializedSelectedDDRStage {
  mlir::memref::AllocOp allocation;
  const SpatialEdgeStrategy *strategy = nullptr;
};

struct TileRegionEmissionRelations {
  llvm::SmallVector<MaterializedSelectedDDRStage, 8> selectedDDRStages;
  StructuredMaterializationRelations materializedBuffers;
};

struct SelectedCollectivePartitionGroup {
  llvm::SmallVector<int64_t, 8> partitionIds;
  int64_t localParticipantIndex = -1;
};

struct CandidateLoopTile {
  llvm::SmallVector<mlir::OpFoldResult, 4> loopOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> ivs;
  llvm::SmallVector<mlir::OpFoldResult, 4> tileSizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizeBounds;
};

mlir::func::FuncOp findSingleStandaloneTensorProgram(mlir::ModuleOp module);

mlir::LogicalResult verifyTensorProgramScope(mlir::func::FuncOp function,
                                             unsigned functionalArgumentCount,
                                             std::string *failureReason);

bool isTensorProgramOutputBoundary(TensorProgramScope scope, mlir::Value value,
                                   unsigned outputIndex);

mlir::LogicalResult validateCandidateTile(mlir::RankedTensorType resultType,
                                          llvm::ArrayRef<int64_t> offsets,
                                          llvm::ArrayRef<int64_t> sizes,
                                          std::string *failureReason);

llvm::SmallVector<unsigned, 2> getReductionLoopDims(mlir::linalg::LinalgOp op);

mlir::LogicalResult buildCandidateLoopTile(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::linalg::LinalgOp op,
    mlir::AffineMap outputMap,
    llvm::ArrayRef<mlir::OpFoldResult> candidateOffsets,
    llvm::ArrayRef<int64_t> candidateSizes,
    llvm::ArrayRef<mlir::OpFoldResult> candidateReductionOffsets,
    llvm::ArrayRef<int64_t> candidateReductionSizes, CandidateLoopTile &tile,
    std::string *failureReason);

mlir::LogicalResult
verifyReductionSplitNumericLegality(mlir::linalg::LinalgOp root,
                                    bool preservesSequentialReductionOrder,
                                    std::string *failureReason);

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    TensorProgramScope scope, mlir::Operation *root, unsigned outputIndex,
    llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes =
        nullptr);

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned outputIndex,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes =
        nullptr);

mlir::FailureOr<mlir::Value> materializeCandidateOperandConsumerTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    unsigned operandNumber,
    llvm::ArrayRef<mlir::OpFoldResult> operandTileOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> operandTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason);

/// Recursively tiles current-SSA producers of slices created for one actual
/// candidate tile. It never recovers correspondence from names.
mlir::LogicalResult fuseCandidateProducerSlices(
    mlir::Operation *tiledConsumer, mlir::Operation *sourceConsumer,
    TensorProgramScope scope,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    mlir::OpBuilder::Listener *insertionListener, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes =
        nullptr);

mlir::FailureOr<mlir::Value>
getCandidateOutputBoundary(TensorProgramScope scope, unsigned outputIndex,
                           std::string *failureReason);

mlir::Value
insertCandidateRootTile(mlir::Operation *root, mlir::Value tileValue,
                        mlir::Value outputDestination,
                        llvm::ArrayRef<int64_t> candidateTileOffsets,
                        llvm::ArrayRef<int64_t> candidateTileSizes);

mlir::Value
insertCandidateRootTile(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value tileValue, mlir::Value outputDestination,
                        llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
                        llvm::ArrayRef<int64_t> candidateTileSizes);

/// Materializes one selected structured result domain through its configured
/// temporal waves and writes every wave directly into an existing full-domain
/// tensor destination.  The returned SSA value is the updated destination;
/// no compact full-spatial-shard assembly is introduced in SPM.
mlir::FailureOr<mlir::Value> materializeCandidateRootTileIntoDestination(
    TensorProgramScope scope, mlir::Operation *root,
    llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    mlir::Value destination, std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes =
        nullptr);

mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>>
collectCandidateRoots(TensorProgramScope scope, bool rejectProducerChains,
                      std::string *failureReason);

mlir::LogicalResult materializeCandidateOutputTileSlices(
    TensorProgramScope scope, llvm::ArrayRef<SpatialOutputShard> outputShards,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    std::string *failureReason,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes =
        nullptr,
    llvm::ArrayRef<mlir::Operation *> preservedOperations = {});

/// Erases only the dead pure tensor producer/view closure left after candidate
/// tiling. Spatial shard materialization uses this cleanup so an untiled source
/// producer cannot become a hidden SPM demand.
void eraseDeadCandidateSupportClosure(
    TensorProgramScope scope,
    llvm::ArrayRef<mlir::Operation *> preservedOperations = {});

void setFailureReason(std::string *failureReason, llvm::StringRef reason);

std::optional<ComputeReduceKind>
matchExactReductionKind(llvm::ArrayRef<mlir::BlockArgument> iterCarriedArgs,
                        unsigned redPos, mlir::Value expectedReducedValue,
                        llvm::StringRef subject, std::string *failureReason);

class TileRegionBodyEmitter {
public:
  explicit TileRegionBodyEmitter(
      std::string *failureReason, int64_t currentLogicalPartition,
      llvm::ArrayRef<CandidatePeerEndpoint> peerEndpoints = {},
      llvm::ArrayRef<CandidateSelectedDDRStage> selectedDDRStages = {},
      TileRegionEmissionRelations *emissionRelations = nullptr,
      llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {});

  mlir::FailureOr<TileRegionOp> emit(TensorProgramScope scope,
                                     mlir::RewriterBase &rewriter);

private:
  std::string *failureReason;
  int64_t currentLogicalPartition = -1;
  llvm::SmallVector<CandidatePeerEndpoint, 8> peerEndpoints;
  llvm::ArrayRef<CandidateSelectedDDRStage> selectedDDRStages;
  TileRegionEmissionRelations *emissionRelations = nullptr;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint32_t, 2>>
      structuredNodeIds;
  llvm::SmallVector<uint32_t, 2> activeStructuredNodes;
  llvm::DenseMap<mlir::Value, BufferVersions> buffers;
  llvm::DenseMap<mlir::Value, mlir::Value> scalarValues;
  llvm::DenseMap<mlir::Value, mlir::Attribute> scalarAttrs;
  llvm::DenseMap<mlir::Value, mlir::Attribute> tensorAttrs;
  llvm::DenseMap<mlir::Value, mlir::Value> compilerOwnedBuffers;
  llvm::DenseMap<mlir::Value, mlir::Value> externalBuffers;
  llvm::DenseSet<mlir::Value> writableExternalBuffers;
  llvm::DenseSet<mlir::Value> selectedDDRStageExternalBuffers;
  llvm::DenseMap<mlir::Value, unsigned> externalOutputIndices;
  llvm::DenseMap<mlir::Value, mlir::Value> directYieldBuffers;
  llvm::DenseMap<mlir::Value, mlir::Value> fillInitScalars;
  llvm::DenseMap<mlir::Value, mlir::Attribute> fillInitAttrs;
  mlir::LogicalResult fail(llvm::StringRef reason);

  mlir::FailureOr<TileRegionOp> failAndReturn(llvm::StringRef reason);

  mlir::FailureOr<mlir::Value> failValue(llvm::StringRef reason);

  mlir::Location getMaterializationLocation(mlir::Value original) const;

  mlir::FailureOr<ElementwiseExprValue>
  failElementwiseExprValue(llvm::StringRef reason);

  mlir::FailureOr<int64_t> failI64(llvm::StringRef reason);

  mlir::FailureOr<SelectedCollectivePartitionGroup>
  failSelectedCollectivePartitionGroup(llvm::StringRef reason);

  mlir::FailureOr<unsigned> failUnsigned(llvm::StringRef reason);

  mlir::FailureOr<mlir::Type> failType(llvm::StringRef reason);

  mlir::MemRefType makeWaferMemRefType(mlir::RankedTensorType tensorType,
                                       MemorySpace space, MemLayout layout);

  mlir::MemRefType makeSPMMemRefType(mlir::RankedTensorType tensorType,
                                     MemLayout layout);

  mlir::MemRefType makeDDRMemRefType(mlir::RankedTensorType tensorType);

  bool isScalarType(mlir::Type type) const;

  StateSnapshot snapshotState() const;

  void restoreState(const StateSnapshot &snapshot);

  mlir::FailureOr<mlir::Type> convertControlFlowType(mlir::Type type);

  mlir::LogicalResult recordControlFlowValue(mlir::Value original,
                                             mlir::Value converted);

  mlir::FailureOr<mlir::Value>
  materializeControlFlowValue(mlir::Value original, mlir::OpBuilder &builder);

  mlir::FailureOr<mlir::Value>
  materializeDdrBoundary(mlir::Value original, mlir::Value converted,
                         bool readOnly, mlir::RewriterBase &rewriter);

  MemLayout alignedLayoutForTensor(mlir::RankedTensorType tensorType) const;

  void record(mlir::Value original, MemLayout layout, mlir::Value buffer);

  mlir::Value lookup(mlir::Value original, MemLayout layout) const;

  mlir::Value lookupAny(mlir::Value original, MemLayout &layout) const;

  mlir::TypedAttr getScalarSplatAttr(mlir::RankedTensorType tensorType,
                                     mlir::Attribute attr) const;

  bool isElidableConstantBoundary(mlir::Value value) const;

  mlir::FailureOr<mlir::Value>
  materializeTensorConstant(mlir::Value original, mlir::Attribute attr,
                            MemLayout targetLayout, mlir::OpBuilder &builder);

  mlir::FailureOr<mlir::Value> getOrMaterialize(mlir::Value original,
                                                MemLayout targetLayout,
                                                mlir::OpBuilder &builder);

  mlir::FailureOr<mlir::Value>
  getOrMaterializeStructuredInput(mlir::Value original, MemLayout targetLayout,
                                  mlir::OpBuilder &builder);
  void recordOperationResultBuffer(uint32_t structuredNodeId,
                                   mlir::Value buffer);
  void recordOperandBuffer(uint32_t structuredNodeId, mlir::Value buffer);
  void recordOutputBuffer(unsigned outputIndex, mlir::Value buffer);

  mlir::FailureOr<SelectedCollectivePartitionGroup>
  getCollectivePartitionGroup(mlir::DenseI64ArrayAttr partitionGroup,
                              mlir::DenseIntElementsAttr partitionGroups);

  mlir::LogicalResult requireSingleTensorCollective(mlir::Operation *op);

  mlir::FailureOr<mlir::Value> materializeCollectiveInputInResultType(
      mlir::Value input, mlir::Type resultElementType, mlir::Location loc,
      mlir::OpBuilder &builder);

  mlir::LogicalResult convertAllGather(LinalgExtCollectiveAllGatherOp op,
                                       mlir::OpBuilder &builder);

  mlir::LogicalResult
  convertReduceScatter(LinalgExtCollectiveReduceScatterOp op,
                       mlir::OpBuilder &builder);

  mlir::LogicalResult convertAllReduce(LinalgExtCollectiveAllReduceOp op,
                                       mlir::OpBuilder &builder);

  mlir::LogicalResult convertAllToAll(LinalgExtCollectiveAllToAllOp op,
                                      mlir::OpBuilder &builder);

  mlir::LogicalResult
  convertCollectivePermute(LinalgExtCollectiveCollectivePermuteOp op,
                           mlir::OpBuilder &builder);

  mlir::LogicalResult convertLinalgExtCollective(mlir::Operation *op,
                                                 mlir::OpBuilder &builder);

  mlir::LogicalResult initializeBoundary(TensorProgramScope scope,
                                         TileRegionOp tileRegion,
                                         mlir::OpBuilder &builder);

  mlir::LogicalResult convertOp(mlir::Operation *op, mlir::OpBuilder &builder);

  mlir::LogicalResult emitPeerEndpoint(const CandidatePeerEndpoint &endpoint,
                                       mlir::OpBuilder &builder);

  mlir::LogicalResult convertStructuredOp(mlir::Operation *operation,
                                          mlir::OpBuilder &builder);

  mlir::LogicalResult convertSupportOp(mlir::Operation *op,
                                       mlir::OpBuilder &builder);

  mlir::LogicalResult convertNestedOp(mlir::Operation *op,
                                      mlir::OpBuilder &builder);

  mlir::LogicalResult convertScfYield(mlir::scf::YieldOp yield,
                                      mlir::OpBuilder &builder);

  mlir::LogicalResult convertScfBlock(mlir::Block &source,
                                      mlir::OpBuilder &builder);

  void eraseImplicitYield(mlir::Block *block);

  mlir::LogicalResult convertScfIf(mlir::scf::IfOp ifOp,
                                   mlir::OpBuilder &builder);

  mlir::LogicalResult convertScfFor(mlir::scf::ForOp forOp,
                                    mlir::OpBuilder &builder);

  mlir::LogicalResult convertTensorExtract(mlir::tensor::ExtractOp extract,
                                           mlir::OpBuilder &builder);

  mlir::LogicalResult convertTensorPad(mlir::tensor::PadOp pad,
                                       mlir::OpBuilder &builder);

  bool allStatic(llvm::ArrayRef<int64_t> values) const;

  mlir::FailureOr<mlir::Value> materializeMemRefSubview(
      mlir::Location loc, mlir::Value sourceMemRef,
      mlir::RankedTensorType tileTensorType,
      llvm::ArrayRef<mlir::OpFoldResult> offsets, llvm::ArrayRef<int64_t> sizes,
      llvm::ArrayRef<int64_t> strides, mlir::OpBuilder &builder);

  bool hasNoObservableDestUseExceptInsert(
      mlir::tensor::InsertSliceOp insertSlice) const;

  bool
  onlyFeedsTensorInsertDestinations(mlir::Value value,
                                    llvm::DenseSet<mlir::Value> &visited) const;

  mlir::LogicalResult
  convertTensorExtractSlice(mlir::tensor::ExtractSliceOp extractSlice,
                            mlir::OpBuilder &builder);

  mlir::LogicalResult
  convertTensorInsertSlice(mlir::tensor::InsertSliceOp insertSlice,
                           mlir::OpBuilder &builder);

  mlir::LogicalResult convertTensorReshape(mlir::Operation *op,
                                           mlir::Value sourceValue,
                                           mlir::Value resultValue,
                                           mlir::OpBuilder &builder);

  mlir::FailureOr<mlir::Value> getScalarValue(mlir::Value original);

  bool isUnreadDpsInitUse(mlir::OpOperand &use) const;

  bool onlyFeedsUnreadDpsInit(mlir::Value value,
                              llvm::DenseSet<mlir::Value> &visited) const;

  bool onlyFeedsUnreadDpsInit(mlir::Value value) const;

  bool onlyFeedsScalarInitializedComputeInit(
      mlir::Value value, llvm::DenseSet<mlir::Value> &visited) const;

  bool onlyFeedsScalarInitializedComputeInit(mlir::Value value) const;

  mlir::LogicalResult verifyNamedLinalgPayloads(TensorProgramScope scope);

  mlir::LogicalResult verifyExactFillPayload(mlir::linalg::FillOp fill);

  mlir::LogicalResult verifyExactGemmPayload(mlir::linalg::LinalgOp op,
                                             llvm::StringRef subject);

  bool hasExactGemmPayload(mlir::linalg::LinalgOp op) const;

  mlir::LogicalResult convertFill(mlir::linalg::FillOp fill,
                                  mlir::OpBuilder &builder);

  bool hasPositiveZeroFilledComputeInit(mlir::linalg::LinalgOp op) const;

  mlir::FailureOr<mlir::Value>
  createAccumulatorCombine(mlir::Location loc, ComputeReduceKind kind,
                           mlir::Value accumulator, mlir::Value partial,
                           mlir::RankedTensorType resultTensorType,
                           mlir::OpBuilder &builder);

  mlir::LogicalResult convertMatmul(mlir::linalg::LinalgOp op,
                                    mlir::OpBuilder &builder);

  mlir::FailureOr<std::pair<GemmOrientation, GemmOrientation>>
  inferRank2GemmOrientations(mlir::linalg::LinalgOp op);

  mlir::FailureOr<unsigned> findOperandDimForLoop(mlir::AffineMap map,
                                                  unsigned loopDim,
                                                  llvm::StringRef role);

  bool mapContainsLoopDim(mlir::AffineMap map, unsigned loopDim) const;

  mlir::FailureOr<BatchedGemmAttrs>
  inferBatchMatmulAttrs(mlir::linalg::LinalgOp op);

  mlir::LogicalResult convertBatchMatmul(mlir::linalg::LinalgOp op,
                                         mlir::OpBuilder &builder);

  mlir::LogicalResult convertConvolution(mlir::linalg::LinalgOp op,
                                         mlir::OpBuilder &builder);

  std::optional<ComputeElementwiseKind>
  inferCompareKind(mlir::arith::CmpFPredicate predicate);

  std::optional<ComputeElementwiseKind>
  inferCompareKind(mlir::arith::CmpIPredicate predicate);

  std::optional<ComputeReduceKind>
  inferReduceKind(mlir::linalg::GenericOp generic);

  mlir::FailureOr<mlir::TypedAttr> getNeutralReduceInit(mlir::Type elementType,
                                                        ComputeReduceKind kind);

  bool hasReductionIterator(mlir::linalg::GenericOp generic) const;

  mlir::LogicalResult
  getReductionInputDims(mlir::linalg::GenericOp generic,
                        llvm::SmallVectorImpl<int64_t> &inputDims);

  mlir::LogicalResult createReduceOp(
      mlir::Location loc, mlir::Type resultType, ComputeReduceKindAttr kindAttr,
      mlir::Value input, llvm::ArrayRef<int64_t> dims, mlir::Value init,
      mlir::Attribute initAttr, mlir::OpBuilder &builder, mlir::Value &result);

  mlir::LogicalResult convertReduceGeneric(mlir::linalg::GenericOp generic,
                                           mlir::OpBuilder &builder);

  std::optional<unsigned>
  getPassthroughInputIndex(mlir::linalg::GenericOp generic);

  bool isIdentityMap(mlir::AffineMap map, int64_t rank) const;

  mlir::AffineMap getIdentityMap(mlir::RankedTensorType tensorType) const;

  bool isScalarSplatValue(mlir::Attribute attr, double expected) const;

  mlir::Attribute getBlockArgumentConstantAttr(mlir::linalg::GenericOp generic,
                                               mlir::BlockArgument arg) const;

  bool isScalarLikeConstant(mlir::linalg::GenericOp generic, mlir::Value value,
                            double expected) const;

  mlir::FailureOr<ElementwiseExprValue> getElementwiseExprValue(
      llvm::DenseMap<mlir::Value, ElementwiseExprValue> &values,
      mlir::Value value);

  mlir::FailureOr<mlir::Value> materializeElementwiseExprOperand(
      ElementwiseExprValue exprValue, mlir::RankedTensorType resultTensorType,
      mlir::Location loc, mlir::OpBuilder &builder);

  mlir::FailureOr<ElementwiseExprValue>
  createElementwiseFillExprValue(mlir::Location loc, mlir::Value scalar,
                                 mlir::RankedTensorType resultTensorType,
                                 mlir::OpBuilder &builder);

  mlir::FailureOr<ElementwiseExprValue>
  createElementwiseOpExprValue(mlir::Location loc, ComputeElementwiseKind kind,
                               llvm::ArrayRef<ElementwiseExprValue> operands,
                               mlir::RankedTensorType resultTensorType,
                               mlir::OpBuilder &builder);

  mlir::LogicalResult convertElementwiseScalarOp(
      mlir::linalg::GenericOp generic, mlir::Operation *op,
      llvm::DenseMap<mlir::Value, ElementwiseExprValue> &values,
      mlir::RankedTensorType resultTensorType, mlir::OpBuilder &builder);

  mlir::LogicalResult
  convertElementwiseGenericExpression(mlir::linalg::GenericOp generic,
                                      mlir::OpBuilder &builder);

  mlir::LogicalResult convertPassthroughGeneric(mlir::linalg::GenericOp generic,
                                                mlir::OpBuilder &builder);

  mlir::Value createZeroScalar(mlir::Location loc, mlir::Type type,
                               mlir::OpBuilder &builder);

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
  matchTwoWayConcatGeneric(mlir::linalg::GenericOp generic,
                           int64_t &concatAxis);

  mlir::LogicalResult
  convertTwoWayConcatGeneric(mlir::linalg::GenericOp generic,
                             llvm::ArrayRef<mlir::Value> concatInputs,
                             int64_t concatAxis, mlir::OpBuilder &builder);

  mlir::FailureOr<bool>
  tryConvertTiledTwoWayConcatGeneric(mlir::linalg::GenericOp generic,
                                     mlir::OpBuilder &builder);

  mlir::LogicalResult convertGeneric(mlir::linalg::GenericOp generic,
                                     mlir::OpBuilder &builder);

  mlir::LogicalResult finishRegion(TensorProgramScope scope,
                                   TileRegionOp tileRegion,
                                   mlir::OpBuilder &builder);
};

mlir::LogicalResult convertTensorProgramToTileRegionModuleInPlace(
    mlir::ModuleOp module, mlir::MLIRContext *context,
    unsigned functionalArgumentCount, int64_t currentLogicalPartition,
    std::string *failureReason, bool suppressDiagnostics = true,
    bool verifyResult = true, bool populateFallbackFailureReason = true,
    llvm::ArrayRef<CandidatePeerEndpoint> peerEndpoints = {},
    llvm::ArrayRef<CandidateSelectedDDRStage> selectedDDRStages = {},
    TileRegionEmissionRelations *emissionRelations = nullptr,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {});

} // namespace wafer::tensor_program_to_tile_region
