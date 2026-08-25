//===- Internal.h - Tensor program to tile-region internals -*- C++ -*-===//
#pragma once

#include "ProducerTileFusion.h"
#include "TensorProgramScope.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/Core/TopologyIds.h"

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

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace wafer {
struct SpatialEdgeFragment;
struct StructuredNodePhysicalRepresentation;
struct StructuredNodeComputeImplementation;
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

struct BufferVersions {
  mlir::Value tensor;
  mlir::Value nTensor;
  mlir::Value cx;
  mlir::Value nCx;
};

struct SelectedNodeRepresentation {
  bool preserveNaturalOperands = false;
  llvm::SmallVector<std::optional<MemLayout>, 4> operandLayouts;
  llvm::SmallVector<uint8_t, 4> sharedOperands;
  llvm::SmallVector<std::optional<MemLayout>, 2> resultLayouts;
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
  /// Exact structured root that owns this endpoint: producer for send,
  /// consumer for receive. This is required because an explicit DDR boundary
  /// intentionally severs the ordinary tensor SSA path.
  uint32_t structuredNodeId = 0;
  /// Query-local identity of the selected exact fragment.  It is used only in
  /// C++ planning/diagnostics; the relation is never encoded in IR.
  const SpatialEdgeFragment *selectedFragment = nullptr;
  /// Independent-DDR baseline endpoints stream this exact persistent carrier
  /// domain through compact SCF wave loops. `carrierBuffer` is then the DDR
  /// stage; offsets/sizes describe the fragment in that stage and tile sizes
  /// bound each actual SPM allocation. Empty tile sizes select the ordinary
  /// single-buffer endpoint path.
  llvm::SmallVector<int64_t, 4> streamOffsets;
  llvm::SmallVector<int64_t, 4> streamSizes;
  llvm::SmallVector<int64_t, 4> streamTileSizes;
};

/// Query-local relation between one selected DDR stage and the tensor-level
/// compiler-owned buffer created for it.  This relation is passed directly to
/// TileRegion lowering; it is never encoded in Location or serialized in IR.
struct CandidateSelectedDDRStage {
  mlir::Value buffer;
  uint32_t producerNode = 0;
  unsigned producerResult = 0;
  StructuredResultIdentityKind producerResultKind =
      StructuredResultIdentityKind::OperationResult;
};

struct PendingPeerToken {
  CandidatePeerEndpointKind kind = CandidatePeerEndpointKind::Send;
  mlir::Value logicalValue;
  mlir::Value token;
};

struct StaticInsertSliceAssembly {
  mlir::Value base;
  llvm::SmallVector<mlir::tensor::InsertSliceOp, 4> inserts;
};

/// Recognizes a verifier-valid static insert_slice chain rooted at
/// tensor.empty whose disjoint inserted boxes cover the complete result.
/// Non-matching values return an empty optional; arithmetic overflow is a
/// compiler failure. The query does not inspect uses or modify IR.
mlir::FailureOr<std::optional<StaticInsertSliceAssembly>>
analyzeCompleteStaticInsertSliceAssembly(mlir::Value value);

/// Concrete TileRegion allocation produced for one selected DDR stage.
struct MaterializedSelectedDDRStage {
  mlir::memref::AllocOp allocation;
  uint32_t producerNode = 0;
  unsigned producerResult = 0;
  StructuredResultIdentityKind producerResultKind =
      StructuredResultIdentityKind::OperationResult;
};

struct TileRegionEmissionRelations {
  llvm::SmallVector<MaterializedSelectedDDRStage, 8> selectedDDRStages;
  StructuredMaterializationRelations materializedBuffers;
};

/// Caller-owned sink for one TileRegion conversion. The body emitter reports
/// concrete emission facts through this narrow API; ownership, deduplication
/// and the aggregate relation contract remain outside the emitter.
class TileRegionEmissionRecorder {
public:
  explicit TileRegionEmissionRecorder(TileRegionEmissionRelations &output)
      : output(output) {}

  void recordSelectedDDRStage(mlir::memref::AllocOp allocation,
                              uint32_t producerNode, unsigned producerResult,
                              StructuredResultIdentityKind producerResultKind);
  void recordOperationResultBuffer(
      uint32_t structuredNodeId, unsigned resultIndex, mlir::Value buffer,
      StructuredResultIdentityKind identityKind =
          StructuredResultIdentityKind::OperationResult);
  void
  recordStructuredComputeOperation(llvm::ArrayRef<uint32_t> structuredNodeIds,
                                   mlir::Operation *operation);
  void recordOperandBuffer(uint32_t structuredNodeId, mlir::Value buffer);
  void recordScratchBuffer(llvm::ArrayRef<uint32_t> structuredNodeIds,
                           mlir::Value buffer);
  void recordOutputBuffer(unsigned outputIndex, mlir::Value buffer);
  void recordCardDDRBuffer(int64_t resourceId, mlir::Value buffer);

private:
  TileRegionEmissionRelations &output;
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

/// Extends one private Tile materialization instance with one caller-owned
/// output destination per functional result.
mlir::LogicalResult appendTileOutputDestinations(mlir::func::FuncOp program,
                                                 std::string *failureReason);

mlir::LogicalResult verifyTensorProgramScope(
    mlir::func::FuncOp function, unsigned functionalArgumentCount,
    std::string *failureReason, unsigned boundaryArgumentCount = 0);

bool isTensorProgramOutputBoundary(TensorProgramScope scope, mlir::Value value,
                                   unsigned outputIndex);

mlir::LogicalResult validateCandidateTile(mlir::RankedTensorType resultType,
                                          llvm::ArrayRef<int64_t> offsets,
                                          llvm::ArrayRef<int64_t> sizes,
                                          std::string *failureReason);

llvm::SmallVector<unsigned, 2> getReductionLoopDims(mlir::linalg::LinalgOp op);

/// Returns true for a projected permutation extended only by constant-zero
/// positions whose indexed tensor extent is exactly one. This is the precise
/// complete-reduction boundary accepted by candidate tiling; arbitrary
/// constants and repeated loop dimensions are not projections.
bool isProjectedPermutationWithUnitConstants(
    mlir::AffineMap map, mlir::RankedTensorType indexedType);

mlir::LogicalResult buildCandidateLoopTile(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::linalg::LinalgOp op,
    mlir::AffineMap outputMap,
    llvm::ArrayRef<mlir::OpFoldResult> candidateOffsets,
    llvm::ArrayRef<int64_t> candidateSizes,
    llvm::ArrayRef<mlir::OpFoldResult> candidateReductionOffsets,
    llvm::ArrayRef<int64_t> candidateReductionSizes, CandidateLoopTile &tile,
    std::string *failureReason);

/// Records that `materialized` is the current-IR realization of every
/// structured node represented by `source`.
void recordStructuredOperationNodeMaterialization(
    mlir::Operation *source, mlir::Operation *materialized,
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes);

/// Builds the selected temporal traversal for one structured result domain.
/// This is the only implementation used by root and producer materialization.
mlir::FailureOr<mlir::Value> materializeConfiguredStructuredTraversal(
    mlir::OpBuilder &builder, TensorProgramScope scope, mlir::Operation *root,
    mlir::linalg::LinalgOp sourceCompute,
    llvm::ArrayRef<mlir::OpFoldResult> requestedOutputOffsets,
    llvm::ArrayRef<int64_t> requestedOutputSizes,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOpNestedTemporalTile> nestedTemporalTiles,
    std::string *failureReason, mlir::Value outputDestination = {},
    llvm::ArrayRef<mlir::OpFoldResult> destinationBaseOffsets = {},
    llvm::SmallVectorImpl<StructuredOperationNodeMapping> *operationNodes =
        nullptr);

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
      TileRegionEmissionRecorder *relationRecorder = nullptr,
      llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
      llvm::ArrayRef<StructuredNodePhysicalRepresentation> representations = {},
      llvm::ArrayRef<StructuredNodeComputeImplementation> implementations = {},
      llvm::ArrayRef<SpatialOutputShard> outputShards = {});

  mlir::FailureOr<TileRegionOp> emit(TensorProgramScope scope,
                                     mlir::RewriterBase &rewriter);

  /// Emits the final TileRegion sequence directly from structured stages.
  /// Roots connected through current tensor SSA stay in one stage; explicit
  /// selected DDR allocations cut that graph. The baseline separately checks
  /// that every resulting stage contains exactly one structured root.
  mlir::FailureOr<TileRegionOp>
  emitStructuredStages(TensorProgramScope scope, mlir::RewriterBase &rewriter);

private:
  std::string *failureReason;
  int64_t currentLogicalPartition = -1;
  llvm::SmallVector<CandidatePeerEndpoint, 8> peerEndpoints;
  llvm::ArrayRef<CandidateSelectedDDRStage> selectedDDRStages;
  llvm::SmallVector<SpatialOutputShard, 4> outputShards;
  llvm::SmallVector<PendingPeerToken, 8> pendingPeerTokens;
  TileRegionEmissionRecorder *relationRecorder = nullptr;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint32_t, 2>>
      structuredNodeIds;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 2>>
      coupledComponentIndices;
  llvm::DenseMap<uint32_t, SelectedNodeRepresentation> selectedRepresentations;
  bool malformedRepresentations = false;
  llvm::DenseMap<uint32_t, StructuredComputeImplementation>
      selectedImplementations;
  bool malformedImplementations = false;
  StructuredComputeImplementation activeImplementation =
      StructuredComputeImplementation::Natural;
  llvm::SmallVector<uint32_t, 2> activeStructuredNodes;
  llvm::DenseSet<uint32_t> currentStageNodes;
  llvm::DenseSet<mlir::Operation *> convertedStructuredOperations;
  llvm::DenseSet<mlir::Operation *> convertingStructuredOperations;
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
  void recordStructuredComputeOperation(mlir::Operation *operation);
  void recordScratchAllocation(mlir::memref::AllocOp allocation);

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

  mlir::LogicalResult awaitPendingPeerToken(size_t index,
                                            mlir::OpBuilder &builder);

  mlir::LogicalResult awaitPendingPeerReceive(mlir::Value logicalValue,
                                              mlir::OpBuilder &builder);

  mlir::LogicalResult makePeerResourceAvailable(CandidatePeerEndpointKind kind,
                                                mlir::OpBuilder &builder);

  mlir::LogicalResult awaitAllPendingPeerTokens(mlir::OpBuilder &builder);

  mlir::LogicalResult convertStructuredOp(mlir::Operation *operation,
                                          mlir::OpBuilder &builder);

  mlir::LogicalResult convertSupportOp(mlir::Operation *op,
                                       mlir::OpBuilder &builder);

  /// Binds a complete, unobserved tensor.insert_slice assembly rooted at
  /// tensor.empty to the matching caller-owned DDR result. Incomplete,
  /// overlapping, dynamic, or otherwise observed assemblies remain on the
  /// ordinary materialization path.
  mlir::LogicalResult
  bindCompleteInsertSliceOutputsToDDR(TensorProgramScope scope);

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

  bool isFullTensorInsertSlice(mlir::tensor::InsertSliceOp insertSlice) const;

  bool isDeferredStaticInsertSliceAssembly(mlir::Value value) const;

  bool onlyFeedsStaticExtractSlicesThroughInsertDestinations(
      mlir::Value value, llvm::DenseSet<mlir::Value> &visited) const;

  mlir::FailureOr<mlir::Value>
  materializeStaticTensorWindow(mlir::Value tensor,
                                llvm::ArrayRef<int64_t> offsets,
                                llvm::ArrayRef<int64_t> sizes,
                                mlir::Location loc, mlir::OpBuilder &builder);

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

  mlir::FailureOr<mlir::Value> getScalarValue(mlir::Value original,
                                              mlir::OpBuilder &builder);

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
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    bool requireOneStructuredRootPerRegion = false,
    llvm::ArrayRef<StructuredNodePhysicalRepresentation> representations = {},
    llvm::ArrayRef<StructuredNodeComputeImplementation> implementations = {},
    llvm::ArrayRef<SpatialOutputShard> outputShards = {});

/// In-place form for a function already owned by its final isolated
/// Card/Tile construction. This avoids manufacturing a synthetic builtin
/// module solely to obtain a conversion anchor.
mlir::LogicalResult convertTensorProgramToTileRegionFunctionInPlace(
    mlir::func::FuncOp function, unsigned functionalArgumentCount,
    int64_t currentLogicalPartition, std::string *failureReason,
    bool suppressDiagnostics = true, bool verifyResult = true,
    bool populateFallbackFailureReason = true,
    llvm::ArrayRef<CandidatePeerEndpoint> peerEndpoints = {},
    llvm::ArrayRef<CandidateSelectedDDRStage> selectedDDRStages = {},
    TileRegionEmissionRelations *emissionRelations = nullptr,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes = {},
    bool requireOneStructuredRootPerRegion = false,
    llvm::ArrayRef<StructuredNodePhysicalRepresentation> representations = {},
    llvm::ArrayRef<StructuredNodeComputeImplementation> implementations = {},
    llvm::ArrayRef<SpatialOutputShard> outputShards = {});

} // namespace wafer::tensor_program_to_tile_region
