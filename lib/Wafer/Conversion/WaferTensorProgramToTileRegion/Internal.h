//===- Internal.h - Tensor program to tile-region internals -*- C++ -*-===//
#pragma once

#include "Wafer/Analysis/Scheduling/LayoutPlanningAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

namespace wafer::tensor_program_to_tile_region {

/// A verified standalone tensor-program scheduling scope. Its entry arguments
/// are inputs followed by output destinations; func.return yields one root per
/// output destination.
class TensorProgramScope {
public:
  explicit TensorProgramScope(mlir::func::FuncOp function)
      : function(function) {}

  mlir::func::FuncOp getFunction() { return function; }
  mlir::Block &getBody() { return function.getBody().front(); }
  mlir::func::ReturnOp getReturn() {
    return mlir::cast<mlir::func::ReturnOp>(getBody().getTerminator());
  }
  unsigned getOutputCount() { return function.getNumResults(); }
  unsigned getInputCount() {
    return function.getNumArguments() - getOutputCount();
  }
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

struct ElementwiseExprValue {
  mlir::Value buffer;
  mlir::AffineMap indexingMap;
};

struct StateSnapshot {
  llvm::DenseMap<mlir::Value, BufferVersions> buffers;
  llvm::DenseMap<mlir::Value, mlir::Value> scalarValues;
  llvm::DenseMap<mlir::Value, mlir::Attribute> scalarAttrs;
  llvm::DenseMap<mlir::Value, mlir::Attribute> tensorAttrs;
  llvm::DenseMap<mlir::Value, mlir::Value> externalBuffers;
  llvm::DenseSet<mlir::Value> writableExternalBuffers;
  llvm::DenseMap<mlir::Value, unsigned> externalOutputIndices;
  llvm::DenseMap<mlir::Value, mlir::Value> directYieldBuffers;
  llvm::DenseMap<mlir::Value, mlir::Value> fillInitScalars;
  llvm::DenseMap<mlir::Value, mlir::Attribute> fillInitAttrs;
};

struct SelectedCollectiveRankGroup {
  llvm::SmallVector<int64_t, 8> ranks;
  int64_t localRank = -1;
};

struct CandidateLoopTile {
  llvm::SmallVector<mlir::OpFoldResult, 4> loopOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> ivs;
  llvm::SmallVector<mlir::OpFoldResult, 4> tileSizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizeBounds;
};

struct ReductionChunk {
  llvm::SmallVector<int64_t, 2> offsets;
  llvm::SmallVector<int64_t, 2> sizes;
};

mlir::func::FuncOp findSingleStandaloneTensorProgram(mlir::ModuleOp module);

mlir::LogicalResult verifyTensorProgramScope(mlir::func::FuncOp function,
                                             std::string *failureReason);

bool isTensorProgramOutputBoundary(TensorProgramScope scope, mlir::Value value,
                                   unsigned outputIndex);

mlir::LogicalResult validateCandidateTile(mlir::RankedTensorType resultType,
                                          llvm::ArrayRef<int64_t> offsets,
                                          llvm::ArrayRef<int64_t> sizes,
                                          std::string *failureReason);

llvm::SmallVector<unsigned, 2> getReductionLoopDims(mlir::linalg::LinalgOp op);

mlir::LogicalResult
buildCandidateLoopTile(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::linalg::LinalgOp op, mlir::AffineMap outputMap,
                       llvm::ArrayRef<mlir::OpFoldResult> candidateOffsets,
                       llvm::ArrayRef<int64_t> candidateSizes,
                       llvm::ArrayRef<int64_t> candidateReductionOffsets,
                       llvm::ArrayRef<int64_t> candidateReductionSizes,
                       CandidateLoopTile &tile, std::string *failureReason);

mlir::FailureOr<uint64_t> getCandidateReductionChunkCount(
    mlir::linalg::LinalgOp root,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason);

mlir::FailureOr<llvm::SmallVector<ReductionChunk, 8>>
buildReductionChunks(mlir::linalg::LinalgOp root,
                     llvm::ArrayRef<int64_t> candidateReductionTileSizes,
                     std::string *failureReason);

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    TensorProgramScope scope, mlir::linalg::LinalgOp root, unsigned outputIndex,
    llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason);

mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    mlir::OpBuilder &builder, TensorProgramScope scope,
    mlir::linalg::LinalgOp root, unsigned outputIndex,
    llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    std::string *failureReason);

mlir::FailureOr<mlir::Value>
getCandidateOutputBoundary(TensorProgramScope scope, unsigned outputIndex,
                           std::string *failureReason);

mlir::Value
insertCandidateRootTile(mlir::linalg::LinalgOp root, mlir::Value tileValue,
                        mlir::Value outputDestination,
                        llvm::ArrayRef<int64_t> candidateTileOffsets,
                        llvm::ArrayRef<int64_t> candidateTileSizes);

mlir::Value
insertCandidateRootTile(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value tileValue, mlir::Value outputDestination,
                        llvm::ArrayRef<mlir::OpFoldResult> candidateTileOffsets,
                        llvm::ArrayRef<int64_t> candidateTileSizes);

mlir::FailureOr<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
collectCandidateRoots(TensorProgramScope scope, bool rejectProducerChains,
                      std::string *failureReason);

mlir::LogicalResult materializeCandidateTileSlices(
    TensorProgramScope scope, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason);

/// Erases only the dead pure tensor producer/view closure left after candidate
/// tiling. The direct single-tile and complete traversal APIs share this
/// cleanup so an untiled source producer cannot become a hidden SPM demand.
void eraseDeadCandidateSupportClosure(TensorProgramScope scope);

mlir::LogicalResult materializeCompleteCandidateTraversal(
    TensorProgramScope scope, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason);

void setFailureReason(std::string *failureReason, llvm::StringRef reason);

std::optional<ComputeReduceKind>
matchExactReductionKind(llvm::ArrayRef<mlir::BlockArgument> iterCarriedArgs,
                        unsigned redPos, mlir::Value expectedReducedValue,
                        llvm::StringRef subject, std::string *failureReason);

class TileRegionBodyEmitter : public WaferTargetImplementationMaterializer {
public:
  explicit TileRegionBodyEmitter(std::string *failureReason,
                                 int64_t currentLogicalRank,
                                 std::optional<TargetImplementationKind>
                                     selectedAlternative = std::nullopt);

  mlir::FailureOr<TileRegionOp> emit(TensorProgramScope scope,
                                     mlir::RewriterBase &rewriter);

private:
  std::string *failureReason;
  int64_t currentLogicalRank = -1;
  std::optional<TargetImplementationKind> selectedAlternative;
  bool selectedAlternativeMaterialized = false;
  llvm::DenseMap<mlir::Value, BufferVersions> buffers;
  llvm::DenseMap<mlir::Value, mlir::Value> scalarValues;
  llvm::DenseMap<mlir::Value, mlir::Attribute> scalarAttrs;
  llvm::DenseMap<mlir::Value, mlir::Attribute> tensorAttrs;
  llvm::DenseMap<mlir::Value, mlir::Value> externalBuffers;
  llvm::DenseSet<mlir::Value> writableExternalBuffers;
  llvm::DenseMap<mlir::Value, unsigned> externalOutputIndices;
  llvm::DenseMap<mlir::Value, mlir::Value> directYieldBuffers;
  llvm::DenseMap<mlir::Value, mlir::Value> fillInitScalars;
  llvm::DenseMap<mlir::Value, mlir::Attribute> fillInitAttrs;

  mlir::LogicalResult fail(llvm::StringRef reason);

  mlir::FailureOr<TileRegionOp> failAndReturn(llvm::StringRef reason);

  mlir::FailureOr<mlir::Value> failValue(llvm::StringRef reason);

  mlir::FailureOr<ElementwiseExprValue>
  failElementwiseExprValue(llvm::StringRef reason);

  mlir::FailureOr<int64_t> failI64(llvm::StringRef reason);

  mlir::FailureOr<SelectedCollectiveRankGroup>
  failSelectedCollectiveRankGroup(llvm::StringRef reason);

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

  mlir::FailureOr<int64_t> getCompactByteSize(mlir::Value buffer,
                                              llvm::StringRef subject);

  mlir::FailureOr<int64_t> getCommunicationId(mlir::IntegerAttr channelId,
                                              llvm::StringRef subject);

  mlir::FailureOr<SelectedCollectiveRankGroup>
  getCollectiveRankGroup(mlir::DenseI64ArrayAttr rankGroup,
                         mlir::DenseIntElementsAttr rankGroups);

  std::optional<ComputeReduceKind>
  inferCollectiveReduceKind(mlir::Region &combiner);

  mlir::LogicalResult requireSingleTensorCollective(mlir::Operation *op);

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

  mlir::LogicalResult convertOp(const OpLayoutPlan &opPlan,
                                mlir::OpBuilder &builder);

  mlir::LogicalResult
  materializeSourceImplementation(mlir::Operation *operation,
                                  mlir::OpBuilder &builder);

  mlir::LogicalResult materializeTargetImplementation(
      mlir::Operation *source, const TargetImplementationCandidate &candidate,
      mlir::OpBuilder &builder) override;

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

  bool allStatic(llvm::ArrayRef<int64_t> values) const;

  mlir::FailureOr<mlir::Value> materializeDdrSubview(
      mlir::Location loc, mlir::Value sourceDdr,
      mlir::RankedTensorType tileTensorType,
      llvm::ArrayRef<mlir::OpFoldResult> offsets, llvm::ArrayRef<int64_t> sizes,
      llvm::ArrayRef<int64_t> strides, mlir::OpBuilder &builder);

  std::optional<unsigned>
  getSingleTensorProgramReturnOperandIndex(mlir::Value value) const;

  bool isLinearInsertChainToTensorProgramReturn(
      mlir::tensor::InsertSliceOp insertSlice,
      unsigned expectedOutputIndex) const;

  bool hasNoObservableDestUseExceptInsert(
      mlir::tensor::InsertSliceOp insertSlice) const;

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

  bool isUnreadLinalgDpsInitUse(mlir::OpOperand &use) const;

  bool onlyFeedsUnreadDpsInit(mlir::Value value,
                              llvm::DenseSet<mlir::Value> &visited) const;

  bool onlyFeedsUnreadDpsInit(mlir::Value value) const;

  bool onlyFeedsGemmOverwriteInit(mlir::Value value,
                                  llvm::DenseSet<mlir::Value> &visited) const;

  bool onlyFeedsGemmOverwriteInit(mlir::Value value) const;

  mlir::LogicalResult verifyNamedLinalgPayloads(TensorProgramScope scope);

  mlir::LogicalResult verifyExactFillPayload(mlir::linalg::FillOp fill);

  mlir::LogicalResult verifyExactGemmPayload(mlir::linalg::LinalgOp op,
                                             llvm::StringRef subject);

  mlir::LogicalResult convertFill(mlir::linalg::FillOp fill,
                                  mlir::OpBuilder &builder);

  mlir::LogicalResult requireZeroFilledGemmInit(mlir::linalg::LinalgOp op,
                                                llvm::StringRef subject);

  bool hasOrderedGemmChunkInit(mlir::linalg::LinalgOp op) const;

  mlir::FailureOr<mlir::Value>
  createOrderedChunkCombine(mlir::Location loc, ComputeReduceKind kind,
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

  std::optional<ComputeElementwiseKind>
  inferCompareKind(mlir::arith::CmpFPredicate predicate);

  std::optional<ComputeElementwiseKind>
  inferCompareKind(mlir::arith::CmpIPredicate predicate);

  std::optional<ComputeReduceKind>
  inferReduceKind(mlir::linalg::GenericOp generic);

  mlir::FailureOr<bool>
  hasOrderedReduceChunkInit(mlir::linalg::GenericOp generic,
                            ComputeReduceKind kind);

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
      mlir::RankedTensorType resultTensorType, bool reciprocalViaDivision,
      mlir::OpBuilder &builder);

  mlir::LogicalResult
  convertElementwiseGenericExpression(mlir::linalg::GenericOp generic,
                                      bool reciprocalViaDivision,
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

  mlir::LogicalResult convertGeneric(mlir::linalg::GenericOp generic,
                                     bool reciprocalViaDivision,
                                     mlir::OpBuilder &builder);

  mlir::LogicalResult finishRegion(TensorProgramScope scope,
                                   TileRegionOp tileRegion,
                                   mlir::OpBuilder &builder);
};

mlir::LogicalResult convertTensorProgramToTileRegionModuleInPlace(
    mlir::ModuleOp module, mlir::MLIRContext *context,
    int64_t currentLogicalRank, std::string *failureReason,
    bool suppressDiagnostics = true, bool verifyResult = true,
    bool populateFallbackFailureReason = true,
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt);

} // namespace wafer::tensor_program_to_tile_region
