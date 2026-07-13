//===- WaferGroupToTileRegion.cpp - Group to tile-region conversion -------===//

#include "Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h"

#include "Wafer/Analysis/Group/LayoutPlanningAnalysis.h"
#include "Wafer/Transforms/Passes.h"

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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

using namespace wafer;

namespace wafer {
#define GEN_PASS_DEF_CONVERTGROUPTOTILEREGIONPASS
#include "Wafer/Transforms/WaferPasses.h.inc"
} // namespace wafer

wafer::detail::CheckedStaticTileProductStatus
wafer::detail::checkedStaticTileProduct(llvm::ArrayRef<int64_t> ranges,
                                        llvm::ArrayRef<int64_t> tileSizes,
                                        uint64_t &product) {
  product = 1;
  if (ranges.size() != tileSizes.size())
    return CheckedStaticTileProductStatus::InvalidInput;

  for (auto [range, tileSize] : llvm::zip(ranges, tileSizes)) {
    if (range <= 0 || tileSize <= 0 || tileSize > range)
      return CheckedStaticTileProductStatus::InvalidInput;

    uint64_t unsignedRange = static_cast<uint64_t>(range);
    uint64_t unsignedTileSize = static_cast<uint64_t>(tileSize);
    uint64_t tileCount = unsignedRange / unsignedTileSize;
    tileCount += unsignedRange % unsignedTileSize != 0;
    if (product > std::numeric_limits<uint64_t>::max() / tileCount)
      return CheckedStaticTileProductStatus::Overflow;
    product *= tileCount;
  }
  return CheckedStaticTileProductStatus::Success;
}

wafer::detail::CompleteCandidateExpansionStatus
wafer::detail::checkCompleteCandidateExpansionBudget(
    uint64_t outputTileCount, llvm::ArrayRef<uint64_t> reductionChunkCounts,
    uint64_t &materializationCount) {
  materializationCount = 0;
  if (outputTileCount == 0 || reductionChunkCounts.empty() ||
      llvm::is_contained(reductionChunkCounts, uint64_t{0}))
    return CompleteCandidateExpansionStatus::InvalidInput;

  for (uint64_t reductionChunkCount : reductionChunkCounts) {
    if (outputTileCount >
        std::numeric_limits<uint64_t>::max() / reductionChunkCount)
      return CompleteCandidateExpansionStatus::CountOverflow;
    uint64_t rootMaterializationCount = outputTileCount * reductionChunkCount;
    if (materializationCount >
        std::numeric_limits<uint64_t>::max() - rootMaterializationCount)
      return CompleteCandidateExpansionStatus::CountOverflow;
    materializationCount += rootMaterializationCount;
  }

  if (materializationCount > kCompleteCandidateMaterializationBudget)
    return CompleteCandidateExpansionStatus::BudgetExceeded;
  return CompleteCandidateExpansionStatus::WithinBudget;
}

namespace {

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

static void setFailureReason(std::string *failureReason,
                             llvm::StringRef reason) {
  if (failureReason)
    *failureReason = reason.str();
}

static std::optional<ComputeReduceKind>
matchExactReductionKind(llvm::ArrayRef<mlir::BlockArgument> iterCarriedArgs,
                        unsigned redPos, mlir::Value expectedReducedValue,
                        llvm::StringRef subject, std::string *failureReason) {
  llvm::SmallVector<mlir::Operation *, 1> combinerOps;
  mlir::Value reducedValue =
      mlir::matchReduction(iterCarriedArgs, redPos, combinerOps);
  if (!reducedValue || reducedValue != expectedReducedValue ||
      combinerOps.size() != 1) {
    setFailureReason(failureReason,
                     (subject +
                      " requires one exact combiner wired to the reduced value "
                      "and accumulator")
                         .str());
    return std::nullopt;
  }

  mlir::Operation *combiner = combinerOps.front();
  mlir::Block *combinerBlock = combiner->getBlock();
  if (!combinerBlock ||
      !llvm::all_of(combinerBlock->without_terminator(),
                    [&](mlir::Operation &op) { return &op == combiner; })) {
    setFailureReason(
        failureReason,
        (subject + " cannot erase additional reduction payload operations")
            .str());
    return std::nullopt;
  }
  // Linalg reductions and Wafer collectives both permit an
  // implementation-selected reduction tree. Choosing that tree is distinct
  // from reassociating an ordinary scalar expression.
  if (mlir::isa<mlir::arith::AddFOp>(combiner))
    return ComputeReduceKind::Sum;
  if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(combiner)) {
    if (addi.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none) {
      setFailureReason(
          failureReason,
          (subject + " cannot preserve integer overflow flags").str());
      return std::nullopt;
    }
    return ComputeReduceKind::Sum;
  }
  if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MaxSIOp>(combiner))
    return ComputeReduceKind::Max;
  if (mlir::isa<mlir::arith::MinimumFOp, mlir::arith::MinSIOp>(combiner))
    return ComputeReduceKind::Min;

  if (mlir::isa<mlir::arith::MaxNumFOp, mlir::arith::MinNumFOp>(combiner)) {
    setFailureReason(failureReason,
                     (subject +
                      " cannot preserve maxnum/minnum NaN semantics with the "
                      "current reduce kind")
                         .str());
    return std::nullopt;
  }
  if (mlir::isa<mlir::arith::MaxUIOp, mlir::arith::MinUIOp>(combiner)) {
    setFailureReason(failureReason,
                     (subject +
                      " cannot preserve unsigned min/max semantics with the "
                      "current reduce kind")
                         .str());
    return std::nullopt;
  }

  setFailureReason(failureReason,
                   (subject + " requires an exact sum, signed min/max, or IEEE "
                              "minimum/maximum combiner")
                       .str());
  return std::nullopt;
}

class TileRegionBodyEmitter {
public:
  explicit TileRegionBodyEmitter(std::string *failureReason,
                                 int64_t currentLogicalRank)
      : failureReason(failureReason), currentLogicalRank(currentLogicalRank) {}

  mlir::FailureOr<TileRegionOp>
  emit(GroupOp group, mlir::ValueRange convertedInputs,
       mlir::ValueRange convertedOuts,
       mlir::ConversionPatternRewriter &rewriter) {
    if (currentLogicalRank < 0)
      return failAndReturn("logical-rank must be non-negative");

    llvm::DenseSet<mlir::Value> boundaryValues;
    for (mlir::Value input : group.getInputs()) {
      if (!boundaryValues.insert(input).second)
        return failAndReturn("group boundary SSA values must be unique");
    }
    for (mlir::Value out : group.getOuts()) {
      if (!boundaryValues.insert(out).second)
        return failAndReturn("group boundary SSA values must be unique");
    }
    if (mlir::failed(verifyNamedLinalgPayloads(group)))
      return mlir::failure();

    GroupLayoutPlan layoutPlan;
    if (mlir::failed(collectGroupLayoutPlan(group, layoutPlan)))
      return failAndReturn("group-to-tile-region layout planning failed");
    if (!layoutPlan.succeeded)
      return failAndReturn(layoutPlan.failureReason);

    llvm::SmallVector<mlir::Value, 4> tileRegionInputs;
    for (auto [original, converted] :
         llvm::zip(group.getInputs(), convertedInputs)) {
      mlir::FailureOr<mlir::Value> boundary = materializeDdrBoundary(
          original, converted, /*readOnly=*/true, rewriter);
      if (mlir::failed(boundary))
        return mlir::failure();
      tileRegionInputs.push_back(*boundary);
    }
    for (auto [original, converted] :
         llvm::zip(group.getOuts(), convertedOuts)) {
      mlir::FailureOr<mlir::Value> boundary = materializeDdrBoundary(
          original, converted, /*readOnly=*/false, rewriter);
      if (mlir::failed(boundary))
        return mlir::failure();
      tileRegionInputs.push_back(*boundary);
    }

    llvm::SmallVector<mlir::Type, 2> tileRegionResultTypes;
    for (mlir::Type resultType : group.getResultTypes()) {
      auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
      if (!tensorType)
        return failAndReturn("group result is not a ranked tensor");
      tileRegionResultTypes.push_back(makeDDRMemRefType(tensorType));
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto tileRegion = rewriter.create<TileRegionOp>(
        group.getLoc(), tileRegionResultTypes, tileRegionInputs);
    mlir::Block *tileBlock = new mlir::Block();
    tileRegion.getBody().push_back(tileBlock);
    for (mlir::Value input : tileRegion.getInputs())
      tileBlock->addArgument(input.getType(), input.getLoc());

    rewriter.setInsertionPointToStart(tileBlock);
    if (mlir::failed(initializeBoundary(group, tileRegion, rewriter)))
      return mlir::failure();

    for (OpLayoutPlan &opPlan : layoutPlan.ops) {
      if (mlir::failed(convertOp(opPlan, rewriter)))
        return mlir::failure();
    }

    if (mlir::failed(finishRegion(group, tileRegion, rewriter)))
      return mlir::failure();

    return tileRegion;
  }

private:
  std::string *failureReason;
  int64_t currentLogicalRank = -1;
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

  mlir::LogicalResult fail(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::FailureOr<TileRegionOp> failAndReturn(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::FailureOr<mlir::Value> failValue(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::FailureOr<ElementwiseExprValue>
  failElementwiseExprValue(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::FailureOr<int64_t> failI64(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::FailureOr<SelectedCollectiveRankGroup>
  failSelectedCollectiveRankGroup(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::FailureOr<unsigned> failUnsigned(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::FailureOr<mlir::Type> failType(llvm::StringRef reason) {
    setFailureReason(failureReason, reason);
    return mlir::failure();
  }

  mlir::MemRefType makeWaferMemRefType(mlir::RankedTensorType tensorType,
                                       MemorySpace space, MemLayout layout) {
    auto *context = tensorType.getContext();
    return mlir::MemRefType::get(tensorType.getShape(),
                                 tensorType.getElementType(),
                                 mlir::MemRefLayoutAttrInterface{},
                                 MemoryAttr::get(context, space, layout));
  }

  mlir::MemRefType makeSPMMemRefType(mlir::RankedTensorType tensorType,
                                     MemLayout layout) {
    return makeWaferMemRefType(tensorType, MemorySpace::SPM, layout);
  }

  mlir::MemRefType makeDDRMemRefType(mlir::RankedTensorType tensorType) {
    return makeWaferMemRefType(tensorType, MemorySpace::DDR, MemLayout::Tensor);
  }

  bool isScalarType(mlir::Type type) const {
    return mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(type);
  }

  StateSnapshot snapshotState() const {
    return {buffers,
            scalarValues,
            scalarAttrs,
            tensorAttrs,
            externalBuffers,
            writableExternalBuffers,
            externalOutputIndices,
            directYieldBuffers,
            fillInitScalars,
            fillInitAttrs};
  }

  void restoreState(const StateSnapshot &snapshot) {
    buffers = snapshot.buffers;
    scalarValues = snapshot.scalarValues;
    scalarAttrs = snapshot.scalarAttrs;
    tensorAttrs = snapshot.tensorAttrs;
    externalBuffers = snapshot.externalBuffers;
    writableExternalBuffers = snapshot.writableExternalBuffers;
    externalOutputIndices = snapshot.externalOutputIndices;
    directYieldBuffers = snapshot.directYieldBuffers;
    fillInitScalars = snapshot.fillInitScalars;
    fillInitAttrs = snapshot.fillInitAttrs;
  }

  mlir::FailureOr<mlir::Type> convertControlFlowType(mlir::Type type) {
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type))
      return makeSPMMemRefType(tensorType, MemLayout::Tensor);
    if (isScalarType(type))
      return type;
    return failType("control-flow value is not a ranked tensor or scalar");
  }

  mlir::LogicalResult recordControlFlowValue(mlir::Value original,
                                             mlir::Value converted) {
    if (mlir::isa<mlir::RankedTensorType>(original.getType())) {
      record(original, MemLayout::Tensor, converted);
      return mlir::success();
    }
    if (isScalarType(original.getType())) {
      scalarValues[original] = converted;
      return mlir::success();
    }
    return fail("control-flow value is not a ranked tensor or scalar");
  }

  mlir::FailureOr<mlir::Value>
  materializeControlFlowValue(mlir::Value original, mlir::OpBuilder &builder) {
    if (mlir::isa<mlir::RankedTensorType>(original.getType()))
      return getOrMaterialize(original, MemLayout::Tensor, builder);
    if (isScalarType(original.getType()))
      return getScalarValue(original);
    return failValue("control-flow yield is not a ranked tensor or scalar");
  }

  mlir::FailureOr<mlir::Value>
  materializeDdrBoundary(mlir::Value original, mlir::Value converted,
                         bool readOnly,
                         mlir::ConversionPatternRewriter &rewriter) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
    if (!tensorType) {
      if (mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(
              original.getType()))
        return converted;
      return failValue("group boundary is not a ranked tensor or scalar");
    }
    if (!readOnly && original.getDefiningOp<mlir::tensor::EmptyOp>()) {
      auto alloc = rewriter.create<mlir::memref::AllocOp>(
          original.getLoc(), makeDDRMemRefType(tensorType));
      return alloc.getResult();
    }
    auto toMemref = rewriter.create<mlir::bufferization::ToMemrefOp>(
        original.getLoc(), makeDDRMemRefType(tensorType), converted, readOnly);
    return toMemref.getMemref();
  }

  MemLayout alignedLayoutForTensor(mlir::RankedTensorType tensorType) const {
    return tensorType.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
  }

  void record(mlir::Value original, MemLayout layout, mlir::Value buffer) {
    BufferVersions &versions = buffers[original];
    switch (layout) {
    case MemLayout::Tensor:
      versions.tensor = buffer;
      break;
    case MemLayout::NTensor:
      versions.nTensor = buffer;
      break;
    case MemLayout::Cx:
      versions.cx = buffer;
      break;
    case MemLayout::NCx:
      versions.nCx = buffer;
      break;
    }
  }

  mlir::Value lookup(mlir::Value original, MemLayout layout) const {
    auto it = buffers.find(original);
    if (it == buffers.end())
      return {};
    const BufferVersions &versions = it->second;
    switch (layout) {
    case MemLayout::Tensor:
      return versions.tensor;
    case MemLayout::NTensor:
      return versions.nTensor;
    case MemLayout::Cx:
      return versions.cx;
    case MemLayout::NCx:
      return versions.nCx;
    }
    llvm_unreachable("unknown memory layout");
  }

  mlir::Value lookupAny(mlir::Value original, MemLayout &layout) const {
    auto it = buffers.find(original);
    if (it == buffers.end())
      return {};
    const BufferVersions &versions = it->second;
    if (versions.tensor) {
      layout = MemLayout::Tensor;
      return versions.tensor;
    }
    if (versions.cx) {
      layout = MemLayout::Cx;
      return versions.cx;
    }
    if (versions.nTensor) {
      layout = MemLayout::NTensor;
      return versions.nTensor;
    }
    if (versions.nCx) {
      layout = MemLayout::NCx;
      return versions.nCx;
    }
    return {};
  }

  mlir::TypedAttr getScalarSplatAttr(mlir::RankedTensorType tensorType,
                                     mlir::Attribute attr) const {
    if (auto typed = mlir::dyn_cast<mlir::TypedAttr>(attr)) {
      if (typed.getType() == tensorType.getElementType())
        return typed;
    }

    auto elements = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
    if (!elements || !elements.isSplat() ||
        elements.getElementType() != tensorType.getElementType())
      return {};

    mlir::Type elementType = tensorType.getElementType();
    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
      return mlir::FloatAttr::get(floatType,
                                  elements.getSplatValue<mlir::APFloat>());
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType))
      return mlir::IntegerAttr::get(intType,
                                    elements.getSplatValue<mlir::APInt>());
    return {};
  }

  mlir::FailureOr<mlir::Value>
  materializeTensorConstant(mlir::Value original, mlir::Attribute attr,
                            MemLayout targetLayout, mlir::OpBuilder &builder) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
    if (!tensorType)
      return failValue(
          "constant tensor materialization requires ranked tensor");

    mlir::TypedAttr scalarAttr = getScalarSplatAttr(tensorType, attr);
    if (!scalarAttr)
      return failValue("constant tensor materialization requires splat attr");

    auto scalar =
        builder.create<mlir::arith::ConstantOp>(original.getLoc(), scalarAttr);
    auto tensorBuffer = builder.create<mlir::memref::AllocOp>(
        original.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor));
    builder.create<ComputeFillOp>(original.getLoc(), tensorBuffer.getResult(),
                                  scalar.getResult());
    record(original, MemLayout::Tensor, tensorBuffer.getResult());
    if (targetLayout == MemLayout::Tensor)
      return tensorBuffer.getResult();

    auto materialized = builder.create<LayoutMaterializeOp>(
        original.getLoc(), makeSPMMemRefType(tensorType, targetLayout),
        tensorBuffer.getResult());
    record(original, targetLayout, materialized.getResult());
    return materialized.getResult();
  }

  mlir::FailureOr<mlir::Value> getOrMaterialize(mlir::Value original,
                                                MemLayout targetLayout,
                                                mlir::OpBuilder &builder) {
    if (mlir::Value existing = lookup(original, targetLayout))
      return existing;

    MemLayout sourceLayout = MemLayout::Tensor;
    mlir::Value source = lookupAny(original, sourceLayout);
    if (!source) {
      if (auto attrIt = tensorAttrs.find(original);
          attrIt != tensorAttrs.end()) {
        mlir::FailureOr<mlir::Value> constant = materializeTensorConstant(
            original, attrIt->second, targetLayout, builder);
        if (mlir::succeeded(constant))
          return *constant;
      }

      auto externalIt = externalBuffers.find(original);
      if (externalIt == externalBuffers.end())
        return failValue("missing buffer for value");

      auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
      if (!tensorType)
        return failValue("cannot materialize non-ranked-tensor value");

      auto load = builder.create<StorageLoadOp>(
          original.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor),
          externalIt->second);
      record(original, MemLayout::Tensor, load.getResult());
      source = load.getResult();
      sourceLayout = MemLayout::Tensor;
    }
    if (sourceLayout == targetLayout)
      return source;

    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
    if (!tensorType)
      return failValue("cannot materialize non-ranked-tensor value");

    mlir::Type resultType = makeSPMMemRefType(tensorType, targetLayout);
    auto materialize = builder.create<LayoutMaterializeOp>(original.getLoc(),
                                                           resultType, source);
    record(original, targetLayout, materialize.getResult());
    return materialize.getResult();
  }

  mlir::FailureOr<int64_t> getCompactByteSize(mlir::Value buffer,
                                              llvm::StringRef subject) {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(buffer.getType());
    if (!memrefType) {
      std::string reason = subject.str() + " buffer is not a memref";
      return failI64(reason);
    }
    std::optional<WaferPhysicalTensorInfo> physicalInfo =
        computeWaferPhysicalTensorInfo(memrefType);
    if (!physicalInfo || physicalInfo->compactBytes <= 0) {
      std::string reason =
          subject.str() + " compact byte size is not representable";
      return failI64(reason);
    }
    return physicalInfo->compactBytes;
  }

  mlir::FailureOr<int64_t>
  getCommunicationId(const WaferLinalgExtCollectiveInfo &info,
                     llvm::StringRef subject) {
    if (!info.hasChannelId) {
      std::string reason =
          subject.str() +
          " materialization requires channel_id for stable DTE identity";
      return failI64(reason);
    }
    return info.channelId;
  }

  mlir::FailureOr<SelectedCollectiveRankGroup>
  getCollectiveRankGroup(const WaferLinalgExtCollectiveInfo &info) {
    auto findLocalRank = [&](llvm::ArrayRef<int64_t> ranks)
        -> std::optional<SelectedCollectiveRankGroup> {
      for (auto [index, logicalRank] : llvm::enumerate(ranks)) {
        if (logicalRank != currentLogicalRank)
          continue;
        SelectedCollectiveRankGroup selected;
        selected.ranks.assign(ranks.begin(), ranks.end());
        selected.localRank = static_cast<int64_t>(index);
        return selected;
      }
      return std::nullopt;
    };

    if (!info.rankGroup.empty()) {
      if (std::optional<SelectedCollectiveRankGroup> selected =
              findLocalRank(info.rankGroup))
        return *selected;
      return failSelectedCollectiveRankGroup(
          "logical-rank is not a member of collective rank_group");
    }

    if (!info.hasRankGroups || info.rankGroupSize <= 0)
      return failSelectedCollectiveRankGroup(
          "collective materialization requires rank_group or rank_groups");
    if (info.rankGroups.size() % static_cast<size_t>(info.rankGroupSize) != 0)
      return failSelectedCollectiveRankGroup(
          "collective rank_groups are malformed");

    for (size_t offset = 0; offset < info.rankGroups.size();
         offset += static_cast<size_t>(info.rankGroupSize)) {
      llvm::ArrayRef<int64_t> ranks(info.rankGroups.data() + offset,
                                    static_cast<size_t>(info.rankGroupSize));
      if (std::optional<SelectedCollectiveRankGroup> selected =
              findLocalRank(ranks))
        return *selected;
    }
    return failSelectedCollectiveRankGroup(
        "logical-rank is not a member of collective rank_groups");
  }

  std::optional<ComputeReduceKind>
  inferCollectiveReduceKind(mlir::Region &combiner) {
    if (!combiner.hasOneBlock()) {
      (void)fail("collective reduction materialization requires one combiner "
                 "block");
      return std::nullopt;
    }
    mlir::Block &block = combiner.front();
    if (block.getNumArguments() != 2 ||
        !mlir::isa<LinalgExtCollectiveYieldOp>(block.getTerminator())) {
      (void)fail("collective reduction materialization requires two combiner "
                 "arguments and one yielded value");
      return std::nullopt;
    }
    return matchExactReductionKind(
        llvm::ArrayRef<mlir::BlockArgument>{block.getArgument(1)},
        /*redPos=*/0, block.getArgument(0),
        "collective reduction materialization", failureReason);
  }

  mlir::LogicalResult requireSingleTensorCollective(mlir::Operation *op) {
    if (op->getNumResults() != 1)
      return fail("collective materialization supports one result");
    return mlir::success();
  }

  mlir::LogicalResult convertAllGather(LinalgExtCollectiveAllGatherOp op,
                                       const WaferLinalgExtCollectiveInfo &info,
                                       mlir::OpBuilder &builder) {
    if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
      return mlir::failure();
    if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
      return fail("all_gather materialization supports one input and one out");
    mlir::FailureOr<int64_t> communicationId =
        getCommunicationId(info, "all_gather");
    if (mlir::failed(communicationId))
      return mlir::failure();

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
    if (!resultTensorType)
      return fail("all_gather result is not a ranked tensor");

    mlir::FailureOr<mlir::Value> localChunk =
        getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
    if (mlir::failed(localChunk))
      return mlir::failure();

    auto gatherBuffer = builder.create<mlir::memref::AllocOp>(
        op.getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Tensor));
    mlir::FailureOr<int64_t> bytes =
        getCompactByteSize(*localChunk, "all_gather local chunk");
    mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
        getCollectiveRankGroup(info);
    if (mlir::failed(bytes) || mlir::failed(rankGroup))
      return mlir::failure();

    mlir::MLIRContext *context = builder.getContext();
    builder.create<CommAllGatherOp>(
        op.getLoc(), *localChunk, gatherBuffer.getResult(),
        builder.getI64IntegerAttr(rankGroup->localRank),
        builder.getI64IntegerAttr(
            static_cast<int64_t>(rankGroup->ranks.size())),
        mlir::DenseI64ArrayAttr::get(context, rankGroup->ranks),
        builder.getI64IntegerAttr(*bytes),
        builder.getI64IntegerAttr(*communicationId));
    record(op.getResult(0), MemLayout::Tensor, gatherBuffer.getResult());
    return mlir::success();
  }

  mlir::LogicalResult
  convertReduceScatter(LinalgExtCollectiveReduceScatterOp op,
                       const WaferLinalgExtCollectiveInfo &info,
                       mlir::OpBuilder &builder) {
    if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
      return mlir::failure();
    if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
      return fail(
          "reduce_scatter materialization supports one input and one out");
    mlir::FailureOr<int64_t> communicationId =
        getCommunicationId(info, "reduce_scatter");
    if (mlir::failed(communicationId))
      return mlir::failure();
    std::optional<ComputeReduceKind> kind =
        inferCollectiveReduceKind(op.getCombiner());
    if (!kind)
      return mlir::failure();

    mlir::FailureOr<mlir::Value> input =
        getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
    mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
        getCollectiveRankGroup(info);
    if (mlir::failed(input) || mlir::failed(rankGroup))
      return mlir::failure();
    if (!info.hasAxis)
      return fail("reduce_scatter materialization requires an axis");

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
    if (!resultTensorType)
      return fail("reduce_scatter result must be ranked");
    auto slotType = makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
    auto recvBuffer =
        builder.create<mlir::memref::AllocOp>(op.getLoc(), slotType);
    mlir::FailureOr<int64_t> bytes =
        getCompactByteSize(recvBuffer.getResult(), "reduce_scatter local slot");
    if (mlir::failed(bytes))
      return mlir::failure();

    auto kindAttr = ComputeReduceKindAttr::get(builder.getContext(), *kind);
    auto result = builder.create<CommReduceScatterOp>(
        op.getLoc(), slotType, kindAttr, *input, recvBuffer.getResult(),
        builder.getI64IntegerAttr(info.axis),
        builder.getI64IntegerAttr(rankGroup->localRank),
        builder.getI64IntegerAttr(
            static_cast<int64_t>(rankGroup->ranks.size())),
        mlir::DenseI64ArrayAttr::get(builder.getContext(), rankGroup->ranks),
        builder.getI64IntegerAttr(*bytes),
        builder.getI64IntegerAttr(*communicationId));
    record(op.getResult(0), MemLayout::Tensor, result.getResult());
    return mlir::success();
  }

  mlir::LogicalResult convertAllReduce(LinalgExtCollectiveAllReduceOp op,
                                       const WaferLinalgExtCollectiveInfo &info,
                                       mlir::OpBuilder &builder) {
    if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
      return mlir::failure();
    if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
      return fail("all_reduce materialization supports one input and one out");
    mlir::FailureOr<int64_t> communicationId =
        getCommunicationId(info, "all_reduce");
    if (mlir::failed(communicationId))
      return mlir::failure();
    std::optional<ComputeReduceKind> kind =
        inferCollectiveReduceKind(op.getCombiner());
    if (!kind)
      return mlir::failure();

    mlir::FailureOr<mlir::Value> input =
        getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
    mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
        getCollectiveRankGroup(info);
    if (mlir::failed(input) || mlir::failed(rankGroup))
      return mlir::failure();

    auto inputType = mlir::cast<mlir::MemRefType>((*input).getType());
    auto recvBuffer =
        builder.create<mlir::memref::AllocOp>(op.getLoc(), inputType);
    mlir::FailureOr<int64_t> bytes =
        getCompactByteSize(*input, "all_reduce input");
    if (mlir::failed(bytes))
      return mlir::failure();

    auto kindAttr = ComputeReduceKindAttr::get(builder.getContext(), *kind);
    auto result = builder.create<CommAllReduceOp>(
        op.getLoc(), inputType, kindAttr, *input, recvBuffer.getResult(),
        builder.getI64IntegerAttr(rankGroup->localRank),
        builder.getI64IntegerAttr(
            static_cast<int64_t>(rankGroup->ranks.size())),
        mlir::DenseI64ArrayAttr::get(builder.getContext(), rankGroup->ranks),
        builder.getI64IntegerAttr(*bytes),
        builder.getI64IntegerAttr(*communicationId));
    record(op.getResult(0), MemLayout::Tensor, result.getResult());
    return mlir::success();
  }

  mlir::LogicalResult convertAllToAll(LinalgExtCollectiveAllToAllOp op,
                                      const WaferLinalgExtCollectiveInfo &info,
                                      mlir::OpBuilder &builder) {
    if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
      return mlir::failure();
    if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
      return fail("all_to_all materialization supports one input and one out");
    mlir::FailureOr<int64_t> communicationId =
        getCommunicationId(info, "all_to_all");
    if (mlir::failed(communicationId))
      return mlir::failure();

    mlir::FailureOr<SelectedCollectiveRankGroup> rankGroup =
        getCollectiveRankGroup(info);
    mlir::FailureOr<mlir::Value> input =
        getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
    if (mlir::failed(rankGroup) || mlir::failed(input))
      return mlir::failure();

    int64_t groupSize = static_cast<int64_t>(rankGroup->ranks.size());
    if (groupSize <= 0 || info.splitCount != groupSize)
      return fail(
          "all_to_all materialization requires split_count to match rank_group "
          "size");
    if (!info.hasSplitAxis || !info.hasConcatAxis)
      return fail("all_to_all materialization requires split and concat axes");

    auto inputTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
        op.getInputs().front().getType());
    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
    if (!inputTensorType || !resultTensorType)
      return fail("all_to_all input and result must be ranked tensors");
    if (!inputTensorType.hasStaticShape() || !resultTensorType.hasStaticShape())
      return fail("all_to_all materialization requires static shapes");

    int64_t rank = inputTensorType.getRank();
    int64_t splitAxis = info.splitAxis;
    int64_t concatAxis = info.concatAxis;
    if (splitAxis < 0 || splitAxis >= rank || concatAxis < 0 ||
        concatAxis >= rank || resultTensorType.getRank() != rank)
      return fail("all_to_all materialization has invalid axes");

    int64_t inputSplitDim = inputTensorType.getDimSize(splitAxis);
    int64_t inputConcatDim = inputTensorType.getDimSize(concatAxis);
    if (inputSplitDim <= 0 || inputConcatDim <= 0 ||
        inputSplitDim % groupSize != 0)
      return fail("all_to_all materialization requires evenly split static "
                  "dimensions");

    llvm::SmallVector<int64_t, 4> slotShape(inputTensorType.getShape().begin(),
                                            inputTensorType.getShape().end());
    slotShape[splitAxis] = inputSplitDim / groupSize;
    llvm::SmallVector<int64_t, 4> resultSlotShape(
        resultTensorType.getShape().begin(), resultTensorType.getShape().end());
    resultSlotShape[concatAxis] = inputConcatDim;
    if (slotShape != resultSlotShape)
      return fail("all_to_all materialization slot shapes do not match");

    auto slotTensorType = mlir::RankedTensorType::get(
        slotShape, inputTensorType.getElementType());
    auto slotType = makeSPMMemRefType(slotTensorType, MemLayout::Tensor);
    auto resultType = makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
    auto resultAlloc =
        builder.create<mlir::memref::AllocOp>(op.getLoc(), resultType);
    mlir::Value resultBuffer = resultAlloc.getResult();

    llvm::SmallVector<int64_t, 4> strides(rank, 1);
    auto makeSourceOffsets = [&](int64_t targetIndex) {
      llvm::SmallVector<int64_t, 4> offsets(rank, 0);
      offsets[splitAxis] = targetIndex * slotShape[splitAxis];
      return offsets;
    };
    auto makeResultOffsets = [&](int64_t sourceIndex) {
      llvm::SmallVector<int64_t, 4> offsets(rank, 0);
      offsets[concatAxis] = sourceIndex * inputConcatDim;
      return offsets;
    };
    auto arrayAttr = [&](llvm::ArrayRef<int64_t> values) {
      return mlir::DenseI64ArrayAttr::get(builder.getContext(), values);
    };

    std::optional<WaferPhysicalTensorInfo> slotPhysicalInfo =
        computeWaferPhysicalTensorInfo(slotType);
    if (!slotPhysicalInfo || slotPhysicalInfo->compactBytes <= 0)
      return fail("all_to_all slot compact byte size is not representable");
    int64_t bytes = slotPhysicalInfo->compactBytes;

    struct PendingSend {
      mlir::Value buffer;
      int64_t targetIndex = -1;
      int64_t peer = -1;
    };
    struct PendingRecv {
      mlir::Value buffer;
      int64_t sourceIndex = -1;
      int64_t peer = -1;
    };
    llvm::SmallVector<PendingSend, 4> sends;
    llvm::SmallVector<PendingRecv, 4> recvs;

    int64_t localRank = rankGroup->localRank;
    for (int64_t targetIndex = 0; targetIndex < groupSize; ++targetIndex) {
      llvm::SmallVector<int64_t, 4> sourceOffsets =
          makeSourceOffsets(targetIndex);
      auto extract = builder.create<MoveExtractSliceOp>(
          op.getLoc(), slotType, *input, arrayAttr(sourceOffsets),
          arrayAttr(slotShape), arrayAttr(strides));

      if (targetIndex == localRank) {
        llvm::SmallVector<int64_t, 4> resultOffsets =
            makeResultOffsets(localRank);
        auto insert = builder.create<MoveInsertSliceOp>(
            op.getLoc(), resultType, extract.getResult(), resultBuffer,
            arrayAttr(resultOffsets), arrayAttr(resultSlotShape),
            arrayAttr(strides));
        resultBuffer = insert.getResult();
        continue;
      }

      sends.push_back(
          {extract.getResult(), targetIndex, rankGroup->ranks[targetIndex]});
    }

    for (int64_t sourceIndex = 0; sourceIndex < groupSize; ++sourceIndex) {
      if (sourceIndex == localRank)
        continue;
      auto recvBuffer =
          builder.create<mlir::memref::AllocOp>(op.getLoc(), slotType);
      recvs.push_back(
          {recvBuffer.getResult(), sourceIndex, rankGroup->ranks[sourceIndex]});
    }

    mlir::Type tokenType = builder.getType<mlir::async::TokenType>();
    if (!sends.empty() || !recvs.empty())
      builder.create<SyncLocalFenceOp>(op.getLoc());
    for (int64_t distance = 1; distance < groupSize; ++distance) {
      int64_t targetIndex = (localRank + distance) % groupSize;
      int64_t sourceIndex = (localRank + groupSize - distance) % groupSize;
      auto sendIt = llvm::find_if(sends, [&](const PendingSend &send) {
        return send.targetIndex == targetIndex;
      });
      auto recvIt = llvm::find_if(recvs, [&](const PendingRecv &recv) {
        return recv.sourceIndex == sourceIndex;
      });
      if (sendIt == sends.end() || recvIt == recvs.end())
        return fail("all_to_all protocol could not recover semantic peer slot");
      auto message = DTEMessageAttr::get(builder.getContext(), *communicationId,
                                         DTEProtocolPhase::AllToAll, distance,
                                         targetIndex);
      auto dteSend = builder.create<InstrDTESendOp>(
          op.getLoc(), tokenType, sendIt->buffer,
          builder.getI64IntegerAttr(sendIt->peer),
          builder.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
      auto recvMessage =
          DTEMessageAttr::get(builder.getContext(), *communicationId,
                              DTEProtocolPhase::AllToAll, distance, localRank);
      auto dteRecv = builder.create<InstrDTERecvOp>(
          op.getLoc(), tokenType, recvIt->buffer,
          builder.getI64IntegerAttr(recvIt->peer),
          builder.getI64IntegerAttr(bytes), recvMessage,
          DirectDTEBindingAttr());
      llvm::SmallVector<mlir::Value, 2> roundTokens{dteSend.getToken(),
                                                    dteRecv.getToken()};
      builder.create<InstrDTEWaitOp>(op.getLoc(), roundTokens);
    }

    for (const PendingRecv &recv : recvs) {
      llvm::SmallVector<int64_t, 4> resultOffsets =
          makeResultOffsets(recv.sourceIndex);
      auto insert = builder.create<MoveInsertSliceOp>(
          op.getLoc(), resultType, recv.buffer, resultBuffer,
          arrayAttr(resultOffsets), arrayAttr(resultSlotShape),
          arrayAttr(strides));
      resultBuffer = insert.getResult();
    }

    record(op.getResult(0), MemLayout::Tensor, resultBuffer);
    return mlir::success();
  }

  mlir::LogicalResult
  convertCollectivePermute(LinalgExtCollectiveCollectivePermuteOp op,
                           const WaferLinalgExtCollectiveInfo &info,
                           mlir::OpBuilder &builder) {
    if (mlir::failed(requireSingleTensorCollective(op.getOperation())))
      return mlir::failure();
    if (op.getInputs().size() != 1 || op.getOuts().size() != 1)
      return fail(
          "collective_permute materialization supports one input and one out");
    mlir::FailureOr<int64_t> communicationId =
        getCommunicationId(info, "collective_permute");
    if (mlir::failed(communicationId))
      return mlir::failure();
    if (info.sourceTargetPairs.empty() || info.sourceTargetPairs.size() % 2)
      return fail("collective_permute materialization requires source/target "
                  "pairs");

    mlir::FailureOr<mlir::Value> input =
        getOrMaterialize(op.getInputs().front(), MemLayout::Tensor, builder);
    if (mlir::failed(input))
      return mlir::failure();

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
    if (!resultTensorType)
      return fail("collective_permute result must be ranked");
    auto resultType = makeSPMMemRefType(resultTensorType, MemLayout::Tensor);

    std::optional<int64_t> sendPeer;
    std::optional<int64_t> recvPeer;
    std::optional<int64_t> sendPayloadSlice;
    std::optional<int64_t> recvPayloadSlice;
    bool localCopy = false;
    for (size_t index = 0; index < info.sourceTargetPairs.size(); index += 2) {
      int64_t source = info.sourceTargetPairs[index];
      int64_t target = info.sourceTargetPairs[index + 1];
      if (source == currentLogicalRank) {
        if (target == currentLogicalRank)
          localCopy = true;
        else {
          sendPeer = target;
          sendPayloadSlice = static_cast<int64_t>(index / 2);
        }
      }
      if (target == currentLogicalRank && source != currentLogicalRank) {
        recvPeer = source;
        recvPayloadSlice = static_cast<int64_t>(index / 2);
      }
    }

    mlir::FailureOr<int64_t> bytes =
        getCompactByteSize(*input, "collective_permute input");
    if (mlir::failed(bytes))
      return mlir::failure();

    mlir::Value resultBuffer;
    if (localCopy) {
      resultBuffer = builder.create<MoveCopyOp>(op.getLoc(), resultType, *input)
                         .getResult();
    } else {
      auto alloc =
          builder.create<mlir::memref::AllocOp>(op.getLoc(), resultType);
      mlir::Value zero = createZeroScalar(
          op.getLoc(), resultTensorType.getElementType(), builder);
      if (!zero)
        return fail(
            "collective_permute zero-fill requires numeric element type");
      builder.create<ComputeFillOp>(op.getLoc(), alloc.getResult(), zero);
      resultBuffer = alloc.getResult();
    }

    llvm::SmallVector<mlir::Value, 2> tokens;
    mlir::Type tokenType = builder.getType<mlir::async::TokenType>();
    if (sendPeer) {
      auto message = DTEMessageAttr::get(builder.getContext(), *communicationId,
                                         DTEProtocolPhase::CollectivePermute,
                                         /*round=*/0, *sendPayloadSlice);
      auto send = builder.create<InstrDTESendOp>(
          op.getLoc(), tokenType, *input, builder.getI64IntegerAttr(*sendPeer),
          builder.getI64IntegerAttr(*bytes), message, DirectDTEBindingAttr());
      tokens.push_back(send.getToken());
    }
    if (recvPeer) {
      auto message = DTEMessageAttr::get(builder.getContext(), *communicationId,
                                         DTEProtocolPhase::CollectivePermute,
                                         /*round=*/0, *recvPayloadSlice);
      auto recv = builder.create<InstrDTERecvOp>(
          op.getLoc(), tokenType, resultBuffer,
          builder.getI64IntegerAttr(*recvPeer),
          builder.getI64IntegerAttr(*bytes), message, DirectDTEBindingAttr());
      tokens.push_back(recv.getToken());
    }
    if (!tokens.empty())
      builder.create<InstrDTEWaitOp>(op.getLoc(), tokens);

    record(op.getResult(0), MemLayout::Tensor, resultBuffer);
    return mlir::success();
  }

  mlir::LogicalResult
  convertLinalgExtCollective(mlir::Operation *op,
                             const WaferLinalgExtCollectiveInfo &info,
                             mlir::OpBuilder &builder) {
    switch (info.kind) {
    case WaferLinalgExtCollectiveKind::AllGather:
      return convertAllGather(mlir::cast<LinalgExtCollectiveAllGatherOp>(op),
                              info, builder);
    case WaferLinalgExtCollectiveKind::ReduceScatter:
      return convertReduceScatter(
          mlir::cast<LinalgExtCollectiveReduceScatterOp>(op), info, builder);
    case WaferLinalgExtCollectiveKind::AllReduce:
      return convertAllReduce(mlir::cast<LinalgExtCollectiveAllReduceOp>(op),
                              info, builder);
    case WaferLinalgExtCollectiveKind::AllToAll:
      return convertAllToAll(mlir::cast<LinalgExtCollectiveAllToAllOp>(op),
                             info, builder);
    case WaferLinalgExtCollectiveKind::CollectivePermute:
      return convertCollectivePermute(
          mlir::cast<LinalgExtCollectiveCollectivePermuteOp>(op), info,
          builder);
    }
    llvm_unreachable("unknown linalg-ext collective kind");
  }

  mlir::LogicalResult initializeBoundary(GroupOp group, TileRegionOp tileRegion,
                                         mlir::OpBuilder &builder) {
    mlir::Block &groupBlock = group.getBody().front();
    mlir::Block &tileBlock = tileRegion.getBody().front();
    if (groupBlock.getNumArguments() != tileBlock.getNumArguments())
      return fail("group boundary argument count mismatch");

    for (auto [groupArg, tileArg] :
         llvm::zip(groupBlock.getArguments(), tileBlock.getArguments())) {
      auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(groupArg.getType());
      if (!tensorType) {
        if (mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(
                groupArg.getType())) {
          unsigned argIndex = groupArg.getArgNumber();
          unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
          if (argIndex < inputCount) {
            if (auto constant = group.getInputs()[argIndex]
                                    .getDefiningOp<mlir::arith::ConstantOp>()) {
              mlir::Operation *cloned = builder.clone(*constant.getOperation());
              scalarValues[groupArg] = cloned->getResult(0);
              scalarAttrs[groupArg] = constant.getValue();
              continue;
            }
          }
          scalarValues[groupArg] = tileArg;
          continue;
        }
        return fail("group boundary is not a ranked tensor or scalar");
      }
      externalBuffers[groupArg] = tileArg;
      unsigned argIndex = groupArg.getArgNumber();
      unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
      if (argIndex < inputCount) {
        if (auto constant = group.getInputs()[argIndex]
                                .getDefiningOp<mlir::arith::ConstantOp>())
          tensorAttrs[groupArg] = constant.getValue();
      }
      if (argIndex >= inputCount) {
        writableExternalBuffers.insert(groupArg);
        externalOutputIndices[groupArg] = argIndex - inputCount;
      }
    }
    return mlir::success();
  }

  mlir::LogicalResult convertOp(const OpLayoutPlan &opPlan,
                                mlir::OpBuilder &builder) {
    if (opPlan.kind == OpTilingDemandKind::Failure)
      return fail(opPlan.failureReason);

    mlir::Operation *op = opPlan.op;
    if (opPlan.kind == OpTilingDemandKind::Support)
      return convertSupportOp(op, builder);
    if (opPlan.kind == OpTilingDemandKind::LinalgExtCollective)
      return convertLinalgExtCollective(op, opPlan.collectiveInfo, builder);

    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(op))
      return convertFill(fill, builder);
    if (mlir::isa<mlir::linalg::MatmulOp>(op))
      return convertMatmul(mlir::cast<mlir::linalg::LinalgOp>(op), builder);
    if (mlir::isa<mlir::linalg::BatchMatmulOp>(op))
      return convertBatchMatmul(mlir::cast<mlir::linalg::LinalgOp>(op),
                                builder);
    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op))
      return convertGeneric(generic, builder);

    return fail("unsupported linalg op " + op->getName().getStringRef().str());
  }

  mlir::LogicalResult convertSupportOp(mlir::Operation *op,
                                       mlir::OpBuilder &builder) {
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      if (constant->getNumResults() == 0)
        return mlir::success();

      mlir::Value originalResult = constant.getResult();
      if (onlyFeedsUnreadDpsInit(originalResult))
        return mlir::success();
      auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(originalResult.getType());
      if (!tensorType) {
        mlir::Operation *cloned = builder.clone(*constant.getOperation());
        scalarValues[originalResult] = cloned->getResult(0);
        scalarAttrs[originalResult] = constant.getValue();
        return mlir::success();
      }

      if (getScalarSplatAttr(tensorType, constant.getValue())) {
        mlir::FailureOr<mlir::Value> materialized = materializeTensorConstant(
            originalResult, constant.getValue(), MemLayout::Tensor, builder);
        if (mlir::failed(materialized))
          return mlir::failure();
        tensorAttrs[originalResult] = constant.getValue();
        return mlir::success();
      }

      mlir::Operation *cloned = builder.clone(*constant.getOperation());
      auto ddr = builder.create<mlir::bufferization::ToMemrefOp>(
          constant.getLoc(), makeDDRMemRefType(tensorType),
          cloned->getResult(0), /*read_only=*/true);
      auto load = builder.create<StorageLoadOp>(
          constant.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor),
          ddr.getMemref());
      record(originalResult, MemLayout::Tensor, load.getResult());
      externalBuffers[originalResult] = ddr.getMemref();
      tensorAttrs[originalResult] = constant.getValue();
      return mlir::success();
    }

    if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(op)) {
      if (onlyFeedsUnreadDpsInit(empty.getResult()))
        return mlir::success();
      auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(empty.getType());
      if (!tensorType)
        return fail("tensor.empty result is not a ranked tensor");
      auto alloc = builder.create<mlir::memref::AllocOp>(
          empty.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor));
      record(empty.getResult(), MemLayout::Tensor, alloc.getResult());
      return mlir::success();
    }

    if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractOp>(op))
      return convertTensorExtract(extract, builder);
    if (auto extractSlice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op))
      return convertTensorExtractSlice(extractSlice, builder);
    if (auto insertSlice = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(op))
      return convertTensorInsertSlice(insertSlice, builder);
    if (auto expandShape = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(op))
      return convertTensorReshape(expandShape.getOperation(),
                                  expandShape.getSrc(), expandShape.getResult(),
                                  builder);
    if (auto collapseShape = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(op))
      return convertTensorReshape(collapseShape.getOperation(),
                                  collapseShape.getSrc(),
                                  collapseShape.getResult(), builder);
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op))
      return convertScfIf(ifOp, builder);
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op))
      return convertScfFor(forOp, builder);

    return mlir::success();
  }

  mlir::LogicalResult convertNestedOp(mlir::Operation *op,
                                      mlir::OpBuilder &builder) {
    if (mlir::isa<WaferLinalgExtCollectiveOpInterface>(op))
      return fail("nested collective materialization is not implemented");
    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(op))
      return convertFill(fill, builder);
    if (mlir::isa<mlir::linalg::MatmulOp>(op))
      return convertMatmul(mlir::cast<mlir::linalg::LinalgOp>(op), builder);
    if (mlir::isa<mlir::linalg::BatchMatmulOp>(op))
      return convertBatchMatmul(mlir::cast<mlir::linalg::LinalgOp>(op),
                                builder);
    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op))
      return convertGeneric(generic, builder);
    if (mlir::isa<mlir::arith::ConstantOp, mlir::tensor::EmptyOp,
                  mlir::tensor::ExtractOp, mlir::tensor::ExtractSliceOp,
                  mlir::tensor::InsertSliceOp, mlir::tensor::ExpandShapeOp,
                  mlir::tensor::CollapseShapeOp, mlir::scf::IfOp,
                  mlir::scf::ForOp>(op))
      return convertSupportOp(op, builder);
    return fail("unsupported op inside structured control-flow " +
                op->getName().getStringRef().str());
  }

  mlir::LogicalResult convertScfYield(mlir::scf::YieldOp yield,
                                      mlir::OpBuilder &builder) {
    llvm::SmallVector<mlir::Value, 4> yielded;
    for (mlir::Value value : yield.getResults()) {
      mlir::FailureOr<mlir::Value> converted =
          materializeControlFlowValue(value, builder);
      if (mlir::failed(converted))
        return mlir::failure();
      yielded.push_back(*converted);
    }
    builder.create<mlir::scf::YieldOp>(yield.getLoc(), yielded);
    return mlir::success();
  }

  mlir::LogicalResult convertScfBlock(mlir::Block &source,
                                      mlir::OpBuilder &builder) {
    for (mlir::Operation &op : source.without_terminator()) {
      if (mlir::failed(convertNestedOp(&op, builder)))
        return mlir::failure();
    }
    auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(source.getTerminator());
    if (!yield)
      return fail("structured control-flow body must terminate with scf.yield");
    return convertScfYield(yield, builder);
  }

  void eraseImplicitYield(mlir::Block *block) {
    if (!block || block->empty())
      return;
    if (mlir::isa<mlir::scf::YieldOp>(block->back()))
      block->back().erase();
  }

  mlir::LogicalResult convertScfIf(mlir::scf::IfOp ifOp,
                                   mlir::OpBuilder &builder) {
    mlir::FailureOr<mlir::Value> condition =
        getScalarValue(ifOp.getCondition());
    if (mlir::failed(condition))
      return mlir::failure();

    llvm::SmallVector<mlir::Type, 4> resultTypes;
    for (mlir::Type type : ifOp->getResultTypes()) {
      mlir::FailureOr<mlir::Type> converted = convertControlFlowType(type);
      if (mlir::failed(converted))
        return mlir::failure();
      resultTypes.push_back(*converted);
    }

    bool hasElse = !ifOp.getElseRegion().empty();
    auto convertedIf = builder.create<mlir::scf::IfOp>(
        ifOp.getLoc(), resultTypes, *condition, hasElse);

    {
      StateSnapshot outer = snapshotState();
      mlir::Block *thenBlock = convertedIf.thenBlock();
      eraseImplicitYield(thenBlock);
      mlir::OpBuilder thenBuilder(thenBlock, thenBlock->end());
      if (mlir::failed(convertScfBlock(*ifOp.thenBlock(), thenBuilder)))
        return mlir::failure();
      restoreState(outer);
    }

    if (hasElse) {
      StateSnapshot outer = snapshotState();
      mlir::Block *elseBlock = convertedIf.elseBlock();
      eraseImplicitYield(elseBlock);
      mlir::OpBuilder elseBuilder(elseBlock, elseBlock->end());
      if (mlir::failed(convertScfBlock(*ifOp.elseBlock(), elseBuilder)))
        return mlir::failure();
      restoreState(outer);
    }

    for (auto [original, converted] :
         llvm::zip(ifOp->getResults(), convertedIf->getResults())) {
      if (mlir::failed(recordControlFlowValue(original, converted)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult convertScfFor(mlir::scf::ForOp forOp,
                                    mlir::OpBuilder &builder) {
    mlir::FailureOr<mlir::Value> lowerBound =
        getScalarValue(forOp.getLowerBound());
    mlir::FailureOr<mlir::Value> upperBound =
        getScalarValue(forOp.getUpperBound());
    mlir::FailureOr<mlir::Value> step = getScalarValue(forOp.getStep());
    if (mlir::failed(lowerBound) || mlir::failed(upperBound) ||
        mlir::failed(step))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> initArgs;
    for (mlir::Value init : forOp.getInitArgs()) {
      mlir::FailureOr<mlir::Value> converted =
          materializeControlFlowValue(init, builder);
      if (mlir::failed(converted))
        return mlir::failure();
      initArgs.push_back(*converted);
    }

    auto convertedFor = builder.create<mlir::scf::ForOp>(
        forOp.getLoc(), *lowerBound, *upperBound, *step, initArgs);
    eraseImplicitYield(convertedFor.getBody());

    StateSnapshot outer = snapshotState();
    scalarValues[forOp.getInductionVar()] = convertedFor.getInductionVar();
    for (auto [original, converted] : llvm::zip(
             forOp.getRegionIterArgs(), convertedFor.getRegionIterArgs())) {
      if (mlir::failed(recordControlFlowValue(original, converted)))
        return mlir::failure();
    }

    mlir::OpBuilder bodyBuilder(convertedFor.getBody(),
                                convertedFor.getBody()->end());
    if (mlir::failed(convertScfBlock(*forOp.getBody(), bodyBuilder)))
      return mlir::failure();
    restoreState(outer);

    for (auto [original, converted] :
         llvm::zip(forOp->getResults(), convertedFor->getResults())) {
      if (mlir::failed(recordControlFlowValue(original, converted)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult convertTensorExtract(mlir::tensor::ExtractOp extract,
                                           mlir::OpBuilder &builder) {
    auto bufferIt = externalBuffers.find(extract.getTensor());
    if (bufferIt == externalBuffers.end())
      return fail("tensor.extract from tile-local tensor is not representable");

    llvm::SmallVector<mlir::Value, 4> indices;
    for (mlir::Value index : extract.getIndices()) {
      auto scalarIt = scalarValues.find(index);
      if (scalarIt == scalarValues.end())
        return fail("missing index value for tensor.extract");
      indices.push_back(scalarIt->second);
    }

    auto load = builder.create<mlir::memref::LoadOp>(extract.getLoc(),
                                                     bufferIt->second, indices);
    scalarValues[extract.getResult()] = load.getResult();
    return mlir::success();
  }

  bool allStatic(llvm::ArrayRef<int64_t> values) const {
    return llvm::all_of(values, [](int64_t value) {
      return value != mlir::ShapedType::kDynamic;
    });
  }

  mlir::FailureOr<mlir::Value> materializeDdrSubview(
      mlir::Location loc, mlir::Value sourceDdr,
      mlir::RankedTensorType tileTensorType, llvm::ArrayRef<int64_t> offsets,
      llvm::ArrayRef<int64_t> sizes, llvm::ArrayRef<int64_t> strides,
      mlir::OpBuilder &builder) {
    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(sourceDdr.getType());
    if (!sourceType)
      return failValue("external tile view source is not a memref");
    if (sourceType.getElementType() != tileTensorType.getElementType())
      return failValue("external tile view element type mismatch");
    if (sourceType.getRank() != static_cast<int64_t>(offsets.size()) ||
        sourceType.getRank() != static_cast<int64_t>(sizes.size()) ||
        sourceType.getRank() != static_cast<int64_t>(strides.size()))
      return failValue("external tile view rank mismatch");

    auto subviewType = mlir::cast<mlir::MemRefType>(
        mlir::memref::SubViewOp::inferRankReducedResultType(
            tileTensorType.getShape(), sourceType, offsets, sizes, strides));
    auto subview = builder.create<mlir::memref::SubViewOp>(
        loc, subviewType, sourceDdr, offsets, sizes, strides);
    return subview.getResult();
  }

  std::optional<unsigned>
  getSingleGroupYieldOperandIndex(mlir::Value value) const {
    if (!value.hasOneUse())
      return std::nullopt;
    mlir::OpOperand &use = *value.getUses().begin();
    if (!mlir::isa<GroupYieldOp>(use.getOwner()))
      return std::nullopt;
    return use.getOperandNumber();
  }

  bool isLinearInsertChainToGroupYield(mlir::tensor::InsertSliceOp insertSlice,
                                       unsigned expectedOutputIndex) const {
    mlir::Value current = insertSlice.getResult();
    while (current.hasOneUse()) {
      mlir::OpOperand &use = *current.getUses().begin();
      if (mlir::isa<GroupYieldOp>(use.getOwner()))
        return use.getOperandNumber() == expectedOutputIndex;

      auto nextInsert =
          mlir::dyn_cast<mlir::tensor::InsertSliceOp>(use.getOwner());
      if (!nextInsert || nextInsert.getDest() != current)
        return false;
      current = nextInsert.getResult();
    }
    return false;
  }

  bool hasNoObservableDestUseExceptInsert(
      mlir::tensor::InsertSliceOp insertSlice) const {
    mlir::Value dest = insertSlice.getDest();
    mlir::OpOperand *destOperand = &insertSlice->getOpOperand(1);
    for (mlir::OpOperand &use : dest.getUses()) {
      if (&use == destOperand)
        continue;
      if (isUnreadLinalgDpsInitUse(use))
        continue;
      auto extractSlice =
          mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(use.getOwner());
      if (extractSlice && extractSlice.getSource() == dest &&
          onlyFeedsUnreadDpsInit(extractSlice.getResult()))
        continue;
      return false;
    }
    return true;
  }

  mlir::LogicalResult
  convertTensorExtractSlice(mlir::tensor::ExtractSliceOp extractSlice,
                            mlir::OpBuilder &builder) {
    if (!allStatic(extractSlice.getStaticOffsets()) ||
        !allStatic(extractSlice.getStaticSizes()) ||
        !allStatic(extractSlice.getStaticStrides()))
      return fail("dynamic tensor.extract_slice is not representable");

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(extractSlice.getType());
    if (!resultTensorType)
      return fail("tensor.extract_slice result is not a ranked tensor");

    if (onlyFeedsUnreadDpsInit(extractSlice.getResult()))
      return mlir::success();

    bool hasFillInitMarker = fillInitAttrs.contains(extractSlice.getSource()) ||
                             fillInitScalars.contains(extractSlice.getSource());
    if (hasFillInitMarker &&
        onlyFeedsGemmOverwriteInit(extractSlice.getResult())) {
      if (auto attrIt = fillInitAttrs.find(extractSlice.getSource());
          attrIt != fillInitAttrs.end())
        fillInitAttrs[extractSlice.getResult()] = attrIt->second;
      if (auto scalarIt = fillInitScalars.find(extractSlice.getSource());
          scalarIt != fillInitScalars.end())
        fillInitScalars[extractSlice.getResult()] = scalarIt->second;
      return mlir::success();
    }

    if (auto externalIt = externalBuffers.find(extractSlice.getSource());
        externalIt != externalBuffers.end()) {
      mlir::FailureOr<mlir::Value> tileView = materializeDdrSubview(
          extractSlice.getLoc(), externalIt->second, resultTensorType,
          extractSlice.getStaticOffsets(), extractSlice.getStaticSizes(),
          extractSlice.getStaticStrides(), builder);
      if (mlir::failed(tileView))
        return mlir::failure();

      auto load = builder.create<StorageLoadOp>(
          extractSlice.getLoc(),
          makeSPMMemRefType(resultTensorType, MemLayout::Tensor), *tileView);
      record(extractSlice.getResult(), MemLayout::Tensor, load.getResult());
      return mlir::success();
    }

    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(extractSlice.getSource(), MemLayout::Tensor, builder);
    if (mlir::failed(source))
      return mlir::failure();

    mlir::MLIRContext *context = extractSlice.getContext();
    auto offsets =
        mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticOffsets());
    auto sizes =
        mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticSizes());
    auto strides =
        mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticStrides());
    auto move = builder.create<MoveExtractSliceOp>(
        extractSlice.getLoc(),
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor), *source,
        offsets, sizes, strides);
    record(extractSlice.getResult(), MemLayout::Tensor, move.getResult());
    if (auto attrIt = fillInitAttrs.find(extractSlice.getSource());
        attrIt != fillInitAttrs.end())
      fillInitAttrs[extractSlice.getResult()] = attrIt->second;
    if (auto scalarIt = fillInitScalars.find(extractSlice.getSource());
        scalarIt != fillInitScalars.end())
      fillInitScalars[extractSlice.getResult()] = scalarIt->second;
    return mlir::success();
  }

  mlir::LogicalResult
  convertTensorInsertSlice(mlir::tensor::InsertSliceOp insertSlice,
                           mlir::OpBuilder &builder) {
    if (!allStatic(insertSlice.getStaticOffsets()) ||
        !allStatic(insertSlice.getStaticSizes()) ||
        !allStatic(insertSlice.getStaticStrides()))
      return fail("dynamic tensor.insert_slice is not representable");

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(insertSlice.getType());
    if (!resultTensorType)
      return fail("tensor.insert_slice result is not a ranked tensor");
    auto sourceTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(insertSlice.getSourceType());
    if (!sourceTensorType)
      return fail("tensor.insert_slice source is not a ranked tensor");

    auto externalIt = externalBuffers.find(insertSlice.getDest());
    auto outputIndexIt = externalOutputIndices.find(insertSlice.getDest());
    std::optional<unsigned> yieldIndex =
        getSingleGroupYieldOperandIndex(insertSlice.getResult());
    bool isDirectYield = outputIndexIt != externalOutputIndices.end() &&
                         yieldIndex && outputIndexIt->second == *yieldIndex;
    bool isLinearInsertChain =
        outputIndexIt != externalOutputIndices.end() &&
        isLinearInsertChainToGroupYield(insertSlice, outputIndexIt->second);
    if (externalIt != externalBuffers.end() &&
        writableExternalBuffers.contains(insertSlice.getDest()) &&
        outputIndexIt != externalOutputIndices.end() &&
        hasNoObservableDestUseExceptInsert(insertSlice) &&
        (isDirectYield || isLinearInsertChain)) {
      mlir::Value externalBuffer = externalIt->second;
      unsigned outputIndex = outputIndexIt->second;
      mlir::FailureOr<mlir::Value> source =
          getOrMaterialize(insertSlice.getSource(), MemLayout::Tensor, builder);
      if (mlir::failed(source))
        return mlir::failure();

      mlir::FailureOr<mlir::Value> tileView = materializeDdrSubview(
          insertSlice.getLoc(), externalBuffer, sourceTensorType,
          insertSlice.getStaticOffsets(), insertSlice.getStaticSizes(),
          insertSlice.getStaticStrides(), builder);
      if (mlir::failed(tileView))
        return mlir::failure();

      builder.create<StorageStoreOp>(insertSlice.getLoc(), *source, *tileView);
      mlir::Value result = insertSlice.getResult();
      externalBuffers[result] = externalBuffer;
      writableExternalBuffers.insert(result);
      externalOutputIndices[result] = outputIndex;
      directYieldBuffers[result] = externalBuffer;
      return mlir::success();
    }

    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(insertSlice.getSource(), MemLayout::Tensor, builder);
    mlir::FailureOr<mlir::Value> dest =
        getOrMaterialize(insertSlice.getDest(), MemLayout::Tensor, builder);
    if (mlir::failed(source) || mlir::failed(dest))
      return mlir::failure();

    mlir::MLIRContext *context = insertSlice.getContext();
    auto offsets =
        mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticOffsets());
    auto sizes =
        mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticSizes());
    auto strides =
        mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticStrides());
    auto move = builder.create<MoveInsertSliceOp>(
        insertSlice.getLoc(),
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor), *source, *dest,
        offsets, sizes, strides);
    record(insertSlice.getResult(), MemLayout::Tensor, move.getResult());
    return mlir::success();
  }

  mlir::LogicalResult convertTensorReshape(mlir::Operation *op,
                                           mlir::Value sourceValue,
                                           mlir::Value resultValue,
                                           mlir::OpBuilder &builder) {
    MemLayout sourceLayout = MemLayout::Tensor;
    mlir::Value source = lookupAny(sourceValue, sourceLayout);
    if (!source) {
      mlir::FailureOr<mlir::Value> loaded =
          getOrMaterialize(sourceValue, MemLayout::Tensor, builder);
      if (mlir::failed(loaded))
        return mlir::failure();
      source = *loaded;
      sourceLayout = MemLayout::Tensor;
    }

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(resultValue.getType());
    if (!resultTensorType)
      return fail("tensor reshape result is not a ranked tensor");

    auto reshape = builder.create<ViewReshapeOp>(
        op->getLoc(), makeSPMMemRefType(resultTensorType, sourceLayout),
        source);
    record(resultValue, sourceLayout, reshape.getResult());
    return mlir::success();
  }

  mlir::FailureOr<mlir::Value> getScalarValue(mlir::Value original) {
    auto it = scalarValues.find(original);
    if (it == scalarValues.end())
      return failValue("missing scalar value for tile compute");
    return it->second;
  }

  bool isUnreadLinalgDpsInitUse(mlir::OpOperand &use) const {
    auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(use.getOwner());
    if (!linalgOp)
      return false;
    llvm::ArrayRef<mlir::BlockArgument> outputArgs =
        linalgOp.getRegionOutputArgs();
    for (int64_t index = 0; index < linalgOp.getNumDpsInits(); ++index) {
      if (linalgOp.getDpsInitOperand(index) != &use)
        continue;
      return static_cast<size_t>(index) < outputArgs.size() &&
             outputArgs[index].use_empty();
    }
    return false;
  }

  bool onlyFeedsUnreadDpsInit(mlir::Value value,
                              llvm::DenseSet<mlir::Value> &visited) const {
    if (value.use_empty() || !visited.insert(value).second)
      return false;
    for (mlir::OpOperand &use : value.getUses()) {
      if (isUnreadLinalgDpsInitUse(use))
        continue;
      auto extractSlice =
          mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(use.getOwner());
      if (!extractSlice || extractSlice.getSource() != value ||
          !onlyFeedsUnreadDpsInit(extractSlice.getResult(), visited))
        return false;
    }
    return true;
  }

  bool onlyFeedsUnreadDpsInit(mlir::Value value) const {
    llvm::DenseSet<mlir::Value> visited;
    return onlyFeedsUnreadDpsInit(value, visited);
  }

  bool onlyFeedsGemmOverwriteInit(mlir::Value value,
                                  llvm::DenseSet<mlir::Value> &visited) const {
    if (value.use_empty() || !visited.insert(value).second)
      return false;

    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *owner = use.getOwner();
      if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::BatchMatmulOp>(
              owner)) {
        auto dpsOp = mlir::cast<mlir::linalg::LinalgOp>(owner);
        if (dpsOp.isDpsInit(&use))
          continue;
        return false;
      }

      auto extractSlice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(owner);
      if (!extractSlice || extractSlice.getSource() != value ||
          !onlyFeedsGemmOverwriteInit(extractSlice.getResult(), visited))
        return false;
    }
    return true;
  }

  bool onlyFeedsGemmOverwriteInit(mlir::Value value) const {
    llvm::DenseSet<mlir::Value> visited;
    return onlyFeedsGemmOverwriteInit(value, visited);
  }

  mlir::LogicalResult verifyNamedLinalgPayloads(GroupOp group) {
    mlir::WalkResult result = group.getBody().walk([&](mlir::Operation *op) {
      if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(op)) {
        if (mlir::failed(verifyExactFillPayload(fill)))
          return mlir::WalkResult::interrupt();
      } else if (mlir::isa<mlir::linalg::MatmulOp>(op)) {
        if (mlir::failed(verifyExactGemmPayload(
                mlir::cast<mlir::linalg::LinalgOp>(op), "matmul")))
          return mlir::WalkResult::interrupt();
      } else if (mlir::isa<mlir::linalg::BatchMatmulOp>(op)) {
        if (mlir::failed(verifyExactGemmPayload(
                mlir::cast<mlir::linalg::LinalgOp>(op), "batch matmul")))
          return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return result.wasInterrupted() ? mlir::failure() : mlir::success();
  }

  mlir::LogicalResult verifyExactFillPayload(mlir::linalg::FillOp fill) {
    mlir::linalg::LinalgOp op = fill;
    if (op->getNumRegions() != 1 || op->getRegion(0).empty() ||
        op.getRegionInputArgs().size() != 1 ||
        op.getRegionOutputArgs().size() != 1)
      return fail("linalg.fill requires the canonical scalar payload");

    mlir::Block &body = op->getRegion(0).front();
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getValues().size() != 1 ||
        yield.getValues().front() != op.getRegionInputArgs().front() ||
        !body.without_terminator().empty())
      return fail("linalg.fill requires the canonical scalar payload");
    return mlir::success();
  }

  mlir::LogicalResult verifyExactGemmPayload(mlir::linalg::LinalgOp op,
                                             llvm::StringRef subject) {
    if (op->getNumRegions() != 1 || op->getRegion(0).empty() ||
        op.getRegionInputArgs().size() != 2 ||
        op.getRegionOutputArgs().size() != 1)
      return fail(
          (subject + " requires an exact multiply-accumulate payload").str());

    mlir::Block &body = op->getRegion(0).front();
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    llvm::SmallVector<mlir::Operation *, 2> payloadOps;
    for (mlir::Operation &payloadOp : body.without_terminator())
      payloadOps.push_back(&payloadOp);
    if (!yield || yield.getValues().size() != 1 || payloadOps.size() != 2)
      return fail(
          (subject + " requires an exact multiply-accumulate payload").str());

    mlir::Value lhs = op.getRegionInputArgs()[0];
    mlir::Value rhs = op.getRegionInputArgs()[1];
    mlir::Value accumulator = op.getRegionOutputArgs()[0];
    auto matchesPair = [](mlir::Value first, mlir::Value second,
                          mlir::Value expectedFirst,
                          mlir::Value expectedSecond) {
      return (first == expectedFirst && second == expectedSecond) ||
             (first == expectedSecond && second == expectedFirst);
    };

    mlir::Value sum;
    if (auto mul = mlir::dyn_cast<mlir::arith::MulFOp>(payloadOps[0])) {
      auto add = mlir::dyn_cast<mlir::arith::AddFOp>(payloadOps[1]);
      if (!add || mul.getFastmath() != mlir::arith::FastMathFlags::none ||
          add.getFastmath() != mlir::arith::FastMathFlags::none ||
          !matchesPair(mul.getLhs(), mul.getRhs(), lhs, rhs) ||
          !matchesPair(add.getLhs(), add.getRhs(), mul.getResult(),
                       accumulator))
        return fail(
            (subject + " requires an exact multiply-accumulate payload").str());
      sum = add.getResult();
    } else if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(payloadOps[0])) {
      auto add = mlir::dyn_cast<mlir::arith::AddIOp>(payloadOps[1]);
      if (!add ||
          mul.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none ||
          add.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none ||
          !matchesPair(mul.getLhs(), mul.getRhs(), lhs, rhs) ||
          !matchesPair(add.getLhs(), add.getRhs(), mul.getResult(),
                       accumulator))
        return fail(
            (subject + " requires an exact multiply-accumulate payload").str());
      sum = add.getResult();
    } else {
      return fail(
          (subject + " requires an exact multiply-accumulate payload").str());
    }
    if (yield.getValues().front() != sum)
      return fail(
          (subject + " requires an exact multiply-accumulate payload").str());
    return mlir::success();
  }

  mlir::LogicalResult convertFill(mlir::linalg::FillOp fill,
                                  mlir::OpBuilder &builder) {
    mlir::linalg::LinalgOp op = fill;
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
        fill->getNumResults() != 1)
      return fail("unsupported linalg.fill arity");
    if (mlir::failed(verifyExactFillPayload(fill)))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> value = getScalarValue(op.getDpsInputs()[0]);
    if (mlir::failed(value))
      return mlir::failure();

    fillInitScalars[fill.getResult(0)] = *value;
    if (auto attrIt = scalarAttrs.find(op.getDpsInputs()[0]);
        attrIt != scalarAttrs.end())
      fillInitAttrs[fill.getResult(0)] = attrIt->second;

    // The target GEMM is overwrite-only. Preserve its source-level identity
    // proof without issuing a dead fill or loading an output buffer that the
    // successful GEMM path cannot consume.
    if (onlyFeedsGemmOverwriteInit(fill.getResult(0)))
      return mlir::success();

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(fill.getResult(0).getType());
    if (!resultTensorType)
      return fail("linalg.fill result is not a ranked tensor");
    auto result = builder.create<mlir::memref::AllocOp>(
        fill.getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Tensor));
    builder.create<ComputeFillOp>(fill.getLoc(), result.getResult(), *value);
    record(fill.getResult(0), MemLayout::Tensor, result.getResult());
    return mlir::success();
  }

  mlir::LogicalResult requireZeroFilledGemmInit(mlir::linalg::LinalgOp op,
                                                llvm::StringRef subject) {
    if (op.getNumDpsInits() != 1)
      return fail((subject + " requires exactly one DPS init").str());

    auto isPositiveZero = [](mlir::Attribute attr) {
      if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
        const llvm::APFloat &value = floatAttr.getValue();
        return value.isZero() && !value.isNegative();
      }
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
        return intAttr.getValue().isZero();
      if (auto elements = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
        if (!elements.isSplat())
          return false;
        if (mlir::isa<mlir::FloatType>(elements.getElementType())) {
          llvm::APFloat value = elements.getSplatValue<mlir::APFloat>();
          return value.isZero() && !value.isNegative();
        }
        if (mlir::isa<mlir::IntegerType>(elements.getElementType()))
          return elements.getSplatValue<mlir::APInt>().isZero();
      }
      return false;
    };

    auto attrIt = fillInitAttrs.find(op.getDpsInits().front());
    if (attrIt == fillInitAttrs.end() || !isPositiveZero(attrIt->second)) {
      return fail((subject +
                   " requires a provable zero-filled DPS init because "
                   "wafer.tile.gemm has overwrite semantics")
                      .str());
    }
    return mlir::success();
  }

  mlir::LogicalResult convertMatmul(mlir::linalg::LinalgOp op,
                                    mlir::OpBuilder &builder) {
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
        op->getNumResults() != 1)
      return fail("unsupported matmul arity");
    if (mlir::failed(verifyExactGemmPayload(op, "matmul")))
      return mlir::failure();
    if (mlir::failed(requireZeroFilledGemmInit(op, "matmul")))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> lhs =
        getOrMaterialize(op.getDpsInputs()[0], MemLayout::Cx, builder);
    mlir::FailureOr<mlir::Value> rhs =
        getOrMaterialize(op.getDpsInputs()[1], MemLayout::Cx, builder);
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultTensorType)
      return fail("matmul result is not a ranked tensor");

    auto gemm = builder.create<ComputeGemmOp>(
        op->getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Cx), *lhs,
        *rhs);
    record(op->getResult(0), MemLayout::Cx, gemm.getResult());
    return mlir::success();
  }

  mlir::FailureOr<unsigned> findOperandDimForLoop(mlir::AffineMap map,
                                                  unsigned loopDim,
                                                  llvm::StringRef role) {
    for (auto [operandDim, expr] : llvm::enumerate(map.getResults())) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr)
        return failUnsigned("batch matmul requires projected permutation " +
                            role.str() + " indexing map");
      if (dimExpr.getPosition() == loopDim)
        return static_cast<unsigned>(operandDim);
    }
    return failUnsigned("batch matmul indexing map is missing " + role.str() +
                        " loop dimension");
  }

  bool mapContainsLoopDim(mlir::AffineMap map, unsigned loopDim) const {
    for (mlir::AffineExpr expr : map.getResults()) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (dimExpr && dimExpr.getPosition() == loopDim)
        return true;
    }
    return false;
  }

  mlir::FailureOr<BatchedGemmAttrs>
  inferBatchMatmulAttrs(mlir::linalg::LinalgOp op) {
    llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
        op.getIteratorTypesArray();
    llvm::SmallVector<unsigned, 1> reductionLoops;
    for (auto [index, iteratorType] : llvm::enumerate(iteratorTypes)) {
      if (iteratorType == mlir::utils::IteratorType::reduction)
        reductionLoops.push_back(static_cast<unsigned>(index));
    }
    if (reductionLoops.size() != 1)
      return mlir::failure();

    llvm::SmallVector<mlir::AffineMap, 4> maps = op.getIndexingMapsArray();
    if (maps.size() != 3)
      return mlir::failure();
    mlir::AffineMap lhsMap = maps[0];
    mlir::AffineMap rhsMap = maps[1];
    mlir::AffineMap resultMap = maps[2];

    std::optional<unsigned> mLoop;
    std::optional<unsigned> nLoop;
    llvm::SmallVector<unsigned, 4> batchLoops;
    unsigned reductionLoop = reductionLoops.front();
    for (unsigned loopDim = 0; loopDim < iteratorTypes.size(); ++loopDim) {
      if (loopDim == reductionLoop)
        continue;
      bool inLhs = mapContainsLoopDim(lhsMap, loopDim);
      bool inRhs = mapContainsLoopDim(rhsMap, loopDim);
      bool inResult = mapContainsLoopDim(resultMap, loopDim);
      if (inLhs && !inRhs && inResult) {
        if (mLoop)
          return mlir::failure();
        mLoop = loopDim;
        continue;
      }
      if (!inLhs && inRhs && inResult) {
        if (nLoop)
          return mlir::failure();
        nLoop = loopDim;
        continue;
      }
      if (inLhs && inRhs && inResult) {
        batchLoops.push_back(loopDim);
        continue;
      }
      return mlir::failure();
    }
    if (!mLoop || !nLoop || batchLoops.empty())
      return mlir::failure();

    BatchedGemmAttrs attrs;
    auto lhsMDim = findOperandDimForLoop(lhsMap, *mLoop, "lhs M");
    auto lhsKDim =
        findOperandDimForLoop(lhsMap, reductionLoop, "lhs contracting");
    auto rhsKDim =
        findOperandDimForLoop(rhsMap, reductionLoop, "rhs contracting");
    auto rhsNDim = findOperandDimForLoop(rhsMap, *nLoop, "rhs N");
    auto resultMDim = findOperandDimForLoop(resultMap, *mLoop, "result M");
    auto resultNDim = findOperandDimForLoop(resultMap, *nLoop, "result N");
    if (mlir::failed(lhsMDim) || mlir::failed(lhsKDim) ||
        mlir::failed(rhsKDim) || mlir::failed(rhsNDim) ||
        mlir::failed(resultMDim) || mlir::failed(resultNDim))
      return mlir::failure();
    attrs.lhsMDim = *lhsMDim;
    attrs.lhsContractingDim = *lhsKDim;
    attrs.rhsContractingDim = *rhsKDim;
    attrs.rhsNDim = *rhsNDim;
    attrs.resultMDim = *resultMDim;
    attrs.resultNDim = *resultNDim;

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultTensorType || !resultTensorType.hasStaticShape())
      return mlir::failure();

    attrs.batchCount = 1;
    for (unsigned batchLoop : batchLoops) {
      auto lhsBatchDim = findOperandDimForLoop(lhsMap, batchLoop, "lhs batch");
      auto rhsBatchDim = findOperandDimForLoop(rhsMap, batchLoop, "rhs batch");
      auto resultBatchDim =
          findOperandDimForLoop(resultMap, batchLoop, "result batch");
      if (mlir::failed(lhsBatchDim) || mlir::failed(rhsBatchDim) ||
          mlir::failed(resultBatchDim))
        return mlir::failure();
      attrs.lhsBatchDims.push_back(*lhsBatchDim);
      attrs.rhsBatchDims.push_back(*rhsBatchDim);
      attrs.resultBatchDims.push_back(*resultBatchDim);
      int64_t dimSize = resultTensorType.getDimSize(*resultBatchDim);
      if (mlir::ShapedType::isDynamic(dimSize) || dimSize <= 0)
        return mlir::failure();
      if (attrs.batchCount > std::numeric_limits<int64_t>::max() / dimSize)
        return mlir::failure();
      attrs.batchCount *= dimSize;
    }
    return attrs;
  }

  mlir::LogicalResult convertBatchMatmul(mlir::linalg::LinalgOp op,
                                         mlir::OpBuilder &builder) {
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
        op->getNumResults() != 1)
      return fail("unsupported batch matmul arity");
    if (mlir::failed(verifyExactGemmPayload(op, "batch matmul")))
      return mlir::failure();
    if (mlir::failed(requireZeroFilledGemmInit(op, "batch matmul")))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> lhs =
        getOrMaterialize(op.getDpsInputs()[0], MemLayout::Cx, builder);
    mlir::FailureOr<mlir::Value> rhs =
        getOrMaterialize(op.getDpsInputs()[1], MemLayout::Cx, builder);
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultTensorType)
      return fail("batch matmul result is not a ranked tensor");

    mlir::FailureOr<BatchedGemmAttrs> attrs = inferBatchMatmulAttrs(op);
    if (mlir::failed(attrs))
      return fail("unsupported batch matmul indexing");

    auto gemm = builder.create<ComputeGemmOp>(
        op->getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Cx), *lhs,
        *rhs);
    gemm->setAttr("batch_count", builder.getI64IntegerAttr(attrs->batchCount));
    gemm->setAttr("lhs_batch_dims",
                  builder.getDenseI64ArrayAttr(attrs->lhsBatchDims));
    gemm->setAttr("lhs_m_dim", builder.getI64IntegerAttr(attrs->lhsMDim));
    gemm->setAttr("lhs_contracting_dim",
                  builder.getI64IntegerAttr(attrs->lhsContractingDim));
    gemm->setAttr("rhs_batch_dims",
                  builder.getDenseI64ArrayAttr(attrs->rhsBatchDims));
    gemm->setAttr("rhs_contracting_dim",
                  builder.getI64IntegerAttr(attrs->rhsContractingDim));
    gemm->setAttr("rhs_n_dim", builder.getI64IntegerAttr(attrs->rhsNDim));
    gemm->setAttr("result_batch_dims",
                  builder.getDenseI64ArrayAttr(attrs->resultBatchDims));
    gemm->setAttr("result_m_dim", builder.getI64IntegerAttr(attrs->resultMDim));
    gemm->setAttr("result_n_dim", builder.getI64IntegerAttr(attrs->resultNDim));
    record(op->getResult(0), MemLayout::Cx, gemm.getResult());
    return mlir::success();
  }

  std::optional<ComputeElementwiseKind>
  inferCompareKind(mlir::arith::CmpFPredicate predicate) {
    switch (predicate) {
    case mlir::arith::CmpFPredicate::OEQ:
      return ComputeElementwiseKind::Eq;
    case mlir::arith::CmpFPredicate::UNE:
      return ComputeElementwiseKind::Ne;
    case mlir::arith::CmpFPredicate::OLT:
      return ComputeElementwiseKind::Lt;
    case mlir::arith::CmpFPredicate::OLE:
      return ComputeElementwiseKind::Le;
    case mlir::arith::CmpFPredicate::OGT:
      return ComputeElementwiseKind::Gt;
    case mlir::arith::CmpFPredicate::OGE:
      return ComputeElementwiseKind::Ge;
    case mlir::arith::CmpFPredicate::UEQ:
    case mlir::arith::CmpFPredicate::ONE:
    case mlir::arith::CmpFPredicate::ULT:
    case mlir::arith::CmpFPredicate::ULE:
    case mlir::arith::CmpFPredicate::UGT:
    case mlir::arith::CmpFPredicate::UGE:
    case mlir::arith::CmpFPredicate::AlwaysFalse:
    case mlir::arith::CmpFPredicate::ORD:
    case mlir::arith::CmpFPredicate::UNO:
    case mlir::arith::CmpFPredicate::AlwaysTrue:
      (void)fail("arith.cmpf predicate does not match the current target "
                 "comparison NaN semantics");
      return std::nullopt;
    }
    llvm_unreachable("unknown cmpf predicate");
  }

  std::optional<ComputeElementwiseKind>
  inferCompareKind(mlir::arith::CmpIPredicate predicate) {
    switch (predicate) {
    case mlir::arith::CmpIPredicate::eq:
      return ComputeElementwiseKind::Eq;
    case mlir::arith::CmpIPredicate::ne:
      return ComputeElementwiseKind::Ne;
    case mlir::arith::CmpIPredicate::slt:
      return ComputeElementwiseKind::Lt;
    case mlir::arith::CmpIPredicate::sle:
      return ComputeElementwiseKind::Le;
    case mlir::arith::CmpIPredicate::sgt:
      return ComputeElementwiseKind::Gt;
    case mlir::arith::CmpIPredicate::sge:
      return ComputeElementwiseKind::Ge;
    case mlir::arith::CmpIPredicate::ult:
    case mlir::arith::CmpIPredicate::ule:
    case mlir::arith::CmpIPredicate::ugt:
    case mlir::arith::CmpIPredicate::uge:
      (void)fail("unsigned arith.cmpi predicate cannot be represented by the "
                 "current signed target comparison kind");
      return std::nullopt;
    }
    llvm_unreachable("unknown cmpi predicate");
  }

  std::optional<ComputeReduceKind>
  inferReduceKind(mlir::linalg::GenericOp generic) {
    if (generic.getRegionInputArgs().size() != 1 ||
        generic.getRegionOutputArgs().size() != 1) {
      (void)fail("linalg.generic reduction requires one input and one "
                 "accumulator");
      return std::nullopt;
    }
    return matchExactReductionKind(generic.getRegionOutputArgs(), /*redPos=*/0,
                                   generic.getRegionInputArgs().front(),
                                   "linalg.generic reduction", failureReason);
  }

  bool hasReductionIterator(mlir::linalg::GenericOp generic) const {
    for (mlir::utils::IteratorType iteratorType :
         generic.getIteratorTypesArray()) {
      if (iteratorType == mlir::utils::IteratorType::reduction)
        return true;
    }
    return false;
  }

  mlir::LogicalResult
  getReductionInputDims(mlir::linalg::GenericOp generic,
                        llvm::SmallVectorImpl<int64_t> &inputDims) {
    inputDims.clear();
    if (generic.getNumDpsInputs() != 1)
      return fail("unsupported reduction input arity");

    llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
        generic.getIndexingMapsArray();
    if (indexingMaps.empty())
      return fail("reduction generic has no indexing map");
    mlir::AffineMap inputMap = indexingMaps[0];

    llvm::SmallVector<unsigned, 4> reductionLoopDims;
    for (auto [index, iteratorType] :
         llvm::enumerate(generic.getIteratorTypesArray())) {
      if (iteratorType == mlir::utils::IteratorType::reduction)
        reductionLoopDims.push_back(static_cast<unsigned>(index));
    }
    if (reductionLoopDims.empty())
      return fail("reduction generic has no reduction dimensions");

    for (unsigned loopDim : reductionLoopDims) {
      std::optional<int64_t> inputDim;
      for (auto [dimIndex, expr] : llvm::enumerate(inputMap.getResults())) {
        auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
        if (!dimExpr)
          return fail("unsupported reduction input indexing map");
        if (dimExpr.getPosition() == loopDim) {
          inputDim = static_cast<int64_t>(dimIndex);
          break;
        }
      }
      if (!inputDim)
        return fail("reduction dimension is not present in input map");
      inputDims.push_back(*inputDim);
    }
    return mlir::success();
  }

  mlir::LogicalResult createReduceOp(
      mlir::Location loc, mlir::Type resultType, ComputeReduceKindAttr kindAttr,
      mlir::Value input, llvm::ArrayRef<int64_t> dims, mlir::Value init,
      mlir::Attribute initAttr, mlir::OpBuilder &builder, mlir::Value &result) {
    mlir::OperationState state(loc, ComputeReduceOp::getOperationName());
    state.addAttribute("kind", kindAttr);
    state.addAttribute(
        "dimensions", mlir::DenseI64ArrayAttr::get(builder.getContext(), dims));
    if (initAttr)
      state.addAttribute("init_value", initAttr);
    state.addOperands(input);
    if (init)
      state.addOperands(init);
    state.addTypes(resultType);
    mlir::Operation *op = builder.create(state);
    result = op->getResult(0);
    return mlir::success();
  }

  mlir::LogicalResult convertReduceGeneric(mlir::linalg::GenericOp generic,
                                           mlir::OpBuilder &builder) {
    if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
        generic->getNumResults() != 1)
      return fail("unsupported reduction generic arity");

    std::optional<ComputeReduceKind> kind = inferReduceKind(generic);
    if (!kind)
      return mlir::failure();

    llvm::SmallVector<int64_t, 4> reduceDims;
    if (mlir::failed(getReductionInputDims(generic, reduceDims)))
      return mlir::failure();

    mlir::Value initTensor = generic.getDpsInits()[0];
    mlir::Value initScalar;
    mlir::Attribute initAttr;
    if (auto attrIt = fillInitAttrs.find(initTensor);
        attrIt != fillInitAttrs.end())
      initAttr = attrIt->second;
    if (auto scalarIt = fillInitScalars.find(initTensor);
        scalarIt != fillInitScalars.end())
      initScalar = scalarIt->second;
    if (initAttr)
      initScalar = {};
    if (!initAttr && !initScalar)
      return fail("missing reduction init scalar");

    auto inputTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
        generic.getDpsInputs()[0].getType());
    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
    if (!inputTensorType || !resultTensorType)
      return fail("reduction generic operands/results must be ranked tensors");

    mlir::FailureOr<mlir::Value> input =
        getOrMaterialize(generic.getDpsInputs()[0],
                         alignedLayoutForTensor(inputTensorType), builder);
    if (mlir::failed(input))
      return mlir::failure();

    mlir::Type reduceResultType = makeSPMMemRefType(
        resultTensorType, alignedLayoutForTensor(resultTensorType));
    mlir::Value reduceResult;
    auto kindAttr = ComputeReduceKindAttr::get(generic.getContext(), *kind);
    if (mlir::failed(createReduceOp(generic.getLoc(), reduceResultType,
                                    kindAttr, *input, reduceDims, initScalar,
                                    initAttr, builder, reduceResult)))
      return mlir::failure();

    record(generic->getResult(0), alignedLayoutForTensor(resultTensorType),
           reduceResult);
    return mlir::success();
  }

  std::optional<unsigned>
  getPassthroughInputIndex(mlir::linalg::GenericOp generic) {
    if (!generic.getBody()->without_terminator().empty())
      return std::nullopt;
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(
        generic.getBody()->getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return std::nullopt;

    auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(yield.getValues()[0]);
    if (!blockArg)
      return std::nullopt;
    unsigned argNumber = blockArg.getArgNumber();
    if (argNumber >= generic.getNumDpsInputs())
      return std::nullopt;
    return argNumber;
  }

  bool isIdentityMap(mlir::AffineMap map, int64_t rank) const {
    return map.getNumDims() == static_cast<unsigned>(rank) &&
           map.getNumSymbols() == 0 && map.isIdentity();
  }

  mlir::AffineMap getIdentityMap(mlir::RankedTensorType tensorType) const {
    return mlir::AffineMap::getMultiDimIdentityMap(tensorType.getRank(),
                                                   tensorType.getContext());
  }

  bool isScalarSplatValue(mlir::Attribute attr, double expected) const {
    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr))
      return floatAttr.getValueAsDouble() == expected;
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return intAttr.getValue().getSExtValue() ==
             static_cast<int64_t>(expected);
    if (auto elements = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
      if (!elements.isSplat())
        return false;
      if (auto floatType =
              mlir::dyn_cast<mlir::FloatType>(elements.getElementType()))
        return elements.getSplatValue<mlir::APFloat>().convertToDouble() ==
               expected;
      if (auto intType =
              mlir::dyn_cast<mlir::IntegerType>(elements.getElementType()))
        return elements.getSplatValue<mlir::APInt>().getSExtValue() ==
               static_cast<int64_t>(expected);
    }
    return false;
  }

  mlir::Attribute getBlockArgumentConstantAttr(mlir::linalg::GenericOp generic,
                                               mlir::BlockArgument arg) const {
    if (arg.getOwner() != generic.getBody() ||
        arg.getArgNumber() >= generic.getNumDpsInputs())
      return {};
    mlir::Value input = generic.getDpsInputs()[arg.getArgNumber()];
    if (auto constant = input.getDefiningOp<mlir::arith::ConstantOp>())
      return constant.getValue();
    if (auto it = tensorAttrs.find(input); it != tensorAttrs.end())
      return it->second;
    return {};
  }

  bool isScalarLikeConstant(mlir::linalg::GenericOp generic, mlir::Value value,
                            double expected) const {
    if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
      return isScalarSplatValue(constant.getValue(), expected);
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (mlir::Attribute attr = getBlockArgumentConstantAttr(generic, arg))
        return isScalarSplatValue(attr, expected);
    }
    return false;
  }

  mlir::FailureOr<ElementwiseExprValue> getElementwiseExprValue(
      llvm::DenseMap<mlir::Value, ElementwiseExprValue> &values,
      mlir::Value value) {
    auto it = values.find(value);
    if (it == values.end())
      return failElementwiseExprValue(
          "unsupported linalg.generic scalar expression");
    return it->second;
  }

  mlir::FailureOr<mlir::Value> materializeElementwiseExprOperand(
      ElementwiseExprValue exprValue, mlir::RankedTensorType resultTensorType,
      mlir::Location loc, mlir::OpBuilder &builder) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(exprValue.buffer.getType());
    if (!sourceType)
      return failValue("elementwise expression operand is not an SPM memref");
    auto operandTensorType = mlir::RankedTensorType::get(
        resultTensorType.getShape(), sourceType.getElementType());
    auto operandType = makeSPMMemRefType(operandTensorType, MemLayout::Tensor);
    if (sourceType == operandType &&
        isIdentityMap(exprValue.indexingMap, resultTensorType.getRank()))
      return exprValue.buffer;

    auto sourceTensorType = mlir::RankedTensorType::get(
        sourceType.getShape(), sourceType.getElementType());
    if (!exprValue.indexingMap.isProjectedPermutation())
      return failValue("elementwise expression indexing map must be a "
                       "projected permutation");

    if (sourceTensorType.getRank() == resultTensorType.getRank()) {
      llvm::SmallVector<int64_t, 4> permutation(resultTensorType.getRank(), -1);
      for (auto [sourceDim, expr] :
           llvm::enumerate(exprValue.indexingMap.getResults())) {
        auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
        if (!dimExpr)
          return failValue("unsupported elementwise transpose map");
        unsigned resultDim = dimExpr.getPosition();
        if (resultDim >= permutation.size())
          return failValue("elementwise transpose map dim out of range");
        permutation[resultDim] = static_cast<int64_t>(sourceDim);
      }
      if (llvm::any_of(permutation, [](int64_t dim) { return dim < 0; }))
        return failValue("elementwise transpose map is incomplete");
      if (sourceType == operandType &&
          llvm::all_of(llvm::enumerate(permutation), [](auto indexed) {
            return static_cast<int64_t>(indexed.index()) == indexed.value();
          }))
        return exprValue.buffer;
      return builder
          .create<MoveTransposeOp>(
              loc, operandType, exprValue.buffer,
              mlir::DenseI64ArrayAttr::get(builder.getContext(), permutation))
          .getResult();
    }

    if (sourceTensorType.getRank() < resultTensorType.getRank()) {
      llvm::SmallVector<int64_t, 4> dimensions;
      for (mlir::AffineExpr expr : exprValue.indexingMap.getResults()) {
        auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
        if (!dimExpr)
          return failValue("unsupported elementwise broadcast map");
        dimensions.push_back(dimExpr.getPosition());
      }
      return builder
          .create<MoveBroadcastOp>(
              loc, operandType, exprValue.buffer,
              mlir::DenseI64ArrayAttr::get(builder.getContext(), dimensions))
          .getResult();
    }

    return failValue("unsupported elementwise expression rank relation");
  }

  mlir::FailureOr<ElementwiseExprValue>
  createElementwiseFillExprValue(mlir::Location loc, mlir::Value scalar,
                                 mlir::RankedTensorType resultTensorType,
                                 mlir::OpBuilder &builder) {
    if (!isScalarType(scalar.getType()))
      return failElementwiseExprValue(
          "elementwise scalar splat source must be scalar typed");
    auto splatTensorType = mlir::RankedTensorType::get(
        resultTensorType.getShape(), scalar.getType());
    auto alloc = builder.create<mlir::memref::AllocOp>(
        loc, makeSPMMemRefType(splatTensorType, MemLayout::Tensor));
    builder.create<ComputeFillOp>(loc, alloc.getResult(), scalar);
    return ElementwiseExprValue{alloc.getResult(),
                                getIdentityMap(resultTensorType)};
  }

  mlir::FailureOr<ElementwiseExprValue>
  createElementwiseOpExprValue(mlir::Location loc, ComputeElementwiseKind kind,
                               llvm::ArrayRef<ElementwiseExprValue> operands,
                               mlir::RankedTensorType resultTensorType,
                               mlir::OpBuilder &builder) {
    llvm::SmallVector<mlir::Value, 2> materializedInputs;
    for (ElementwiseExprValue operand : operands) {
      mlir::FailureOr<mlir::Value> materialized =
          materializeElementwiseExprOperand(operand, resultTensorType, loc,
                                            builder);
      if (mlir::failed(materialized))
        return mlir::failure();
      materializedInputs.push_back(*materialized);
    }

    auto kindAttr = ComputeElementwiseKindAttr::get(builder.getContext(), kind);
    auto elementwise = builder.create<ComputeElementwiseOp>(
        loc, makeSPMMemRefType(resultTensorType, MemLayout::Tensor), kindAttr,
        materializedInputs);
    return ElementwiseExprValue{elementwise.getResult(),
                                getIdentityMap(resultTensorType)};
  }

  mlir::LogicalResult convertElementwiseScalarOp(
      mlir::linalg::GenericOp generic, mlir::Operation *op,
      llvm::DenseMap<mlir::Value, ElementwiseExprValue> &values,
      mlir::RankedTensorType resultTensorType, mlir::OpBuilder &builder) {
    auto lookup = [&](mlir::Value value) {
      auto it = values.find(value);
      if (it != values.end())
        return mlir::FailureOr<ElementwiseExprValue>(it->second);

      if (isScalarType(value.getType())) {
        mlir::FailureOr<mlir::Value> scalar = getScalarValue(value);
        if (mlir::failed(scalar))
          return mlir::FailureOr<ElementwiseExprValue>(mlir::failure());
        mlir::FailureOr<ElementwiseExprValue> splat =
            createElementwiseFillExprValue(value.getLoc(), *scalar,
                                           resultTensorType, builder);
        if (mlir::failed(splat))
          return splat;
        values[value] = *splat;
        return splat;
      }

      return failElementwiseExprValue(
          "unsupported linalg.generic scalar expression");
    };
    auto createUnary = [&](mlir::Value input,
                           ComputeElementwiseKind kind) -> mlir::LogicalResult {
      mlir::FailureOr<ElementwiseExprValue> operand = lookup(input);
      if (mlir::failed(operand))
        return mlir::failure();
      mlir::FailureOr<ElementwiseExprValue> result =
          createElementwiseOpExprValue(op->getLoc(), kind, {*operand},
                                       resultTensorType, builder);
      if (mlir::failed(result))
        return mlir::failure();
      values[op->getResult(0)] = *result;
      return mlir::success();
    };
    auto createBinary =
        [&](mlir::Value lhs, mlir::Value rhs,
            ComputeElementwiseKind kind) -> mlir::LogicalResult {
      mlir::FailureOr<ElementwiseExprValue> lhsValue = lookup(lhs);
      mlir::FailureOr<ElementwiseExprValue> rhsValue = lookup(rhs);
      if (mlir::failed(lhsValue) || mlir::failed(rhsValue))
        return mlir::failure();
      mlir::FailureOr<ElementwiseExprValue> result =
          createElementwiseOpExprValue(op->getLoc(), kind,
                                       {*lhsValue, *rhsValue}, resultTensorType,
                                       builder);
      if (mlir::failed(result))
        return mlir::failure();
      values[op->getResult(0)] = *result;
      return mlir::success();
    };
    auto createTernary =
        [&](mlir::Value first, mlir::Value second, mlir::Value third,
            ComputeElementwiseKind kind) -> mlir::LogicalResult {
      mlir::FailureOr<ElementwiseExprValue> firstValue = lookup(first);
      mlir::FailureOr<ElementwiseExprValue> secondValue = lookup(second);
      mlir::FailureOr<ElementwiseExprValue> thirdValue = lookup(third);
      if (mlir::failed(firstValue) || mlir::failed(secondValue) ||
          mlir::failed(thirdValue))
        return mlir::failure();
      mlir::FailureOr<ElementwiseExprValue> result =
          createElementwiseOpExprValue(op->getLoc(), kind,
                                       {*firstValue, *secondValue, *thirdValue},
                                       resultTensorType, builder);
      if (mlir::failed(result))
        return mlir::failure();
      values[op->getResult(0)] = *result;
      return mlir::success();
    };

    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      if (constant->getNumResults() != 1 || !isScalarType(constant.getType()))
        return fail("unsupported linalg.generic constant expression");
      mlir::Operation *cloned = builder.clone(*constant.getOperation());
      mlir::FailureOr<ElementwiseExprValue> value =
          createElementwiseFillExprValue(constant.getLoc(),
                                         cloned->getResult(0), resultTensorType,
                                         builder);
      if (mlir::failed(value))
        return mlir::failure();
      values[constant.getResult()] = *value;
      return mlir::success();
    }

    if (auto fromElements = mlir::dyn_cast<mlir::tensor::FromElementsOp>(op)) {
      if (fromElements.getElements().size() != 1 ||
          fromElements->getNumResults() != 1)
        return fail("unsupported tensor.from_elements in elementwise body");
      mlir::FailureOr<ElementwiseExprValue> value =
          lookup(fromElements.getElements().front());
      if (mlir::failed(value))
        return mlir::failure();
      values[fromElements.getResult()] = *value;
      return mlir::success();
    }

    if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractOp>(op)) {
      if (!extract.getIndices().empty() || extract->getNumResults() != 1)
        return fail("unsupported tensor.extract in elementwise body");
      mlir::FailureOr<ElementwiseExprValue> value = lookup(extract.getTensor());
      if (mlir::failed(value))
        return mlir::failure();
      values[extract.getResult()] = *value;
      return mlir::success();
    }

    if (auto addf = mlir::dyn_cast<mlir::arith::AddFOp>(op))
      return createBinary(addf.getLhs(), addf.getRhs(),
                          ComputeElementwiseKind::Add);
    if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(op))
      return createBinary(addi.getLhs(), addi.getRhs(),
                          ComputeElementwiseKind::Add);
    if (auto subf = mlir::dyn_cast<mlir::arith::SubFOp>(op))
      return createBinary(subf.getLhs(), subf.getRhs(),
                          ComputeElementwiseKind::Sub);
    if (auto subi = mlir::dyn_cast<mlir::arith::SubIOp>(op))
      return createBinary(subi.getLhs(), subi.getRhs(),
                          ComputeElementwiseKind::Sub);
    if (auto mulf = mlir::dyn_cast<mlir::arith::MulFOp>(op))
      return createBinary(mulf.getLhs(), mulf.getRhs(),
                          ComputeElementwiseKind::Mul);
    if (auto muli = mlir::dyn_cast<mlir::arith::MulIOp>(op))
      return createBinary(muli.getLhs(), muli.getRhs(),
                          ComputeElementwiseKind::Mul);
    if (auto divf = mlir::dyn_cast<mlir::arith::DivFOp>(op)) {
      if (isScalarLikeConstant(generic, divf.getLhs(), 1.0))
        return createUnary(divf.getRhs(), ComputeElementwiseKind::Recip);
      return createBinary(divf.getLhs(), divf.getRhs(),
                          ComputeElementwiseKind::Div);
    }
    if (auto divsi = mlir::dyn_cast<mlir::arith::DivSIOp>(op))
      return createBinary(divsi.getLhs(), divsi.getRhs(),
                          ComputeElementwiseKind::Div);
    if (mlir::isa<mlir::arith::DivUIOp>(op))
      return fail("unsigned integer division cannot be represented by the "
                  "current target elementwise kind");
    if (auto maxf = mlir::dyn_cast<mlir::arith::MaximumFOp>(op))
      return createBinary(maxf.getLhs(), maxf.getRhs(),
                          ComputeElementwiseKind::Max);
    if (mlir::isa<mlir::arith::MaxNumFOp>(op))
      return fail("arith.maxnumf NaN semantics cannot be represented by the "
                  "current target elementwise kind");
    if (auto maxsi = mlir::dyn_cast<mlir::arith::MaxSIOp>(op))
      return createBinary(maxsi.getLhs(), maxsi.getRhs(),
                          ComputeElementwiseKind::Max);
    if (mlir::isa<mlir::arith::MaxUIOp>(op))
      return fail("unsigned integer maximum cannot be represented by the "
                  "current target elementwise kind");
    if (auto minf = mlir::dyn_cast<mlir::arith::MinimumFOp>(op))
      return createBinary(minf.getLhs(), minf.getRhs(),
                          ComputeElementwiseKind::Min);
    if (mlir::isa<mlir::arith::MinNumFOp>(op))
      return fail("arith.minnumf NaN semantics cannot be represented by the "
                  "current target elementwise kind");
    if (auto minsi = mlir::dyn_cast<mlir::arith::MinSIOp>(op))
      return createBinary(minsi.getLhs(), minsi.getRhs(),
                          ComputeElementwiseKind::Min);
    if (mlir::isa<mlir::arith::MinUIOp>(op))
      return fail("unsigned integer minimum cannot be represented by the "
                  "current target elementwise kind");
    if (auto negf = mlir::dyn_cast<mlir::arith::NegFOp>(op))
      return createUnary(negf.getOperand(), ComputeElementwiseKind::Neg);
    if (auto exp = mlir::dyn_cast<mlir::math::ExpOp>(op))
      return createUnary(exp.getOperand(), ComputeElementwiseKind::Exp);
    if (auto sqrt = mlir::dyn_cast<mlir::math::SqrtOp>(op))
      return createUnary(sqrt.getOperand(), ComputeElementwiseKind::Sqrt);
    if (auto rsqrt = mlir::dyn_cast<mlir::math::RsqrtOp>(op))
      return createUnary(rsqrt.getOperand(), ComputeElementwiseKind::Rsqrt);
    if (auto tanh = mlir::dyn_cast<mlir::math::TanhOp>(op))
      return createUnary(tanh.getOperand(), ComputeElementwiseKind::Tanh);
    if (auto cmpf = mlir::dyn_cast<mlir::arith::CmpFOp>(op)) {
      std::optional<ComputeElementwiseKind> kind =
          inferCompareKind(cmpf.getPredicate());
      if (!kind)
        return mlir::failure();
      return createBinary(cmpf.getLhs(), cmpf.getRhs(), *kind);
    }
    if (auto cmpi = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
      std::optional<ComputeElementwiseKind> kind =
          inferCompareKind(cmpi.getPredicate());
      if (!kind)
        return mlir::failure();
      return createBinary(cmpi.getLhs(), cmpi.getRhs(), *kind);
    }
    if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(op))
      return createTernary(select.getCondition(), select.getTrueValue(),
                           select.getFalseValue(),
                           ComputeElementwiseKind::Select);
    if (auto powf = mlir::dyn_cast<mlir::math::PowFOp>(op)) {
      if (!isScalarLikeConstant(generic, powf.getRhs(), 2.0))
        return fail("only powf with exponent 2 is supported in elementwise "
                    "body");
      return createBinary(powf.getLhs(), powf.getLhs(),
                          ComputeElementwiseKind::Mul);
    }

    return fail("unsupported linalg.generic body op " +
                op->getName().getStringRef().str());
  }

  mlir::LogicalResult
  convertElementwiseGenericExpression(mlir::linalg::GenericOp generic,
                                      mlir::OpBuilder &builder) {
    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
    if (!resultTensorType)
      return fail("generic result is not a ranked tensor");

    llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
        generic.getIndexingMapsArray();
    if (indexingMaps.size() !=
        generic.getNumDpsInputs() + generic.getNumDpsInits())
      return fail("elementwise generic indexing map count mismatch");
    mlir::AffineMap resultMap = indexingMaps.back();
    if (!isIdentityMap(resultMap, resultTensorType.getRank()))
      return fail("elementwise generic result map must be identity");

    llvm::DenseMap<mlir::Value, ElementwiseExprValue> values;
    mlir::Block &body = *generic.getBody();
    unsigned bodyArgIndex = 0;
    for (mlir::Value input : generic.getDpsInputs()) {
      mlir::FailureOr<mlir::Value> buffer =
          getOrMaterialize(input, MemLayout::Tensor, builder);
      if (mlir::failed(buffer))
        return mlir::failure();
      values[body.getArgument(bodyArgIndex)] =
          ElementwiseExprValue{*buffer, indexingMaps[bodyArgIndex]};
      ++bodyArgIndex;
    }
    for (mlir::Value init : generic.getDpsInits()) {
      mlir::BlockArgument bodyArg = body.getArgument(bodyArgIndex);
      if (bodyArg.use_empty()) {
        ++bodyArgIndex;
        continue;
      }
      mlir::FailureOr<mlir::Value> buffer =
          getOrMaterialize(init, MemLayout::Tensor, builder);
      if (mlir::failed(buffer))
        return mlir::failure();
      values[bodyArg] =
          ElementwiseExprValue{*buffer, indexingMaps[bodyArgIndex]};
      ++bodyArgIndex;
    }

    for (mlir::Operation &op : body.without_terminator()) {
      if (mlir::failed(convertElementwiseScalarOp(generic, &op, values,
                                                  resultTensorType, builder)))
        return mlir::failure();
    }

    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return fail("unsupported linalg.generic yield");
    mlir::FailureOr<ElementwiseExprValue> yielded =
        getElementwiseExprValue(values, yield.getValues().front());
    if (mlir::failed(yielded))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> result = materializeElementwiseExprOperand(
        *yielded, resultTensorType, generic.getLoc(), builder);
    if (mlir::failed(result))
      return mlir::failure();
    record(generic->getResult(0), MemLayout::Tensor, *result);
    return mlir::success();
  }

  mlir::LogicalResult convertPassthroughGeneric(mlir::linalg::GenericOp generic,
                                                mlir::OpBuilder &builder) {
    if (generic.getNumDpsInits() != 1 || generic->getNumResults() != 1)
      return fail("unsupported passthrough generic arity");
    std::optional<unsigned> inputIndex = getPassthroughInputIndex(generic);
    if (!inputIndex)
      return fail("unsupported linalg.generic passthrough body");

    llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
        generic.getIndexingMapsArray();
    if (indexingMaps.size() != generic.getNumDpsInputs() + 1)
      return fail("passthrough generic indexing map count mismatch");

    mlir::Value inputValue = generic.getDpsInputs()[*inputIndex];
    auto inputTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(inputValue.getType());
    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
    if (!inputTensorType || !resultTensorType)
      return fail(
          "passthrough generic operands/results must be ranked tensors");

    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(inputValue, MemLayout::Tensor, builder);
    if (mlir::failed(source))
      return mlir::failure();

    mlir::AffineMap inputMap = indexingMaps[*inputIndex];
    mlir::AffineMap resultMap = indexingMaps.back();
    if (!isIdentityMap(resultMap, resultTensorType.getRank()))
      return fail("passthrough generic result map must be identity");

    mlir::MemRefType resultType =
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
    mlir::Value result;
    if (inputTensorType == resultTensorType &&
        isIdentityMap(inputMap, resultTensorType.getRank())) {
      result = builder.create<MoveCopyOp>(generic.getLoc(), resultType, *source)
                   .getResult();
    } else if (inputTensorType.getRank() == resultTensorType.getRank()) {
      llvm::SmallVector<int64_t, 4> permutation(resultTensorType.getRank(), -1);
      for (auto [sourceDim, expr] : llvm::enumerate(inputMap.getResults())) {
        auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
        if (!dimExpr)
          return fail("unsupported passthrough transpose map");
        unsigned resultDim = dimExpr.getPosition();
        if (resultDim >= permutation.size())
          return fail("passthrough transpose map dim out of range");
        permutation[resultDim] = static_cast<int64_t>(sourceDim);
      }
      if (llvm::any_of(permutation, [](int64_t dim) { return dim < 0; }))
        return fail("passthrough transpose map is incomplete");
      result =
          builder
              .create<MoveTransposeOp>(generic.getLoc(), resultType, *source,
                                       mlir::DenseI64ArrayAttr::get(
                                           generic.getContext(), permutation))
              .getResult();
    } else if (inputTensorType.getRank() < resultTensorType.getRank()) {
      llvm::SmallVector<int64_t, 4> dimensions;
      for (mlir::AffineExpr expr : inputMap.getResults()) {
        auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
        if (!dimExpr)
          return fail("unsupported passthrough broadcast map");
        dimensions.push_back(dimExpr.getPosition());
      }
      result =
          builder
              .create<MoveBroadcastOp>(generic.getLoc(), resultType, *source,
                                       mlir::DenseI64ArrayAttr::get(
                                           generic.getContext(), dimensions))
              .getResult();
    } else {
      return fail("unsupported passthrough movement rank relation");
    }

    record(generic->getResult(0), MemLayout::Tensor, result);
    return mlir::success();
  }

  mlir::Value createZeroScalar(mlir::Location loc, mlir::Type type,
                               mlir::OpBuilder &builder) {
    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
      auto zero = mlir::FloatAttr::get(
          floatType, llvm::APFloat::getZero(floatType.getFloatSemantics()));
      return builder.create<mlir::arith::ConstantOp>(loc, zero).getResult();
    }
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
      auto zero = mlir::IntegerAttr::get(intType, 0);
      return builder.create<mlir::arith::ConstantOp>(loc, zero).getResult();
    }
    if (mlir::isa<mlir::IndexType>(type))
      return builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
    return {};
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
  matchTwoWayConcatGeneric(mlir::linalg::GenericOp generic,
                           int64_t &concatAxis) {
    concatAxis = -1;
    if (generic.getNumDpsInputs() != 0 || generic.getNumDpsInits() != 1 ||
        generic->getNumResults() != 1)
      return mlir::failure();

    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
    if (!resultTensorType || !resultTensorType.hasStaticShape())
      return mlir::failure();
    int64_t rank = resultTensorType.getRank();

    llvm::SmallVector<mlir::AffineMap, 1> maps = generic.getIndexingMapsArray();
    if (maps.size() != 1 || !isIdentityMap(maps.front(), rank))
      return mlir::failure();
    llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
        generic.getIteratorTypesArray();
    if (iteratorTypes.size() != static_cast<size_t>(rank) ||
        !llvm::all_of(iteratorTypes, [](mlir::utils::IteratorType type) {
          return type == mlir::utils::IteratorType::parallel;
        }))
      return mlir::failure();

    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(
        generic.getBody()->getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return mlir::failure();
    auto ifOp = yield.getValues().front().getDefiningOp<mlir::scf::IfOp>();
    if (!ifOp || ifOp->getNumResults() != 1 || !ifOp.thenBlock() ||
        !ifOp.elseBlock() || ifOp->getBlock() != generic.getBody())
      return mlir::failure();

    auto thenYield =
        mlir::dyn_cast<mlir::scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield =
        mlir::dyn_cast<mlir::scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    if (!thenYield || !elseYield || thenYield.getResults().size() != 1 ||
        elseYield.getResults().size() != 1)
      return mlir::failure();
    auto firstExtract =
        thenYield.getResults().front().getDefiningOp<mlir::tensor::ExtractOp>();
    auto secondExtract =
        elseYield.getResults().front().getDefiningOp<mlir::tensor::ExtractOp>();
    if (!firstExtract || !secondExtract ||
        firstExtract->getBlock() != ifOp.thenBlock() ||
        secondExtract->getBlock() != ifOp.elseBlock())
      return mlir::failure();

    auto firstType = mlir::dyn_cast<mlir::RankedTensorType>(
        firstExtract.getTensor().getType());
    auto secondType = mlir::dyn_cast<mlir::RankedTensorType>(
        secondExtract.getTensor().getType());
    if (!firstType || !secondType || !firstType.hasStaticShape() ||
        !secondType.hasStaticShape() || firstType.getRank() != rank ||
        secondType.getRank() != rank ||
        firstType.getElementType() != resultTensorType.getElementType() ||
        secondType.getElementType() != resultTensorType.getElementType())
      return mlir::failure();

    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t first = firstType.getDimSize(dim);
      int64_t second = secondType.getDimSize(dim);
      int64_t result = resultTensorType.getDimSize(dim);
      if (first == result && second == result)
        continue;
      if (first + second == result && concatAxis < 0) {
        concatAxis = dim;
        continue;
      }
      return mlir::failure();
    }
    if (concatAxis < 0 || firstType.getDimSize(concatAxis) <= 0 ||
        secondType.getDimSize(concatAxis) <= 0)
      return mlir::failure();

    auto cmp = ifOp.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != mlir::arith::CmpIPredicate::ult ||
        cmp->getBlock() != generic.getBody())
      return mlir::failure();
    auto boundaryConstant =
        cmp.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
    auto boundaryAttr =
        boundaryConstant
            ? mlir::dyn_cast<mlir::IntegerAttr>(boundaryConstant.getValue())
            : mlir::IntegerAttr{};
    if (!boundaryAttr || !mlir::isa<mlir::IndexType>(cmp.getRhs().getType()) ||
        boundaryAttr.getInt() != firstType.getDimSize(concatAxis))
      return mlir::failure();

    llvm::DenseSet<mlir::Operation *> bodySkeleton;
    bodySkeleton.insert(cmp.getOperation());
    bodySkeleton.insert(ifOp.getOperation());
    auto matchesLinalgIndex = [&](mlir::Value value, int64_t dim) {
      auto index = value.getDefiningOp<mlir::linalg::IndexOp>();
      if (!index || index->getParentOp() != generic.getOperation() ||
          index.getDim() != static_cast<uint64_t>(dim))
        return false;
      bodySkeleton.insert(index.getOperation());
      return true;
    };
    if (!matchesLinalgIndex(cmp.getLhs(), concatAxis) ||
        firstExtract.getIndices().size() != static_cast<size_t>(rank) ||
        secondExtract.getIndices().size() != static_cast<size_t>(rank))
      return mlir::failure();

    mlir::arith::SubIOp axisSubtract;
    for (int64_t dim = 0; dim < rank; ++dim) {
      if (!matchesLinalgIndex(firstExtract.getIndices()[dim], dim))
        return mlir::failure();
      if (dim != concatAxis) {
        if (!matchesLinalgIndex(secondExtract.getIndices()[dim], dim))
          return mlir::failure();
        continue;
      }

      auto subtract =
          secondExtract.getIndices()[dim].getDefiningOp<mlir::arith::SubIOp>();
      if (!subtract || subtract->getBlock() != ifOp.elseBlock() ||
          !matchesLinalgIndex(subtract.getLhs(), dim) ||
          subtract.getRhs() != cmp.getRhs())
        return mlir::failure();
      axisSubtract = subtract;
    }

    for (mlir::Operation &op : generic.getBody()->without_terminator()) {
      if (!bodySkeleton.contains(&op))
        return mlir::failure();
    }
    for (mlir::Operation &op : ifOp.thenBlock()->without_terminator()) {
      if (&op != firstExtract.getOperation())
        return mlir::failure();
    }
    for (mlir::Operation &op : ifOp.elseBlock()->without_terminator()) {
      if (&op != axisSubtract.getOperation() &&
          &op != secondExtract.getOperation())
        return mlir::failure();
    }

    return llvm::SmallVector<mlir::Value, 2>{firstExtract.getTensor(),
                                             secondExtract.getTensor()};
  }

  mlir::LogicalResult
  convertTwoWayConcatGeneric(mlir::linalg::GenericOp generic,
                             llvm::ArrayRef<mlir::Value> concatInputs,
                             int64_t concatAxis, mlir::OpBuilder &builder) {
    auto resultTensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
    if (!resultTensorType)
      return fail("concat generic result is not a ranked tensor");
    if (concatInputs.size() != 2)
      return fail("concat generic requires two inputs");

    mlir::FailureOr<mlir::Value> first =
        getOrMaterialize(concatInputs[0], MemLayout::Tensor, builder);
    mlir::FailureOr<mlir::Value> second =
        getOrMaterialize(concatInputs[1], MemLayout::Tensor, builder);
    if (mlir::failed(first) || mlir::failed(second))
      return mlir::failure();

    mlir::MemRefType resultType =
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
    auto seed =
        builder.create<mlir::memref::AllocOp>(generic.getLoc(), resultType);
    mlir::Value zero = createZeroScalar(
        generic.getLoc(), resultTensorType.getElementType(), builder);
    if (!zero)
      return fail("concat generic requires numeric element type");
    builder.create<ComputeFillOp>(generic.getLoc(), seed.getResult(), zero);

    auto firstType = mlir::cast<mlir::MemRefType>((*first).getType());
    auto secondType = mlir::cast<mlir::MemRefType>((*second).getType());
    llvm::SmallVector<int64_t, 4> offsets(resultTensorType.getRank(), 0);
    llvm::SmallVector<int64_t, 4> strides(resultTensorType.getRank(), 1);
    llvm::SmallVector<int64_t, 4> firstSizes(firstType.getShape().begin(),
                                             firstType.getShape().end());
    llvm::SmallVector<int64_t, 4> secondSizes(secondType.getShape().begin(),
                                              secondType.getShape().end());

    mlir::MLIRContext *context = generic.getContext();
    auto firstInsert = builder.create<MoveInsertSliceOp>(
        generic.getLoc(), resultType, *first, seed.getResult(),
        mlir::DenseI64ArrayAttr::get(context, offsets),
        mlir::DenseI64ArrayAttr::get(context, firstSizes),
        mlir::DenseI64ArrayAttr::get(context, strides));

    offsets[concatAxis] = firstType.getDimSize(concatAxis);
    auto secondInsert = builder.create<MoveInsertSliceOp>(
        generic.getLoc(), resultType, *second, firstInsert.getResult(),
        mlir::DenseI64ArrayAttr::get(context, offsets),
        mlir::DenseI64ArrayAttr::get(context, secondSizes),
        mlir::DenseI64ArrayAttr::get(context, strides));

    record(generic->getResult(0), MemLayout::Tensor, secondInsert.getResult());
    return mlir::success();
  }

  mlir::LogicalResult convertGeneric(mlir::linalg::GenericOp generic,
                                     mlir::OpBuilder &builder) {
    if (generic.getNumDpsInits() != 1 || generic->getNumResults() != 1)
      return fail("unsupported linalg.generic arity");

    if (hasReductionIterator(generic))
      return convertReduceGeneric(generic, builder);
    if (getPassthroughInputIndex(generic))
      return convertPassthroughGeneric(generic, builder);
    int64_t concatAxis = -1;
    if (mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> concatInputs =
            matchTwoWayConcatGeneric(generic, concatAxis);
        mlir::succeeded(concatInputs))
      return convertTwoWayConcatGeneric(generic, *concatInputs, concatAxis,
                                        builder);
    return convertElementwiseGenericExpression(generic, builder);
  }

  mlir::LogicalResult finishRegion(GroupOp group, TileRegionOp tileRegion,
                                   mlir::OpBuilder &builder) {
    auto yield =
        mlir::dyn_cast<GroupYieldOp>(group.getBody().front().getTerminator());
    if (!yield)
      return fail("group terminator is not wafer.group.yield");

    unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
    llvm::SmallVector<mlir::Value, 2> yieldedValues;
    mlir::Block &tileBlock = tileRegion.getBody().front();
    for (auto [index, value] : llvm::enumerate(yield.getValues())) {
      if (inputCount + index >= tileBlock.getNumArguments())
        return fail("group result has no output boundary");
      mlir::Value output = tileBlock.getArgument(inputCount + index);

      if (auto directIt = directYieldBuffers.find(value);
          directIt != directYieldBuffers.end()) {
        if (directIt->second != output)
          return fail("direct boundary storeback target mismatch");
        yieldedValues.push_back(output);
        continue;
      }

      mlir::FailureOr<mlir::Value> tensorBuffer =
          getOrMaterialize(value, MemLayout::Tensor, builder);
      if (mlir::failed(tensorBuffer))
        return mlir::failure();

      builder.create<StorageStoreOp>(value.getLoc(), *tensorBuffer, output);
      yieldedValues.push_back(output);
    }

    builder.create<TileYieldOp>(group.getLoc(), yieldedValues);
    return mlir::success();
  }
};

struct GroupToTileRegionLoweringPattern
    : public mlir::OpConversionPattern<GroupOp> {
  GroupToTileRegionLoweringPattern(mlir::MLIRContext *context,
                                   std::string *failureReason,
                                   int64_t currentLogicalRank)
      : mlir::OpConversionPattern<GroupOp>(context),
        failureReason(failureReason), currentLogicalRank(currentLogicalRank) {}

  mlir::LogicalResult
  matchAndRewrite(GroupOp group, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    TileRegionBodyEmitter emitter(failureReason, currentLogicalRank);
    mlir::FailureOr<TileRegionOp> tileRegion =
        emitter.emit(group, adaptor.getInputs(), adaptor.getOuts(), rewriter);
    if (mlir::failed(tileRegion))
      return mlir::failure();

    rewriter.setInsertionPointAfter((*tileRegion).getOperation());
    llvm::SmallVector<mlir::Value, 2> replacements;
    for (mlir::Value result : (*tileRegion).getResults()) {
      auto tensor = rewriter.create<mlir::bufferization::ToTensorOp>(
          group.getLoc(), result, /*restrict=*/true, /*writeable=*/true);
      replacements.push_back(tensor.getResult());
    }

    rewriter.replaceOp(group, replacements);
    return mlir::success();
  }

  std::string *failureReason;
  int64_t currentLogicalRank = -1;
};

static mlir::OwningOpRef<mlir::ModuleOp>
cloneGroupToStandaloneModule(GroupOp group) {
  mlir::Location loc = group.getLoc();
  mlir::OwningOpRef<mlir::ModuleOp> standaloneModule =
      mlir::ModuleOp::create(loc);
  mlir::OpBuilder moduleBuilder(standaloneModule->getBodyRegion());

  llvm::SmallVector<mlir::Type, 4> inputTypes;
  for (mlir::Value input : group.getInputs())
    inputTypes.push_back(input.getType());
  for (mlir::Value output : group.getOuts())
    inputTypes.push_back(output.getType());

  auto funcType =
      moduleBuilder.getFunctionType(inputTypes, group.getResultTypes());
  auto func = moduleBuilder.create<mlir::func::FuncOp>(
      loc, "group_to_tile_region", funcType);
  mlir::Block *entry = func.addEntryBlock();

  mlir::IRMapping mapping;
  unsigned argumentIndex = 0;
  for (mlir::Value input : group.getInputs())
    mapping.map(input, entry->getArgument(argumentIndex++));
  for (mlir::Value output : group.getOuts())
    mapping.map(output, entry->getArgument(argumentIndex++));

  mlir::OpBuilder builder(entry, entry->end());
  auto clonedGroup =
      mlir::cast<GroupOp>(builder.clone(*group.getOperation(), mapping));
  builder.create<mlir::func::ReturnOp>(loc, clonedGroup.getResults());
  return standaloneModule;
}

static GroupOp findSingleStandaloneGroup(mlir::ModuleOp module) {
  GroupOp found;
  module.walk([&](GroupOp group) {
    if (!found)
      found = group;
  });
  return found;
}

static bool isGroupOutputBoundary(GroupOp group, mlir::Value value,
                                  unsigned outputIndex) {
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &group.getBody().front())
    return false;
  unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
  return blockArg.getArgNumber() == inputCount + outputIndex;
}

static mlir::LogicalResult validateCandidateTile(
    mlir::RankedTensorType resultType, llvm::ArrayRef<int64_t> offsets,
    llvm::ArrayRef<int64_t> sizes, std::string *failureReason) {
  if (offsets.size() != static_cast<size_t>(resultType.getRank()) ||
      sizes.size() != static_cast<size_t>(resultType.getRank())) {
    setFailureReason(failureReason,
                     "candidate tile rank does not match group result rank");
    return mlir::failure();
  }

  for (auto [dim, values] : llvm::enumerate(llvm::zip(offsets, sizes))) {
    int64_t offset = std::get<0>(values);
    int64_t size = std::get<1>(values);
    int64_t bound = resultType.getDimSize(dim);
    if (mlir::ShapedType::isDynamic(bound)) {
      setFailureReason(failureReason,
                       "candidate tile requires static result shape");
      return mlir::failure();
    }
    if (offset < 0 || size <= 0 || offset + size > bound) {
      setFailureReason(failureReason,
                       "candidate tile is outside group result bounds");
      return mlir::failure();
    }
  }
  return mlir::success();
}

struct CandidateLoopTile {
  llvm::SmallVector<mlir::OpFoldResult, 4> loopOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> ivs;
  llvm::SmallVector<mlir::OpFoldResult, 4> tileSizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizeBounds;
};

static llvm::SmallVector<unsigned, 2>
getReductionLoopDims(mlir::linalg::LinalgOp op) {
  llvm::SmallVector<unsigned, 2> dims;
  for (auto [index, iteratorType] :
       llvm::enumerate(op.getIteratorTypesArray())) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      dims.push_back(static_cast<unsigned>(index));
  }
  return dims;
}

static mlir::LogicalResult
buildCandidateLoopTile(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::linalg::LinalgOp op, mlir::AffineMap outputMap,
                       llvm::ArrayRef<int64_t> candidateOffsets,
                       llvm::ArrayRef<int64_t> candidateSizes,
                       llvm::ArrayRef<int64_t> candidateReductionOffsets,
                       llvm::ArrayRef<int64_t> candidateReductionSizes,
                       CandidateLoopTile &tile, std::string *failureReason) {
  llvm::SmallVector<int64_t, 4> loopRanges = op.getStaticLoopRanges();
  if (llvm::any_of(loopRanges, [](int64_t value) {
        return mlir::ShapedType::isDynamic(value);
      })) {
    setFailureReason(failureReason,
                     "candidate tile requires static linalg loop ranges");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 2> reductionLoopDims = getReductionLoopDims(op);
  bool hasReductionSplit =
      !candidateReductionOffsets.empty() || !candidateReductionSizes.empty();
  if (candidateReductionOffsets.size() != candidateReductionSizes.size() ||
      (hasReductionSplit &&
       candidateReductionOffsets.size() != reductionLoopDims.size())) {
    setFailureReason(failureReason, "candidate reduction split rank mismatch");
    return mlir::failure();
  }

  llvm::DenseMap<unsigned, unsigned> resultDimForLoopDim;
  for (auto [resultDim, expr] : llvm::enumerate(outputMap.getResults())) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr) {
      setFailureReason(failureReason,
                       "candidate tile requires permutation-only output map");
      return mlir::failure();
    }
    resultDimForLoopDim[dimExpr.getPosition()] =
        static_cast<unsigned>(resultDim);
  }

  llvm::DenseMap<unsigned, unsigned> reductionOrdinalForLoopDim;
  for (auto [ordinal, loopDim] : llvm::enumerate(reductionLoopDims))
    reductionOrdinalForLoopDim[loopDim] = static_cast<unsigned>(ordinal);

  mlir::OpFoldResult zero = builder.getIndexAttr(0);
  for (auto [loopDim, loopRange] : llvm::enumerate(loopRanges)) {
    tile.sizeBounds.push_back(builder.getIndexAttr(loopRange));
    tile.loopOffsets.push_back(zero);
    tile.tileSizes.push_back(zero);

    auto resultDimIt = resultDimForLoopDim.find(static_cast<unsigned>(loopDim));
    if (resultDimIt != resultDimForLoopDim.end()) {
      unsigned resultDim = resultDimIt->second;
      tile.loopOffsets.back() =
          builder.getIndexAttr(candidateOffsets[resultDim]);
      tile.tileSizes.back() = builder.getIndexAttr(candidateSizes[resultDim]);
      tile.ivs.push_back(tile.loopOffsets.back());
      continue;
    }

    auto reductionDimIt =
        reductionOrdinalForLoopDim.find(static_cast<unsigned>(loopDim));
    if (reductionDimIt == reductionOrdinalForLoopDim.end())
      continue;
    if (!hasReductionSplit)
      continue;

    unsigned reductionOrdinal = reductionDimIt->second;
    int64_t reductionOffset = candidateReductionOffsets[reductionOrdinal];
    int64_t reductionSize = candidateReductionSizes[reductionOrdinal];
    if (reductionOffset < 0 || reductionSize <= 0 ||
        reductionOffset + reductionSize > loopRange) {
      setFailureReason(failureReason,
                       "candidate reduction split is outside loop bounds");
      return mlir::failure();
    }
    tile.loopOffsets.back() = builder.getIndexAttr(reductionOffset);
    tile.tileSizes.back() = builder.getIndexAttr(reductionSize);
    tile.ivs.push_back(tile.loopOffsets.back());
  }

  return mlir::success();
}

struct ReductionChunk {
  llvm::SmallVector<int64_t, 2> offsets;
  llvm::SmallVector<int64_t, 2> sizes;
};

static mlir::FailureOr<uint64_t> getCandidateReductionChunkCount(
    mlir::linalg::LinalgOp root,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  if (candidateReductionTileSizes.empty())
    return uint64_t{1};

  llvm::SmallVector<unsigned, 2> reductionLoopDims = getReductionLoopDims(root);
  if (candidateReductionTileSizes.size() != reductionLoopDims.size()) {
    setFailureReason(failureReason, "candidate reduction split rank mismatch");
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 4> loopRanges = root.getStaticLoopRanges();
  llvm::SmallVector<int64_t, 2> reductionRanges;
  reductionRanges.reserve(reductionLoopDims.size());
  for (unsigned loopDim : reductionLoopDims) {
    if (loopDim >= loopRanges.size() ||
        mlir::ShapedType::isDynamic(loopRanges[loopDim])) {
      setFailureReason(failureReason,
                       "candidate reduction split requires static loop ranges");
      return mlir::failure();
    }
    reductionRanges.push_back(loopRanges[loopDim]);
  }

  for (auto [range, splitSize] :
       llvm::zip(reductionRanges, candidateReductionTileSizes)) {
    if (range <= 0 || splitSize <= 0 || splitSize > range) {
      setFailureReason(failureReason,
                       "candidate reduction split is outside loop bounds");
      return mlir::failure();
    }
  }

  uint64_t chunkCount = 0;
  switch (wafer::detail::checkedStaticTileProduct(
      reductionRanges, candidateReductionTileSizes, chunkCount)) {
  case wafer::detail::CheckedStaticTileProductStatus::Success:
    return chunkCount;
  case wafer::detail::CheckedStaticTileProductStatus::Overflow:
    setFailureReason(
        failureReason,
        "candidate reduction split expansion count is not representable");
    return mlir::failure();
  case wafer::detail::CheckedStaticTileProductStatus::InvalidInput:
    setFailureReason(failureReason,
                     "candidate reduction split has invalid static ranges");
    return mlir::failure();
  }
  llvm_unreachable("unknown checked tile product status");
}

static void
buildReductionChunkProducts(llvm::ArrayRef<int64_t> ranges,
                            llvm::ArrayRef<int64_t> splitSizes, unsigned dim,
                            llvm::SmallVectorImpl<int64_t> &currentOffsets,
                            llvm::SmallVectorImpl<int64_t> &currentSizes,
                            llvm::SmallVectorImpl<ReductionChunk> &chunks) {
  if (dim == ranges.size()) {
    chunks.push_back(
        ReductionChunk{llvm::SmallVector<int64_t, 2>(currentOffsets.begin(),
                                                     currentOffsets.end()),
                       llvm::SmallVector<int64_t, 2>(currentSizes.begin(),
                                                     currentSizes.end())});
    return;
  }

  for (int64_t offset = 0; offset < ranges[dim];) {
    int64_t size = std::min(splitSizes[dim], ranges[dim] - offset);
    currentOffsets.push_back(offset);
    currentSizes.push_back(size);
    buildReductionChunkProducts(ranges, splitSizes, dim + 1, currentOffsets,
                                currentSizes, chunks);
    currentOffsets.pop_back();
    currentSizes.pop_back();
    offset += size;
  }
}

static mlir::FailureOr<llvm::SmallVector<ReductionChunk, 8>>
buildReductionChunks(mlir::linalg::LinalgOp root,
                     llvm::ArrayRef<int64_t> candidateReductionTileSizes,
                     std::string *failureReason) {
  mlir::FailureOr<uint64_t> chunkCount = getCandidateReductionChunkCount(
      root, candidateReductionTileSizes, failureReason);
  if (mlir::failed(chunkCount))
    return mlir::failure();
  if (candidateReductionTileSizes.empty())
    return llvm::SmallVector<ReductionChunk, 8>{ReductionChunk{}};

  if (*chunkCount > wafer::detail::kCompleteCandidateMaterializationBudget) {
    setFailureReason(
        failureReason,
        "candidate reduction split exceeds the eager materialization budget "
        "(4096 chunks); this is an implementation resource limit, not an IR "
        "or target legality restriction");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 2> reductionLoopDims = getReductionLoopDims(root);
  llvm::SmallVector<int64_t, 4> loopRanges = root.getStaticLoopRanges();
  llvm::SmallVector<int64_t, 2> reductionRanges;
  for (unsigned loopDim : reductionLoopDims)
    reductionRanges.push_back(loopRanges[loopDim]);

  llvm::SmallVector<ReductionChunk, 8> chunks;
  chunks.reserve(static_cast<size_t>(*chunkCount));
  llvm::SmallVector<int64_t, 2> currentOffsets;
  llvm::SmallVector<int64_t, 2> currentSizes;
  buildReductionChunkProducts(reductionRanges, candidateReductionTileSizes,
                              /*dim=*/0, currentOffsets, currentSizes, chunks);
  return chunks;
}

static std::optional<ComputeReduceKind>
inferCandidateReduceKind(mlir::linalg::GenericOp generic,
                         std::string *failureReason) {
  if (generic.getRegionInputArgs().size() != 1 ||
      generic.getRegionOutputArgs().size() != 1) {
    setFailureReason(
        failureReason,
        "candidate reduction split requires one input and one accumulator");
    return std::nullopt;
  }
  return matchExactReductionKind(generic.getRegionOutputArgs(), /*redPos=*/0,
                                 generic.getRegionInputArgs().front(),
                                 "candidate reduction split", failureReason);
}

static mlir::FailureOr<ComputeReduceKind>
getCandidateCombineKind(mlir::linalg::LinalgOp root,
                        std::string *failureReason) {
  if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::BatchMatmulOp>(
          root.getOperation()))
    return ComputeReduceKind::Sum;
  if (auto generic =
          mlir::dyn_cast<mlir::linalg::GenericOp>(root.getOperation())) {
    std::optional<ComputeReduceKind> kind =
        inferCandidateReduceKind(generic, failureReason);
    if (!kind)
      return mlir::failure();
    return *kind;
  }

  setFailureReason(
      failureReason,
      "candidate reduction split requires matmul, batch_matmul, or generic "
      "reduction root");
  return mlir::failure();
}

static mlir::FailureOr<mlir::TypedAttr>
getNeutralScalarAttr(mlir::Type elementType, ComputeReduceKind kind,
                     std::string *failureReason) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
    switch (kind) {
    case ComputeReduceKind::Sum:
      return mlir::cast<mlir::TypedAttr>(mlir::FloatAttr::get(floatType, 0.0));
    case ComputeReduceKind::Max:
      return mlir::cast<mlir::TypedAttr>(mlir::FloatAttr::get(
          floatType, llvm::APFloat::getInf(floatType.getFloatSemantics(),
                                           /*Negative=*/true)));
    case ComputeReduceKind::Min:
      return mlir::cast<mlir::TypedAttr>(mlir::FloatAttr::get(
          floatType, llvm::APFloat::getInf(floatType.getFloatSemantics(),
                                           /*Negative=*/false)));
    case ComputeReduceKind::Avg:
      break;
    }
  }

  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    unsigned width = intType.getWidth();
    switch (kind) {
    case ComputeReduceKind::Sum:
      return mlir::cast<mlir::TypedAttr>(
          mlir::IntegerAttr::get(intType, llvm::APInt(width, 0)));
    case ComputeReduceKind::Max:
      return mlir::cast<mlir::TypedAttr>(mlir::IntegerAttr::get(
          intType, llvm::APInt::getSignedMinValue(width)));
    case ComputeReduceKind::Min:
      return mlir::cast<mlir::TypedAttr>(mlir::IntegerAttr::get(
          intType, llvm::APInt::getSignedMaxValue(width)));
    case ComputeReduceKind::Avg:
      break;
    }
  }

  setFailureReason(
      failureReason,
      "candidate reduction split requires float or integer accumulator type");
  return mlir::failure();
}

static mlir::FailureOr<mlir::Value>
createNeutralInitTensor(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::RankedTensorType resultType,
                        ComputeReduceKind kind, std::string *failureReason) {
  mlir::FailureOr<mlir::TypedAttr> attr =
      getNeutralScalarAttr(resultType.getElementType(), kind, failureReason);
  if (mlir::failed(attr))
    return mlir::failure();

  auto constant = builder.create<mlir::arith::ConstantOp>(loc, *attr);
  auto empty = builder.create<mlir::tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType());
  auto fill = builder.create<mlir::linalg::FillOp>(loc, constant.getResult(),
                                                   empty.getResult());
  return fill.getResult(0);
}

static mlir::FailureOr<mlir::Value>
createPartialCombine(mlir::OpBuilder &builder, mlir::Location loc,
                     ComputeReduceKind kind, mlir::Value accumulator,
                     mlir::Value partial, mlir::Value outputInit,
                     std::string *failureReason) {
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(partial.getType());
  if (!resultType || accumulator.getType() != partial.getType() ||
      outputInit.getType() != partial.getType()) {
    setFailureReason(failureReason,
                     "candidate reduction split accumulator type mismatch");
    return mlir::failure();
  }

  mlir::MLIRContext *context = builder.getContext();
  if (!mlir::isa<mlir::FloatType, mlir::IntegerType>(
          resultType.getElementType())) {
    setFailureReason(failureReason,
                     "candidate reduction split requires float or integer "
                     "accumulator element type");
    return mlir::failure();
  }

  mlir::AffineMap identity =
      mlir::AffineMap::getMultiDimIdentityMap(resultType.getRank(), context);
  llvm::SmallVector<mlir::AffineMap, 3> indexingMaps = {identity, identity,
                                                        identity};
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
      resultType.getRank(), mlir::utils::IteratorType::parallel);

  auto add = builder.create<mlir::linalg::GenericOp>(
      loc, resultType, mlir::ValueRange{accumulator, partial},
      mlir::ValueRange{outputInit}, indexingMaps, iteratorTypes,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange blockArgs) {
        mlir::Value value;
        mlir::Type elementType = resultType.getElementType();
        bool isFloat = mlir::isa<mlir::FloatType>(elementType);
        bool isInteger = mlir::isa<mlir::IntegerType>(elementType);
        if (!isFloat && !isInteger)
          return;

        switch (kind) {
        case ComputeReduceKind::Sum:
          if (isFloat) {
            value = nestedBuilder.create<mlir::arith::AddFOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          } else {
            value = nestedBuilder.create<mlir::arith::AddIOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          }
          break;
        case ComputeReduceKind::Max:
          if (isFloat) {
            value = nestedBuilder.create<mlir::arith::MaximumFOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          } else {
            value = nestedBuilder.create<mlir::arith::MaxSIOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          }
          break;
        case ComputeReduceKind::Min:
          if (isFloat) {
            value = nestedBuilder.create<mlir::arith::MinimumFOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          } else {
            value = nestedBuilder.create<mlir::arith::MinSIOp>(
                nestedLoc, blockArgs[0], blockArgs[1]);
          }
          break;
        case ComputeReduceKind::Avg:
          return;
        }
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, value);
      });

  return add->getResult(0);
}

static mlir::FailureOr<mlir::Value> materializeCandidateRootTileValue(
    GroupOp group, mlir::linalg::LinalgOp root, unsigned outputIndex,
    llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  if (root.getNumDpsInits() != 1 || root->getNumResults() != 1) {
    setFailureReason(failureReason,
                     "candidate tile materialization requires one DPS output");
    return mlir::failure();
  }
  if (!root.hasOnlyProjectedPermutations()) {
    setFailureReason(
        failureReason,
        "candidate tile materialization requires permutation-only maps");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 2> reductionLoopDims = getReductionLoopDims(root);
  bool hasDirectOutputInit =
      isGroupOutputBoundary(group, root.getDpsInits().front(), outputIndex);
  if (!hasDirectOutputInit && reductionLoopDims.empty()) {
    setFailureReason(failureReason,
                     "candidate tile materialization requires direct output "
                     "boundary init for non-reduction roots");
    return mlir::failure();
  }

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType) {
    setFailureReason(failureReason,
                     "candidate tile materialization result is not ranked");
    return mlir::failure();
  }
  if (mlir::failed(validateCandidateTile(resultType, candidateTileOffsets,
                                         candidateTileSizes, failureReason)))
    return mlir::failure();

  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
      root.getIndexingMapsArray();
  unsigned outputMapIndex = static_cast<unsigned>(root.getNumDpsInputs());
  if (outputMapIndex >= indexingMaps.size()) {
    setFailureReason(failureReason,
                     "candidate tile materialization missing output map");
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 2> rootReductionTileSizes;
  if (!reductionLoopDims.empty())
    rootReductionTileSizes.assign(candidateReductionTileSizes.begin(),
                                  candidateReductionTileSizes.end());

  std::optional<ComputeReduceKind> combineKind;
  if (!rootReductionTileSizes.empty()) {
    mlir::FailureOr<ComputeReduceKind> kind =
        getCandidateCombineKind(root, failureReason);
    if (mlir::failed(kind))
      return mlir::failure();
    combineKind = *kind;
  }

  mlir::FailureOr<llvm::SmallVector<ReductionChunk, 8>> reductionChunks =
      buildReductionChunks(root, rootReductionTileSizes, failureReason);
  if (mlir::failed(reductionChunks))
    return mlir::failure();

  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::Value, 4> valuesToTile(root->operand_begin(),
                                                 root->operand_end());
  unsigned initOperandIndex = static_cast<unsigned>(root.getNumDpsInputs());
  mlir::Value outputInitTile;
  mlir::Value neutralInitTile;
  mlir::Value accumulator;

  for (const ReductionChunk &chunk : *reductionChunks) {
    CandidateLoopTile loopTile;
    if (mlir::failed(buildCandidateLoopTile(
            builder, root.getLoc(), root, indexingMaps[outputMapIndex],
            candidateTileOffsets, candidateTileSizes, chunk.offsets,
            chunk.sizes, loopTile, failureReason)))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> tiledOperands =
        mlir::linalg::makeTiledShapes(builder, root.getLoc(), root,
                                      valuesToTile, loopTile.ivs,
                                      loopTile.tileSizes, loopTile.sizeBounds,
                                      /*omitPartialTileCheck=*/true);
    if (!outputInitTile) {
      outputInitTile = tiledOperands[initOperandIndex];
    } else if (initOperandIndex < tiledOperands.size()) {
      mlir::Operation *unusedInitSlice =
          tiledOperands[initOperandIndex].getDefiningOp();
      if (!neutralInitTile) {
        auto initType =
            mlir::dyn_cast<mlir::RankedTensorType>(outputInitTile.getType());
        if (!initType || !combineKind) {
          setFailureReason(
              failureReason,
              "candidate reduction split neutral init type mismatch");
          return mlir::failure();
        }
        mlir::FailureOr<mlir::Value> neutral = createNeutralInitTensor(
            builder, root.getLoc(), initType, *combineKind, failureReason);
        if (mlir::failed(neutral))
          return mlir::failure();
        neutralInitTile = *neutral;
      }
      tiledOperands[initOperandIndex] = neutralInitTile;
      if (unusedInitSlice && unusedInitSlice->use_empty())
        unusedInitSlice->erase();
    }

    llvm::SmallVector<mlir::Type, 2> resultTypes =
        mlir::linalg::getTensorOutputTypes(root, tiledOperands);
    if (resultTypes.size() != 1) {
      setFailureReason(
          failureReason,
          "candidate tile materialization expected one tiled result type");
      return mlir::failure();
    }

    mlir::Operation *tiled =
        mlir::clone(builder, root.getOperation(), resultTypes, tiledOperands);
    auto tiledLinalg = mlir::cast<mlir::linalg::LinalgOp>(tiled);
    mlir::linalg::offsetIndices(builder, tiledLinalg, loopTile.loopOffsets);

    builder.setInsertionPointAfter(tiled);
    mlir::Value partial = tiled->getResult(0);
    if (!accumulator) {
      accumulator = partial;
      continue;
    }

    if (!combineKind) {
      setFailureReason(failureReason,
                       "candidate reduction split missing combine kind");
      return mlir::failure();
    }
    mlir::FailureOr<mlir::Value> combined =
        createPartialCombine(builder, root.getLoc(), *combineKind, accumulator,
                             partial, outputInitTile, failureReason);
    if (mlir::failed(combined))
      return mlir::failure();
    accumulator = *combined;
    builder.setInsertionPointAfter(accumulator.getDefiningOp());
  }

  if (!accumulator) {
    setFailureReason(failureReason,
                     "candidate tile materialization produced no tiled result");
    return mlir::failure();
  }
  return accumulator;
}

static mlir::FailureOr<mlir::Value>
getCandidateOutputBoundary(GroupOp group, unsigned outputIndex,
                           std::string *failureReason) {
  unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
  mlir::Block &body = group.getBody().front();
  if (inputCount + outputIndex >= body.getNumArguments()) {
    setFailureReason(failureReason,
                     "candidate tile materialization missing output boundary");
    return mlir::failure();
  }
  return body.getArgument(inputCount + outputIndex);
}

static mlir::Value
insertCandidateRootTile(mlir::linalg::LinalgOp root, mlir::Value tileValue,
                        mlir::Value outputDestination,
                        llvm::ArrayRef<int64_t> candidateTileOffsets,
                        llvm::ArrayRef<int64_t> candidateTileSizes) {
  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> strides;
  offsets.reserve(candidateTileOffsets.size());
  sizes.reserve(candidateTileSizes.size());
  strides.reserve(candidateTileSizes.size());
  for (auto [offset, size] :
       llvm::zip(candidateTileOffsets, candidateTileSizes)) {
    offsets.push_back(builder.getIndexAttr(offset));
    sizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }

  auto inserted = builder.create<mlir::tensor::InsertSliceOp>(
      root.getLoc(), tileValue, outputDestination, offsets, sizes, strides);
  return inserted.getResult();
}

static mlir::FailureOr<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
collectCandidateRoots(GroupOp group, bool rejectProducerChains,
                      std::string *failureReason) {
  auto yield =
      mlir::dyn_cast<GroupYieldOp>(group.getBody().front().getTerminator());
  if (!yield || yield.getValues().empty()) {
    setFailureReason(failureReason,
                     "candidate tile materialization requires group results");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::linalg::LinalgOp, 4> roots;
  llvm::DenseSet<mlir::Operation *> seenRoots;
  for (auto [index, value] : llvm::enumerate(yield.getValues())) {
    auto root =
        mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(value.getDefiningOp());
    if (!root) {
      setFailureReason(failureReason,
                       "candidate tile materialization requires linalg roots");
      return mlir::failure();
    }
    if (root->getBlock() != &group.getBody().front() ||
        !seenRoots.insert(root.getOperation()).second ||
        root->getNumResults() != 1 || root.getNumDpsInits() != 1 ||
        root->getResult(0) != value) {
      setFailureReason(failureReason,
                       "candidate multi-output coverage requires distinct "
                       "single-result yielded roots");
      return mlir::failure();
    }

    unsigned matchingYieldUses = 0;
    for (mlir::OpOperand &use : value.getUses()) {
      if (use.getOwner() == yield.getOperation() &&
          use.getOperandNumber() == index) {
        ++matchingYieldUses;
        continue;
      }
      setFailureReason(failureReason, "candidate multi-output coverage "
                                      "requires independent yielded roots");
      return mlir::failure();
    }
    if (matchingYieldUses != 1) {
      setFailureReason(failureReason, "candidate multi-output coverage "
                                      "requires independent yielded roots");
      return mlir::failure();
    }

    if (rejectProducerChains) {
      for (mlir::Value input : root.getDpsInputs()) {
        if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
          continue;
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(input);
        if (blockArg && blockArg.getOwner() == &group.getBody().front())
          continue;
        setFailureReason(
            failureReason,
            "complete candidate traversal does not support tensor producer "
            "chains");
        return mlir::failure();
      }

      mlir::Value init = root.getDpsInits().front();
      bool hasDirectOutputInit =
          isGroupOutputBoundary(group, init, static_cast<unsigned>(index));
      bool hasReduction = !getReductionLoopDims(root).empty();
      if (!hasDirectOutputInit &&
          (!hasReduction || !init.getDefiningOp<mlir::linalg::FillOp>())) {
        setFailureReason(
            failureReason,
            "complete candidate traversal does not support output producer "
            "chains");
        return mlir::failure();
      }
    }
    roots.push_back(root);
  }

  return roots;
}

struct CandidateOutputTile {
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
};

static mlir::FailureOr<uint64_t>
getCandidateOutputTileCount(mlir::RankedTensorType resultType,
                            llvm::ArrayRef<int64_t> candidateTileSizes,
                            std::string *failureReason) {
  if (candidateTileSizes.size() != static_cast<size_t>(resultType.getRank())) {
    setFailureReason(failureReason,
                     "candidate tile rank does not match group result rank");
    return mlir::failure();
  }

  for (auto [bound, tileSize] :
       llvm::zip(resultType.getShape(), candidateTileSizes)) {
    if (mlir::ShapedType::isDynamic(bound)) {
      setFailureReason(failureReason,
                       "complete candidate traversal requires static result "
                       "shape");
      return mlir::failure();
    }
    if (bound <= 0 || tileSize <= 0 || tileSize > bound) {
      setFailureReason(
          failureReason,
          "complete candidate traversal tile size is outside result bounds");
      return mlir::failure();
    }
  }

  uint64_t tileCount = 0;
  switch (wafer::detail::checkedStaticTileProduct(
      resultType.getShape(), candidateTileSizes, tileCount)) {
  case wafer::detail::CheckedStaticTileProductStatus::Success:
    return tileCount;
  case wafer::detail::CheckedStaticTileProductStatus::Overflow:
    setFailureReason(
        failureReason,
        "complete candidate traversal output tile count is not representable");
    return mlir::failure();
  case wafer::detail::CheckedStaticTileProductStatus::InvalidInput:
    setFailureReason(
        failureReason,
        "complete candidate traversal has invalid static output ranges");
    return mlir::failure();
  }
  llvm_unreachable("unknown checked tile product status");
}

static mlir::LogicalResult checkCompleteCandidateExpansionBudget(
    llvm::ArrayRef<mlir::linalg::LinalgOp> roots,
    mlir::RankedTensorType resultType,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  mlir::FailureOr<uint64_t> outputTileCount = getCandidateOutputTileCount(
      resultType, candidateTileSizes, failureReason);
  if (mlir::failed(outputTileCount))
    return mlir::failure();

  llvm::SmallVector<uint64_t, 4> reductionChunkCounts;
  reductionChunkCounts.reserve(roots.size());
  for (mlir::linalg::LinalgOp root : roots) {
    if (candidateReductionTileSizes.empty() ||
        getReductionLoopDims(root).empty()) {
      reductionChunkCounts.push_back(1);
      continue;
    }
    mlir::FailureOr<uint64_t> chunkCount = getCandidateReductionChunkCount(
        root, candidateReductionTileSizes, failureReason);
    if (mlir::failed(chunkCount))
      return mlir::failure();
    reductionChunkCounts.push_back(*chunkCount);
  }

  uint64_t materializationCount = 0;
  switch (wafer::detail::checkCompleteCandidateExpansionBudget(
      *outputTileCount, reductionChunkCounts, materializationCount)) {
  case wafer::detail::CompleteCandidateExpansionStatus::WithinBudget:
    return mlir::success();
  case wafer::detail::CompleteCandidateExpansionStatus::BudgetExceeded:
    setFailureReason(
        failureReason,
        "complete candidate traversal exceeds the eager materialization "
        "budget; this is an implementation resource limit, not an IR or "
        "target legality restriction");
    return mlir::failure();
  case wafer::detail::CompleteCandidateExpansionStatus::CountOverflow:
    setFailureReason(
        failureReason,
        "complete candidate traversal expansion count is not representable");
    return mlir::failure();
  case wafer::detail::CompleteCandidateExpansionStatus::InvalidInput:
    setFailureReason(
        failureReason,
        "complete candidate traversal has invalid static expansion counts");
    return mlir::failure();
  }
  llvm_unreachable("unknown complete candidate expansion status");
}

static void buildCandidateOutputTileProducts(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> tileSizes,
    unsigned dim, llvm::SmallVectorImpl<int64_t> &currentOffsets,
    llvm::SmallVectorImpl<int64_t> &currentSizes,
    llvm::SmallVectorImpl<CandidateOutputTile> &tiles) {
  if (dim == shape.size()) {
    tiles.push_back(
        CandidateOutputTile{llvm::SmallVector<int64_t, 4>(
                                currentOffsets.begin(), currentOffsets.end()),
                            llvm::SmallVector<int64_t, 4>(currentSizes.begin(),
                                                          currentSizes.end())});
    return;
  }

  for (int64_t offset = 0; offset < shape[dim];) {
    int64_t size = std::min(tileSizes[dim], shape[dim] - offset);
    currentOffsets.push_back(offset);
    currentSizes.push_back(size);
    buildCandidateOutputTileProducts(shape, tileSizes, dim + 1, currentOffsets,
                                     currentSizes, tiles);
    currentOffsets.pop_back();
    currentSizes.pop_back();
    offset += size;
  }
}

static mlir::FailureOr<llvm::SmallVector<CandidateOutputTile, 8>>
buildCandidateOutputTiles(mlir::RankedTensorType resultType,
                          llvm::ArrayRef<int64_t> candidateTileSizes,
                          std::string *failureReason) {
  mlir::FailureOr<uint64_t> tileCount = getCandidateOutputTileCount(
      resultType, candidateTileSizes, failureReason);
  if (mlir::failed(tileCount))
    return mlir::failure();

  llvm::SmallVector<CandidateOutputTile, 8> tiles;
  tiles.reserve(static_cast<size_t>(*tileCount));
  llvm::SmallVector<int64_t, 4> currentOffsets;
  llvm::SmallVector<int64_t, 4> currentSizes;
  buildCandidateOutputTileProducts(resultType.getShape(), candidateTileSizes,
                                   /*dim=*/0, currentOffsets, currentSizes,
                                   tiles);
  return tiles;
}

static mlir::LogicalResult materializeCandidateTileSlices(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      collectCandidateRoots(group, /*rejectProducerChains=*/false,
                            failureReason);
  if (mlir::failed(roots))
    return mlir::failure();

  auto yield =
      mlir::cast<GroupYieldOp>(group.getBody().front().getTerminator());

  llvm::SmallVector<mlir::Value, 4> insertedValues;
  for (auto [index, root] : llvm::enumerate(*roots)) {
    mlir::FailureOr<mlir::Value> tileValue = materializeCandidateRootTileValue(
        group, root, static_cast<unsigned>(index), candidateTileOffsets,
        candidateTileSizes, candidateReductionTileSizes, failureReason);
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        group, static_cast<unsigned>(index), failureReason);
    if (mlir::failed(tileValue) || mlir::failed(outputBoundary))
      return mlir::failure();
    insertedValues.push_back(
        insertCandidateRootTile(root, *tileValue, *outputBoundary,
                                candidateTileOffsets, candidateTileSizes));
  }

  for (auto [index, inserted] : llvm::enumerate(insertedValues))
    yield->setOperand(index, inserted);
  for (mlir::linalg::LinalgOp root : *roots)
    root->erase();
  return mlir::success();
}

static mlir::LogicalResult materializeCompleteCandidateTraversal(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      collectCandidateRoots(group, /*rejectProducerChains=*/true,
                            failureReason);
  if (mlir::failed(roots))
    return mlir::failure();

  auto firstResultType = mlir::dyn_cast<mlir::RankedTensorType>(
      (*roots).front()->getResult(0).getType());
  if (!firstResultType) {
    setFailureReason(failureReason,
                     "complete candidate traversal result is not ranked");
    return mlir::failure();
  }

  bool hasReductionRoot = false;
  for (mlir::linalg::LinalgOp root : *roots) {
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
    if (!resultType || resultType.getShape() != firstResultType.getShape()) {
      setFailureReason(
          failureReason,
          "complete candidate traversal requires equal static result shapes");
      return mlir::failure();
    }
    hasReductionRoot |= !getReductionLoopDims(root).empty();
  }
  if (!candidateReductionTileSizes.empty() && !hasReductionRoot) {
    setFailureReason(failureReason,
                     "candidate reduction split requires a reduction root");
    return mlir::failure();
  }
  if (mlir::failed(checkCompleteCandidateExpansionBudget(
          *roots, firstResultType, candidateTileSizes,
          candidateReductionTileSizes, failureReason)))
    return mlir::failure();

  mlir::FailureOr<llvm::SmallVector<CandidateOutputTile, 8>> tiles =
      buildCandidateOutputTiles(firstResultType, candidateTileSizes,
                                failureReason);
  if (mlir::failed(tiles))
    return mlir::failure();

  auto yield =
      mlir::cast<GroupYieldOp>(group.getBody().front().getTerminator());
  llvm::SmallVector<mlir::Value, 4> completeOutputs;
  for (auto [index, root] : llvm::enumerate(*roots)) {
    mlir::FailureOr<mlir::Value> outputBoundary = getCandidateOutputBoundary(
        group, static_cast<unsigned>(index), failureReason);
    if (mlir::failed(outputBoundary))
      return mlir::failure();
    mlir::Value output = *outputBoundary;
    for (const CandidateOutputTile &tile : *tiles) {
      mlir::FailureOr<mlir::Value> tileValue =
          materializeCandidateRootTileValue(
              group, root, static_cast<unsigned>(index), tile.offsets,
              tile.sizes, candidateReductionTileSizes, failureReason);
      if (mlir::failed(tileValue))
        return mlir::failure();
      output = insertCandidateRootTile(root, *tileValue, output, tile.offsets,
                                       tile.sizes);
    }
    completeOutputs.push_back(output);
  }

  for (auto [index, output] : llvm::enumerate(completeOutputs))
    yield->setOperand(index, output);
  for (mlir::linalg::LinalgOp root : *roots)
    root->erase();
  return mlir::success();
}

static void configureGroupToTileRegionTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<mlir::arith::ArithDialect,
                         mlir::bufferization::BufferizationDialect,
                         mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                         mlir::scf::SCFDialect, wafer::WaferDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addIllegalOp<GroupOp, GroupYieldOp>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

static mlir::LogicalResult convertGroupToTileRegionModuleInPlace(
    mlir::ModuleOp module, mlir::MLIRContext *context,
    int64_t currentLogicalRank, std::string *failureReason) {
  mlir::ConversionTarget target(*context);
  configureGroupToTileRegionTarget(target);

  mlir::RewritePatternSet patterns(context);
  patterns.add<GroupToTileRegionLoweringPattern>(context, failureReason,
                                                 currentLogicalRank);

  bool conversionSucceeded = false;
  {
    mlir::ScopedDiagnosticHandler handler(
        context, [](mlir::Diagnostic &) { return mlir::success(); });
    conversionSucceeded = mlir::succeeded(
        mlir::applyFullConversion(module, target, std::move(patterns)));
  }

  if (!conversionSucceeded) {
    if (!failureReason || failureReason->empty())
      setFailureReason(failureReason, "group-to-tile-region lowering failed");
    return mlir::failure();
  }

  if (mlir::failed(mlir::verify(module))) {
    setFailureReason(failureReason,
                     "lowered tile-region module failed verifier");
    return mlir::failure();
  }

  return mlir::success();
}

struct ConvertGroupToTileRegionPass
    : public wafer::impl::ConvertGroupToTileRegionPassBase<
          ConvertGroupToTileRegionPass> {
  using wafer::impl::ConvertGroupToTileRegionPassBase<
      ConvertGroupToTileRegionPass>::ConvertGroupToTileRegionPassBase;

  void runOnOperation() final {
    if (logicalRank < 0) {
      getOperation()->emitError()
          << "missing_logical_rank: group-to-tile-region conversion requires "
             "an explicit non-negative logical-rank";
      signalPassFailure();
      return;
    }
    mlir::MLIRContext *context = &getContext();
    mlir::ConversionTarget target(*context);
    configureGroupToTileRegionTarget(target);

    std::string failureReason;
    mlir::RewritePatternSet patterns(context);
    patterns.add<GroupToTileRegionLoweringPattern>(context, &failureReason,
                                                   logicalRank);

    if (mlir::succeeded(mlir::applyFullConversion(getOperation(), target,
                                                  std::move(patterns))))
      return;

    if (!failureReason.empty())
      getOperation().emitError(failureReason);
    else
      getOperation().emitError("group to tile-region conversion failed");
    signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult wafer::lowerGroupToTileRegionModule(
    GroupOp group, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason, int64_t currentLogicalRank) {
  if (failureReason)
    failureReason->clear();

  module = cloneGroupToStandaloneModule(group);
  return convertGroupToTileRegionModuleInPlace(
      *module, group.getContext(), currentLogicalRank, failureReason);
}

mlir::LogicalResult wafer::lowerCandidateGroupToTileRegionModule(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank) {
  if (failureReason)
    failureReason->clear();

  module = cloneGroupToStandaloneModule(group);
  GroupOp clonedGroup = findSingleStandaloneGroup(*module);
  if (!clonedGroup) {
    setFailureReason(failureReason, "standalone module has no wafer.group");
    return mlir::failure();
  }

  if (mlir::failed(materializeCandidateTileSlices(
          clonedGroup, candidateTileOffsets, candidateTileSizes,
          candidateReductionTileSizes, failureReason)))
    return mlir::failure();

  return convertGroupToTileRegionModuleInPlace(
      *module, group.getContext(), currentLogicalRank, failureReason);
}

mlir::LogicalResult wafer::lowerCompleteCandidateGroupToTileRegionModule(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank) {
  if (failureReason)
    failureReason->clear();

  mlir::OwningOpRef<mlir::ModuleOp> candidateModule =
      cloneGroupToStandaloneModule(group);
  GroupOp clonedGroup = findSingleStandaloneGroup(*candidateModule);
  if (!clonedGroup) {
    setFailureReason(failureReason, "standalone module has no wafer.group");
    return mlir::failure();
  }

  if (mlir::failed(materializeCompleteCandidateTraversal(
          clonedGroup, candidateTileSizes, candidateReductionTileSizes,
          failureReason)))
    return mlir::failure();

  if (mlir::failed(convertGroupToTileRegionModuleInPlace(
          *candidateModule, group.getContext(), currentLogicalRank,
          failureReason)))
    return mlir::failure();

  module = std::move(candidateModule);
  return mlir::success();
}

void wafer::dumpGroupToTileRegionModule(mlir::ModuleOp module,
                                        llvm::StringRef groupLabel,
                                        llvm::raw_ostream &os) {
  os << "wafer.group_to_tile_region group " << groupLabel << "\n";
  module.print(os);
  os << "\n";
}
