//===- MovementLowering.cpp - Tile-region movement lowering ------------===//

#include "Internal.h"

#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <optional>
#include <string>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace {

constexpr int64_t kNchw2NhwcPermutation[] = {0, 2, 3, 1};
constexpr int64_t kNhwc2NchwPermutation[] = {0, 3, 1, 2};

class ScratchRecorderHolder {
protected:
  explicit ScratchRecorderHolder(
      TileRegionToInstrBufferRecorder *bufferRecorder)
      : bufferRecorder(bufferRecorder) {}
  TileRegionToInstrBufferRecorder *bufferRecorder = nullptr;
};

struct MovementEndpoint {
  mlir::Value base;
  analysis::IndexRelation viewToBase;

  mlir::MemRefType getType() const {
    return mlir::cast<mlir::MemRefType>(base.getType());
  }
};

mlir::FailureOr<MovementEndpoint>
resolveStaticMovementEndpoint(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type)
    return mlir::failure();
  auto relation = analysis::IndexRelation::identity(type.getShape());
  if (!relation.isExact())
    return mlir::failure();
  while (auto subview = value.getDefiningOp<mlir::memref::SubViewOp>()) {
    auto viewType = subview.getType();
    auto baseType = subview.getSourceType();
    auto dropped = subview.getDroppedDims();
    llvm::SmallVector<int64_t, 4> fullViewShape;
    unsigned viewAxis = 0;
    for (unsigned axis = 0; axis < dropped.size(); ++axis)
      fullViewShape.push_back(
          dropped.test(axis) ? 1 : viewType.getDimSize(viewAxis++));
    auto expand = analysis::IndexRelation::staticReshape(viewType.getShape(),
                                                         fullViewShape);
    auto slice = analysis::IndexRelation::slice(
        fullViewShape, baseType.getShape(), subview.getMixedOffsets(),
        subview.getMixedStrides());
    if (!expand.isExact() || !slice.isExact())
      return mlir::failure();
    auto step = expand.get()->compose(*slice.get());
    if (!step.isExact())
      return mlir::failure();
    relation = relation.get()->compose(*step.get());
    if (!relation.isExact())
      return mlir::failure();
    value = subview.getSource();
  }
  return MovementEndpoint{value, std::move(*relation.relation)};
}

// Preserve an existing standard endpoint whose dynamic base address is carried
// by its current SSA view. Static and blocked subviews use the common relation.
mlir::FailureOr<MovementEndpoint> resolveMovementEndpoint(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type)
    return mlir::failure();
  auto memory = getWaferMemoryAttr(type);
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (memory && isStandardViewCompatibleLayout(memory.getLayout()) &&
      mlir::succeeded(mlir::getStridesAndOffset(type, strides, offset)) &&
      mlir::ShapedType::isDynamic(offset)) {
    auto identity = analysis::IndexRelation::identity(type.getShape());
    if (!identity.isExact())
      return mlir::failure();
    return MovementEndpoint{value, std::move(*identity.relation)};
  }
  return resolveStaticMovementEndpoint(value);
}

class TileLoadLowering : public mlir::OpRewritePattern<StorageLoadOp> {
public:
  TileLoadLowering(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<StorageLoadOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(StorageLoadOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType = mlir::cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(destType.getShape());
    if (!relation.isExact())
      return failPattern(rewriter, op,
                         "tile.load identity relation is not exact");

    if (mlir::succeeded(analysis::TransferRealizability::proveCompactDma(
            sourceType, destType, *relation.get()))) {
      mlir::FailureOr<MovementDescriptor> descriptor =
          getStridedTensorDescriptor(rewriter, op, op.getSource().getType(),
                                     "tile.load source");
      if (mlir::failed(descriptor))
        return mlir::failure();

      createRDMA(rewriter, op.getLoc(), op.getSource(), op.getDest(),
                 *descriptor);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(rewriter, op, sourceType, destType,
                                       destType.getShape(), *relation.get(),
                                       *relation.get(), MovementEngine::RDMA,
                                       "tile.load");
    if (mlir::failed(descriptors))
      return mlir::failure();

    createMappedRDMADescriptors(rewriter, op.getLoc(), op.getSource(),
                                op.getDest(), *descriptors);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class TileStoreLowering : public mlir::OpRewritePattern<StorageStoreOp> {
public:
  TileStoreLowering(mlir::MLIRContext *context,
                    MovementDescriptorCache *descriptorCache)
      : mlir::OpRewritePattern<StorageStoreOp>(context),
        descriptorCache(descriptorCache) {}

  mlir::LogicalResult
  matchAndRewrite(StorageStoreOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType = mlir::cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(destType.getShape());
    if (!relation.isExact())
      return failPattern(rewriter, op,
                         "tile.store identity relation is not exact");

    auto source = resolveMovementEndpoint(op.getSource());
    if (mlir::failed(source))
      return failPattern(rewriter, op,
                         "tile.store source requires an exact view relation");
    mlir::Value descriptorSource = source->base;
    mlir::MemRefType descriptorSourceType = source->getType();
    const analysis::IndexRelation *descriptorSourceRelation =
        &source->viewToBase;

    if (descriptorSource == op.getSource() &&
        mlir::succeeded(analysis::TransferRealizability::proveCompactDma(
            sourceType, destType, *relation.get()))) {
      mlir::FailureOr<MovementDescriptor> descriptor =
          getStridedTensorDescriptor(rewriter, op, op.getDest().getType(),
                                     "tile.store dest");
      if (mlir::failed(descriptor))
        return mlir::failure();
      createWDMA(rewriter, op.getLoc(), op.getSource(), op.getDest(),
                 *descriptor);
    } else {
      mlir::FailureOr<SharedMovementDescriptorPlan> descriptors =
          descriptorCache->getOrCreate(
              rewriter, op, descriptorSourceType, destType, destType.getShape(),
              *descriptorSourceRelation, *relation.get(), MovementEngine::WDMA,
              "tile.store");
      if (mlir::failed(descriptors))
        return mlir::failure();
      createMappedWDMADescriptors(rewriter, op.getLoc(), descriptorSource,
                                  op.getDest(), **descriptors);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  MovementDescriptorCache *descriptorCache;
};

class LayoutMaterializeLowering
    : public mlir::OpRewritePattern<LayoutMaterializeOp>,
      private ScratchRecorderHolder {
public:
  LayoutMaterializeLowering(mlir::MLIRContext *context,
                            TileRegionToInstrBufferRecorder *bufferRecorder,
                            MovementDescriptorCache *descriptorCache)
      : mlir::OpRewritePattern<LayoutMaterializeOp>(context),
        ScratchRecorderHolder(bufferRecorder),
        descriptorCache(descriptorCache) {}

  mlir::LogicalResult
  matchAndRewrite(LayoutMaterializeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op,
                         "layout materialize lowering requires memref types");

    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!relation.isExact())
      return failPattern(
          rewriter, op,
          "layout materialization gather/scatter is not exactly realizable");

    std::optional<DynamicSubviewDescriptor> dynamicSubview =
        getDynamicSubviewDescriptor(op.getSource(), op);
    mlir::MemRefType descriptorSourceType =
        dynamicSubview ? dynamicSubview->relativeType : sourceType;
    SharedMovementDescriptorPlan descriptorPlan;
    std::optional<llvm::SmallVector<MovementDescriptorPair, 4>> direct =
        getExactTensorToBlockedDescriptors(
            descriptorSourceType, resultType,
            mlir::AffineMap::getMultiDimIdentityMap(resultType.getRank(),
                                                    rewriter.getContext()));
    if (!direct)
      direct =
          getExactBlockedToTensorDescriptors(descriptorSourceType, resultType);
    if (direct) {
      descriptorPlan =
          std::make_shared<MovementDescriptorPlan>(std::move(*direct));
    } else {
      mlir::FailureOr<SharedMovementDescriptorPlan> descriptors =
          descriptorCache->getOrCreate(
              rewriter, op, descriptorSourceType, resultType,
              resultType.getShape(), *relation.get(), *relation.get(),
              MovementEngine::GatherScatter, "layout materialize lowering");
      if (mlir::failed(descriptors))
        return mlir::failure();
      descriptorPlan = *descriptors;
    }

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, bufferRecorder);
    if (mlir::failed(dest))
      return mlir::failure();

    if (dynamicSubview) {
      mlir::Value dynamicSourceOffset = materializeDynamicSubviewByteOffset(
          *dynamicSubview, rewriter, op.getLoc());
      if (!dynamicSourceOffset)
        return mlir::failure();
      if (mlir::failed(emitGatherScatterDescriptorPlan(
              rewriter, op.getLoc(), op, dynamicSubview->sourceBase, *dest,
              *descriptorPlan, bufferRecorder, /*ddrResource=*/{},
              dynamicSourceOffset)))
        return mlir::failure();
    } else {
      if (mlir::failed(emitGatherScatterDescriptorPlan(
              rewriter, op.getLoc(), op, op.getSource(), *dest, *descriptorPlan,
              bufferRecorder)))
        return mlir::failure();
    }
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  MovementDescriptorCache *descriptorCache = nullptr;
};

class TileCopyLowering : public mlir::OpRewritePattern<MoveCopyOp>,
                         private ScratchRecorderHolder {
public:
  TileCopyLowering(mlir::MLIRContext *context,
                   TileRegionToInstrBufferRecorder *bufferRecorder)
      : mlir::OpRewritePattern<MoveCopyOp>(context),
        ScratchRecorderHolder(bufferRecorder) {}

  mlir::LogicalResult
  matchAndRewrite(MoveCopyOp op, mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op,
                         "tile.copy lowering requires memref types");
    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!relation.isExact())
      return failPattern(rewriter, op,
                         "tile.copy identity relation is not exact");
    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(
            rewriter, op, sourceType, resultType, resultType.getShape(),
            *relation.get(), *relation.get(), MovementEngine::GatherScatter,
            "tile.copy lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, bufferRecorder);
    if (mlir::failed(dest))
      return mlir::failure();

    if (mlir::failed(emitGatherScatterDescriptorPlan(
            rewriter, op.getLoc(), op, op.getSource(), *dest, *descriptors,
            bufferRecorder, op.getDdrResourceAttr())))
      return mlir::failure();
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }
};

class TileCopyIntoLowering : public mlir::OpRewritePattern<MoveCopyIntoOp> {
public:
  TileCopyIntoLowering(mlir::MLIRContext *context,
                       TileRegionToInstrBufferRecorder *bufferRecorder,
                       MovementDescriptorCache *descriptorCache)
      : mlir::OpRewritePattern<MoveCopyIntoOp>(context),
        bufferRecorder(bufferRecorder), descriptorCache(descriptorCache) {}

  mlir::LogicalResult
  matchAndRewrite(MoveCopyIntoOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::dyn_cast<mlir::MemRefType>(op.getDest().getType());
    if (!sourceType || !destType)
      return failPattern(rewriter, op,
                         "tile.copy_into lowering requires memref types");
    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(destType.getShape());
    if (!relation.isExact())
      return failPattern(rewriter, op,
                         "tile.copy_into identity relation is not exact");

    auto source = resolveMovementEndpoint(op.getSource());
    if (mlir::failed(source))
      return failPattern(
          rewriter, op,
          "tile.copy_into source requires an exact view relation");
    mlir::Value descriptorDest = op.getDest();
    mlir::MemRefType descriptorDestType = destType;
    const analysis::IndexRelation *descriptorDestRelation = relation.get();
    auto destination = resolveStaticMovementEndpoint(op.getDest());
    std::optional<DynamicSubviewDescriptor> dynamicDest;
    if (mlir::succeeded(destination)) {
      descriptorDest = destination->base;
      descriptorDestType = destination->getType();
      descriptorDestRelation = &destination->viewToBase;
    } else {
      dynamicDest =
          getDynamicSubviewDescriptor(op.getDest(), op.getOperation());
      if (!dynamicDest)
        return failPattern(
            rewriter, op,
            "tile.copy_into subview requires an exact static slice relation "
            "or a supported dynamic Tensor-layout byte offset");
      descriptorDest = dynamicDest->sourceBase;
      descriptorDestType = dynamicDest->relativeType;
    }
    mlir::FailureOr<SharedMovementDescriptorPlan> descriptors =
        descriptorCache->getOrCreate(
            rewriter, op, source->getType(), descriptorDestType,
            destType.getShape(), source->viewToBase, *descriptorDestRelation,
            MovementEngine::GatherScatter, "tile.copy_into lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    if (dynamicDest) {
      mlir::Value dynamicOffset = materializeDynamicSubviewByteOffset(
          *dynamicDest, rewriter, op.getLoc());
      if (!dynamicOffset)
        return mlir::failure();
      if (mlir::failed(emitGatherScatterDescriptorPlan(
              rewriter, op.getLoc(), op, source->base, descriptorDest,
              **descriptors, bufferRecorder, /*ddrResource=*/{}, {},
              dynamicOffset)))
        return mlir::failure();
    } else {
      if (mlir::failed(emitGatherScatterDescriptorPlan(
              rewriter, op.getLoc(), op, source->base, descriptorDest,
              **descriptors, bufferRecorder)))
        return mlir::failure();
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  TileRegionToInstrBufferRecorder *bufferRecorder = nullptr;
  MovementDescriptorCache *descriptorCache;
};

class MemRefCopyLowering : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
public:
  MemRefCopyLowering(mlir::MLIRContext *context,
                     TileRegionToInstrBufferRecorder *bufferRecorder,
                     MovementDescriptorCache *descriptorCache)
      : mlir::OpRewritePattern<mlir::memref::CopyOp>(context),
        bufferRecorder(bufferRecorder), descriptorCache(descriptorCache) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CopyOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    if (!op->getParentOfType<TileRegionOp>())
      return failPattern(
          rewriter, op,
          "memref.copy requires an existing TileRegion movement owner");
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::dyn_cast<mlir::MemRefType>(op.getTarget().getType());
    MemoryAttr sourceMemory =
        sourceType ? getWaferMemoryAttr(sourceType) : MemoryAttr{};
    MemoryAttr destMemory =
        destType ? getWaferMemoryAttr(destType) : MemoryAttr{};
    if (!sourceType || !destType || !sourceMemory || !destMemory ||
        sourceType.getShape() != destType.getShape())
      return failPattern(rewriter, op,
                         "memref.copy requires equal static Wafer memrefs");
    auto sourceView = op.getSource().getDefiningOp<mlir::memref::SubViewOp>();
    auto destView = op.getTarget().getDefiningOp<mlir::memref::SubViewOp>();
    if (op.getSource() == op.getTarget() ||
        (sourceView && destView &&
         mlir::OperationEquivalence::isEquivalentTo(
             sourceView, destView, mlir::OperationEquivalence::exactValueMatch,
             nullptr, mlir::OperationEquivalence::IgnoreLocations))) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(destType.getShape());
    if (!relation.isExact())
      return failPattern(rewriter, op,
                         "memref.copy identity relation is not exact");

    auto source = resolveMovementEndpoint(op.getSource());
    auto destination = resolveMovementEndpoint(op.getTarget());
    if (mlir::failed(source) || mlir::failed(destination))
      return failPattern(rewriter, op,
                         "memref.copy requires exact endpoint relations");

    auto record = [&](mlir::Operation *lowered) {
      if (bufferRecorder)
        bufferRecorder->recordLoweredOperation(op, lowered);
    };
    if (sourceMemory.getSpace() == MemorySpace::DDR &&
        destMemory.getSpace() == MemorySpace::SPM) {
      if (source->base == op.getSource() &&
          destination->base == op.getTarget() &&
          mlir::succeeded(analysis::TransferRealizability::proveCompactDma(
              sourceType, destType, *relation.get()))) {
        auto descriptor = getStridedTensorDescriptor(rewriter, op, sourceType,
                                                     "memref.copy RDMA source");
        if (mlir::failed(descriptor))
          return mlir::failure();
        record(createRDMA(rewriter, op.getLoc(), op.getSource(), op.getTarget(),
                          *descriptor));
      } else {
        auto descriptors = descriptorCache->getOrCreate(
            rewriter, op, source->getType(), destination->getType(),
            destType.getShape(), source->viewToBase, destination->viewToBase,
            MovementEngine::RDMA, "memref.copy RDMA");
        if (mlir::failed(descriptors))
          return mlir::failure();
        for (InstrRDMAOp lowered :
             createMappedRDMADescriptors(rewriter, op.getLoc(), source->base,
                                         destination->base, **descriptors))
          record(lowered);
      }
    } else if (sourceMemory.getSpace() == MemorySpace::SPM &&
               destMemory.getSpace() == MemorySpace::DDR) {
      if (source->base == op.getSource() &&
          destination->base == op.getTarget() &&
          mlir::succeeded(analysis::TransferRealizability::proveCompactDma(
              sourceType, destType, *relation.get()))) {
        auto descriptor = getStridedTensorDescriptor(
            rewriter, op, destType, "memref.copy WDMA destination");
        if (mlir::failed(descriptor))
          return mlir::failure();
        record(createWDMA(rewriter, op.getLoc(), op.getSource(), op.getTarget(),
                          *descriptor));
      } else {
        auto descriptors = descriptorCache->getOrCreate(
            rewriter, op, source->getType(), destination->getType(),
            destType.getShape(), source->viewToBase, destination->viewToBase,
            MovementEngine::WDMA, "memref.copy WDMA");
        if (mlir::failed(descriptors))
          return mlir::failure();
        for (InstrWDMAOp lowered :
             createMappedWDMADescriptors(rewriter, op.getLoc(), source->base,
                                         destination->base, **descriptors))
          record(lowered);
      }
    } else if (sourceMemory.getSpace() == MemorySpace::SPM &&
               destMemory.getSpace() == MemorySpace::SPM) {
      auto descriptors = descriptorCache->getOrCreate(
          rewriter, op, source->getType(), destination->getType(),
          destType.getShape(), source->viewToBase, destination->viewToBase,
          MovementEngine::GatherScatter, "memref.copy SPM");
      if (mlir::failed(descriptors))
        return mlir::failure();
      if (mlir::failed(emitGatherScatterDescriptorPlan(
              rewriter, op.getLoc(), op, source->base, destination->base,
              **descriptors, bufferRecorder)))
        return mlir::failure();
    } else if (sourceMemory.getSpace() == MemorySpace::DDR &&
               destMemory.getSpace() == MemorySpace::DDR) {
      auto stagingType = mlir::MemRefType::get(
          sourceType.getShape(), sourceType.getElementType(),
          mlir::MemRefLayoutAttrInterface{},
          MemoryAttr::get(rewriter.getContext(), MemorySpace::SPM,
                          sourceMemory.getLayout()));
      auto readDescriptors = descriptorCache->getOrCreate(
          rewriter, op, source->getType(), stagingType, stagingType.getShape(),
          source->viewToBase, *relation.get(), MovementEngine::RDMA,
          "memref.copy DDR staging read");
      auto writeDescriptors = descriptorCache->getOrCreate(
          rewriter, op, stagingType, destination->getType(),
          stagingType.getShape(), *relation.get(), destination->viewToBase,
          MovementEngine::WDMA, "memref.copy DDR staging write");
      if (mlir::failed(readDescriptors) || mlir::failed(writeDescriptors))
        return mlir::failure();
      auto emitStagedCopy = [&](mlir::Value source, mlir::Value dest) {
        mlir::FailureOr<mlir::Value> staging = createDestAlloc(
            op.getLoc(), stagingType, rewriter, op, bufferRecorder);
        if (mlir::failed(staging))
          return mlir::failure();
        for (InstrRDMAOp lowered : createMappedRDMADescriptors(
                 rewriter, op.getLoc(), source, *staging, **readDescriptors))
          record(lowered);
        for (InstrWDMAOp lowered : createMappedWDMADescriptors(
                 rewriter, op.getLoc(), *staging, dest, **writeDescriptors))
          record(lowered);
        return mlir::success();
      };
      if (mlir::failed(emitStagedCopy(source->base, destination->base)))
        return mlir::failure();
    } else {
      return failPattern(rewriter, op,
                         "memref.copy has unsupported Wafer memory spaces");
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  TileRegionToInstrBufferRecorder *bufferRecorder = nullptr;
  MovementDescriptorCache *descriptorCache;
};

class MoveExtractSliceLowering
    : public mlir::OpRewritePattern<MoveExtractSliceOp>,
      private ScratchRecorderHolder {
public:
  MoveExtractSliceLowering(mlir::MLIRContext *context,
                           TileRegionToInstrBufferRecorder *bufferRecorder)
      : mlir::OpRewritePattern<MoveExtractSliceOp>(context),
        ScratchRecorderHolder(bufferRecorder) {}

  mlir::LogicalResult
  matchAndRewrite(MoveExtractSliceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op,
                         "tile.extract_slice lowering requires memref types");

    llvm::ArrayRef<int64_t> offsets = op.getOffsets();
    llvm::ArrayRef<int64_t> sizes = op.getSizes();
    llvm::ArrayRef<int64_t> strides = op.getStrides();
    llvm::ArrayRef<int64_t> resultShape = resultType.getShape();
    std::optional<llvm::SmallDenseSet<unsigned>> rankReductionMask =
        mlir::computeRankReductionMask(sizes, resultShape);
    if (!rankReductionMask)
      return failPattern(rewriter, op,
                         "tile.extract_slice cannot map rank reduction");
    llvm::SmallVector<mlir::AffineExpr, 4> sourceResults;
    sourceResults.reserve(sourceType.getRank());
    unsigned reducedDim = 0;
    for (unsigned fullDim = 0; fullDim < sizes.size(); ++fullDim) {
      mlir::AffineExpr expression =
          mlir::getAffineConstantExpr(offsets[fullDim], rewriter.getContext());
      if (!rankReductionMask->contains(fullDim)) {
        expression = expression + mlir::getAffineDimExpr(
                                      reducedDim++, rewriter.getContext()) *
                                      strides[fullDim];
      }
      sourceResults.push_back(expression);
    }
    analysis::IndexRelationResult sourceRelation =
        analysis::IndexRelation::fromAffineMap(
            mlir::AffineMap::get(resultShape.size(), 0, sourceResults,
                                 rewriter.getContext()),
            resultShape, sourceType.getShape());
    analysis::IndexRelationResult destRelation =
        analysis::IndexRelation::identity(resultShape);
    if (!sourceRelation.isExact() || !destRelation.isExact())
      return failPattern(rewriter, op,
                         "tile.extract_slice relation is not exact");

    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(
            rewriter, op, sourceType, resultType, resultShape,
            *sourceRelation.get(), *destRelation.get(),
            MovementEngine::GatherScatter, "tile.extract_slice lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, bufferRecorder);
    if (mlir::failed(dest))
      return mlir::failure();

    if (mlir::failed(emitGatherScatterDescriptorPlan(
            rewriter, op.getLoc(), op, op.getSource(), *dest, *descriptors,
            bufferRecorder)))
      return mlir::failure();
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }
};

class MoveInsertSliceLowering
    : public mlir::OpRewritePattern<MoveInsertSliceOp>,
      private ScratchRecorderHolder {
public:
  MoveInsertSliceLowering(mlir::MLIRContext *context,
                          TileRegionToInstrBufferRecorder *bufferRecorder,
                          MovementDescriptorCache *descriptorCache)
      : mlir::OpRewritePattern<MoveInsertSliceOp>(context),
        ScratchRecorderHolder(bufferRecorder),
        descriptorCache(descriptorCache) {}

  mlir::LogicalResult
  matchAndRewrite(MoveInsertSliceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::dyn_cast<mlir::MemRefType>(op.getDest().getType());
    if (!sourceType || !destType)
      return failPattern(rewriter, op,
                         "tile.insert_slice lowering requires memref types");

    llvm::ArrayRef<int64_t> offsets = op.getOffsets();
    llvm::ArrayRef<int64_t> sizes = op.getSizes();
    llvm::ArrayRef<int64_t> strides = op.getStrides();
    llvm::ArrayRef<int64_t> sourceShape = sourceType.getShape();
    std::optional<llvm::SmallDenseSet<unsigned>> rankReductionMask =
        mlir::computeRankReductionMask(sizes, sourceShape);
    if (!rankReductionMask)
      return failPattern(rewriter, op,
                         "tile.insert_slice cannot map rank reduction");
    llvm::SmallVector<mlir::AffineExpr, 4> destResults;
    destResults.reserve(destType.getRank());
    unsigned reducedDim = 0;
    for (unsigned fullDim = 0; fullDim < sizes.size(); ++fullDim) {
      mlir::AffineExpr expression =
          mlir::getAffineConstantExpr(offsets[fullDim], rewriter.getContext());
      if (!rankReductionMask->contains(fullDim)) {
        expression = expression + mlir::getAffineDimExpr(
                                      reducedDim++, rewriter.getContext()) *
                                      strides[fullDim];
      }
      destResults.push_back(expression);
    }
    analysis::IndexRelationResult sourceRelation =
        analysis::IndexRelation::identity(sourceShape);
    analysis::IndexRelationResult destRelation =
        analysis::IndexRelation::fromAffineMap(
            mlir::AffineMap::get(sourceShape.size(), 0, destResults,
                                 rewriter.getContext()),
            sourceShape, destType.getShape());
    if (!sourceRelation.isExact() || !destRelation.isExact())
      return failPattern(rewriter, op,
                         "tile.insert_slice relation is not exact");
    mlir::FailureOr<SharedMovementDescriptorPlan> insertDescriptors =
        descriptorCache->getOrCreate(
            rewriter, op, sourceType, destType, sourceShape,
            *sourceRelation.get(), *destRelation.get(),
            MovementEngine::GatherScatter, "tile.insert_slice lowering");
    if (mlir::failed(insertDescriptors))
      return mlir::failure();

    if (mlir::failed(emitGatherScatterDescriptorPlan(
            rewriter, op.getLoc(), op, op.getSource(), op.getDest(),
            **insertDescriptors, bufferRecorder, op.getDdrResourceAttr())))
      return mlir::failure();
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  MovementDescriptorCache *descriptorCache = nullptr;
};

class MoveTransposeLowering : public mlir::OpRewritePattern<MoveTransposeOp>,
                              private ScratchRecorderHolder {
public:
  MoveTransposeLowering(mlir::MLIRContext *context,
                        TileRegionToInstrBufferRecorder *bufferRecorder)
      : mlir::OpRewritePattern<MoveTransposeOp>(context),
        ScratchRecorderHolder(bufferRecorder) {}

  mlir::LogicalResult
  matchAndRewrite(MoveTransposeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op,
                         "tile.transpose lowering requires memref types");

    llvm::ArrayRef<int64_t> permutation = op.getPermutation();
    if (permutation.size() != static_cast<size_t>(sourceType.getRank()))
      return failPattern(rewriter, op,
                         "tile.transpose permutation rank mismatch");
    llvm::SmallVector<mlir::AffineExpr, 4> sourceResults(sourceType.getRank());
    llvm::SmallVector<bool, 4> seenSourceDims(sourceType.getRank(), false);
    for (auto [resultDim, sourceDim] : llvm::enumerate(permutation)) {
      if (sourceDim < 0 || sourceDim >= sourceType.getRank() ||
          seenSourceDims[sourceDim])
        return failPattern(rewriter, op,
                           "tile.transpose permutation is invalid");
      seenSourceDims[sourceDim] = true;
      sourceResults[sourceDim] =
          mlir::getAffineDimExpr(resultDim, rewriter.getContext());
    }
    analysis::IndexRelationResult sourceRelation =
        analysis::IndexRelation::fromAffineMap(
            mlir::AffineMap::get(resultType.getRank(), 0, sourceResults,
                                 rewriter.getContext()),
            resultType.getShape(), sourceType.getShape());
    analysis::IndexRelationResult destRelation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!sourceRelation.isExact() || !destRelation.isExact())
      return failPattern(rewriter, op, "tile.transpose relation is not exact");

    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(
            rewriter, op, sourceType, resultType, resultType.getShape(),
            *sourceRelation.get(), *destRelation.get(),
            MovementEngine::GatherScatter, "tile.transpose lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, bufferRecorder);
    if (mlir::failed(dest))
      return mlir::failure();

    if (mlir::failed(emitGatherScatterDescriptorPlan(
            rewriter, op.getLoc(), op, op.getSource(), *dest, *descriptors,
            bufferRecorder)))
      return mlir::failure();
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }
};

class InstrTDMADataMoveLowering
    : public mlir::OpRewritePattern<InstrTDMADataMoveOp> {
public:
  InstrTDMADataMoveLowering(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<InstrTDMADataMoveOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(InstrTDMADataMoveOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    InstrDataMoveKind kind = op.getKindAttr().getValue();
    if (!requiresGatherScatterMaterialization(kind))
      return mlir::failure();

    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::dyn_cast<mlir::MemRefType>(op.getDest().getType());
    if (!sourceType || !destType)
      return failPattern(rewriter, op,
                         "tdma_data_move lowering requires memref operands");
    if (mlir::failed(verifyStaticShapeAttrMatchesMemRef(
            rewriter, op, sourceType, op.getSourceShapeAttr(), "source",
            "tdma_data_move lowering")) ||
        mlir::failed(verifyStaticShapeAttrMatchesMemRef(
            rewriter, op, destType, op.getDestShapeAttr(), "dest",
            "tdma_data_move lowering")))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        lowerToDescriptors(op, sourceType, destType, kind, rewriter);
    if (mlir::failed(descriptors))
      return mlir::failure();

    if (mlir::failed(emitGatherScatterDescriptorPlan(
            rewriter, op.getLoc(), op, op.getSource(), op.getDest(),
            *descriptors)))
      return mlir::failure();
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>>
  lowerToDescriptors(InstrTDMADataMoveOp op, mlir::MemRefType sourceType,
                     mlir::MemRefType destType, InstrDataMoveKind kind,
                     mlir::PatternRewriter &rewriter) const {
    llvm::SmallVector<mlir::AffineExpr, 4> sourceResults(sourceType.getRank());
    auto setPermutation = [&](llvm::ArrayRef<int64_t> permutation) {
      if (sourceType.getRank() != destType.getRank() ||
          permutation.size() != static_cast<size_t>(sourceType.getRank()))
        return false;
      llvm::SmallVector<bool, 4> seen(sourceType.getRank(), false);
      for (auto [destDim, sourceDim] : llvm::enumerate(permutation)) {
        if (sourceDim < 0 || sourceDim >= sourceType.getRank() ||
            seen[sourceDim])
          return false;
        seen[sourceDim] = true;
        sourceResults[sourceDim] =
            mlir::getAffineDimExpr(destDim, rewriter.getContext());
      }
      return true;
    };

    llvm::StringRef opLabel = "tdma_data_move lowering";
    switch (kind) {
    case InstrDataMoveKind::Transpose:
      if (!op.getPermutationAttr())
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op,
            "tdma_data_move transpose lowering requires permutation attr");
      if (!setPermutation(op.getPermutationAttr().asArrayRef()))
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op,
            "tdma_data_move transpose lowering requires a valid permutation");
      opLabel = "tdma_data_move transpose lowering";
      break;
    case InstrDataMoveKind::Nchw2Nhwc:
      if (!setPermutation(kNchw2NhwcPermutation))
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op,
            "tdma_data_move nchw2nhwc lowering requires rank four");
      opLabel = "tdma_data_move nchw2nhwc lowering";
      break;
    case InstrDataMoveKind::Nhwc2Nchw:
      if (!setPermutation(kNhwc2NchwPermutation))
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op,
            "tdma_data_move nhwc2nchw lowering requires rank four");
      opLabel = "tdma_data_move nhwc2nchw lowering";
      break;
    case InstrDataMoveKind::TensorNom:
      if (!setPermutation(
              llvm::to_vector(llvm::seq<int64_t>(0, sourceType.getRank()))))
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op,
            "tdma_data_move tensor_nom lowering requires equal ranks");
      opLabel = "tdma_data_move tensor_nom lowering";
      break;
    case InstrDataMoveKind::Mirror:
      if (!op.getAxesAttr())
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op, "tdma_data_move mirror lowering requires axes attr");
      if (!setPermutation(
              llvm::to_vector(llvm::seq<int64_t>(0, sourceType.getRank()))))
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op,
            "tdma_data_move mirror lowering requires equal ranks");
      for (int64_t axis : op.getAxesAttr().asArrayRef()) {
        if (axis < 0 || axis >= sourceType.getRank())
          return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
              rewriter, op,
              "tdma_data_move mirror lowering axis is out of range");
        sourceResults[axis] =
            mlir::getAffineConstantExpr(sourceType.getDimSize(axis) - 1,
                                        rewriter.getContext()) -
            mlir::getAffineDimExpr(axis, rewriter.getContext());
      }
      opLabel = "tdma_data_move mirror lowering";
      break;
    case InstrDataMoveKind::Rotate90:
    case InstrDataMoveKind::Rotate180:
    case InstrDataMoveKind::Rotate270:
      if (!op.getAxesAttr())
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op, "tdma_data_move rotate lowering requires axes attr");
      if (!setPermutation(
              llvm::to_vector(llvm::seq<int64_t>(0, sourceType.getRank()))) ||
          op.getAxesAttr().size() != 2)
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op,
            "tdma_data_move rotate lowering requires equal ranks and two axes");
      {
        int64_t axis0 = op.getAxesAttr().asArrayRef()[0];
        int64_t axis1 = op.getAxesAttr().asArrayRef()[1];
        if (axis0 < 0 || axis1 < 0 || axis0 >= sourceType.getRank() ||
            axis1 >= sourceType.getRank() || axis0 == axis1)
          return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
              rewriter, op, "tdma_data_move rotate lowering axes are invalid");
        mlir::AffineExpr d0 =
            mlir::getAffineDimExpr(axis0, rewriter.getContext());
        mlir::AffineExpr d1 =
            mlir::getAffineDimExpr(axis1, rewriter.getContext());
        if (kind == InstrDataMoveKind::Rotate90) {
          sourceResults[axis0] =
              mlir::getAffineConstantExpr(sourceType.getDimSize(axis0) - 1,
                                          rewriter.getContext()) -
              d1;
          sourceResults[axis1] = d0;
        } else if (kind == InstrDataMoveKind::Rotate180) {
          sourceResults[axis0] =
              mlir::getAffineConstantExpr(sourceType.getDimSize(axis0) - 1,
                                          rewriter.getContext()) -
              d0;
          sourceResults[axis1] =
              mlir::getAffineConstantExpr(sourceType.getDimSize(axis1) - 1,
                                          rewriter.getContext()) -
              d1;
        } else {
          sourceResults[axis0] = d1;
          sourceResults[axis1] =
              mlir::getAffineConstantExpr(sourceType.getDimSize(axis1) - 1,
                                          rewriter.getContext()) -
              d0;
        }
      }
      opLabel = "tdma_data_move rotate lowering";
      break;
    case InstrDataMoveKind::Pad:
    case InstrDataMoveKind::Img2Col:
      return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
          rewriter, op,
          "tdma_data_move pad/img2col remains in the production target "
          "surface");
    }

    analysis::IndexRelationResult sourceRelation =
        analysis::IndexRelation::fromAffineMap(
            mlir::AffineMap::get(destType.getRank(), 0, sourceResults,
                                 rewriter.getContext()),
            destType.getShape(), sourceType.getShape());
    analysis::IndexRelationResult destRelation =
        analysis::IndexRelation::identity(destType.getShape());
    if (!sourceRelation.isExact() || !destRelation.isExact())
      return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
          rewriter, op,
          llvm::Twine(opLabel).concat(" relation is not exact").str());
    return getRelationMovementDescriptors(
        rewriter, op, sourceType, destType, destType.getShape(),
        *sourceRelation.get(), *destRelation.get(),
        MovementEngine::GatherScatter, opLabel);
  }
};

class MoveBroadcastLowering : public mlir::OpRewritePattern<MoveBroadcastOp>,
                              private ScratchRecorderHolder {
public:
  MoveBroadcastLowering(mlir::MLIRContext *context,
                        TileRegionToInstrBufferRecorder *bufferRecorder)
      : mlir::OpRewritePattern<MoveBroadcastOp>(context),
        ScratchRecorderHolder(bufferRecorder) {}

  mlir::LogicalResult
  matchAndRewrite(MoveBroadcastOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op,
                         "tile.broadcast lowering requires memref types");

    llvm::ArrayRef<int64_t> dimensions = op.getDimensions();
    if (dimensions.size() != static_cast<size_t>(sourceType.getRank()))
      return failPattern(rewriter, op,
                         "tile.broadcast dimension rank mismatch");
    llvm::SmallVector<mlir::AffineExpr, 4> sourceResults;
    sourceResults.reserve(sourceType.getRank());
    llvm::SmallVector<bool, 4> usedResultDims(resultType.getRank(), false);
    for (int64_t resultDim : dimensions) {
      if (resultDim < 0 || resultDim >= resultType.getRank() ||
          usedResultDims[resultDim])
        return failPattern(rewriter, op,
                           "tile.broadcast dimensions are invalid");
      usedResultDims[resultDim] = true;
      sourceResults.push_back(
          mlir::getAffineDimExpr(resultDim, rewriter.getContext()));
    }

    // A broadcast that only inserts unit dimensions is an exact reshape. Keep
    // it as a metadata view when both physical layouts describe the same
    // storage instead of manufacturing a gather/scatter movement.
    if (sourceType.getNumElements() == resultType.getNumElements() &&
        mlir::succeeded(
            analysis::TransferRealizability::proveStaticReshapeMetadataView(
                sourceType, resultType,
                /*destinationMayWrite=*/true))) {
      llvm::SmallVector<int64_t> sizes(resultType.getShape().begin(),
                                       resultType.getShape().end());
      mlir::FailureOr<llvm::SmallVector<int64_t>> strides =
          getStaticCompactStrides(rewriter, op, resultType);
      if (mlir::failed(strides))
        return mlir::failure();
      auto view = rewriter.create<mlir::memref::ReinterpretCastOp>(
          op.getLoc(), resultType, op.getSource(), /*offset=*/0, sizes,
          *strides);
      rewriter.replaceOp(op, view.getResult());
      return mlir::success();
    }

    analysis::IndexRelationResult sourceRelation =
        analysis::IndexRelation::fromAffineMap(
            mlir::AffineMap::get(resultType.getRank(), 0, sourceResults,
                                 rewriter.getContext()),
            resultType.getShape(), sourceType.getShape());
    analysis::IndexRelationResult destRelation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!sourceRelation.isExact() || !destRelation.isExact())
      return failPattern(rewriter, op, "tile.broadcast relation is not exact");
    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(
            rewriter, op, sourceType, resultType, resultType.getShape(),
            *sourceRelation.get(), *destRelation.get(),
            MovementEngine::GatherScatter, "tile.broadcast lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, bufferRecorder);
    if (mlir::failed(dest))
      return mlir::failure();

    if (mlir::failed(emitGatherScatterDescriptorPlan(
            rewriter, op.getLoc(), op, op.getSource(), *dest, *descriptors,
            bufferRecorder)))
      return mlir::failure();
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }
};
class ViewReshapeLowering : public mlir::OpRewritePattern<ViewReshapeOp> {
public:
  ViewReshapeLowering(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<ViewReshapeOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(ViewReshapeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    if (op.getSource().getType() == op.getResult().getType()) {
      rewriter.replaceOp(op, op.getSource());
      return mlir::success();
    }

    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    if (!sourceType)
      return failPattern(rewriter, op,
                         "tile.reshape lowering requires memref source type");
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!resultType)
      return failPattern(rewriter, op,
                         "tile.reshape lowering requires memref result type");

    MemoryAttr sourceMemory = wafer::getWaferMemoryAttr(sourceType);
    MemoryAttr resultMemory = wafer::getWaferMemoryAttr(resultType);
    if (!sourceMemory || !resultMemory)
      return failPattern(rewriter, op,
                         "tile.reshape lowering requires Wafer memref types");

    analysis::IndexRelationResult relation =
        analysis::IndexRelation::staticReshape(resultType.getShape(),
                                               sourceType.getShape());
    if (!relation.isExact())
      return failPattern(rewriter, op,
                         "tile.reshape requires an exact index relation");
    if (mlir::succeeded(
            analysis::TransferRealizability::proveStaticReshapeMetadataView(
                sourceType, resultType,
                /*destinationMayWrite=*/true))) {
      llvm::SmallVector<int64_t> sizes(resultType.getShape().begin(),
                                       resultType.getShape().end());
      mlir::FailureOr<llvm::SmallVector<int64_t>> strides =
          getStaticCompactStrides(rewriter, op, resultType);
      if (mlir::failed(strides))
        return mlir::failure();

      auto view = rewriter.create<mlir::memref::ReinterpretCastOp>(
          op.getLoc(), resultType, op.getSource(), /*offset=*/0, sizes,
          *strides);
      rewriter.replaceOp(op, view.getResult());
      return mlir::success();
    }
    return failPattern(
        rewriter, op,
        "tile.reshape is an alias view but the selected physical layouts do "
        "not preserve an identical element mapping; materialize movement "
        "before reshape");
  }
};

class MoveReshapeLowering : public mlir::OpRewritePattern<MoveReshapeOp>,
                            private ScratchRecorderHolder {
public:
  MoveReshapeLowering(mlir::MLIRContext *context,
                      TileRegionToInstrBufferRecorder *bufferRecorder)
      : mlir::OpRewritePattern<MoveReshapeOp>(context),
        ScratchRecorderHolder(bufferRecorder) {}

  mlir::LogicalResult
  matchAndRewrite(MoveReshapeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op,
                         "tile.reshape_copy lowering requires memref types");

    std::optional<CanonicalReshapeMovementRelations> movementRelations =
        getCanonicalReshapeMovementRelations(rewriter.getContext(),
                                             sourceType.getShape(),
                                             resultType.getShape());
    if (!movementRelations)
      return failPattern(
          rewriter, op,
          "tile.reshape_copy canonical relation cannot be represented by a "
          "rectangular affine refinement");
    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(rewriter, op, sourceType, resultType,
                                       movementRelations->iterationShape,
                                       movementRelations->iterationToSource,
                                       movementRelations->iterationToDest,
                                       MovementEngine::GatherScatter,
                                       "tile.reshape_copy lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, bufferRecorder);
    if (mlir::failed(dest))
      return mlir::failure();
    if (mlir::failed(emitGatherScatterDescriptorPlan(
            rewriter, op.getLoc(), op, op.getSource(), *dest, *descriptors,
            bufferRecorder)))
      return mlir::failure();
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }
};

} // namespace

void wafer::tile_region_to_instr::populateMovementLoweringPatterns(
    mlir::RewritePatternSet &patterns,
    TileRegionToInstrBufferRecorder *bufferRecorder,
    MovementDescriptorCache *descriptorCache) {
  mlir::MLIRContext *context = patterns.getContext();
  patterns.add<TileLoadLowering, InstrTDMADataMoveLowering>(context);
  patterns.add<TileStoreLowering>(context, descriptorCache);
  patterns.add<TileCopyIntoLowering>(context, bufferRecorder, descriptorCache);
  patterns.add<MemRefCopyLowering>(context, bufferRecorder, descriptorCache);
  patterns.add<MoveInsertSliceLowering>(context, bufferRecorder,
                                        descriptorCache);
  patterns.add<TileCopyLowering, MoveExtractSliceLowering, MoveReshapeLowering,
               MoveTransposeLowering, MoveBroadcastLowering>(context,
                                                             bufferRecorder);
  patterns.add<LayoutMaterializeLowering>(context, bufferRecorder,
                                          descriptorCache);
}

void wafer::tile_region_to_instr::populateViewReshapeLoweringPattern(
    mlir::RewritePatternSet &patterns) {
  patterns.add<ViewReshapeLowering>(patterns.getContext());
}
