//===- WaferTileRegionToInstr.cpp - Tile-region to instr conversion ------===//

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>

using namespace wafer;

namespace wafer {
#define GEN_PASS_DEF_CONVERTTILEREGIONTOINSTRPASS
#include "Wafer/Transforms/WaferPasses.h.inc"
} // namespace wafer

namespace {

struct MovementDescriptor {
  int64_t byteCount = 0;
  int64_t innerBytes = 0;
  int64_t byteOffset = 0;
  llvm::SmallVector<int64_t, 3> strides;
  llvm::SmallVector<int64_t, 3> iterations;
};

struct LogicalMovementSegment {
  int64_t sourceOffset = 0;
  int64_t destOffset = 0;
  int64_t bytes = 0;
};

static std::optional<int64_t>
getStaticPositiveElementCount(llvm::ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    if (dim == mlir::ShapedType::kDynamic || dim <= 0)
      return std::nullopt;
    if (count > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    count *= dim;
  }
  return count;
}

static void setFailureReason(std::string *failureReason,
                             llvm::StringRef reason) {
  if (failureReason)
    *failureReason = reason.str();
}

static mlir::LogicalResult failPattern(mlir::PatternRewriter &rewriter,
                                       mlir::Operation *op,
                                       std::string *failureReason,
                                       llvm::StringRef reason) {
  setFailureReason(failureReason, reason);
  return rewriter.notifyMatchFailure(op, reason);
}

template <typename T>
static mlir::FailureOr<T>
failFailureOr(mlir::PatternRewriter &rewriter, mlir::Operation *op,
              std::string *failureReason, llvm::StringRef reason) {
  (void)failPattern(rewriter, op, failureReason, reason);
  return mlir::failure();
}

static std::optional<mlir::RankedTensorType>
getLogicalTensorTypeFromMemRef(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return std::nullopt;
  return mlir::RankedTensorType::get(memrefType.getShape(),
                                     memrefType.getElementType());
}

static mlir::FailureOr<mlir::Value>
createDestAlloc(mlir::Location loc, mlir::Type type,
                mlir::PatternRewriter &rewriter, mlir::Operation *op,
                std::string *failureReason) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<mlir::Value>(
        rewriter, op, failureReason,
        "tile-region to instr lowering requires memref result storage");
  return rewriter.create<mlir::memref::AllocOp>(loc, memrefType).getResult();
}

static mlir::FailureOr<MovementDescriptor>
getContiguousDescriptor(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                        mlir::Type type, std::string *failureReason) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
        "instruction descriptor requires a Wafer memref type");

  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes <= 0)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
        "instruction descriptor requires static positive physical byte size");

  MovementDescriptor descriptor;
  descriptor.byteCount = info->physicalBytes;
  descriptor.innerBytes = info->physicalBytes;
  descriptor.strides.assign({0, 0, 0});
  descriptor.iterations.assign({1, 1, 1});
  return descriptor;
}

static void createRDMA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                       mlir::Value source, mlir::Value dest,
                       const MovementDescriptor &descriptor) {
  rewriter.create<InstrRDMAOp>(loc, source, dest, descriptor.byteCount,
                               descriptor.innerBytes, descriptor.strides,
                               descriptor.iterations);
}

static void createWDMA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                       mlir::Value source, mlir::Value dest,
                       const MovementDescriptor &descriptor) {
  rewriter.create<InstrWDMAOp>(loc, source, dest, descriptor.byteCount,
                               descriptor.innerBytes, descriptor.strides,
                               descriptor.iterations);
}

static void createGatherScatter(mlir::PatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value source,
                                mlir::Value dest,
                                const MovementDescriptor &sourceDescriptor,
                                const MovementDescriptor &destDescriptor) {
  mlir::IntegerAttr sourceOffset =
      sourceDescriptor.byteOffset == 0
          ? mlir::IntegerAttr{}
          : rewriter.getI64IntegerAttr(sourceDescriptor.byteOffset);
  mlir::IntegerAttr destOffset =
      destDescriptor.byteOffset == 0
          ? mlir::IntegerAttr{}
          : rewriter.getI64IntegerAttr(destDescriptor.byteOffset);
  rewriter.create<InstrGatherScatterOp>(
      loc, source, dest, destDescriptor.byteCount, destDescriptor.innerBytes,
      sourceOffset, destOffset, sourceDescriptor.strides,
      sourceDescriptor.iterations, destDescriptor.strides,
      destDescriptor.iterations);
}

static void
createGatherScatterSegments(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value source, mlir::Value dest,
                            llvm::ArrayRef<LogicalMovementSegment> segments) {
  for (const LogicalMovementSegment &segment : segments) {
    MovementDescriptor sourceDescriptor;
    sourceDescriptor.byteCount = segment.bytes;
    sourceDescriptor.innerBytes = segment.bytes;
    sourceDescriptor.byteOffset = segment.sourceOffset;
    sourceDescriptor.strides.assign({0, 0, 0});
    sourceDescriptor.iterations.assign({1, 1, 1});

    MovementDescriptor destDescriptor;
    destDescriptor.byteCount = segment.bytes;
    destDescriptor.innerBytes = segment.bytes;
    destDescriptor.byteOffset = segment.destOffset;
    destDescriptor.strides.assign({0, 0, 0});
    destDescriptor.iterations.assign({1, 1, 1});

    createGatherScatter(rewriter, loc, source, dest, sourceDescriptor,
                        destDescriptor);
  }
}

static void copyOptionalAttr(mlir::Operation *from, mlir::Operation *to,
                             llvm::StringRef name) {
  if (mlir::Attribute attr = from->getAttr(name))
    to->setAttr(name, attr);
}

static mlir::IntegerAttr getI64Attr(mlir::PatternRewriter &rewriter,
                                    int64_t value) {
  return rewriter.getI64IntegerAttr(value);
}

static mlir::FailureOr<int64_t>
getStaticPhysicalBytes(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                       mlir::Type type, std::string *failureReason,
                       llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        llvm::Twine(role).concat(" must be a Wafer memref").str());
  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes <= 0)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        llvm::Twine(role)
            .concat(" requires static positive physical byte size")
            .str());
  return info->physicalBytes;
}

static mlir::FailureOr<llvm::SmallVector<int64_t>>
getStaticCompactStrides(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                        mlir::MemRefType type, std::string *failureReason) {
  llvm::SmallVector<int64_t> strides(type.getRank(), 1);
  int64_t runningStride = 1;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    strides[dim] = runningStride;
    int64_t size = type.getDimSize(dim);
    if (size == mlir::ShapedType::kDynamic)
      return failFailureOr<llvm::SmallVector<int64_t>>(
          rewriter, op, failureReason,
          "tile.reshape lowering requires static result shape");
    runningStride *= size;
  }
  return strides;
}

static bool isStandardViewCompatibleLayout(MemLayout layout) {
  return layout == MemLayout::Tensor || layout == MemLayout::NTensor;
}

static mlir::FailureOr<llvm::SmallVector<int64_t>>
delinearizeIndex(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                 llvm::ArrayRef<int64_t> shape, int64_t linearIndex,
                 std::string *failureReason, llvm::StringRef opLabel) {
  llvm::SmallVector<int64_t> indices(shape.size(), 0);
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    int64_t size = shape[dim];
    if (size == mlir::ShapedType::kDynamic || size <= 0)
      return failFailureOr<llvm::SmallVector<int64_t>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel).concat(" requires static positive shape").str());
    indices[dim] = linearIndex % size;
    linearIndex /= size;
  }
  if (linearIndex != 0)
    return failFailureOr<llvm::SmallVector<int64_t>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel).concat(" cannot delinearize logical index").str());
  return indices;
}

template <typename SourceIndexFn, typename DestIndexFn>
static mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getStaticMappedMovementSegments(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    llvm::ArrayRef<int64_t> iterationShape, SourceIndexFn sourceIndexFn,
    DestIndexFn destIndexFn, std::string *failureReason,
    llvm::StringRef opLabel) {
  std::optional<WaferPhysicalTensorInfo> sourceInfo =
      wafer::computeWaferPhysicalTensorInfo(sourceType);
  std::optional<WaferPhysicalTensorInfo> destInfo =
      wafer::computeWaferPhysicalTensorInfo(destType);
  if (!sourceInfo || !destInfo || sourceInfo->physicalBytes <= 0 ||
      destInfo->physicalBytes <= 0)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires static positive physical byte sizes")
            .str());
  if (sourceInfo->elementBytes <= 0 ||
      sourceInfo->elementBytes != destInfo->elementBytes ||
      sourceInfo->bitPackedElement || destInfo->bitPackedElement)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires byte-addressable elements")
            .str());

  std::optional<int64_t> elementCount =
      getStaticPositiveElementCount(iterationShape);
  if (!elementCount)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires static positive iteration shape")
            .str());

  int64_t elementBytes = sourceInfo->elementBytes;
  llvm::SmallVector<LogicalMovementSegment> segments;
  for (int64_t linearIndex = 0; linearIndex < *elementCount; ++linearIndex) {
    mlir::FailureOr<llvm::SmallVector<int64_t>> iterationIndices =
        delinearizeIndex(rewriter, op, iterationShape, linearIndex,
                         failureReason, opLabel);
    if (mlir::failed(iterationIndices))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<int64_t>> sourceIndices =
        sourceIndexFn(*iterationIndices);
    mlir::FailureOr<llvm::SmallVector<int64_t>> destIndices =
        destIndexFn(*iterationIndices);
    if (mlir::failed(sourceIndices) || mlir::failed(destIndices))
      return mlir::failure();

    std::optional<int64_t> sourceOffset =
        wafer::computeWaferPhysicalElementByteOffset(sourceType,
                                                     *sourceIndices);
    std::optional<int64_t> destOffset =
        wafer::computeWaferPhysicalElementByteOffset(destType, *destIndices);
    if (!sourceOffset || !destOffset)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" cannot compute physical element offset")
              .str());
    if (*sourceOffset + elementBytes > sourceInfo->physicalBytes ||
        *destOffset + elementBytes > destInfo->physicalBytes)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" segment exceeds static physical byte range")
              .str());

    if (!segments.empty()) {
      LogicalMovementSegment &last = segments.back();
      if (last.sourceOffset + last.bytes == *sourceOffset &&
          last.destOffset + last.bytes == *destOffset) {
        last.bytes += elementBytes;
        continue;
      }
    }
    segments.push_back({*sourceOffset, *destOffset, elementBytes});
  }

  return segments;
}

static mlir::FailureOr<llvm::SmallVector<int64_t>>
expandRankReducedSliceIndices(mlir::PatternRewriter &rewriter,
                              mlir::Operation *op,
                              llvm::ArrayRef<int64_t> fullShape,
                              llvm::ArrayRef<int64_t> reducedShape,
                              llvm::ArrayRef<int64_t> reducedIndices,
                              std::string *failureReason,
                              llvm::StringRef opLabel) {
  std::optional<llvm::SmallDenseSet<unsigned>> rankReductionMask =
      mlir::computeRankReductionMask(fullShape, reducedShape);
  if (!rankReductionMask)
    return failFailureOr<llvm::SmallVector<int64_t>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel).concat(" cannot map rank-reduced slice").str());
  if (reducedShape.size() != reducedIndices.size())
    return failFailureOr<llvm::SmallVector<int64_t>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel).concat(" has mismatched reduced indices").str());

  llvm::SmallVector<int64_t> fullIndices(fullShape.size(), 0);
  size_t reducedDim = 0;
  for (unsigned fullDim = 0; fullDim < fullShape.size(); ++fullDim) {
    if (rankReductionMask->contains(fullDim)) {
      fullIndices[fullDim] = 0;
      continue;
    }
    if (reducedDim >= reducedIndices.size())
      return failFailureOr<llvm::SmallVector<int64_t>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel).concat(" has incomplete slice index").str());
    fullIndices[fullDim] = reducedIndices[reducedDim++];
  }
  if (reducedDim != reducedIndices.size())
    return failFailureOr<llvm::SmallVector<int64_t>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel).concat(" has unused reduced slice index").str());
  return fullIndices;
}

static mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getStaticLogicalMovementSegments(mlir::PatternRewriter &rewriter,
                                 mlir::Operation *op,
                                 mlir::MemRefType sourceType,
                                 mlir::MemRefType destType,
                                 std::string *failureReason,
                                 llvm::StringRef opLabel) {
  std::optional<WaferPhysicalTensorInfo> sourceInfo =
      wafer::computeWaferPhysicalTensorInfo(sourceType);
  std::optional<WaferPhysicalTensorInfo> destInfo =
      wafer::computeWaferPhysicalTensorInfo(destType);
  if (!sourceInfo || !destInfo || sourceInfo->compactBytes <= 0 ||
      destInfo->compactBytes <= 0 || sourceInfo->physicalBytes <= 0 ||
      destInfo->physicalBytes <= 0)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires static positive byte sizes")
            .str());
  if (sourceInfo->compactBytes != destInfo->compactBytes)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires equal static compact byte counts")
            .str());
  if (sourceInfo->elementBytes <= 0 ||
      sourceInfo->elementBytes != destInfo->elementBytes ||
      sourceInfo->bitPackedElement || destInfo->bitPackedElement ||
      sourceInfo->compactBytes % sourceInfo->elementBytes != 0)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires byte-addressable elements")
            .str());

  int64_t elementBytes = sourceInfo->elementBytes;
  int64_t elementCount = sourceInfo->compactBytes / elementBytes;

  // Reshape preserves canonical logical linear order, not per-dimension index
  // equality.  Delinearize the same logical element number through the source
  // and destination shapes, then ask the physical layout helper where that
  // logical element lives in each buffer.
  llvm::SmallVector<LogicalMovementSegment> segments;
  for (int64_t linearIndex = 0; linearIndex < elementCount; ++linearIndex) {
    mlir::FailureOr<llvm::SmallVector<int64_t>> sourceIndices =
        delinearizeIndex(rewriter, op, sourceType.getShape(), linearIndex,
                         failureReason, opLabel);
    mlir::FailureOr<llvm::SmallVector<int64_t>> destIndices = delinearizeIndex(
        rewriter, op, destType.getShape(), linearIndex, failureReason, opLabel);
    if (mlir::failed(sourceIndices) || mlir::failed(destIndices))
      return mlir::failure();

    std::optional<int64_t> sourceOffset =
        wafer::computeWaferPhysicalElementByteOffset(sourceType,
                                                     *sourceIndices);
    std::optional<int64_t> destOffset =
        wafer::computeWaferPhysicalElementByteOffset(destType, *destIndices);
    if (!sourceOffset || !destOffset)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" cannot compute physical element offset")
              .str());
    if (*sourceOffset + elementBytes > sourceInfo->physicalBytes ||
        *destOffset + elementBytes > destInfo->physicalBytes)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" segment exceeds static physical byte range")
              .str());

    if (!segments.empty()) {
      LogicalMovementSegment &last = segments.back();
      if (last.sourceOffset + last.bytes == *sourceOffset &&
          last.destOffset + last.bytes == *destOffset) {
        last.bytes += elementBytes;
        continue;
      }
    }
    segments.push_back({*sourceOffset, *destOffset, elementBytes});
  }

  return segments;
}

static bool
isMetadataOnlyLogicalMovement(mlir::MemRefType sourceType,
                              mlir::MemRefType destType,
                              llvm::ArrayRef<LogicalMovementSegment> segments) {
  std::optional<WaferPhysicalTensorInfo> sourceInfo =
      wafer::computeWaferPhysicalTensorInfo(sourceType);
  std::optional<WaferPhysicalTensorInfo> destInfo =
      wafer::computeWaferPhysicalTensorInfo(destType);
  if (!sourceInfo || !destInfo ||
      sourceInfo->physicalBytes != destInfo->physicalBytes)
    return false;
  return llvm::all_of(segments, [](const LogicalMovementSegment &segment) {
    return segment.sourceOffset == segment.destOffset;
  });
}

static mlir::FailureOr<int64_t>
getStaticDim(mlir::PatternRewriter &rewriter, mlir::Operation *op,
             mlir::RankedTensorType type, int64_t dim,
             std::string *failureReason, llvm::StringRef role) {
  int64_t value = type.getDimSize(dim);
  if (value == mlir::ShapedType::kDynamic)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        llvm::Twine(role).concat(" requires static GEMM dimensions").str());
  return value;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 3>>
inferRank2GemmMKN(ComputeGemmOp op, mlir::PatternRewriter &rewriter,
                  std::string *failureReason) {
  std::optional<mlir::RankedTensorType> lhs =
      getLogicalTensorTypeFromMemRef(op.getLhs().getType());
  std::optional<mlir::RankedTensorType> rhs =
      getLogicalTensorTypeFromMemRef(op.getRhs().getType());
  std::optional<mlir::RankedTensorType> result =
      getLogicalTensorTypeFromMemRef(op.getResult().getType());
  if (!lhs || !rhs || !result)
    return failFailureOr<llvm::SmallVector<int64_t, 3>>(
        rewriter, op, failureReason,
        "tile.gemm lowering requires Wafer memref operands");
  if (lhs->getRank() != 2 || rhs->getRank() != 2 || result->getRank() != 2)
    return failFailureOr<llvm::SmallVector<int64_t, 3>>(
        rewriter, op, failureReason,
        "batched tile.gemm lowering requires explicit batched instruction "
        "dims");

  mlir::FailureOr<int64_t> m =
      getStaticDim(rewriter, op, *lhs, 0, failureReason, "rank-2 tile.gemm");
  mlir::FailureOr<int64_t> k =
      getStaticDim(rewriter, op, *lhs, 1, failureReason, "rank-2 tile.gemm");
  mlir::FailureOr<int64_t> n =
      getStaticDim(rewriter, op, *rhs, 1, failureReason, "rank-2 tile.gemm");
  if (mlir::failed(m) || mlir::failed(k) || mlir::failed(n))
    return mlir::failure();
  return llvm::SmallVector<int64_t, 3>{*m, *k, *n};
}

class TileLoadLowering : public mlir::OpRewritePattern<StorageLoadOp> {
public:
  TileLoadLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<StorageLoadOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(StorageLoadOp op,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();
    mlir::FailureOr<MovementDescriptor> descriptor = getContiguousDescriptor(
        rewriter, op, op.getResult().getType(), failureReason);
    if (mlir::failed(descriptor))
      return mlir::failure();

    createRDMA(rewriter, op.getLoc(), op.getSource(), *dest, *descriptor);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class TileStoreLowering : public mlir::OpRewritePattern<StorageStoreOp> {
public:
  TileStoreLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<StorageStoreOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(StorageStoreOp op,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::FailureOr<MovementDescriptor> descriptor = getContiguousDescriptor(
        rewriter, op, op.getSource().getType(), failureReason);
    if (mlir::failed(descriptor))
      return mlir::failure();

    createWDMA(rewriter, op.getLoc(), op.getSource(), op.getDest(),
               *descriptor);
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class LayoutMaterializeLowering
    : public mlir::OpRewritePattern<LayoutMaterializeOp> {
public:
  LayoutMaterializeLowering(mlir::MLIRContext *context,
                            std::string *failureReason)
      : mlir::OpRewritePattern<LayoutMaterializeOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(LayoutMaterializeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "layout materialize lowering requires memref types");

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        getStaticLogicalMovementSegments(rewriter, op, sourceType, resultType,
                                         failureReason,
                                         "layout materialize lowering");
    if (mlir::failed(segments))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *dest,
                                *segments);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class TileCopyLowering : public mlir::OpRewritePattern<MoveCopyOp> {
public:
  TileCopyLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<MoveCopyOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(MoveCopyOp op, mlir::PatternRewriter &rewriter) const final {
    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    mlir::FailureOr<MovementDescriptor> descriptor = getContiguousDescriptor(
        rewriter, op, op.getSource().getType(), failureReason);
    if (mlir::failed(descriptor))
      return mlir::failure();

    createGatherScatter(rewriter, op.getLoc(), op.getSource(), *dest,
                        *descriptor, *descriptor);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class MoveExtractSliceLowering
    : public mlir::OpRewritePattern<MoveExtractSliceOp> {
public:
  MoveExtractSliceLowering(mlir::MLIRContext *context,
                           std::string *failureReason)
      : mlir::OpRewritePattern<MoveExtractSliceOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(MoveExtractSliceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.extract_slice lowering requires memref types");

    llvm::ArrayRef<int64_t> offsets = op.getOffsets();
    llvm::ArrayRef<int64_t> sizes = op.getSizes();
    llvm::ArrayRef<int64_t> strides = op.getStrides();
    llvm::ArrayRef<int64_t> resultShape = resultType.getShape();

    auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> resultIndices)
        -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
      mlir::FailureOr<llvm::SmallVector<int64_t>> fullSliceIndices =
          expandRankReducedSliceIndices(rewriter, op, sizes, resultShape,
                                        resultIndices, failureReason,
                                        "tile.extract_slice lowering");
      if (mlir::failed(fullSliceIndices))
        return mlir::failure();
      llvm::SmallVector<int64_t> sourceIndices(sizes.size(), 0);
      for (size_t dim = 0; dim < sizes.size(); ++dim)
        sourceIndices[dim] =
            offsets[dim] + (*fullSliceIndices)[dim] * strides[dim];
      return sourceIndices;
    };
    auto destIndexFn = [](llvm::ArrayRef<int64_t> resultIndices)
        -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
      return llvm::SmallVector<int64_t>(resultIndices.begin(),
                                        resultIndices.end());
    };

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        getStaticMappedMovementSegments(
            rewriter, op, sourceType, resultType, resultShape, sourceIndexFn,
            destIndexFn, failureReason, "tile.extract_slice lowering");
    if (mlir::failed(segments))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *dest,
                                *segments);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class MoveInsertSliceLowering
    : public mlir::OpRewritePattern<MoveInsertSliceOp> {
public:
  MoveInsertSliceLowering(mlir::MLIRContext *context,
                          std::string *failureReason)
      : mlir::OpRewritePattern<MoveInsertSliceOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(MoveInsertSliceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::dyn_cast<mlir::MemRefType>(op.getDest().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !destType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.insert_slice lowering requires memref types");

    auto identityIndexFn = [](llvm::ArrayRef<int64_t> indices)
        -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
      return llvm::SmallVector<int64_t>(indices.begin(), indices.end());
    };

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> copySegments =
        getStaticMappedMovementSegments(rewriter, op, destType, resultType,
                                        resultType.getShape(), identityIndexFn,
                                        identityIndexFn, failureReason,
                                        "tile.insert_slice dest copy lowering");
    if (mlir::failed(copySegments))
      return mlir::failure();

    llvm::ArrayRef<int64_t> offsets = op.getOffsets();
    llvm::ArrayRef<int64_t> sizes = op.getSizes();
    llvm::ArrayRef<int64_t> strides = op.getStrides();
    llvm::ArrayRef<int64_t> sourceShape = sourceType.getShape();

    auto sourceIndexFn = [](llvm::ArrayRef<int64_t> sourceIndices)
        -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
      return llvm::SmallVector<int64_t>(sourceIndices.begin(),
                                        sourceIndices.end());
    };
    auto destIndexFn = [&](llvm::ArrayRef<int64_t> sourceIndices)
        -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
      mlir::FailureOr<llvm::SmallVector<int64_t>> fullSliceIndices =
          expandRankReducedSliceIndices(rewriter, op, sizes, sourceShape,
                                        sourceIndices, failureReason,
                                        "tile.insert_slice lowering");
      if (mlir::failed(fullSliceIndices))
        return mlir::failure();
      llvm::SmallVector<int64_t> destIndices(sizes.size(), 0);
      for (size_t dim = 0; dim < sizes.size(); ++dim)
        destIndices[dim] =
            offsets[dim] + (*fullSliceIndices)[dim] * strides[dim];
      return destIndices;
    };

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> insertSegments =
        getStaticMappedMovementSegments(
            rewriter, op, sourceType, resultType, sourceShape, sourceIndexFn,
            destIndexFn, failureReason, "tile.insert_slice lowering");
    if (mlir::failed(insertSegments))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> result = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(result))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getDest(), *result,
                                *copySegments);
    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *result,
                                *insertSegments);
    rewriter.replaceOp(op, *result);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class MoveTransposeLowering : public mlir::OpRewritePattern<MoveTransposeOp> {
public:
  MoveTransposeLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<MoveTransposeOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(MoveTransposeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.transpose lowering requires memref types");

    llvm::ArrayRef<int64_t> permutation = op.getPermutation();
    auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> resultIndices)
        -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
      llvm::SmallVector<int64_t> sourceIndices(sourceType.getRank(), 0);
      for (auto [resultDim, sourceDim] : llvm::enumerate(permutation))
        sourceIndices[sourceDim] = resultIndices[resultDim];
      return sourceIndices;
    };
    auto destIndexFn = [](llvm::ArrayRef<int64_t> resultIndices)
        -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
      return llvm::SmallVector<int64_t>(resultIndices.begin(),
                                        resultIndices.end());
    };

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        getStaticMappedMovementSegments(rewriter, op, sourceType, resultType,
                                        resultType.getShape(), sourceIndexFn,
                                        destIndexFn, failureReason,
                                        "tile.transpose lowering");
    if (mlir::failed(segments))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *dest,
                                *segments);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class MoveBroadcastLowering : public mlir::OpRewritePattern<MoveBroadcastOp> {
public:
  MoveBroadcastLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<MoveBroadcastOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(MoveBroadcastOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.broadcast lowering requires memref types");

    llvm::ArrayRef<int64_t> dimensions = op.getDimensions();
    auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> resultIndices)
        -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
      llvm::SmallVector<int64_t> sourceIndices(sourceType.getRank(), 0);
      for (auto [sourceDim, resultDim] : llvm::enumerate(dimensions))
        sourceIndices[sourceDim] = resultIndices[resultDim];
      return sourceIndices;
    };
    auto destIndexFn = [](llvm::ArrayRef<int64_t> resultIndices)
        -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
      return llvm::SmallVector<int64_t>(resultIndices.begin(),
                                        resultIndices.end());
    };

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        getStaticMappedMovementSegments(rewriter, op, sourceType, resultType,
                                        resultType.getShape(), sourceIndexFn,
                                        destIndexFn, failureReason,
                                        "tile.broadcast lowering");
    if (mlir::failed(segments))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *dest,
                                *segments);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class FillLowering : public mlir::OpRewritePattern<ComputeFillOp> {
public:
  using mlir::OpRewritePattern<ComputeFillOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ComputeFillOp op,
                  mlir::PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<InstrFillOp>(op, op.getDest(), op.getValue());
    return mlir::success();
  }
};

class ElementwiseLowering
    : public mlir::OpRewritePattern<ComputeElementwiseOp> {
public:
  ElementwiseLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<ComputeElementwiseOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeElementwiseOp op,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    auto instr = rewriter.create<InstrElementwiseOp>(
        op.getLoc(), op.getKindAttr(), op.getInputs(), *dest);
    copyOptionalAttr(op, instr, "indexing_maps");
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class ReduceLowering : public mlir::OpRewritePattern<ComputeReduceOp> {
public:
  ReduceLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<ComputeReduceOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeReduceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    auto instr = rewriter.create<InstrReduceOp>(
        op.getLoc(), op.getKindAttr(), op.getInput(), *dest, op.getInit());
    copyOptionalAttr(op, instr, "dimensions");
    copyOptionalAttr(op, instr, "init_value");
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class GemmLowering : public mlir::OpRewritePattern<ComputeGemmOp> {
public:
  GemmLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<ComputeGemmOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(ComputeGemmOp op,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<int64_t, 3>> mkn =
        inferRank2GemmMKN(op, rewriter, failureReason);
    if (mlir::failed(mkn))
      return mlir::failure();

    auto instr = rewriter.create<InstrGemmOp>(
        op.getLoc(), op.getLhs(), op.getRhs(), *dest,
        getI64Attr(rewriter, (*mkn)[0]), getI64Attr(rewriter, (*mkn)[1]),
        getI64Attr(rewriter, (*mkn)[2]),
        /*batch_count=*/mlir::IntegerAttr{},
        /*lhs_batch_dims=*/mlir::DenseI64ArrayAttr{},
        /*lhs_m_dim=*/mlir::IntegerAttr{},
        /*lhs_contracting_dim=*/mlir::IntegerAttr{},
        /*rhs_batch_dims=*/mlir::DenseI64ArrayAttr{},
        /*rhs_contracting_dim=*/mlir::IntegerAttr{},
        /*rhs_n_dim=*/mlir::IntegerAttr{},
        /*result_batch_dims=*/mlir::DenseI64ArrayAttr{},
        /*result_m_dim=*/mlir::IntegerAttr{},
        /*result_n_dim=*/mlir::IntegerAttr{});
    copyOptionalAttr(op, instr, "batch_count");
    copyOptionalAttr(op, instr, "lhs_batch_dims");
    copyOptionalAttr(op, instr, "lhs_m_dim");
    copyOptionalAttr(op, instr, "lhs_contracting_dim");
    copyOptionalAttr(op, instr, "rhs_batch_dims");
    copyOptionalAttr(op, instr, "rhs_contracting_dim");
    copyOptionalAttr(op, instr, "rhs_n_dim");
    copyOptionalAttr(op, instr, "result_batch_dims");
    copyOptionalAttr(op, instr, "result_m_dim");
    copyOptionalAttr(op, instr, "result_n_dim");
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class ViewReshapeLowering : public mlir::OpRewritePattern<ViewReshapeOp> {
public:
  ViewReshapeLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<ViewReshapeOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(ViewReshapeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    if (op.getSource().getType() == op.getResult().getType()) {
      rewriter.replaceOp(op, op.getSource());
      return mlir::success();
    }

    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    if (!sourceType)
      return failPattern(rewriter, op, failureReason,
                         "tile.reshape lowering requires memref source type");
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.reshape lowering requires memref result type");

    MemoryAttr sourceMemory = wafer::getWaferMemoryAttr(sourceType);
    MemoryAttr resultMemory = wafer::getWaferMemoryAttr(resultType);
    if (!sourceMemory || !resultMemory)
      return failPattern(rewriter, op, failureReason,
                         "tile.reshape lowering requires Wafer memref types");

    if (isStandardViewCompatibleLayout(sourceMemory.getLayout()) &&
        isStandardViewCompatibleLayout(resultMemory.getLayout())) {
      mlir::FailureOr<int64_t> sourceBytes =
          getStaticPhysicalBytes(rewriter, op, op.getSource().getType(),
                                 failureReason, "tile.reshape source");
      mlir::FailureOr<int64_t> resultBytes =
          getStaticPhysicalBytes(rewriter, op, op.getResult().getType(),
                                 failureReason, "tile.reshape result");
      if (mlir::failed(sourceBytes) || mlir::failed(resultBytes))
        return mlir::failure();
      if (*sourceBytes != *resultBytes)
        return failPattern(rewriter, op, failureReason,
                           "tile.reshape standard view lowering requires "
                           "equal static physical byte counts");

      llvm::SmallVector<int64_t> sizes(resultType.getShape().begin(),
                                       resultType.getShape().end());
      mlir::FailureOr<llvm::SmallVector<int64_t>> strides =
          getStaticCompactStrides(rewriter, op, resultType, failureReason);
      if (mlir::failed(strides))
        return mlir::failure();

      auto view = rewriter.create<mlir::memref::ReinterpretCastOp>(
          op.getLoc(), resultType, op.getSource(), /*offset=*/0, sizes,
          *strides);
      rewriter.replaceOp(op, view.getResult());
      return mlir::success();
    }

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        getStaticLogicalMovementSegments(rewriter, op, sourceType, resultType,
                                         failureReason,
                                         "tile.reshape lowering");
    if (mlir::failed(segments))
      return mlir::failure();

    if (isMetadataOnlyLogicalMovement(sourceType, resultType, *segments)) {
      llvm::SmallVector<int64_t> sizes(resultType.getShape().begin(),
                                       resultType.getShape().end());
      mlir::FailureOr<llvm::SmallVector<int64_t>> strides =
          getStaticCompactStrides(rewriter, op, resultType, failureReason);
      if (mlir::failed(strides))
        return mlir::failure();

      auto view = rewriter.create<mlir::memref::ReinterpretCastOp>(
          op.getLoc(), resultType, op.getSource(), /*offset=*/0, sizes,
          *strides);
      rewriter.replaceOp(op, view.getResult());
      return mlir::success();
    }

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *dest,
                                *segments);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

template <typename OpT>
class UnsupportedCommLowering : public mlir::OpRewritePattern<OpT> {
public:
  UnsupportedCommLowering(mlir::MLIRContext *context,
                          std::string *failureReason)
      : mlir::OpRewritePattern<OpT>(context), failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(OpT op, mlir::PatternRewriter &rewriter) const final {
    return failPattern(rewriter, op, failureReason,
                       "tile communication lowering requires "
                       "placement/local-rank facts");
  }

private:
  std::string *failureReason;
};

static void configureTileRegionToInstrTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                         mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                         mlir::scf::SCFDialect>();
  target.addLegalOp<mlir::ModuleOp, TileRegionOp, TileYieldOp, SyncLocalDrainOp,
                    InstrRDMAOp, InstrWDMAOp, InstrGatherScatterOp, InstrFillOp,
                    InstrElementwiseOp, InstrReduceOp, InstrConvertOp,
                    InstrGemmOp>();
  target.addIllegalOp<StorageLoadOp, StorageStoreOp, LayoutMaterializeOp,
                      ComputeFillOp, ComputeGemmOp, ComputeElementwiseOp,
                      ComputeReduceOp, MoveCopyOp, MoveExtractSliceOp,
                      MoveInsertSliceOp, MoveTransposeOp, MoveBroadcastOp,
                      ViewReshapeOp, CommSendOp, CommRecvOp, CommWaitOp,
                      CommAllGatherOp, CommReduceScatterOp, CommAllReduceOp>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

static void populateTileRegionToInstrPatterns(mlir::RewritePatternSet &patterns,
                                              std::string *failureReason) {
  mlir::MLIRContext *context = patterns.getContext();
  patterns
      .add<TileLoadLowering, TileStoreLowering, LayoutMaterializeLowering,
           TileCopyLowering, MoveExtractSliceLowering, MoveInsertSliceLowering,
           MoveTransposeLowering, MoveBroadcastLowering, ElementwiseLowering,
           ReduceLowering, GemmLowering, ViewReshapeLowering>(context,
                                                              failureReason);
  patterns.add<FillLowering>(context);
  patterns.add<UnsupportedCommLowering<CommSendOp>,
               UnsupportedCommLowering<CommRecvOp>,
               UnsupportedCommLowering<CommWaitOp>,
               UnsupportedCommLowering<CommAllGatherOp>,
               UnsupportedCommLowering<CommReduceScatterOp>,
               UnsupportedCommLowering<CommAllReduceOp>>(context,
                                                         failureReason);
}

struct ConvertTileRegionToInstrPass
    : public wafer::impl::ConvertTileRegionToInstrPassBase<
          ConvertTileRegionToInstrPass> {
  using wafer::impl::ConvertTileRegionToInstrPassBase<
      ConvertTileRegionToInstrPass>::ConvertTileRegionToInstrPassBase;

  void runOnOperation() final {
    std::string failureReason;
    if (mlir::succeeded(wafer::convertTileRegionToInstrModule(getOperation(),
                                                              &failureReason)))
      return;

    if (!failureReason.empty())
      getOperation().emitError(failureReason);
    else
      getOperation().emitError("tile-region to instruction conversion failed");
    signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult
wafer::convertTileRegionToInstrModule(mlir::ModuleOp module,
                                      std::string *failureReason) {
  if (failureReason)
    failureReason->clear();

  mlir::MLIRContext *context = module.getContext();
  mlir::ConversionTarget target(*context);
  configureTileRegionToInstrTarget(target);

  mlir::RewritePatternSet patterns(context);
  populateTileRegionToInstrPatterns(patterns, failureReason);

  bool conversionSucceeded = false;
  {
    mlir::ScopedDiagnosticHandler handler(
        context, [](mlir::Diagnostic &) { return mlir::success(); });
    conversionSucceeded = mlir::succeeded(
        mlir::applyFullConversion(module, target, std::move(patterns)));
  }

  if (!conversionSucceeded) {
    if (!failureReason || failureReason->empty())
      setFailureReason(failureReason,
                       "tile-region to instruction conversion failed");
    return mlir::failure();
  }
  return mlir::success();
}
