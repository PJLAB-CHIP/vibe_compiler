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
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
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

  auto delinearize =
      [&](mlir::MemRefType type,
          int64_t linearIndex) -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
    llvm::SmallVector<int64_t> indices(type.getRank(), 0);
    for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
      int64_t size = type.getDimSize(dim);
      if (size == mlir::ShapedType::kDynamic || size <= 0)
        return failFailureOr<llvm::SmallVector<int64_t>>(
            rewriter, op, failureReason,
            llvm::Twine(opLabel)
                .concat(" requires static positive shape")
                .str());
      indices[dim] = linearIndex % size;
      linearIndex /= size;
    }
    if (linearIndex != 0)
      return failFailureOr<llvm::SmallVector<int64_t>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" cannot delinearize logical index")
              .str());
    return indices;
  };

  llvm::SmallVector<LogicalMovementSegment> segments;
  for (int64_t linearIndex = 0; linearIndex < elementCount; ++linearIndex) {
    mlir::FailureOr<llvm::SmallVector<int64_t>> sourceIndices =
        delinearize(sourceType, linearIndex);
    mlir::FailureOr<llvm::SmallVector<int64_t>> destIndices =
        delinearize(destType, linearIndex);
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

    for (const LogicalMovementSegment &segment : *segments) {
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

      createGatherScatter(rewriter, op.getLoc(), op.getSource(), *dest,
                          sourceDescriptor, destDescriptor);
    }
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

template <typename OpT>
class UnsupportedMovementLowering : public mlir::OpRewritePattern<OpT> {
public:
  UnsupportedMovementLowering(mlir::MLIRContext *context,
                              std::string *failureReason,
                              llvm::StringRef reason)
      : mlir::OpRewritePattern<OpT>(context), failureReason(failureReason),
        reason(reason.str()) {}

  mlir::LogicalResult
  matchAndRewrite(OpT op, mlir::PatternRewriter &rewriter) const final {
    return failPattern(rewriter, op, failureReason, reason);
  }

private:
  std::string *failureReason;
  std::string reason;
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

    for (const LogicalMovementSegment &segment : *segments) {
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

      createGatherScatter(rewriter, op.getLoc(), op.getSource(), *dest,
                          sourceDescriptor, destDescriptor);
    }
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
  patterns.add<TileLoadLowering, TileStoreLowering, LayoutMaterializeLowering,
               TileCopyLowering, ElementwiseLowering, ReduceLowering,
               GemmLowering, ViewReshapeLowering>(context, failureReason);
  patterns.add<FillLowering>(context);
  patterns.add<UnsupportedMovementLowering<MoveExtractSliceOp>>(
      context, failureReason,
      "wafer.tile.extract_slice cannot lower to TDMA-backed "
      "wafer.instr.gather_scatter: unsupported slice descriptor");
  patterns.add<UnsupportedMovementLowering<MoveInsertSliceOp>>(
      context, failureReason,
      "wafer.tile.insert_slice cannot lower to TDMA-backed "
      "wafer.instr.gather_scatter: unsupported slice descriptor");
  patterns.add<UnsupportedMovementLowering<MoveTransposeOp>>(
      context, failureReason,
      "wafer.tile.transpose cannot lower to TDMA-backed "
      "wafer.instr.gather_scatter: unsupported permutation");
  patterns.add<UnsupportedMovementLowering<MoveBroadcastOp>>(
      context, failureReason,
      "wafer.tile.broadcast cannot lower to TDMA-backed "
      "wafer.instr.gather_scatter: unsupported broadcast descriptor");
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
