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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

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

struct PackedMovementDescriptor {
  MovementDescriptor source;
  MovementDescriptor dest;
};

struct DescriptorLoop {
  int64_t strideBytes = 0;
  int64_t iterations = 1;
};

constexpr int64_t kNchw2NhwcPermutation[] = {0, 2, 3, 1};
constexpr int64_t kNhwc2NchwPermutation[] = {0, 3, 1, 2};

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

static std::optional<int64_t> checkedMulI64(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return std::nullopt;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return std::nullopt;
  return lhs * rhs;
}

static std::optional<int64_t> checkedAddI64(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return std::nullopt;
  if (rhs > std::numeric_limits<int64_t>::max() - lhs)
    return std::nullopt;
  return lhs + rhs;
}

static std::optional<int64_t> checkedAddScaledI64(int64_t base, int64_t stride,
                                                  int64_t iteration) {
  std::optional<int64_t> scaled = checkedMulI64(stride, iteration);
  if (!scaled)
    return std::nullopt;
  return checkedAddI64(base, *scaled);
}

static std::optional<int64_t>
computeDescriptorPayloadBytes(int64_t innerBytes,
                              llvm::ArrayRef<int64_t> iterations) {
  int64_t total = innerBytes;
  for (int64_t iteration : iterations) {
    std::optional<int64_t> next = checkedMulI64(total, iteration);
    if (!next)
      return std::nullopt;
    total = *next;
  }
  return total;
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

static mlir::FailureOr<MovementDescriptor>
getStridedTensorDescriptor(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                           mlir::Type type, std::string *failureReason,
                           llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
        llvm::Twine(role).concat(" requires a Wafer memref type").str());

  std::optional<WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  if (!info)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
        llvm::Twine(role)
            .concat(" requires static byte-addressable tensor")
            .str());

  if (info->bitPackedElement) {
    if (info->compactBytes <= 0 || info->physicalBytes != info->compactBytes)
      return failFailureOr<MovementDescriptor>(
          rewriter, op, failureReason,
          llvm::Twine(role)
              .concat(" requires contiguous bitpacked tensor")
              .str());
    MovementDescriptor descriptor;
    descriptor.byteCount = info->compactBytes;
    descriptor.innerBytes = info->compactBytes;
    descriptor.strides.assign({0, 0, 0});
    descriptor.iterations.assign({1, 1, 1});
    return descriptor;
  }

  if (info->compactBytes <= 0 || info->elementBytes <= 0)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
        llvm::Twine(role)
            .concat(" requires static byte-addressable tensor")
            .str());

  llvm::SmallVector<int64_t> memrefStrides;
  int64_t memrefOffset = 0;
  if (mlir::failed(
          mlir::getStridesAndOffset(memrefType, memrefStrides, memrefOffset)) ||
      static_cast<int64_t>(memrefStrides.size()) != memrefType.getRank())
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
        llvm::Twine(role)
            .concat(" requires static strided memref layout")
            .str());

  int64_t innerElements = 1;
  llvm::SmallVector<DescriptorLoop, 3> loops;
  for (int64_t dim = memrefType.getRank() - 1; dim >= 0; --dim) {
    int64_t dimSize = memrefType.getDimSize(dim);
    int64_t dimStride = memrefStrides[dim];
    if (dimSize == mlir::ShapedType::kDynamic || dimSize <= 0 ||
        dimStride == mlir::ShapedType::kDynamic || dimStride < 0)
      return failFailureOr<MovementDescriptor>(
          rewriter, op, failureReason,
          llvm::Twine(role)
              .concat(" requires static positive shape and non-negative "
                      "strides")
              .str());

    if (dimSize == 1)
      continue;

    if (loops.empty() && dimStride == innerElements) {
      std::optional<int64_t> nextInner = checkedMulI64(innerElements, dimSize);
      if (!nextInner)
        return failFailureOr<MovementDescriptor>(
            rewriter, op, failureReason,
            llvm::Twine(role).concat(" descriptor inner span overflows").str());
      innerElements = *nextInner;
      continue;
    }

    std::optional<int64_t> strideBytes =
        checkedMulI64(dimStride, info->elementBytes);
    if (!strideBytes)
      return failFailureOr<MovementDescriptor>(
          rewriter, op, failureReason,
          llvm::Twine(role).concat(" descriptor stride overflows").str());

    if (!loops.empty()) {
      std::optional<int64_t> collapsedStride =
          checkedMulI64(loops.back().strideBytes, loops.back().iterations);
      if (collapsedStride && *collapsedStride == *strideBytes) {
        std::optional<int64_t> collapsedIterations =
            checkedMulI64(loops.back().iterations, dimSize);
        if (!collapsedIterations)
          return failFailureOr<MovementDescriptor>(
              rewriter, op, failureReason,
              llvm::Twine(role)
                  .concat(" descriptor iteration overflows")
                  .str());
        loops.back().iterations = *collapsedIterations;
        continue;
      }
    }

    if (loops.size() == 3)
      return failFailureOr<MovementDescriptor>(
          rewriter, op, failureReason,
          llvm::Twine(role)
              .concat(" requires at most three strided descriptor levels")
              .str());
    loops.push_back({*strideBytes, dimSize});
  }

  std::optional<int64_t> innerBytes =
      checkedMulI64(innerElements, info->elementBytes);
  if (!innerBytes)
    return failFailureOr<MovementDescriptor>(
        rewriter, op, failureReason,
        llvm::Twine(role)
            .concat(" descriptor inner byte count overflows")
            .str());

  MovementDescriptor descriptor;
  descriptor.byteCount = info->compactBytes;
  descriptor.innerBytes = *innerBytes;
  descriptor.strides.assign({0, 0, 0});
  descriptor.iterations.assign({1, 1, 1});
  for (auto [index, loop] : llvm::enumerate(loops)) {
    descriptor.strides[index] = loop.strideBytes;
    descriptor.iterations[index] = loop.iterations;
  }
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

static bool
matchesPackedDescriptor(llvm::ArrayRef<LogicalMovementSegment> segments,
                        size_t start, llvm::ArrayRef<int64_t> sourceStrides,
                        llvm::ArrayRef<int64_t> destStrides,
                        llvm::ArrayRef<int64_t> iterations) {
  if (sourceStrides.size() != 3 || destStrides.size() != 3 ||
      iterations.size() != 3 || start >= segments.size())
    return false;

  std::optional<int64_t> totalSegments =
      computeDescriptorPayloadBytes(/*innerBytes=*/1, iterations);
  if (!totalSegments)
    return false;
  if (*totalSegments <= 0 ||
      start + static_cast<size_t>(*totalSegments) > segments.size())
    return false;

  const LogicalMovementSegment &base = segments[start];
  for (int64_t linear = 0; linear < *totalSegments; ++linear) {
    const LogicalMovementSegment &segment = segments[start + linear];
    if (segment.bytes != base.bytes)
      return false;

    int64_t i0 = linear % iterations[0];
    int64_t i1 = (linear / iterations[0]) % iterations[1];
    int64_t i2 = linear / (iterations[0] * iterations[1]);

    std::optional<int64_t> expectedSource =
        checkedAddScaledI64(base.sourceOffset, sourceStrides[0], i0);
    std::optional<int64_t> expectedDest =
        checkedAddScaledI64(base.destOffset, destStrides[0], i0);
    if (!expectedSource || !expectedDest)
      return false;
    expectedSource = checkedAddScaledI64(*expectedSource, sourceStrides[1], i1);
    expectedDest = checkedAddScaledI64(*expectedDest, destStrides[1], i1);
    if (!expectedSource || !expectedDest)
      return false;
    expectedSource = checkedAddScaledI64(*expectedSource, sourceStrides[2], i2);
    expectedDest = checkedAddScaledI64(*expectedDest, destStrides[2], i2);
    if (!expectedSource || !expectedDest)
      return false;

    if (segment.sourceOffset != *expectedSource ||
        segment.destOffset != *expectedDest)
      return false;
  }
  return true;
}

static int64_t
inferPackedIteration(llvm::ArrayRef<LogicalMovementSegment> segments,
                     size_t start, int64_t blockSize,
                     llvm::SmallVectorImpl<int64_t> &sourceStrides,
                     llvm::SmallVectorImpl<int64_t> &destStrides,
                     llvm::SmallVectorImpl<int64_t> &iterations, int64_t dim) {
  if (blockSize <= 0)
    return 1;
  size_t nextBlockStart = start + static_cast<size_t>(blockSize);
  if (nextBlockStart >= segments.size())
    return 1;

  const LogicalMovementSegment &base = segments[start];
  const LogicalMovementSegment &nextBase = segments[nextBlockStart];
  if (nextBase.bytes != base.bytes ||
      nextBase.sourceOffset < base.sourceOffset ||
      nextBase.destOffset < base.destOffset)
    return 1;

  sourceStrides[dim] = nextBase.sourceOffset - base.sourceOffset;
  destStrides[dim] = nextBase.destOffset - base.destOffset;

  int64_t inferred = 1;
  while (true) {
    llvm::SmallVector<int64_t, 3> candidateIterations(iterations.begin(),
                                                      iterations.end());
    candidateIterations[dim] = inferred + 1;
    if (!matchesPackedDescriptor(segments, start, sourceStrides, destStrides,
                                 candidateIterations))
      break;
    iterations[dim] = inferred + 1;
    ++inferred;
  }
  if (inferred == 1) {
    sourceStrides[dim] = 0;
    destStrides[dim] = 0;
  }
  return inferred;
}

static PackedMovementDescriptor
packMovementDescriptor(llvm::ArrayRef<LogicalMovementSegment> segments,
                       size_t start) {
  const LogicalMovementSegment &base = segments[start];
  llvm::SmallVector<int64_t, 3> sourceStrides({0, 0, 0});
  llvm::SmallVector<int64_t, 3> destStrides({0, 0, 0});
  llvm::SmallVector<int64_t, 3> iterations({1, 1, 1});

  inferPackedIteration(segments, start, /*blockSize=*/1, sourceStrides,
                       destStrides, iterations, /*dim=*/0);
  std::optional<int64_t> dim1Block = computeDescriptorPayloadBytes(
      /*innerBytes=*/1, llvm::ArrayRef<int64_t>(iterations).take_front(1));
  if (dim1Block)
    inferPackedIteration(segments, start, *dim1Block, sourceStrides,
                         destStrides, iterations, /*dim=*/1);
  std::optional<int64_t> dim2Block = computeDescriptorPayloadBytes(
      /*innerBytes=*/1, llvm::ArrayRef<int64_t>(iterations).take_front(2));
  if (dim2Block)
    inferPackedIteration(segments, start, *dim2Block, sourceStrides,
                         destStrides, iterations, /*dim=*/2);

  std::optional<int64_t> byteCount =
      computeDescriptorPayloadBytes(base.bytes, iterations);
  if (!byteCount) {
    sourceStrides.assign({0, 0, 0});
    destStrides.assign({0, 0, 0});
    iterations.assign({1, 1, 1});
    byteCount = base.bytes;
  }

  MovementDescriptor sourceDescriptor;
  sourceDescriptor.byteCount = *byteCount;
  sourceDescriptor.innerBytes = base.bytes;
  sourceDescriptor.byteOffset = base.sourceOffset;
  sourceDescriptor.strides = sourceStrides;
  sourceDescriptor.iterations = iterations;

  MovementDescriptor destDescriptor;
  destDescriptor.byteCount = *byteCount;
  destDescriptor.innerBytes = base.bytes;
  destDescriptor.byteOffset = base.destOffset;
  destDescriptor.strides = destStrides;
  destDescriptor.iterations = iterations;

  return {sourceDescriptor, destDescriptor};
}

static void
createGatherScatterSegments(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value source, mlir::Value dest,
                            llvm::ArrayRef<LogicalMovementSegment> segments) {
  for (size_t index = 0; index < segments.size();) {
    PackedMovementDescriptor packed = packMovementDescriptor(segments, index);
    createGatherScatter(rewriter, loc, source, dest, packed.source,
                        packed.dest);

    std::optional<int64_t> descriptorSegments = computeDescriptorPayloadBytes(
        /*innerBytes=*/1, packed.source.iterations);
    if (!descriptorSegments || *descriptorSegments <= 0)
      descriptorSegments = 1;
    index += static_cast<size_t>(*descriptorSegments);
  }
}

static mlir::FailureOr<uint64_t> preflightPackedMovementCommands(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    llvm::ArrayRef<LogicalMovementSegment> segments,
    std::string *failureReason, llvm::StringRef opLabel) {
  if (segments.empty())
    return failFailureOr<uint64_t>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel).concat(" produced no movement segments").str());

  auto fitsTargetField = [](int64_t value) {
    return value >= 0 &&
           static_cast<uint64_t>(value) <=
               std::numeric_limits<uint32_t>::max();
  };
  auto descriptorFitsTarget = [&](const MovementDescriptor &descriptor) {
    if (descriptor.byteCount <= 0 || descriptor.innerBytes <= 0 ||
        !fitsTargetField(descriptor.byteCount) ||
        !fitsTargetField(descriptor.innerBytes) ||
        !fitsTargetField(descriptor.byteOffset))
      return false;
    return llvm::all_of(descriptor.strides, fitsTargetField) &&
           llvm::all_of(descriptor.iterations, [&](int64_t iteration) {
             return iteration > 0 && fitsTargetField(iteration);
           });
  };

  uint64_t commandCount = 0;
  for (size_t index = 0; index < segments.size();) {
    PackedMovementDescriptor packed = packMovementDescriptor(segments, index);
    if (!descriptorFitsTarget(packed.source) ||
        !descriptorFitsTarget(packed.dest))
      return failFailureOr<uint64_t>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" descriptor exceeds uint32 target fields")
              .str());

    std::optional<int64_t> descriptorSegments = computeDescriptorPayloadBytes(
        /*innerBytes=*/1, packed.source.iterations);
    if (!descriptorSegments || *descriptorSegments <= 0)
      descriptorSegments = 1;
    index += static_cast<size_t>(*descriptorSegments);
    if (commandCount == std::numeric_limits<uint64_t>::max())
      return failFailureOr<uint64_t>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel).concat(" command count overflows").str());
    ++commandCount;
  }
  return commandCount;
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
readRequiredI64Attr(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                    llvm::StringRef name, std::string *failureReason) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (!attr)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        llvm::Twine("batched tile.gemm lowering requires ")
            .concat(name)
            .concat(" attr")
            .str());
  return attr.getInt();
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

static bool requiresGatherScatterMaterialization(InstrDataMoveKind kind) {
  switch (kind) {
  case InstrDataMoveKind::Mirror:
  case InstrDataMoveKind::Transpose:
  case InstrDataMoveKind::Rotate90:
  case InstrDataMoveKind::Rotate180:
  case InstrDataMoveKind::Rotate270:
  case InstrDataMoveKind::Nchw2Nhwc:
  case InstrDataMoveKind::Nhwc2Nchw:
  case InstrDataMoveKind::TensorNom:
    return true;
  case InstrDataMoveKind::Pad:
  case InstrDataMoveKind::Img2Col:
    return false;
  }
  llvm_unreachable("unknown instr data move kind");
}

static mlir::LogicalResult verifyStaticShapeAttrMatchesMemRef(
    mlir::PatternRewriter &rewriter, mlir::Operation *op, mlir::MemRefType type,
    mlir::DenseI64ArrayAttr shapeAttr, llvm::StringRef role,
    std::string *failureReason, llvm::StringRef opLabel) {
  if (type.getRank() != static_cast<int64_t>(shapeAttr.size()))
    return failPattern(rewriter, op, failureReason,
                       llvm::Twine(opLabel)
                           .concat(" requires ")
                           .concat(role)
                           .concat("_shape rank to match the ")
                           .concat(role)
                           .concat(" memref rank")
                           .str());
  for (auto [dim, attrSize] : llvm::enumerate(shapeAttr.asArrayRef())) {
    int64_t memrefSize = type.getDimSize(dim);
    if (memrefSize == mlir::ShapedType::kDynamic || memrefSize != attrSize)
      return failPattern(rewriter, op, failureReason,
                         llvm::Twine(opLabel)
                             .concat(" requires static ")
                             .concat(role)
                             .concat(" memref shape to match ")
                             .concat(role)
                             .concat("_shape")
                             .str());
  }
  return mlir::success();
}

static mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getPermutationDataMoveSegments(mlir::PatternRewriter &rewriter,
                               InstrTDMADataMoveOp op,
                               mlir::MemRefType sourceType,
                               mlir::MemRefType destType,
                               llvm::ArrayRef<int64_t> permutation,
                               std::string *failureReason,
                               llvm::StringRef opLabel) {
  if (sourceType.getRank() != destType.getRank() ||
      sourceType.getRank() != static_cast<int64_t>(permutation.size()))
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires rank-compatible operands")
            .str());
  for (auto [destDim, sourceDim] : llvm::enumerate(permutation)) {
    int64_t sourceSize = sourceType.getDimSize(sourceDim);
    int64_t destSize = destType.getDimSize(destDim);
    if (sourceSize == mlir::ShapedType::kDynamic ||
        destSize == mlir::ShapedType::kDynamic || sourceSize != destSize)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" requires permutation-compatible static shapes")
              .str());
  }

  auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> destIndices)
      -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
    llvm::SmallVector<int64_t> sourceIndices(sourceType.getRank(), 0);
    for (auto [destDim, sourceDim] : llvm::enumerate(permutation))
      sourceIndices[sourceDim] = destIndices[destDim];
    return sourceIndices;
  };
  auto destIndexFn = [](llvm::ArrayRef<int64_t> destIndices)
      -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
    return llvm::SmallVector<int64_t>(destIndices.begin(), destIndices.end());
  };

  return getStaticMappedMovementSegments(rewriter, op, sourceType, destType,
                                         destType.getShape(), sourceIndexFn,
                                         destIndexFn, failureReason, opLabel);
}

static mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getMirrorDataMoveSegments(mlir::PatternRewriter &rewriter,
                          InstrTDMADataMoveOp op, mlir::MemRefType sourceType,
                          mlir::MemRefType destType,
                          llvm::ArrayRef<int64_t> axes,
                          std::string *failureReason, llvm::StringRef opLabel) {
  if (sourceType.getRank() != destType.getRank())
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires rank-compatible operands")
            .str());
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim) {
    int64_t sourceSize = sourceType.getDimSize(dim);
    int64_t destSize = destType.getDimSize(dim);
    if (sourceSize == mlir::ShapedType::kDynamic ||
        destSize == mlir::ShapedType::kDynamic || sourceSize != destSize)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" requires equal static source/dest shapes")
              .str());
  }

  llvm::SmallDenseSet<int64_t, 4> mirroredAxes;
  mirroredAxes.insert(axes.begin(), axes.end());
  auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> destIndices)
      -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
    llvm::SmallVector<int64_t> sourceIndices(destIndices.begin(),
                                             destIndices.end());
    for (int64_t axis : mirroredAxes)
      sourceIndices[axis] = sourceType.getDimSize(axis) - 1 - destIndices[axis];
    return sourceIndices;
  };
  auto destIndexFn = [](llvm::ArrayRef<int64_t> destIndices)
      -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
    return llvm::SmallVector<int64_t>(destIndices.begin(), destIndices.end());
  };

  return getStaticMappedMovementSegments(rewriter, op, sourceType, destType,
                                         destType.getShape(), sourceIndexFn,
                                         destIndexFn, failureReason, opLabel);
}

static mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getRotateDataMoveSegments(mlir::PatternRewriter &rewriter,
                          InstrTDMADataMoveOp op, mlir::MemRefType sourceType,
                          mlir::MemRefType destType, InstrDataMoveKind kind,
                          llvm::ArrayRef<int64_t> axes,
                          std::string *failureReason, llvm::StringRef opLabel) {
  if (sourceType.getRank() != destType.getRank() || axes.size() != 2)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires rank-compatible operands")
            .str());
  int64_t axis0 = axes[0];
  int64_t axis1 = axes[1];
  for (int64_t dim = 0; dim < sourceType.getRank(); ++dim) {
    int64_t sourceSize = sourceType.getDimSize(dim);
    int64_t destSize = destType.getDimSize(dim);
    if (sourceSize == mlir::ShapedType::kDynamic ||
        destSize == mlir::ShapedType::kDynamic)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel).concat(" requires static shapes").str());
    if (dim != axis0 && dim != axis1 && sourceSize != destSize)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" requires non-rotated dimensions to match")
              .str());
  }

  int64_t sourceAxis0Size = sourceType.getDimSize(axis0);
  int64_t sourceAxis1Size = sourceType.getDimSize(axis1);
  int64_t destAxis0Size = destType.getDimSize(axis0);
  int64_t destAxis1Size = destType.getDimSize(axis1);
  if (kind == InstrDataMoveKind::Rotate180) {
    if (sourceAxis0Size != destAxis0Size || sourceAxis1Size != destAxis1Size)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" requires equal rotated-axis shapes")
              .str());
  } else if (sourceAxis0Size != destAxis1Size ||
             sourceAxis1Size != destAxis0Size) {
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires swapped rotated-axis shapes")
            .str());
  }

  auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> destIndices)
      -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
    llvm::SmallVector<int64_t> sourceIndices(destIndices.begin(),
                                             destIndices.end());
    switch (kind) {
    case InstrDataMoveKind::Rotate90:
      sourceIndices[axis0] = sourceAxis0Size - 1 - destIndices[axis1];
      sourceIndices[axis1] = destIndices[axis0];
      break;
    case InstrDataMoveKind::Rotate180:
      sourceIndices[axis0] = sourceAxis0Size - 1 - destIndices[axis0];
      sourceIndices[axis1] = sourceAxis1Size - 1 - destIndices[axis1];
      break;
    case InstrDataMoveKind::Rotate270:
      sourceIndices[axis0] = destIndices[axis1];
      sourceIndices[axis1] = sourceAxis1Size - 1 - destIndices[axis0];
      break;
    default:
      llvm_unreachable("expected rotate data_move kind");
    }
    return sourceIndices;
  };
  auto destIndexFn = [](llvm::ArrayRef<int64_t> destIndices)
      -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
    return llvm::SmallVector<int64_t>(destIndices.begin(), destIndices.end());
  };

  return getStaticMappedMovementSegments(rewriter, op, sourceType, destType,
                                         destType.getShape(), sourceIndexFn,
                                         destIndexFn, failureReason, opLabel);
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
inferGemmMKN(ComputeGemmOp op, mlir::PatternRewriter &rewriter,
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

  if (lhs->getRank() != 2 || rhs->getRank() != 2 || result->getRank() != 2) {
    mlir::FailureOr<int64_t> lhsMDim =
        readRequiredI64Attr(rewriter, op, "lhs_m_dim", failureReason);
    mlir::FailureOr<int64_t> lhsKDim =
        readRequiredI64Attr(rewriter, op, "lhs_contracting_dim", failureReason);
    mlir::FailureOr<int64_t> rhsNDim =
        readRequiredI64Attr(rewriter, op, "rhs_n_dim", failureReason);
    if (mlir::failed(lhsMDim) || mlir::failed(lhsKDim) || mlir::failed(rhsNDim))
      return mlir::failure();
    mlir::FailureOr<int64_t> m = getStaticDim(
        rewriter, op, *lhs, *lhsMDim, failureReason, "batched tile.gemm");
    mlir::FailureOr<int64_t> k = getStaticDim(
        rewriter, op, *lhs, *lhsKDim, failureReason, "batched tile.gemm");
    mlir::FailureOr<int64_t> n = getStaticDim(
        rewriter, op, *rhs, *rhsNDim, failureReason, "batched tile.gemm");
    if (mlir::failed(m) || mlir::failed(k) || mlir::failed(n))
      return mlir::failure();
    return llvm::SmallVector<int64_t, 3>{*m, *k, *n};
  }

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
    mlir::FailureOr<MovementDescriptor> descriptor =
        getStridedTensorDescriptor(rewriter, op, op.getSource().getType(),
                                   failureReason, "tile.load source");
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
    mlir::FailureOr<MovementDescriptor> descriptor = getStridedTensorDescriptor(
        rewriter, op, op.getDest().getType(), failureReason, "tile.store dest");
    if (mlir::failed(descriptor))
      return mlir::failure();

    createWDMA(rewriter, op.getLoc(), op.getSource(), op.getDest(),
               *descriptor);
    // A tile store is the last local-engine use of the tile-local source in
    // the current schedule.  Make that completion boundary explicit so SPM
    // planning can end the source lifetime before the next traversal tile.
    // The terminal fence remains a safety net for paths without a store.
    rewriter.create<SyncLocalFenceOp>(op.getLoc());
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

class InstrTDMADataMoveLowering
    : public mlir::OpRewritePattern<InstrTDMADataMoveOp> {
public:
  InstrTDMADataMoveLowering(mlir::MLIRContext *context,
                            std::string *failureReason)
      : mlir::OpRewritePattern<InstrTDMADataMoveOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(InstrTDMADataMoveOp op,
                  mlir::PatternRewriter &rewriter) const final {
    InstrDataMoveKind kind = op.getKindAttr().getValue();
    if (!requiresGatherScatterMaterialization(kind))
      return mlir::failure();

    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::dyn_cast<mlir::MemRefType>(op.getDest().getType());
    if (!sourceType || !destType)
      return failPattern(rewriter, op, failureReason,
                         "tdma_data_move lowering requires memref operands");
    if (mlir::failed(verifyStaticShapeAttrMatchesMemRef(
            rewriter, op, sourceType, op.getSourceShapeAttr(), "source",
            failureReason, "tdma_data_move lowering")) ||
        mlir::failed(verifyStaticShapeAttrMatchesMemRef(
            rewriter, op, destType, op.getDestShapeAttr(), "dest",
            failureReason, "tdma_data_move lowering")))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        lowerToSegments(op, sourceType, destType, kind, rewriter);
    if (mlir::failed(segments))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(),
                                op.getDest(), *segments);
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
  lowerToSegments(InstrTDMADataMoveOp op, mlir::MemRefType sourceType,
                  mlir::MemRefType destType, InstrDataMoveKind kind,
                  mlir::PatternRewriter &rewriter) const {
    switch (kind) {
    case InstrDataMoveKind::Transpose:
      if (!op.getPermutationAttr())
        return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
            rewriter, op, failureReason,
            "tdma_data_move transpose lowering requires permutation attr");
      return getPermutationDataMoveSegments(
          rewriter, op, sourceType, destType,
          op.getPermutationAttr().asArrayRef(), failureReason,
          "tdma_data_move transpose lowering");
    case InstrDataMoveKind::Nchw2Nhwc:
      return getPermutationDataMoveSegments(
          rewriter, op, sourceType, destType,
          llvm::ArrayRef<int64_t>(kNchw2NhwcPermutation), failureReason,
          "tdma_data_move nchw2nhwc lowering");
    case InstrDataMoveKind::Nhwc2Nchw:
      return getPermutationDataMoveSegments(
          rewriter, op, sourceType, destType,
          llvm::ArrayRef<int64_t>(kNhwc2NchwPermutation), failureReason,
          "tdma_data_move nhwc2nchw lowering");
    case InstrDataMoveKind::TensorNom:
      return getStaticLogicalMovementSegments(
          rewriter, op, sourceType, destType, failureReason,
          "tdma_data_move tensor_nom lowering");
    case InstrDataMoveKind::Mirror:
      if (!op.getAxesAttr())
        return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
            rewriter, op, failureReason,
            "tdma_data_move mirror lowering requires axes attr");
      return getMirrorDataMoveSegments(
          rewriter, op, sourceType, destType, op.getAxesAttr().asArrayRef(),
          failureReason, "tdma_data_move mirror lowering");
    case InstrDataMoveKind::Rotate90:
    case InstrDataMoveKind::Rotate180:
    case InstrDataMoveKind::Rotate270:
      if (!op.getAxesAttr())
        return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
            rewriter, op, failureReason,
            "tdma_data_move rotate lowering requires axes attr");
      return getRotateDataMoveSegments(rewriter, op, sourceType, destType, kind,
                                       op.getAxesAttr().asArrayRef(),
                                       failureReason,
                                       "tdma_data_move rotate lowering");
    case InstrDataMoveKind::Pad:
    case InstrDataMoveKind::Img2Col:
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          "tdma_data_move pad/img2col remains in the production target "
          "surface");
    }
    llvm_unreachable("unknown instr data move kind");
  }

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

static mlir::FailureOr<InstrElementwiseKindAttr> getInstrElementwiseKindAttr(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    ComputeElementwiseKindAttr computeKind, std::string *failureReason);

static mlir::FailureOr<InstrElementwiseKindAttr> getAccumulationElementwiseKind(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    ComputeReduceKindAttr reduceKind, std::string *failureReason,
    llvm::StringRef opLabel);

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

// A select fed by a private, constant-filled predicate does not require a
// target boolean fill or mask operation.  Canonicalize that exact tile-level
// dataflow to an explicit fresh copy before conversion patterns can lower the
// fill independently.  Keeping this as a separate typed prepass makes the
// rewrite independent of cross-root dialect-conversion pattern ordering.
class ConstantPredicateSelectToCopy
    : public mlir::OpRewritePattern<ComputeElementwiseOp> {
public:
  using mlir::OpRewritePattern<ComputeElementwiseOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ComputeElementwiseOp op,
                  mlir::PatternRewriter &rewriter) const final {
    if (op.getKind() != ComputeElementwiseKind::Select ||
        op.getInputs().size() != 3)
      return mlir::failure();

    mlir::Value predicate = op.getInputs().front();
    auto predicateAlloc = predicate.getDefiningOp<mlir::memref::AllocOp>();
    if (!predicateAlloc)
      return mlir::failure();

    mlir::OpOperand *selectPredicateUse = &op->getOpOperand(0);
    ComputeFillOp predicateFill;
    unsigned selectUseCount = 0;
    unsigned fillUseCount = 0;
    for (mlir::OpOperand &use : predicate.getUses()) {
      if (&use == selectPredicateUse) {
        ++selectUseCount;
        continue;
      }
      auto fill = mlir::dyn_cast<ComputeFillOp>(use.getOwner());
      if (!fill || &use != &fill.getDestMutable())
        return mlir::failure();
      predicateFill = fill;
      ++fillUseCount;
    }
    if (selectUseCount != 1 || fillUseCount != 1 || !predicateFill)
      return mlir::failure();

    if (predicateFill->getBlock() != op->getBlock() ||
        !predicateFill->isBeforeInBlock(op))
      return mlir::failure();

    auto constant =
        predicateFill.getValue().getDefiningOp<mlir::arith::ConstantOp>();
    auto valueAttr =
        constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                 : mlir::IntegerAttr{};
    if (!constant || !constant.getType().isInteger(1) || !valueAttr ||
        !valueAttr.getType().isInteger(1))
      return mlir::failure();

    unsigned selectedInputIndex = valueAttr.getValue().isZero() ? 2 : 1;
    mlir::Value selected = op.getInputs()[selectedInputIndex];
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    auto selectedType = mlir::dyn_cast<mlir::MemRefType>(selected.getType());
    if (!resultType || !selectedType || selectedType != resultType)
      return mlir::failure();

    if (mlir::Attribute rawMaps = op->getAttr("indexing_maps")) {
      auto maps = mlir::dyn_cast<mlir::ArrayAttr>(rawMaps);
      if (!maps || maps.size() != op.getInputs().size() + 1)
        return mlir::failure();
      auto hasIdentityMap = [&](unsigned mapIndex) {
        auto mapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(maps[mapIndex]);
        if (!mapAttr)
          return false;
        mlir::AffineMap map = mapAttr.getValue();
        return map.getNumDims() == resultType.getRank() &&
               map.getNumSymbols() == 0 &&
               map.getNumResults() == resultType.getRank() && map.isIdentity();
      };
      if (!hasIdentityMap(selectedInputIndex) ||
          !hasIdentityMap(maps.size() - 1))
        return mlir::failure();
    }

    auto copy = rewriter.create<MoveCopyOp>(op.getLoc(), resultType, selected);
    rewriter.replaceOp(op, copy.getResult());
    rewriter.eraseOp(predicateFill);
    rewriter.eraseOp(predicateAlloc);
    if (constant->use_empty())
      rewriter.eraseOp(constant);
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
    if (op.getKind() == ComputeElementwiseKind::Select) {
      if (op.getInputs().size() != 3)
        return failPattern(
            rewriter, op, failureReason,
            "target select lowering requires predicate, true and false "
            "operands");
      auto destType =
          mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
      if (!destType || !mlir::isa<mlir::FloatType>(destType.getElementType()))
        return failPattern(
            rewriter, op, failureReason,
            "target select lowering currently requires floating-point values");
    }

    struct InputMovementPlan {
      mlir::Value source;
      mlir::MemRefType materializedType;
      llvm::SmallVector<LogicalMovementSegment> segments;
    };

    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.elementwise lowering requires a memref result");

    // Preflight every map before creating an allocation or an instruction.
    // A failed conversion therefore cannot leave a partially materialized
    // operand sequence in the pattern rewriter.
    llvm::SmallVector<InputMovementPlan, 3> movementPlans;
    movementPlans.reserve(op.getInputs().size());
    mlir::Attribute rawIndexingMaps = op->getAttr("indexing_maps");
    mlir::ArrayAttr indexingMaps;
    if (rawIndexingMaps) {
      indexingMaps = mlir::dyn_cast<mlir::ArrayAttr>(rawIndexingMaps);
      if (!indexingMaps)
        return failPattern(
            rewriter, op, failureReason,
            "tile.elementwise indexing_maps must be an array attribute");
      if (indexingMaps.size() != op.getInputs().size() + 1)
        return failPattern(
            rewriter, op, failureReason,
            "tile.elementwise indexing map count must match inputs plus "
            "result");

      auto resultMapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(
          indexingMaps[indexingMaps.size() - 1]);
      if (!resultMapAttr ||
          resultMapAttr.getValue().getNumDims() != resultType.getRank() ||
          resultMapAttr.getValue().getNumSymbols() != 0 ||
          !resultMapAttr.getValue().isIdentity())
        return failPattern(
            rewriter, op, failureReason,
            "tile.elementwise result indexing map must be identity");
    }

    for (auto [index, input] : llvm::enumerate(op.getInputs())) {
      InputMovementPlan plan;
      plan.source = input;
      if (!indexingMaps) {
        movementPlans.push_back(std::move(plan));
        continue;
      }

      auto sourceType = mlir::dyn_cast<mlir::MemRefType>(input.getType());
      auto inputMapAttr =
          mlir::dyn_cast<mlir::AffineMapAttr>(indexingMaps[index]);
      if (!sourceType || !inputMapAttr)
        return failPattern(
            rewriter, op, failureReason,
            "tile.elementwise indexing map materialization requires memref "
            "inputs and affine maps");

      mlir::AffineMap inputMap = inputMapAttr.getValue();
      if (inputMap.getNumDims() != resultType.getRank() ||
          inputMap.getNumSymbols() != 0 ||
          inputMap.getNumResults() != sourceType.getRank() ||
          !inputMap.isProjectedPermutation())
        return failPattern(
            rewriter, op, failureReason,
            "tile.elementwise input indexing map must be a projected "
            "permutation of result dimensions");

      if (inputMap.isIdentity() &&
          sourceType.getShape() == resultType.getShape()) {
        movementPlans.push_back(std::move(plan));
        continue;
      }

      plan.materializedType = mlir::MemRefType::get(
          resultType.getShape(), sourceType.getElementType(),
          resultType.getLayout(), resultType.getMemorySpace());
      auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> resultIndices)
          -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
        llvm::SmallVector<int64_t> sourceIndices;
        sourceIndices.reserve(inputMap.getNumResults());
        for (mlir::AffineExpr expr : inputMap.getResults()) {
          auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
          if (!dimExpr || dimExpr.getPosition() >= resultIndices.size())
            return mlir::failure();
          sourceIndices.push_back(resultIndices[dimExpr.getPosition()]);
        }
        return sourceIndices;
      };
      auto destIndexFn = [](llvm::ArrayRef<int64_t> resultIndices)
          -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
        return llvm::SmallVector<int64_t>(resultIndices.begin(),
                                          resultIndices.end());
      };
      mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
          getStaticMappedMovementSegments(
              rewriter, op, sourceType, plan.materializedType,
              resultType.getShape(), sourceIndexFn, destIndexFn, failureReason,
              "tile.elementwise indexing map materialization");
      if (mlir::failed(segments))
        return mlir::failure();
      plan.segments = std::move(*segments);
      movementPlans.push_back(std::move(plan));
    }

    InstrElementwiseKindAttr instrKind;
    if (op.getKind() != ComputeElementwiseKind::Select) {
      mlir::FailureOr<InstrElementwiseKindAttr> resolvedInstrKind =
          getInstrElementwiseKindAttr(rewriter, op, op.getKindAttr(),
                                      failureReason);
      if (mlir::failed(resolvedInstrKind))
        return mlir::failure();
      instrKind = *resolvedInstrKind;
    }

    // Select has a movement-based target sequence. Preflight its copy
    // descriptors as well, still before emitting any effect.
    mlir::FailureOr<MovementDescriptor> falseDescriptor;
    mlir::FailureOr<MovementDescriptor> selectDestDescriptor;
    if (op.getKind() == ComputeElementwiseKind::Select) {
      mlir::Type falseType = movementPlans[2].source.getType();
      if (movementPlans[2].materializedType)
        falseType = movementPlans[2].materializedType;
      falseDescriptor =
          getContiguousDescriptor(rewriter, op, falseType, failureReason);
      selectDestDescriptor =
          getContiguousDescriptor(rewriter, op, resultType, failureReason);
      if (mlir::failed(falseDescriptor) || mlir::failed(selectDestDescriptor) ||
          falseDescriptor->byteCount != selectDestDescriptor->byteCount)
        return failPattern(
            rewriter, op, failureReason,
            "target select lowering requires equal contiguous false and "
            "destination payloads");
    }

    llvm::SmallVector<mlir::Value, 3> inputs;
    inputs.reserve(movementPlans.size());
    bool materializedMappedInput = false;
    for (const InputMovementPlan &plan : movementPlans) {
      if (!plan.materializedType) {
        inputs.push_back(plan.source);
        continue;
      }
      mlir::Value materialized =
          rewriter
              .create<mlir::memref::AllocOp>(op.getLoc(), plan.materializedType)
              .getResult();
      createGatherScatterSegments(rewriter, op.getLoc(), plan.source,
                                  materialized, plan.segments);
      inputs.push_back(materialized);
      materializedMappedInput = true;
    }
    if (materializedMappedInput)
      rewriter.create<SyncLocalFenceOp>(op.getLoc());

    mlir::Value dest =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), resultType)
            .getResult();

    if (op.getKind() == ComputeElementwiseKind::Select) {
      createGatherScatter(rewriter, op.getLoc(), inputs[2], dest,
                          *falseDescriptor, *selectDestDescriptor);
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
      mlir::Value mask =
          rewriter.create<mlir::memref::AllocOp>(op.getLoc(), resultType)
              .getResult();
      rewriter.create<InstrBit2FpOp>(op.getLoc(), inputs[0], mask);
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
      rewriter.create<InstrMaskMoveOp>(op.getLoc(), inputs[1], mask, dest);
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
      rewriter.replaceOp(op, dest);
      return mlir::success();
    }

    rewriter.create<InstrElementwiseOp>(op.getLoc(), instrKind, inputs, dest);
    if (materializedMappedInput)
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
    rewriter.replaceOp(op, dest);
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
    auto inputType =
        mlir::dyn_cast<mlir::MemRefType>(op.getInput().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!inputType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce lowering requires memref operands");
    if (inputType.getElementType() != resultType.getElementType())
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering requires matching element types");

    std::optional<int64_t> inputElementCount =
        getStaticPositiveElementCount(inputType.getShape());
    std::optional<int64_t> resultElementCount =
        getStaticPositiveElementCount(resultType.getShape());
    if (!inputElementCount || !resultElementCount)
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering requires static positive input/result shapes");
    if (static_cast<uint64_t>(*resultElementCount) >
        std::numeric_limits<uint32_t>::max())
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering result element count exceeds uint32 target "
          "field");

    mlir::Type elementType = inputType.getElementType();
    unsigned scalarWidth = 0;
    if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType))
      scalarWidth = integerType.getWidth();
    else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
      scalarWidth = floatType.getWidth();
    else
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering requires target-encodable integer or float "
          "elements");
    if (scalarWidth > 32)
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering element width exceeds uint32 scalar ABI");
    std::optional<WaferPhysicalTensorInfo> inputPhysical =
        wafer::computeWaferPhysicalTensorInfo(inputType);
    std::optional<WaferPhysicalTensorInfo> resultPhysical =
        wafer::computeWaferPhysicalTensorInfo(resultType);
    if (!inputPhysical || !resultPhysical ||
        inputPhysical->bitPackedElement || resultPhysical->bitPackedElement ||
        inputPhysical->elementBytes <= 0 || resultPhysical->elementBytes <= 0)
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering requires byte-addressable elements");

    bool targetEncodableElement =
        (mlir::isa<mlir::IntegerType>(elementType) &&
         (scalarWidth == 8 || scalarWidth == 16 || scalarWidth == 32)) ||
        mlir::isa<mlir::Float16Type, mlir::BFloat16Type,
                  mlir::Float32Type>(elementType);
    if (!targetEncodableElement)
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce element type is not encodable by the target "
          "data-format ABI");

    mlir::FailureOr<InstrElementwiseKindAttr> accumulationKind =
        getAccumulationElementwiseKind(rewriter, op, op.getKindAttr(),
                                       failureReason, "tile.reduce");
    if (mlir::failed(accumulationKind))
      return mlir::failure();

    mlir::Attribute initValue = op->getAttr("init_value");
    bool hasInitOperand = static_cast<bool>(op.getInit());
    if (hasInitOperand == static_cast<bool>(initValue))
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce lowering requires exactly one constant init source");
    if (initValue) {
      auto typedInit = mlir::dyn_cast<mlir::TypedAttr>(initValue);
      if (!typedInit || typedInit.getType() != elementType)
        return failPattern(
            rewriter, op, failureReason,
            "tile.reduce init_value type must match input element type");
    } else {
      auto constant = op.getInit().getDefiningOp<mlir::arith::ConstantOp>();
      auto typedInit = constant
                           ? mlir::dyn_cast<mlir::TypedAttr>(constant.getValue())
                           : mlir::TypedAttr{};
      if (!constant || !typedInit || typedInit.getType() != elementType)
        return failPattern(
            rewriter, op, failureReason,
            "tile.reduce init operand must be a matching arith.constant");
    }

    auto dimensionsAttr =
        op->getAttrOfType<mlir::DenseI64ArrayAttr>("dimensions");
    if (!dimensionsAttr)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce lowering requires dimensions attr");

    llvm::SmallVector<int64_t, 4> reducedDims(
        dimensionsAttr.asArrayRef().begin(),
        dimensionsAttr.asArrayRef().end());
    if (reducedDims.empty())
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce lowering requires non-empty dimensions");
    llvm::sort(reducedDims);
    if (std::adjacent_find(reducedDims.begin(), reducedDims.end()) !=
        reducedDims.end())
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce lowering dimensions must be unique");
    for (int64_t dim : reducedDims)
      if (dim < 0 || dim >= inputType.getRank())
        return failPattern(
            rewriter, op, failureReason,
            "tile.reduce lowering dimension is outside input rank");

    llvm::SmallVector<int64_t, 4> nonReducedDims;
    llvm::SmallVector<int64_t, 4> reductionShape;
    for (int64_t inputDim = 0; inputDim < inputType.getRank(); ++inputDim) {
      if (llvm::is_contained(reducedDims, inputDim))
        reductionShape.push_back(inputType.getDimSize(inputDim));
      else
        nonReducedDims.push_back(inputDim);
    }
    if (static_cast<int64_t>(nonReducedDims.size()) != resultType.getRank())
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce result rank does not match non-reduced dimensions");
    for (auto [resultDim, inputDim] : llvm::enumerate(nonReducedDims))
      if (resultType.getDimSize(resultDim) != inputType.getDimSize(inputDim))
        return failPattern(
            rewriter, op, failureReason,
            "tile.reduce result shape does not match non-reduced dimensions");

    std::optional<int64_t> reductionTupleCount =
        getStaticPositiveElementCount(reductionShape);
    if (!reductionTupleCount)
      return failPattern(
          rewriter, op, failureReason,
          "tile.reduce reduction tuple count overflows or is not positive");
    constexpr uint64_t budget =
        wafer::detail::kStaticTerminalOperationBudget;
    if (static_cast<uint64_t>(*reductionTupleCount) > (budget - 4) / 4)
      return failPattern(
          rewriter, op, failureReason,
          "static_terminal_budget_exceeded: ordered tile.reduce minimum "
          "terminal operation count exceeds 4096");

    auto tensorType = mlir::MemRefType::get(
        resultType.getShape(), elementType, mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(rewriter.getContext(), MemorySpace::SPM,
                        MemLayout::Tensor));

    struct SlicePlan {
      llvm::SmallVector<LogicalMovementSegment> segments;
    };
    llvm::SmallVector<SlicePlan, 8> slicePlans;
    slicePlans.reserve(static_cast<size_t>(*reductionTupleCount));
    uint64_t terminalOperationCount = 2; // Initial fill and completion.

    for (int64_t linearTuple = 0; linearTuple < *reductionTupleCount;
         ++linearTuple) {
      mlir::FailureOr<llvm::SmallVector<int64_t>> tuple = delinearizeIndex(
          rewriter, op, reductionShape, linearTuple, failureReason,
          "tile.reduce ordered tuple");
      if (mlir::failed(tuple))
        return mlir::failure();

      auto sourceIndexFn =
          [&](llvm::ArrayRef<int64_t> resultIndices)
          -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
        llvm::SmallVector<int64_t> sourceIndices(inputType.getRank(), 0);
        size_t reducedIndex = 0;
        size_t resultIndex = 0;
        for (int64_t inputDim = 0; inputDim < inputType.getRank(); ++inputDim) {
          if (llvm::is_contained(reducedDims, inputDim))
            sourceIndices[inputDim] = (*tuple)[reducedIndex++];
          else
            sourceIndices[inputDim] = resultIndices[resultIndex++];
        }
        return sourceIndices;
      };
      auto destIndexFn =
          [](llvm::ArrayRef<int64_t> resultIndices)
          -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
        return llvm::SmallVector<int64_t>(resultIndices.begin(),
                                          resultIndices.end());
      };

      mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
          getStaticMappedMovementSegments(
              rewriter, op, inputType, tensorType, resultType.getShape(),
              sourceIndexFn, destIndexFn, failureReason,
              "tile.reduce ordered slice movement");
      if (mlir::failed(segments))
        return mlir::failure();
      mlir::FailureOr<uint64_t> commandCount =
          preflightPackedMovementCommands(
              rewriter, op, *segments, failureReason,
              "tile.reduce ordered slice movement");
      if (mlir::failed(commandCount))
        return mlir::failure();
      if (*commandCount > budget - terminalOperationCount ||
          3 > budget - terminalOperationCount - *commandCount)
        return failPattern(
            rewriter, op, failureReason,
            "static_terminal_budget_exceeded: ordered tile.reduce terminal "
            "operation count exceeds 4096");
      terminalOperationCount += *commandCount + 3;
      slicePlans.push_back({std::move(*segments)});
    }

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> finalSegments =
        getStaticLogicalMovementSegments(
            rewriter, op, tensorType, resultType, failureReason,
            "tile.reduce final logical movement");
    if (mlir::failed(finalSegments))
      return mlir::failure();
    mlir::FailureOr<uint64_t> finalCommandCount =
        preflightPackedMovementCommands(
            rewriter, op, *finalSegments, failureReason,
            "tile.reduce final logical movement");
    if (mlir::failed(finalCommandCount))
      return mlir::failure();
    if (*finalCommandCount > budget - terminalOperationCount ||
        1 > budget - terminalOperationCount - *finalCommandCount)
      return failPattern(
          rewriter, op, failureReason,
          "static_terminal_budget_exceeded: ordered tile.reduce terminal "
          "operation count exceeds 4096");

    // All legality, geometry, packing and budget checks above are deliberately
    // completed before creating any effectful instruction.
    mlir::Value init = op.getInit();
    if (!init)
      init = rewriter
                 .create<mlir::arith::ConstantOp>(
                     op.getLoc(), mlir::cast<mlir::TypedAttr>(initValue))
                 .getResult();
    mlir::Value accumulatorA =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), tensorType);
    mlir::Value accumulatorB =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), tensorType);
    mlir::Value slice =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), tensorType);
    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), resultType, rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    rewriter.create<InstrFillOp>(op.getLoc(), accumulatorA, init);
    rewriter.create<SyncLocalFenceOp>(op.getLoc());
    mlir::Value currentAccumulator = accumulatorA;
    mlir::Value nextAccumulator = accumulatorB;
    for (const SlicePlan &plan : slicePlans) {
      createGatherScatterSegments(rewriter, op.getLoc(), op.getInput(), slice,
                                  plan.segments);
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
      llvm::SmallVector<mlir::Value, 2> inputs{currentAccumulator, slice};
      rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                          inputs, nextAccumulator);
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
      std::swap(currentAccumulator, nextAccumulator);
    }
    createGatherScatterSegments(rewriter, op.getLoc(), currentAccumulator,
                                *dest, *finalSegments);
    rewriter.create<SyncLocalFenceOp>(op.getLoc());
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
        inferGemmMKN(op, rewriter, failureReason);
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

static mlir::FailureOr<int64_t>
inferAllGatherAxis(mlir::PatternRewriter &rewriter, CommAllGatherOp op,
                   mlir::MemRefType localType, mlir::MemRefType gatherType,
                   std::string *failureReason) {
  if (localType.getRank() != gatherType.getRank())
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        "tile.all_gather lowering requires equal-rank local and gather "
        "buffers");
  if (localType.getRank() == 0)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        "tile.all_gather lowering requires a non-scalar gather buffer");
  if (localType.getElementType() != gatherType.getElementType())
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        "tile.all_gather lowering requires matching element types");
  MemoryAttr localMemory = wafer::getWaferMemoryAttr(localType);
  MemoryAttr gatherMemory = wafer::getWaferMemoryAttr(gatherType);
  if (!localMemory || !gatherMemory ||
      localMemory.getSpace() != MemorySpace::SPM ||
      gatherMemory.getSpace() != MemorySpace::SPM ||
      localMemory.getLayout() != gatherMemory.getLayout() ||
      !isStandardViewCompatibleLayout(localMemory.getLayout()))
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        "tile.all_gather lowering requires tensor or ntensor SPM layouts");

  int64_t groupSize = op.getGroupSizeAttr().getInt();
  int64_t axis = -1;
  for (int64_t dim = 0; dim < localType.getRank(); ++dim) {
    int64_t localDim = localType.getDimSize(dim);
    int64_t gatherDim = gatherType.getDimSize(dim);
    if (localDim == mlir::ShapedType::kDynamic ||
        gatherDim == mlir::ShapedType::kDynamic)
      return failFailureOr<int64_t>(
          rewriter, op, failureReason,
          "tile.all_gather lowering requires static buffer shapes");

    std::optional<int64_t> expectedGatherDim =
        checkedMulI64(localDim, groupSize);
    if (!expectedGatherDim)
      return failFailureOr<int64_t>(
          rewriter, op, failureReason,
          "tile.all_gather lowering axis size overflows");

    if (gatherDim == *expectedGatherDim) {
      if (axis >= 0)
        return failFailureOr<int64_t>(
            rewriter, op, failureReason,
            "tile.all_gather lowering requires a unique gather axis");
      axis = dim;
      continue;
    }

    if (gatherDim != localDim)
      return failFailureOr<int64_t>(
          rewriter, op, failureReason,
          "tile.all_gather gather buffer shape must match local shape except "
          "on the gathered axis");
  }

  if (axis < 0)
    return failFailureOr<int64_t>(
        rewriter, op, failureReason,
        "tile.all_gather lowering could not infer gather axis");
  return axis;
}

static mlir::FailureOr<mlir::Value>
createAxisSlotView(mlir::PatternRewriter &rewriter, mlir::Location loc,
                   mlir::Operation *op, mlir::MemRefType slotType,
                   mlir::Value fullBuffer, int64_t axis, int64_t slot,
                   std::string *failureReason, llvm::StringRef opLabel) {
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides;
  offsets.reserve(slotType.getRank());
  sizes.reserve(slotType.getRank());
  strides.reserve(slotType.getRank());

  for (int64_t dim = 0; dim < slotType.getRank(); ++dim) {
    int64_t size = slotType.getDimSize(dim);
    if (size == mlir::ShapedType::kDynamic)
      return failFailureOr<mlir::Value>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" slot view requires static local shape")
              .str());
    int64_t offset = dim == axis ? slot * size : 0;
    offsets.push_back(rewriter.getIndexAttr(offset));
    sizes.push_back(rewriter.getIndexAttr(size));
    strides.push_back(rewriter.getIndexAttr(1));
  }

  return rewriter
      .create<mlir::memref::SubViewOp>(loc, fullBuffer, offsets, sizes, strides)
      .getResult();
}

static mlir::LogicalResult
createContiguousSPMCopy(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Operation *op, mlir::Value source,
                        mlir::Value dest, std::string *failureReason,
                        llvm::StringRef role) {
  mlir::FailureOr<MovementDescriptor> sourceDescriptor =
      getContiguousDescriptor(rewriter, op, source.getType(), failureReason);
  mlir::FailureOr<MovementDescriptor> destDescriptor =
      getContiguousDescriptor(rewriter, op, dest.getType(), failureReason);
  if (mlir::failed(sourceDescriptor) || mlir::failed(destDescriptor))
    return mlir::failure();
  if (sourceDescriptor->byteCount != destDescriptor->byteCount)
    return failPattern(
        rewriter, op, failureReason,
        llvm::Twine(role)
            .concat(" requires equal source and destination byte counts")
            .str());

  createGatherScatter(rewriter, loc, source, dest, *sourceDescriptor,
                      *destDescriptor);
  return mlir::success();
}

static mlir::LogicalResult
createLogicalSPMCopy(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     mlir::Operation *op, mlir::Value source, mlir::Value dest,
                     std::string *failureReason, llvm::StringRef role) {
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto destType = mlir::dyn_cast<mlir::MemRefType>(dest.getType());
  if (!sourceType || !destType)
    return failPattern(
        rewriter, op, failureReason,
        llvm::Twine(role).concat(" requires memref operands").str());
  mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
      getStaticLogicalMovementSegments(rewriter, op, sourceType, destType,
                                       failureReason, role);
  if (mlir::failed(segments))
    return mlir::failure();
  createGatherScatterSegments(rewriter, loc, source, dest, *segments);
  return mlir::success();
}

static mlir::MemRefType getContiguousSPMBufferType(mlir::MemRefType type) {
  return mlir::MemRefType::get(type.getShape(), type.getElementType(),
                               mlir::MemRefLayoutAttrInterface{},
                               type.getMemorySpace());
}

static mlir::FailureOr<InstrElementwiseKindAttr> getInstrElementwiseKindAttr(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    ComputeElementwiseKindAttr computeKind, std::string *failureReason) {
  InstrElementwiseKind instrKind;
  switch (computeKind.getValue()) {
  case ComputeElementwiseKind::Add:
    instrKind = InstrElementwiseKind::Add;
    break;
  case ComputeElementwiseKind::Sub:
    instrKind = InstrElementwiseKind::Sub;
    break;
  case ComputeElementwiseKind::Mul:
    instrKind = InstrElementwiseKind::Mul;
    break;
  case ComputeElementwiseKind::Div:
    instrKind = InstrElementwiseKind::Div;
    break;
  case ComputeElementwiseKind::Max:
    instrKind = InstrElementwiseKind::Max;
    break;
  case ComputeElementwiseKind::Min:
    instrKind = InstrElementwiseKind::Min;
    break;
  case ComputeElementwiseKind::Neg:
    instrKind = InstrElementwiseKind::Neg;
    break;
  case ComputeElementwiseKind::Recip:
    instrKind = InstrElementwiseKind::Recip;
    break;
  case ComputeElementwiseKind::Sqrt:
    instrKind = InstrElementwiseKind::Sqrt;
    break;
  case ComputeElementwiseKind::Rsqrt:
    instrKind = InstrElementwiseKind::Rsqrt;
    break;
  case ComputeElementwiseKind::Exp:
    instrKind = InstrElementwiseKind::Exp;
    break;
  case ComputeElementwiseKind::Tanh:
    instrKind = InstrElementwiseKind::Tanh;
    break;
  case ComputeElementwiseKind::Eq:
    instrKind = InstrElementwiseKind::Eq;
    break;
  case ComputeElementwiseKind::Ne:
    instrKind = InstrElementwiseKind::Ne;
    break;
  case ComputeElementwiseKind::Lt:
    instrKind = InstrElementwiseKind::Lt;
    break;
  case ComputeElementwiseKind::Le:
    instrKind = InstrElementwiseKind::Le;
    break;
  case ComputeElementwiseKind::Gt:
    instrKind = InstrElementwiseKind::Gt;
    break;
  case ComputeElementwiseKind::Ge:
    instrKind = InstrElementwiseKind::Ge;
    break;
  case ComputeElementwiseKind::Select:
    return failFailureOr<InstrElementwiseKindAttr>(
        rewriter, op, failureReason,
        "tile.elementwise select must lower to target movement sequence before "
        "instruction elementwise");
  }
  return InstrElementwiseKindAttr::get(rewriter.getContext(), instrKind);
}

static mlir::FailureOr<InstrElementwiseKindAttr> getAccumulationElementwiseKind(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    ComputeReduceKindAttr reduceKind, std::string *failureReason,
    llvm::StringRef opLabel) {
  InstrElementwiseKind elementwiseKind;
  switch (reduceKind.getValue()) {
  case ComputeReduceKind::Sum:
    elementwiseKind = InstrElementwiseKind::Add;
    break;
  case ComputeReduceKind::Max:
    elementwiseKind = InstrElementwiseKind::Max;
    break;
  case ComputeReduceKind::Min:
    elementwiseKind = InstrElementwiseKind::Min;
    break;
  case ComputeReduceKind::Avg:
    return failFailureOr<InstrElementwiseKindAttr>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" lowering does not support avg accumulation")
            .str());
  }

  return InstrElementwiseKindAttr::get(rewriter.getContext(), elementwiseKind);
}

static int64_t getHighestTreeMask(int64_t groupSize) {
  int64_t mask = 1;
  while (mask < groupSize)
    mask <<= 1;
  return mask >> 1;
}

static int64_t getLowestSetBit(int64_t value) { return value & -value; }

class AllGatherLowering : public mlir::OpRewritePattern<CommAllGatherOp> {
public:
  AllGatherLowering(mlir::MLIRContext *context, std::string *failureReason,
                    AllGatherSchedule schedule)
      : mlir::OpRewritePattern<CommAllGatherOp>(context),
        failureReason(failureReason), schedule(schedule) {}

  mlir::LogicalResult
  matchAndRewrite(CommAllGatherOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto localType =
        mlir::dyn_cast<mlir::MemRefType>(op.getLocalChunk().getType());
    auto gatherType =
        mlir::dyn_cast<mlir::MemRefType>(op.getGatherBuffer().getType());
    if (!localType || !gatherType)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_gather lowering requires memref buffers");

    int64_t groupSize = op.getGroupSizeAttr().getInt();
    int64_t localRank = op.getLocalRankAttr().getInt();
    llvm::ArrayRef<int64_t> rankGroup = op.getRankGroupAttr().asArrayRef();
    if (groupSize <= 1 || localRank < 0 || localRank >= groupSize ||
        static_cast<int64_t>(rankGroup.size()) != groupSize)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_gather lowering requires valid rank facts");

    mlir::FailureOr<int64_t> axis =
        inferAllGatherAxis(rewriter, op, localType, gatherType, failureReason);
    if (mlir::failed(axis))
      return mlir::failure();

    llvm::SmallVector<mlir::Value> slots(groupSize);
    auto getSlot = [&](int64_t slot) -> mlir::FailureOr<mlir::Value> {
      if (slots[slot])
        return slots[slot];
      mlir::FailureOr<mlir::Value> view = createAxisSlotView(
          rewriter, op.getLoc(), op, localType, op.getGatherBuffer(), *axis,
          slot, failureReason, "tile.all_gather");
      if (mlir::failed(view))
        return mlir::failure();
      slots[slot] = *view;
      return slots[slot];
    };

    mlir::FailureOr<mlir::Value> localSlot = getSlot(localRank);
    if (mlir::failed(localSlot))
      return mlir::failure();

    mlir::MemRefType commSlotType = getContiguousSPMBufferType(localType);
    auto localCommSlot =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), commSlotType);
    if (mlir::failed(
            createLogicalSPMCopy(rewriter, op.getLoc(), op, op.getLocalChunk(),
                                 localCommSlot.getResult(), failureReason,
                                 "tile.all_gather local contiguous copy")))
      return mlir::failure();
    if (mlir::failed(createLogicalSPMCopy(
            rewriter, op.getLoc(), op, localCommSlot.getResult(), *localSlot,
            failureReason, "tile.all_gather local slot copy")))
      return mlir::failure();
    rewriter.create<SyncLocalFenceOp>(op.getLoc());

    int64_t bytes = op.getBytesAttr().getInt();
    if (schedule == AllGatherSchedule::Direct) {
      for (int64_t distance = 1; distance < groupSize; ++distance) {
        int64_t sendPeerIndex = (localRank + distance) % groupSize;
        int64_t recvPeerIndex = (localRank + groupSize - distance) % groupSize;
        mlir::FailureOr<mlir::Value> recvSlot = getSlot(recvPeerIndex);
        if (mlir::failed(recvSlot))
          return mlir::failure();

        auto recvCommSlot =
            rewriter.create<mlir::memref::AllocOp>(op.getLoc(), commSlotType);
        auto sendMessage = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllGatherDirect, distance, localRank);
        auto recvMessage = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllGatherDirect, distance, recvPeerIndex);
        auto send = rewriter.create<InstrDTESendOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            localCommSlot.getResult(),
            rewriter.getI64IntegerAttr(rankGroup[sendPeerIndex]),
            rewriter.getI64IntegerAttr(bytes), sendMessage,
            DirectDTEBindingAttr());
        auto recv = rewriter.create<InstrDTERecvOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            recvCommSlot.getResult(),
            rewriter.getI64IntegerAttr(rankGroup[recvPeerIndex]),
            rewriter.getI64IntegerAttr(bytes), recvMessage,
            DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                                 recv.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
        if (mlir::failed(createLogicalSPMCopy(
                rewriter, op.getLoc(), op, recvCommSlot.getResult(), *recvSlot,
                failureReason, "tile.all_gather received slot copy")))
          return mlir::failure();
      }

      rewriter.eraseOp(op);
      return mlir::success();
    }

    int64_t nextPeer = rankGroup[(localRank + 1) % groupSize];
    int64_t prevPeer = rankGroup[(localRank + groupSize - 1) % groupSize];
    mlir::Value sendSlot = localCommSlot.getResult();
    for (int64_t step = 0; step < groupSize - 1; ++step) {
      int64_t recvSlotIndex = (localRank + groupSize - step - 1) % groupSize;
      mlir::FailureOr<mlir::Value> recvSlot = getSlot(recvSlotIndex);
      if (mlir::failed(recvSlot))
        return mlir::failure();

      auto recvCommSlot =
          rewriter.create<mlir::memref::AllocOp>(op.getLoc(), commSlotType);
      int64_t sendPayloadSlice = (localRank + groupSize - step) % groupSize;
      auto sendMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllGatherRing, step, sendPayloadSlice);
      auto recvMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllGatherRing, step, recvSlotIndex);
      auto send = rewriter.create<InstrDTESendOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(), sendSlot,
          rewriter.getI64IntegerAttr(nextPeer),
          rewriter.getI64IntegerAttr(bytes), sendMessage,
          DirectDTEBindingAttr());
      auto recv = rewriter.create<InstrDTERecvOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
          recvCommSlot.getResult(), rewriter.getI64IntegerAttr(prevPeer),
          rewriter.getI64IntegerAttr(bytes), recvMessage,
          DirectDTEBindingAttr());
      llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                               recv.getToken()};
      rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
      if (mlir::failed(createLogicalSPMCopy(
              rewriter, op.getLoc(), op, recvCommSlot.getResult(), *recvSlot,
              failureReason, "tile.all_gather received slot copy")))
        return mlir::failure();
      sendSlot = recvCommSlot.getResult();
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  std::string *failureReason;
  AllGatherSchedule schedule;
};

class ReduceScatterLowering
    : public mlir::OpRewritePattern<CommReduceScatterOp> {
public:
  ReduceScatterLowering(mlir::MLIRContext *context, std::string *failureReason,
                        ReduceScatterSchedule schedule)
      : mlir::OpRewritePattern<CommReduceScatterOp>(context),
        failureReason(failureReason), schedule(schedule) {}

  mlir::LogicalResult
  matchAndRewrite(CommReduceScatterOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto inputType = mlir::dyn_cast<mlir::MemRefType>(op.getInput().getType());
    auto recvType =
        mlir::dyn_cast<mlir::MemRefType>(op.getRecvBuffer().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!inputType || !recvType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce_scatter lowering requires memref "
                         "buffers");
    if (recvType != resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce_scatter lowering requires matching "
                         "recv/result buffer types");

    MemoryAttr inputMemory = wafer::getWaferMemoryAttr(inputType);
    MemoryAttr recvMemory = wafer::getWaferMemoryAttr(recvType);
    MemoryAttr resultMemory = wafer::getWaferMemoryAttr(resultType);
    if (!inputMemory || !recvMemory || !resultMemory ||
        inputMemory.getSpace() != MemorySpace::SPM ||
        recvMemory.getSpace() != MemorySpace::SPM ||
        resultMemory.getSpace() != MemorySpace::SPM ||
        inputMemory.getLayout() != MemLayout::Tensor ||
        recvMemory.getLayout() != MemLayout::Tensor ||
        resultMemory.getLayout() != MemLayout::Tensor)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce_scatter lowering requires tensor SPM "
                         "buffers");

    int64_t groupSize = op.getGroupSizeAttr().getInt();
    int64_t localRank = op.getLocalRankAttr().getInt();
    llvm::ArrayRef<int64_t> rankGroup = op.getRankGroupAttr().asArrayRef();
    if (groupSize <= 1 || localRank < 0 || localRank >= groupSize ||
        static_cast<int64_t>(rankGroup.size()) != groupSize)
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce_scatter lowering requires valid rank "
                         "facts");

    int64_t axis = op.getAxisAttr().getInt();
    if (axis < 0 || axis >= inputType.getRank() ||
        inputType.getRank() != resultType.getRank())
      return failPattern(rewriter, op, failureReason,
                         "tile.reduce_scatter lowering requires valid axis");
    for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
      int64_t inputDim = inputType.getDimSize(dim);
      int64_t resultDim = resultType.getDimSize(dim);
      if (inputDim == mlir::ShapedType::kDynamic ||
          resultDim == mlir::ShapedType::kDynamic)
        return failPattern(rewriter, op, failureReason,
                           "tile.reduce_scatter lowering requires static "
                           "buffer shapes");
      if (dim == axis) {
        std::optional<int64_t> expectedInputDim =
            checkedMulI64(resultDim, groupSize);
        if (!expectedInputDim || inputDim != *expectedInputDim)
          return failPattern(rewriter, op, failureReason,
                             "tile.reduce_scatter input axis size must equal "
                             "result axis size times group_size");
        continue;
      }
      if (inputDim != resultDim)
        return failPattern(rewriter, op, failureReason,
                           "tile.reduce_scatter non-axis dimensions must "
                           "match");
    }

    mlir::FailureOr<InstrElementwiseKindAttr> accumulationKind =
        getAccumulationElementwiseKind(rewriter, op, op.getKindAttr(),
                                       failureReason, "tile.reduce_scatter");
    if (mlir::failed(accumulationKind))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> accumulator = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(accumulator))
      return mlir::failure();

    llvm::SmallVector<mlir::Value> inputSlots(groupSize);
    auto getInputSlot = [&](int64_t slot) -> mlir::FailureOr<mlir::Value> {
      if (inputSlots[slot])
        return inputSlots[slot];
      mlir::FailureOr<mlir::Value> view = createAxisSlotView(
          rewriter, op.getLoc(), op, resultType, op.getInput(), axis, slot,
          failureReason, "tile.reduce_scatter");
      if (mlir::failed(view))
        return mlir::failure();
      inputSlots[slot] = *view;
      return inputSlots[slot];
    };

    mlir::FailureOr<mlir::Value> localSlot = getInputSlot(localRank);
    if (mlir::failed(localSlot))
      return mlir::failure();
    if (mlir::failed(createContiguousSPMCopy(
            rewriter, op.getLoc(), op, *localSlot, *accumulator, failureReason,
            "tile.reduce_scatter accumulator init")))
      return mlir::failure();
    rewriter.create<SyncLocalFenceOp>(op.getLoc());

    switch (schedule) {
    case ReduceScatterSchedule::Direct:
      break;
    }

    int64_t bytes = op.getBytesAttr().getInt();
    for (int64_t distance = 1; distance < groupSize; ++distance) {
      int64_t sendSlotIndex = (localRank + distance) % groupSize;
      int64_t recvRankIndex = (localRank + groupSize - distance) % groupSize;
      mlir::FailureOr<mlir::Value> sendSlot = getInputSlot(sendSlotIndex);
      if (mlir::failed(sendSlot))
        return mlir::failure();

      auto sendMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::ReduceScatterDirect, distance, sendSlotIndex);
      auto recvMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::ReduceScatterDirect, distance, localRank);
      auto send = rewriter.create<InstrDTESendOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(), *sendSlot,
          rewriter.getI64IntegerAttr(rankGroup[sendSlotIndex]),
          rewriter.getI64IntegerAttr(bytes), sendMessage,
          DirectDTEBindingAttr());
      auto recv = rewriter.create<InstrDTERecvOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
          op.getRecvBuffer(),
          rewriter.getI64IntegerAttr(rankGroup[recvRankIndex]),
          rewriter.getI64IntegerAttr(bytes), recvMessage,
          DirectDTEBindingAttr());
      llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                               recv.getToken()};
      rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);

      llvm::SmallVector<mlir::Value, 2> inputs{*accumulator,
                                               op.getRecvBuffer()};
      rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                          inputs, *accumulator);
    }

    rewriter.replaceOp(op, *accumulator);
    return mlir::success();
  }

private:
  std::string *failureReason;
  ReduceScatterSchedule schedule;
};

class AllReduceLowering : public mlir::OpRewritePattern<CommAllReduceOp> {
public:
  AllReduceLowering(mlir::MLIRContext *context, std::string *failureReason,
                    AllReduceSchedule schedule)
      : mlir::OpRewritePattern<CommAllReduceOp>(context),
        failureReason(failureReason), schedule(schedule) {}

  mlir::LogicalResult
  matchAndRewrite(CommAllReduceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto inputType = mlir::dyn_cast<mlir::MemRefType>(op.getInput().getType());
    auto recvType =
        mlir::dyn_cast<mlir::MemRefType>(op.getRecvBuffer().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!inputType || !recvType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_reduce lowering requires memref buffers");
    if (inputType != recvType || inputType != resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_reduce lowering requires matching buffer "
                         "types");

    MemoryAttr inputMemory = wafer::getWaferMemoryAttr(inputType);
    if (!inputMemory || inputMemory.getSpace() != MemorySpace::SPM ||
        inputMemory.getLayout() != MemLayout::Tensor)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_reduce lowering requires tensor SPM "
                         "buffers");

    int64_t groupSize = op.getGroupSizeAttr().getInt();
    int64_t localRank = op.getLocalRankAttr().getInt();
    llvm::ArrayRef<int64_t> rankGroup = op.getRankGroupAttr().asArrayRef();
    if (groupSize <= 1 || localRank < 0 || localRank >= groupSize ||
        static_cast<int64_t>(rankGroup.size()) != groupSize)
      return failPattern(rewriter, op, failureReason,
                         "tile.all_reduce lowering requires valid rank facts");

    mlir::FailureOr<InstrElementwiseKindAttr> accumulationKind =
        getAccumulationElementwiseKind(rewriter, op, op.getKindAttr(),
                                       failureReason, "tile.all_reduce");
    if (mlir::failed(accumulationKind))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> accumulator = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(accumulator))
      return mlir::failure();

    int64_t bytes = op.getBytesAttr().getInt();
    if (schedule == AllReduceSchedule::Tree) {
      if (mlir::failed(createContiguousSPMCopy(
              rewriter, op.getLoc(), op, op.getInput(), *accumulator,
              failureReason, "tile.all_reduce accumulator init")))
        return mlir::failure();
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
      for (int64_t mask = 1; mask < groupSize; mask <<= 1) {
        if ((localRank & mask) != 0) {
          int64_t parentRank = localRank ^ mask;
          auto message = DTEMessageAttr::get(
              rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
              DTEProtocolPhase::AllReduceTreeReduce, mask, localRank);
          auto send = rewriter.create<InstrDTESendOp>(
              op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
              *accumulator, rewriter.getI64IntegerAttr(rankGroup[parentRank]),
              rewriter.getI64IntegerAttr(bytes), message,
              DirectDTEBindingAttr());
          llvm::SmallVector<mlir::Value, 1> tokens{send.getToken()};
          rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
          break;
        }

        int64_t childRank = localRank | mask;
        if (childRank >= groupSize)
          continue;
        auto message = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllReduceTreeReduce, mask, childRank);
        auto recv = rewriter.create<InstrDTERecvOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            op.getRecvBuffer(),
            rewriter.getI64IntegerAttr(rankGroup[childRank]),
            rewriter.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 1> tokens{recv.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);

        llvm::SmallVector<mlir::Value, 2> inputs{*accumulator,
                                                 op.getRecvBuffer()};
        rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                            inputs, *accumulator);
        rewriter.create<SyncLocalFenceOp>(op.getLoc());
      }

      bool hasFinalResult = localRank == 0;
      int64_t receiveMask = localRank == 0 ? 0 : getLowestSetBit(localRank);
      for (int64_t mask = getHighestTreeMask(groupSize); mask >= 1;
           mask >>= 1) {
        if (!hasFinalResult && receiveMask == mask) {
          int64_t parentRank = localRank ^ mask;
          auto message = DTEMessageAttr::get(
              rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
              DTEProtocolPhase::AllReduceTreeBroadcast, mask, localRank);
          auto recv = rewriter.create<InstrDTERecvOp>(
              op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
              *accumulator, rewriter.getI64IntegerAttr(rankGroup[parentRank]),
              rewriter.getI64IntegerAttr(bytes), message,
              DirectDTEBindingAttr());
          llvm::SmallVector<mlir::Value, 1> tokens{recv.getToken()};
          rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
          hasFinalResult = true;
          continue;
        }

        if (!hasFinalResult || (localRank & mask) != 0)
          continue;
        int64_t childRank = localRank | mask;
        if (childRank >= groupSize || getLowestSetBit(childRank) != mask)
          continue;
        auto message = DTEMessageAttr::get(
            rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
            DTEProtocolPhase::AllReduceTreeBroadcast, mask, childRank);
        auto send = rewriter.create<InstrDTESendOp>(
            op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
            *accumulator, rewriter.getI64IntegerAttr(rankGroup[childRank]),
            rewriter.getI64IntegerAttr(bytes), message, DirectDTEBindingAttr());
        llvm::SmallVector<mlir::Value, 1> tokens{send.getToken()};
        rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);
      }

      rewriter.replaceOp(op, *accumulator);
      return mlir::success();
    }

    auto forwardBuffer =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), inputType);
    if (mlir::failed(createContiguousSPMCopy(
            rewriter, op.getLoc(), op, op.getInput(), *accumulator,
            failureReason, "tile.all_reduce accumulator init")))
      return mlir::failure();
    if (mlir::failed(createContiguousSPMCopy(
            rewriter, op.getLoc(), op, op.getInput(), forwardBuffer.getResult(),
            failureReason, "tile.all_reduce forward init")))
      return mlir::failure();
    rewriter.create<SyncLocalFenceOp>(op.getLoc());

    int64_t nextPeer = rankGroup[(localRank + 1) % groupSize];
    int64_t prevPeer = rankGroup[(localRank + groupSize - 1) % groupSize];
    for (int64_t step = 0; step < groupSize - 1; ++step) {
      int64_t sendPayloadSlice = (localRank + groupSize - step) % groupSize;
      int64_t recvPayloadSlice = (localRank + groupSize - step - 1) % groupSize;
      auto sendMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllReduceRing, step, sendPayloadSlice);
      auto recvMessage = DTEMessageAttr::get(
          rewriter.getContext(), op.getCommunicationIdAttr().getInt(),
          DTEProtocolPhase::AllReduceRing, step, recvPayloadSlice);
      auto send = rewriter.create<InstrDTESendOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
          forwardBuffer.getResult(), rewriter.getI64IntegerAttr(nextPeer),
          rewriter.getI64IntegerAttr(bytes), sendMessage,
          DirectDTEBindingAttr());
      auto recv = rewriter.create<InstrDTERecvOp>(
          op.getLoc(), rewriter.getType<mlir::async::TokenType>(),
          op.getRecvBuffer(), rewriter.getI64IntegerAttr(prevPeer),
          rewriter.getI64IntegerAttr(bytes), recvMessage,
          DirectDTEBindingAttr());
      llvm::SmallVector<mlir::Value, 2> tokens{send.getToken(),
                                               recv.getToken()};
      rewriter.create<InstrDTEWaitOp>(op.getLoc(), tokens);

      llvm::SmallVector<mlir::Value, 2> inputs{*accumulator,
                                               op.getRecvBuffer()};
      rewriter.create<InstrElementwiseOp>(op.getLoc(), *accumulationKind,
                                          inputs, *accumulator);

      if (step + 1 == groupSize - 1)
        continue;
      if (mlir::failed(createContiguousSPMCopy(
              rewriter, op.getLoc(), op, op.getRecvBuffer(),
              forwardBuffer.getResult(), failureReason,
              "tile.all_reduce forward copy")))
        return mlir::failure();
      rewriter.create<SyncLocalFenceOp>(op.getLoc());
    }

    rewriter.replaceOp(op, *accumulator);
    return mlir::success();
  }

private:
  std::string *failureReason;
  AllReduceSchedule schedule;
};

static void configureTileRegionToInstrTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                         mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                         mlir::scf::SCFDialect>();
  target.addLegalOp<mlir::ModuleOp, TileRegionOp, TileYieldOp, SyncLocalFenceOp,
                    InstrRDMAOp, InstrWDMAOp, InstrGatherScatterOp, InstrFillOp,
                    InstrElementwiseOp, InstrBit2FpOp, InstrMaskMoveOp,
                    InstrReduceOp, InstrConvertOp, InstrGemmOp, InstrDTESendOp,
                    InstrDTERecvOp, InstrDTEWaitOp>();
  target.addDynamicallyLegalOp<InstrTDMADataMoveOp>([](InstrTDMADataMoveOp op) {
    return !requiresGatherScatterMaterialization(op.getKindAttr().getValue());
  });
  target.addIllegalOp<
      StorageLoadOp, StorageStoreOp, LayoutMaterializeOp, ComputeFillOp,
      ComputeGemmOp, ComputeElementwiseOp, ComputeReduceOp, MoveCopyOp,
      MoveExtractSliceOp, MoveInsertSliceOp, MoveTransposeOp, MoveBroadcastOp,
      ViewReshapeOp, CommAllGatherOp, CommReduceScatterOp, CommAllReduceOp>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

static void
populateTileRegionToInstrPatterns(mlir::RewritePatternSet &patterns,
                                  const TileRegionToInstrOptions &options,
                                  std::string *failureReason) {
  mlir::MLIRContext *context = patterns.getContext();
  patterns
      .add<TileLoadLowering, TileStoreLowering, LayoutMaterializeLowering,
           TileCopyLowering, MoveExtractSliceLowering, MoveInsertSliceLowering,
           MoveTransposeLowering, InstrTDMADataMoveLowering,
           MoveBroadcastLowering, ElementwiseLowering, ReduceLowering,
           GemmLowering, ViewReshapeLowering>(context, failureReason);
  patterns.add<FillLowering>(context);
  patterns.add<AllGatherLowering>(context, failureReason,
                                  options.allGatherSchedule);
  patterns.add<ReduceScatterLowering>(context, failureReason,
                                      options.reduceScatterSchedule);
  patterns.add<AllReduceLowering>(context, failureReason,
                                  options.allReduceSchedule);
}

static mlir::LogicalResult parseTileRegionToInstrOptions(
    llvm::StringRef allGatherSchedule, llvm::StringRef allReduceSchedule,
    llvm::StringRef reduceScatterSchedule, TileRegionToInstrOptions &options,
    std::string *failureReason) {
  if (allGatherSchedule == "auto" || allGatherSchedule == "ring") {
    options.allGatherSchedule = AllGatherSchedule::Ring;
  } else if (allGatherSchedule == "direct") {
    options.allGatherSchedule = AllGatherSchedule::Direct;
  } else {
    setFailureReason(failureReason,
                     llvm::Twine("unsupported all_gather schedule: ")
                         .concat(allGatherSchedule)
                         .str());
    return mlir::failure();
  }

  if (allReduceSchedule == "auto" || allReduceSchedule == "ring") {
    options.allReduceSchedule = AllReduceSchedule::Ring;
  } else if (allReduceSchedule == "tree") {
    options.allReduceSchedule = AllReduceSchedule::Tree;
  } else {
    setFailureReason(failureReason,
                     llvm::Twine("unsupported all_reduce schedule: ")
                         .concat(allReduceSchedule)
                         .str());
    return mlir::failure();
  }

  if (reduceScatterSchedule == "auto" || reduceScatterSchedule == "direct") {
    options.reduceScatterSchedule = ReduceScatterSchedule::Direct;
  } else {
    setFailureReason(failureReason,
                     llvm::Twine("unsupported reduce_scatter schedule: ")
                         .concat(reduceScatterSchedule)
                         .str());
    return mlir::failure();
  }

  return mlir::success();
}

static mlir::LogicalResult
materializeTerminalLocalFences(mlir::ModuleOp module) {
  mlir::WalkResult result = module.walk([&](TileRegionOp tileRegion) {
    if (!tileRegion.getBody().hasOneBlock()) {
      tileRegion.emitError()
          << "instruction_completion_failure: terminal local completion "
             "requires a single-block wafer.tile.region";
      return mlir::WalkResult::interrupt();
    }

    mlir::Block &body = tileRegion.getBody().front();
    mlir::Operation *terminator = body.getTerminator();
    if (!terminator) {
      tileRegion.emitError()
          << "instruction_completion_failure: wafer.tile.region has no "
             "terminator for terminal local completion";
      return mlir::WalkResult::interrupt();
    }
    if (mlir::isa_and_nonnull<SyncLocalFenceOp>(terminator->getPrevNode()))
      return mlir::WalkResult::advance();

    mlir::OpBuilder builder(terminator);
    builder.create<SyncLocalFenceOp>(terminator->getLoc());
    return mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

struct ConvertTileRegionToInstrPass
    : public wafer::impl::ConvertTileRegionToInstrPassBase<
          ConvertTileRegionToInstrPass> {
  using wafer::impl::ConvertTileRegionToInstrPassBase<
      ConvertTileRegionToInstrPass>::ConvertTileRegionToInstrPassBase;

  void runOnOperation() final {
    std::string failureReason;
    TileRegionToInstrOptions options;
    if (mlir::failed(parseTileRegionToInstrOptions(
            allGatherSchedule, allReduceSchedule, reduceScatterSchedule,
            options, &failureReason))) {
      getOperation().emitError(failureReason);
      signalPassFailure();
      return;
    }

    if (mlir::succeeded(wafer::convertTileRegionToInstrModule(
            getOperation(), options, &failureReason)))
      return;

    if (!failureReason.empty())
      getOperation().emitError(failureReason);
    else
      getOperation().emitError("tile-region to instruction conversion failed");
    signalPassFailure();
  }
};

} // namespace

wafer::detail::StaticTerminalOperationBudgetStatus
wafer::detail::checkStaticTerminalOperationBudget(
    mlir::Operation *root, uint64_t &operationCount) {
  operationCount = 0;
  bool overflow = false;
  root->walk([&](mlir::Operation *operation) {
    if (overflow ||
        !mlir::isa<WaferInstructionOpInterface, SyncLocalFenceOp>(operation))
      return;
    if (operationCount == std::numeric_limits<uint64_t>::max()) {
      overflow = true;
      return;
    }
    ++operationCount;
  });
  if (overflow)
    return StaticTerminalOperationBudgetStatus::CountOverflow;
  if (operationCount > kStaticTerminalOperationBudget)
    return StaticTerminalOperationBudgetStatus::BudgetExceeded;
  return StaticTerminalOperationBudgetStatus::WithinBudget;
}

mlir::LogicalResult
wafer::convertTileRegionToInstrModule(mlir::ModuleOp module,
                                      std::string *failureReason) {
  return wafer::convertTileRegionToInstrModule(
      module, TileRegionToInstrOptions{}, failureReason);
}

mlir::LogicalResult
wafer::convertTileRegionToInstrModule(mlir::ModuleOp module,
                                      const TileRegionToInstrOptions &options,
                                      std::string *failureReason) {
  if (failureReason)
    failureReason->clear();

  mlir::MLIRContext *context = module.getContext();
  llvm::SmallVector<mlir::Operation *, 4> selectCandidates;
  module.walk([&](ComputeElementwiseOp op) {
    if (op.getKind() == ComputeElementwiseKind::Select)
      selectCandidates.push_back(op);
  });
  if (!selectCandidates.empty()) {
    mlir::RewritePatternSet canonicalizationPatterns(context);
    canonicalizationPatterns.add<ConstantPredicateSelectToCopy>(context);
    mlir::FrozenRewritePatternSet frozenPatterns(
        std::move(canonicalizationPatterns));
    mlir::GreedyRewriteConfig config;
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
    if (mlir::failed(mlir::applyOpPatternsAndFold(selectCandidates,
                                                  frozenPatterns, config))) {
      setFailureReason(
          failureReason,
          "tile constant-predicate select canonicalization failed");
      return mlir::failure();
    }
  }

  mlir::ConversionTarget target(*context);
  configureTileRegionToInstrTarget(target);

  mlir::RewritePatternSet patterns(context);
  populateTileRegionToInstrPatterns(patterns, options, failureReason);

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
  return materializeTerminalLocalFences(module);
}
