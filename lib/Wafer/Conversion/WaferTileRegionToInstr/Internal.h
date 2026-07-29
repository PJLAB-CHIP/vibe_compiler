//===- Internal.h - Tile-region to instr implementation --------*- C++ -*-===//

#ifndef WAFER_LIB_CONVERSION_WAFERTILEREGIONTOINSTR_INTERNAL_H
#define WAFER_LIB_CONVERSION_WAFERTILEREGIONTOINSTR_INTERNAL_H

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace wafer::tile_region_to_instr {

enum class AllGatherSchedule { Ring, Direct };
enum class AllReduceSchedule { Auto, Ring, Tree };
enum class ReduceScatterSchedule { Direct, Ring };

struct TileRegionToInstrOptions {
  AllGatherSchedule allGatherSchedule = AllGatherSchedule::Ring;
  AllReduceSchedule allReduceSchedule = AllReduceSchedule::Auto;
  ReduceScatterSchedule reduceScatterSchedule = ReduceScatterSchedule::Direct;
};

/// Compiler-private materialization point used only by actual-clone candidate
/// generation. The installed conversion API always lowers the baseline form.
mlir::LogicalResult convertTileRegionToInstrModule(
    mlir::ModuleOp module, const TileRegionToInstrOptions &options,
    std::string *failureReason = nullptr);

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

using StaticPhysicalOffsetCalculator =
    wafer::WaferStaticPhysicalOffsetCalculator;

std::optional<int64_t>
getStaticPositiveElementCount(llvm::ArrayRef<int64_t> shape);
std::optional<int64_t> checkedMulI64(int64_t lhs, int64_t rhs);

void setFailureReason(std::string *failureReason, llvm::StringRef reason);
mlir::LogicalResult failPattern(mlir::PatternRewriter &rewriter,
                                mlir::Operation *op, std::string *failureReason,
                                llvm::StringRef reason);

inline NCCWorkerAttr getDefaultNCCWorkerAttr(mlir::OpBuilder &builder) {
  return NCCWorkerAttr::get(builder.getContext(), NCCWorker::Worker0);
}

template <typename T>
mlir::FailureOr<T>
failFailureOr(mlir::PatternRewriter &rewriter, mlir::Operation *op,
              std::string *failureReason, llvm::StringRef reason) {
  (void)failPattern(rewriter, op, failureReason, reason);
  return mlir::failure();
}

std::optional<mlir::RankedTensorType>
getLogicalTensorTypeFromMemRef(mlir::Type type);
mlir::FailureOr<mlir::Value> createDestAlloc(mlir::Location loc,
                                             mlir::Type type,
                                             mlir::PatternRewriter &rewriter,
                                             mlir::Operation *op,
                                             std::string *failureReason);

mlir::FailureOr<MovementDescriptor>
getContiguousDescriptor(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                        mlir::Type type, std::string *failureReason);
mlir::FailureOr<MovementDescriptor>
getStridedTensorDescriptor(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                           mlir::Type type, std::string *failureReason,
                           llvm::StringRef role);

void createRDMA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                mlir::Value source, mlir::Value dest,
                const MovementDescriptor &descriptor);
void createWDMA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                mlir::Value source, mlir::Value dest,
                const MovementDescriptor &descriptor);
void createMappedRDMASegments(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value source,
                              mlir::Value dest,
                              llvm::ArrayRef<LogicalMovementSegment> segments);
void createMappedWDMASegments(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value source,
                              mlir::Value dest,
                              llvm::ArrayRef<LogicalMovementSegment> segments);
mlir::LogicalResult
preflightMappedDMASegments(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                           llvm::ArrayRef<LogicalMovementSegment> segments,
                           std::string *failureReason, llvm::StringRef opLabel);
void createGatherScatter(mlir::PatternRewriter &rewriter, mlir::Location loc,
                         mlir::Value source, mlir::Value dest,
                         const MovementDescriptor &sourceDescriptor,
                         const MovementDescriptor &destDescriptor);
void createGatherScatterSegments(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<LogicalMovementSegment> segments,
    bool mayReorderDisjointSegments = false);

mlir::FailureOr<uint64_t> preflightPackedMovementCommands(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    llvm::ArrayRef<LogicalMovementSegment> segments, std::string *failureReason,
    llvm::StringRef opLabel);

void copyOptionalAttr(mlir::Operation *from, mlir::Operation *to,
                      llvm::StringRef name);
mlir::IntegerAttr getI64Attr(mlir::PatternRewriter &rewriter, int64_t value);
mlir::FailureOr<int64_t> readRequiredI64Attr(mlir::PatternRewriter &rewriter,
                                             mlir::Operation *op,
                                             llvm::StringRef name,
                                             std::string *failureReason);
mlir::FailureOr<int64_t> getStaticPhysicalBytes(mlir::PatternRewriter &rewriter,
                                                mlir::Operation *op,
                                                mlir::Type type,
                                                std::string *failureReason,
                                                llvm::StringRef role);
mlir::FailureOr<llvm::SmallVector<int64_t>>
getStaticCompactStrides(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                        mlir::MemRefType type, std::string *failureReason);
bool isStandardViewCompatibleLayout(MemLayout layout);
mlir::FailureOr<llvm::SmallVector<int64_t>>
delinearizeIndex(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                 llvm::ArrayRef<int64_t> shape, int64_t linearIndex,
                 std::string *failureReason, llvm::StringRef opLabel);

template <typename SourceIndexFn, typename DestIndexFn>
mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getStaticMappedMovementSegments(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    mlir::MemRefType sourceType, mlir::MemRefType destType,
    llvm::ArrayRef<int64_t> iterationShape, SourceIndexFn sourceIndexFn,
    DestIndexFn destIndexFn, std::string *failureReason,
    llvm::StringRef opLabel) {
  std::optional<WaferPhysicalTensorInfo> sourceInfoStorage =
      computeWaferPhysicalTensorInfo(sourceType);
  std::optional<WaferPhysicalTensorInfo> destInfoStorage =
      computeWaferPhysicalTensorInfo(destType);
  if (!sourceInfoStorage || !destInfoStorage ||
      sourceInfoStorage->physicalBytes <= 0 ||
      destInfoStorage->physicalBytes <= 0)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires static positive physical byte sizes")
            .str());
  const WaferPhysicalTensorInfo &sourceInfo = *sourceInfoStorage;
  const WaferPhysicalTensorInfo &destInfo = *destInfoStorage;
  if (sourceInfo.elementBytes <= 0 ||
      sourceInfo.elementBytes != destInfo.elementBytes ||
      sourceInfo.bitPackedElement || destInfo.bitPackedElement)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires byte-addressable elements")
            .str());

  std::optional<StaticPhysicalOffsetCalculator> sourceOffsets =
      StaticPhysicalOffsetCalculator::create(sourceType);
  std::optional<StaticPhysicalOffsetCalculator> destOffsets =
      StaticPhysicalOffsetCalculator::create(destType);
  if (!sourceOffsets || !destOffsets)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires static positive physical byte sizes")
            .str());

  std::optional<int64_t> elementCount =
      getStaticPositiveElementCount(iterationShape);
  if (!elementCount)
    return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
        rewriter, op, failureReason,
        llvm::Twine(opLabel)
            .concat(" requires static positive iteration shape")
            .str());

  int64_t elementBytes = sourceInfo.elementBytes;
  llvm::SmallVector<LogicalMovementSegment> segments;
  llvm::SmallVector<int64_t, 4> iterationIndices(iterationShape.size(), 0);
  llvm::SmallVector<int64_t, 4> sourceIndices;
  llvm::SmallVector<int64_t, 4> destIndices;
  sourceIndices.reserve(sourceType.getRank());
  destIndices.reserve(destType.getRank());
  for (int64_t linearIndex = 0; linearIndex < *elementCount; ++linearIndex) {
    sourceIndices.clear();
    destIndices.clear();
    if (mlir::failed(sourceIndexFn(iterationIndices, sourceIndices)) ||
        mlir::failed(destIndexFn(iterationIndices, destIndices)))
      return mlir::failure();

    if (sourceIndices.size() != static_cast<size_t>(sourceType.getRank()) ||
        destIndices.size() != static_cast<size_t>(destType.getRank()))
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" cannot compute physical element offset")
              .str());
    int64_t sourceOffset =
        sourceOffsets->getByteOffsetForValidIndices(sourceIndices);
    int64_t destOffset = destOffsets->getByteOffsetForValidIndices(destIndices);
    if (sourceOffset < 0 || destOffset < 0 ||
        sourceOffset > sourceInfo.physicalBytes - elementBytes ||
        destOffset > destInfo.physicalBytes - elementBytes)
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel)
              .concat(" segment exceeds static physical byte range")
              .str());

    bool coalesced = false;
    if (!segments.empty()) {
      LogicalMovementSegment &last = segments.back();
      if (last.sourceOffset + last.bytes == sourceOffset &&
          last.destOffset + last.bytes == destOffset) {
        last.bytes += elementBytes;
        coalesced = true;
      }
    }
    if (!coalesced)
      segments.push_back({sourceOffset, destOffset, elementBytes});

    // The domain was validated once above.  Maintain its canonical
    // lexicographic multi-index incrementally instead of re-delinearizing the
    // linear ordinal (and allocating new index vectors) for every element.
    if (linearIndex + 1 != *elementCount) {
      for (int64_t dim = static_cast<int64_t>(iterationShape.size()) - 1;
           dim >= 0; --dim) {
        if (++iterationIndices[dim] < iterationShape[dim])
          break;
        iterationIndices[dim] = 0;
      }
    }
  }

  return segments;
}

mlir::FailureOr<llvm::SmallVector<int64_t>> expandRankReducedSliceIndices(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    llvm::ArrayRef<int64_t> fullShape, llvm::ArrayRef<int64_t> reducedShape,
    llvm::ArrayRef<int64_t> reducedIndices, std::string *failureReason,
    llvm::StringRef opLabel);
mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getStaticLogicalMovementSegments(mlir::PatternRewriter &rewriter,
                                 mlir::Operation *op,
                                 mlir::MemRefType sourceType,
                                 mlir::MemRefType destType,
                                 std::string *failureReason,
                                 llvm::StringRef opLabel);
bool requiresGatherScatterMaterialization(InstrDataMoveKind kind);
mlir::LogicalResult verifyStaticShapeAttrMatchesMemRef(
    mlir::PatternRewriter &rewriter, mlir::Operation *op, mlir::MemRefType type,
    mlir::DenseI64ArrayAttr shapeAttr, llvm::StringRef role,
    std::string *failureReason, llvm::StringRef opLabel);
mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getPermutationDataMoveSegments(mlir::PatternRewriter &rewriter,
                               InstrTDMADataMoveOp op,
                               mlir::MemRefType sourceType,
                               mlir::MemRefType destType,
                               llvm::ArrayRef<int64_t> permutation,
                               std::string *failureReason,
                               llvm::StringRef opLabel);
mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getMirrorDataMoveSegments(mlir::PatternRewriter &rewriter,
                          InstrTDMADataMoveOp op, mlir::MemRefType sourceType,
                          mlir::MemRefType destType,
                          llvm::ArrayRef<int64_t> axes,
                          std::string *failureReason, llvm::StringRef opLabel);
mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
getRotateDataMoveSegments(mlir::PatternRewriter &rewriter,
                          InstrTDMADataMoveOp op, mlir::MemRefType sourceType,
                          mlir::MemRefType destType, InstrDataMoveKind kind,
                          llvm::ArrayRef<int64_t> axes,
                          std::string *failureReason, llvm::StringRef opLabel);

mlir::FailureOr<InstrElementwiseKindAttr> getAccumulationElementwiseKind(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    ComputeReduceKindAttr reduceKind, std::string *failureReason,
    llvm::StringRef opLabel);

void populateMovementLoweringPatterns(mlir::RewritePatternSet &patterns,
                                      std::string *failureReason);
void populateViewReshapeLoweringPattern(mlir::RewritePatternSet &patterns,
                                        std::string *failureReason);
void populateComputeLoweringPatterns(mlir::RewritePatternSet &patterns,
                                     std::string *failureReason);
void populateFillLoweringPattern(mlir::RewritePatternSet &patterns);
void populateConstantPredicateSelectCanonicalizationPattern(
    mlir::RewritePatternSet &patterns);
void populatePeerLoweringPatterns(mlir::RewritePatternSet &patterns,
                                  std::string *failureReason);
void populateCollectiveLoweringPatterns(mlir::RewritePatternSet &patterns,
                                        const TileRegionToInstrOptions &options,
                                        std::string *failureReason);

} // namespace wafer::tile_region_to_instr

#endif // WAFER_LIB_CONVERSION_WAFERTILEREGIONTOINSTR_INTERNAL_H
