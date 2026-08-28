//===- Internal.h - Tile-region to instr implementation --------*- C++ -*-===//

#ifndef WAFER_LIB_CONVERSION_WAFERTILEREGIONTOINSTR_INTERNAL_H
#define WAFER_LIB_CONVERSION_WAFERTILEREGIONTOINSTR_INTERNAL_H

#include "Wafer/Analysis/Linalg/IndexRelation.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace wafer::tile_region_to_instr {

/// Attributes detailed conversion time to the typed source operation being
/// rewritten. This stays invocation-local and is a no-op unless the owning
/// compiler request explicitly enabled detailed timing.
class ScopedLoweringPatternTiming {
public:
  explicit ScopedLoweringPatternTiming(mlir::Operation *operation)
      : timing("lowering-pattern", "tile-region-to-instr",
               operation->getName().getStringRef()) {}

private:
  wafer::support::ScopedCompileTimingSpan timing;
};

struct MovementDescriptor {
  int64_t byteCount = 0;
  int64_t innerBytes = 0;
  int64_t byteOffset = 0;
  llvm::SmallVector<int64_t, 3> strides;
  llvm::SmallVector<int64_t, 3> iterations;
};

struct MovementDescriptorPair {
  MovementDescriptor source;
  MovementDescriptor dest;
};

using MovementDescriptorPlan = llvm::SmallVector<MovementDescriptorPair>;
using SharedMovementDescriptorPlan =
    std::shared_ptr<const MovementDescriptorPlan>;

enum class MovementEngine { RDMA, WDMA, GatherScatter };

/// Request-local cache of exact descriptor plans. Entries contain only typed
/// endpoint geometry and total projected IndexRelations; they own no
/// Operation/Value handles and die with the TileRegion-to-Instr lowering
/// session. Failed/unsupported queries are never cached.
class MovementDescriptorCache {
public:
  bool hasOrProveIdentityPhysicalTraversal(mlir::MemRefType sourceType,
                                           mlir::MemRefType destType);

  mlir::FailureOr<SharedMovementDescriptorPlan>
  getOrCreate(mlir::PatternRewriter &rewriter, mlir::Operation *op,
              mlir::MemRefType sourceType, mlir::MemRefType destType,
              llvm::ArrayRef<int64_t> iterationShape,
              const analysis::IndexRelation &iterationToSource,
              const analysis::IndexRelation &iterationToDest,
              MovementEngine engine, llvm::StringRef opLabel);

private:
  struct Entry {
    mlir::MemRefType sourceType;
    mlir::MemRefType destType;
    llvm::SmallVector<int64_t, 4> iterationShape;
    mlir::AffineMap iterationToSource;
    mlir::AffineMap iterationToDest;
    MovementEngine engine;
    SharedMovementDescriptorPlan plan;
  };

  llvm::SmallVector<Entry, 16> entries;
  llvm::DenseMap<size_t, llvm::SmallVector<unsigned, 2>> entriesByHash;
  llvm::DenseMap<size_t, uint8_t> cacheAdmissionCounts;
  llvm::DenseSet<std::pair<mlir::Type, mlir::Type>>
      provenIdentityPhysicalTraversals;
};

using CanonicalReshapeMovementRelations = analysis::CanonicalReshapeRelations;

inline std::optional<CanonicalReshapeMovementRelations>
getCanonicalReshapeMovementRelations(mlir::MLIRContext *context,
                                     llvm::ArrayRef<int64_t> sourceShape,
                                     llvm::ArrayRef<int64_t> destShape) {
  return analysis::getCanonicalReshapeRelations(context, sourceShape,
                                                destShape);
}

std::optional<int64_t>
getStaticPositiveElementCount(llvm::ArrayRef<int64_t> shape);
std::optional<int64_t> checkedMulI64(int64_t lhs, int64_t rhs);

mlir::LogicalResult failPattern(mlir::PatternRewriter &rewriter,
                                mlir::Operation *op, llvm::StringRef reason);

inline NCCWorkerAttr getDefaultNCCWorkerAttr(mlir::OpBuilder &builder) {
  return NCCWorkerAttr::get(builder.getContext(), NCCWorker::Worker0);
}

template <typename T>
mlir::FailureOr<T> failFailureOr(mlir::PatternRewriter &rewriter,
                                 mlir::Operation *op, llvm::StringRef reason) {
  (void)failPattern(rewriter, op, reason);
  return mlir::failure();
}

std::optional<mlir::RankedTensorType>
getLogicalTensorTypeFromMemRef(mlir::Type type);
mlir::FailureOr<mlir::Value>
createDestAlloc(mlir::Location loc, mlir::Type type,
                mlir::PatternRewriter &rewriter, mlir::Operation *op,
                TileRegionToInstrBufferRecorder *bufferRecorder);

mlir::FailureOr<MovementDescriptor>
getContiguousDescriptor(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                        mlir::Type type);
mlir::FailureOr<MovementDescriptor>
getStridedTensorDescriptor(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                           mlir::Type type, llvm::StringRef role);

InstrRDMAOp createRDMA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                       mlir::Value source, mlir::Value dest,
                       const MovementDescriptor &descriptor);
InstrWDMAOp createWDMA(mlir::PatternRewriter &rewriter, mlir::Location loc,
                       mlir::Value source, mlir::Value dest,
                       const MovementDescriptor &descriptor);
InstrGatherScatterOp createGatherScatter(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, const MovementDescriptor &sourceDescriptor,
    const MovementDescriptor &destDescriptor,
    mlir::Value dynamicSourceOffset = {}, mlir::Value dynamicDestOffset = {});
mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>>
getRelationMovementDescriptors(mlir::PatternRewriter &rewriter,
                               mlir::Operation *op, mlir::MemRefType sourceType,
                               mlir::MemRefType destType,
                               llvm::ArrayRef<int64_t> iterationShape,
                               const analysis::IndexRelation &iterationToSource,
                               const analysis::IndexRelation &iterationToDest,
                               MovementEngine engine, llvm::StringRef opLabel);
llvm::SmallVector<InstrGatherScatterOp, 4> createGatherScatterDescriptors(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<MovementDescriptorPair> descriptors);
llvm::SmallVector<InstrRDMAOp, 4> createMappedRDMADescriptors(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<MovementDescriptorPair> descriptors);
llvm::SmallVector<InstrWDMAOp, 4> createMappedWDMADescriptors(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<MovementDescriptorPair> descriptors);
mlir::IntegerAttr getI64Attr(mlir::PatternRewriter &rewriter, int64_t value);
mlir::FailureOr<int64_t> readRequiredI64Attr(mlir::PatternRewriter &rewriter,
                                             mlir::Operation *op,
                                             llvm::StringRef name);
mlir::FailureOr<int64_t> getStaticPhysicalBytes(mlir::PatternRewriter &rewriter,
                                                mlir::Operation *op,
                                                mlir::Type type,
                                                llvm::StringRef role);
mlir::FailureOr<llvm::SmallVector<int64_t>>
getStaticCompactStrides(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                        mlir::MemRefType type);
bool isStandardViewCompatibleLayout(MemLayout layout);
mlir::FailureOr<llvm::SmallVector<int64_t>>
delinearizeIndex(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                 llvm::ArrayRef<int64_t> shape, int64_t linearIndex,
                 llvm::StringRef opLabel);

bool requiresGatherScatterMaterialization(InstrDataMoveKind kind);
mlir::LogicalResult verifyStaticShapeAttrMatchesMemRef(
    mlir::PatternRewriter &rewriter, mlir::Operation *op, mlir::MemRefType type,
    mlir::DenseI64ArrayAttr shapeAttr, llvm::StringRef role,
    llvm::StringRef opLabel);
mlir::FailureOr<InstrElementwiseKindAttr> getAccumulationElementwiseKind(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    ComputeReduceKindAttr reduceKind, llvm::StringRef opLabel);

void populateMovementLoweringPatterns(
    mlir::RewritePatternSet &patterns,
    TileRegionToInstrBufferRecorder *bufferRecorder,
    MovementDescriptorCache *descriptorCache);
void populateViewReshapeLoweringPattern(mlir::RewritePatternSet &patterns);
void populateComputeLoweringPatterns(
    mlir::RewritePatternSet &patterns,
    TileRegionToInstrBufferRecorder *bufferRecorder,
    MovementDescriptorCache *descriptorCache);
void populateFillLoweringPattern(mlir::RewritePatternSet &patterns);
void populatePeerLoweringPatterns(mlir::RewritePatternSet &patterns);

} // namespace wafer::tile_region_to_instr

#endif // WAFER_LIB_CONVERSION_WAFERTILEREGIONTOINSTR_INTERNAL_H
