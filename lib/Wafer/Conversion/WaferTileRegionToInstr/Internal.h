//===- Internal.h - Tile-region to instr implementation --------*- C++ -*-===//

#ifndef WAFER_LIB_CONVERSION_WAFERTILEREGIONTOINSTR_INTERNAL_H
#define WAFER_LIB_CONVERSION_WAFERTILEREGIONTOINSTR_INTERNAL_H

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Target/TargetProfile.h"

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

enum class AllGatherSchedule { Ring, Direct };
enum class AllReduceSchedule { Auto, Ring, Tree };
enum class ReduceScatterSchedule { Direct, Ring };

struct TileRegionToInstrOptions {
  AllGatherSchedule allGatherSchedule = AllGatherSchedule::Ring;
  AllReduceSchedule allReduceSchedule = AllReduceSchedule::Auto;
  ReduceScatterSchedule reduceScatterSchedule = ReduceScatterSchedule::Direct;
  /// When present, target-dependent instruction choices must already match
  /// the selected closed profile.  The standalone conversion pass leaves this
  /// empty and checks only target-independent instruction IR legality.
  std::optional<TargetProfileId> targetProfile;
};

/// Compiler-private materialization point used only by actual-clone candidate
/// generation. The installed conversion API always lowers the baseline form.
mlir::LogicalResult
convertTileRegionToInstrModule(mlir::ModuleOp module,
                               const TileRegionToInstrOptions &options,
                               std::string *failureReason = nullptr);

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

enum class MovementEngine { RDMA, WDMA, GatherScatter };

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
void createGatherScatter(mlir::PatternRewriter &rewriter, mlir::Location loc,
                         mlir::Value source, mlir::Value dest,
                         const MovementDescriptor &sourceDescriptor,
                         const MovementDescriptor &destDescriptor);
mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>>
getRelationMovementDescriptors(mlir::PatternRewriter &rewriter,
                               mlir::Operation *op, mlir::MemRefType sourceType,
                               mlir::MemRefType destType,
                               llvm::ArrayRef<int64_t> iterationShape,
                               const analysis::IndexRelation &iterationToSource,
                               const analysis::IndexRelation &iterationToDest,
                               MovementEngine engine,
                               std::string *failureReason,
                               llvm::StringRef opLabel);
void createGatherScatterDescriptors(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<MovementDescriptorPair> descriptors);
void createMappedRDMADescriptors(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<MovementDescriptorPair> descriptors);
void createMappedWDMADescriptors(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value source,
    mlir::Value dest, llvm::ArrayRef<MovementDescriptorPair> descriptors);
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

bool requiresGatherScatterMaterialization(InstrDataMoveKind kind);
mlir::LogicalResult verifyStaticShapeAttrMatchesMemRef(
    mlir::PatternRewriter &rewriter, mlir::Operation *op, mlir::MemRefType type,
    mlir::DenseI64ArrayAttr shapeAttr, llvm::StringRef role,
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
                                     const TileRegionToInstrOptions &options,
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
