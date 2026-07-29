//===- WaferTensorProgramToTileRegion.h - Tensor program lowering -*- C++
//-*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H

#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer {

/// Describes how a source operation may terminate a candidate traversal.
/// This is a structural materialization capability, not a profitability or
/// target-legality result.
enum class CandidateTraversalRootCapability {
  Tiled,
  FullTraversalOnly,
  Unsupported,
};

CandidateTraversalRootCapability
classifyCandidateTraversalRoot(mlir::Operation *operation);

/// Selects the tile seed used while materializing one discardable complete
/// candidate clone. This is transient transformation input: it is never
/// persisted in IR, rank-frontier metadata, or an artifact.
enum class CandidateTileTraversalKind : uint8_t {
  ResultDriven,
  OperandDriven,
  PartialReduction,
};

/// The iteration-domain tile corresponding to one operand tile. This is a
/// transient analysis result derived from the current TilingInterface; callers
/// must not retain it across IR mutation.
struct OperandTileIterationDomain {
  llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
};

/// The actual consumer implementation produced from one operand tile.
/// Ownership of operations remains with the caller's IR.
struct OperandTileMaterialization {
  OperandTileIterationDomain iterationDomain;
  llvm::SmallVector<mlir::Operation *, 2> tiledOperations;
  llvm::SmallVector<mlir::Value, 2> tiledValues;
  llvm::SmallVector<mlir::Operation *, 4> generatedSlices;
};

/// Maps an operand tile into the consumer iteration domain through
/// TilingInterface::getIterationDomainTileFromOperandTile.
mlir::FailureOr<OperandTileIterationDomain> mapOperandTileToIterationDomain(
    mlir::Operation *consumer, mlir::OpBuilder &builder, unsigned operandNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, std::string *failureReason);

/// Materializes an actual consumer tile from an operand tile through
/// TilingInterface::getTiledImplementationFromOperandTile. The returned
/// iteration-domain relation is recomputed from the same current IR.
mlir::FailureOr<OperandTileMaterialization> materializeConsumerFromOperandTile(
    mlir::Operation *consumer, mlir::OpBuilder &builder, unsigned operandNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, std::string *failureReason);

/// The actual partial-reduction and merge implementation produced through
/// PartialReductionOpInterface. Ownership of operations remains with the
/// caller's IR.
struct PartialReductionTileMaterialization {
  llvm::SmallVector<int, 2> reductionDimensions;
  llvm::SmallVector<mlir::Value, 2> initialValues;
  llvm::SmallVector<mlir::Operation *, 2> partialOperations;
  llvm::SmallVector<mlir::Value, 2> partialValues;
  llvm::SmallVector<mlir::Operation *, 4> generatedSlices;
  llvm::SmallVector<mlir::Operation *, 2> mergeOperations;
  llvm::SmallVector<mlir::Value, 2> mergedValues;
};

/// Materializes a partial reduction for one iteration-domain tile and merges
/// it to the corresponding result tile. Numeric regrouping legality is checked
/// before any IR is created and fails closed when it cannot be established.
/// Transformation callers should invoke this on a discardable candidate clone
/// so a later interface failure remains atomic at candidate granularity.
mlir::FailureOr<PartialReductionTileMaterialization>
materializePartialReductionTile(
    mlir::Operation *reduction, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> iterationSizes,
    std::string *failureReason);

/// As above, but merges into caller-provided result-tile destinations. This is
/// used to chain independently materialized reduction chunks through ordinary
/// SSA. Destination count and types are verified before interface mutation.
mlir::FailureOr<PartialReductionTileMaterialization>
materializePartialReductionTile(
    mlir::Operation *reduction, mlir::OpBuilder &builder,
    llvm::ArrayRef<mlir::OpFoldResult> iterationOffsets,
    llvm::ArrayRef<mlir::OpFoldResult> iterationSizes,
    mlir::ValueRange resultTileDestinations, std::string *failureReason);

namespace detail {

enum class CheckedStaticTileProductStatus { Success, InvalidInput, Overflow };

/// Computes product_i ceil(ranges[i] / tileSizes[i]) without signed or unsigned
/// overflow. Positive ranges and in-bounds positive tile sizes are required.
CheckedStaticTileProductStatus
checkedStaticTileProduct(llvm::ArrayRef<int64_t> ranges,
                         llvm::ArrayRef<int64_t> tileSizes, uint64_t &product);

/// Clones a standalone structured scheduling function into an owning module.
/// The function contract is: one entry block; entry arguments are read-only
/// inputs followed by output destinations; the number of output destinations
/// equals the function result count; func.return yields those result roots.
mlir::OwningOpRef<mlir::ModuleOp>
cloneTensorProgramToStandaloneModule(mlir::func::FuncOp function);

} // namespace detail

mlir::LogicalResult lowerTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason, int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt,
    bool useDirectMappedBoundaryTransfer = false);

/// Verifies that replacing one structured reduction by more than one ordered
/// chunk, including neutral-initialized partials and chunk-result combines, is
/// permitted by the source IR's numeric semantics. This is a transformation
/// legality gate, not a target capability or profitability query.
mlir::LogicalResult
verifyCandidateReductionSplitNumericLegality(mlir::linalg::LinalgOp root,
                                             std::string *failureReason);

mlir::LogicalResult lowerCandidateTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt,
    bool useDirectMappedBoundaryTransfer = false);

/// Materializes a compact structured traversal of a standalone tensor
/// program and lowers it to tile-region IR.
mlir::LogicalResult lowerCompleteCandidateTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt,
    bool useDirectMappedBoundaryTransfer = false,
    CandidateTileTraversalKind traversalKind =
        CandidateTileTraversalKind::ResultDriven);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H
