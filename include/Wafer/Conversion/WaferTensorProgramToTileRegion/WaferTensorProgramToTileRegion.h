//===- WaferTensorProgramToTileRegion.h - Tensor program lowering -*- C++
//-*-===//

#ifndef WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H
#define WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H

#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

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
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt);

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
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt);

/// Materializes a compact structured traversal of a standalone tensor
/// program and lowers it to tile-region IR.
mlir::LogicalResult lowerCompleteCandidateTensorProgramToTileRegionModule(
    mlir::func::FuncOp function, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative = std::nullopt);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTENSORPROGRAMTOTILEREGION_H
