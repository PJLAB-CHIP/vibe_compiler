//===- WaferGroupToTileRegion.h - Group-to-tile-region API -----*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERGROUPTOTILEREGION_WAFERGROUPTOTILEREGION_H
#define WAFER_CONVERSION_WAFERGROUPTOTILEREGION_WAFERGROUPTOTILEREGION_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace wafer {

namespace detail {

/// Complete candidate traversal currently materializes every output-tile and
/// reduction-chunk pair eagerly. This budget is an implementation resource
/// guard, not an IR or target legality restriction. A compact loop form should
/// replace the eager representation before increasing it substantially.
inline constexpr uint64_t kCompleteCandidateMaterializationBudget = 4096;

enum class CheckedStaticTileProductStatus { Success, InvalidInput, Overflow };

/// Computes product_i ceil(ranges[i] / tileSizes[i]) without signed or unsigned
/// overflow. Positive ranges and in-bounds positive tile sizes are required.
CheckedStaticTileProductStatus
checkedStaticTileProduct(llvm::ArrayRef<int64_t> ranges,
                         llvm::ArrayRef<int64_t> tileSizes, uint64_t &product);

enum class CompleteCandidateExpansionStatus {
  WithinBudget,
  InvalidInput,
  CountOverflow,
  BudgetExceeded,
};

/// Computes the sum, over yielded roots, of
/// outputTileCount * reductionChunkCount. A chunk count of one represents an
/// unsplit or non-reduction root.
CompleteCandidateExpansionStatus checkCompleteCandidateExpansionBudget(
    uint64_t outputTileCount, llvm::ArrayRef<uint64_t> reductionChunkCounts,
    uint64_t &materializationCount);

/// Isolates one group behind a stable function argument/result boundary for
/// candidate evaluation. Constant group operands retain an in-function SSA
/// defining op while their ABI argument slots remain present for commit.
mlir::OwningOpRef<mlir::ModuleOp> cloneGroupToStandaloneModule(GroupOp group);

} // namespace detail

mlir::LogicalResult lowerGroupToTileRegionModule(
    GroupOp group, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason, int64_t currentLogicalRank);

mlir::LogicalResult lowerCandidateGroupToTileRegionModule(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank);

/// Materializes every static output tile described by the candidate sizes,
/// including non-divisible tails, before lowering the standalone group. The
/// supported boundary is distinct single-result yielded roots with equal
/// static result shapes; unsupported producer relations fail closed.
mlir::LogicalResult lowerCompleteCandidateGroupToTileRegionModule(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileSizes,
    llvm::ArrayRef<int64_t> candidateReductionTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *failureReason,
    int64_t currentLogicalRank);

void dumpGroupToTileRegionModule(mlir::ModuleOp module,
                                 llvm::StringRef groupLabel,
                                 llvm::raw_ostream &os);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERGROUPTOTILEREGION_WAFERGROUPTOTILEREGION_H
