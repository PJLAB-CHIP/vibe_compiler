//===- WaferGroupToTileRegion.h - Group-to-tile-region API -----*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERGROUPTOTILEREGION_WAFERGROUPTOTILEREGION_H
#define WAFER_CONVERSION_WAFERGROUPTOTILEREGION_WAFERGROUPTOTILEREGION_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace wafer {

mlir::LogicalResult
lowerGroupToTileRegionModule(GroupOp group,
                             mlir::OwningOpRef<mlir::ModuleOp> &module,
                             std::string *failureReason = nullptr);

mlir::LogicalResult lowerCandidateGroupToTileRegionModule(
    GroupOp group, llvm::ArrayRef<int64_t> candidateTileOffsets,
    llvm::ArrayRef<int64_t> candidateTileSizes,
    mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *failureReason = nullptr);

void dumpGroupToTileRegionModule(mlir::ModuleOp module,
                                 llvm::StringRef groupLabel,
                                 llvm::raw_ostream &os);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERGROUPTOTILEREGION_WAFERGROUPTOTILEREGION_H
