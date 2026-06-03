//===- TileRegionCandidate.h - Provisional tile-region candidate -*- C++ -*-===//

#ifndef WAFER_CONVERSION_TILEREGIONCANDIDATE_TILEREGIONCANDIDATE_H
#define WAFER_CONVERSION_TILEREGIONCANDIDATE_TILEREGIONCANDIDATE_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace wafer {

struct TileRegionCandidate {
  GroupOp group;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  bool succeeded = true;
  std::string failureReason;
};

mlir::LogicalResult buildTileRegionCandidate(GroupOp group,
                                             TileRegionCandidate &candidate);

void dumpTileRegionCandidate(const TileRegionCandidate &candidate,
                             llvm::StringRef groupLabel, llvm::raw_ostream &os);

} // namespace wafer

#endif // WAFER_CONVERSION_TILEREGIONCANDIDATE_TILEREGIONCANDIDATE_H
