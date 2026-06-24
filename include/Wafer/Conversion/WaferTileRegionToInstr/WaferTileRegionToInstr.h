//===- WaferTileRegionToInstr.h - Tile-region to instr API -----*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H
#define WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <string>

namespace wafer {

enum class AllGatherSchedule {
  Ring,
  Direct,
};

enum class AllReduceSchedule {
  Ring,
  Tree,
};

enum class ReduceScatterSchedule {
  Direct,
};

struct TileRegionToInstrOptions {
  AllGatherSchedule allGatherSchedule = AllGatherSchedule::Ring;
  AllReduceSchedule allReduceSchedule = AllReduceSchedule::Ring;
  ReduceScatterSchedule reduceScatterSchedule = ReduceScatterSchedule::Direct;
};

mlir::LogicalResult
convertTileRegionToInstrModule(mlir::ModuleOp module,
                               std::string *failureReason = nullptr);

mlir::LogicalResult
convertTileRegionToInstrModule(mlir::ModuleOp module,
                               const TileRegionToInstrOptions &options,
                               std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H
