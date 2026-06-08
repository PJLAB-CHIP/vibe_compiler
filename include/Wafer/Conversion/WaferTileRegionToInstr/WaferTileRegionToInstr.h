//===- WaferTileRegionToInstr.h - Tile-region to instr API -----*- C++ -*-===//

#ifndef WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H
#define WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <string>

namespace wafer {

mlir::LogicalResult
convertTileRegionToInstrModule(mlir::ModuleOp module,
                               std::string *failureReason = nullptr);

} // namespace wafer

#endif // WAFER_CONVERSION_WAFERTILEREGIONTOINSTR_WAFERTILEREGIONTOINSTR_H
