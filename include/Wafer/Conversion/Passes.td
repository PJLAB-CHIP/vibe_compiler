//===- Passes.td - Wafer conversion pass declarations -------*- tablegen -*-===//

#ifndef WAFER_CONVERSION_PASSES_TD
#define WAFER_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

include "Wafer/Conversion/StableHLOToLinalg/Passes.td"
include "Wafer/Conversion/TileToInstr/Passes.td"
include "Wafer/Conversion/InstrToLLVM/Passes.td"

#endif // WAFER_CONVERSION_PASSES_TD
