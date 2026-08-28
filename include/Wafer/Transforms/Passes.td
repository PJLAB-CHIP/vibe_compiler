//===- Passes.td - Wafer pass declaration umbrella --------*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_PASSES_TD
#define WAFER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

include "Wafer/Transforms/StableHLO/Passes.td"
include "Wafer/Transforms/Linalg/Passes.td"
include "Wafer/Transforms/Module/Passes.td"
include "Wafer/Transforms/Instr/Passes.td"
include "Wafer/Conversion/StableHLOToLinalg/Passes.td"
include "Wafer/Conversion/TileToInstr/Passes.td"
include "Wafer/Conversion/InstrToLLVM/Passes.td"

#endif // WAFER_TRANSFORMS_PASSES_TD
