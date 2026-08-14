//===- Passes.td - Wafer pass declaration umbrella --------*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_PASSES_TD
#define WAFER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

include "Wafer/Transforms/FrontendPasses.td"
include "Wafer/Transforms/MemoryPasses.td"
include "Wafer/Transforms/TargetPasses.td"
include "Wafer/Transforms/TileLoweringPasses.td"

#endif // WAFER_TRANSFORMS_PASSES_TD
