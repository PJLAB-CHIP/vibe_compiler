//===- Patterns.td - Declarative TileRegion to Instr rewrites -*- tablegen -*-===//

#ifndef WAFER_CONVERSION_TILEREGION_TO_INSTR_PATTERNS
#define WAFER_CONVERSION_TILEREGION_TO_INSTR_PATTERNS

include "mlir/IR/PatternBase.td"
include "Wafer/IR/WaferOps.td"

def LowerComputeFill : Pat<
  (Wafer_ComputeFillOp $dest, $value, $fill_domain),
  (Wafer_InstrFillOp $dest, $value, $fill_domain,
    ConstantAttr<Wafer_NCCWorkerAttr, "::wafer::NCCWorker::Worker0">)>;

#endif // WAFER_CONVERSION_TILEREGION_TO_INSTR_PATTERNS
