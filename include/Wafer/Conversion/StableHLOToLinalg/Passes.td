//===- Passes.td - StableHLO to Linalg conversion ---------*- tablegen -*-===//

#ifndef WAFER_CONVERSION_STABLEHLOTOLINALG_PASSES_TD
#define WAFER_CONVERSION_STABLEHLOTOLINALG_PASSES_TD

def NormalizeStablehloCollectivesPass
    : Pass<"wafer-normalize-stablehlo-collectives", "mlir::ModuleOp"> {
  let summary =
      "Normalize post-SPMD StableHLO collectives to Wafer linalg extension collectives";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::tensor::TensorDialect",
                           "wafer::WaferDialect"];
}

def LowerStaticStablehloConcatenatePass
    : Pass<"wafer-lower-static-stablehlo-concatenate", "mlir::ModuleOp"> {
  let summary =
      "Lower static StableHLO concatenate to tensor insertion SSA";
  let dependentDialects = ["mlir::tensor::TensorDialect"];
}

#endif // WAFER_CONVERSION_STABLEHLOTOLINALG_PASSES_TD
