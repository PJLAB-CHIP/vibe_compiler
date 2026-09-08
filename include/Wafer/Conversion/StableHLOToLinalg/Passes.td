//===- Passes.td - StableHLO to Linalg conversion ---------*- tablegen -*-===//

#ifndef WAFER_CONVERSION_STABLEHLOTOLINALG_PASSES_TD
#define WAFER_CONVERSION_STABLEHLOTOLINALG_PASSES_TD

def PromoteStablehloLogisticPass
    : Pass<"wafer-promote-stablehlo-logistic", "mlir::func::FuncOp"> {
  let summary = "Compute low-precision StableHLO logistic in F32 before rounding its result";
}

def FoldConvolutionInputCastsPass
    : Pass<"wafer-fold-convolution-input-casts", "mlir::func::FuncOp"> {
  let summary = "Fuse exact input widening into convolution scalar bodies";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::tensor::TensorDialect"];
}

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
