//===- Passes.td - StableHLO-level transforms -------------*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_STABLEHLO_PASSES_TD
#define WAFER_TRANSFORMS_STABLEHLO_PASSES_TD

def VerifyFrontendProgramPass
    : Pass<"wafer-verify-frontend-program", "mlir::ModuleOp"> {
  let summary = "Verify imported frontend IR markers and bounded boundaries";
}

def ExpandStablehloBatchNormInferencePass
    : Pass<"wafer-expand-stablehlo-batch-norm-inference", "mlir::func::FuncOp"> {
  let summary = "Expand static floating-point BatchNorm inference in its source dtype";
}

def FoldDefaultStablehloExecutionIdsPass
    : Pass<"wafer-fold-default-stablehlo-execution-ids", "mlir::ModuleOp"> {
  let summary =
      "Fold replica/partition IDs for the verified single-replica partition";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def FoldConstantIntegerTensorCastsPass
    : Pass<"wafer-fold-constant-integer-tensor-casts", "mlir::ModuleOp"> {
  let summary =
      "Fold constant integer tensor unrealized casts after import";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def ApplyDefaultSpmdShardingPass
    : Pass<"wafer-apply-default-spmd-sharding", "mlir::ModuleOp"> {
  let summary =
      "Seed unannotated StableHLO function inputs with Wafer SDY shardings";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "wafer::WaferDialect"];
  let options = [
    Option<"executionMeshName", "execution-mesh", "std::string",
           /*default=*/"\"default_mesh\"",
           "wafer.execution.mesh symbol used to derive default SDY input "
           "shardings">
  ];
}

#endif // WAFER_TRANSFORMS_STABLEHLO_PASSES_TD
