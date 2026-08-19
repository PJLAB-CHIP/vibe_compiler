//===- FrontendPasses.td - Imported IR transforms ---------*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_FRONTENDPASSES_TD
#define WAFER_TRANSFORMS_FRONTENDPASSES_TD

def VerifyFrontendProgramPass
    : Pass<"wafer-verify-frontend-program", "mlir::ModuleOp"> {
  let summary = "Verify imported frontend IR markers and bounded boundaries";
}

def NormalizeStablehloCollectivesPass
    : Pass<"wafer-normalize-stablehlo-collectives", "mlir::ModuleOp"> {
  let summary =
      "Normalize post-SPMD StableHLO collectives to Wafer linalg extension collectives";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect"
  ];
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

def FoldStaticTensorOpsPass
    : Pass<"wafer-fold-static-tensor-ops", "mlir::ModuleOp"> {
  let summary =
      "Evaluate bounded static tensor and structured Linalg expressions";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::tensor::TensorDialect"];
}

def LowerStaticStablehloConcatenatePass
    : Pass<"wafer-lower-static-stablehlo-concatenate", "mlir::ModuleOp"> {
  let summary =
      "Lower static StableHLO concatenate to tensor insertion SSA";
  let dependentDialects = ["mlir::tensor::TensorDialect"];
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

#endif // WAFER_TRANSFORMS_FRONTENDPASSES_TD
