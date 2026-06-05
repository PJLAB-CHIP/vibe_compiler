//===- Passes.td - Wafer pass declarations ----------------*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_PASSES_TD
#define WAFER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def FormLogicalGroupsPass
    : Pass<"wafer-form-logical-groups", "mlir::ModuleOp"> {
  let summary = "Form root-seeded tensor-level logical wafer.group ops";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::math::MathDialect", "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect",
  ];
}

def DumpGroupTilingDemandPass
    : Pass<"wafer-dump-group-tiling-demand", "mlir::ModuleOp"> {
  let summary = "Dump wafer.group tiling demand analysis";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect",
  ];
}

def DumpGroupLayoutPlanPass
    : Pass<"wafer-dump-group-layout-plan", "mlir::ModuleOp"> {
  let summary = "Dump wafer.group layout planning analysis";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect",
  ];
}

def DumpGroupToTileRegionPass
    : Pass<"wafer-dump-group-to-tile-region", "mlir::ModuleOp"> {
  let summary = "Dump wafer.group to wafer.tile.region lowering IR";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::async::AsyncDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::func::FuncDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect",
  ];
}

def NormalizeStablehloCollectivesPass
    : Pass<"wafer-normalize-stablehlo-collectives", "mlir::ModuleOp"> {
  let summary =
      "Normalize post-SPMD StableHLO collectives to Wafer tensor collectives";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect",
  ];
}

def ConvertGroupToTileRegionPass
    : Pass<"wafer-convert-group-to-tile-region", "mlir::ModuleOp"> {
  let summary = "Convert logical wafer.group ops to wafer.tile.region IR";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::func::FuncDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::math::MathDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect",
  ];
}

#endif // WAFER_TRANSFORMS_PASSES_TD
