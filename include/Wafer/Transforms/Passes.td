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
                           "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect",
  ];
}

def DumpCandidateDdrTileViewsPass
    : Pass<"wafer-dump-candidate-ddr-tile-views", "mlir::ModuleOp"> {
  let summary = "Dump R3.2e candidate DDR tile-view evaluation IR";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::func::FuncDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::math::MathDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect",
  ];
  let options = [
    Option<"candidateTileOffsets", "candidate-tile-offsets", "std::string",
           /*default=*/"\"\"",
           "Comma-separated candidate output tile offsets">,
    Option<"candidateTileSizes", "candidate-tile-sizes", "std::string",
           /*default=*/"\"\"",
           "Comma-separated candidate output tile sizes">
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
                           "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect",
  ];
}

def ConvertTileRegionToInstrPass
    : Pass<"wafer-convert-tile-region-to-instr", "mlir::ModuleOp"> {
  let summary = "Convert executable wafer.tile.region ops to wafer.instr IR";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::async::AsyncDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "wafer::WaferDialect",
  ];
}

def PlaceSPMBuffersPass
    : Pass<"wafer-place-spm-buffers", "mlir::ModuleOp"> {
  let summary =
      "Place Wafer SPM memref allocations with tile-local byte offsets";
  let dependentDialects = ["mlir::memref::MemRefDialect",
                           "wafer::WaferDialect",
  ];
  let options = [
    Option<"spmBase", "spm-base", "int64_t", /*default=*/"65536",
           "Inclusive byte offset for ordinary tile-local SPM placement">,
    Option<"spmLimit", "spm-limit", "int64_t", /*default=*/"3080192",
           "Exclusive byte limit for ordinary tile-local SPM placement">,
    Option<"spmAlignment", "spm-alignment", "int64_t", /*default=*/"256",
           "Preferred byte alignment for ordinary tile-local SPM placement">
  ];
}

#endif // WAFER_TRANSFORMS_PASSES_TD
