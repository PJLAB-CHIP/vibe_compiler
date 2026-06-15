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
  let summary = "Dump candidate DDR tile-view evaluation IR";
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

def SelectGroupTilePass
    : Pass<"wafer-select-group-tile", "mlir::ModuleOp"> {
  let summary = "Run closed-loop group tile selection and commit selected lowering";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::async::AsyncDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::func::FuncDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::math::MathDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect",
  ];
  let options = [
    Option<"tileSearch", "tile-search", "std::string",
           /*default=*/"\"first-legal\"",
           "Tile candidate search mode: first-legal or min-estimated-time">,
    Option<"maxCandidatesPerDim", "max-candidates-per-dim", "int64_t",
           /*default=*/"8",
           "Maximum refinement ladder entries kept per traversal/reduction dimension">,
    Option<"preferredTileSizes", "preferred-tile-sizes", "std::string",
           /*default=*/"\"128,64,32,16,8,4,2,1\"",
           "Comma-separated tile-size hints used as refinement tie-breakers">,
    Option<"spmBase", "spm-base", "int64_t", /*default=*/"65536",
           "Inclusive byte offset for ordinary tile-local SPM planning">,
    Option<"spmLimit", "spm-limit", "int64_t", /*default=*/"3080192",
           "Exclusive byte limit for ordinary tile-local SPM planning">,
    Option<"spmAlignment", "spm-alignment", "int64_t", /*default=*/"256",
           "Preferred byte alignment for ordinary tile-local SPM planning">,
    Option<"ddrCapacityBytes", "ddr-capacity-bytes", "int64_t",
           /*default=*/"9223372036854775807",
           "Total DDR bytes available to the planned DDR memory footprint">,
    Option<"ddrLargestContiguousBytes", "ddr-largest-contiguous-bytes",
           "int64_t", /*default=*/"9223372036854775807",
           "Largest contiguous DDR range available to a single root demand">,
    Option<"ddrBandwidthLimitBytes", "ddr-bandwidth-limit-bytes", "int64_t",
           /*default=*/"9223372036854775807",
           "Total DDR movement bytes allowed by this planning gate">,
    Option<"ddrAlignmentBytes", "ddr-alignment-bytes", "int64_t",
           /*default=*/"256",
           "Preferred byte alignment for compiler-managed DDR planning">,
    Option<"computeOpsPerCycle", "compute-ops-per-cycle", "int64_t",
           /*default=*/"1024",
           "Rough compute throughput used by min-estimated-time">,
    Option<"ddrBytesPerCycle", "ddr-bytes-per-cycle", "int64_t",
           /*default=*/"256",
           "Rough DDR bandwidth used by min-estimated-time">,
    Option<"spmBytesPerCycle", "spm-bytes-per-cycle", "int64_t",
           /*default=*/"1024",
           "Rough SPM/local movement bandwidth used by min-estimated-time">,
    Option<"instrIssueCycles", "instr-issue-cycles", "int64_t",
           /*default=*/"1",
           "Per-instruction issue overhead used by min-estimated-time">,
    Option<"assumeDdrComputeOverlap", "assume-ddr-compute-overlap", "bool",
           /*default=*/"false",
           "Use max(compute, DDR) instead of compute + DDR in rough timing">,
    Option<"printCandidateSummary", "print-candidate-summary", "bool",
           /*default=*/"false",
           "Print selected candidate summaries to stderr">
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

def PlanSPMMemoryPass
    : Pass<"wafer-plan-spm-memory", "mlir::ModuleOp"> {
  let summary =
      "Plan Wafer SPM memref allocations with tile-local byte offsets";
  let dependentDialects = ["mlir::memref::MemRefDialect",
                           "wafer::WaferDialect",
  ];
  let options = [
    Option<"spmBase", "spm-base", "int64_t", /*default=*/"65536",
           "Inclusive byte offset for ordinary tile-local SPM planning">,
    Option<"spmLimit", "spm-limit", "int64_t", /*default=*/"3080192",
           "Exclusive byte limit for ordinary tile-local SPM planning">,
    Option<"spmAlignment", "spm-alignment", "int64_t", /*default=*/"256",
           "Preferred byte alignment for ordinary tile-local SPM planning">
  ];
}

def PlanDDRMemoryPass
    : Pass<"wafer-plan-ddr-memory", "mlir::ModuleOp"> {
  let summary =
      "Plan and verify DDR memory demands in instruction-level Wafer IR";
  let dependentDialects = ["mlir::async::AsyncDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect", "wafer::WaferDialect",
  ];
  let options = [
    Option<"ddrCapacityBytes", "ddr-capacity-bytes", "int64_t",
           /*default=*/"9223372036854775807",
           "Total DDR bytes available to the planned DDR memory footprint">,
    Option<"ddrLargestContiguousBytes", "ddr-largest-contiguous-bytes",
           "int64_t", /*default=*/"9223372036854775807",
           "Largest contiguous DDR range available to a single root demand">,
    Option<"ddrBandwidthLimitBytes", "ddr-bandwidth-limit-bytes", "int64_t",
           /*default=*/"9223372036854775807",
           "Total DDR movement bytes allowed by this planning gate">,
    Option<"ddrAlignmentBytes", "ddr-alignment-bytes", "int64_t",
           /*default=*/"256",
           "Preferred byte alignment for compiler-managed DDR planning">
  ];
}

#endif // WAFER_TRANSFORMS_PASSES_TD
