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
    Option<"tileSearchEffort", "tile-search-effort", "std::string",
           /*default=*/"\"default\"",
           "Tile candidate search effort: quick, default or deep">,
    Option<"maxCandidatesPerDim", "max-candidates-per-dim", "int64_t",
           /*default=*/"-1",
           "Maximum candidate tile sizes kept per traversal/reduction "
           "dimension; -1 uses tile-search-effort policy and 0 keeps all "
           "shape-driven refinements">,
    Option<"preferredTileSizes", "preferred-tile-sizes", "std::string",
           /*default=*/"\"\"",
           "Comma-separated tile-size hints used as refinement tie-breakers; "
           "empty uses tile-search-effort policy">,
    Option<"maxSearchCandidates", "max-search-candidates", "int64_t",
           /*default=*/"-1",
           "Soft total candidate-visit cap, enforced only after a passing "
           "candidate exists; -1 uses tile-search-effort policy and 0 stops "
           "optimization after the first passing candidate">,
    Option<"searchBeamWidth", "search-beam-width", "int64_t",
           /*default=*/"-1",
           "Maximum refinement neighbors enqueued after the first passing "
           "candidate; -1 uses tile-search-effort policy and 0 disables the "
           "beam cap">,
    Option<"candidateParallelism", "candidate-parallelism", "int64_t",
           /*default=*/"1",
           "Maximum candidate evaluations run concurrently in min-estimated-time">,
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
