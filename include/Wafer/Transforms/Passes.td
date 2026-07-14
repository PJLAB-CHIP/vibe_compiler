//===- Passes.td - Wafer pass declarations ----------------*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_PASSES_TD
#define WAFER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def MaterializeTargetTopologyPass
    : Pass<"wafer-materialize-target-topology", "mlir::ModuleOp"> {
  let summary = "Materialize regular Wafer target topology facts";
  let dependentDialects = ["wafer::WaferDialect"];
  let options = [
    Option<"topologyName", "topology-name", "std::string",
           /*default=*/"\"default\"",
           "Symbol name for the materialized target topology">,
    Option<"cardYCount", "card-y", "int64_t", /*default=*/"1",
           "Number of Wafer card rows in the target topology">,
    Option<"cardXCount", "card-x", "int64_t", /*default=*/"1",
           "Number of Wafer card columns in the target topology">,
    Option<"cardInterconnect", "card-interconnect", "std::string",
           /*default=*/"\"mesh\"",
           "Card-level C2C topology kind: mesh or torus">,
    Option<"tileYCount", "tile-y", "int64_t", /*default=*/"4",
           "Number of tile rows per card in the target topology">,
    Option<"tileXCount", "tile-x", "int64_t", /*default=*/"4",
           "Number of tile columns per card in the target topology">,
    Option<"unavailableTiles", "unavailable-tiles", "std::string",
           /*default=*/"\"\"",
           "Comma-separated card_y,card_x,tile_y,tile_x tuples for unavailable "
           "tile endpoints">
  ];
}

def MaterializeExecutionMeshPass
    : Pass<"wafer-materialize-execution-mesh", "mlir::ModuleOp"> {
  let summary = "Materialize Wafer program execution mesh facts";
  let dependentDialects = ["wafer::WaferDialect"];
  let options = [
    Option<"meshName", "mesh-name", "std::string",
           /*default=*/"\"default_mesh\"",
           "Symbol name for the materialized execution mesh">,
    Option<"topologyName", "topology-name", "std::string",
           /*default=*/"\"default\"",
           "Referenced target topology symbol name">,
    Option<"policy", "policy", "std::string",
           /*default=*/"\"all_available\"",
           "Execution mesh endpoint policy: all_available or explicit">,
    Option<"axes", "axes", "std::string",
           /*default=*/"\"\"",
           "Comma-separated logical mesh axis names; empty uses rank">,
    Option<"shape", "shape", "std::string",
           /*default=*/"\"\"",
           "Comma-separated logical mesh axis sizes; empty uses rank count">,
    Option<"endpoints", "endpoints", "std::string",
           /*default=*/"\"\"",
           "Comma-separated card_y,card_x,tile_y,tile_x tuples for explicit "
           "execution endpoints">
  ];
}

def StripTargetMetadataPass
    : Pass<"wafer-strip-target-metadata", "mlir::ModuleOp"> {
  let summary =
      "Strip consumed Wafer target topology and execution mesh metadata";
  let dependentDialects = ["wafer::WaferDialect"];
}

def LowerInstrToTargetLLVMPass
    : Pass<"wafer-lower-instr-to-target-llvm", "mlir::ModuleOp"> {
  let summary =
      "Lower memory-planned wafer.instr ops to target LLVM CRT calls";
  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "wafer::WaferDialect"];
  let options = [
    Option<"targetProfile", "target-profile", "std::string",
           /*default=*/"\"\"",
           "Required registered target profile for debug direct lowering">,
    Option<"defaultDDRArenaArgumentIndex",
           "default-ddr-arena-argument-index", "int64_t", /*default=*/"-1",
           "Entry argument carrying the default DDR arena base; -1 rejects "
           "compiler-managed DDR allocations">,
    Option<"logicalRank", "logical-rank", "int64_t", /*default=*/"-1",
           "Logical rank whose accepted Direct DTE endpoints are lowered; "
           "required when Direct DTE instructions are present">,
    Option<"transportStatusArgumentIndex",
           "transport-status-argument-index", "int64_t", /*default=*/"-1",
           "Entry i64 argument carrying the launch-observable Direct DTE "
           "status address; required when Direct DTE instructions are present">
  ];
}

def FormLogicalGroupsPass
    : Pass<"wafer-form-logical-groups", "mlir::ModuleOp"> {
  let summary = "Form root-seeded tensor-level logical wafer.group ops";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::math::MathDialect", "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect"
  ];
}

def DumpGroupTilingDemandPass
    : Pass<"wafer-dump-group-tiling-demand", "mlir::ModuleOp"> {
  let summary = "Dump wafer.group tiling demand analysis";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect"
  ];
}

def DumpGroupLayoutPlanPass
    : Pass<"wafer-dump-group-layout-plan", "mlir::ModuleOp"> {
  let summary = "Dump wafer.group layout planning analysis";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect"
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
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect"
  ];
  let options = [
    Option<"logicalRank", "logical-rank", "int64_t", /*default=*/"-1",
           "Explicit logical execution rank used to materialize group-local "
           "communication rank facts; required for modules containing groups">
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
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect"
  ];
  let options = [
    Option<"logicalRank", "logical-rank", "int64_t", /*default=*/"-1",
           "Explicit logical execution rank used to materialize candidate "
           "communication facts; required for modules containing groups">,
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
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect"
  ];
  let options = [
    Option<"logicalRank", "logical-rank", "int64_t", /*default=*/"-1",
           "Explicit logical execution rank used to materialize group-local "
           "communication facts; required for modules containing groups">,
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
           "Maximum candidate evaluations run concurrently in "
           "min-estimated-time; ignored by first-legal">,
    Option<"printCandidateSummary", "print-candidate-summary", "bool",
           /*default=*/"false",
           "Print selected candidate summaries to stderr">
  ];
}

def NormalizeStablehloCollectivesPass
    : Pass<"wafer-normalize-stablehlo-collectives", "mlir::ModuleOp"> {
  let summary =
      "Normalize post-SPMD StableHLO collectives to Wafer linalg extension collectives";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect"
  ];
}

def ConvertGroupToTileRegionPass
    : Pass<"wafer-convert-group-to-tile-region", "mlir::ModuleOp"> {
  let summary = "Convert logical wafer.group ops to wafer.tile.region IR";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::async::AsyncDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::func::FuncDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::math::MathDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect", "wafer::WaferDialect"
  ];
  let options = [
    Option<"logicalRank", "logical-rank", "int64_t", /*default=*/"-1",
           "Explicit logical execution rank used to materialize group-local "
           "communication rank facts; required for modules containing groups">
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
                           "wafer::WaferDialect"
  ];
  let options = [
    Option<"allGatherSchedule", "all-gather-schedule", "std::string",
           /*default=*/"\"auto\"",
           "All-gather schedule policy: auto, ring or direct">,
    Option<"allReduceSchedule", "all-reduce-schedule", "std::string",
           /*default=*/"\"auto\"",
           "All-reduce schedule policy: auto, ring or tree">,
    Option<"reduceScatterSchedule", "reduce-scatter-schedule", "std::string",
           /*default=*/"\"auto\"",
           "Reduce-scatter schedule policy: auto or direct">
  ];
}

def PlanSPMMemoryPass
    : Pass<"wafer-plan-spm-memory", "mlir::ModuleOp"> {
  let summary =
      "Plan Wafer SPM memref allocations with tile-local byte offsets";
  let dependentDialects = ["mlir::memref::MemRefDialect",
                           "wafer::WaferDialect"
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
                           "mlir::scf::SCFDialect", "wafer::WaferDialect"
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
