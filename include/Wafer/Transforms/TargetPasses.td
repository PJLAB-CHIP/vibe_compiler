//===- TargetPasses.td - Target materialization/lowering ---*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_TARGETPASSES_TD
#define WAFER_TRANSFORMS_TARGETPASSES_TD

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
  let summary = "Materialize Wafer logical card-partition mesh facts";
  let dependentDialects = ["wafer::WaferDialect"];
  let options = [
    Option<"meshName", "mesh-name", "std::string",
           /*default=*/"\"default_mesh\"",
           "Symbol name for the materialized execution mesh">,
    Option<"axes", "axes", "std::string",
           /*default=*/"\"card_partition\"",
           "Comma-separated logical card-partition axis names">,
    Option<"shape", "shape", "std::string",
           /*default=*/"\"\"",
           "Comma-separated logical mesh axis sizes; must be explicit">
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
  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::LLVM::LLVMDialect",
                           "wafer::WaferDialect"];
  let options = [
    Option<"defaultDDRArenaArgumentIndex",
           "default-ddr-arena-argument-index", "int64_t", /*default=*/"-1",
           "Entry argument carrying the default DDR arena base; -1 rejects "
           "compiler-managed DDR allocations">,
    Option<"physicalCardId", "physical-card-id", "int64_t", /*default=*/"-1",
           "Physical card containing the accepted Tile program; required "
           "when Direct DTE instructions are present">,
    Option<"physicalTileId", "physical-tile-id", "int64_t", /*default=*/"-1",
           "Physical Tile whose accepted Direct DTE endpoints are lowered; "
           "required when Direct DTE instructions are present">,
    Option<"transportStatusArgumentIndex",
           "transport-status-argument-index", "int64_t", /*default=*/"-1",
           "Entry i64 argument carrying the launch-observable Direct DTE "
           "status address; its presence is the Direct DTE lifecycle contract">,
    Option<"transportPreparedBeforeEntry",
           "transport-prepared-before-entry", "bool", /*default=*/"false",
           "Debug direct-lowering projection stating that transport "
           "initialization completed in an earlier launch phase">,
    Option<"profileRecordArgumentIndex",
           "profile-record-argument-index", "int64_t", /*default=*/"-1",
           "Entry i64 argument carrying the per-launch profile record base; "
           "-1 disables profile instrumentation">
  ];
}

#endif // WAFER_TRANSFORMS_TARGETPASSES_TD
