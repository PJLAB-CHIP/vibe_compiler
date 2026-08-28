//===- Passes.td - Tile to Instr conversion ---------------*- tablegen -*-===//

#ifndef WAFER_CONVERSION_TILETOINSTR_PASSES_TD
#define WAFER_CONVERSION_TILETOINSTR_PASSES_TD

def ConvertTileRegionToInstrPass
    : Pass<"wafer-convert-tile-region-to-instr", "wafer::TileRegionOp"> {
  let summary = "Convert one executable wafer.tile.region to wafer.instr IR";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::async::AsyncDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "wafer::WaferDialect"];
  let statistics = [
    Statistic<"numDataflowOperationsLowered", "dataflow-ops-lowered",
              "Number of Tile dataflow operations eliminated by conversion">
  ];
}

def ConvertBufferizationCopiesToInstrPass
    : Pass<"wafer-convert-bufferization-copies-to-instr", "mlir::ModuleOp"> {
  let summary =
      "Convert owned bufferization copies and reject ownerless copies";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::async::AsyncDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "wafer::WaferDialect"];
}

#endif // WAFER_CONVERSION_TILETOINSTR_PASSES_TD
