//===- TileLoweringPasses.td - Tile/Instr transforms -------*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_TILELOWERINGPASSES_TD
#define WAFER_TRANSFORMS_TILELOWERINGPASSES_TD

def ConvertTileRegionToInstrPass
    : Pass<"wafer-convert-tile-region-to-instr", "wafer::TileRegionOp"> {
  let summary = "Convert one executable wafer.tile.region to wafer.instr IR";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::async::AsyncDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "wafer::WaferDialect"
  ];
  let statistics = [
    Statistic<"numDataflowOperationsLowered", "dataflow-ops-lowered",
              "Number of Tile dataflow operations eliminated by conversion">
  ];
}

def PlaceRequiredNCCJoinsPass
    : Pass<"wafer-place-required-ncc-joins", "mlir::func::FuncOp"> {
  let summary =
      "Place joins required by outstanding NCC accesses in one function";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::async::AsyncDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "wafer::WaferDialect"
  ];
  let statistics = [
    Statistic<"numRequiredJoinsInserted", "required-joins-inserted",
              "Number of required NCC joins inserted in the function">
  ];
}

def RebuildRequiredNCCJoinsPass
    : Pass<"wafer-rebuild-required-ncc-joins", "mlir::func::FuncOp"> {
  let summary =
      "Erase derived NCC joins and recompute required joins in one function";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::async::AsyncDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "wafer::WaferDialect"
  ];
  let statistics = [
    Statistic<"numDerivedJoinsRemoved", "derived-joins-removed",
              "Number of existing derived NCC joins removed before rebuilding">,
    Statistic<"numRequiredJoinsInserted", "required-joins-inserted",
              "Number of required NCC joins present after rebuilding">
  ];
}

#endif // WAFER_TRANSFORMS_TILELOWERINGPASSES_TD
