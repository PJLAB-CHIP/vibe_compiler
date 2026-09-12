//===- Passes.td - Tile current-IR transforms -------------*- tablegen -*-===//

#ifndef WAFER_TRANSFORMS_TILE_PASSES_TD
#define WAFER_TRANSFORMS_TILE_PASSES_TD

def ResolveCurrentLayoutsAndBufferizePass
    : Pass<"wafer-resolve-current-layouts-and-bufferize", "mlir::ModuleOp"> {
  let summary =
      "Resolve current value/use layouts and bufferize TileRegion compute";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::bufferization::BufferizationDialect",
                           "mlir::func::FuncDialect",
                           "mlir::linalg::LinalgDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect",
                           "wafer::WaferDialect"];
}

def HoistInvariantPhysicalMovementPass
    : Pass<"wafer-hoist-invariant-physical-movement", "mlir::func::FuncOp"> {
  let summary = "Reuse private physical copies with invariant source contents";
}

#endif // WAFER_TRANSFORMS_TILE_PASSES_TD
