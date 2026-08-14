//===- ViewOps.td - Wafer view operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_VIEWOPS_TD
#define WAFER_IR_OPS_VIEWOPS_TD

def Wafer_ViewReshapeOp
    : Wafer_Op<"tile.reshape", [Pure, ViewLikeOpInterface,
                                 WaferTileDataflowOpInterface]> {
  let summary = "tile-local logical reshape";
  let description = [{
    `wafer.tile.reshape` changes the logical tensor shape attached to storage
    while preserving canonical logical linear element order, static element
    count, element type, memory space, and layout marker. Result multi-indices
    are reinterpreted through the new shape; they are not required to equal the
    source multi-indices. The operation is always an aliasing metadata view;
    when a selected physical layout cannot preserve the same physical element
    mapping, the producer must materialize explicit movement before creating
    this view.
  }];

  let arguments = (ins Wafer_AnySPMMemRef:$source);
  let results = (outs Wafer_AnySPMMemRef:$result);
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

#endif // WAFER_IR_OPS_VIEWOPS_TD
