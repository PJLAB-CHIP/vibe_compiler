//===- ViewOps.td - Wafer view operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_VIEWOPS_TD
#define WAFER_IR_OPS_VIEWOPS_TD

def Wafer_ViewReshapeOp : Wafer_Op<"tile.reshape", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>
  ]> {
  let summary = "tile-local logical reshape";
  let description = [{
    `wafer.tile.reshape` changes the logical tensor shape attached to storage
    while preserving canonical logical linear element order, static element
    count, element type, memory space, and layout marker. Result multi-indices
    are reinterpreted through the new shape; they are not required to equal the
    source multi-indices. Instruction lowering may represent that logical
    reindexing as a metadata view when the current physical layout can alias the
    same linear element order, or materialize it with explicit movement when the
    target physical interpretation requires data rearrangement.
  }];

  let arguments = (ins Wafer_AnySPMMemRef:$source);
  let results = (outs Wafer_AnySPMMemRef:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

#endif // WAFER_IR_OPS_VIEWOPS_TD
