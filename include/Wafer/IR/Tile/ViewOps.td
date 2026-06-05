//===- ViewOps.td - Wafer view operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_VIEWOPS_TD
#define WAFER_IR_OPS_VIEWOPS_TD

def Wafer_ViewReshapeOp : Wafer_Op<"tile.reshape", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    NoMemoryEffect
  ]> {
  let summary = "storage shape-only reshape view";
  let description = [{
    `wafer.tile.reshape` changes the logical tensor shape attached to storage
    without moving data. It is legal only when static element count is
    preserved and physical layout/memory space stay unchanged. If a reshape
    needs physical layout materialization, it must use explicit movement.
  }];

  let arguments = (ins Wafer_AnyStorage:$source);
  let results = (outs Wafer_AnyStorage:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

#endif // WAFER_IR_OPS_VIEWOPS_TD
