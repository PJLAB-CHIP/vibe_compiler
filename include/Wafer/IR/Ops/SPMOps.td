//===- SPMOps.td - Wafer SPM operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_SPMOPS_TD
#define WAFER_IR_OPS_SPMOPS_TD

def Wafer_LoadTileOp : Wafer_Op<"load_tile", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_DDRResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "load an external tensor tile into a tile buffer";
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_StoreTileOp : Wafer_Op<"store_tile", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_DDRResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "store a tile buffer to an external tensor boundary";
  let arguments = (ins AnyType:$source, AnyType:$dest);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `,` $dest attr-dict `:` type($source) `->` type($dest)";
}

#endif // WAFER_IR_OPS_SPMOPS_TD
