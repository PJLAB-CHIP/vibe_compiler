//===- SPMOps.td - Wafer SPM operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_SPMOPS_TD
#define WAFER_IR_OPS_SPMOPS_TD

def Wafer_AllocTileOp : Wafer_Op<"alloc_tile", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemWrite<Wafer_SPMResource>]>
  ]> {
  let summary = "create an abstract tile-local buffer allocation demand";
  let description = [{
    `wafer.alloc_tile` creates a tile-local buffer value before physical SPM
    offset allocation. It records shape, layout, and memory-space demand in
    the result type only; R3.2e assigns the concrete SPM window.
  }];

  let results = (outs Wafer_AnyTileBuffer:$result);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Wafer_LoadTileOp : Wafer_Op<"load_tile", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_DDRResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "load an external tensor tile into a tile buffer";
  let arguments = (ins AnyRankedTensor:$source);
  let results = (outs Wafer_AnyTileBuffer:$result);
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
  let arguments = (ins Wafer_AnyTileBuffer:$source, AnyRankedTensor:$dest);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `,` $dest attr-dict `:` type($source) `->` type($dest)";
}

#endif // WAFER_IR_OPS_SPMOPS_TD
