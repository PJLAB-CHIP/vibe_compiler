//===- SPMOps.td - Wafer SPM operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_SPMOPS_TD
#define WAFER_IR_OPS_SPMOPS_TD

def Wafer_StorageLoadOp : Wafer_Op<"tile.load", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_DDRResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
]> {
  let summary = "load an external tensor boundary into an SPM buffer";
  let arguments = (ins AnyRankedTensor:$source);
  let results = (outs Wafer_AnySPMMemRef:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_StorageStoreOp : Wafer_Op<"tile.store", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_DDRResource>,
                   MemWrite<Wafer_MovementResource>]>
]> {
  let summary = "store an SPM buffer to an external tensor boundary";
  let arguments = (ins Wafer_AnySPMMemRef:$source, AnyRankedTensor:$dest);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `,` $dest attr-dict `:` type($source) `->` type($dest)";
}

#endif // WAFER_IR_OPS_SPMOPS_TD
