//===- SPMOps.td - Wafer SPM operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_SPMOPS_TD
#define WAFER_IR_OPS_SPMOPS_TD

def Wafer_StorageAllocOp : Wafer_Op<"storage.alloc", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemWrite<Wafer_SPMResource>]>
  ]> {
  let summary = "create an abstract tile-local storage allocation demand";
  let description = [{
    `wafer.storage.alloc` creates a storage value before physical SPM offset
    allocation. It records shape, layout, and memory-space demand in the result
    type only; R3.2e assigns the concrete SPM window.
  }];

  let results = (outs Wafer_AnyStorage:$result);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Wafer_StorageLoadOp : Wafer_Op<"storage.load", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_DDRResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "load an external tensor slice into storage";
  let arguments = (ins AnyRankedTensor:$source);
  let results = (outs Wafer_AnyStorage:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_StorageStoreOp : Wafer_Op<"storage.store", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_DDRResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "store storage to an external tensor boundary";
  let arguments = (ins Wafer_AnyStorage:$source, AnyRankedTensor:$dest);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `,` $dest attr-dict `:` type($source) `->` type($dest)";
}

#endif // WAFER_IR_OPS_SPMOPS_TD
