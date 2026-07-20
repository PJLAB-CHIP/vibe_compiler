//===- SyncOps.td - Wafer Sync operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_SYNCOPS_TD
#define WAFER_IR_OPS_SYNCOPS_TD

def Wafer_SyncLocalFenceOp : Wafer_Op<"instr.local_fence", [
    MemoryEffects<[MemWrite<Wafer_SyncResource>]>
  ]> {
  let summary = "local compute/movement visibility fence";
  let assemblyFormat = "attr-dict";
}

#endif // WAFER_IR_OPS_SYNCOPS_TD
