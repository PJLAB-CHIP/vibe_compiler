//===- SyncOps.td - Wafer Sync operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_SYNCOPS_TD
#define WAFER_IR_OPS_SYNCOPS_TD

def Wafer_SyncLocalDrainOp : Wafer_Op<"sync.local_drain", [
    WaferResourceEffectInterface
  ]> {
  let summary = "local compute/movement visibility drain";
  let assemblyFormat = "attr-dict";
}

#endif // WAFER_IR_OPS_SYNCOPS_TD
