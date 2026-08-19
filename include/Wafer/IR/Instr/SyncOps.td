//===- SyncOps.td - Wafer Sync operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_SYNCOPS_TD
#define WAFER_IR_OPS_SYNCOPS_TD

def Wafer_SyncNCCJoinOp
    : Wafer_Op<
          "instr.ncc_join", [
            DeclareOpInterfaceMethods<WaferNCCCompletionOpInterface>,
            MemoryEffects<[MemWrite<Wafer_SyncResource>]>
          ]> {
  let summary = "typed NCC worker-participant completion join";
  let description = [{
    Completes the outstanding NCC issue stream for exactly the canonical,
    non-empty `participants` worker set. This operation does not complete a
    Direct DTE event, make cacheable host memory visible, or act as a multi-tile
    barrier.
  }];
  let arguments = (ins DenseI64ArrayAttr:$participants);
  let builders = [OpBuilder<(ins "::wafer::NCCWorker":$participant), [{
      int64_t ordinal = static_cast<int64_t>(participant);
      build($_builder, $_state, ::llvm::ArrayRef<int64_t>(ordinal));
    }]>];
  let hasVerifier = 1;
  let assemblyFormat = "$participants attr-dict";
}

#endif // WAFER_IR_OPS_SYNCOPS_TD
