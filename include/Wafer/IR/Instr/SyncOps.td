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
  let arguments = (ins DenseI64ArrayAttr:$participants,
                       OptionalAttr<UnitAttr>:$schedule_boundary);
  let builders = [OpBuilder<(ins "::wafer::NCCWorker":$participant), [{
      int64_t ordinal = static_cast<int64_t>(participant);
      build($_builder, $_state, ::llvm::ArrayRef<int64_t>(ordinal),
            ::mlir::UnitAttr{});
    }]>];
  let hasVerifier = 1;
  let assemblyFormat = "$participants attr-dict";
}

class Wafer_DDRPublicationOp<string mnemonic, list<SideEffect> readyEffects>
    : Wafer_Op<mnemonic, [MemoryEffects<[MemWrite<Wafer_SyncResource>]>]> {
  let arguments = (ins Arg<AnyMemRef, "published DDR data", [MemRead]>:$data,
      Arg<AnyMemRef, "invocation-owned completion word", readyEffects>:$ready);
  let assemblyFormat =
      "$data `,` $ready attr-dict `:` type($data) `,` type($ready)";
  let hasVerifier = 1;
}

def Wafer_SyncDDRPublishOp
    : Wafer_DDRPublicationOp<"instr.ddr_publish", [MemWrite]> {
  let summary = "publish completed immutable shared DDR data";
}

def Wafer_SyncDDRAcquireOp
    : Wafer_DDRPublicationOp<"instr.ddr_acquire", [MemRead]> {
  let summary = "wait for the matching shared DDR publication before reading";
}

#endif // WAFER_IR_OPS_SYNCOPS_TD
