//===- CommOps.td - Wafer Comm operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_COMMOPS_TD
#define WAFER_IR_OPS_COMMOPS_TD

include "mlir/Dialect/Async/IR/AsyncTypes.td"

class Wafer_CommPeerOp<string mnemonic, list<Trait> resourceEffects,
                       list<SideEffect> bufferEffects>
    : Wafer_Op<mnemonic, [WaferTileDataflowOpInterface] # resourceEffects> {
  let description = [{
    Target-abstract movement of one fixed-size SPM tile between two physical
    Tiles on the selected card. `peer` is an available physical `tile_id` and
    `message` is the stable match identity. Routes, FSMs, pipeline stages,
    launch slots, ownership roles and costs are deliberately absent.

    The returned token makes issue-to-completion lifetime explicit. A
    matching `async.await` is converted to an exact
    `wafer.instr.dte_wait`; it does not complete any NCC worker domain.
  }];
  let arguments = (ins Arg<Wafer_AnySPMMemRef, "", bufferEffects>:$buffer,
      I64Attr:$peer, I64Attr:$bytes, Wafer_DTEMessageAttr:$message);
  let results = (outs Async_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat = "$buffer attr-dict `:` type($buffer) `->` type($token)";
}

def Wafer_CommPeerSendOp
    : Wafer_CommPeerOp<"tile.peer_send",
                       [MemoryEffects<[MemRead<Wafer_SPMResource>,
                                       MemWrite<Wafer_CommunicationResource>]>],
                       [MemRead]> {
  let summary = "target-abstract fixed-size peer tile send";
}

def Wafer_CommPeerRecvOp
    : Wafer_CommPeerOp<"tile.peer_recv",
                       [MemoryEffects<[MemWrite<Wafer_SPMResource>,
                                       MemWrite<Wafer_CommunicationResource>]>],
                       [MemWrite]> {
  let summary = "target-abstract fixed-size peer tile receive";
}

#endif // WAFER_IR_OPS_COMMOPS_TD
