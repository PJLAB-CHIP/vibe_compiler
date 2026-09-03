//===- DTEOps.td - Wafer DTE instruction operation definitions -*- tablegen -*-===//

#ifndef WAFER_IR_INSTR_DTEOPS_TD
#define WAFER_IR_INSTR_DTEOPS_TD

include "mlir/Dialect/Async/IR/AsyncTypes.td"

class Wafer_InstrDTEP2POp<string mnemonic, list<Trait> resourceEffects,
                          list<SideEffect> bufferEffects>
    : Wafer_InstrOp<mnemonic, resourceEffects> {
  let description = [{
    Fixed-size unicast Direct DTE instruction. `peer` is an available physical
    `tile_id` on the selected card. `message` is the stable match identity
    emitted by protocol materialization; it is not a physical channel or
    packet id.
  }];
  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", bufferEffects>:$buffer,
    Optional<I64>:$binding_selector,
    OptionalAttr<I64Attr>:$buffer_offset,
    I64Attr:$peer,
    I64Attr:$bytes,
    Wafer_DTEMessageAttr:$message,
    OptionalAttr<Wafer_DirectDTEBindingAttr>:$binding
  );
  let results = (outs Async_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat =
      "$buffer (`,` $binding_selector^)? attr-dict `:` type($buffer) `->` type($token)";
}

def Wafer_InstrDTESendOp
    : Wafer_InstrDTEP2POp<"instr.dte_send",
        [MemoryEffects<[MemRead<Wafer_SPMResource>,
                        MemWrite<Wafer_CommunicationResource>]>],
        [MemRead]> {
  let summary = "instruction-level Direct DTE fixed-size unicast send";
}

def Wafer_InstrDTERecvOp
    : Wafer_InstrDTEP2POp<"instr.dte_recv",
        [MemoryEffects<[MemWrite<Wafer_SPMResource>,
                        MemWrite<Wafer_CommunicationResource>]>],
        [MemWrite]> {
  let summary = "instruction-level Direct DTE fixed-size unicast receive";
}

class Wafer_InstrDTEMultiSendOp<string mnemonic>
    : Wafer_InstrOp<mnemonic, [
        MemoryEffects<[MemRead<Wafer_SPMResource>,
                       MemWrite<Wafer_CommunicationResource>]>
      ]> {
  let description = [{
    One native Direct DTE multi-destination sender issue. `peers` and
    `messages` are parallel arrays with deterministic order. The canonical
    broadcast grouper sorts physical Tiles; scatter orders destinations by
    increasing contiguous source segment. `bytes` is
    the payload size for each destination. The optional `bindings` array is
    attached atomically by DeviceExecutable transport verification and has one
    `DirectDTEBindingAttr` per destination.

    This operation is formed from actual unicast Instr edges before fresh
    completion. It is not an abstract collective or a persistent schedule.
  }];
  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$buffer,
    I64Attr:$source_offset,
    DenseI64ArrayAttr:$peers,
    I64Attr:$bytes,
    ArrayAttr:$messages,
    OptionalAttr<ArrayAttr>:$bindings
  );
  let results = (outs Async_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat =
      "$buffer attr-dict `:` type($buffer) `->` type($token)";
}

def Wafer_InstrDTEBroadcastOp
    : Wafer_InstrDTEMultiSendOp<"instr.dte_broadcast"> {
  let summary = "instruction-level native Direct DTE broadcast send";
}

def Wafer_InstrDTEScatterOp
    : Wafer_InstrDTEMultiSendOp<"instr.dte_scatter"> {
  let summary = "instruction-level native Direct DTE equal-segment scatter";
}

def Wafer_InstrDTEWaitOp : Wafer_InstrOp<"instr.dte_wait", [
    MemoryEffects<[MemRead<Wafer_CommunicationResource>,
                   MemWrite<Wafer_CommunicationResource>]>
  ]> {
  let summary = "instruction-level Direct DTE completion wait";
  let arguments = (ins Variadic<Async_TokenType>:$tokens);
  let hasVerifier = 1;
  let assemblyFormat = "$tokens attr-dict `:` type($tokens)";
}

#endif // WAFER_IR_INSTR_DTEOPS_TD
