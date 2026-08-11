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
