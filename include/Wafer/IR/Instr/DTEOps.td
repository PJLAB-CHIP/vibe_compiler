//===- DTEOps.td - Wafer DTE instruction operation definitions -*- tablegen -*-===//

#ifndef WAFER_IR_INSTR_DTEOPS_TD
#define WAFER_IR_INSTR_DTEOPS_TD

include "mlir/Dialect/Async/IR/AsyncTypes.td"

class Wafer_InstrDTEP2POp<string mnemonic> : Wafer_InstrOp<mnemonic, [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_CommunicationResource>]>
  ]> {
  let description = [{
    Fixed-size unicast Direct DTE instruction. `peer` is a logical
    execution-rank id in the current `wafer.execution.mesh`, not an encoded
    physical tile id. The physical endpoint is derived from
    `wafer.execution.mesh` and `wafer.target.topology` at the DTE instruction
    lowering boundary. `message` is the stable logical match identity emitted
    by protocol materialization; it is not a physical channel or packet id.
  }];
  let arguments = (ins
    Wafer_AnySPMMemRef:$buffer,
    I64Attr:$peer,
    I64Attr:$bytes,
    Wafer_DTEMessageAttr:$message,
    OptionalAttr<Wafer_DirectDTEBindingAttr>:$binding
  );
  let results = (outs Async_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat =
      "$buffer attr-dict `:` type($buffer) `->` type($token)";
}

def Wafer_InstrDTESendOp : Wafer_InstrDTEP2POp<"instr.dte_send"> {
  let summary = "instruction-level Direct DTE fixed-size unicast send";
}

def Wafer_InstrDTERecvOp : Wafer_InstrDTEP2POp<"instr.dte_recv"> {
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
