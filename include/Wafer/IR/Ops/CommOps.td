//===- CommOps.td - Wafer Comm operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_COMMOPS_TD
#define WAFER_IR_OPS_COMMOPS_TD

def Wafer_CommSendOp : Wafer_CommP2POp<"comm.send"> {
  let summary = "fixed-size unicast p2p send";
}

def Wafer_CommRecvOp : Wafer_CommP2POp<"comm.recv"> {
  let summary = "fixed-size unicast p2p receive";
}

def Wafer_CommWaitOp : Wafer_Op<"comm.wait", [
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_CommunicationResource>]>
  ]> {
  let summary = "wait for communication completion tokens";
  let arguments = (ins Variadic<AnyType>:$tokens);
  let hasVerifier = 1;
  let assemblyFormat = "$tokens attr-dict `:` type($tokens)";
}

def Wafer_CommAllGatherOp : Wafer_Op<"comm.all_gather", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_CommunicationResource>]>
  ]> {
  let summary = "target-abstract fixed-size ring all-gather request";
  let description = [{
    `wafer.comm.all_gather` is the collective-level communication boundary
    between logical StableHLO all-gather semantics and explicit p2p schedules.
    It carries the local shard buffer, the complete gather buffer, rank/group
    facts, and the fixed byte size of one shard. It does not choose raw DTE
    register modes or hide a communication plan side table; lowering rewrites
    it into explicit `wafer.comm.send` / `recv` / `wait` steps.
  }];

  let arguments = (ins
    AnyType:$local_chunk,
    AnyType:$gather_buffer,
    I64Attr:$local_rank,
    I64Attr:$group_size,
    I64Attr:$bytes
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$local_chunk `into` $gather_buffer attr-dict `:` type($local_chunk) `->` type($gather_buffer)";
}

class Wafer_CommReduceCollectiveOp<string mnemonic> : Wafer_Op<mnemonic, [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>,
                   MemWrite<Wafer_CommunicationResource>]>
  ]> {
  let arguments = (ins
    Wafer_ComputeReduceKindAttr:$kind,
    AnyType:$input,
    AnyType:$recv_buffer,
    I64Attr:$local_rank,
    I64Attr:$group_size,
    I64Attr:$bytes
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $input `using` $recv_buffer attr-dict `:` functional-type(operands, results)";
}

def Wafer_CommReduceScatterOp : Wafer_CommReduceCollectiveOp<"comm.reduce_scatter"> {
  let summary = "target-abstract fixed-size ring reduce-scatter slot request";
  let description = [{
    `wafer.comm.reduce_scatter` represents one local scatter slot of a
    collective reduction. Lowering must express communication as p2p ring
    steps and express local accumulation with compute ops; reduction semantics
    are not hidden in DTE protocol.
  }];
}

def Wafer_CommAllReduceOp : Wafer_CommReduceCollectiveOp<"comm.all_reduce"> {
  let summary = "target-abstract fixed-size ring all-reduce request";
  let description = [{
    `wafer.comm.all_reduce` represents a same-shaped collective reduction.
    V0 lowering uses unicast ring communication plus explicit local
    elementwise accumulation instead of raw DTE non-unicast or packet-side
    reduction.
  }];
}

#endif // WAFER_IR_OPS_COMMOPS_TD
