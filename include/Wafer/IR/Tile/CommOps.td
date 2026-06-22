//===- CommOps.td - Wafer Comm operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_COMMOPS_TD
#define WAFER_IR_OPS_COMMOPS_TD

include "mlir/Dialect/Async/IR/AsyncTypes.td"

class Wafer_CommP2POp<string mnemonic> : Wafer_Op<mnemonic, [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_CommunicationResource>]>
  ]> {
  let description = [{
    Fixed-size unicast tile communication. `peer` is a logical execution-rank
    id in the current `wafer.execution.mesh`, not an encoded physical tile id.
    The physical endpoint is derived from `wafer.execution.mesh` and
    `wafer.target.topology` at the communication / Direct DTE lowering
    boundary.
  }];
  let arguments = (ins
    Wafer_AnySPMMemRef:$buffer,
    I64Attr:$peer,
    I64Attr:$bytes
  );
  let results = (outs Async_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat =
      "$buffer attr-dict `:` type($buffer) `->` type($token)";
}

def Wafer_CommSendOp : Wafer_CommP2POp<"tile.send"> {
  let summary = "fixed-size unicast p2p send";
}

def Wafer_CommRecvOp : Wafer_CommP2POp<"tile.recv"> {
  let summary = "fixed-size unicast p2p receive";
}

def Wafer_CommWaitOp : Wafer_Op<"tile.wait", [
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_CommunicationResource>]>
  ]> {
  let summary = "wait for communication completion tokens";
  let arguments = (ins Variadic<Async_TokenType>:$tokens);
  let hasVerifier = 1;
  let assemblyFormat = "$tokens attr-dict `:` type($tokens)";
}

def Wafer_CommAllGatherOp : Wafer_Op<"tile.all_gather", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_CommunicationResource>]>
  ]> {
  let summary = "target-abstract fixed-size ring all-gather request";
  let description = [{
    `wafer.tile.all_gather` is the collective-level communication boundary
    between logical StableHLO all-gather semantics and explicit p2p schedules.
    It carries the local shard buffer, the complete gather buffer,
    `rank_group` logical execution ranks, `local_rank` as the group-local
    index into that rank group, and the fixed byte size of one shard. It does
    not choose raw DTE register modes, encode physical endpoints, or hide a
    communication plan side table; lowering rewrites it into explicit
    `wafer.tile.send` / `recv` / `wait` steps.
  }];

  let arguments = (ins
    Wafer_AnySPMMemRef:$local_chunk,
    Wafer_AnySPMMemRef:$gather_buffer,
    I64Attr:$local_rank,
    I64Attr:$group_size,
    DenseI64ArrayAttr:$rank_group,
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
    Wafer_AnySPMMemRef:$input,
    Wafer_AnySPMMemRef:$recv_buffer,
    I64Attr:$local_rank,
    I64Attr:$group_size,
    DenseI64ArrayAttr:$rank_group,
    I64Attr:$bytes
  );
  let results = (outs Wafer_AnySPMMemRef:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $input `using` $recv_buffer attr-dict `:` functional-type(operands, results)";
}

def Wafer_CommReduceScatterOp : Wafer_CommReduceCollectiveOp<"tile.reduce_scatter"> {
  let summary = "target-abstract fixed-size ring reduce-scatter slot request";
  let description = [{
    `wafer.tile.reduce_scatter` represents one local scatter slot of a
    collective reduction. `rank_group` contains logical execution ranks and
    `local_rank` is the group-local index. Physical endpoints are derived from
    the execution mesh and target topology during lowering. Lowering must
    express communication as p2p ring steps and express local accumulation with
    compute ops; reduction semantics are not hidden in DTE protocol.
  }];
}

def Wafer_CommAllReduceOp : Wafer_CommReduceCollectiveOp<"tile.all_reduce"> {
  let summary = "target-abstract fixed-size ring all-reduce request";
  let description = [{
    `wafer.tile.all_reduce` represents a same-shaped collective reduction.
    `rank_group` contains logical execution ranks and `local_rank` is the
    group-local index. Physical endpoints are derived from the execution mesh
    and target topology during lowering. V0 lowering uses unicast ring
    communication plus explicit local elementwise accumulation instead of raw
    DTE non-unicast or packet-side reduction.
  }];
}

#endif // WAFER_IR_OPS_COMMOPS_TD
