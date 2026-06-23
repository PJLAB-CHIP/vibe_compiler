//===- CommOps.td - Wafer Comm operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_COMMOPS_TD
#define WAFER_IR_OPS_COMMOPS_TD

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
    `wafer.instr.dte_send` / `dte_recv` / `dte_wait` instructions.
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
