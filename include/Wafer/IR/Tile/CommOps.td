//===- CommOps.td - Wafer Comm operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_COMMOPS_TD
#define WAFER_IR_OPS_COMMOPS_TD

def Wafer_CommAllGatherOp : Wafer_Op<"tile.all_gather", [
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
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$local_chunk,
    Arg<Wafer_AnySPMMemRef, "", [MemWrite]>:$gather_buffer,
    I64Attr:$local_rank,
    I64Attr:$group_size,
    DenseI64ArrayAttr:$rank_group,
    I64Attr:$bytes,
    I64Attr:$communication_id
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$local_chunk `into` $gather_buffer attr-dict `:` type($local_chunk) `->` type($gather_buffer)";
}

class Wafer_CommReduceCollectiveOp<string mnemonic> : Wafer_Op<mnemonic, [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>,
                   MemWrite<Wafer_CommunicationResource>]>
  ]> {
  let arguments = (ins
    Wafer_ComputeReduceKindAttr:$kind,
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$input,
    Arg<Wafer_AnySPMMemRef, "", [MemRead, MemWrite]>:$recv_buffer,
    I64Attr:$local_rank,
    I64Attr:$group_size,
    DenseI64ArrayAttr:$rank_group,
    I64Attr:$bytes,
    I64Attr:$communication_id
  );
  let results = (outs Wafer_AnySPMMemRef:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $input `using` $recv_buffer attr-dict `:` functional-type(operands, results)";
}

def Wafer_CommReduceScatterOp : Wafer_Op<"tile.reduce_scatter", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>,
                   MemWrite<Wafer_CommunicationResource>]>
  ]> {
  let summary = "target-abstract fixed-size reduce-scatter full-input request";
  let description = [{
    `wafer.tile.reduce_scatter` represents a rank-local reduce-scatter request
    over the full local input buffer. The result and `recv_buffer` have the
    current rank's local scatter-slot shape. `axis` selects the scatter
    dimension in the full input; along that axis, input size must equal result
    size times `group_size`. `rank_group` contains logical execution ranks and
    `local_rank` is the group-local index. Lowering rewrites this op into
    explicit p2p DTE steps plus local accumulation; reduction semantics are not
    hidden in DTE protocol.
  }];
  let arguments = (ins
    Wafer_ComputeReduceKindAttr:$kind,
    Arg<Wafer_AnySPMMemRef, "", [MemRead]>:$input,
    Arg<Wafer_AnySPMMemRef, "", [MemRead, MemWrite]>:$recv_buffer,
    I64Attr:$axis,
    I64Attr:$local_rank,
    I64Attr:$group_size,
    DenseI64ArrayAttr:$rank_group,
    I64Attr:$bytes,
    I64Attr:$communication_id
  );
  let results = (outs Wafer_AnySPMMemRef:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $input `using` $recv_buffer attr-dict `:` functional-type(operands, results)";
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
