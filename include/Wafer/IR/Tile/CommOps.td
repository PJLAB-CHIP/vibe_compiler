//===- CommOps.td - Wafer Comm operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_COMMOPS_TD
#define WAFER_IR_OPS_COMMOPS_TD

include "mlir/Dialect/Async/IR/AsyncTypes.td"

class Wafer_CommPeerOp<string mnemonic, list<Trait> resourceEffects,
                       list<SideEffect> bufferEffects>
    : Wafer_Op<mnemonic, resourceEffects> {
  let description = [{
    Target-abstract movement of one fixed-size SPM tile between two logical
    execution ranks. `peer` is a logical rank in the current execution mesh
    and `message` is the stable logical match identity. Physical endpoints,
    routes, FSMs, algorithms, pipeline stages, slots, ownership roles and
    costs are deliberately absent.

    The returned token makes issue-to-completion lifetime explicit. A
    matching `async.await` is converted to an exact
    `wafer.instr.dte_wait`; it does not complete any NCC worker domain.
  }];
  let arguments = (ins
    Arg<Wafer_AnySPMMemRef, "", bufferEffects>:$buffer,
    I64Attr:$peer,
    I64Attr:$bytes,
    Wafer_DTEMessageAttr:$message
  );
  let results = (outs Async_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat =
      "$buffer attr-dict `:` type($buffer) `->` type($token)";
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

def Wafer_CommAllGatherOp : Wafer_Op<"tile.all_gather", [
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_CommunicationResource>]>
  ]> {
  let summary = "target-abstract fixed-size all-gather request";
  let description = [{
    `wafer.tile.all_gather` is the collective-level communication boundary
    between logical StableHLO all-gather semantics and explicit p2p schedules.
    It carries the local shard buffer, the complete gather buffer,
    `rank_group` logical execution ranks, `local_rank` as the group-local
    index into that rank group, and the fixed byte size of one shard. It does
    not choose raw DTE register modes, encode physical endpoints, or hide a
    communication plan side table. Lowering derives bounded algorithm
    parameters from the current execution mesh and target topology, then
    rewrites the request into explicit `wafer.instr.dte_send` / `dte_recv` /
    `dte_wait` instructions. The selected algorithm is not part of this op's
    semantics.
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
  let summary = "target-abstract fixed-size all-reduce request";
  let description = [{
    `wafer.tile.all_reduce` represents a same-shaped collective reduction.
    `rank_group` contains logical execution ranks and `local_rank` is the
    group-local index. Physical endpoints and bounded ring/tree parameters are
    derived from the execution mesh and target topology during lowering.
    Lowering uses explicit unicast communication and local elementwise
    accumulation instead of hiding reduction in DTE protocol state.
  }];
}

#endif // WAFER_IR_OPS_COMMOPS_TD
