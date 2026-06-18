//===- ShardOps.td - Wafer shard binding operation definitions -*- tablegen -*-===//

#ifndef WAFER_IR_RESOURCE_SHARDOPS_TD
#define WAFER_IR_RESOURCE_SHARDOPS_TD

def Wafer_ShardBindingOp : Wafer_Op<"shard.binding"> {
  let summary = "launch-visible tensor shard binding by logical rank";
  let description = [{
    `wafer.shard.binding` records how a launch-visible tensor boundary value is
    sliced across dense logical ranks. It references an entry function symbol
    and function argument index, then stores per-rank static slice bounds as
    rank-major arrays. The op is a boundary shard fact: it does not store
    payload file paths, physical placement, SPM/DDR offsets, package manifest
    fields, or runtime handles.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$kernel,
    I64Attr:$argument_index,
    I64Attr:$logical_rank_count,
    DenseI64ArrayAttr:$global_shape,
    DenseI64ArrayAttr:$local_shape,
    DenseI64ArrayAttr:$shard_ranks,
    DenseI64ArrayAttr:$shard_offsets,
    DenseI64ArrayAttr:$shard_sizes,
    DenseI64ArrayAttr:$shard_strides
  );

  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";
}

#endif // WAFER_IR_RESOURCE_SHARDOPS_TD
