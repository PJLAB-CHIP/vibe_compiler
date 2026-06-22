//===- ShardOps.td - Wafer boundary shard operation definitions -*- tablegen -*-===//

#ifndef WAFER_IR_RESOURCE_SHARDOPS_TD
#define WAFER_IR_RESOURCE_SHARDOPS_TD

def Wafer_BoundaryShardsOp : Wafer_Op<"boundary.shards", [Symbol]> {
  let summary = "launch-visible tensor boundary shards by logical rank";
  let description = [{
    `wafer.boundary.shards` records how a launch-visible function boundary
    tensor value is sliced across dense logical ranks. Function argument/result
    attributes reference this symbol with `wafer.boundary_shards`; ownership is
    carried by the boundary value, not by a kernel/index side table. The op
    stores per-rank static slice bounds as rank-major arrays and does not store
    payload file paths, physical endpoint mapping, SPM/DDR offsets, package manifest
    fields, or runtime handles.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$execution_mesh,
    DenseI64ArrayAttr:$global_shape,
    DenseI64ArrayAttr:$local_shape,
    DenseI64ArrayAttr:$shard_ranks,
    DenseI64ArrayAttr:$shard_offsets,
    DenseI64ArrayAttr:$shard_sizes,
    DenseI64ArrayAttr:$shard_strides
  );

  let hasVerifier = 1;
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // WAFER_IR_RESOURCE_SHARDOPS_TD
