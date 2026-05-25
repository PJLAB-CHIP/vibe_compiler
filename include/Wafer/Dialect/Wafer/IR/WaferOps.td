//===- WaferOps.td - Wafer operation definitions ----------*- tablegen -*-===//

#ifndef WAFER_DIALECT_WAFER_IR_WAFEROPS_TD
#define WAFER_DIALECT_WAFER_IR_WAFEROPS_TD

include "Wafer/Dialect/Wafer/IR/WaferBase.td"
include "Wafer/Dialect/Wafer/IR/WaferAttrs.td"
include "Wafer/Dialect/Wafer/IR/WaferInterfaces.td"
include "Wafer/Dialect/Wafer/IR/WaferTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Wafer_GroupOp : Wafer_Op<"group", [
    AttrSizedOperandSegments,
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"GroupYieldOp">
  ]> {
  let summary = "tensor-level Wafer grouping and scheduling boundary";
  let description = [{
    `wafer.group` carries a tensor-level fusion/scheduling candidate. It owns
    explicit inputs, explicit destination-style outputs, and a single body
    region. Tile sizes, physical addresses, queues, packets, runtime buffers,
    and planner-local search state are deliberately outside this operation.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyTensor>:$outs
  );
  let results = (outs Variadic<AnyTensor>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    $body attr-dict `:` type($results)
  }];
}

def Wafer_GroupYieldOp : Wafer_Op<"group_yield", [
    HasParent<"GroupOp">,
    Pure,
    ReturnLike,
    Terminator
  ]> {
  let summary = "terminate a wafer.group body";
  let description = [{
    `wafer.group_yield` returns tensor-level values from a `wafer.group` body.
    The parent `wafer.group` verifier checks the yielded value count and types
    against the group results.
  }];

  let arguments = (ins Variadic<AnyTensor>:$values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{ attr-dict ($values^ `:` type($values))? }];
}

def Wafer_TileRegionOp : Wafer_Op<"tile_region", [
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"TileYieldOp">
  ]> {
  let summary = "tile-local Wafer execution region";
  let description = [{
    `wafer.tile_region` is the post-group device-side execution scope for
    tile-local buffers, layout materialization, movement, target-abstract
    compute, communication, and sync/effect ordering. It does not store group
    planner search state, raw packets, runtime buffer handles, or C ABI calls.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $inputs `:` type($inputs) `)` `->` `(` type($results) `)`
    $body attr-dict
  }];
}

def Wafer_TileYieldOp : Wafer_Op<"tile_yield", [
    HasParent<"TileRegionOp">,
    Pure,
    ReturnLike,
    Terminator
  ]> {
  let summary = "terminate a wafer.tile_region body";
  let description = [{
    `wafer.tile_yield` returns values from a `wafer.tile_region` body. The
    parent verifier checks value count, value types, and tile-buffer boundary
    ownership.
  }];

  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{ attr-dict ($values^ `:` type($values))? }];
}

def Wafer_LayoutMaterializeOp : Wafer_Op<"layout.materialize", [
    WaferLayoutOpInterface
  ]> {
  let summary = "explicit tile-buffer layout materialization";
  let description = [{
    `wafer.layout.materialize` represents real device-side data movement from
    one Wafer physical layout family to another. It preserves the logical
    tensor type and memory space; only the tile-buffer `mem_layout` changes.
    Same-layout materialization is not a semantic no-op to keep around and
    must be removed by canonicalization instead.
  }];

  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);

  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_LoadTileOp : Wafer_Op<"load_tile", [
    WaferLayoutOpInterface,
    WaferResourceEffectInterface
  ]> {
  let summary = "load an external tensor tile into a tile buffer";
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_StoreTileOp : Wafer_Op<"store_tile", [
    WaferLayoutOpInterface,
    WaferResourceEffectInterface
  ]> {
  let summary = "store a tile buffer to an external tensor boundary";
  let arguments = (ins AnyType:$source, AnyType:$dest);
  let hasVerifier = 1;
  let assemblyFormat =
      "$source `,` $dest attr-dict `:` type($source) `->` type($dest)";
}

def Wafer_DdrExternalBindingOp : Wafer_Op<"ddr.external_binding"> {
  let summary = "launch-visible external DDR binding demand";
  let description = [{
    `wafer.ddr.external_binding` records the compact external tensor binding
    contract that the launch/runtime layer must satisfy for user-visible input
    and output tensors. It carries byte size, alignment and visibility policy,
    but not a physical DDR address, BO handle, pool placement, or allocation
    trace.
  }];

  let arguments = (ins
    Wafer_DdrBindingKindAttr:$kind,
    AnyTensor:$value,
    I64Attr:$bytes,
    I64Attr:$alignment,
    BoolAttr:$read_only,
    BoolAttr:$host_visible
  );

  let hasVerifier = 1;
  let assemblyFormat = "$kind $value attr-dict `:` type($value)";
}

def Wafer_PlacementMapOp : Wafer_Op<"placement.map"> {
  let summary = "accepted logical-rank to physical-tile mapping";
  let description = [{
    `wafer.placement.map` records an accepted placement mapping from dense
    logical ranks to Wafer physical tile coordinates. Coordinates are flattened
    as `[card_y, card_x, tile_y, tile_x]` tuples in logical rank order. The op
    carries only the accepted mapping and target topology facts needed for
    verification; candidate search trace, cost scores, SPM/DDR allocation, DTE
    packets, and runtime handles are not placement-map semantics.
  }];

  let arguments = (ins
    I64Attr:$logical_rank_count,
    DenseI64ArrayAttr:$physical_tile_coords,
    I64Attr:$card_y_count,
    I64Attr:$card_x_count,
    I64Attr:$tile_y_count,
    I64Attr:$tile_x_count,
    DenseI64ArrayAttr:$bad_tile_ids
  );

  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";
}

def Wafer_AbiRdma1DOp : Wafer_Op<"abi.rdma_1d", [
    WaferResourceEffectInterface
  ]> {
  let summary = "C ABI skeleton DDR-to-SPM 1D DMA issue";
  let arguments = (ins
    Wafer_AbiWaitPolicyAttr:$wait_policy,
    AnyType:$source,
    I64Attr:$bytes
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$wait_policy $source attr-dict `:` type($source) `->` type($result)";
}

def Wafer_AbiWdma1DOp : Wafer_Op<"abi.wdma_1d", [
    WaferResourceEffectInterface
  ]> {
  let summary = "C ABI skeleton SPM-to-DDR 1D DMA issue";
  let arguments = (ins
    Wafer_AbiWaitPolicyAttr:$wait_policy,
    AnyType:$source,
    AnyType:$dest,
    I64Attr:$bytes
  );
  let hasVerifier = 1;
  let assemblyFormat =
      "$wait_policy $source `,` $dest attr-dict `:` type($source) `->` type($dest)";
}

def Wafer_AbiGemmOp : Wafer_Op<"abi.gemm", [
    WaferResourceEffectInterface
  ]> {
  let summary = "C ABI skeleton GEMM issue";
  let arguments = (ins
    Wafer_AbiWaitPolicyAttr:$wait_policy,
    AnyType:$lhs,
    AnyType:$rhs,
    I64Attr:$m,
    I64Attr:$k,
    I64Attr:$n
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$wait_policy $lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def Wafer_AbiElementwiseOp : Wafer_Op<"abi.elementwise", [
    WaferResourceEffectInterface
  ]> {
  let summary = "C ABI skeleton elementwise issue";
  let arguments = (ins
    Wafer_AbiWaitPolicyAttr:$wait_policy,
    Wafer_ComputeElementwiseKindAttr:$kind,
    Variadic<AnyType>:$inputs
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$wait_policy $kind $inputs attr-dict `:` functional-type(operands, results)";
}

def Wafer_ComputeGemmOp : Wafer_Op<"compute.gemm", [
    WaferLayoutOpInterface,
    WaferResourceEffectInterface
  ]> {
  let summary = "target-abstract tile GEMM";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def Wafer_ComputeElementwiseOp : Wafer_Op<"compute.elementwise", [
    WaferLayoutOpInterface,
    WaferResourceEffectInterface
  ]> {
  let summary = "target-abstract tile elementwise compute";
  let description = [{
    `wafer.compute.elementwise` represents tile-local scalar elementwise
    compute over already materialized SPM tile buffers. The V0 operation keeps
    broadcast planning outside the IR contract: all operands and the result
    must have the same logical tensor type, memory space, and physical layout.
  }];

  let arguments = (ins
    Wafer_ComputeElementwiseKindAttr:$kind,
    Variadic<AnyType>:$inputs
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $inputs attr-dict `:` functional-type(operands, results)";
}

class Wafer_CommP2POp<string mnemonic> : Wafer_Op<mnemonic, [
    WaferResourceEffectInterface
  ]> {
  let arguments = (ins
    AnyType:$buffer,
    I64Attr:$peer,
    I64Attr:$bytes
  );
  let results = (outs AnyType:$token);
  let hasVerifier = 1;
  let assemblyFormat =
      "$buffer attr-dict `:` type($buffer) `->` type($token)";
}

def Wafer_CommSendOp : Wafer_CommP2POp<"comm.send"> {
  let summary = "fixed-size unicast p2p send";
}

def Wafer_CommRecvOp : Wafer_CommP2POp<"comm.recv"> {
  let summary = "fixed-size unicast p2p receive";
}

def Wafer_CommWaitOp : Wafer_Op<"comm.wait", [
    WaferResourceEffectInterface
  ]> {
  let summary = "wait for communication completion tokens";
  let arguments = (ins Variadic<AnyType>:$tokens);
  let hasVerifier = 1;
  let assemblyFormat = "$tokens attr-dict `:` type($tokens)";
}

def Wafer_SyncLocalDrainOp : Wafer_Op<"sync.local_drain", [
    WaferResourceEffectInterface
  ]> {
  let summary = "local compute/movement visibility drain";
  let assemblyFormat = "attr-dict";
}

def Wafer_LaunchOp : Wafer_Op<"launch", [
    AttrSizedOperandSegments
  ]> {
  let summary = "host/device Wafer launch boundary";
  let description = [{
    `wafer.launch` is the host-visible boundary for a compiled Wafer package.
    It binds ranked tensor inputs and outputs to a device package reference and
    carries coarse resource summaries needed by the runtime boundary. It does
    not expose tile-local buffers, packet streams, physical addresses, or other
    lower-level execution details back into tensor IR.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$kernel,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs,
    StrAttr:$package_ref,
    I64Attr:$spm_bytes,
    I64Attr:$ddr_bytes
  );
  let results = (outs Variadic<AnyType>:$results);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $kernel
    `inputs` `(` $inputs `:` type($inputs) `)`
    `outputs` `(` $outputs `:` type($outputs) `)`
    attr-dict `:` type($results)
  }];
}

#endif // WAFER_DIALECT_WAFER_IR_WAFEROPS_TD
