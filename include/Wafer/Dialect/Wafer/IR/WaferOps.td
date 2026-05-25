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

#endif // WAFER_DIALECT_WAFER_IR_WAFEROPS_TD
