//===- TileRegionOps.td - Wafer TileRegion operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_TILEREGIONOPS_TD
#define WAFER_IR_OPS_TILEREGIONOPS_TD

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
  let hasRegionVerifier = 1;
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

#endif // WAFER_IR_OPS_TILEREGIONOPS_TD
