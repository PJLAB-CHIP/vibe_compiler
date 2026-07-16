//===- TileRegionOps.td - Wafer TileRegion operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_TILEREGIONOPS_TD
#define WAFER_IR_OPS_TILEREGIONOPS_TD

def Wafer_TileRegionOp : Wafer_Op<"tile.region", [
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"TileYieldOp">
  ]> {
  let summary = "tile-local Wafer execution region";
  let description = [{
    `wafer.tile.region` is a target-abstract device-side execution scope for
    tile-local storage values, layout materialization, movement, target-abstract
    compute, communication, and sync/effect ordering. It does not store
    scheduling search state, raw packets, runtime buffer handles, or target CRT
    calls. SPM storage may cross sibling region boundaries only as an explicit
    SSA operand/result. An SPM entry block argument aliases its corresponding
    operand, and an SPM result must alias either such an entry argument or an
    SPM allocation owned by this region.
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

def Wafer_TileYieldOp : Wafer_Op<"tile.yield", [
    HasParent<"TileRegionOp">,
    Pure,
    ReturnLike,
    Terminator
  ]> {
  let summary = "terminate a wafer.tile.region body";
  let description = [{
    `wafer.tile.yield` returns values from a `wafer.tile.region` body. The
    parent verifier checks value count, value types, and storage boundary
    ownership.
  }];

  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{ attr-dict ($values^ `:` type($values))? }];
}

#endif // WAFER_IR_OPS_TILEREGIONOPS_TD
