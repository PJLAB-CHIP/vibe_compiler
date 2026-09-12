//===- TileRegionOps.td - Wafer TileRegion operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_TILEREGIONOPS_TD
#define WAFER_IR_OPS_TILEREGIONOPS_TD

def Wafer_TileRegionOp : Wafer_Op<"tile.region", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
      ["getEntrySuccessorOperands", "getRegionInvocationBounds"]>,
    IsolatedFromAbove,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"TileYieldOp">
  ]> {
  let summary = "tile-local Wafer execution region";
  let description = [{
    `wafer.tile.region` is a target-abstract device-side execution scope for one
    selected Tile-local region group. The same operation evolves in place from
    structural tensor IR, through layout-resolved IR, to physical local-storage
    and movement IR. It does not store scheduling search state, raw packets,
    runtime buffer handles, or target CRT calls.

    Structural shaped operands/results are ranked tensors. Physical shaped
    operands/results are explicit Wafer DDR memrefs. A physical region may
    opt into an explicit `resident` boundary when a same-Tile producer/
    consumer chain carries a current SPM allocation with a verified
    lifetime/alias owner; this is residency, not an execution-region or fusion
    decision. Stage-scoped checks enforce the structural/layout-resolved/
    physical forms; the operation verifier only checks their common local
    operand/result/region contract. Scalar, event, and control values may cross
    the boundary according to their typed contracts.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       OptionalAttr<UnitAttr>:$resident);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let assemblyFormat = [{
    `(` ($inputs^ `:` type($inputs))? `)` `->`
    custom<ParenthesizedTypeList>(type($results))
    $body attr-dict
  }];
}

def Wafer_TileYieldOp : Wafer_Op<"tile.yield", [
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
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
