//===- LayoutOps.td - Wafer Layout operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_LAYOUTOPS_TD
#define WAFER_IR_OPS_LAYOUTOPS_TD

def Wafer_LayoutMaterializeOp : Wafer_Op<"layout.materialize", [
    DeclareOpInterfaceMethods<WaferLayoutMaterializationOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "explicit tile-buffer layout materialization";
  let description = [{
    `wafer.layout.materialize` represents real device-side data movement from
    one Wafer physical layout family to another. It preserves the logical
    tensor type and memory space; only the tile-buffer `mem_layout` changes.
    Same-layout materialization is not a meaningful IR edge; the verifier
    rejects it instead of relying on canonicalization for legality.
  }];

  let arguments = (ins Wafer_AnyTileBuffer:$source);
  let results = (outs Wafer_AnyTileBuffer:$result);

  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

#endif // WAFER_IR_OPS_LAYOUTOPS_TD
