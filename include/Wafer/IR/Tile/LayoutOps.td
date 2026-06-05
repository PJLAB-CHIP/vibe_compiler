//===- LayoutOps.td - Wafer Layout operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_LAYOUTOPS_TD
#define WAFER_IR_OPS_LAYOUTOPS_TD

def Wafer_LayoutMaterializeOp : Wafer_Op<"tile.materialize_layout", [
    DeclareOpInterfaceMethods<WaferLayoutMaterializationOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_MovementResource>]>
  ]> {
  let summary = "explicit storage layout materialization";
  let description = [{
    `wafer.tile.materialize_layout` represents real device-side data movement from
    one Wafer physical layout family to another. It preserves the logical
    tensor type and memory space; only the storage `mem_layout` changes.
    Same-layout materialization is not a meaningful IR edge; the verifier
    rejects it instead of relying on canonicalization for legality.
  }];

  let arguments = (ins Wafer_AnyStorage:$source);
  let results = (outs Wafer_AnyStorage:$result);

  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

#endif // WAFER_IR_OPS_LAYOUTOPS_TD
