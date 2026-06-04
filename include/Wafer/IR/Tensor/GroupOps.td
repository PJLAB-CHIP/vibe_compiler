//===- GroupOps.td - Wafer Group operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_GROUPOPS_TD
#define WAFER_IR_OPS_GROUPOPS_TD

def Wafer_GroupOp
    : Wafer_Op<"group", [AttrSizedOperandSegments,
                         DeclareOpInterfaceMethods<WaferTilingInterface>,
                         IsolatedFromAbove, RecursiveMemoryEffects,
                         SingleBlockImplicitTerminator<"GroupYieldOp">]> {
  let summary = "tensor-level Wafer grouping and scheduling boundary";
  let description = [{
    `wafer.group` carries a tensor-level fusion/scheduling boundary. It owns
    explicit inputs, explicit destination-style outputs, and a single body
    region. Downstream planning may accept, split, or reject that boundary.
    Tile sizes, physical addresses, queues, packets, runtime buffers, and
    planner-local search state are deliberately outside this operation.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs, Variadic<AnyTensor>:$outs);
  let results = (outs Variadic<AnyTensor>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    $body attr-dict `:` type($results)
  }];
}

def Wafer_GroupYieldOp : Wafer_Op<"group_yield", [HasParent<"GroupOp">, Pure,
                                                  ReturnLike, Terminator]> {
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

#endif // WAFER_IR_OPS_GROUPOPS_TD
