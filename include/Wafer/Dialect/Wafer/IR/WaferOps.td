//===- WaferOps.td - Wafer operation definitions ----------*- tablegen -*-===//

#ifndef WAFER_DIALECT_WAFER_IR_WAFEROPS_TD
#define WAFER_DIALECT_WAFER_IR_WAFEROPS_TD

include "Wafer/Dialect/Wafer/IR/WaferBase.td"
include "Wafer/Dialect/Wafer/IR/WaferAttrs.td"
include "Wafer/Dialect/Wafer/IR/WaferInterfaces.td"
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

#endif // WAFER_DIALECT_WAFER_IR_WAFEROPS_TD
