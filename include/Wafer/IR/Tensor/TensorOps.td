//===- TensorOps.td - Selected tensor-stage operations -------*- tablegen -*-===//

#ifndef WAFER_IR_TENSOR_TENSOROPS_TD
#define WAFER_IR_TENSOR_TENSOROPS_TD

def Wafer_TensorCompletionOp
    : Wafer_Op<"tensor.completion", [
        Pure,
        DeclareOpInterfaceMethods<WaferTensorIndexingOpInterface>
      ]> {
  let summary = "selected tensor-stage completion with value forwarding";
  let description = [{
    Marks completion of the explicitly selected NCC worker set after one
    serialized tensor-stage work unit and forwards its live tensor values
    unchanged. The marker itself has only SSA semantics; selected TileRegion
    lowering materializes the Sync effect as `wafer.instr.ncc_join`. This keeps
    unselected scopes removable while carrying the schedule decision without
    use-count, source-order, location, or SPM-pressure inference.
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$states,
                       DenseI64ArrayAttr:$participants);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasVerifier = 1;
  let assemblyFormat =
      "$states `participants` `(` $participants `)` attr-dict `:` "
      "functional-type(operands, results)";
}

#endif // WAFER_IR_TENSOR_TENSOROPS_TD
