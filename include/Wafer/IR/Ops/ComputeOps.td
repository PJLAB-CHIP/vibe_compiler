//===- ComputeOps.td - Wafer Compute operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_COMPUTEOPS_TD
#define WAFER_IR_OPS_COMPUTEOPS_TD

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
    its operand/result relation explicit: without `indexing_maps`, all operands
    and the result must have the same logical tensor type; with
    `indexing_maps`, operands may use verifier-proven projected-permutation
    broadcast from result dimensions. All operands and the result must use SPM
    tensor tile buffers with matching element types.
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

def Wafer_ComputeReduceOp : Wafer_Op<"compute.reduce", [
    WaferLayoutOpInterface,
    WaferResourceEffectInterface
  ]> {
  let summary = "target-abstract tile-local reduction compute";
  let description = [{
    `wafer.compute.reduce` represents a tile-local reduction over an SPM tensor
    tile. The result shape is the input shape with `dimensions` removed, and
    `init_value` records the scalar initialization value recovered from the
    structured `linalg.reduce` output.
  }];

  let arguments = (ins
    Wafer_ComputeReduceKindAttr:$kind,
    AnyType:$input
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $input attr-dict `:` functional-type(operands, results)";
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

#endif // WAFER_IR_OPS_COMPUTEOPS_TD
