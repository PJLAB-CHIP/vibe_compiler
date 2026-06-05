//===- ComputeOps.td - Wafer Compute operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_COMPUTEOPS_TD
#define WAFER_IR_OPS_COMPUTEOPS_TD

include "mlir/Dialect/Async/IR/AsyncTypes.td"

def Wafer_ComputeFillOp : Wafer_Op<"compute.fill", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "target-abstract tile fill compute";
  let description = [{
    `wafer.compute.fill` initializes an existing tile-local destination storage
    with a scalar value. The operation is explicit because the initialized
    buffer participates in SPM lifetime and resource planning; it does not
    assign a physical SPM offset.
  }];

  let arguments = (ins Wafer_AnyStorage:$dest, AnyType:$value);
  let hasVerifier = 1;
  let assemblyFormat =
      "$dest `,` $value attr-dict `:` type($dest) `,` type($value)";
}

def Wafer_ComputeGemmOp : Wafer_Op<"compute.gemm", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "target-abstract tile GEMM";
  let arguments = (ins Wafer_AnyStorage:$lhs, Wafer_AnyStorage:$rhs);
  let results = (outs Wafer_AnyStorage:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def Wafer_ComputeElementwiseOp : Wafer_Op<"compute.elementwise", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
  ]> {
  let summary = "target-abstract tile elementwise compute";
  let description = [{
    `wafer.compute.elementwise` represents tile-local scalar elementwise
    compute over already materialized SPM storage values. The V0 operation keeps
    its operand/result relation explicit: without `indexing_maps`, all operands
    and the result must have the same logical tensor type; with
    `indexing_maps`, operands may use verifier-proven projected-permutation
    broadcast from result dimensions. All operands and the result must use SPM
    tensor storage values with matching element types.
  }];

  let arguments = (ins
    Wafer_ComputeElementwiseKindAttr:$kind,
    Variadic<Wafer_AnyStorage>:$inputs
  );
  let results = (outs Wafer_AnyStorage:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $inputs attr-dict `:` functional-type(operands, results)";
}

def Wafer_ComputeReduceOp : Wafer_Op<"compute.reduce", [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_ComputeResource>]>
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
    Wafer_AnyStorage:$input,
    Optional<AnyType>:$init
  );
  let results = (outs Wafer_AnyStorage:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "$kind $input (`,` $init^)? attr-dict `:` functional-type(operands, results)";
}

class Wafer_CommP2POp<string mnemonic> : Wafer_Op<mnemonic, [
    DeclareOpInterfaceMethods<WaferLayoutOpInterface>,
    DeclareOpInterfaceMethods<WaferResourceEffectInterface>,
    MemoryEffects<[MemRead<Wafer_SPMResource>,
                   MemWrite<Wafer_SPMResource>,
                   MemWrite<Wafer_CommunicationResource>]>
  ]> {
  let arguments = (ins
    Wafer_AnyStorage:$buffer,
    I64Attr:$peer,
    I64Attr:$bytes
  );
  let results = (outs Async_TokenType:$token);
  let hasVerifier = 1;
  let assemblyFormat =
      "$buffer attr-dict `:` type($buffer) `->` type($token)";
}

#endif // WAFER_IR_OPS_COMPUTEOPS_TD
