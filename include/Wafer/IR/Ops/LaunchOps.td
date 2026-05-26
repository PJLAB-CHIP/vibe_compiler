//===- LaunchOps.td - Wafer Launch operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_LAUNCHOPS_TD
#define WAFER_IR_OPS_LAUNCHOPS_TD

def Wafer_LaunchOp : Wafer_Op<"launch", [
    AttrSizedOperandSegments
  ]> {
  let summary = "host/device Wafer launch boundary";
  let description = [{
    `wafer.launch` is the host-visible boundary for a compiled Wafer package.
    It binds ranked tensor inputs and outputs to a device package reference and
    carries coarse resource summaries needed by the runtime boundary. It does
    not expose tile-local buffers, packet streams, physical addresses, or other
    lower-level execution details back into tensor IR.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$kernel,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs,
    StrAttr:$package_ref,
    I64Attr:$spm_bytes,
    I64Attr:$ddr_bytes
  );
  let results = (outs Variadic<AnyType>:$results);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $kernel
    `inputs` `(` $inputs `:` type($inputs) `)`
    `outputs` `(` $outputs `:` type($outputs) `)`
    attr-dict `:` type($results)
  }];
}

#endif // WAFER_IR_OPS_LAUNCHOPS_TD
