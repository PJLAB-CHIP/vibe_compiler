//===- DDROps.td - Wafer DDR operation definitions ----*- tablegen -*-===//

#ifndef WAFER_IR_OPS_DDROPS_TD
#define WAFER_IR_OPS_DDROPS_TD

def Wafer_DdrExternalBindingOp : Wafer_Op<"ddr.external_binding"> {
  let summary = "launch-visible external DDR binding demand";
  let description = [{
    `wafer.ddr.external_binding` records the compact external tensor binding
    contract that the launch/runtime layer must satisfy for user-visible input
    and output tensors. It carries byte size, alignment and visibility policy,
    but not a physical DDR address, BO handle, pool placement, or allocation
    trace.
  }];

  let arguments = (ins
    Wafer_DdrBindingKindAttr:$kind,
    AnyTensor:$value,
    I64Attr:$bytes,
    I64Attr:$alignment,
    BoolAttr:$read_only,
    BoolAttr:$host_visible
  );

  let hasVerifier = 1;
  let assemblyFormat = "$kind $value attr-dict `:` type($value)";
}

#endif // WAFER_IR_OPS_DDROPS_TD
