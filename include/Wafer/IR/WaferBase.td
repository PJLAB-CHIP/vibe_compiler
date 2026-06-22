//===- WaferBase.td - Wafer dialect base definitions -------*- tablegen -*-===//

#ifndef WAFER_IR_WAFERBASE_TD
#define WAFER_IR_WAFERBASE_TD

include "mlir/IR/OpBase.td"

def Wafer_Dialect : Dialect {
  let name = "wafer";
  let cppNamespace = "::wafer";
  let summary = "Wafer compiler IR dialect";
  let description = [{
    The Wafer dialect owns target-specific compiler IR objects that are stable
    enough to verify and lower. Planner-local search state, host runtime
    packaging details, and raw packet fields are intentionally outside this
    IR layer.
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 0;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
}

class Wafer_Op<string mnemonic, list<Trait> traits = []>
    : Op<Wafer_Dialect, mnemonic, traits>;

#endif // WAFER_IR_WAFERBASE_TD
