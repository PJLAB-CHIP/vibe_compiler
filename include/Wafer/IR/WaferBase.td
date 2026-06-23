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
}

class Wafer_Op<string mnemonic, list<Trait> traits = []>
    : Op<Wafer_Dialect, mnemonic, traits>;

def WaferLinalgExt_Dialect : Dialect {
  let name = "wafer_linalg_ext";
  let cppNamespace = "::wafer";
  let summary = "Wafer linalg extension dialect";
  let description = [{
    The Wafer linalg extension dialect carries destination-style tensor
    operations that extend the linalg-level structured tensor layer before
    Wafer group and tile-region materialization. Target-local storage,
    endpoints, DTE tokens, packet fields, and runtime handles belong to lower
    Wafer IR layers.
  }];
}

class WaferLinalgExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<WaferLinalgExt_Dialect, mnemonic, traits>;

#endif // WAFER_IR_WAFERBASE_TD
