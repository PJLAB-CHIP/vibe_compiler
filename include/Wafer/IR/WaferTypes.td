//===- WaferTypes.td - Wafer type definitions -------------*- tablegen -*-===//

#ifndef WAFER_IR_WAFERTYPES_TD
#define WAFER_IR_WAFERTYPES_TD

include "Wafer/IR/WaferAttrs.td"
include "Wafer/IR/WaferBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"

def Wafer_AnyWaferMemRef : Type<
  CPred<"::wafer::isWaferMemRefType($_self)">,
  "Wafer-tagged memref">;

def Wafer_AnySPMMemRef : Type<
  CPred<"::wafer::isWaferSPMMemRefType($_self)">,
  "Wafer SPM memref">;

def Wafer_AnyDDRMemRef : Type<
  CPred<"::wafer::isWaferDDRMemRefType($_self)">,
  "Wafer DDR memref">;

#endif // WAFER_IR_WAFERTYPES_TD
