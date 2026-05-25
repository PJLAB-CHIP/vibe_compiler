//===- WaferInterfaces.td - Wafer operation interfaces ----*- tablegen -*-===//

#ifndef WAFER_DIALECT_WAFER_IR_WAFERINTERFACES_TD
#define WAFER_DIALECT_WAFER_IR_WAFERINTERFACES_TD

include "Wafer/Dialect/Wafer/IR/WaferBase.td"

def WaferTilingInterface : OpInterface<"WaferTilingInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for operations that expose structural tiling facts derived from
    the current IR. Implementations must not return planner side tables or
    cached search state.
  }];
  let methods = [
    InterfaceMethod<
      "Verify that the operation's tiling contract is represented in IR.",
      "::mlir::LogicalResult", "verifyWaferTilingContract", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::success();">
  ];
}

def WaferLayoutOpInterface : OpInterface<"WaferLayoutOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for operations that define a verifiable relation between operand
    and result layout attributes. Layout materialization remains explicit data
    movement and is not modeled as metadata rewriting.
  }];
  let methods = [
    InterfaceMethod<
      "Verify the operation's layout relation using only the operation IR.",
      "::mlir::LogicalResult", "verifyWaferLayoutContract", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::success();">
  ];
}

def WaferResourceEffectInterface : OpInterface<"WaferResourceEffectInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for operations that expose resource/effect constraints which are
    verifiable from the operation, its operands, results, regions, and attrs.
    Planner estimates remain analysis and must not be surfaced here.
  }];
  let methods = [
    InterfaceMethod<
      "Verify resource/effect constraints represented by this operation.",
      "::mlir::LogicalResult", "verifyWaferResourceEffectContract", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::success();">
  ];
}

#endif // WAFER_DIALECT_WAFER_IR_WAFERINTERFACES_TD
