//===- WaferInterfaces.td - Wafer operation interfaces ----*- tablegen -*-===//

#ifndef WAFER_IR_WAFERINTERFACES_TD
#define WAFER_IR_WAFERINTERFACES_TD

include "Wafer/IR/WaferBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Wafer_SPMResource : Resource<"::wafer::WaferSPMResource">;
def Wafer_DDRResource : Resource<"::wafer::WaferDDRResource">;
def Wafer_MovementResource : Resource<"::wafer::WaferMovementResource">;
def Wafer_ComputeResource : Resource<"::wafer::WaferComputeResource">;
def Wafer_CommunicationResource
    : Resource<"::wafer::WaferCommunicationResource">;
def Wafer_SyncResource : Resource<"::wafer::WaferSyncResource">;

def WaferTilingInterface : OpInterface<"WaferTilingInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for operations that expose structural tiling facts derived from
    the current IR. Implementations must not return planner side tables or
    cached search state.
  }];
  let methods = [
    InterfaceMethod<
      "Collect the tensor values that participate in Wafer tiling.",
      "void", "collectWaferTilingDemand",
      (ins "::llvm::SmallVectorImpl<::wafer::WaferTilingDemand> &":$demands)>,
    InterfaceMethod<
      "Verify that the operation's tiling contract is represented in IR.",
      "::mlir::LogicalResult", "verifyWaferTilingContract", (ins)>
  ];
}

def WaferTensorCollectiveOpInterface
    : OpInterface<"WaferTensorCollectiveOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for tensor-level collective operations. The returned facts must
    be derived from operands, results, regions, and attributes already present
    in IR; planner choices and runtime schedules do not belong here.
  }];
  let methods = [
    InterfaceMethod<
      "Collect the tensor collective facts represented by this operation.",
      "void", "collectWaferTensorCollectiveInfo",
      (ins "::wafer::WaferTensorCollectiveInfo &":$info)>,
    InterfaceMethod<
      "Verify the tensor collective contract represented by this operation.",
      "::mlir::LogicalResult", "verifyWaferTensorCollectiveContract",
      (ins)>
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
      "Collect exact layout requirements represented by this operation.",
      "void", "collectWaferLayoutRequirements",
      (ins "::llvm::SmallVectorImpl<::wafer::WaferLayoutRequirement> &":$requirements)>,
    InterfaceMethod<
      "Verify the operation's layout relation using only the operation IR.",
      "::mlir::LogicalResult", "verifyWaferLayoutContract", (ins)>
  ];
}

def WaferLayoutMaterializationOpInterface
    : OpInterface<"WaferLayoutMaterializationOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for explicit layout conversion edges. These operations represent
    real movement between physical layout families and are queried separately
    from layout-sensitive compute or boundary ops.
  }];
  let methods = [
    InterfaceMethod<
      "Collect source/result layouts participating in this materialization.",
      "void", "collectWaferMaterializationLayouts",
      (ins "::llvm::SmallVectorImpl<::wafer::WaferLayoutRequirement> &":$requirements)>,
    InterfaceMethod<
      "Verify the materialization layout relation using only operation IR.",
      "::mlir::LogicalResult", "verifyWaferLayoutMaterializationContract",
      (ins)>
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
      "Collect Wafer-specific resource effects represented by this operation.",
      "void", "collectWaferResourceEffects",
      (ins "::llvm::SmallVectorImpl<::wafer::WaferResourceEffect> &":$effects)>,
    InterfaceMethod<
      "Verify resource/effect constraints represented by this operation.",
      "::mlir::LogicalResult", "verifyWaferResourceEffectContract", (ins)>
  ];
}

#endif // WAFER_IR_WAFERINTERFACES_TD
