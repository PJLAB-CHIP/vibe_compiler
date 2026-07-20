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

def WaferTargetImplementationOpInterface
    : OpInterface<"WaferTargetImplementationOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for bounded target implementation enumeration on source
    structured operations. Standard MLIR interfaces remain authoritative for
    source semantics, tiling, DPS roles, layout, and effects.
  }];
  let methods = [
    InterfaceMethod<
      "Collect deterministic target implementation candidates for this source operation.",
      "void", "collectTargetImplementationCandidates",
      (ins "const ::wafer::WaferTargetCapabilities &":$capabilities,
           "::llvm::SmallVectorImpl<::wafer::TargetImplementationCandidate> &":$candidates)>,
    InterfaceMethod<
      "Materialize one selected candidate into typed target-abstract IR.",
      "::mlir::LogicalResult", "materializeSelectedTargetImplementation",
      (ins "const ::wafer::TargetImplementationCandidate &":$candidate,
           "::wafer::WaferTargetImplementationMaterializer &":$materializer,
           "::mlir::OpBuilder &":$builder)>
  ];
}

def WaferLinalgExtCollectiveOpInterface
    : OpInterface<"WaferLinalgExtCollectiveOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for Wafer linalg-extension collective operations. The returned
    facts must be derived from operands, results, regions, and attributes
    already present in IR; planner choices and runtime schedules do not belong
    here.
  }];
  let methods = [
    InterfaceMethod<
      "Collect the linalg-extension collective facts represented by this operation.",
      "void", "collectWaferLinalgExtCollectiveInfo",
      (ins "::wafer::WaferLinalgExtCollectiveInfo &":$info)>,
    InterfaceMethod<
      "Verify the linalg-extension collective contract represented by this operation.",
      "::mlir::LogicalResult", "verifyWaferLinalgExtCollectiveContract",
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

def WaferInstructionOpInterface : OpInterface<"WaferInstructionOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for instruction-level Wafer operations. Implementations expose
    only facts represented by the operation itself: instruction family and
    verifier-checkable instruction contract. Scheduling state, worker id, SPM
    offsets, packet fields, and ABI symbols are intentionally outside this
    interface.
  }];
  let methods = [
    InterfaceMethod<
      "Return the hardware instruction family selected by this instruction op.",
      "::wafer::InstrFamily", "getInstructionFamily", (ins)>,
    InterfaceMethod<
      "Verify the instruction contract represented by this operation.",
      "::mlir::LogicalResult", "verifyInstructionContract", (ins)>
  ];
}

#endif // WAFER_IR_WAFERINTERFACES_TD
