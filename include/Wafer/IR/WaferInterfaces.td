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

def WaferLinalgExtCollectiveOpInterface
    : OpInterface<"WaferLinalgExtCollectiveOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Minimal family query for generic traversal of Wafer linalg-extension
    collectives. Partition groups, channels, axes, and combiner regions remain
    directly represented by the concrete typed operation.
  }];
  let methods = [
    InterfaceMethod<
      "Return the collective semantic family represented by this operation.",
      "::wafer::WaferLinalgExtCollectiveKind", "getCollectiveKind", (ins)>
  ];
}

def WaferInstructionOpInterface : OpInterface<"WaferInstructionOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for instruction-level Wafer operations. Implementations expose
    only the instruction family consumed by generic instruction traversal.
    Operation invariants remain in the typed op verifier. Scheduling state,
    worker id, SPM offsets, packet fields, and ABI symbols are intentionally
    outside this interface.
  }];
  let methods = [
    InterfaceMethod<
      "Return the hardware instruction family selected by this instruction op.",
      "::wafer::InstrFamily", "getInstructionFamily", (ins)>
  ];
}

def WaferNCCIssueOpInterface : OpInterface<"WaferNCCIssueOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for ordinary NCC issues with an explicit typed worker domain.
    Completion joins and Direct DTE operations do not implement this
    interface.
  }];
  let methods = [
    InterfaceMethod<
      "Return the worker domain that receives this NCC issue.",
      "::wafer::NCCWorker", "getIssueWorker", (ins)>
  ];
}

#endif // WAFER_IR_WAFERINTERFACES_TD
