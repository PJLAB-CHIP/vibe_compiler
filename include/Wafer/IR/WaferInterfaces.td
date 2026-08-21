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

def WaferCoupledReductionOpInterface
    : OpInterface<"WaferCoupledReductionOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Exposes the source-owned grouping of reduction components when they must
    be initialized, combined, and finalized as one semantic state. The result
    contains only iterator identities, affine component maps, and element
    types; it never
    returns physical planning or target objects and does not build IR.
  }];
  let methods = [
    InterfaceMethod<
      "Return the verifier-closed coupled reduction description.",
      "::wafer::CoupledReductionDescription",
      "getCoupledReductionDescription", (ins)>
  ];
}

def WaferTileDataflowOpInterface
    : OpInterface<"WaferTileDataflowOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Marker interface for target-abstract Tile dataflow operations that must be
    consumed before Instr IR is legal. Structural TileRegion/TileYield ops,
    metadata and instruction-level operations intentionally do not implement
    this interface. Dialect conversion and its stage-legality check query this
    single classification so newly added Tile dataflow operations fail closed.
  }];
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

def WaferNCCCompletionOpInterface
    : OpInterface<"WaferNCCCompletionOpInterface"> {
  let cppNamespace = "::wafer";
  let description = [{
    Interface for an instruction whose typed fields determine its complete
    NCC behavior instead of the generic ordered-issue rule. Current
    implementations are participant joins and the peripheral op, whose kind
    distinguishes synchronous writeback from ordinary issue. Other ordinary
    issues continue to use WaferNCCIssueOpInterface generically.
  }];
  let methods = [
    InterfaceMethod<
      "Return this operation's complete typed NCC completion semantics.",
      "::wafer::NCCOperationCompletion", "getNCCCompletion", (ins)>
  ];
}

#endif // WAFER_IR_WAFERINTERFACES_TD
