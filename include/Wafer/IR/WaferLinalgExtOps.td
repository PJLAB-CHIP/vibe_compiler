//===- WaferLinalgExtOps.td - Wafer linalg extension ops -*- tablegen -*-===//

#ifndef WAFER_IR_WAFERLINALGEXTOPS_TD
#define WAFER_IR_WAFERLINALGEXTOPS_TD

include "Wafer/IR/WaferBase.td"
include "Wafer/IR/WaferInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "Wafer/IR/LinalgExt/CollectiveOps.td"

#endif // WAFER_IR_WAFERLINALGEXTOPS_TD
