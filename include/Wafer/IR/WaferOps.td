//===- WaferOps.td - Wafer operation definitions ----------*- tablegen -*-===//

#ifndef WAFER_IR_WAFEROPS_TD
#define WAFER_IR_WAFEROPS_TD

include "Wafer/IR/WaferBase.td"
include "Wafer/IR/WaferAttrs.td"
include "Wafer/IR/WaferInterfaces.td"
include "Wafer/IR/WaferTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "Wafer/IR/Ops/GroupOps.td"
include "Wafer/IR/Ops/TileRegionOps.td"
include "Wafer/IR/Ops/LayoutOps.td"
include "Wafer/IR/Ops/SPMOps.td"
include "Wafer/IR/Ops/DDROps.td"
include "Wafer/IR/Ops/PlacementOps.td"
include "Wafer/IR/Ops/AbiOps.td"
include "Wafer/IR/Ops/ComputeOps.td"
include "Wafer/IR/Ops/TensorCollectiveOps.td"
include "Wafer/IR/Ops/CommOps.td"
include "Wafer/IR/Ops/SyncOps.td"
include "Wafer/IR/Ops/LaunchOps.td"

#endif // WAFER_IR_WAFEROPS_TD
