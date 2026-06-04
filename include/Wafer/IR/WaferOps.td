//===- WaferOps.td - Wafer operation definitions ----------*- tablegen -*-===//

#ifndef WAFER_IR_WAFEROPS_TD
#define WAFER_IR_WAFEROPS_TD

include "Wafer/IR/WaferBase.td"
include "Wafer/IR/WaferAttrs.td"
include "Wafer/IR/WaferInterfaces.td"
include "Wafer/IR/WaferTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "Wafer/IR/Tensor/GroupOps.td"
include "Wafer/IR/Tensor/TensorCollectiveOps.td"
include "Wafer/IR/Tile/TileRegionOps.td"
include "Wafer/IR/Tile/LayoutOps.td"
include "Wafer/IR/Tile/ComputeOps.td"
include "Wafer/IR/Tile/MoveOps.td"
include "Wafer/IR/Tile/ViewOps.td"
include "Wafer/IR/Tile/CommOps.td"
include "Wafer/IR/Resource/SPMOps.td"
include "Wafer/IR/Resource/DDROps.td"
include "Wafer/IR/Resource/PlacementOps.td"
include "Wafer/IR/Instr/SyncOps.td"
include "Wafer/IR/Runtime/LaunchOps.td"
include "Wafer/IR/Debug/AbiOps.td"

#endif // WAFER_IR_WAFEROPS_TD
