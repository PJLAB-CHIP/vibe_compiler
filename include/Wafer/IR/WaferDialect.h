//===- WaferDialect.h - Wafer dialect declarations -------------*- C++ -*-===//

#ifndef WAFER_IR_WAFERDIALECT_H
#define WAFER_IR_WAFERDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Wafer/IR/WaferEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Wafer/IR/WaferAttrs.h.inc"

#include "Wafer/IR/WaferOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Wafer/IR/WaferTypes.h.inc"

#include "Wafer/IR/WaferInterfaces.h"

#define GET_OP_CLASSES
#include "Wafer/IR/WaferOps.h.inc"

namespace wafer {

inline constexpr char kWaferCommSlotAttrName[] = "slot";

} // namespace wafer

#endif // WAFER_IR_WAFERDIALECT_H
