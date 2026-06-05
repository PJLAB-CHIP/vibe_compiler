//===- WaferDialect.h - Wafer dialect declarations -------------*- C++ -*-===//

#ifndef WAFER_IR_WAFERDIALECT_H
#define WAFER_IR_WAFERDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include <cstdint>
#include <optional>

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

struct WaferPhysicalTensorInfo {
  mlir::RankedTensorType logicalTensorType;
  MemorySpace memorySpace;
  MemLayout layout;
  int64_t compactBytes = -1;
  int64_t physicalBytes = -1;
  int64_t cBlock = 0;
  int64_t alignedC = -1;
  int64_t tailC = 0;
  bool bitPackedElement = false;
};

bool isWaferMemRefType(mlir::Type type);
bool isWaferSPMMemRefType(mlir::Type type);
bool isWaferDDRMemRefType(mlir::Type type);
MemoryAttr getWaferMemoryAttr(mlir::MemRefType type);
std::optional<WaferPhysicalTensorInfo>
computeWaferPhysicalTensorInfo(mlir::MemRefType type);

} // namespace wafer

#endif // WAFER_IR_WAFERDIALECT_H
