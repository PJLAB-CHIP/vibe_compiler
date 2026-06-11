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
#include "llvm/ADT/ArrayRef.h"

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
inline constexpr char kWaferSPMOffsetAttrName[] = "wafer.spm.offset";
inline constexpr char kWaferDDRRequirementAttrName[] = "wafer.ddr.requirement";
inline constexpr char kWaferDDRRangeAttrName[] = "wafer.ddr.range";
inline constexpr char kWaferDDRAccessAttrName[] = "wafer.ddr.access";
inline constexpr int64_t kWaferSPMBankLineBytes = 256;

struct WaferPhysicalTensorInfo {
  mlir::RankedTensorType logicalTensorType;
  MemorySpace memorySpace;
  MemLayout layout;
  int64_t compactBytes = -1;
  int64_t physicalBytes = -1;
  int64_t physicalElements = -1;
  int64_t elementBytes = -1;
  int64_t cBlock = 0;
  int64_t cxBlocks = 0;
  int64_t c0 = 0;
  int64_t alignedC = -1;
  int64_t tailC = 0;
  int64_t outerElements = -1;
  int64_t hwElements = -1;
  int64_t batchElements = -1;
  int64_t bankAlignElements = -1;
  bool bitPackedElement = false;
};

bool isWaferMemRefType(mlir::Type type);
bool isWaferSPMMemRefType(mlir::Type type);
bool isWaferDDRMemRefType(mlir::Type type);
MemoryAttr getWaferMemoryAttr(mlir::MemRefType type);
std::optional<WaferPhysicalTensorInfo>
computeWaferPhysicalTensorInfo(mlir::MemRefType type);
std::optional<int64_t>
computeWaferPhysicalElementByteOffset(mlir::MemRefType type,
                                      llvm::ArrayRef<int64_t> logicalIndices);

} // namespace wafer

#endif // WAFER_IR_WAFERDIALECT_H
