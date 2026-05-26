//===- WaferDialect.cpp - Wafer dialect implementation -------------------===//

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <limits>
#include <optional>

using namespace wafer;

#include "Wafer/IR/WaferEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Wafer/IR/WaferAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Wafer/IR/WaferTypes.cpp.inc"

#include "Wafer/IR/WaferOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Wafer/IR/WaferOps.cpp.inc"

mlir::LogicalResult
TileBufferType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                       mlir::Type tensorType, mlir::Attribute layout,
                       mlir::Attribute memorySpace) {
  if (!mlir::isa<mlir::RankedTensorType>(tensorType))
    return emitError() << "tile_buffer logical type must be a ranked tensor";
  if (!mlir::isa<MemLayoutAttr>(layout))
    return emitError() << "tile_buffer layout must be a wafer mem_layout attr";
  if (!mlir::isa<MemorySpaceAttr>(memorySpace))
    return emitError()
           << "tile_buffer memory space must be a wafer memory_space attr";
  return mlir::success();
}

void WaferDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Wafer/IR/WaferAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Wafer/IR/WaferTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Wafer/IR/WaferOps.cpp.inc"
      >();
}
