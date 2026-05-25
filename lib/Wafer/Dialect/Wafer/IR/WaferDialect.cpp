//===- WaferDialect.cpp - Wafer dialect implementation -------------------===//

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace wafer;

#include "Wafer/Dialect/Wafer/IR/WaferEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Wafer/Dialect/Wafer/IR/WaferAttrs.cpp.inc"

#include "Wafer/Dialect/Wafer/IR/WaferOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Wafer/Dialect/Wafer/IR/WaferOps.cpp.inc"

void WaferDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Wafer/Dialect/Wafer/IR/WaferAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Wafer/Dialect/Wafer/IR/WaferOps.cpp.inc"
      >();
}
