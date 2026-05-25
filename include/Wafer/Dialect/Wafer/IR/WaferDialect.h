//===- WaferDialect.h - Wafer dialect declarations -------------*- C++ -*-===//

#ifndef WAFER_DIALECT_WAFER_IR_WAFERDIALECT_H
#define WAFER_DIALECT_WAFER_IR_WAFERDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "Wafer/Dialect/Wafer/IR/WaferEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Wafer/Dialect/Wafer/IR/WaferAttrs.h.inc"

#include "Wafer/Dialect/Wafer/IR/WaferOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Wafer/Dialect/Wafer/IR/WaferOps.h.inc"

#endif // WAFER_DIALECT_WAFER_IR_WAFERDIALECT_H
