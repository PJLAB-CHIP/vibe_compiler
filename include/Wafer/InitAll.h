//===- InitAll.h - Wafer dialect registration ------------------*- C++ -*-===//

#ifndef WAFER_INITALL_H
#define WAFER_INITALL_H

#include "Wafer/Dialect/Wafer/IR/WaferDialect.h"

#include "mlir/IR/DialectRegistry.h"

namespace wafer {

inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<wafer::WaferDialect>();
}

} // namespace wafer

#endif // WAFER_INITALL_H
