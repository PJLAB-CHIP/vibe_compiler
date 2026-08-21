//===- InitWaferDialects.h - Wafer dialect registration --------*- C++ -*-===//

#ifndef WAFER_INITWAFERDIALECTS_H
#define WAFER_INITWAFERDIALECTS_H

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/DialectRegistry.h"

namespace wafer {

/// Register the dialects that define the core Wafer IR surface. Importer and
/// target translation extensions belong to their dedicated registry profiles.
inline void registerWaferCoreDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::async::AsyncDialect, wafer::WaferDialect>();
  registerWaferTensorIndexingExternalModels(registry);
}

} // namespace wafer

#endif // WAFER_INITWAFERDIALECTS_H
