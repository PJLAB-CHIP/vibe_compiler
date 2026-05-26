//===- InitImporterDialects.h - Optional importer dialect hooks -*- C++ -*-===//

#ifndef WAFER_FRONTEND_INITIMPORTERDIALECTS_H
#define WAFER_FRONTEND_INITIMPORTERDIALECTS_H

#include "mlir/IR/DialectRegistry.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/Register.h"
#endif

namespace wafer {

inline void registerImporterDialects(mlir::DialectRegistry &registry) {
#ifdef WAFER_ENABLE_STABLEHLO
  mlir::stablehlo::registerAllDialects(registry);
#else
  (void)registry;
#endif
}

} // namespace wafer

#endif // WAFER_FRONTEND_INITIMPORTERDIALECTS_H
