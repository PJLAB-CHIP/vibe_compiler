//===- InitImporterDialects.h - Optional importer dialect hooks -*- C++ -*-===//

#ifndef WAFER_INITIMPORTERDIALECTS_H
#define WAFER_INITIMPORTERDIALECTS_H

#include "mlir/IR/DialectRegistry.h"

namespace wafer {

inline void registerImporterDialects(mlir::DialectRegistry &registry) {
  (void)registry;
}

} // namespace wafer

#endif // WAFER_INITIMPORTERDIALECTS_H
