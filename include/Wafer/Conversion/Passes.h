//===- Passes.h - Wafer conversion pass registration ----------*- C++ -*-===//

#ifndef WAFER_CONVERSION_PASSES_H
#define WAFER_CONVERSION_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace wafer {

std::unique_ptr<mlir::Pass> createLowerToCAbiSkeletonPass();
void registerWaferConversionPasses();

} // namespace wafer

#endif // WAFER_CONVERSION_PASSES_H
