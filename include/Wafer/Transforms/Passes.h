//===- Passes.h - Wafer transform pass registration ------------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_PASSES_H
#define WAFER_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace wafer {

std::unique_ptr<mlir::Pass> createLowerStablehloDotPass();
std::unique_ptr<mlir::Pass> createNormalizeConstantsPass();
void registerWaferPasses();

} // namespace wafer

#endif // WAFER_TRANSFORMS_PASSES_H
