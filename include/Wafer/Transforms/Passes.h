//===- Passes.h - Wafer transform pass registration ------------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_PASSES_H
#define WAFER_TRANSFORMS_PASSES_H

#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace wafer {

std::unique_ptr<mlir::Pass> createLowerStablehloDotPass();
std::unique_ptr<mlir::Pass> createLowerStablehloElementwisePass();
std::unique_ptr<mlir::Pass> createLowerStablehloReducePass();
std::unique_ptr<mlir::Pass> createLowerStablehloShapePass();
std::unique_ptr<mlir::Pass> createNormalizeStablehloCollectivesPass();
std::unique_ptr<mlir::Pass> createNormalizeConstantsPass();
#ifdef WAFER_ENABLE_SHARDY
std::unique_ptr<mlir::Pass> createApplyDefaultSpmdShardingPass();
std::unique_ptr<mlir::Pass>
createApplyDefaultSpmdShardingPass(int64_t tileCount);
#endif
void registerWaferTransformPasses();

} // namespace wafer

#endif // WAFER_TRANSFORMS_PASSES_H
