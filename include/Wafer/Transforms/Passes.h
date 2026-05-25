//===- Passes.h - Wafer transform pass registration ------------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_PASSES_H
#define WAFER_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace wafer {

std::unique_ptr<mlir::Pass> createLowerStablehloDotPass();
std::unique_ptr<mlir::Pass> createLowerStablehloElementwisePass();
std::unique_ptr<mlir::Pass> createLowerStablehloReducePass();
std::unique_ptr<mlir::Pass> createLowerStablehloShapePass();
std::unique_ptr<mlir::Pass> createNormalizeConstantsPass();
std::unique_ptr<mlir::Pass> createFormGroupsPass();
std::unique_ptr<mlir::Pass> createCheckRootTileCandidatesPass();
std::unique_ptr<mlir::Pass> createCheckNormSchedulePass();
std::unique_ptr<mlir::Pass> createCheckSoftmaxSchedulePass();
std::unique_ptr<mlir::Pass> createCheckProjectionResidualSchedulePass();
std::unique_ptr<mlir::Pass> createMaterializeSingleTilePass();
std::unique_ptr<mlir::Pass> createMaterializeMultiTileNoCommPass();
std::unique_ptr<mlir::Pass> createCompactLayoutAssignmentPass();
std::unique_ptr<mlir::Pass> createCheckSPMAllocationPass();
std::unique_ptr<mlir::Pass> createMaterializeDDRExternalBindingsPass();
std::unique_ptr<mlir::Pass> createLowerToCAbiSkeletonPass();
void registerWaferPasses();

} // namespace wafer

#endif // WAFER_TRANSFORMS_PASSES_H
