//===- Passes.h - Wafer transform pass registration ------------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_PASSES_H
#define WAFER_TRANSFORMS_PASSES_H

#include <memory>
#include <string>

namespace mlir {
class Pass;
} // namespace mlir

namespace wafer {

struct SPMMemoryPlanningFailure;

#define GEN_PASS_DECL
#include "Wafer/Transforms/WaferPasses.h.inc"

std::unique_ptr<mlir::Pass> createLegalizeStablehloToLinalgPass();

/// Compiler-adapter form of the SPM assignment pass. It executes through the
/// PassManager/AnalysisManager path while returning typed rejection evidence
/// to the caller coordinating whole-executable verification.
std::unique_ptr<mlir::Pass>
createPlanSPMMemoryPassWithFailure(const PlanSPMMemoryPassOptions &options,
                                   SPMMemoryPlanningFailure *failure);
void registerWaferTransformPasses();

} // namespace wafer

#endif // WAFER_TRANSFORMS_PASSES_H
