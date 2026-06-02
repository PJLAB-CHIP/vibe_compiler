//===- Pipelines.cpp - Wafer named pipeline registration -----------------===//

#include "Wafer/Pipelines/Pipelines.h"

#include "Wafer/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"

#ifdef WAFER_ENABLE_SHARDY
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#endif

namespace wafer {
namespace {

#ifdef WAFER_ENABLE_SHARDY
struct StablehloShardingPropagationPipelineOptions
    : public mlir::PassPipelineOptions<
          StablehloShardingPropagationPipelineOptions> {
  Option<int64_t> defaultTileCount{
      *this, "default-tile-count",
      llvm::cl::desc("logical Wafer tile mesh size for default SPMD input "
                     "sharding seeds"),
      llvm::cl::init(16)};
};
#endif

static void addStablehloToLinalgBody(mlir::OpPassManager &pm) {
  pm.addPass(createLowerStablehloReducePass());
  pm.addPass(createNormalizeConstantsPass());
  pm.addPass(createLowerStablehloDotPass());
  pm.addPass(createLowerStablehloElementwisePass());
  pm.addPass(createLowerStablehloShapePass());
}

} // namespace

void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm) {
  addStablehloToLinalgBody(pm);
}

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm,
                                               int64_t defaultTileCount) {
  pm.addPass(createApplyDefaultSpmdShardingPass(defaultTileCount));
  mlir::sdy::addPropagationPipeline(pm);
}
#endif

void registerWaferPipelines() {
  static bool registered = [] {
    mlir::PassPipelineRegistration<>(
        "wafer-lower-stablehlo-to-linalg",
        "Lower StableHLO tensor IR to structured Linalg/Tensor IR",
        [](mlir::OpPassManager &pm) { buildStablehloToLinalgPipeline(pm); });
#ifdef WAFER_ENABLE_SHARDY
    mlir::PassPipelineRegistration<StablehloShardingPropagationPipelineOptions>(
        "wafer-propagate-stablehlo-sharding",
        "Apply Wafer default StableHLO/SDY sharding seeds when needed and run "
        "Shardy propagation",
        [](mlir::OpPassManager &pm,
           const StablehloShardingPropagationPipelineOptions &options) {
          buildStablehloShardingPropagationPipeline(pm,
                                                    options.defaultTileCount);
        });
#endif
    return true;
  }();
  (void)registered;
}

} // namespace wafer
