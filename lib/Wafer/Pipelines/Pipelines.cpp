//===- Pipelines.cpp - Wafer named pipeline registration -----------------===//

#include "Wafer/Pipelines/Pipelines.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#ifdef WAFER_ENABLE_SHARDY
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#endif

namespace wafer {
namespace {

static void addStablehloToLinalgBody(mlir::OpPassManager &pm) {
  pm.addPass(createNormalizeStablehloCollectivesPass());
  pm.addPass(createLowerStaticStablehloConcatenatePass());
  pm.addPass(createLegalizeStablehloToLinalgPass());
  pm.addPass(createNormalizeStablehloCollectivesPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createNormalizeStablehloCollectivesPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

static mlir::bufferization::OneShotBufferizationOptions
getFunctionBoundaryBufferizationOptions() {
  mlir::bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.allowReturnAllocsFromLoops = true;
  options.inferFunctionResultLayout = true;
  options.bufferAlignment = 64;
  options.defaultMemorySpaceFn =
      [](mlir::TensorType tensorType) -> std::optional<mlir::Attribute> {
    return MemoryAttr::get(tensorType.getContext(), MemorySpace::DDR,
                           MemLayout::Tensor);
  };
  options.unknownTypeConverterFn =
      [](mlir::Value value, mlir::Attribute memorySpace,
         const mlir::bufferization::BufferizationOptions &options)
      -> mlir::BaseMemRefType {
    (void)options;
    return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(
        mlir::cast<mlir::TensorType>(value.getType()), memorySpace);
  };
  options.functionArgTypeConverterFn =
      [](mlir::TensorType tensorType, mlir::Attribute memorySpace,
         mlir::func::FuncOp funcOp,
         const mlir::bufferization::BufferizationOptions &options)
      -> mlir::BaseMemRefType {
    (void)memorySpace;
    (void)funcOp;
    (void)options;
    if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(tensorType)) {
      return mlir::MemRefType::get(ranked.getShape(), ranked.getElementType(),
                                   mlir::MemRefLayoutAttrInterface{},
                                   MemoryAttr::get(ranked.getContext(),
                                                   MemorySpace::DDR,
                                                   MemLayout::Tensor));
    }
    return mlir::UnrankedMemRefType::get(
        tensorType.getElementType(),
        MemoryAttr::get(tensorType.getContext(), MemorySpace::DDR,
                        MemLayout::Tensor));
  };
  return options;
}

static void addFunctionBoundaryBufferization(mlir::OpPassManager &pm) {
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(
      getFunctionBoundaryBufferizationOptions()));
}

} // namespace

void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm) {
  addStablehloToLinalgBody(pm);
}

void buildPreparePhysicalTileCandidatePipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  addFunctionBoundaryBufferization(pm);
  pm.addPass(mlir::createCanonicalizerPass());
}

void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm) {
  mlir::OpPassManager &functionPM = pm.nest<mlir::func::FuncOp>();
  functionPM.nest<TileRegionOp>().addPass(
      createConvertTileRegionToInstrPass());
  functionPM.addPass(createNormalizeNCCCompletionPass());
}

void buildPlanSPMMemoryPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createPlanSPMMemoryPass());
}

void buildPlanDDRMemoryPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createPlanDDRMemoryPass());
}

#ifdef WAFER_ENABLE_SHARDY
void buildStablehloShardingPropagationPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createApplyDefaultSpmdShardingPass());
  mlir::sdy::addPropagationPipeline(pm);
}
#endif

void registerWaferPipelines() {
  static bool registered = [] {
    mlir::PassPipelineRegistration<>(
        "wafer-lower-stablehlo-to-linalg",
        "Lower StableHLO tensor IR to structured Linalg/Tensor IR",
        [](mlir::OpPassManager &pm) { buildStablehloToLinalgPipeline(pm); });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-tile-region-to-instr",
        "Debug-only lowering of executable wafer.tile.region ops to "
        "wafer.instr IR",
        [](mlir::OpPassManager &pm) {
          buildLowerTileRegionToInstrPipeline(pm);
        });
#ifdef WAFER_ENABLE_SHARDY
    mlir::PassPipelineRegistration<>(
        "wafer-propagate-stablehlo-sharding",
        "Apply Wafer default StableHLO/SDY sharding seeds when needed and run "
        "Shardy propagation",
        [](mlir::OpPassManager &pm) {
          buildStablehloShardingPropagationPipeline(pm);
        });
#endif
    return true;
  }();
  (void)registered;
}

} // namespace wafer
