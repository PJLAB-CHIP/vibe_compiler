//===- Pipelines.cpp - Wafer named pipeline registration -----------------===//

#include "Wafer/Pipelines/Pipelines.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  pm.addPass(createNormalizeStablehloCollectivesPass());
  pm.addPass(createLegalizeStablehloToLinalgPass());
}

struct PlacementPipelineOptions
    : public mlir::PassPipelineOptions<PlacementPipelineOptions> {
  Option<int64_t> logicalRankCount{
      *this, "logical-rank-count",
      llvm::cl::desc("number of dense logical ranks to place; 0 infers from "
                     "wafer.shard.binding or falls back to one rank"),
      llvm::cl::init(0)};
  Option<int64_t> cardYCount{
      *this, "card-y",
      llvm::cl::desc("number of Wafer card rows in the target topology"),
      llvm::cl::init(1)};
  Option<int64_t> cardXCount{
      *this, "card-x",
      llvm::cl::desc("number of Wafer card columns in the target topology"),
      llvm::cl::init(1)};
  Option<int64_t> tileYCount{
      *this, "tile-y",
      llvm::cl::desc("number of tile rows per card in the target topology"),
      llvm::cl::init(4)};
  Option<int64_t> tileXCount{
      *this, "tile-x",
      llvm::cl::desc("number of tile columns per card in the target topology"),
      llvm::cl::init(4)};
  Option<std::string> badTileIds{
      *this, "bad-tile-ids",
      llvm::cl::desc("comma-separated flat physical tile ids excluded from "
                     "placement"),
      llvm::cl::init("")};
};

} // namespace

void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm) {
  addStablehloToLinalgBody(pm);
}

void buildFormLogicalGroupsPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createFormLogicalGroupsPass());
}

void buildLowerGroupsToTileRegionPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createConvertGroupToTileRegionPass());

  mlir::bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.allowReturnAllocsFromLoops = true;
  options.inferFunctionResultLayout = true;
  options.bufferAlignment = 64;
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

  pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
}

void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createConvertTileRegionToInstrPass());
}

void buildLowerGroupsToInstrPipeline(mlir::OpPassManager &pm) {
  buildLowerGroupsToTileRegionPipeline(pm);
  buildLowerTileRegionToInstrPipeline(pm);
}

void buildPlanSPMMemoryPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createPlanSPMMemoryPass());
}

void buildPlanDDRMemoryPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createPlanDDRMemoryPass());
}

void buildLowerGroupsToMemoryPlannedInstrPipeline(mlir::OpPassManager &pm) {
  buildLowerGroupsToInstrPipeline(pm);
  buildPlanSPMMemoryPipeline(pm);
}

void buildLowerGroupsToDDRMemoryPlannedInstrPipeline(mlir::OpPassManager &pm) {
  buildLowerGroupsToMemoryPlannedInstrPipeline(pm);
  buildPlanDDRMemoryPipeline(pm);
}

void buildLowerGroupsToSelectedInstrPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createSelectGroupTilePass());
}

static void buildLowerGroupsToPlacementPipeline(
    mlir::OpPassManager &pm, const PlacementPipelineOptions &options) {
  buildLowerGroupsToSelectedInstrPipeline(pm);
  pm.addPass(createPlanPlacementPass(
      options.logicalRankCount, options.cardYCount, options.cardXCount,
      options.tileYCount, options.tileXCount, options.badTileIds));
}

void buildLowerGroupsToPlacementPipeline(mlir::OpPassManager &pm) {
  PlacementPipelineOptions options;
  buildLowerGroupsToPlacementPipeline(pm, options);
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
    mlir::PassPipelineRegistration<>(
        "wafer-lower-groups-to-tile-region",
        "Lower logical wafer.group ops to memref-backed wafer.tile.region IR",
        [](mlir::OpPassManager &pm) {
          buildLowerGroupsToTileRegionPipeline(pm);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-tile-region-to-instr",
        "Lower executable wafer.tile.region ops to wafer.instr IR",
        [](mlir::OpPassManager &pm) {
          buildLowerTileRegionToInstrPipeline(pm);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-groups-to-instr",
        "Lower logical wafer.group ops to instruction-level Wafer IR",
        [](mlir::OpPassManager &pm) { buildLowerGroupsToInstrPipeline(pm); });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-groups-to-memory-planned-instr",
        "Lower logical wafer.group ops to memory-planned instruction-level "
        "Wafer IR",
        [](mlir::OpPassManager &pm) {
          buildLowerGroupsToMemoryPlannedInstrPipeline(pm);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-groups-to-ddr-memory-planned-instr",
        "Lower logical wafer.group ops through SPM and DDR memory planning to "
        "instruction-level Wafer IR",
        [](mlir::OpPassManager &pm) {
          buildLowerGroupsToDDRMemoryPlannedInstrPipeline(pm);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-groups-to-selected-instr",
        "Select group tile candidates and commit memory-planned "
        "instruction-level Wafer IR",
        [](mlir::OpPassManager &pm) {
          buildLowerGroupsToSelectedInstrPipeline(pm);
        });
    mlir::PassPipelineRegistration<PlacementPipelineOptions>(
        "wafer-lower-groups-to-placement",
        "Select group tile candidates and append accepted physical placement",
        [](mlir::OpPassManager &pm,
           const PlacementPipelineOptions &options) {
          buildLowerGroupsToPlacementPipeline(pm, options);
        });
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
