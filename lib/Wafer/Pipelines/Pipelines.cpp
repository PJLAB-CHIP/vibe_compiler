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
#include "llvm/Support/CommandLine.h"

#ifdef WAFER_ENABLE_SHARDY
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#endif

namespace wafer {
namespace {

static void addStablehloToLinalgBody(mlir::OpPassManager &pm) {
  pm.addPass(createNormalizeStablehloCollectivesPass());
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

void buildFormLogicalGroupsPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createFormLogicalGroupsPass());
}

void buildLowerGroupsToTileRegionPipeline(mlir::OpPassManager &pm,
                                          int64_t logicalRank) {
  ConvertGroupToTileRegionPassOptions options;
  options.logicalRank = logicalRank;
  pm.addPass(createConvertGroupToTileRegionPass(options));
  addFunctionBoundaryBufferization(pm);
}

void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createConvertTileRegionToInstrPass());
}

void buildLowerGroupsToInstrPipeline(mlir::OpPassManager &pm,
                                     int64_t logicalRank) {
  buildLowerGroupsToTileRegionPipeline(pm, logicalRank);
  buildLowerTileRegionToInstrPipeline(pm);
}

void buildPlanSPMMemoryPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createPlanSPMMemoryPass());
}

void buildPlanDDRMemoryPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createPlanDDRMemoryPass());
}

void buildLowerGroupsToMemoryPlannedInstrPipeline(mlir::OpPassManager &pm,
                                                  int64_t logicalRank) {
  buildLowerGroupsToInstrPipeline(pm, logicalRank);
  buildPlanSPMMemoryPipeline(pm);
}

void buildLowerGroupsToDDRMemoryPlannedInstrPipeline(mlir::OpPassManager &pm,
                                                     int64_t logicalRank) {
  buildLowerGroupsToMemoryPlannedInstrPipeline(pm, logicalRank);
  buildPlanDDRMemoryPipeline(pm);
}

void buildLowerGroupsToTargetLLVMPipeline(mlir::OpPassManager &pm,
                                          int64_t logicalRank) {
  buildLowerGroupsToSelectedInstrPipeline(pm, logicalRank);
  pm.addPass(createLowerInstrToTargetLLVMPass());
}

void buildLowerGroupsToSelectedInstrPipeline(mlir::OpPassManager &pm,
                                             int64_t logicalRank) {
  SelectGroupTilePassOptions options;
  options.logicalRank = logicalRank;
  pm.addPass(createSelectGroupTilePass(options));
  addFunctionBoundaryBufferization(pm);
  pm.addPass(mlir::createCanonicalizerPass());
  buildPlanSPMMemoryPipeline(pm);
  buildPlanDDRMemoryPipeline(pm);
  pm.addPass(mlir::createCanonicalizerPass());
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
        "wafer-lower-groups-to-tile-region",
        "Debug rank-0 direct lowering of logical wafer.group ops to "
        "memref-backed wafer.tile.region IR",
        [](mlir::OpPassManager &pm) {
          buildLowerGroupsToTileRegionPipeline(pm, /*logicalRank=*/0);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-tile-region-to-instr",
        "Debug-only lowering of executable wafer.tile.region ops to "
        "wafer.instr IR",
        [](mlir::OpPassManager &pm) {
          buildLowerTileRegionToInstrPipeline(pm);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-groups-to-instr",
        "Debug rank-0 direct lowering of logical wafer.group ops to "
        "instruction-level Wafer IR",
        [](mlir::OpPassManager &pm) {
          buildLowerGroupsToInstrPipeline(pm, /*logicalRank=*/0);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-groups-to-memory-planned-instr",
        "Debug rank-0 direct lowering of logical wafer.group ops to "
        "memory-planned instruction-level Wafer IR",
        [](mlir::OpPassManager &pm) {
          buildLowerGroupsToMemoryPlannedInstrPipeline(pm,
                                                       /*logicalRank=*/0);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-groups-to-ddr-memory-planned-instr",
        "Debug rank-0 direct lowering of logical wafer.group ops through SPM "
        "and DDR memory planning to instruction-level Wafer IR",
        [](mlir::OpPassManager &pm) {
          buildLowerGroupsToDDRMemoryPlannedInstrPipeline(pm,
                                                          /*logicalRank=*/0);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-groups-to-target-llvm",
        "Debug rank-0 replay: select complete logical wafer.group candidates, "
        "bufferize function boundaries, and lower the accepted instruction "
        "artifact to target "
        "LLVM CRT calls",
        [](mlir::OpPassManager &pm) {
          buildLowerGroupsToTargetLLVMPipeline(pm, /*logicalRank=*/0);
        });
    mlir::PassPipelineRegistration<>(
        "wafer-lower-groups-to-selected-instr",
        "Debug rank-0 replay: select complete group tile candidates and "
        "commit function-boundary bufferized, memory-planned "
        "instruction-level Wafer IR",
        [](mlir::OpPassManager &pm) {
          buildLowerGroupsToSelectedInstrPipeline(pm, /*logicalRank=*/0);
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
