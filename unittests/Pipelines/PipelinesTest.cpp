//===- PipelinesTest.cpp - Production pipeline contracts ----------------===//

#include "Wafer/Conversion/TileToInstr/Pipelines.h"
#include "Wafer/CodeGen/TargetCodeGen.h"

#include "Wafer/CodeGen/DeviceExecutableInternal.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/ProgramData/ProgramData.h"
#include "Wafer/Transforms/Instr/MemoryPlanningPipelines.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Conversion/InstrToLLVM/InstrToLLVM.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

static std::string printPipeline(mlir::OpPassManager &manager) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  manager.printAsTextualPipeline(stream);
  stream.flush();
  return text;
}

TEST(PipelinesTest, TileLoweringBuilderUsesTheProductionNestedStructure) {
  mlir::MLIRContext context;
  mlir::PassManager production(&context);
  wafer::buildLowerTileRegionToInstrPipeline(production);
  const std::string pipeline = printPipeline(production);
  EXPECT_NE(pipeline.find("wafer-convert-tile-region-to-instr"),
            std::string::npos);
  EXPECT_NE(pipeline.find("wafer-convert-bufferization-copies-to-instr"),
            std::string::npos);
  EXPECT_NE(pipeline.find("wafer-rebuild-required-ncc-joins"),
            std::string::npos);
}

TEST(PipelinesTest, InstrFunctionBufferizationExposesLeafAndAssignsNoOffsets) {
  mlir::MLIRContext context;
  mlir::PassManager manager(&context);
  wafer::buildBufferizeInstrFunctionsPipeline(manager);

  std::string pipeline;
  llvm::raw_string_ostream os(pipeline);
  manager.printAsTextualPipeline(os);
  os.flush();

  EXPECT_NE(pipeline.find("wafer-bufferize-instr-function-boundaries"),
            std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("drop-equivalent-buffer-results"), std::string::npos)
      << pipeline;
  const size_t firstCanonicalize = pipeline.find("canonicalize");
  const size_t cse = pipeline.find("cse", firstCanonicalize);
  const size_t secondCanonicalize = pipeline.find("canonicalize", cse);
  EXPECT_NE(firstCanonicalize, std::string::npos) << pipeline;
  EXPECT_NE(cse, std::string::npos) << pipeline;
  EXPECT_NE(secondCanonicalize, std::string::npos) << pipeline;
  EXPECT_LT(firstCanonicalize, cse) << pipeline;
  EXPECT_LT(cse, secondCanonicalize) << pipeline;
  EXPECT_EQ(pipeline.find("wafer-plan-spm-memory"), std::string::npos)
      << pipeline;
  EXPECT_EQ(pipeline.find("wafer-plan-ddr-memory"), std::string::npos)
      << pipeline;

  mlir::PassManager leafManager(&context);
  leafManager.addPass(wafer::createBufferizeInstrFunctionBoundariesPass());
  std::string leafPipeline;
  llvm::raw_string_ostream leafStream(leafPipeline);
  leafManager.printAsTextualPipeline(leafStream);
  leafStream.flush();
  EXPECT_NE(leafPipeline.find("wafer-bufferize-instr-function-boundaries"),
            std::string::npos)
      << leafPipeline;
  EXPECT_EQ(leafPipeline.find("canonicalize"), std::string::npos)
      << leafPipeline;
}

TEST(PipelinesTest, MemoryAssignmentBuildersExposeAtomicPasses) {
  mlir::MLIRContext context;
  mlir::PassManager manager(&context);
  wafer::PlanSPMMemoryPassOptions spm;
  wafer::PlanDDRMemoryPassOptions ddr;
  manager.addPass(wafer::createPlanSPMMemoryPass(spm));
  manager.addPass(wafer::createPlanDDRMemoryPass(ddr));

  std::string pipeline;
  llvm::raw_string_ostream os(pipeline);
  manager.printAsTextualPipeline(os);
  os.flush();

  EXPECT_NE(pipeline.find("wafer-plan-spm-memory"), std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("wafer-plan-ddr-memory"), std::string::npos)
      << pipeline;
}

TEST(PipelinesTest, ProductionAtomicPassAddersExposeTheirPasses) {
  mlir::MLIRContext context;
  mlir::PassManager manager(&context);
  wafer::MaterializeExecutionMeshPassOptions mesh;
  mesh.shape = "1";
  manager.addPass(wafer::createMaterializeTargetTopologyPass());
  manager.addPass(wafer::createMaterializeExecutionMeshPass(mesh));
  manager.addPass(mlir::createLowerAffinePass());
  wafer::TargetConversionRequest target;
  target.profileRecordArgumentIndex = 5;
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(target));

  std::string pipeline;
  llvm::raw_string_ostream os(pipeline);
  manager.printAsTextualPipeline(os);
  os.flush();

  EXPECT_NE(pipeline.find("wafer-materialize-target-topology"),
            std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("wafer-materialize-execution-mesh"),
            std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("lower-affine"), std::string::npos) << pipeline;
  EXPECT_NE(pipeline.find("wafer-lower-instr-to-target-llvm"),
            std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("profile-record-argument-index=5"), std::string::npos)
      << pipeline;
}

TEST(PipelinesTest, NCCJoinPassesUseFunctionAnchors) {
  mlir::MLIRContext context;
  mlir::PassManager manager(&context);
  mlir::OpPassManager &functionManager = manager.nest<mlir::func::FuncOp>();
  functionManager.addPass(wafer::createPlaceRequiredNCCJoinsPass());
  functionManager.addPass(wafer::createRebuildRequiredNCCJoinsPass());

  std::string pipeline = printPipeline(manager);
  EXPECT_NE(pipeline.find("func.func(wafer-place-required-ncc-joins,"),
            std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("wafer-rebuild-required-ncc-joins"),
            std::string::npos)
      << pipeline;
}

} // namespace
