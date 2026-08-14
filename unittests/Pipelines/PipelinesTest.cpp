//===- PipelinesTest.cpp - Production pipeline contracts ----------------===//

#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Compiler/TargetArtifact.h"

#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

template <typename OpT> unsigned countOps(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](OpT) { ++count; });
  return count;
}

static std::string printPipeline(mlir::OpPassManager &manager) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  manager.printAsTextualPipeline(stream);
  stream.flush();
  return text;
}

TEST(PipelinesTest, NamedTileLoweringExpandsToProductionBuilderStructure) {
  wafer::registerWaferTransformPasses();
  wafer::registerWaferPipelines();

  mlir::MLIRContext context;
  mlir::PassManager production(&context);
  wafer::buildLowerTileRegionToInstrPipeline(production);

  mlir::PassManager textual(&context);
  std::string errors;
  llvm::raw_string_ostream errorStream(errors);
  ASSERT_TRUE(mlir::succeeded(mlir::parsePassPipeline(
      "wafer-lower-tile-region-to-instr", textual, errorStream)))
      << errors;
  EXPECT_EQ(printPipeline(production), printPipeline(textual));
}

wafer::frontend::ProgramBoundaryBinding
replicatedBoundary(int64_t index, llvm::ArrayRef<int64_t> shape,
                   llvm::StringRef dtype) {
  wafer::frontend::ProgramPartitionSlice slice;
  slice.partitionId = 0;
  slice.replicaId = 0;
  slice.offsets.assign(shape.size(), 0);
  slice.sizes.assign(shape.begin(), shape.end());
  slice.strides.assign(shape.size(), 1);

  wafer::frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  binding.globalShape.assign(shape.begin(), shape.end());
  binding.localShape.assign(shape.begin(), shape.end());
  binding.dtype = dtype.str();
  binding.partitionSlices.push_back(std::move(slice));
  return binding;
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
  EXPECT_EQ(pipeline.find("wafer-plan-spm-memory"), std::string::npos)
      << pipeline;
  EXPECT_EQ(pipeline.find("wafer-plan-ddr-memory"), std::string::npos)
      << pipeline;

  mlir::PassManager leafManager(&context);
  wafer::addInstrFunctionBoundaryBufferizationPass(leafManager);
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

TEST(PipelinesTest, MemoryPlanningPreparationExposesModuleAndFunctionLeaves) {
  wafer::registerWaferTransformPasses();
  wafer::registerWaferPipelines();

  mlir::MLIRContext context;
  mlir::PassManager production(&context);
  wafer::buildPrepareInstrForMemoryPlanningPipeline(production);

  mlir::PassManager textual(&context);
  std::string errors;
  llvm::raw_string_ostream errorStream(errors);
  ASSERT_TRUE(mlir::succeeded(mlir::parsePassPipeline(
      "wafer-prepare-instr-for-memory-planning", textual, errorStream)))
      << errors;
  const std::string pipeline = printPipeline(production);
  EXPECT_EQ(pipeline, printPipeline(textual));
  EXPECT_NE(pipeline.find("wafer-bufferize-instr-function-boundaries"),
            std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("func.func(wafer-rebuild-required-ncc-joins)"),
            std::string::npos)
      << pipeline;
  EXPECT_EQ(pipeline.find("wafer-plan-spm-memory"), std::string::npos)
      << pipeline;
}

TEST(PipelinesTest, MemoryAssignmentBuildersExposeAtomicPasses) {
  mlir::MLIRContext context;
  mlir::PassManager manager(&context);
  wafer::PlanSPMMemoryPassOptions spm;
  wafer::PlanDDRMemoryPassOptions ddr;
  wafer::addAssignSPMOffsetsPass(manager, spm);
  wafer::addAssignDDROffsetsPass(manager, ddr);

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
  wafer::addMaterializeTargetTopologyPass(manager);
  wafer::addMaterializeExecutionMeshPass(manager, mesh);
  wafer::addLowerAffineControlAndIndexingPass(manager);
  wafer::TargetConversionRequest target;
  target.profileRecordArgumentIndex = 5;
  wafer::addLowerInstrToTargetLLVMPass(manager, target);

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
  mlir::OpPassManager &functionManager =
      manager.nest<mlir::func::FuncOp>();
  wafer::addRequiredNCCJoinPlacementPass(functionManager);
  wafer::addRecomputeRequiredNCCJoinPlacementPass(functionManager);

  std::string pipeline = printPipeline(manager);
  EXPECT_NE(pipeline.find("func.func(wafer-place-required-ncc-joins,"),
            std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("wafer-rebuild-required-ncc-joins"),
            std::string::npos)
      << pipeline;
}

TEST(PipelinesTest, StructuredProgramLowersWholePhysicalTileDomainToTarget) {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["card"], shape = array<i64: 1>}
  func.func @convert(%input: tensor<16x4xf16>) -> tensor<16x4xf32> {
    %output = tensor.empty() : tensor<16x4xf32>
    %converted = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%input : tensor<16x4xf16>)
        outs(%output : tensor<16x4xf32>) {
    ^bb0(%value: f16, %old: f32):
      %extended = arith.extf %value : f16 to f32
      linalg.yield %extended : f32
    } -> tensor<16x4xf32>
    return %converted : tensor<16x4xf32>
  }
}
)mlir",
                                              context.get());
  ASSERT_TRUE(module);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {replicatedBoundary(/*index=*/0, {16, 4}, "f16")};
  program.distributedOutputs = {
      replicatedBoundary(/*index=*/0, {16, 4}, "f32")};
  llvm::Expected<wafer::compiler::ExecutionConfig> executionConfig =
      wafer::compiler::ExecutionConfig::createForSingleCard(
          /*numPartitions=*/1, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig));

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  llvm::Expected<wafer::compiler::ExecutableBundle> executable =
      wafer::compiler::detail::buildExecutableBundle(
          context, *module, std::move(program), *executionConfig,
          wafer::OptimizationConfig::search(), diagnostics, std::nullopt);
  ASSERT_TRUE(static_cast<bool>(executable))
      << diagnosticsText
      << (executable ? "" : llvm::toString(executable.takeError()));
  EXPECT_NE(diagnosticsText.find(
                "compile-stats stage=whole-card-executable-synthesis"),
            std::string::npos);
  EXPECT_NE(diagnosticsText.find("whole-card-search policy=search"),
            std::string::npos);
  EXPECT_NE(diagnosticsText.find("physical_tile_count=16"), std::string::npos);
  // The executable bundle becomes the MLIRContext owner on success. Destroy
  // the source module before that owner so its uniqued state stays live.
  module = nullptr;
  const auto &tiles = executable->getPhysicalTileExecutables();
  ASSERT_EQ(tiles.size(), 16u);
  for (size_t index = 0; index < tiles.size(); ++index) {
    const wafer::compiler::PhysicalTileExecutable &tile = tiles[index];
    EXPECT_EQ(tile.getPhysicalCardId(), wafer::PhysicalCardId(0));
    EXPECT_EQ(tile.getPhysicalTileId(),
              wafer::PhysicalTileId(static_cast<int64_t>(index)));
    mlir::ModuleOp instrModule = tile.getModule();
    // TileRegion remains the explicit Tile-local execution container; all
    // dataflow operations inside it have crossed to Instr IR.
    EXPECT_EQ(countOps<wafer::TileRegionOp>(instrModule), 1u);
    EXPECT_EQ(countOps<wafer::InstrConvertOp>(instrModule), 1u);
    EXPECT_EQ(countOps<wafer::InstrRDMAOp>(instrModule), 1u);
    EXPECT_EQ(countOps<wafer::InstrWDMAOp>(instrModule), 1u);
    EXPECT_EQ(countOps<wafer::SyncNCCJoinOp>(instrModule), 1u);
  }

  llvm::Expected<wafer::compiler::TargetLLVMModuleBundle> target =
      wafer::compiler::compileExecutableBundleToTargetLLVMModules(*executable,
                                                                  diagnostics);
  ASSERT_TRUE(static_cast<bool>(target))
      << diagnosticsText << (target ? "" : llvm::toString(target.takeError()));
  EXPECT_EQ(target->getModules().size(), 16u);
}

} // namespace
