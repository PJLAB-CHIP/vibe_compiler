//===- PipelinesTest.cpp - Production pipeline contracts ----------------===//

#include "Wafer/Pipelines/Pipelines.h"
#include "Wafer/Compiler/TargetArtifact.h"

#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

template <typename OpT> unsigned countOps(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](OpT) { ++count; });
  return count;
}

wafer::frontend::ProgramBoundaryBinding
replicatedBoundary(int64_t index, llvm::ArrayRef<int64_t> shape,
                   llvm::StringRef dtype) {
  wafer::frontend::ProgramRankSlice slice;
  slice.logicalRank = 0;
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
  binding.rankSlices.push_back(std::move(slice));
  return binding;
}

TEST(PipelinesTest, ScheduledRankFinalizationDoesNotSelectAnotherCandidate) {
  mlir::MLIRContext context;
  mlir::PassManager manager(&context);
  wafer::buildFinalizeScheduledTensorProgramPipeline(manager);

  std::string pipeline;
  llvm::raw_string_ostream os(pipeline);
  manager.printAsTextualPipeline(os);
  os.flush();

  EXPECT_EQ(pipeline.find("wafer-schedule-tensor-program"), std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("one-shot-bufferize"), std::string::npos) << pipeline;
  EXPECT_NE(pipeline.find("wafer-plan-spm-memory"), std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("wafer-plan-ddr-memory"), std::string::npos)
      << pipeline;
}

TEST(PipelinesTest,
     StructuredProgramLowersResidentSharedProducerWinnerToTarget) {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>, policy = "explicit", shape = array<i64: 1>, topology = @default}
  func.func @convert_with_mixed_shape_consumers(%input: tensor<2x4xf16>)
      -> (tensor<2x4xf32>, tensor<2xf32>) {
    %converted_empty = tensor.empty() : tensor<2x4xf32>
    %converted = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%input : tensor<2x4xf16>)
        outs(%converted_empty : tensor<2x4xf32>) {
    ^bb0(%value: f16, %old: f32):
      %extended = arith.extf %value : f16 to f32
      linalg.yield %extended : f32
    } -> tensor<2x4xf32>
    %zero = arith.constant 0.0 : f32
    %reduced_empty = tensor.empty() : tensor<2xf32>
    %reduced_init = linalg.fill ins(%zero : f32)
        outs(%reduced_empty : tensor<2xf32>) -> tensor<2xf32>
    %reduced = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%converted : tensor<2x4xf32>)
        outs(%reduced_init : tensor<2xf32>) {
    ^bb0(%value: f32, %accumulator: f32):
      %sum = arith.addf %accumulator, %value : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>
    %squared_empty = tensor.empty() : tensor<2x4xf32>
    %squared = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%converted : tensor<2x4xf32>)
        outs(%squared_empty : tensor<2x4xf32>) {
    ^bb0(%value: f32, %old: f32):
      %square = arith.mulf %value, %value : f32
      linalg.yield %square : f32
    } -> tensor<2x4xf32>
    return %squared, %reduced : tensor<2x4xf32>, tensor<2xf32>
  }
}
)mlir",
                                              context.get());
  ASSERT_TRUE(module);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {replicatedBoundary(/*index=*/0, {2, 4}, "f16")};
  program.distributedOutputs = {replicatedBoundary(/*index=*/0, {2, 4}, "f32"),
                                replicatedBoundary(/*index=*/1, {2}, "f32")};
  llvm::Expected<wafer::compiler::ExecutionConfig> executionConfig =
      wafer::compiler::ExecutionConfig::createForSingleCard(
          /*executionRankCount=*/1,
          wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
          wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(executionConfig));

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  llvm::Expected<wafer::compiler::ExecutableBundle> executable =
      wafer::compiler::detail::buildExecutableBundle(
          context, *module, std::move(program), *executionConfig, diagnostics,
          std::nullopt);
  ASSERT_TRUE(static_cast<bool>(executable))
      << diagnosticsText
      << (executable ? "" : llvm::toString(executable.takeError()));
  EXPECT_NE(
      diagnosticsText.find("compile-stats stage=rank-candidate-generation"),
      std::string::npos);
  EXPECT_NE(diagnosticsText.find("compile-stats stage=whole-variant-selection"),
            std::string::npos);
  EXPECT_NE(diagnosticsText.find("planned_attempt_limit=153"),
            std::string::npos);
  // The executable bundle becomes the MLIRContext owner on success. Destroy
  // the source module before that owner so its uniqued state stays live.
  module = nullptr;
  ASSERT_EQ(executable->getRankExecutables().size(), 1u);
  mlir::ModuleOp scheduled =
      executable->getRankExecutables().front().getModule();

  llvm::SmallVector<wafer::TileRegionOp, 2> regions;
  scheduled.walk(
      [&](wafer::TileRegionOp region) { regions.push_back(region); });
  std::string scheduledText;
  llvm::raw_string_ostream scheduledStream(scheduledText);
  scheduled.print(scheduledStream);
  scheduledStream.flush();
  ASSERT_EQ(regions.size(), 3u) << scheduledText;
  bool hasSPMResult = false;
  bool hasSPMOperand = false;
  for (wafer::TileRegionOp region : regions) {
    for (mlir::Value result : region.getResults())
      hasSPMResult |= wafer::isWaferSPMMemRefType(result.getType());
    for (mlir::Value operand : region.getOperands())
      hasSPMOperand |= wafer::isWaferSPMMemRefType(operand.getType());
  }
  // The interface-driven complete traversal materializes the shared convert
  // once, then carries its typed SPM result to both consumers. This explicit
  // SSA handoff replaces the two independent DDR reloads.
  EXPECT_TRUE(hasSPMResult);
  EXPECT_TRUE(hasSPMOperand);
  EXPECT_EQ(countOps<wafer::InstrRDMAOp>(scheduled), 1u);
  EXPECT_EQ(countOps<wafer::InstrWDMAOp>(scheduled), 2u);
  EXPECT_EQ(countOps<wafer::SyncLocalFenceOp>(scheduled), 0u);
  // The producer and each consumer retain a terminal participant join.
  // Cross-region join sinking is a separate optimization.
  EXPECT_EQ(countOps<wafer::SyncNCCJoinOp>(scheduled), 3u);

  llvm::Expected<wafer::compiler::TargetLLVMModuleBundle> target =
      wafer::compiler::compileExecutableBundleToTargetLLVMModules(*executable,
                                                                  diagnostics);
  ASSERT_TRUE(static_cast<bool>(target))
      << diagnosticsText << (target ? "" : llvm::toString(target.takeError()));
  EXPECT_EQ(target->getModules().size(), 1u);
}

} // namespace
