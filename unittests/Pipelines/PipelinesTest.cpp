//===- PipelinesTest.cpp - Production pipeline contracts ----------------===//

#include "Wafer/Pipelines/Pipelines.h"
#include "Pipelines/EquivalentInputVariantInternal.h"
#include "Pipelines/QualificationInternal.h"
#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Transforms/StructuredOptimization.h"

#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>

namespace {

static std::set<uint32_t> gatewayKeys(const mlir::PassManager &manager) {
  std::set<uint32_t> keys;
  for (const mlir::Pass &pass : manager.getPasses())
    if (auto description = wafer::describeOptimizationInvocationPass(pass))
      keys.insert(description->key.semanticId);
  return keys;
}

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

TEST(PipelinesTest, ClosedLoopRankSchedulerUsesEstimatedTimeRanking) {
  mlir::MLIRContext context;
  mlir::PassManager manager(&context);
  wafer::buildScheduleTensorProgramToSelectedInstrPipeline(manager,
                                                           /*logicalRank=*/7);

  std::string pipeline;
  for (mlir::Pass &pass : manager.getPasses()) {
    auto description = wafer::describeOptimizationInvocationPass(pass);
    if (description &&
        description->key == wafer::mechanism::TileDataflowMaterialization)
      pipeline = description->ownerTextualPipeline;
  }

  EXPECT_NE(pipeline.find("wafer-schedule-tensor-program"), std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("logical-rank=7"), std::string::npos) << pipeline;
  EXPECT_NE(pipeline.find("tile-search=min-estimated-time"), std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("candidate-parallelism=4"), std::string::npos)
      << pipeline;
  EXPECT_EQ(pipeline.find("tile-search=first-legal"), std::string::npos)
      << pipeline;
}

TEST(PipelinesTest, ScheduledRankFinalizationDoesNotSelectAnotherCandidate) {
  mlir::MLIRContext context;
  mlir::PassManager manager(&context);
  wafer::buildFinalizeScheduledTensorProgramPipeline(manager);

  std::string pipeline;
  for (mlir::Pass &pass : manager.getPasses()) {
    if (auto description = wafer::describeOptimizationInvocationPass(pass)) {
      pipeline += description->ownerTextualPipeline;
      pipeline += ',';
    } else {
      llvm::raw_string_ostream os(pipeline);
      pass.printAsTextualPipeline(os);
      os << ',';
    }
  }

  EXPECT_EQ(pipeline.find("wafer-schedule-tensor-program"), std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("one-shot-bufferize"), std::string::npos) << pipeline;
  EXPECT_NE(pipeline.find("wafer-plan-spm-memory"), std::string::npos)
      << pipeline;
  EXPECT_NE(pipeline.find("wafer-plan-ddr-memory"), std::string::npos)
      << pipeline;
}

TEST(PipelinesTest,
     EmptyProposalKeepsRequiredPipelineAndDisablesOptionalOptimizations) {
  mlir::MLIRContext context;
  wafer::OptimizationQualificationProposal proposal =
      wafer::getCurrentOptimizationQualificationProposal();
  std::string diagnostic;

  mlir::PassManager stableAllOff(&context);
  ASSERT_TRUE(wafer::qualification_internal::buildStablehloToLinalgPipeline(
      stableAllOff, proposal, wafer::getAllOffOptimizationConfiguration(),
      &diagnostic))
      << diagnostic;
  std::set<uint32_t> stableOffKeys = gatewayKeys(stableAllOff);
  EXPECT_EQ(stableOffKeys.count(
                wafer::mechanism::RequiredTensorNormalization.semanticId),
            1u);
  EXPECT_EQ(stableOffKeys.count(
                wafer::mechanism::PostLegalizationCanonicalization.semanticId),
            1u);
  EXPECT_EQ(stableOffKeys.count(
                wafer::mechanism::StructuredTensorCanonicalization.semanticId),
            1u);
  EXPECT_FALSE(
      stableOffKeys.count(wafer::mechanism::StablehloCleanup.semanticId));
  EXPECT_FALSE(stableOffKeys.count(
      wafer::mechanism::StructuredTensorCleanup.semanticId));

  ASSERT_TRUE(proposal.fixedBindings.empty());
  ASSERT_TRUE(proposal.cleanupBindings.empty());

  mlir::PassManager stableAllOn(&context);
  ASSERT_TRUE(wafer::qualification_internal::buildStablehloToLinalgPipeline(
      stableAllOn, proposal, wafer::getAllOnOptimizationConfiguration(),
      &diagnostic))
      << diagnostic;
  std::set<uint32_t> stableOnKeys = gatewayKeys(stableAllOn);
  EXPECT_EQ(stableOnKeys, stableOffKeys);
  EXPECT_FALSE(
      stableOnKeys.count(wafer::mechanism::StablehloCleanup.semanticId));
  EXPECT_FALSE(
      stableOnKeys.count(wafer::mechanism::StructuredTensorCleanup.semanticId));

  mlir::PassManager finalizeAllOff(&context);
  ASSERT_TRUE(wafer::qualification_internal::
                  buildFinalizeScheduledTensorProgramPipeline(
                      finalizeAllOff, proposal,
                      wafer::getAllOffOptimizationConfiguration(), &diagnostic))
      << diagnostic;
  std::set<uint32_t> finalizeOffKeys = gatewayKeys(finalizeAllOff);
  EXPECT_EQ(finalizeOffKeys.count(
                wafer::mechanism::SelectedPayloadNormalization.semanticId),
            1u);
  EXPECT_EQ(finalizeOffKeys.count(
                wafer::mechanism::FunctionBoundaryBufferization.semanticId),
            1u);
  EXPECT_FALSE(finalizeOffKeys.count(
      wafer::mechanism::PreBufferizationCleanup.semanticId));
  EXPECT_FALSE(finalizeOffKeys.count(
      wafer::mechanism::PostBufferizationCleanup.semanticId));
  EXPECT_FALSE(finalizeOffKeys.count(
      wafer::mechanism::PostMemoryPlanningCleanup.semanticId));

  mlir::PassManager production(&context);
  wafer::buildFinalizeScheduledTensorProgramPipeline(production);
  std::set<uint32_t> productionKeys = gatewayKeys(production);
  EXPECT_EQ(productionKeys.count(
                wafer::mechanism::SelectedPayloadNormalization.semanticId),
            1u);
  EXPECT_EQ(productionKeys.count(
                wafer::mechanism::FunctionBoundaryBufferization.semanticId),
            1u);
  EXPECT_FALSE(productionKeys.count(
      wafer::mechanism::PreBufferizationCleanup.semanticId));
  EXPECT_FALSE(productionKeys.count(
      wafer::mechanism::PostBufferizationCleanup.semanticId));
  EXPECT_FALSE(productionKeys.count(
      wafer::mechanism::PostMemoryPlanningCleanup.semanticId));
}

TEST(PipelinesTest,
     MetamorphicInputConstructsEquivalentSSAConsumedByRequiredNormalization) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @entry(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
    return %arg : tensor<2x4xf32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::PassManager constructor(&context);
  constructor.addPass(
      wafer::qualification_internal::createEquivalentInputVariantPass(
          wafer::EquivalentInputVariantV1::Metamorphic));
  ASSERT_TRUE(mlir::succeeded(constructor.run(*module)));
  EXPECT_EQ(countOps<mlir::tensor::ExtractSliceOp>(*module), 1u);

  wafer::TensorNormalizationOutcome normalized =
      wafer::normalizeRequiredTensorModule(*module);
  ASSERT_EQ(normalized.status, wafer::TensorNormalizationStatus::Success);
  EXPECT_TRUE(normalized.changed);
  EXPECT_EQ(countOps<mlir::tensor::ExtractSliceOp>(*module), 0u);
  EXPECT_EQ(wafer::verifyRequiredTensorNormalForm(*module).status,
            wafer::TensorNormalizationStatus::Success);
}

TEST(PipelinesTest,
     StructuredProgramWithoutOptionalCleanupLowersAcceptedArtifactToTarget) {
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
    %negated_empty = tensor.empty() : tensor<2x4xf32>
    %negated = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%converted : tensor<2x4xf32>)
        outs(%negated_empty : tensor<2x4xf32>) {
    ^bb0(%value: f32, %old: f32):
      %negative = arith.negf %value : f32
      linalg.yield %negative : f32
    } -> tensor<2x4xf32>
    return %negated, %reduced : tensor<2x4xf32>, tensor<2xf32>
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
          wafer::TargetProfileId::waferTx81SingleCardKernelV1());
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
  // The executable bundle becomes the MLIRContext owner on success. Destroy
  // the source module before that owner so its uniqued state stays live.
  module = nullptr;
  ASSERT_EQ(executable->getRankExecutables().size(), 1u);
  mlir::ModuleOp scheduled =
      executable->getRankExecutables().front().getModule();

  llvm::SmallVector<wafer::TileRegionOp, 2> regions;
  scheduled.walk(
      [&](wafer::TileRegionOp region) { regions.push_back(region); });
  ASSERT_EQ(regions.size(), 3u);
  bool hasSPMResult = false;
  bool hasSPMOperand = false;
  for (wafer::TileRegionOp region : regions) {
    for (mlir::Value result : region.getResults())
      hasSPMResult |= wafer::isWaferSPMMemRefType(result.getType());
    for (mlir::Value operand : region.getOperands())
      hasSPMOperand |= wafer::isWaferSPMMemRefType(operand.getType());
  }
  // CandidateCommitCleanup is not a member of the qualified proposal. The
  // required pipeline still proves and selects this resident handoff without
  // relying on optional canonicalization for legality.
  EXPECT_TRUE(hasSPMResult);
  EXPECT_TRUE(hasSPMOperand);
  EXPECT_EQ(countOps<wafer::InstrRDMAOp>(scheduled), 1u);
  EXPECT_EQ(countOps<wafer::InstrWDMAOp>(scheduled), 2u);

  llvm::Expected<wafer::compiler::TargetLLVMModuleBundle> target =
      wafer::compiler::compileExecutableBundleToTargetLLVMModules(*executable,
                                                                  diagnostics);
  ASSERT_TRUE(static_cast<bool>(target))
      << diagnosticsText << (target ? "" : llvm::toString(target.takeError()));
  EXPECT_EQ(target->getModules().size(), 1u);
}

} // namespace
