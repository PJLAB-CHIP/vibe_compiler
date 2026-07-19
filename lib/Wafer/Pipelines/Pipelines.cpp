//===- Pipelines.cpp - Wafer named pipeline registration -----------------===//

#include "Wafer/Pipelines/Pipelines.h"

#include "EquivalentInputVariantInternal.h"
#include "QualificationInternal.h"

#include "Wafer/Support/OptimizationQualification.h"
#include "Wafer/Transforms/Passes.h"
#include "Wafer/Transforms/StructuredOptimization.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#ifdef WAFER_ENABLE_SHARDY
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#endif

namespace wafer {
namespace {

static bool
optimizationEnabled(const OptimizationQualificationProposal &proposal,
                    const OptimizationConfiguration &configuration,
                    MechanismKey key, std::string *diagnostic) {
  std::optional<bool> enabled =
      isOptimizationMechanismEnabled(proposal, configuration, key, diagnostic);
  return enabled.value_or(false);
}

static void
addStablehloToLinalgBody(mlir::OpPassManager &pm,
                         const OptimizationQualificationProposal &proposal,
                         const OptimizationConfiguration &configuration,
                         EquivalentInputVariantV1 inputVariant) {
  pm.addPass(createOptimizationInvocationPass(
      mechanism::StablehloCollectiveNormalization,
      OptimizationCutPoint::PostSPMDStableHLOModule,
      [] { return createNormalizeStablehloCollectivesPass(); },
      /*invocationOrdinal=*/0));
  pm.addPass(createOptimizationInvocationPass(
      mechanism::StablehloStructuredLegalization,
      OptimizationCutPoint::PostSPMDStableHLOModule,
      [] { return createLegalizeStablehloToLinalgPass(); }));
  pm.addPass(createOptimizationInvocationPass(
      mechanism::StablehloCollectiveNormalization,
      OptimizationCutPoint::PostSPMDStableHLOModule,
      [] { return createNormalizeStablehloCollectivesPass(); },
      /*invocationOrdinal=*/1));
  pm.addPass(createOptimizationInvocationPass(
      mechanism::PostLegalizationCanonicalization,
      OptimizationCutPoint::StructuredTensorModule,
      [] { return mlir::createCanonicalizerPass(); }));
  pm.addPass(createOptimizationInvocationPass(
      mechanism::StablehloCollectiveNormalization,
      OptimizationCutPoint::PostSPMDStableHLOModule,
      [] { return createNormalizeStablehloCollectivesPass(); },
      /*invocationOrdinal=*/2));
  pm.addPass(createOptimizationInvocationPass(
      mechanism::StructuredTensorCanonicalization,
      OptimizationCutPoint::StructuredTensorModule,
      [] { return mlir::createCanonicalizerPass(); }));
  if (inputVariant == EquivalentInputVariantV1::Metamorphic)
    pm.addPass(
        qualification_internal::createEquivalentInputVariantPass(inputVariant));
  pm.addPass(createOptimizationInvocationPass(
      mechanism::RequiredTensorNormalization,
      OptimizationCutPoint::StructuredTensorModule,
      [] { return createRequiredTensorNormalizationPass(); }));
  if (optimizationEnabled(proposal, configuration, mechanism::StablehloCleanup,
                          nullptr))
    pm.addPass(createOptimizationInvocationPass(
        mechanism::StablehloCleanup,
        OptimizationCutPoint::StructuredTensorModule,
        [] { return mlir::createCanonicalizerPass(); }));
  if (optimizationEnabled(proposal, configuration,
                          mechanism::StructuredTensorCleanup, nullptr))
    pm.addPass(createOptimizationInvocationPass(
        mechanism::StructuredTensorCleanup,
        OptimizationCutPoint::StructuredTensorModule,
        [] { return mlir::createCanonicalizerPass(); }));
}

static void addFunctionBoundaryBufferization(mlir::OpPassManager &pm,
                                             uint64_t invocationOrdinal) {
  pm.addPass(createOptimizationInvocationPass(
      mechanism::FunctionBoundaryBufferization,
      OptimizationCutPoint::SelectedPhysicalPayloadModule,
      [] { return createFunctionBoundaryBufferizationPass(); },
      invocationOrdinal));
}

} // namespace

void buildStablehloToLinalgPipeline(mlir::OpPassManager &pm) {
  OptimizationQualificationProposal proposal =
      getCurrentOptimizationQualificationProposal();
  addStablehloToLinalgBody(pm, proposal, getAllOnOptimizationConfiguration(),
                           EquivalentInputVariantV1::Original);
}

void buildScheduleTensorProgramToSelectedInstrPipeline(mlir::OpPassManager &pm,
                                                       int64_t logicalRank) {
  OptimizationQualificationProposal proposal =
      getCurrentOptimizationQualificationProposal();
  (void)
      qualification_internal::buildScheduleTensorProgramToSelectedInstrPipeline(
          pm, logicalRank, proposal, getAllOnOptimizationConfiguration());
}

static void addScheduleTensorProgramToSelectedInstrBody(
    mlir::OpPassManager &pm, int64_t logicalRank,
    const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration) {
  // Candidate scopes are compiler-private SSA/dataflow views over the
  // structured tensor program.  The scheduling pipeline does not materialize
  // a wrapper operation or a second tensor-program artifact.
  ScheduleTensorProgramPassOptions options;
  options.logicalRank = logicalRank;
  options.tileSearch = "min-estimated-time";
  // Production already lowers the independent rank domain in parallel.  A
  // small nested candidate width uses the remaining host cores without
  // creating the unbounded rank x candidate fanout that hardware-sized
  // workloads would otherwise invite.
  options.candidateParallelism = 4;
  bool candidateCleanup = optimizationEnabled(
      proposal, configuration, mechanism::CandidateCommitCleanup, nullptr);
  pm.addPass(createOptimizationInvocationPass(
      mechanism::TileDataflowMaterialization,
      OptimizationCutPoint::StructuredTensorModule,
      [options, candidateCleanup] {
        return createScheduleTensorProgramPassForOptimizationQualification(
            options, candidateCleanup);
      }));
  (void)qualification_internal::buildFinalizeScheduledTensorProgramPipeline(
      pm, proposal, configuration);
}

void buildFinalizeScheduledTensorProgramPipeline(mlir::OpPassManager &pm) {
  OptimizationQualificationProposal proposal =
      getCurrentOptimizationQualificationProposal();
  (void)qualification_internal::buildFinalizeScheduledTensorProgramPipeline(
      pm, proposal, getAllOnOptimizationConfiguration());
}

static void addFinalizeScheduledTensorProgramBody(
    mlir::OpPassManager &pm, const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration,
    uint64_t invocationOrdinal) {
  if (optimizationEnabled(proposal, configuration,
                          mechanism::PreBufferizationCleanup, nullptr))
    pm.addPass(createOptimizationInvocationPass(
        mechanism::PreBufferizationCleanup,
        OptimizationCutPoint::SelectedPhysicalPayloadModule,
        [] { return mlir::createCanonicalizerPass(); }, invocationOrdinal));
  addFunctionBoundaryBufferization(pm, invocationOrdinal);
  pm.addPass(createOptimizationInvocationPass(
      mechanism::SelectedPayloadNormalization,
      OptimizationCutPoint::SelectedPhysicalPayloadModule,
      [] { return createSelectedPayloadNormalizationPass(); },
      invocationOrdinal));
  if (optimizationEnabled(proposal, configuration,
                          mechanism::PostBufferizationCleanup, nullptr))
    pm.addPass(createOptimizationInvocationPass(
        mechanism::PostBufferizationCleanup,
        OptimizationCutPoint::SelectedPhysicalPayloadModule,
        [] { return mlir::createCanonicalizerPass(); }, invocationOrdinal));
  buildPlanSPMMemoryPipeline(pm);
  buildPlanDDRMemoryPipeline(pm);
  if (optimizationEnabled(proposal, configuration,
                          mechanism::PostMemoryPlanningCleanup, nullptr))
    pm.addPass(createOptimizationInvocationPass(
        mechanism::PostMemoryPlanningCleanup,
        OptimizationCutPoint::FinalInstructionModule,
        [] { return mlir::createCanonicalizerPass(); }, invocationOrdinal));
}

namespace qualification_internal {

bool buildStablehloToLinalgPipeline(
    mlir::OpPassManager &pm, const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration, std::string *diagnostic,
    EquivalentInputVariantV1 inputVariant) {
  if (!validateOptimizationConfiguration(proposal, configuration, diagnostic))
    return false;
  if (inputVariant != EquivalentInputVariantV1::Original &&
      inputVariant != EquivalentInputVariantV1::Metamorphic) {
    if (diagnostic)
      *diagnostic = "unknown equivalent input variant";
    return false;
  }
  addStablehloToLinalgBody(pm, proposal, configuration, inputVariant);
  return true;
}

bool buildScheduleTensorProgramToSelectedInstrPipeline(
    mlir::OpPassManager &pm, int64_t logicalRank,
    const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration, std::string *diagnostic) {
  if (!validateOptimizationConfiguration(proposal, configuration, diagnostic))
    return false;
  addScheduleTensorProgramToSelectedInstrBody(pm, logicalRank, proposal,
                                              configuration);
  return true;
}

bool buildFinalizeScheduledTensorProgramPipeline(
    mlir::OpPassManager &pm, const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration, std::string *diagnostic,
    uint64_t invocationOrdinal) {
  if (!validateOptimizationConfiguration(proposal, configuration, diagnostic))
    return false;
  addFinalizeScheduledTensorProgramBody(pm, proposal, configuration,
                                        invocationOrdinal);
  return true;
}

} // namespace qualification_internal

void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createOptimizationInvocationPass(
      mechanism::InstructionLowering,
      OptimizationCutPoint::SelectedPhysicalPayloadModule,
      [] { return createConvertTileRegionToInstrPass(); }));
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
