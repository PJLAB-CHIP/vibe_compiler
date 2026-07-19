//===- OptimizationQualificationDriver.cpp - Adoption gateway tests ------===//

#include "../../lib/Wafer/Transforms/StructuredOptimization/StructuredOptimizationInternal.h"
#include "Wafer/Analysis/SchedulableCallClosure.h"
#include "Wafer/Support/CanonicalIRSnapshot.h"
#include "Wafer/Support/OptimizationAdoption.h"
#include "Wafer/Support/OptimizationInvocation.h"
#include "Wafer/Support/OptimizationMechanism.h"
#include "Wafer/Support/OptimizationQualification.h"
#include "Wafer/Support/OptimizationQualificationArchive.h"
#include "Wafer/Support/OptimizationQualificationArchiveStore.h"
#include "Wafer/Support/OptimizationQualificationEvidence.h"
#include "Wafer/Transforms/StructuredOptimization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

class CollectingRecorder final : public wafer::OptimizationInvocationRecorder {
public:
  bool beginInvocation(const wafer::InvocationPreparationV1 &preparation,
                       std::string *diagnostic = nullptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    return journal_.beginInvocation(preparation, diagnostic);
  }

  bool recordInvocationTerminal(const wafer::InvocationTelemetryV1 &telemetry,
                                std::string *diagnostic = nullptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    return journal_.recordInvocationTerminal(telemetry, diagnostic);
  }

  std::vector<wafer::InvocationTelemetryV1> records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return journal_.committedTerminals();
  }

private:
  mutable std::mutex mutex_;
  wafer::OptimizationInvocationJournalV1 journal_;
};

class SetAuditTestAttributePass final
    : public mlir::PassWrapper<SetAuditTestAttributePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetAuditTestAttributePass)

  void runOnOperation() override {
    getOperation()->setAttr("test.audit", mlir::UnitAttr::get(&getContext()));
  }
};

class NoChangePass final
    : public mlir::PassWrapper<NoChangePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NoChangePass)
  void runOnOperation() override {}
};

wafer::AdoptionDigest filledDigest(uint8_t value) {
  wafer::AdoptionDigest digest;
  digest.fill(value);
  return digest;
}

wafer::ExactStaticVectorEvidenceV1 makeStaticVector(uint64_t firstMetric) {
  wafer::RegistryRefV1 registry = wafer::getCurrentStaticMetricRegistryRefV1();
  wafer::ExactStaticVectorEvidenceV1 vector;
  vector.registrySchema = registry.registrySchema;
  vector.registryDigest = registry.registryDigest;
  vector.orderedMetrics = {{1, firstMetric}, {2, 100}, {3, 100}, {4, 100}};
  return vector;
}

wafer::OptimizationConfigurationV1 evidenceAllOnConfiguration() {
  return {wafer::OptimizationGroupSelectionV1::AllOn, std::nullopt,
          wafer::OptimizationGroupSelectionV1::AllOn, std::nullopt};
}

wafer::OptimizationConfigurationV1 evidenceAllOffConfiguration() {
  return {wafer::OptimizationGroupSelectionV1::AllOff, std::nullopt,
          wafer::OptimizationGroupSelectionV1::AllOff, std::nullopt};
}

wafer::StaticComparisonEvidenceV1 makeStaticComparison(
    const wafer::QualificationCaseKeyV1 &caseKey,
    wafer::OptimizationComparisonKindV1 kind,
    std::optional<wafer::MechanismKey> disabledMechanism = std::nullopt) {
  wafer::StaticComparisonEvidenceV1 comparison;
  comparison.proposalDigest = wafer::digestOptimizationQualificationProposalV1(
      wafer::getCurrentOptimizationQualificationProposal());
  comparison.comparisonKey.caseKey = caseKey;
  comparison.comparisonKey.comparisonKind = kind;
  comparison.comparisonKey.disabledMechanismKey = disabledMechanism;
  comparison.comparisonKey.configurationB = evidenceAllOnConfiguration();
  if (kind == wafer::OptimizationComparisonKindV1::GlobalAllOffVsAllOn) {
    comparison.comparisonKey.configurationA = evidenceAllOffConfiguration();
  } else if (kind ==
             wafer::OptimizationComparisonKindV1::FixedDisableOneVsAllOn) {
    comparison.comparisonKey.configurationA = {
        wafer::OptimizationGroupSelectionV1::DisableOne, disabledMechanism,
        wafer::OptimizationGroupSelectionV1::AllOn, std::nullopt};
  } else {
    comparison.comparisonKey.configurationA = {
        wafer::OptimizationGroupSelectionV1::AllOn, std::nullopt,
        wafer::OptimizationGroupSelectionV1::DisableOne, disabledMechanism};
  }
  comparison.vectorA = makeStaticVector(100);
  comparison.vectorB = makeStaticVector(99);
  comparison.comparisonResult =
      wafer::StaticComparisonResultV1::BStrictlyBetter;
  return comparison;
}

void appendPassingABBA(const wafer::StaticComparisonEvidenceV1 &comparison,
                       std::vector<wafer::ABBASampleV1> &samples) {
  for (uint32_t block = 0; block < 5; ++block)
    for (uint32_t position = 0; position < 4; ++position)
      samples.push_back({comparison.proposalDigest, comparison.comparisonKey,
                         block,
                         static_cast<wafer::ABBASamplePosition>(position), 100,
                         100, wafer::ProcessStatusV1::Success});
}

wafer::GateEvidenceV1 makePassingGate(
    wafer::RegistryRefV1 gate, const wafer::QualificationCaseKeyV1 &caseKey,
    wafer::OptimizationConfigurationV1 configuration, uint8_t digestByte) {
  wafer::GateEvidenceV1 evidence;
  evidence.gate = gate;
  evidence.caseKey = caseKey;
  evidence.configuration = std::move(configuration);
  evidence.status = wafer::GateStatusV1::Passed;
  evidence.evidenceDigest = filledDigest(digestByte);
  return evidence;
}

wafer::OptimizationBatchObservationV1 makeQualifiedBatchEvidence() {
  wafer::OptimizationBatchObservationV1 batch;
  batch.proposalDigest = wafer::digestOptimizationQualificationProposalV1(
      wafer::getCurrentOptimizationQualificationProposal());
  batch.qualificationIdentity = {filledDigest(1), filledDigest(2),
                                 filledDigest(3), filledDigest(4),
                                 filledDigest(5)};
  batch.qualificationPolicy =
      wafer::getCurrentOptimizationSetQualificationPolicyRefV1();
  batch.batchStatus = wafer::OptimizationBatchStatusV1::Qualified;
  batch.qualificationRunDigest = filledDigest(6);
  batch.optimizationPublicationAttemptDigest = filledDigest(7);

  uint8_t gateDigest = 20;
  for (const wafer::QualificationCaseKeyV1 &original :
       wafer::getCurrentMandatoryQualificationCasesV1()) {
    for (wafer::EquivalentInputVariantV1 variant :
         {wafer::EquivalentInputVariantV1::Original,
          wafer::EquivalentInputVariantV1::Metamorphic})
      for (bool cleanupOn : {false, true})
        batch.equivalentIR2x2GateResults.orderedGateResults.push_back(
            makePassingGate(
                wafer::getCurrentEquivalentIRGateRefV1(),
                {original.corpus, original.rankCount, variant},
                {wafer::OptimizationGroupSelectionV1::AllOff, std::nullopt,
                 cleanupOn ? wafer::OptimizationGroupSelectionV1::AllOn
                           : wafer::OptimizationGroupSelectionV1::AllOff,
                 std::nullopt},
                gateDigest++));
    batch.productionAllOnGateResults.orderedGateResults.push_back(
        makePassingGate(wafer::getCurrentProductionAllOnGateRefV1(), original,
                        evidenceAllOnConfiguration(), gateDigest++));
  }
  auto caseLess = [](const auto &lhs, const auto &rhs) {
    const auto &a = lhs.comparisonKey.caseKey;
    const auto &b = rhs.comparisonKey.caseKey;
    return std::tie(a.corpus.id, a.rankCount, a.inputVariant) <
           std::tie(b.corpus.id, b.rankCount, b.inputVariant);
  };
  std::sort(batch.globalStaticComparisons.begin(),
            batch.globalStaticComparisons.end(), caseLess);
  for (const wafer::StaticComparisonEvidenceV1 &comparison :
       batch.globalStaticComparisons)
    appendPassingABBA(comparison, batch.globalABBASamples);
  auto gateLess = [](const wafer::GateEvidenceV1 &lhs,
                     const wafer::GateEvidenceV1 &rhs) {
    return std::tie(lhs.gate.id, lhs.caseKey.corpus.id, lhs.caseKey.rankCount,
                    lhs.caseKey.inputVariant, lhs.configuration.fixedSelection,
                    lhs.configuration.cleanupSelection) <
           std::tie(rhs.gate.id, rhs.caseKey.corpus.id, rhs.caseKey.rankCount,
                    rhs.caseKey.inputVariant, rhs.configuration.fixedSelection,
                    rhs.configuration.cleanupSelection);
  };
  std::sort(batch.equivalentIR2x2GateResults.orderedGateResults.begin(),
            batch.equivalentIR2x2GateResults.orderedGateResults.end(),
            gateLess);
  std::sort(batch.productionAllOnGateResults.orderedGateResults.begin(),
            batch.productionAllOnGateResults.orderedGateResults.end(),
            gateLess);
  return batch;
}

wafer::OptimizationInvocationScopeContextV1 debugRecordingScope() {
  wafer::OptimizationInvocationScopeContextV1 scope;
  scope.scopeKind = wafer::InvocationScopeKindV1::DebugReplay;
  scope.scopeDigest = filledDigest(42);
  return scope;
}

wafer::InvocationTelemetryV1 makeRewriteTerminal(uint64_t ordinal = 0) {
  wafer::InvocationTelemetryV1 terminal;
  terminal.identity.scopeKind = wafer::InvocationScopeKindV1::QualificationRun;
  terminal.identity.scopeDigest = filledDigest(1);
  terminal.identity.mechanismKey =
      wafer::mechanism::RequiredTensorNormalization;
  terminal.identity.invocationSite = {1, 1, filledDigest(2)};
  terminal.identity.cutPoint =
      wafer::OptimizationCutPoint::StructuredTensorModule;
  terminal.identity.invocationOrdinal = ordinal;
  terminal.qualificationCase = wafer::QualificationCaseKeyV1{
      {1, 1, filledDigest(3)}, 16, wafer::EquivalentInputVariantV1::Original};
  terminal.specDigest = wafer::digestAdoptionSpecV1(
      *wafer::lookupAdoptionSpec(terminal.identity.mechanismKey));
  terminal.inputSnapshotDigest = filledDigest(4);
  terminal.outcome = wafer::InvocationOutcome::Applied;
  terminal.rewriteCount = 2;
  terminal.workSummary.workPolicyDigest = filledDigest(5);
  terminal.workSummary.orderedCounters = {{1, 7}, {2, 2}};
  return terminal;
}

TEST(OptimizationQualificationDriverTest,
     SemanticMechanismRegistryIsAppendOnlyAndSorted) {
  std::vector<wafer::MechanismDescriptor> descriptors =
      wafer::getAllMechanismDescriptors();
  ASSERT_FALSE(descriptors.empty());
  uint32_t previous = 0;
  for (const wafer::MechanismDescriptor &descriptor : descriptors) {
    EXPECT_GT(descriptor.key.semanticId, previous);
    EXPECT_NE(descriptor.displayLabel, nullptr);
    EXPECT_NE(*descriptor.displayLabel, '\0');
    EXPECT_EQ(wafer::lookupMechanismDescriptor(descriptor.key)->key,
              descriptor.key);
    previous = descriptor.key.semanticId;
  }
  EXPECT_FALSE(wafer::lookupMechanismDescriptor(wafer::MechanismKey{0}));
  EXPECT_FALSE(
      wafer::lookupMechanismDescriptor(wafer::MechanismKey{previous + 1}));
}

TEST(OptimizationQualificationDriverTest,
     AdoptionSpecsAreCanonicalAllAndOnlyInventoryRows) {
  std::vector<wafer::AdoptionSpec> specs = wafer::getAllAdoptionSpecs();
  EXPECT_EQ(specs.size(), wafer::getAllMechanismDescriptors().size());
  EXPECT_TRUE(wafer::auditOptimizationAdoptionInventory().empty());
  uint32_t previous = 0;
  for (const wafer::AdoptionSpec &spec : specs) {
    EXPECT_GT(spec.mechanismKey.semanticId, previous);
    std::vector<uint8_t> bytes = wafer::encodeAdoptionSpecV1(spec);
    ASSERT_FALSE(bytes.empty());
    std::string diagnostic;
    EXPECT_TRUE(wafer::validateCanonicalAdoptionSpecV1(bytes, &diagnostic))
        << diagnostic;
    EXPECT_EQ(wafer::toHex(wafer::digestAdoptionSpecV1(spec)).size(), 64u);

    std::vector<uint8_t> corrupted = bytes;
    corrupted.back() ^= 1;
    EXPECT_FALSE(
        wafer::validateCanonicalAdoptionSpecV1(corrupted, &diagnostic));
    previous = spec.mechanismKey.semanticId;
  }

  std::vector<uint8_t> key = wafer::encodeMechanismKeyV1(
      wafer::mechanism::RequiredTensorNormalization);
  const std::string prefix("wafer.mechanism-key\0", 20);
  ASSERT_GE(key.size(), prefix.size());
  EXPECT_TRUE(std::equal(prefix.begin(), prefix.end(), key.begin()));
  EXPECT_EQ(key[key.size() - 4], 0u);
  EXPECT_EQ(key[key.size() - 1], 5u);
}

TEST(OptimizationQualificationDriverTest,
     InvocationTerminalHasCanonicalIdentityAndReadback) {
  wafer::InvocationTelemetryV1 terminal = makeRewriteTerminal();
  std::string diagnostic;
  ASSERT_TRUE(wafer::validateInvocationTelemetryV1(terminal, &diagnostic))
      << diagnostic;
  std::vector<uint8_t> bytes = wafer::encodeInvocationTelemetryV1(terminal);
  ASSERT_FALSE(bytes.empty());
  wafer::InvocationTelemetryV1 decoded;
  ASSERT_TRUE(
      wafer::decodeCanonicalInvocationTelemetryV1(bytes, decoded, &diagnostic))
      << diagnostic;
  EXPECT_EQ(wafer::encodeInvocationTelemetryV1(decoded), bytes);
  EXPECT_EQ(wafer::digestInvocationIdentityV1(decoded.identity),
            wafer::digestInvocationIdentityV1(terminal.identity));
  EXPECT_EQ(wafer::digestInvocationTelemetryV1(decoded),
            wafer::digestInvocationTelemetryV1(terminal));

  std::vector<uint8_t> corrupted = bytes;
  corrupted.back() ^= 1;
  EXPECT_FALSE(wafer::decodeCanonicalInvocationTelemetryV1(corrupted, decoded,
                                                           &diagnostic));
}

TEST(OptimizationQualificationDriverTest,
     InvocationJournalRejectsNoBeginDuplicateAndPostSealCommit) {
  wafer::InvocationTelemetryV1 first = makeRewriteTerminal();
  wafer::InvocationTelemetryV1 second = makeRewriteTerminal(1);
  wafer::InvocationPreparationV1 firstPreparation{
      first.identity, first.qualificationCase, first.specDigest,
      first.inputSnapshotDigest};
  wafer::InvocationPreparationV1 secondPreparation{
      second.identity, second.qualificationCase, second.specDigest,
      second.inputSnapshotDigest};
  wafer::OptimizationInvocationJournalV1 journal;
  std::string diagnostic;

  EXPECT_FALSE(journal.commitInvocationTerminal(first, &diagnostic));
  ASSERT_TRUE(journal.beginInvocation(firstPreparation, &diagnostic))
      << diagnostic;
  EXPECT_FALSE(journal.beginInvocation(firstPreparation, &diagnostic));
  ASSERT_TRUE(journal.beginInvocation(secondPreparation, &diagnostic))
      << diagnostic;
  ASSERT_TRUE(journal.commitInvocationTerminal(first, &diagnostic))
      << diagnostic;
  EXPECT_FALSE(journal.commitInvocationTerminal(first, &diagnostic));
  EXPECT_FALSE(journal.sealScope(first.identity.scopeKind,
                                 first.identity.scopeDigest, &diagnostic));
  ASSERT_TRUE(journal.commitInvocationTerminal(second, &diagnostic))
      << diagnostic;
  ASSERT_TRUE(journal.sealScope(first.identity.scopeKind,
                                first.identity.scopeDigest, &diagnostic))
      << diagnostic;
  EXPECT_FALSE(journal.commitInvocationTerminal(second, &diagnostic));
  wafer::InvocationTelemetryV1 third = makeRewriteTerminal(2);
  wafer::InvocationPreparationV1 thirdPreparation{
      third.identity, third.qualificationCase, third.specDigest,
      third.inputSnapshotDigest};
  EXPECT_FALSE(journal.beginInvocation(thirdPreparation, &diagnostic));
  EXPECT_EQ(journal.committedTerminals().size(), 2u);
}

TEST(OptimizationQualificationDriverTest,
     GatewayRejectsWrongCutAndEvidenceShapeBeforeEmission) {
  std::string diagnostic;
  wafer::OptimizationInvocationTokenV1 wrongCutToken;
  EXPECT_FALSE(wafer::beginOptimizationInvocationV1(
      wafer::mechanism::StructuredTensorCleanup,
      wafer::OptimizationCutPoint::TargetLLVMModule,
      /*invocationOrdinal=*/0, filledDigest(8), wrongCutToken, &diagnostic));
  EXPECT_NE(diagnostic.find("wrong optimization cut point"), std::string::npos);

  wafer::OptimizationInvocationTokenV1 token;
  ASSERT_TRUE(wafer::beginOptimizationInvocationV1(
      wafer::mechanism::StructuredTensorCleanup,
      wafer::OptimizationCutPoint::StructuredTensorModule,
      /*invocationOrdinal=*/0, filledDigest(8), token, &diagnostic));
  wafer::OptimizationInvocationTelemetry wrongCut;
  wrongCut.key = wafer::mechanism::StructuredTensorCleanup;
  wrongCut.cutPoint = wafer::OptimizationCutPoint::TargetLLVMModule;
  wrongCut.outcome = wafer::InvocationOutcome::NoChange;
  wrongCut.inputSnapshotDigest = filledDigest(8);
  EXPECT_FALSE(
      wafer::commitOptimizationInvocationV1(token, wrongCut, &diagnostic));
  EXPECT_NE(diagnostic.find("does not match begin"), std::string::npos);

  wafer::OptimizationInvocationTelemetry wrongEvidence;
  wrongEvidence.key = wafer::mechanism::StructuredTensorCleanup;
  wrongEvidence.cutPoint = wafer::OptimizationCutPoint::StructuredTensorModule;
  wrongEvidence.outcome = wafer::InvocationOutcome::Applied;
  wrongEvidence.rewriteCount = 1;
  wrongEvidence.successfulBackendActionCount = 1;
  wrongEvidence.inputSnapshotDigest = filledDigest(8);
  diagnostic.clear();
  EXPECT_FALSE(
      wafer::commitOptimizationInvocationV1(token, wrongEvidence, &diagnostic));
  EXPECT_NE(diagnostic.find("backend-action evidence"), std::string::npos);

  wrongEvidence.outcome = wafer::InvocationOutcome::Invalid;
  wrongEvidence.rewriteCount = 0;
  wrongEvidence.successfulBackendActionCount = 0;
  ASSERT_TRUE(
      wafer::commitOptimizationInvocationV1(token, wrongEvidence, &diagnostic))
      << diagnostic;
}

TEST(OptimizationQualificationDriverTest,
     OnlyOneProcessWideInvocationRecorderMayBeInstalled) {
  auto firstRecorder = std::make_shared<CollectingRecorder>();
  auto secondRecorder = std::make_shared<CollectingRecorder>();
  wafer::ScopedOptimizationInvocationRecorder first(firstRecorder,
                                                    debugRecordingScope());
  wafer::ScopedOptimizationInvocationRecorder second(secondRecorder,
                                                     debugRecordingScope());
  EXPECT_TRUE(first.installed());
  EXPECT_FALSE(second.installed());
}

TEST(OptimizationQualificationDriverTest,
     GatewayComposesCheckedExecutionAndOwnerLocalOrdinals) {
  auto recorder = std::make_shared<CollectingRecorder>();
  wafer::OptimizationInvocationScopeContextV1 scope = debugRecordingScope();
  scope.invocationOrdinalBase =
      3 * wafer::kOptimizationInvocationLocalOrdinalLimit;
  wafer::ScopedOptimizationInvocationRecorder scoped(recorder, scope);
  ASSERT_TRUE(scoped.installed());

  wafer::OptimizationInvocationTokenV1 token;
  std::string diagnostic;
  ASSERT_TRUE(wafer::beginOptimizationInvocationV1(
      wafer::mechanism::StructuredTensorCleanup,
      wafer::OptimizationCutPoint::StructuredTensorModule,
      /*invocationOrdinal=*/7, filledDigest(8), token, &diagnostic));
  wafer::OptimizationInvocationTelemetry telemetry;
  telemetry.key = wafer::mechanism::StructuredTensorCleanup;
  telemetry.cutPoint = wafer::OptimizationCutPoint::StructuredTensorModule;
  telemetry.outcome = wafer::InvocationOutcome::NoChange;
  telemetry.invocationOrdinal = 7;
  telemetry.inputSnapshotDigest = filledDigest(8);
  ASSERT_TRUE(
      wafer::commitOptimizationInvocationV1(token, telemetry, &diagnostic));
  auto records = recorder->records();
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(records.front().identity.invocationOrdinal,
            scope.invocationOrdinalBase + 7);

  wafer::OptimizationInvocationTokenV1 outOfRange;
  EXPECT_FALSE(wafer::beginOptimizationInvocationV1(
      wafer::mechanism::StructuredTensorCleanup,
      wafer::OptimizationCutPoint::StructuredTensorModule,
      wafer::kOptimizationInvocationLocalOrdinalLimit, filledDigest(8),
      outOfRange, &diagnostic));
  EXPECT_NE(diagnostic.find("owner-local"), std::string::npos);
}

TEST(OptimizationQualificationDriverTest,
     PassGatewayEmitsAppliedAndNoChangeTerminals) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto recorder = std::make_shared<CollectingRecorder>();
  wafer::ScopedOptimizationInvocationRecorder scoped(recorder,
                                                     debugRecordingScope());
  ASSERT_TRUE(scoped.installed());

  mlir::PassManager manager(&context);
  manager.addPass(wafer::createOptimizationInvocationPass(
      wafer::mechanism::StructuredTensorCleanup,
      wafer::OptimizationCutPoint::StructuredTensorModule,
      [] { return std::make_unique<SetAuditTestAttributePass>(); }));
  manager.addPass(wafer::createOptimizationInvocationPass(
      wafer::mechanism::StructuredTensorCleanup,
      wafer::OptimizationCutPoint::StructuredTensorModule,
      [] { return std::make_unique<NoChangePass>(); }, 1));
  ASSERT_TRUE(mlir::succeeded(manager.run(*module)));

  std::vector<wafer::InvocationTelemetryV1> records = recorder->records();
  ASSERT_EQ(records.size(), 2u);
  std::sort(
      records.begin(), records.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.identity.invocationOrdinal < rhs.identity.invocationOrdinal;
      });
  EXPECT_EQ(records[0].outcome, wafer::InvocationOutcome::Applied);
  EXPECT_EQ(records[0].rewriteCount, 1u);
  EXPECT_EQ(records[1].outcome, wafer::InvocationOutcome::NoChange);
  EXPECT_EQ(records[1].rewriteCount, 0u);
}

TEST(OptimizationQualificationDriverTest,
     DebugRegisteredRawPassUsesInstrumentationGateway) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @cse() -> (i32, i32) {
    %a = arith.constant 7 : i32
    %b = arith.constant 7 : i32
    return %a, %b : i32, i32
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);
  auto recorder = std::make_shared<CollectingRecorder>();
  wafer::ScopedOptimizationInvocationRecorder scoped(recorder,
                                                     debugRecordingScope());
  ASSERT_TRUE(scoped.installed());

  mlir::PassManager manager(&context);
  manager.addInstrumentation(
      wafer::createDebugOptimizationInvocationInstrumentation());
  manager.addPass(mlir::createCSEPass());
  ASSERT_TRUE(mlir::succeeded(manager.run(*module)));
  auto records = recorder->records();
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(records.front().identity.mechanismKey,
            wafer::mechanism::ScalarCommonSubexpressionElimination);
  EXPECT_EQ(records.front().outcome, wafer::InvocationOutcome::Applied);
  EXPECT_EQ(records.front().rewriteCount, 1u);
}

TEST(OptimizationQualificationDriverTest,
     RequiredTensorNormalizationIsTransactionalAndIdempotent) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @normalize(%arg: tensor<4xf32>) -> tensor<4xf32> {
    %scalar = arith.constant 1.0 : f32
    %wrapped = tensor.from_elements %scalar : tensor<f32>
    %unwrapped = tensor.extract %wrapped[] : tensor<f32>
    %slice = tensor.extract_slice %arg[0] [4] [1]
      : tensor<4xf32> to tensor<4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loop = scf.for %i = %c0 to %c1 step %c1 iter_args(%value = %slice)
        -> tensor<4xf32> {
      scf.yield %value : tensor<4xf32>
    }
    return %loop : tensor<4xf32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);

  wafer::TensorNormalizationOutcome first =
      wafer::normalizeRequiredTensorModule(*module);
  ASSERT_EQ(first.status, wafer::TensorNormalizationStatus::Success);
  EXPECT_TRUE(first.changed);
  EXPECT_EQ(first.work.committedRewrites, 3u);
  wafer::TensorNormalizationOutcome second =
      wafer::normalizeRequiredTensorModule(*module);
  ASSERT_EQ(second.status, wafer::TensorNormalizationStatus::Success);
  EXPECT_FALSE(second.changed);
  EXPECT_EQ(second.work.committedRewrites, 0u);
  EXPECT_EQ(wafer::verifyRequiredTensorNormalForm(*module).status,
            wafer::TensorNormalizationStatus::Success);
}

TEST(OptimizationQualificationDriverTest,
     RequiredTensorNormalizationOwnsStaticShapeScaffolding) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @normalize_shape(%arg: tensor<2x3xf32>,
                             %init: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %c0 = arith.constant 0 : index
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%arg : tensor<2x3xf32>)
        outs(%init : tensor<2x3xf32>) {
      ^bb0(%value: f32, %old: f32):
        %index = linalg.index 0 : index
        %duplicate = linalg.index 0 : index
        %dim = tensor.dim %arg, %c0 : tensor<2x3xf32>
        %sum = arith.addi %c0, %dim : index
        %difference = arith.subi %duplicate, %c0 : index
        linalg.yield %value : f32
    } -> tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);

  wafer::TensorNormalizationOutcome first =
      wafer::normalizeRequiredTensorModule(*module);
  ASSERT_EQ(first.status, wafer::TensorNormalizationStatus::Success);
  EXPECT_TRUE(first.changed);
  EXPECT_EQ(first.work.committedRewrites, 4u);
  EXPECT_EQ(wafer::verifyRequiredTensorNormalForm(*module).status,
            wafer::TensorNormalizationStatus::Success);

  wafer::TensorNormalizationOutcome second =
      wafer::normalizeRequiredTensorModule(*module);
  ASSERT_EQ(second.status, wafer::TensorNormalizationStatus::Success);
  EXPECT_FALSE(second.changed);
  EXPECT_EQ(second.work.committedRewrites, 0u);
}

TEST(OptimizationQualificationDriverTest,
     RequiredNormalizerUsesOnlyTheTypedEntryPrivateCallClosure) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func private @helper(%arg: tensor<4xf32>) -> tensor<4xf32> {
    %slice = tensor.extract_slice %arg[0] [4] [1]
      : tensor<4xf32> to tensor<4xf32>
    return %slice : tensor<4xf32>
  }
  func.func @entry(%arg: tensor<4xf32>) -> tensor<4xf32> {
    %result = func.call @helper(%arg) : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
  func.func private @unused(%arg: tensor<?xf32>) -> tensor<?xf32> {
    return %arg : tensor<?xf32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);
  wafer::SchedulableCallClosure closure =
      wafer::analyzeSchedulableCallClosure(*module);
  ASSERT_EQ(closure.status, wafer::SchedulableCallClosureStatus::Success);
  ASSERT_EQ(closure.functions.size(), 2u);
  EXPECT_EQ(closure.functions[0].getSymName(), "helper");
  EXPECT_EQ(closure.functions[1].getSymName(), "entry");

  wafer::TensorNormalizationOutcome outcome =
      wafer::normalizeRequiredTensorModule(*module);
  ASSERT_EQ(outcome.status, wafer::TensorNormalizationStatus::Success);
  EXPECT_TRUE(outcome.changed);
  EXPECT_EQ(outcome.work.committedRewrites, 1u);
  EXPECT_EQ(wafer::verifyRequiredTensorNormalForm(*module).status,
            wafer::TensorNormalizationStatus::Success);
  EXPECT_TRUE(module->lookupSymbol<mlir::func::FuncOp>("unused"));
}

TEST(OptimizationQualificationDriverTest,
     RequiredNormalizerRejectsRecursiveClosureWithoutMutatingSource) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @entry(%arg: i32) -> i32 {
    %result = func.call @entry(%arg) : (i32) -> i32
    return %result : i32
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);
  wafer::CanonicalIRSnapshotV1 before;
  wafer::CanonicalIRSnapshotV1 after;
  std::string diagnostic;
  ASSERT_TRUE(wafer::createCanonicalIRSnapshotV1(
      *module, wafer::CanonicalIRSnapshotMode::MutationGuard, before,
      &diagnostic));
  wafer::TensorNormalizationOutcome outcome =
      wafer::normalizeRequiredTensorModule(*module);
  EXPECT_EQ(outcome.status,
            wafer::TensorNormalizationStatus::UnsupportedSemantic);
  ASSERT_TRUE(outcome.diagnostic);
  EXPECT_EQ(outcome.diagnostic->reason,
            "schedulable path contains a recursive call cycle");
  ASSERT_TRUE(wafer::createCanonicalIRSnapshotV1(
      *module, wafer::CanonicalIRSnapshotMode::MutationGuard, after,
      &diagnostic));
  EXPECT_EQ(before.structuralBytes, after.structuralBytes);
}

TEST(OptimizationQualificationDriverTest,
     ModuleWideNormalizerKeepsProductionSingleEntryAnalysisStrict) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @first(%arg: tensor<4xf32>) -> tensor<4xf32> {
    %slice = tensor.extract_slice %arg[0] [4] [1]
      : tensor<4xf32> to tensor<4xf32>
    return %slice : tensor<4xf32>
  }
  func.func @second(%arg: tensor<8xf32>) -> tensor<8xf32> {
    %slice = tensor.extract_slice %arg[0] [8] [1]
      : tensor<8xf32> to tensor<8xf32>
    return %slice : tensor<8xf32>
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);

  wafer::SchedulableCallClosure production =
      wafer::analyzeSchedulableCallClosure(*module);
  EXPECT_EQ(production.status, wafer::SchedulableCallClosureStatus::InvalidIR);
  EXPECT_EQ(production.reason,
            wafer::SchedulableCallClosureReason::MultipleEntries);

  wafer::SchedulableCallClosure moduleWide =
      wafer::analyzeAllSchedulableCallClosures(*module);
  ASSERT_EQ(moduleWide.status, wafer::SchedulableCallClosureStatus::Success);
  ASSERT_EQ(moduleWide.functions.size(), 2u);
  wafer::TensorNormalizationOutcome outcome =
      wafer::normalizeRequiredTensorModule(*module);
  ASSERT_EQ(outcome.status, wafer::TensorNormalizationStatus::Success);
  EXPECT_EQ(outcome.work.committedRewrites, 2u);
  EXPECT_EQ(wafer::verifyRequiredTensorNormalForm(*module).status,
            wafer::TensorNormalizationStatus::Success);
}

TEST(OptimizationQualificationDriverTest,
     SelectedPayloadNormalizationOwnsSubviewAndTraversalPostcondition) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @normalize(%arg: memref<4xf32>) {
    %view = memref.subview %arg[0] [4] [1]
      : memref<4xf32> to memref<4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
      memref.prefetch %view[%i], read, locality<1>, data
          : memref<4xf32>
    }
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);

  wafer::SelectedPayloadNormalizationOutcome first =
      wafer::normalizeSelectedPhysicalPayload(*module);
  ASSERT_EQ(first.status, wafer::SelectedPayloadNormalizationStatus::Success);
  EXPECT_TRUE(first.changed);
  EXPECT_EQ(first.work.committedRewrites, 3u);
  wafer::SelectedPayloadNormalizationOutcome second =
      wafer::normalizeSelectedPhysicalPayload(*module);
  ASSERT_EQ(second.status, wafer::SelectedPayloadNormalizationStatus::Success);
  EXPECT_FALSE(second.changed);
  EXPECT_EQ(second.work.committedRewrites, 0u);
  EXPECT_EQ(wafer::verifySelectedPhysicalPayloadNormalForm(*module).status,
            wafer::SelectedPayloadNormalizationStatus::Success);
}

static std::vector<wafer::ABBASample>
makeABBASamples(uint64_t aWall, uint64_t bWall, uint64_t aRss, uint64_t bRss) {
  std::vector<wafer::ABBASample> samples;
  for (uint32_t block = 0; block < 5; ++block) {
    samples.push_back({block, wafer::ABBASamplePosition::ALeft, aWall + block,
                       aRss + block, true});
    samples.push_back({block, wafer::ABBASamplePosition::BLeft, bWall + block,
                       bRss + block, true});
    samples.push_back({block, wafer::ABBASamplePosition::BRight, bWall + block,
                       bRss + block, true});
    samples.push_back({block, wafer::ABBASamplePosition::ARight, aWall + block,
                       aRss + block, true});
  }
  return samples;
}

TEST(OptimizationQualificationDriverTest,
     ExactABBAQualificationAppliesFrozenMedianMadGuard) {
  auto evaluation = wafer::evaluateOptimizationSetABBA(
      makeABBASamples(/*aWall=*/100, /*bWall=*/104,
                      /*aRss=*/1000, /*bRss=*/900));
  ASSERT_TRUE(evaluation.decision) << evaluation.diagnostic;
  EXPECT_TRUE(evaluation.decision->accepted);
  EXPECT_TRUE(evaluation.decision->significantHostBenefit);
  EXPECT_EQ(evaluation.decision->wall.baselineMedian,
            (wafer::ExactRationalValue{"102", "1"}));
  EXPECT_EQ(evaluation.decision->wall.deltaMedian,
            (wafer::ExactRationalValue{"4", "1"}));
  EXPECT_EQ(evaluation.decision->wall.medianAbsoluteDeviation,
            (wafer::ExactRationalValue{"0", "1"}));
  EXPECT_EQ(evaluation.decision->wall.guard,
            (wafer::ExactRationalValue{"51", "10"}));
  EXPECT_FALSE(evaluation.decision->wall.significantImprovement);
  EXPECT_TRUE(evaluation.decision->peakRss.significantImprovement);
}

TEST(OptimizationQualificationDriverTest,
     ExactABBAQualificationRejectsRegressionAndMalformedEvidence) {
  std::vector<wafer::ABBASample> samples =
      makeABBASamples(/*aWall=*/100, /*bWall=*/107,
                      /*aRss=*/1000, /*bRss=*/1000);
  auto regression = wafer::evaluateOptimizationSetABBA(samples);
  ASSERT_TRUE(regression.decision) << regression.diagnostic;
  EXPECT_FALSE(regression.decision->accepted);

  samples[3].position = wafer::ABBASamplePosition::BRight;
  auto wrongOrder = wafer::evaluateOptimizationSetABBA(samples);
  EXPECT_FALSE(wrongOrder.decision);
  EXPECT_NE(wrongOrder.diagnostic.find("canonical"), std::string::npos);

  samples = makeABBASamples(100, 100, 1000, 1000);
  samples[7].processSucceeded = false;
  auto failedProcess = wafer::evaluateOptimizationSetABBA(samples);
  EXPECT_FALSE(failedProcess.decision);
  EXPECT_NE(failedProcess.diagnostic.find("non-success"), std::string::npos);
}

TEST(OptimizationQualificationDriverTest,
     OptimizationSetPolicyIsCanonicalAndBindsClosedRegistries) {
  wafer::OptimizationSetQualificationPolicyV1 policy =
      wafer::getCurrentOptimizationSetQualificationPolicyV1();
  std::string diagnostic;
  ASSERT_TRUE(
      wafer::validateOptimizationSetQualificationPolicyV1(policy, &diagnostic))
      << diagnostic;
  EXPECT_EQ(policy.mandatoryRankCounts, (std::vector<uint32_t>{1, 16}));
  EXPECT_EQ(wafer::getCurrentMandatoryQualificationCasesV1().size(), 3u);
  std::vector<wafer::RegistryRefV1> corpusRows =
      wafer::getCurrentMandatoryCorpusRowsV1();
  ASSERT_EQ(corpusRows.size(), 2u);
  for (const wafer::RegistryRefV1 &row : corpusRows) {
    EXPECT_NE(row.id, 0u);
    EXPECT_EQ(row.registrySchema,
              policy.mandatoryCorpusRegistry.registrySchema);
    EXPECT_EQ(row.registryDigest,
              policy.mandatoryCorpusRegistry.registryDigest);
  }

  std::vector<uint8_t> bytes =
      wafer::encodeOptimizationSetQualificationPolicyV1(policy);
  ASSERT_FALSE(bytes.empty());
  wafer::OptimizationSetQualificationPolicyV1 decoded;
  ASSERT_TRUE(wafer::decodeCanonicalOptimizationSetQualificationPolicyV1(
      bytes, decoded, &diagnostic))
      << diagnostic;
  EXPECT_EQ(wafer::encodeOptimizationSetQualificationPolicyV1(decoded), bytes);
  wafer::QualificationPolicyRefV1 ref =
      wafer::getCurrentOptimizationSetQualificationPolicyRefV1();
  EXPECT_EQ(ref.policySchema, policy.schemaVersion);
  EXPECT_EQ(ref.canonicalPolicyDigest,
            wafer::digestOptimizationSetQualificationPolicyV1(policy));

  bytes.front() ^= 1;
  EXPECT_FALSE(wafer::decodeCanonicalOptimizationSetQualificationPolicyV1(
      bytes, decoded, &diagnostic));
}

TEST(OptimizationQualificationDriverTest,
     OptimizationSetPolicyRejectsUnregisteredGuardAndInsufficientCaps) {
  wafer::OptimizationSetQualificationPolicyV1 policy =
      wafer::getCurrentOptimizationSetQualificationPolicyV1();
  std::string diagnostic;

  policy.mandatoryRankCounts = {16, 1};
  EXPECT_FALSE(
      wafer::validateOptimizationSetQualificationPolicyV1(policy, &diagnostic));
  EXPECT_NE(diagnostic.find("sorted"), std::string::npos);

  policy = wafer::getCurrentOptimizationSetQualificationPolicyV1();
  policy.wallGuardFractionDenominator = 0;
  EXPECT_FALSE(
      wafer::validateOptimizationSetQualificationPolicyV1(policy, &diagnostic));
  EXPECT_NE(diagnostic.find("fixed ABBA guard"), std::string::npos);

  policy = wafer::getCurrentOptimizationSetQualificationPolicyV1();
  policy.totalProcessLaunchCap = 1;
  EXPECT_FALSE(
      wafer::validateOptimizationSetQualificationPolicyV1(policy, &diagnostic));
  EXPECT_NE(diagnostic.find("process launch cap"), std::string::npos);

  policy = wafer::getCurrentOptimizationSetQualificationPolicyV1();
  policy.totalGatewayInvocationCap = 1;
  EXPECT_FALSE(
      wafer::validateOptimizationSetQualificationPolicyV1(policy, &diagnostic));
  EXPECT_NE(diagnostic.find("gateway invocation cap"), std::string::npos);
}

TEST(OptimizationQualificationDriverTest,
     OptimizationBatchEvidenceIsCanonicalAndAllAndOnly) {
  wafer::OptimizationBatchObservationV1 batch = makeQualifiedBatchEvidence();
  std::string diagnostic;
  ASSERT_TRUE(wafer::validateOptimizationBatchObservationV1(batch, &diagnostic))
      << diagnostic;
  std::vector<uint8_t> bytes =
      wafer::encodeOptimizationBatchObservationV1(batch);
  ASSERT_FALSE(bytes.empty());
  wafer::OptimizationBatchObservationV1 decoded;
  ASSERT_TRUE(wafer::decodeCanonicalOptimizationBatchObservationV1(
      bytes, decoded, &diagnostic))
      << diagnostic;
  EXPECT_EQ(wafer::encodeOptimizationBatchObservationV1(decoded), bytes);

  ASSERT_TRUE(batch.globalStaticComparisons.empty());
  ASSERT_TRUE(batch.globalABBASamples.empty());
  batch.globalStaticComparisons.push_back(makeStaticComparison(
      wafer::getCurrentMandatoryQualificationCasesV1().front(),
      wafer::OptimizationComparisonKindV1::GlobalAllOffVsAllOn));
  EXPECT_FALSE(
      wafer::validateOptimizationBatchObservationV1(batch, &diagnostic));
  EXPECT_NE(diagnostic.find("mandatory domain"), std::string::npos);
}

TEST(OptimizationQualificationDriverTest,
     EmptyProposalHasNoPerKeyQualificationDomain) {
  wafer::OptimizationQualificationProposal proposal =
      wafer::getCurrentOptimizationQualificationProposal();
  ASSERT_TRUE(proposal.fixedBindings.empty());
  ASSERT_TRUE(proposal.cleanupBindings.empty());
  EXPECT_EQ(wafer::lookupAdoptionSpec(wafer::mechanism::StablehloCleanup)
                ->adoptionMode,
            wafer::AdoptionMode::None);
  EXPECT_EQ(wafer::lookupAdoptionSpec(wafer::mechanism::CandidateCommitCleanup)
                ->adoptionMode,
            wafer::AdoptionMode::None);
  std::string diagnostic;
  EXPECT_EQ(wafer::isOptimizationMechanismEnabled(
                proposal, wafer::getAllOnOptimizationConfiguration(),
                wafer::mechanism::StablehloCleanup, &diagnostic),
            std::nullopt);
  EXPECT_NE(
      diagnostic.find("absent from the optimization qualification proposal"),
      std::string::npos);
}

TEST(OptimizationQualificationDriverTest,
     QualificationArchiveRecordsRoundTripAsOneDigestChain) {
  wafer::AdoptionQualificationInputV1 input;
  for (const wafer::AdoptionSpec &spec : wafer::getAllAdoptionSpecs())
    input.specBindings.push_back(
        {spec.mechanismKey, wafer::digestAdoptionSpecV1(spec)});
  uint8_t snapshotByte = 1;
  wafer::OptimizationSetQualificationPolicyV1 policy =
      wafer::getCurrentOptimizationSetQualificationPolicyV1();
  for (const wafer::QualificationCaseKeyV1 &original :
       wafer::getCurrentMandatoryQualificationCasesV1())
    for (wafer::EquivalentInputVariantV1 variant :
         {wafer::EquivalentInputVariantV1::Original,
          wafer::EquivalentInputVariantV1::Metamorphic})
      input.qualificationCases.push_back(
          {{original.corpus, original.rankCount, variant},
           filledDigest(snapshotByte++)});
  input.optimizationProposalDigest =
      wafer::digestOptimizationQualificationProposalV1(
          wafer::getCurrentOptimizationQualificationProposal());
  std::vector<uint8_t> inputBytes =
      wafer::encodeAdoptionQualificationInputV1(input);
  ASSERT_FALSE(inputBytes.empty());
  wafer::AdoptionQualificationInputV1 decodedInput;
  std::string diagnostic;
  ASSERT_TRUE(wafer::decodeCanonicalAdoptionQualificationInputV1(
      inputBytes, decodedInput, &diagnostic))
      << diagnostic;
  EXPECT_EQ(wafer::encodeAdoptionQualificationInputV1(decodedInput),
            inputBytes);

  wafer::AdoptionQualificationRunV1 run;
  run.qualificationInputDigest =
      wafer::digestAdoptionQualificationInputV1(input);
  run.qualificationIdentity = {filledDigest(20), filledDigest(21),
                               filledDigest(22), filledDigest(23),
                               filledDigest(24)};
  run.qualificationPolicy =
      wafer::getCurrentOptimizationSetQualificationPolicyRefV1();
  run.runSeriesOrdinal = 7;
  run.attemptOrdinal = 0;
  std::vector<uint8_t> runBytes = wafer::encodeAdoptionQualificationRunV1(run);
  ASSERT_FALSE(runBytes.empty());
  wafer::AdoptionQualificationRunV1 decodedRun;
  ASSERT_TRUE(wafer::decodeCanonicalAdoptionQualificationRunV1(
      runBytes, decodedRun, &diagnostic))
      << diagnostic;

  wafer::AdoptionQualificationResultManifestV1 manifest;
  manifest.qualificationRunDigest =
      wafer::digestAdoptionQualificationRunV1(run);
  manifest.invocationTerminals = {{filledDigest(30), filledDigest(31)}};
  manifest.observationBindings = {
      {wafer::mechanism::RequiredTensorNormalization, filledDigest(32)}};
  manifest.optimizationBatchObservationDigest = filledDigest(33);
  std::vector<uint8_t> manifestBytes =
      wafer::encodeAdoptionQualificationResultManifestV1(manifest);
  ASSERT_FALSE(manifestBytes.empty());
  wafer::AdoptionQualificationResultManifestV1 decodedManifest;
  ASSERT_TRUE(wafer::decodeCanonicalAdoptionQualificationResultManifestV1(
      manifestBytes, decodedManifest, &diagnostic))
      << diagnostic;

  wafer::AdoptionQualificationRunTerminalV1 runTerminal;
  runTerminal.qualificationRunDigest = manifest.qualificationRunDigest;
  runTerminal.outcome =
      wafer::AdoptionQualificationRunOutcomeV1::CompletedEvidence;
  runTerminal.resultManifestDigest =
      wafer::digestAdoptionQualificationResultManifestV1(manifest);
  std::vector<uint8_t> runTerminalBytes =
      wafer::encodeAdoptionQualificationRunTerminalV1(runTerminal);
  ASSERT_FALSE(runTerminalBytes.empty());
  wafer::AdoptionQualificationRunTerminalV1 decodedRunTerminal;
  ASSERT_TRUE(wafer::decodeCanonicalAdoptionQualificationRunTerminalV1(
      runTerminalBytes, decodedRunTerminal, &diagnostic))
      << diagnostic;

  wafer::OptimizationSetPublicationAttemptV1 attempt;
  attempt.qualificationRunDigest = manifest.qualificationRunDigest;
  attempt.proposalDigest = *input.optimizationProposalDigest;
  std::vector<uint8_t> attemptBytes =
      wafer::encodeOptimizationSetPublicationAttemptV1(attempt);
  ASSERT_FALSE(attemptBytes.empty());
  wafer::OptimizationSetPublicationAttemptV1 decodedAttempt;
  ASSERT_TRUE(wafer::decodeCanonicalOptimizationSetPublicationAttemptV1(
      attemptBytes, decodedAttempt, &diagnostic))
      << diagnostic;

  wafer::QualifiedOptimizationSetV1 qualifiedSet;
  qualifiedSet.proposalDigest = *input.optimizationProposalDigest;
  qualifiedSet.batchObservationDigest = filledDigest(33);
  wafer::OptimizationQualificationProposal proposal =
      wafer::getCurrentOptimizationQualificationProposal();
  uint8_t observationByte = 80;
  for (const wafer::MechanismSpecBinding &binding : proposal.fixedBindings)
    qualifiedSet.observationBindings.push_back(
        {binding.mechanismKey, filledDigest(observationByte++)});
  for (const wafer::MechanismSpecBinding &binding : proposal.cleanupBindings)
    qualifiedSet.observationBindings.push_back(
        {binding.mechanismKey, filledDigest(observationByte++)});
  std::sort(qualifiedSet.observationBindings.begin(),
            qualifiedSet.observationBindings.end(),
            [](const wafer::MechanismObservationBindingV1 &lhs,
               const wafer::MechanismObservationBindingV1 &rhs) {
              return lhs.mechanismKey < rhs.mechanismKey;
            });
  std::vector<uint8_t> qualifiedSetBytes =
      wafer::encodeQualifiedOptimizationSetV1(qualifiedSet);
  ASSERT_FALSE(qualifiedSetBytes.empty());
  wafer::QualifiedOptimizationSetV1 decodedQualifiedSet;
  ASSERT_TRUE(wafer::decodeCanonicalQualifiedOptimizationSetV1(
      qualifiedSetBytes, decodedQualifiedSet, &diagnostic))
      << diagnostic;
  EXPECT_EQ(wafer::encodeQualifiedOptimizationSetV1(decodedQualifiedSet),
            qualifiedSetBytes);

  wafer::OptimizationSetPublicationTerminalV1 publicationTerminal;
  publicationTerminal.optimizationPublicationAttemptDigest =
      wafer::digestOptimizationSetPublicationAttemptV1(attempt);
  publicationTerminal.outcome =
      wafer::OptimizationSetPublicationOutcomeV1::QualifiedPublished;
  publicationTerminal.batchObservationDigest = filledDigest(33);
  publicationTerminal.candidateSetDigest =
      wafer::digestQualifiedOptimizationSetV1(qualifiedSet);
  std::vector<uint8_t> publicationTerminalBytes =
      wafer::encodeOptimizationSetPublicationTerminalV1(publicationTerminal);
  ASSERT_FALSE(publicationTerminalBytes.empty());
  wafer::OptimizationSetPublicationTerminalV1 decodedPublicationTerminal;
  ASSERT_TRUE(wafer::decodeCanonicalOptimizationSetPublicationTerminalV1(
      publicationTerminalBytes, decodedPublicationTerminal, &diagnostic))
      << diagnostic;

  wafer::ActiveQualifiedOptimizationSetRefV1 active;
  active.generation = 1;
  active.setDigest = *publicationTerminal.candidateSetDigest;
  active.qualificationRunDigest = manifest.qualificationRunDigest;
  active.adoptionQualificationRunTerminalDigest =
      wafer::digestAdoptionQualificationRunTerminalV1(runTerminal);
  active.optimizationPublicationAttemptDigest =
      publicationTerminal.optimizationPublicationAttemptDigest;
  active.optimizationPublicationTerminalDigest =
      wafer::digestOptimizationSetPublicationTerminalV1(publicationTerminal);
  std::vector<uint8_t> activeBytes =
      wafer::encodeActiveQualifiedOptimizationSetRefV1(active);
  ASSERT_FALSE(activeBytes.empty());
  wafer::ActiveQualifiedOptimizationSetRefV1 decodedActive;
  ASSERT_TRUE(wafer::decodeCanonicalActiveQualifiedOptimizationSetRefV1(
      activeBytes, decodedActive, &diagnostic))
      << diagnostic;
  EXPECT_EQ(wafer::encodeActiveQualifiedOptimizationSetRefV1(decodedActive),
            activeBytes);
}

TEST(OptimizationQualificationDriverTest,
     CompletedEvidenceChainIsFreshlyAggregatedFromTerminals) {
  wafer::CompletedQualificationEvidenceV1 chain;
  for (const wafer::AdoptionSpec &spec : wafer::getAllAdoptionSpecs())
    chain.input.specBindings.push_back(
        {spec.mechanismKey, wafer::digestAdoptionSpecV1(spec)});
  uint8_t snapshot = 1;
  for (const wafer::QualificationCaseKeyV1 &original :
       wafer::getCurrentMandatoryQualificationCasesV1())
    for (wafer::EquivalentInputVariantV1 variant :
         {wafer::EquivalentInputVariantV1::Original,
          wafer::EquivalentInputVariantV1::Metamorphic})
      chain.input.qualificationCases.push_back(
          {{original.corpus, original.rankCount, variant},
           filledDigest(snapshot++)});
  wafer::OptimizationQualificationProposal proposal =
      wafer::getCurrentOptimizationQualificationProposal();
  chain.input.optimizationProposalDigest =
      wafer::digestOptimizationQualificationProposalV1(proposal);

  chain.run.qualificationInputDigest =
      wafer::digestAdoptionQualificationInputV1(chain.input);
  chain.run.qualificationIdentity = {filledDigest(30), filledDigest(31),
                                     filledDigest(32), filledDigest(33),
                                     filledDigest(34)};
  chain.run.qualificationPolicy =
      wafer::getCurrentOptimizationSetQualificationPolicyRefV1();
  chain.run.runSeriesOrdinal = 1;
  wafer::AdoptionDigest runDigest =
      wafer::digestAdoptionQualificationRunV1(chain.run);

  wafer::OptimizationSetPublicationAttemptV1 attempt;
  attempt.qualificationRunDigest = runDigest;
  attempt.proposalDigest = *chain.input.optimizationProposalDigest;
  chain.publicationAttempt = attempt;
  wafer::AdoptionDigest attemptDigest =
      wafer::digestOptimizationSetPublicationAttemptV1(attempt);

  wafer::AdoptionSpec requiredSpec =
      *wafer::lookupAdoptionSpec(wafer::mechanism::RequiredTensorNormalization);
  const wafer::QualificationInputCaseV1 &terminalCase =
      chain.input.qualificationCases.front();
  wafer::InvocationTelemetryV1 terminal;
  terminal.identity.scopeKind = wafer::InvocationScopeKindV1::QualificationRun;
  terminal.identity.scopeDigest = runDigest;
  terminal.identity.mechanismKey = requiredSpec.mechanismKey;
  terminal.identity.invocationSite =
      *wafer::lookupOptimizationInvocationSiteV1(requiredSpec.mechanismKey);
  terminal.identity.cutPoint = requiredSpec.cutPoint;
  terminal.qualificationCase = terminalCase.caseKey;
  terminal.specDigest = wafer::digestAdoptionSpecV1(requiredSpec);
  terminal.inputSnapshotDigest = terminalCase.inputSnapshotDigest;
  terminal.outcome = wafer::InvocationOutcome::Applied;
  terminal.rewriteCount = 1;
  terminal.workSummary.workPolicyDigest =
      wafer::digestAdoptionWorkPolicyV1(requiredSpec.workPolicyKind);
  terminal.workSummary.orderedCounters = {{1, 1}};
  chain.invocationTerminals.push_back(terminal);

  wafer::ClosedReasonV1 rejectedReason{
      wafer::getGlobalClosedReasonRefV1(
          wafer::GlobalClosedReasonV1::QualificationEvidenceRejected),
      std::nullopt};
  uint64_t nextInvocationOrdinal = 1;
  for (const wafer::AdoptionSpec &spec : wafer::getAllAdoptionSpecs()) {
    wafer::QualificationObservationV1 observation;
    observation.mechanismKey = spec.mechanismKey;
    observation.specDigest = wafer::digestAdoptionSpecV1(spec);
    observation.qualificationIdentity = chain.run.qualificationIdentity;
    observation.qualificationPolicy = chain.run.qualificationPolicy;
    observation.compileWorkSummary.workPolicyDigest =
        wafer::digestAdoptionWorkPolicyV1(spec.workPolicyKind);
    observation.qualificationRunDigest = runDigest;
    if (spec.mechanismKey == requiredSpec.mechanismKey) {
      observation.qualificationStatus = wafer::QualificationStatusV1::Qualified;
      observation.outcomeCounts = {
          {wafer::InvocationOutcome::Applied, std::nullopt, 1}};
      observation.invocationEvidence = {{terminal.identity.invocationSite,
                                         terminal.identity.cutPoint,
                                         *terminal.qualificationCase,
                                         1,
                                         1,
                                         {}}};
      observation.compileWorkSummary.orderedCounters = {{1, 1}};
      observation.mechanismGateResults.orderedGateResults.push_back(
          makePassingGate(wafer::getCurrentProductionAllOnGateRefV1(),
                          *terminal.qualificationCase,
                          evidenceAllOnConfiguration(), 60));
    } else if (spec.adoptionMode == wafer::AdoptionMode::FixedOptimization ||
               spec.adoptionMode == wafer::AdoptionMode::BestEffortCleanup) {
      observation.qualificationStatus = wafer::QualificationStatusV1::Qualified;
      observation.optimizationProposalDigest =
          *chain.input.optimizationProposalDigest;
      observation.optimizationPublicationAttemptDigest = attemptDigest;
      observation.outcomeCounts = {
          {wafer::InvocationOutcome::Applied, std::nullopt,
           2 * wafer::getCurrentMandatoryQualificationCasesV1().size()}};
      wafer::OptimizationComparisonKindV1 kind =
          spec.adoptionMode == wafer::AdoptionMode::FixedOptimization
              ? wafer::OptimizationComparisonKindV1::FixedDisableOneVsAllOn
              : wafer::OptimizationComparisonKindV1::CleanupDisableOneVsAllOn;
      for (const wafer::QualificationCaseKeyV1 &original :
           wafer::getCurrentMandatoryQualificationCasesV1()) {
        auto comparison =
            makeStaticComparison(original, kind, spec.mechanismKey);
        observation.staticComparisons.push_back(std::move(comparison));
        for (wafer::EquivalentInputVariantV1 variant :
             {wafer::EquivalentInputVariantV1::Original,
              wafer::EquivalentInputVariantV1::Metamorphic}) {
          wafer::QualificationCaseKeyV1 caseKey{original.corpus,
                                                original.rankCount, variant};
          observation.invocationEvidence.push_back(
              {*wafer::lookupOptimizationInvocationSiteV1(spec.mechanismKey),
               spec.cutPoint,
               caseKey,
               1,
               1,
               {}});
          observation.mechanismGateResults.orderedGateResults.push_back(
              makePassingGate(
                  wafer::getCurrentProductionAllOnGateRefV1(), caseKey,
                  evidenceAllOnConfiguration(),
                  static_cast<uint8_t>(70 + spec.mechanismKey.semanticId)));
          auto inputCase = std::find_if(
              chain.input.qualificationCases.begin(),
              chain.input.qualificationCases.end(),
              [&](const wafer::QualificationInputCaseV1 &candidate) {
                return candidate.caseKey.corpus.id == caseKey.corpus.id &&
                       candidate.caseKey.rankCount == caseKey.rankCount &&
                       candidate.caseKey.inputVariant == caseKey.inputVariant;
              });
          ASSERT_NE(inputCase, chain.input.qualificationCases.end());
          wafer::InvocationTelemetryV1 cleanupTerminal;
          cleanupTerminal.identity.scopeKind =
              wafer::InvocationScopeKindV1::QualificationRun;
          cleanupTerminal.identity.scopeDigest = runDigest;
          cleanupTerminal.identity.mechanismKey = spec.mechanismKey;
          cleanupTerminal.identity.invocationSite =
              *wafer::lookupOptimizationInvocationSiteV1(spec.mechanismKey);
          cleanupTerminal.identity.cutPoint = spec.cutPoint;
          cleanupTerminal.identity.invocationOrdinal = nextInvocationOrdinal++;
          cleanupTerminal.qualificationCase = caseKey;
          cleanupTerminal.specDigest = wafer::digestAdoptionSpecV1(spec);
          cleanupTerminal.inputSnapshotDigest = inputCase->inputSnapshotDigest;
          cleanupTerminal.outcome = wafer::InvocationOutcome::Applied;
          cleanupTerminal.rewriteCount = 1;
          cleanupTerminal.workSummary.workPolicyDigest =
              wafer::digestAdoptionWorkPolicyV1(spec.workPolicyKind);
          chain.invocationTerminals.push_back(std::move(cleanupTerminal));
        }
      }
      std::sort(observation.staticComparisons.begin(),
                observation.staticComparisons.end(),
                [](const auto &lhs, const auto &rhs) {
                  return std::tie(lhs.comparisonKey.caseKey.corpus.id,
                                  lhs.comparisonKey.caseKey.rankCount) <
                         std::tie(rhs.comparisonKey.caseKey.corpus.id,
                                  rhs.comparisonKey.caseKey.rankCount);
                });
      for (const auto &comparison : observation.staticComparisons)
        appendPassingABBA(comparison, observation.abbaSamples);
      std::sort(observation.invocationEvidence.begin(),
                observation.invocationEvidence.end(),
                [](const wafer::InvocationEvidenceV1 &lhs,
                   const wafer::InvocationEvidenceV1 &rhs) {
                  return std::tie(lhs.invocationSite.id, lhs.cutPoint,
                                  lhs.caseKey.corpus.id, lhs.caseKey.rankCount,
                                  lhs.caseKey.inputVariant) <
                         std::tie(rhs.invocationSite.id, rhs.cutPoint,
                                  rhs.caseKey.corpus.id, rhs.caseKey.rankCount,
                                  rhs.caseKey.inputVariant);
                });
      std::sort(
          observation.mechanismGateResults.orderedGateResults.begin(),
          observation.mechanismGateResults.orderedGateResults.end(),
          [](const wafer::GateEvidenceV1 &lhs,
             const wafer::GateEvidenceV1 &rhs) {
            return std::tie(lhs.gate.id, lhs.caseKey.corpus.id,
                            lhs.caseKey.rankCount, lhs.caseKey.inputVariant) <
                   std::tie(rhs.gate.id, rhs.caseKey.corpus.id,
                            rhs.caseKey.rankCount, rhs.caseKey.inputVariant);
          });
    } else {
      observation.qualificationStatus = wafer::QualificationStatusV1::Rejected;
      observation.closedReason = rejectedReason;
    }
    ASSERT_FALSE(wafer::encodeQualificationObservationV1(observation).empty());
    chain.observations.push_back(std::move(observation));
  }

  chain.optimizationBatch = makeQualifiedBatchEvidence();
  chain.optimizationBatch->qualificationIdentity =
      chain.run.qualificationIdentity;
  chain.optimizationBatch->qualificationPolicy = chain.run.qualificationPolicy;
  chain.optimizationBatch->qualificationRunDigest = runDigest;
  chain.optimizationBatch->optimizationPublicationAttemptDigest = attemptDigest;

  chain.resultManifest.qualificationRunDigest = runDigest;
  std::sort(chain.invocationTerminals.begin(), chain.invocationTerminals.end(),
            [](const wafer::InvocationTelemetryV1 &lhs,
               const wafer::InvocationTelemetryV1 &rhs) {
              return wafer::digestInvocationIdentityV1(lhs.identity) <
                     wafer::digestInvocationIdentityV1(rhs.identity);
            });
  for (const wafer::InvocationTelemetryV1 &invocation :
       chain.invocationTerminals)
    chain.resultManifest.invocationTerminals.push_back(
        {wafer::digestInvocationIdentityV1(invocation.identity),
         wafer::digestInvocationTelemetryV1(invocation)});
  for (const wafer::QualificationObservationV1 &observation :
       chain.observations)
    chain.resultManifest.observationBindings.push_back(
        {observation.mechanismKey,
         wafer::digestQualificationObservationV1(observation)});
  chain.resultManifest.optimizationBatchObservationDigest =
      wafer::digestOptimizationBatchObservationV1(*chain.optimizationBatch);
  chain.runTerminal.qualificationRunDigest = runDigest;
  chain.runTerminal.outcome =
      wafer::AdoptionQualificationRunOutcomeV1::CompletedEvidence;
  chain.runTerminal.resultManifestDigest =
      wafer::digestAdoptionQualificationResultManifestV1(chain.resultManifest);

  std::string diagnostic;
  ASSERT_TRUE(
      wafer::validateCompletedQualificationEvidenceV1(chain, &diagnostic))
      << diagnostic;
  llvm::SmallString<256> archiveRoot;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory(
      "/tmp/wafer-qualification-archive", archiveRoot));
  auto cleanupArchive = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(archiveRoot); });
  wafer::OptimizationQualificationArchiveStoreV1 store(archiveRoot.str().str());
  ASSERT_TRUE(store.publishCompletedRun(chain, &diagnostic)) << diagnostic;
  wafer::CompletedQualificationEvidenceV1 readback;
  ASSERT_TRUE(store.readCompletedRun(runDigest, readback, &diagnostic))
      << diagnostic;
  EXPECT_EQ(wafer::digestAdoptionQualificationRunV1(readback.run), runDigest);
  EXPECT_FALSE(store.publishCompletedRun(chain, &diagnostic));
  EXPECT_NE(diagnostic.find("permanently sealed"), std::string::npos);

  wafer::QualifiedOptimizationSetV1 candidateSet;
  candidateSet.proposalDigest = *chain.input.optimizationProposalDigest;
  candidateSet.batchObservationDigest =
      wafer::digestOptimizationBatchObservationV1(*chain.optimizationBatch);
  for (const wafer::MechanismSpecBinding &binding : proposal.fixedBindings) {
    auto observation =
        std::find_if(chain.observations.begin(), chain.observations.end(),
                     [&](const wafer::QualificationObservationV1 &value) {
                       return value.mechanismKey == binding.mechanismKey;
                     });
    ASSERT_NE(observation, chain.observations.end());
    candidateSet.observationBindings.push_back(
        {binding.mechanismKey,
         wafer::digestQualificationObservationV1(*observation)});
  }
  for (const wafer::MechanismSpecBinding &binding : proposal.cleanupBindings) {
    auto observation =
        std::find_if(chain.observations.begin(), chain.observations.end(),
                     [&](const wafer::QualificationObservationV1 &value) {
                       return value.mechanismKey == binding.mechanismKey;
                     });
    ASSERT_NE(observation, chain.observations.end());
    candidateSet.observationBindings.push_back(
        {binding.mechanismKey,
         wafer::digestQualificationObservationV1(*observation)});
  }
  std::sort(candidateSet.observationBindings.begin(),
            candidateSet.observationBindings.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.mechanismKey < rhs.mechanismKey;
            });
  wafer::OptimizationSetPublicationResultV1 publication;
  ASSERT_TRUE(store.publishOptimizationSet(runDigest, candidateSet, publication,
                                           &diagnostic))
      << diagnostic;
  ASSERT_EQ(publication.terminal.outcome,
            wafer::OptimizationSetPublicationOutcomeV1::QualifiedPublished);
  ASSERT_TRUE(publication.activeRef);
  wafer::ActiveQualifiedOptimizationSelectionV1 selection;
  ASSERT_TRUE(store.loadActiveQualifiedOptimizationSet(selection, &diagnostic))
      << diagnostic;
  EXPECT_EQ(selection.activeRef.setDigest,
            wafer::digestQualifiedOptimizationSetV1(candidateSet));
  EXPECT_FALSE(store.publishOptimizationSet(runDigest, candidateSet,
                                            publication, &diagnostic));
  EXPECT_NE(diagnostic.find("attempt is already permanently sealed"),
            std::string::npos);

  chain.resultManifest.invocationTerminals.front().terminalDigest[0] ^= 1;
  chain.runTerminal.resultManifestDigest =
      wafer::digestAdoptionQualificationResultManifestV1(chain.resultManifest);
  EXPECT_FALSE(
      wafer::validateCompletedQualificationEvidenceV1(chain, &diagnostic));
  EXPECT_NE(diagnostic.find("fresh all-and-only"), std::string::npos);
}

TEST(OptimizationQualificationDriverTest,
     TypedOptimizationConfigurationsAreFrozenToOwnProposalGroup) {
  wafer::OptimizationQualificationProposal proposal =
      wafer::getCurrentOptimizationQualificationProposal();
  std::string diagnostic;
  ASSERT_TRUE(
      wafer::validateOptimizationQualificationProposal(proposal, &diagnostic))
      << diagnostic;
  EXPECT_TRUE(proposal.fixedBindings.empty());
  ASSERT_TRUE(proposal.cleanupBindings.empty());
  std::vector<uint8_t> proposalBytes =
      wafer::encodeOptimizationQualificationProposalV1(proposal);
  ASSERT_FALSE(proposalBytes.empty());
  EXPECT_TRUE(wafer::validateCanonicalOptimizationQualificationProposalV1(
      proposalBytes, &diagnostic))
      << diagnostic;
  EXPECT_EQ(
      wafer::toHex(wafer::digestOptimizationQualificationProposalV1(proposal))
          .size(),
      64u);
  proposalBytes.back() ^= 1;
  EXPECT_FALSE(wafer::validateCanonicalOptimizationQualificationProposalV1(
      proposalBytes, &diagnostic));

  wafer::OptimizationConfiguration missing =
      wafer::getDisableOneOptimizationConfiguration(
          proposal, wafer::mechanism::StablehloCleanup, &diagnostic);
  EXPECT_NE(
      diagnostic.find("absent from the optimization qualification proposal"),
      std::string::npos);
  EXPECT_TRUE(
      wafer::validateOptimizationConfiguration(proposal, missing, &diagnostic));
  EXPECT_EQ(
      wafer::isOptimizationMechanismEnabled(
          proposal, missing, wafer::mechanism::StablehloCleanup, &diagnostic),
      std::nullopt);
  EXPECT_EQ(wafer::isOptimizationMechanismEnabled(
                proposal, wafer::getAllOffOptimizationConfiguration(),
                wafer::mechanism::RequiredTensorNormalization, &diagnostic),
            std::nullopt);
}

TEST(OptimizationQualificationDriverTest,
     CanonicalSnapshotSeparatesSemanticAndMutationGuardLocations) {
  mlir::MLIRContext context;
  auto first = mlir::ModuleOp::create(
      mlir::FileLineColLoc::get(&context, "first.mlir", 1, 2));
  auto second = mlir::ModuleOp::create(
      mlir::FileLineColLoc::get(&context, "second.mlir", 9, 8));

  wafer::CanonicalIRSnapshotV1 firstSemantic;
  wafer::CanonicalIRSnapshotV1 secondSemantic;
  wafer::CanonicalIRSnapshotV1 firstGuard;
  wafer::CanonicalIRSnapshotV1 secondGuard;
  std::string diagnostic;
  ASSERT_TRUE(wafer::createCanonicalIRSnapshotV1(
      first, wafer::CanonicalIRSnapshotMode::SemanticStructure, firstSemantic,
      &diagnostic))
      << diagnostic;
  ASSERT_TRUE(wafer::createCanonicalIRSnapshotV1(
      second, wafer::CanonicalIRSnapshotMode::SemanticStructure, secondSemantic,
      &diagnostic))
      << diagnostic;
  EXPECT_EQ(firstSemantic.structuralBytes, secondSemantic.structuralBytes);
  EXPECT_EQ(firstSemantic.sha256Digest, secondSemantic.sha256Digest);

  ASSERT_TRUE(wafer::createCanonicalIRSnapshotV1(
      first, wafer::CanonicalIRSnapshotMode::MutationGuard, firstGuard,
      &diagnostic))
      << diagnostic;
  ASSERT_TRUE(wafer::createCanonicalIRSnapshotV1(
      second, wafer::CanonicalIRSnapshotMode::MutationGuard, secondGuard,
      &diagnostic))
      << diagnostic;
  EXPECT_NE(firstGuard.structuralBytes, secondGuard.structuralBytes);
  EXPECT_NE(firstGuard.sha256Digest, secondGuard.sha256Digest);

  second->setAttr("semantic.marker",
                  mlir::StringAttr::get(&context, "changed"));
  ASSERT_TRUE(wafer::createCanonicalIRSnapshotV1(
      second, wafer::CanonicalIRSnapshotMode::SemanticStructure, secondSemantic,
      &diagnostic))
      << diagnostic;
  EXPECT_NE(firstSemantic.structuralBytes, secondSemantic.structuralBytes);
}

TEST(OptimizationQualificationDriverTest,
     CanonicalSnapshotRejectsAttributesWithoutRegisteredCodec) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  module->setAttr(
      "unsupported",
      mlir::OpaqueAttr::get(mlir::StringAttr::get(&context, "unregistered"),
                            "payload", mlir::NoneType::get(&context)));
  wafer::CanonicalIRSnapshotV1 snapshot;
  std::string diagnostic;
  EXPECT_FALSE(wafer::createCanonicalIRSnapshotV1(
      module, wafer::CanonicalIRSnapshotMode::MutationGuard, snapshot,
      &diagnostic));
  EXPECT_NE(diagnostic.find("no registered canonical codec"),
            std::string::npos);
  EXPECT_TRUE(snapshot.structuralBytes.empty());
}

TEST(OptimizationQualificationDriverTest,
     NormalizerFailureKeepsMutationGuardByteIdentical) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(&context);
  auto function = mlir::func::FuncOp::create(
      builder.getUnknownLoc(), "invalid_return",
      builder.getFunctionType({}, {builder.getI32Type()}));
  module.getBody()->push_back(function);
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

  wafer::CanonicalIRSnapshotV1 before;
  wafer::CanonicalIRSnapshotV1 after;
  std::string diagnostic;
  ASSERT_TRUE(wafer::createCanonicalIRSnapshotV1(
      module, wafer::CanonicalIRSnapshotMode::MutationGuard, before,
      &diagnostic))
      << diagnostic;
  wafer::TensorNormalizationOutcome tensor =
      wafer::normalizeRequiredTensorModule(module);
  EXPECT_EQ(tensor.status, wafer::TensorNormalizationStatus::InvalidIR);
  ASSERT_TRUE(wafer::createCanonicalIRSnapshotV1(
      module, wafer::CanonicalIRSnapshotMode::MutationGuard, after,
      &diagnostic))
      << diagnostic;
  EXPECT_EQ(before.structuralBytes, after.structuralBytes);

  wafer::SelectedPayloadNormalizationOutcome payload =
      wafer::normalizeSelectedPhysicalPayload(module);
  EXPECT_EQ(payload.status,
            wafer::SelectedPayloadNormalizationStatus::InvalidIR);
  ASSERT_TRUE(wafer::createCanonicalIRSnapshotV1(
      module, wafer::CanonicalIRSnapshotMode::MutationGuard, after,
      &diagnostic))
      << diagnostic;
  EXPECT_EQ(before.structuralBytes, after.structuralBytes);
}

TEST(OptimizationQualificationDriverTest,
     NormalizerWorkCountersRejectOverflowWithoutPartialMerge) {
  using namespace wafer::structured_optimization::detail;
  uint64_t value = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(addWorkCounter(value, 1));
  EXPECT_EQ(value, std::numeric_limits<uint64_t>::max());
  uint64_t product = 17;
  EXPECT_FALSE(
      multiplyWorkCounter(std::numeric_limits<uint64_t>::max(), 2, product));
  EXPECT_EQ(product, 17u);

  wafer::TensorNormalizationWorkSummary destination;
  destination.fuelConsumed = 9;
  destination.operationVisits = std::numeric_limits<uint64_t>::max();
  wafer::TensorNormalizationWorkSummary source;
  source.fuelConsumed = 3;
  source.operationVisits = 1;
  EXPECT_FALSE(mergeWork(destination, source));
  EXPECT_EQ(destination.fuelConsumed, 9u);
  EXPECT_EQ(destination.operationVisits, std::numeric_limits<uint64_t>::max());
}

} // namespace
