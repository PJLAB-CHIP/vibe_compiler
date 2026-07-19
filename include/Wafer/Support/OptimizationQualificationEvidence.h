//===- OptimizationQualificationEvidence.h - Evidence records -*- C++ -*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONEVIDENCE_H
#define WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONEVIDENCE_H

#include "Wafer/Support/OptimizationQualification.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer {

enum class QualificationStatusV1 : uint32_t {
  Unassessed = 0,
  NoOpObserved = 1,
  DownstreamBlocked = 2,
  Qualified = 3,
  Rejected = 4,
};

enum class OptimizationBatchStatusV1 : uint32_t { Qualified = 0, Rejected = 1 };

enum class OptimizationComparisonKindV1 : uint32_t {
  GlobalAllOffVsAllOn = 0,
  FixedDisableOneVsAllOn = 1,
  CleanupDisableOneVsAllOn = 2,
};

enum class StaticComparisonResultV1 : uint32_t {
  BStrictlyBetter = 0,
  BEqual = 1,
  BRegressed = 2,
  IncomparableUnknown = 3,
};

enum class GateStatusV1 : uint32_t {
  Passed = 0,
  Failed = 1,
  Skipped = 2,
  Unsupported = 3,
  Timeout = 4,
  Cancelled = 5,
};

enum class ProcessStatusV1 : uint32_t { Success = 0 };

enum class OptimizationGroupSelectionV1 : uint32_t {
  AllOn = 0,
  AllOff = 1,
  DisableOne = 2,
};

struct OutcomeCountV1 {
  InvocationOutcome outcome = InvocationOutcome::Invalid;
  std::optional<ClosedReasonV1> reason;
  uint64_t count = 0;
};

struct InvocationEvidenceV1 {
  RegistryRefV1 invocationSite;
  OptimizationCutPoint cutPoint = OptimizationCutPoint::FrontendProgramImport;
  QualificationCaseKeyV1 caseKey;
  uint64_t invocationCount = 0;
  uint64_t rewriteCount = 0;
  std::vector<BackendActionEvidenceV1> backendActions;
};

struct OptimizationConfigurationV1 {
  OptimizationGroupSelectionV1 fixedSelection =
      OptimizationGroupSelectionV1::AllOn;
  std::optional<MechanismKey> fixedDisabledKey;
  OptimizationGroupSelectionV1 cleanupSelection =
      OptimizationGroupSelectionV1::AllOn;
  std::optional<MechanismKey> cleanupDisabledKey;
};

struct OptimizationComparisonKeyV1 {
  QualificationCaseKeyV1 caseKey;
  OptimizationComparisonKindV1 comparisonKind =
      OptimizationComparisonKindV1::GlobalAllOffVsAllOn;
  std::optional<MechanismKey> disabledMechanismKey;
  OptimizationConfigurationV1 configurationA;
  OptimizationConfigurationV1 configurationB;
};

struct GateEvidenceV1 {
  RegistryRefV1 gate;
  QualificationCaseKeyV1 caseKey;
  OptimizationConfigurationV1 configuration;
  GateStatusV1 status = GateStatusV1::Failed;
  AdoptionDigest evidenceDigest{};
  std::optional<ClosedReasonV1> terminalReason;
};

struct GateEvidenceBundleV1 {
  std::vector<GateEvidenceV1> orderedGateResults;
};

struct MetricEvidenceV1 {
  uint32_t metricId = 0;
  std::optional<uint64_t> value;
};

struct ExactStaticVectorEvidenceV1 {
  uint16_t registrySchema = 0;
  AdoptionDigest registryDigest{};
  std::vector<MetricEvidenceV1> orderedMetrics;
};

struct StaticComparisonEvidenceV1 {
  AdoptionDigest proposalDigest{};
  OptimizationComparisonKeyV1 comparisonKey;
  ExactStaticVectorEvidenceV1 vectorA;
  ExactStaticVectorEvidenceV1 vectorB;
  StaticComparisonResultV1 comparisonResult =
      StaticComparisonResultV1::IncomparableUnknown;
};

struct ABBASampleV1 {
  AdoptionDigest proposalDigest{};
  OptimizationComparisonKeyV1 comparisonKey;
  uint32_t blockIndex = 0;
  ABBASamplePosition sequencePosition = ABBASamplePosition::ALeft;
  uint64_t wallNs = 0;
  uint64_t peakRssBytes = 0;
  ProcessStatusV1 processStatus = ProcessStatusV1::Success;
};

struct QualificationObservationV1 {
  uint16_t schemaVersion = 1;
  MechanismKey mechanismKey;
  AdoptionDigest specDigest{};
  QualificationIdentityV1 qualificationIdentity;
  QualificationPolicyRefV1 qualificationPolicy;
  std::vector<OutcomeCountV1> outcomeCounts;
  std::vector<InvocationEvidenceV1> invocationEvidence;
  GateEvidenceBundleV1 mechanismGateResults;
  WorkSummaryV1 compileWorkSummary;
  std::vector<StaticComparisonEvidenceV1> staticComparisons;
  std::vector<ABBASampleV1> abbaSamples;
  QualificationStatusV1 qualificationStatus = QualificationStatusV1::Unassessed;
  std::optional<ClosedReasonV1> closedReason;
  std::optional<AdoptionDigest> optimizationProposalDigest;
  AdoptionDigest qualificationRunDigest{};
  std::optional<AdoptionDigest> optimizationPublicationAttemptDigest;
};

struct OptimizationBatchObservationV1 {
  uint16_t schemaVersion = 1;
  AdoptionDigest proposalDigest{};
  QualificationIdentityV1 qualificationIdentity;
  QualificationPolicyRefV1 qualificationPolicy;
  std::vector<StaticComparisonEvidenceV1> globalStaticComparisons;
  std::vector<ABBASampleV1> globalABBASamples;
  GateEvidenceBundleV1 equivalentIR2x2GateResults;
  GateEvidenceBundleV1 productionAllOnGateResults;
  OptimizationBatchStatusV1 batchStatus = OptimizationBatchStatusV1::Rejected;
  std::optional<ClosedReasonV1> closedReason;
  AdoptionDigest qualificationRunDigest{};
  AdoptionDigest optimizationPublicationAttemptDigest{};
};

RegistryRefV1 getCurrentEquivalentIRGateRefV1();
RegistryRefV1 getCurrentProductionAllOnGateRefV1();

std::vector<uint8_t>
encodeQualificationObservationV1(const QualificationObservationV1 &value);
AdoptionDigest
digestQualificationObservationV1(const QualificationObservationV1 &value);
bool decodeCanonicalQualificationObservationV1(
    const std::vector<uint8_t> &bytes, QualificationObservationV1 &value,
    std::string *diagnostic = nullptr);
bool validateQualificationObservationV1(const QualificationObservationV1 &value,
                                        std::string *diagnostic = nullptr);

std::vector<uint8_t> encodeOptimizationBatchObservationV1(
    const OptimizationBatchObservationV1 &value);
AdoptionDigest digestOptimizationBatchObservationV1(
    const OptimizationBatchObservationV1 &value);
bool decodeCanonicalOptimizationBatchObservationV1(
    const std::vector<uint8_t> &bytes, OptimizationBatchObservationV1 &value,
    std::string *diagnostic = nullptr);
bool validateOptimizationBatchObservationV1(
    const OptimizationBatchObservationV1 &value,
    std::string *diagnostic = nullptr);

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONEVIDENCE_H
