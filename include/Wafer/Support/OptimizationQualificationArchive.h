//===- OptimizationQualificationArchive.h - Run/publication records -*- C++
//-*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONARCHIVE_H
#define WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONARCHIVE_H

#include "Wafer/Support/OptimizationQualificationEvidence.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer {

struct QualificationInputCaseV1 {
  QualificationCaseKeyV1 caseKey;
  AdoptionDigest inputSnapshotDigest{};
};

struct InvocationTerminalBindingV1 {
  AdoptionDigest invocationId{};
  AdoptionDigest terminalDigest{};
};

struct AdoptionQualificationInputV1 {
  uint16_t schemaVersion = 1;
  std::vector<MechanismSpecBinding> specBindings;
  std::vector<QualificationInputCaseV1> qualificationCases;
  std::optional<AdoptionDigest> optimizationProposalDigest;
};

struct AdoptionQualificationRunV1 {
  uint16_t schemaVersion = 1;
  AdoptionDigest qualificationInputDigest{};
  QualificationIdentityV1 qualificationIdentity;
  QualificationPolicyRefV1 qualificationPolicy;
  uint64_t runSeriesOrdinal = 0;
  uint32_t attemptOrdinal = 0;
};

enum class AdoptionQualificationRunOutcomeV1 : uint32_t {
  CompletedEvidence = 0,
  Cancelled = 1,
  HostEnvironmentInvalidated = 2,
  ResourceExhausted = 3,
  Invalid = 4,
};

struct AdoptionQualificationRunTerminalV1 {
  uint16_t schemaVersion = 1;
  AdoptionDigest qualificationRunDigest{};
  AdoptionQualificationRunOutcomeV1 outcome =
      AdoptionQualificationRunOutcomeV1::Invalid;
  std::optional<AdoptionDigest> resultManifestDigest;
  std::optional<ClosedReasonV1> closedReason;
};

struct AdoptionQualificationResultManifestV1 {
  uint16_t schemaVersion = 1;
  AdoptionDigest qualificationRunDigest{};
  std::vector<InvocationTerminalBindingV1> invocationTerminals;
  std::vector<MechanismObservationBindingV1> observationBindings;
  std::optional<AdoptionDigest> optimizationBatchObservationDigest;
};

struct OptimizationSetPublicationAttemptV1 {
  uint16_t schemaVersion = 1;
  AdoptionDigest qualificationRunDigest{};
  AdoptionDigest proposalDigest{};
  std::optional<AdoptionDigest> expectedActiveRefDigest;
};

enum class OptimizationSetPublicationOutcomeV1 : uint32_t {
  QualifiedPublished = 0,
  RejectedEvidence = 1,
  PublicationConflict = 2,
  PublicationFailed = 3,
};

struct OptimizationSetPublicationTerminalV1 {
  uint16_t schemaVersion = 1;
  AdoptionDigest optimizationPublicationAttemptDigest{};
  OptimizationSetPublicationOutcomeV1 outcome =
      OptimizationSetPublicationOutcomeV1::PublicationFailed;
  std::optional<AdoptionDigest> batchObservationDigest;
  std::optional<AdoptionDigest> candidateSetDigest;
  std::optional<ClosedReasonV1> closedReason;
};

struct ActiveQualifiedOptimizationSetRefV1 {
  uint16_t schemaVersion = 1;
  uint64_t generation = 0;
  std::optional<AdoptionDigest> parentSetDigest;
  AdoptionDigest setDigest{};
  AdoptionDigest qualificationRunDigest{};
  AdoptionDigest adoptionQualificationRunTerminalDigest{};
  AdoptionDigest optimizationPublicationAttemptDigest{};
  AdoptionDigest optimizationPublicationTerminalDigest{};
};

#define WAFER_DECLARE_QUALIFICATION_ARCHIVE_CODEC(Type)                        \
  std::vector<uint8_t> encode##Type(const Type &value);                        \
  AdoptionDigest digest##Type(const Type &value);                              \
  bool decodeCanonical##Type(const std::vector<uint8_t> &bytes, Type &value,   \
                             std::string *diagnostic = nullptr)

WAFER_DECLARE_QUALIFICATION_ARCHIVE_CODEC(AdoptionQualificationInputV1);
WAFER_DECLARE_QUALIFICATION_ARCHIVE_CODEC(AdoptionQualificationRunV1);
WAFER_DECLARE_QUALIFICATION_ARCHIVE_CODEC(AdoptionQualificationRunTerminalV1);
WAFER_DECLARE_QUALIFICATION_ARCHIVE_CODEC(
    AdoptionQualificationResultManifestV1);
WAFER_DECLARE_QUALIFICATION_ARCHIVE_CODEC(OptimizationSetPublicationAttemptV1);
WAFER_DECLARE_QUALIFICATION_ARCHIVE_CODEC(OptimizationSetPublicationTerminalV1);
WAFER_DECLARE_QUALIFICATION_ARCHIVE_CODEC(ActiveQualifiedOptimizationSetRefV1);

#undef WAFER_DECLARE_QUALIFICATION_ARCHIVE_CODEC

/// Complete in-memory evidence transaction for a successful qualification
/// run. Validation re-derives every manifest digest and invocation aggregate;
/// the records are not trusted merely because each one parses in isolation.
struct CompletedQualificationEvidenceV1 {
  AdoptionQualificationInputV1 input;
  AdoptionQualificationRunV1 run;
  std::optional<OptimizationSetPublicationAttemptV1> publicationAttempt;
  std::vector<InvocationTelemetryV1> invocationTerminals;
  std::vector<QualificationObservationV1> observations;
  std::optional<OptimizationBatchObservationV1> optimizationBatch;
  AdoptionQualificationResultManifestV1 resultManifest;
  AdoptionQualificationRunTerminalV1 runTerminal;
};

bool validateCompletedQualificationEvidenceV1(
    const CompletedQualificationEvidenceV1 &value,
    std::string *diagnostic = nullptr);

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONQUALIFICATIONARCHIVE_H
