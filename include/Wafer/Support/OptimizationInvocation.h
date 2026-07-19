//===- OptimizationInvocation.h - Canonical invocation records -*- C++ -*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONINVOCATION_H
#define WAFER_SUPPORT_OPTIMIZATIONINVOCATION_H

#include "Wafer/Support/OptimizationAdoption.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace wafer {

struct RegistryRefV1 {
  uint32_t id = 0;
  uint16_t registrySchema = 0;
  AdoptionDigest registryDigest{};
};

/// Closed invocation-site registry row for one live mechanism adapter. The
/// registry reference is independent of the mechanism key even when the
/// current revision has one site per semantic mechanism.
std::optional<RegistryRefV1>
lookupOptimizationInvocationSiteV1(MechanismKey mechanismKey);
std::vector<std::string> auditOptimizationInvocationSiteRegistryV1();

enum class GlobalClosedReasonV1 : uint32_t {
  UnsupportedSemantic = 1,
  ResourceExhausted = 2,
  InvalidOwnerTerminal = 3,
  Cancelled = 4,
  NoDeterministicBenefit = 5,
  DownstreamGateBlocked = 6,
  QualificationEvidenceRejected = 7,
  HostEnvironmentInvalidated = 8,
  PublicationConflict = 9,
  PublicationFailure = 10,
};

RegistryRefV1 getGlobalClosedReasonRefV1(GlobalClosedReasonV1 reason);
bool isGlobalClosedReasonRefV1(const RegistryRefV1 &ref,
                               GlobalClosedReasonV1 reason);

enum class EquivalentInputVariantV1 : uint32_t {
  Original = 0,
  Metamorphic = 1,
};

struct QualificationCaseKeyV1 {
  RegistryRefV1 corpus;
  uint32_t rankCount = 0;
  EquivalentInputVariantV1 inputVariant = EquivalentInputVariantV1::Original;
};

struct QualificationIdentityV1 {
  AdoptionDigest build{};
  AdoptionDigest toolchain{};
  AdoptionDigest hostKernelAffinityGovernor{};
  AdoptionDigest corpus{};
  AdoptionDigest featureConfig{};
};

struct QualificationPolicyRefV1 {
  uint32_t policyId = 0;
  uint16_t policySchema = 0;
  AdoptionDigest canonicalPolicyDigest{};
};

struct ClosedReasonV1 {
  RegistryRefV1 reason;
  std::optional<AdoptionDigest> detailDigest;
};

struct WorkCounterV1 {
  uint32_t counterId = 0;
  uint64_t value = 0;
};

struct WorkSummaryV1 {
  AdoptionDigest workPolicyDigest{};
  std::vector<WorkCounterV1> orderedCounters;
};

enum class BackendActionStatusV1 : uint32_t {
  Success = 0,
  Failed = 1,
  Timeout = 2,
  Cancelled = 3,
};

struct BackendActionEvidenceV1 {
  AdoptionDigest invocationId{};
  uint32_t actionOrdinal = 0;
  std::vector<std::string> argv;
  AdoptionDigest observedToolDigest{};
  AdoptionDigest observedOutputDigest{};
  BackendActionStatusV1 terminalStatus = BackendActionStatusV1::Failed;
};

enum class InvocationScopeKindV1 : uint32_t {
  ProductionCompile = 0,
  DebugReplay = 1,
  QualificationRun = 2,
};

struct InvocationIdentityV1 {
  InvocationScopeKindV1 scopeKind = InvocationScopeKindV1::ProductionCompile;
  AdoptionDigest scopeDigest{};
  MechanismKey mechanismKey;
  RegistryRefV1 invocationSite;
  OptimizationCutPoint cutPoint = OptimizationCutPoint::FrontendProgramImport;
  uint64_t invocationOrdinal = 0;
};

struct InvocationPreparationV1 {
  InvocationIdentityV1 identity;
  std::optional<QualificationCaseKeyV1> qualificationCase;
  AdoptionDigest specDigest{};
  AdoptionDigest inputSnapshotDigest{};
};

struct OptimizationInvocationScopeContextV1 {
  InvocationScopeKindV1 scopeKind = InvocationScopeKindV1::ProductionCompile;
  AdoptionDigest scopeDigest{};
  std::optional<QualificationCaseKeyV1> qualificationCase;
  /// Checked high-order namespace for one process execution within a
  /// qualification run. Owner adapters emit only local ordinals below the
  /// fixed limit; the gateway composes the globally unique terminal ordinal.
  uint64_t invocationOrdinalBase = 0;
};

inline constexpr uint64_t kOptimizationInvocationLocalOrdinalLimit = uint64_t{1}
                                                                     << 40;

struct InvocationTelemetryV1 {
  uint16_t schemaVersion = 1;
  InvocationIdentityV1 identity;
  std::optional<QualificationCaseKeyV1> qualificationCase;
  AdoptionDigest specDigest{};
  AdoptionDigest inputSnapshotDigest{};
  InvocationOutcome outcome = InvocationOutcome::Invalid;
  std::optional<ClosedReasonV1> terminalReason;
  uint64_t rewriteCount = 0;
  WorkSummaryV1 workSummary;
  std::vector<BackendActionEvidenceV1> backendActions;
};

std::vector<uint8_t>
encodeInvocationIdentityV1(const InvocationIdentityV1 &identity);
AdoptionDigest digestInvocationIdentityV1(const InvocationIdentityV1 &identity);

std::vector<uint8_t>
encodeInvocationTelemetryV1(const InvocationTelemetryV1 &telemetry);
AdoptionDigest
digestInvocationTelemetryV1(const InvocationTelemetryV1 &telemetry);

/// Parses, validates, and re-encodes exact canonical terminal bytes. Unknown,
/// missing, reordered, mistyped, duplicate, or noncanonical fields fail.
bool decodeCanonicalInvocationTelemetryV1(const std::vector<uint8_t> &bytes,
                                          InvocationTelemetryV1 &telemetry,
                                          std::string *diagnostic = nullptr);

bool validateInvocationPreparationV1(const InvocationPreparationV1 &preparation,
                                     std::string *diagnostic = nullptr);
bool validateInvocationTelemetryV1(const InvocationTelemetryV1 &telemetry,
                                   std::string *diagnostic = nullptr);

/// In-memory begin/terminal state machine used by an archive transaction.
/// Begin entries are deliberately not externally visible. A scope can be
/// sealed only after every prepared identity has one terminal; after sealing,
/// all begin/commit attempts fail closed.
class OptimizationInvocationJournalV1 final
    : public OptimizationInvocationRecorder {
public:
  bool beginInvocation(const InvocationPreparationV1 &preparation,
                       std::string *diagnostic = nullptr) override;
  bool commitInvocationTerminal(const InvocationTelemetryV1 &telemetry,
                                std::string *diagnostic = nullptr);
  bool recordInvocationTerminal(const InvocationTelemetryV1 &telemetry,
                                std::string *diagnostic = nullptr) override {
    return commitInvocationTerminal(telemetry, diagnostic);
  }
  bool sealScope(InvocationScopeKindV1 scopeKind,
                 const AdoptionDigest &scopeDigest,
                 std::string *diagnostic = nullptr);

  std::vector<InvocationTelemetryV1> committedTerminals() const;

private:
  struct Entry {
    InvocationPreparationV1 preparation;
    std::optional<InvocationTelemetryV1> terminal;
  };
  struct DigestLess {
    bool operator()(const AdoptionDigest &lhs,
                    const AdoptionDigest &rhs) const {
      return lhs < rhs;
    }
  };

  std::map<AdoptionDigest, Entry, DigestLess> entries_;
  std::map<AdoptionDigest, InvocationScopeKindV1, DigestLess> sealedScopes_;
  mutable std::mutex mutex_;
};

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONINVOCATION_H
