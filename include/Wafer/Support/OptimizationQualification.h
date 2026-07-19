//===- OptimizationQualification.h - Exact optimization qualification -*- C++
//-*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONQUALIFICATION_H
#define WAFER_SUPPORT_OPTIMIZATIONQUALIFICATION_H

#include "Wafer/Support/OptimizationAdoption.h"
#include "Wafer/Support/OptimizationInvocation.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer {

struct MechanismSpecBinding {
  MechanismKey mechanismKey;
  AdoptionDigest specDigest{};
};

struct MechanismObservationBindingV1 {
  MechanismKey mechanismKey;
  AdoptionDigest observationDigest{};
};

/// Frozen candidate set for one fixed-optimization qualification run.  The two
/// groups are disjoint; an empty union is the canonical result when
/// exploratory evidence closes every optional mechanism before publication.
/// A binding is valid only when its immutable spec digest and AdoptionMode
/// match its group.
struct OptimizationQualificationProposal {
  uint16_t schemaVersion = 1;
  std::vector<MechanismSpecBinding> fixedBindings;
  std::vector<MechanismSpecBinding> cleanupBindings;
};

enum class OptimizationGroupSelectionKind : uint8_t {
  AllOn,
  AllOff,
  DisableOne,
};

struct OptimizationGroupSelection {
  OptimizationGroupSelectionKind kind = OptimizationGroupSelectionKind::AllOn;
  std::optional<MechanismKey> disabledKey;
};

struct OptimizationConfiguration {
  OptimizationGroupSelection fixed;
  OptimizationGroupSelection cleanup;
};

/// Constructs the immutable current proposal from the typed adoption registry.
/// It snapshots exact spec digests rather than interpreting future registry
/// additions as members of an already-running AllOn configuration.
OptimizationQualificationProposal getCurrentOptimizationQualificationProposal();

std::vector<uint8_t> encodeOptimizationQualificationProposalV1(
    const OptimizationQualificationProposal &proposal);
AdoptionDigest digestOptimizationQualificationProposalV1(
    const OptimizationQualificationProposal &proposal);
bool validateCanonicalOptimizationQualificationProposalV1(
    const std::vector<uint8_t> &bytes, std::string *diagnostic = nullptr);

/// Validates canonical binding order, group/mode membership, exact immutable
/// spec digests, and disjointness.
bool validateOptimizationQualificationProposal(
    const OptimizationQualificationProposal &proposal,
    std::string *diagnostic = nullptr);

/// Immutable, all-and-only qualification result for one frozen proposal.
/// The active reference points at the digest of this record; production never
/// infers membership from the mutable mechanism registry.
struct QualifiedOptimizationSetV1 {
  uint16_t schemaVersion = 1;
  AdoptionDigest proposalDigest{};
  AdoptionDigest batchObservationDigest{};
  std::vector<MechanismObservationBindingV1> observationBindings;
};

std::vector<uint8_t> encodeQualifiedOptimizationSetV1(
    const QualifiedOptimizationSetV1 &qualifiedSet);
AdoptionDigest digestQualifiedOptimizationSetV1(
    const QualifiedOptimizationSetV1 &qualifiedSet);
bool decodeCanonicalQualifiedOptimizationSetV1(
    const std::vector<uint8_t> &bytes, QualifiedOptimizationSetV1 &qualifiedSet,
    std::string *diagnostic = nullptr);
bool validateQualifiedOptimizationSetV1(
    const QualifiedOptimizationSetV1 &qualifiedSet,
    std::string *diagnostic = nullptr);

struct OptimizationSetQualificationPolicyV1 {
  uint16_t schemaVersion = 1;
  RegistryRefV1 mandatoryCorpusRegistry;
  std::vector<uint32_t> mandatoryRankCounts;
  RegistryRefV1 staticMetricRegistry;
  uint32_t abbaBlockCount = 5;
  uint32_t wallGuardFractionNumerator = 1;
  uint32_t wallGuardFractionDenominator = 20;
  uint32_t madMultiplier = 3;
  uint32_t hostEnvironmentRetryCap = 0;
  uint64_t totalProcessLaunchCap = 0;
  uint64_t totalGatewayInvocationCap = 0;
};

RegistryRefV1 getCurrentMandatoryCorpusRegistryRefV1();
std::vector<RegistryRefV1> getCurrentMandatoryCorpusRowsV1();
/// Returns the explicit original-input case domain. Corpus and rank are
/// paired here; callers must not form their Cartesian product because some
/// frozen corpora have a fixed execution mesh.
std::vector<QualificationCaseKeyV1> getCurrentMandatoryQualificationCasesV1();
RegistryRefV1 getCurrentStaticMetricRegistryRefV1();
OptimizationSetQualificationPolicyV1
getCurrentOptimizationSetQualificationPolicyV1();
QualificationPolicyRefV1 getCurrentOptimizationSetQualificationPolicyRefV1();

std::vector<uint8_t> encodeOptimizationSetQualificationPolicyV1(
    const OptimizationSetQualificationPolicyV1 &policy);
AdoptionDigest digestOptimizationSetQualificationPolicyV1(
    const OptimizationSetQualificationPolicyV1 &policy);
bool validateOptimizationSetQualificationPolicyV1(
    const OptimizationSetQualificationPolicyV1 &policy,
    std::string *diagnostic = nullptr);
bool decodeCanonicalOptimizationSetQualificationPolicyV1(
    const std::vector<uint8_t> &bytes,
    OptimizationSetQualificationPolicyV1 &policy,
    std::string *diagnostic = nullptr);

OptimizationConfiguration getAllOnOptimizationConfiguration();
OptimizationConfiguration getAllOffOptimizationConfiguration();
OptimizationConfiguration getDisableOneOptimizationConfiguration(
    const OptimizationQualificationProposal &proposal, MechanismKey key,
    std::string *diagnostic = nullptr);

/// Validates that DisableOne carries exactly one own-group proposal member and
/// that AllOn/AllOff carry no key.  Required normalization is outside this
/// selection and therefore cannot be disabled through a configuration.
bool validateOptimizationConfiguration(
    const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration,
    std::string *diagnostic = nullptr);

/// Returns whether one fixed/cleanup proposal member is selected by the
/// configuration. Non-proposal and non-optimization keys fail closed instead of
/// being guessed from pass names or registration order.
std::optional<bool> isOptimizationMechanismEnabled(
    const OptimizationQualificationProposal &proposal,
    const OptimizationConfiguration &configuration, MechanismKey key,
    std::string *diagnostic = nullptr);

/// Canonical ABBA positions.  Their ordinal order is part of the evidence
/// validator: every block is exactly A-left, B-left, B-right, A-right.
enum class ABBASamplePosition : uint8_t {
  ALeft = 0,
  BLeft = 1,
  BRight = 2,
  ARight = 3,
};

struct ABBASample {
  uint32_t blockIndex = 0;
  ABBASamplePosition position = ABBASamplePosition::ALeft;
  uint64_t wallNs = 0;
  uint64_t peakRssBytes = 0;
  bool processSucceeded = false;
};

/// A canonical reduced exact rational.  Decimal strings avoid exposing the
/// implementation bigint while preserving arbitrary precision in records and
/// diagnostics.  Zero is always "0/1" and the denominator is positive.
struct ExactRationalValue {
  std::string signedNumerator;
  std::string positiveDenominator;

  friend bool operator==(const ExactRationalValue &lhs,
                         const ExactRationalValue &rhs) {
    return lhs.signedNumerator == rhs.signedNumerator &&
           lhs.positiveDenominator == rhs.positiveDenominator;
  }
};

struct ABBAHostMetricDecision {
  ExactRationalValue baselineMedian;
  ExactRationalValue deltaMedian;
  ExactRationalValue medianAbsoluteDeviation;
  ExactRationalValue guard;
  bool nonRegressed = false;
  bool significantImprovement = false;
};

struct OptimizationSetHostDecision {
  ABBAHostMetricDecision wall;
  ABBAHostMetricDecision peakRss;
  bool accepted = false;
  bool significantHostBenefit = false;
};

struct OptimizationSetHostEvaluation {
  std::optional<OptimizationSetHostDecision> decision;
  std::string diagnostic;
};

/// Implements OptimizationSetQualificationPolicyV1's frozen host guard:
/// five ordered ABBA blocks, guard=max(m_A/20, 3*MAD), independently for wall
/// and peak RSS.  The implementation uses reduced arbitrary-precision signed
/// rationals throughout; floating point and rounded integer medians never
/// participate in the decision.
OptimizationSetHostEvaluation
evaluateOptimizationSetABBA(const std::vector<ABBASample> &samples);

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONQUALIFICATION_H
