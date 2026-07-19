//===- OptimizationAdoption.h - Canonical adoption inventory -*- C++ -*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONADOPTION_H
#define WAFER_SUPPORT_OPTIMIZATIONADOPTION_H

#include "Wafer/Support/OptimizationMechanism.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer {

using AdoptionDigest = OptimizationDigest;

enum class ProviderOrigin : uint32_t {
  Internal,
  Upstream,
  Vendored,
  Toolchain
};
enum class Availability : uint32_t { Absent, PresentUnresolved, Resolved };
enum class Exposure : uint32_t {
  DebugRegistered,
  LibraryCallable,
  BackendExecutable,
};
enum class AdoptionMode : uint32_t {
  None,
  RequiredNormalization,
  FixedOptimization,
  BestEffortCleanup,
  CandidateLocal,
  TargetSpecific,
  TargetBackend,
};
enum class OperationDomainKind : uint8_t { IR, ArtifactAction };

/// Closed ordinal sets.  Their elements are stable compiler registry IDs, not
/// printer spellings.  IR domains use semanticFamilies/requiredInterfaces/
/// allowedDialects; action domains use those same storage slots as executor,
/// action and input-artifact sets, plus outputArtifacts.
struct OperationDomain {
  OperationDomainKind kind = OperationDomainKind::IR;
  std::vector<uint32_t> semanticFamilies;
  std::vector<uint32_t> requiredInterfaces;
  std::vector<uint32_t> allowedDialects;
  std::vector<uint32_t> outputArtifacts;
};

struct AdoptionSpec {
  uint16_t schemaVersion = 1;
  MechanismKey mechanismKey;
  ProviderOrigin providerOrigin = ProviderOrigin::Internal;
  std::string providerId;
  std::string providerRevision;
  std::optional<AdoptionDigest> providerContentDigest;
  OptimizationCutPoint cutPoint = OptimizationCutPoint::FrontendProgramImport;
  OperationDomain operationDomain;
  Availability availability = Availability::Absent;
  std::vector<Exposure> exposureSet;
  AdoptionMode adoptionMode = AdoptionMode::None;
  uint32_t preconditionContractId = 0;
  uint32_t numericContractId = 0;
  uint32_t effectContractId = 0;
  uint32_t workPolicyKind = 0;
  uint32_t ownerId = 0;
  InvocationEvidenceKind evidenceKind = InvocationEvidenceKind::Rewrite;
};

std::vector<AdoptionSpec> getAllAdoptionSpecs();
std::optional<AdoptionSpec> lookupAdoptionSpec(MechanismKey key);

/// Exact §6.2 owner encodings and SHA-256 digest.
std::vector<uint8_t> encodeMechanismKeyV1(MechanismKey key);
std::vector<uint8_t> encodeOperationDomainV1(const OperationDomain &domain);
std::vector<uint8_t> encodeAdoptionSpecV1(const AdoptionSpec &spec);
AdoptionDigest digestAdoptionSpecV1(const AdoptionSpec &spec);

/// Canonical definition digest embedded in WorkPolicyRefV1 for one closed
/// policy kind. This is also the exact digest carried by invocation work
/// summaries; callers must not synthesize a second policy identity.
AdoptionDigest digestAdoptionWorkPolicyV1(uint32_t workPolicyKind);

/// Accepts only the all-and-only canonical bytes of the immutable registry
/// row.  Unknown/missing fields, wrong type tags, noncanonical set order,
/// descriptor/spec cut or evidence drift, and digest-changing mutations fail.
bool validateCanonicalAdoptionSpecV1(const std::vector<uint8_t> &bytes,
                                     std::string *diagnostic = nullptr);

/// Bidirectional inventory shape: one immutable spec for every descriptor and
/// exact cut/evidence agreement.  Live/qualified/rejected closure belongs to
/// invocation observations and is deliberately not copied into AdoptionSpec.
/// Returns sorted diagnostics.
std::vector<std::string> auditOptimizationAdoptionInventory();

std::string toHex(const AdoptionDigest &digest);

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONADOPTION_H
