//===- ProfileCompanion.h - Verified profiler companion -------*- C++ -*-===//

#ifndef WAFER_RUNTIME_PROFILECOMPANION_H
#define WAFER_RUNTIME_PROFILECOMPANION_H

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Runtime/PackageManifest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvm::json {
class OStream;
} // namespace llvm::json

namespace wafer {

struct TargetCallDescriptor;

namespace runtime {

inline constexpr uint32_t kProfileCompanionSchemaVersion = 9;
inline constexpr int64_t kProfileCompanionCardCount = 1;
inline constexpr int64_t kProfileCompanionTileCount = 16;
inline constexpr llvm::StringLiteral kProfileSiteCorrelationBasis =
    "typed-target-call-ordinal-ssa-identity-occurrence-v2";
inline constexpr llvm::StringLiteral kProfileRecordABI =
    WAFER_TX81_PROFILER_RECORD_ABI_V4;
inline constexpr llvm::StringLiteral kProfileStaticCostModelName =
    "tx81-static-peak-lower-bound-v1";
inline constexpr llvm::StringLiteral kProfileStaticCostModelScope =
    "complete-final-instruction-program-per-physical-tile";
inline constexpr llvm::StringLiteral kProfileCompanionActivationFileName =
    "activation.json";
inline constexpr llvm::StringLiteral kProfileCompanionPlanFileName =
    "plan.json";
inline constexpr llvm::StringLiteral kProfileCompanionSiteMapFileName =
    "site-map.json";

enum class ProfileCaptureKind {
  Count,
  Trace,
};

enum class ProfileTSMEngine {
  CT,
  NE,
  RDMA,
  WDMA,
  TDMA,
  DirectDTE,
};

enum class ProfileTargetSiteKind {
  NCCCommand,
  NCCCompletion,
  DirectDTEControl,
  DirectDTEIssue,
  DirectDTEWait,
};

/// One exact, final-IR-derived static work dimension. Known uint64 values use a
/// decimal string on the JSON wire so the complete unsigned range survives
/// round-trip through every consumer. Non-known dimensions carry null.
struct ProfileStaticCostMetric {
  std::string knowledge;
  std::optional<uint64_t> value;
  std::string reason;
};

struct ProfileStaticCostRates {
  uint64_t cardDDRBytesPerSecond = 0;
  uint64_t directionalNoCBytesPerSecond = 0;
  uint64_t f16Bf16NPULogicalOpsPerSecondPerTile = 0;
  uint64_t f16Bf16VectorLogicalOpsPerSecondPerTile = 0;
  uint64_t f32VectorLogicalOpsPerSecondPerTile = 0;
  /// The current target has no calibrated SPM rate.
  std::optional<uint64_t> spmMovementBytesPerSecond;
};

struct ProfileStaticDirectionalNoCWork {
  ProfileStaticCostMetric north;
  ProfileStaticCostMetric east;
  ProfileStaticCostMetric south;
  ProfileStaticCostMetric west;
};

struct ProfileStaticTileWork {
  ProfileStaticCostMetric npuF16Bf16LogicalOps;
  ProfileStaticCostMetric npuOtherLogicalOps;
  ProfileStaticCostMetric vectorF16Bf16LogicalOps;
  ProfileStaticCostMetric vectorF32LogicalOps;
  ProfileStaticCostMetric vectorOtherLogicalOps;
  ProfileStaticCostMetric ddrReadBytes;
  ProfileStaticCostMetric ddrWriteBytes;
  ProfileStaticCostMetric spmMovementBytes;
  ProfileStaticCostMetric nocTransmitBytes;
  ProfileStaticCostMetric nocReceiveBytes;
  ProfileStaticDirectionalNoCWork directionalNoCTransmitBytes;
};

struct ProfileStaticTileCost {
  PhysicalCardId cardId{0};
  PhysicalTileId tileId{0};
  LaunchSlotId launchSlot;
  ProfileStaticTileWork work;
};

/// Static target rates plus exact work derived from the accepted final
/// instruction program. It is evidence for lower-bound comparison, not an
/// issue-latency, overlap, contention, or wall-time prediction.
struct ProfileStaticCostModel {
  std::string model;
  std::string scope;
  ProfileStaticCostRates rates;
  std::vector<ProfileStaticTileCost> tiles;
};

struct ProfileTargetCallSite {
  uint64_t siteId = 0;
  uint64_t targetCallOrdinal = 0;
  std::string targetCallSymbol;
  ProfileTargetSiteKind siteKind = ProfileTargetSiteKind::NCCCommand;
  /// Present only for NCCCommand (one of the five NCC engines) and Direct DTE
  /// issue/wait observation sites (DirectDTE). Completion/control sites have
  /// no engine.
  std::optional<ProfileTSMEngine> engine;
  std::string correlationKey;
  std::optional<uint64_t> functionOrdinal;
  std::optional<uint64_t> blockOrdinal;
  std::optional<uint64_t> instructionOrdinal;
};

struct ProfileTileSiteMap {
  PhysicalCardId cardId{0};
  PhysicalTileId tileId{0};
  LaunchSlotId launchSlot;
  std::vector<ProfileTargetCallSite> sites;
};

class ProfileProductionArtifact {
public:
  ProfileProductionArtifact(std::string manifestDigest,
                            ProfileStaticCostModel staticCostModel,
                            std::string packageDirectory,
                            VerifiedPackageManifest package);
  ProfileProductionArtifact(ProfileProductionArtifact &&) = default;
  ProfileProductionArtifact &operator=(ProfileProductionArtifact &&) = default;
  ProfileProductionArtifact(const ProfileProductionArtifact &) = delete;
  ProfileProductionArtifact &
  operator=(const ProfileProductionArtifact &) = delete;

  llvm::StringRef getManifestDigest() const { return manifestDigest; }
  const ProfileStaticCostModel &getStaticCostModel() const {
    return staticCostModel;
  }
  llvm::StringRef getPackageDirectory() const { return packageDirectory; }
  const VerifiedPackageManifest &getPackage() const { return package; }

private:
  std::string manifestDigest;
  ProfileStaticCostModel staticCostModel;
  std::string packageDirectory;
  VerifiedPackageManifest package;
};

class ProfileCapturePackage {
public:
  ProfileCapturePackage(ProfileCaptureKind capture,
                        std::string packageReference,
                        std::string manifestDigest, std::string recordABI,
                        uint64_t recordBytes, std::string packageDirectory,
                        VerifiedPackageManifest package);
  ProfileCapturePackage(ProfileCapturePackage &&) = default;
  ProfileCapturePackage &operator=(ProfileCapturePackage &&) = default;
  ProfileCapturePackage(const ProfileCapturePackage &) = delete;
  ProfileCapturePackage &operator=(const ProfileCapturePackage &) = delete;

  ProfileCaptureKind getCaptureKind() const { return capture; }
  llvm::StringRef getPackageReference() const { return packageReference; }
  llvm::StringRef getManifestDigest() const { return manifestDigest; }
  llvm::StringRef getRecordABI() const { return recordABI; }
  uint64_t getRecordBytes() const { return recordBytes; }
  llvm::StringRef getPackageDirectory() const { return packageDirectory; }
  const VerifiedPackageManifest &getPackage() const { return package; }

private:
  ProfileCaptureKind capture;
  std::string packageReference;
  std::string manifestDigest;
  std::string recordABI;
  uint64_t recordBytes;
  std::string packageDirectory;
  VerifiedPackageManifest package;
};

class VerifiedProfileCompanion {
public:
  VerifiedProfileCompanion(VerifiedProfileCompanion &&) = default;
  VerifiedProfileCompanion &operator=(VerifiedProfileCompanion &&) = default;
  VerifiedProfileCompanion(const VerifiedProfileCompanion &) = delete;
  VerifiedProfileCompanion &
  operator=(const VerifiedProfileCompanion &) = delete;

  uint32_t getSchemaVersion() const { return schemaVersion; }
  int64_t getCardCount() const { return cardCount; }
  int64_t getTileCount() const { return tileCount; }
  llvm::StringRef getRoot() const { return root; }
  const ProfileProductionArtifact &getProductionArtifact() const {
    return productionArtifact;
  }
  llvm::ArrayRef<ProfileCapturePackage> getCaptures() const { return captures; }
  llvm::ArrayRef<ProfileTileSiteMap> getSiteMap() const { return siteMap; }

  const ProfileCapturePackage *findCapture(ProfileCaptureKind capture) const;
  uint64_t getSiteCount() const;

private:
  friend llvm::Expected<VerifiedProfileCompanion>
  loadVerifiedProfileCompanion(llvm::StringRef, llvm::StringRef,
                               const PackageParseLimits &);

  VerifiedProfileCompanion(std::string root,
                           ProfileProductionArtifact productionArtifact,
                           std::vector<ProfileCapturePackage> captures,
                           std::vector<ProfileTileSiteMap> siteMap)
      : root(std::move(root)),
        productionArtifact(std::move(productionArtifact)),
        captures(std::move(captures)), siteMap(std::move(siteMap)) {}

  uint32_t schemaVersion = kProfileCompanionSchemaVersion;
  int64_t cardCount = kProfileCompanionCardCount;
  int64_t tileCount = kProfileCompanionTileCount;
  std::string root;
  ProfileProductionArtifact productionArtifact;
  std::vector<ProfileCapturePackage> captures;
  std::vector<ProfileTileSiteMap> siteMap;
};

llvm::StringRef stringifyProfileCaptureKind(ProfileCaptureKind capture);
llvm::StringRef stringifyProfileTSMEngine(ProfileTSMEngine engine);
llvm::StringRef stringifyProfileTargetSiteKind(ProfileTargetSiteKind kind);

/// Emits the canonical strict JSON representation used in the profile plan
/// and top-level profile evidence.
void writeProfileStaticCostModel(llvm::json::OStream &json,
                                 const ProfileStaticCostModel &model);

/// Returns the profiler site semantic owned by one closed target-call
/// descriptor. Every descriptor in the public target-call registry maps to
/// exactly one site kind; this function never recovers semantics from symbols.
ProfileTargetSiteKind
getProfileTargetSiteKind(const TargetCallDescriptor &descriptor);

/// Loads and fail-closed verifies one compiler-published profiler companion.
/// `productionPackageRoot` is the ordinary package selected by the user; the
/// companion production digest must resolve to exactly it.
llvm::Expected<VerifiedProfileCompanion>
loadVerifiedProfileCompanion(llvm::StringRef companionRoot,
                             llvm::StringRef productionPackageRoot,
                             const PackageParseLimits &limits = {});

/// Looks for the compiler-owned sibling `<productionPackageRoot>.profile`.
/// Absence is not an error. Once the sibling exists, every malformed or stale
/// artifact is an error rather than a fallback to unprofiled execution.
llvm::Expected<std::optional<VerifiedProfileCompanion>>
loadSiblingProfileCompanionIfPresent(llvm::StringRef productionPackageRoot,
                                     const PackageParseLimits &limits = {});

} // namespace runtime
} // namespace wafer

#endif // WAFER_RUNTIME_PROFILECOMPANION_H
