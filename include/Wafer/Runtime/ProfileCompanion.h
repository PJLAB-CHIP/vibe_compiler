//===- ProfileCompanion.h - Verified profiler companion -------*- C++ -*-===//

#ifndef WAFER_RUNTIME_PROFILECOMPANION_H
#define WAFER_RUNTIME_PROFILECOMPANION_H

#include "Wafer/Runtime/PackageManifest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer::runtime {

inline constexpr uint32_t kProfileCompanionSchemaVersion = 3;
inline constexpr int64_t kProfileCompanionRankCount = 16;
inline constexpr llvm::StringLiteral kProfileSiteCorrelationBasis =
    "heuristic-target-call-signature-occurrence-v1";
inline constexpr llvm::StringLiteral kProfileCompanionActivationFileName =
    "activation.json";
inline constexpr llvm::StringLiteral kProfileCompanionPlanFileName =
    "plan.json";
inline constexpr llvm::StringLiteral kProfileCompanionVariantsFileName =
    "variants.json";
inline constexpr llvm::StringLiteral kProfileCompanionSiteMapFileName =
    "site-map.json";

enum class ProfileVariantRole {
  FinalArtifact,
};

enum class ProfileCaptureKind {
  Summary,
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

struct ProfileTargetCallSite {
  uint64_t siteId = 0;
  uint64_t targetCallOrdinal = 0;
  std::string targetCallSymbol;
  ProfileTSMEngine engine = ProfileTSMEngine::CT;
  std::string correlationKey;
  std::optional<uint64_t> functionOrdinal;
  std::optional<uint64_t> blockOrdinal;
  std::optional<uint64_t> instructionOrdinal;
};

struct ProfileRankSiteMap {
  int64_t logicalRank = -1;
  std::vector<ProfileTargetCallSite> sites;
};

struct ProfileVariantSiteMap {
  std::string variantId;
  std::vector<ProfileRankSiteMap> ranks;
};

class ProfileVariantPackage {
public:
  ProfileVariantPackage(std::string id, ProfileVariantRole role,
                        std::string packageReference,
                        std::string manifestDigest,
                        std::string packageDirectory,
                        VerifiedPackageManifest package);
  ProfileVariantPackage(ProfileVariantPackage &&) = default;
  ProfileVariantPackage &operator=(ProfileVariantPackage &&) = default;
  ProfileVariantPackage(const ProfileVariantPackage &) = delete;
  ProfileVariantPackage &operator=(const ProfileVariantPackage &) = delete;

  llvm::StringRef getId() const { return id; }
  ProfileVariantRole getRole() const { return role; }
  llvm::StringRef getPackageReference() const { return packageReference; }
  llvm::StringRef getManifestDigest() const { return manifestDigest; }
  llvm::StringRef getPackageDirectory() const { return packageDirectory; }
  const VerifiedPackageManifest &getPackage() const { return package; }

private:
  std::string id;
  ProfileVariantRole role;
  std::string packageReference;
  std::string manifestDigest;
  std::string packageDirectory;
  VerifiedPackageManifest package;
};

class ProfileCapturePackage {
public:
  ProfileCapturePackage(std::string variantId, ProfileCaptureKind capture,
                        std::string packageReference,
                        std::string manifestDigest, uint64_t recordBytes,
                        std::string packageDirectory,
                        VerifiedPackageManifest package);
  ProfileCapturePackage(ProfileCapturePackage &&) = default;
  ProfileCapturePackage &operator=(ProfileCapturePackage &&) = default;
  ProfileCapturePackage(const ProfileCapturePackage &) = delete;
  ProfileCapturePackage &operator=(const ProfileCapturePackage &) = delete;

  llvm::StringRef getVariantId() const { return variantId; }
  ProfileCaptureKind getCaptureKind() const { return capture; }
  llvm::StringRef getPackageReference() const { return packageReference; }
  llvm::StringRef getManifestDigest() const { return manifestDigest; }
  uint64_t getRecordBytes() const { return recordBytes; }
  llvm::StringRef getPackageDirectory() const { return packageDirectory; }
  const VerifiedPackageManifest &getPackage() const { return package; }

private:
  std::string variantId;
  ProfileCaptureKind capture;
  std::string packageReference;
  std::string manifestDigest;
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
  int64_t getRankCount() const { return rankCount; }
  llvm::StringRef getRoot() const { return root; }
  llvm::StringRef getProductionManifestDigest() const {
    return productionManifestDigest;
  }
  llvm::ArrayRef<ProfileVariantPackage> getVariants() const { return variants; }
  llvm::ArrayRef<ProfileCapturePackage> getCaptures() const { return captures; }
  llvm::ArrayRef<ProfileVariantSiteMap> getSiteMaps() const { return siteMaps; }

  const ProfileVariantPackage *findVariant(ProfileVariantRole role) const;
  const ProfileCapturePackage *findCapture(llvm::StringRef variantId,
                                           ProfileCaptureKind capture) const;
  const ProfileVariantSiteMap *findSiteMap(llvm::StringRef variantId) const;
  uint64_t getSiteCount() const;

private:
  friend llvm::Expected<VerifiedProfileCompanion>
  loadVerifiedProfileCompanion(llvm::StringRef, llvm::StringRef,
                               const PackageParseLimits &);

  VerifiedProfileCompanion(std::string root,
                           std::string productionManifestDigest,
                           std::vector<ProfileVariantPackage> variants,
                           std::vector<ProfileCapturePackage> captures,
                           std::vector<ProfileVariantSiteMap> siteMaps)
      : root(std::move(root)),
        productionManifestDigest(std::move(productionManifestDigest)),
        variants(std::move(variants)), captures(std::move(captures)),
        siteMaps(std::move(siteMaps)) {}

  uint32_t schemaVersion = kProfileCompanionSchemaVersion;
  int64_t rankCount = kProfileCompanionRankCount;
  std::string root;
  std::string productionManifestDigest;
  std::vector<ProfileVariantPackage> variants;
  std::vector<ProfileCapturePackage> captures;
  std::vector<ProfileVariantSiteMap> siteMaps;
};

llvm::StringRef stringifyProfileVariantRole(ProfileVariantRole role);
llvm::StringRef stringifyProfileCaptureKind(ProfileCaptureKind capture);
llvm::StringRef stringifyProfileTSMEngine(ProfileTSMEngine engine);

/// Loads and fail-closed verifies one compiler-published profiler companion.
/// `productionPackageRoot` is the ordinary package selected by the user; the
/// final-artifact reference in the companion must resolve to exactly it.
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

} // namespace wafer::runtime

#endif // WAFER_RUNTIME_PROFILECOMPANION_H
