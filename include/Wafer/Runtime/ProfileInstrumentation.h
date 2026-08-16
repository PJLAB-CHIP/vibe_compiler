//===- ProfileInstrumentation.h - Verified profiler instrumentation -------*-
// C++ -*-===//

#ifndef WAFER_RUNTIME_PROFILEINSTRUMENTATION_H
#define WAFER_RUNTIME_PROFILEINSTRUMENTATION_H

#include "Wafer/Package/PackageManifest.h"
#include "Wafer/Package/ProfileInstrumentationModel.h"

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

class ProfiledPackage {
public:
  ProfiledPackage(std::string manifestDigest,
                  ProfileStaticCostModel staticCostModel,
                  std::string packageDirectory,
                  VerifiedPackageManifest package);
  ProfiledPackage(ProfiledPackage &&) = default;
  ProfiledPackage &operator=(ProfiledPackage &&) = default;
  ProfiledPackage(const ProfiledPackage &) = delete;
  ProfiledPackage &operator=(const ProfiledPackage &) = delete;

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

class VerifiedProfileInstrumentation {
public:
  VerifiedProfileInstrumentation(VerifiedProfileInstrumentation &&) = default;
  VerifiedProfileInstrumentation &
  operator=(VerifiedProfileInstrumentation &&) = default;
  VerifiedProfileInstrumentation(const VerifiedProfileInstrumentation &) =
      delete;
  VerifiedProfileInstrumentation &
  operator=(const VerifiedProfileInstrumentation &) = delete;

  int64_t getCardCount() const { return cardCount; }
  int64_t getTileCount() const { return tileCount; }
  llvm::StringRef getRoot() const { return root; }
  const ProfiledPackage &getProfiledPackage() const { return profiledPackage; }
  llvm::ArrayRef<ProfileCapturePackage> getCaptures() const { return captures; }
  llvm::ArrayRef<ProfileTileSiteMap> getSiteMap() const { return siteMap; }

  const ProfileCapturePackage *findCapture(ProfileCaptureKind capture) const;
  uint64_t getSiteCount() const;

private:
  friend llvm::Expected<VerifiedProfileInstrumentation>
  loadVerifiedProfileInstrumentation(llvm::StringRef, llvm::StringRef,
                                     const PackageParseLimits &);

  VerifiedProfileInstrumentation(std::string root,
                                 ProfiledPackage profiledPackage,
                                 std::vector<ProfileCapturePackage> captures,
                                 std::vector<ProfileTileSiteMap> siteMap)
      : root(std::move(root)), profiledPackage(std::move(profiledPackage)),
        captures(std::move(captures)), siteMap(std::move(siteMap)) {}

  int64_t cardCount = kProfileInstrumentationCardCount;
  int64_t tileCount = kProfileInstrumentationTileCount;
  std::string root;
  ProfiledPackage profiledPackage;
  std::vector<ProfileCapturePackage> captures;
  std::vector<ProfileTileSiteMap> siteMap;
};

/// Loads and fail-closed verifies one compiler-written profiler
/// instrumentation. `primaryPackageRoot` is the ordinary package selected by
/// the user; the instrumentation primary-package digest must resolve to exactly
/// it.
llvm::Expected<VerifiedProfileInstrumentation>
loadVerifiedProfileInstrumentation(llvm::StringRef instrumentationRoot,
                                   llvm::StringRef primaryPackageRoot,
                                   const PackageParseLimits &limits = {});

/// Looks for the compiler-owned sibling `<primaryPackageRoot>.profile`.
/// Absence is not an error. Once the sibling exists, every malformed or stale
/// file or package mismatch is an error rather than a fallback to unprofiled
/// execution.
llvm::Expected<std::optional<VerifiedProfileInstrumentation>>
loadSiblingProfileInstrumentationIfPresent(
    llvm::StringRef primaryPackageRoot, const PackageParseLimits &limits = {});

} // namespace runtime
} // namespace wafer

#endif // WAFER_RUNTIME_PROFILEINSTRUMENTATION_H
