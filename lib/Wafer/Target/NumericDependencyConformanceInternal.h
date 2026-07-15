//===- NumericDependencyConformanceInternal.h - Private helpers -*- C++ -*-===//

#ifndef WAFER_LIB_TARGET_NUMERICDEPENDENCYCONFORMANCEINTERNAL_H
#define WAFER_LIB_TARGET_NUMERICDEPENDENCYCONFORMANCEINTERNAL_H

#include "Wafer/Target/NumericDependencyConformance.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

namespace wafer::numeric_dependency_conformance_internal {

using ErrorCode = NumericDependencyConformanceErrorCode;

inline constexpr llvm::StringLiteral kRecordKind = "wafer-numeric-model-deps";
inline constexpr llvm::StringLiteral kRecordStatus = "conformance-passed";
inline constexpr llvm::StringLiteral kConformancePolicy =
    "wafer-numeric-model-conformance-v1";

inline constexpr llvm::StringLiteral kRequiredPins[] = {
    "softfloat", "testfloat", "m4", "gmp", "mpfr"};

inline constexpr llvm::StringLiteral kRequiredArtifacts[] = {
    "m4",
    "softfloat",
    "softfloat-header",
    "softfloat-types-header",
    "testsoftfloat",
    "gmp",
    "gmp-soname",
    "gmp-header",
    "mpfr",
    "mpfr-soname",
    "mpfr-header",
    "license-softfloat",
    "license-testfloat",
    "license-m4",
    "license-gmp-copying",
    "license-gmp-gpl-v2",
    "license-gmp-gpl-v3",
    "license-gmp-lgpl-v3",
    "license-mpfr-copying",
    "license-mpfr-lesser",
};

struct ArtifactPathPolicy {
  llvm::StringLiteral name;
  llvm::StringLiteral path;
  bool executable;
};

inline constexpr ArtifactPathPolicy kArtifactPathPolicies[] = {
    {"m4", "install/m4/bin/m4", true},
    {"softfloat", "install/softfloat/lib/libsoftfloat.a", false},
    {"softfloat-header", "install/softfloat/include/softfloat.h", false},
    {"softfloat-types-header", "install/softfloat/include/softfloat_types.h",
     false},
    {"testsoftfloat", "install/testfloat/bin/testsoftfloat", true},
    {"gmp", "install/gmp/lib/libgmp.so.10.5.0", false},
    {"gmp-soname", "install/gmp/lib/libgmp.so.10", false},
    {"gmp-header", "install/gmp/include/gmp.h", false},
    {"mpfr", "install/mpfr/lib/libmpfr.so.6.2.2", false},
    {"mpfr-soname", "install/mpfr/lib/libmpfr.so.6", false},
    {"mpfr-header", "install/mpfr/include/mpfr.h", false},
    {"license-softfloat", "install/licenses/softfloat.txt", false},
    {"license-testfloat", "install/licenses/testfloat.txt", false},
    {"license-m4", "install/licenses/m4.txt", false},
    {"license-gmp-copying", "install/licenses/gmp-copying.txt", false},
    {"license-gmp-gpl-v2", "install/licenses/gmp-gpl-v2.txt", false},
    {"license-gmp-gpl-v3", "install/licenses/gmp-gpl-v3.txt", false},
    {"license-gmp-lgpl-v3", "install/licenses/gmp-lgpl-v3.txt", false},
    {"license-mpfr-copying", "install/licenses/mpfr-copying.txt", false},
    {"license-mpfr-lesser", "install/licenses/mpfr-lesser.txt", false},
};

inline constexpr llvm::StringLiteral kRequiredGates[] = {
    "m4-configure",
    "m4-build",
    "m4-check",
    "m4-install",
    "m4-version",
    "gmp-configure",
    "gmp-build",
    "gmp-check",
    "gmp-install",
    "mpfr-configure",
    "mpfr-build",
    "mpfr-check",
    "mpfr-install",
    "softfloat-build",
    "testfloat-build",
    "softfloat-policy-compile",
    "softfloat-tls-default-nan",
    "testsoftfloat-f16-mulAdd",
    "testsoftfloat-f32-mulAdd",
    "testsoftfloat-all1",
    "testsoftfloat-all2",
    "managed-version-thread-safe-compile",
    "managed-version-thread-safe",
};

inline constexpr llvm::StringLiteral kRequiredTools[] = {
    "cc", "cxx", "make",    "ar",    "ranlib",
    "nm", "ld",  "readelf", "shell", "false"};

inline constexpr llvm::StringLiteral kRequiredEnvironments[] = {
    "base", "managed", "mpfr", "version-runtime"};

struct LicensePolicy {
  llvm::StringLiteral name;
  llvm::StringLiteral dependency;
  llvm::StringLiteral source;
};

inline constexpr LicensePolicy kRequiredLicenses[] = {
    {"license-softfloat", "softfloat", "COPYING.txt"},
    {"license-testfloat", "testfloat", "COPYING.txt"},
    {"license-m4", "m4", "COPYING"},
    {"license-gmp-copying", "gmp", "COPYING"},
    {"license-gmp-gpl-v2", "gmp", "COPYINGv2"},
    {"license-gmp-gpl-v3", "gmp", "COPYINGv3"},
    {"license-gmp-lgpl-v3", "gmp", "COPYING.LESSERv3"},
    {"license-mpfr-copying", "mpfr", "COPYING"},
    {"license-mpfr-lesser", "mpfr", "COPYING.LESSER"},
};

llvm::Error invalid(ErrorCode code, const llvm::Twine &detail);
bool isLowerSHA256(llvm::StringRef digest);

struct FileReadback {
  std::string digest;
  std::string contents;
  uint64_t size = 0;
  uint32_t mode = 0;
};

llvm::Expected<FileReadback> readRegularFile(llvm::StringRef path,
                                             uint64_t maximumBytes,
                                             const llvm::Twine &label,
                                             bool captureContents = false);

llvm::Expected<std::string>
runAndCapture(llvm::StringRef program, llvm::ArrayRef<std::string> arguments,
              llvm::ArrayRef<std::string> environment,
              const llvm::Twine &label);
std::vector<std::string> commandEnvironment(llvm::StringRef program);

struct ManagedRoot {
  std::string requested;
  std::string resolved;
};

llvm::Expected<ManagedRoot> verifyManagedRoot(llvm::StringRef input);

llvm::Expected<std::string>
resolveAbsoluteRegularPath(llvm::StringRef input, const llvm::Twine &label);

struct ManagedPath {
  std::string relative;
  std::string resolved;
  llvm::sys::fs::file_status status;
};

enum class RequiredFileType { RegularFile, Directory };

llvm::Expected<ManagedPath> resolveManagedPath(const ManagedRoot &root,
                                               llvm::StringRef input,
                                               RequiredFileType requiredType,
                                               const llvm::Twine &label,
                                               bool requireRelative);

llvm::Error requireExactKeys(const llvm::json::Object &object,
                             std::initializer_list<llvm::StringRef> expected,
                             const llvm::Twine &label);
llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Value &value, const llvm::Twine &label);
llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Value &value, const llvm::Twine &label);
llvm::Expected<std::string> requireString(const llvm::json::Value &value,
                                          const llvm::Twine &label);
llvm::Expected<uint64_t> requireUnsigned(const llvm::json::Value &value,
                                         const llvm::Twine &label);
llvm::Expected<std::vector<std::string>>
requireStringArray(const llvm::json::Value &value, const llvm::Twine &label);
llvm::Expected<std::optional<std::string>>
requireNullableString(const llvm::json::Value &value, const llvm::Twine &label);
llvm::Error expectString(llvm::StringRef actual, llvm::StringRef expected,
                         const llvm::Twine &label);
llvm::Error expectStringArray(llvm::ArrayRef<std::string> actual,
                              std::initializer_list<llvm::StringRef> expected,
                              const llvm::Twine &label);
llvm::Error scanDuplicateJSONKeys(llvm::StringRef input);

llvm::Expected<std::string>
digestSourceTree(llvm::StringRef sourceRoot,
                 const NumericDependencyReadLimits &limits,
                 const llvm::Twine &label);

llvm::Expected<NumericDependencyArtifactIdentity>
parseArtifactIdentity(llvm::StringRef name, const llvm::json::Value &value,
                      const ManagedRoot &root, llvm::StringRef readelfPath);

struct ParsedBuildIdentity {
  NumericDependencyBuildIdentity build;
  std::vector<NumericDependencyToolIdentity> tools;
  std::vector<NumericDependencyEnvironmentIdentity> environments;
};

llvm::Expected<ParsedBuildIdentity>
parseBuildIdentity(const llvm::json::Value &value);

llvm::Error
validateGateContract(const NumericDependencyConformanceRecord &record,
                     const NumericDependencyConformanceGateIdentity &gate);
llvm::Error
validateSharedObjectPair(llvm::StringRef stem,
                         const NumericDependencyArtifactIdentity &real,
                         const NumericDependencyArtifactIdentity &loader);

} // namespace wafer::numeric_dependency_conformance_internal

#endif // WAFER_LIB_TARGET_NUMERICDEPENDENCYCONFORMANCEINTERNAL_H
