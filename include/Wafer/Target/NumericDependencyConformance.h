//===- NumericDependencyConformance.h - Numeric dependency identity -*- C++
//-*-===//

#ifndef WAFER_TARGET_NUMERICDEPENDENCYCONFORMANCE_H
#define WAFER_TARGET_NUMERICDEPENDENCYCONFORMANCE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer {

inline constexpr uint32_t kNumericDependencyConformanceSchemaVersion = 2;

/// Stable failure classes for dependency identity preflight.  Diagnostics use
/// these spellings rather than parser- or operating-system-specific text so a
/// caller can classify a failure without inspecting mutable external state.
enum class NumericDependencyConformanceErrorCode {
  InvalidArgument,
  UnsafePath,
  Symlink,
  FileType,
  IO,
  RecordTooLarge,
  RecordDigestMismatch,
  JSONSyntax,
  DuplicateField,
  MissingField,
  UnknownField,
  TypeMismatch,
  SchemaMismatch,
  PolicyMismatch,
  ClosureMismatch,
  SizeMismatch,
  DigestMismatch,
  SourceTreeMismatch,
  ResourceLimit,
  LoadedObjectUnavailable,
  LoadedObjectMismatch,
};

llvm::StringRef stringifyNumericDependencyConformanceErrorCode(
    NumericDependencyConformanceErrorCode code);

struct NumericDependencyELFIdentity {
  std::string elfClass;
  std::string byteOrder;
  std::string type;
  std::string machine;
  std::optional<std::string> buildId;
  std::optional<std::string> soname;
  std::vector<std::string> needed;
  std::vector<std::string> rpath;
  std::vector<std::string> runpath;
};

struct NumericDependencyArtifactIdentity {
  std::string name;
  std::string relativePath;
  std::string resolvedPath;
  std::string sha256;
  uint64_t size = 0;
  std::string fileType;
  uint32_t mode = 0;
  std::optional<NumericDependencyELFIdentity> elf;
};

struct NumericDependencySourceIdentity {
  std::string name;
  std::string version;
  std::string url;
  std::string archiveSHA256;
  std::string archiveRelativePath;
  std::string archiveResolvedPath;
  std::string sourceRelativePath;
  std::string sourceResolvedPath;
  std::string sourceTreeSHA256;
};

struct NumericDependencyBuildIdentity {
  std::string platform;
  std::string softFloatSpecialization;
  std::string softFloatThreadLocal;
  std::string softFloatRaiseFlags;
  std::vector<std::string> m4ConfigureOptions;
  std::vector<std::string> gmpConfigureOptions;
  std::vector<std::string> mpfrConfigureOptions;
  std::string softFloatLinkage;
  std::string gmpLinkage;
  std::string mpfrLinkage;
  std::string managedM4RelativePath;
  std::string pkgConfigPolicy;
  uint64_t jobs = 0;
  std::string toolchainPolicy;
  std::string mpfrPatches;
  std::string elfIdentityPolicy;
};

struct NumericDependencyToolIdentity {
  std::string name;
  std::string resolvedPath;
  std::string sha256;
  uint64_t size = 0;
  std::string fileType;
  uint32_t mode = 0;
  std::string versionFirstLine;
  std::string versionOutputSHA256;
};

struct NumericDependencyEnvironmentIdentity {
  std::string name;
  std::vector<std::pair<std::string, std::string>> variables;
};

struct NumericDependencyLicenseIdentity {
  std::string name;
  std::string dependency;
  std::string sourceRelativePath;
  std::string sourceResolvedPath;
  std::string artifactName;
  std::string sha256;
};

struct NumericDependencyConformanceGateIdentity {
  std::string name;
  std::vector<std::string> command;
  std::string cwd;
  std::string environment;
  int64_t exitCode = -1;
  NumericDependencyArtifactIdentity log;
};

struct NumericDependencyReadLimits {
  uint64_t maxRecordBytes = 16 * 1024 * 1024;
  uint64_t maxSourceFiles = 1'000'000;
  uint64_t maxSourceBytes = UINT64_C(16) * 1024 * 1024 * 1024;
};

/// Immutable, process-local readback of a fully verified managed record.
/// Construction is atomic: no instance is returned until the record, every
/// source tree, and every artifact/log/license/header identity has passed.
class NumericDependencyConformanceRecord {
public:
  NumericDependencyConformanceRecord(NumericDependencyConformanceRecord &&) =
      default;
  NumericDependencyConformanceRecord &
  operator=(NumericDependencyConformanceRecord &&) = delete;
  NumericDependencyConformanceRecord(
      const NumericDependencyConformanceRecord &) = delete;
  NumericDependencyConformanceRecord &
  operator=(const NumericDependencyConformanceRecord &) = delete;

  uint32_t getSchemaVersion() const { return schemaVersion; }
  llvm::StringRef getKind() const { return kind; }
  llvm::StringRef getStatus() const { return status; }
  llvm::StringRef getManagedRoot() const { return managedRoot; }
  llvm::StringRef getRecordPath() const { return recordPath; }
  llvm::StringRef getRecordSHA256() const { return recordSHA256; }
  llvm::StringRef getConformancePolicy() const { return conformancePolicy; }
  const NumericDependencyBuildIdentity &getBuildIdentity() const {
    return build;
  }
  llvm::ArrayRef<NumericDependencySourceIdentity> getSources() const {
    return sources;
  }
  llvm::ArrayRef<NumericDependencyArtifactIdentity> getArtifacts() const {
    return artifacts;
  }
  llvm::ArrayRef<NumericDependencyToolIdentity> getTools() const {
    return tools;
  }
  llvm::ArrayRef<NumericDependencyEnvironmentIdentity> getEnvironments() const {
    return environments;
  }
  llvm::ArrayRef<NumericDependencyLicenseIdentity> getLicenses() const {
    return licenses;
  }
  llvm::ArrayRef<NumericDependencyConformanceGateIdentity> getGates() const {
    return gates;
  }

  const NumericDependencySourceIdentity *findSource(llvm::StringRef name) const;
  const NumericDependencyArtifactIdentity *
  findArtifact(llvm::StringRef name) const;
  const NumericDependencyToolIdentity *findTool(llvm::StringRef name) const;
  const NumericDependencyEnvironmentIdentity *
  findEnvironment(llvm::StringRef name) const;

private:
  friend llvm::Expected<NumericDependencyConformanceRecord>
  readNumericDependencyConformanceRecord(llvm::StringRef, llvm::StringRef,
                                         llvm::StringRef,
                                         const NumericDependencyReadLimits &);

  NumericDependencyConformanceRecord() = default;

  uint32_t schemaVersion = 0;
  std::string kind;
  std::string status;
  std::string managedRoot;
  std::string recordPath;
  std::string recordSHA256;
  std::vector<NumericDependencySourceIdentity> sources;
  NumericDependencyBuildIdentity build;
  std::vector<NumericDependencyToolIdentity> tools;
  std::vector<NumericDependencyEnvironmentIdentity> environments;
  std::vector<NumericDependencyArtifactIdentity> artifacts;
  std::vector<NumericDependencyLicenseIdentity> licenses;
  std::string conformancePolicy;
  std::vector<NumericDependencyConformanceGateIdentity> gates;
};

/// Read and verify schema version 2 of numeric-model-deps.json.  The expected
/// digest must be the configure-time trusted identity emitted only after the
/// Python producer/validator accepts archive provenance (the raw 64-character,
/// lowercase SHA-256 stored by WaferNumericModelDeps.cmake); computing it from
/// an otherwise untrusted record is not a trust boundary.  Readback replays the
/// recorded host-tool version commands and the recorded readelf tool under a
/// closed environment, then verifies current source/artifact evidence.
llvm::Expected<NumericDependencyConformanceRecord>
readNumericDependencyConformanceRecord(
    llvm::StringRef managedRoot, llvm::StringRef recordPath,
    llvm::StringRef expectedRecordSHA256,
    const NumericDependencyReadLimits &limits = {});

enum class NumericLoadedObjectKind { MPFR, GMP };

llvm::StringRef stringifyNumericLoadedObjectKind(NumericLoadedObjectKind kind);

struct NumericLoadedObjectIdentity {
  NumericLoadedObjectKind kind = NumericLoadedObjectKind::MPFR;
  std::string resolvedPath;
  std::string sha256;
};

/// Injection boundary for runtime loader readback.  A feature-enabled caller
/// may use the dladdr provider below or supply a stricter platform provider.
class NumericLoadedObjectIdentityProvider {
public:
  virtual ~NumericLoadedObjectIdentityProvider() = default;

  virtual llvm::Expected<NumericLoadedObjectIdentity>
  identify(NumericLoadedObjectKind kind) const = 0;
};

/// Generic POSIX symbol-origin provider.  The caller supplies representative
/// symbol addresses from the already loaded MPFR and GMP objects; no MPFR/GMP
/// header or link dependency is introduced by this interface.
class DladdrNumericLoadedObjectIdentityProvider final
    : public NumericLoadedObjectIdentityProvider {
public:
  DladdrNumericLoadedObjectIdentityProvider(const void *mpfrSymbolAddress,
                                            const void *gmpSymbolAddress)
      : mpfrSymbolAddress(mpfrSymbolAddress),
        gmpSymbolAddress(gmpSymbolAddress) {}

  llvm::Expected<NumericLoadedObjectIdentity>
  identify(NumericLoadedObjectKind kind) const override;

private:
  const void *mpfrSymbolAddress;
  const void *gmpSymbolAddress;
};

/// Successful execution-time binding of the verified record to the actual
/// loaded MPFR/GMP symbol providers.  The provenance digest covers the record
/// digest plus both resolved paths and content digests.
class NumericDependencyExecutionIdentity {
public:
  NumericDependencyExecutionIdentity(NumericDependencyExecutionIdentity &&) =
      default;
  NumericDependencyExecutionIdentity &
  operator=(NumericDependencyExecutionIdentity &&) = delete;
  NumericDependencyExecutionIdentity(
      const NumericDependencyExecutionIdentity &) = delete;
  NumericDependencyExecutionIdentity &
  operator=(const NumericDependencyExecutionIdentity &) = delete;

  llvm::StringRef getRecordSHA256() const { return recordSHA256; }
  llvm::StringRef getProvenanceSHA256() const { return provenanceSHA256; }
  const NumericLoadedObjectIdentity &getMPFR() const { return mpfr; }
  const NumericLoadedObjectIdentity &getGMP() const { return gmp; }

private:
  friend llvm::Expected<NumericDependencyExecutionIdentity>
  verifyNumericDependencyExecutionIdentity(
      const NumericDependencyConformanceRecord &,
      const NumericLoadedObjectIdentityProvider &);

  NumericDependencyExecutionIdentity(std::string recordSHA256,
                                     std::string provenanceSHA256,
                                     NumericLoadedObjectIdentity mpfr,
                                     NumericLoadedObjectIdentity gmp)
      : recordSHA256(std::move(recordSHA256)),
        provenanceSHA256(std::move(provenanceSHA256)), mpfr(std::move(mpfr)),
        gmp(std::move(gmp)) {}

  std::string recordSHA256;
  std::string provenanceSHA256;
  NumericLoadedObjectIdentity mpfr;
  NumericLoadedObjectIdentity gmp;
};

/// Verify that the actual MPFR/GMP symbol providers are exactly one of the
/// record's real or materialized loader artifacts, with matching file SHA-256.
/// This function is side-effect free and returns no partial execution identity.
llvm::Expected<NumericDependencyExecutionIdentity>
verifyNumericDependencyExecutionIdentity(
    const NumericDependencyConformanceRecord &record,
    const NumericLoadedObjectIdentityProvider &provider);

} // namespace wafer

#endif // WAFER_TARGET_NUMERICDEPENDENCYCONFORMANCE_H
