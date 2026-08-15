//===- NumericDependencyConformance.h - Numeric dependency records -*- C++
//-*-===//

#ifndef WAFER_TARGET_NUMERICDEPENDENCYCONFORMANCE_H
#define WAFER_TARGET_NUMERICDEPENDENCYCONFORMANCE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer {

inline constexpr uint32_t kNumericDependencyConformanceSchemaVersion = 2;

/// Stable failure classes for dependency record validation. Diagnostics use
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

class NumericDependencyConformanceError final
    : public llvm::ErrorInfo<NumericDependencyConformanceError> {
public:
  static char ID;

  NumericDependencyConformanceError(NumericDependencyConformanceErrorCode code,
                                    std::string detail)
      : code(code), detail(std::move(detail)) {}

  NumericDependencyConformanceErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  NumericDependencyConformanceErrorCode code;
  std::string detail;
};

struct NumericDependencyELFRecord {
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

struct NumericDependencyFileRecord {
  std::string name;
  std::string relativePath;
  std::string resolvedPath;
  std::string sha256;
  uint64_t size = 0;
  std::string fileType;
  uint32_t mode = 0;
  std::optional<NumericDependencyELFRecord> elf;
};

struct NumericDependencySourceRecord {
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

struct NumericDependencyBuildConfig {
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
  std::string elfValidationPolicy;
};

struct NumericDependencyToolRecord {
  std::string name;
  std::string resolvedPath;
  std::string sha256;
  uint64_t size = 0;
  std::string fileType;
  uint32_t mode = 0;
  std::string versionFirstLine;
  std::string versionOutputSHA256;
};

struct NumericDependencyEnvironmentRecord {
  std::string name;
  std::vector<std::pair<std::string, std::string>> variables;
};

struct NumericDependencyLicenseRecord {
  std::string name;
  std::string dependency;
  std::string sourceRelativePath;
  std::string sourceResolvedPath;
  std::string fileName;
  std::string sha256;
};

struct NumericDependencyConformanceGateRecord {
  std::string name;
  std::vector<std::string> command;
  std::string cwd;
  std::string environment;
  int64_t exitCode = -1;
  NumericDependencyFileRecord log;
};

struct NumericDependencyReadLimits {
  uint64_t maxRecordBytes = 16 * 1024 * 1024;
  uint64_t maxSourceFiles = 1'000'000;
  uint64_t maxSourceBytes = UINT64_C(16) * 1024 * 1024 * 1024;
};

/// Immutable, process-local readback of a fully verified managed record.
/// Construction is atomic: no instance is returned until the record, every
/// source tree, and every managed file, log, license, and header has passed.
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
  const NumericDependencyBuildConfig &getBuildConfig() const { return build; }
  llvm::ArrayRef<NumericDependencySourceRecord> getSources() const {
    return sources;
  }
  llvm::ArrayRef<NumericDependencyFileRecord> getFiles() const { return files; }
  llvm::ArrayRef<NumericDependencyToolRecord> getTools() const { return tools; }
  llvm::ArrayRef<NumericDependencyEnvironmentRecord> getEnvironments() const {
    return environments;
  }
  llvm::ArrayRef<NumericDependencyLicenseRecord> getLicenses() const {
    return licenses;
  }
  llvm::ArrayRef<NumericDependencyConformanceGateRecord> getGates() const {
    return gates;
  }

  const NumericDependencySourceRecord *findSource(llvm::StringRef name) const;
  const NumericDependencyFileRecord *findFile(llvm::StringRef name) const;
  const NumericDependencyToolRecord *findTool(llvm::StringRef name) const;
  const NumericDependencyEnvironmentRecord *
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
  std::vector<NumericDependencySourceRecord> sources;
  NumericDependencyBuildConfig build;
  std::vector<NumericDependencyToolRecord> tools;
  std::vector<NumericDependencyEnvironmentRecord> environments;
  std::vector<NumericDependencyFileRecord> files;
  std::vector<NumericDependencyLicenseRecord> licenses;
  std::string conformancePolicy;
  std::vector<NumericDependencyConformanceGateRecord> gates;
};

/// Read and verify schema version 2 of numeric-model-deps.json.  The expected
/// digest must be the configure-time trusted digest emitted only after the
/// Python producer verifies each source archive (the raw 64-character,
/// lowercase SHA-256 stored by WaferNumericModelDeps.cmake); computing it from
/// an otherwise untrusted record is not a trust boundary. Readback replays the
/// recorded host-tool version commands and the recorded readelf tool under a
/// closed environment, then verifies the current sources and managed files.
llvm::Expected<NumericDependencyConformanceRecord>
readNumericDependencyConformanceRecord(
    llvm::StringRef managedRoot, llvm::StringRef recordPath,
    llvm::StringRef expectedRecordSHA256,
    const NumericDependencyReadLimits &limits = {});

enum class NumericLoadedObjectKind { MPFR, GMP };

llvm::StringRef stringifyNumericLoadedObjectKind(NumericLoadedObjectKind kind);

struct NumericLoadedObject {
  NumericLoadedObjectKind kind = NumericLoadedObjectKind::MPFR;
  std::string resolvedPath;
  std::string sha256;
};

/// Injection boundary for runtime loader readback.  A feature-enabled caller
/// may use the dladdr provider below or supply a stricter platform provider.
class NumericLoadedObjectProvider {
public:
  virtual ~NumericLoadedObjectProvider() = default;

  virtual llvm::Expected<NumericLoadedObject>
  getLoadedObject(NumericLoadedObjectKind kind) const = 0;
};

/// Generic POSIX symbol-origin provider.  The caller supplies representative
/// symbol addresses from the already loaded MPFR and GMP objects; no MPFR/GMP
/// header or link dependency is introduced by this interface.
class DladdrNumericLoadedObjectProvider final
    : public NumericLoadedObjectProvider {
public:
  DladdrNumericLoadedObjectProvider(const void *mpfrSymbolAddress,
                                    const void *gmpSymbolAddress)
      : mpfrSymbolAddress(mpfrSymbolAddress),
        gmpSymbolAddress(gmpSymbolAddress) {}

  llvm::Expected<NumericLoadedObject>
  getLoadedObject(NumericLoadedObjectKind kind) const override;

private:
  const void *mpfrSymbolAddress;
  const void *gmpSymbolAddress;
};

/// Successful execution-time binding of the verified record to the actual
/// loaded MPFR/GMP symbol providers. The binding digest covers the record
/// digest plus both resolved paths and content digests.
class NumericDependencyExecutionBinding {
public:
  NumericDependencyExecutionBinding(NumericDependencyExecutionBinding &&) =
      default;
  NumericDependencyExecutionBinding &
  operator=(NumericDependencyExecutionBinding &&) = delete;
  NumericDependencyExecutionBinding(const NumericDependencyExecutionBinding &) =
      delete;
  NumericDependencyExecutionBinding &
  operator=(const NumericDependencyExecutionBinding &) = delete;

  llvm::StringRef getRecordSHA256() const { return recordSHA256; }
  llvm::StringRef getBindingSHA256() const { return bindingSHA256; }
  const NumericLoadedObject &getMPFR() const { return mpfr; }
  const NumericLoadedObject &getGMP() const { return gmp; }

private:
  friend llvm::Expected<NumericDependencyExecutionBinding>
  bindNumericDependenciesToLoadedObjects(
      const NumericDependencyConformanceRecord &,
      const NumericLoadedObjectProvider &);

  NumericDependencyExecutionBinding(std::string recordSHA256,
                                    std::string bindingSHA256,
                                    NumericLoadedObject mpfr,
                                    NumericLoadedObject gmp)
      : recordSHA256(std::move(recordSHA256)),
        bindingSHA256(std::move(bindingSHA256)), mpfr(std::move(mpfr)),
        gmp(std::move(gmp)) {}

  std::string recordSHA256;
  std::string bindingSHA256;
  NumericLoadedObject mpfr;
  NumericLoadedObject gmp;
};

/// Verify that the actual MPFR/GMP symbol providers are exactly one of the
/// record's real or materialized loader files, with matching file SHA-256.
/// This function is side-effect free and returns no partial binding.
llvm::Expected<NumericDependencyExecutionBinding>
bindNumericDependenciesToLoadedObjects(
    const NumericDependencyConformanceRecord &record,
    const NumericLoadedObjectProvider &provider);

} // namespace wafer

#endif // WAFER_TARGET_NUMERICDEPENDENCYCONFORMANCE_H
