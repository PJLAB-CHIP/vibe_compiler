//===- NumericDependencyConformance.cpp - Public conformance orchestration ===//

#include "Wafer/Target/Numeric/Dependency/NumericDependencyConformance.h"

#include "NumericDependencyConformanceInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer {
using namespace numeric_dependency_conformance_internal;

llvm::Error
numeric_dependency_conformance_internal::invalid(ErrorCode code,
                                                 const llvm::Twine &detail) {
  return llvm::make_error<NumericDependencyConformanceError>(code,
                                                             detail.str());
}

llvm::StringRef stringifyNumericDependencyConformanceErrorCode(
    NumericDependencyConformanceErrorCode code) {
  switch (code) {
  case ErrorCode::InvalidArgument:
    return "invalid-argument";
  case ErrorCode::UnsafePath:
    return "unsafe-path";
  case ErrorCode::Symlink:
    return "symlink";
  case ErrorCode::FileType:
    return "file-type";
  case ErrorCode::IO:
    return "io";
  case ErrorCode::RecordTooLarge:
    return "record-too-large";
  case ErrorCode::RecordDigestMismatch:
    return "record-digest-mismatch";
  case ErrorCode::JSONSyntax:
    return "json-syntax";
  case ErrorCode::DuplicateField:
    return "duplicate-field";
  case ErrorCode::MissingField:
    return "missing-field";
  case ErrorCode::UnknownField:
    return "unknown-field";
  case ErrorCode::TypeMismatch:
    return "type-mismatch";
  case ErrorCode::PolicyMismatch:
    return "policy-mismatch";
  case ErrorCode::ClosureMismatch:
    return "closure-mismatch";
  case ErrorCode::SizeMismatch:
    return "size-mismatch";
  case ErrorCode::DigestMismatch:
    return "digest-mismatch";
  case ErrorCode::SourceTreeMismatch:
    return "source-tree-mismatch";
  case ErrorCode::ResourceLimit:
    return "resource-limit";
  case ErrorCode::LoadedObjectUnavailable:
    return "loaded-object-unavailable";
  case ErrorCode::LoadedObjectMismatch:
    return "loaded-object-mismatch";
  }
  llvm_unreachable("unknown numeric dependency conformance error code");
}

char NumericDependencyConformanceError::ID;

void NumericDependencyConformanceError::log(llvm::raw_ostream &stream) const {
  stream << "numeric dependency conformance "
         << stringifyNumericDependencyConformanceErrorCode(code) << ": "
         << detail;
}

std::error_code NumericDependencyConformanceError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

const NumericDependencySourceRecord *
NumericDependencyConformanceRecord::findSource(llvm::StringRef name) const {
  auto iterator =
      llvm::find_if(sources, [&](const NumericDependencySourceRecord &source) {
        return source.name == name;
      });
  return iterator == sources.end() ? nullptr : &*iterator;
}

const NumericDependencyFileRecord *
NumericDependencyConformanceRecord::findFile(llvm::StringRef name) const {
  auto iterator =
      llvm::find_if(files, [&](const NumericDependencyFileRecord &file) {
        return file.name == name;
      });
  return iterator == files.end() ? nullptr : &*iterator;
}

const NumericDependencyToolRecord *
NumericDependencyConformanceRecord::findTool(llvm::StringRef name) const {
  auto iterator =
      llvm::find_if(tools, [&](const NumericDependencyToolRecord &tool) {
        return tool.name == name;
      });
  return iterator == tools.end() ? nullptr : &*iterator;
}

const NumericDependencyEnvironmentRecord *
NumericDependencyConformanceRecord::findEnvironment(
    llvm::StringRef name) const {
  auto iterator = llvm::find_if(
      environments, [&](const NumericDependencyEnvironmentRecord &environment) {
        return environment.name == name;
      });
  return iterator == environments.end() ? nullptr : &*iterator;
}

llvm::Expected<NumericDependencyConformanceRecord>
readNumericDependencyConformanceRecord(
    llvm::StringRef managedRoot, llvm::StringRef recordPath,
    llvm::StringRef expectedRecordSHA256,
    const NumericDependencyReadLimits &limits) {
  if (!isLowerSHA256(expectedRecordSHA256))
    return invalid(ErrorCode::InvalidArgument,
                   "expected record digest is not lowercase SHA-256");
  if (limits.maxRecordBytes == 0 || limits.maxSourceFiles == 0 ||
      limits.maxSourceBytes == 0)
    return invalid(ErrorCode::InvalidArgument,
                   "dependency read limits must be nonzero");

  llvm::Expected<ManagedRoot> root = verifyManagedRoot(managedRoot);
  if (!root)
    return root.takeError();
  llvm::Expected<ManagedPath> resolvedRecord = resolveManagedPath(
      *root, recordPath, RequiredFileType::RegularFile,
      "numeric dependency record", /*requireRelative=*/false);
  if (!resolvedRecord)
    return resolvedRecord.takeError();
  llvm::Expected<FileReadback> recordReadback =
      readRegularFile(resolvedRecord->resolved, limits.maxRecordBytes,
                      "numeric dependency record", /*captureContents=*/true);
  if (!recordReadback) {
    std::optional<ErrorCode> readErrorCode;
    std::string readErrorDetail;
    llvm::Error unhandled =
        llvm::handleErrors(recordReadback.takeError(),
                           [&](const NumericDependencyConformanceError &error) {
                             readErrorCode = error.getCode();
                             readErrorDetail = error.getDetail().str();
                           });
    if (unhandled)
      return std::move(unhandled);
    if (readErrorCode == ErrorCode::ResourceLimit)
      return invalid(ErrorCode::RecordTooLarge,
                     "numeric dependency record exceeds byte limit");
    return invalid(ErrorCode::IO, readErrorDetail);
  }
  if (recordReadback->digest != expectedRecordSHA256)
    return invalid(ErrorCode::RecordDigestMismatch,
                   "numeric dependency record SHA256 mismatch");

  llvm::StringRef recordBytes(recordReadback->contents);

  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(recordBytes);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return invalid(ErrorCode::JSONSyntax,
                   "numeric dependency record is not valid JSON");
  }
  if (llvm::Error error = scanDuplicateJSONKeys(recordBytes))
    return error;
  llvm::Expected<const llvm::json::Object *> top =
      requireObject(*parsed, "numeric dependency record");
  if (!top)
    return top.takeError();
  if (llvm::Error error =
          requireExactKeys(**top,
                           {"kind", "status", "pins", "build", "artifacts",
                            "licenses", "conformance"},
                           "numeric dependency record"))
    return error;
  llvm::Expected<std::string> kind =
      requireString(*(*top)->get("kind"), "numeric record kind");
  if (!kind)
    return kind.takeError();
  llvm::Expected<std::string> status =
      requireString(*(*top)->get("status"), "numeric record status");
  if (!status)
    return status.takeError();
  if (*kind != kRecordKind || *status != kRecordStatus)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric dependency record is not conformance-passed");

  NumericDependencyConformanceRecord result;
  result.kind = std::move(*kind);
  result.status = std::move(*status);
  result.managedRoot = root->resolved;
  result.recordPath = resolvedRecord->resolved;
  result.recordSHA256 = expectedRecordSHA256.str();

  llvm::Expected<const llvm::json::Object *> pins =
      requireObject(*(*top)->get("pins"), "numeric dependency pins");
  if (!pins)
    return pins.takeError();
  if (llvm::Error error = requireExactKeys(
          **pins, {"softfloat", "testfloat", "m4", "gmp", "mpfr"},
          "numeric dependency pins"))
    return error;

  llvm::StringSet<> pinArchivePaths;
  llvm::StringSet<> pinSourcePaths;
  for (llvm::StringRef name : kRequiredPins) {
    llvm::Expected<const llvm::json::Object *> pin =
        requireObject(*(*pins)->get(name), "numeric pin " + name);
    if (!pin)
      return pin.takeError();
    if (llvm::Error error =
            requireExactKeys(**pin,
                             {"version", "url", "archive_sha256", "archive",
                              "source", "source_tree_sha256"},
                             "numeric pin " + name))
      return error;

    NumericDependencySourceRecord sourceRecord;
    sourceRecord.name = name.str();
#define READ_PIN_STRING(Field, Key)                                            \
  do {                                                                         \
    llvm::Expected<std::string> value =                                        \
        requireString(*(*pin)->get(Key), "numeric pin " + name + "." Key);     \
    if (!value)                                                                \
      return value.takeError();                                                \
    sourceRecord.Field = std::move(*value);                                    \
  } while (false)
    READ_PIN_STRING(version, "version");
    READ_PIN_STRING(url, "url");
    READ_PIN_STRING(archiveSHA256, "archive_sha256");
    READ_PIN_STRING(archiveRelativePath, "archive");
    READ_PIN_STRING(sourceRelativePath, "source");
    READ_PIN_STRING(sourceTreeSHA256, "source_tree_sha256");
#undef READ_PIN_STRING
    if (sourceRecord.version.empty() ||
        !llvm::StringRef(sourceRecord.url).starts_with("https://") ||
        !isLowerSHA256(sourceRecord.archiveSHA256) ||
        !isLowerSHA256(sourceRecord.sourceTreeSHA256))
      return invalid(ErrorCode::TypeMismatch,
                     "numeric pin " + name + " record is malformed");
    if (!pinArchivePaths.insert(sourceRecord.archiveRelativePath).second ||
        !pinSourcePaths.insert(sourceRecord.sourceRelativePath).second)
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric pin paths are not unique");

    llvm::Expected<ManagedPath> archive = resolveManagedPath(
        *root, sourceRecord.archiveRelativePath, RequiredFileType::RegularFile,
        "numeric source archive " + name, /*requireRelative=*/true);
    if (!archive)
      return archive.takeError();
    llvm::StringRef urlFile =
        llvm::StringRef(sourceRecord.url).rsplit('/').second;
    if (urlFile.empty() ||
        llvm::sys::path::filename(archive->resolved) != urlFile)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric source archive name does not match URL");
    llvm::Expected<FileReadback> archiveReadback =
        readRegularFile(archive->resolved, std::numeric_limits<uint64_t>::max(),
                        "numeric source archive " + name);
    if (!archiveReadback)
      return archiveReadback.takeError();
    if (archiveReadback->digest != sourceRecord.archiveSHA256)
      return invalid(ErrorCode::DigestMismatch,
                     "numeric source archive SHA256 mismatch for " + name);
    sourceRecord.archiveResolvedPath = archive->resolved;

    llvm::Expected<ManagedPath> source = resolveManagedPath(
        *root, sourceRecord.sourceRelativePath, RequiredFileType::Directory,
        "numeric source tree " + name, /*requireRelative=*/true);
    if (!source)
      return source.takeError();
    llvm::Expected<std::string> sourceTreeDigest = digestSourceTree(
        source->resolved, limits, "numeric source tree " + name);
    if (!sourceTreeDigest)
      return sourceTreeDigest.takeError();
    if (*sourceTreeDigest != sourceRecord.sourceTreeSHA256)
      return invalid(ErrorCode::SourceTreeMismatch,
                     "numeric source tree SHA256 mismatch for " + name);
    sourceRecord.sourceResolvedPath = source->resolved;
    result.sources.push_back(std::move(sourceRecord));
  }

  llvm::Expected<ParsedBuildConfig> build =
      parseBuildConfig(*(*top)->get("build"));
  if (!build)
    return build.takeError();
  result.build = std::move(build->build);
  result.tools = std::move(build->tools);
  result.environments = std::move(build->environments);
  const NumericDependencyToolRecord *readelfTool = result.findTool("readelf");
  if (!readelfTool)
    return invalid(ErrorCode::ClosureMismatch,
                   "numeric readelf tool record is missing");

  llvm::Expected<const llvm::json::Object *> fileEntries =
      requireObject(*(*top)->get("artifacts"), "numeric dependency files");
  if (!fileEntries)
    return fileEntries.takeError();
  llvm::StringSet<> allowedFiles;
  for (llvm::StringRef name : kRequiredFiles)
    allowedFiles.insert(name);
  for (const auto &entry : **fileEntries)
    if (!allowedFiles.contains(entry.first))
      return invalid(ErrorCode::UnknownField,
                     "numeric dependency files has unknown field '" +
                         entry.first.str() + "'");
  for (llvm::StringRef name : kRequiredFiles)
    if (!(*fileEntries)->get(name))
      return invalid(ErrorCode::MissingField,
                     "numeric dependency files is missing field '" + name +
                         "'");

  llvm::StringSet<> filePaths;
  for (llvm::StringRef name : kRequiredFiles) {
    llvm::Expected<NumericDependencyFileRecord> fileRecord = parseFileRecord(
        name, *(*fileEntries)->get(name), *root, readelfTool->resolvedPath);
    if (!fileRecord)
      return fileRecord.takeError();
    auto policy = llvm::find_if(kRequiredFilePolicies,
                                [&](const RequiredFilePolicy &candidate) {
                                  return candidate.name == name;
                                });
    if (policy == std::end(kRequiredFilePolicies) ||
        fileRecord->relativePath != policy->path)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric dependency file path mismatch for " + name);
    if (policy->executable && (fileRecord->mode & 0111) == 0)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric dependency file is not executable: " + name);
    if (!filePaths.insert(fileRecord->relativePath).second)
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric dependency file paths are not unique");
    result.files.push_back(std::move(*fileRecord));
  }

  const NumericDependencyFileRecord *managedM4 = result.findFile("m4");
  const NumericDependencyFileRecord *softFloat = result.findFile("softfloat");
  const NumericDependencyFileRecord *gmp = result.findFile("gmp");
  const NumericDependencyFileRecord *gmpLoader = result.findFile("gmp-soname");
  const NumericDependencyFileRecord *mpfr = result.findFile("mpfr");
  const NumericDependencyFileRecord *mpfrLoader =
      result.findFile("mpfr-soname");
  if (!managedM4 || !softFloat || !gmp || !gmpLoader || !mpfr || !mpfrLoader)
    return invalid(ErrorCode::ClosureMismatch,
                   "numeric dependency file closure is incomplete");
  if (managedM4->relativePath != result.build.managedM4RelativePath)
    return invalid(ErrorCode::PolicyMismatch,
                   "managed m4 file does not match build policy");
  if (llvm::sys::path::extension(softFloat->resolvedPath) != ".a")
    return invalid(ErrorCode::PolicyMismatch,
                   "SoftFloat file is not a static archive");
  if (llvm::Error error = validateSharedObjectPair("gmp", *gmp, *gmpLoader))
    return error;
  if (llvm::Error error = validateSharedObjectPair("mpfr", *mpfr, *mpfrLoader))
    return error;
  if (!mpfr->elf || !gmpLoader->elf || !gmpLoader->elf->soname ||
      !llvm::is_contained(mpfr->elf->needed, *gmpLoader->elf->soname))
    return invalid(ErrorCode::PolicyMismatch,
                   "MPFR ELF metadata does not require recorded GMP SONAME");
  std::vector<std::string> mpfrRuntimePaths = mpfr->elf->rpath;
  mpfrRuntimePaths.insert(mpfrRuntimePaths.end(), mpfr->elf->runpath.begin(),
                          mpfr->elf->runpath.end());
  llvm::SmallString<256> expectedGMPRuntimePath(root->resolved);
  llvm::sys::path::append(expectedGMPRuntimePath, "install", "gmp", "lib");
  if (mpfrRuntimePaths !=
      std::vector<std::string>{expectedGMPRuntimePath.str().str()})
    return invalid(ErrorCode::PolicyMismatch,
                   "MPFR ELF runtime path does not exactly name managed GMP");

  llvm::Expected<const llvm::json::Object *> licenses =
      requireObject(*(*top)->get("licenses"), "numeric dependency licenses");
  if (!licenses)
    return licenses.takeError();
  llvm::StringSet<> allowedLicenses;
  for (const LicensePolicy &license : kRequiredLicenses)
    allowedLicenses.insert(license.name);
  for (const auto &entry : **licenses)
    if (!allowedLicenses.contains(entry.first))
      return invalid(ErrorCode::UnknownField,
                     "numeric dependency licenses has unknown field '" +
                         entry.first.str() + "'");
  for (const LicensePolicy &policy : kRequiredLicenses) {
    const llvm::json::Value *licenseValue = (*licenses)->get(policy.name);
    if (!licenseValue)
      return invalid(ErrorCode::MissingField,
                     "numeric dependency licenses is missing field '" +
                         policy.name + "'");
    llvm::Expected<const llvm::json::Object *> license =
        requireObject(*licenseValue, "numeric license " + policy.name);
    if (!license)
      return license.takeError();
    if (llvm::Error error = requireExactKeys(
            **license, {"dependency", "source", "artifact", "sha256"},
            "numeric license " + policy.name))
      return error;
    llvm::Expected<std::string> dependency = requireString(
        *(*license)->get("dependency"), "numeric license dependency");
    if (!dependency)
      return dependency.takeError();
    llvm::Expected<std::string> source =
        requireString(*(*license)->get("source"), "numeric license source");
    if (!source)
      return source.takeError();
    llvm::Expected<std::string> fileName =
        requireString(*(*license)->get("artifact"), "numeric license file");
    if (!fileName)
      return fileName.takeError();
    llvm::Expected<std::string> licenseDigest =
        requireString(*(*license)->get("sha256"), "numeric license SHA256");
    if (!licenseDigest)
      return licenseDigest.takeError();
    if (*dependency != policy.dependency || *fileName != policy.name ||
        !isLowerSHA256(*licenseDigest))
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric license binding mismatch for " + policy.name);
    const NumericDependencySourceRecord *dependencySource =
        result.findSource(*dependency);
    const NumericDependencyFileRecord *licenseFile = result.findFile(*fileName);
    if (!dependencySource || !licenseFile ||
        licenseFile->sha256 != *licenseDigest || licenseFile->elf)
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric license file binding is incomplete");
    const std::string managedLicenseSource =
        dependencySource->sourceRelativePath + "/" + policy.source.str();
    if (*source != managedLicenseSource)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric license source mismatch for " + policy.name);
    llvm::Expected<ManagedPath> sourcePath = resolveManagedPath(
        *root, *source, RequiredFileType::RegularFile,
        "numeric license source " + policy.name, /*requireRelative=*/true);
    if (!sourcePath)
      return sourcePath.takeError();
    llvm::Expected<FileReadback> sourceReadback = readRegularFile(
        sourcePath->resolved, std::numeric_limits<uint64_t>::max(),
        "numeric license source " + policy.name);
    if (!sourceReadback)
      return sourceReadback.takeError();
    if (sourceReadback->digest != *licenseDigest)
      return invalid(ErrorCode::DigestMismatch,
                     "numeric license source/file digest mismatch for " +
                         policy.name);
    result.licenses.push_back({policy.name.str(), std::move(*dependency),
                               std::move(*source), sourcePath->resolved,
                               std::move(*fileName),
                               std::move(*licenseDigest)});
  }

  llvm::Expected<const llvm::json::Object *> conformance =
      requireObject(*(*top)->get("conformance"), "numeric conformance section");
  if (!conformance)
    return conformance.takeError();
  if (llvm::Error error = requireExactKeys(**conformance, {"policy", "gates"},
                                           "numeric conformance section"))
    return error;
  llvm::Expected<std::string> policy = requireString(
      *(*conformance)->get("policy"), "numeric conformance policy");
  if (!policy)
    return policy.takeError();
  if (*policy != kConformancePolicy)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric conformance policy mismatch");
  result.conformancePolicy = std::move(*policy);
  llvm::Expected<const llvm::json::Array *> gates =
      requireArray(*(*conformance)->get("gates"), "numeric conformance gates");
  if (!gates)
    return gates.takeError();
  llvm::StringSet<> gateNames;
  llvm::StringSet<> gateLogPaths;
  for (const llvm::json::Value &gateValue : **gates) {
    llvm::Expected<const llvm::json::Object *> gate =
        requireObject(gateValue, "numeric conformance gate");
    if (!gate)
      return gate.takeError();
    if (llvm::Error error = requireExactKeys(
            **gate,
            {"name", "command", "cwd", "environment", "exit_code", "log"},
            "numeric conformance gate"))
      return error;
    llvm::Expected<std::string> name =
        requireString(*(*gate)->get("name"), "numeric conformance gate name");
    if (!name)
      return name.takeError();
    if (!gateNames.insert(*name).second)
      return invalid(ErrorCode::ClosureMismatch,
                     "duplicate numeric conformance gate '" + *name + "'");
    llvm::Expected<std::vector<std::string>> command = requireStringArray(
        *(*gate)->get("command"), "numeric conformance command");
    if (!command)
      return command.takeError();
    llvm::Expected<std::string> cwd =
        requireString(*(*gate)->get("cwd"), "numeric conformance cwd");
    if (!cwd)
      return cwd.takeError();
    llvm::Expected<std::string> environment = requireString(
        *(*gate)->get("environment"), "numeric conformance environment");
    if (!environment)
      return environment.takeError();
    llvm::Expected<uint64_t> exitCode = requireUnsigned(
        *(*gate)->get("exit_code"), "numeric conformance exit_code");
    if (!exitCode)
      return exitCode.takeError();
    if (*exitCode > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return invalid(ErrorCode::TypeMismatch,
                     "numeric conformance exit_code is out of range");
    llvm::Expected<NumericDependencyFileRecord> log =
        parseFileRecord("conformance-log:" + *name, *(*gate)->get("log"), *root,
                        readelfTool->resolvedPath);
    if (!log)
      return log.takeError();
    if (log->relativePath != "conformance/" + *name + ".log")
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric conformance log path mismatch for " + *name);
    if (log->size == 0)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric conformance log is empty for " + *name);
    if (!gateLogPaths.insert(log->relativePath).second)
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric conformance log paths are not unique");
    NumericDependencyConformanceGateRecord gateRecord{
        std::move(*name),
        std::move(*command),
        std::move(*cwd),
        std::move(*environment),
        static_cast<int64_t>(*exitCode),
        std::move(*log)};
    if (llvm::Error error = validateGateContract(result, gateRecord))
      return error;
    result.gates.push_back(std::move(gateRecord));
  }
  llvm::StringSet<> requiredGateNames;
  for (llvm::StringRef name : kRequiredGates)
    requiredGateNames.insert(name);
  if (gateNames.size() != requiredGateNames.size())
    return invalid(ErrorCode::ClosureMismatch,
                   "numeric conformance gate closure mismatch");
  for (llvm::StringRef name : kRequiredGates)
    if (!gateNames.contains(name))
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric conformance gate closure is missing " + name);

  // Re-read the record after all external evidence.  A concurrent record
  // replacement cannot produce a verified snapshot under the old digest.
  llvm::Expected<FileReadback> finalRecordReadback =
      readRegularFile(resolvedRecord->resolved, limits.maxRecordBytes,
                      "numeric dependency record");
  if (!finalRecordReadback)
    return finalRecordReadback.takeError();
  if (finalRecordReadback->digest != expectedRecordSHA256)
    return invalid(ErrorCode::RecordDigestMismatch,
                   "numeric dependency record changed during verification");
  return result;
}

} // namespace wafer
