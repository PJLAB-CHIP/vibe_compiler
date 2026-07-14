//===- NumericDependencyConformance.cpp - Numeric dependency identity -----===//

#include "Wafer/Target/NumericDependencyConformance.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace wafer {
namespace {

using ErrorCode = NumericDependencyConformanceErrorCode;

constexpr llvm::StringLiteral kRecordKind = "wafer-numeric-model-deps";
constexpr llvm::StringLiteral kRecordStatus = "conformance-passed";
constexpr llvm::StringLiteral kConformancePolicy =
    "wafer-numeric-model-conformance-v1";

constexpr llvm::StringLiteral kRequiredPins[] = {"softfloat", "testfloat", "m4",
                                                 "gmp", "mpfr"};

constexpr llvm::StringLiteral kRequiredArtifacts[] = {
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

constexpr ArtifactPathPolicy kArtifactPathPolicies[] = {
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

constexpr llvm::StringLiteral kRequiredGates[] = {
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

constexpr llvm::StringLiteral kRequiredTools[] = {
    "cc", "cxx", "make",    "ar",    "ranlib",
    "nm", "ld",  "readelf", "shell", "false"};

constexpr llvm::StringLiteral kRequiredEnvironments[] = {
    "base", "managed", "mpfr", "version-runtime"};

struct LicensePolicy {
  llvm::StringLiteral name;
  llvm::StringLiteral dependency;
  llvm::StringLiteral source;
};

constexpr LicensePolicy kRequiredLicenses[] = {
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

llvm::Error invalid(ErrorCode code, const llvm::Twine &detail) {
  return llvm::createStringError(
      llvm::errc::invalid_argument, "numeric dependency conformance %s: %s",
      stringifyNumericDependencyConformanceErrorCode(code).str().c_str(),
      detail.str().c_str());
}

bool isLowerSHA256(llvm::StringRef digest) {
  return digest.size() == 64 && llvm::all_of(digest, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool isLowerHex(llvm::StringRef value) {
  return !value.empty() && llvm::all_of(value, [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

std::string sha256(llvm::StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

struct FileReadback {
  std::string digest;
  std::string contents;
  uint64_t size = 0;
  uint32_t mode = 0;
};

llvm::Expected<FileReadback> readRegularFile(llvm::StringRef path,
                                             uint64_t maximumBytes,
                                             const llvm::Twine &label,
                                             bool captureContents = false) {
  const std::string pathStorage = path.str();
  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  const int descriptor = ::open(pathStorage.c_str(), flags);
  if (descriptor < 0)
    return invalid(ErrorCode::IO,
                   label + " cannot be opened: " + std::strerror(errno));

  struct DescriptorCloser {
    int descriptor;
    ~DescriptorCloser() { ::close(descriptor); }
  } closer{descriptor};
  struct stat before{};
  if (::fstat(descriptor, &before) != 0)
    return invalid(ErrorCode::IO,
                   label + " cannot be statted: " + std::strerror(errno));
  if (!S_ISREG(before.st_mode))
    return invalid(ErrorCode::FileType, label + " is not a regular file");
  if (before.st_size < 0 ||
      static_cast<uint64_t>(before.st_size) > maximumBytes)
    return invalid(ErrorCode::ResourceLimit, label + " exceeds byte limit");

  llvm::SHA256 hasher;
  std::string contents;
  if (captureContents)
    contents.reserve(static_cast<size_t>(before.st_size));
  std::vector<uint8_t> buffer(1024 * 1024);
  uint64_t total = 0;
  while (true) {
    const ssize_t bytesRead = ::read(descriptor, buffer.data(), buffer.size());
    if (bytesRead < 0) {
      if (errno == EINTR)
        continue;
      return invalid(ErrorCode::IO,
                     label + " cannot be read: " + std::strerror(errno));
    }
    if (bytesRead == 0)
      break;
    if (static_cast<uint64_t>(bytesRead) > maximumBytes - total)
      return invalid(ErrorCode::ResourceLimit, label + " exceeds byte limit");
    total += static_cast<uint64_t>(bytesRead);
    hasher.update(
        llvm::ArrayRef<uint8_t>(buffer.data(), static_cast<size_t>(bytesRead)));
    if (captureContents)
      contents.append(reinterpret_cast<const char *>(buffer.data()),
                      static_cast<size_t>(bytesRead));
  }

  struct stat after{};
  if (::fstat(descriptor, &after) != 0)
    return invalid(ErrorCode::IO,
                   label + " cannot be restatted: " + std::strerror(errno));
  if (before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_mode != after.st_mode ||
      total != static_cast<uint64_t>(before.st_size))
    return invalid(ErrorCode::IO, label + " changed while being read");
  return FileReadback{llvm::toHex(hasher.final(), /*LowerCase=*/true),
                      std::move(contents), total,
                      static_cast<uint32_t>(before.st_mode & 07777)};
}

llvm::Expected<std::string>
runAndCapture(llvm::StringRef program, llvm::ArrayRef<std::string> arguments,
              llvm::ArrayRef<std::string> environment,
              const llvm::Twine &label) {
  constexpr size_t maximumOutputBytes = 16 * 1024 * 1024;
  const std::string stableLabel = label.str();
  const std::string programStorage = program.str();
  if (arguments.empty() || arguments.front() != program)
    return invalid(ErrorCode::InvalidArgument,
                   stableLabel + " has an invalid argument vector");

  std::vector<char *> argumentPointers;
  argumentPointers.reserve(arguments.size() + 1);
  for (const std::string &argument : arguments)
    argumentPointers.push_back(const_cast<char *>(argument.c_str()));
  argumentPointers.push_back(nullptr);
  std::vector<char *> environmentPointers;
  environmentPointers.reserve(environment.size() + 1);
  for (const std::string &entry : environment)
    environmentPointers.push_back(const_cast<char *>(entry.c_str()));
  environmentPointers.push_back(nullptr);

  int descriptors[2];
  if (::pipe(descriptors) != 0)
    return invalid(ErrorCode::IO, stableLabel +
                                      " output pipe cannot be created: " +
                                      std::strerror(errno));
  const pid_t child = ::fork();
  if (child < 0) {
    ::close(descriptors[0]);
    ::close(descriptors[1]);
    return invalid(ErrorCode::IO, stableLabel + " process cannot be created: " +
                                      std::strerror(errno));
  }
  if (child == 0) {
    ::close(descriptors[0]);
    if (::dup2(descriptors[1], STDOUT_FILENO) < 0 ||
        ::dup2(descriptors[1], STDERR_FILENO) < 0)
      ::_exit(126);
    ::close(descriptors[1]);
    ::execve(programStorage.c_str(), argumentPointers.data(),
             environmentPointers.data());
    ::_exit(127);
  }

  ::close(descriptors[1]);
  std::string output;
  output.reserve(4096);
  std::vector<char> buffer(64 * 1024);
  bool exceededLimit = false;
  bool readFailed = false;
  int readError = 0;
  while (true) {
    const ssize_t count = ::read(descriptors[0], buffer.data(), buffer.size());
    if (count < 0) {
      if (errno == EINTR)
        continue;
      readFailed = true;
      readError = errno;
      break;
    }
    if (count == 0)
      break;
    if (static_cast<size_t>(count) > maximumOutputBytes - output.size()) {
      exceededLimit = true;
      break;
    }
    output.append(buffer.data(), static_cast<size_t>(count));
  }
  ::close(descriptors[0]);
  if (exceededLimit || readFailed)
    ::kill(child, SIGKILL);

  int status = 0;
  while (::waitpid(child, &status, 0) < 0) {
    if (errno == EINTR)
      continue;
    return invalid(
        ErrorCode::IO,
        stableLabel + " process cannot be waited for: " + std::strerror(errno));
  }
  if (readFailed)
    return invalid(ErrorCode::IO, stableLabel + " output cannot be read: " +
                                      std::strerror(readError));
  if (exceededLimit)
    return invalid(ErrorCode::ResourceLimit,
                   stableLabel + " output exceeds byte limit");
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    return invalid(ErrorCode::PolicyMismatch,
                   stableLabel + " did not exit successfully");
  return output;
}

std::vector<std::string> commandEnvironment(llvm::StringRef program) {
  return {"LC_ALL=C", "LANG=C",
          ("PATH=" + llvm::sys::path::parent_path(program)).str()};
}

struct ManagedRoot {
  std::string requested;
  std::string resolved;
};

void removeTrailingSeparators(llvm::SmallVectorImpl<char> &path) {
  while (path.size() > 1 && llvm::sys::path::is_separator(path.back()))
    path.pop_back();
}

llvm::Expected<ManagedRoot> verifyManagedRoot(llvm::StringRef input) {
  if (input.empty() || input.contains('\0') ||
      !llvm::sys::path::is_absolute(input))
    return invalid(ErrorCode::InvalidArgument,
                   "managed root must be an absolute path");

  llvm::SmallString<256> requested(input);
  llvm::sys::path::remove_dots(requested, /*remove_dot_dot=*/true);
  removeTrailingSeparators(requested);

  llvm::SmallString<256> candidate("/");
  llvm::sys::fs::file_status status;
  llvm::SmallVector<llvm::StringRef, 16> components;
  llvm::StringRef(requested).split(components, '/', /*MaxSplit=*/-1,
                                   /*KeepEmpty=*/false);
  if (components.empty()) {
    if (std::error_code error =
            llvm::sys::fs::status(candidate, status, /*follow=*/false))
      return invalid(ErrorCode::IO,
                     "managed root cannot be statted: " + error.message());
  } else {
    for (llvm::StringRef component : components) {
      llvm::sys::path::append(candidate, component);
      if (std::error_code error =
              llvm::sys::fs::status(candidate, status, /*follow=*/false))
        return invalid(ErrorCode::IO,
                       "managed root component cannot be statted: " +
                           error.message());
      if (status.type() == llvm::sys::fs::file_type::symlink_file)
        return invalid(ErrorCode::Symlink,
                       "managed root contains a symlink component");
      if (status.type() != llvm::sys::fs::file_type::directory_file)
        return invalid(ErrorCode::FileType,
                       "managed root contains a non-directory component");
    }
  }

  llvm::SmallString<256> resolved;
  if (std::error_code error = llvm::sys::fs::real_path(requested, resolved))
    return invalid(ErrorCode::IO,
                   "managed root cannot be resolved: " + error.message());
  removeTrailingSeparators(resolved);
  if (requested != resolved)
    return invalid(ErrorCode::UnsafePath,
                   "managed root is not a canonical absolute path");
  return ManagedRoot{requested.str().str(), resolved.str().str()};
}

llvm::Error validateRelativePath(llvm::StringRef relative,
                                 const llvm::Twine &label) {
  if (relative.empty() || relative.contains('\0') || relative.contains('\\') ||
      llvm::sys::path::is_absolute(relative))
    return invalid(ErrorCode::UnsafePath,
                   label + " is not a safe relative path");
  llvm::SmallVector<llvm::StringRef, 16> components;
  relative.split(components, '/', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  for (llvm::StringRef component : components)
    if (component.empty() || component == "." || component == "..")
      return invalid(ErrorCode::UnsafePath,
                     label + " contains an unsafe path component");
  return llvm::Error::success();
}

std::optional<llvm::StringRef> relativeToPrefix(llvm::StringRef path,
                                                llvm::StringRef prefix) {
  if (prefix == "/") {
    if (!path.starts_with('/'))
      return std::nullopt;
    return path.drop_front();
  }
  if (!path.starts_with(prefix) || path.size() <= prefix.size() ||
      !llvm::sys::path::is_separator(path[prefix.size()]))
    return std::nullopt;
  return path.drop_front(prefix.size() + 1);
}

struct ManagedPath {
  std::string relative;
  std::string resolved;
  llvm::sys::fs::file_status status;
};

enum class RequiredFileType { RegularFile, Directory };

llvm::Expected<std::string>
resolveAbsoluteRegularPath(llvm::StringRef input, const llvm::Twine &label) {
  if (input.empty() || !llvm::sys::path::is_absolute(input) ||
      input.contains('\0') || input.contains('\\'))
    return invalid(ErrorCode::UnsafePath,
                   label + " is not a safe absolute path");
  llvm::SmallVector<llvm::StringRef, 16> components;
  input.split(components, '/', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  llvm::SmallString<256> candidate("/");
  bool sawComponent = false;
  llvm::sys::fs::file_status status;
  for (llvm::StringRef component : components) {
    if (component.empty() && !sawComponent)
      continue;
    sawComponent = true;
    if (component.empty() || component == "." || component == "..")
      return invalid(ErrorCode::UnsafePath,
                     label + " contains an unsafe path component");
    llvm::sys::path::append(candidate, component);
    if (std::error_code error =
            llvm::sys::fs::status(candidate, status, /*follow=*/false))
      return invalid(ErrorCode::IO,
                     label + " cannot be statted: " + error.message());
    if (status.type() == llvm::sys::fs::file_type::symlink_file)
      return invalid(ErrorCode::Symlink,
                     label + " contains a symlink component");
  }
  if (!sawComponent || status.type() != llvm::sys::fs::file_type::regular_file)
    return invalid(ErrorCode::FileType, label + " is not a regular file");
  llvm::SmallString<256> resolved;
  if (std::error_code error = llvm::sys::fs::real_path(candidate, resolved))
    return invalid(ErrorCode::IO,
                   label + " cannot be resolved: " + error.message());
  return resolved.str().str();
}

llvm::Expected<ManagedPath> resolveManagedPath(const ManagedRoot &root,
                                               llvm::StringRef input,
                                               RequiredFileType requiredType,
                                               const llvm::Twine &label,
                                               bool requireRelative) {
  llvm::StringRef relative = input;
  if (llvm::sys::path::is_absolute(input)) {
    if (requireRelative)
      return invalid(ErrorCode::UnsafePath,
                     label + " must be relative to the managed root");
    std::optional<llvm::StringRef> fromRequested =
        relativeToPrefix(input, root.requested);
    std::optional<llvm::StringRef> fromResolved =
        relativeToPrefix(input, root.resolved);
    if (!fromRequested && !fromResolved)
      return invalid(ErrorCode::UnsafePath, label + " escapes managed root");
    relative = fromRequested ? *fromRequested : *fromResolved;
  }
  if (llvm::Error error = validateRelativePath(relative, label))
    return error;

  llvm::SmallVector<llvm::StringRef, 16> components;
  relative.split(components, '/', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  llvm::SmallString<256> candidate(root.resolved);
  llvm::sys::fs::file_status status;
  for (size_t index = 0; index < components.size(); ++index) {
    llvm::sys::path::append(candidate, components[index]);
    if (std::error_code error =
            llvm::sys::fs::status(candidate, status, /*follow=*/false))
      return invalid(ErrorCode::IO,
                     label + " cannot be statted: " + error.message());
    if (status.type() == llvm::sys::fs::file_type::symlink_file)
      return invalid(ErrorCode::Symlink,
                     label + " contains a symlink component");
    if (index + 1 != components.size() &&
        status.type() != llvm::sys::fs::file_type::directory_file)
      return invalid(ErrorCode::FileType,
                     label + " has a non-directory parent component");
  }

  llvm::sys::fs::file_type expected =
      requiredType == RequiredFileType::RegularFile
          ? llvm::sys::fs::file_type::regular_file
          : llvm::sys::fs::file_type::directory_file;
  if (status.type() != expected)
    return invalid(ErrorCode::FileType,
                   label + (requiredType == RequiredFileType::RegularFile
                                ? " is not a regular file"
                                : " is not a directory"));

  llvm::SmallString<256> resolved;
  if (std::error_code error = llvm::sys::fs::real_path(candidate, resolved))
    return invalid(ErrorCode::IO,
                   label + " cannot be resolved: " + error.message());
  if (!relativeToPrefix(resolved, root.resolved))
    return invalid(ErrorCode::UnsafePath,
                   label + " resolves outside managed root");
  return ManagedPath{relative.str(), resolved.str().str(), status};
}

llvm::Error requireExactKeys(const llvm::json::Object &object,
                             std::initializer_list<llvm::StringRef> expected,
                             const llvm::Twine &label) {
  llvm::StringSet<> allowed;
  for (llvm::StringRef key : expected)
    allowed.insert(key);
  for (const auto &entry : object)
    if (!allowed.contains(entry.first))
      return invalid(ErrorCode::UnknownField,
                     label + " has unknown field '" + entry.first.str() + "'");
  for (llvm::StringRef key : expected)
    if (!object.get(key))
      return invalid(ErrorCode::MissingField,
                     label + " is missing field '" + key + "'");
  return llvm::Error::success();
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Value &value, const llvm::Twine &label) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return invalid(ErrorCode::TypeMismatch, label + " must be an object");
  return object;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Value &value, const llvm::Twine &label) {
  const llvm::json::Array *array = value.getAsArray();
  if (!array)
    return invalid(ErrorCode::TypeMismatch, label + " must be an array");
  return array;
}

llvm::Expected<std::string> requireString(const llvm::json::Value &value,
                                          const llvm::Twine &label) {
  std::optional<llvm::StringRef> string = value.getAsString();
  if (!string)
    return invalid(ErrorCode::TypeMismatch, label + " must be a string");
  return string->str();
}

llvm::Expected<uint64_t> requireUnsigned(const llvm::json::Value &value,
                                         const llvm::Twine &label) {
  std::optional<int64_t> integer = value.getAsInteger();
  if (!integer || *integer < 0)
    return invalid(ErrorCode::TypeMismatch,
                   label + " must be a non-negative integer");
  return static_cast<uint64_t>(*integer);
}

llvm::Expected<std::vector<std::string>>
requireStringArray(const llvm::json::Value &value, const llvm::Twine &label) {
  llvm::Expected<const llvm::json::Array *> array = requireArray(value, label);
  if (!array)
    return array.takeError();
  std::vector<std::string> result;
  result.reserve((*array)->size());
  for (size_t index = 0; index < (*array)->size(); ++index) {
    llvm::Expected<std::string> item =
        requireString((**array)[index], label + " item");
    if (!item)
      return item.takeError();
    result.push_back(std::move(*item));
  }
  return result;
}

llvm::Expected<std::optional<std::string>>
requireNullableString(const llvm::json::Value &value,
                      const llvm::Twine &label) {
  if (value.getAsNull())
    return std::optional<std::string>();
  llvm::Expected<std::string> string = requireString(value, label);
  if (!string)
    return string.takeError();
  return std::optional<std::string>(std::move(*string));
}

llvm::Error expectString(llvm::StringRef actual, llvm::StringRef expected,
                         const llvm::Twine &label) {
  if (actual != expected)
    return invalid(ErrorCode::PolicyMismatch,
                   label + " does not match frozen policy");
  return llvm::Error::success();
}

llvm::Error expectStringArray(llvm::ArrayRef<std::string> actual,
                              std::initializer_list<llvm::StringRef> expected,
                              const llvm::Twine &label) {
  if (actual.size() != expected.size())
    return invalid(ErrorCode::PolicyMismatch,
                   label + " does not match frozen policy");
  size_t index = 0;
  for (llvm::StringRef item : expected)
    if (actual[index++] != item)
      return invalid(ErrorCode::PolicyMismatch,
                     label + " does not match frozen policy");
  return llvm::Error::success();
}

/// LLVM's JSON object intentionally keeps the last value for a repeated key.
/// The managed identity format is fail-closed, so scan the already
/// syntax-validated document and reject duplicate decoded keys at every depth.
class DuplicateKeyScanner {
public:
  explicit DuplicateKeyScanner(llvm::StringRef input) : input(input) {}

  llvm::Error scan() {
    if (llvm::Error error = parseValue(/*depth=*/0))
      return error;
    skipWhitespace();
    if (offset != input.size())
      return invalid(ErrorCode::JSONSyntax,
                     "record contains trailing JSON data");
    return llvm::Error::success();
  }

private:
  void skipWhitespace() {
    while (offset < input.size() &&
           std::isspace(static_cast<unsigned char>(input[offset])))
      ++offset;
  }

  llvm::Error expect(char character) {
    skipWhitespace();
    if (offset >= input.size() || input[offset] != character)
      return invalid(ErrorCode::JSONSyntax, "record JSON token scan failed");
    ++offset;
    return llvm::Error::success();
  }

  llvm::Expected<std::string> parseString() {
    skipWhitespace();
    if (offset >= input.size() || input[offset] != '"')
      return invalid(ErrorCode::JSONSyntax, "record JSON string scan failed");
    const size_t start = offset++;
    bool escaped = false;
    while (offset < input.size()) {
      const char character = input[offset++];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (character == '\\') {
        escaped = true;
        continue;
      }
      if (character == '"') {
        llvm::Expected<llvm::json::Value> decoded =
            llvm::json::parse(input.slice(start, offset));
        if (!decoded) {
          llvm::consumeError(decoded.takeError());
          return invalid(ErrorCode::JSONSyntax,
                         "record JSON string cannot be decoded");
        }
        std::optional<llvm::StringRef> value = decoded->getAsString();
        if (!value)
          return invalid(ErrorCode::JSONSyntax,
                         "record JSON key is not a string");
        return value->str();
      }
    }
    return invalid(ErrorCode::JSONSyntax, "record JSON string is unterminated");
  }

  llvm::Error parseObject(unsigned depth) {
    if (llvm::Error error = expect('{'))
      return error;
    llvm::StringSet<> keys;
    skipWhitespace();
    if (offset < input.size() && input[offset] == '}') {
      ++offset;
      return llvm::Error::success();
    }
    while (true) {
      llvm::Expected<std::string> key = parseString();
      if (!key)
        return key.takeError();
      if (!keys.insert(*key).second)
        return invalid(ErrorCode::DuplicateField,
                       "record contains duplicate field '" + *key + "'");
      if (llvm::Error error = expect(':'))
        return error;
      if (llvm::Error error = parseValue(depth + 1))
        return error;
      skipWhitespace();
      if (offset < input.size() && input[offset] == '}') {
        ++offset;
        return llvm::Error::success();
      }
      if (llvm::Error error = expect(','))
        return error;
    }
  }

  llvm::Error parseArray(unsigned depth) {
    if (llvm::Error error = expect('['))
      return error;
    skipWhitespace();
    if (offset < input.size() && input[offset] == ']') {
      ++offset;
      return llvm::Error::success();
    }
    while (true) {
      if (llvm::Error error = parseValue(depth + 1))
        return error;
      skipWhitespace();
      if (offset < input.size() && input[offset] == ']') {
        ++offset;
        return llvm::Error::success();
      }
      if (llvm::Error error = expect(','))
        return error;
    }
  }

  llvm::Error parseValue(unsigned depth) {
    if (depth > 128)
      return invalid(ErrorCode::ResourceLimit,
                     "record JSON nesting exceeds limit");
    skipWhitespace();
    if (offset >= input.size())
      return invalid(ErrorCode::JSONSyntax, "record JSON value is missing");
    if (input[offset] == '{')
      return parseObject(depth);
    if (input[offset] == '[')
      return parseArray(depth);
    if (input[offset] == '"') {
      llvm::Expected<std::string> ignored = parseString();
      if (!ignored)
        return ignored.takeError();
      return llvm::Error::success();
    }

    const size_t start = offset;
    while (offset < input.size() && input[offset] != ',' &&
           input[offset] != '}' && input[offset] != ']' &&
           !std::isspace(static_cast<unsigned char>(input[offset])))
      ++offset;
    if (offset == start)
      return invalid(ErrorCode::JSONSyntax,
                     "record JSON primitive scan failed");
    return llvm::Error::success();
  }

  llvm::StringRef input;
  size_t offset = 0;
};

struct TreeEntry {
  std::string relative;
  bool executable = false;
  std::string digest;
  uint64_t size = 0;
};

llvm::Expected<std::string>
digestSourceTree(llvm::StringRef sourceRoot,
                 const NumericDependencyReadLimits &limits,
                 const llvm::Twine &label) {
  const std::string labelStorage = label.str();
  const llvm::StringRef stableLabel(labelStorage);
  std::vector<TreeEntry> entries;
  uint64_t totalBytes = 0;
  std::error_code iteratorError;
  llvm::sys::fs::recursive_directory_iterator iterator(
      sourceRoot, iteratorError, /*follow_symlinks=*/false);
  llvm::sys::fs::recursive_directory_iterator end;
  if (iteratorError)
    return invalid(ErrorCode::IO, stableLabel + " cannot be enumerated: " +
                                      iteratorError.message());

  while (iterator != end) {
    const std::string path = iterator->path();
    llvm::sys::fs::file_status status;
    if (std::error_code error =
            llvm::sys::fs::status(path, status, /*follow=*/false))
      return invalid(ErrorCode::IO, stableLabel + " entry cannot be statted: " +
                                        error.message());
    if (status.type() == llvm::sys::fs::file_type::symlink_file)
      return invalid(ErrorCode::Symlink, stableLabel + " contains a symlink");
    if (status.type() != llvm::sys::fs::file_type::directory_file &&
        status.type() != llvm::sys::fs::file_type::regular_file)
      return invalid(ErrorCode::FileType,
                     stableLabel + " contains a special file");

    if (status.type() == llvm::sys::fs::file_type::regular_file) {
      if (entries.size() >= limits.maxSourceFiles)
        return invalid(ErrorCode::ResourceLimit,
                       stableLabel + " exceeds source-file limit");
      llvm::Expected<FileReadback> readback = readRegularFile(
          path, limits.maxSourceBytes - totalBytes, stableLabel + " file");
      if (!readback)
        return readback.takeError();
      if (readback->size > limits.maxSourceBytes - totalBytes)
        return invalid(ErrorCode::ResourceLimit,
                       stableLabel + " exceeds source-byte limit");
      totalBytes += readback->size;
      std::optional<llvm::StringRef> relative =
          relativeToPrefix(path, sourceRoot);
      if (!relative)
        return invalid(ErrorCode::UnsafePath,
                       stableLabel + " enumeration escaped source root");
      const bool executable = (status.permissions() & llvm::sys::fs::all_exe) !=
                              llvm::sys::fs::no_perms;
      entries.push_back({relative->str(), executable,
                         std::move(readback->digest), readback->size});
    }

    iterator.increment(iteratorError);
    if (iteratorError)
      return invalid(ErrorCode::IO, stableLabel + " cannot be enumerated: " +
                                        iteratorError.message());
  }
  if (entries.empty())
    return invalid(ErrorCode::ClosureMismatch,
                   stableLabel + " source tree is empty");

  llvm::sort(entries, [](const TreeEntry &lhs, const TreeEntry &rhs) {
    return lhs.relative < rhs.relative;
  });
  llvm::SHA256 hasher;
  for (const TreeEntry &entry : entries) {
    hasher.update(entry.relative);
    hasher.update(llvm::StringRef("\0", 1));
    hasher.update(entry.executable ? "1" : "0");
    hasher.update(llvm::StringRef("\0", 1));
    const std::string binaryDigest = llvm::fromHex(entry.digest);
    hasher.update(llvm::StringRef(binaryDigest.data(), binaryDigest.size()));
    hasher.update("\n");
  }
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

llvm::Expected<bool> hasELFMagic(llvm::StringRef path,
                                 const llvm::Twine &label) {
  const std::string pathStorage = path.str();
  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  const int descriptor = ::open(pathStorage.c_str(), flags);
  if (descriptor < 0)
    return invalid(ErrorCode::IO, label +
                                      " cannot be opened for ELF inspection: " +
                                      std::strerror(errno));
  char magic[4] = {};
  size_t offset = 0;
  while (offset != sizeof(magic)) {
    const ssize_t count =
        ::read(descriptor, magic + offset, sizeof(magic) - offset);
    if (count < 0) {
      if (errno == EINTR)
        continue;
      const int readError = errno;
      ::close(descriptor);
      return invalid(ErrorCode::IO, label +
                                        " cannot be read for ELF inspection: " +
                                        std::strerror(readError));
    }
    if (count == 0)
      break;
    offset += static_cast<size_t>(count);
  }
  ::close(descriptor);
  return offset == sizeof(magic) && magic[0] == '\x7f' && magic[1] == 'E' &&
         magic[2] == 'L' && magic[3] == 'F';
}

llvm::Expected<std::string> findELFHeader(llvm::StringRef output,
                                          llvm::StringRef name,
                                          const llvm::Twine &label) {
  llvm::SmallVector<llvm::StringRef, 64> lines;
  output.split(lines, '\n');
  const std::string prefix = (name + ":").str();
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (line.consume_front(prefix))
      return line.trim().str();
  }
  return invalid(ErrorCode::PolicyMismatch,
                 label + " readelf output is missing " + name);
}

std::vector<std::string> findELFDynamicValues(llvm::StringRef output,
                                              llvm::StringRef tag,
                                              bool splitPaths) {
  llvm::SmallVector<llvm::StringRef, 64> lines;
  output.split(lines, '\n');
  const std::string marker = ("(" + tag + ")").str();
  std::vector<std::string> values;
  for (llvm::StringRef line : lines) {
    const size_t markerPosition = line.find(marker);
    if (markerPosition == llvm::StringRef::npos)
      continue;
    const size_t open = line.find('[', markerPosition + marker.size());
    const size_t close = open == llvm::StringRef::npos
                             ? llvm::StringRef::npos
                             : line.find(']', open + 1);
    if (open == llvm::StringRef::npos || close == llvm::StringRef::npos)
      continue;
    llvm::StringRef value = line.slice(open + 1, close);
    if (!splitPaths) {
      values.push_back(value.str());
      continue;
    }
    llvm::SmallVector<llvm::StringRef, 8> paths;
    value.split(paths, ':', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (llvm::StringRef path : paths)
      values.push_back(path.str());
  }
  return values;
}

llvm::Expected<NumericDependencyELFIdentity>
inspectELFIdentity(llvm::StringRef path, llvm::StringRef readelfPath,
                   const llvm::Twine &label) {
  const std::vector<std::string> arguments = {readelfPath.str(), "--wide",
                                              "--file-header",   "--notes",
                                              "--dynamic",       path.str()};
  const std::vector<std::string> environment = commandEnvironment(readelfPath);
  llvm::Expected<std::string> output = runAndCapture(
      readelfPath, arguments, environment, label + " readelf inspection");
  if (!output)
    return output.takeError();

  NumericDependencyELFIdentity result;
  llvm::Expected<std::string> elfClass = findELFHeader(*output, "Class", label);
  if (!elfClass)
    return elfClass.takeError();
  llvm::Expected<std::string> data = findELFHeader(*output, "Data", label);
  if (!data)
    return data.takeError();
  llvm::Expected<std::string> type = findELFHeader(*output, "Type", label);
  if (!type)
    return type.takeError();
  llvm::Expected<std::string> machine =
      findELFHeader(*output, "Machine", label);
  if (!machine)
    return machine.takeError();
  result.elfClass = std::move(*elfClass);
  if (llvm::StringRef(*data).contains("little endian"))
    result.byteOrder = "little";
  else if (llvm::StringRef(*data).contains("big endian"))
    result.byteOrder = "big";
  else
    return invalid(ErrorCode::PolicyMismatch,
                   label + " readelf output has unknown byte order");
  result.type = llvm::StringRef(*type).ltrim().split(' ').first.str();
  result.machine = std::move(*machine);

  const size_t buildMarker = output->find("Build ID:");
  if (buildMarker != std::string::npos) {
    llvm::StringRef build =
        llvm::StringRef(*output)
            .drop_front(buildMarker + llvm::StringRef("Build ID:").size())
            .ltrim();
    build = build.take_while(llvm::isHexDigit);
    if (!build.empty()) {
      std::string lowered = build.str();
      llvm::transform(lowered, lowered.begin(), [](char character) {
        return static_cast<char>(
            std::tolower(static_cast<unsigned char>(character)));
      });
      result.buildId = std::move(lowered);
    }
  }
  std::vector<std::string> sonames =
      findELFDynamicValues(*output, "SONAME", /*splitPaths=*/false);
  if (!sonames.empty())
    result.soname = std::move(sonames.front());
  result.needed = findELFDynamicValues(*output, "NEEDED", /*splitPaths=*/false);
  result.rpath = findELFDynamicValues(*output, "RPATH", /*splitPaths=*/true);
  result.runpath =
      findELFDynamicValues(*output, "RUNPATH", /*splitPaths=*/true);
  return result;
}

bool equalELFIdentity(const NumericDependencyELFIdentity &lhs,
                      const NumericDependencyELFIdentity &rhs) {
  return lhs.elfClass == rhs.elfClass && lhs.byteOrder == rhs.byteOrder &&
         lhs.type == rhs.type && lhs.machine == rhs.machine &&
         lhs.buildId == rhs.buildId && lhs.soname == rhs.soname &&
         lhs.needed == rhs.needed && lhs.rpath == rhs.rpath &&
         lhs.runpath == rhs.runpath;
}

llvm::Expected<NumericDependencyArtifactIdentity>
parseArtifactIdentity(llvm::StringRef name, const llvm::json::Value &value,
                      const ManagedRoot &root, llvm::StringRef readelfPath) {
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, "numeric artifact " + name);
  if (!object)
    return object.takeError();
  if (llvm::Error error = requireExactKeys(
          **object, {"path", "sha256", "size", "file_type", "mode", "elf"},
          "numeric artifact " + name))
    return error;

  llvm::Expected<std::string> relative =
      requireString(*(*object)->get("path"), "numeric artifact path");
  if (!relative)
    return relative.takeError();
  llvm::Expected<std::string> digest =
      requireString(*(*object)->get("sha256"), "numeric artifact SHA256");
  if (!digest)
    return digest.takeError();
  if (!isLowerSHA256(*digest))
    return invalid(ErrorCode::TypeMismatch,
                   "numeric artifact SHA256 is not lowercase SHA-256");
  llvm::Expected<uint64_t> size =
      requireUnsigned(*(*object)->get("size"), "numeric artifact size");
  if (!size)
    return size.takeError();
  llvm::Expected<std::string> fileType =
      requireString(*(*object)->get("file_type"), "numeric artifact file_type");
  if (!fileType)
    return fileType.takeError();
  if (*fileType != "regular")
    return invalid(ErrorCode::FileType,
                   "numeric artifact file_type is not regular");
  llvm::Expected<uint64_t> mode =
      requireUnsigned(*(*object)->get("mode"), "numeric artifact mode");
  if (!mode)
    return mode.takeError();
  if (*mode > 07777)
    return invalid(ErrorCode::TypeMismatch,
                   "numeric artifact mode is out of range");

  std::optional<NumericDependencyELFIdentity> elf;
  const llvm::json::Value &elfValue = *(*object)->get("elf");
  if (!elfValue.getAsNull()) {
    llvm::Expected<const llvm::json::Object *> elfObject =
        requireObject(elfValue, "numeric artifact ELF identity");
    if (!elfObject)
      return elfObject.takeError();
    if (llvm::Error error = requireExactKeys(**elfObject,
                                             {"class", "byte_order", "type",
                                              "machine", "build_id", "soname",
                                              "needed", "rpath", "runpath"},
                                             "numeric artifact ELF identity"))
      return error;
    NumericDependencyELFIdentity parsedELF;
#define READ_ELF_STRING(Field, Key)                                            \
  do {                                                                         \
    llvm::Expected<std::string> parsed =                                       \
        requireString(*(*elfObject)->get(Key), "numeric artifact ELF " Key);   \
    if (!parsed)                                                               \
      return parsed.takeError();                                               \
    parsedELF.Field = std::move(*parsed);                                      \
  } while (false)
    READ_ELF_STRING(elfClass, "class");
    READ_ELF_STRING(byteOrder, "byte_order");
    READ_ELF_STRING(type, "type");
    READ_ELF_STRING(machine, "machine");
#undef READ_ELF_STRING
    llvm::Expected<std::optional<std::string>> buildId = requireNullableString(
        *(*elfObject)->get("build_id"), "numeric artifact ELF build_id");
    if (!buildId)
      return buildId.takeError();
    llvm::Expected<std::optional<std::string>> soname = requireNullableString(
        *(*elfObject)->get("soname"), "numeric artifact ELF soname");
    if (!soname)
      return soname.takeError();
    parsedELF.buildId = std::move(*buildId);
    parsedELF.soname = std::move(*soname);
    if (parsedELF.buildId &&
        (parsedELF.buildId->empty() ||
         !llvm::all_of(*parsedELF.buildId, llvm::isHexDigit)))
      return invalid(ErrorCode::TypeMismatch,
                     "numeric artifact ELF build_id is malformed");
    llvm::Expected<std::vector<std::string>> needed = requireStringArray(
        *(*elfObject)->get("needed"), "numeric artifact ELF needed");
    if (!needed)
      return needed.takeError();
    llvm::Expected<std::vector<std::string>> rpath = requireStringArray(
        *(*elfObject)->get("rpath"), "numeric artifact ELF rpath");
    if (!rpath)
      return rpath.takeError();
    llvm::Expected<std::vector<std::string>> runpath = requireStringArray(
        *(*elfObject)->get("runpath"), "numeric artifact ELF runpath");
    if (!runpath)
      return runpath.takeError();
    parsedELF.needed = std::move(*needed);
    parsedELF.rpath = std::move(*rpath);
    parsedELF.runpath = std::move(*runpath);
    if (parsedELF.elfClass.empty() || parsedELF.byteOrder.empty() ||
        parsedELF.type.empty() || parsedELF.machine.empty())
      return invalid(ErrorCode::TypeMismatch,
                     "numeric artifact ELF identity is incomplete");
    elf = std::move(parsedELF);
  }

  llvm::Expected<ManagedPath> path =
      resolveManagedPath(root, *relative, RequiredFileType::RegularFile,
                         "numeric artifact " + name, /*requireRelative=*/true);
  if (!path)
    return path.takeError();
  llvm::Expected<FileReadback> readback =
      readRegularFile(path->resolved, std::numeric_limits<uint64_t>::max(),
                      "numeric artifact " + name);
  if (!readback)
    return readback.takeError();
  if (readback->size != *size)
    return invalid(ErrorCode::SizeMismatch,
                   "numeric artifact " + name + " size mismatch");
  if (readback->digest != *digest)
    return invalid(ErrorCode::DigestMismatch,
                   "numeric artifact " + name + " SHA256 mismatch");
  if (readback->mode != *mode)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric artifact " + name + " mode mismatch");

  llvm::Expected<bool> isELF =
      hasELFMagic(path->resolved, "numeric artifact " + name);
  if (!isELF)
    return isELF.takeError();
  if (*isELF) {
    if (!elf)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric artifact " + name +
                         " omits its actual ELF identity");
    llvm::Expected<NumericDependencyELFIdentity> actualELF = inspectELFIdentity(
        path->resolved, readelfPath, "numeric artifact " + name);
    if (!actualELF)
      return actualELF.takeError();
    if (!equalELFIdentity(*elf, *actualELF))
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric artifact " + name + " ELF identity mismatch");
  } else if (elf) {
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric artifact " + name +
                       " records ELF identity for a non-ELF file");
  }

  llvm::Expected<FileReadback> finalReadback =
      readRegularFile(path->resolved, std::numeric_limits<uint64_t>::max(),
                      "numeric artifact " + name);
  if (!finalReadback)
    return finalReadback.takeError();
  if (finalReadback->digest != readback->digest ||
      finalReadback->size != readback->size ||
      finalReadback->mode != readback->mode)
    return invalid(ErrorCode::IO,
                   "numeric artifact " + name +
                       " changed during ELF identity inspection");
  return NumericDependencyArtifactIdentity{name.str(),
                                           path->relative,
                                           path->resolved,
                                           std::move(*digest),
                                           *size,
                                           std::move(*fileType),
                                           static_cast<uint32_t>(*mode),
                                           std::move(elf)};
}

struct ParsedBuildIdentity {
  NumericDependencyBuildIdentity build;
  std::vector<NumericDependencyToolIdentity> tools;
  std::vector<NumericDependencyEnvironmentIdentity> environments;
};

llvm::Expected<NumericDependencyToolIdentity>
parseToolIdentity(llvm::StringRef name, const llvm::json::Value &value) {
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, "numeric tool " + name);
  if (!object)
    return object.takeError();
  if (llvm::Error error =
          requireExactKeys(**object,
                           {"path", "sha256", "size", "file_type", "mode",
                            "version_first_line", "version_output_sha256"},
                           "numeric tool " + name))
    return error;

  NumericDependencyToolIdentity result;
  result.name = name.str();
#define READ_TOOL_STRING(Field, Key)                                           \
  do {                                                                         \
    llvm::Expected<std::string> parsed =                                       \
        requireString(*(*object)->get(Key), "numeric tool " + name + "." Key); \
    if (!parsed)                                                               \
      return parsed.takeError();                                               \
    result.Field = std::move(*parsed);                                         \
  } while (false)
  READ_TOOL_STRING(resolvedPath, "path");
  READ_TOOL_STRING(sha256, "sha256");
  READ_TOOL_STRING(fileType, "file_type");
  READ_TOOL_STRING(versionFirstLine, "version_first_line");
  READ_TOOL_STRING(versionOutputSHA256, "version_output_sha256");
#undef READ_TOOL_STRING
  llvm::Expected<uint64_t> size =
      requireUnsigned(*(*object)->get("size"), "numeric tool size");
  if (!size)
    return size.takeError();
  llvm::Expected<uint64_t> mode =
      requireUnsigned(*(*object)->get("mode"), "numeric tool mode");
  if (!mode)
    return mode.takeError();
  if (*mode > 07777 || result.fileType != "regular" ||
      !isLowerSHA256(result.sha256) ||
      !isLowerSHA256(result.versionOutputSHA256) ||
      llvm::StringRef(result.versionFirstLine).contains('\n') ||
      llvm::StringRef(result.versionFirstLine).contains('\r'))
    return invalid(ErrorCode::TypeMismatch,
                   "numeric tool identity is malformed for " + name);
  llvm::Expected<std::string> resolved =
      resolveAbsoluteRegularPath(result.resolvedPath, "numeric tool " + name);
  if (!resolved)
    return resolved.takeError();
  llvm::Expected<FileReadback> readback = readRegularFile(
      *resolved, std::numeric_limits<uint64_t>::max(), "numeric tool " + name);
  if (!readback)
    return readback.takeError();
  if (readback->size != *size)
    return invalid(ErrorCode::SizeMismatch,
                   "numeric tool size mismatch for " + name);
  if (readback->digest != result.sha256)
    return invalid(ErrorCode::DigestMismatch,
                   "numeric tool SHA256 mismatch for " + name);
  if (readback->mode != *mode)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric tool mode mismatch for " + name);

  std::string versionOutput;
  if (name == "false") {
    versionOutput = "managed-false-command\n";
  } else {
    std::vector<std::string> arguments = {*resolved};
    if (name == "shell") {
      arguments.push_back("-c");
      arguments.push_back("printf 'managed-shell\\n'");
    } else {
      arguments.push_back("--version");
    }
    const std::vector<std::string> environment = commandEnvironment(*resolved);
    llvm::Expected<std::string> captured =
        runAndCapture(*resolved, arguments, environment,
                      "numeric tool version command " + name);
    if (!captured)
      return captured.takeError();
    versionOutput = std::move(*captured);
  }
  llvm::StringRef firstLine(versionOutput);
  const size_t lineEnd = firstLine.find_first_of("\r\n");
  if (lineEnd != llvm::StringRef::npos)
    firstLine = firstLine.take_front(lineEnd);
  if (result.versionFirstLine != firstLine ||
      result.versionOutputSHA256 != sha256(versionOutput))
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric tool version identity mismatch for " + name);
  result.size = *size;
  result.mode = static_cast<uint32_t>(*mode);
  result.resolvedPath = std::move(*resolved);
  return result;
}

llvm::Expected<NumericDependencyEnvironmentIdentity>
parseEnvironmentIdentity(llvm::StringRef name, const llvm::json::Value &value) {
  llvm::Expected<const llvm::json::Object *> object =
      requireObject(value, "numeric environment " + name);
  if (!object)
    return object.takeError();
  NumericDependencyEnvironmentIdentity result;
  result.name = name.str();
  llvm::StringSet<> keys;
  for (const auto &entry : **object) {
    llvm::StringRef key = entry.first;
    if (key.empty() || key.contains('=') || !keys.insert(entry.first).second)
      return invalid(ErrorCode::TypeMismatch,
                     "numeric environment variable name is invalid");
    llvm::Expected<std::string> environmentValue = requireString(
        entry.second, "numeric environment " + name + "." + entry.first.str());
    if (!environmentValue)
      return environmentValue.takeError();
    if (llvm::StringRef(*environmentValue).contains('\0'))
      return invalid(ErrorCode::TypeMismatch,
                     "numeric environment value contains NUL");
    result.variables.emplace_back(entry.first.str(),
                                  std::move(*environmentValue));
  }
  return result;
}

llvm::StringMap<std::string>
environmentMap(const NumericDependencyEnvironmentIdentity &environment) {
  llvm::StringMap<std::string> result;
  for (const auto &entry : environment.variables)
    result.insert(entry);
  return result;
}

llvm::Error
verifyEnvironment(const NumericDependencyEnvironmentIdentity &actual,
                  const llvm::StringMap<std::string> &expected) {
  llvm::StringMap<std::string> values = environmentMap(actual);
  if (values.size() != expected.size())
    return invalid(ErrorCode::PolicyMismatch, "numeric environment " +
                                                  actual.name +
                                                  " variable closure mismatch");
  for (const auto &entry : expected) {
    auto iterator = values.find(entry.first());
    if (iterator == values.end() || iterator->second != entry.second)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric environment " + actual.name +
                         " does not match frozen policy");
  }
  return llvm::Error::success();
}

llvm::Expected<ParsedBuildIdentity>
parseBuildIdentity(const llvm::json::Value &value) {
  llvm::Expected<const llvm::json::Object *> build =
      requireObject(value, "numeric dependency build");
  if (!build)
    return build.takeError();
  if (llvm::Error error = requireExactKeys(
          **build,
          {"platform", "softfloat_specialization", "softfloat_thread_local",
           "softfloat_raise_flags", "configure_options", "linkage",
           "managed_m4", "pkg_config", "jobs", "toolchain", "environments",
           "mpfr_patches", "elf_identity_policy"},
          "numeric dependency build"))
    return error;

  ParsedBuildIdentity parsedResult;
  NumericDependencyBuildIdentity &result = parsedResult.build;
#define READ_BUILD_STRING(Field, Key)                                          \
  do {                                                                         \
    llvm::Expected<std::string> parsed =                                       \
        requireString(*(*build)->get(Key), "numeric build " Key);              \
    if (!parsed)                                                               \
      return parsed.takeError();                                               \
    result.Field = std::move(*parsed);                                         \
  } while (false)
  READ_BUILD_STRING(platform, "platform");
  READ_BUILD_STRING(softFloatSpecialization, "softfloat_specialization");
  READ_BUILD_STRING(softFloatThreadLocal, "softfloat_thread_local");
  READ_BUILD_STRING(softFloatRaiseFlags, "softfloat_raise_flags");
  READ_BUILD_STRING(managedM4RelativePath, "managed_m4");
  READ_BUILD_STRING(pkgConfigPolicy, "pkg_config");
  READ_BUILD_STRING(mpfrPatches, "mpfr_patches");
  READ_BUILD_STRING(elfIdentityPolicy, "elf_identity_policy");
#undef READ_BUILD_STRING

  llvm::Expected<uint64_t> jobs =
      requireUnsigned(*(*build)->get("jobs"), "numeric build jobs");
  if (!jobs)
    return jobs.takeError();
  if (*jobs == 0)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric build jobs must be positive");
  result.jobs = *jobs;

  llvm::Expected<const llvm::json::Object *> toolchain =
      requireObject(*(*build)->get("toolchain"), "numeric toolchain");
  if (!toolchain)
    return toolchain.takeError();
  if (llvm::Error error = requireExactKeys(**toolchain, {"policy", "tools"},
                                           "numeric toolchain"))
    return error;
  llvm::Expected<std::string> toolchainPolicy =
      requireString(*(*toolchain)->get("policy"), "numeric toolchain policy");
  if (!toolchainPolicy)
    return toolchainPolicy.takeError();
  result.toolchainPolicy = std::move(*toolchainPolicy);
  if (result.toolchainPolicy != "wafer-host-numeric-build-environment-v1")
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric toolchain policy mismatch");
  llvm::Expected<const llvm::json::Object *> tools =
      requireObject(*(*toolchain)->get("tools"), "numeric tools");
  if (!tools)
    return tools.takeError();
  llvm::StringSet<> allowedTools;
  for (llvm::StringRef name : kRequiredTools)
    allowedTools.insert(name);
  for (const auto &entry : **tools)
    if (!allowedTools.contains(entry.first))
      return invalid(ErrorCode::UnknownField,
                     "numeric tools has unknown field '" + entry.first.str() +
                         "'");
  for (llvm::StringRef name : kRequiredTools) {
    const llvm::json::Value *toolValue = (*tools)->get(name);
    if (!toolValue)
      return invalid(ErrorCode::MissingField,
                     "numeric tools is missing field '" + name + "'");
    llvm::Expected<NumericDependencyToolIdentity> tool =
        parseToolIdentity(name, *toolValue);
    if (!tool)
      return tool.takeError();
    parsedResult.tools.push_back(std::move(*tool));
  }

  llvm::Expected<const llvm::json::Object *> environments =
      requireObject(*(*build)->get("environments"), "numeric environments");
  if (!environments)
    return environments.takeError();
  llvm::StringSet<> allowedEnvironments;
  for (llvm::StringRef name : kRequiredEnvironments)
    allowedEnvironments.insert(name);
  for (const auto &entry : **environments)
    if (!allowedEnvironments.contains(entry.first))
      return invalid(ErrorCode::UnknownField,
                     "numeric environments has unknown field '" +
                         entry.first.str() + "'");
  for (llvm::StringRef name : kRequiredEnvironments) {
    const llvm::json::Value *environmentValue = (*environments)->get(name);
    if (!environmentValue)
      return invalid(ErrorCode::MissingField,
                     "numeric environments is missing field '" + name + "'");
    llvm::Expected<NumericDependencyEnvironmentIdentity> environment =
        parseEnvironmentIdentity(name, *environmentValue);
    if (!environment)
      return environment.takeError();
    parsedResult.environments.push_back(std::move(*environment));
  }

  llvm::Expected<const llvm::json::Object *> options = requireObject(
      *(*build)->get("configure_options"), "numeric configure options");
  if (!options)
    return options.takeError();
  if (llvm::Error error = requireExactKeys(**options, {"m4", "gmp", "mpfr"},
                                           "numeric configure options"))
    return error;
  llvm::Expected<std::vector<std::string>> m4Options = requireStringArray(
      *(*options)->get("m4"), "numeric m4 configure options");
  if (!m4Options)
    return m4Options.takeError();
  llvm::Expected<std::vector<std::string>> gmpOptions = requireStringArray(
      *(*options)->get("gmp"), "numeric GMP configure options");
  if (!gmpOptions)
    return gmpOptions.takeError();
  llvm::Expected<std::vector<std::string>> mpfrOptions = requireStringArray(
      *(*options)->get("mpfr"), "numeric MPFR configure options");
  if (!mpfrOptions)
    return mpfrOptions.takeError();
  result.m4ConfigureOptions = std::move(*m4Options);
  result.gmpConfigureOptions = std::move(*gmpOptions);
  result.mpfrConfigureOptions = std::move(*mpfrOptions);

  llvm::Expected<const llvm::json::Object *> linkage =
      requireObject(*(*build)->get("linkage"), "numeric linkage policy");
  if (!linkage)
    return linkage.takeError();
  if (llvm::Error error = requireExactKeys(
          **linkage, {"softfloat", "gmp", "mpfr"}, "numeric linkage policy"))
    return error;
#define READ_LINKAGE_STRING(Field, Key)                                        \
  do {                                                                         \
    llvm::Expected<std::string> parsed =                                       \
        requireString(*(*linkage)->get(Key), "numeric linkage " Key);          \
    if (!parsed)                                                               \
      return parsed.takeError();                                               \
    result.Field = std::move(*parsed);                                         \
  } while (false)
  READ_LINKAGE_STRING(softFloatLinkage, "softfloat");
  READ_LINKAGE_STRING(gmpLinkage, "gmp");
  READ_LINKAGE_STRING(mpfrLinkage, "mpfr");
#undef READ_LINKAGE_STRING

  if (llvm::Error error = expectString(result.platform, "Linux-x86_64-GCC",
                                       "numeric build platform"))
    return error;
  if (llvm::Error error =
          expectString(result.softFloatSpecialization, "ARM-VFPv2-defaultNaN",
                       "SoftFloat specialization"))
    return error;
  if (llvm::Error error = expectString(result.softFloatThreadLocal,
                                       "_Thread_local", "SoftFloat TLS"))
    return error;
  if (llvm::Error error =
          expectString(result.softFloatRaiseFlags, "non-trapping",
                       "SoftFloat raise-flags policy"))
    return error;
  if (llvm::Error error = expectStringArray(result.m4ConfigureOptions,
                                            {"--disable-dependency-tracking"},
                                            "m4 configure options"))
    return error;
  if (llvm::Error error = expectStringArray(
          result.gmpConfigureOptions,
          {"--enable-shared", "--disable-static", "--with-pic"},
          "GMP configure options"))
    return error;
  if (llvm::Error error = expectStringArray(
          result.mpfrConfigureOptions,
          {"--enable-shared", "--disable-static", "--enable-thread-safe",
           "--disable-dependency-tracking", "--with-gmp=${GMP_PREFIX}"},
          "MPFR configure options"))
    return error;
  if (llvm::Error error =
          expectString(result.softFloatLinkage, "static", "SoftFloat linkage"))
    return error;
  if (llvm::Error error =
          expectString(result.gmpLinkage, "shared-only", "GMP linkage"))
    return error;
  if (llvm::Error error =
          expectString(result.mpfrLinkage, "shared-only", "MPFR linkage"))
    return error;
  if (llvm::Error error = expectString(result.managedM4RelativePath,
                                       "install/m4/bin/m4", "managed m4 path"))
    return error;
  if (llvm::Error error =
          expectString(result.pkgConfigPolicy, "disabled", "pkg-config policy"))
    return error;
  if (llvm::Error error =
          expectString(result.mpfrPatches, "", "MPFR patch identity"))
    return error;
  if (llvm::Error error = expectString(result.elfIdentityPolicy,
                                       "sha256-build-id-soname-needed-rpath-v1",
                                       "ELF identity policy"))
    return error;

  auto findTool =
      [&](llvm::StringRef name) -> const NumericDependencyToolIdentity * {
    auto iterator = llvm::find_if(
        parsedResult.tools, [&](const NumericDependencyToolIdentity &tool) {
          return tool.name == name;
        });
    return iterator == parsedResult.tools.end() ? nullptr : &*iterator;
  };
  auto findEnvironment = [&](llvm::StringRef name)
      -> const NumericDependencyEnvironmentIdentity * {
    auto iterator = llvm::find_if(
        parsedResult.environments,
        [&](const NumericDependencyEnvironmentIdentity &environment) {
          return environment.name == name;
        });
    return iterator == parsedResult.environments.end() ? nullptr : &*iterator;
  };
  for (llvm::StringRef name : kRequiredTools)
    if (!findTool(name))
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric tool closure changed during parsing");

  llvm::SmallVector<std::string, 10> toolDirectories;
  for (llvm::StringRef name : kRequiredTools) {
    std::string directory =
        llvm::sys::path::parent_path(findTool(name)->resolvedPath).str();
    if (!llvm::is_contained(toolDirectories, directory))
      toolDirectories.push_back(std::move(directory));
  }
  std::string toolPath;
  for (llvm::StringRef directory : toolDirectories) {
    if (!toolPath.empty())
      toolPath.push_back(':');
    toolPath.append(directory);
  }
  constexpr llvm::StringLiteral rootToken = "${NUMERIC_ROOT}";
  llvm::StringMap<std::string> base;
  base["AR"] = findTool("ar")->resolvedPath;
  base["CC"] = findTool("cc")->resolvedPath;
  base["CXX"] = findTool("cxx")->resolvedPath;
  base["HOME"] = (rootToken + "/build/home").str();
  base["LANG"] = "C";
  base["LC_ALL"] = "C";
  base["LD"] = findTool("ld")->resolvedPath;
  base["MAKE"] = findTool("make")->resolvedPath;
  base["NM"] = findTool("nm")->resolvedPath;
  base["PATH"] = toolPath;
  base["RANLIB"] = findTool("ranlib")->resolvedPath;
  base["SHELL"] = findTool("shell")->resolvedPath;
  base["SOURCE_DATE_EPOCH"] = "0";
  base["TMPDIR"] = (rootToken + "/build/tmp").str();
  base["TZ"] = "UTC";

  llvm::StringMap<std::string> managed = base;
  managed["M4"] = (rootToken + "/install/m4/bin/m4").str();
  managed["PATH"] =
      (rootToken + "/install/m4/bin:" + llvm::Twine(toolPath)).str();
  managed["PKG_CONFIG"] = findTool("false")->resolvedPath;
  llvm::StringMap<std::string> mpfr = managed;
  mpfr["CPPFLAGS"] = ("-I" + rootToken + "/install/gmp/include").str();
  mpfr["LDFLAGS"] = ("-L" + rootToken + "/install/gmp/lib -Wl,-rpath," +
                     rootToken + "/install/gmp/lib")
                        .str();
  mpfr["LD_LIBRARY_PATH"] = (rootToken + "/install/gmp/lib").str();
  llvm::StringMap<std::string> versionRuntime = managed;
  versionRuntime["LD_LIBRARY_PATH"] =
      (rootToken + "/install/mpfr/lib:" + rootToken + "/install/gmp/lib").str();

  const std::pair<llvm::StringRef, const llvm::StringMap<std::string> *>
      expectedEnvironments[] = {{"base", &base},
                                {"managed", &managed},
                                {"mpfr", &mpfr},
                                {"version-runtime", &versionRuntime}};
  for (const auto &expected : expectedEnvironments) {
    const NumericDependencyEnvironmentIdentity *actual =
        findEnvironment(expected.first);
    if (!actual)
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric environment closure changed during parsing");
    if (llvm::Error error = verifyEnvironment(*actual, *expected.second))
      return error;
  }
  return parsedResult;
}

struct GateContract {
  std::vector<std::string> command;
  std::string cwd;
  std::string environment;
};

llvm::Expected<GateContract>
expectedGateContract(const NumericDependencyConformanceRecord &record,
                     llvm::StringRef name) {
  constexpr llvm::StringLiteral root = "${NUMERIC_ROOT}";
  const NumericDependencyToolIdentity *make = record.findTool("make");
  const NumericDependencyToolIdentity *cc = record.findTool("cc");
  const NumericDependencySourceIdentity *m4Source = record.findSource("m4");
  const NumericDependencySourceIdentity *gmpSource = record.findSource("gmp");
  const NumericDependencySourceIdentity *mpfrSource = record.findSource("mpfr");
  const NumericDependencyArtifactIdentity *gmp = record.findArtifact("gmp");
  const NumericDependencyArtifactIdentity *mpfr = record.findArtifact("mpfr");
  if (!make || !cc || !m4Source || !gmpSource || !mpfrSource || !gmp || !mpfr)
    return invalid(ErrorCode::ClosureMismatch,
                   "gate contract inputs are incomplete");
  const std::string jobs =
      "-j" + std::to_string(record.getBuildIdentity().jobs);

  struct AutotoolsPolicy {
    llvm::StringRef dependency;
    const NumericDependencySourceIdentity *source;
    llvm::ArrayRef<std::string> options;
    llvm::StringRef environment;
  };
  const AutotoolsPolicy dependencies[] = {
      {"m4", m4Source, record.getBuildIdentity().m4ConfigureOptions, "base"},
      {"gmp", gmpSource, record.getBuildIdentity().gmpConfigureOptions,
       "managed"},
      {"mpfr", mpfrSource, record.getBuildIdentity().mpfrConfigureOptions,
       "mpfr"},
  };
  for (const AutotoolsPolicy &dependency : dependencies) {
    const std::string prefix = dependency.dependency.str() + "-";
    const std::string cwd = (root + "/build/" + dependency.dependency).str();
    if (name == prefix + "configure") {
      std::vector<std::string> command = {
          (root + "/" + dependency.source->sourceRelativePath + "/configure")
              .str(),
          ("--prefix=" + root + "/install/" + dependency.dependency).str()};
      for (const std::string &rawOption : dependency.options) {
        std::string option = rawOption;
        const llvm::StringRef marker = "${GMP_PREFIX}";
        const size_t position = option.find(marker.str());
        if (position != std::string::npos)
          option.replace(position, marker.size(),
                         (root + "/install/gmp").str());
        command.push_back(std::move(option));
      }
      return GateContract{std::move(command), cwd,
                          dependency.environment.str()};
    }
    if (name == prefix + "build")
      return GateContract{
          {make->resolvedPath, jobs}, cwd, dependency.environment.str()};
    if (name == prefix + "check")
      return GateContract{{make->resolvedPath, jobs, "check"},
                          cwd,
                          dependency.environment.str()};
    if (name == prefix + "install")
      return GateContract{
          {make->resolvedPath, "install"}, cwd, dependency.environment.str()};
  }

  if (name == "m4-version")
    return GateContract{
        {(root + "/install/m4/bin/m4").str(), "--version"}, root.str(), "base"};
  if (name == "softfloat-build")
    return GateContract{
        {make->resolvedPath, jobs,
         ("SOURCE_DIR=" + root + "/sources/SoftFloat-3e/source").str(),
         "SPECIALIZE_TYPE=ARM-VFPv2-defaultNaN"},
        (root + "/build/softfloat").str(),
        "managed"};
  if (name == "testfloat-build")
    return GateContract{
        {make->resolvedPath, jobs, "testsoftfloat",
         ("SOURCE_DIR=" + root + "/sources/TestFloat-3e/source").str(),
         ("SOFTFLOAT_INCLUDE_DIR=" + root +
          "/sources/SoftFloat-3e/source/include")
             .str(),
         ("SOFTFLOAT_LIB=" + root + "/build/softfloat/softfloat.a").str()},
        (root + "/build/testfloat").str(),
        "managed"};
  if (name == "softfloat-policy-compile")
    return GateContract{
        {cc->resolvedPath, "-std=c11", "-DTHREAD_LOCAL=_Thread_local",
         ("-I" + root + "/install/softfloat/include").str(),
         (root + "/build/softfloat/wafer-softfloat-policy-probe.c").str(),
         (root + "/install/softfloat/lib/libsoftfloat.a").str(), "-pthread",
         "-o", (root + "/build/softfloat/wafer-softfloat-policy-probe").str()},
        root.str(),
        "managed"};
  if (name == "softfloat-tls-default-nan")
    return GateContract{
        {(root + "/build/softfloat/wafer-softfloat-policy-probe").str()},
        root.str(),
        "managed"};

  llvm::StringRef selector;
  if (name == "testsoftfloat-f16-mulAdd")
    selector = "f16_mulAdd";
  else if (name == "testsoftfloat-f32-mulAdd")
    selector = "f32_mulAdd";
  else if (name == "testsoftfloat-all1")
    selector = "-all1";
  else if (name == "testsoftfloat-all2")
    selector = "-all2";
  if (!selector.empty())
    return GateContract{{(root + "/install/testfloat/bin/testsoftfloat").str(),
                         "-seed", "1", "-level", "1", "-errorstop",
                         "-rnear_even", "-tininessafter", selector.str()},
                        root.str(),
                        "managed"};

  if (name == "managed-version-thread-safe-compile")
    return GateContract{
        {cc->resolvedPath, ("-I" + root + "/install/gmp/include").str(),
         ("-I" + root + "/install/mpfr/include").str(),
         (root + "/build/managed-version-thread-safe.c").str(),
         (root + "/" + mpfr->relativePath).str(),
         (root + "/" + gmp->relativePath).str(),
         ("-Wl,-rpath," + root + "/install/mpfr/lib").str(),
         ("-Wl,-rpath," + root + "/install/gmp/lib").str(), "-pthread", "-o",
         (root + "/build/managed-version-thread-safe").str()},
        root.str(),
        "managed"};
  if (name == "managed-version-thread-safe")
    return GateContract{{(root + "/build/managed-version-thread-safe").str()},
                        root.str(),
                        "version-runtime"};
  return invalid(ErrorCode::ClosureMismatch,
                 "unknown numeric conformance gate " + name);
}

llvm::Error
validateGateContract(const NumericDependencyConformanceRecord &record,
                     const NumericDependencyConformanceGateIdentity &gate) {
  if (gate.exitCode != 0)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric conformance gate exit_code is not zero");
  llvm::Expected<GateContract> expected =
      expectedGateContract(record, gate.name);
  if (!expected)
    return expected.takeError();
  if (gate.command != expected->command || gate.cwd != expected->cwd ||
      gate.environment != expected->environment)
    return invalid(ErrorCode::PolicyMismatch,
                   "numeric conformance gate invocation mismatch for " +
                       gate.name);
  if (!record.findEnvironment(gate.environment))
    return invalid(ErrorCode::ClosureMismatch,
                   "numeric conformance gate references unknown environment");
  return llvm::Error::success();
}

llvm::Error
validateSharedObjectPair(llvm::StringRef stem,
                         const NumericDependencyArtifactIdentity &real,
                         const NumericDependencyArtifactIdentity &loader) {
  llvm::StringRef realName = llvm::sys::path::filename(real.resolvedPath);
  llvm::StringRef loaderName = llvm::sys::path::filename(loader.resolvedPath);
  const std::string prefix = ("lib" + stem + ".so.").str();
  if (!realName.starts_with(prefix) || !loaderName.starts_with(prefix) ||
      real.resolvedPath == loader.resolvedPath)
    return invalid(ErrorCode::PolicyMismatch,
                   stem + " real/loader shared-object identity is invalid");
  if (real.sha256 != loader.sha256 || real.size != loader.size)
    return invalid(ErrorCode::DigestMismatch,
                   stem + " real/loader shared-object content differs");
  if (real.mode != loader.mode || real.fileType != loader.fileType ||
      !real.elf || !loader.elf)
    return invalid(ErrorCode::PolicyMismatch,
                   stem + " real/loader ELF identity is incomplete");
  const NumericDependencyELFIdentity &lhs = *real.elf;
  const NumericDependencyELFIdentity &rhs = *loader.elf;
  if (lhs.elfClass != rhs.elfClass || lhs.byteOrder != rhs.byteOrder ||
      lhs.type != rhs.type || lhs.machine != rhs.machine ||
      lhs.buildId != rhs.buildId || lhs.soname != rhs.soname ||
      lhs.needed != rhs.needed || lhs.rpath != rhs.rpath ||
      lhs.runpath != rhs.runpath || lhs.type != "DYN" ||
      lhs.byteOrder != "little" || !lhs.soname || *lhs.soname != loaderName ||
      !lhs.buildId || !isLowerHex(*lhs.buildId) ||
      !llvm::StringRef(*lhs.soname).starts_with(prefix))
    return invalid(ErrorCode::PolicyMismatch,
                   stem + " real/loader ELF identity mismatch");
  return llvm::Error::success();
}

llvm::Expected<NumericLoadedObjectIdentity>
verifyLoadedObject(const NumericDependencyConformanceRecord &record,
                   NumericLoadedObjectIdentity identity,
                   llvm::StringRef realArtifactName,
                   llvm::StringRef loaderArtifactName) {
  const NumericDependencyArtifactIdentity *real =
      record.findArtifact(realArtifactName);
  const NumericDependencyArtifactIdentity *loader =
      record.findArtifact(loaderArtifactName);
  if (!real || !loader)
    return invalid(ErrorCode::ClosureMismatch,
                   "verified record lost loaded-object artifacts");
  if (!isLowerSHA256(identity.sha256))
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "provider returned an invalid loaded-object SHA256");

  llvm::Expected<ManagedRoot> root = verifyManagedRoot(record.getManagedRoot());
  if (!root)
    return root.takeError();
  llvm::Expected<ManagedPath> loaded = resolveManagedPath(
      *root, identity.resolvedPath, RequiredFileType::RegularFile,
      "loaded " + stringifyNumericLoadedObjectKind(identity.kind),
      /*requireRelative=*/false);
  if (!loaded)
    return loaded.takeError();
  llvm::Expected<FileReadback> readback = readRegularFile(
      loaded->resolved, std::numeric_limits<uint64_t>::max(),
      "loaded " + stringifyNumericLoadedObjectKind(identity.kind));
  if (!readback)
    return readback.takeError();
  if (identity.sha256 != readback->digest)
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "provider loaded-object digest does not match its file");
  if (loaded->resolved != real->resolvedPath &&
      loaded->resolved != loader->resolvedPath)
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "loaded object is not the recorded real/loader artifact");
  if (readback->digest != real->sha256 || readback->digest != loader->sha256)
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "loaded object content does not match managed record");
  identity.resolvedPath = loaded->resolved;
  return identity;
}

} // namespace

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
  case ErrorCode::SchemaMismatch:
    return "schema-mismatch";
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

llvm::StringRef stringifyNumericLoadedObjectKind(NumericLoadedObjectKind kind) {
  switch (kind) {
  case NumericLoadedObjectKind::MPFR:
    return "mpfr";
  case NumericLoadedObjectKind::GMP:
    return "gmp";
  }
  llvm_unreachable("unknown loaded numeric object kind");
}

const NumericDependencySourceIdentity *
NumericDependencyConformanceRecord::findSource(llvm::StringRef name) const {
  auto iterator = llvm::find_if(
      sources, [&](const NumericDependencySourceIdentity &source) {
        return source.name == name;
      });
  return iterator == sources.end() ? nullptr : &*iterator;
}

const NumericDependencyArtifactIdentity *
NumericDependencyConformanceRecord::findArtifact(llvm::StringRef name) const {
  auto iterator = llvm::find_if(
      artifacts, [&](const NumericDependencyArtifactIdentity &artifact) {
        return artifact.name == name;
      });
  return iterator == artifacts.end() ? nullptr : &*iterator;
}

const NumericDependencyToolIdentity *
NumericDependencyConformanceRecord::findTool(llvm::StringRef name) const {
  auto iterator =
      llvm::find_if(tools, [&](const NumericDependencyToolIdentity &tool) {
        return tool.name == name;
      });
  return iterator == tools.end() ? nullptr : &*iterator;
}

const NumericDependencyEnvironmentIdentity *
NumericDependencyConformanceRecord::findEnvironment(
    llvm::StringRef name) const {
  auto iterator = llvm::find_if(
      environments,
      [&](const NumericDependencyEnvironmentIdentity &environment) {
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
    std::string message = llvm::toString(recordReadback.takeError());
    if (message.find("resource-limit") != std::string::npos)
      return invalid(ErrorCode::RecordTooLarge,
                     "numeric dependency record exceeds byte limit");
    return invalid(ErrorCode::IO, message);
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
  DuplicateKeyScanner duplicateScanner(recordBytes);
  if (llvm::Error error = duplicateScanner.scan())
    return error;
  llvm::Expected<const llvm::json::Object *> top =
      requireObject(*parsed, "numeric dependency record");
  if (!top)
    return top.takeError();
  if (llvm::Error error =
          requireExactKeys(**top,
                           {"schema_version", "kind", "status", "pins", "build",
                            "artifacts", "licenses", "conformance"},
                           "numeric dependency record"))
    return error;

  llvm::Expected<uint64_t> schema = requireUnsigned(
      *(*top)->get("schema_version"), "numeric record schema_version");
  if (!schema)
    return schema.takeError();
  if (*schema != kNumericDependencyConformanceSchemaVersion)
    return invalid(ErrorCode::SchemaMismatch,
                   "numeric dependency record schema mismatch");
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
  result.schemaVersion = static_cast<uint32_t>(*schema);
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

    NumericDependencySourceIdentity identity;
    identity.name = name.str();
#define READ_PIN_STRING(Field, Key)                                            \
  do {                                                                         \
    llvm::Expected<std::string> value =                                        \
        requireString(*(*pin)->get(Key), "numeric pin " + name + "." Key);     \
    if (!value)                                                                \
      return value.takeError();                                                \
    identity.Field = std::move(*value);                                        \
  } while (false)
    READ_PIN_STRING(version, "version");
    READ_PIN_STRING(url, "url");
    READ_PIN_STRING(archiveSHA256, "archive_sha256");
    READ_PIN_STRING(archiveRelativePath, "archive");
    READ_PIN_STRING(sourceRelativePath, "source");
    READ_PIN_STRING(sourceTreeSHA256, "source_tree_sha256");
#undef READ_PIN_STRING
    if (identity.version.empty() ||
        !llvm::StringRef(identity.url).starts_with("https://") ||
        !isLowerSHA256(identity.archiveSHA256) ||
        !isLowerSHA256(identity.sourceTreeSHA256))
      return invalid(ErrorCode::TypeMismatch,
                     "numeric pin " + name + " identity is malformed");
    if (!pinArchivePaths.insert(identity.archiveRelativePath).second ||
        !pinSourcePaths.insert(identity.sourceRelativePath).second)
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric pin paths are not unique");

    llvm::Expected<ManagedPath> archive = resolveManagedPath(
        *root, identity.archiveRelativePath, RequiredFileType::RegularFile,
        "numeric source archive " + name, /*requireRelative=*/true);
    if (!archive)
      return archive.takeError();
    llvm::StringRef urlFile = llvm::StringRef(identity.url).rsplit('/').second;
    if (urlFile.empty() ||
        llvm::sys::path::filename(archive->resolved) != urlFile)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric source archive name does not match URL");
    llvm::Expected<FileReadback> archiveReadback =
        readRegularFile(archive->resolved, std::numeric_limits<uint64_t>::max(),
                        "numeric source archive " + name);
    if (!archiveReadback)
      return archiveReadback.takeError();
    if (archiveReadback->digest != identity.archiveSHA256)
      return invalid(ErrorCode::DigestMismatch,
                     "numeric source archive SHA256 mismatch for " + name);
    identity.archiveResolvedPath = archive->resolved;

    llvm::Expected<ManagedPath> source = resolveManagedPath(
        *root, identity.sourceRelativePath, RequiredFileType::Directory,
        "numeric source tree " + name, /*requireRelative=*/true);
    if (!source)
      return source.takeError();
    llvm::Expected<std::string> sourceTreeDigest = digestSourceTree(
        source->resolved, limits, "numeric source tree " + name);
    if (!sourceTreeDigest)
      return sourceTreeDigest.takeError();
    if (*sourceTreeDigest != identity.sourceTreeSHA256)
      return invalid(ErrorCode::SourceTreeMismatch,
                     "numeric source tree SHA256 mismatch for " + name);
    identity.sourceResolvedPath = source->resolved;
    result.sources.push_back(std::move(identity));
  }

  llvm::Expected<ParsedBuildIdentity> build =
      parseBuildIdentity(*(*top)->get("build"));
  if (!build)
    return build.takeError();
  result.build = std::move(build->build);
  result.tools = std::move(build->tools);
  result.environments = std::move(build->environments);
  const NumericDependencyToolIdentity *readelfTool = result.findTool("readelf");
  if (!readelfTool)
    return invalid(ErrorCode::ClosureMismatch,
                   "numeric readelf tool identity is missing");

  llvm::Expected<const llvm::json::Object *> artifacts =
      requireObject(*(*top)->get("artifacts"), "numeric dependency artifacts");
  if (!artifacts)
    return artifacts.takeError();
  llvm::StringSet<> artifactAllowed;
  for (llvm::StringRef name : kRequiredArtifacts)
    artifactAllowed.insert(name);
  for (const auto &entry : **artifacts)
    if (!artifactAllowed.contains(entry.first))
      return invalid(ErrorCode::UnknownField,
                     "numeric dependency artifacts has unknown field '" +
                         entry.first.str() + "'");
  for (llvm::StringRef name : kRequiredArtifacts)
    if (!(*artifacts)->get(name))
      return invalid(ErrorCode::MissingField,
                     "numeric dependency artifacts is missing field '" + name +
                         "'");

  llvm::StringSet<> artifactPaths;
  for (llvm::StringRef name : kRequiredArtifacts) {
    llvm::Expected<NumericDependencyArtifactIdentity> artifact =
        parseArtifactIdentity(name, *(*artifacts)->get(name), *root,
                              readelfTool->resolvedPath);
    if (!artifact)
      return artifact.takeError();
    auto policy = llvm::find_if(kArtifactPathPolicies,
                                [&](const ArtifactPathPolicy &candidate) {
                                  return candidate.name == name;
                                });
    if (policy == std::end(kArtifactPathPolicies) ||
        artifact->relativePath != policy->path)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric artifact path mismatch for " + name);
    if (policy->executable && (artifact->mode & 0111) == 0)
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric artifact is not executable: " + name);
    if (!artifactPaths.insert(artifact->relativePath).second)
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric artifact paths are not unique");
    result.artifacts.push_back(std::move(*artifact));
  }

  const NumericDependencyArtifactIdentity *managedM4 =
      result.findArtifact("m4");
  const NumericDependencyArtifactIdentity *softFloat =
      result.findArtifact("softfloat");
  const NumericDependencyArtifactIdentity *gmp = result.findArtifact("gmp");
  const NumericDependencyArtifactIdentity *gmpLoader =
      result.findArtifact("gmp-soname");
  const NumericDependencyArtifactIdentity *mpfr = result.findArtifact("mpfr");
  const NumericDependencyArtifactIdentity *mpfrLoader =
      result.findArtifact("mpfr-soname");
  if (!managedM4 || !softFloat || !gmp || !gmpLoader || !mpfr || !mpfrLoader)
    return invalid(ErrorCode::ClosureMismatch,
                   "numeric artifact closure is incomplete");
  if (managedM4->relativePath != result.build.managedM4RelativePath)
    return invalid(ErrorCode::PolicyMismatch,
                   "managed m4 artifact does not match build policy");
  if (llvm::sys::path::extension(softFloat->resolvedPath) != ".a")
    return invalid(ErrorCode::PolicyMismatch,
                   "SoftFloat artifact is not a static archive");
  if (llvm::Error error = validateSharedObjectPair("gmp", *gmp, *gmpLoader))
    return error;
  if (llvm::Error error = validateSharedObjectPair("mpfr", *mpfr, *mpfrLoader))
    return error;
  if (!mpfr->elf || !gmpLoader->elf || !gmpLoader->elf->soname ||
      !llvm::is_contained(mpfr->elf->needed, *gmpLoader->elf->soname))
    return invalid(ErrorCode::PolicyMismatch,
                   "MPFR ELF identity does not require recorded GMP SONAME");
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
    llvm::Expected<std::string> artifactName =
        requireString(*(*license)->get("artifact"), "numeric license artifact");
    if (!artifactName)
      return artifactName.takeError();
    llvm::Expected<std::string> licenseDigest =
        requireString(*(*license)->get("sha256"), "numeric license SHA256");
    if (!licenseDigest)
      return licenseDigest.takeError();
    if (*dependency != policy.dependency || *artifactName != policy.name ||
        !isLowerSHA256(*licenseDigest))
      return invalid(ErrorCode::PolicyMismatch,
                     "numeric license binding mismatch for " + policy.name);
    const NumericDependencySourceIdentity *dependencySource =
        result.findSource(*dependency);
    const NumericDependencyArtifactIdentity *licenseArtifact =
        result.findArtifact(*artifactName);
    if (!dependencySource || !licenseArtifact ||
        licenseArtifact->sha256 != *licenseDigest || licenseArtifact->elf)
      return invalid(ErrorCode::ClosureMismatch,
                     "numeric license artifact binding is incomplete");
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
                     "numeric license source/artifact digest mismatch for " +
                         policy.name);
    result.licenses.push_back({policy.name.str(), std::move(*dependency),
                               std::move(*source), sourcePath->resolved,
                               std::move(*artifactName),
                               std::move(*licenseDigest)});
  }

  llvm::Expected<const llvm::json::Object *> conformance = requireObject(
      *(*top)->get("conformance"), "numeric conformance identity");
  if (!conformance)
    return conformance.takeError();
  if (llvm::Error error = requireExactKeys(**conformance, {"policy", "gates"},
                                           "numeric conformance identity"))
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
    llvm::Expected<NumericDependencyArtifactIdentity> log =
        parseArtifactIdentity("conformance-log:" + *name, *(*gate)->get("log"),
                              *root, readelfTool->resolvedPath);
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
    NumericDependencyConformanceGateIdentity gateIdentity{
        std::move(*name),
        std::move(*command),
        std::move(*cwd),
        std::move(*environment),
        static_cast<int64_t>(*exitCode),
        std::move(*log)};
    if (llvm::Error error = validateGateContract(result, gateIdentity))
      return error;
    result.gates.push_back(std::move(gateIdentity));
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

llvm::Expected<NumericLoadedObjectIdentity>
DladdrNumericLoadedObjectIdentityProvider::identify(
    NumericLoadedObjectKind kind) const {
  const void *symbol = kind == NumericLoadedObjectKind::MPFR ? mpfrSymbolAddress
                                                             : gmpSymbolAddress;
  if (!symbol)
    return invalid(ErrorCode::LoadedObjectUnavailable,
                   "representative " + stringifyNumericLoadedObjectKind(kind) +
                       " symbol address is null");
#if defined(__unix__) || defined(__APPLE__)
  Dl_info information{};
  if (dladdr(symbol, &information) == 0 || !information.dli_fname ||
      !*information.dli_fname)
    return invalid(ErrorCode::LoadedObjectUnavailable,
                   "dladdr could not resolve loaded " +
                       stringifyNumericLoadedObjectKind(kind));
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(information.dli_fname,
                                                    status, /*follow=*/false))
    return invalid(ErrorCode::LoadedObjectUnavailable,
                   "dladdr path cannot be statted: " + error.message());
  if (status.type() == llvm::sys::fs::file_type::symlink_file)
    return invalid(ErrorCode::Symlink, "dladdr path is a symlink");
  if (status.type() != llvm::sys::fs::file_type::regular_file)
    return invalid(ErrorCode::FileType, "dladdr path is not a regular file");
  llvm::SmallString<256> resolved;
  if (std::error_code error =
          llvm::sys::fs::real_path(information.dli_fname, resolved))
    return invalid(ErrorCode::LoadedObjectUnavailable,
                   "dladdr path cannot be resolved: " + error.message());
  llvm::Expected<FileReadback> readback = readRegularFile(
      resolved, std::numeric_limits<uint64_t>::max(), "dladdr loaded object");
  if (!readback)
    return readback.takeError();
  return NumericLoadedObjectIdentity{kind, resolved.str().str(),
                                     std::move(readback->digest)};
#else
  return invalid(ErrorCode::LoadedObjectUnavailable,
                 "dladdr loaded-object readback is unavailable on this host");
#endif
}

llvm::Expected<NumericDependencyExecutionIdentity>
verifyNumericDependencyExecutionIdentity(
    const NumericDependencyConformanceRecord &record,
    const NumericLoadedObjectIdentityProvider &provider) {
  llvm::Expected<NumericLoadedObjectIdentity> mpfr =
      provider.identify(NumericLoadedObjectKind::MPFR);
  if (!mpfr)
    return mpfr.takeError();
  if (mpfr->kind != NumericLoadedObjectKind::MPFR)
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "provider returned the wrong MPFR object kind");
  llvm::Expected<NumericLoadedObjectIdentity> gmp =
      provider.identify(NumericLoadedObjectKind::GMP);
  if (!gmp)
    return gmp.takeError();
  if (gmp->kind != NumericLoadedObjectKind::GMP)
    return invalid(ErrorCode::LoadedObjectMismatch,
                   "provider returned the wrong GMP object kind");

  llvm::Expected<NumericLoadedObjectIdentity> verifiedMPFR =
      verifyLoadedObject(record, std::move(*mpfr), "mpfr", "mpfr-soname");
  if (!verifiedMPFR)
    return verifiedMPFR.takeError();
  llvm::Expected<NumericLoadedObjectIdentity> verifiedGMP =
      verifyLoadedObject(record, std::move(*gmp), "gmp", "gmp-soname");
  if (!verifiedGMP)
    return verifiedGMP.takeError();

  llvm::SHA256 provenance;
  provenance.update("wafer-numeric-dependency-execution-v1");
  provenance.update(llvm::StringRef("\0", 1));
  provenance.update(record.getRecordSHA256());
  for (const NumericLoadedObjectIdentity *identity :
       {&*verifiedMPFR, &*verifiedGMP}) {
    provenance.update(llvm::StringRef("\0", 1));
    provenance.update(stringifyNumericLoadedObjectKind(identity->kind));
    provenance.update(llvm::StringRef("\0", 1));
    provenance.update(identity->resolvedPath);
    provenance.update(llvm::StringRef("\0", 1));
    provenance.update(identity->sha256);
  }
  return NumericDependencyExecutionIdentity(
      record.getRecordSHA256().str(),
      llvm::toHex(provenance.final(), /*LowerCase=*/true),
      std::move(*verifiedMPFR), std::move(*verifiedGMP));
}

} // namespace wafer
