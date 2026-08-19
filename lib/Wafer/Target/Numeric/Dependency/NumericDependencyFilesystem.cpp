//===- NumericDependencyFilesystem.cpp - Secure managed files -*- C++ -*-===//

#include "NumericDependencyConformanceInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace wafer::numeric_dependency_conformance_internal {

llvm::Expected<FileReadback> readRegularFile(llvm::StringRef path,
                                             uint64_t maximumBytes,
                                             const llvm::Twine &label,
                                             bool captureContents) {
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

} // namespace wafer::numeric_dependency_conformance_internal
