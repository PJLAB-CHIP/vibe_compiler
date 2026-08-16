//===- PackageManifestReadback.cpp - Canonical manifest filesystem I/O ---===//

#include "PackageManifestInternal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

namespace wafer::runtime {

llvm::Expected<VerifiedPackageManifest>
parseCanonicalPackageJson(llvm::StringRef json, llvm::StringRef packageRoot,
                          const PackageParseLimits &limits) {
  llvm::Expected<PackageManifest> manifest =
      detail::parseManifest(json, limits);
  if (!manifest)
    return manifest.takeError();
  llvm::Expected<VerifiedPackageManifest> verified =
      verifyPackageManifest(std::move(*manifest), packageRoot, limits);
  if (!verified)
    return verified.takeError();
  if (serializeCanonicalPackageJson(*verified) != json)
    return detail::invalid("package manifest JSON is not canonical");
  return std::move(*verified);
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
detail::openPackageMember(llvm::StringRef path,
                          llvm::StringRef description) {
  llvm::sys::fs::file_t descriptor = llvm::sys::fs::kInvalidFile;
  if (std::error_code error = llvm::sys::fs::openFileForRead(
          path, descriptor, llvm::sys::fs::OF_None))
    return llvm::createStringError(error, "failed to open " + description);
  auto close = llvm::make_scope_exit([&] {
    if (descriptor != llvm::sys::fs::kInvalidFile)
      (void)llvm::sys::fs::closeFile(descriptor);
  });

  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(descriptor, status))
    return llvm::createStringError(error, "failed to stat " + description);
  if (!llvm::sys::fs::is_regular_file(status))
    return detail::invalid(description + " is not a regular file");
  // `IsVolatile` forces a read-backed snapshot instead of a file-backed mmap.
  // A read-only mmap would survive unlink/rename but would still observe an
  // in-place write to the same inode, which is not content ownership.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getOpenFile(descriptor, path, status.getSize(),
                                      /*RequiresNullTerminator=*/true,
                                      /*IsVolatile=*/true);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to snapshot " + description);
  if (std::error_code error = llvm::sys::fs::closeFile(descriptor))
    return llvm::createStringError(error, "failed to close " + description);
  return std::move(*buffer);
}

static std::string digestBuffer(const llvm::MemoryBuffer &buffer) {
  llvm::SHA256 hasher;
  hasher.update(buffer.getBuffer());
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

llvm::Expected<detail::BoundExecutablePackage>
detail::bindExecutablePackage(llvm::StringRef packageRoot,
                              const PackageParseLimits &limits) {
  // The strict loader closes the whole package root: exactly the canonical
  // manifest, the modules directory and data/program-data.bin.
  if (packageRoot.empty())
    return detail::invalid("package root must not be empty");
  llvm::SmallString<256> path(packageRoot);
  llvm::sys::path::append(path, kPackageManifestFileName);
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      openPackageMember(path, "package manifest member");
  if (!buffer)
    return buffer.takeError();
  llvm::Expected<VerifiedPackageManifest> parsed =
      parseCanonicalPackageJson((*buffer)->getBuffer(), packageRoot, limits);
  if (!parsed)
    return parsed.takeError();
  if (llvm::Error error =
          detail::verifyPackageRoot(parsed->getManifest(), packageRoot))
    return std::move(error);

  const PackageManifest &manifest = parsed->getManifest();
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> modules;
  modules.reserve(manifest.modules.size());
  for (const PackageModuleRecord &module : manifest.modules) {
    llvm::SmallString<256> modulePath(packageRoot);
    llvm::sys::path::append(modulePath, module.relativePath);
    llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> moduleBuffer =
        openPackageMember(modulePath, "package module member '" +
                                          module.relativePath + "'");
    if (!moduleBuffer)
      return moduleBuffer.takeError();
    if (digestBuffer(**moduleBuffer) != module.digest)
      return detail::invalid(
          "opened package module member digest mismatch: " +
          module.relativePath);
    modules.push_back(std::move(*moduleBuffer));
  }

  llvm::SmallString<256> programDataPath(packageRoot);
  llvm::sys::path::append(programDataPath, manifest.programData.relativePath);
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> programData =
      openPackageMember(programDataPath, "package program data member");
  if (!programData)
    return programData.takeError();
  if ((*programData)->getBufferSize() != manifest.programData.totalBytes)
    return detail::invalid(
        "opened package program data member size mismatch");
  if (digestBuffer(**programData) != manifest.programData.digest)
    return detail::invalid(
        "opened package program data member digest mismatch");

  return detail::BoundExecutablePackage{
      std::move(*parsed), std::move(*buffer), std::move(modules),
      std::move(*programData)};
}

llvm::Expected<ExecutablePackage>
loadExecutablePackage(llvm::StringRef packageRoot,
                      const PackageParseLimits &limits) {
  llvm::SmallString<256> canonicalRoot;
  if (std::error_code error =
          llvm::sys::fs::real_path(packageRoot, canonicalRoot))
    return llvm::createStringError(error,
                                   "failed to resolve package root");
  llvm::Expected<detail::BoundExecutablePackage> bound =
      detail::bindExecutablePackage(canonicalRoot, limits);
  if (!bound)
    return bound.takeError();
  return detail::ExecutablePackageFactory::make(canonicalRoot.str().str(),
                                                std::move(*bound));
}

} // namespace wafer::runtime
