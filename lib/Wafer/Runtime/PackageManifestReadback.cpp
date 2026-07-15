//===- PackageManifestReadback.cpp - Canonical manifest filesystem I/O ---===//

#include "PackageManifestInternal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

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

llvm::Expected<VerifiedPackageManifest>
loadVerifiedPackageManifest(llvm::StringRef packageRoot,
                            const PackageParseLimits &limits) {
  llvm::SmallString<256> path(packageRoot);
  llvm::sys::path::append(path, kPackageManifestFileName);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read package manifest: " + path);
  return parseCanonicalPackageJson((*buffer)->getBuffer(), packageRoot, limits);
}

} // namespace wafer::runtime
