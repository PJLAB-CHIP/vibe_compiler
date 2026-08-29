//===- PackageManifestInternal.h - Package manifest internals -*- C++ -*-===//

#ifndef WAFER_PACKAGE_MANIFEST_PACKAGEMANIFESTINTERNAL_H
#define WAFER_PACKAGE_MANIFEST_PACKAGEMANIFESTINTERNAL_H

#include "Wafer/Package/Manifest/PackageManifest.h"

#include "llvm/ADT/Twine.h"

namespace wafer::runtime::detail {

llvm::Error invalid(llvm::Twine message);

llvm::Error
verifyPackageRoot(const PackageManifest &manifest,
                  llvm::StringRef packageRoot);

llvm::Expected<PackageManifest> parseManifest(llvm::StringRef json,
                                              const PackageParseLimits &limits);

} // namespace wafer::runtime::detail

#endif // WAFER_PACKAGE_MANIFEST_PACKAGEMANIFESTINTERNAL_H
