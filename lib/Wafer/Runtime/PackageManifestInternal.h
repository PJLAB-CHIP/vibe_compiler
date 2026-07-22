//===- PackageManifestInternal.h - Runtime manifest internals --*- C++ -*-===//

#ifndef WAFER_RUNTIME_PACKAGEMANIFESTINTERNAL_H
#define WAFER_RUNTIME_PACKAGEMANIFESTINTERNAL_H

#include "Wafer/Runtime/PackageManifest.h"

#include "llvm/ADT/Twine.h"

namespace wafer::runtime::detail {

llvm::Error invalid(llvm::Twine message);

const PackageResourceRecord *
findResource(llvm::ArrayRef<PackageResourceRecord> resources, ResourceId id);

const PackageModuleRecord *
findModule(llvm::ArrayRef<PackageModuleRecord> modules, ModuleId id);

const PackageModuleExportRecord *
findModuleExport(const PackageModuleRecord &module,
                 PackageModuleExportRole role);

const PackageCompletionRecord *
findCompletion(llvm::ArrayRef<PackageCompletionRecord> completions,
               CompletionId id);

llvm::Expected<PackageManifest> parseManifest(llvm::StringRef json,
                                              const PackageParseLimits &limits);

} // namespace wafer::runtime::detail

#endif // WAFER_RUNTIME_PACKAGEMANIFESTINTERNAL_H
