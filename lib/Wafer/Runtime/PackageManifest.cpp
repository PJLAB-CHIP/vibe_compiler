//===- PackageManifest.cpp - Typed manifest schema support ---------------===//

#include "PackageManifestInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"

namespace wafer::runtime {
namespace detail {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

const PackageResourceRecord *
findResource(llvm::ArrayRef<PackageResourceRecord> resources, ResourceId id) {
  auto iterator = llvm::find_if(
      resources, [&](const auto &resource) { return resource.id == id; });
  return iterator == resources.end() ? nullptr : &*iterator;
}

const PackageModuleRecord *
findModule(llvm::ArrayRef<PackageModuleRecord> modules, ModuleId id) {
  auto iterator = llvm::find_if(
      modules, [&](const auto &module) { return module.id == id; });
  return iterator == modules.end() ? nullptr : &*iterator;
}

const PackageCompletionRecord *
findCompletion(llvm::ArrayRef<PackageCompletionRecord> completions,
               CompletionId id) {
  auto iterator = llvm::find_if(
      completions, [&](const auto &completion) { return completion.id == id; });
  return iterator == completions.end() ? nullptr : &*iterator;
}

} // namespace detail

llvm::StringRef stringifyPackageResourceRole(PackageResourceRole role) {
  switch (role) {
  case PackageResourceRole::UserInput:
    return "user_input";
  case PackageResourceRole::Parameter:
    return "parameter";
  case PackageResourceRole::Constant:
    return "constant";
  case PackageResourceRole::Output:
    return "output";
  case PackageResourceRole::Workspace:
    return "workspace";
  case PackageResourceRole::TransportStatus:
    return "transport_status";
  }
  llvm_unreachable("unknown package resource role");
}

llvm::StringRef stringifyPackageAccessMode(PackageAccessMode access) {
  switch (access) {
  case PackageAccessMode::ReadOnly:
    return "read_only";
  case PackageAccessMode::WriteOnly:
    return "write_only";
  case PackageAccessMode::ReadWrite:
    return "read_write";
  }
  llvm_unreachable("unknown package access mode");
}

} // namespace wafer::runtime
