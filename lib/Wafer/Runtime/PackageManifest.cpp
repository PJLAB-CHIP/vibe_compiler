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

const PackageModuleExportRecord *
findModuleExport(const PackageModuleRecord &module,
                 PackageModuleExportRole role) {
  auto iterator = llvm::find_if(
      module.exports, [&](const auto &moduleExport) {
        return moduleExport.role == role;
      });
  return iterator == module.exports.end() ? nullptr : &*iterator;
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

llvm::StringRef
stringifyPackageModuleExportRole(PackageModuleExportRole role) {
  switch (role) {
  case PackageModuleExportRole::Prepare:
    return "prepare";
  case PackageModuleExportRole::Main:
    return "main";
  }
  llvm_unreachable("unknown package module export role");
}

llvm::StringRef
stringifyPackageEntryCompletionKind(PackageEntryCompletionKind kind) {
  switch (kind) {
  case PackageEntryCompletionKind::ReturnAfterLocalDrain:
    return "return_after_local_drain";
  }
  llvm_unreachable("unknown package entry completion kind");
}

} // namespace wafer::runtime
