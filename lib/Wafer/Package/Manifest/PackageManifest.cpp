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

} // namespace detail

const PackageModuleRecord *
findModule(llvm::ArrayRef<PackageModuleRecord> modules, ModuleId id) {
  auto iterator = llvm::find_if(
      modules, [&](const auto &module) { return module.id == id; });
  return iterator == modules.end() ? nullptr : &*iterator;
}

const PackageModuleExportRecord *
findModuleExport(const PackageModuleRecord &module,
                 PackageModuleExportRole role) {
  auto iterator = llvm::find_if(module.exports, [&](const auto &moduleExport) {
    return moduleExport.role == role;
  });
  return iterator == module.exports.end() ? nullptr : &*iterator;
}

const ProgramTensorRecord *
findProgramTensor(llvm::ArrayRef<ProgramTensorRecord> tensors,
                  ProgramTensorId id) {
  auto iterator = llvm::find_if(
      tensors, [&](const auto &tensor) { return tensor.id == id; });
  return iterator == tensors.end() ? nullptr : &*iterator;
}

const TargetTensorRecord *
findTargetTensor(llvm::ArrayRef<TargetTensorRecord> tensors,
                 TargetTensorId id) {
  auto iterator = llvm::find_if(
      tensors, [&](const auto &tensor) { return tensor.id == id; });
  return iterator == tensors.end() ? nullptr : &*iterator;
}

const ExternalPortRecord *findPort(llvm::ArrayRef<ExternalPortRecord> ports,
                                   PortId id) {
  auto iterator =
      llvm::find_if(ports, [&](const auto &port) { return port.id == id; });
  return iterator == ports.end() ? nullptr : &*iterator;
}

llvm::StringRef stringifyProgramTensorRole(ProgramTensorRole role) {
  switch (role) {
  case ProgramTensorRole::Parameter:
    return "parameter";
  case ProgramTensorRole::Constant:
    return "constant";
  }
  llvm_unreachable("unknown program tensor role");
}

llvm::StringRef stringifyPackageAccessMode(PackageAccessMode access) {
  switch (access) {
  case PackageAccessMode::None:
    return "none";
  case PackageAccessMode::ReadOnly:
    return "read_only";
  case PackageAccessMode::WriteOnly:
    return "write_only";
  case PackageAccessMode::ReadWrite:
    return "read_write";
  }
  llvm_unreachable("unknown package access mode");
}

llvm::StringRef stringifyPackageMemLayout(PackageMemLayout layout) {
  switch (layout) {
  case PackageMemLayout::Tensor:
    return "tensor";
  case PackageMemLayout::NTensor:
    return "ntensor";
  case PackageMemLayout::Cx:
    return "cx";
  case PackageMemLayout::NCx:
    return "ncx";
  }
  llvm_unreachable("unknown package memory layout");
}

PhysicalTensorLayout getPhysicalTensorLayout(PackageMemLayout layout) {
  switch (layout) {
  case PackageMemLayout::Tensor:
    return PhysicalTensorLayout::Tensor;
  case PackageMemLayout::NTensor:
    return PhysicalTensorLayout::NTensor;
  case PackageMemLayout::Cx:
    return PhysicalTensorLayout::Cx;
  case PackageMemLayout::NCx:
    return PhysicalTensorLayout::NCx;
  }
  llvm_unreachable("unknown package memory layout");
}

llvm::StringRef stringifyPackageModuleExportRole(PackageModuleExportRole role) {
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
