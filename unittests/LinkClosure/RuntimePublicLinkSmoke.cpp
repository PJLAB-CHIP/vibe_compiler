//===- RuntimePublicLinkSmoke.cpp - WaferRuntime public API link ----------===//
//
// Minimal translation unit that references one function per public header
// group of WaferRuntime, so the public link closure is exercised end-to-end
// from a consumer that only links the library.
//
//===----------------------------------------------------------------------===//

#include "Wafer/Package/Manifest/PackageManifest.h"

#include "llvm/Support/raw_ostream.h"

int main() {
  using namespace wafer::runtime;
  if (stringifyPackageMemLayout(PackageMemLayout::Tensor) != "tensor")
    return 1;
  if (stringifyPackageMemLayout(PackageMemLayout::NTensor) != "ntensor")
    return 1;
  if (stringifyPackageMemLayout(PackageMemLayout::Cx) != "cx")
    return 1;
  if (stringifyPackageMemLayout(PackageMemLayout::NCx) != "ncx")
    return 1;
  if (stringifyProgramTensorRole(ProgramTensorRole::Parameter) != "parameter")
    return 1;
  if (stringifyProgramTensorRole(ProgramTensorRole::Constant) != "constant")
    return 1;
  if (stringifyPackageAccessMode(PackageAccessMode::ReadOnly) != "read_only")
    return 1;
  if (stringifyPackageAccessMode(PackageAccessMode::WriteOnly) != "write_only")
    return 1;
  if (stringifyPackageAccessMode(PackageAccessMode::ReadWrite) != "read_write")
    return 1;
  if (stringifyPackageModuleExportRole(PackageModuleExportRole::Prepare) !=
      "prepare")
    return 1;
  if (stringifyPackageModuleExportRole(PackageModuleExportRole::Main) !=
      "main")
    return 1;
  if (stringifyPackageEntryCompletionKind(
          PackageEntryCompletionKind::ReturnAfterLocalDrain) !=
      "return_after_local_drain")
    return 1;
  if (wafer::kCurrentTargetModuleFormat != "elf-riscv64")
    return 1;
  if (wafer::runtime::kDirectDTEStatusABI != "wafer-direct-dte-status")
    return 1;
  return 0;
}
