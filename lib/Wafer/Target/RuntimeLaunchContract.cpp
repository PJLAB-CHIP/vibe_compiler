//===- RuntimeLaunchContract.cpp - Typed runtime launch contract --------===//

#include "Wafer/Target/RuntimeLaunchContract.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"

#include <initializer_list>

namespace wafer {
namespace {

template <typename T>
llvm::Expected<T> invalidValue(llvm::StringRef kind,
                               llvm::StringRef canonicalSpelling) {
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown %s '%s'", kind.str().c_str(),
                                 canonicalSpelling.str().c_str());
}

bool hasPhases(llvm::ArrayRef<RuntimeLaunchPhaseRole> actual,
               std::initializer_list<RuntimeLaunchPhaseRole> expected) {
  return actual == llvm::ArrayRef<RuntimeLaunchPhaseRole>(expected);
}

} // namespace

llvm::Expected<RuntimeLaunchContract> RuntimeLaunchContract::createKernel(
    KernelLaunchForm form, KernelEntryABI entryABI,
    llvm::ArrayRef<RuntimeLaunchPhaseRole> phases) {
  const bool valid = (form == KernelLaunchForm::Grid &&
                      (entryABI == KernelEntryABI::TileMajorPointerTable ||
                       entryABI == KernelEntryABI::TileRowPointerTable) &&
                      hasPhases(phases, {RuntimeLaunchPhaseRole::Main})) ||
                     (form == KernelLaunchForm::Cluster &&
                      (entryABI == KernelEntryABI::TileMajorPointerTable ||
                       entryABI == KernelEntryABI::TileRowPointerTable) &&
                      hasPhases(phases, {RuntimeLaunchPhaseRole::Prepare,
                                         RuntimeLaunchPhaseRole::Main}));
  if (!valid)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "kernel runtime launch form, entry ABI and ordered phases are "
        "incompatible");
  return RuntimeLaunchContract(KernelRuntimeLaunchContract{
      form, entryABI, std::vector<RuntimeLaunchPhaseRole>(phases)});
}

bool operator==(const RuntimeLaunchContract &lhs,
                const RuntimeLaunchContract &rhs) {
  return lhs.contract.form == rhs.contract.form &&
         lhs.contract.entryABI == rhs.contract.entryABI &&
         lhs.contract.phases == rhs.contract.phases;
}

llvm::StringRef stringifyKernelLaunchForm(KernelLaunchForm form) {
  switch (form) {
  case KernelLaunchForm::Grid:
    return "grid";
  case KernelLaunchForm::Cluster:
    return "cluster";
  }
  llvm_unreachable("unknown kernel launch form");
}

llvm::Expected<KernelLaunchForm>
parseKernelLaunchForm(llvm::StringRef canonicalSpelling) {
  if (canonicalSpelling == "grid")
    return KernelLaunchForm::Grid;
  if (canonicalSpelling == "cluster")
    return KernelLaunchForm::Cluster;
  return invalidValue<KernelLaunchForm>("kernel launch form",
                                        canonicalSpelling);
}

llvm::StringRef stringifyKernelEntryABI(KernelEntryABI entryABI) {
  switch (entryABI) {
  case KernelEntryABI::TileMajorPointerTable:
    return "tile-major-pointer-table";
  case KernelEntryABI::TileRowPointerTable:
    return "tile-row-pointer-table";
  }
  llvm_unreachable("unknown kernel entry ABI");
}

llvm::Expected<KernelEntryABI>
parseKernelEntryABI(llvm::StringRef canonicalSpelling) {
  if (canonicalSpelling == "tile-major-pointer-table")
    return KernelEntryABI::TileMajorPointerTable;
  if (canonicalSpelling == "tile-row-pointer-table")
    return KernelEntryABI::TileRowPointerTable;
  return invalidValue<KernelEntryABI>("kernel entry ABI", canonicalSpelling);
}

llvm::StringRef stringifyRuntimeLaunchPhaseRole(RuntimeLaunchPhaseRole phase) {
  switch (phase) {
  case RuntimeLaunchPhaseRole::Prepare:
    return "prepare";
  case RuntimeLaunchPhaseRole::Main:
    return "main";
  }
  llvm_unreachable("unknown runtime launch phase role");
}

llvm::Expected<RuntimeLaunchPhaseRole>
parseRuntimeLaunchPhaseRole(llvm::StringRef canonicalSpelling) {
  if (canonicalSpelling == "prepare")
    return RuntimeLaunchPhaseRole::Prepare;
  if (canonicalSpelling == "main")
    return RuntimeLaunchPhaseRole::Main;
  return invalidValue<RuntimeLaunchPhaseRole>("runtime launch phase role",
                                              canonicalSpelling);
}

} // namespace wafer
