//===- RuntimeLaunchContract.h - Typed runtime launch contract -*- C++ -*-===//

#ifndef WAFER_TARGET_RUNTIMELAUNCHCONTRACT_H
#define WAFER_TARGET_RUNTIMELAUNCHCONTRACT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace wafer {

/// Digest-qualified TX81 V5.6 command-packet payload limit. The 0x800-byte
/// packet reserves 0x24 bytes outside the kernel argument block.
inline constexpr uint64_t kTx81KernelArgumentBytesMax = 0x7dc;

/// Digest-qualified TX81 V5.6 cluster C-INS argument limit. The cluster task
/// table reserves 0x30 bytes from its 0x800-byte packet, independently of the
/// ordinary kernel command limit above.
inline constexpr uint64_t kTx81ClusterKernelArgumentBytesMax = 0x7d0;

enum class KernelLaunchForm : uint8_t { Grid, Cluster };

enum class KernelEntryABI : uint8_t {
  TileMajorPointerTable,
  TileRowPointerTable,
};

enum class RuntimeLaunchPhaseRole : uint8_t { Prepare, Main };

struct KernelRuntimeLaunchContract {
  KernelLaunchForm form;
  KernelEntryABI entryABI;
  std::vector<RuntimeLaunchPhaseRole> phases;
};

/// Closed kernel runtime launch contract. Values can only be created through
/// the validating factory, so malformed form/entry-ABI/phase cross-products
/// cannot enter generated target modules or package construction. The current
/// product provider only implements the `txLaunchKernel` family; model launch
/// kinds are retired.
class RuntimeLaunchContract {
public:
  RuntimeLaunchContract() = delete;

  static llvm::Expected<RuntimeLaunchContract>
  createKernel(KernelLaunchForm form, KernelEntryABI entryABI,
               llvm::ArrayRef<RuntimeLaunchPhaseRole> phases);

  const KernelRuntimeLaunchContract &getKernel() const { return contract; }
  llvm::ArrayRef<RuntimeLaunchPhaseRole> getPhases() const {
    return contract.phases;
  }

  friend bool operator==(const RuntimeLaunchContract &lhs,
                         const RuntimeLaunchContract &rhs);
  friend bool operator!=(const RuntimeLaunchContract &lhs,
                         const RuntimeLaunchContract &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit RuntimeLaunchContract(KernelRuntimeLaunchContract kernel)
      : contract(std::move(kernel)) {}

  KernelRuntimeLaunchContract contract;
};

llvm::StringRef stringifyKernelLaunchForm(KernelLaunchForm form);
llvm::Expected<KernelLaunchForm>
parseKernelLaunchForm(llvm::StringRef canonicalSpelling);

llvm::StringRef stringifyKernelEntryABI(KernelEntryABI entryABI);
llvm::Expected<KernelEntryABI>
parseKernelEntryABI(llvm::StringRef canonicalSpelling);

llvm::StringRef stringifyRuntimeLaunchPhaseRole(RuntimeLaunchPhaseRole phase);
llvm::Expected<RuntimeLaunchPhaseRole>
parseRuntimeLaunchPhaseRole(llvm::StringRef canonicalSpelling);

} // namespace wafer

#endif // WAFER_TARGET_RUNTIMELAUNCHCONTRACT_H
