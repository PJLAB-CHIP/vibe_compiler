//===- RuntimeLaunchContract.h - Typed runtime launch contract -*- C++ -*-===//

#ifndef WAFER_TARGET_RUNTIMELAUNCHCONTRACT_H
#define WAFER_TARGET_RUNTIMELAUNCHCONTRACT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace wafer {

/// Digest-qualified TX81 V5.6 command-packet payload limit. The 0x800-byte
/// packet reserves 0x24 bytes outside the kernel argument block.
inline constexpr uint64_t kTx81KernelArgumentBytesMax = 0x7dc;

/// Digest-qualified TX81 V5.6 cluster C-INS argument limit. The cluster task
/// table reserves 0x30 bytes from its 0x800-byte packet, independently of the
/// ordinary kernel command limit above.
inline constexpr uint64_t kTx81ClusterKernelArgumentBytesMax = 0x7d0;

/// The only product-visible runtime launch kinds. Kernel command geometry,
/// entry parameter encoding and phase structure are nested typed facts rather
/// than additional launch kinds.
enum class RuntimeLaunchKind : uint8_t { Kernel, Model };

enum class KernelLaunchForm : uint8_t { Grid, Cluster };

enum class KernelEntryABI : uint8_t {
  TileMajorPointerTable,
  TileRowPointerTable,
};

enum class ModelEntryABI : uint8_t { Tx81ModelBootParam };

enum class RuntimeLaunchPhaseRole : uint8_t { Prepare, Main };

struct KernelRuntimeLaunchContract {
  KernelLaunchForm form;
  KernelEntryABI entryABI;
  std::vector<RuntimeLaunchPhaseRole> phases;
};

struct ModelRuntimeLaunchContract {
  ModelEntryABI entryABI;
  std::vector<RuntimeLaunchPhaseRole> phases;
};

/// Closed tagged runtime launch contract. Values can only be created through
/// the validating factories, so malformed form/entry-ABI/phase cross-products
/// cannot enter generated target modules or package construction.
class RuntimeLaunchContract {
public:
  RuntimeLaunchContract() = delete;

  static llvm::Expected<RuntimeLaunchContract>
  createKernel(KernelLaunchForm form, KernelEntryABI entryABI,
               llvm::ArrayRef<RuntimeLaunchPhaseRole> phases);

  static llvm::Expected<RuntimeLaunchContract>
  createModel(ModelEntryABI entryABI,
              llvm::ArrayRef<RuntimeLaunchPhaseRole> phases);

  RuntimeLaunchKind getKind() const;
  const KernelRuntimeLaunchContract *getKernel() const;
  const ModelRuntimeLaunchContract *getModel() const;
  llvm::ArrayRef<RuntimeLaunchPhaseRole> getPhases() const;

  friend bool operator==(const RuntimeLaunchContract &lhs,
                         const RuntimeLaunchContract &rhs);
  friend bool operator!=(const RuntimeLaunchContract &lhs,
                         const RuntimeLaunchContract &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit RuntimeLaunchContract(KernelRuntimeLaunchContract kernel)
      : contract(std::move(kernel)) {}
  explicit RuntimeLaunchContract(ModelRuntimeLaunchContract model)
      : contract(std::move(model)) {}

  std::variant<KernelRuntimeLaunchContract, ModelRuntimeLaunchContract>
      contract;
};

llvm::StringRef stringifyRuntimeLaunchKind(RuntimeLaunchKind kind);
llvm::Expected<RuntimeLaunchKind>
parseRuntimeLaunchKind(llvm::StringRef canonicalSpelling);

llvm::StringRef stringifyKernelLaunchForm(KernelLaunchForm form);
llvm::Expected<KernelLaunchForm>
parseKernelLaunchForm(llvm::StringRef canonicalSpelling);

llvm::StringRef stringifyKernelEntryABI(KernelEntryABI entryABI);
llvm::Expected<KernelEntryABI>
parseKernelEntryABI(llvm::StringRef canonicalSpelling);

llvm::StringRef stringifyModelEntryABI(ModelEntryABI entryABI);
llvm::Expected<ModelEntryABI>
parseModelEntryABI(llvm::StringRef canonicalSpelling);

llvm::StringRef stringifyRuntimeLaunchPhaseRole(RuntimeLaunchPhaseRole phase);
llvm::Expected<RuntimeLaunchPhaseRole>
parseRuntimeLaunchPhaseRole(llvm::StringRef canonicalSpelling);

} // namespace wafer

#endif // WAFER_TARGET_RUNTIMELAUNCHCONTRACT_H
