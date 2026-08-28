//===- TargetIdentity.h - Current Wafer target identity --------*- C++ -*-===//

#ifndef WAFER_TARGET_TARGETIDENTITY_H
#define WAFER_TARGET_TARGETIDENTITY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace wafer {

/// Serialized identifier for the current Wafer target. Compilation does not
/// select among profiles; this value keeps module, package, and runtime checks
/// typed and exact.
class TargetIdentityId {
public:
  TargetIdentityId() = delete;

  static constexpr TargetIdentityId waferTx81SingleCard() {
    return TargetIdentityId(Value::WaferTx81SingleCard);
  }

  friend constexpr bool operator==(TargetIdentityId lhs, TargetIdentityId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(TargetIdentityId lhs, TargetIdentityId rhs) {
    return !(lhs == rhs);
  }

private:
  enum class Value : uint8_t { WaferTx81SingleCard };

  explicit constexpr TargetIdentityId(Value value) : value(value) {}
  Value value;
};

/// Serialized identifier for the current worker-aware kernel ABI.
class KernelRuntimeABIId {
public:
  KernelRuntimeABIId() = delete;

  static constexpr KernelRuntimeABIId waferTx81Kernel() {
    return KernelRuntimeABIId(Value::WaferTx81Kernel);
  }

  friend constexpr bool operator==(KernelRuntimeABIId lhs,
                                   KernelRuntimeABIId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(KernelRuntimeABIId lhs,
                                   KernelRuntimeABIId rhs) {
    return !(lhs == rhs);
  }

private:
  enum class Value : uint8_t { WaferTx81Kernel };

  explicit constexpr KernelRuntimeABIId(Value value) : value(value) {}
  Value value;
};

inline constexpr TargetIdentityId kCurrentTargetIdentity =
    TargetIdentityId::waferTx81SingleCard();
inline constexpr KernelRuntimeABIId kCurrentKernelRuntimeABI =
    KernelRuntimeABIId::waferTx81Kernel();
inline constexpr llvm::StringLiteral kCurrentTargetModuleFormat = "elf-riscv64";

llvm::Expected<TargetIdentityId>
parseTargetIdentityId(llvm::StringRef canonicalSpelling);
llvm::Expected<KernelRuntimeABIId>
parseKernelRuntimeABIId(llvm::StringRef canonicalSpelling);

llvm::StringRef stringifyTargetIdentityId(TargetIdentityId id);
llvm::StringRef stringifyKernelRuntimeABIId(KernelRuntimeABIId id);

} // namespace wafer

#endif // WAFER_TARGET_TARGETIDENTITY_H
