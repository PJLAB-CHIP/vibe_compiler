//===- TargetProfile.h - Closed Wafer target profile registry ---*- C++ -*-===//

#ifndef WAFER_TARGET_TARGETPROFILE_H
#define WAFER_TARGET_TARGETPROFILE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace wafer {

/// Closed compiler target-profile identity. Values can only be obtained from
/// the registry or from the corresponding explicitly named factory; there is
/// deliberately no default or unknown value.
class TargetProfileId {
public:
  TargetProfileId() = delete;

  static constexpr TargetProfileId waferTx81SingleCardKernelV1() {
    return TargetProfileId(Value::WaferTx81SingleCardKernelV1);
  }
  static constexpr TargetProfileId waferTx81SingleCardKernelV2() {
    return TargetProfileId(Value::WaferTx81SingleCardKernelV2);
  }
  static constexpr TargetProfileId waferTx81SingleCardKernelV3() {
    return TargetProfileId(Value::WaferTx81SingleCardKernelV3);
  }

  friend constexpr bool operator==(TargetProfileId lhs, TargetProfileId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(TargetProfileId lhs, TargetProfileId rhs) {
    return !(lhs == rhs);
  }

private:
  enum class Value : uint8_t {
    WaferTx81SingleCardKernelV1,
    WaferTx81SingleCardKernelV2,
    WaferTx81SingleCardKernelV3
  };

  explicit constexpr TargetProfileId(Value value) : value(value) {}

  Value value;
};

/// Closed target identity selected by a target profile.
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

/// Closed kernel runtime ABI identity selected by a target profile.
class KernelRuntimeABIId {
public:
  KernelRuntimeABIId() = delete;

  static constexpr KernelRuntimeABIId waferTx81KernelV1() {
    return KernelRuntimeABIId(Value::WaferTx81KernelV1);
  }
  static constexpr KernelRuntimeABIId waferTx81KernelV2() {
    return KernelRuntimeABIId(Value::WaferTx81KernelV2);
  }
  static constexpr KernelRuntimeABIId waferTx81KernelV3() {
    return KernelRuntimeABIId(Value::WaferTx81KernelV3);
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
  enum class Value : uint8_t {
    WaferTx81KernelV1,
    WaferTx81KernelV2,
    WaferTx81KernelV3
  };

  explicit constexpr KernelRuntimeABIId(Value value) : value(value) {}

  Value value;
};

/// One immutable row in the closed target-profile registry. The canonical
/// spelling is an opaque key; consumers must not recover fields by parsing it.
struct TargetProfileRecord {
  TargetProfileId id;
  llvm::StringLiteral canonicalSpelling;
  TargetIdentityId targetIdentity;
  llvm::StringLiteral targetIdentitySpelling;
  KernelRuntimeABIId kernelRuntimeABI;
  llvm::StringLiteral kernelRuntimeABISpelling;
  llvm::StringLiteral moduleFormat;
  TargetProfileId formatCompatibilityProfile;
  /// Numeric arithmetic compatibility identity. ABI-only revisions may
  /// inherit a proven numeric registry while retaining their distinct target
  /// profile and kernel ABI identities.
  TargetProfileId numericCompatibilityProfile;
};

llvm::ArrayRef<TargetProfileRecord> getRegisteredTargetProfiles();

llvm::Expected<TargetProfileId>
parseTargetProfileId(llvm::StringRef canonicalSpelling);
llvm::Expected<TargetIdentityId>
parseTargetIdentityId(llvm::StringRef canonicalSpelling);
llvm::Expected<KernelRuntimeABIId>
parseKernelRuntimeABIId(llvm::StringRef canonicalSpelling);

const TargetProfileRecord &getTargetProfileRecord(TargetProfileId id);
llvm::StringRef stringifyTargetProfileId(TargetProfileId id);
llvm::StringRef stringifyTargetIdentityId(TargetIdentityId id);
llvm::StringRef stringifyKernelRuntimeABIId(KernelRuntimeABIId id);

} // namespace wafer

#endif // WAFER_TARGET_TARGETPROFILE_H
