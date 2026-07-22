//===- TargetLaunchABI.h - Closed target launch ABI registry ---*- C++ -*-===//

#ifndef WAFER_TARGET_TARGETLAUNCHABI_H
#define WAFER_TARGET_TARGETLAUNCHABI_H

#include "Wafer/Target/TargetProfile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace wafer {

/// Digest-qualified TX81 V5.6 command-packet payload limit. The 0x800-byte
/// packet reserves 0x24 bytes outside the kernel argument block.
inline constexpr uint64_t kTx81KernelArgumentBytesMax = 0x7dc;

/// Closed identity for the host/device entry and submission ABI. Launch ABI
/// is deliberately orthogonal to target hardware and numeric profiles.
class TargetLaunchABIId {
public:
  TargetLaunchABIId() = delete;

  static constexpr TargetLaunchABIId perRankPointerBlockV1() {
    return TargetLaunchABIId(Value::PerRankPointerBlockV1);
  }
  static constexpr TargetLaunchABIId tx81KernelGridPointerTableV1() {
    return TargetLaunchABIId(Value::Tx81KernelGridPointerTableV1);
  }
  static constexpr TargetLaunchABIId tx81ModelBootParamV1() {
    return TargetLaunchABIId(Value::Tx81ModelBootParamV1);
  }

  friend constexpr bool operator==(TargetLaunchABIId lhs,
                                   TargetLaunchABIId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(TargetLaunchABIId lhs,
                                   TargetLaunchABIId rhs) {
    return !(lhs == rhs);
  }

private:
  enum class Value : uint8_t {
    PerRankPointerBlockV1,
    Tx81KernelGridPointerTableV1,
    Tx81ModelBootParamV1,
  };

  explicit constexpr TargetLaunchABIId(Value value) : value(value) {}

  Value value;
};

struct TargetLaunchABIRecord {
  TargetLaunchABIId id;
  llvm::StringLiteral canonicalSpelling;
};

llvm::ArrayRef<TargetLaunchABIRecord> getRegisteredTargetLaunchABIs();
llvm::Expected<TargetLaunchABIId>
parseTargetLaunchABIId(llvm::StringRef canonicalSpelling);
const TargetLaunchABIRecord &getTargetLaunchABIRecord(TargetLaunchABIId id);
llvm::StringRef stringifyTargetLaunchABIId(TargetLaunchABIId id);
bool isTargetLaunchABICompatible(TargetLaunchABIId launchABI,
                                 TargetProfileId targetProfile);

} // namespace wafer

#endif // WAFER_TARGET_TARGETLAUNCHABI_H
