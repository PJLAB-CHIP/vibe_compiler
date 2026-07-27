//===- TargetNumericCapability.h - Target numeric admission ----*- C++ -*-===//

#ifndef WAFER_TARGET_TARGETNUMERICCAPABILITY_H
#define WAFER_TARGET_TARGETNUMERICCAPABILITY_H

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/TargetFormat.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace wafer {

/// Numeric semantic contract requested by the accepted instruction program.
/// Current Instr IR carries exact source semantics and has no relaxed
/// signed-zero marker, so target lowering may only request SourceExact.
enum class TargetNumericSemanticRequirement : uint8_t { SourceExact };

enum class TargetNumericCapabilitySupport : uint8_t {
  Unsupported,
  Supported,
};

enum class TargetNumericUnsupportedReason : uint8_t {
  None,
  IntegerElementwisePolicyUnproven,
  ElementwiseParameterPolicyUnproven,
  ExactSignedZeroPolicyUnproven,
};

/// One closed CT elementwise numeric-admission row. This is separate from
/// TargetFormatEncodingRecord: an instruction can have a valid ABI format code
/// while its arithmetic semantics remain unsafe to execute.
struct TargetCTElementwiseNumericCapabilityRecord {
  TargetProfileId profile;
  InstrElementwiseKind kind;
  LogicalFormat inputFormat;
  TargetNumericSemanticRequirement requirement;
  TargetNumericCapabilitySupport support;
  TargetNumericUnsupportedReason unsupportedReason;

  constexpr bool isSupported() const {
    return support == TargetNumericCapabilitySupport::Supported;
  }
};

llvm::ArrayRef<TargetCTElementwiseNumericCapabilityRecord>
getTargetCTElementwiseNumericCapabilityRecords();

const TargetCTElementwiseNumericCapabilityRecord *
findTargetCTElementwiseNumericCapability(
    TargetProfileId profile, InstrElementwiseKind kind,
    LogicalFormat inputFormat, TargetNumericSemanticRequirement requirement);

llvm::StringRef stringifyTargetNumericSemanticRequirement(
    TargetNumericSemanticRequirement requirement);
llvm::StringRef
stringifyTargetNumericUnsupportedReason(TargetNumericUnsupportedReason reason);

} // namespace wafer

#endif // WAFER_TARGET_TARGETNUMERICCAPABILITY_H
