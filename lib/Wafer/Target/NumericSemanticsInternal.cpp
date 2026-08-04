//===- NumericSemanticsInternal.cpp - Private numeric registry helpers ===//

#include "NumericSemanticsInternal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SHA256.h"

#include <string>
#include <vector>

namespace wafer::numeric_semantics_internal {
std::string digestCanonical(llvm::StringRef canonical) {
  llvm::SHA256 hasher;
  hasher.update(canonical);
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

bool isValidDigest(llvm::StringRef digest) {
  if (digest.size() != 71 || !digest.starts_with("sha256:"))
    return false;
  for (char character : digest.drop_front(7))
    if (!llvm::isHexDigit(character) || (character >= 'A' && character <= 'F'))
      return false;
  return true;
}

bool isMPFRElementwiseOperation(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::Sqrt:
  case NumericElementwiseOperation::Rsqrt:
  case NumericElementwiseOperation::Log2:
  case NumericElementwiseOperation::Ln:
  case NumericElementwiseOperation::Pow2:
  case NumericElementwiseOperation::Exp:
  case NumericElementwiseOperation::Sin:
  case NumericElementwiseOperation::Cos:
  case NumericElementwiseOperation::Tanh:
  case NumericElementwiseOperation::Sigmoid:
  case NumericElementwiseOperation::Softplus:
    return true;
  default:
    return false;
  }
}

std::optional<NumericModelImplementationReason>
getElementwiseUnsupportedReason(NumericElementwiseOperation operation,
                                LogicalFormat inputFormat) {
  if (inputFormat == LogicalFormat::I8)
    return NumericModelImplementationReason::IntegerElementwisePolicyUnproven;
  switch (operation) {
  case NumericElementwiseOperation::ExpLp:
    return NumericModelImplementationReason::ExpLpParameterPolicyUnproven;
  case NumericElementwiseOperation::SatRelu:
    return NumericModelImplementationReason::SatReluParameterPolicyUnproven;
  case NumericElementwiseOperation::LeakyRelu:
    return NumericModelImplementationReason::LeakyReluParameterPolicyUnproven;
  default:
    return std::nullopt;
  }
}

LogicalFormat
getElementwiseDestinationFormat(NumericElementwiseOperation operation,
                                LogicalFormat inputFormat) {
  return isNumericElementwiseRelation(operation) ? LogicalFormat::Bool
                                                 : inputFormat;
}

bool isFloating(LogicalFormat format) {
  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(format);
  return descriptor &&
         descriptor->category == LogicalFormatCategory::BinaryFloatingPoint;
}

bool isInteger(LogicalFormat format) {
  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(format);
  return descriptor &&
         (descriptor->category == LogicalFormatCategory::SignedInteger ||
          descriptor->category == LogicalFormatCategory::UnsignedInteger);
}

llvm::ArrayRef<LogicalFormat>
getCompilerNumericFormats(TargetFormatEngine engine) {
  // This is the formal numeric-model domain, not target instruction legality.
  // The current target encoding registry admits every LogicalFormat.
  static constexpr LogicalFormat modeledFormats[] = {
      LogicalFormat::I8, LogicalFormat::F16, LogicalFormat::BF16,
      LogicalFormat::F32};
  switch (engine) {
  case TargetFormatEngine::CT:
    return modeledFormats;
  case TargetFormatEngine::NE:
    return modeledFormats;
  case TargetFormatEngine::RDMA:
  case TargetFormatEngine::WDMA:
  case TargetFormatEngine::TDMA:
    llvm_unreachable("numeric command registry requested a movement engine");
  }
  llvm_unreachable("target format engine is not registered");
}

} // namespace wafer::numeric_semantics_internal
