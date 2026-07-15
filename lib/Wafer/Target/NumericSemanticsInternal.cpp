//===- NumericSemanticsInternal.cpp - Private numeric registry helpers ===//

#include "NumericSemanticsInternal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SHA256.h"

#include <string>
#include <vector>

namespace wafer::numeric_semantics_internal {
namespace {

constexpr TargetProfileId kTargetProfile =
    TargetProfileId::waferTx81SingleCardKernelV1();

} // namespace

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
  static const std::vector<LogicalFormat> ctFormats = [] {
    std::vector<LogicalFormat> result;
    for (const LogicalFormatDescriptor &descriptor :
         getLogicalFormatDescriptors()) {
      const TargetFormatEncodingRecord *record = findTargetFormatEncoding(
          kTargetProfile, TargetFormatEngine::CT, descriptor.format);
      if (record && record->isSupported() &&
          descriptor.category != LogicalFormatCategory::Boolean)
        result.push_back(descriptor.format);
    }
    if (result.size() != 4)
      llvm::report_fatal_error(
          "CT compiler numeric format closure is not four rows");
    return result;
  }();
  static const std::vector<LogicalFormat> neFormats = [] {
    std::vector<LogicalFormat> result;
    for (const LogicalFormatDescriptor &descriptor :
         getLogicalFormatDescriptors()) {
      const TargetFormatEncodingRecord *record = findTargetFormatEncoding(
          kTargetProfile, TargetFormatEngine::NE, descriptor.format);
      if (record && record->isSupported() &&
          descriptor.category != LogicalFormatCategory::Boolean)
        result.push_back(descriptor.format);
    }
    if (result.size() != 4)
      llvm::report_fatal_error(
          "NE compiler numeric format closure is not four rows");
    return result;
  }();
  switch (engine) {
  case TargetFormatEngine::CT:
    return ctFormats;
  case TargetFormatEngine::NE:
    return neFormats;
  case TargetFormatEngine::RDMA:
  case TargetFormatEngine::WDMA:
  case TargetFormatEngine::TDMA:
    llvm_unreachable("numeric command registry requested a movement engine");
  }
  llvm_unreachable("target format engine is not registered");
}

} // namespace wafer::numeric_semantics_internal
