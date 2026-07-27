//===- TargetNumericCapability.cpp - Target numeric admission -----------===//

#include "Wafer/Target/TargetNumericCapability.h"

#include "llvm/Support/ErrorHandling.h"

#include <set>
#include <tuple>
#include <vector>

namespace wafer {
namespace {

using Requirement = TargetNumericSemanticRequirement;
using Support = TargetNumericCapabilitySupport;
using Reason = TargetNumericUnsupportedReason;

constexpr TargetProfileId kProfile =
    TargetProfileId::waferTx81SingleCardKernelV1();

constexpr InstrElementwiseKind kElementwiseKinds[] = {
    InstrElementwiseKind::Abs,      InstrElementwiseKind::Recip,
    InstrElementwiseKind::Square,   InstrElementwiseKind::Sqrt,
    InstrElementwiseKind::Rsqrt,    InstrElementwiseKind::Neg,
    InstrElementwiseKind::Max,      InstrElementwiseKind::Min,
    InstrElementwiseKind::Add,      InstrElementwiseKind::Sub,
    InstrElementwiseKind::Mul,      InstrElementwiseKind::Div,
    InstrElementwiseKind::Eq,       InstrElementwiseKind::Ne,
    InstrElementwiseKind::Ge,       InstrElementwiseKind::Gt,
    InstrElementwiseKind::Le,       InstrElementwiseKind::Lt,
    InstrElementwiseKind::LogicNot, InstrElementwiseKind::LogicAnd,
    InstrElementwiseKind::LogicOr,  InstrElementwiseKind::LogicXor,
    InstrElementwiseKind::Log2,     InstrElementwiseKind::Ln,
    InstrElementwiseKind::Pow2,     InstrElementwiseKind::Exp,
    InstrElementwiseKind::ExpLp,    InstrElementwiseKind::Sin,
    InstrElementwiseKind::Cos,      InstrElementwiseKind::Tanh,
    InstrElementwiseKind::Sigmoid,  InstrElementwiseKind::Relu,
    InstrElementwiseKind::SatRelu,  InstrElementwiseKind::LeakyRelu,
    InstrElementwiseKind::Softplus,
};

constexpr LogicalFormat kValueFormats[] = {
    LogicalFormat::I8,
    LogicalFormat::F16,
    LogicalFormat::BF16,
    LogicalFormat::F32,
};

constexpr bool isLogic(InstrElementwiseKind kind) {
  switch (kind) {
  case InstrElementwiseKind::LogicNot:
  case InstrElementwiseKind::LogicAnd:
  case InstrElementwiseKind::LogicOr:
  case InstrElementwiseKind::LogicXor:
    return true;
  default:
    return false;
  }
}

constexpr Reason getUnsupportedReason(InstrElementwiseKind kind,
                                      LogicalFormat inputFormat) {
  if (inputFormat == LogicalFormat::I8)
    return Reason::IntegerElementwisePolicyUnproven;
  switch (kind) {
  case InstrElementwiseKind::ExpLp:
  case InstrElementwiseKind::SatRelu:
  case InstrElementwiseKind::LeakyRelu:
    return Reason::ElementwiseParameterPolicyUnproven;
  case InstrElementwiseKind::Neg:
    // The current TX81 evidence canonicalizes a zero result to +0, while
    // accepted source Neg requires the sign of zero to be inverted exactly.
    return Reason::ExactSignedZeroPolicyUnproven;
  default:
    return Reason::None;
  }
}

TargetCTElementwiseNumericCapabilityRecord
makeRecord(InstrElementwiseKind kind, LogicalFormat inputFormat) {
  Reason reason = getUnsupportedReason(kind, inputFormat);
  return {kProfile,
          kind,
          inputFormat,
          Requirement::SourceExact,
          reason == Reason::None ? Support::Supported : Support::Unsupported,
          reason};
}

std::vector<TargetCTElementwiseNumericCapabilityRecord> buildRecords() {
  std::vector<TargetCTElementwiseNumericCapabilityRecord> records;
  records.reserve(128);
  for (InstrElementwiseKind kind : kElementwiseKinds) {
    if (isLogic(kind)) {
      records.push_back(makeRecord(kind, LogicalFormat::Bool));
      continue;
    }
    for (LogicalFormat format : kValueFormats)
      records.push_back(makeRecord(kind, format));
  }

  if (records.size() != 128)
    llvm::report_fatal_error(
        "target CT elementwise numeric registry is not 128 rows");
  std::set<std::tuple<unsigned, unsigned, unsigned>> keys;
  for (const TargetCTElementwiseNumericCapabilityRecord &record : records) {
    if (!keys.insert({static_cast<unsigned>(record.kind),
                      static_cast<unsigned>(record.inputFormat),
                      static_cast<unsigned>(record.requirement)})
             .second)
      llvm::report_fatal_error(
          "target CT elementwise numeric registry has duplicate rows");
    const TargetFormatEncodingRecord *format = findTargetFormatEncoding(
        record.profile, TargetFormatEngine::CT, record.inputFormat);
    if (!format || !format->isSupported())
      llvm::report_fatal_error(
          "target CT elementwise numeric row has no encodable CT format");
    if ((record.unsupportedReason == Reason::None) != record.isSupported())
      llvm::report_fatal_error(
          "target CT elementwise numeric row has inconsistent support");
  }
  return records;
}

} // namespace

llvm::ArrayRef<TargetCTElementwiseNumericCapabilityRecord>
getTargetCTElementwiseNumericCapabilityRecords() {
  static const std::vector<TargetCTElementwiseNumericCapabilityRecord> records =
      buildRecords();
  return records;
}

const TargetCTElementwiseNumericCapabilityRecord *
findTargetCTElementwiseNumericCapability(
    TargetProfileId profile, InstrElementwiseKind kind,
    LogicalFormat inputFormat, TargetNumericSemanticRequirement requirement) {
  profile = getTargetProfileRecord(profile).numericCompatibilityProfile;
  const TargetCTElementwiseNumericCapabilityRecord *match = nullptr;
  for (const TargetCTElementwiseNumericCapabilityRecord &record :
       getTargetCTElementwiseNumericCapabilityRecords()) {
    if (record.profile != profile || record.kind != kind ||
        record.inputFormat != inputFormat || record.requirement != requirement)
      continue;
    if (match)
      llvm::report_fatal_error(
          "target CT elementwise numeric lookup is ambiguous");
    match = &record;
  }
  return match;
}

llvm::StringRef stringifyTargetNumericSemanticRequirement(
    TargetNumericSemanticRequirement requirement) {
  switch (requirement) {
  case Requirement::SourceExact:
    return "source-exact";
  }
  llvm_unreachable("unknown target numeric semantic requirement");
}

llvm::StringRef
stringifyTargetNumericUnsupportedReason(TargetNumericUnsupportedReason reason) {
  switch (reason) {
  case Reason::None:
    return "none";
  case Reason::IntegerElementwisePolicyUnproven:
    return "integer-elementwise-policy-unproven";
  case Reason::ElementwiseParameterPolicyUnproven:
    return "elementwise-parameter-policy-unproven";
  case Reason::ExactSignedZeroPolicyUnproven:
    return "exact-signed-zero-policy-unproven";
  }
  llvm_unreachable("unknown target numeric unsupported reason");
}

} // namespace wafer
