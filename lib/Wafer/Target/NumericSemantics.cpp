//===- NumericSemantics.cpp - Closed numeric capability resolution ------===//

#include "Wafer/Target/NumericSemantics.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace wafer {
namespace {

constexpr ModelProfileId kFormalDeterministicV1 =
    ModelProfileId::formalDeterministicV1();
constexpr TargetProfileId kTargetProfile =
    TargetProfileId::waferTx81SingleCardKernelV1();

constexpr NumericRoundingMode kRoundingModes[] = {
    NumericRoundingMode::NearestEven,    NumericRoundingMode::TowardZero,
    NumericRoundingMode::TowardPositive, NumericRoundingMode::TowardNegative,
    NumericRoundingMode::Stochastic,
};
constexpr NumericRoundingMode kDeterministicRoundingModes[] = {
    NumericRoundingMode::NearestEven,
    NumericRoundingMode::TowardZero,
    NumericRoundingMode::TowardPositive,
    NumericRoundingMode::TowardNegative,
};
constexpr NumericElementwiseOperation kElementwiseOperations[] = {
    NumericElementwiseOperation::Abs,
    NumericElementwiseOperation::Recip,
    NumericElementwiseOperation::Square,
    NumericElementwiseOperation::Sqrt,
    NumericElementwiseOperation::Rsqrt,
    NumericElementwiseOperation::Neg,
    NumericElementwiseOperation::Max,
    NumericElementwiseOperation::Min,
    NumericElementwiseOperation::Add,
    NumericElementwiseOperation::Sub,
    NumericElementwiseOperation::Mul,
    NumericElementwiseOperation::Div,
    NumericElementwiseOperation::Eq,
    NumericElementwiseOperation::Ne,
    NumericElementwiseOperation::Ge,
    NumericElementwiseOperation::Gt,
    NumericElementwiseOperation::Le,
    NumericElementwiseOperation::Lt,
    NumericElementwiseOperation::LogicNot,
    NumericElementwiseOperation::LogicAnd,
    NumericElementwiseOperation::LogicOr,
    NumericElementwiseOperation::LogicXor,
    NumericElementwiseOperation::Log2,
    NumericElementwiseOperation::Ln,
    NumericElementwiseOperation::Pow2,
    NumericElementwiseOperation::Exp,
    NumericElementwiseOperation::ExpLp,
    NumericElementwiseOperation::Sin,
    NumericElementwiseOperation::Cos,
    NumericElementwiseOperation::Tanh,
    NumericElementwiseOperation::Sigmoid,
    NumericElementwiseOperation::Relu,
    NumericElementwiseOperation::SatRelu,
    NumericElementwiseOperation::LeakyRelu,
    NumericElementwiseOperation::Softplus,
};
constexpr NumericReduceOperation kReduceOperations[] = {
    NumericReduceOperation::Sum,
    NumericReduceOperation::Max,
    NumericReduceOperation::Min,
    NumericReduceOperation::Avg,
};

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

std::string makeModelPolicyDigest(const ModelProfileRecord &record) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream
      << "wafer-model-policy-v1\n"
      << "model=" << record.canonicalSpelling << '\n'
      << "decode-byte-order="
      << static_cast<unsigned>(record.numericDecodePolicy.byteOrder) << '\n'
      << "decode-bit-order="
      << static_cast<unsigned>(record.numericDecodePolicy.bitOrder) << '\n'
      << "decode-noncanonical="
      << static_cast<unsigned>(record.numericDecodePolicy.nonCanonicalEncoding)
      << '\n'
      << "encode-byte-order="
      << static_cast<unsigned>(record.numericEncodePolicy.byteOrder) << '\n'
      << "encode-bit-order="
      << static_cast<unsigned>(record.numericEncodePolicy.bitOrder) << '\n'
      << "encode-noncanonical="
      << static_cast<unsigned>(record.numericEncodePolicy.nonCanonicalEncoding)
      << '\n'
      << "model-only=" << static_cast<unsigned>(record.modelOnly) << '\n';
  stream.flush();
  return digestCanonical(canonical);
}

bool isKnownLayout(NumericTensorLayout layout) {
  switch (layout) {
  case NumericTensorLayout::Tensor:
  case NumericTensorLayout::NTensor:
  case NumericTensorLayout::Cx:
  case NumericTensorLayout::NCx:
    return true;
  }
  return false;
}

bool isKnownRoundingMode(NumericRoundingMode mode) {
  switch (mode) {
  case NumericRoundingMode::NearestEven:
  case NumericRoundingMode::TowardZero:
  case NumericRoundingMode::TowardPositive:
  case NumericRoundingMode::TowardNegative:
  case NumericRoundingMode::Stochastic:
    return true;
  }
  return false;
}

bool isKnownElementwiseOperation(NumericElementwiseOperation operation) {
  return std::find(std::begin(kElementwiseOperations),
                   std::end(kElementwiseOperations),
                   operation) != std::end(kElementwiseOperations);
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

bool isKnownReduceOperation(NumericReduceOperation operation) {
  return std::find(std::begin(kReduceOperations), std::end(kReduceOperations),
                   operation) != std::end(kReduceOperations);
}

bool isKnownReduceDimension(NativeCTReduceDimension dimension) {
  switch (dimension) {
  case NativeCTReduceDimension::Trailing0:
  case NativeCTReduceDimension::Trailing1:
  case NativeCTReduceDimension::Trailing2:
  case NativeCTReduceDimension::Trailing3:
  case NativeCTReduceDimension::Trailing2And1:
  case NativeCTReduceDimension::Trailing2And1And0:
    return true;
  }
  return false;
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

bool isAlignedLayout(NumericTensorLayout layout) {
  return layout == NumericTensorLayout::Cx ||
         layout == NumericTensorLayout::NCx;
}

bool checkedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

std::string makeTensorDigest(LogicalFormat format, NumericTensorLayout layout,
                             llvm::ArrayRef<uint64_t> shape,
                             uint64_t elementCount) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream << "wafer-numeric-tensor-key-v1\n"
         << "format=" << stringifyLogicalFormat(format) << '\n'
         << "layout=" << stringifyNumericTensorLayout(layout) << '\n'
         << "rank=" << shape.size() << '\n';
  for (auto [index, dimension] : llvm::enumerate(shape))
    stream << "dim-" << index << '=' << dimension << '\n';
  stream << "element-count=" << elementCount << '\n';
  stream.flush();
  return digestCanonical(canonical);
}

void writeTensorDigest(llvm::raw_ostream &stream, llvm::StringRef role,
                       const NumericTensorKey &tensor) {
  stream << role << "-tensor-digest=" << tensor.getDigest() << '\n';
}

std::string makeCommandKeyDigest(TargetProfileId targetProfile,
                                 const NumericCommandPayload &payload) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream << "wafer-numeric-command-key-v2\n"
         << "target=" << stringifyTargetProfileId(targetProfile) << '\n';
  std::visit(
      [&](const auto &command) {
        using Command = std::decay_t<decltype(command)>;
        if constexpr (std::is_same_v<Command, NumericCTConvertCommand>) {
          const TargetConvertRoute *route =
              findTargetConvertRoute(targetProfile, command.opcode);
          if (!route)
            llvm::report_fatal_error("validated convert key lost its route");
          stream << "family=ct-convert\n"
                 << "opcode=" << command.opcode << '\n'
                 << "route=" << route->canonicalSpelling << '\n';
          writeTensorDigest(stream, "source", command.source);
          writeTensorDigest(stream, "destination", command.destination);
          if (!command.parameter) {
            stream << "parameter=none\n";
          } else if (std::optional<NumericRoundingMode> mode =
                         command.parameter->getRoundingMode()) {
            stream << "rounding-mode=" << stringifyNumericRoundingMode(*mode)
                   << '\n';
          } else {
            stream << "zero-point=" << *command.parameter->getZeroPoint()
                   << '\n';
          }
        } else if constexpr (std::is_same_v<Command,
                                            NumericCTElementwiseCommand>) {
          stream << "family=ct-elementwise\n"
                 << "operation="
                 << stringifyNumericElementwiseOperation(command.operation)
                 << '\n'
                 << "arity=" << command.inputs.size() << '\n';
          for (auto [index, input] : llvm::enumerate(command.inputs))
            writeTensorDigest(
                stream, (llvm::Twine("input-") + llvm::Twine(index)).str(),
                input);
          writeTensorDigest(stream, "destination", command.destination);
        } else if constexpr (std::is_same_v<Command, NumericNEGemmCommand>) {
          stream << "family=ne-gemm\n";
          writeTensorDigest(stream, "lhs", command.lhs);
          writeTensorDigest(stream, "rhs", command.rhs);
          writeTensorDigest(stream, "destination", command.destination);
          stream << "m=" << command.m << '\n'
                 << "k=" << command.k << '\n'
                 << "n=" << command.n << '\n'
                 << "batch-count=" << command.batchCount << '\n';
          for (auto [index, dimension] :
               llvm::enumerate(command.axes.lhsBatchDimensions))
            stream << "lhs-batch-dim-" << index << '=' << dimension << '\n';
          stream << "lhs-m-dim=" << command.axes.lhsMDimension << '\n'
                 << "lhs-contracting-dim="
                 << command.axes.lhsContractingDimension << '\n';
          for (auto [index, dimension] :
               llvm::enumerate(command.axes.rhsBatchDimensions))
            stream << "rhs-batch-dim-" << index << '=' << dimension << '\n';
          stream << "rhs-contracting-dim="
                 << command.axes.rhsContractingDimension << '\n'
                 << "rhs-n-dim=" << command.axes.rhsNDimension << '\n';
          for (auto [index, dimension] :
               llvm::enumerate(command.axes.destinationBatchDimensions))
            stream << "destination-batch-dim-" << index << '=' << dimension
                   << '\n';
          stream << "destination-m-dim=" << command.axes.destinationMDimension
                 << '\n'
                 << "destination-n-dim=" << command.axes.destinationNDimension
                 << '\n';
        } else if constexpr (std::is_same_v<Command,
                                            NumericNativeCTReduceCommand>) {
          stream << "family=native-ct-reduce\n"
                 << "operation="
                 << stringifyNumericReduceOperation(command.operation) << '\n'
                 << "dimension="
                 << stringifyNativeCTReduceDimension(command.dimension) << '\n';
          writeTensorDigest(stream, "input", command.input);
          writeTensorDigest(stream, "destination", command.destination);
        }
      },
      payload);
  stream.flush();
  return digestCanonical(canonical);
}

std::string makeSemanticsDigest(
    const ModelProfileRecord &model, const NumericSemanticsIdentity &identity,
    std::optional<NumericRoundingMode> roundingMode,
    NumericRoundingPointPolicy roundingPointPolicy,
    FloatToIntegerPolicy floatToIntegerPolicy,
    FloatingNaNPolicy floatingNaNPolicy,
    FloatingSignedZeroPolicy floatingSignedZeroPolicy,
    FloatingSubnormalPolicy floatingSubnormalPolicy,
    FloatingTininessPolicy floatingTininessPolicy,
    NumericExceptionFlagPolicy exceptionFlagPolicy,
    FloatingNaNSignalingPolicy floatingNaNSignalingPolicy,
    FloatingDenormalModePolicy floatingDenormalModePolicy,
    FloatingOverflowPolicy floatingOverflowPolicy,
    NumericSaturationPolicy saturationPolicy,
    NumericTranscendentalEvaluationPolicy transcendentalEvaluationPolicy,
    NumericGemmAccumulatorPolicy gemmAccumulatorPolicy,
    NumericGemmAccumulatorInitializationPolicy
        gemmAccumulatorInitializationPolicy,
    NumericGemmReductionOrderPolicy gemmReductionOrderPolicy) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream << "wafer-numeric-semantics-v3\n"
         << "model-policy-digest=" << model.policyDigest << '\n';
  std::visit(
      [&](const auto &typedIdentity) {
        using Identity = std::decay_t<decltype(typedIdentity)>;
        stream << "target="
               << stringifyTargetProfileId(typedIdentity.getTargetProfile())
               << '\n'
               << "family="
               << stringifyNumericCommandFamily(typedIdentity.getFamily())
               << '\n';
        if constexpr (std::is_same_v<Identity,
                                     NumericCTConvertSemanticsIdentity>) {
          const TargetConvertRoute &route = typedIdentity.getCTConvertRoute();
          stream << "opcode=" << route.opcode << '\n'
                 << "route=" << route.canonicalSpelling << '\n'
                 << "source=" << stringifyLogicalFormat(route.source) << '\n'
                 << "destination=" << stringifyLogicalFormat(route.destination)
                 << '\n';
        } else if constexpr (std::is_same_v<
                                 Identity,
                                 NumericCTElementwiseSemanticsIdentity>) {
          stream << "operation="
                 << stringifyNumericElementwiseOperation(
                        typedIdentity.getOperation())
                 << '\n'
                 << "input-format="
                 << stringifyLogicalFormat(typedIdentity.getInputFormat())
                 << '\n'
                 << "destination-format="
                 << stringifyLogicalFormat(typedIdentity.getDestinationFormat())
                 << '\n';
        } else if constexpr (std::is_same_v<Identity,
                                            NumericNEGemmSemanticsIdentity>) {
          stream << "format="
                 << stringifyLogicalFormat(typedIdentity.getFormat()) << '\n';
        }
      },
      identity);
  if (roundingMode)
    stream << "rounding-mode=" << stringifyNumericRoundingMode(*roundingMode)
           << '\n';
  else
    stream << "rounding-mode=none\n";
  stream << "rounding-point=" << static_cast<unsigned>(roundingPointPolicy)
         << '\n'
         << "float-to-int=" << static_cast<unsigned>(floatToIntegerPolicy)
         << '\n'
         << "nan=" << static_cast<unsigned>(floatingNaNPolicy) << '\n'
         << "signed-zero=" << static_cast<unsigned>(floatingSignedZeroPolicy)
         << '\n'
         << "subnormal=" << static_cast<unsigned>(floatingSubnormalPolicy)
         << '\n'
         << "tininess=" << static_cast<unsigned>(floatingTininessPolicy) << '\n'
         << "flags=" << static_cast<unsigned>(exceptionFlagPolicy) << '\n'
         << "nan-signaling="
         << static_cast<unsigned>(floatingNaNSignalingPolicy) << '\n'
         << "denormal-mode="
         << static_cast<unsigned>(floatingDenormalModePolicy) << '\n'
         << "overflow=" << static_cast<unsigned>(floatingOverflowPolicy) << '\n'
         << "saturation=" << static_cast<unsigned>(saturationPolicy) << '\n'
         << "transcendental-evaluation="
         << static_cast<unsigned>(transcendentalEvaluationPolicy) << '\n'
         << "gemm-accumulator=" << static_cast<unsigned>(gemmAccumulatorPolicy)
         << '\n'
         << "gemm-accumulator-initialization="
         << static_cast<unsigned>(gemmAccumulatorInitializationPolicy) << '\n'
         << "gemm-reduction-order="
         << static_cast<unsigned>(gemmReductionOrderPolicy) << '\n';
  stream.flush();
  return digestCanonical(canonical);
}

void writeSelector(llvm::raw_ostream &stream, TargetProfileId targetProfile,
                   const NumericCapabilitySelector &selector) {
  std::visit(
      [&](const auto &typedSelector) {
        using Selector = std::decay_t<decltype(typedSelector)>;
        if constexpr (std::is_same_v<Selector,
                                     NumericCTConvertPatternSelector>) {
          const TargetConvertRoute *route =
              findTargetConvertRoute(targetProfile, typedSelector.opcode);
          if (!route)
            llvm::report_fatal_error("convert pattern lost its route");
          stream << "opcode=" << typedSelector.opcode << '\n'
                 << "route=" << route->canonicalSpelling << '\n'
                 << "parameter-pattern="
                 << static_cast<unsigned>(typedSelector.parameter.getKind())
                 << '\n';
          if (std::optional<NumericRoundingMode> mode =
                  typedSelector.parameter.getExactRoundingMode())
            stream << "exact-rounding=" << stringifyNumericRoundingMode(*mode)
                   << '\n';
        } else if constexpr (std::is_same_v<
                                 Selector,
                                 NumericCTElementwisePatternSelector>) {
          stream << "operation="
                 << stringifyNumericElementwiseOperation(
                        typedSelector.operation)
                 << '\n'
                 << "input-format="
                 << stringifyLogicalFormat(typedSelector.inputFormat) << '\n';
        } else if constexpr (std::is_same_v<Selector,
                                            NumericNEGemmPatternSelector>) {
          stream << "format=" << stringifyLogicalFormat(typedSelector.format)
                 << '\n';
        } else if constexpr (std::is_same_v<
                                 Selector,
                                 NumericNativeCTReducePatternSelector>) {
          stream << "operation="
                 << stringifyNumericReduceOperation(typedSelector.operation)
                 << '\n'
                 << "format=" << stringifyLogicalFormat(typedSelector.format)
                 << '\n';
        }
      },
      selector);
}

std::string
makePatternDigest(const ModelProfileRecord &model,
                  TargetProfileId targetProfile, NumericCommandFamily family,
                  const NumericCapabilitySelector &selector,
                  NumericModelImplementationCapability modelCapability,
                  NumericCompilerEmittabilityCapability compilerCapability,
                  NumericEvidenceCapability evidenceCapability,
                  const NumericSemanticsProfile *semantics,
                  std::optional<FormalKernelKind> formalKernel,
                  std::optional<NumericComparatorKind> comparator,
                  std::optional<FormalNumericBackendKind> formalBackend) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream << "wafer-numeric-capability-pattern-v2\n"
         << "model-policy-digest=" << model.policyDigest << '\n'
         << "target=" << stringifyTargetProfileId(targetProfile) << '\n'
         << "family=" << stringifyNumericCommandFamily(family) << '\n';
  writeSelector(stream, targetProfile, selector);
  stream << "model-status=" << static_cast<unsigned>(modelCapability.status)
         << '\n'
         << "model-reason="
         << stringifyNumericModelImplementationReason(modelCapability.reason)
         << '\n'
         << "compiler-status="
         << static_cast<unsigned>(compilerCapability.status) << '\n'
         << "compiler-reason="
         << stringifyNumericCompilerEmittabilityReason(
                compilerCapability.reason)
         << '\n'
         << "evidence-status="
         << static_cast<unsigned>(evidenceCapability.status) << '\n'
         << "evidence-reason="
         << stringifyNumericEvidenceReason(evidenceCapability.reason) << '\n'
         << "semantics="
         << (semantics ? semantics->getDigest() : llvm::StringRef("none"))
         << '\n';
  if (formalKernel)
    stream << "kernel=" << stringifyFormalKernelKind(*formalKernel) << '\n';
  else
    stream << "kernel=none\n";
  if (comparator)
    stream << "comparator=" << static_cast<unsigned>(*comparator) << '\n';
  else
    stream << "comparator=none\n";
  if (formalBackend)
    stream << "backend=" << stringifyFormalNumericBackendKind(*formalBackend)
           << '\n';
  else
    stream << "backend=none\n";
  stream.flush();
  return digestCanonical(canonical);
}

std::string makeResolutionDigest(const ModelProfileRecord &model,
                                 const NumericCommandKey &key,
                                 const NumericCapabilityPattern &pattern) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream << "wafer-resolved-numeric-command-v1\n"
         << "model-policy-digest=" << model.policyDigest << '\n'
         << "key-digest=" << key.getDigest() << '\n'
         << "pattern-digest=" << pattern.getDigest() << '\n';
  stream.flush();
  return digestCanonical(canonical);
}

llvm::Error requireEngineFormat(TargetProfileId targetProfile,
                                TargetFormatEngine engine, LogicalFormat format,
                                llvm::StringRef role) {
  const TargetFormatEncodingRecord *record =
      findTargetFormatEncoding(targetProfile, engine, format);
  if (!record || !record->isSupported())
    return llvm::createStringError(
        llvm::errc::not_supported,
        "%s format '%s' is not compiler-emittable for target engine '%s'",
        role.str().c_str(), stringifyLogicalFormat(format).str().c_str(),
        stringifyTargetFormatEngine(engine).str().c_str());
  return llvm::Error::success();
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

const NumericSemanticsProfile *
findCTConvertSemantics(ModelProfileId modelProfile,
                       TargetProfileId targetProfile, uint16_t opcode,
                       NumericRoundingMode effectiveRounding) {
  const NumericSemanticsProfile *match = nullptr;
  for (const NumericSemanticsProfile &profile :
       getRegisteredNumericSemanticsProfiles()) {
    const NumericCTConvertSemanticsIdentity *identity =
        profile.getCTConvertIdentity();
    if (identity && profile.getModelProfile() == modelProfile &&
        identity->getTargetProfile() == targetProfile &&
        identity->getCTConvertOpcode() == opcode &&
        identity->getEffectiveRoundingMode() == effectiveRounding) {
      if (match)
        llvm::report_fatal_error(
            "duplicate reusable numeric semantics identity");
      match = &profile;
    }
  }
  return match;
}

const NumericSemanticsProfile *findCTElementwiseSemantics(
    ModelProfileId modelProfile, TargetProfileId targetProfile,
    NumericElementwiseOperation operation, LogicalFormat inputFormat) {
  const NumericSemanticsProfile *match = nullptr;
  for (const NumericSemanticsProfile &profile :
       getRegisteredNumericCTElementwiseSemanticsProfiles()) {
    const NumericCTElementwiseSemanticsIdentity *identity =
        profile.getCTElementwiseIdentity();
    if (identity && profile.getModelProfile() == modelProfile &&
        identity->getTargetProfile() == targetProfile &&
        identity->getOperation() == operation &&
        identity->getInputFormat() == inputFormat) {
      if (match)
        llvm::report_fatal_error(
            "duplicate reusable elementwise semantics identity");
      match = &profile;
    }
  }
  return match;
}

const NumericSemanticsProfile *
findNEGemmSemantics(ModelProfileId modelProfile, TargetProfileId targetProfile,
                    LogicalFormat format) {
  const NumericSemanticsProfile *match = nullptr;
  for (const NumericSemanticsProfile &profile :
       getRegisteredNumericNEGemmSemanticsProfiles()) {
    const NumericNEGemmSemanticsIdentity *identity =
        profile.getNEGemmIdentity();
    if (identity && profile.getModelProfile() == modelProfile &&
        identity->getTargetProfile() == targetProfile &&
        identity->getFormat() == format) {
      if (match)
        llvm::report_fatal_error("duplicate reusable GEMM semantics identity");
      match = &profile;
    }
  }
  return match;
}

std::vector<size_t> getReducedDimensions(NativeCTReduceDimension dimension,
                                         size_t rank) {
  auto trailing = [&](size_t index) -> std::optional<size_t> {
    if (index >= rank)
      return std::nullopt;
    return rank - 1 - index;
  };
  std::vector<size_t> dimensions;
  switch (dimension) {
  case NativeCTReduceDimension::Trailing0:
    if (auto value = trailing(0))
      dimensions.push_back(*value);
    break;
  case NativeCTReduceDimension::Trailing1:
    if (auto value = trailing(1))
      dimensions.push_back(*value);
    break;
  case NativeCTReduceDimension::Trailing2:
    if (auto value = trailing(2))
      dimensions.push_back(*value);
    break;
  case NativeCTReduceDimension::Trailing3:
    if (auto value = trailing(3))
      dimensions.push_back(*value);
    break;
  case NativeCTReduceDimension::Trailing2And1: {
    std::optional<size_t> first = trailing(2);
    std::optional<size_t> second = trailing(1);
    if (first && second)
      dimensions = {*first, *second};
    break;
  }
  case NativeCTReduceDimension::Trailing2And1And0: {
    std::optional<size_t> first = trailing(2);
    std::optional<size_t> second = trailing(1);
    std::optional<size_t> third = trailing(0);
    if (first && second && third)
      dimensions = {*first, *second, *third};
    break;
  }
  }
  return dimensions;
}

bool selectorsOverlap(const NumericCapabilityPattern &lhs,
                      const NumericCapabilityPattern &rhs) {
  if (lhs.getModelProfile() != rhs.getModelProfile() ||
      lhs.getTargetProfile() != rhs.getTargetProfile() ||
      lhs.getFamily() != rhs.getFamily())
    return false;
  if (const auto *lhsConvert = lhs.getCTConvertSelector()) {
    const auto *rhsConvert = rhs.getCTConvertSelector();
    if (!rhsConvert || lhsConvert->opcode != rhsConvert->opcode)
      return false;
    if (lhsConvert->parameter.getKind() != rhsConvert->parameter.getKind())
      return false;
    return lhsConvert->parameter.getKind() !=
               NumericCapabilityParameterPatternKind::ExactRoundingMode ||
           lhsConvert->parameter.getExactRoundingMode() ==
               rhsConvert->parameter.getExactRoundingMode();
  }
  return lhs.getSelector() == rhs.getSelector();
}

llvm::Error
validatePatterns(llvm::ArrayRef<NumericCapabilityPattern> patterns) {
  size_t convertSupported = 0;
  size_t convertStochastic = 0;
  size_t convertZeroPoint = 0;
  size_t elementwise = 0;
  size_t elementwiseSupported = 0;
  size_t elementwiseInteger = 0;
  size_t elementwiseExpLp = 0;
  size_t elementwiseSatRelu = 0;
  size_t elementwiseLeakyRelu = 0;
  size_t gemm = 0;
  size_t gemmSupported = 0;
  size_t gemmInteger = 0;
  size_t reduce = 0;
  std::set<std::string> digests;
  std::set<std::string> elementwiseSelectors;
  std::set<std::string> gemmSelectors;
  std::set<std::string> reduceSelectors;
  std::set<const NumericSemanticsProfile *> referencedSemantics;

  auto requireSupportedExecution =
      [&](const NumericCapabilityPattern &pattern,
          NumericCommandFamily expectedFamily, FormalKernelKind expectedKernel,
          FormalNumericBackendKind expectedBackend) -> llvm::Error {
    const NumericSemanticsProfile *semantics = pattern.getSemantics();
    if (pattern.getModelCapability().status !=
            NumericModelImplementationStatus::Implemented ||
        pattern.getModelCapability().reason !=
            NumericModelImplementationReason::None ||
        !semantics || semantics->getModelProfile() != kFormalDeterministicV1 ||
        semantics->getFamily() != expectedFamily ||
        pattern.getFormalKernelKind() != expectedKernel ||
        pattern.getComparatorKind() != NumericComparatorKind::RawExact ||
        pattern.getFormalBackendKind() != expectedBackend)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "supported numeric pattern has incomplete or cross-family execution "
          "identity");
    if (!referencedSemantics.insert(semantics).second)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "supported numeric patterns reuse one semantics identity");
    return llvm::Error::success();
  };

  auto requireUnsupported =
      [&](const NumericCapabilityPattern &pattern,
          NumericModelImplementationReason expectedReason) -> llvm::Error {
    if (pattern.getModelCapability().status !=
            NumericModelImplementationStatus::Absent ||
        pattern.getModelCapability().reason != expectedReason ||
        pattern.getSemantics() || pattern.getFormalKernelKind() ||
        pattern.getComparatorKind() || pattern.getFormalBackendKind())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "unsupported numeric pattern has the wrong stable reason or carries "
          "execution identity");
    return llvm::Error::success();
  };

  for (size_t index = 0; index < patterns.size(); ++index) {
    const NumericCapabilityPattern &pattern = patterns[index];
    if (pattern.getModelProfile() != kFormalDeterministicV1 ||
        pattern.getTargetProfile() != kTargetProfile)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "numeric capability pattern has an unexpected model or target");
    if (!isValidDigest(pattern.getDigest()) ||
        !digests.insert(pattern.getDigest().str()).second)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "numeric capability pattern digest is invalid or duplicated");
    if (pattern.getCompilerCapability().status !=
            NumericCompilerEmittabilityStatus::Emittable ||
        pattern.getCompilerCapability().reason !=
            NumericCompilerEmittabilityReason::None ||
        pattern.getEvidenceCapability().status !=
            NumericEvidenceStatus::ModelOnlyUncorrelated ||
        pattern.getEvidenceCapability().reason !=
            NumericEvidenceReason::HardwareCorrelationNotRun)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "numeric capability pattern lost a compiler or evidence axis");

    for (size_t other = index + 1; other < patterns.size(); ++other)
      if (selectorsOverlap(pattern, patterns[other]))
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "numeric capability registry contains overlapping selectors");

    if (const auto *selector = pattern.getCTConvertSelector()) {
      if (pattern.getFamily() != NumericCommandFamily::CTConvert)
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "convert selector has wrong family");
      if (pattern.isSupported()) {
        ++convertSupported;
        if (llvm::Error error = requireSupportedExecution(
                pattern, NumericCommandFamily::CTConvert,
                FormalKernelKind::Convert,
                FormalNumericBackendKind::LLVMAPFloatAPInt))
          return error;
        const NumericCTConvertSemanticsIdentity *identity =
            pattern.getSemantics()->getCTConvertIdentity();
        std::optional<NumericRoundingMode> expectedRounding =
            selector->parameter.getExactRoundingMode();
        if (selector->parameter.getKind() ==
            NumericCapabilityParameterPatternKind::NoParameter)
          expectedRounding = NumericRoundingMode::NearestEven;
        if (!identity || identity->getTargetProfile() != kTargetProfile ||
            identity->getCTConvertOpcode() != selector->opcode ||
            !expectedRounding ||
            identity->getEffectiveRoundingMode() != *expectedRounding ||
            pattern.getSemantics()->getRoundingModePolicy() !=
                expectedRounding ||
            pattern.getSemantics()->getRoundingPointPolicy() !=
                NumericRoundingPointPolicy::ConversionResult)
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "supported convert selector does not match its typed semantics");
        const TargetConvertRoute &route = identity->getCTConvertRoute();
        const bool floatToInteger =
            isFloating(route.source) && isInteger(route.destination);
        const bool floatToFloat =
            isFloating(route.source) && isFloating(route.destination);
        const bool hasFloatingEndpoint =
            isFloating(route.source) || isFloating(route.destination);
        if (pattern.getSemantics()->getFloatToIntegerPolicy() !=
                (floatToInteger ? FloatToIntegerPolicy::
                                      FiniteInRangeRejectNaNInfOverflowNoWrite
                                : FloatToIntegerPolicy::NotApplicable) ||
            pattern.getSemantics()->getFloatingNaNPolicy() !=
                (floatToFloat ? FloatingNaNPolicy::CanonicalPositiveQuietNaN
                              : FloatingNaNPolicy::NotApplicable) ||
            pattern.getSemantics()->getFloatingSignedZeroPolicy() !=
                (floatToFloat ? FloatingSignedZeroPolicy::Preserve
                              : FloatingSignedZeroPolicy::NotApplicable) ||
            pattern.getSemantics()->getFloatingSubnormalPolicy() !=
                (hasFloatingEndpoint
                     ? FloatingSubnormalPolicy::Gradual
                     : FloatingSubnormalPolicy::NotApplicable) ||
            pattern.getSemantics()->getFloatingTininessPolicy() !=
                (isFloating(route.destination)
                     ? FloatingTininessPolicy::AfterRounding
                     : FloatingTininessPolicy::NotApplicable) ||
            pattern.getSemantics()->getExceptionFlagPolicy() !=
                (hasFloatingEndpoint
                     ? NumericExceptionFlagPolicy::ModelOnly
                     : NumericExceptionFlagPolicy::NotApplicable) ||
            pattern.getSemantics()->getFloatingNaNSignalingPolicy() !=
                (isFloating(route.source)
                     ? FloatingNaNSignalingPolicy::
                           SignalingRaisesInvalidQuietDoesNot
                     : FloatingNaNSignalingPolicy::NotApplicable) ||
            pattern.getSemantics()->getFloatingDenormalModePolicy() !=
                (hasFloatingEndpoint
                     ? FloatingDenormalModePolicy::GradualNoDAZNoFTZ
                     : FloatingDenormalModePolicy::NotApplicable) ||
            pattern.getSemantics()->getFloatingOverflowPolicy() !=
                (isFloating(route.destination)
                     ? FloatingOverflowPolicy::IEEE754AccordingToRoundingMode
                     : FloatingOverflowPolicy::NotApplicable) ||
            pattern.getSemantics()->getSaturationPolicy() !=
                NumericSaturationPolicy::Disabled ||
            pattern.getSemantics()->getTranscendentalEvaluationPolicy() !=
                NumericTranscendentalEvaluationPolicy::NotApplicable ||
            pattern.getSemantics()->getGemmAccumulatorPolicy() !=
                NumericGemmAccumulatorPolicy::NotApplicable ||
            pattern.getSemantics()->getGemmAccumulatorInitializationPolicy() !=
                NumericGemmAccumulatorInitializationPolicy::NotApplicable ||
            pattern.getSemantics()->getGemmReductionOrderPolicy() !=
                NumericGemmReductionOrderPolicy::NotApplicable)
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "supported convert semantics has an incomplete numeric policy");
      } else {
        if (selector->parameter.getKind() ==
            NumericCapabilityParameterPatternKind::AnyZeroPoint) {
          ++convertZeroPoint;
          if (llvm::Error error = requireUnsupported(
                  pattern,
                  NumericModelImplementationReason::ZeroPointFormulaUnproven))
            return error;
        } else {
          ++convertStochastic;
          if (selector->parameter.getExactRoundingMode() !=
              NumericRoundingMode::Stochastic)
            return llvm::createStringError(
                llvm::errc::invalid_argument,
                "stochastic pattern has the wrong selector");
          if (llvm::Error error = requireUnsupported(
                  pattern,
                  NumericModelImplementationReason::StochasticStateUnproven))
            return error;
        }
      }
      continue;
    }

    if (const auto *selector = pattern.getCTElementwiseSelector()) {
      ++elementwise;
      if (pattern.getFamily() != NumericCommandFamily::CTElementwise)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "elementwise selector has the wrong family");
      elementwiseSelectors.insert(
          (llvm::Twine(
               stringifyNumericElementwiseOperation(selector->operation)) +
           ":" + stringifyLogicalFormat(selector->inputFormat))
              .str());
      std::optional<NumericModelImplementationReason> unsupportedReason =
          getElementwiseUnsupportedReason(selector->operation,
                                          selector->inputFormat);
      if (unsupportedReason) {
        if (llvm::Error error = requireUnsupported(pattern, *unsupportedReason))
          return error;
        switch (*unsupportedReason) {
        case NumericModelImplementationReason::IntegerElementwisePolicyUnproven:
          ++elementwiseInteger;
          break;
        case NumericModelImplementationReason::ExpLpParameterPolicyUnproven:
          ++elementwiseExpLp;
          break;
        case NumericModelImplementationReason::SatReluParameterPolicyUnproven:
          ++elementwiseSatRelu;
          break;
        case NumericModelImplementationReason::LeakyReluParameterPolicyUnproven:
          ++elementwiseLeakyRelu;
          break;
        default:
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "elementwise selector has a non-elementwise unsupported reason");
        }
        continue;
      }

      ++elementwiseSupported;
      const FormalNumericBackendKind expectedBackend =
          isMPFRElementwiseOperation(selector->operation)
              ? FormalNumericBackendKind::MPFR
              : FormalNumericBackendKind::LLVMAPFloatAPInt;
      if (llvm::Error error = requireSupportedExecution(
              pattern, NumericCommandFamily::CTElementwise,
              FormalKernelKind::Elementwise, expectedBackend))
        return error;
      const NumericCTElementwiseSemanticsIdentity *identity =
          pattern.getSemantics()->getCTElementwiseIdentity();
      const LogicalFormat expectedDestination = getElementwiseDestinationFormat(
          selector->operation, selector->inputFormat);
      if (!identity || identity->getTargetProfile() != kTargetProfile ||
          identity->getOperation() != selector->operation ||
          identity->getInputFormat() != selector->inputFormat ||
          identity->getDestinationFormat() != expectedDestination)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "supported elementwise selector does not match its typed "
            "semantics");
      const bool logic = isNumericElementwiseLogic(selector->operation);
      const bool relation = isNumericElementwiseRelation(selector->operation);
      FloatingSignedZeroPolicy expectedSignedZero =
          logic      ? FloatingSignedZeroPolicy::NotApplicable
          : relation ? FloatingSignedZeroPolicy::PredicateOnly
                     : FloatingSignedZeroPolicy::IEEE754OperationDefined;
      if (selector->operation == NumericElementwiseOperation::Max)
        expectedSignedZero =
            FloatingSignedZeroPolicy::MaximumPositiveUnlessBothNegative;
      else if (selector->operation == NumericElementwiseOperation::Min)
        expectedSignedZero =
            FloatingSignedZeroPolicy::MinimumNegativeUnlessBothPositive;
      const NumericTranscendentalEvaluationPolicy expectedTranscendental =
          isMPFRElementwiseOperation(selector->operation)
              ? NumericTranscendentalEvaluationPolicy::
                    CorrectlyRoundedMathematicalResultAdaptiveMPFRFinalRNE
              : NumericTranscendentalEvaluationPolicy::NotApplicable;
      if (pattern.getSemantics()->getRoundingModePolicy() !=
              (logic || relation ? std::nullopt
                                 : std::optional<NumericRoundingMode>(
                                       NumericRoundingMode::NearestEven)) ||
          pattern.getSemantics()->getRoundingPointPolicy() !=
              (logic || relation
                   ? NumericRoundingPointPolicy::NotApplicable
                   : NumericRoundingPointPolicy::ElementwiseResult) ||
          pattern.getSemantics()->getFloatToIntegerPolicy() !=
              FloatToIntegerPolicy::NotApplicable ||
          pattern.getSemantics()->getFloatingNaNPolicy() !=
              (logic ? FloatingNaNPolicy::NotApplicable
               : relation
                   ? FloatingNaNPolicy::OrderedRelationFalseExceptNotEqualTrue
                   : FloatingNaNPolicy::CanonicalPositiveQuietNaN) ||
          pattern.getSemantics()->getFloatingSignedZeroPolicy() !=
              expectedSignedZero ||
          pattern.getSemantics()->getFloatingSubnormalPolicy() !=
              (logic ? FloatingSubnormalPolicy::NotApplicable
                     : FloatingSubnormalPolicy::Gradual) ||
          pattern.getSemantics()->getFloatingTininessPolicy() !=
              (logic || relation ? FloatingTininessPolicy::NotApplicable
                                 : FloatingTininessPolicy::AfterRounding) ||
          pattern.getSemantics()->getExceptionFlagPolicy() !=
              (logic ? NumericExceptionFlagPolicy::NotApplicable
                     : NumericExceptionFlagPolicy::ModelOnly) ||
          pattern.getSemantics()->getFloatingNaNSignalingPolicy() !=
              (logic ? FloatingNaNSignalingPolicy::NotApplicable
                     : FloatingNaNSignalingPolicy::
                           SignalingRaisesInvalidQuietDoesNot) ||
          pattern.getSemantics()->getFloatingDenormalModePolicy() !=
              (logic ? FloatingDenormalModePolicy::NotApplicable
                     : FloatingDenormalModePolicy::GradualNoDAZNoFTZ) ||
          pattern.getSemantics()->getFloatingOverflowPolicy() !=
              (logic || relation
                   ? FloatingOverflowPolicy::NotApplicable
                   : FloatingOverflowPolicy::IEEE754AccordingToRoundingMode) ||
          pattern.getSemantics()->getSaturationPolicy() !=
              (logic || relation ? NumericSaturationPolicy::NotApplicable
                                 : NumericSaturationPolicy::Disabled) ||
          pattern.getSemantics()->getTranscendentalEvaluationPolicy() !=
              expectedTranscendental ||
          pattern.getSemantics()->getGemmAccumulatorPolicy() !=
              NumericGemmAccumulatorPolicy::NotApplicable ||
          pattern.getSemantics()->getGemmAccumulatorInitializationPolicy() !=
              NumericGemmAccumulatorInitializationPolicy::NotApplicable ||
          pattern.getSemantics()->getGemmReductionOrderPolicy() !=
              NumericGemmReductionOrderPolicy::NotApplicable)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "supported elementwise semantics has an incomplete numeric "
            "policy");
      continue;
    }
    if (const auto *selector = pattern.getNEGemmSelector()) {
      ++gemm;
      if (pattern.getFamily() != NumericCommandFamily::NEGemm)
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "GEMM selector has the wrong family");
      gemmSelectors.insert(stringifyLogicalFormat(selector->format).str());
      if (selector->format == LogicalFormat::I8) {
        ++gemmInteger;
        if (llvm::Error error = requireUnsupported(
                pattern, NumericModelImplementationReason::
                             IntegerGemmAccumulatorPolicyUnproven))
          return error;
        continue;
      }
      ++gemmSupported;
      if (llvm::Error error = requireSupportedExecution(
              pattern, NumericCommandFamily::NEGemm, FormalKernelKind::Gemm,
              FormalNumericBackendKind::LLVMAPFloatAPInt))
        return error;
      const NumericNEGemmSemanticsIdentity *identity =
          pattern.getSemantics()->getNEGemmIdentity();
      if (!identity || identity->getTargetProfile() != kTargetProfile ||
          identity->getFormat() != selector->format ||
          pattern.getSemantics()->getRoundingModePolicy() !=
              NumericRoundingMode::NearestEven ||
          pattern.getSemantics()->getRoundingPointPolicy() !=
              NumericRoundingPointPolicy::GemmFusedMultiplyAddAndDestination ||
          pattern.getSemantics()->getGemmAccumulatorPolicy() !=
              NumericGemmAccumulatorPolicy::F32FusedMultiplyAdd ||
          pattern.getSemantics()->getGemmAccumulatorInitializationPolicy() !=
              NumericGemmAccumulatorInitializationPolicy::PositiveZero ||
          pattern.getSemantics()->getGemmReductionOrderPolicy() !=
              NumericGemmReductionOrderPolicy::IncreasingK ||
          pattern.getSemantics()->getFloatToIntegerPolicy() !=
              FloatToIntegerPolicy::NotApplicable ||
          pattern.getSemantics()->getFloatingNaNPolicy() !=
              FloatingNaNPolicy::CanonicalPositiveQuietNaN ||
          pattern.getSemantics()->getFloatingSignedZeroPolicy() !=
              FloatingSignedZeroPolicy::
                  GemmPositiveZeroAccumulatorThenIEEE754 ||
          pattern.getSemantics()->getFloatingSubnormalPolicy() !=
              FloatingSubnormalPolicy::Gradual ||
          pattern.getSemantics()->getFloatingTininessPolicy() !=
              FloatingTininessPolicy::AfterRounding ||
          pattern.getSemantics()->getExceptionFlagPolicy() !=
              NumericExceptionFlagPolicy::ModelOnly ||
          pattern.getSemantics()->getFloatingNaNSignalingPolicy() !=
              FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot ||
          pattern.getSemantics()->getFloatingDenormalModePolicy() !=
              FloatingDenormalModePolicy::GradualNoDAZNoFTZ ||
          pattern.getSemantics()->getFloatingOverflowPolicy() !=
              FloatingOverflowPolicy::IEEE754AccordingToRoundingMode ||
          pattern.getSemantics()->getSaturationPolicy() !=
              NumericSaturationPolicy::Disabled ||
          pattern.getSemantics()->getTranscendentalEvaluationPolicy() !=
              NumericTranscendentalEvaluationPolicy::NotApplicable)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "supported GEMM selector does not match its typed semantics");
      continue;
    }
    if (const auto *selector = pattern.getNativeCTReduceSelector()) {
      ++reduce;
      if (pattern.getFamily() != NumericCommandFamily::NativeCTReduce)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "native reduction selector has the wrong family");
      if (llvm::Error error = requireUnsupported(
              pattern,
              NumericModelImplementationReason::NativeReductionPolicyUnproven))
        return error;
      reduceSelectors.insert(
          (llvm::Twine(stringifyNumericReduceOperation(selector->operation)) +
           ":" + stringifyLogicalFormat(selector->format))
              .str());
      continue;
    }
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "numeric pattern has no typed selector");
  }

  std::set<std::string> expectedElementwise;
  for (NumericElementwiseOperation operation :
       getNumericElementwiseOperations()) {
    if (isNumericElementwiseLogic(operation)) {
      expectedElementwise.insert(
          (llvm::Twine(stringifyNumericElementwiseOperation(operation)) +
           ":bool")
              .str());
      continue;
    }
    for (LogicalFormat format :
         getCompilerNumericFormats(TargetFormatEngine::CT))
      expectedElementwise.insert(
          (llvm::Twine(stringifyNumericElementwiseOperation(operation)) + ":" +
           stringifyLogicalFormat(format))
              .str());
  }
  std::set<std::string> expectedGemm;
  for (LogicalFormat format : getCompilerNumericFormats(TargetFormatEngine::NE))
    expectedGemm.insert(stringifyLogicalFormat(format).str());
  std::set<std::string> expectedReduce;
  for (NumericReduceOperation operation : getNumericReduceOperations())
    for (LogicalFormat format :
         getCompilerNumericFormats(TargetFormatEngine::CT))
      expectedReduce.insert(
          (llvm::Twine(stringifyNumericReduceOperation(operation)) + ":" +
           stringifyLogicalFormat(format))
              .str());

  if (patterns.size() != 276 || convertSupported != 101 ||
      convertStochastic != 23 || convertZeroPoint != 4 || elementwise != 128 ||
      elementwiseSupported != 88 || elementwiseInteger != 31 ||
      elementwiseExpLp != 3 || elementwiseSatRelu != 3 ||
      elementwiseLeakyRelu != 3 || gemm != 4 || gemmSupported != 3 ||
      gemmInteger != 1 || reduce != 16 ||
      elementwiseSelectors != expectedElementwise ||
      gemmSelectors != expectedGemm || reduceSelectors != expectedReduce)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "numeric capability registry is missing or "
                                   "has an extra finite selector");

  auto requireCompleteSemanticsRegistry =
      [&](llvm::ArrayRef<NumericSemanticsProfile> semanticsProfiles,
          NumericCommandFamily expectedFamily) -> llvm::Error {
    for (const NumericSemanticsProfile &semantics : semanticsProfiles)
      if (semantics.getFamily() != expectedFamily ||
          !isValidDigest(semantics.getDigest()) ||
          !referencedSemantics.count(&semantics))
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "reusable family semantics does not have exactly one matching "
            "pattern");
    return llvm::Error::success();
  };
  if (getRegisteredNumericSemanticsProfiles().size() != 101 ||
      getRegisteredNumericCTElementwiseSemanticsProfiles().size() != 88 ||
      getRegisteredNumericNEGemmSemanticsProfiles().size() != 3 ||
      referencedSemantics.size() != 192)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "numeric semantics registry is missing or has an extra supported row");
  if (llvm::Error error = requireCompleteSemanticsRegistry(
          getRegisteredNumericSemanticsProfiles(),
          NumericCommandFamily::CTConvert))
    return error;
  if (llvm::Error error = requireCompleteSemanticsRegistry(
          getRegisteredNumericCTElementwiseSemanticsProfiles(),
          NumericCommandFamily::CTElementwise))
    return error;
  if (llvm::Error error = requireCompleteSemanticsRegistry(
          getRegisteredNumericNEGemmSemanticsProfiles(),
          NumericCommandFamily::NEGemm))
    return error;
  return llvm::Error::success();
}

} // namespace

llvm::ArrayRef<ModelProfileRecord> getRegisteredModelProfiles() {
  static const std::vector<ModelProfileRecord> profiles = [] {
    ModelProfileRecord record{
        kFormalDeterministicV1,
        "wafer-model-formal-deterministic-v1",
        {LogicalByteOrder::LittleEndian,
         LogicalBitOrder::LeastSignificantBitFirstWithinByte,
         NonCanonicalEncodingPolicy::Reject},
        {LogicalByteOrder::LittleEndian,
         LogicalBitOrder::LeastSignificantBitFirstWithinByte,
         NonCanonicalEncodingPolicy::ClearUnusedBits},
        std::string(),
        true,
    };
    record.policyDigest = makeModelPolicyDigest(record);
    if (!isValidDigest(record.policyDigest))
      llvm::report_fatal_error("invalid model policy digest");
    std::vector<ModelProfileRecord> result;
    result.push_back(std::move(record));
    return result;
  }();
  return profiles;
}

llvm::Expected<ModelProfileId>
parseModelProfileId(llvm::StringRef canonicalSpelling) {
  for (const ModelProfileRecord &record : getRegisteredModelProfiles())
    if (record.canonicalSpelling == canonicalSpelling)
      return record.id;
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown model profile '%s'",
                                 canonicalSpelling.str().c_str());
}

const ModelProfileRecord &getModelProfileRecord(ModelProfileId id) {
  for (const ModelProfileRecord &record : getRegisteredModelProfiles())
    if (record.id == id)
      return record;
  llvm_unreachable("closed ModelProfileId is not registered");
}

llvm::StringRef stringifyModelProfileId(ModelProfileId id) {
  return getModelProfileRecord(id).canonicalSpelling;
}

llvm::StringRef stringifyNumericTensorLayout(NumericTensorLayout layout) {
  switch (layout) {
  case NumericTensorLayout::Tensor:
    return "tensor";
  case NumericTensorLayout::NTensor:
    return "ntensor";
  case NumericTensorLayout::Cx:
    return "cx";
  case NumericTensorLayout::NCx:
    return "ncx";
  }
  llvm_unreachable("numeric tensor layout is not registered");
}

llvm::Expected<NumericTensorKey>
NumericTensorKey::create(LogicalFormat format, NumericTensorLayout layout,
                         std::vector<uint64_t> shape) {
  if (!findLogicalFormatDescriptor(format))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "numeric tensor has an unknown format");
  if (!isKnownLayout(layout))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "numeric tensor has an unknown layout");
  uint64_t elementCount = 1;
  for (uint64_t dimension : shape) {
    if (dimension > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "numeric tensor dimension must fit the static IR dimension domain");
    if (!checkedMultiply(elementCount, dimension, elementCount))
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "numeric tensor static element count overflows uint64_t");
  }
  std::string digest = makeTensorDigest(format, layout, shape, elementCount);
  if (!isValidDigest(digest))
    llvm::report_fatal_error("invalid numeric tensor digest");
  return NumericTensorKey(format, layout, std::move(shape), elementCount,
                          std::move(digest));
}

llvm::ArrayRef<NumericRoundingMode> getNumericRoundingModes() {
  return kRoundingModes;
}

llvm::Expected<NumericRoundingMode> parseNumericRoundingMode(uint8_t value) {
  if (value <= static_cast<uint8_t>(NumericRoundingMode::Stochastic))
    return static_cast<NumericRoundingMode>(value);
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "unknown numeric rounding mode %u",
                                 static_cast<unsigned>(value));
}

llvm::StringRef stringifyNumericRoundingMode(NumericRoundingMode mode) {
  switch (mode) {
  case NumericRoundingMode::NearestEven:
    return "nearest-even";
  case NumericRoundingMode::TowardZero:
    return "toward-zero";
  case NumericRoundingMode::TowardPositive:
    return "toward-positive-infinity";
  case NumericRoundingMode::TowardNegative:
    return "toward-negative-infinity";
  case NumericRoundingMode::Stochastic:
    return "stochastic";
  }
  llvm_unreachable("numeric rounding mode is not registered");
}

std::optional<NumericRoundingMode>
NumericConvertParameter::getRoundingMode() const {
  if (kind != Kind::RoundingMode)
    return std::nullopt;
  return static_cast<NumericRoundingMode>(payload);
}

std::optional<uint32_t> NumericConvertParameter::getZeroPoint() const {
  if (kind != Kind::ZeroPoint)
    return std::nullopt;
  return payload;
}

llvm::ArrayRef<NumericElementwiseOperation> getNumericElementwiseOperations() {
  return kElementwiseOperations;
}

llvm::StringRef
stringifyNumericElementwiseOperation(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::Abs:
    return "abs";
  case NumericElementwiseOperation::Recip:
    return "recip";
  case NumericElementwiseOperation::Square:
    return "square";
  case NumericElementwiseOperation::Sqrt:
    return "sqrt";
  case NumericElementwiseOperation::Rsqrt:
    return "rsqrt";
  case NumericElementwiseOperation::Neg:
    return "neg";
  case NumericElementwiseOperation::Max:
    return "max";
  case NumericElementwiseOperation::Min:
    return "min";
  case NumericElementwiseOperation::Add:
    return "add";
  case NumericElementwiseOperation::Sub:
    return "sub";
  case NumericElementwiseOperation::Mul:
    return "mul";
  case NumericElementwiseOperation::Div:
    return "div";
  case NumericElementwiseOperation::Eq:
    return "eq";
  case NumericElementwiseOperation::Ne:
    return "ne";
  case NumericElementwiseOperation::Ge:
    return "ge";
  case NumericElementwiseOperation::Gt:
    return "gt";
  case NumericElementwiseOperation::Le:
    return "le";
  case NumericElementwiseOperation::Lt:
    return "lt";
  case NumericElementwiseOperation::LogicNot:
    return "logic_not";
  case NumericElementwiseOperation::LogicAnd:
    return "logic_and";
  case NumericElementwiseOperation::LogicOr:
    return "logic_or";
  case NumericElementwiseOperation::LogicXor:
    return "logic_xor";
  case NumericElementwiseOperation::Log2:
    return "log2";
  case NumericElementwiseOperation::Ln:
    return "ln";
  case NumericElementwiseOperation::Pow2:
    return "pow2";
  case NumericElementwiseOperation::Exp:
    return "exp";
  case NumericElementwiseOperation::ExpLp:
    return "exp_lp";
  case NumericElementwiseOperation::Sin:
    return "sin";
  case NumericElementwiseOperation::Cos:
    return "cos";
  case NumericElementwiseOperation::Tanh:
    return "tanh";
  case NumericElementwiseOperation::Sigmoid:
    return "sigmoid";
  case NumericElementwiseOperation::Relu:
    return "relu";
  case NumericElementwiseOperation::SatRelu:
    return "satrelu";
  case NumericElementwiseOperation::LeakyRelu:
    return "leakyrelu";
  case NumericElementwiseOperation::Softplus:
    return "softplus";
  }
  llvm_unreachable("numeric elementwise operation is not registered");
}

unsigned getNumericElementwiseArity(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::Abs:
  case NumericElementwiseOperation::Recip:
  case NumericElementwiseOperation::Square:
  case NumericElementwiseOperation::Sqrt:
  case NumericElementwiseOperation::Rsqrt:
  case NumericElementwiseOperation::Neg:
  case NumericElementwiseOperation::LogicNot:
  case NumericElementwiseOperation::Log2:
  case NumericElementwiseOperation::Ln:
  case NumericElementwiseOperation::Pow2:
  case NumericElementwiseOperation::Exp:
  case NumericElementwiseOperation::ExpLp:
  case NumericElementwiseOperation::Sin:
  case NumericElementwiseOperation::Cos:
  case NumericElementwiseOperation::Tanh:
  case NumericElementwiseOperation::Sigmoid:
  case NumericElementwiseOperation::Relu:
  case NumericElementwiseOperation::SatRelu:
  case NumericElementwiseOperation::LeakyRelu:
  case NumericElementwiseOperation::Softplus:
    return 1;
  case NumericElementwiseOperation::Max:
  case NumericElementwiseOperation::Min:
  case NumericElementwiseOperation::Add:
  case NumericElementwiseOperation::Sub:
  case NumericElementwiseOperation::Mul:
  case NumericElementwiseOperation::Div:
  case NumericElementwiseOperation::Eq:
  case NumericElementwiseOperation::Ne:
  case NumericElementwiseOperation::Ge:
  case NumericElementwiseOperation::Gt:
  case NumericElementwiseOperation::Le:
  case NumericElementwiseOperation::Lt:
  case NumericElementwiseOperation::LogicAnd:
  case NumericElementwiseOperation::LogicOr:
  case NumericElementwiseOperation::LogicXor:
    return 2;
  }
  llvm_unreachable("numeric elementwise operation is not registered");
}

bool isNumericElementwiseRelation(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::Eq:
  case NumericElementwiseOperation::Ne:
  case NumericElementwiseOperation::Ge:
  case NumericElementwiseOperation::Gt:
  case NumericElementwiseOperation::Le:
  case NumericElementwiseOperation::Lt:
    return true;
  default:
    return false;
  }
}

bool isNumericElementwiseLogic(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::LogicNot:
  case NumericElementwiseOperation::LogicAnd:
  case NumericElementwiseOperation::LogicOr:
  case NumericElementwiseOperation::LogicXor:
    return true;
  default:
    return false;
  }
}

llvm::ArrayRef<NumericReduceOperation> getNumericReduceOperations() {
  return kReduceOperations;
}

llvm::StringRef stringifyNumericReduceOperation(NumericReduceOperation kind) {
  switch (kind) {
  case NumericReduceOperation::Sum:
    return "sum";
  case NumericReduceOperation::Max:
    return "max";
  case NumericReduceOperation::Min:
    return "min";
  case NumericReduceOperation::Avg:
    return "avg";
  }
  llvm_unreachable("numeric reduce operation is not registered");
}

llvm::StringRef
stringifyNativeCTReduceDimension(NativeCTReduceDimension dimension) {
  switch (dimension) {
  case NativeCTReduceDimension::Trailing0:
    return "trailing-0";
  case NativeCTReduceDimension::Trailing1:
    return "trailing-1";
  case NativeCTReduceDimension::Trailing2:
    return "trailing-2";
  case NativeCTReduceDimension::Trailing3:
    return "trailing-3";
  case NativeCTReduceDimension::Trailing2And1:
    return "trailing-2-and-1";
  case NativeCTReduceDimension::Trailing2And1And0:
    return "trailing-2-and-1-and-0";
  }
  llvm_unreachable("native CT reduce dimension is not registered");
}

llvm::StringRef stringifyNumericCommandFamily(NumericCommandFamily family) {
  switch (family) {
  case NumericCommandFamily::CTConvert:
    return "ct-convert";
  case NumericCommandFamily::CTElementwise:
    return "ct-elementwise";
  case NumericCommandFamily::NEGemm:
    return "ne-gemm";
  case NumericCommandFamily::NativeCTReduce:
    return "native-ct-reduce";
  }
  llvm_unreachable("numeric command family is not registered");
}

llvm::Expected<NumericGemmAxes> getCanonicalNumericGemmAxes(uint64_t rank) {
  if (rank < 2)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "canonical numeric GEMM axes require rank at least 2");
  const uint64_t matrixBase = rank - 2;
  std::vector<uint64_t> batchDimensions;
  if (matrixBase > batchDimensions.max_size())
    return llvm::createStringError(
        llvm::errc::result_out_of_range,
        "canonical numeric GEMM batch-axis count exceeds the host size domain");
  batchDimensions.reserve(static_cast<size_t>(matrixBase));
  for (uint64_t dimension = 0; dimension < matrixBase; ++dimension)
    batchDimensions.push_back(dimension);
  return NumericGemmAxes{batchDimensions, matrixBase, matrixBase + 1,
                         batchDimensions, matrixBase, matrixBase + 1,
                         batchDimensions, matrixBase, matrixBase + 1};
}

NumericCommandFamily NumericCommandKey::getFamily() const {
  if (std::holds_alternative<NumericCTConvertCommand>(payload))
    return NumericCommandFamily::CTConvert;
  if (std::holds_alternative<NumericCTElementwiseCommand>(payload))
    return NumericCommandFamily::CTElementwise;
  if (std::holds_alternative<NumericNEGemmCommand>(payload))
    return NumericCommandFamily::NEGemm;
  return NumericCommandFamily::NativeCTReduce;
}

const NumericCTConvertCommand *NumericCommandKey::getCTConvert() const {
  return std::get_if<NumericCTConvertCommand>(&payload);
}
const NumericCTElementwiseCommand *NumericCommandKey::getCTElementwise() const {
  return std::get_if<NumericCTElementwiseCommand>(&payload);
}
const NumericNEGemmCommand *NumericCommandKey::getNEGemm() const {
  return std::get_if<NumericNEGemmCommand>(&payload);
}
const NumericNativeCTReduceCommand *
NumericCommandKey::getNativeCTReduce() const {
  return std::get_if<NumericNativeCTReduceCommand>(&payload);
}

llvm::Expected<NumericCommandKey> NumericCommandKey::createCTConvert(
    TargetProfileId targetProfile, uint16_t opcode, NumericTensorKey source,
    NumericTensorKey destination,
    std::optional<NumericConvertParameter> parameter) {
  const TargetConvertRoute *route =
      findTargetConvertRoute(targetProfile, opcode);
  if (!route)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "unknown CT convert route for target profile '%s' and opcode %u",
        stringifyTargetProfileId(targetProfile).str().c_str(),
        static_cast<unsigned>(opcode));
  if (source.getFormat() != route->source ||
      destination.getFormat() != route->destination)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "CT convert route endpoints do not match source/destination tensors");
  if (source.getElementCount() != destination.getElementCount())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "CT convert source and destination must have the same static element "
        "count");
  if (destination.getElementCount() > std::numeric_limits<uint32_t>::max())
    return llvm::createStringError(
        llvm::errc::result_out_of_range,
        "CT convert destination element count must fit uint32_t");

  switch (route->parameterKind) {
  case TargetConvertParameterKind::None:
    if (parameter)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "parameterless CT convert route '%s' "
                                     "rejects every optional parameter",
                                     route->canonicalSpelling.str().c_str());
    break;
  case TargetConvertParameterKind::RoundingMode: {
    if (!parameter)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "CT convert route '%s' requires exactly one rounding-mode parameter",
          route->canonicalSpelling.str().c_str());
    std::optional<NumericRoundingMode> mode = parameter->getRoundingMode();
    if (!mode)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "CT convert route '%s' requires a rounding-mode parameter, not a "
          "zero-point parameter",
          route->canonicalSpelling.str().c_str());
    if (!isKnownRoundingMode(*mode))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "CT convert route '%s' carries an unknown rounding mode %u",
          route->canonicalSpelling.str().c_str(), static_cast<unsigned>(*mode));
    break;
  }
  case TargetConvertParameterKind::ZeroPoint:
    if (!parameter)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "CT convert route '%s' requires exactly "
                                     "one uint32 zero-point parameter",
                                     route->canonicalSpelling.str().c_str());
    if (!parameter->getZeroPoint())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "CT convert route '%s' requires a zero-point parameter, not a "
          "rounding-mode parameter",
          route->canonicalSpelling.str().c_str());
    break;
  }

  NumericCommandPayload payload = NumericCTConvertCommand{
      opcode, std::move(source), std::move(destination), parameter};
  std::string digest = makeCommandKeyDigest(targetProfile, payload);
  return NumericCommandKey(targetProfile, std::move(payload),
                           std::move(digest));
}

llvm::Expected<NumericCommandKey> NumericCommandKey::createCTElementwise(
    TargetProfileId targetProfile, NumericElementwiseOperation operation,
    std::vector<NumericTensorKey> inputs, NumericTensorKey destination) {
  if (!isKnownElementwiseOperation(operation))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown numeric elementwise operation");
  const unsigned arity = getNumericElementwiseArity(operation);
  if (inputs.size() != arity)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "numeric elementwise operation expects %u input(s), got %zu", arity,
        inputs.size());
  if (destination.getElementCount() > std::numeric_limits<uint32_t>::max())
    return llvm::createStringError(
        llvm::errc::result_out_of_range,
        "CT elementwise destination element count must fit uint32_t");
  if (llvm::Error error = requireEngineFormat(
          targetProfile, TargetFormatEngine::CT, destination.getFormat(),
          "CT elementwise destination"))
    return std::move(error);

  for (const NumericTensorKey &input : inputs) {
    if (input.getShape() != destination.getShape() ||
        input.getElementCount() != destination.getElementCount())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "CT elementwise input shapes must match destination shape");
    if (llvm::Error error =
            requireEngineFormat(targetProfile, TargetFormatEngine::CT,
                                input.getFormat(), "CT elementwise input"))
      return std::move(error);
  }

  if (isNumericElementwiseLogic(operation)) {
    if (destination.getFormat() != LogicalFormat::Bool ||
        std::any_of(inputs.begin(), inputs.end(),
                    [](const NumericTensorKey &key) {
                      return key.getFormat() != LogicalFormat::Bool;
                    }))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "logical CT elementwise operations "
                                     "require BOOL inputs and destination");
  } else if (isNumericElementwiseRelation(operation)) {
    if (destination.getFormat() != LogicalFormat::Bool ||
        inputs.front().getFormat() == LogicalFormat::Bool ||
        std::any_of(inputs.begin(), inputs.end(),
                    [&](const NumericTensorKey &key) {
                      return key.getFormat() != inputs.front().getFormat();
                    }))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "relation CT elementwise operations require matching numeric inputs "
          "and BOOL destination");
  } else if (destination.getFormat() == LogicalFormat::Bool ||
             std::any_of(inputs.begin(), inputs.end(),
                         [&](const NumericTensorKey &key) {
                           return key.getFormat() != destination.getFormat();
                         })) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "numeric CT elementwise input and destination formats must match");
  }

  NumericCommandPayload payload = NumericCTElementwiseCommand{
      operation, std::move(inputs), std::move(destination)};
  std::string digest = makeCommandKeyDigest(targetProfile, payload);
  return NumericCommandKey(targetProfile, std::move(payload),
                           std::move(digest));
}

llvm::Expected<NumericCommandKey> NumericCommandKey::createNEGemm(
    TargetProfileId targetProfile, NumericTensorKey lhs, NumericTensorKey rhs,
    NumericTensorKey destination, uint32_t m, uint32_t k, uint32_t n,
    uint32_t batchCount, NumericGemmAxes axes) {
  if (lhs.getFormat() != rhs.getFormat() ||
      lhs.getFormat() != destination.getFormat())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM lhs, rhs and destination formats must match");
  if (llvm::Error error = requireEngineFormat(
          targetProfile, TargetFormatEngine::NE, lhs.getFormat(), "NE GEMM"))
    return std::move(error);
  if (!isAlignedLayout(lhs.getLayout()) || !isAlignedLayout(rhs.getLayout()) ||
      !isAlignedLayout(destination.getLayout()))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM tensors must use cx or ncx layout markers");

  const size_t rank = lhs.getShape().size();
  if (rank < 2 || rhs.getShape().size() != rank ||
      destination.getShape().size() != rank)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM tensors must have the same rank of at least 2");
  static_assert(std::numeric_limits<size_t>::max() <=
                    std::numeric_limits<uint64_t>::max(),
                "numeric GEMM rank must convert losslessly to its axis domain");
  llvm::Expected<NumericGemmAxes> canonicalAxes =
      getCanonicalNumericGemmAxes(static_cast<uint64_t>(rank));
  if (!canonicalAxes)
    return canonicalAxes.takeError();
  if (!(axes == *canonicalAxes))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM requires canonical leading batch and trailing M/K/N axes");
  if (m == 0 || k == 0 || n == 0 || batchCount == 0 ||
      m > std::numeric_limits<uint16_t>::max() ||
      k > std::numeric_limits<uint16_t>::max() ||
      n > std::numeric_limits<uint16_t>::max() ||
      batchCount > std::numeric_limits<uint16_t>::max())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM m, k, n and batch count must be positive uint16 values");
  for (const NumericTensorKey *tensor : {&lhs, &rhs, &destination})
    if (std::any_of(tensor->getShape().begin(), tensor->getShape().end(),
                    [](uint64_t dimension) { return dimension == 0; }))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "NE GEMM tensor dimensions must be positive");

  const size_t matrixBase = rank - 2;
  uint64_t inferredBatch = 1;
  for (size_t index = 0; index < matrixBase; ++index) {
    if (lhs.getShape()[index] != rhs.getShape()[index] ||
        lhs.getShape()[index] != destination.getShape()[index])
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "NE GEMM canonical leading batch dimensions must match");
    if (!checkedMultiply(inferredBatch, lhs.getShape()[index], inferredBatch) ||
        inferredBatch > std::numeric_limits<uint16_t>::max())
      return llvm::createStringError(
          llvm::errc::result_out_of_range,
          "NE GEMM canonical batch product must fit uint16_t");
  }
  if (lhs.getShape()[matrixBase] != m || lhs.getShape()[matrixBase + 1] != k ||
      rhs.getShape()[matrixBase] != k || rhs.getShape()[matrixBase + 1] != n ||
      destination.getShape()[matrixBase] != m ||
      destination.getShape()[matrixBase + 1] != n)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM requires canonical trailing lhs[M,K], rhs[K,N], dest[M,N] "
        "axes matching m/k/n");
  if (inferredBatch != batchCount)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "NE GEMM batch count must equal the leading batch shape product");

  NumericCommandPayload payload =
      NumericNEGemmCommand{std::move(lhs),
                           std::move(rhs),
                           std::move(destination),
                           static_cast<uint16_t>(m),
                           static_cast<uint16_t>(k),
                           static_cast<uint16_t>(n),
                           static_cast<uint16_t>(batchCount),
                           std::move(axes)};
  std::string digest = makeCommandKeyDigest(targetProfile, payload);
  return NumericCommandKey(targetProfile, std::move(payload),
                           std::move(digest));
}

llvm::Expected<NumericCommandKey> NumericCommandKey::createNativeCTReduce(
    TargetProfileId targetProfile, NumericReduceOperation operation,
    NumericTensorKey input, NumericTensorKey destination,
    NativeCTReduceDimension dimension) {
  if (!isKnownReduceOperation(operation))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown numeric reduce operation");
  if (!isKnownReduceDimension(dimension))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown native CT reduce dimension");
  if (input.getFormat() != destination.getFormat())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce input and destination formats must match");
  if (input.getFormat() == LogicalFormat::Bool)
    return llvm::createStringError(
        llvm::errc::not_supported,
        "native CT reduce does not accept the BOOL-specific CT format row");
  if (llvm::Error error =
          requireEngineFormat(targetProfile, TargetFormatEngine::CT,
                              input.getFormat(), "native CT reduce"))
    return std::move(error);

  const size_t rank = input.getShape().size();
  if (rank < 1 || rank > 4)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce input rank must be in [1, 4]");
  NumericTensorLayout expectedInputLayout =
      rank > 2 ? NumericTensorLayout::NCx : NumericTensorLayout::Cx;
  NumericTensorLayout expectedDestinationLayout =
      destination.getShape().size() > 2 ? NumericTensorLayout::NCx
                                        : NumericTensorLayout::Cx;
  if (input.getLayout() != expectedInputLayout ||
      destination.getLayout() != expectedDestinationLayout)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce rank <= 2 requires cx and rank > 2 requires ncx");
  for (uint64_t value : input.getShape())
    if (value == 0 || value > std::numeric_limits<uint16_t>::max())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "native CT reduce input dimensions must be positive uint16 values");

  std::vector<size_t> reduced = getReducedDimensions(dimension, rank);
  if (reduced.empty())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce dimension is not valid for the input rank");
  std::vector<uint64_t> expectedShape;
  for (size_t index = 0; index < rank; ++index)
    if (std::find(reduced.begin(), reduced.end(), index) == reduced.end())
      expectedShape.push_back(input.getShape()[index]);
  if (destination.getShape() != llvm::ArrayRef<uint64_t>(expectedShape))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "native CT reduce destination shape must contain exactly the "
        "non-reduced input dimensions");

  NumericCommandPayload payload = NumericNativeCTReduceCommand{
      operation, std::move(input), std::move(destination), dimension};
  std::string digest = makeCommandKeyDigest(targetProfile, payload);
  return NumericCommandKey(targetProfile, std::move(payload),
                           std::move(digest));
}

llvm::StringRef stringifyFormalKernelKind(FormalKernelKind kind) {
  switch (kind) {
  case FormalKernelKind::Convert:
    return "convert";
  case FormalKernelKind::Elementwise:
    return "elementwise";
  case FormalKernelKind::Gemm:
    return "gemm";
  }
  llvm_unreachable("formal kernel kind is not registered");
}

llvm::StringRef
stringifyFormalNumericBackendKind(FormalNumericBackendKind kind) {
  switch (kind) {
  case FormalNumericBackendKind::LLVMAPFloatAPInt:
    return "llvm-apfloat-apint";
  case FormalNumericBackendKind::MPFR:
    return "mpfr";
  }
  llvm_unreachable("formal numeric backend kind is not registered");
}

const TargetConvertRoute &
NumericCTConvertSemanticsIdentity::getCTConvertRoute() const {
  const TargetConvertRoute *route =
      findTargetConvertRoute(targetProfile, opcode);
  if (!route)
    llvm_unreachable("numeric route policy lost its CT convert route");
  return *route;
}

NumericCommandFamily NumericSemanticsProfile::getFamily() const {
  return std::visit(
      [](const auto &typedIdentity) { return typedIdentity.getFamily(); },
      identity);
}

const NumericCTConvertSemanticsIdentity *
NumericSemanticsProfile::getCTConvertIdentity() const {
  return std::get_if<NumericCTConvertSemanticsIdentity>(&identity);
}

const NumericCTElementwiseSemanticsIdentity *
NumericSemanticsProfile::getCTElementwiseIdentity() const {
  return std::get_if<NumericCTElementwiseSemanticsIdentity>(&identity);
}

const NumericNEGemmSemanticsIdentity *
NumericSemanticsProfile::getNEGemmIdentity() const {
  return std::get_if<NumericNEGemmSemanticsIdentity>(&identity);
}

const NumericRoutePolicyIdentity &
NumericSemanticsProfile::getRoutePolicyIdentity() const {
  const NumericCTConvertSemanticsIdentity *convert = getCTConvertIdentity();
  if (!convert)
    llvm_unreachable("non-convert semantics requested a convert identity");
  return *convert;
}

NumericRoundingMode NumericSemanticsProfile::getRoundingMode() const {
  const NumericCTConvertSemanticsIdentity *convert = getCTConvertIdentity();
  if (!convert || !roundingMode)
    llvm_unreachable("non-convert semantics requested convert rounding");
  return convert->getEffectiveRoundingMode();
}

llvm::ArrayRef<NumericSemanticsProfile>
getRegisteredNumericSemanticsProfiles() {
  static const std::vector<NumericSemanticsProfile> profiles = [] {
    std::vector<NumericSemanticsProfile> result;
    result.reserve(101);
    size_t zeroPointRouteCount = 0;
    size_t parameterlessRouteCount = 0;
    size_t roundingRouteCount = 0;
    const ModelProfileRecord &model =
        getModelProfileRecord(kFormalDeterministicV1);

    for (const TargetConvertRoute &route : getTargetConvertRoutes()) {
      if (route.profile != kTargetProfile)
        llvm::report_fatal_error(
            "numeric semantics registry saw an unexpected target profile");
      if (route.parameterKind == TargetConvertParameterKind::ZeroPoint) {
        ++zeroPointRouteCount;
        continue;
      }
      llvm::ArrayRef<NumericRoundingMode> modes;
      if (route.parameterKind == TargetConvertParameterKind::None) {
        ++parameterlessRouteCount;
        modes =
            llvm::ArrayRef<NumericRoundingMode>(kDeterministicRoundingModes, 1);
      } else {
        ++roundingRouteCount;
        modes = kDeterministicRoundingModes;
      }
      for (NumericRoundingMode mode : modes) {
        NumericSemanticsIdentity identity = NumericCTConvertSemanticsIdentity(
            route.profile, route.opcode, mode);
        const bool floatToInteger =
            isFloating(route.source) && isInteger(route.destination);
        const bool floatToFloat =
            isFloating(route.source) && isFloating(route.destination);
        const bool hasFloatingEndpoint =
            isFloating(route.source) || isFloating(route.destination);
        FloatToIntegerPolicy floatToIntegerPolicy =
            floatToInteger
                ? FloatToIntegerPolicy::FiniteInRangeRejectNaNInfOverflowNoWrite
                : FloatToIntegerPolicy::NotApplicable;
        FloatingNaNPolicy floatingNaNPolicy =
            floatToFloat ? FloatingNaNPolicy::CanonicalPositiveQuietNaN
                         : FloatingNaNPolicy::NotApplicable;
        FloatingSubnormalPolicy floatingSubnormalPolicy =
            hasFloatingEndpoint ? FloatingSubnormalPolicy::Gradual
                                : FloatingSubnormalPolicy::NotApplicable;
        FloatingTininessPolicy floatingTininessPolicy =
            isFloating(route.destination)
                ? FloatingTininessPolicy::AfterRounding
                : FloatingTininessPolicy::NotApplicable;
        NumericExceptionFlagPolicy exceptionFlagPolicy =
            hasFloatingEndpoint ? NumericExceptionFlagPolicy::ModelOnly
                                : NumericExceptionFlagPolicy::NotApplicable;
        const FloatingSignedZeroPolicy floatingSignedZeroPolicy =
            floatToFloat ? FloatingSignedZeroPolicy::Preserve
                         : FloatingSignedZeroPolicy::NotApplicable;
        std::string digest = makeSemanticsDigest(
            model, identity, mode, NumericRoundingPointPolicy::ConversionResult,
            floatToIntegerPolicy, floatingNaNPolicy, floatingSignedZeroPolicy,
            floatingSubnormalPolicy, floatingTininessPolicy,
            exceptionFlagPolicy,
            isFloating(route.source)
                ? FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot
                : FloatingNaNSignalingPolicy::NotApplicable,
            hasFloatingEndpoint ? FloatingDenormalModePolicy::GradualNoDAZNoFTZ
                                : FloatingDenormalModePolicy::NotApplicable,
            isFloating(route.destination)
                ? FloatingOverflowPolicy::IEEE754AccordingToRoundingMode
                : FloatingOverflowPolicy::NotApplicable,
            NumericSaturationPolicy::Disabled,
            NumericTranscendentalEvaluationPolicy::NotApplicable,
            NumericGemmAccumulatorPolicy::NotApplicable,
            NumericGemmAccumulatorInitializationPolicy::NotApplicable,
            NumericGemmReductionOrderPolicy::NotApplicable);
        result.push_back(NumericSemanticsProfile(
            kFormalDeterministicV1, std::move(identity), mode,
            NumericRoundingPointPolicy::ConversionResult, floatToIntegerPolicy,
            floatingNaNPolicy, floatingSignedZeroPolicy,
            floatingSubnormalPolicy, floatingTininessPolicy,
            exceptionFlagPolicy,
            isFloating(route.source)
                ? FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot
                : FloatingNaNSignalingPolicy::NotApplicable,
            hasFloatingEndpoint ? FloatingDenormalModePolicy::GradualNoDAZNoFTZ
                                : FloatingDenormalModePolicy::NotApplicable,
            isFloating(route.destination)
                ? FloatingOverflowPolicy::IEEE754AccordingToRoundingMode
                : FloatingOverflowPolicy::NotApplicable,
            NumericSaturationPolicy::Disabled,
            NumericTranscendentalEvaluationPolicy::NotApplicable,
            NumericGemmAccumulatorPolicy::NotApplicable,
            NumericGemmAccumulatorInitializationPolicy::NotApplicable,
            NumericGemmReductionOrderPolicy::NotApplicable, std::move(digest)));
      }
    }
    if (zeroPointRouteCount != 4 || parameterlessRouteCount != 9 ||
        roundingRouteCount != 23 || result.size() != 101)
      llvm::report_fatal_error(
          "numeric semantics closure is not 4 zero-point routes excluded, 9 "
          "plain policies and 23 x 4 deterministic policies");
    for (size_t index = 0; index < result.size(); ++index) {
      if (!isValidDigest(result[index].getDigest()))
        llvm::report_fatal_error(
            "numeric semantics registry produced an invalid digest");
      for (size_t other = index + 1; other < result.size(); ++other) {
        if (result[index].getRoutePolicyIdentity() ==
            result[other].getRoutePolicyIdentity())
          llvm::report_fatal_error(
              "numeric semantics registry contains a duplicate route policy");
        if (result[index].getDigest() == result[other].getDigest())
          llvm::report_fatal_error(
              "numeric semantics registry contains a duplicate digest");
      }
    }
    return result;
  }();
  return profiles;
}

llvm::ArrayRef<NumericSemanticsProfile>
getRegisteredNumericCTElementwiseSemanticsProfiles() {
  static const std::vector<NumericSemanticsProfile> profiles = [] {
    std::vector<NumericSemanticsProfile> result;
    result.reserve(88);
    const ModelProfileRecord &model =
        getModelProfileRecord(kFormalDeterministicV1);

    auto append = [&](NumericElementwiseOperation operation,
                      LogicalFormat inputFormat) {
      const LogicalFormat destinationFormat =
          getElementwiseDestinationFormat(operation, inputFormat);
      NumericSemanticsIdentity identity = NumericCTElementwiseSemanticsIdentity(
          kTargetProfile, operation, inputFormat, destinationFormat);
      const bool logic = isNumericElementwiseLogic(operation);
      const bool relation = isNumericElementwiseRelation(operation);
      const std::optional<NumericRoundingMode> roundingMode =
          logic || relation ? std::nullopt
                            : std::optional<NumericRoundingMode>(
                                  NumericRoundingMode::NearestEven);
      const NumericRoundingPointPolicy roundingPointPolicy =
          logic || relation ? NumericRoundingPointPolicy::NotApplicable
                            : NumericRoundingPointPolicy::ElementwiseResult;
      const FloatingNaNPolicy floatingNaNPolicy =
          logic      ? FloatingNaNPolicy::NotApplicable
          : relation ? FloatingNaNPolicy::OrderedRelationFalseExceptNotEqualTrue
                     : FloatingNaNPolicy::CanonicalPositiveQuietNaN;
      FloatingSignedZeroPolicy floatingSignedZeroPolicy =
          logic      ? FloatingSignedZeroPolicy::NotApplicable
          : relation ? FloatingSignedZeroPolicy::PredicateOnly
                     : FloatingSignedZeroPolicy::IEEE754OperationDefined;
      if (operation == NumericElementwiseOperation::Max)
        floatingSignedZeroPolicy =
            FloatingSignedZeroPolicy::MaximumPositiveUnlessBothNegative;
      else if (operation == NumericElementwiseOperation::Min)
        floatingSignedZeroPolicy =
            FloatingSignedZeroPolicy::MinimumNegativeUnlessBothPositive;
      const FloatingSubnormalPolicy floatingSubnormalPolicy =
          logic ? FloatingSubnormalPolicy::NotApplicable
                : FloatingSubnormalPolicy::Gradual;
      const FloatingTininessPolicy floatingTininessPolicy =
          logic || relation ? FloatingTininessPolicy::NotApplicable
                            : FloatingTininessPolicy::AfterRounding;
      const NumericExceptionFlagPolicy exceptionFlagPolicy =
          logic ? NumericExceptionFlagPolicy::NotApplicable
                : NumericExceptionFlagPolicy::ModelOnly;
      const FloatingNaNSignalingPolicy floatingNaNSignalingPolicy =
          logic
              ? FloatingNaNSignalingPolicy::NotApplicable
              : FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot;
      const FloatingDenormalModePolicy floatingDenormalModePolicy =
          logic ? FloatingDenormalModePolicy::NotApplicable
                : FloatingDenormalModePolicy::GradualNoDAZNoFTZ;
      const FloatingOverflowPolicy floatingOverflowPolicy =
          logic || relation
              ? FloatingOverflowPolicy::NotApplicable
              : FloatingOverflowPolicy::IEEE754AccordingToRoundingMode;
      const NumericSaturationPolicy saturationPolicy =
          logic || relation ? NumericSaturationPolicy::NotApplicable
                            : NumericSaturationPolicy::Disabled;
      const NumericTranscendentalEvaluationPolicy
          transcendentalEvaluationPolicy =
              isMPFRElementwiseOperation(operation)
                  ? NumericTranscendentalEvaluationPolicy::
                        CorrectlyRoundedMathematicalResultAdaptiveMPFRFinalRNE
                  : NumericTranscendentalEvaluationPolicy::NotApplicable;
      std::string digest = makeSemanticsDigest(
          model, identity, roundingMode, roundingPointPolicy,
          FloatToIntegerPolicy::NotApplicable, floatingNaNPolicy,
          floatingSignedZeroPolicy, floatingSubnormalPolicy,
          floatingTininessPolicy, exceptionFlagPolicy,
          floatingNaNSignalingPolicy, floatingDenormalModePolicy,
          floatingOverflowPolicy, saturationPolicy,
          transcendentalEvaluationPolicy,
          NumericGemmAccumulatorPolicy::NotApplicable,
          NumericGemmAccumulatorInitializationPolicy::NotApplicable,
          NumericGemmReductionOrderPolicy::NotApplicable);
      result.push_back(NumericSemanticsProfile(
          kFormalDeterministicV1, std::move(identity), roundingMode,
          roundingPointPolicy, FloatToIntegerPolicy::NotApplicable,
          floatingNaNPolicy, floatingSignedZeroPolicy, floatingSubnormalPolicy,
          floatingTininessPolicy, exceptionFlagPolicy,
          floatingNaNSignalingPolicy, floatingDenormalModePolicy,
          floatingOverflowPolicy, saturationPolicy,
          transcendentalEvaluationPolicy,
          NumericGemmAccumulatorPolicy::NotApplicable,
          NumericGemmAccumulatorInitializationPolicy::NotApplicable,
          NumericGemmReductionOrderPolicy::NotApplicable, std::move(digest)));
    };

    for (NumericElementwiseOperation operation :
         getNumericElementwiseOperations()) {
      if (isNumericElementwiseLogic(operation)) {
        append(operation, LogicalFormat::Bool);
        continue;
      }
      for (LogicalFormat format :
           getCompilerNumericFormats(TargetFormatEngine::CT)) {
        if (getElementwiseUnsupportedReason(operation, format))
          continue;
        append(operation, format);
      }
    }
    if (result.size() != 88)
      llvm::report_fatal_error(
          "elementwise numeric semantics closure is not 88 supported rows");
    for (size_t index = 0; index < result.size(); ++index) {
      if (!isValidDigest(result[index].getDigest()) ||
          result[index].getFamily() != NumericCommandFamily::CTElementwise)
        llvm::report_fatal_error(
            "elementwise numeric semantics registry has an invalid row");
      for (size_t other = index + 1; other < result.size(); ++other) {
        if (result[index].getIdentity() == result[other].getIdentity())
          llvm::report_fatal_error(
              "elementwise numeric semantics registry has a duplicate "
              "identity");
        if (result[index].getDigest() == result[other].getDigest())
          llvm::report_fatal_error(
              "elementwise numeric semantics registry has a duplicate "
              "digest");
      }
    }
    return result;
  }();
  return profiles;
}

llvm::ArrayRef<NumericSemanticsProfile>
getRegisteredNumericNEGemmSemanticsProfiles() {
  static const std::vector<NumericSemanticsProfile> profiles = [] {
    std::vector<NumericSemanticsProfile> result;
    result.reserve(3);
    const ModelProfileRecord &model =
        getModelProfileRecord(kFormalDeterministicV1);
    for (LogicalFormat format :
         getCompilerNumericFormats(TargetFormatEngine::NE)) {
      if (format == LogicalFormat::I8)
        continue;
      NumericSemanticsIdentity identity =
          NumericNEGemmSemanticsIdentity(kTargetProfile, format);
      std::string digest = makeSemanticsDigest(
          model, identity, NumericRoundingMode::NearestEven,
          NumericRoundingPointPolicy::GemmFusedMultiplyAddAndDestination,
          FloatToIntegerPolicy::NotApplicable,
          FloatingNaNPolicy::CanonicalPositiveQuietNaN,
          FloatingSignedZeroPolicy::GemmPositiveZeroAccumulatorThenIEEE754,
          FloatingSubnormalPolicy::Gradual,
          FloatingTininessPolicy::AfterRounding,
          NumericExceptionFlagPolicy::ModelOnly,
          FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot,
          FloatingDenormalModePolicy::GradualNoDAZNoFTZ,
          FloatingOverflowPolicy::IEEE754AccordingToRoundingMode,
          NumericSaturationPolicy::Disabled,
          NumericTranscendentalEvaluationPolicy::NotApplicable,
          NumericGemmAccumulatorPolicy::F32FusedMultiplyAdd,
          NumericGemmAccumulatorInitializationPolicy::PositiveZero,
          NumericGemmReductionOrderPolicy::IncreasingK);
      result.push_back(NumericSemanticsProfile(
          kFormalDeterministicV1, std::move(identity),
          NumericRoundingMode::NearestEven,
          NumericRoundingPointPolicy::GemmFusedMultiplyAddAndDestination,
          FloatToIntegerPolicy::NotApplicable,
          FloatingNaNPolicy::CanonicalPositiveQuietNaN,
          FloatingSignedZeroPolicy::GemmPositiveZeroAccumulatorThenIEEE754,
          FloatingSubnormalPolicy::Gradual,
          FloatingTininessPolicy::AfterRounding,
          NumericExceptionFlagPolicy::ModelOnly,
          FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot,
          FloatingDenormalModePolicy::GradualNoDAZNoFTZ,
          FloatingOverflowPolicy::IEEE754AccordingToRoundingMode,
          NumericSaturationPolicy::Disabled,
          NumericTranscendentalEvaluationPolicy::NotApplicable,
          NumericGemmAccumulatorPolicy::F32FusedMultiplyAdd,
          NumericGemmAccumulatorInitializationPolicy::PositiveZero,
          NumericGemmReductionOrderPolicy::IncreasingK, std::move(digest)));
    }
    if (result.size() != 3)
      llvm::report_fatal_error(
          "GEMM numeric semantics closure is not three supported rows");
    for (size_t index = 0; index < result.size(); ++index) {
      if (!isValidDigest(result[index].getDigest()) ||
          result[index].getFamily() != NumericCommandFamily::NEGemm)
        llvm::report_fatal_error(
            "GEMM numeric semantics registry has an invalid row");
      for (size_t other = index + 1; other < result.size(); ++other) {
        if (result[index].getIdentity() == result[other].getIdentity())
          llvm::report_fatal_error(
              "GEMM numeric semantics registry has a duplicate identity");
        if (result[index].getDigest() == result[other].getDigest())
          llvm::report_fatal_error(
              "GEMM numeric semantics registry has a duplicate digest");
      }
    }
    return result;
  }();
  return profiles;
}

llvm::StringRef stringifyNumericModelImplementationReason(
    NumericModelImplementationReason reason) {
  switch (reason) {
  case NumericModelImplementationReason::None:
    return "none";
  case NumericModelImplementationReason::StochasticStateUnproven:
    return "stochastic-state-unproven";
  case NumericModelImplementationReason::ZeroPointFormulaUnproven:
    return "zero-point-formula-unproven";
  case NumericModelImplementationReason::FormalKernelNotImplemented:
    return "formal-kernel-not-implemented";
  case NumericModelImplementationReason::IntegerElementwisePolicyUnproven:
    return "integer-elementwise-policy-unproven";
  case NumericModelImplementationReason::ExpLpParameterPolicyUnproven:
    return "exp-lp-parameter-policy-unproven";
  case NumericModelImplementationReason::SatReluParameterPolicyUnproven:
    return "sat-relu-parameter-policy-unproven";
  case NumericModelImplementationReason::LeakyReluParameterPolicyUnproven:
    return "leaky-relu-parameter-policy-unproven";
  case NumericModelImplementationReason::IntegerGemmAccumulatorPolicyUnproven:
    return "integer-gemm-accumulator-policy-unproven";
  case NumericModelImplementationReason::NativeReductionPolicyUnproven:
    return "native-reduction-policy-unproven";
  }
  llvm_unreachable("numeric model implementation reason is not registered");
}

llvm::StringRef stringifyNumericCompilerEmittabilityReason(
    NumericCompilerEmittabilityReason reason) {
  switch (reason) {
  case NumericCompilerEmittabilityReason::None:
    return "none";
  }
  llvm_unreachable("numeric compiler reason is not registered");
}

llvm::StringRef stringifyNumericEvidenceReason(NumericEvidenceReason reason) {
  switch (reason) {
  case NumericEvidenceReason::HardwareCorrelationNotRun:
    return "hardware-correlation-not-run";
  }
  llvm_unreachable("numeric evidence reason is not registered");
}

bool NumericCapabilityParameterPattern::matches(
    const std::optional<NumericConvertParameter> &parameter) const {
  switch (kind) {
  case NumericCapabilityParameterPatternKind::NoParameter:
    return !parameter;
  case NumericCapabilityParameterPatternKind::ExactRoundingMode:
    return parameter && parameter->getRoundingMode() == exactRoundingModeValue;
  case NumericCapabilityParameterPatternKind::AnyZeroPoint:
    return parameter && parameter->getZeroPoint().has_value();
  }
  return false;
}

const NumericCTConvertPatternSelector *
NumericCapabilityPattern::getCTConvertSelector() const {
  return std::get_if<NumericCTConvertPatternSelector>(&selector);
}
const NumericCTElementwisePatternSelector *
NumericCapabilityPattern::getCTElementwiseSelector() const {
  return std::get_if<NumericCTElementwisePatternSelector>(&selector);
}
const NumericNEGemmPatternSelector *
NumericCapabilityPattern::getNEGemmSelector() const {
  return std::get_if<NumericNEGemmPatternSelector>(&selector);
}
const NumericNativeCTReducePatternSelector *
NumericCapabilityPattern::getNativeCTReduceSelector() const {
  return std::get_if<NumericNativeCTReducePatternSelector>(&selector);
}

bool NumericCapabilityPattern::matches(
    ModelProfileId candidateModel,
    const NumericCommandKey &candidateKey) const {
  if (modelProfile != candidateModel ||
      targetProfile != candidateKey.getTargetProfile() ||
      family != candidateKey.getFamily())
    return false;
  if (const auto *typedSelector = getCTConvertSelector()) {
    const NumericCTConvertCommand *command = candidateKey.getCTConvert();
    return command && command->opcode == typedSelector->opcode &&
           typedSelector->parameter.matches(command->parameter);
  }
  if (const auto *typedSelector = getCTElementwiseSelector()) {
    const NumericCTElementwiseCommand *command =
        candidateKey.getCTElementwise();
    return command && command->operation == typedSelector->operation &&
           !command->inputs.empty() &&
           command->inputs.front().getFormat() == typedSelector->inputFormat;
  }
  if (const auto *typedSelector = getNEGemmSelector()) {
    const NumericNEGemmCommand *command = candidateKey.getNEGemm();
    return command && command->lhs.getFormat() == typedSelector->format;
  }
  const auto *typedSelector = getNativeCTReduceSelector();
  const NumericNativeCTReduceCommand *command =
      candidateKey.getNativeCTReduce();
  return typedSelector && command &&
         command->operation == typedSelector->operation &&
         command->input.getFormat() == typedSelector->format;
}

llvm::ArrayRef<NumericCapabilityPattern>
getRegisteredNumericCapabilityPatterns() {
  static const std::vector<NumericCapabilityPattern> patterns = [] {
    std::vector<NumericCapabilityPattern> result;
    result.reserve(276);
    const ModelProfileRecord &model =
        getModelProfileRecord(kFormalDeterministicV1);
    constexpr NumericCompilerEmittabilityCapability compilerCapability = {
        NumericCompilerEmittabilityStatus::Emittable,
        NumericCompilerEmittabilityReason::None,
    };
    constexpr NumericEvidenceCapability evidenceCapability = {
        NumericEvidenceStatus::ModelOnlyUncorrelated,
        NumericEvidenceReason::HardwareCorrelationNotRun,
    };

    auto appendPattern = [&](NumericCommandFamily family,
                             NumericCapabilitySelector selector,
                             NumericModelImplementationCapability modelAxis,
                             const NumericSemanticsProfile *semantics,
                             std::optional<FormalKernelKind> kernel,
                             std::optional<FormalNumericBackendKind> backend) {
      std::optional<NumericComparatorKind> comparator;
      if (modelAxis.status == NumericModelImplementationStatus::Implemented)
        comparator = NumericComparatorKind::RawExact;
      std::string digest =
          makePatternDigest(model, kTargetProfile, family, selector, modelAxis,
                            compilerCapability, evidenceCapability, semantics,
                            kernel, comparator, backend);
      result.push_back(NumericCapabilityPattern(
          kFormalDeterministicV1, kTargetProfile, family, std::move(selector),
          modelAxis, compilerCapability, evidenceCapability, semantics, kernel,
          comparator, backend, std::move(digest)));
    };

    for (const TargetConvertRoute &route : getTargetConvertRoutes()) {
      switch (route.parameterKind) {
      case TargetConvertParameterKind::ZeroPoint:
        appendPattern(
            NumericCommandFamily::CTConvert,
            NumericCTConvertPatternSelector{
                route.opcode,
                NumericCapabilityParameterPattern::anyZeroPoint()},
            {NumericModelImplementationStatus::Absent,
             NumericModelImplementationReason::ZeroPointFormulaUnproven},
            nullptr, std::nullopt, std::nullopt);
        break;
      case TargetConvertParameterKind::None: {
        const NumericSemanticsProfile *semantics = findCTConvertSemantics(
            kFormalDeterministicV1, route.profile, route.opcode,
            NumericRoundingMode::NearestEven);
        if (!semantics)
          llvm::report_fatal_error(
              "plain capability pattern has no reusable semantics");
        appendPattern(
            NumericCommandFamily::CTConvert,
            NumericCTConvertPatternSelector{
                route.opcode, NumericCapabilityParameterPattern::noParameter()},
            {NumericModelImplementationStatus::Implemented,
             NumericModelImplementationReason::None},
            semantics, FormalKernelKind::Convert,
            FormalNumericBackendKind::LLVMAPFloatAPInt);
        break;
      }
      case TargetConvertParameterKind::RoundingMode:
        for (NumericRoundingMode mode : kRoundingModes) {
          if (mode == NumericRoundingMode::Stochastic) {
            appendPattern(
                NumericCommandFamily::CTConvert,
                NumericCTConvertPatternSelector{
                    route.opcode,
                    NumericCapabilityParameterPattern::exactRoundingMode(mode)},
                {NumericModelImplementationStatus::Absent,
                 NumericModelImplementationReason::StochasticStateUnproven},
                nullptr, std::nullopt, std::nullopt);
            continue;
          }
          const NumericSemanticsProfile *semantics = findCTConvertSemantics(
              kFormalDeterministicV1, route.profile, route.opcode, mode);
          if (!semantics)
            llvm::report_fatal_error(
                "deterministic capability pattern has no reusable semantics");
          appendPattern(
              NumericCommandFamily::CTConvert,
              NumericCTConvertPatternSelector{
                  route.opcode,
                  NumericCapabilityParameterPattern::exactRoundingMode(mode)},
              {NumericModelImplementationStatus::Implemented,
               NumericModelImplementationReason::None},
              semantics, FormalKernelKind::Convert,
              FormalNumericBackendKind::LLVMAPFloatAPInt);
        }
        break;
      }
    }

    for (NumericElementwiseOperation operation :
         getNumericElementwiseOperations()) {
      if (isNumericElementwiseLogic(operation)) {
        const NumericSemanticsProfile *semantics =
            findCTElementwiseSemantics(kFormalDeterministicV1, kTargetProfile,
                                       operation, LogicalFormat::Bool);
        if (!semantics)
          llvm::report_fatal_error(
              "BOOL logic capability pattern has no reusable semantics");
        appendPattern(
            NumericCommandFamily::CTElementwise,
            NumericCTElementwisePatternSelector{operation, LogicalFormat::Bool},
            {NumericModelImplementationStatus::Implemented,
             NumericModelImplementationReason::None},
            semantics, FormalKernelKind::Elementwise,
            FormalNumericBackendKind::LLVMAPFloatAPInt);
        continue;
      }
      for (LogicalFormat format :
           getCompilerNumericFormats(TargetFormatEngine::CT)) {
        if (std::optional<NumericModelImplementationReason> reason =
                getElementwiseUnsupportedReason(operation, format)) {
          appendPattern(NumericCommandFamily::CTElementwise,
                        NumericCTElementwisePatternSelector{operation, format},
                        {NumericModelImplementationStatus::Absent, *reason},
                        nullptr, std::nullopt, std::nullopt);
          continue;
        }
        const NumericSemanticsProfile *semantics = findCTElementwiseSemantics(
            kFormalDeterministicV1, kTargetProfile, operation, format);
        if (!semantics)
          llvm::report_fatal_error(
              "elementwise capability pattern has no reusable semantics");
        appendPattern(NumericCommandFamily::CTElementwise,
                      NumericCTElementwisePatternSelector{operation, format},
                      {NumericModelImplementationStatus::Implemented,
                       NumericModelImplementationReason::None},
                      semantics, FormalKernelKind::Elementwise,
                      isMPFRElementwiseOperation(operation)
                          ? FormalNumericBackendKind::MPFR
                          : FormalNumericBackendKind::LLVMAPFloatAPInt);
      }
    }
    for (LogicalFormat format :
         getCompilerNumericFormats(TargetFormatEngine::NE)) {
      if (format == LogicalFormat::I8) {
        appendPattern(NumericCommandFamily::NEGemm,
                      NumericNEGemmPatternSelector{format},
                      {NumericModelImplementationStatus::Absent,
                       NumericModelImplementationReason::
                           IntegerGemmAccumulatorPolicyUnproven},
                      nullptr, std::nullopt, std::nullopt);
        continue;
      }
      const NumericSemanticsProfile *semantics =
          findNEGemmSemantics(kFormalDeterministicV1, kTargetProfile, format);
      if (!semantics)
        llvm::report_fatal_error(
            "GEMM capability pattern has no reusable semantics");
      appendPattern(NumericCommandFamily::NEGemm,
                    NumericNEGemmPatternSelector{format},
                    {NumericModelImplementationStatus::Implemented,
                     NumericModelImplementationReason::None},
                    semantics, FormalKernelKind::Gemm,
                    FormalNumericBackendKind::LLVMAPFloatAPInt);
    }
    for (NumericReduceOperation operation : getNumericReduceOperations())
      for (LogicalFormat format :
           getCompilerNumericFormats(TargetFormatEngine::CT))
        appendPattern(
            NumericCommandFamily::NativeCTReduce,
            NumericNativeCTReducePatternSelector{operation, format},
            {NumericModelImplementationStatus::Absent,
             NumericModelImplementationReason::NativeReductionPolicyUnproven},
            nullptr, std::nullopt, std::nullopt);

    if (llvm::Error error = validatePatterns(result)) {
      std::string message = llvm::toString(std::move(error));
      llvm::report_fatal_error(llvm::StringRef(message));
    }
    return result;
  }();
  return patterns;
}

llvm::Error validateNumericCapabilityPatternsForTesting(
    llvm::ArrayRef<NumericCapabilityPattern> patterns) {
  return validatePatterns(patterns);
}

llvm::Expected<ResolvedNumericCommand>
resolveNumericCommand(ModelProfileId modelProfile,
                      NumericCommandKey commandKey) {
  bool modelRegistered = false;
  for (const ModelProfileRecord &record : getRegisteredModelProfiles())
    modelRegistered |= record.id == modelProfile;
  if (!modelRegistered)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "model profile is not registered");

  const NumericCapabilityPattern *match = nullptr;
  for (const NumericCapabilityPattern &pattern :
       getRegisteredNumericCapabilityPatterns()) {
    if (!pattern.matches(modelProfile, commandKey))
      continue;
    if (match)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "exact numeric command key matches multiple capability patterns");
    match = &pattern;
  }
  if (!match)
    return llvm::createStringError(
        llvm::errc::not_supported,
        "exact numeric command key has no capability pattern");

  const ModelProfileRecord &model = getModelProfileRecord(modelProfile);
  std::string digest = makeResolutionDigest(model, commandKey, *match);
  if (!isValidDigest(digest))
    llvm::report_fatal_error("invalid numeric resolution digest");
  return ResolvedNumericCommand(std::move(commandKey), match,
                                std::move(digest));
}

} // namespace wafer
