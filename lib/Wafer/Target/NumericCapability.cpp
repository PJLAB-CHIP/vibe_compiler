//===- NumericCapability.cpp - Numeric capability registry ---------------===//

#include "Wafer/Target/NumericSemantics.h"

#include "NumericSemanticsInternal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace wafer {
using numeric_semantics_internal::digestCanonical;
using numeric_semantics_internal::getCompilerNumericFormats;
using numeric_semantics_internal::getElementwiseDestinationFormat;
using numeric_semantics_internal::getElementwiseUnsupportedReason;
using numeric_semantics_internal::isFloating;
using numeric_semantics_internal::isInteger;
using numeric_semantics_internal::isMPFRElementwiseOperation;
using numeric_semantics_internal::isValidDigest;

namespace {

constexpr ModelProfileId kFormalDeterministicV1 =
    ModelProfileId::formalDeterministicV1();

void writeSelector(llvm::raw_ostream &stream,
                   const NumericCapabilitySelector &selector) {
  std::visit(
      [&](const auto &typedSelector) {
        using Selector = std::decay_t<decltype(typedSelector)>;
        if constexpr (std::is_same_v<Selector,
                                     NumericCTConvertPatternSelector>) {
          const TargetConvertRoute *route =
              findTargetConvertRoute(typedSelector.opcode);
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
makePatternDigest(const ModelProfileRecord &model, NumericCommandFamily family,
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
  stream << "wafer-numeric-capability-pattern\n"
         << "model-policy-digest=" << model.policyDigest << '\n'
         << "family=" << stringifyNumericCommandFamily(family) << '\n';
  writeSelector(stream, selector);
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

const NumericSemanticsProfile *
findCTConvertSemantics(ModelProfileId modelProfile, uint16_t opcode,
                       NumericRoundingMode effectiveRounding) {
  const NumericSemanticsProfile *match = nullptr;
  for (const NumericSemanticsProfile &profile :
       getRegisteredNumericCTConvertSemanticsProfiles()) {
    const NumericCTConvertSemanticsKey *key = profile.getCTConvertKey();
    if (key && profile.getModelProfile() == modelProfile &&
        key->getCTConvertOpcode() == opcode &&
        key->getEffectiveRoundingMode() == effectiveRounding) {
      if (match)
        llvm::report_fatal_error("duplicate reusable numeric semantics key");
      match = &profile;
    }
  }
  return match;
}

const NumericSemanticsProfile *
findCTElementwiseSemantics(ModelProfileId modelProfile,
                           NumericElementwiseOperation operation,
                           LogicalFormat inputFormat) {
  const NumericSemanticsProfile *match = nullptr;
  for (const NumericSemanticsProfile &profile :
       getRegisteredNumericCTElementwiseSemanticsProfiles()) {
    const NumericCTElementwiseSemanticsKey *key = profile.getCTElementwiseKey();
    if (key && profile.getModelProfile() == modelProfile &&
        key->getOperation() == operation &&
        key->getInputFormat() == inputFormat) {
      if (match)
        llvm::report_fatal_error(
            "duplicate reusable elementwise semantics key");
      match = &profile;
    }
  }
  return match;
}

const NumericSemanticsProfile *findNEGemmSemantics(ModelProfileId modelProfile,
                                                   LogicalFormat format) {
  const NumericSemanticsProfile *match = nullptr;
  for (const NumericSemanticsProfile &profile :
       getRegisteredNumericNEGemmSemanticsProfiles()) {
    const NumericNEGemmSemanticsKey *key = profile.getNEGemmKey();
    if (key && profile.getModelProfile() == modelProfile &&
        key->getFormat() == format) {
      if (match)
        llvm::report_fatal_error("duplicate reusable GEMM semantics key");
      match = &profile;
    }
  }
  return match;
}

const NumericSemanticsProfile *
findNativeCTReduceSemantics(ModelProfileId modelProfile,
                            NumericReduceOperation operation,
                            LogicalFormat format) {
  const NumericSemanticsProfile *match = nullptr;
  for (const NumericSemanticsProfile &profile :
       getRegisteredNumericNativeCTReduceSemanticsProfiles()) {
    const NumericNativeCTReduceSemanticsKey *key =
        profile.getNativeCTReduceKey();
    if (key && profile.getModelProfile() == modelProfile &&
        key->getOperation() == operation && key->getFormat() == format) {
      if (match)
        llvm::report_fatal_error(
            "duplicate reusable native reduction semantics key");
      match = &profile;
    }
  }
  return match;
}

bool selectorsOverlap(const NumericCapabilityPattern &lhs,
                      const NumericCapabilityPattern &rhs) {
  if (lhs.getModelProfile() != rhs.getModelProfile() ||
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
  size_t reduceSupported = 0;
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
          "key");
    if (!referencedSemantics.insert(semantics).second)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "supported numeric patterns reuse one semantics key");
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
          "execution key");
    return llvm::Error::success();
  };

  for (size_t index = 0; index < patterns.size(); ++index) {
    const NumericCapabilityPattern &pattern = patterns[index];
    if (pattern.getModelProfile() != kFormalDeterministicV1)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "numeric capability pattern has an unexpected model");
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
        const NumericCTConvertSemanticsKey *key =
            pattern.getSemantics()->getCTConvertKey();
        std::optional<NumericRoundingMode> expectedRounding =
            selector->parameter.getExactRoundingMode();
        if (selector->parameter.getKind() ==
            NumericCapabilityParameterPatternKind::NoParameter)
          expectedRounding = NumericRoundingMode::NearestEven;
        if (!key || key->getCTConvertOpcode() != selector->opcode ||
            !expectedRounding ||
            key->getEffectiveRoundingMode() != *expectedRounding ||
            pattern.getSemantics()->getRoundingModePolicy() !=
                expectedRounding ||
            pattern.getSemantics()->getRoundingPointPolicy() !=
                NumericRoundingPointPolicy::ConversionResult)
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "supported convert selector does not match its typed semantics");
        const TargetConvertRoute &route = key->getCTConvertRoute();
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
                NumericGemmReductionOrderPolicy::NotApplicable ||
            pattern.getSemantics()
                    ->getReductionAccumulatorInitializationPolicy() !=
                NumericReductionAccumulatorInitializationPolicy::
                    NotApplicable ||
            pattern.getSemantics()->getReductionOrderPolicy() !=
                NumericReductionOrderPolicy::NotApplicable)
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
      const NumericCTElementwiseSemanticsKey *key =
          pattern.getSemantics()->getCTElementwiseKey();
      const LogicalFormat expectedDestination = getElementwiseDestinationFormat(
          selector->operation, selector->inputFormat);
      if (!key || key->getOperation() != selector->operation ||
          key->getInputFormat() != selector->inputFormat ||
          key->getDestinationFormat() != expectedDestination)
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
              NumericGemmReductionOrderPolicy::NotApplicable ||
          pattern.getSemantics()
                  ->getReductionAccumulatorInitializationPolicy() !=
              NumericReductionAccumulatorInitializationPolicy::NotApplicable ||
          pattern.getSemantics()->getReductionOrderPolicy() !=
              NumericReductionOrderPolicy::NotApplicable)
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
      const NumericNEGemmSemanticsKey *key =
          pattern.getSemantics()->getNEGemmKey();
      if (!key || key->getFormat() != selector->format ||
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
              NumericTranscendentalEvaluationPolicy::NotApplicable ||
          pattern.getSemantics()
                  ->getReductionAccumulatorInitializationPolicy() !=
              NumericReductionAccumulatorInitializationPolicy::NotApplicable ||
          pattern.getSemantics()->getReductionOrderPolicy() !=
              NumericReductionOrderPolicy::NotApplicable)
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
      const bool supported =
          selector->operation == NumericReduceOperation::Sum &&
          selector->format == LogicalFormat::F32;
      if (!supported) {
        if (llvm::Error error =
                requireUnsupported(pattern, NumericModelImplementationReason::
                                                NativeReductionPolicyUnproven))
          return error;
      } else {
        ++reduceSupported;
        if (llvm::Error error = requireSupportedExecution(
                pattern, NumericCommandFamily::NativeCTReduce,
                FormalKernelKind::Reduce,
                FormalNumericBackendKind::LLVMAPFloatAPInt))
          return error;
        const NumericNativeCTReduceSemanticsKey *key =
            pattern.getSemantics()->getNativeCTReduceKey();
        if (!key || key->getOperation() != NumericReduceOperation::Sum ||
            key->getFormat() != LogicalFormat::F32 ||
            pattern.getSemantics()->getRoundingModePolicy() !=
                NumericRoundingMode::NearestEven ||
            pattern.getSemantics()->getRoundingPointPolicy() !=
                NumericRoundingPointPolicy::ReductionStep ||
            pattern.getSemantics()->getFloatingNaNPolicy() !=
                FloatingNaNPolicy::CanonicalPositiveQuietNaN ||
            pattern.getSemantics()->getFloatingSignedZeroPolicy() !=
                FloatingSignedZeroPolicy::
                    ReductionPositiveZeroAccumulatorThenIEEE754 ||
            pattern.getSemantics()->getFloatToIntegerPolicy() !=
                FloatToIntegerPolicy::NotApplicable ||
            pattern.getSemantics()->getFloatingSubnormalPolicy() !=
                FloatingSubnormalPolicy::Gradual ||
            pattern.getSemantics()->getFloatingTininessPolicy() !=
                FloatingTininessPolicy::AfterRounding ||
            pattern.getSemantics()->getExceptionFlagPolicy() !=
                NumericExceptionFlagPolicy::ModelOnly ||
            pattern.getSemantics()->getFloatingNaNSignalingPolicy() !=
                FloatingNaNSignalingPolicy::
                    SignalingRaisesInvalidQuietDoesNot ||
            pattern.getSemantics()->getFloatingDenormalModePolicy() !=
                FloatingDenormalModePolicy::GradualNoDAZNoFTZ ||
            pattern.getSemantics()->getFloatingOverflowPolicy() !=
                FloatingOverflowPolicy::IEEE754AccordingToRoundingMode ||
            pattern.getSemantics()->getSaturationPolicy() !=
                NumericSaturationPolicy::Disabled ||
            pattern.getSemantics()->getTranscendentalEvaluationPolicy() !=
                NumericTranscendentalEvaluationPolicy::NotApplicable ||
            pattern.getSemantics()->getGemmAccumulatorPolicy() !=
                NumericGemmAccumulatorPolicy::NotApplicable ||
            pattern.getSemantics()->getGemmAccumulatorInitializationPolicy() !=
                NumericGemmAccumulatorInitializationPolicy::NotApplicable ||
            pattern.getSemantics()->getGemmReductionOrderPolicy() !=
                NumericGemmReductionOrderPolicy::NotApplicable ||
            pattern.getSemantics()
                    ->getReductionAccumulatorInitializationPolicy() !=
                NumericReductionAccumulatorInitializationPolicy::PositiveZero ||
            pattern.getSemantics()->getReductionOrderPolicy() !=
                NumericReductionOrderPolicy::
                    IncreasingLogicalRowMajorInputIndex)
          return llvm::createStringError(
              llvm::errc::invalid_argument,
              "supported native reduction selector does not match its typed "
              "semantics");
      }
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
      gemmInteger != 1 || reduce != 16 || reduceSupported != 1 ||
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
  if (getRegisteredNumericCTConvertSemanticsProfiles().size() != 101 ||
      getRegisteredNumericCTElementwiseSemanticsProfiles().size() != 88 ||
      getRegisteredNumericNEGemmSemanticsProfiles().size() != 3 ||
      getRegisteredNumericNativeCTReduceSemanticsProfiles().size() != 1 ||
      referencedSemantics.size() != 193)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "numeric semantics registry is missing or has an extra supported row");
  if (llvm::Error error = requireCompleteSemanticsRegistry(
          getRegisteredNumericCTConvertSemanticsProfiles(),
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
  if (llvm::Error error = requireCompleteSemanticsRegistry(
          getRegisteredNumericNativeCTReduceSemanticsProfiles(),
          NumericCommandFamily::NativeCTReduce))
    return error;
  return llvm::Error::success();
}

} // namespace

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
  if (modelProfile != candidateModel || family != candidateKey.getFamily())
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
      std::string digest = makePatternDigest(
          model, family, selector, modelAxis, compilerCapability,
          evidenceCapability, semantics, kernel, comparator, backend);
      result.push_back(NumericCapabilityPattern(
          kFormalDeterministicV1, family, std::move(selector), modelAxis,
          compilerCapability, evidenceCapability, semantics, kernel, comparator,
          backend, std::move(digest)));
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
        const NumericSemanticsProfile *semantics =
            findCTConvertSemantics(kFormalDeterministicV1, route.opcode,
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
        for (NumericRoundingMode mode : getNumericRoundingModes()) {
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
              kFormalDeterministicV1, route.opcode, mode);
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
        const NumericSemanticsProfile *semantics = findCTElementwiseSemantics(
            kFormalDeterministicV1, operation, LogicalFormat::Bool);
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
            kFormalDeterministicV1, operation, format);
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
          findNEGemmSemantics(kFormalDeterministicV1, format);
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
           getCompilerNumericFormats(TargetFormatEngine::CT)) {
        const bool supported = operation == NumericReduceOperation::Sum &&
                               format == LogicalFormat::F32;
        if (!supported) {
          appendPattern(
              NumericCommandFamily::NativeCTReduce,
              NumericNativeCTReducePatternSelector{operation, format},
              {NumericModelImplementationStatus::Absent,
               NumericModelImplementationReason::NativeReductionPolicyUnproven},
              nullptr, std::nullopt, std::nullopt);
          continue;
        }
        const NumericSemanticsProfile *semantics = findNativeCTReduceSemantics(
            kFormalDeterministicV1, operation, format);
        if (!semantics)
          llvm::report_fatal_error(
              "native reduction capability pattern has no reusable semantics");
        appendPattern(NumericCommandFamily::NativeCTReduce,
                      NumericNativeCTReducePatternSelector{operation, format},
                      {NumericModelImplementationStatus::Implemented,
                       NumericModelImplementationReason::None},
                      semantics, FormalKernelKind::Reduce,
                      FormalNumericBackendKind::LLVMAPFloatAPInt);
      }

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
