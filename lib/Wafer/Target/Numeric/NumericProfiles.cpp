//===- NumericProfiles.cpp - Numeric semantics profiles ------------------===//

#include "Wafer/Target/Numeric/NumericSemantics.h"

#include "NumericSemanticsInternal.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

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

constexpr ModelProfileId kFormalDeterministic =
    ModelProfileId::formalDeterministic();
constexpr NumericRoundingMode kDeterministicRoundingModes[] = {
    NumericRoundingMode::NearestEven,
    NumericRoundingMode::TowardZero,
    NumericRoundingMode::TowardPositive,
    NumericRoundingMode::TowardNegative,
};

std::string makeModelPolicyDigest(const ModelProfileRecord &record) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream
      << "wafer-model-policy\n"
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

std::string makeSemanticsDigest(
    const ModelProfileRecord &model, const NumericSemanticsKey &key,
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
    NumericGemmReductionOrderPolicy gemmReductionOrderPolicy,
    NumericReductionAccumulatorInitializationPolicy
        reductionAccumulatorInitializationPolicy,
    NumericReductionOrderPolicy reductionOrderPolicy) {
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  stream << "wafer-numeric-semantics\n"
         << "model-policy-digest=" << model.policyDigest << '\n';
  std::visit(
      [&](const auto &typedKey) {
        using KeyType = std::decay_t<decltype(typedKey)>;
        stream << "family="
               << stringifyNumericCommandFamily(typedKey.getFamily()) << '\n';
        if constexpr (std::is_same_v<KeyType, NumericCTConvertSemanticsKey>) {
          const TargetConvertRoute &route = typedKey.getCTConvertRoute();
          stream << "opcode=" << route.opcode << '\n'
                 << "route=" << route.canonicalSpelling << '\n'
                 << "source=" << stringifyLogicalFormat(route.source) << '\n'
                 << "destination=" << stringifyLogicalFormat(route.destination)
                 << '\n';
        } else if constexpr (std::is_same_v<KeyType,
                                            NumericCTElementwiseSemanticsKey>) {
          stream << "operation="
                 << stringifyNumericElementwiseOperation(
                        typedKey.getOperation())
                 << '\n'
                 << "input-format="
                 << stringifyLogicalFormat(typedKey.getInputFormat()) << '\n'
                 << "destination-format="
                 << stringifyLogicalFormat(typedKey.getDestinationFormat())
                 << '\n';
        } else if constexpr (std::is_same_v<KeyType,
                                            NumericNEGemmSemanticsKey>) {
          stream << "format=" << stringifyLogicalFormat(typedKey.getFormat())
                 << '\n';
        } else if constexpr (std::is_same_v<
                                 KeyType, NumericNativeCTReduceSemanticsKey>) {
          stream << "operation="
                 << stringifyNumericReduceOperation(typedKey.getOperation())
                 << '\n'
                 << "format=" << stringifyLogicalFormat(typedKey.getFormat())
                 << '\n';
        }
      },
      key);
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
         << static_cast<unsigned>(gemmReductionOrderPolicy) << '\n'
         << "reduction-accumulator-initialization="
         << static_cast<unsigned>(reductionAccumulatorInitializationPolicy)
         << '\n'
         << "reduction-order=" << static_cast<unsigned>(reductionOrderPolicy)
         << '\n';
  stream.flush();
  return digestCanonical(canonical);
}

} // namespace

llvm::ArrayRef<ModelProfileRecord> getRegisteredModelProfiles() {
  static const std::vector<ModelProfileRecord> profiles = [] {
    ModelProfileRecord record{
        kFormalDeterministic,
        "wafer-model-formal-deterministic",
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

llvm::StringRef stringifyFormalKernelKind(FormalKernelKind kind) {
  switch (kind) {
  case FormalKernelKind::Convert:
    return "convert";
  case FormalKernelKind::Elementwise:
    return "elementwise";
  case FormalKernelKind::Gemm:
    return "gemm";
  case FormalKernelKind::Reduce:
    return "reduce";
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
NumericCTConvertSemanticsKey::getCTConvertRoute() const {
  const TargetConvertRoute *route = findTargetConvertRoute(opcode);
  if (!route)
    llvm_unreachable("numeric route policy lost its CT convert route");
  return *route;
}

NumericCommandFamily NumericSemanticsProfile::getFamily() const {
  return std::visit([](const auto &typedKey) { return typedKey.getFamily(); },
                    key);
}

const NumericCTConvertSemanticsKey *
NumericSemanticsProfile::getCTConvertKey() const {
  return std::get_if<NumericCTConvertSemanticsKey>(&key);
}

const NumericCTElementwiseSemanticsKey *
NumericSemanticsProfile::getCTElementwiseKey() const {
  return std::get_if<NumericCTElementwiseSemanticsKey>(&key);
}

const NumericNEGemmSemanticsKey *NumericSemanticsProfile::getNEGemmKey() const {
  return std::get_if<NumericNEGemmSemanticsKey>(&key);
}

const NumericNativeCTReduceSemanticsKey *
NumericSemanticsProfile::getNativeCTReduceKey() const {
  return std::get_if<NumericNativeCTReduceSemanticsKey>(&key);
}

NumericRoundingMode NumericSemanticsProfile::getRoundingMode() const {
  const NumericCTConvertSemanticsKey *convert = getCTConvertKey();
  if (!convert || !roundingMode)
    llvm_unreachable("non-convert semantics requested convert rounding");
  return convert->getEffectiveRoundingMode();
}

llvm::ArrayRef<NumericSemanticsProfile>
getRegisteredNumericCTConvertSemanticsProfiles() {
  static const std::vector<NumericSemanticsProfile> profiles = [] {
    std::vector<NumericSemanticsProfile> result;
    result.reserve(101);
    size_t zeroPointRouteCount = 0;
    size_t parameterlessRouteCount = 0;
    size_t roundingRouteCount = 0;
    const ModelProfileRecord &model =
        getModelProfileRecord(kFormalDeterministic);

    for (const TargetConvertRoute &route : getTargetConvertRoutes()) {
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
        NumericSemanticsKey key =
            NumericCTConvertSemanticsKey(route.opcode, mode);
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
            model, key, mode, NumericRoundingPointPolicy::ConversionResult,
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
            NumericGemmReductionOrderPolicy::NotApplicable,
            NumericReductionAccumulatorInitializationPolicy::NotApplicable,
            NumericReductionOrderPolicy::NotApplicable);
        result.push_back(NumericSemanticsProfile(
            kFormalDeterministic, std::move(key), mode,
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
            NumericGemmReductionOrderPolicy::NotApplicable,
            NumericReductionAccumulatorInitializationPolicy::NotApplicable,
            NumericReductionOrderPolicy::NotApplicable, std::move(digest)));
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
        if (*result[index].getCTConvertKey() ==
            *result[other].getCTConvertKey())
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
        getModelProfileRecord(kFormalDeterministic);

    auto append = [&](NumericElementwiseOperation operation,
                      LogicalFormat inputFormat) {
      const LogicalFormat destinationFormat =
          getElementwiseDestinationFormat(operation, inputFormat);
      NumericSemanticsKey key = NumericCTElementwiseSemanticsKey(
          operation, inputFormat, destinationFormat);
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
          model, key, roundingMode, roundingPointPolicy,
          FloatToIntegerPolicy::NotApplicable, floatingNaNPolicy,
          floatingSignedZeroPolicy, floatingSubnormalPolicy,
          floatingTininessPolicy, exceptionFlagPolicy,
          floatingNaNSignalingPolicy, floatingDenormalModePolicy,
          floatingOverflowPolicy, saturationPolicy,
          transcendentalEvaluationPolicy,
          NumericGemmAccumulatorPolicy::NotApplicable,
          NumericGemmAccumulatorInitializationPolicy::NotApplicable,
          NumericGemmReductionOrderPolicy::NotApplicable,
          NumericReductionAccumulatorInitializationPolicy::NotApplicable,
          NumericReductionOrderPolicy::NotApplicable);
      result.push_back(NumericSemanticsProfile(
          kFormalDeterministic, std::move(key), roundingMode,
          roundingPointPolicy, FloatToIntegerPolicy::NotApplicable,
          floatingNaNPolicy, floatingSignedZeroPolicy, floatingSubnormalPolicy,
          floatingTininessPolicy, exceptionFlagPolicy,
          floatingNaNSignalingPolicy, floatingDenormalModePolicy,
          floatingOverflowPolicy, saturationPolicy,
          transcendentalEvaluationPolicy,
          NumericGemmAccumulatorPolicy::NotApplicable,
          NumericGemmAccumulatorInitializationPolicy::NotApplicable,
          NumericGemmReductionOrderPolicy::NotApplicable,
          NumericReductionAccumulatorInitializationPolicy::NotApplicable,
          NumericReductionOrderPolicy::NotApplicable, std::move(digest)));
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
        if (result[index].getSemanticsKey() == result[other].getSemanticsKey())
          llvm::report_fatal_error(
              "elementwise numeric semantics registry has a duplicate "
              "key");
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
        getModelProfileRecord(kFormalDeterministic);
    for (LogicalFormat format :
         getCompilerNumericFormats(TargetFormatEngine::NE)) {
      if (format == LogicalFormat::I8)
        continue;
      NumericSemanticsKey key = NumericNEGemmSemanticsKey(format);
      std::string digest = makeSemanticsDigest(
          model, key, NumericRoundingMode::NearestEven,
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
          NumericGemmReductionOrderPolicy::IncreasingK,
          NumericReductionAccumulatorInitializationPolicy::NotApplicable,
          NumericReductionOrderPolicy::NotApplicable);
      result.push_back(NumericSemanticsProfile(
          kFormalDeterministic, std::move(key),
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
          NumericGemmReductionOrderPolicy::IncreasingK,
          NumericReductionAccumulatorInitializationPolicy::NotApplicable,
          NumericReductionOrderPolicy::NotApplicable, std::move(digest)));
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
        if (result[index].getSemanticsKey() == result[other].getSemanticsKey())
          llvm::report_fatal_error(
              "GEMM numeric semantics registry has a duplicate key");
        if (result[index].getDigest() == result[other].getDigest())
          llvm::report_fatal_error(
              "GEMM numeric semantics registry has a duplicate digest");
      }
    }
    return result;
  }();
  return profiles;
}

llvm::ArrayRef<NumericSemanticsProfile>
getRegisteredNumericNativeCTReduceSemanticsProfiles() {
  static const std::vector<NumericSemanticsProfile> profiles = [] {
    const ModelProfileRecord &model =
        getModelProfileRecord(kFormalDeterministic);
    NumericSemanticsKey key = NumericNativeCTReduceSemanticsKey(
        NumericReduceOperation::Sum, LogicalFormat::F32);
    std::string digest = makeSemanticsDigest(
        model, key, NumericRoundingMode::NearestEven,
        NumericRoundingPointPolicy::ReductionStep,
        FloatToIntegerPolicy::NotApplicable,
        FloatingNaNPolicy::CanonicalPositiveQuietNaN,
        FloatingSignedZeroPolicy::ReductionPositiveZeroAccumulatorThenIEEE754,
        FloatingSubnormalPolicy::Gradual, FloatingTininessPolicy::AfterRounding,
        NumericExceptionFlagPolicy::ModelOnly,
        FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot,
        FloatingDenormalModePolicy::GradualNoDAZNoFTZ,
        FloatingOverflowPolicy::IEEE754AccordingToRoundingMode,
        NumericSaturationPolicy::Disabled,
        NumericTranscendentalEvaluationPolicy::NotApplicable,
        NumericGemmAccumulatorPolicy::NotApplicable,
        NumericGemmAccumulatorInitializationPolicy::NotApplicable,
        NumericGemmReductionOrderPolicy::NotApplicable,
        NumericReductionAccumulatorInitializationPolicy::PositiveZero,
        NumericReductionOrderPolicy::IncreasingLogicalRowMajorInputIndex);
    std::vector<NumericSemanticsProfile> result;
    result.push_back(NumericSemanticsProfile(
        kFormalDeterministic, std::move(key), NumericRoundingMode::NearestEven,
        NumericRoundingPointPolicy::ReductionStep,
        FloatToIntegerPolicy::NotApplicable,
        FloatingNaNPolicy::CanonicalPositiveQuietNaN,
        FloatingSignedZeroPolicy::ReductionPositiveZeroAccumulatorThenIEEE754,
        FloatingSubnormalPolicy::Gradual, FloatingTininessPolicy::AfterRounding,
        NumericExceptionFlagPolicy::ModelOnly,
        FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot,
        FloatingDenormalModePolicy::GradualNoDAZNoFTZ,
        FloatingOverflowPolicy::IEEE754AccordingToRoundingMode,
        NumericSaturationPolicy::Disabled,
        NumericTranscendentalEvaluationPolicy::NotApplicable,
        NumericGemmAccumulatorPolicy::NotApplicable,
        NumericGemmAccumulatorInitializationPolicy::NotApplicable,
        NumericGemmReductionOrderPolicy::NotApplicable,
        NumericReductionAccumulatorInitializationPolicy::PositiveZero,
        NumericReductionOrderPolicy::IncreasingLogicalRowMajorInputIndex,
        std::move(digest)));
    if (!isValidDigest(result.front().getDigest()) ||
        result.front().getFamily() != NumericCommandFamily::NativeCTReduce)
      llvm::report_fatal_error(
          "native CT reduction numeric semantics registry has an invalid row");
    return result;
  }();
  return profiles;
}

} // namespace wafer
