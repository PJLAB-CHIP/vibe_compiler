//===- FormalNumericValidation.cpp - Resolved command validation -----===//

#include "FormalNumericInternal.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::formal_detail {

static bool isLLVMElementwiseOperation(NumericElementwiseOperation operation) {
  switch (operation) {
  case NumericElementwiseOperation::Abs:
  case NumericElementwiseOperation::Recip:
  case NumericElementwiseOperation::Square:
  case NumericElementwiseOperation::Neg:
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
  case NumericElementwiseOperation::LogicNot:
  case NumericElementwiseOperation::LogicAnd:
  case NumericElementwiseOperation::LogicOr:
  case NumericElementwiseOperation::LogicXor:
  case NumericElementwiseOperation::Relu:
    return true;
  case NumericElementwiseOperation::Sqrt:
  case NumericElementwiseOperation::Rsqrt:
  case NumericElementwiseOperation::Log2:
  case NumericElementwiseOperation::Ln:
  case NumericElementwiseOperation::Pow2:
  case NumericElementwiseOperation::Exp:
  case NumericElementwiseOperation::ExpLp:
  case NumericElementwiseOperation::Sin:
  case NumericElementwiseOperation::Cos:
  case NumericElementwiseOperation::Tanh:
  case NumericElementwiseOperation::Sigmoid:
  case NumericElementwiseOperation::SatRelu:
  case NumericElementwiseOperation::LeakyRelu:
  case NumericElementwiseOperation::Softplus:
    return false;
  }
  return false;
}

static llvm::Error
validateResolvedExecution(const ResolvedNumericCommand &command,
                          NumericCommandFamily family, FormalKernelKind kernel,
                          FormalNumericBackendKind backend) {
  if (command.getFamily() != family || !command.isSupported() ||
      !command.getSemantics() || command.getFormalKernelKind() != kernel ||
      command.getComparatorKind() != NumericComparatorKind::RawExact ||
      command.getFormalBackendKind() != backend)
    return formalError(
        FormalNumericErrorCode::UnsupportedResolvedCommand,
        "the resolved command has no complete matching formal key");
  return llvm::Error::success();
}

llvm::Error
validateElementwiseResolvedCommand(const ResolvedNumericCommand &command) {
  if (llvm::Error error = validateResolvedExecution(
          command, NumericCommandFamily::CTElementwise,
          FormalKernelKind::Elementwise,
          FormalNumericBackendKind::LLVMAPFloatAPInt))
    return error;
  const NumericCTElementwiseCommand *elementwise =
      command.getCommandKey().getCTElementwise();
  const NumericSemanticsProfile &semantics = *command.getSemantics();
  const NumericCTElementwiseSemanticsKey *key = semantics.getCTElementwiseKey();
  if (!elementwise || !key ||
      semantics.getModelProfile() != command.getPattern().getModelProfile() ||
      key->getOperation() != elementwise->operation ||
      !isLLVMElementwiseOperation(elementwise->operation) ||
      elementwise->inputs.empty() ||
      key->getInputFormat() != elementwise->inputs.front().getFormat() ||
      key->getDestinationFormat() != elementwise->destination.getFormat() ||
      elementwise->inputs.size() !=
          getNumericElementwiseArity(elementwise->operation))
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "resolved elementwise semantics and exact key disagree");
  for (const NumericTensorKey &input : elementwise->inputs)
    if (input.getFormat() != key->getInputFormat())
      return formalError(
          FormalNumericErrorCode::UnsupportedResolvedCommand,
          "resolved elementwise key has heterogeneous input formats");

  const bool logic = isNumericElementwiseLogic(elementwise->operation);
  const bool relation = isNumericElementwiseRelation(elementwise->operation);
  if ((!logic && key->getInputFormat() != LogicalFormat::F16 &&
       key->getInputFormat() != LogicalFormat::BF16 &&
       key->getInputFormat() != LogicalFormat::F32) ||
      (logic && key->getInputFormat() != LogicalFormat::Bool) ||
      key->getDestinationFormat() !=
          (relation ? LogicalFormat::Bool : key->getInputFormat()))
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "elementwise format is outside the LLVM model subset");

  FloatingSignedZeroPolicy signedZeroPolicy =
      logic      ? FloatingSignedZeroPolicy::NotApplicable
      : relation ? FloatingSignedZeroPolicy::PredicateOnly
                 : FloatingSignedZeroPolicy::IEEE754OperationDefined;
  if (elementwise->operation == NumericElementwiseOperation::Max)
    signedZeroPolicy =
        FloatingSignedZeroPolicy::MaximumPositiveUnlessBothNegative;
  else if (elementwise->operation == NumericElementwiseOperation::Min)
    signedZeroPolicy =
        FloatingSignedZeroPolicy::MinimumNegativeUnlessBothPositive;
  const std::optional<NumericRoundingMode> roundingMode =
      logic || relation ? std::nullopt
                        : std::optional<NumericRoundingMode>(
                              NumericRoundingMode::NearestEven);
  if (semantics.getRoundingModePolicy() != roundingMode ||
      semantics.getRoundingPointPolicy() !=
          (logic || relation ? NumericRoundingPointPolicy::NotApplicable
                             : NumericRoundingPointPolicy::ElementwiseResult) ||
      semantics.getFloatToIntegerPolicy() !=
          FloatToIntegerPolicy::NotApplicable ||
      semantics.getFloatingNaNPolicy() !=
          (logic ? FloatingNaNPolicy::NotApplicable
           : relation
               ? FloatingNaNPolicy::OrderedRelationFalseExceptNotEqualTrue
               : FloatingNaNPolicy::CanonicalPositiveQuietNaN) ||
      semantics.getFloatingSignedZeroPolicy() != signedZeroPolicy ||
      semantics.getFloatingSubnormalPolicy() !=
          (logic ? FloatingSubnormalPolicy::NotApplicable
                 : FloatingSubnormalPolicy::Gradual) ||
      semantics.getFloatingTininessPolicy() !=
          (logic || relation ? FloatingTininessPolicy::NotApplicable
                             : FloatingTininessPolicy::AfterRounding) ||
      semantics.getExceptionFlagPolicy() !=
          (logic ? NumericExceptionFlagPolicy::NotApplicable
                 : NumericExceptionFlagPolicy::ModelOnly) ||
      semantics.getFloatingNaNSignalingPolicy() !=
          (logic ? FloatingNaNSignalingPolicy::NotApplicable
                 : FloatingNaNSignalingPolicy::
                       SignalingRaisesInvalidQuietDoesNot) ||
      semantics.getFloatingDenormalModePolicy() !=
          (logic ? FloatingDenormalModePolicy::NotApplicable
                 : FloatingDenormalModePolicy::GradualNoDAZNoFTZ) ||
      semantics.getFloatingOverflowPolicy() !=
          (logic || relation
               ? FloatingOverflowPolicy::NotApplicable
               : FloatingOverflowPolicy::IEEE754AccordingToRoundingMode) ||
      semantics.getSaturationPolicy() !=
          (logic || relation ? NumericSaturationPolicy::NotApplicable
                             : NumericSaturationPolicy::Disabled) ||
      semantics.getTranscendentalEvaluationPolicy() !=
          NumericTranscendentalEvaluationPolicy::NotApplicable ||
      semantics.getGemmAccumulatorPolicy() !=
          NumericGemmAccumulatorPolicy::NotApplicable ||
      semantics.getGemmAccumulatorInitializationPolicy() !=
          NumericGemmAccumulatorInitializationPolicy::NotApplicable ||
      semantics.getGemmReductionOrderPolicy() !=
          NumericGemmReductionOrderPolicy::NotApplicable)
    return formalError(
        FormalNumericErrorCode::UnsupportedResolvedCommand,
        "resolved elementwise semantics has an incompatible numeric policy");
  return llvm::Error::success();
}

llvm::Error validateGemmResolvedCommand(const ResolvedNumericCommand &command) {
  if (llvm::Error error = validateResolvedExecution(
          command, NumericCommandFamily::NEGemm, FormalKernelKind::Gemm,
          FormalNumericBackendKind::LLVMAPFloatAPInt))
    return error;
  const NumericNEGemmCommand *gemm = command.getCommandKey().getNEGemm();
  const NumericSemanticsProfile &semantics = *command.getSemantics();
  const NumericNEGemmSemanticsKey *key = semantics.getNEGemmKey();
  if (!gemm || !key ||
      semantics.getModelProfile() != command.getPattern().getModelProfile() ||
      key->getFormat() != gemm->lhs.getFormat() ||
      gemm->rhs.getFormat() != key->getFormat() ||
      gemm->destination.getFormat() != key->getFormat() ||
      (key->getFormat() != LogicalFormat::F16 &&
       key->getFormat() != LogicalFormat::BF16 &&
       key->getFormat() != LogicalFormat::F32) ||
      semantics.getRoundingModePolicy() != NumericRoundingMode::NearestEven ||
      semantics.getRoundingPointPolicy() !=
          NumericRoundingPointPolicy::GemmFusedMultiplyAddAndDestination ||
      semantics.getFloatToIntegerPolicy() !=
          FloatToIntegerPolicy::NotApplicable ||
      semantics.getFloatingNaNPolicy() !=
          FloatingNaNPolicy::CanonicalPositiveQuietNaN ||
      semantics.getFloatingSignedZeroPolicy() !=
          FloatingSignedZeroPolicy::GemmPositiveZeroAccumulatorThenIEEE754 ||
      semantics.getFloatingSubnormalPolicy() !=
          FloatingSubnormalPolicy::Gradual ||
      semantics.getFloatingTininessPolicy() !=
          FloatingTininessPolicy::AfterRounding ||
      semantics.getExceptionFlagPolicy() !=
          NumericExceptionFlagPolicy::ModelOnly ||
      semantics.getFloatingNaNSignalingPolicy() !=
          FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot ||
      semantics.getFloatingDenormalModePolicy() !=
          FloatingDenormalModePolicy::GradualNoDAZNoFTZ ||
      semantics.getFloatingOverflowPolicy() !=
          FloatingOverflowPolicy::IEEE754AccordingToRoundingMode ||
      semantics.getSaturationPolicy() != NumericSaturationPolicy::Disabled ||
      semantics.getTranscendentalEvaluationPolicy() !=
          NumericTranscendentalEvaluationPolicy::NotApplicable ||
      semantics.getGemmAccumulatorPolicy() !=
          NumericGemmAccumulatorPolicy::F32FusedMultiplyAdd ||
      semantics.getGemmAccumulatorInitializationPolicy() !=
          NumericGemmAccumulatorInitializationPolicy::PositiveZero ||
      semantics.getGemmReductionOrderPolicy() !=
          NumericGemmReductionOrderPolicy::IncreasingK)
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "resolved GEMM semantics and exact key disagree");
  return llvm::Error::success();
}

llvm::Error
validateReduceResolvedCommand(const ResolvedNumericCommand &command) {
  if (llvm::Error error = validateResolvedExecution(
          command, NumericCommandFamily::NativeCTReduce,
          FormalKernelKind::Reduce, FormalNumericBackendKind::LLVMAPFloatAPInt))
    return error;
  const NumericNativeCTReduceCommand *reduce =
      command.getCommandKey().getNativeCTReduce();
  const NumericSemanticsProfile &semantics = *command.getSemantics();
  const NumericNativeCTReduceSemanticsKey *key =
      semantics.getNativeCTReduceKey();
  if (!reduce || !key ||
      semantics.getModelProfile() != command.getPattern().getModelProfile() ||
      key->getOperation() != NumericReduceOperation::Sum ||
      reduce->operation != key->getOperation() ||
      key->getFormat() != LogicalFormat::F32 ||
      reduce->input.getFormat() != key->getFormat() ||
      reduce->destination.getFormat() != key->getFormat() ||
      semantics.getRoundingModePolicy() != NumericRoundingMode::NearestEven ||
      semantics.getRoundingPointPolicy() !=
          NumericRoundingPointPolicy::ReductionStep ||
      semantics.getFloatToIntegerPolicy() !=
          FloatToIntegerPolicy::NotApplicable ||
      semantics.getFloatingNaNPolicy() !=
          FloatingNaNPolicy::CanonicalPositiveQuietNaN ||
      semantics.getFloatingSignedZeroPolicy() !=
          FloatingSignedZeroPolicy::
              ReductionPositiveZeroAccumulatorThenIEEE754 ||
      semantics.getFloatingSubnormalPolicy() !=
          FloatingSubnormalPolicy::Gradual ||
      semantics.getFloatingTininessPolicy() !=
          FloatingTininessPolicy::AfterRounding ||
      semantics.getExceptionFlagPolicy() !=
          NumericExceptionFlagPolicy::ModelOnly ||
      semantics.getFloatingNaNSignalingPolicy() !=
          FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot ||
      semantics.getFloatingDenormalModePolicy() !=
          FloatingDenormalModePolicy::GradualNoDAZNoFTZ ||
      semantics.getFloatingOverflowPolicy() !=
          FloatingOverflowPolicy::IEEE754AccordingToRoundingMode ||
      semantics.getSaturationPolicy() != NumericSaturationPolicy::Disabled ||
      semantics.getTranscendentalEvaluationPolicy() !=
          NumericTranscendentalEvaluationPolicy::NotApplicable ||
      semantics.getGemmAccumulatorPolicy() !=
          NumericGemmAccumulatorPolicy::NotApplicable ||
      semantics.getGemmAccumulatorInitializationPolicy() !=
          NumericGemmAccumulatorInitializationPolicy::NotApplicable ||
      semantics.getGemmReductionOrderPolicy() !=
          NumericGemmReductionOrderPolicy::NotApplicable ||
      semantics.getReductionAccumulatorInitializationPolicy() !=
          NumericReductionAccumulatorInitializationPolicy::PositiveZero ||
      semantics.getReductionOrderPolicy() !=
          NumericReductionOrderPolicy::IncreasingLogicalRowMajorInputIndex)
    return formalError(FormalNumericErrorCode::UnsupportedResolvedCommand,
                       "resolved native reduction semantics and exact key "
                       "disagree");
  return llvm::Error::success();
}

} // namespace wafer::formal_detail
