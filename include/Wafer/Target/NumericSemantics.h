//===- NumericSemantics.h - Numeric capability registry --------*- C++ -*-===//

#ifndef WAFER_TARGET_NUMERICSEMANTICS_H
#define WAFER_TARGET_NUMERICSEMANTICS_H

#include "Wafer/Target/NumericCodec.h"
#include "Wafer/Target/PhysicalLayout.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace wafer {

/// Closed identifier for one explicitly selected model-only semantics profile.
/// It deliberately has no default or unknown value and does not affect
/// compiler target legality.
class ModelProfileId {
public:
  ModelProfileId() = delete;

  static constexpr ModelProfileId formalDeterministic() {
    return ModelProfileId(Value::FormalDeterministic);
  }

  friend constexpr bool operator==(ModelProfileId lhs, ModelProfileId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(ModelProfileId lhs, ModelProfileId rhs) {
    return !(lhs == rhs);
  }

private:
  enum class Value : uint8_t { FormalDeterministic };

  explicit constexpr ModelProfileId(Value value) : value(value) {}

  Value value;
};

struct ModelProfileRecord {
  ModelProfileId id;
  llvm::StringLiteral canonicalSpelling;
  LogicalScalarCodecPolicy numericDecodePolicy;
  LogicalScalarCodecPolicy numericEncodePolicy;
  /// Stable digest of every model-wide policy field in this record.
  std::string policyDigest;
  /// This profile is a deterministic formal model candidate, not hardware
  /// evidence for any target revision.
  bool modelOnly;
};

llvm::ArrayRef<ModelProfileRecord> getRegisteredModelProfiles();
llvm::Expected<ModelProfileId>
parseModelProfileId(llvm::StringRef canonicalSpelling);
const ModelProfileRecord &getModelProfileRecord(ModelProfileId id);
llvm::StringRef stringifyModelProfileId(ModelProfileId id);

/// Validated logical tensor key: format, the shared Wafer physical layout
/// family, complete static shape, checked element count and a digest covering
/// all four facts. The key carries only the pure target layout enum; physical
/// geometry remains derived from dtype and shape by the shared calculator.
class NumericTensorKey {
public:
  NumericTensorKey() = delete;

  static llvm::Expected<NumericTensorKey> create(LogicalFormat format,
                                                 PhysicalTensorLayout layout,
                                                 std::vector<uint64_t> shape);

  LogicalFormat getFormat() const { return format; }
  PhysicalTensorLayout getLayout() const { return layout; }
  llvm::ArrayRef<uint64_t> getShape() const { return shape; }
  uint64_t getElementCount() const { return elementCount; }
  llvm::StringRef getDigest() const { return tensorDigest; }

  friend bool operator==(const NumericTensorKey &lhs,
                         const NumericTensorKey &rhs) {
    return lhs.format == rhs.format && lhs.layout == rhs.layout &&
           lhs.shape == rhs.shape && lhs.elementCount == rhs.elementCount &&
           lhs.tensorDigest == rhs.tensorDigest;
  }
  friend bool operator!=(const NumericTensorKey &lhs,
                         const NumericTensorKey &rhs) {
    return !(lhs == rhs);
  }

private:
  NumericTensorKey(LogicalFormat format, PhysicalTensorLayout layout,
                   std::vector<uint64_t> shape, uint64_t elementCount,
                   std::string tensorDigest)
      : format(format), layout(layout), shape(std::move(shape)),
        elementCount(elementCount), tensorDigest(std::move(tensorDigest)) {}

  LogicalFormat format;
  PhysicalTensorLayout layout;
  std::vector<uint64_t> shape;
  uint64_t elementCount;
  std::string tensorDigest;
};

/// Typed values of the target RND_MODE field. Numeric ordinals are part of the
/// target command contract; mode 4 remains a typed exact command value even
/// though the first model profile resolves it to an unsupported pattern.
enum class NumericRoundingMode : uint8_t {
  NearestEven = 0,
  TowardZero = 1,
  TowardPositive = 2,
  TowardNegative = 3,
  Stochastic = 4,
};

llvm::ArrayRef<NumericRoundingMode> getNumericRoundingModes();
llvm::Expected<NumericRoundingMode> parseNumericRoundingMode(uint8_t value);
llvm::StringRef stringifyNumericRoundingMode(NumericRoundingMode mode);

/// A present CT-convert optional parameter. Parameterless routes use
/// std::nullopt; a present value cannot be untyped.
class NumericConvertParameter {
public:
  enum class Kind : uint8_t { RoundingMode, ZeroPoint };

  NumericConvertParameter() = delete;

  static constexpr NumericConvertParameter
  roundingMode(NumericRoundingMode mode) {
    return NumericConvertParameter(Kind::RoundingMode,
                                   static_cast<uint32_t>(mode));
  }
  static constexpr NumericConvertParameter zeroPoint(uint32_t value) {
    return NumericConvertParameter(Kind::ZeroPoint, value);
  }

  Kind getKind() const { return kind; }
  std::optional<NumericRoundingMode> getRoundingMode() const;
  std::optional<uint32_t> getZeroPoint() const;

  friend constexpr bool operator==(NumericConvertParameter lhs,
                                   NumericConvertParameter rhs) {
    return lhs.kind == rhs.kind && lhs.payload == rhs.payload;
  }
  friend constexpr bool operator!=(NumericConvertParameter lhs,
                                   NumericConvertParameter rhs) {
    return !(lhs == rhs);
  }

private:
  constexpr NumericConvertParameter(Kind kind, uint32_t payload)
      : kind(kind), payload(payload) {}

  Kind kind;
  uint32_t payload;
};

/// Target-independent semantic kinds accepted by terminal CT elementwise
/// commands. Ordinals are registry implementation details and have no target
/// ABI meaning.
enum class NumericElementwiseOperation : uint8_t {
  Abs,
  Recip,
  Square,
  Sqrt,
  Rsqrt,
  Neg,
  Max,
  Min,
  Add,
  Sub,
  Mul,
  Div,
  Eq,
  Ne,
  Ge,
  Gt,
  Le,
  Lt,
  LogicNot,
  LogicAnd,
  LogicOr,
  LogicXor,
  Log2,
  Ln,
  Pow2,
  Exp,
  ExpLp,
  Sin,
  Cos,
  Tanh,
  Sigmoid,
  Relu,
  SatRelu,
  LeakyRelu,
  Softplus,
};

llvm::ArrayRef<NumericElementwiseOperation> getNumericElementwiseOperations();
llvm::StringRef
stringifyNumericElementwiseOperation(NumericElementwiseOperation operation);
unsigned getNumericElementwiseArity(NumericElementwiseOperation operation);
bool isNumericElementwiseRelation(NumericElementwiseOperation operation);
bool isNumericElementwiseLogic(NumericElementwiseOperation operation);

/// Target-independent semantic kinds accepted by native CT reduction.
/// Ordinals have no target ABI meaning.
enum class NumericReduceOperation : uint8_t { Sum, Max, Min, Avg };

llvm::ArrayRef<NumericReduceOperation> getNumericReduceOperations();
llvm::StringRef stringifyNumericReduceOperation(NumericReduceOperation kind);

/// Typed target reduction dimension selector. Names describe logical trailing
/// dimensions rather than importing historical frontend axis names.
enum class NativeCTReduceDimension : uint8_t {
  Trailing0 = 0,
  Trailing1 = 1,
  Trailing2 = 2,
  Trailing3 = 3,
  Trailing2And1 = 4,
  Trailing2And1And0 = 5,
};

llvm::StringRef
stringifyNativeCTReduceDimension(NativeCTReduceDimension dimension);
std::vector<size_t>
getNativeCTReduceLogicalDimensions(NativeCTReduceDimension dimension,
                                   size_t rank);

enum class NumericCommandFamily : uint8_t {
  CTConvert,
  CTElementwise,
  NEGemm,
  NativeCTReduce,
};

llvm::StringRef stringifyNumericCommandFamily(NumericCommandFamily family);

struct NumericCTConvertCommand {
  uint16_t opcode;
  NumericTensorKey source;
  NumericTensorKey destination;
  std::optional<NumericConvertParameter> parameter;

  friend bool operator==(const NumericCTConvertCommand &lhs,
                         const NumericCTConvertCommand &rhs) {
    return lhs.opcode == rhs.opcode && lhs.source == rhs.source &&
           lhs.destination == rhs.destination && lhs.parameter == rhs.parameter;
  }
};

struct NumericCTElementwiseCommand {
  NumericElementwiseOperation operation;
  std::vector<NumericTensorKey> inputs;
  NumericTensorKey destination;

  friend bool operator==(const NumericCTElementwiseCommand &lhs,
                         const NumericCTElementwiseCommand &rhs) {
    return lhs.operation == rhs.operation && lhs.inputs == rhs.inputs &&
           lhs.destination == rhs.destination;
  }
};

/// Exact GEMM dimension mapping. Factories accept the mapping explicitly and
/// reject every non-canonical permutation; this keeps the command key
/// faithful to the terminal instruction fields without admitting alternate
/// predicate paths in capability matching.
struct NumericGemmAxes {
  std::vector<uint64_t> lhsBatchDimensions;
  uint64_t lhsMDimension = 0;
  uint64_t lhsContractingDimension = 0;
  std::vector<uint64_t> rhsBatchDimensions;
  uint64_t rhsContractingDimension = 0;
  uint64_t rhsNDimension = 0;
  std::vector<uint64_t> destinationBatchDimensions;
  uint64_t destinationMDimension = 0;
  uint64_t destinationNDimension = 0;

  friend bool operator==(const NumericGemmAxes &lhs,
                         const NumericGemmAxes &rhs) {
    return lhs.lhsBatchDimensions == rhs.lhsBatchDimensions &&
           lhs.lhsMDimension == rhs.lhsMDimension &&
           lhs.lhsContractingDimension == rhs.lhsContractingDimension &&
           lhs.rhsBatchDimensions == rhs.rhsBatchDimensions &&
           lhs.rhsContractingDimension == rhs.rhsContractingDimension &&
           lhs.rhsNDimension == rhs.rhsNDimension &&
           lhs.destinationBatchDimensions == rhs.destinationBatchDimensions &&
           lhs.destinationMDimension == rhs.destinationMDimension &&
           lhs.destinationNDimension == rhs.destinationNDimension;
  }
};

llvm::Expected<NumericGemmAxes> getCanonicalNumericGemmAxes(uint64_t rank);

struct NumericNEGemmCommand {
  NumericTensorKey lhs;
  NumericTensorKey rhs;
  NumericTensorKey destination;
  uint16_t m;
  uint16_t k;
  uint16_t n;
  uint16_t batchCount;
  NumericGemmAxes axes;
  TargetGemmOrientation lhsOrientation;
  TargetGemmOrientation rhsOrientation;

  friend bool operator==(const NumericNEGemmCommand &lhs,
                         const NumericNEGemmCommand &rhs) {
    return lhs.lhs == rhs.lhs && lhs.rhs == rhs.rhs &&
           lhs.destination == rhs.destination && lhs.m == rhs.m &&
           lhs.k == rhs.k && lhs.n == rhs.n &&
           lhs.batchCount == rhs.batchCount && lhs.axes == rhs.axes &&
           lhs.lhsOrientation == rhs.lhsOrientation &&
           lhs.rhsOrientation == rhs.rhsOrientation;
  }
};

struct NumericNativeCTReduceCommand {
  NumericReduceOperation operation;
  NumericTensorKey input;
  NumericTensorKey destination;
  NativeCTReduceDimension dimension;

  friend bool operator==(const NumericNativeCTReduceCommand &lhs,
                         const NumericNativeCTReduceCommand &rhs) {
    return lhs.operation == rhs.operation && lhs.input == rhs.input &&
           lhs.destination == rhs.destination && lhs.dimension == rhs.dimension;
  }
};

using NumericCommandPayload =
    std::variant<NumericCTConvertCommand, NumericCTElementwiseCommand,
                 NumericNEGemmCommand, NumericNativeCTReduceCommand>;

/// Validated exact command key. Shapes, layouts, parameters and all
/// family-specific fields remain exact here; capability wildcarding belongs to
/// the separate finite pattern registry.
class NumericCommandKey {
public:
  NumericCommandKey() = delete;

  static llvm::Expected<NumericCommandKey>
  createCTConvert(uint16_t opcode, NumericTensorKey source,
                  NumericTensorKey destination,
                  std::optional<NumericConvertParameter> parameter);
  static llvm::Expected<NumericCommandKey>
  createCTElementwise(NumericElementwiseOperation operation,
                      std::vector<NumericTensorKey> inputs,
                      NumericTensorKey destination);
  static llvm::Expected<NumericCommandKey> createNEGemm(
      NumericTensorKey lhs, NumericTensorKey rhs, NumericTensorKey destination,
      uint32_t m, uint32_t k, uint32_t n, uint32_t batchCount,
      NumericGemmAxes axes,
      TargetGemmOrientation lhsOrientation = TargetGemmOrientation::Normal,
      TargetGemmOrientation rhsOrientation = TargetGemmOrientation::Normal);
  static llvm::Expected<NumericCommandKey>
  createNativeCTReduce(NumericReduceOperation operation, NumericTensorKey input,
                       NumericTensorKey destination,
                       NativeCTReduceDimension dimension);

  NumericCommandFamily getFamily() const;
  const NumericCommandPayload &getPayload() const { return payload; }
  const NumericCTConvertCommand *getCTConvert() const;
  const NumericCTElementwiseCommand *getCTElementwise() const;
  const NumericNEGemmCommand *getNEGemm() const;
  const NumericNativeCTReduceCommand *getNativeCTReduce() const;
  llvm::StringRef getDigest() const { return keyDigest; }

  friend bool operator==(const NumericCommandKey &lhs,
                         const NumericCommandKey &rhs) {
    return lhs.payload == rhs.payload && lhs.keyDigest == rhs.keyDigest;
  }
  friend bool operator!=(const NumericCommandKey &lhs,
                         const NumericCommandKey &rhs) {
    return !(lhs == rhs);
  }

private:
  NumericCommandKey(NumericCommandPayload payload, std::string keyDigest)
      : payload(std::move(payload)), keyDigest(std::move(keyDigest)) {}

  NumericCommandPayload payload;
  std::string keyDigest;
};

enum class NumericComparatorKind : uint8_t { RawExact };
enum class FormalKernelKind : uint8_t { Convert, Elementwise, Gemm, Reduce };
enum class FormalNumericBackendKind : uint8_t { LLVMAPFloatAPInt, MPFR };

llvm::StringRef stringifyFormalKernelKind(FormalKernelKind kind);
llvm::StringRef
stringifyFormalNumericBackendKind(FormalNumericBackendKind kind);

enum class FloatToIntegerPolicy : uint8_t {
  NotApplicable,
  FiniteInRangeRejectNaNInfOverflowNoWrite,
};
enum class FloatingNaNPolicy : uint8_t {
  NotApplicable,
  CanonicalPositiveQuietNaN,
  OrderedRelationFalseExceptNotEqualTrue,
};
enum class FloatingSubnormalPolicy : uint8_t { NotApplicable, Gradual };
enum class FloatingTininessPolicy : uint8_t { NotApplicable, AfterRounding };
enum class NumericExceptionFlagPolicy : uint8_t { NotApplicable, ModelOnly };
enum class FloatingNaNSignalingPolicy : uint8_t {
  NotApplicable,
  SignalingRaisesInvalidQuietDoesNot,
};
enum class FloatingDenormalModePolicy : uint8_t {
  NotApplicable,
  GradualNoDAZNoFTZ,
};
enum class FloatingOverflowPolicy : uint8_t {
  NotApplicable,
  IEEE754AccordingToRoundingMode,
};
enum class NumericSaturationPolicy : uint8_t {
  NotApplicable,
  Disabled,
};
enum class NumericTranscendentalEvaluationPolicy : uint8_t {
  NotApplicable,
  CorrectlyRoundedMathematicalResultAdaptiveMPFRFinalRNE,
};
enum class NumericRoundingPointPolicy : uint8_t {
  NotApplicable,
  ConversionResult,
  ElementwiseResult,
  GemmFusedMultiplyAddAndDestination,
  ReductionStep,
};
enum class FloatingSignedZeroPolicy : uint8_t {
  NotApplicable,
  Preserve,
  IEEE754OperationDefined,
  MaximumPositiveUnlessBothNegative,
  MinimumNegativeUnlessBothPositive,
  PredicateOnly,
  GemmPositiveZeroAccumulatorThenIEEE754,
  ReductionPositiveZeroAccumulatorThenIEEE754,
};
enum class NumericGemmAccumulatorPolicy : uint8_t {
  NotApplicable,
  F32FusedMultiplyAdd,
};
enum class NumericGemmAccumulatorInitializationPolicy : uint8_t {
  NotApplicable,
  PositiveZero,
};
enum class NumericGemmReductionOrderPolicy : uint8_t {
  NotApplicable,
  IncreasingK,
};
enum class NumericReductionAccumulatorInitializationPolicy : uint8_t {
  NotApplicable,
  PositiveZero,
};
enum class NumericReductionOrderPolicy : uint8_t {
  NotApplicable,
  IncreasingLogicalRowMajorInputIndex,
};

/// Finite reusable key for one CT convert route and effective policy. It
/// intentionally contains no tensor or exact command parameter.
class NumericCTConvertSemanticsKey {
public:
  NumericCTConvertSemanticsKey() = delete;
  NumericCTConvertSemanticsKey(uint16_t opcode,
                               NumericRoundingMode effectiveRounding)
      : opcode(opcode), effectiveRounding(effectiveRounding) {}

  NumericCommandFamily getFamily() const {
    return NumericCommandFamily::CTConvert;
  }
  uint16_t getCTConvertOpcode() const { return opcode; }
  const TargetConvertRoute &getCTConvertRoute() const;
  NumericRoundingMode getEffectiveRoundingMode() const {
    return effectiveRounding;
  }

  friend bool operator==(const NumericCTConvertSemanticsKey &lhs,
                         const NumericCTConvertSemanticsKey &rhs) {
    return lhs.opcode == rhs.opcode &&
           lhs.effectiveRounding == rhs.effectiveRounding;
  }

private:
  uint16_t opcode;
  NumericRoundingMode effectiveRounding;
};

class NumericCTElementwiseSemanticsKey {
public:
  NumericCTElementwiseSemanticsKey() = delete;
  NumericCTElementwiseSemanticsKey(NumericElementwiseOperation operation,
                                   LogicalFormat inputFormat,
                                   LogicalFormat destinationFormat)
      : operation(operation), inputFormat(inputFormat),
        destinationFormat(destinationFormat) {}

  NumericCommandFamily getFamily() const {
    return NumericCommandFamily::CTElementwise;
  }
  NumericElementwiseOperation getOperation() const { return operation; }
  LogicalFormat getInputFormat() const { return inputFormat; }
  LogicalFormat getDestinationFormat() const { return destinationFormat; }

  friend bool operator==(const NumericCTElementwiseSemanticsKey &lhs,
                         const NumericCTElementwiseSemanticsKey &rhs) {
    return lhs.operation == rhs.operation &&
           lhs.inputFormat == rhs.inputFormat &&
           lhs.destinationFormat == rhs.destinationFormat;
  }

private:
  NumericElementwiseOperation operation;
  LogicalFormat inputFormat;
  LogicalFormat destinationFormat;
};

class NumericNEGemmSemanticsKey {
public:
  NumericNEGemmSemanticsKey() = delete;
  explicit NumericNEGemmSemanticsKey(LogicalFormat format) : format(format) {}

  NumericCommandFamily getFamily() const {
    return NumericCommandFamily::NEGemm;
  }
  LogicalFormat getFormat() const { return format; }

  friend bool operator==(const NumericNEGemmSemanticsKey &lhs,
                         const NumericNEGemmSemanticsKey &rhs) {
    return lhs.format == rhs.format;
  }

private:
  LogicalFormat format;
};

class NumericNativeCTReduceSemanticsKey {
public:
  NumericNativeCTReduceSemanticsKey() = delete;
  NumericNativeCTReduceSemanticsKey(NumericReduceOperation operation,
                                    LogicalFormat format)
      : operation(operation), format(format) {}

  NumericCommandFamily getFamily() const {
    return NumericCommandFamily::NativeCTReduce;
  }
  NumericReduceOperation getOperation() const { return operation; }
  LogicalFormat getFormat() const { return format; }

  friend bool operator==(const NumericNativeCTReduceSemanticsKey &lhs,
                         const NumericNativeCTReduceSemanticsKey &rhs) {
    return lhs.operation == rhs.operation && lhs.format == rhs.format;
  }

private:
  NumericReduceOperation operation;
  LogicalFormat format;
};

using NumericSemanticsKey =
    std::variant<NumericCTConvertSemanticsKey, NumericCTElementwiseSemanticsKey,
                 NumericNEGemmSemanticsKey, NumericNativeCTReduceSemanticsKey>;

/// Immutable reusable family-typed numeric semantics. Tensor shape/layout and
/// exact command parameters remain in NumericCommandKey.
class NumericSemanticsProfile {
public:
  NumericSemanticsProfile() = delete;

  ModelProfileId getModelProfile() const { return modelProfile; }
  NumericCommandFamily getFamily() const;
  const NumericSemanticsKey &getSemanticsKey() const { return key; }
  const NumericCTConvertSemanticsKey *getCTConvertKey() const;
  const NumericCTElementwiseSemanticsKey *getCTElementwiseKey() const;
  const NumericNEGemmSemanticsKey *getNEGemmKey() const;
  const NumericNativeCTReduceSemanticsKey *getNativeCTReduceKey() const;
  NumericRoundingMode getRoundingMode() const;
  std::optional<NumericRoundingMode> getRoundingModePolicy() const {
    return roundingMode;
  }
  NumericRoundingPointPolicy getRoundingPointPolicy() const {
    return roundingPointPolicy;
  }
  FloatToIntegerPolicy getFloatToIntegerPolicy() const {
    return floatToIntegerPolicy;
  }
  FloatingNaNPolicy getFloatingNaNPolicy() const { return floatingNaNPolicy; }
  FloatingSignedZeroPolicy getFloatingSignedZeroPolicy() const {
    return floatingSignedZeroPolicy;
  }
  FloatingSubnormalPolicy getFloatingSubnormalPolicy() const {
    return floatingSubnormalPolicy;
  }
  FloatingTininessPolicy getFloatingTininessPolicy() const {
    return floatingTininessPolicy;
  }
  NumericExceptionFlagPolicy getExceptionFlagPolicy() const {
    return exceptionFlagPolicy;
  }
  FloatingNaNSignalingPolicy getFloatingNaNSignalingPolicy() const {
    return floatingNaNSignalingPolicy;
  }
  FloatingDenormalModePolicy getFloatingDenormalModePolicy() const {
    return floatingDenormalModePolicy;
  }
  FloatingOverflowPolicy getFloatingOverflowPolicy() const {
    return floatingOverflowPolicy;
  }
  NumericSaturationPolicy getSaturationPolicy() const {
    return saturationPolicy;
  }
  NumericTranscendentalEvaluationPolicy
  getTranscendentalEvaluationPolicy() const {
    return transcendentalEvaluationPolicy;
  }
  NumericGemmAccumulatorPolicy getGemmAccumulatorPolicy() const {
    return gemmAccumulatorPolicy;
  }
  NumericGemmAccumulatorInitializationPolicy
  getGemmAccumulatorInitializationPolicy() const {
    return gemmAccumulatorInitializationPolicy;
  }
  NumericGemmReductionOrderPolicy getGemmReductionOrderPolicy() const {
    return gemmReductionOrderPolicy;
  }
  NumericReductionAccumulatorInitializationPolicy
  getReductionAccumulatorInitializationPolicy() const {
    return reductionAccumulatorInitializationPolicy;
  }
  NumericReductionOrderPolicy getReductionOrderPolicy() const {
    return reductionOrderPolicy;
  }
  llvm::StringRef getDigest() const { return semanticDigest; }

private:
  NumericSemanticsProfile(
      ModelProfileId modelProfile, NumericSemanticsKey key,
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
      NumericReductionOrderPolicy reductionOrderPolicy,
      std::string semanticDigest)
      : modelProfile(modelProfile), key(std::move(key)),
        roundingMode(roundingMode), roundingPointPolicy(roundingPointPolicy),
        floatToIntegerPolicy(floatToIntegerPolicy),
        floatingNaNPolicy(floatingNaNPolicy),
        floatingSignedZeroPolicy(floatingSignedZeroPolicy),
        floatingSubnormalPolicy(floatingSubnormalPolicy),
        floatingTininessPolicy(floatingTininessPolicy),
        exceptionFlagPolicy(exceptionFlagPolicy),
        floatingNaNSignalingPolicy(floatingNaNSignalingPolicy),
        floatingDenormalModePolicy(floatingDenormalModePolicy),
        floatingOverflowPolicy(floatingOverflowPolicy),
        saturationPolicy(saturationPolicy),
        transcendentalEvaluationPolicy(transcendentalEvaluationPolicy),
        gemmAccumulatorPolicy(gemmAccumulatorPolicy),
        gemmAccumulatorInitializationPolicy(
            gemmAccumulatorInitializationPolicy),
        gemmReductionOrderPolicy(gemmReductionOrderPolicy),
        reductionAccumulatorInitializationPolicy(
            reductionAccumulatorInitializationPolicy),
        reductionOrderPolicy(reductionOrderPolicy),
        semanticDigest(std::move(semanticDigest)) {}

  ModelProfileId modelProfile;
  NumericSemanticsKey key;
  std::optional<NumericRoundingMode> roundingMode;
  NumericRoundingPointPolicy roundingPointPolicy;
  FloatToIntegerPolicy floatToIntegerPolicy;
  FloatingNaNPolicy floatingNaNPolicy;
  FloatingSignedZeroPolicy floatingSignedZeroPolicy;
  FloatingSubnormalPolicy floatingSubnormalPolicy;
  FloatingTininessPolicy floatingTininessPolicy;
  NumericExceptionFlagPolicy exceptionFlagPolicy;
  FloatingNaNSignalingPolicy floatingNaNSignalingPolicy;
  FloatingDenormalModePolicy floatingDenormalModePolicy;
  FloatingOverflowPolicy floatingOverflowPolicy;
  NumericSaturationPolicy saturationPolicy;
  NumericTranscendentalEvaluationPolicy transcendentalEvaluationPolicy;
  NumericGemmAccumulatorPolicy gemmAccumulatorPolicy;
  NumericGemmAccumulatorInitializationPolicy
      gemmAccumulatorInitializationPolicy;
  NumericGemmReductionOrderPolicy gemmReductionOrderPolicy;
  NumericReductionAccumulatorInitializationPolicy
      reductionAccumulatorInitializationPolicy;
  NumericReductionOrderPolicy reductionOrderPolicy;
  std::string semanticDigest;

  friend llvm::ArrayRef<NumericSemanticsProfile>
  getRegisteredNumericCTConvertSemanticsProfiles();
  friend llvm::ArrayRef<NumericSemanticsProfile>
  getRegisteredNumericCTElementwiseSemanticsProfiles();
  friend llvm::ArrayRef<NumericSemanticsProfile>
  getRegisteredNumericNEGemmSemanticsProfiles();
  friend llvm::ArrayRef<NumericSemanticsProfile>
  getRegisteredNumericNativeCTReduceSemanticsProfiles();
};

llvm::ArrayRef<NumericSemanticsProfile>
getRegisteredNumericCTConvertSemanticsProfiles();
llvm::ArrayRef<NumericSemanticsProfile>
getRegisteredNumericCTElementwiseSemanticsProfiles();
llvm::ArrayRef<NumericSemanticsProfile>
getRegisteredNumericNEGemmSemanticsProfiles();
llvm::ArrayRef<NumericSemanticsProfile>
getRegisteredNumericNativeCTReduceSemanticsProfiles();

enum class NumericCapabilitySupport : uint8_t { Unsupported, Supported };
enum class NumericModelImplementationStatus : uint8_t { Absent, Implemented };
enum class NumericModelImplementationReason : uint8_t {
  None,
  StochasticStateUnproven,
  ZeroPointFormulaUnproven,
  FormalKernelNotImplemented,
  IntegerElementwisePolicyUnproven,
  ExpLpParameterPolicyUnproven,
  SatReluParameterPolicyUnproven,
  LeakyReluParameterPolicyUnproven,
  IntegerGemmAccumulatorPolicyUnproven,
  NativeReductionPolicyUnproven,
};
struct NumericModelImplementationCapability {
  NumericModelImplementationStatus status;
  NumericModelImplementationReason reason;
};

enum class NumericCompilerEmittabilityStatus : uint8_t {
  NotEmittable,
  Emittable,
};
enum class NumericCompilerEmittabilityReason : uint8_t { None };
struct NumericCompilerEmittabilityCapability {
  NumericCompilerEmittabilityStatus status;
  NumericCompilerEmittabilityReason reason;
};

enum class NumericEvidenceStatus : uint8_t { ModelOnlyUncorrelated };
enum class NumericEvidenceReason : uint8_t { HardwareCorrelationNotRun };
struct NumericEvidenceCapability {
  NumericEvidenceStatus status;
  NumericEvidenceReason reason;
};

llvm::StringRef stringifyNumericModelImplementationReason(
    NumericModelImplementationReason reason);
llvm::StringRef stringifyNumericCompilerEmittabilityReason(
    NumericCompilerEmittabilityReason reason);
llvm::StringRef stringifyNumericEvidenceReason(NumericEvidenceReason reason);

enum class NumericCapabilityParameterPatternKind : uint8_t {
  NoParameter,
  ExactRoundingMode,
  AnyZeroPoint,
};

class NumericCapabilityParameterPattern {
public:
  NumericCapabilityParameterPattern() = delete;

  static constexpr NumericCapabilityParameterPattern noParameter() {
    return NumericCapabilityParameterPattern(
        NumericCapabilityParameterPatternKind::NoParameter, std::nullopt);
  }
  static constexpr NumericCapabilityParameterPattern
  exactRoundingMode(NumericRoundingMode mode) {
    return NumericCapabilityParameterPattern(
        NumericCapabilityParameterPatternKind::ExactRoundingMode, mode);
  }
  static constexpr NumericCapabilityParameterPattern anyZeroPoint() {
    return NumericCapabilityParameterPattern(
        NumericCapabilityParameterPatternKind::AnyZeroPoint, std::nullopt);
  }

  NumericCapabilityParameterPatternKind getKind() const { return kind; }
  std::optional<NumericRoundingMode> getExactRoundingMode() const {
    return exactRoundingModeValue;
  }
  bool matches(const std::optional<NumericConvertParameter> &parameter) const;

  friend constexpr bool operator==(NumericCapabilityParameterPattern lhs,
                                   NumericCapabilityParameterPattern rhs) {
    return lhs.kind == rhs.kind &&
           lhs.exactRoundingModeValue == rhs.exactRoundingModeValue;
  }

private:
  constexpr NumericCapabilityParameterPattern(
      NumericCapabilityParameterPatternKind kind,
      std::optional<NumericRoundingMode> exactRoundingModeValue)
      : kind(kind), exactRoundingModeValue(exactRoundingModeValue) {}

  NumericCapabilityParameterPatternKind kind;
  std::optional<NumericRoundingMode> exactRoundingModeValue;
};

struct NumericCTConvertPatternSelector {
  uint16_t opcode;
  NumericCapabilityParameterPattern parameter;
  friend bool operator==(NumericCTConvertPatternSelector lhs,
                         NumericCTConvertPatternSelector rhs) {
    return lhs.opcode == rhs.opcode && lhs.parameter == rhs.parameter;
  }
};
struct NumericCTElementwisePatternSelector {
  NumericElementwiseOperation operation;
  LogicalFormat inputFormat;
  friend bool operator==(NumericCTElementwisePatternSelector lhs,
                         NumericCTElementwisePatternSelector rhs) {
    return lhs.operation == rhs.operation && lhs.inputFormat == rhs.inputFormat;
  }
};
struct NumericNEGemmPatternSelector {
  LogicalFormat format;
  friend bool operator==(NumericNEGemmPatternSelector lhs,
                         NumericNEGemmPatternSelector rhs) {
    return lhs.format == rhs.format;
  }
};
struct NumericNativeCTReducePatternSelector {
  NumericReduceOperation operation;
  LogicalFormat format;
  friend bool operator==(NumericNativeCTReducePatternSelector lhs,
                         NumericNativeCTReducePatternSelector rhs) {
    return lhs.operation == rhs.operation && lhs.format == rhs.format;
  }
};

using NumericCapabilitySelector = std::variant<
    NumericCTConvertPatternSelector, NumericCTElementwisePatternSelector,
    NumericNEGemmPatternSelector, NumericNativeCTReducePatternSelector>;

/// One closed dispatch pattern. Shapes, layouts and family parameters remain
/// in the exact key; this finite selector has no predicate priority.
class NumericCapabilityPattern {
public:
  NumericCapabilityPattern() = delete;

  ModelProfileId getModelProfile() const { return modelProfile; }
  NumericCommandFamily getFamily() const { return family; }
  const NumericCapabilitySelector &getSelector() const { return selector; }
  const NumericCTConvertPatternSelector *getCTConvertSelector() const;
  const NumericCTElementwisePatternSelector *getCTElementwiseSelector() const;
  const NumericNEGemmPatternSelector *getNEGemmSelector() const;
  const NumericNativeCTReducePatternSelector *getNativeCTReduceSelector() const;
  NumericCapabilitySupport getSupport() const {
    return modelCapability.status ==
                   NumericModelImplementationStatus::Implemented
               ? NumericCapabilitySupport::Supported
               : NumericCapabilitySupport::Unsupported;
  }
  bool isSupported() const {
    return getSupport() == NumericCapabilitySupport::Supported;
  }
  NumericModelImplementationCapability getModelCapability() const {
    return modelCapability;
  }
  NumericCompilerEmittabilityCapability getCompilerCapability() const {
    return compilerCapability;
  }
  NumericEvidenceCapability getEvidenceCapability() const {
    return evidenceCapability;
  }
  const NumericSemanticsProfile *getSemantics() const { return semantics; }
  std::optional<FormalKernelKind> getFormalKernelKind() const {
    return formalKernel;
  }
  std::optional<NumericComparatorKind> getComparatorKind() const {
    return comparator;
  }
  std::optional<FormalNumericBackendKind> getFormalBackendKind() const {
    return formalBackend;
  }
  llvm::StringRef getDigest() const { return patternDigest; }

  bool matches(ModelProfileId candidateModel,
               const NumericCommandKey &candidateKey) const;

private:
  NumericCapabilityPattern(
      ModelProfileId modelProfile, NumericCommandFamily family,
      NumericCapabilitySelector selector,
      NumericModelImplementationCapability modelCapability,
      NumericCompilerEmittabilityCapability compilerCapability,
      NumericEvidenceCapability evidenceCapability,
      const NumericSemanticsProfile *semantics,
      std::optional<FormalKernelKind> formalKernel,
      std::optional<NumericComparatorKind> comparator,
      std::optional<FormalNumericBackendKind> formalBackend,
      std::string patternDigest)
      : modelProfile(modelProfile), family(family),
        selector(std::move(selector)), modelCapability(modelCapability),
        compilerCapability(compilerCapability),
        evidenceCapability(evidenceCapability), semantics(semantics),
        formalKernel(formalKernel), comparator(comparator),
        formalBackend(formalBackend), patternDigest(std::move(patternDigest)) {}

  ModelProfileId modelProfile;
  NumericCommandFamily family;
  NumericCapabilitySelector selector;
  NumericModelImplementationCapability modelCapability;
  NumericCompilerEmittabilityCapability compilerCapability;
  NumericEvidenceCapability evidenceCapability;
  const NumericSemanticsProfile *semantics;
  std::optional<FormalKernelKind> formalKernel;
  std::optional<NumericComparatorKind> comparator;
  std::optional<FormalNumericBackendKind> formalBackend;
  std::string patternDigest;

  friend llvm::ArrayRef<NumericCapabilityPattern>
  getRegisteredNumericCapabilityPatterns();
};

llvm::ArrayRef<NumericCapabilityPattern>
getRegisteredNumericCapabilityPatterns();

/// Test-facing structural validator used to prove that a supplied registry has
/// neither overlapping selectors nor a missing member of the finite closure.
llvm::Error validateNumericCapabilityPatternsForTesting(
    llvm::ArrayRef<NumericCapabilityPattern> patterns);

/// Immutable result of resolving one exact key against exactly one finite
/// capability pattern. The exact key and selector pattern have separate
/// digests; resolutionDigest binds both.
class ResolvedNumericCommand {
public:
  ResolvedNumericCommand() = delete;

  const NumericCommandKey &getCommandKey() const { return commandKey; }
  const NumericCapabilityPattern &getPattern() const { return *pattern; }
  NumericCommandFamily getFamily() const { return commandKey.getFamily(); }
  bool isSupported() const { return pattern->isSupported(); }
  const NumericSemanticsProfile *getSemantics() const {
    return pattern->getSemantics();
  }
  std::optional<FormalKernelKind> getFormalKernelKind() const {
    return pattern->getFormalKernelKind();
  }
  std::optional<NumericComparatorKind> getComparatorKind() const {
    return pattern->getComparatorKind();
  }
  std::optional<FormalNumericBackendKind> getFormalBackendKind() const {
    return pattern->getFormalBackendKind();
  }
  llvm::StringRef getDigest() const { return resolutionDigest; }

private:
  ResolvedNumericCommand(NumericCommandKey commandKey,
                         const NumericCapabilityPattern *pattern,
                         std::string resolutionDigest)
      : commandKey(std::move(commandKey)), pattern(pattern),
        resolutionDigest(std::move(resolutionDigest)) {}

  NumericCommandKey commandKey;
  const NumericCapabilityPattern *pattern;
  std::string resolutionDigest;

  friend llvm::Expected<ResolvedNumericCommand>
  resolveNumericCommand(ModelProfileId modelProfile,
                        NumericCommandKey commandKey);
};

llvm::Expected<ResolvedNumericCommand>
resolveNumericCommand(ModelProfileId modelProfile,
                      NumericCommandKey commandKey);

} // namespace wafer

#endif // WAFER_TARGET_NUMERICSEMANTICS_H
