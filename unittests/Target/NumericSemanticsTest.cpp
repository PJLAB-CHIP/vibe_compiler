//===- NumericSemanticsTest.cpp - Numeric resolution closure tests -------===//

#include "Wafer/Target/NumericSemantics.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using wafer::FloatingDenormalModePolicy;
using wafer::FloatingNaNPolicy;
using wafer::FloatingNaNSignalingPolicy;
using wafer::FloatingOverflowPolicy;
using wafer::FloatingSignedZeroPolicy;
using wafer::FloatingSubnormalPolicy;
using wafer::FloatingTininessPolicy;
using wafer::FloatToIntegerPolicy;
using wafer::FormalKernelKind;
using wafer::FormalNumericBackendKind;
using wafer::ModelProfileId;
using wafer::NativeCTReduceDimension;
using wafer::NumericCapabilityParameterPatternKind;
using wafer::NumericCapabilitySupport;
using wafer::NumericCommandFamily;
using wafer::NumericCommandKey;
using wafer::NumericComparatorKind;
using wafer::NumericCompilerEmittabilityReason;
using wafer::NumericCompilerEmittabilityStatus;
using wafer::NumericConvertParameter;
using wafer::NumericElementwiseOperation;
using wafer::NumericEvidenceReason;
using wafer::NumericEvidenceStatus;
using wafer::NumericExceptionFlagPolicy;
using wafer::NumericGemmAccumulatorInitializationPolicy;
using wafer::NumericGemmAccumulatorPolicy;
using wafer::NumericGemmAxes;
using wafer::NumericGemmReductionOrderPolicy;
using wafer::NumericModelImplementationReason;
using wafer::NumericModelImplementationStatus;
using wafer::NumericReduceOperation;
using wafer::NumericRoundingMode;
using wafer::NumericRoundingPointPolicy;
using wafer::NumericSaturationPolicy;
using wafer::NumericTensorKey;
using wafer::NumericTensorLayout;
using wafer::NumericTranscendentalEvaluationPolicy;
using wafer::ResolvedNumericCommand;
using wafer::TargetConvertParameterKind;

constexpr wafer::TargetProfileId kTargetProfile =
    wafer::TargetProfileId::waferTx81SingleCardKernelV1();
constexpr ModelProfileId kModelProfile =
    ModelProfileId::formalDeterministicV1();
constexpr wafer::LogicalFormat kComputeFormats[] = {
    wafer::LogicalFormat::I8, wafer::LogicalFormat::F16,
    wafer::LogicalFormat::BF16, wafer::LogicalFormat::F32};

template <typename T, typename = void>
struct HasExactCommandKeyAccessor : std::false_type {};
template <typename T>
struct HasExactCommandKeyAccessor<
    T, std::void_t<decltype(std::declval<const T &>().getCommandKey())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasShapeLessConvertFactory : std::false_type {};
template <typename T>
struct HasShapeLessConvertFactory<
    T, std::void_t<decltype(T::createCTConvert(
           kTargetProfile, uint16_t{143},
           std::optional<NumericConvertParameter>{}))>> : std::true_type {};

static_assert(!std::is_default_constructible_v<ModelProfileId>);
static_assert(!std::is_default_constructible_v<NumericTensorKey>);
static_assert(!std::is_default_constructible_v<NumericCommandKey>);
static_assert(!std::is_default_constructible_v<wafer::NumericSemanticsProfile>);
static_assert(!std::is_default_constructible_v<ResolvedNumericCommand>);
static_assert(
    !HasExactCommandKeyAccessor<wafer::NumericSemanticsProfile>::value,
    "reusable semantics must not own an exact command key");
static_assert(!HasShapeLessConvertFactory<NumericCommandKey>::value,
              "shape-less convert factory must remain deleted");

bool isDigest(llvm::StringRef digest) {
  if (digest.size() != 71 || !digest.starts_with("sha256:"))
    return false;
  for (char character : digest.drop_front(7))
    if (!std::isdigit(static_cast<unsigned char>(character)) &&
        !(character >= 'a' && character <= 'f'))
      return false;
  return true;
}

template <typename T> std::string expectError(llvm::Expected<T> expected) {
  if (expected) {
    ADD_FAILURE() << "expected an LLVM Error";
    return {};
  }
  return llvm::toString(expected.takeError());
}

std::string expectError(llvm::Error error) {
  if (!error) {
    ADD_FAILURE() << "expected an LLVM Error";
    return {};
  }
  return llvm::toString(std::move(error));
}

NumericTensorKey
makeTensor(wafer::LogicalFormat format, NumericTensorLayout layout,
           std::vector<uint64_t> shape = std::vector<uint64_t>{2, 3}) {
  llvm::Expected<NumericTensorKey> tensor =
      NumericTensorKey::create(format, layout, std::move(shape));
  if (tensor)
    return std::move(*tensor);
  ADD_FAILURE() << llvm::toString(tensor.takeError());
  return llvm::cantFail(NumericTensorKey::create(
      wafer::LogicalFormat::F16, NumericTensorLayout::Tensor, {1}));
}

std::optional<NumericCommandKey> expectConvertKey(
    uint16_t opcode,
    std::optional<NumericConvertParameter> parameter = std::nullopt,
    std::vector<uint64_t> shape = {2, 3},
    NumericTensorLayout sourceLayout = NumericTensorLayout::Tensor,
    NumericTensorLayout destinationLayout = NumericTensorLayout::Tensor) {
  const wafer::TargetConvertRoute *route =
      wafer::findTargetConvertRoute(kTargetProfile, opcode);
  if (!route) {
    ADD_FAILURE() << "missing CT convert route " << opcode;
    return std::nullopt;
  }
  NumericTensorKey source = makeTensor(route->source, sourceLayout, shape);
  NumericTensorKey destination =
      makeTensor(route->destination, destinationLayout, shape);
  llvm::Expected<NumericCommandKey> key = NumericCommandKey::createCTConvert(
      kTargetProfile, opcode, std::move(source), std::move(destination),
      parameter);
  if (!key) {
    ADD_FAILURE() << llvm::toString(key.takeError());
    return std::nullopt;
  }
  return std::move(*key);
}

std::optional<ResolvedNumericCommand>
expectResolution(const NumericCommandKey &key) {
  llvm::Expected<ResolvedNumericCommand> resolution =
      wafer::resolveNumericCommand(kModelProfile, key);
  if (!resolution) {
    ADD_FAILURE() << llvm::toString(resolution.takeError());
    return std::nullopt;
  }
  return std::move(*resolution);
}

NumericGemmAxes canonicalAxes(uint64_t rank) {
  llvm::Expected<NumericGemmAxes> axes =
      wafer::getCanonicalNumericGemmAxes(rank);
  if (axes)
    return std::move(*axes);
  ADD_FAILURE() << llvm::toString(axes.takeError());
  return llvm::cantFail(wafer::getCanonicalNumericGemmAxes(2));
}

bool usesMPFR(NumericElementwiseOperation operation) {
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
expectedElementwiseUnsupportedReason(NumericElementwiseOperation operation,
                                     wafer::LogicalFormat format) {
  if (format == wafer::LogicalFormat::I8)
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

TEST(NumericSemanticsTest, ModelProfileOwnsStableCompletePolicyDigest) {
  llvm::ArrayRef<wafer::ModelProfileRecord> profiles =
      wafer::getRegisteredModelProfiles();
  ASSERT_EQ(profiles.size(), 1u);
  const wafer::ModelProfileRecord &profile = profiles.front();
  EXPECT_EQ(profile.id, kModelProfile);
  EXPECT_EQ(profile.canonicalSpelling, "wafer-model-formal-deterministic-v1");
  EXPECT_EQ(profile.numericDecodePolicy,
            (wafer::LogicalScalarCodecPolicy{
                wafer::LogicalByteOrder::LittleEndian,
                wafer::LogicalBitOrder::LeastSignificantBitFirstWithinByte,
                wafer::NonCanonicalEncodingPolicy::Reject}));
  EXPECT_EQ(profile.numericEncodePolicy,
            (wafer::LogicalScalarCodecPolicy{
                wafer::LogicalByteOrder::LittleEndian,
                wafer::LogicalBitOrder::LeastSignificantBitFirstWithinByte,
                wafer::NonCanonicalEncodingPolicy::ClearUnusedBits}));
  EXPECT_TRUE(profile.modelOnly);
  EXPECT_EQ(profile.policyDigest,
            "sha256:f6b38b0231d2459cd7d5cb345639474e61daf2a76c253163ffe1aa3a"
            "a0e46d82");
  EXPECT_EQ(wafer::stringifyModelProfileId(kModelProfile),
            profile.canonicalSpelling);
  EXPECT_EQ(&wafer::getModelProfileRecord(kModelProfile), &profile);

  llvm::Expected<ModelProfileId> parsed =
      wafer::parseModelProfileId("wafer-model-formal-deterministic-v1");
  ASSERT_TRUE(static_cast<bool>(parsed))
      << (parsed ? std::string() : llvm::toString(parsed.takeError()));
  EXPECT_EQ(*parsed, kModelProfile);
  EXPECT_NE(expectError(wafer::parseModelProfileId("unknown"))
                .find("unknown model profile"),
            std::string::npos);
}

TEST(NumericSemanticsTest, TensorKeyOwnsCheckedStaticShapeLayoutAndDigest) {
  std::set<std::string> digests;
  for (NumericTensorLayout layout :
       {NumericTensorLayout::Tensor, NumericTensorLayout::NTensor,
        NumericTensorLayout::Cx, NumericTensorLayout::NCx}) {
    NumericTensorKey tensor =
        makeTensor(wafer::LogicalFormat::BF16, layout, {2, 3, 5});
    EXPECT_EQ(tensor.getFormat(), wafer::LogicalFormat::BF16);
    EXPECT_EQ(tensor.getLayout(), layout);
    EXPECT_EQ(tensor.getShape(), llvm::ArrayRef<uint64_t>({2, 3, 5}));
    EXPECT_EQ(tensor.getElementCount(), 30u);
    EXPECT_TRUE(isDigest(tensor.getDigest()));
    EXPECT_TRUE(digests.insert(tensor.getDigest().str()).second);
  }

  NumericTensorKey scalar =
      makeTensor(wafer::LogicalFormat::F32, NumericTensorLayout::Tensor, {});
  NumericTensorKey empty = makeTensor(wafer::LogicalFormat::F32,
                                      NumericTensorLayout::Tensor, {4, 0, 7});
  EXPECT_TRUE(scalar.getShape().empty());
  EXPECT_EQ(scalar.getElementCount(), 1u);
  EXPECT_EQ(empty.getElementCount(), 0u);
  EXPECT_NE(scalar.getDigest(), empty.getDigest());

  std::vector<uint64_t> highRankShape(UINT32_C(0x10000), 1);
  llvm::Expected<NumericTensorKey> highRank = NumericTensorKey::create(
      wafer::LogicalFormat::F16, NumericTensorLayout::NTensor, highRankShape);
  ASSERT_TRUE(static_cast<bool>(highRank))
      << (highRank ? std::string() : llvm::toString(highRank.takeError()));
  EXPECT_EQ(highRank->getShape().size(), UINT32_C(0x10000));
  EXPECT_EQ(highRank->getElementCount(), 1u);

  std::string error = expectError(
      NumericTensorKey::create(static_cast<wafer::LogicalFormat>(255),
                               NumericTensorLayout::Tensor, {1}));
  EXPECT_NE(error.find("unknown format"), std::string::npos);
  error = expectError(NumericTensorKey::create(
      wafer::LogicalFormat::F16, static_cast<NumericTensorLayout>(255), {1}));
  EXPECT_NE(error.find("unknown layout"), std::string::npos);
  error = expectError(NumericTensorKey::create(
      wafer::LogicalFormat::F16, NumericTensorLayout::Tensor,
      {static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), 3}));
  EXPECT_NE(error.find("overflows uint64"), std::string::npos);
}

TEST(NumericSemanticsTest, OperationEnumsClose35ElementwiseAndFourReduceKinds) {
  llvm::ArrayRef<NumericElementwiseOperation> operations =
      wafer::getNumericElementwiseOperations();
  ASSERT_EQ(operations.size(), 35u);
  size_t unary = 0;
  size_t binary = 0;
  size_t relations = 0;
  size_t logic = 0;
  std::set<std::string> spellings;
  for (NumericElementwiseOperation operation : operations) {
    const unsigned arity = wafer::getNumericElementwiseArity(operation);
    ASSERT_TRUE(arity == 1 || arity == 2);
    unary += arity == 1;
    binary += arity == 2;
    relations += wafer::isNumericElementwiseRelation(operation);
    logic += wafer::isNumericElementwiseLogic(operation);
    EXPECT_TRUE(
        spellings
            .insert(
                wafer::stringifyNumericElementwiseOperation(operation).str())
            .second);
  }
  EXPECT_EQ(unary, 20u);
  EXPECT_EQ(binary, 15u);
  EXPECT_EQ(relations, 6u);
  EXPECT_EQ(logic, 4u);
  EXPECT_EQ(
      spellings,
      (std::set<std::string>{
          "abs",      "add",       "cos",       "div",      "eq",        "exp",
          "exp_lp",   "ge",        "gt",        "le",       "leakyrelu", "ln",
          "log2",     "logic_and", "logic_not", "logic_or", "logic_xor", "lt",
          "max",      "min",       "mul",       "ne",       "neg",       "pow2",
          "recip",    "relu",      "rsqrt",     "satrelu",  "sigmoid",   "sin",
          "softplus", "sqrt",      "square",    "sub",      "tanh",
      }));

  llvm::ArrayRef<NumericReduceOperation> reductions =
      wafer::getNumericReduceOperations();
  ASSERT_EQ(reductions.size(), 4u);
  spellings.clear();
  for (NumericReduceOperation operation : reductions)
    EXPECT_TRUE(
        spellings
            .insert(wafer::stringifyNumericReduceOperation(operation).str())
            .second);
  EXPECT_EQ(spellings, (std::set<std::string>{"avg", "max", "min", "sum"}));

  EXPECT_EQ(
      wafer::stringifyNumericCommandFamily(NumericCommandFamily::CTElementwise),
      "ct-elementwise");
  EXPECT_EQ(wafer::stringifyNumericTensorLayout(NumericTensorLayout::NCx),
            "ncx");
  EXPECT_EQ(wafer::stringifyNativeCTReduceDimension(
                NativeCTReduceDimension::Trailing2And1And0),
            "trailing-2-and-1-and-0");
  EXPECT_EQ(
      wafer::stringifyNumericModelImplementationReason(
          NumericModelImplementationReason::IntegerElementwisePolicyUnproven),
      "integer-elementwise-policy-unproven");
  EXPECT_EQ(wafer::stringifyNumericModelImplementationReason(
                NumericModelImplementationReason::ExpLpParameterPolicyUnproven),
            "exp-lp-parameter-policy-unproven");
  EXPECT_EQ(
      wafer::stringifyNumericModelImplementationReason(
          NumericModelImplementationReason::SatReluParameterPolicyUnproven),
      "sat-relu-parameter-policy-unproven");
  EXPECT_EQ(
      wafer::stringifyNumericModelImplementationReason(
          NumericModelImplementationReason::LeakyReluParameterPolicyUnproven),
      "leaky-relu-parameter-policy-unproven");
  EXPECT_EQ(wafer::stringifyNumericModelImplementationReason(
                NumericModelImplementationReason::
                    IntegerGemmAccumulatorPolicyUnproven),
            "integer-gemm-accumulator-policy-unproven");
}

TEST(NumericSemanticsTest, ExactConvertKeysValidateTensorAndParameterFacts) {
  size_t zeroPointCount = 0;
  size_t parameterlessCount = 0;
  size_t roundingCount = 0;
  std::set<std::string> digests;
  for (const wafer::TargetConvertRoute &route :
       wafer::getTargetConvertRoutes()) {
    std::optional<NumericConvertParameter> parameter;
    switch (route.parameterKind) {
    case TargetConvertParameterKind::None:
      ++parameterlessCount;
      break;
    case TargetConvertParameterKind::RoundingMode:
      ++roundingCount;
      parameter = NumericConvertParameter::roundingMode(
          NumericRoundingMode::TowardNegative);
      break;
    case TargetConvertParameterKind::ZeroPoint:
      ++zeroPointCount;
      parameter = NumericConvertParameter::zeroPoint(UINT32_C(0xdeadbeef));
      break;
    }

    std::optional<NumericCommandKey> key =
        expectConvertKey(route.opcode, parameter);
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->getFamily(), NumericCommandFamily::CTConvert);
    const wafer::NumericCTConvertCommand *convert = key->getCTConvert();
    ASSERT_NE(convert, nullptr);
    EXPECT_EQ(convert->opcode, route.opcode);
    EXPECT_EQ(convert->source.getFormat(), route.source);
    EXPECT_EQ(convert->destination.getFormat(), route.destination);
    EXPECT_EQ(convert->parameter, parameter);
    EXPECT_EQ(key->getCTElementwise(), nullptr);
    EXPECT_EQ(key->getNEGemm(), nullptr);
    EXPECT_EQ(key->getNativeCTReduce(), nullptr);
    EXPECT_TRUE(isDigest(key->getDigest()));
    EXPECT_TRUE(digests.insert(key->getDigest().str()).second);
  }
  EXPECT_EQ(zeroPointCount, 4u);
  EXPECT_EQ(parameterlessCount, 9u);
  EXPECT_EQ(roundingCount, 23u);

  const wafer::TargetConvertRoute &route =
      *wafer::findTargetConvertRoute(kTargetProfile, 144);
  NumericTensorKey source =
      makeTensor(route.source, NumericTensorLayout::Tensor, {2, 3});
  NumericTensorKey destination =
      makeTensor(route.destination, NumericTensorLayout::Tensor, {2, 4});
  std::string error = expectError(NumericCommandKey::createCTConvert(
      kTargetProfile, route.opcode, source, destination,
      NumericConvertParameter::roundingMode(NumericRoundingMode::NearestEven)));
  EXPECT_NE(error.find("same static element count"), std::string::npos);

  NumericTensorKey wrongSource = makeTensor(
      wafer::LogicalFormat::F32, NumericTensorLayout::Tensor, {2, 3});
  destination =
      makeTensor(route.destination, NumericTensorLayout::Tensor, {2, 3});
  error = expectError(NumericCommandKey::createCTConvert(
      kTargetProfile, route.opcode, wrongSource, destination,
      NumericConvertParameter::roundingMode(NumericRoundingMode::NearestEven)));
  EXPECT_NE(error.find("endpoints"), std::string::npos);

  source = makeTensor(route.source, NumericTensorLayout::Tensor,
                      {UINT64_C(0x100000000)});
  destination = makeTensor(route.destination, NumericTensorLayout::Tensor,
                           {UINT64_C(0x100000000)});
  error = expectError(NumericCommandKey::createCTConvert(
      kTargetProfile, route.opcode, source, destination,
      NumericConvertParameter::roundingMode(NumericRoundingMode::NearestEven)));
  EXPECT_NE(error.find("fit uint32"), std::string::npos);

  source = makeTensor(route.source, NumericTensorLayout::Tensor, {1});
  destination = makeTensor(route.destination, NumericTensorLayout::Tensor, {1});
  error = expectError(NumericCommandKey::createCTConvert(
      kTargetProfile, route.opcode, source, destination, std::nullopt));
  EXPECT_NE(error.find("requires exactly one rounding-mode"),
            std::string::npos);
  error = expectError(NumericCommandKey::createCTConvert(
      kTargetProfile, route.opcode, source, destination,
      NumericConvertParameter::zeroPoint(0)));
  EXPECT_NE(error.find("not a zero-point"), std::string::npos);

  error = expectError(NumericCommandKey::createCTConvert(
      kTargetProfile, std::numeric_limits<uint16_t>::max(), source, destination,
      NumericConvertParameter::roundingMode(NumericRoundingMode::NearestEven)));
  EXPECT_NE(error.find("unknown CT convert route"), std::string::npos);

  const wafer::TargetConvertRoute *plainRoute = nullptr;
  const wafer::TargetConvertRoute *zeroPointRoute = nullptr;
  for (const wafer::TargetConvertRoute &candidate :
       wafer::getTargetConvertRoutes()) {
    if (candidate.parameterKind == TargetConvertParameterKind::None)
      plainRoute = &candidate;
    if (candidate.parameterKind == TargetConvertParameterKind::ZeroPoint)
      zeroPointRoute = &candidate;
  }
  ASSERT_NE(plainRoute, nullptr);
  ASSERT_NE(zeroPointRoute, nullptr);
  error = expectError(NumericCommandKey::createCTConvert(
      kTargetProfile, plainRoute->opcode,
      makeTensor(plainRoute->source, NumericTensorLayout::Tensor, {1}),
      makeTensor(plainRoute->destination, NumericTensorLayout::Tensor, {1}),
      NumericConvertParameter::roundingMode(NumericRoundingMode::NearestEven)));
  EXPECT_NE(error.find("parameterless"), std::string::npos);
  error = expectError(NumericCommandKey::createCTConvert(
      kTargetProfile, zeroPointRoute->opcode,
      makeTensor(zeroPointRoute->source, NumericTensorLayout::Tensor, {1}),
      makeTensor(zeroPointRoute->destination, NumericTensorLayout::Tensor, {1}),
      NumericConvertParameter::roundingMode(NumericRoundingMode::NearestEven)));
  EXPECT_NE(error.find("not a rounding-mode"), std::string::npos);
}

TEST(NumericSemanticsTest,
     ReusableConvertSemanticsClose101PoliciesWithoutExactKeys) {
  llvm::ArrayRef<wafer::NumericSemanticsProfile> profiles =
      wafer::getRegisteredNumericSemanticsProfiles();
  ASSERT_EQ(profiles.size(), 101u);
  std::set<std::string> digests;
  std::set<std::string> identities;
  for (const wafer::NumericSemanticsProfile &profile : profiles) {
    EXPECT_EQ(profile.getModelProfile(), kModelProfile);
    EXPECT_TRUE(isDigest(profile.getDigest()));
    EXPECT_TRUE(digests.insert(profile.getDigest().str()).second);
    const wafer::NumericRoutePolicyIdentity &identity =
        profile.getRoutePolicyIdentity();
    const wafer::TargetConvertRoute &route = identity.getCTConvertRoute();
    EXPECT_EQ(identity.getTargetProfile(), kTargetProfile);
    EXPECT_EQ(identity.getFamily(), NumericCommandFamily::CTConvert);
    EXPECT_EQ(identity.getCTConvertOpcode(), route.opcode);
    EXPECT_NE(route.parameterKind, TargetConvertParameterKind::ZeroPoint);
    EXPECT_NE(profile.getRoundingMode(), NumericRoundingMode::Stochastic);
    EXPECT_TRUE(identities
                    .insert(std::to_string(route.opcode) + ":" +
                            std::to_string(static_cast<unsigned>(
                                profile.getRoundingMode())))
                    .second);
  }
  EXPECT_EQ(digests.size(), 101u);
  EXPECT_EQ(identities.size(), 101u);
}

TEST(NumericSemanticsTest,
     FamilyTypedElementwiseAndGemmSemanticsCarryCompletePolicies) {
  llvm::ArrayRef<wafer::NumericSemanticsProfile> elementwiseProfiles =
      wafer::getRegisteredNumericCTElementwiseSemanticsProfiles();
  ASSERT_EQ(elementwiseProfiles.size(), 88u);
  std::set<std::string> digests;
  size_t logicCount = 0;
  size_t relationCount = 0;
  size_t transcendentalCount = 0;
  for (const wafer::NumericSemanticsProfile &profile : elementwiseProfiles) {
    EXPECT_TRUE(isDigest(profile.getDigest()));
    EXPECT_TRUE(digests.insert(profile.getDigest().str()).second);
    EXPECT_EQ(profile.getModelProfile(), kModelProfile);
    EXPECT_EQ(profile.getFamily(), NumericCommandFamily::CTElementwise);
    EXPECT_EQ(profile.getCTConvertIdentity(), nullptr);
    EXPECT_EQ(profile.getNEGemmIdentity(), nullptr);
    const wafer::NumericCTElementwiseSemanticsIdentity *identity =
        profile.getCTElementwiseIdentity();
    ASSERT_NE(identity, nullptr);
    EXPECT_EQ(identity->getTargetProfile(), kTargetProfile);
    EXPECT_NE(identity->getInputFormat(), wafer::LogicalFormat::I8);
    const NumericElementwiseOperation operation = identity->getOperation();
    const bool logic = wafer::isNumericElementwiseLogic(operation);
    const bool relation = wafer::isNumericElementwiseRelation(operation);
    logicCount += logic;
    relationCount += relation;
    transcendentalCount += usesMPFR(operation);
    EXPECT_EQ(identity->getDestinationFormat(),
              relation ? wafer::LogicalFormat::Bool
                       : identity->getInputFormat());
    EXPECT_EQ(profile.getRoundingModePolicy(),
              logic || relation ? std::nullopt
                                : std::optional<NumericRoundingMode>(
                                      NumericRoundingMode::NearestEven));
    EXPECT_EQ(profile.getRoundingPointPolicy(),
              logic || relation
                  ? NumericRoundingPointPolicy::NotApplicable
                  : NumericRoundingPointPolicy::ElementwiseResult);
    EXPECT_EQ(profile.getFloatToIntegerPolicy(),
              FloatToIntegerPolicy::NotApplicable);
    EXPECT_EQ(profile.getFloatingNaNPolicy(),
              logic ? FloatingNaNPolicy::NotApplicable
              : relation
                  ? FloatingNaNPolicy::OrderedRelationFalseExceptNotEqualTrue
                  : FloatingNaNPolicy::CanonicalPositiveQuietNaN);
    FloatingSignedZeroPolicy expectedSignedZero =
        logic      ? FloatingSignedZeroPolicy::NotApplicable
        : relation ? FloatingSignedZeroPolicy::PredicateOnly
                   : FloatingSignedZeroPolicy::IEEE754OperationDefined;
    if (operation == NumericElementwiseOperation::Max)
      expectedSignedZero =
          FloatingSignedZeroPolicy::MaximumPositiveUnlessBothNegative;
    else if (operation == NumericElementwiseOperation::Min)
      expectedSignedZero =
          FloatingSignedZeroPolicy::MinimumNegativeUnlessBothPositive;
    EXPECT_EQ(profile.getFloatingSignedZeroPolicy(), expectedSignedZero);
    EXPECT_EQ(profile.getFloatingSubnormalPolicy(),
              logic ? FloatingSubnormalPolicy::NotApplicable
                    : FloatingSubnormalPolicy::Gradual);
    EXPECT_EQ(profile.getFloatingTininessPolicy(),
              logic || relation ? FloatingTininessPolicy::NotApplicable
                                : FloatingTininessPolicy::AfterRounding);
    EXPECT_EQ(profile.getExceptionFlagPolicy(),
              logic ? NumericExceptionFlagPolicy::NotApplicable
                    : NumericExceptionFlagPolicy::ModelOnly);
    EXPECT_EQ(
        profile.getFloatingNaNSignalingPolicy(),
        logic ? FloatingNaNSignalingPolicy::NotApplicable
              : FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot);
    EXPECT_EQ(profile.getFloatingDenormalModePolicy(),
              logic ? FloatingDenormalModePolicy::NotApplicable
                    : FloatingDenormalModePolicy::GradualNoDAZNoFTZ);
    EXPECT_EQ(profile.getFloatingOverflowPolicy(),
              logic || relation
                  ? FloatingOverflowPolicy::NotApplicable
                  : FloatingOverflowPolicy::IEEE754AccordingToRoundingMode);
    EXPECT_EQ(profile.getSaturationPolicy(),
              logic || relation ? NumericSaturationPolicy::NotApplicable
                                : NumericSaturationPolicy::Disabled);
    EXPECT_EQ(profile.getTranscendentalEvaluationPolicy(),
              usesMPFR(operation)
                  ? NumericTranscendentalEvaluationPolicy::
                        CorrectlyRoundedMathematicalResultAdaptiveMPFRFinalRNE
                  : NumericTranscendentalEvaluationPolicy::NotApplicable);
    EXPECT_EQ(profile.getGemmAccumulatorPolicy(),
              NumericGemmAccumulatorPolicy::NotApplicable);
    EXPECT_EQ(profile.getGemmAccumulatorInitializationPolicy(),
              NumericGemmAccumulatorInitializationPolicy::NotApplicable);
    EXPECT_EQ(profile.getGemmReductionOrderPolicy(),
              NumericGemmReductionOrderPolicy::NotApplicable);
  }
  EXPECT_EQ(logicCount, 4u);
  EXPECT_EQ(relationCount, 18u);
  EXPECT_EQ(transcendentalCount, 33u);

  llvm::ArrayRef<wafer::NumericSemanticsProfile> gemmProfiles =
      wafer::getRegisteredNumericNEGemmSemanticsProfiles();
  ASSERT_EQ(gemmProfiles.size(), 3u);
  std::set<wafer::LogicalFormat> gemmFormats;
  for (const wafer::NumericSemanticsProfile &profile : gemmProfiles) {
    EXPECT_TRUE(isDigest(profile.getDigest()));
    EXPECT_TRUE(digests.insert(profile.getDigest().str()).second);
    EXPECT_EQ(profile.getFamily(), NumericCommandFamily::NEGemm);
    EXPECT_EQ(profile.getCTConvertIdentity(), nullptr);
    EXPECT_EQ(profile.getCTElementwiseIdentity(), nullptr);
    const wafer::NumericNEGemmSemanticsIdentity *identity =
        profile.getNEGemmIdentity();
    ASSERT_NE(identity, nullptr);
    EXPECT_TRUE(gemmFormats.insert(identity->getFormat()).second);
    EXPECT_EQ(profile.getRoundingModePolicy(),
              NumericRoundingMode::NearestEven);
    EXPECT_EQ(profile.getRoundingPointPolicy(),
              NumericRoundingPointPolicy::GemmFusedMultiplyAddAndDestination);
    EXPECT_EQ(profile.getFloatingNaNPolicy(),
              FloatingNaNPolicy::CanonicalPositiveQuietNaN);
    EXPECT_EQ(profile.getFloatingSignedZeroPolicy(),
              FloatingSignedZeroPolicy::GemmPositiveZeroAccumulatorThenIEEE754);
    EXPECT_EQ(profile.getFloatingSubnormalPolicy(),
              FloatingSubnormalPolicy::Gradual);
    EXPECT_EQ(profile.getFloatingTininessPolicy(),
              FloatingTininessPolicy::AfterRounding);
    EXPECT_EQ(profile.getExceptionFlagPolicy(),
              NumericExceptionFlagPolicy::ModelOnly);
    EXPECT_EQ(profile.getFloatingNaNSignalingPolicy(),
              FloatingNaNSignalingPolicy::SignalingRaisesInvalidQuietDoesNot);
    EXPECT_EQ(profile.getFloatingDenormalModePolicy(),
              FloatingDenormalModePolicy::GradualNoDAZNoFTZ);
    EXPECT_EQ(profile.getFloatingOverflowPolicy(),
              FloatingOverflowPolicy::IEEE754AccordingToRoundingMode);
    EXPECT_EQ(profile.getSaturationPolicy(), NumericSaturationPolicy::Disabled);
    EXPECT_EQ(profile.getTranscendentalEvaluationPolicy(),
              NumericTranscendentalEvaluationPolicy::NotApplicable);
    EXPECT_EQ(profile.getGemmAccumulatorPolicy(),
              NumericGemmAccumulatorPolicy::F32FusedMultiplyAdd);
    EXPECT_EQ(profile.getGemmAccumulatorInitializationPolicy(),
              NumericGemmAccumulatorInitializationPolicy::PositiveZero);
    EXPECT_EQ(profile.getGemmReductionOrderPolicy(),
              NumericGemmReductionOrderPolicy::IncreasingK);
  }
  EXPECT_EQ(gemmFormats,
            (std::set<wafer::LogicalFormat>{wafer::LogicalFormat::F16,
                                            wafer::LogicalFormat::BF16,
                                            wafer::LogicalFormat::F32}));
  EXPECT_EQ(digests.size(), 91u);
}

TEST(NumericSemanticsTest, PatternRegistryClosesExactly276TypedSelectors) {
  llvm::ArrayRef<wafer::NumericCapabilityPattern> patterns =
      wafer::getRegisteredNumericCapabilityPatterns();
  ASSERT_EQ(patterns.size(), 276u);
  size_t convertSupported = 0;
  size_t convertStochastic = 0;
  size_t convertZeroPoint = 0;
  size_t elementwiseSupported = 0;
  size_t elementwiseInteger = 0;
  size_t elementwiseExpLp = 0;
  size_t elementwiseSatRelu = 0;
  size_t elementwiseLeakyRelu = 0;
  size_t gemmSupported = 0;
  size_t gemmInteger = 0;
  size_t reduce = 0;
  size_t reduceSupported = 0;
  std::set<std::string> digests;
  std::set<const wafer::NumericSemanticsProfile *> semantics;

  for (const wafer::NumericCapabilityPattern &pattern : patterns) {
    EXPECT_TRUE(isDigest(pattern.getDigest()));
    EXPECT_TRUE(digests.insert(pattern.getDigest().str()).second);
    EXPECT_EQ(pattern.getModelProfile(), kModelProfile);
    EXPECT_EQ(pattern.getTargetProfile(), kTargetProfile);
    EXPECT_EQ(pattern.getCompilerCapability().status,
              NumericCompilerEmittabilityStatus::Emittable);
    EXPECT_EQ(pattern.getCompilerCapability().reason,
              NumericCompilerEmittabilityReason::None);
    EXPECT_EQ(pattern.getEvidenceCapability().status,
              NumericEvidenceStatus::ModelOnlyUncorrelated);
    EXPECT_EQ(pattern.getEvidenceCapability().reason,
              NumericEvidenceReason::HardwareCorrelationNotRun);

    if (const auto *selector = pattern.getCTConvertSelector()) {
      EXPECT_EQ(pattern.getFamily(), NumericCommandFamily::CTConvert);
      if (pattern.isSupported()) {
        ++convertSupported;
        ASSERT_NE(pattern.getSemantics(), nullptr);
        EXPECT_TRUE(semantics.insert(pattern.getSemantics()).second);
        EXPECT_EQ(pattern.getFormalKernelKind(), FormalKernelKind::Convert);
        EXPECT_EQ(pattern.getComparatorKind(), NumericComparatorKind::RawExact);
        EXPECT_EQ(pattern.getFormalBackendKind(),
                  FormalNumericBackendKind::LLVMAPFloatAPInt);
      } else if (selector->parameter.getKind() ==
                 NumericCapabilityParameterPatternKind::AnyZeroPoint) {
        ++convertZeroPoint;
        EXPECT_EQ(pattern.getModelCapability().reason,
                  NumericModelImplementationReason::ZeroPointFormulaUnproven);
      } else {
        ++convertStochastic;
        EXPECT_EQ(selector->parameter.getExactRoundingMode(),
                  NumericRoundingMode::Stochastic);
        EXPECT_EQ(pattern.getModelCapability().reason,
                  NumericModelImplementationReason::StochasticStateUnproven);
      }
      continue;
    }

    if (const auto *selector = pattern.getCTElementwiseSelector()) {
      EXPECT_EQ(pattern.getFamily(), NumericCommandFamily::CTElementwise);
      const std::optional<NumericModelImplementationReason> reason =
          expectedElementwiseUnsupportedReason(selector->operation,
                                               selector->inputFormat);
      if (reason) {
        EXPECT_FALSE(pattern.isSupported());
        EXPECT_EQ(pattern.getModelCapability().status,
                  NumericModelImplementationStatus::Absent);
        EXPECT_EQ(pattern.getModelCapability().reason, *reason);
        EXPECT_EQ(pattern.getSemantics(), nullptr);
        EXPECT_FALSE(pattern.getFormalKernelKind());
        EXPECT_FALSE(pattern.getComparatorKind());
        EXPECT_FALSE(pattern.getFormalBackendKind());
        switch (*reason) {
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
          ADD_FAILURE() << "unexpected elementwise unsupported reason";
        }
      } else {
        ++elementwiseSupported;
        ASSERT_TRUE(pattern.isSupported());
        ASSERT_NE(pattern.getSemantics(), nullptr);
        EXPECT_TRUE(semantics.insert(pattern.getSemantics()).second);
        EXPECT_EQ(pattern.getFormalKernelKind(), FormalKernelKind::Elementwise);
        EXPECT_EQ(pattern.getComparatorKind(), NumericComparatorKind::RawExact);
        EXPECT_EQ(pattern.getFormalBackendKind(),
                  usesMPFR(selector->operation)
                      ? FormalNumericBackendKind::MPFR
                      : FormalNumericBackendKind::LLVMAPFloatAPInt);
      }
      continue;
    }
    if (const auto *selector = pattern.getNEGemmSelector()) {
      EXPECT_EQ(pattern.getFamily(), NumericCommandFamily::NEGemm);
      if (selector->format == wafer::LogicalFormat::I8) {
        ++gemmInteger;
        EXPECT_FALSE(pattern.isSupported());
        EXPECT_EQ(pattern.getModelCapability().status,
                  NumericModelImplementationStatus::Absent);
        EXPECT_EQ(pattern.getModelCapability().reason,
                  NumericModelImplementationReason::
                      IntegerGemmAccumulatorPolicyUnproven);
        EXPECT_EQ(pattern.getSemantics(), nullptr);
        EXPECT_FALSE(pattern.getFormalKernelKind());
        EXPECT_FALSE(pattern.getComparatorKind());
        EXPECT_FALSE(pattern.getFormalBackendKind());
      } else {
        ++gemmSupported;
        ASSERT_TRUE(pattern.isSupported());
        ASSERT_NE(pattern.getSemantics(), nullptr);
        EXPECT_TRUE(semantics.insert(pattern.getSemantics()).second);
        EXPECT_EQ(pattern.getFormalKernelKind(), FormalKernelKind::Gemm);
        EXPECT_EQ(pattern.getComparatorKind(), NumericComparatorKind::RawExact);
        EXPECT_EQ(pattern.getFormalBackendKind(),
                  FormalNumericBackendKind::LLVMAPFloatAPInt);
      }
      continue;
    }
    ASSERT_NE(pattern.getNativeCTReduceSelector(), nullptr);
    ++reduce;
    EXPECT_EQ(pattern.getFamily(), NumericCommandFamily::NativeCTReduce);
    const auto *selector = pattern.getNativeCTReduceSelector();
    if (selector->operation == NumericReduceOperation::Sum &&
        selector->format == wafer::LogicalFormat::F32) {
      ++reduceSupported;
      EXPECT_TRUE(pattern.isSupported());
      ASSERT_NE(pattern.getSemantics(), nullptr);
      EXPECT_TRUE(semantics.insert(pattern.getSemantics()).second);
      EXPECT_EQ(pattern.getFormalKernelKind(), FormalKernelKind::Reduce);
      EXPECT_EQ(pattern.getComparatorKind(), NumericComparatorKind::RawExact);
      EXPECT_EQ(pattern.getFormalBackendKind(),
                FormalNumericBackendKind::LLVMAPFloatAPInt);
      const auto *identity =
          pattern.getSemantics()->getNativeCTReduceIdentity();
      ASSERT_NE(identity, nullptr);
      EXPECT_EQ(identity->getOperation(), NumericReduceOperation::Sum);
      EXPECT_EQ(identity->getFormat(), wafer::LogicalFormat::F32);
    } else {
      EXPECT_FALSE(pattern.isSupported());
      EXPECT_EQ(pattern.getModelCapability().status,
                NumericModelImplementationStatus::Absent);
      EXPECT_EQ(
          pattern.getModelCapability().reason,
          NumericModelImplementationReason::NativeReductionPolicyUnproven);
      EXPECT_EQ(pattern.getSemantics(), nullptr);
      EXPECT_FALSE(pattern.getFormalKernelKind());
      EXPECT_FALSE(pattern.getComparatorKind());
      EXPECT_FALSE(pattern.getFormalBackendKind());
    }
  }
  EXPECT_EQ(convertSupported, 101u);
  EXPECT_EQ(convertStochastic, 23u);
  EXPECT_EQ(convertZeroPoint, 4u);
  EXPECT_EQ(elementwiseSupported, 88u);
  EXPECT_EQ(elementwiseInteger, 31u);
  EXPECT_EQ(elementwiseExpLp, 3u);
  EXPECT_EQ(elementwiseSatRelu, 3u);
  EXPECT_EQ(elementwiseLeakyRelu, 3u);
  EXPECT_EQ(gemmSupported, 3u);
  EXPECT_EQ(gemmInteger, 1u);
  EXPECT_EQ(reduce, 16u);
  EXPECT_EQ(reduceSupported, 1u);
  EXPECT_EQ(digests.size(), 276u);
  EXPECT_EQ(semantics.size(), 193u);
}

TEST(NumericSemanticsTest, RegistryValidatorDetectsMissingAndOverlap) {
  llvm::ArrayRef<wafer::NumericCapabilityPattern> patterns =
      wafer::getRegisteredNumericCapabilityPatterns();
  llvm::Error valid =
      wafer::validateNumericCapabilityPatternsForTesting(patterns);
  if (valid)
    ADD_FAILURE() << llvm::toString(std::move(valid));

  std::string error = expectError(
      wafer::validateNumericCapabilityPatternsForTesting(patterns.drop_back()));
  EXPECT_NE(error.find("missing or has an extra"), std::string::npos);

  std::vector<wafer::NumericCapabilityPattern> duplicate(patterns.begin(),
                                                         patterns.end());
  duplicate.push_back(patterns.front());
  error = expectError(
      wafer::validateNumericCapabilityPatternsForTesting(duplicate));
  EXPECT_NE(error.find("overlapping selectors"), std::string::npos);
}

TEST(NumericSemanticsTest, ConvertResolutionPreserves101And128Closure) {
  std::set<const wafer::NumericCapabilityPattern *> matchedPatterns;
  size_t supported = 0;
  size_t stochastic = 0;
  for (const wafer::TargetConvertRoute &route :
       wafer::getTargetConvertRoutes()) {
    if (route.parameterKind == TargetConvertParameterKind::ZeroPoint)
      continue;
    llvm::ArrayRef<NumericRoundingMode> modes;
    NumericRoundingMode plain = NumericRoundingMode::NearestEven;
    if (route.parameterKind == TargetConvertParameterKind::None)
      modes = llvm::ArrayRef<NumericRoundingMode>(&plain, 1);
    else
      modes = wafer::getNumericRoundingModes();
    for (NumericRoundingMode mode : modes) {
      std::optional<NumericConvertParameter> parameter;
      if (route.parameterKind == TargetConvertParameterKind::RoundingMode)
        parameter = NumericConvertParameter::roundingMode(mode);
      std::optional<NumericCommandKey> key =
          expectConvertKey(route.opcode, parameter);
      ASSERT_TRUE(key.has_value());
      std::optional<ResolvedNumericCommand> resolution = expectResolution(*key);
      ASSERT_TRUE(resolution.has_value());
      matchedPatterns.insert(&resolution->getPattern());
      EXPECT_EQ(resolution->getFamily(), NumericCommandFamily::CTConvert);
      EXPECT_TRUE(isDigest(resolution->getDigest()));
      if (mode == NumericRoundingMode::Stochastic) {
        ++stochastic;
        EXPECT_FALSE(resolution->isSupported());
      } else {
        ++supported;
        EXPECT_TRUE(resolution->isSupported());
        ASSERT_NE(resolution->getSemantics(), nullptr);
        EXPECT_EQ(resolution->getSemantics()->getRoundingMode(), mode);
      }
    }
  }
  EXPECT_EQ(supported, 101u);
  EXPECT_EQ(stochastic, 23u);
  EXPECT_EQ(matchedPatterns.size(), 124u);

  for (const wafer::TargetConvertRoute &route :
       wafer::getTargetConvertRoutes()) {
    if (route.parameterKind != TargetConvertParameterKind::ZeroPoint)
      continue;
    std::optional<NumericCommandKey> zero =
        expectConvertKey(route.opcode, NumericConvertParameter::zeroPoint(0));
    std::optional<NumericCommandKey> maximum = expectConvertKey(
        route.opcode, NumericConvertParameter::zeroPoint(UINT32_MAX));
    ASSERT_TRUE(zero && maximum);
    std::optional<ResolvedNumericCommand> zeroResolution =
        expectResolution(*zero);
    std::optional<ResolvedNumericCommand> maximumResolution =
        expectResolution(*maximum);
    ASSERT_TRUE(zeroResolution && maximumResolution);
    EXPECT_EQ(&zeroResolution->getPattern(), &maximumResolution->getPattern());
    EXPECT_NE(zero->getDigest(), maximum->getDigest());
    EXPECT_NE(zeroResolution->getDigest(), maximumResolution->getDigest());
    matchedPatterns.insert(&zeroResolution->getPattern());
  }
  EXPECT_EQ(matchedPatterns.size(), 128u);
}

TEST(NumericSemanticsTest,
     ConvertShapeAndLayoutChangeOnlyExactAndResolutionDigest) {
  const auto parameter =
      NumericConvertParameter::roundingMode(NumericRoundingMode::NearestEven);
  std::optional<NumericCommandKey> base =
      expectConvertKey(/*int16_bf16=*/144, parameter, {2, 3});
  const wafer::TargetConvertRoute &route =
      *wafer::findTargetConvertRoute(kTargetProfile, /*int16_bf16=*/144);
  llvm::Expected<NumericCommandKey> shapeExpected =
      NumericCommandKey::createCTConvert(
          kTargetProfile, route.opcode,
          makeTensor(route.source, NumericTensorLayout::Tensor, {2, 3}),
          makeTensor(route.destination, NumericTensorLayout::Tensor, {1, 6}),
          parameter);
  ASSERT_TRUE(static_cast<bool>(shapeExpected))
      << (shapeExpected ? std::string()
                        : llvm::toString(shapeExpected.takeError()));
  std::optional<NumericCommandKey> shape = std::move(*shapeExpected);
  std::optional<NumericCommandKey> layout = expectConvertKey(
      /*int16_bf16=*/144, parameter, {2, 3}, NumericTensorLayout::NTensor,
      NumericTensorLayout::Cx);
  ASSERT_TRUE(base && shape && layout);
  const wafer::NumericCTConvertCommand *reshapedConvert = shape->getCTConvert();
  ASSERT_NE(reshapedConvert, nullptr);
  EXPECT_EQ(reshapedConvert->source.getShape(),
            llvm::ArrayRef<uint64_t>({2, 3}));
  EXPECT_EQ(reshapedConvert->destination.getShape(),
            llvm::ArrayRef<uint64_t>({1, 6}));
  EXPECT_EQ(reshapedConvert->source.getElementCount(),
            reshapedConvert->destination.getElementCount());
  std::optional<ResolvedNumericCommand> baseResolution =
      expectResolution(*base);
  std::optional<ResolvedNumericCommand> shapeResolution =
      expectResolution(*shape);
  std::optional<ResolvedNumericCommand> layoutResolution =
      expectResolution(*layout);
  ASSERT_TRUE(baseResolution && shapeResolution && layoutResolution);

  EXPECT_NE(base->getDigest(), shape->getDigest());
  EXPECT_NE(base->getDigest(), layout->getDigest());
  EXPECT_NE(baseResolution->getDigest(), shapeResolution->getDigest());
  EXPECT_NE(baseResolution->getDigest(), layoutResolution->getDigest());
  EXPECT_EQ(&baseResolution->getPattern(), &shapeResolution->getPattern());
  EXPECT_EQ(&baseResolution->getPattern(), &layoutResolution->getPattern());
  EXPECT_EQ(baseResolution->getSemantics(), shapeResolution->getSemantics());
  EXPECT_EQ(baseResolution->getSemantics(), layoutResolution->getSemantics());

  std::optional<ResolvedNumericCommand> repeated = expectResolution(*base);
  ASSERT_TRUE(repeated.has_value());
  EXPECT_EQ(repeated->getDigest(), baseResolution->getDigest());
}

TEST(NumericSemanticsTest, ElementwiseClosesAll35ArityAndFormatSelectors) {
  size_t exactKeys = 0;
  size_t supported = 0;
  size_t integerUnsupported = 0;
  size_t parameterUnsupported = 0;
  std::set<const wafer::NumericCapabilityPattern *> patterns;
  for (NumericElementwiseOperation operation :
       wafer::getNumericElementwiseOperations()) {
    std::vector<wafer::LogicalFormat> formats;
    if (wafer::isNumericElementwiseLogic(operation))
      formats.push_back(wafer::LogicalFormat::Bool);
    else
      formats.assign(std::begin(kComputeFormats), std::end(kComputeFormats));

    for (wafer::LogicalFormat format : formats) {
      const wafer::LogicalFormat destinationFormat =
          wafer::isNumericElementwiseRelation(operation)
              ? wafer::LogicalFormat::Bool
              : format;
      NumericTensorKey input =
          makeTensor(format, NumericTensorLayout::Tensor, {2, 3, 5});
      std::vector<NumericTensorKey> inputs(
          wafer::getNumericElementwiseArity(operation), input);
      NumericTensorKey destination =
          makeTensor(destinationFormat, NumericTensorLayout::Tensor, {2, 3, 5});
      llvm::Expected<NumericCommandKey> key =
          NumericCommandKey::createCTElementwise(kTargetProfile, operation,
                                                 std::move(inputs),
                                                 std::move(destination));
      ASSERT_TRUE(static_cast<bool>(key))
          << (key ? std::string() : llvm::toString(key.takeError()));
      EXPECT_EQ(key->getFamily(), NumericCommandFamily::CTElementwise);
      const wafer::NumericCTElementwiseCommand *command =
          key->getCTElementwise();
      ASSERT_NE(command, nullptr);
      EXPECT_EQ(key->getCTConvert(), nullptr);
      EXPECT_EQ(key->getNEGemm(), nullptr);
      EXPECT_EQ(key->getNativeCTReduce(), nullptr);
      EXPECT_EQ(command->inputs.size(),
                wafer::getNumericElementwiseArity(operation));
      llvm::Expected<ResolvedNumericCommand> resolution =
          wafer::resolveNumericCommand(kModelProfile, std::move(*key));
      ASSERT_TRUE(static_cast<bool>(resolution))
          << (resolution ? std::string()
                         : llvm::toString(resolution.takeError()));
      const auto *selector =
          resolution->getPattern().getCTElementwiseSelector();
      ASSERT_NE(selector, nullptr);
      EXPECT_EQ(selector->operation, operation);
      EXPECT_EQ(selector->inputFormat, format);
      const std::optional<NumericModelImplementationReason> reason =
          expectedElementwiseUnsupportedReason(operation, format);
      if (reason) {
        EXPECT_FALSE(resolution->isSupported());
        EXPECT_EQ(resolution->getPattern().getModelCapability().reason,
                  *reason);
        EXPECT_EQ(resolution->getSemantics(), nullptr);
        if (*reason ==
            NumericModelImplementationReason::IntegerElementwisePolicyUnproven)
          ++integerUnsupported;
        else
          ++parameterUnsupported;
      } else {
        ++supported;
        ASSERT_TRUE(resolution->isSupported());
        const wafer::NumericSemanticsProfile *semantics =
            resolution->getSemantics();
        ASSERT_NE(semantics, nullptr);
        EXPECT_EQ(semantics->getFamily(), NumericCommandFamily::CTElementwise);
        const wafer::NumericCTElementwiseSemanticsIdentity *identity =
            semantics->getCTElementwiseIdentity();
        ASSERT_NE(identity, nullptr);
        EXPECT_EQ(semantics->getCTConvertIdentity(), nullptr);
        EXPECT_EQ(semantics->getNEGemmIdentity(), nullptr);
        EXPECT_EQ(identity->getOperation(), operation);
        EXPECT_EQ(identity->getInputFormat(), format);
        EXPECT_EQ(identity->getDestinationFormat(), destinationFormat);
        EXPECT_EQ(resolution->getFormalKernelKind(),
                  FormalKernelKind::Elementwise);
        EXPECT_EQ(resolution->getComparatorKind(),
                  NumericComparatorKind::RawExact);
        EXPECT_EQ(resolution->getFormalBackendKind(),
                  usesMPFR(operation)
                      ? FormalNumericBackendKind::MPFR
                      : FormalNumericBackendKind::LLVMAPFloatAPInt);
      }
      patterns.insert(&resolution->getPattern());
      ++exactKeys;
    }
  }
  EXPECT_EQ(exactKeys, 128u);
  EXPECT_EQ(patterns.size(), 128u);
  EXPECT_EQ(supported, 88u);
  EXPECT_EQ(integerUnsupported, 31u);
  EXPECT_EQ(parameterUnsupported, 9u);
}

TEST(NumericSemanticsTest,
     ElementwiseFactoryRejectsInvalidSurfaceBeforeResolve) {
  NumericTensorKey f16 = makeTensor(wafer::LogicalFormat::F16,
                                    NumericTensorLayout::Tensor, {2, 3});
  NumericTensorKey f16Destination = f16;
  std::string error = expectError(NumericCommandKey::createCTElementwise(
      kTargetProfile, NumericElementwiseOperation::Add, {f16}, f16Destination));
  EXPECT_NE(error.find("expects 2 input"), std::string::npos);

  NumericTensorKey mismatched = makeTensor(wafer::LogicalFormat::F16,
                                           NumericTensorLayout::Tensor, {3, 2});
  error = expectError(NumericCommandKey::createCTElementwise(
      kTargetProfile, NumericElementwiseOperation::Add, {f16, f16},
      mismatched));
  EXPECT_NE(error.find("shapes must match"), std::string::npos);

  NumericTensorKey ntensor = makeTensor(wafer::LogicalFormat::F16,
                                        NumericTensorLayout::NTensor, {2, 3});
  llvm::Expected<NumericCommandKey> layoutVariant =
      NumericCommandKey::createCTElementwise(kTargetProfile,
                                             NumericElementwiseOperation::Neg,
                                             {ntensor}, f16Destination);
  ASSERT_TRUE(static_cast<bool>(layoutVariant))
      << (layoutVariant ? std::string()
                        : llvm::toString(layoutVariant.takeError()));
  ASSERT_NE(layoutVariant->getCTElementwise(), nullptr);
  EXPECT_EQ(layoutVariant->getCTElementwise()->inputs.front().getLayout(),
            NumericTensorLayout::NTensor);
  EXPECT_EQ(layoutVariant->getCTElementwise()->destination.getLayout(),
            NumericTensorLayout::Tensor);
  llvm::Expected<NumericCommandKey> tensorVariant =
      NumericCommandKey::createCTElementwise(kTargetProfile,
                                             NumericElementwiseOperation::Neg,
                                             {f16}, f16Destination);
  ASSERT_TRUE(static_cast<bool>(tensorVariant));
  EXPECT_NE(layoutVariant->getDigest(), tensorVariant->getDigest());
  llvm::Expected<ResolvedNumericCommand> layoutResolution =
      wafer::resolveNumericCommand(kModelProfile, std::move(*layoutVariant));
  llvm::Expected<ResolvedNumericCommand> tensorResolution =
      wafer::resolveNumericCommand(kModelProfile, std::move(*tensorVariant));
  ASSERT_TRUE(static_cast<bool>(layoutResolution));
  ASSERT_TRUE(static_cast<bool>(tensorResolution));
  EXPECT_EQ(&layoutResolution->getPattern(), &tensorResolution->getPattern());
  EXPECT_NE(layoutResolution->getDigest(), tensorResolution->getDigest());

  NumericTensorKey boolean = makeTensor(wafer::LogicalFormat::Bool,
                                        NumericTensorLayout::Tensor, {2, 3});
  error = expectError(NumericCommandKey::createCTElementwise(
      kTargetProfile, NumericElementwiseOperation::Add, {boolean, boolean},
      boolean));
  EXPECT_NE(error.find("numeric CT elementwise"), std::string::npos);
  error = expectError(NumericCommandKey::createCTElementwise(
      kTargetProfile, NumericElementwiseOperation::Eq, {boolean, boolean},
      boolean));
  EXPECT_NE(error.find("matching numeric inputs"), std::string::npos);
  error = expectError(NumericCommandKey::createCTElementwise(
      kTargetProfile, NumericElementwiseOperation::LogicAnd, {f16, f16}, f16));
  EXPECT_NE(error.find("BOOL inputs"), std::string::npos);

  NumericTensorKey i16 = makeTensor(wafer::LogicalFormat::I16,
                                    NumericTensorLayout::Tensor, {2, 3});
  error = expectError(NumericCommandKey::createCTElementwise(
      kTargetProfile, NumericElementwiseOperation::Neg, {i16}, i16));
  EXPECT_NE(error.find("not compiler-emittable"), std::string::npos);

  error = expectError(NumericCommandKey::createCTElementwise(
      kTargetProfile, static_cast<NumericElementwiseOperation>(255), {f16},
      f16Destination));
  EXPECT_NE(error.find("unknown numeric elementwise"), std::string::npos);
}

TEST(NumericSemanticsTest, GemmValidatesFormatsLayoutsShapesBatchAndAxes) {
  std::string error = expectError(
      wafer::getCanonicalNumericGemmAxes(std::numeric_limits<uint64_t>::max()));
  EXPECT_NE(error.find("host size domain"), std::string::npos);

  std::set<const wafer::NumericCapabilityPattern *> patterns;
  size_t supported = 0;
  size_t integerUnsupported = 0;
  for (wafer::LogicalFormat format : kComputeFormats) {
    NumericTensorKey lhs = makeTensor(format, NumericTensorLayout::Cx, {2, 3});
    NumericTensorKey rhs = makeTensor(format, NumericTensorLayout::NCx, {3, 4});
    NumericTensorKey destination =
        makeTensor(format, NumericTensorLayout::Cx, {2, 4});
    llvm::Expected<NumericCommandKey> key = NumericCommandKey::createNEGemm(
        kTargetProfile, lhs, rhs, destination, 2, 3, 4, 1, canonicalAxes(2));
    ASSERT_TRUE(static_cast<bool>(key))
        << (key ? std::string() : llvm::toString(key.takeError()));
    EXPECT_EQ(key->getFamily(), NumericCommandFamily::NEGemm);
    ASSERT_NE(key->getNEGemm(), nullptr);
    EXPECT_EQ(key->getCTConvert(), nullptr);
    EXPECT_EQ(key->getCTElementwise(), nullptr);
    EXPECT_EQ(key->getNativeCTReduce(), nullptr);
    llvm::Expected<ResolvedNumericCommand> resolution =
        wafer::resolveNumericCommand(kModelProfile, std::move(*key));
    ASSERT_TRUE(static_cast<bool>(resolution))
        << (resolution ? std::string()
                       : llvm::toString(resolution.takeError()));
    if (format == wafer::LogicalFormat::I8) {
      ++integerUnsupported;
      EXPECT_FALSE(resolution->isSupported());
      EXPECT_EQ(resolution->getPattern().getModelCapability().reason,
                NumericModelImplementationReason::
                    IntegerGemmAccumulatorPolicyUnproven);
      EXPECT_EQ(resolution->getSemantics(), nullptr);
    } else {
      ++supported;
      ASSERT_TRUE(resolution->isSupported());
      const wafer::NumericSemanticsProfile *semantics =
          resolution->getSemantics();
      ASSERT_NE(semantics, nullptr);
      EXPECT_EQ(semantics->getFamily(), NumericCommandFamily::NEGemm);
      const wafer::NumericNEGemmSemanticsIdentity *identity =
          semantics->getNEGemmIdentity();
      ASSERT_NE(identity, nullptr);
      EXPECT_EQ(semantics->getCTConvertIdentity(), nullptr);
      EXPECT_EQ(semantics->getCTElementwiseIdentity(), nullptr);
      EXPECT_EQ(identity->getFormat(), format);
      EXPECT_EQ(resolution->getFormalKernelKind(), FormalKernelKind::Gemm);
      EXPECT_EQ(resolution->getComparatorKind(),
                NumericComparatorKind::RawExact);
      EXPECT_EQ(resolution->getFormalBackendKind(),
                FormalNumericBackendKind::LLVMAPFloatAPInt);
    }
    patterns.insert(&resolution->getPattern());
  }
  EXPECT_EQ(patterns.size(), 4u);
  EXPECT_EQ(supported, 3u);
  EXPECT_EQ(integerUnsupported, 1u);

  NumericTensorKey lhs3 = makeTensor(wafer::LogicalFormat::F16,
                                     NumericTensorLayout::NCx, {5, 2, 3});
  NumericTensorKey rhs3 = makeTensor(wafer::LogicalFormat::F16,
                                     NumericTensorLayout::NCx, {5, 3, 4});
  NumericTensorKey destination3 = makeTensor(
      wafer::LogicalFormat::F16, NumericTensorLayout::NCx, {5, 2, 4});
  llvm::Expected<NumericCommandKey> batched = NumericCommandKey::createNEGemm(
      kTargetProfile, lhs3, rhs3, destination3, 2, 3, 4, 5, canonicalAxes(3));
  ASSERT_TRUE(static_cast<bool>(batched))
      << (batched ? std::string() : llvm::toString(batched.takeError()));
  const wafer::NumericNEGemmCommand *command = batched->getNEGemm();
  ASSERT_NE(command, nullptr);
  EXPECT_EQ(command->batchCount, 5u);
  EXPECT_EQ(command->axes, canonicalAxes(3));

  std::vector<uint64_t> lhs12(10, 1), rhs12(10, 1), destination12(10, 1);
  lhs12.insert(lhs12.end(), {2, 3});
  rhs12.insert(rhs12.end(), {3, 4});
  destination12.insert(destination12.end(), {2, 4});
  llvm::Expected<NumericCommandKey> rank12 = NumericCommandKey::createNEGemm(
      kTargetProfile,
      makeTensor(wafer::LogicalFormat::BF16, NumericTensorLayout::NCx, lhs12),
      makeTensor(wafer::LogicalFormat::BF16, NumericTensorLayout::NCx, rhs12),
      makeTensor(wafer::LogicalFormat::BF16, NumericTensorLayout::NCx,
                 destination12),
      2, 3, 4, 1, canonicalAxes(12));
  EXPECT_TRUE(static_cast<bool>(rank12))
      << (rank12 ? std::string() : llvm::toString(rank12.takeError()));

  NumericGemmAxes noncanonical = canonicalAxes(3);
  noncanonical.lhsBatchDimensions.front() = 1;
  error = expectError(NumericCommandKey::createNEGemm(
      kTargetProfile, lhs3, rhs3, destination3, 2, 3, 4, 5, noncanonical));
  EXPECT_NE(error.find("canonical leading batch"), std::string::npos);
  error = expectError(NumericCommandKey::createNEGemm(
      kTargetProfile, lhs3, rhs3, destination3, 2, 3, 4, 4, canonicalAxes(3)));
  EXPECT_NE(error.find("batch count"), std::string::npos);
  error = expectError(NumericCommandKey::createNEGemm(
      kTargetProfile, lhs3, rhs3, destination3, 2, 3, UINT32_C(0x10000), 5,
      canonicalAxes(3)));
  EXPECT_NE(error.find("positive uint16"), std::string::npos);

  NumericTensorKey tensorLayout = makeTensor(
      wafer::LogicalFormat::F16, NumericTensorLayout::Tensor, {5, 2, 3});
  error = expectError(NumericCommandKey::createNEGemm(
      kTargetProfile, tensorLayout, rhs3, destination3, 2, 3, 4, 5,
      canonicalAxes(3)));
  EXPECT_NE(error.find("cx or ncx"), std::string::npos);
  NumericTensorKey wrongRhs = makeTensor(wafer::LogicalFormat::F16,
                                         NumericTensorLayout::NCx, {5, 2, 4});
  error = expectError(NumericCommandKey::createNEGemm(
      kTargetProfile, lhs3, wrongRhs, destination3, 2, 3, 4, 5,
      canonicalAxes(3)));
  EXPECT_NE(error.find("canonical trailing"), std::string::npos);

  NumericTensorKey mismatchedFormat = makeTensor(
      wafer::LogicalFormat::BF16, NumericTensorLayout::NCx, {5, 3, 4});
  error = expectError(NumericCommandKey::createNEGemm(
      kTargetProfile, lhs3, mismatchedFormat, destination3, 2, 3, 4, 5,
      canonicalAxes(3)));
  EXPECT_NE(error.find("formats must match"), std::string::npos);

  NumericTensorKey rank1 =
      makeTensor(wafer::LogicalFormat::F16, NumericTensorLayout::Cx, {2});
  error = expectError(NumericCommandKey::createNEGemm(
      kTargetProfile, rank1, rank1, rank1, 1, 1, 1, 1, NumericGemmAxes{}));
  EXPECT_NE(error.find("rank of at least 2"), std::string::npos);

  std::vector<uint64_t> shape13(13, 1);
  NumericTensorKey rank13 =
      makeTensor(wafer::LogicalFormat::F16, NumericTensorLayout::NCx, shape13);
  llvm::Expected<NumericCommandKey> rank13Command =
      NumericCommandKey::createNEGemm(kTargetProfile, rank13, rank13, rank13, 1,
                                      1, 1, 1, canonicalAxes(13));
  EXPECT_TRUE(static_cast<bool>(rank13Command))
      << (rank13Command ? std::string()
                        : llvm::toString(rank13Command.takeError()));
  llvm::Expected<ResolvedNumericCommand> rank13Resolution =
      wafer::resolveNumericCommand(kModelProfile, std::move(*rank13Command));
  ASSERT_TRUE(static_cast<bool>(rank13Resolution))
      << (rank13Resolution ? std::string()
                           : llvm::toString(rank13Resolution.takeError()));
  ASSERT_NE(rank13Resolution->getPattern().getNEGemmSelector(), nullptr);
  EXPECT_EQ(rank13Resolution->getPattern().getNEGemmSelector()->format,
            wafer::LogicalFormat::F16);

  NumericTensorKey overflowingBatch = makeTensor(
      wafer::LogicalFormat::F16, NumericTensorLayout::NCx, {65536, 1, 1});
  error = expectError(NumericCommandKey::createNEGemm(
      kTargetProfile, overflowingBatch, overflowingBatch, overflowingBatch, 1,
      1, 1, 65535, canonicalAxes(3)));
  EXPECT_NE(error.find("batch product must fit uint16"), std::string::npos);

  NumericTensorKey mismatchedBatch = makeTensor(
      wafer::LogicalFormat::F16, NumericTensorLayout::NCx, {6, 3, 4});
  error = expectError(NumericCommandKey::createNEGemm(
      kTargetProfile, lhs3, mismatchedBatch, destination3, 2, 3, 4, 5,
      canonicalAxes(3)));
  EXPECT_NE(error.find("leading batch dimensions must match"),
            std::string::npos);
}

TEST(NumericSemanticsTest, ReduceClosesFourKindsByFourLegalFormats) {
  std::set<const wafer::NumericCapabilityPattern *> patterns;
  size_t commands = 0;
  for (NumericReduceOperation operation : wafer::getNumericReduceOperations()) {
    for (wafer::LogicalFormat format : kComputeFormats) {
      NumericTensorKey input =
          makeTensor(format, NumericTensorLayout::NCx, {2, 3, 4, 5});
      NumericTensorKey destination =
          makeTensor(format, NumericTensorLayout::NCx, {2, 3, 4});
      llvm::Expected<NumericCommandKey> key =
          NumericCommandKey::createNativeCTReduce(
              kTargetProfile, operation, std::move(input),
              std::move(destination), NativeCTReduceDimension::Trailing0);
      ASSERT_TRUE(static_cast<bool>(key))
          << (key ? std::string() : llvm::toString(key.takeError()));
      EXPECT_EQ(key->getFamily(), NumericCommandFamily::NativeCTReduce);
      ASSERT_NE(key->getNativeCTReduce(), nullptr);
      EXPECT_EQ(key->getCTConvert(), nullptr);
      EXPECT_EQ(key->getCTElementwise(), nullptr);
      EXPECT_EQ(key->getNEGemm(), nullptr);
      llvm::Expected<ResolvedNumericCommand> resolution =
          wafer::resolveNumericCommand(kModelProfile, std::move(*key));
      ASSERT_TRUE(static_cast<bool>(resolution))
          << (resolution ? std::string()
                         : llvm::toString(resolution.takeError()));
      const bool supported = operation == NumericReduceOperation::Sum &&
                             format == wafer::LogicalFormat::F32;
      EXPECT_EQ(resolution->isSupported(), supported);
      EXPECT_EQ(resolution->getPattern().getModelCapability().reason,
                supported ? NumericModelImplementationReason::None
                          : NumericModelImplementationReason::
                                NativeReductionPolicyUnproven);
      patterns.insert(&resolution->getPattern());
      ++commands;
    }
  }
  EXPECT_EQ(commands, 16u);
  EXPECT_EQ(patterns.size(), 16u);

  NumericTensorKey rank3 = makeTensor(wafer::LogicalFormat::F32,
                                      NumericTensorLayout::NCx, {2, 3, 4});
  NumericTensorKey trailing =
      makeTensor(wafer::LogicalFormat::F32, NumericTensorLayout::Cx, {4});
  llvm::Expected<NumericCommandKey> combined =
      NumericCommandKey::createNativeCTReduce(
          kTargetProfile, NumericReduceOperation::Sum, rank3, trailing,
          NativeCTReduceDimension::Trailing2And1);
  EXPECT_TRUE(static_cast<bool>(combined))
      << (combined ? std::string() : llvm::toString(combined.takeError()));

  NumericTensorKey scalar =
      makeTensor(wafer::LogicalFormat::F32, NumericTensorLayout::Cx, {});
  llvm::Expected<NumericCommandKey> all =
      NumericCommandKey::createNativeCTReduce(
          kTargetProfile, NumericReduceOperation::Sum, rank3, scalar,
          NativeCTReduceDimension::Trailing2And1And0);
  EXPECT_TRUE(static_cast<bool>(all))
      << (all ? std::string() : llvm::toString(all.takeError()));
}

TEST(NumericSemanticsTest, ReduceRejectsInvalidDimShapeLayoutAndFormat) {
  NumericTensorKey input = makeTensor(wafer::LogicalFormat::F16,
                                      NumericTensorLayout::NCx, {2, 3, 4});
  NumericTensorKey goodDestination =
      makeTensor(wafer::LogicalFormat::F16, NumericTensorLayout::Cx, {2, 3});
  std::string error = expectError(NumericCommandKey::createNativeCTReduce(
      kTargetProfile, NumericReduceOperation::Sum, input, goodDestination,
      NativeCTReduceDimension::Trailing3));
  EXPECT_NE(error.find("not valid for the input rank"), std::string::npos);

  NumericTensorKey rank5 = makeTensor(
      wafer::LogicalFormat::F16, NumericTensorLayout::NCx, {2, 3, 4, 5, 6});
  NumericTensorKey rank5Destination = makeTensor(
      wafer::LogicalFormat::F16, NumericTensorLayout::NCx, {2, 4, 5, 6});
  error = expectError(NumericCommandKey::createNativeCTReduce(
      kTargetProfile, NumericReduceOperation::Sum, rank5, rank5Destination,
      NativeCTReduceDimension::Trailing3));
  EXPECT_NE(error.find("rank must be in [1, 4]"), std::string::npos);

  NumericTensorKey wrongShape =
      makeTensor(wafer::LogicalFormat::F16, NumericTensorLayout::Cx, {3, 2});
  error = expectError(NumericCommandKey::createNativeCTReduce(
      kTargetProfile, NumericReduceOperation::Sum, input, wrongShape,
      NativeCTReduceDimension::Trailing0));
  EXPECT_NE(error.find("non-reduced"), std::string::npos);

  NumericTensorKey wrongLayout =
      makeTensor(wafer::LogicalFormat::F16, NumericTensorLayout::Cx, {2, 3, 4});
  error = expectError(NumericCommandKey::createNativeCTReduce(
      kTargetProfile, NumericReduceOperation::Sum, wrongLayout, goodDestination,
      NativeCTReduceDimension::Trailing0));
  EXPECT_NE(error.find("rank > 2 requires ncx"), std::string::npos);

  NumericTensorKey booleanInput = makeTensor(
      wafer::LogicalFormat::Bool, NumericTensorLayout::NCx, {2, 3, 4});
  NumericTensorKey booleanDestination =
      makeTensor(wafer::LogicalFormat::Bool, NumericTensorLayout::Cx, {2, 3});
  error = expectError(NumericCommandKey::createNativeCTReduce(
      kTargetProfile, NumericReduceOperation::Sum, booleanInput,
      booleanDestination, NativeCTReduceDimension::Trailing0));
  EXPECT_NE(error.find("does not accept"), std::string::npos);
  error = expectError(NumericCommandKey::createNativeCTReduce(
      kTargetProfile, static_cast<NumericReduceOperation>(255), input,
      goodDestination, NativeCTReduceDimension::Trailing0));
  EXPECT_NE(error.find("unknown numeric reduce"), std::string::npos);
  error = expectError(NumericCommandKey::createNativeCTReduce(
      kTargetProfile, NumericReduceOperation::Sum, input, goodDestination,
      static_cast<NativeCTReduceDimension>(255)));
  EXPECT_NE(error.find("unknown native CT reduce"), std::string::npos);
}

TEST(NumericSemanticsTest,
     NewFamilyShapeAndDimensionChangeOnlyExactResolutionIdentity) {
  NumericTensorKey inputA = makeTensor(wafer::LogicalFormat::F32,
                                       NumericTensorLayout::NCx, {2, 3, 4});
  NumericTensorKey destinationA =
      makeTensor(wafer::LogicalFormat::F32, NumericTensorLayout::Cx, {2, 3});
  NumericTensorKey inputB = makeTensor(wafer::LogicalFormat::F32,
                                       NumericTensorLayout::NCx, {5, 6, 7});
  NumericTensorKey destinationB =
      makeTensor(wafer::LogicalFormat::F32, NumericTensorLayout::Cx, {5, 7});
  llvm::Expected<NumericCommandKey> keyA =
      NumericCommandKey::createNativeCTReduce(
          kTargetProfile, NumericReduceOperation::Max, inputA, destinationA,
          NativeCTReduceDimension::Trailing0);
  llvm::Expected<NumericCommandKey> keyB =
      NumericCommandKey::createNativeCTReduce(
          kTargetProfile, NumericReduceOperation::Max, inputB, destinationB,
          NativeCTReduceDimension::Trailing1);
  ASSERT_TRUE(static_cast<bool>(keyA));
  ASSERT_TRUE(static_cast<bool>(keyB));
  EXPECT_NE(keyA->getDigest(), keyB->getDigest());
  llvm::Expected<ResolvedNumericCommand> resolutionA =
      wafer::resolveNumericCommand(kModelProfile, std::move(*keyA));
  llvm::Expected<ResolvedNumericCommand> resolutionB =
      wafer::resolveNumericCommand(kModelProfile, std::move(*keyB));
  ASSERT_TRUE(static_cast<bool>(resolutionA));
  ASSERT_TRUE(static_cast<bool>(resolutionB));
  EXPECT_EQ(&resolutionA->getPattern(), &resolutionB->getPattern());
  EXPECT_EQ(resolutionA->getPattern().getDigest(),
            resolutionB->getPattern().getDigest());
  EXPECT_NE(resolutionA->getDigest(), resolutionB->getDigest());
  EXPECT_EQ(resolutionA->getSemantics(), nullptr);
  EXPECT_EQ(resolutionB->getSemantics(), nullptr);
}

TEST(NumericSemanticsTest, ConvertPoliciesRemainExplicitPerCategory) {
  auto resolveProfile =
      [](uint16_t opcode) -> const wafer::NumericSemanticsProfile * {
    std::optional<NumericCommandKey> key =
        expectConvertKey(opcode, NumericConvertParameter::roundingMode(
                                     NumericRoundingMode::NearestEven));
    if (!key)
      return nullptr;
    std::optional<ResolvedNumericCommand> resolution = expectResolution(*key);
    if (!resolution || !resolution->isSupported()) {
      ADD_FAILURE() << "expected a supported deterministic resolution";
      return nullptr;
    }
    return resolution->getSemantics();
  };

  const wafer::NumericSemanticsProfile *floatToInteger = resolveProfile(163);
  ASSERT_NE(floatToInteger, nullptr);
  EXPECT_EQ(floatToInteger->getFloatToIntegerPolicy(),
            FloatToIntegerPolicy::FiniteInRangeRejectNaNInfOverflowNoWrite);
  EXPECT_EQ(floatToInteger->getFloatingNaNPolicy(),
            FloatingNaNPolicy::NotApplicable);

  const wafer::NumericSemanticsProfile *floatToFloat = resolveProfile(166);
  ASSERT_NE(floatToFloat, nullptr);
  EXPECT_EQ(floatToFloat->getFloatingNaNPolicy(),
            FloatingNaNPolicy::CanonicalPositiveQuietNaN);
  EXPECT_EQ(floatToFloat->getFloatingSubnormalPolicy(),
            FloatingSubnormalPolicy::Gradual);
  EXPECT_EQ(floatToFloat->getFloatingTininessPolicy(),
            FloatingTininessPolicy::AfterRounding);
  EXPECT_EQ(floatToFloat->getExceptionFlagPolicy(),
            NumericExceptionFlagPolicy::ModelOnly);
}

} // namespace
