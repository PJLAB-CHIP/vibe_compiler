//===- FormalTensorNumericTest.cpp - Atomic tensor numeric tests ----------===//

#include "Wafer/Target/FormalTensorNumeric.h"

#include "gtest/gtest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace wafer;

constexpr ModelProfileId kModelProfile =
    ModelProfileId::formalDeterministicV1();

static_assert(!std::is_default_constructible_v<FormalNumericWorkBudget>);

template <typename T> std::string expectError(llvm::Expected<T> value) {
  if (value) {
    ADD_FAILURE() << "expected an LLVM Error";
    return {};
  }
  return llvm::toString(value.takeError());
}

NumericTensorKey makeTensor(LogicalFormat format, MemLayout layout,
                            std::vector<uint64_t> shape) {
  llvm::Expected<NumericTensorKey> key =
      NumericTensorKey::create(format, layout, std::move(shape));
  if (key)
    return std::move(*key);
  ADD_FAILURE() << llvm::toString(key.takeError());
  return llvm::cantFail(
      NumericTensorKey::create(LogicalFormat::F32, MemLayout::Tensor, {1}));
}

ResolvedNumericCommand resolve(NumericCommandKey key) {
  llvm::Expected<ResolvedNumericCommand> resolved =
      resolveNumericCommand(kModelProfile, std::move(key));
  if (resolved)
    return std::move(*resolved);
  ADD_FAILURE() << llvm::toString(resolved.takeError());
  llvm_unreachable("test command must resolve");
}

uint16_t findConvertOpcode(LogicalFormat source, LogicalFormat destination,
                           TargetConvertParameterKind parameterKind) {
  for (const TargetConvertRoute &route : getTargetConvertRoutes())
    if (route.source == source && route.destination == destination &&
        route.parameterKind == parameterKind)
      return route.opcode;
  ADD_FAILURE() << "missing requested convert route";
  return 0;
}

ResolvedNumericCommand makeConvert(LogicalFormat source,
                                   LogicalFormat destination,
                                   std::vector<uint64_t> shape) {
  const uint16_t opcode = findConvertOpcode(
      source, destination, TargetConvertParameterKind::RoundingMode);
  NumericTensorKey sourceKey = makeTensor(source, MemLayout::Tensor, shape);
  NumericTensorKey destinationKey =
      makeTensor(destination, MemLayout::Tensor, std::move(shape));
  return resolve(llvm::cantFail(NumericCommandKey::createCTConvert(
      opcode, std::move(sourceKey), std::move(destinationKey),
      NumericConvertParameter::roundingMode(
          NumericRoundingMode::NearestEven))));
}

ResolvedNumericCommand makePlainConvert(LogicalFormat source,
                                        LogicalFormat destination,
                                        std::vector<uint64_t> shape) {
  const uint16_t opcode =
      findConvertOpcode(source, destination, TargetConvertParameterKind::None);
  NumericTensorKey sourceKey = makeTensor(source, MemLayout::Tensor, shape);
  NumericTensorKey destinationKey =
      makeTensor(destination, MemLayout::Tensor, std::move(shape));
  return resolve(llvm::cantFail(NumericCommandKey::createCTConvert(
      opcode, std::move(sourceKey), std::move(destinationKey),
      std::nullopt)));
}

ResolvedNumericCommand makeElementwise(NumericElementwiseOperation operation,
                                       LogicalFormat input,
                                       LogicalFormat destination,
                                       std::vector<uint64_t> shape) {
  const unsigned arity = getNumericElementwiseArity(operation);
  std::vector<NumericTensorKey> inputs;
  for (unsigned index = 0; index < arity; ++index)
    inputs.push_back(makeTensor(input, MemLayout::Tensor, shape));
  return resolve(llvm::cantFail(NumericCommandKey::createCTElementwise(
      operation, std::move(inputs),
      makeTensor(destination, MemLayout::Tensor, std::move(shape)))));
}

ResolvedNumericCommand makeGemm(LogicalFormat format, uint32_t batch,
                                uint32_t m, uint32_t k, uint32_t n) {
  std::vector<uint64_t> lhsShape{batch, m, k};
  std::vector<uint64_t> rhsShape{batch, k, n};
  std::vector<uint64_t> destinationShape{batch, m, n};
  return resolve(llvm::cantFail(NumericCommandKey::createNEGemm(
      makeTensor(format, MemLayout::NCx, std::move(lhsShape)),
      makeTensor(format, MemLayout::NCx, std::move(rhsShape)),
      makeTensor(format, MemLayout::NCx, std::move(destinationShape)), m, k, n,
      batch, llvm::cantFail(getCanonicalNumericGemmAxes(/*rank=*/3)))));
}

ResolvedNumericCommand makeOrientedGemm(LogicalFormat format, uint32_t m,
                                        uint32_t k, uint32_t n,
                                        GemmOrientation lhsOrientation,
                                        GemmOrientation rhsOrientation) {
  std::vector<uint64_t> lhsShape = lhsOrientation == GemmOrientation::Normal
                                       ? std::vector<uint64_t>{m, k}
                                       : std::vector<uint64_t>{k, m};
  std::vector<uint64_t> rhsShape = rhsOrientation == GemmOrientation::Normal
                                       ? std::vector<uint64_t>{k, n}
                                       : std::vector<uint64_t>{n, k};
  return resolve(llvm::cantFail(NumericCommandKey::createNEGemm(
      makeTensor(format, MemLayout::Cx, std::move(lhsShape)),
      makeTensor(format, MemLayout::Cx, std::move(rhsShape)),
      makeTensor(format, MemLayout::Cx, {m, n}), m, k, n,
      /*batchCount=*/1, llvm::cantFail(getCanonicalNumericGemmAxes(/*rank=*/2)),
      lhsOrientation, rhsOrientation)));
}

std::vector<llvm::ArrayRef<RawLogicalValue>>
views(const std::vector<std::vector<RawLogicalValue>> &storage) {
  std::vector<llvm::ArrayRef<RawLogicalValue>> result;
  result.reserve(storage.size());
  for (const std::vector<RawLogicalValue> &input : storage)
    result.push_back(input);
  return result;
}

FormalTensorNumericResult
execute(FormalNumericExecutionContext &context,
        const ResolvedNumericCommand &command,
        const std::vector<std::vector<RawLogicalValue>> &storage,
        FormalNumericWorkBudget budget = FormalNumericWorkBudget::create(
            /*maximumScalarEvaluations=*/1024,
            /*maximumFusedMultiplyAdds=*/1024)) {
  std::vector<llvm::ArrayRef<RawLogicalValue>> inputViews = views(storage);
  llvm::Expected<FormalTensorNumericResult> result =
      executeFormalTensorNumeric(context, command, inputViews, budget);
  if (result)
    return std::move(*result);
  ADD_FAILURE() << llvm::toString(result.takeError());
  return {};
}

TEST(FormalTensorNumericTest, ConvertCommitsOnlyCompleteTensorAndFlags) {
  ResolvedNumericCommand command =
      makeConvert(LogicalFormat::F32, LogicalFormat::F16, {3});
  std::vector<std::vector<RawLogicalValue>> inputs{{
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0x3f801000)},
      {LogicalFormat::F32, UINT64_C(0x7f800001)},
  }};
  FormalNumericExecutionContext context;
  FormalTensorNumericResult result = execute(context, command, inputs);
  ASSERT_EQ(result.values.size(), 3u);
  EXPECT_EQ(result.values[0].bits, UINT64_C(0x3c00));
  EXPECT_EQ(result.values[1].bits, UINT64_C(0x3c00));
  EXPECT_EQ(result.values[2].bits, UINT64_C(0x7e00));
  EXPECT_TRUE(result.flags.inexact);
  EXPECT_TRUE(result.flags.invalid);
  EXPECT_EQ(context.getAggregateFlags(), result.flags);
}

TEST(FormalTensorNumericTest,
     FloatToIntegerLateRejectLeavesInvocationContextUnchanged) {
  ResolvedNumericCommand command =
      makeConvert(LogicalFormat::F32, LogicalFormat::I32, {2});
  std::vector<std::vector<RawLogicalValue>> inputs{{
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0x7fc00000)},
  }};
  FormalNumericExecutionContext context;
  std::vector<llvm::ArrayRef<RawLogicalValue>> inputViews = views(inputs);
  std::string error = expectError(executeFormalTensorNumeric(
      context, command, inputViews,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/2,
                                      /*maximumFusedMultiplyAdds=*/0)));
  EXPECT_NE(error.find("float-to-integer-non-finite"), std::string::npos);
  EXPECT_FALSE(context.getAggregateFlags().any());
}

TEST(FormalTensorNumericTest,
     BasicRelationLogicAndMPFRFamiliesUseResolvedBackend) {
  FormalNumericExecutionContext context;
  {
    ResolvedNumericCommand add =
        makeElementwise(NumericElementwiseOperation::Add, LogicalFormat::F32,
                        LogicalFormat::F32, {2});
    FormalTensorNumericResult result =
        execute(context, add,
                {{{LogicalFormat::F32, UINT64_C(0x3f800000)},
                  {LogicalFormat::F32, UINT64_C(0x40000000)}},
                 {{LogicalFormat::F32, UINT64_C(0x40400000)},
                  {LogicalFormat::F32, UINT64_C(0x40800000)}}});
    ASSERT_EQ(result.values.size(), 2u);
    EXPECT_EQ(result.values[0].bits, UINT64_C(0x40800000));
    EXPECT_EQ(result.values[1].bits, UINT64_C(0x40c00000));
  }
  {
    ResolvedNumericCommand notEqual =
        makeElementwise(NumericElementwiseOperation::Ne, LogicalFormat::F16,
                        LogicalFormat::Bool, {2});
    FormalTensorNumericResult result =
        execute(context, notEqual,
                {{{LogicalFormat::F16, UINT64_C(0x7e00)},
                  {LogicalFormat::F16, UINT64_C(0x3c00)}},
                 {{LogicalFormat::F16, UINT64_C(0x3c00)},
                  {LogicalFormat::F16, UINT64_C(0x3c00)}}});
    ASSERT_EQ(result.values.size(), 2u);
    EXPECT_EQ(result.values[0].format, LogicalFormat::Bool);
    EXPECT_EQ(result.values[0].bits, UINT64_C(1));
    EXPECT_EQ(result.values[1].bits, UINT64_C(0));
  }
  {
    ResolvedNumericCommand logical =
        makeElementwise(NumericElementwiseOperation::LogicXor,
                        LogicalFormat::Bool, LogicalFormat::Bool, {3});
    FormalTensorNumericResult result = execute(context, logical,
                                               {{{LogicalFormat::Bool, 0},
                                                 {LogicalFormat::Bool, 1},
                                                 {LogicalFormat::Bool, 1}},
                                                {{LogicalFormat::Bool, 0},
                                                 {LogicalFormat::Bool, 0},
                                                 {LogicalFormat::Bool, 1}}});
    ASSERT_EQ(result.values.size(), 3u);
    EXPECT_EQ(result.values[0].bits, UINT64_C(0));
    EXPECT_EQ(result.values[1].bits, UINT64_C(1));
    EXPECT_EQ(result.values[2].bits, UINT64_C(0));
  }
  {
    ResolvedNumericCommand exponential =
        makeElementwise(NumericElementwiseOperation::Exp, LogicalFormat::BF16,
                        LogicalFormat::BF16, {2});
    FormalTensorNumericResult result =
        execute(context, exponential,
                {{{LogicalFormat::BF16, UINT64_C(0)},
                  {LogicalFormat::BF16, UINT64_C(0x7f81)}}});
    ASSERT_EQ(result.values.size(), 2u);
    EXPECT_EQ(result.values[0].bits, UINT64_C(0x3f80));
    EXPECT_EQ(result.values[1].bits, UINT64_C(0x7fc0));
    EXPECT_TRUE(result.flags.invalid);
  }
}

TEST(FormalTensorNumericTest,
     GemmUsesBatchRowMajorIncreasingKAndOriginalDestinationFormat) {
  ResolvedNumericCommand command =
      makeGemm(LogicalFormat::F16, /*batch=*/2, /*m=*/2, /*k=*/2, /*n=*/2);
  std::vector<std::vector<RawLogicalValue>> inputs{
      {
          {LogicalFormat::F16, UINT64_C(0x3c00)},
          {LogicalFormat::F16, UINT64_C(0x4000)},
          {LogicalFormat::F16, UINT64_C(0x4200)},
          {LogicalFormat::F16, UINT64_C(0x4400)},
          {LogicalFormat::F16, UINT64_C(0x3c00)},
          {LogicalFormat::F16, UINT64_C(0)},
          {LogicalFormat::F16, UINT64_C(0)},
          {LogicalFormat::F16, UINT64_C(0x3c00)},
      },
      {
          {LogicalFormat::F16, UINT64_C(0x3c00)},
          {LogicalFormat::F16, UINT64_C(0)},
          {LogicalFormat::F16, UINT64_C(0)},
          {LogicalFormat::F16, UINT64_C(0x3c00)},
          {LogicalFormat::F16, UINT64_C(0x4000)},
          {LogicalFormat::F16, UINT64_C(0)},
          {LogicalFormat::F16, UINT64_C(0)},
          {LogicalFormat::F16, UINT64_C(0x4000)},
      },
  };
  FormalNumericExecutionContext context;
  FormalTensorNumericResult result = execute(context, command, inputs);
  ASSERT_EQ(result.values.size(), 8u);
  const uint64_t expected[] = {
      UINT64_C(0x3c00), UINT64_C(0x4000), UINT64_C(0x4200), UINT64_C(0x4400),
      UINT64_C(0x4000), UINT64_C(0),      UINT64_C(0),      UINT64_C(0x4000)};
  for (size_t index = 0; index < result.values.size(); ++index) {
    EXPECT_EQ(result.values[index].format, LogicalFormat::F16);
    EXPECT_EQ(result.values[index].bits, expected[index]);
  }
}

TEST(FormalTensorNumericTest,
     NonSquareGemmExecutesAllTypedOperandOrientations) {
  const std::vector<RawLogicalValue> normalLhs{
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0x40000000)},
      {LogicalFormat::F32, UINT64_C(0x40400000)},
      {LogicalFormat::F32, UINT64_C(0x40800000)},
      {LogicalFormat::F32, UINT64_C(0x40a00000)},
      {LogicalFormat::F32, UINT64_C(0x40c00000)}};
  const std::vector<RawLogicalValue> transposedLhs{normalLhs[0], normalLhs[3],
                                                   normalLhs[1], normalLhs[4],
                                                   normalLhs[2], normalLhs[5]};
  const std::vector<RawLogicalValue> normalRhs{
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0)},
      {LogicalFormat::F32, UINT64_C(0)},
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0)},
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0)},
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0)},
      {LogicalFormat::F32, UINT64_C(0)},
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0x3f800000)}};
  const std::vector<RawLogicalValue> transposedRhs{
      normalRhs[0],  normalRhs[4], normalRhs[8], normalRhs[1],
      normalRhs[5],  normalRhs[9], normalRhs[2], normalRhs[6],
      normalRhs[10], normalRhs[3], normalRhs[7], normalRhs[11]};
  const std::array<uint64_t, 8> expected{
      UINT64_C(0x3f800000), UINT64_C(0x40000000), UINT64_C(0x40400000),
      UINT64_C(0x40c00000), UINT64_C(0x40800000), UINT64_C(0x40a00000),
      UINT64_C(0x40c00000), UINT64_C(0x41700000)};

  for (GemmOrientation lhsOrientation :
       {GemmOrientation::Normal, GemmOrientation::Transpose}) {
    for (GemmOrientation rhsOrientation :
         {GemmOrientation::Normal, GemmOrientation::Transpose}) {
      ResolvedNumericCommand command =
          makeOrientedGemm(LogicalFormat::F32, /*m=*/2, /*k=*/3, /*n=*/4,
                           lhsOrientation, rhsOrientation);
      const std::vector<RawLogicalValue> &lhs =
          lhsOrientation == GemmOrientation::Normal ? normalLhs : transposedLhs;
      const std::vector<RawLogicalValue> &rhs =
          rhsOrientation == GemmOrientation::Normal ? normalRhs : transposedRhs;
      FormalNumericExecutionContext context;
      FormalTensorNumericResult result = execute(context, command, {lhs, rhs});
      ASSERT_EQ(result.values.size(), expected.size());
      for (size_t index = 0; index < expected.size(); ++index)
        EXPECT_EQ(result.values[index].bits, expected[index])
            << "lhs_orientation=" << static_cast<int>(lhsOrientation)
            << " rhs_orientation=" << static_cast<int>(rhsOrientation)
            << " index=" << index;
    }
  }
}

TEST(FormalTensorNumericTest,
     BudgetAndInputPreflightPrecedeEvaluationAndLeaveContextUnchanged) {
  ResolvedNumericCommand command =
      makeGemm(LogicalFormat::F32, /*batch=*/1, /*m=*/2, /*k=*/3, /*n=*/2);
  std::vector<std::vector<RawLogicalValue>> wrongSizedInputs{
      {{LogicalFormat::F32, 0}}, {{LogicalFormat::F32, 0}}};
  std::vector<llvm::ArrayRef<RawLogicalValue>> inputViews =
      views(wrongSizedInputs);
  FormalNumericExecutionContext context;
  std::string error = expectError(executeFormalTensorNumeric(
      context, command, inputViews,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/15,
                                      /*maximumFusedMultiplyAdds=*/12)));
  EXPECT_NE(error.find("scalar-work-budget-exceeded"), std::string::npos);
  EXPECT_FALSE(context.getAggregateFlags().any());

  error = expectError(executeFormalTensorNumeric(
      context, command, inputViews,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/16,
                                      /*maximumFusedMultiplyAdds=*/11)));
  EXPECT_NE(error.find("multiply-accumulate-work-budget-exceeded"),
            std::string::npos);
  EXPECT_FALSE(context.getAggregateFlags().any());

  error = expectError(executeFormalTensorNumeric(
      context, command, inputViews,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/16,
                                      /*maximumFusedMultiplyAdds=*/12)));
  EXPECT_NE(error.find("input-element-count-mismatch"), std::string::npos);
  EXPECT_FALSE(context.getAggregateFlags().any());
}

TEST(FormalTensorNumericTest,
     InvalidTF32FailsAndNativeF32SumPublishesOnlyCompleteResult) {
  FormalNumericExecutionContext context;
  ResolvedNumericCommand convert =
      makePlainConvert(LogicalFormat::TF32, LogicalFormat::F32, {1});
  std::vector<std::vector<RawLogicalValue>> noncanonicalTF32{
      {{LogicalFormat::TF32, UINT64_C(0x3f801001)}}};
  std::vector<llvm::ArrayRef<RawLogicalValue>> inputViews =
      views(noncanonicalTF32);
  std::string error = expectError(executeFormalTensorNumeric(
      context, convert, inputViews,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1,
                                      /*maximumFusedMultiplyAdds=*/0)));
  EXPECT_NE(error.find("invalid-input-encoding"), std::string::npos);
  EXPECT_FALSE(context.getAggregateFlags().any());

  ResolvedNumericCommand relation =
      makeElementwise(NumericElementwiseOperation::Eq, LogicalFormat::F32,
                      LogicalFormat::Bool, {1});
  std::vector<std::vector<RawLogicalValue>> wrongFormat{
      {{LogicalFormat::TF32, UINT64_C(0x3f801000)}},
      {{LogicalFormat::F32, UINT64_C(0x3f800000)}}};
  inputViews = views(wrongFormat);
  error = expectError(executeFormalTensorNumeric(
      context, relation, inputViews,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1,
                                      /*maximumFusedMultiplyAdds=*/0)));
  EXPECT_NE(error.find("input-format-mismatch"), std::string::npos);
  EXPECT_FALSE(context.getAggregateFlags().any());

  NumericTensorKey reduceInput =
      makeTensor(LogicalFormat::F32, MemLayout::Cx, {2, 2});
  NumericTensorKey reduceDestination =
      makeTensor(LogicalFormat::F32, MemLayout::Cx, {2});
  ResolvedNumericCommand reduce =
      resolve(llvm::cantFail(NumericCommandKey::createNativeCTReduce(
          NumericReduceOperation::Sum, std::move(reduceInput),
          std::move(reduceDestination), NativeCTReduceDimension::Trailing0)));
  std::array<RawLogicalValue, 4> values{
      RawLogicalValue{LogicalFormat::F32, UINT64_C(0x3f800000)},
      RawLogicalValue{LogicalFormat::F32, UINT64_C(0x40000000)},
      RawLogicalValue{LogicalFormat::F32, UINT64_C(0x40400000)},
      RawLogicalValue{LogicalFormat::F32, UINT64_C(0x40800000)}};
  std::array<llvm::ArrayRef<RawLogicalValue>, 1> reduceInputs{
      llvm::ArrayRef<RawLogicalValue>(values)};
  llvm::Expected<FormalTensorNumericResult> reduceResult =
      executeFormalTensorNumeric(
          context, reduce, reduceInputs,
          FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/4,
                                          /*maximumFusedMultiplyAdds=*/0));
  ASSERT_TRUE(static_cast<bool>(reduceResult))
      << (reduceResult ? std::string()
                       : llvm::toString(reduceResult.takeError()));
  ASSERT_EQ(reduceResult->values.size(), 2u);
  EXPECT_EQ(reduceResult->values[0].bits, UINT64_C(0x40400000));
  EXPECT_EQ(reduceResult->values[1].bits, UINT64_C(0x40e00000));
  EXPECT_FALSE(context.getAggregateFlags().any());
}

TEST(FormalTensorNumericTest, ExactComparatorIncludesBitsFormatsAndFlags) {
  FormalTensorNumericResult baseline{{{LogicalFormat::F16, UINT64_C(0x3c00)},
                                      {LogicalFormat::Bool, UINT64_C(1)}},
                                     {}};
  EXPECT_TRUE(llvm::cantFail(
      compareFormalTensorNumericResultsExact(baseline, baseline)));

  FormalTensorNumericResult differentBits = baseline;
  differentBits.values[0].bits = UINT64_C(0x3c01);
  EXPECT_FALSE(llvm::cantFail(
      compareFormalTensorNumericResultsExact(baseline, differentBits)));

  FormalTensorNumericResult differentFlags = baseline;
  differentFlags.flags.inexact = true;
  EXPECT_FALSE(llvm::cantFail(
      compareFormalTensorNumericResultsExact(baseline, differentFlags)));

  FormalTensorNumericResult invalid = baseline;
  invalid.values[1] = {LogicalFormat::Bool, UINT64_C(2)};
  invalid.flags.inexact = true;
  EXPECT_NE(
      expectError(compareFormalTensorNumericResultsExact(baseline, invalid))
          .find("invalid-comparator-operand-encoding"),
      std::string::npos);
}

} // namespace
