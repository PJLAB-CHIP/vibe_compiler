//===- ManagedReferenceTargetModelTest.cpp - Tensor reference tests ------===//

#include "Wafer/Model/TargetBulkModel.h"

#include "Wafer/Target/FormalTensorNumeric.h"
#include "Wafer/Target/PhysicalTensorCodec.h"

#include "gtest/gtest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::model;

constexpr TargetProfileId kTargetProfile =
    TargetProfileId::waferTx81SingleCardKernelV1();
constexpr ModelProfileId kModelProfile =
    ModelProfileId::formalDeterministicV1();

NumericTensorKey makeTensor(LogicalFormat format, MemLayout layout,
                            std::vector<uint64_t> shape) {
  return llvm::cantFail(
      NumericTensorKey::create(format, layout, std::move(shape)));
}

ResolvedNumericCommand resolve(NumericCommandKey command) {
  return llvm::cantFail(
      resolveNumericCommand(kModelProfile, std::move(command)));
}

TargetModelNumericTensor makeStorage(const NumericTensorKey &key,
                                     llvm::ArrayRef<RawLogicalValue> values) {
  return {key, llvm::cantFail(packPhysicalTensorLogicalValues(key, values,
                                                              UINT8_C(0xa5)))};
}

TargetModelNumericTensor makeTemplate(const NumericTensorKey &key) {
  const uint64_t bytes = llvm::cantFail(getPhysicalTensorStorageBytes(key));
  return {key, std::vector<uint8_t>(static_cast<size_t>(bytes), UINT8_C(0xa5))};
}

std::vector<RawLogicalValue>
unpack(const TargetModelManagedReferenceResult &result) {
  return llvm::cantFail(unpackPhysicalTensorLogicalValues(
      result.destination.key, result.destination.storage));
}

template <typename T> std::string expectError(llvm::Expected<T> value) {
  if (value) {
    ADD_FAILURE() << "expected an LLVM Error";
    return {};
  }
  return llvm::toString(value.takeError());
}

std::unique_ptr<ManagedReferenceTargetModelBackend> makeBackend() {
  return llvm::cantFail(
      ManagedReferenceTargetModelBackend::create(BulkNumericWorkBudget::create(
          /*maximumTotalBytes=*/UINT64_C(1) << 20,
          /*maximumScratchpadBytes=*/UINT64_C(1) << 20,
          /*maximumReorderBytes=*/UINT64_C(1) << 20)));
}

TEST(ManagedReferenceTargetModelTest,
     FiniteF16ElementwiseMatchesFormalValuesWithoutScalarFallback) {
  NumericTensorKey key = makeTensor(LogicalFormat::F16, MemLayout::Tensor, {6});
  ResolvedNumericCommand command =
      resolve(llvm::cantFail(NumericCommandKey::createCTElementwise(
          kTargetProfile, NumericElementwiseOperation::Add, {key, key}, key)));
  const std::vector<RawLogicalValue> lhs{
      {LogicalFormat::F16, UINT64_C(0x0000)},
      {LogicalFormat::F16, UINT64_C(0x8000)},
      {LogicalFormat::F16, UINT64_C(0x3c00)},
      {LogicalFormat::F16, UINT64_C(0xc000)},
      {LogicalFormat::F16, UINT64_C(0x0001)},
      {LogicalFormat::F16, UINT64_C(0x7bff)}};
  const std::vector<RawLogicalValue> rhs{
      {LogicalFormat::F16, UINT64_C(0x8000)},
      {LogicalFormat::F16, UINT64_C(0x0000)},
      {LogicalFormat::F16, UINT64_C(0x3800)},
      {LogicalFormat::F16, UINT64_C(0x4000)},
      {LogicalFormat::F16, UINT64_C(0x0001)},
      {LogicalFormat::F16, UINT64_C(0xfbff)}};
  TargetModelNumericRequest request{
      command,
      {makeStorage(key, lhs), makeStorage(key, rhs)},
      makeTemplate(key)};
  std::unique_ptr<ManagedReferenceTargetModelBackend> backend = makeBackend();
  TargetModelManagedReferenceResult managed = llvm::cantFail(
      backend->execute(request, FormalNumericWorkBudget::create(
                                    /*maximumScalarEvaluations=*/6,
                                    /*maximumFusedMultiplyAdds=*/0)));

  std::vector<llvm::ArrayRef<RawLogicalValue>> formalInputs{lhs, rhs};
  FormalNumericExecutionContext context;
  FormalTensorNumericResult formal = llvm::cantFail(executeFormalTensorNumeric(
      context, command, formalInputs,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/6,
                                      /*maximumFusedMultiplyAdds=*/0)));
  std::vector<RawLogicalValue> values = unpack(managed);
  ASSERT_EQ(values.size(), formal.values.size());
  for (size_t index = 0; index < values.size(); ++index) {
    EXPECT_EQ(values[index].format, formal.values[index].format);
    EXPECT_EQ(values[index].bits, formal.values[index].bits);
  }
  EXPECT_EQ(managed.evidence.scalarEvaluations, 6u);
  EXPECT_FALSE(managed.evidence.environmentDigest.empty());
  EXPECT_EQ(managed.evidence.implementation,
            "native-non-nan-f16-f32-tensor-v1");
}

TEST(ManagedReferenceTargetModelTest,
     F32ToF16ConvertAndF32SumReduceMatchFormalValues) {
  std::unique_ptr<ManagedReferenceTargetModelBackend> backend = makeBackend();
  NumericTensorKey f32 = makeTensor(LogicalFormat::F32, MemLayout::Tensor, {6});
  NumericTensorKey f16 = makeTensor(LogicalFormat::F16, MemLayout::Tensor, {6});
  const std::vector<RawLogicalValue> convertInput{
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0xc0000000)},
      {LogicalFormat::F32, UINT64_C(0x3f000000)},
      {LogicalFormat::F32, UINT64_C(0x33800000)},
      {LogicalFormat::F32, UINT64_C(0x477ff000)},
      {LogicalFormat::F32, UINT64_C(0xc77ff000)}};
  ResolvedNumericCommand convert =
      resolve(llvm::cantFail(NumericCommandKey::createCTConvert(
          kTargetProfile, /*fp32_fp16=*/166, f32, f16,
          NumericConvertParameter::roundingMode(
              NumericRoundingMode::NearestEven))));
  TargetModelNumericRequest convertRequest{
      convert, {makeStorage(f32, convertInput)}, makeTemplate(f16)};
  TargetModelManagedReferenceResult converted = llvm::cantFail(backend->execute(
      convertRequest,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/6,
                                      /*maximumFusedMultiplyAdds=*/0)));
  std::vector<llvm::ArrayRef<RawLogicalValue>> convertViews{convertInput};
  FormalNumericExecutionContext convertContext;
  FormalTensorNumericResult formalConvert =
      llvm::cantFail(executeFormalTensorNumeric(
          convertContext, convert, convertViews,
          FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/6,
                                          /*maximumFusedMultiplyAdds=*/0)));
  std::vector<RawLogicalValue> convertedValues = unpack(converted);
  ASSERT_EQ(convertedValues.size(), formalConvert.values.size());
  for (size_t index = 0; index < convertedValues.size(); ++index)
    EXPECT_EQ(convertedValues[index].bits, formalConvert.values[index].bits);

  NumericTensorKey reduceInput =
      makeTensor(LogicalFormat::F32, MemLayout::Cx, {2, 2});
  NumericTensorKey reduceOutput =
      makeTensor(LogicalFormat::F32, MemLayout::Cx, {2});
  const std::vector<RawLogicalValue> reduceValues{
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0x40000000)},
      {LogicalFormat::F32, UINT64_C(0x40400000)},
      {LogicalFormat::F32, UINT64_C(0x40800000)}};
  ResolvedNumericCommand reduce =
      resolve(llvm::cantFail(NumericCommandKey::createNativeCTReduce(
          kTargetProfile, NumericReduceOperation::Sum, reduceInput,
          reduceOutput, NativeCTReduceDimension::Trailing0)));
  TargetModelNumericRequest reduceRequest{
      reduce,
      {makeStorage(reduceInput, reduceValues)},
      makeTemplate(reduceOutput)};
  TargetModelManagedReferenceResult reduced = llvm::cantFail(backend->execute(
      reduceRequest,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/4,
                                      /*maximumFusedMultiplyAdds=*/0)));
  std::vector<llvm::ArrayRef<RawLogicalValue>> reduceViews{reduceValues};
  FormalNumericExecutionContext reduceContext;
  FormalTensorNumericResult formalReduce =
      llvm::cantFail(executeFormalTensorNumeric(
          reduceContext, reduce, reduceViews,
          FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/4,
                                          /*maximumFusedMultiplyAdds=*/0)));
  std::vector<RawLogicalValue> reducedValues = unpack(reduced);
  ASSERT_EQ(reducedValues.size(), formalReduce.values.size());
  for (size_t index = 0; index < reducedValues.size(); ++index)
    EXPECT_EQ(reducedValues[index].bits, formalReduce.values[index].bits);
}

TEST(ManagedReferenceTargetModelTest,
     SupportsMaskInfinityButRejectsNaNAndScalarBudget) {
  NumericTensorKey key = makeTensor(LogicalFormat::F32, MemLayout::Tensor, {1});
  ResolvedNumericCommand command =
      resolve(llvm::cantFail(NumericCommandKey::createCTElementwise(
          kTargetProfile, NumericElementwiseOperation::Exp, {key}, key)));
  TargetModelNumericRequest request{
      command,
      {makeStorage(key, {{LogicalFormat::F32, UINT64_C(0xff800000)}})},
      makeTemplate(key)};
  std::unique_ptr<ManagedReferenceTargetModelBackend> backend = makeBackend();
  TargetModelManagedReferenceResult infinity = llvm::cantFail(backend->execute(
      request,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1,
                                      /*maximumFusedMultiplyAdds=*/0)));
  std::vector<RawLogicalValue> infinityValues = unpack(infinity);
  ASSERT_EQ(infinityValues.size(), 1u);
  EXPECT_EQ(infinityValues.front().bits, UINT64_C(0));

  request.inputs[0] =
      makeStorage(key, {{LogicalFormat::F32, UINT64_C(0x7fc00000)}});
  std::string error = expectError(
      backend->execute(request, FormalNumericWorkBudget::create(
                                    /*maximumScalarEvaluations=*/1,
                                    /*maximumFusedMultiplyAdds=*/0)));
  EXPECT_NE(error.find("non-NaN domain"), std::string::npos);

  request.inputs[0] =
      makeStorage(key, {{LogicalFormat::F32, UINT64_C(0x3f800000)}});
  error = expectError(
      backend->execute(request, FormalNumericWorkBudget::create(
                                    /*maximumScalarEvaluations=*/0,
                                    /*maximumFusedMultiplyAdds=*/0)));
  EXPECT_NE(error.find("scalar evaluation budget exceeded"), std::string::npos);
}

} // namespace
