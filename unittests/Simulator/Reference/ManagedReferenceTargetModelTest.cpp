//===- ManagedReferenceTargetModelTest.cpp - Tensor reference tests ------===//

#include "Wafer/Simulator/Reference/TargetNumericBackend.h"

#include "Wafer/Target/PhysicalTensor/PhysicalTensorCodec.h"
#include "Wafer/Simulator/Reference/FormalTensorNumeric.h"

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

PhysicalTensorDescriptor makeTensor(LogicalFormat format,
                                    PhysicalTensorLayout layout,
                                    std::vector<uint64_t> shape) {
  return llvm::cantFail(
      PhysicalTensorDescriptor::create(format, layout, std::move(shape)));
}

TargetModelNumericTensor makeStorage(const PhysicalTensorDescriptor &key,
                                     llvm::ArrayRef<RawLogicalValue> values) {
  return {key, llvm::cantFail(packPhysicalTensorLogicalValues(key, values,
                                                              UINT8_C(0xa5)))};
}

TargetModelNumericTensor makeTemplate(const PhysicalTensorDescriptor &key) {
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
  return llvm::cantFail(ManagedReferenceTargetModelBackend::create(
      OneDNNNumericWorkBudget::create(
          /*maximumTotalBytes=*/UINT64_C(1) << 20,
          /*maximumScratchpadBytes=*/UINT64_C(1) << 20,
          /*maximumReorderBytes=*/UINT64_C(1) << 20)));
}

TEST(ManagedReferenceTargetModelTest,
     F32ExtremaMatchFormalIncludingZerosAndPadding) {
  auto backend = makeBackend();
  for (uint64_t extent : {1024u, 1025u, 1031u})
    for (auto kind : {TargetReduceOperation::Max, TargetReduceOperation::Min}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(static_cast<unsigned>(kind));
      auto input = makeTensor(LogicalFormat::F32, PhysicalTensorLayout::NCx,
                              {1, 1, 3, extent});
      auto output = makeTensor(LogicalFormat::F32, PhysicalTensorLayout::NCx,
                               {1, 1, 3, 1});
      auto operation = llvm::cantFail(createFormalReduceOperation(
          kind, input, output, TargetReduceDimension::Trailing0));
      std::vector<RawLogicalValue> values(
          3 * extent, {LogicalFormat::F32, UINT64_C(0xbf800000)});
      for (uint64_t i = 0; i < extent; ++i) {
        values[extent + i].bits = UINT64_C(0x3f800000);
        values[2 * extent + i].bits = i % 2 ? UINT64_C(0x80000000) : 0;
      }
      values[extent - 1].bits = UINT64_C(0xff800000);
      values[2 * extent - 1].bits = UINT64_C(0x7f800000);
      TargetModelReduceRequest request{
          operation, {{makeStorage(input, values)}, makeTemplate(output)}};
      auto result = backend->execute(
          request, FormalNumericWorkBudget::create(3 * extent, 0));
      ASSERT_TRUE(static_cast<bool>(result))
          << llvm::toString(result.takeError());
      std::vector<RawLogicalValue> expected =
          kind == TargetReduceOperation::Max
              ? std::vector<RawLogicalValue>{{LogicalFormat::F32,
                                              UINT64_C(0xbf800000)},
                                             {LogicalFormat::F32,
                                              UINT64_C(0x7f800000)},
                                             {LogicalFormat::F32, 0}}
              : std::vector<RawLogicalValue>{
                    {LogicalFormat::F32, UINT64_C(0xff800000)},
                    {LogicalFormat::F32, UINT64_C(0x3f800000)},
                    {LogicalFormat::F32, UINT64_C(0x80000000)}};
      EXPECT_EQ(result->destination.storage,
                makeStorage(output, expected).storage);
      FormalNumericExecutionContext context;
      std::vector<llvm::ArrayRef<RawLogicalValue>> inputs{values};
      auto formal = llvm::cantFail(executeFormalTensorNumeric(
          context, operation, inputs,
          FormalNumericWorkBudget::create(3 * extent, 0)));
      auto actual = unpack(*result);
      ASSERT_EQ(actual.size(), formal.values.size());
      for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_EQ(actual[i].bits, formal.values[i].bits);
      EXPECT_NE(
          expectError(backend->execute(request, FormalNumericWorkBudget::create(
                                                    3 * extent - 1, 0)))
              .find("budget"),
          std::string::npos);
      auto destinationBefore = request.tensors.destinationTemplate.storage;
      values.back().bits = UINT64_C(0x7fc12345);
      request.tensors.inputs.front() = makeStorage(input, values);
      EXPECT_NE(
          expectError(backend->execute(request, FormalNumericWorkBudget::create(
                                                    3 * extent, 0)))
              .find("NaN"),
          std::string::npos);
      EXPECT_EQ(request.tensors.destinationTemplate.storage, destinationBefore);
    }
}

TEST(ManagedReferenceTargetModelTest,
     FiniteF16ElementwiseMatchesFormalValuesWithoutScalarFallback) {
  PhysicalTensorDescriptor key =
      makeTensor(LogicalFormat::F16, PhysicalTensorLayout::Tensor, {6});
  FormalElementwiseOperation operation =
      llvm::cantFail(createFormalElementwiseOperation(
          TargetElementwiseOperation::Add, {key, key}, key));
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
  TargetModelElementwiseRequest request{
      operation,
      {{makeStorage(key, lhs), makeStorage(key, rhs)}, makeTemplate(key)}};
  std::unique_ptr<ManagedReferenceTargetModelBackend> backend = makeBackend();
  TargetModelManagedReferenceResult managed = llvm::cantFail(
      backend->execute(request, FormalNumericWorkBudget::create(
                                    /*maximumScalarEvaluations=*/6,
                                    /*maximumFusedMultiplyAdds=*/0)));

  std::vector<llvm::ArrayRef<RawLogicalValue>> formalInputs{lhs, rhs};
  FormalNumericExecutionContext context;
  FormalTensorNumericResult formal = llvm::cantFail(executeFormalTensorNumeric(
      context, operation, formalInputs,
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
  EXPECT_EQ(managed.evidence.implementation, "native-non-nan-f16-f32-tensor");
}

TEST(ManagedReferenceTargetModelTest,
     F32ToF16ConvertAndF32SumReduceMatchFormalValues) {
  std::unique_ptr<ManagedReferenceTargetModelBackend> backend = makeBackend();
  PhysicalTensorDescriptor f32 =
      makeTensor(LogicalFormat::F32, PhysicalTensorLayout::Tensor, {6});
  PhysicalTensorDescriptor f16 =
      makeTensor(LogicalFormat::F16, PhysicalTensorLayout::Tensor, {6});
  const std::vector<RawLogicalValue> convertInput{
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0xc0000000)},
      {LogicalFormat::F32, UINT64_C(0x3f000000)},
      {LogicalFormat::F32, UINT64_C(0x33800000)},
      {LogicalFormat::F32, UINT64_C(0x477ff000)},
      {LogicalFormat::F32, UINT64_C(0xc77ff000)}};
  FormalConvertOperation convert = llvm::cantFail(createFormalConvertOperation(
      /*fp32_fp16=*/166, f32, f16,
      TargetConvertParameter::roundingMode(TargetRoundingMode::NearestEven)));
  TargetModelConvertRequest convertRequest{
      convert, {{makeStorage(f32, convertInput)}, makeTemplate(f16)}};
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

  PhysicalTensorDescriptor reduceInput =
      makeTensor(LogicalFormat::F32, PhysicalTensorLayout::NCx, {1, 1, 2, 2});
  PhysicalTensorDescriptor reduceOutput =
      makeTensor(LogicalFormat::F32, PhysicalTensorLayout::NCx, {1, 1, 2, 1});
  const std::vector<RawLogicalValue> reduceValues{
      {LogicalFormat::F32, UINT64_C(0x3f800000)},
      {LogicalFormat::F32, UINT64_C(0x40000000)},
      {LogicalFormat::F32, UINT64_C(0x40400000)},
      {LogicalFormat::F32, UINT64_C(0x40800000)}};
  FormalReduceOperation reduce = llvm::cantFail(createFormalReduceOperation(
      TargetReduceOperation::Sum, reduceInput, reduceOutput,
      TargetReduceDimension::Trailing0));
  TargetModelReduceRequest reduceRequest{
      reduce,
      {{makeStorage(reduceInput, reduceValues)}, makeTemplate(reduceOutput)}};
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
  PhysicalTensorDescriptor key =
      makeTensor(LogicalFormat::F32, PhysicalTensorLayout::Tensor, {1});
  FormalElementwiseOperation operation =
      llvm::cantFail(createFormalElementwiseOperation(
          TargetElementwiseOperation::Exp, {key}, key));
  TargetModelElementwiseRequest request{
      operation,
      {{makeStorage(key, {{LogicalFormat::F32, UINT64_C(0xff800000)}})},
       makeTemplate(key)}};
  std::unique_ptr<ManagedReferenceTargetModelBackend> backend = makeBackend();
  TargetModelManagedReferenceResult infinity = llvm::cantFail(backend->execute(
      request,
      FormalNumericWorkBudget::create(/*maximumScalarEvaluations=*/1,
                                      /*maximumFusedMultiplyAdds=*/0)));
  std::vector<RawLogicalValue> infinityValues = unpack(infinity);
  ASSERT_EQ(infinityValues.size(), 1u);
  EXPECT_EQ(infinityValues.front().bits, UINT64_C(0));

  request.tensors.inputs[0] =
      makeStorage(key, {{LogicalFormat::F32, UINT64_C(0x7fc00000)}});
  std::string error = expectError(
      backend->execute(request, FormalNumericWorkBudget::create(
                                    /*maximumScalarEvaluations=*/1,
                                    /*maximumFusedMultiplyAdds=*/0)));
  EXPECT_NE(error.find("non-NaN domain"), std::string::npos);

  request.tensors.inputs[0] =
      makeStorage(key, {{LogicalFormat::F32, UINT64_C(0x3f800000)}});
  error = expectError(
      backend->execute(request, FormalNumericWorkBudget::create(
                                    /*maximumScalarEvaluations=*/0,
                                    /*maximumFusedMultiplyAdds=*/0)));
  EXPECT_NE(error.find("scalar evaluation budget exceeded"), std::string::npos);
}

} // namespace
