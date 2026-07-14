//===- ReferenceExecutorTest.cpp - Accepted-rank semantics tests ---------===//

#include "Wafer/Compiler/ReferenceExecutor.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"
#include "../../lib/Wafer/Compiler/TargetArtifactInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> bytesOf(const std::vector<float> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

std::vector<float> floatsOf(llvm::ArrayRef<uint8_t> bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::vector<float> fixedNonZeroPayload(size_t count, uint32_t &state,
                                       float divisor) {
  std::vector<float> values;
  values.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    int32_t value = static_cast<int32_t>((state >> 24) & 0x7f) - 64;
    if (value >= -3 && value <= 3)
      value += value < 0 ? -4 : 4;
    values.push_back(static_cast<float>(value) / divisor);
  }
  return values;
}

struct ResidualMlpOracle {
  std::vector<float> hidden;
  std::vector<float> output;
};

ResidualMlpOracle computeResidualMlpOracle(llvm::ArrayRef<float> input,
                                           llvm::ArrayRef<float> weight1,
                                           llvm::ArrayRef<float> bias1,
                                           llvm::ArrayRef<float> weight2,
                                           llvm::ArrayRef<float> bias2) {
  constexpr int64_t rows = 2;
  constexpr int64_t inputChannels = 2;
  constexpr int64_t hiddenChannels = 3;
  constexpr int64_t outputChannels = 2;
  ResidualMlpOracle oracle;
  oracle.hidden.resize(rows * hiddenChannels);
  oracle.output.resize(rows * outputChannels);
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t hidden = 0; hidden < hiddenChannels; ++hidden) {
      float value = 0.0f;
      for (int64_t channel = 0; channel < inputChannels; ++channel)
        value += input[row * inputChannels + channel] *
                 weight1[channel * hiddenChannels + hidden];
      oracle.hidden[row * hiddenChannels + hidden] =
          std::tanh(value + bias1[hidden]);
    }
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t output = 0; output < outputChannels; ++output) {
      float value = 0.0f;
      for (int64_t hidden = 0; hidden < hiddenChannels; ++hidden)
        value += oracle.hidden[row * hiddenChannels + hidden] *
                 weight2[hidden * outputChannels + output];
      oracle.output[row * outputChannels + output] =
          value + bias2[output] + input[row * inputChannels + output];
    }
  return oracle;
}

bool differs(llvm::ArrayRef<float> lhs, llvm::ArrayRef<float> rhs,
             float threshold = 1.0e-6f) {
  for (auto [left, right] : llvm::zip_equal(lhs, rhs))
    if (std::abs(left - right) > threshold)
      return true;
  return false;
}

wafer::frontend::ProgramRankSlice
singleRankSlice(llvm::ArrayRef<int64_t> shape) {
  wafer::frontend::ProgramRankSlice slice;
  slice.logicalRank = 0;
  slice.replicaId = 0;
  slice.offsets.assign(shape.size(), 0);
  slice.sizes.assign(shape.begin(), shape.end());
  slice.strides.assign(shape.size(), 1);
  return slice;
}

wafer::frontend::ProgramBoundaryBinding boundary(int64_t index,
                                                 bool output = false) {
  wafer::frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  binding.globalShape = {8};
  binding.localShape = {8};
  binding.dtype = "f32";
  binding.rankSlices.push_back(singleRankSlice({8}));
  (void)output;
  return binding;
}

wafer::frontend::ProgramBoundaryBinding
shapedBoundary(int64_t index, llvm::ArrayRef<int64_t> shape) {
  wafer::frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  binding.globalShape.assign(shape.begin(), shape.end());
  binding.localShape.assign(shape.begin(), shape.end());
  binding.dtype = "f32";
  binding.rankSlices.push_back(singleRankSlice(shape));
  return binding;
}

wafer::frontend::ProgramBoundaryBinding partitionedBoundary(int64_t index) {
  wafer::frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = wafer::frontend::ProgramDistributionKind::Partitioned;
  binding.globalShape = {64};
  binding.localShape = {4};
  binding.dtype = "f32";
  for (int64_t rank = 0; rank < 16; ++rank) {
    wafer::frontend::ProgramRankSlice slice;
    slice.logicalRank = rank;
    slice.replicaId = 0;
    slice.offsets = {rank * 4};
    slice.sizes = {4};
    slice.strides = {1};
    binding.rankSlices.push_back(std::move(slice));
  }
  return binding;
}

wafer::frontend::ProgramBoundaryBinding replicatedBoundary16(int64_t index) {
  wafer::frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  binding.globalShape = {4};
  binding.localShape = {4};
  binding.dtype = "f32";
  for (int64_t rank = 0; rank < 16; ++rank) {
    wafer::frontend::ProgramRankSlice slice;
    slice.logicalRank = rank;
    slice.replicaId = rank;
    slice.offsets = {0};
    slice.sizes = {4};
    slice.strides = {1};
    binding.rankSlices.push_back(std::move(slice));
  }
  return binding;
}

wafer::frontend::ProgramParameterBinding
parameter(int64_t argumentIndex, llvm::ArrayRef<int64_t> shape) {
  wafer::frontend::ProgramParameterBinding binding;
  binding.argumentIndex = argumentIndex;
  binding.name = "parameter" + std::to_string(argumentIndex);
  binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  binding.globalShape.assign(shape.begin(), shape.end());
  binding.localShape.assign(shape.begin(), shape.end());
  binding.dtype = "f32";
  binding.rankSlices.push_back(singleRankSlice(shape));
  return binding;
}

TEST(ReferenceExecutorTest,
     ExecutesAcceptedElementwiseRankAndReturnsFullTensor) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto grouped = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>, policy = "explicit", shape = array<i64: 1>, topology = @default}
  func.func private @choose_first(%lhs: tensor<8xf32>,
                                  %rhs: tensor<8xf32>) -> tensor<8xf32> {
    return %lhs : tensor<8xf32>
  }
  func.func @main(%lhs: tensor<8xf32>,
                  %rhs: tensor<8xf32>) -> tensor<8xf32> {
    %selected = func.call @choose_first(%lhs, %rhs)
        : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %out = tensor.empty() : tensor<8xf32>
    %group = wafer.group ins(%selected, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%out : tensor<8xf32>) {
    ^bb0(%lhs_arg: tensor<8xf32>, %rhs_arg: tensor<8xf32>,
         %out_arg: tensor<8xf32>):
      %sum = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%lhs_arg, %rhs_arg : tensor<8xf32>, tensor<8xf32>)
          outs(%out_arg : tensor<8xf32>) {
        ^bb0(%a: f32, %b: f32, %old: f32):
          %value = arith.addf %a, %b : f32
          linalg.yield %value : f32
        } -> tensor<8xf32>
      wafer.group.yield %sum : tensor<8xf32>
    } : tensor<8xf32>
    return %group : tensor<8xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(grouped);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0), boundary(1)};
  program.distributedOutputs = {boundary(0, true)};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  ASSERT_TRUE(static_cast<bool>(config));
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  auto bundle = wafer::compiler::detail::buildExecutableBundle(
      context, *grouped, std::move(program), *config, diagnostics,
      std::nullopt);
  if (!bundle)
    FAIL() << diagnosticsText << llvm::toString(bundle.takeError());
  // The accepted bundle now owns the shared context; destroy the source clone
  // while that context is still alive.
  grouped = nullptr;
  ASSERT_EQ(bundle->getRankExecutables().size(), 1u);
  EXPECT_TRUE(mlir::succeeded(wafer::compiler::detail::lowerTargetABIForTesting(
      bundle->getRankExecutables().front(), bundle->getExecutionConfig())));

  auto lhs = wafer::compiler::ReferenceTensor::create(
      "f32", {8}, bytesOf({1, 2, 3, 4, 5, 6, 7, 8}));
  auto rhs = wafer::compiler::ReferenceTensor::create(
      "f32", {8}, bytesOf({8, 7, 6, 5, 4, 3, 2, 1}));
  ASSERT_TRUE(static_cast<bool>(lhs));
  ASSERT_TRUE(static_cast<bool>(rhs));
  std::vector<wafer::compiler::ReferenceInputBinding> inputs;
  inputs.push_back(
      {wafer::compiler::ProgramResourceRole::UserInput, 0, std::move(*lhs)});
  inputs.push_back(
      {wafer::compiler::ProgramResourceRole::UserInput, 1, std::move(*rhs)});
  auto result =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, inputs);
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_EQ(result->getOutputs().size(), 1u);
  EXPECT_EQ(floatsOf(result->getOutputs().front().tensor.getBytes()),
            std::vector<float>(8, 9.0f));

  auto prepared =
      wafer::compiler::prepareReferenceRank(*bundle, /*logicalRank=*/0);
  if (!prepared)
    FAIL() << llvm::toString(prepared.takeError());
  EXPECT_EQ(prepared->getLogicalRank(), 0);
  EXPECT_GT(prepared->getProjectedOperationCount(), 0u);

  mlir::func::FuncOp helper =
      bundle->getRankExecutables()
          .front()
          .getModule()
          .lookupSymbol<mlir::func::FuncOp>("choose_first");
  ASSERT_TRUE(helper);
  mlir::func::ReturnOp helperReturn = mlir::cast<mlir::func::ReturnOp>(
      helper.getBody().front().getTerminator());
  helperReturn->setOperand(0, helper.getArgument(1));
  auto immutableCallResult =
      wafer::compiler::executeReferenceProgram(*prepared, inputs);
  if (!immutableCallResult)
    FAIL() << llvm::toString(immutableCallResult.takeError());
  EXPECT_EQ(
      floatsOf(immutableCallResult->getOutputs().front().tensor.getBytes()),
      std::vector<float>(8, 9.0f));
  auto mutatedCallResult =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, inputs);
  if (!mutatedCallResult)
    FAIL() << llvm::toString(mutatedCallResult.takeError());
  EXPECT_EQ(floatsOf(mutatedCallResult->getOutputs().front().tensor.getBytes()),
            std::vector<float>({16, 14, 12, 10, 8, 6, 4, 2}));
  helperReturn->setOperand(0, helper.getArgument(0));

  mlir::OpBuilder helperBuilder(helper.getContext());
  helperBuilder.setInsertionPoint(helperReturn);
  auto recursiveCall = helperBuilder.create<mlir::func::CallOp>(
      helper.getLoc(), helper, helper.getArguments());
  auto recursiveProgram =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, {});
  ASSERT_FALSE(static_cast<bool>(recursiveProgram));
  std::string recursiveMessage = llvm::toString(recursiveProgram.takeError());
  EXPECT_NE(recursiveMessage.find("capability preflight"), std::string::npos);
  EXPECT_NE(recursiveMessage.find("recursive"), std::string::npos);
  recursiveCall.erase();

  mlir::func::CallOp directCall;
  bundle->getRankExecutables().front().getModule().walk(
      [&](mlir::func::CallOp call) {
        if (!directCall && call.getCallee() == "choose_first")
          directCall = call;
      });
  ASSERT_TRUE(directCall);
  mlir::Attribute originalCallee = directCall.getCalleeAttr();
  directCall->setAttr("callee", mlir::FlatSymbolRefAttr::get(
                                    directCall.getContext(), "missing"));
  auto unresolvedProgram =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, {});
  ASSERT_FALSE(static_cast<bool>(unresolvedProgram));
  std::string unresolvedMessage = llvm::toString(unresolvedProgram.takeError());
  EXPECT_NE(unresolvedMessage.find("capability preflight"), std::string::npos);
  EXPECT_NE(unresolvedMessage.find("unresolved"), std::string::npos);
  directCall->setAttr("callee", originalCallee);

  helperBuilder.setInsertionPoint(helperReturn);
  auto privateDDR = helperBuilder.create<mlir::memref::AllocOp>(
      helper.getLoc(),
      mlir::cast<mlir::MemRefType>(helper.getArgument(0).getType()),
      mlir::ValueRange{});
  privateDDR->setAttr(wafer::kWaferDDROffsetAttrName,
                      wafer::DDROffsetAttr::get(helper.getContext(), 4 << 20));
  auto privateDDRProgram =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, {});
  ASSERT_FALSE(static_cast<bool>(privateDDRProgram));
  std::string privateDDRMessage = llvm::toString(privateDDRProgram.takeError());
  EXPECT_NE(privateDDRMessage.find("capability preflight"), std::string::npos);
  EXPECT_NE(privateDDRMessage.find("cannot own compiler-managed DDR storage"),
            std::string::npos);
  privateDDR.erase();

  wafer::InstrRDMAOp rdma;
  bundle->getRankExecutables().front().getModule().walk(
      [&](wafer::InstrRDMAOp operation) {
        if (!rdma)
          rdma = operation;
      });
  ASSERT_TRUE(rdma);
  mlir::IntegerAttr originalByteCount = rdma.getByteCountAttr();
  rdma->setAttr("byte_count",
                mlir::IntegerAttr::get(originalByteCount.getType(), 999));
  auto immutableResult =
      wafer::compiler::executeReferenceProgram(*prepared, inputs);
  if (!immutableResult)
    FAIL() << llvm::toString(immutableResult.takeError());
  ASSERT_EQ(immutableResult->getOutputs().size(), 1u);
  EXPECT_EQ(floatsOf(immutableResult->getOutputs().front().tensor.getBytes()),
            std::vector<float>(8, 9.0f));

  auto badDescriptor =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, inputs);
  ASSERT_FALSE(static_cast<bool>(badDescriptor));
  EXPECT_NE(llvm::toString(badDescriptor.takeError()).find("descriptor"),
            std::string::npos);
  rdma->setAttr("byte_count", originalByteCount);

  mlir::func::FuncOp entry = bundle->getRankExecutables()
                                 .front()
                                 .getModule()
                                 .lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(entry);
  mlir::OpBuilder builder(entry.getContext());
  builder.setInsertionPoint(entry.getBody().front().getTerminator());
  auto unsupportedConstant =
      builder.create<mlir::arith::ConstantIndexOp>(entry.getLoc(), 1);
  auto unsupportedAdd = builder.create<mlir::arith::AddIOp>(
      entry.getLoc(), unsupportedConstant, unsupportedConstant);
  auto unsupportedProgram =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, {});
  ASSERT_FALSE(static_cast<bool>(unsupportedProgram));
  std::string unsupportedMessage =
      llvm::toString(unsupportedProgram.takeError());
  EXPECT_NE(unsupportedMessage.find("capability preflight"), std::string::npos);
  EXPECT_NE(unsupportedMessage.find("arith.addi"), std::string::npos);
  unsupportedAdd.erase();
  unsupportedConstant.erase();

  mlir::Value originalRDMASource = rdma.getSource();
  auto sourceType = mlir::cast<mlir::MemRefType>(originalRDMASource.getType());
  builder.setInsertionPoint(rdma);
  llvm::SmallVector<mlir::OpFoldResult> offsets(sourceType.getRank(),
                                                builder.getIndexAttr(0));
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides(sourceType.getRank(),
                                                builder.getIndexAttr(1));
  for (int64_t dim : sourceType.getShape())
    sizes.push_back(builder.getIndexAttr(dim));
  auto fullSubview = builder.create<mlir::memref::SubViewOp>(
      rdma.getLoc(), originalRDMASource, offsets, sizes, strides);
  rdma.getSourceMutable().assign(fullSubview.getResult());
  auto viewProgram =
      wafer::compiler::prepareReferenceRank(*bundle, /*logicalRank=*/0);
  if (!viewProgram)
    FAIL() << llvm::toString(viewProgram.takeError());
  auto viewResult =
      wafer::compiler::executeReferenceProgram(*viewProgram, inputs);
  if (!viewResult)
    FAIL() << llvm::toString(viewResult.takeError());
  EXPECT_EQ(floatsOf(viewResult->getOutputs().front().tensor.getBytes()),
            std::vector<float>(8, 9.0f));
  rdma.getSourceMutable().assign(originalRDMASource);
  fullSubview.erase();

  wafer::InstrElementwiseOp add;
  bundle->getRankExecutables().front().getModule().walk(
      [&](wafer::InstrElementwiseOp operation) {
        if (!add && operation.getKind() == wafer::InstrElementwiseKind::Add &&
            operation.getInputs().size() == 2)
          add = operation;
      });
  ASSERT_TRUE(add);
  mlir::Value originalAddInput = add.getInputs().front();
  auto addInputType = mlir::cast<mlir::MemRefType>(originalAddInput.getType());
  auto i32Type = mlir::MemRefType::get(
      addInputType.getShape(), builder.getI32Type(), addInputType.getLayout(),
      addInputType.getMemorySpace());
  builder.setInsertionPoint(add);
  auto roundedInteger = builder.create<mlir::memref::AllocOp>(
      add.getLoc(), i32Type, mlir::ValueRange{});
  roundedInteger->setAttr(wafer::kWaferSPMOffsetAttrName,
                          wafer::SPMOffsetAttr::get(add.getContext(), 1 << 20));
  auto roundedFloat = builder.create<mlir::memref::AllocOp>(
      add.getLoc(), addInputType, mlir::ValueRange{});
  roundedFloat->setAttr(wafer::kWaferSPMOffsetAttrName,
                        wafer::SPMOffsetAttr::get(add.getContext(), 2 << 20));
  auto toInteger = builder.create<wafer::InstrConvertOp>(
      add.getLoc(), wafer::InstrConvertKind::Fp32Int32, originalAddInput,
      roundedInteger, /*zero_point=*/mlir::IntegerAttr{},
      builder.getI64IntegerAttr(0));
  auto backToFloat = builder.create<wafer::InstrConvertOp>(
      add.getLoc(), wafer::InstrConvertKind::Int32Fp32, roundedInteger,
      roundedFloat, /*zero_point=*/mlir::IntegerAttr{},
      builder.getI64IntegerAttr(0));
  add->setOperand(0, roundedFloat);

  std::vector<wafer::compiler::ReferenceInputBinding> roundingInputs = inputs;
  auto fractional = wafer::compiler::ReferenceTensor::create(
      "f32", {8},
      bytesOf({1.5f, -1.5f, 2.5f, -2.5f, 0.5f, -0.5f, 10.25f, -10.25f}));
  auto zeros = wafer::compiler::ReferenceTensor::create(
      "f32", {8}, bytesOf(std::vector<float>(8, 0.0f)));
  ASSERT_TRUE(static_cast<bool>(fractional));
  ASSERT_TRUE(static_cast<bool>(zeros));
  roundingInputs[0].tensor = std::move(*fractional);
  roundingInputs[1].tensor = std::move(*zeros);

  struct RoundingCase {
    int64_t mode;
    std::vector<float> expected;
  };
  const std::vector<RoundingCase> roundingCases = {
      {0, {2, -2, 2, -2, 0, 0, 10, -10}},
      {1, {1, -1, 2, -2, 0, 0, 10, -10}},
      {2, {2, -1, 3, -2, 1, 0, 11, -10}},
      {3, {1, -2, 2, -3, 0, -1, 10, -11}},
  };
  for (const RoundingCase &roundingCase : roundingCases) {
    toInteger.setRoundingMode(roundingCase.mode);
    backToFloat.setRoundingMode(roundingCase.mode);
    auto roundingProgram =
        wafer::compiler::prepareReferenceRank(*bundle, /*logicalRank=*/0);
    if (!roundingProgram)
      FAIL() << llvm::toString(roundingProgram.takeError());
    auto roundingResult = wafer::compiler::executeReferenceProgram(
        *roundingProgram, roundingInputs);
    if (!roundingResult)
      FAIL() << llvm::toString(roundingResult.takeError());
    EXPECT_EQ(floatsOf(roundingResult->getOutputs().front().tensor.getBytes()),
              roundingCase.expected);
  }

  toInteger.setRoundingMode(2);
  backToFloat.setRoundingMode(2);
  auto immutableRoundingProgram =
      wafer::compiler::prepareReferenceRank(*bundle, /*logicalRank=*/0);
  if (!immutableRoundingProgram)
    FAIL() << llvm::toString(immutableRoundingProgram.takeError());
  toInteger.setRoundingMode(1);
  backToFloat.setRoundingMode(1);
  auto immutableRoundingResult = wafer::compiler::executeReferenceProgram(
      *immutableRoundingProgram, roundingInputs);
  if (!immutableRoundingResult)
    FAIL() << llvm::toString(immutableRoundingResult.takeError());
  EXPECT_EQ(
      floatsOf(immutableRoundingResult->getOutputs().front().tensor.getBytes()),
      roundingCases[2].expected);

  auto nonFinite = wafer::compiler::ReferenceTensor::create(
      "f32", {8},
      bytesOf({std::numeric_limits<float>::infinity(), 0, 0, 0, 0, 0, 0, 0}));
  ASSERT_TRUE(static_cast<bool>(nonFinite));
  auto nonFiniteInputs = roundingInputs;
  nonFiniteInputs[0].tensor = std::move(*nonFinite);
  auto nonFiniteResult = wafer::compiler::executeReferenceProgram(
      *immutableRoundingProgram, nonFiniteInputs);
  ASSERT_FALSE(static_cast<bool>(nonFiniteResult));
  EXPECT_NE(llvm::toString(nonFiniteResult.takeError()).find("NaN, Inf"),
            std::string::npos);

  toInteger.setRoundingMode(4);
  auto missingStochasticSeed =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, {});
  ASSERT_FALSE(static_cast<bool>(missingStochasticSeed));
  std::string stochasticMessage =
      llvm::toString(missingStochasticSeed.takeError());
  EXPECT_NE(stochasticMessage.find("explicit execution seed"),
            std::string::npos);

  auto stochasticProgram =
      wafer::compiler::prepareReferenceRank(*bundle, /*logicalRank=*/0);
  if (!stochasticProgram)
    FAIL() << llvm::toString(stochasticProgram.takeError());
  wafer::compiler::ReferenceExecutionOptions stochasticOptions;
  stochasticOptions.stochasticSeed = 12345;
  auto stochastic = wafer::compiler::executeReferenceProgram(
      *stochasticProgram, roundingInputs, stochasticOptions);
  if (!stochastic)
    FAIL() << llvm::toString(stochastic.takeError());
  EXPECT_EQ(floatsOf(stochastic->getOutputs().front().tensor.getBytes()),
            std::vector<float>({2, -1, 3, -2, 0, 0, 11, -10}));
  auto repeatedStochastic = wafer::compiler::executeReferenceProgram(
      *stochasticProgram, roundingInputs, stochasticOptions);
  if (!repeatedStochastic)
    FAIL() << llvm::toString(repeatedStochastic.takeError());
  EXPECT_EQ(repeatedStochastic->getOutputs().front().tensor.getBytes(),
            stochastic->getOutputs().front().tensor.getBytes());

  toInteger.setRoundingMode(0);
  auto f16Type = mlir::MemRefType::get(
      addInputType.getShape(), builder.getF16Type(), addInputType.getLayout(),
      addInputType.getMemorySpace());
  builder.setInsertionPoint(add);
  auto stochasticHalf = builder.create<mlir::memref::AllocOp>(
      add.getLoc(), f16Type, mlir::ValueRange{});
  stochasticHalf->setAttr(wafer::kWaferSPMOffsetAttrName,
                          wafer::SPMOffsetAttr::get(add.getContext(), 3 << 20));
  auto floatToHalf = builder.create<wafer::InstrConvertOp>(
      add.getLoc(), wafer::InstrConvertKind::Fp32Fp16, originalAddInput,
      stochasticHalf, /*zero_point=*/mlir::IntegerAttr{},
      builder.getI64IntegerAttr(4));
  auto halfToFloat = builder.create<wafer::InstrConvertOp>(
      add.getLoc(), wafer::InstrConvertKind::Fp16Fp32, stochasticHalf,
      roundedFloat, /*zero_point=*/mlir::IntegerAttr{},
      /*rounding_mode=*/mlir::IntegerAttr{});
  add->setOperand(0, roundedFloat);
  auto floatingStochasticInput = wafer::compiler::ReferenceTensor::create(
      "f32", {8},
      bytesOf({1.0004f, 1.0006f, 1.00048828125f, -1.0004f, -1.0006f, 2.0008f,
               2.0012f, -2.0008f}));
  ASSERT_TRUE(static_cast<bool>(floatingStochasticInput));
  auto floatingStochasticInputs = roundingInputs;
  floatingStochasticInputs[0].tensor = std::move(*floatingStochasticInput);
  auto floatingStochastic = wafer::compiler::executeReferenceRank(
      *bundle, /*logicalRank=*/0, floatingStochasticInputs, stochasticOptions);
  if (!floatingStochastic)
    FAIL() << llvm::toString(floatingStochastic.takeError());
  EXPECT_EQ(
      floatsOf(floatingStochastic->getOutputs().front().tensor.getBytes()),
      std::vector<float>({1.0009765625f, 1.0009765625f, 1.0009765625f, -1.0f,
                          -1.0009765625f, 2.001953125f, 2.001953125f, -2.0f}));

  floatToHalf.setRoundingMode(0);
  auto integerToHalf = builder.create<wafer::InstrConvertOp>(
      add.getLoc(), wafer::InstrConvertKind::Int32Fp16, roundedInteger,
      stochasticHalf, /*zero_point=*/mlir::IntegerAttr{},
      builder.getI64IntegerAttr(4));
  auto integerHalfToFloat = builder.create<wafer::InstrConvertOp>(
      add.getLoc(), wafer::InstrConvertKind::Fp16Fp32, stochasticHalf,
      roundedFloat, /*zero_point=*/mlir::IntegerAttr{},
      /*rounding_mode=*/mlir::IntegerAttr{});
  auto integerStochasticInput = wafer::compiler::ReferenceTensor::create(
      "f32", {8},
      bytesOf({2049.0f, -2049.0f, 2051.0f, -2051.0f, 4097.0f, -4097.0f, 8195.0f,
               -8195.0f}));
  ASSERT_TRUE(static_cast<bool>(integerStochasticInput));
  auto integerStochasticInputs = roundingInputs;
  integerStochasticInputs[0].tensor = std::move(*integerStochasticInput);
  auto integerStochastic = wafer::compiler::executeReferenceRank(
      *bundle, /*logicalRank=*/0, integerStochasticInputs, stochasticOptions);
  if (!integerStochastic)
    FAIL() << llvm::toString(integerStochastic.takeError());
  EXPECT_EQ(
      floatsOf(integerStochastic->getOutputs().front().tensor.getBytes()),
      std::vector<float>({2050, -2048, 2052, -2050, 4096, -4096, 8200, -8192}));

  add->setOperand(0, originalAddInput);
  integerHalfToFloat.erase();
  integerToHalf.erase();
  halfToFloat.erase();
  floatToHalf.erase();
  stochasticHalf.erase();
  backToFloat.erase();
  toInteger.erase();
  roundedFloat.erase();
  roundedInteger.erase();

  size_t declaredConvertKinds = 0;
  size_t acceptedConvertKinds = 0;
  size_t rejectedZeroPointKinds = 0;
  size_t acceptedStochasticKinds = 0;
  for (uint32_t rawKind = 0;
       rawKind <= wafer::getMaxEnumValForInstrConvertKind(); ++rawKind) {
    std::optional<wafer::InstrConvertKind> kind =
        wafer::symbolizeInstrConvertKind(rawKind);
    if (!kind)
      continue;
    SCOPED_TRACE(wafer::stringifyInstrConvertKind(*kind).str());
    ++declaredConvertKinds;

    auto [sourceElement, destElement] =
        wafer::getInstrConvertTypePair(add.getContext(), *kind);
    auto matrixSourceType = mlir::MemRefType::get(
        addInputType.getShape(), sourceElement, addInputType.getLayout(),
        addInputType.getMemorySpace());
    auto matrixDestType = mlir::MemRefType::get(
        addInputType.getShape(), destElement, addInputType.getLayout(),
        addInputType.getMemorySpace());
    builder.setInsertionPoint(add);
    auto matrixSource = builder.create<mlir::memref::AllocOp>(
        add.getLoc(), matrixSourceType, mlir::ValueRange{});
    matrixSource->setAttr(wafer::kWaferSPMOffsetAttrName,
                          wafer::SPMOffsetAttr::get(add.getContext(), 1 << 20));
    auto matrixDest = builder.create<mlir::memref::AllocOp>(
        add.getLoc(), matrixDestType, mlir::ValueRange{});
    matrixDest->setAttr(wafer::kWaferSPMOffsetAttrName,
                        wafer::SPMOffsetAttr::get(add.getContext(), 2 << 20));

    mlir::IntegerAttr zeroPoint;
    mlir::IntegerAttr roundingMode;
    wafer::InstrConvertParameterKind parameterKind =
        wafer::getInstrConvertParameterKind(*kind);
    if (parameterKind == wafer::InstrConvertParameterKind::ZeroPoint)
      zeroPoint = builder.getI64IntegerAttr(0);
    else if (parameterKind == wafer::InstrConvertParameterKind::RoundingMode)
      roundingMode = builder.getI64IntegerAttr(0);
    auto matrixConvert = builder.create<wafer::InstrConvertOp>(
        add.getLoc(), *kind, matrixSource, matrixDest, zeroPoint, roundingMode);

    if (parameterKind == wafer::InstrConvertParameterKind::ZeroPoint) {
      auto rejected =
          wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, {});
      ASSERT_FALSE(static_cast<bool>(rejected));
      std::string message = llvm::toString(rejected.takeError());
      EXPECT_NE(message.find("capability preflight"), std::string::npos);
      EXPECT_NE(message.find("zero-point convert formula"), std::string::npos);
      ++rejectedZeroPointKinds;
    } else {
      auto matrixProgram =
          wafer::compiler::prepareReferenceRank(*bundle, /*logicalRank=*/0);
      if (!matrixProgram)
        FAIL() << llvm::toString(matrixProgram.takeError());
      auto matrixResult =
          wafer::compiler::executeReferenceProgram(*matrixProgram, inputs);
      if (!matrixResult)
        FAIL() << llvm::toString(matrixResult.takeError());
      EXPECT_EQ(floatsOf(matrixResult->getOutputs().front().tensor.getBytes()),
                std::vector<float>(8, 9.0f));
      ++acceptedConvertKinds;

      if (parameterKind == wafer::InstrConvertParameterKind::RoundingMode) {
        matrixConvert.setRoundingMode(4);
        auto stochasticMatrixProgram =
            wafer::compiler::prepareReferenceRank(*bundle, /*logicalRank=*/0);
        if (!stochasticMatrixProgram)
          FAIL() << llvm::toString(stochasticMatrixProgram.takeError());
        wafer::compiler::ReferenceExecutionOptions options;
        options.stochasticSeed = 12345;
        auto stochasticMatrixResult = wafer::compiler::executeReferenceProgram(
            *stochasticMatrixProgram, inputs, options);
        if (!stochasticMatrixResult)
          FAIL() << llvm::toString(stochasticMatrixResult.takeError());
        ++acceptedStochasticKinds;
      }
    }

    matrixConvert.erase();
    matrixDest.erase();
    matrixSource.erase();
  }
  EXPECT_GT(declaredConvertKinds, 0u);
  EXPECT_GT(acceptedConvertKinds, 0u);
  EXPECT_GT(rejectedZeroPointKinds, 0u);
  EXPECT_GT(acceptedStochasticKinds, 0u);
  EXPECT_EQ(acceptedConvertKinds + rejectedZeroPointKinds,
            declaredConvertKinds);

  auto tf32Type = mlir::MemRefType::get(
      addInputType.getShape(), mlir::FloatTF32Type::get(add.getContext()),
      addInputType.getLayout(), addInputType.getMemorySpace());
  builder.setInsertionPoint(add);
  auto tf32Buffer = builder.create<mlir::memref::AllocOp>(
      add.getLoc(), tf32Type, mlir::ValueRange{});
  tf32Buffer->setAttr(wafer::kWaferSPMOffsetAttrName,
                      wafer::SPMOffsetAttr::get(add.getContext(), 1 << 20));
  auto tf32RoundTrip = builder.create<mlir::memref::AllocOp>(
      add.getLoc(), addInputType, mlir::ValueRange{});
  tf32RoundTrip->setAttr(wafer::kWaferSPMOffsetAttrName,
                         wafer::SPMOffsetAttr::get(add.getContext(), 2 << 20));
  auto toTF32 = builder.create<wafer::InstrConvertOp>(
      add.getLoc(), wafer::InstrConvertKind::Fp32Tf32, originalAddInput,
      tf32Buffer, /*zero_point=*/mlir::IntegerAttr{},
      builder.getI64IntegerAttr(0));
  auto fromTF32 = builder.create<wafer::InstrConvertOp>(
      add.getLoc(), wafer::InstrConvertKind::Tf32Fp32, tf32Buffer,
      tf32RoundTrip, /*zero_point=*/mlir::IntegerAttr{},
      /*rounding_mode=*/mlir::IntegerAttr{});
  add->setOperand(0, tf32RoundTrip);
  auto tf32Input = wafer::compiler::ReferenceTensor::create(
      "f32", {8},
      bytesOf({1.0f, 1.0004f, 1.0006f, 1.00048828125f, -1.0004f, -1.0006f,
               2.0008f, 2.0012f}));
  auto tf32Zero = wafer::compiler::ReferenceTensor::create(
      "f32", {8}, bytesOf(std::vector<float>(8, 0.0f)));
  ASSERT_TRUE(static_cast<bool>(tf32Input));
  ASSERT_TRUE(static_cast<bool>(tf32Zero));
  std::vector<wafer::compiler::ReferenceInputBinding> tf32Inputs = inputs;
  tf32Inputs[0].tensor = std::move(*tf32Input);
  tf32Inputs[1].tensor = std::move(*tf32Zero);
  auto tf32Result = wafer::compiler::executeReferenceRank(
      *bundle, /*logicalRank=*/0, tf32Inputs);
  if (!tf32Result)
    FAIL() << llvm::toString(tf32Result.takeError());
  EXPECT_EQ(floatsOf(tf32Result->getOutputs().front().tensor.getBytes()),
            std::vector<float>({1.0f, 1.0f, 1.0009765625f, 1.0f, -1.0f,
                                -1.0009765625f, 2.0f, 2.001953125f}));
  add->setOperand(0, originalAddInput);
  fromTF32.erase();
  toTF32.erase();
  tf32RoundTrip.erase();
  tf32Buffer.erase();

  mlir::Value originalAddRhs = add.getInputs()[1];
  builder.setInsertionPoint(add);
  auto branchCondition = builder.create<mlir::arith::ConstantIntOp>(
      add.getLoc(), /*value=*/1, /*width=*/1);
  auto selected = builder.create<mlir::scf::IfOp>(
      add.getLoc(), mlir::TypeRange{addInputType}, branchCondition,
      /*withElseRegion=*/true);
  builder.setInsertionPointToEnd(selected.thenBlock());
  builder.create<mlir::scf::YieldOp>(add.getLoc(), originalAddInput);
  builder.setInsertionPointToEnd(selected.elseBlock());
  builder.create<mlir::scf::YieldOp>(add.getLoc(), originalAddRhs);
  add->setOperand(0, selected.getResult(0));

  auto trueBranchProgram =
      wafer::compiler::prepareReferenceRank(*bundle, /*logicalRank=*/0);
  if (!trueBranchProgram)
    FAIL() << llvm::toString(trueBranchProgram.takeError());
  branchCondition->setAttr("value", builder.getBoolAttr(false));
  auto immutableTrueBranch =
      wafer::compiler::executeReferenceProgram(*trueBranchProgram, inputs);
  if (!immutableTrueBranch)
    FAIL() << llvm::toString(immutableTrueBranch.takeError());
  EXPECT_EQ(
      floatsOf(immutableTrueBranch->getOutputs().front().tensor.getBytes()),
      std::vector<float>(8, 9.0f));
  auto falseBranch =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, inputs);
  if (!falseBranch)
    FAIL() << llvm::toString(falseBranch.takeError());
  EXPECT_EQ(floatsOf(falseBranch->getOutputs().front().tensor.getBytes()),
            std::vector<float>({16, 14, 12, 10, 8, 6, 4, 2}));
  add->setOperand(0, originalAddInput);
  selected.erase();
  branchCondition.erase();

  builder.setInsertionPoint(add);
  auto loopTemp = builder.create<mlir::memref::AllocOp>(
      add.getLoc(), addInputType, mlir::ValueRange{});
  loopTemp->setAttr(wafer::kWaferSPMOffsetAttrName,
                    wafer::SPMOffsetAttr::get(add.getContext(), 3 << 20));
  auto lowerBound =
      builder.create<mlir::arith::ConstantIndexOp>(add.getLoc(), 0);
  auto upperBound =
      builder.create<mlir::arith::ConstantIndexOp>(add.getLoc(), 0);
  auto loopStep = builder.create<mlir::arith::ConstantIndexOp>(add.getLoc(), 1);
  auto loop = builder.create<mlir::scf::ForOp>(
      add.getLoc(), lowerBound, upperBound, loopStep,
      mlir::ValueRange{originalAddInput});
  if (!loop.getBody()->empty())
    loop.getBody()->back().erase();
  builder.setInsertionPointToEnd(loop.getBody());
  builder.create<wafer::InstrElementwiseOp>(
      add.getLoc(), wafer::InstrElementwiseKind::Add,
      mlir::ValueRange{loop.getRegionIterArg(0), originalAddRhs}, loopTemp);
  builder.create<mlir::scf::YieldOp>(add.getLoc(), loopTemp.getResult());
  add->setOperand(0, loop.getResult(0));

  const std::vector<std::vector<float>> loopExpected = {
      std::vector<float>(8, 9.0f),
      {17, 16, 15, 14, 13, 12, 11, 10},
      {25, 23, 21, 19, 17, 15, 13, 11},
  };
  for (int64_t tripCount = 0; tripCount <= 2; ++tripCount) {
    upperBound->setAttr("value", builder.getIndexAttr(tripCount));
    auto loopProgram =
        wafer::compiler::prepareReferenceRank(*bundle, /*logicalRank=*/0);
    if (!loopProgram)
      FAIL() << llvm::toString(loopProgram.takeError());
    auto loopResult =
        wafer::compiler::executeReferenceProgram(*loopProgram, inputs);
    if (!loopResult)
      FAIL() << llvm::toString(loopResult.takeError());
    EXPECT_EQ(floatsOf(loopResult->getOutputs().front().tensor.getBytes()),
              loopExpected[tripCount]);
  }
  upperBound->setAttr("value", builder.getIndexAttr(2));
  auto immutableLoopProgram =
      wafer::compiler::prepareReferenceRank(*bundle, /*logicalRank=*/0);
  if (!immutableLoopProgram)
    FAIL() << llvm::toString(immutableLoopProgram.takeError());
  upperBound->setAttr("value", builder.getIndexAttr(0));
  auto immutableLoopResult =
      wafer::compiler::executeReferenceProgram(*immutableLoopProgram, inputs);
  if (!immutableLoopResult)
    FAIL() << llvm::toString(immutableLoopResult.takeError());
  EXPECT_EQ(
      floatsOf(immutableLoopResult->getOutputs().front().tensor.getBytes()),
      loopExpected[2]);
  add->setOperand(0, originalAddInput);
  loop.erase();
  loopStep.erase();
  upperBound.erase();
  lowerBound.erase();
  loopTemp.erase();

  mlir::memref::AllocOp spmAlloc;
  bundle->getRankExecutables().front().getModule().walk(
      [&](mlir::memref::AllocOp operation) {
        if (!spmAlloc && wafer::isWaferSPMMemRefType(operation.getType()))
          spmAlloc = operation;
      });
  ASSERT_TRUE(spmAlloc);
  mlir::Attribute originalOffset =
      spmAlloc->removeAttr(wafer::kWaferSPMOffsetAttrName);
  auto badOffset =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, inputs);
  ASSERT_FALSE(static_cast<bool>(badOffset));
  EXPECT_NE(llvm::toString(badOffset.takeError()).find("accepted offset"),
            std::string::npos);
  spmAlloc->setAttr(wafer::kWaferSPMOffsetAttrName, originalOffset);

  auto wrongDType = wafer::compiler::ReferenceTensor::create(
      "f16", {8}, std::vector<uint8_t>(16, 0));
  ASSERT_TRUE(static_cast<bool>(wrongDType));
  inputs.front().tensor = std::move(*wrongDType);
  auto badDType =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, inputs);
  ASSERT_FALSE(static_cast<bool>(badDType));
  EXPECT_NE(llvm::toString(badDType.takeError()).find("typed rank binding"),
            std::string::npos);

  wafer::TileRegionOp tile = add->getParentOfType<wafer::TileRegionOp>();
  ASSERT_TRUE(tile);
  mlir::Value originalTileInput = tile.getInputs().front();
  mlir::Value alternateTileInput = tile.getInputs()[1];
  mlir::Block *entryBlock = &entry.getBody().front();
  mlir::Block *mergeBlock = entryBlock->splitBlock(tile);
  mlir::Block *leftBlock =
      builder.createBlock(&entry.getBody(), mlir::Region::iterator(mergeBlock));
  mlir::Block *rightBlock =
      builder.createBlock(&entry.getBody(), mlir::Region::iterator(mergeBlock));
  mlir::BlockArgument forwarded =
      mergeBlock->addArgument(originalTileInput.getType(), tile.getLoc());
  tile->setOperand(0, forwarded);

  builder.setInsertionPointToEnd(entryBlock);
  auto cfgCondition = builder.create<mlir::arith::ConstantIntOp>(
      entry.getLoc(), /*value=*/1, /*width=*/1);
  builder.create<mlir::cf::CondBranchOp>(entry.getLoc(), cfgCondition,
                                         leftBlock, mlir::ValueRange{},
                                         rightBlock, mlir::ValueRange{});
  builder.setInsertionPointToEnd(leftBlock);
  auto leftBranch = builder.create<mlir::cf::BranchOp>(
      entry.getLoc(), mergeBlock, mlir::ValueRange{originalTileInput});
  builder.setInsertionPointToEnd(rightBlock);
  builder.create<mlir::cf::BranchOp>(entry.getLoc(), mergeBlock,
                                     mlir::ValueRange{alternateTileInput});

  auto cfgInputs = roundingInputs;
  auto trueCFGProgram =
      wafer::compiler::prepareReferenceRank(*bundle, /*logicalRank=*/0);
  if (!trueCFGProgram)
    FAIL() << llvm::toString(trueCFGProgram.takeError());
  cfgCondition->setAttr("value", builder.getBoolAttr(false));
  auto immutableTrueCFG =
      wafer::compiler::executeReferenceProgram(*trueCFGProgram, cfgInputs);
  if (!immutableTrueCFG)
    FAIL() << llvm::toString(immutableTrueCFG.takeError());
  EXPECT_EQ(floatsOf(immutableTrueCFG->getOutputs().front().tensor.getBytes()),
            floatsOf(cfgInputs[0].tensor.getBytes()));
  auto falseCFG = wafer::compiler::executeReferenceRank(
      *bundle, /*logicalRank=*/0, cfgInputs);
  if (!falseCFG)
    FAIL() << llvm::toString(falseCFG.takeError());
  EXPECT_EQ(floatsOf(falseCFG->getOutputs().front().tensor.getBytes()),
            std::vector<float>(8, 0.0f));

  leftBranch.erase();
  builder.setInsertionPointToEnd(leftBlock);
  builder.create<mlir::cf::BranchOp>(entry.getLoc(), leftBlock,
                                     mlir::ValueRange{});
  auto cyclicCFG =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, {});
  ASSERT_FALSE(static_cast<bool>(cyclicCFG));
  std::string cyclicMessage = llvm::toString(cyclicCFG.takeError());
  EXPECT_NE(cyclicMessage.find("capability preflight"), std::string::npos);
  EXPECT_NE(cyclicMessage.find("cyclic CFG"), std::string::npos);

  directCall.getResult(0).replaceAllUsesWith(directCall.getOperand(0));
  directCall.erase();
  auto unreachableHelper =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, {});
  ASSERT_FALSE(static_cast<bool>(unreachableHelper));
  std::string unreachableMessage =
      llvm::toString(unreachableHelper.takeError());
  EXPECT_NE(unreachableMessage.find("capability preflight"), std::string::npos);
  EXPECT_NE(unreachableMessage.find("outside the entry call closure"),
            std::string::npos);
}

TEST(ReferenceExecutorTest, RejectsMalformedCompactTensor) {
  auto tensor = wafer::compiler::ReferenceTensor::create("f32", {2}, {0, 0});
  ASSERT_FALSE(tensor);
  EXPECT_NE(llvm::toString(tensor.takeError()).find("byte count"),
            std::string::npos);
}

TEST(ReferenceExecutorTest,
     ExecutesOrderedReductionCompositesWithoutNativeReduce) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto grouped = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>, policy = "explicit", shape = array<i64: 1>, topology = @default}
  func.func @main(%input: tensor<7x4xf32>)
      -> (tensor<7xf32>, tensor<7xf32>, tensor<7xf32>) {
    %sum_out = tensor.empty() : tensor<7xf32>
    %sum_result = wafer.group ins(%input : tensor<7x4xf32>)
        outs(%sum_out : tensor<7xf32>) {
    ^bb0(%input_arg: tensor<7x4xf32>, %out_arg: tensor<7xf32>):
      %init_scalar = arith.constant 5.000000e-01 : f32
      %empty = tensor.empty() : tensor<7xf32>
      %init = linalg.fill ins(%init_scalar : f32)
          outs(%empty : tensor<7xf32>) -> tensor<7xf32>
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0)>
          ],
          iterator_types = ["parallel", "reduction"]
        } ins(%input_arg : tensor<7x4xf32>)
          outs(%init : tensor<7xf32>) {
        ^bb0(%value: f32, %acc: f32):
          %next = arith.addf %value, %acc : f32
          linalg.yield %next : f32
        } -> tensor<7xf32>
      wafer.group.yield %sum : tensor<7xf32>
    } : tensor<7xf32>

    %max_out = tensor.empty() : tensor<7xf32>
    %max_result = wafer.group ins(%input : tensor<7x4xf32>)
        outs(%max_out : tensor<7xf32>) {
    ^bb0(%input_arg: tensor<7x4xf32>, %out_arg: tensor<7xf32>):
      %init_scalar = arith.constant -0.000000e+00 : f32
      %empty = tensor.empty() : tensor<7xf32>
      %init = linalg.fill ins(%init_scalar : f32)
          outs(%empty : tensor<7xf32>) -> tensor<7xf32>
      %max = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0)>
          ],
          iterator_types = ["parallel", "reduction"]
        } ins(%input_arg : tensor<7x4xf32>)
          outs(%init : tensor<7xf32>) {
        ^bb0(%value: f32, %acc: f32):
          %next = arith.maximumf %value, %acc : f32
          linalg.yield %next : f32
        } -> tensor<7xf32>
      wafer.group.yield %max : tensor<7xf32>
    } : tensor<7xf32>

    %min_out = tensor.empty() : tensor<7xf32>
    %min_result = wafer.group ins(%input : tensor<7x4xf32>)
        outs(%min_out : tensor<7xf32>) {
    ^bb0(%input_arg: tensor<7x4xf32>, %out_arg: tensor<7xf32>):
      %init_scalar = arith.constant 0.000000e+00 : f32
      %empty = tensor.empty() : tensor<7xf32>
      %init = linalg.fill ins(%init_scalar : f32)
          outs(%empty : tensor<7xf32>) -> tensor<7xf32>
      %min = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0)>
          ],
          iterator_types = ["parallel", "reduction"]
        } ins(%input_arg : tensor<7x4xf32>)
          outs(%init : tensor<7xf32>) {
        ^bb0(%value: f32, %acc: f32):
          %next = arith.minimumf %value, %acc : f32
          linalg.yield %next : f32
        } -> tensor<7xf32>
      wafer.group.yield %min : tensor<7xf32>
    } : tensor<7xf32>

    return %sum_result, %max_result, %min_result
        : tensor<7xf32>, tensor<7xf32>, tensor<7xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(grouped);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {shapedBoundary(0, {7, 4})};
  program.distributedOutputs = {shapedBoundary(0, {7}),
                                shapedBoundary(1, {7}),
                                shapedBoundary(2, {7})};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  ASSERT_TRUE(static_cast<bool>(config));
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  auto bundle = wafer::compiler::detail::buildExecutableBundle(
      context, *grouped, std::move(program), *config, diagnostics,
      std::nullopt);
  if (!bundle)
    FAIL() << diagnosticsText << llvm::toString(bundle.takeError());
  grouped = nullptr;
  ASSERT_EQ(bundle->getRankExecutables().size(), 1u);

  unsigned nativeReduceCount = 0;
  unsigned addCount = 0;
  unsigned maxCount = 0;
  unsigned minCount = 0;
  bundle->getRankExecutables().front().getModule().walk(
      [&](mlir::Operation *operation) {
        if (mlir::isa<wafer::InstrReduceOp>(operation))
          ++nativeReduceCount;
        if (auto elementwise =
                mlir::dyn_cast<wafer::InstrElementwiseOp>(operation)) {
          switch (elementwise.getKind()) {
          case wafer::InstrElementwiseKind::Add:
            ++addCount;
            break;
          case wafer::InstrElementwiseKind::Max:
            ++maxCount;
            break;
          case wafer::InstrElementwiseKind::Min:
            ++minCount;
            break;
          default:
            break;
          }
        }
      });
  EXPECT_EQ(nativeReduceCount, 0u);
  EXPECT_EQ(addCount, 4u);
  EXPECT_EQ(maxCount, 4u);
  EXPECT_EQ(minCount, 4u);

  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto input = wafer::compiler::ReferenceTensor::create(
      "f32", {7, 4},
      bytesOf({1.0e20f, 3.0f, -1.0e20f, 4.0f,
               1.0f, 2.0f, 3.0f, 4.0f,
               1.0f, nan, 2.0f, 3.0f,
               -0.0f, 0.0f, -0.0f, 0.0f,
               -1.0f, -2.0f, -3.0f, -4.0f,
               -0.0f, -0.0f, -0.0f, -0.0f,
               0.0f, 0.0f, 0.0f, 0.0f}));
  ASSERT_TRUE(static_cast<bool>(input));
  std::vector<wafer::compiler::ReferenceInputBinding> inputs;
  inputs.push_back(
      {wafer::compiler::ProgramResourceRole::UserInput, 0, std::move(*input)});
  auto result =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, inputs);
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_EQ(result->getOutputs().size(), 3u);
  std::vector<float> sum =
      floatsOf(result->getOutputs()[0].tensor.getBytes());
  ASSERT_EQ(sum.size(), 7u);
  EXPECT_EQ(sum[0], 4.0f);
  EXPECT_EQ(sum[1], 10.5f);
  EXPECT_TRUE(std::isnan(sum[2]));
  EXPECT_EQ(sum[3], 0.5f);
  EXPECT_EQ(sum[4], -9.5f);
  EXPECT_EQ(sum[5], 0.5f);
  EXPECT_EQ(sum[6], 0.5f);

  std::vector<float> maximum =
      floatsOf(result->getOutputs()[1].tensor.getBytes());
  ASSERT_EQ(maximum.size(), 7u);
  EXPECT_EQ(maximum[0], 1.0e20f);
  EXPECT_EQ(maximum[1], 4.0f);
  EXPECT_TRUE(std::isnan(maximum[2]));
  EXPECT_EQ(maximum[3], 0.0f);
  EXPECT_FALSE(std::signbit(maximum[3]));
  EXPECT_EQ(maximum[4], 0.0f);
  EXPECT_TRUE(std::signbit(maximum[4]));
  EXPECT_EQ(maximum[5], 0.0f);
  EXPECT_TRUE(std::signbit(maximum[5]));
  EXPECT_EQ(maximum[6], 0.0f);
  EXPECT_FALSE(std::signbit(maximum[6]));

  std::vector<float> minimum =
      floatsOf(result->getOutputs()[2].tensor.getBytes());
  ASSERT_EQ(minimum.size(), 7u);
  EXPECT_EQ(minimum[0], -1.0e20f);
  EXPECT_EQ(minimum[1], 0.0f);
  EXPECT_FALSE(std::signbit(minimum[1]));
  EXPECT_TRUE(std::isnan(minimum[2]));
  EXPECT_EQ(minimum[3], 0.0f);
  EXPECT_TRUE(std::signbit(minimum[3]));
  EXPECT_EQ(minimum[4], -4.0f);
  EXPECT_EQ(minimum[5], 0.0f);
  EXPECT_TRUE(std::signbit(minimum[5]));
  EXPECT_EQ(minimum[6], 0.0f);
  EXPECT_FALSE(std::signbit(minimum[6]));
}

TEST(ReferenceExecutorTest, ExecutesSelectedResidualMlpSemantics) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto grouped = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>, policy = "explicit", shape = array<i64: 1>, topology = @default}
  func.func @main(%input: tensor<2x2xf32>, %w1: tensor<2x3xf32>,
                  %b1: tensor<3xf32>, %w2: tensor<3x2xf32>,
                  %b2: tensor<2xf32>) -> tensor<2x2xf32> {
    %out = tensor.empty() : tensor<2x2xf32>
    %group = wafer.group
        ins(%input, %w1, %b1, %w2, %b2 : tensor<2x2xf32>,
            tensor<2x3xf32>, tensor<3xf32>, tensor<3x2xf32>, tensor<2xf32>)
        outs(%out : tensor<2x2xf32>) {
    ^bb0(%x: tensor<2x2xf32>, %weight1: tensor<2x3xf32>,
         %bias1: tensor<3xf32>, %weight2: tensor<3x2xf32>,
         %bias2: tensor<2xf32>, %output: tensor<2x2xf32>):
      %zero = arith.constant 0.0 : f32
      %hidden_empty = tensor.empty() : tensor<2x3xf32>
      %hidden_init = linalg.fill ins(%zero : f32)
          outs(%hidden_empty : tensor<2x3xf32>) -> tensor<2x3xf32>
      %hidden = linalg.matmul
          ins(%x, %weight1 : tensor<2x2xf32>, tensor<2x3xf32>)
          outs(%hidden_init : tensor<2x3xf32>) -> tensor<2x3xf32>
      %biased = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%hidden, %bias1 : tensor<2x3xf32>, tensor<3xf32>)
          outs(%hidden_empty : tensor<2x3xf32>) {
        ^bb0(%value: f32, %bias: f32, %old: f32):
          %sum = arith.addf %value, %bias : f32
          linalg.yield %sum : f32
        } -> tensor<2x3xf32>
      %activated = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%biased : tensor<2x3xf32>)
          outs(%hidden_empty : tensor<2x3xf32>) {
        ^bb0(%value: f32, %old: f32):
          %result = math.tanh %value : f32
          linalg.yield %result : f32
        } -> tensor<2x3xf32>
      %projected_init = linalg.fill ins(%zero : f32)
          outs(%output : tensor<2x2xf32>) -> tensor<2x2xf32>
      %projected = linalg.matmul
          ins(%activated, %weight2 : tensor<2x3xf32>, tensor<3x2xf32>)
          outs(%projected_init : tensor<2x2xf32>) -> tensor<2x2xf32>
      %projected_bias = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%projected, %bias2 : tensor<2x2xf32>, tensor<2xf32>)
          outs(%output : tensor<2x2xf32>) {
        ^bb0(%value: f32, %bias: f32, %old: f32):
          %sum = arith.addf %value, %bias : f32
          linalg.yield %sum : f32
        } -> tensor<2x2xf32>
      %residual = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%projected_bias, %x : tensor<2x2xf32>, tensor<2x2xf32>)
          outs(%output : tensor<2x2xf32>) {
        ^bb0(%value: f32, %skip: f32, %old: f32):
          %sum = arith.addf %value, %skip : f32
          linalg.yield %sum : f32
        } -> tensor<2x2xf32>
      wafer.group.yield %residual : tensor<2x2xf32>
    } : tensor<2x2xf32>
    return %group : tensor<2x2xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(grouped);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  program.programUserInputCount = 1;
  program.programParameterCount = 4;
  program.distributedInputs = {shapedBoundary(0, {2, 2})};
  program.parameters = {parameter(1, {2, 3}), parameter(2, {3}),
                        parameter(3, {3, 2}), parameter(4, {2})};
  program.distributedOutputs = {shapedBoundary(0, {2, 2})};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  ASSERT_TRUE(static_cast<bool>(config));
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  auto bundle = wafer::compiler::detail::buildExecutableBundle(
      context, *grouped, std::move(program), *config, diagnostics,
      std::nullopt);
  if (!bundle)
    FAIL() << diagnosticsText << llvm::toString(bundle.takeError());
  grouped = nullptr;

  struct InputSpec {
    wafer::compiler::ProgramResourceRole role;
    int64_t index;
    std::vector<int64_t> shape;
    std::vector<float> values;
  };
  uint32_t payloadState = 0x00c0ffeeu;
  std::vector<float> inputValues = fixedNonZeroPayload(4, payloadState, 32.0f);
  std::vector<float> weight1Values =
      fixedNonZeroPayload(6, payloadState, 64.0f);
  std::vector<float> bias1Values = fixedNonZeroPayload(3, payloadState, 128.0f);
  std::vector<float> weight2Values =
      fixedNonZeroPayload(6, payloadState, 64.0f);
  std::vector<float> bias2Values = fixedNonZeroPayload(2, payloadState, 128.0f);
  ResidualMlpOracle oracle = computeResidualMlpOracle(
      inputValues, weight1Values, bias1Values, weight2Values, bias2Values);

  for (int64_t hidden = 0; hidden < 3; ++hidden) {
    std::vector<float> withoutChannel = weight2Values;
    withoutChannel[hidden * 2] = 0.0f;
    withoutChannel[hidden * 2 + 1] = 0.0f;
    EXPECT_TRUE(differs(oracle.output,
                        computeResidualMlpOracle(inputValues, weight1Values,
                                                 bias1Values, withoutChannel,
                                                 bias2Values)
                            .output));
  }
  EXPECT_TRUE(differs(oracle.output,
                      computeResidualMlpOracle(inputValues, weight1Values,
                                               std::vector<float>(3, 0.0f),
                                               weight2Values, bias2Values)
                          .output));
  EXPECT_TRUE(differs(oracle.output,
                      computeResidualMlpOracle(inputValues, weight1Values,
                                               bias1Values, weight2Values,
                                               std::vector<float>(2, 0.0f))
                          .output));

  std::vector<InputSpec> specs = {
      {wafer::compiler::ProgramResourceRole::UserInput, 0, {2, 2}, inputValues},
      {wafer::compiler::ProgramResourceRole::Parameter,
       1,
       {2, 3},
       weight1Values},
      {wafer::compiler::ProgramResourceRole::Parameter, 2, {3}, bias1Values},
      {wafer::compiler::ProgramResourceRole::Parameter,
       3,
       {3, 2},
       weight2Values},
      {wafer::compiler::ProgramResourceRole::Parameter, 4, {2}, bias2Values}};
  std::vector<wafer::compiler::ReferenceInputBinding> inputs;
  for (InputSpec &spec : specs) {
    auto tensor = wafer::compiler::ReferenceTensor::create(
        "f32", spec.shape, bytesOf(spec.values));
    ASSERT_TRUE(static_cast<bool>(tensor));
    inputs.push_back({spec.role, spec.index, std::move(*tensor)});
  }
  auto result =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, inputs);
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_EQ(result->getOutputs().size(), 1u);
  std::vector<float> actual =
      floatsOf(result->getOutputs().front().tensor.getBytes());
  ASSERT_EQ(actual.size(), oracle.output.size());
  for (auto [value, reference] : llvm::zip_equal(actual, oracle.output))
    EXPECT_NEAR(value, reference, 1.0e-5f);
}

TEST(ReferenceExecutorTest,
     ExecutesAcceptedDirectDTEDomainAndReassemblesPartitionedOutput) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto grouped = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64>, policy = "all_available", shape = array<i64: 16>, topology = @default}
  func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %iteration = %c0 to %c2 step %c1
        iter_args(%current = %input) -> tensor<4xf32> {
      %out = tensor.empty() : tensor<4xf32>
      %group = wafer.group ins(%current : tensor<4xf32>)
          outs(%out : tensor<4xf32>) {
      ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
        %permuted = wafer.linalg_ext.collective.collective_permute
            ins(%arg0 : tensor<4xf32>) outs(%arg1 : tensor<4xf32>)
            {source_target_pairs = array<i64: 0, 1, 1, 0, 2, 3, 3, 2,
                                              4, 5, 5, 4, 6, 7, 7, 6,
                                              8, 9, 9, 8, 10, 11, 11, 10,
                                              12, 13, 13, 12, 14, 15, 15, 14>,
             channel_id = 91 : i64} -> tensor<4xf32>
        wafer.group.yield %permuted : tensor<4xf32>
      } : tensor<4xf32>
      scf.yield %group : tensor<4xf32>
    }
    return %result : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(grouped);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 1;
  program.distributedInputs = {partitionedBoundary(0)};
  program.distributedOutputs = {partitionedBoundary(0)};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  if (!config)
    FAIL() << llvm::toString(config.takeError());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  auto bundle = wafer::compiler::detail::buildExecutableBundle(
      context, *grouped, std::move(program), *config, diagnostics,
      std::nullopt);
  if (!bundle)
    FAIL() << diagnosticsText << llvm::toString(bundle.takeError());
  grouped = nullptr;
  ASSERT_EQ(bundle->getRankExecutables().size(), 16u);
  for (const wafer::compiler::RankExecutable &rank :
       bundle->getRankExecutables())
    EXPECT_EQ(rank.getTransportContract(),
              wafer::compiler::TransportContract::DirectDTE);

  std::vector<wafer::compiler::ReferenceRankInvocation> invocations;
  for (int64_t rank = 0; rank < 16; ++rank) {
    std::vector<float> values;
    for (int64_t element = 0; element < 4; ++element)
      values.push_back(static_cast<float>(rank * 4 + element));
    auto input =
        wafer::compiler::ReferenceTensor::create("f32", {4}, bytesOf(values));
    ASSERT_TRUE(static_cast<bool>(input));
    wafer::compiler::ReferenceRankInvocation invocation;
    invocation.logicalRank = rank;
    invocation.inputs.push_back(
        {wafer::compiler::ProgramResourceRole::UserInput, 0,
         std::move(*input)});
    invocations.push_back(std::move(invocation));
  }
  std::reverse(invocations.begin(), invocations.end());
  auto result = wafer::compiler::executeReferenceBundle(*bundle, invocations);
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_EQ(result->getRankResults().size(), 16u);
  EXPECT_EQ(
      floatsOf(result->getRankResults()[0].getOutputs()[0].tensor.getBytes()),
      (std::vector<float>{0, 1, 2, 3}));
  EXPECT_EQ(
      floatsOf(result->getRankResults()[1].getOutputs()[0].tensor.getBytes()),
      (std::vector<float>{4, 5, 6, 7}));
  ASSERT_EQ(result->getGlobalOutputs().size(), 1u);
  std::vector<float> expectedGlobal;
  for (int64_t rank = 0; rank < 16; ++rank)
    for (int64_t element = 0; element < 4; ++element)
      expectedGlobal.push_back(static_cast<float>(rank * 4 + element));
  EXPECT_EQ(floatsOf(result->getGlobalOutputs()[0].tensor.getBytes()),
            expectedGlobal);

  const int64_t savedLogicalRank = invocations[1].logicalRank;
  invocations[1].logicalRank = invocations[0].logicalRank;
  auto duplicateDomain =
      wafer::compiler::executeReferenceBundle(*bundle, invocations);
  EXPECT_FALSE(static_cast<bool>(duplicateDomain));
  if (!duplicateDomain)
    EXPECT_NE(llvm::toString(duplicateDomain.takeError()).find("all-and-only"),
              std::string::npos);

  invocations[1].logicalRank = savedLogicalRank;
  wafer::InstrDTERecvOp rankZeroRecv;
  bundle->getRankExecutables()[0].getModule().walk(
      [&](wafer::InstrDTERecvOp recv) {
        if (!rankZeroRecv)
          rankZeroRecv = recv;
      });
  ASSERT_TRUE(rankZeroRecv);
  mlir::IntegerAttr originalBytes = rankZeroRecv.getBytesAttr();
  mlir::IntegerAttr originalPeer = rankZeroRecv.getPeerAttr();

  rankZeroRecv->setAttr("bytes",
                        mlir::IntegerAttr::get(originalBytes.getType(), 8));
  auto wrongBytes =
      wafer::compiler::executeReferenceBundle(*bundle, invocations);
  EXPECT_FALSE(static_cast<bool>(wrongBytes));
  if (!wrongBytes)
    EXPECT_NE(llvm::toString(wrongBytes.takeError()).find("byte count"),
              std::string::npos);
  rankZeroRecv->setAttr("bytes", originalBytes);

  rankZeroRecv->setAttr("peer",
                        mlir::IntegerAttr::get(originalPeer.getType(), 2));
  auto unmatched =
      wafer::compiler::executeReferenceBundle(*bundle, invocations);
  EXPECT_FALSE(static_cast<bool>(unmatched));
  if (!unmatched) {
    std::string message = llvm::toString(unmatched.takeError());
    EXPECT_NE(message.find("no-progress/deadlock"), std::string::npos);
    EXPECT_NE(message.find("rank=0"), std::string::npos);
    EXPECT_NE(message.find("send=missing"), std::string::npos);
  }
  rankZeroRecv->setAttr("peer", originalPeer);

  wafer::InstrDTEWaitOp rankZeroWait;
  bundle->getRankExecutables()[0].getModule().walk(
      [&](wafer::InstrDTEWaitOp wait) {
        if (!rankZeroWait)
          rankZeroWait = wait;
      });
  ASSERT_TRUE(rankZeroWait);
  mlir::Operation *firstIssue = nullptr;
  bundle->getRankExecutables()[0].getModule().walk(
      [&](mlir::Operation *operation) {
        if (!firstIssue &&
            mlir::isa<wafer::InstrDTESendOp, wafer::InstrDTERecvOp>(operation))
          firstIssue = operation;
      });
  ASSERT_NE(firstIssue, nullptr);
  mlir::OpBuilder earlyWaitBuilder(firstIssue);
  earlyWaitBuilder.setInsertionPoint(firstIssue);
  mlir::Operation *earlyWait =
      earlyWaitBuilder.clone(*rankZeroWait.getOperation());
  auto unmatchedToken =
      wafer::compiler::executeReferenceBundle(*bundle, invocations);
  EXPECT_FALSE(static_cast<bool>(unmatchedToken));
  if (!unmatchedToken) {
    std::string message = llvm::toString(unmatchedToken.takeError());
    EXPECT_NE(message.find("operand has no projected SSA definition"),
              std::string::npos)
        << message;
  }
  earlyWait->erase();

  mlir::OpBuilder builder(rankZeroRecv);
  builder.setInsertionPointAfter(rankZeroRecv);
  builder.clone(*rankZeroRecv.getOperation());
  auto duplicateRecv =
      wafer::compiler::executeReferenceBundle(*bundle, invocations);
  EXPECT_FALSE(static_cast<bool>(duplicateRecv));
  if (!duplicateRecv)
    EXPECT_NE(llvm::toString(duplicateRecv.takeError())
                  .find("duplicate Direct DTE recv instance"),
              std::string::npos);
}

TEST(ReferenceExecutorTest, ExecutesTransportFreeReplicatedMultiRankDomain) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  wafer::registerAllDialects(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto grouped = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64>, policy = "all_available", shape = array<i64: 16>, topology = @default}
  func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
    return %input : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(grouped);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 1;
  program.distributedInputs = {replicatedBoundary16(0)};
  program.distributedOutputs = {replicatedBoundary16(0)};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  if (!config)
    FAIL() << llvm::toString(config.takeError());
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  auto bundle = wafer::compiler::detail::buildExecutableBundle(
      context, *grouped, std::move(program), *config, diagnostics,
      std::nullopt);
  if (!bundle)
    FAIL() << diagnosticsText << llvm::toString(bundle.takeError());
  grouped = nullptr;
  for (const wafer::compiler::RankExecutable &rank :
       bundle->getRankExecutables())
    EXPECT_EQ(rank.getTransportContract(),
              wafer::compiler::TransportContract::None);

  std::vector<wafer::compiler::ReferenceRankInvocation> invocations;
  for (int64_t rank = 0; rank < 16; ++rank) {
    auto input = wafer::compiler::ReferenceTensor::create(
        "f32", {4}, bytesOf({1, 2, 3, 4}));
    ASSERT_TRUE(static_cast<bool>(input));
    wafer::compiler::ReferenceRankInvocation invocation;
    invocation.logicalRank = rank;
    invocation.inputs.push_back(
        {wafer::compiler::ProgramResourceRole::UserInput, 0,
         std::move(*input)});
    invocations.push_back(std::move(invocation));
  }
  std::reverse(invocations.begin(), invocations.end());
  auto result = wafer::compiler::executeReferenceBundle(*bundle, invocations);
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_EQ(result->getRankResults().size(), 16u);
  ASSERT_EQ(result->getGlobalOutputs().size(), 1u);
  EXPECT_EQ(floatsOf(result->getGlobalOutputs()[0].tensor.getBytes()),
            (std::vector<float>{1, 2, 3, 4}));
}

} // namespace
