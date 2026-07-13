//===- ReferenceExecutorTest.cpp - Accepted-rank semantics tests ---------===//

#include "Wafer/Compiler/ReferenceExecutor.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"

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
  binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  binding.globalShape.assign(shape.begin(), shape.end());
  binding.localShape.assign(shape.begin(), shape.end());
  binding.dtype = "f32";
  binding.rankSlices.push_back(singleRankSlice(shape));
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
  func.func @main(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                  %out: tensor<8xf32>) -> tensor<8xf32> {
    %group = wafer.group ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
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
  program.programUserInputCount = 3;
  program.distributedInputs = {boundary(0), boundary(1), boundary(2)};
  program.distributedOutputs = {boundary(0, true)};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
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

  auto lhs = wafer::compiler::ReferenceTensor::create(
      "f32", {8}, bytesOf({1, 2, 3, 4, 5, 6, 7, 8}));
  auto rhs = wafer::compiler::ReferenceTensor::create(
      "f32", {8}, bytesOf({8, 7, 6, 5, 4, 3, 2, 1}));
  auto out = wafer::compiler::ReferenceTensor::create(
      "f32", {8}, bytesOf(std::vector<float>(8, -100.0f)));
  ASSERT_TRUE(static_cast<bool>(lhs));
  ASSERT_TRUE(static_cast<bool>(rhs));
  ASSERT_TRUE(static_cast<bool>(out));
  std::vector<wafer::compiler::ReferenceInputBinding> inputs;
  inputs.push_back(
      {wafer::compiler::ProgramResourceRole::UserInput, 0, std::move(*lhs)});
  inputs.push_back(
      {wafer::compiler::ProgramResourceRole::UserInput, 1, std::move(*rhs)});
  inputs.push_back(
      {wafer::compiler::ProgramResourceRole::UserInput, 2, std::move(*out)});
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
  auto stochastic =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, {});
  ASSERT_FALSE(static_cast<bool>(stochastic));
  std::string stochasticMessage = llvm::toString(stochastic.takeError());
  EXPECT_NE(stochasticMessage.find("capability preflight"), std::string::npos);
  EXPECT_NE(stochasticMessage.find("stochastic"), std::string::npos);

  add->setOperand(0, originalAddInput);
  backToFloat.erase();
  toInteger.erase();
  roundedFloat.erase();
  roundedInteger.erase();

  auto i8Type = mlir::MemRefType::get(
      addInputType.getShape(), builder.getI8Type(), addInputType.getLayout(),
      addInputType.getMemorySpace());
  auto f16Type = mlir::MemRefType::get(
      addInputType.getShape(), builder.getF16Type(), addInputType.getLayout(),
      addInputType.getMemorySpace());
  builder.setInsertionPoint(add);
  auto quantized = builder.create<mlir::memref::AllocOp>(add.getLoc(), i8Type,
                                                         mlir::ValueRange{});
  quantized->setAttr(wafer::kWaferSPMOffsetAttrName,
                     wafer::SPMOffsetAttr::get(add.getContext(), 1 << 20));
  auto dequantized = builder.create<mlir::memref::AllocOp>(
      add.getLoc(), f16Type, mlir::ValueRange{});
  dequantized->setAttr(wafer::kWaferSPMOffsetAttrName,
                       wafer::SPMOffsetAttr::get(add.getContext(), 2 << 20));
  auto zeroPointConvert = builder.create<wafer::InstrConvertOp>(
      add.getLoc(), wafer::InstrConvertKind::Int8Fp16, quantized, dequantized,
      builder.getI64IntegerAttr(0), /*rounding_mode=*/mlir::IntegerAttr{});
  auto zeroPointProgram =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, {});
  ASSERT_FALSE(static_cast<bool>(zeroPointProgram));
  std::string zeroPointMessage = llvm::toString(zeroPointProgram.takeError());
  EXPECT_NE(zeroPointMessage.find("capability preflight"), std::string::npos);
  EXPECT_NE(zeroPointMessage.find("zero-point convert formula"),
            std::string::npos);
  zeroPointConvert.erase();
  dequantized.erase();
  quantized.erase();

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
}

TEST(ReferenceExecutorTest, RejectsMalformedCompactTensor) {
  auto tensor = wafer::compiler::ReferenceTensor::create("f32", {2}, {0, 0});
  ASSERT_FALSE(tensor);
  EXPECT_NE(llvm::toString(tensor.takeError()).find("byte count"),
            std::string::npos);
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
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
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
  std::vector<InputSpec> specs = {
      {wafer::compiler::ProgramResourceRole::UserInput,
       0,
       {2, 2},
       {1, 2, 3, 4}},
      {wafer::compiler::ProgramResourceRole::Parameter,
       1,
       {2, 3},
       {1, 0, 1, 0, 1, -1}},
      {wafer::compiler::ProgramResourceRole::Parameter, 2, {3}, {0, 0, 0}},
      {wafer::compiler::ProgramResourceRole::Parameter,
       3,
       {3, 2},
       {1, 0, 0, 1, 0, 0}},
      {wafer::compiler::ProgramResourceRole::Parameter, 4, {2}, {0.5, -0.5}}};
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
  std::vector<float> expected = {1.5f + std::tanh(1.0f), 1.5f + std::tanh(2.0f),
                                 3.5f + std::tanh(3.0f),
                                 3.5f + std::tanh(4.0f)};
  ASSERT_EQ(actual.size(), expected.size());
  for (auto [value, reference] : llvm::zip_equal(actual, expected))
    EXPECT_NEAR(value, reference, 1.0e-5f);
}

} // namespace
