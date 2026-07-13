//===- ReferenceExecutorTest.cpp - Accepted-rank semantics tests ---------===//

#include "Wafer/Compiler/ReferenceExecutor.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
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
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
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
  auto badDescriptor =
      wafer::compiler::executeReferenceRank(*bundle, /*logicalRank=*/0, inputs);
  ASSERT_FALSE(static_cast<bool>(badDescriptor));
  EXPECT_NE(llvm::toString(badDescriptor.takeError()).find("descriptor"),
            std::string::npos);
  rdma->setAttr("byte_count", originalByteCount);

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
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
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
