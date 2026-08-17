//===- CardExecutableSynthesisTest.cpp -----------------------------===//

#include "../../lib/Wafer/Compiler/CardExecutableSynthesis.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "Wafer/Compiler/ProgramData.h"
#include "../../lib/Wafer/Compiler/CardExecutableInternal.h"
#include "../../lib/Wafer/Compiler/SelectedBufferMaterialization.h"
#include "../../lib/Wafer/Compiler/StructuredDAGSchedulePlan.h"

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Support/CompileWorkStatistics.h"
#include "Wafer/Support/OptimizationConfig.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace {

static wafer::frontend::ProgramPartitionSlice
singlePartitionSlice(llvm::ArrayRef<int64_t> shape) {
  wafer::frontend::ProgramPartitionSlice slice;
  slice.partitionId = 0;
  slice.replicaId = 0;
  slice.offsets.assign(shape.size(), 0);
  slice.sizes.assign(shape.begin(), shape.end());
  slice.strides.assign(shape.size(), 1);
  return slice;
}

static wafer::frontend::ProgramBoundaryBinding
boundary(int64_t index, llvm::ArrayRef<int64_t> shape) {
  wafer::frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  binding.globalShape.assign(shape.begin(), shape.end());
  binding.localShape.assign(shape.begin(), shape.end());
  binding.dtype = "f16";
  binding.partitionSlices.push_back(singlePartitionSlice(shape));
  return binding;
}

static wafer::frontend::FrontendProgramVerificationResult programMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {1, 32}), boundary(1, {1, 32})};
  program.distributedOutputs = {boundary(0, {1, 32})};
  return program;
}

static wafer::frontend::FrontendProgramVerificationResult branchMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {1, 32}), boundary(1, {2, 8})};
  program.distributedOutputs = {boundary(0, {1, 32}), boundary(1, {2, 8})};
  return program;
}

static wafer::frontend::FrontendProgramVerificationResult
dependentProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary(0, {1, 320})};
  program.distributedOutputs = {boundary(0, {1, 320})};
  return program;
}

static wafer::frontend::FrontendProgramVerificationResult
largeTemporalProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {64, 64, 64, 64}),
                               boundary(1, {64, 64, 64, 64})};
  program.distributedOutputs = {boundary(0, {64, 64, 64, 64})};
  return program;
}

static wafer::frontend::FrontendProgramVerificationResult
largeProducerStageProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {1, 4096}),
                               boundary(1, {4096, 4096})};
  program.distributedOutputs = {boundary(0, {1, 4096})};
  return program;
}

static wafer::frontend::FrontendProgramVerificationResult
layoutPipelineProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {128, 128}),
                               boundary(1, {128, 128})};
  program.distributedOutputs = {boundary(0, {128, 128})};
  return program;
}

static wafer::frontend::FrontendProgramVerificationResult
twoReductionAxisProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {1, 10, 11}), boundary(1, {1})};
  program.distributedOutputs = {boundary(0, {1})};
  return program;
}

static wafer::frontend::FrontendProgramVerificationResult
broadcastProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary(0, {16})};
  program.distributedOutputs = {boundary(0, {16, 8})};
  return program;
}

static wafer::frontend::FrontendProgramVerificationResult
reductionDemandProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary(0, {16, 8})};
  program.distributedOutputs = {boundary(0, {16})};
  return program;
}

static wafer::frontend::FrontendProgramVerificationResult
windowDemandProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {1, 18, 18, 1}),
                               boundary(1, {3, 3, 1, 1})};
  program.distributedOutputs = {boundary(0, {1, 16, 16, 1})};
  return program;
}

static wafer::frontend::FrontendProgramVerificationResult
stridedDemandProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary(0, {32})};
  program.distributedOutputs = {boundary(0, {16})};
  return program;
}

static wafer::frontend::FrontendProgramVerificationResult
multiPieceDemandProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {16}), boundary(1, {4})};
  program.distributedOutputs = {boundary(0, {16})};
  return program;
}

struct ParsedProgram {
  std::shared_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

static ParsedProgram parseProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<1x32xf16>, %rhs: tensor<1x32xf16>)
      -> tensor<1x32xf16> {
    %out = tensor.empty() : tensor<1x32xf16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%lhs, %rhs : tensor<1x32xf16>, tensor<1x32xf16>)
        outs(%out : tensor<1x32xf16>) {
      ^bb0(%a: f16, %b: f16, %old: f16):
        %value = arith.addf %a, %b : f16
        linalg.yield %value : f16
    } -> tensor<1x32xf16>
    return %sum : tensor<1x32xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static ParsedProgram parseBranchProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<1x32xf16>, %rhs: tensor<2x8xf16>)
      -> (tensor<1x32xf16>, tensor<2x8xf16>) {
    %out0 = tensor.empty() : tensor<1x32xf16>
    %out1 = tensor.empty() : tensor<2x8xf16>
    %first = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%lhs : tensor<1x32xf16>) outs(%out0 : tensor<1x32xf16>) {
      ^bb0(%value: f16, %old: f16):
        %result = arith.addf %value, %value : f16
        linalg.yield %result : f16
    } -> tensor<1x32xf16>
    %second = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%rhs : tensor<2x8xf16>) outs(%out1 : tensor<2x8xf16>) {
      ^bb0(%value: f16, %old: f16):
        %result = arith.mulf %value, %value : f16
        linalg.yield %result : f16
    } -> tensor<2x8xf16>
    return %first, %second : tensor<1x32xf16>, tensor<2x8xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static ParsedProgram parseDependentProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<1x320xf16>) -> tensor<1x320xf16> {
    %scratch = tensor.empty() : tensor<1x320xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%input : tensor<1x320xf16>)
        outs(%scratch : tensor<1x320xf16>) {
      ^bb0(%value: f16, %old: f16):
        %result = arith.addf %value, %value : f16
        linalg.yield %result : f16
    } -> tensor<1x320xf16>
    %out = tensor.empty() : tensor<1x320xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%producer : tensor<1x320xf16>)
        outs(%out : tensor<1x320xf16>) {
      ^bb0(%value: f16, %old: f16):
        %result = arith.mulf %value, %value : f16
        linalg.yield %result : f16
    } -> tensor<1x320xf16>
    return %consumer : tensor<1x320xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static ParsedProgram parseThreeStageDependentProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<1x320xf16>) -> tensor<1x320xf16> {
    %scratch0 = tensor.empty() : tensor<1x320xf16>
    %first = linalg.map ins(%input : tensor<1x320xf16>)
        outs(%scratch0 : tensor<1x320xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %scratch1 = tensor.empty() : tensor<1x320xf16>
    %second = linalg.map ins(%first : tensor<1x320xf16>)
        outs(%scratch1 : tensor<1x320xf16>) (%value: f16) {
      %next = arith.mulf %value, %value : f16
      linalg.yield %next : f16
    }
    %out = tensor.empty() : tensor<1x320xf16>
    %third = linalg.map ins(%second : tensor<1x320xf16>)
        outs(%out : tensor<1x320xf16>) (%value: f16) {
      %next = arith.subf %value, %value : f16
      linalg.yield %next : f16
    }
    return %third : tensor<1x320xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static ParsedProgram parseLargeTemporalProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<64x64x64x64xf16>,
                  %rhs: tensor<64x64x64x64xf16>)
      -> tensor<64x64x64x64xf16> {
    %out = tensor.empty() : tensor<64x64x64x64xf16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%lhs, %rhs : tensor<64x64x64x64xf16>,
                         tensor<64x64x64x64xf16>)
        outs(%out : tensor<64x64x64x64xf16>) {
      ^bb0(%a: f16, %b: f16, %old: f16):
        %value = arith.addf %a, %b : f16
        linalg.yield %value : f16
    } -> tensor<64x64x64x64xf16>
    return %sum : tensor<64x64x64x64xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static ParsedProgram parseLargeProducerStageProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<1x4096xf16>,
                  %weight: tensor<4096x4096xf16>)
      -> tensor<1x4096xf16> {
    %transposed_init = tensor.empty() : tensor<4096x4096xf16>
    %transposed = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%weight : tensor<4096x4096xf16>)
        outs(%transposed_init : tensor<4096x4096xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<4096x4096xf16>
    %result_init = tensor.empty() : tensor<1x4096xf16>
    %zero = arith.constant 0.0 : f16
    %filled = linalg.fill ins(%zero : f16)
        outs(%result_init : tensor<1x4096xf16>) -> tensor<1x4096xf16>
    %result = linalg.matmul
        ins(%input, %transposed : tensor<1x4096xf16>, tensor<4096x4096xf16>)
        outs(%filled : tensor<1x4096xf16>) -> tensor<1x4096xf16>
    return %result : tensor<1x4096xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static ParsedProgram parseLayoutPipelineProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<128x128xf16>, %rhs: tensor<128x128xf16>)
      -> tensor<128x128xf16> {
    %matmulOut = tensor.empty() : tensor<128x128xf16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%matmulOut : tensor<128x128xf16>) -> tensor<128x128xf16>
    %product = linalg.matmul ins(%lhs, %rhs : tensor<128x128xf16>,
                                tensor<128x128xf16>)
        outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
    %mapOut = tensor.empty() : tensor<128x128xf16>
    %result = linalg.map ins(%product : tensor<128x128xf16>)
        outs(%mapOut : tensor<128x128xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    return %result : tensor<128x128xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static ParsedProgram parseTwoReductionAxisProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<1x10x11xf16>,
                  %init: tensor<1xf16>) -> tensor<1xf16> {
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0)>],
        iterator_types = ["parallel", "reduction", "reduction"]
      } ins(%input : tensor<1x10x11xf16>)
        outs(%init : tensor<1xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %sum = arith.addf %value, %acc  : f16
        linalg.yield %sum : f16
    } -> tensor<1xf16>
    return %result : tensor<1xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

// The broadcast source must be fully resident on every Tile for every
// temporal breakpoint: 1536x1024xf16 is 3 MiB, above the fixed SPM capacity,
// so even the minimum legal tile proves a capacity overflow. This is the
// minimum-tile negative case of the deterministic baseline.
static ParsedProgram parseBroadcastProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<16xf16>) -> tensor<16x8xf16> {
    %producerOut = tensor.empty() : tensor<16xf16>
    %producer = linalg.map ins(%input : tensor<16xf16>)
        outs(%producerOut : tensor<16xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %resultOut = tensor.empty() : tensor<16x8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%producer : tensor<16xf16>)
        outs(%resultOut : tensor<16x8xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %old : f16
        linalg.yield %next : f16
    } -> tensor<16x8xf16>
    return %result : tensor<16x8xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static ParsedProgram parseReductionDemandProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<16x8xf16>) -> tensor<16xf16> {
    %producerOut = tensor.empty() : tensor<16x8xf16>
    %producer = linalg.map ins(%input : tensor<16x8xf16>)
        outs(%producerOut : tensor<16x8xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %resultOut = tensor.empty() : tensor<16xf16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%resultOut : tensor<16xf16>) -> tensor<16xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%producer : tensor<16x8xf16>) outs(%init : tensor<16xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<16xf16>
    return %result : tensor<16xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static ParsedProgram parseWindowDemandProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<1x18x18x1xf16>,
                  %filter: tensor<3x3x1x1xf16>) -> tensor<1x16x16x1xf16> {
    %producerOut = tensor.empty() : tensor<1x18x18x1xf16>
    %producer = linalg.map ins(%input : tensor<1x18x18x1xf16>)
        outs(%producerOut : tensor<1x18x18x1xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %resultOut = tensor.empty() : tensor<1x16x16x1xf16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%resultOut : tensor<1x16x16x1xf16>) -> tensor<1x16x16x1xf16>
    %result = linalg.conv_2d_nhwc_hwcf
        {dilations = dense<1> : tensor<2xi64>,
         strides = dense<1> : tensor<2xi64>}
        ins(%producer, %filter : tensor<1x18x18x1xf16>, tensor<3x3x1x1xf16>)
        outs(%init : tensor<1x16x16x1xf16>) -> tensor<1x16x16x1xf16>
    return %result : tensor<1x16x16x1xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static ParsedProgram parseStridedDemandProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<32xf16>) -> tensor<16xf16> {
    %producerOut = tensor.empty() : tensor<32xf16>
    %producer = linalg.map ins(%input : tensor<32xf16>)
        outs(%producerOut : tensor<32xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %view = tensor.extract_slice %producer[0] [16] [2]
        : tensor<32xf16> to tensor<16xf16>
    %resultOut = tensor.empty() : tensor<16xf16>
    %result = linalg.map ins(%view : tensor<16xf16>)
        outs(%resultOut : tensor<16xf16>) (%value: f16) {
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    }
    return %result : tensor<16xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static ParsedProgram parseMultiPieceDemandProgram() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%left: tensor<16xf16>, %right: tensor<4xf16>)
      -> tensor<16xf16> {
    %leftOut = tensor.empty() : tensor<16xf16>
    %producer = linalg.map ins(%left : tensor<16xf16>)
        outs(%leftOut : tensor<16xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %withMiddle = tensor.insert_slice %right into %producer[6] [4] [1]
        : tensor<4xf16> into tensor<16xf16>
    %resultOut = tensor.empty() : tensor<16xf16>
    %result = linalg.map ins(%withMiddle : tensor<16xf16>)
        outs(%resultOut : tensor<16xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    return %result : tensor<16xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

static size_t countOccurrences(llvm::StringRef text, llvm::StringRef needle) {
  size_t count = 0;
  while (true) {
    size_t position = text.find(needle);
    if (position == llvm::StringRef::npos)
      return count;
    ++count;
    text = text.drop_front(position + needle.size());
  }
}

static wafer::compiler::ExecutionConfig executionConfig() {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  EXPECT_TRUE(static_cast<bool>(config));
  return *config;
}

enum class TestOperationRelationMode {
  Complete,
  MissingFirstNode,
  MissingLastNode,
  FusedFirstTwo,
  Ambiguous,
};

static mlir::FailureOr<wafer::compiler::detail::CardExecutableLoweringResult>
makePlanTestExecutable(
    mlir::MLIRContext &context, size_t nodeCount,
    llvm::ArrayRef<wafer::compiler::detail::StructuredDAGNodePlacement> placements,
    TestOperationRelationMode mode,
    llvm::SmallVectorImpl<
        wafer::compiler::detail::AcceptedOperationNodeRelation>
        &operationNodeRelations) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  llvm::SmallVector<wafer::analysis::TileInstructionProgram, 16>
      programs;
  modules.reserve(16);
  programs.reserve(16);
  for (int64_t tileId = 0; tileId < 16; ++tileId) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @entry() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
        %a, %a into %b
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
        %b, %b into %c
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
        %c, %c into %a
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
        mlir::ParserConfig(&context));
    if (!module)
      return mlir::failure();

    llvm::SmallVector<wafer::InstrElementwiseOp, 3> markers;
    module->walk([&](wafer::InstrElementwiseOp operation) {
      markers.push_back(operation);
    });
    if (markers.size() != 3)
      return mlir::failure();

    size_t markerIndex = 0;
    for (const wafer::compiler::detail::StructuredDAGNodePlacement &placement :
         placements) {
      if (!llvm::is_contained(placement.tiles, wafer::TileId(tileId)))
        continue;
      if (placement.node >= nodeCount || markerIndex >= markers.size())
        return mlir::failure();
      if ((mode == TestOperationRelationMode::MissingFirstNode &&
           placement.node == 0) ||
          (mode == TestOperationRelationMode::MissingLastNode &&
           placement.node + 1 == nodeCount))
        continue;
      mlir::Operation *marker = markers[markerIndex++].getOperation();
      operationNodeRelations.push_back({marker, placement.node});
    }
    if ((mode == TestOperationRelationMode::FusedFirstTwo ||
         mode == TestOperationRelationMode::Ambiguous) &&
        tileId == 0 && nodeCount >= 2) {
      mlir::Operation *marker = markers.front().getOperation();
      for (uint32_t node = 0; node < 2; ++node)
        if (!llvm::any_of(operationNodeRelations, [&](const auto &relation) {
              return relation.operation == marker &&
                     relation.structuredNodeId == node;
            }))
          operationNodeRelations.push_back({marker, node});
    }

    mlir::func::FuncOp entry =
        module->lookupSymbol<mlir::func::FuncOp>("entry");
    if (!entry)
      return mlir::failure();
    programs.push_back({wafer::TileId(tileId), entry.getOperation()});
    modules.push_back(std::move(module));
  }

  wafer::analysis::CardInstructionProgramCost cost =
      wafer::analysis::analyzeCardInstructionProgramCost(
          programs, wafer::analysis::getTargetScheduleCostPolicy());
  std::vector<wafer::compiler::TileExecutable> tiles;
  tiles.reserve(modules.size());
  for (size_t tile = 0; tile < modules.size(); ++tile)
    tiles.push_back(
        wafer::compiler::CardExecutableBuilder::makeTileExecutable(
            wafer::CardId(0),
            wafer::TileId(static_cast<int64_t>(tile)),
            wafer::LaunchSlotId(static_cast<int64_t>(tile)),
            std::move(modules[tile]), "entry", {},
            wafer::compiler::TransportContract::None));
  wafer::RuntimeLaunchContract launch =
      llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
          wafer::KernelLaunchForm::Grid,
          wafer::KernelEntryABI::TileMajorPointerTable,
          {wafer::RuntimeLaunchPhaseRole::Main}));
  return wafer::compiler::detail::CardExecutableLoweringResult(
      std::move(tiles), std::move(launch), std::move(cost));
}

static mlir::FailureOr<wafer::compiler::detail::StructuredDAGAnalysis>
analyzePlanTestDAG(mlir::ModuleOp module) {
  mlir::func::FuncOp function = *module.getOps<mlir::func::FuncOp>().begin();
  return wafer::compiler::detail::StructuredDAGAnalysis::create(function);
}

static void expectCompleteTileDomain(
    const wafer::compiler::detail::CardExecutableSynthesisResult &result) {
  const auto &executable = result.executable;
  ASSERT_EQ(executable.tiles.size(), 16u);
  ASSERT_EQ(result.tileDataflowIRTrace.size(), executable.tiles.size());
  for (size_t index = 0; index < executable.tiles.size(); ++index) {
    const wafer::compiler::TileExecutable &tile =
        executable.tiles[index];
    llvm::StringRef tileDataflowIR = result.tileDataflowIRTrace[index];
    EXPECT_EQ(tile.getCardId(), wafer::CardId(0));
    EXPECT_EQ(tile.getTileId(),
              wafer::TileId(static_cast<int64_t>(index)));
    EXPECT_FALSE(tileDataflowIR.empty());
    EXPECT_NE(tileDataflowIR.find("wafer.tile.region"), llvm::StringRef::npos);
    EXPECT_NE(tileDataflowIR.find("wafer.tile.load"), llvm::StringRef::npos);
    EXPECT_NE(tileDataflowIR.find("wafer.tile.store"), llvm::StringRef::npos);
  }
}

static void expectDemandProgramCompletesExecutableGate(
    ParsedProgram &parsed,
    const wafer::frontend::FrontendProgramVerificationResult &metadata) {
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::synthesizeCardExecutable(
      *parsed.module, metadata, executionConfig(),
      wafer::OptimizationConfig::none(), diagnostics, programData, &statistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  diagnostics.flush();

  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  expectCompleteTileDomain(*executable);
  EXPECT_GT(statistics.exactDemandSatisfiedEdges, 0u);
  EXPECT_EQ(statistics.demandAbortStatus,
            wafer::analysis::ExactDemandStatus::Satisfied);
  EXPECT_EQ(statistics.exactGates.cardModuleCompilationInvocations, 1u);
  EXPECT_EQ(statistics.acceptedCandidates, 0u);
  // Baseline structural contract: every TileRegion carries exactly one
  // structured compute root; an in-region operand demand of a foreign node
  // never counts as a second root.
  EXPECT_EQ(diagnosticsText.find("multiple structured compute roots"),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardExecutableSynthesisTest,
     AcceptedPlanConservesResidualAndRunsDisjointNodePhasesTogether) {
  ParsedProgram parsed = parseBranchProgram();
  ASSERT_TRUE(parsed.module);
  auto dag = analyzePlanTestDAG(*parsed.module);
  ASSERT_TRUE(mlir::succeeded(dag));
  ASSERT_EQ(dag->getNodes().size(), 2u);

  llvm::SmallVector<wafer::compiler::detail::StructuredDAGNodePlacement, 2>
      placements;
  placements.push_back({0, 0, {wafer::TileId(0)}});
  placements.push_back({1, 0, {wafer::TileId(1)}});
  llvm::SmallVector<wafer::compiler::detail::AcceptedOperationNodeRelation, 4>
      relations;
  auto executable = makePlanTestExecutable(
      *parsed.context, dag->getNodes().size(), placements,
      TestOperationRelationMode::Complete, relations);
  ASSERT_TRUE(mlir::succeeded(executable));

  llvm::SmallVector<wafer::analysis::CardInstructionProgramCost, 3>
      phaseCosts;
  std::string failureReason;
  auto plan = wafer::compiler::detail::buildAcceptedStructuredDAGSchedulePlan(
      *dag, placements, relations, *executable, phaseCosts, &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  ASSERT_EQ(phaseCosts.size(), 3u);
  EXPECT_EQ(executable->resourceCost.aggregateInstructionCount.value, 48u);
  EXPECT_EQ(phaseCosts[0].aggregateInstructionCount.value, 46u);
  EXPECT_EQ(phaseCosts[1].aggregateInstructionCount.value, 1u);
  EXPECT_EQ(phaseCosts[2].aggregateInstructionCount.value, 1u);
  ASSERT_EQ(plan->getSteps().size(), 4u);
  EXPECT_EQ(plan->getSteps().back().kind,
            wafer::analysis::StaticScheduleStep::Kind::Stage);
  EXPECT_EQ(plan->getSteps().back().stage.independentBranches.size(), 2u);
}

TEST(CardExecutableSynthesisTest,
     AcceptedPlanKeepsDependentNodePhasesInDAGOrder) {
  ParsedProgram parsed = parseDependentProgram();
  ASSERT_TRUE(parsed.module);
  auto dag = analyzePlanTestDAG(*parsed.module);
  ASSERT_TRUE(mlir::succeeded(dag));
  ASSERT_EQ(dag->getNodes().size(), 2u);
  ASSERT_EQ(dag->getEdges().size(), 1u);

  llvm::SmallVector<wafer::compiler::detail::StructuredDAGNodePlacement, 2>
      placements;
  placements.push_back({0, 0, {wafer::TileId(0)}});
  placements.push_back({1, 0, {wafer::TileId(0)}});
  llvm::SmallVector<wafer::compiler::detail::AcceptedOperationNodeRelation, 4>
      relations;
  auto executable = makePlanTestExecutable(
      *parsed.context, dag->getNodes().size(), placements,
      TestOperationRelationMode::Complete, relations);
  ASSERT_TRUE(mlir::succeeded(executable));

  llvm::SmallVector<wafer::analysis::CardInstructionProgramCost, 3>
      phaseCosts;
  std::string failureReason;
  auto plan = wafer::compiler::detail::buildAcceptedStructuredDAGSchedulePlan(
      *dag, placements, relations, *executable, phaseCosts, &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  ASSERT_EQ(plan->getSteps().size(), 5u);
  for (size_t step = 3; step < plan->getSteps().size(); ++step) {
    EXPECT_EQ(plan->getSteps()[step].kind,
              wafer::analysis::StaticScheduleStep::Kind::Stage);
    EXPECT_EQ(plan->getSteps()[step].stage.independentBranches.size(), 1u);
  }
}

TEST(CardExecutableSynthesisTest,
     AcceptedPlanAllowsEliminatedInternalNodeCoveredByItsSuccessor) {
  ParsedProgram parsed = parseDependentProgram();
  ASSERT_TRUE(parsed.module);
  auto dag = analyzePlanTestDAG(*parsed.module);
  ASSERT_TRUE(mlir::succeeded(dag));
  ASSERT_EQ(dag->getNodes().size(), 2u);
  ASSERT_EQ(dag->getEdges().size(), 1u);

  llvm::SmallVector<wafer::compiler::detail::StructuredDAGNodePlacement, 2>
      placements;
  placements.push_back({0, 0, {wafer::TileId(0)}});
  placements.push_back({1, 0, {wafer::TileId(0)}});
  llvm::SmallVector<wafer::compiler::detail::AcceptedOperationNodeRelation, 4>
      relations;
  auto executable = makePlanTestExecutable(
      *parsed.context, dag->getNodes().size(), placements,
      TestOperationRelationMode::MissingFirstNode, relations);
  ASSERT_TRUE(mlir::succeeded(executable));

  llvm::SmallVector<wafer::analysis::CardInstructionProgramCost, 3>
      phaseCosts;
  std::string failureReason;
  auto plan = wafer::compiler::detail::buildAcceptedStructuredDAGSchedulePlan(
      *dag, placements, relations, *executable, phaseCosts, &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  ASSERT_EQ(phaseCosts.size(), 3u);
  EXPECT_EQ(phaseCosts[0].aggregateInstructionCount.value, 47u);
  EXPECT_EQ(phaseCosts[1].aggregateInstructionCount.value, 0u);
  EXPECT_EQ(phaseCosts[2].aggregateInstructionCount.value, 1u);
}

TEST(CardExecutableSynthesisTest,
     AcceptedPlanAttributesFusedOperationOnceToUniqueDownstreamNode) {
  ParsedProgram parsed = parseDependentProgram();
  ASSERT_TRUE(parsed.module);
  auto dag = analyzePlanTestDAG(*parsed.module);
  ASSERT_TRUE(mlir::succeeded(dag));
  ASSERT_EQ(dag->getNodes().size(), 2u);
  ASSERT_EQ(dag->getEdges().size(), 1u);

  llvm::SmallVector<wafer::compiler::detail::StructuredDAGNodePlacement, 2>
      placements;
  placements.push_back({0, 0, {wafer::TileId(0)}});
  placements.push_back({1, 0, {wafer::TileId(0)}});
  llvm::SmallVector<wafer::compiler::detail::AcceptedOperationNodeRelation, 4>
      relations;
  auto executable = makePlanTestExecutable(
      *parsed.context, dag->getNodes().size(), placements,
      TestOperationRelationMode::FusedFirstTwo, relations);
  ASSERT_TRUE(mlir::succeeded(executable));

  llvm::SmallVector<wafer::analysis::CardInstructionProgramCost, 3>
      phaseCosts;
  std::string failureReason;
  auto plan = wafer::compiler::detail::buildAcceptedStructuredDAGSchedulePlan(
      *dag, placements, relations, *executable, phaseCosts, &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  ASSERT_EQ(phaseCosts.size(), 3u);
  EXPECT_EQ(phaseCosts[0].aggregateInstructionCount.value, 46u);
  EXPECT_EQ(phaseCosts[1].aggregateInstructionCount.value, 0u);
  EXPECT_EQ(phaseCosts[2].aggregateInstructionCount.value, 2u);
}

TEST(CardExecutableSynthesisTest,
     AcceptedPlanRejectsLostAndAmbiguousOperationRelations) {
  ParsedProgram parsed = parseBranchProgram();
  ASSERT_TRUE(parsed.module);
  auto dag = analyzePlanTestDAG(*parsed.module);
  ASSERT_TRUE(mlir::succeeded(dag));
  ASSERT_EQ(dag->getNodes().size(), 2u);
  llvm::SmallVector<wafer::compiler::detail::StructuredDAGNodePlacement, 2>
      placements;
  placements.push_back({0, 0, {wafer::TileId(0)}});
  placements.push_back({1, 0, {wafer::TileId(1)}});

  for (TestOperationRelationMode mode :
       {TestOperationRelationMode::MissingLastNode,
        TestOperationRelationMode::Ambiguous}) {
    llvm::SmallVector<wafer::compiler::detail::AcceptedOperationNodeRelation, 4>
        relations;
    auto executable = makePlanTestExecutable(
        *parsed.context, dag->getNodes().size(), placements, mode, relations);
    ASSERT_TRUE(mlir::succeeded(executable));
    llvm::SmallVector<wafer::analysis::CardInstructionProgramCost, 3>
        phaseCosts;
    std::string failureReason;
    auto plan = wafer::compiler::detail::buildAcceptedStructuredDAGSchedulePlan(
        *dag, placements, relations, *executable, phaseCosts, &failureReason);
    EXPECT_TRUE(mlir::failed(plan));
    EXPECT_FALSE(failureReason.empty());
  }
}

TEST(CardExecutableSynthesisTest,
     NoneMaterializesOneBalancedBaselineThroughExactGates) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::synthesizeCardExecutable(
      *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::none(), diagnostics, programData, &statistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  expectCompleteTileDomain(*executable);

  EXPECT_EQ(statistics.structuredNodeCount, 1u);
  EXPECT_EQ(statistics.structuredEdgeCount, 0u);
  // The deterministic baseline never constructs search candidates and never
  // writes candidate proposal/selection statistics.
  EXPECT_EQ(statistics.candidateProposals, 0u);
  EXPECT_EQ(statistics.cheapPrunedCandidates, 0u);
  EXPECT_EQ(statistics.shortlistedCandidates, 0u);
  EXPECT_EQ(statistics.materializedCandidates, 0u);
  EXPECT_EQ(statistics.materializationRejections, 0u);
  EXPECT_EQ(statistics.acceptedCandidates, 0u);
  EXPECT_EQ(statistics.plannedCandidates, 0u);
  EXPECT_EQ(statistics.schedulePlanRejections, 0u);
  EXPECT_EQ(statistics.selectedOutputMappingCount, 0u);
  EXPECT_EQ(statistics.selectedUniqueActiveTileCount, 0u);
  EXPECT_EQ(statistics.selectedParallelComponentCount, 0u);
  EXPECT_EQ(statistics.exactGates.tileModuleLoweringAttempts, 1u);
  EXPECT_EQ(statistics.exactGates.cardModuleCompilationInvocations, 1u);
  EXPECT_EQ(statistics.exactGates.targetLoweringVerificationInvocations, 1u);
  EXPECT_EQ(statistics.exactGates.targetTileLoweringVerificationInvocations,
            16u);
  EXPECT_EQ(statistics.baselineCardModuleMaterializations, 1u);
  EXPECT_EQ(statistics.baselineScopedCardModuleMaterializations, 0u);
  EXPECT_EQ(statistics.baselineRegionSPMCapacityChecks, 16u);
  EXPECT_EQ(statistics.baselineRegionSPMCapacityOverflowProofs, 0u);
  EXPECT_GT(statistics.baselineMaximumRegionSPMQueryWorkers, 1u);
  EXPECT_EQ(statistics.resourceScheduleMemoHits, 0u);
  EXPECT_EQ(statistics.resourceScheduleMemoMisses, 0u);
  EXPECT_EQ(statistics.selectedExecutableRematerializations, 0u);
  EXPECT_EQ(statistics.selectedExecutableRematerializationGates
                .tileModuleLoweringAttempts,
            0u);
  EXPECT_EQ(diagnosticsText.find("card-executable-search policy=none"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(
      diagnosticsText.find("card-executable-baseline-controller search_states=0 "
                           "placement_enumeration=0 "
                           "candidate_family=0"),
      std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("card-executable-baseline-admission"),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardExecutableSynthesisTest,
     NoneProbesIndependentStructuredOwnersInOneActualRegion) {
  ParsedProgram parsed = parseBranchProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::synthesizeCardExecutable(
      *parsed.module, branchMetadata(), executionConfig(),
      wafer::OptimizationConfig::none(), diagnostics, programData, &statistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  ASSERT_EQ(executable->executable.tiles.size(), 16u);
  EXPECT_EQ(statistics.structuredNodeCount, 2u);
  EXPECT_EQ(statistics.structuredEdgeCount, 0u);
  EXPECT_EQ(statistics.materializedCandidates, 0u);
  EXPECT_EQ(statistics.exactGates.cardModuleCompilationInvocations, 1u);
  EXPECT_EQ(statistics.selectedActualFusedLogicalEdges, 0u);
  // Two independent structured roots on one Tile form multiple sequential
  // regions: the shared Tile (Tile 0) carries one region per root.
  EXPECT_EQ(diagnosticsText.find("multiple structured compute roots"),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardExecutableSynthesisTest,
     NoneCarriesBroadcastDemandThroughTheCompleteExecutableGate) {
  ParsedProgram parsed = parseBroadcastProgram();
  expectDemandProgramCompletesExecutableGate(parsed,
                                             broadcastProgramMetadata());
}

TEST(CardExecutableSynthesisTest,
     NoneCarriesReductionDemandThroughTheCompleteExecutableGate) {
  ParsedProgram parsed = parseReductionDemandProgram();
  expectDemandProgramCompletesExecutableGate(parsed,
                                             reductionDemandProgramMetadata());
}

TEST(CardExecutableSynthesisTest,
     NoneCarriesWindowDemandThroughTheCompleteExecutableGate) {
  ParsedProgram parsed = parseWindowDemandProgram();
  expectDemandProgramCompletesExecutableGate(parsed,
                                             windowDemandProgramMetadata());
}

TEST(CardExecutableSynthesisTest,
     NoneCarriesStridedDemandThroughTheCompleteExecutableGate) {
  ParsedProgram parsed = parseStridedDemandProgram();
  expectDemandProgramCompletesExecutableGate(parsed,
                                             stridedDemandProgramMetadata());
}

TEST(CardExecutableSynthesisTest,
     NoneCarriesMultiPieceDemandThroughTheCompleteExecutableGate) {
  ParsedProgram parsed = parseMultiPieceDemandProgram();
  expectDemandProgramCompletesExecutableGate(parsed,
                                             multiPieceDemandProgramMetadata());
}

TEST(CardExecutableSynthesisTest,
     NoneProducesStableCardModuleAndCardExecutableIR) {
  ParsedProgram firstProgram = parseProgram();
  ParsedProgram secondProgram = parseProgram();
  ASSERT_TRUE(firstProgram.module);
  ASSERT_TRUE(secondProgram.module);
  std::string firstDiagnosticsText;
  std::string secondDiagnosticsText;
  llvm::raw_string_ostream firstDiagnostics(firstDiagnosticsText);
  llvm::raw_string_ostream secondDiagnostics(secondDiagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics firstStats;
  wafer::compiler::ProgramDataHandoff programData;
  wafer::compiler::detail::CardExecutableSynthesisStatistics secondStats;
  auto first = wafer::compiler::detail::synthesizeCardExecutable(
      *firstProgram.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::none(), firstDiagnostics, programData, &firstStats, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  auto second = wafer::compiler::detail::synthesizeCardExecutable(
      *secondProgram.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::none(), secondDiagnostics, programData, &secondStats, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  firstDiagnostics.flush();
  secondDiagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(first)) << firstDiagnosticsText;
  ASSERT_TRUE(mlir::succeeded(second)) << secondDiagnosticsText;
  ASSERT_EQ(first->executable.tiles.size(), second->executable.tiles.size());
  EXPECT_EQ(first->tileDataflowIRTrace, second->tileDataflowIRTrace);
  for (auto [firstTile, secondTile] :
       llvm::zip(first->executable.tiles, second->executable.tiles)) {
    std::string firstExecutableIR;
    std::string secondExecutableIR;
    llvm::raw_string_ostream firstStream(firstExecutableIR);
    llvm::raw_string_ostream secondStream(secondExecutableIR);
    firstTile.getModule().print(firstStream);
    secondTile.getModule().print(secondStream);
    firstStream.flush();
    secondStream.flush();
    EXPECT_EQ(firstExecutableIR, secondExecutableIR);
  }
  EXPECT_EQ(firstStats.baselineCardModuleMaterializations,
            secondStats.baselineCardModuleMaterializations);
  EXPECT_EQ(firstStats.baselineScopedCardModuleMaterializations,
            secondStats.baselineScopedCardModuleMaterializations);
  EXPECT_EQ(firstStats.baselineRegionSPMCapacityChecks,
            secondStats.baselineRegionSPMCapacityChecks);
  EXPECT_EQ(firstStats.exactGates.cardModuleCompilationInvocations, 1u);
  EXPECT_EQ(secondStats.exactGates.cardModuleCompilationInvocations, 1u);
}

TEST(CardExecutableSynthesisTest,
     SelectedDoubleBufferMaterializesExactRotatingSlots) {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @pipeline() {
    %input = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %unused = %c0 to %c1 step %c1 {
      scf.yield
    }
    scf.for %iv = %c0 to %c2 step %c1 {
      %slot = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %slot
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %slot, %slot into %slot
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield
    }
    return
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      source, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  unsigned slotAllocations = 0;
  std::string failureReason;
  auto workSession =
      std::make_shared<wafer::support::CompileWorkStatisticsSession>();
  wafer::support::ScopedCompileWorkStatisticsActivation workActivation(
      workSession);
  ASSERT_TRUE(
      mlir::succeeded(wafer::compiler::detail::materializeSelectedBuffering(
          module, /*requestedBufferCount=*/2, &slotAllocations,
          &failureReason)))
      << failureReason;
  EXPECT_EQ(slotAllocations, 2u);
  EXPECT_EQ(workSession->snapshot().selectedBufferModuleClones, 1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  unsigned rotatingArguments = 0;
  module->walk([&](mlir::scf::ForOp loop) {
    rotatingArguments =
        std::max(rotatingArguments, loop.getNumRegionIterArgs());
  });
  EXPECT_GE(rotatingArguments, 2u);

  auto mismatched = mlir::parseSourceString<mlir::ModuleOp>(
      source, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(mismatched);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  mismatched->print(beforeStream);
  beforeStream.flush();
  EXPECT_TRUE(
      mlir::failed(wafer::compiler::detail::materializeSelectedBuffering(
          mismatched, /*requestedBufferCount=*/3, nullptr, &failureReason)));
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  mismatched->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

TEST(CardExecutableSynthesisTest,
     SearchUsesFactorizedCoordinatesAndSelectsAcceptedCohort) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::synthesizeCardExecutable(
      *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, programData, &statistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  expectCompleteTileDomain(*executable);

  EXPECT_GE(statistics.candidateProposals, statistics.shortlistedCandidates);
  EXPECT_GT(statistics.candidateProposals, 0u);
  EXPECT_GT(statistics.resourceScheduleMemoHits, 0u);
  EXPECT_GT(statistics.resourceScheduleMemoMisses, 0u);
  EXPECT_EQ(statistics.materializedCandidates +
                statistics.strictDominatedCandidates +
                statistics.incumbentClosedFeedbackCandidates +
                statistics.feedbackBeamDeferredCandidates +
                statistics.preBufferEquivalentRejections +
                statistics.bufferStructureEquivalentRejections,
            statistics.shortlistedCandidates);
  EXPECT_EQ(statistics.feedbackBeamDeferredCandidates, 0u);
  EXPECT_EQ(statistics.materializationRejections, 0u);
  EXPECT_EQ(statistics.acceptedCandidates, statistics.materializedCandidates);
  EXPECT_EQ(statistics.plannedCandidates, statistics.acceptedCandidates);
  EXPECT_EQ(statistics.schedulePlanRejections, 0u);
  EXPECT_EQ(statistics.exactGates.tileModuleLoweringAttempts,
            statistics.materializedCandidates);
  EXPECT_EQ(statistics.exactGates.cardModuleCompilationInvocations,
            statistics.materializedCandidates);
  EXPECT_EQ(statistics.exactGates.targetLoweringVerificationInvocations,
            statistics.acceptedCandidates);
  EXPECT_EQ(statistics.exactGates.targetTileLoweringVerificationInvocations,
            statistics.acceptedCandidates * 16);
  EXPECT_EQ(statistics.selectedExecutableRematerializations, 1u);
  EXPECT_EQ(statistics.selectedExecutableRematerializationGates
                .tileModuleLoweringAttempts,
            1u);
  EXPECT_EQ(statistics.selectedExecutableRematerializationGates
                .cardModuleCompilationInvocations,
            1u);
  EXPECT_EQ(statistics.selectedStableOrdinal, 0u);
  EXPECT_EQ(statistics.selectedOutputMappingCount, 1u);
  EXPECT_EQ(statistics.selectedUniqueActiveTileCount, 16u);
  EXPECT_EQ(statistics.selectedParallelComponentCount, 1u);
  EXPECT_GT(statistics.enabledDurationTerms, 0u);
  EXPECT_GT(statistics.selectedMakespanPicoseconds, 0u);
  EXPECT_NE(diagnosticsText.find("card-executable-search policy=search"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("card-executable-selection admitted="),
            std::string::npos)
      << diagnosticsText;

  std::string repeatDiagnosticsText;
  llvm::raw_string_ostream repeatDiagnostics(repeatDiagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics
      repeatStatistics;
  auto repeated = wafer::compiler::detail::synthesizeCardExecutable(
      *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), repeatDiagnostics,
      programData, &repeatStatistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  repeatDiagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(repeated)) << repeatDiagnosticsText;
  EXPECT_EQ(repeatStatistics.candidateProposals, statistics.candidateProposals);
  EXPECT_EQ(repeatStatistics.resourceScheduleMemoHits,
            statistics.resourceScheduleMemoHits);
  EXPECT_EQ(repeatStatistics.resourceScheduleMemoMisses,
            statistics.resourceScheduleMemoMisses);
  EXPECT_EQ(repeatStatistics.cheapPrunedCandidates,
            statistics.cheapPrunedCandidates);
  EXPECT_EQ(repeatStatistics.materializedCandidates,
            statistics.materializedCandidates);
  EXPECT_EQ(repeatStatistics.selectedStableOrdinal,
            statistics.selectedStableOrdinal);
  EXPECT_EQ(repeatStatistics.selectedMakespanPicoseconds,
            statistics.selectedMakespanPicoseconds);
  ASSERT_EQ(repeated->executable.tiles.size(),
            executable->executable.tiles.size());
  EXPECT_EQ(repeated->tileDataflowIRTrace, executable->tileDataflowIRTrace);
}

TEST(CardExecutableSynthesisTest,
     SearchActualizesIndependentComponentPlacementCandidate) {
  ParsedProgram parsed = parseBranchProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::synthesizeCardExecutable(
      *parsed.module, branchMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, programData, &statistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  EXPECT_EQ(statistics.dependencyComponentCount, 2u);
  EXPECT_TRUE(statistics.independentComponentPlacementProven);
  EXPECT_GT(statistics.independentComponentCandidateProposals, 0u);
  EXPECT_GT(statistics.independentComponentCandidateMaterializations, 0u);
  EXPECT_GT(statistics.independentComponentCandidateAcceptances, 0u)
      << diagnosticsText;
  EXPECT_GT(statistics.materializedCandidates, 0u);
  EXPECT_NE(diagnosticsText.find("independent_component_acceptances="),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardExecutableSynthesisTest,
     SearchActualizesBoundedNodePlacementCandidate) {
  ParsedProgram parsed = parseDependentProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::synthesizeCardExecutable(
      *parsed.module, dependentProgramMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, programData, &statistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  EXPECT_EQ(statistics.plannedCandidates,
            statistics.acceptedCandidates - statistics.schedulePlanRejections);
  EXPECT_GT(statistics.nodePlacementGroups, 0u);
  EXPECT_GT(statistics.nodePlacementStatesExpanded, 0u);
  EXPECT_GT(statistics.nodePlacementCandidateProposals, 0u);
  EXPECT_GT(statistics.nodePlacementCandidateMaterializations, 0u);
  EXPECT_GT(statistics.nodePlacementCandidateAcceptances, 0u)
      << diagnosticsText;
  EXPECT_GT(statistics.alternativeEdgeActionCandidateProposals, 0u);
  EXPECT_GT(statistics.alternativeEdgeActionCandidateMaterializations, 0u);
  EXPECT_GT(statistics.alternativeEdgeActionCandidateAcceptances, 0u)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("node_placement_proposals="),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardExecutableSynthesisTest,
     SearchActualizesThreeStageTilePipeline) {
  ParsedProgram parsed = parseThreeStageDependentProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::synthesizeCardExecutable(
      *parsed.module, dependentProgramMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, programData, &statistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  EXPECT_GT(statistics.multiStagePlacementCandidateProposals, 0u);
  EXPECT_GT(statistics.multiStagePlacementCandidateMaterializations, 0u);
  EXPECT_GT(statistics.multiStagePlacementCandidateAcceptances, 0u)
      << diagnosticsText;
  EXPECT_GT(statistics.bufferedCandidateProposals, 0u);
  EXPECT_GT(statistics.bufferedCandidateMaterializations, 0u);
  EXPECT_GT(statistics.applicableFusionLogicalEdges, 0u);
  EXPECT_GT(statistics.actualFusionCandidateAcceptances, 0u) << diagnosticsText;
  EXPECT_GT(statistics.selectedActualFusedLogicalEdges, 0u) << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("multi_stage_placement_proposals="),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("multi_stage_placement_acceptances="),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardExecutableSynthesisTest,
     SearchAdmitsLayoutAndRejectsBufferingWithoutOneExactLoop) {
  ParsedProgram parsed = parseLayoutPipelineProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::synthesizeCardExecutable(
      *parsed.module, layoutPipelineProgramMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, programData, &statistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  EXPECT_GT(statistics.layoutConversionCandidateProposals, 0u);
  EXPECT_GT(statistics.layoutConversionCandidateMaterializations, 0u);
  EXPECT_GT(statistics.layoutConversionCandidateAcceptances, 0u)
      << diagnosticsText;
  EXPECT_GT(statistics.layoutBufferedCandidateProposals, 0u);
  EXPECT_GT(statistics.layoutBufferedCandidateMaterializations, 0u);
  EXPECT_EQ(statistics.bufferedCandidateAcceptances, 0u) << diagnosticsText;
  EXPECT_EQ(statistics.rotatingSlotAllocationsMaterialized, 0u)
      << diagnosticsText;
  EXPECT_GT(statistics.applicableFusionLogicalEdges, 0u);
  EXPECT_GT(statistics.actualFusionCandidateAcceptances, 0u) << diagnosticsText;
  EXPECT_GT(statistics.selectedActualFusedLogicalEdges, 0u) << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("layout_buffered_proposals="),
            std::string::npos);
  EXPECT_NE(diagnosticsText.find(
                "no static loop containing every exact logical-edge endpoint"),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardExecutableSynthesisTest,
     SearchTemporalDerivationMakesStrictProgressAcrossAlignedClasses) {
  const wafer::TargetMemoryPolicy memory =
      wafer::getDefaultWaferTargetPolicy().memory;

  // This is the conv-mixed-DAG reduction boundary: 16-way spatial mapping
  // leaves a 1x2 output tile, while the component's large convolution inputs
  // make its aligned tensor-byte scale exceed SPM at that extent. 16 -> 17
  // waves still maps extent 32 back to tile size 2, so the old waves+1 update
  // did not progress. The adjacent distinct class is tile size 1.
  EXPECT_EQ(wafer::compiler::detail::deriveCapacityTemporalShape(
                {1, 2}, /*elementBytes=*/2,
                /*tensorMultiplicity=*/16384, memory,
                /*additionalWaveRefinements=*/0),
            (llvm::SmallVector<int64_t, 4>{1, 1}));

  // Overflow in bytes-per-element-set must saturate before division; wrapping
  // would incorrectly enlarge the capacity budget and retain extent 2.
  EXPECT_EQ(wafer::compiler::detail::deriveCapacityTemporalShape(
                {1, 2}, std::numeric_limits<uint64_t>::max(),
                /*tensorMultiplicity=*/2, memory,
                /*additionalWaveRefinements=*/0),
            (llvm::SmallVector<int64_t, 4>{1, 1}));
}

TEST(CardExecutableSynthesisTest,
     SearchProposesAndMaterializesMultipleReductionIteratorAxes) {
  ParsedProgram parsed = parseTwoReductionAxisProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::synthesizeCardExecutable(
      *parsed.module, twoReductionAxisProgramMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, programData, &statistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  EXPECT_GT(statistics.multiReductionAxisCandidateProposals, 0u);
  EXPECT_GT(statistics.multiReductionAxisCandidateMaterializations, 0u);
  EXPECT_LE(statistics.multiReductionAxisCandidateMaterializations,
            statistics.materializedCandidates);
  EXPECT_NE(diagnosticsText.find("multi_reduction_axis_proposals="),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardExecutableSynthesisTest,
     NoneJointlyRefinesExplicitProducerStageAndConsumerDemand) {
  ParsedProgram parsed = parseLargeProducerStageProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics statistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::synthesizeCardExecutable(
      *parsed.module, largeProducerStageProgramMetadata(), executionConfig(),
      wafer::OptimizationConfig::none(), diagnostics, programData, &statistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  EXPECT_EQ(statistics.materializedCandidates, 0u);
  EXPECT_EQ(statistics.exactGates.cardModuleCompilationInvocations, 1u);
  // P7: the baseline temporal fallback is a functional legalization step
  // and never writes the search feedback bag.
  EXPECT_EQ(statistics.allocationFeedbackTransitions, 0u);
  EXPECT_NE(diagnosticsText.find("card-executable-baseline-temporal-refinement"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_EQ(diagnosticsText.find("exhausted its temporal domain"),
            std::string::npos)
      << diagnosticsText;
}

TEST(CardExecutableSynthesisTest,
     SearchUsesFiniteTemporalTraversalWhenOneWaveExceedsSPM) {
  ParsedProgram baselineProgram = parseLargeTemporalProgram();
  ASSERT_TRUE(baselineProgram.module);
  std::string baselineDiagnosticsText;
  llvm::raw_string_ostream baselineDiagnostics(baselineDiagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics
      baselineStatistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto baseline = wafer::compiler::detail::synthesizeCardExecutable(
      *baselineProgram.module, largeTemporalProgramMetadata(),
      executionConfig(), wafer::OptimizationConfig::none(), baselineDiagnostics, programData,
      &baselineStatistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  baselineDiagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(baseline)) << baselineDiagnosticsText;
  // The deterministic controller consumes proven region-capacity overflows
  // through the finite temporal domain, then compiles the complete
  // CardExecutable
  // exactly once. It never constructs the search placement enumeration or
  // shortlist.
  EXPECT_EQ(baselineStatistics.shortlistedCandidates, 0u);
  EXPECT_EQ(baselineStatistics.materializedCandidates, 0u);
  EXPECT_EQ(baselineStatistics.materializationRejections, 0u);
  EXPECT_EQ(baselineStatistics.baselineCardModuleMaterializations, 2u);
  EXPECT_GT(baselineStatistics.baselineScopedCardModuleMaterializations, 0u);
  EXPECT_GT(baselineStatistics.baselineRegionSPMCapacityOverflowProofs, 0u);
  EXPECT_GT(baselineStatistics.baselineRegionSPMCapacityOverflowProofs,
            baselineStatistics.allocationFeedbackTransitions);
  EXPECT_GT(baselineStatistics.baselineMaximumRegionSPMQueryWorkers, 1u);
  // Exact allocator conflicts can carry several tied structured node
  // relations.
  // The deterministic controller composes their temporal changes in one
  // mutable baseline state. It never creates candidate siblings, progressive
  // promotions, priority selections or lookahead states.
  EXPECT_EQ(baselineStatistics.allocationFeedbackCandidates, 0u);
  EXPECT_EQ(baselineStatistics.spmFailureProbeAttempts, 0u);
  EXPECT_EQ(baselineStatistics.allocationFeedbackProgressivePromotions, 0u);
  EXPECT_EQ(baselineStatistics.allocationFeedbackLookaheadCandidates, 0u);
  EXPECT_EQ(baselineStatistics.allocationFeedbackPrioritySelections, 0u);
  EXPECT_EQ(baselineStatistics.spmFailureProbeTilesSkipped, 0u);
  EXPECT_NE(baselineDiagnosticsText.find(
                "tile-region-spm-capacity outcome=capacity-exceeded"),
            std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_EQ(baselineStatistics.exactGates.cardModuleCompilationInvocations,
            1u);
  EXPECT_EQ(baselineDiagnosticsText.find(
                "card-executable-compilation outcome=exact-rejection"),
            std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_NE(
      baselineDiagnosticsText.find("card-exact-spm-conflict-certificate"),
      std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_EQ(
      baselineDiagnosticsText.find("tile-execution-allocation-feedback-joint"),
      std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_EQ(baselineStatistics.acceptedCandidates, 0u);
  EXPECT_EQ(baselineStatistics.selectedExecutableRematerializations, 0u);
  EXPECT_EQ(baselineStatistics.selectedTemporalWaveLowerBound, 0u);
  EXPECT_LE(baselineStatistics.selectedPeakAlignedResidencyEstimate,
            3080192u - 65536u);
  for (llvm::StringRef tileDataflowIR : baseline->tileDataflowIRTrace) {
    EXPECT_NE(tileDataflowIR.find("scf.for"), llvm::StringRef::npos)
        << tileDataflowIR.str();
    const size_t stores = countOccurrences(tileDataflowIR, "wafer.tile.store");
    EXPECT_GT(stores, 1u);
    EXPECT_LE(stores, 81u);
    EXPECT_NE(tileDataflowIR.find("memref.dealloc"), llvm::StringRef::npos)
        << tileDataflowIR.str();
  }

  ParsedProgram searchProgram = parseLargeTemporalProgram();
  ASSERT_TRUE(searchProgram.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics statistics;
  auto executable = wafer::compiler::detail::synthesizeCardExecutable(
      *searchProgram.module, largeTemporalProgramMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, programData, &statistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  ASSERT_EQ(executable->executable.tiles.size(), 16u);
  EXPECT_EQ(statistics.plannedCandidates,
            statistics.acceptedCandidates - statistics.schedulePlanRejections);
  EXPECT_GE(statistics.candidateProposals, statistics.shortlistedCandidates);
  EXPECT_GT(statistics.materializedCandidates, 0u);
  EXPECT_GT(statistics.selectedTemporalWaveLowerBound, 1u);
  EXPECT_GT(statistics.selectedInstructionExecutionLowerBound, 1u);
  EXPECT_GT(statistics.selectedPeakOutputTileFootprintEstimate, 0u);
  EXPECT_NE(
      baselineDiagnosticsText.find("card-executable-baseline-temporal-refinement"),
      std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_EQ(
      baselineDiagnosticsText.find("tile-execution-allocation-feedback-priority"),
      std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_LE(statistics.selectedPeakAlignedResidencyEstimate, 3080192u - 65536u);
  // Prologue, repeated steady iterations and tail execute sequentially.  They
  // must not multiply simultaneous SPM residency; this generic has three
  // tensor operands plus one tensor result live in the conservative estimate.
  const uint64_t alignedTileBytes =
      ((statistics.selectedPeakOutputTileFootprintEstimate + 255) / 256) * 256;
  EXPECT_EQ(statistics.selectedPeakAlignedResidencyEstimate,
            alignedTileBytes * 4);
  for (llvm::StringRef tileDataflowIR : executable->tileDataflowIRTrace) {
    EXPECT_NE(tileDataflowIR.find("scf.for"), llvm::StringRef::npos)
        << tileDataflowIR.str();
    // Up to four iterator dimensions contribute compact
    // prologue/steady/tail sites; runtime wave count is never host-expanded.
    const size_t stores = countOccurrences(tileDataflowIR, "wafer.tile.store");
    EXPECT_GT(stores, 1u);
    EXPECT_LE(stores, 81u) << tileDataflowIR.str();
    EXPECT_NE(tileDataflowIR.find("memref.dealloc"), llvm::StringRef::npos)
        << tileDataflowIR.str();
  }

  ParsedProgram repeatedProgram = parseLargeTemporalProgram();
  ASSERT_TRUE(repeatedProgram.module);
  std::string repeatedDiagnosticsText;
  llvm::raw_string_ostream repeatedDiagnostics(repeatedDiagnosticsText);
  wafer::compiler::detail::CardExecutableSynthesisStatistics
      repeatedStatistics;
  auto repeated = wafer::compiler::detail::synthesizeCardExecutable(
      *repeatedProgram.module, largeTemporalProgramMetadata(),
      executionConfig(), wafer::OptimizationConfig::search(),
      repeatedDiagnostics, programData, &repeatedStatistics, /*tilePipelineParallelism=*/0, /*requestTileIRTrace=*/true);
  repeatedDiagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(repeated)) << repeatedDiagnosticsText;
  EXPECT_EQ(repeatedStatistics.candidateProposals,
            statistics.candidateProposals);
  EXPECT_EQ(repeatedStatistics.materializedCandidates,
            statistics.materializedCandidates);
  EXPECT_EQ(repeatedStatistics.selectedStableOrdinal,
            statistics.selectedStableOrdinal);
  EXPECT_EQ(repeatedStatistics.selectedTemporalWaveLowerBound,
            statistics.selectedTemporalWaveLowerBound);
  ASSERT_EQ(repeated->executable.tiles.size(),
            executable->executable.tiles.size());
  EXPECT_EQ(repeated->tileDataflowIRTrace, executable->tileDataflowIRTrace);
}

} // namespace
