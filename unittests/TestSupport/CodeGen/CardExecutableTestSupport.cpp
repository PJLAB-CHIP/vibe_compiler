//===- CardExecutableTestSupport.cpp - Compiler executable fixtures -------===//

#include "CardExecutableTestSupport.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <limits>
#include <utility>

namespace wafer::compiler::testing {

wafer::frontend::ProgramPartitionSlice
singlePartitionSlice(llvm::ArrayRef<int64_t> shape) {
  wafer::frontend::ProgramPartitionSlice slice;
  slice.partitionId = 0;
  slice.replicaId = 0;
  slice.offsets.assign(shape.size(), 0);
  slice.sizes.assign(shape.begin(), shape.end());
  slice.strides.assign(shape.size(), 1);
  return slice;
}

wafer::frontend::ProgramBoundaryBinding
boundary(int64_t index, llvm::ArrayRef<int64_t> shape) {
  wafer::frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  binding.globalShape.assign(shape.begin(), shape.end());
  binding.localShape.assign(shape.begin(), shape.end());
  binding.dtype = ProgramElementType::F16;
  binding.partitionSlices.push_back(singlePartitionSlice(shape));
  return binding;
}

wafer::frontend::FrontendProgramVerificationResult programMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {2, 1024, 128}),
                               boundary(1, {2, 1024, 128})};
  program.distributedOutputs = {boundary(0, {2, 1024, 128})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult branchMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {1, 32}), boundary(1, {2, 8})};
  program.distributedOutputs = {boundary(0, {1, 32}), boundary(1, {2, 8})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult dependentProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary(0, {1, 320})};
  program.distributedOutputs = {boundary(0, {1, 320})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult
largeTemporalProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {2, 1025, 8192}),
                               boundary(1, {2, 1025, 8192})};
  program.distributedOutputs = {boundary(0, {2, 1025, 8192})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult
largeProducerStageProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary(0, {2, 1025, 8192})};
  program.distributedOutputs = {boundary(0, {2, 1025, 8192})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult
largeTransposedWeightProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {16, 4096}),
                               boundary(1, {11008, 4096})};
  program.distributedOutputs = {boundary(0, {16, 11008})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult
layoutPipelineProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {128, 128}),
                               boundary(1, {128, 128})};
  program.distributedOutputs = {boundary(0, {128, 128})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult
twoReductionAxisProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {1, 10, 11}), boundary(1, {1})};
  program.distributedOutputs = {boundary(0, {1})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult broadcastProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary(0, {16})};
  program.distributedOutputs = {boundary(0, {16, 8})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult
reductionDemandProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary(0, {16, 8})};
  program.distributedOutputs = {boundary(0, {16})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult
windowDemandProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {1, 18, 18, 1}),
                               boundary(1, {3, 3, 1, 1})};
  program.distributedOutputs = {boundary(0, {1, 16, 16, 1})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult
stridedDemandProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 1;
  program.distributedInputs = {boundary(0, {32})};
  program.distributedOutputs = {boundary(0, {16})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult
multiPieceDemandProgramMetadata() {
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {16}), boundary(1, {4})};
  program.distributedOutputs = {boundary(0, {16})};
  return program;
}

wafer::frontend::FrontendProgramVerificationResult
multiProducerJoinProgramMetadata(int64_t extent) {
  const int64_t leftExtent = extent / 2;
  const int64_t rightExtent = extent - leftExtent;
  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 2;
  program.distributedInputs = {boundary(0, {2, leftExtent, 128}),
                               boundary(1, {2, rightExtent, 128})};
  program.distributedOutputs = {boundary(0, {2, extent, 128})};
  return program;
}

ParsedProgram parseProgram() {
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
  func.func @main(%lhs: tensor<2x1024x128xf16>,
                  %rhs: tensor<2x1024x128xf16>)
      -> tensor<2x1024x128xf16> {
    %out = tensor.empty() : tensor<2x1024x128xf16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%lhs, %rhs : tensor<2x1024x128xf16>,
                            tensor<2x1024x128xf16>)
        outs(%out : tensor<2x1024x128xf16>) {
      ^bb0(%a: f16, %b: f16, %old: f16):
        %value = arith.addf %a, %b : f16
        linalg.yield %value : f16
    } -> tensor<2x1024x128xf16>
    return %sum : tensor<2x1024x128xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

ParsedProgram parseBranchProgram() {
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

ParsedProgram parseDependentProgram() {
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

ParsedProgram parseThreeStageDependentProgram() {
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

ParsedProgram parseLargeTemporalProgram() {
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
  func.func @main(%lhs: tensor<2x1025x8192xf16>,
                  %rhs: tensor<2x1025x8192xf16>)
      -> tensor<2x1025x8192xf16> {
    %out = tensor.empty() : tensor<2x1025x8192xf16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%lhs, %rhs : tensor<2x1025x8192xf16>,
                            tensor<2x1025x8192xf16>)
        outs(%out : tensor<2x1025x8192xf16>) {
      ^bb0(%a: f16, %b: f16, %old: f16):
        %value = arith.addf %a, %b : f16
        linalg.yield %value : f16
    } -> tensor<2x1025x8192xf16>
    return %sum : tensor<2x1025x8192xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

ParsedProgram parseLargeProducerStageProgram() {
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
  func.func @main(%input: tensor<2x1025x8192xf16>)
      -> tensor<2x1025x8192xf16> {
    %producer_init = tensor.empty() : tensor<2x1025x8192xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%input : tensor<2x1025x8192xf16>)
        outs(%producer_init : tensor<2x1025x8192xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1025x8192xf16>
    %result_init = tensor.empty() : tensor<2x1025x8192xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%producer : tensor<2x1025x8192xf16>)
        outs(%result_init : tensor<2x1025x8192xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1025x8192xf16>
    return %result : tensor<2x1025x8192xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

ParsedProgram parseLargeTransposedWeightProgram() {
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
  func.func @main(%activation: tensor<16x4096xf16>,
                  %weight: tensor<11008x4096xf16>)
      -> tensor<16x11008xf16> {
    %weight_out = tensor.empty() : tensor<4096x11008xf16>
    %transposed = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%weight : tensor<11008x4096xf16>)
        outs(%weight_out : tensor<4096x11008xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<4096x11008xf16>
    %zero = arith.constant 0.0 : f16
    %result_out = tensor.empty() : tensor<16x11008xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%result_out : tensor<16x11008xf16>) -> tensor<16x11008xf16>
    %result = linalg.matmul
        ins(%activation, %transposed : tensor<16x4096xf16>,
             tensor<4096x11008xf16>)
        outs(%init : tensor<16x11008xf16>) -> tensor<16x11008xf16>
    return %result : tensor<16x11008xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

ParsedProgram parseLayoutPipelineProgram() {
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

ParsedProgram parseTwoReductionAxisProgram() {
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
ParsedProgram parseBroadcastProgram() {
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
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%resultOut : tensor<16x8xf16>) -> tensor<16x8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%producer : tensor<16xf16>)
        outs(%init : tensor<16x8xf16>) {
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

ParsedProgram parseReductionDemandProgram() {
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

ParsedProgram parseWindowDemandProgram() {
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

ParsedProgram parseStridedDemandProgram() {
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

ParsedProgram parseMultiPieceDemandProgram() {
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

ParsedProgram parseMultiProducerJoinProgram(int64_t extent) {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  const int64_t leftExtent = extent / 2;
  const int64_t rightExtent = extent - leftExtent;
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%left: tensor<2x)mlir"
         << leftExtent << "x128xf16>, %right: tensor<2x" << rightExtent
         << "x128xf16>) -> tensor<2x" << extent << "x128xf16> {\n"
         << "    %leftOut = tensor.empty() : tensor<2x" << leftExtent
         << "x128xf16>\n"
         << "    %producerA = linalg.generic {indexing_maps = [#id, #id], "
            "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
         << "        ins(%left : tensor<2x" << leftExtent
         << "x128xf16>) outs(%leftOut : tensor<2x" << leftExtent
         << "x128xf16>) {\n"
         << "      ^bb0(%value: f16, %old: f16):\n"
         << "        %next = arith.addf %value, %value : f16\n"
         << "        linalg.yield %next : f16\n"
         << "    } -> tensor<2x" << leftExtent << "x128xf16>\n"
         << "    %rightOut = tensor.empty() : tensor<2x" << rightExtent
         << "x128xf16>\n"
         << "    %producerB = linalg.generic {indexing_maps = [#id, #id], "
            "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
         << "        ins(%right : tensor<2x" << rightExtent
         << "x128xf16>) outs(%rightOut : tensor<2x" << rightExtent
         << "x128xf16>) {\n"
         << "      ^bb0(%value: f16, %old: f16):\n"
         << "        %next = arith.mulf %value, %value : f16\n"
         << "        linalg.yield %next : f16\n"
         << "    } -> tensor<2x" << rightExtent << "x128xf16>\n"
         << "    %empty = tensor.empty() : tensor<2x" << extent
         << "x128xf16>\n"
         << "    %lower = tensor.insert_slice %producerA into %empty"
         << "[0, 0, 0] [2, " << leftExtent
         << ", 128] [1, 1, 1] : tensor<2x" << leftExtent
         << "x128xf16> into tensor<2x" << extent << "x128xf16>\n"
         << "    %assembled = tensor.insert_slice %producerB into %lower"
         << "[0, " << leftExtent << ", 0] [2, " << rightExtent
         << ", 128] [1, 1, 1] : tensor<2x" << rightExtent
         << "x128xf16> into tensor<2x" << extent << "x128xf16>\n"
         << "    %resultOut = tensor.empty() : tensor<2x" << extent
         << "x128xf16>\n"
         << "    %result = linalg.generic {indexing_maps = [#id, #id], "
            "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
         << "        ins(%assembled : tensor<2x" << extent
         << "x128xf16>) outs(%resultOut : tensor<2x" << extent
         << "x128xf16>) {\n"
         << "      ^bb0(%value: f16, %old: f16):\n"
         << "        %next = arith.addf %value, %value : f16\n"
         << "        linalg.yield %next : f16\n"
         << "    } -> tensor<2x" << extent << "x128xf16>\n"
         << "    return %result : tensor<2x" << extent << "x128xf16>\n"
         << "  }\n"
         << "}\n";
  stream.flush();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      source,
      mlir::ParserConfig(context.get()));
  return ParsedProgram{std::move(context), std::move(module)};
}

size_t countOccurrences(llvm::StringRef text, llvm::StringRef needle) {
  size_t count = 0;
  while (true) {
    size_t position = text.find(needle);
    if (position == llvm::StringRef::npos)
      return count;
    ++count;
    text = text.drop_front(position + needle.size());
  }
}

wafer::compiler::ExecutionConfig executionConfig() {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  EXPECT_TRUE(static_cast<bool>(config));
  return *config;
}

void expectCompleteTileDomain(
    const wafer::compiler::detail::CardExecutableLoweringResult &executable,
    llvm::ArrayRef<std::string> tileDataflowIRTrace) {
  ASSERT_EQ(executable.tiles.size(), 16u);
  ASSERT_EQ(tileDataflowIRTrace.size(), executable.tiles.size());
  for (size_t index = 0; index < executable.tiles.size(); ++index) {
    const wafer::compiler::TileExecutable &tile = executable.tiles[index];
    llvm::StringRef tileDataflowIR = tileDataflowIRTrace[index];
    EXPECT_EQ(tile.getCardId(), wafer::CardId(0));
    EXPECT_EQ(tile.getTileId(), wafer::TileId(static_cast<int64_t>(index)));
    EXPECT_FALSE(tileDataflowIR.empty());
    EXPECT_NE(tileDataflowIR.find("wafer.tile.region"), llvm::StringRef::npos);
    EXPECT_NE(tileDataflowIR.find("wafer.tile.load"), llvm::StringRef::npos);
    EXPECT_NE(tileDataflowIR.find("wafer.tile.store"), llvm::StringRef::npos);
  }
}

void expectDemandProgramCompletesExecutableGate(
    ParsedProgram &parsed,
    const wafer::frontend::FrontendProgramVerificationResult &metadata) {
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::BaselineStatistics baselineStatistics;
  wafer::compiler::ProgramDataHandoff programData;
  auto executable = wafer::compiler::detail::compileCardBaseline(
      *parsed.module, metadata, executionConfig(), diagnostics, programData,
      &baselineStatistics, /*tilePipelineParallelism=*/0,
      /*captureTileDataflowIRTrace=*/true);
  diagnostics.flush();

  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  expectCompleteTileDomain(executable->executable,
                           executable->tileDataflowIRTrace);
  EXPECT_GT(baselineStatistics.exactDemandSatisfiedEdges, 0u);
  EXPECT_EQ(baselineStatistics.exactGates.cardModuleCompilationInvocations, 1u);
  // Baseline structural contract: every TileRegion carries exactly one
  // structured compute root; an in-region operand demand of a foreign node
  // never counts as a second root.
  EXPECT_EQ(diagnosticsText.find("multiple structured compute roots"),
            std::string::npos)
      << diagnosticsText;
}

} // namespace wafer::compiler::testing
