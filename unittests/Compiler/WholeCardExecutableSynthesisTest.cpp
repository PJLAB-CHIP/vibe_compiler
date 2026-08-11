//===- WholeCardExecutableSynthesisTest.cpp -----------------------------===//

#include "../../lib/Wafer/Compiler/WholeCardExecutableSynthesis.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"
#include "../../lib/Wafer/Compiler/SelectedBufferMaterialization.h"
#include "../../lib/Wafer/Compiler/WholeDAGSchedulePlan.h"

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Compiler/Compilation.h"
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
        %sum = arith.addf %value, %acc fastmath<reassoc> : f16
        linalg.yield %sum : f16
    } -> tensor<1xf16>
    return %result : tensor<1xf16>
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
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::RuntimeLaunchKind::Kernel);
  EXPECT_TRUE(static_cast<bool>(config));
  return *config;
}

enum class TestLineageMode {
  Complete,
  MissingFirstNode,
  MissingLastNode,
  FusedFirstTwo,
  Ambiguous,
};

static mlir::FailureOr<wafer::compiler::detail::AcceptedWholeCardExecutable>
makePlanTestExecutable(
    mlir::MLIRContext &context,
    llvm::ArrayRef<wafer::CardProgramSourceOperationLineage> sourceLineage,
    llvm::ArrayRef<wafer::compiler::detail::WholeDAGNodePlacement> placements,
    TestLineageMode mode) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  llvm::SmallVector<wafer::analysis::PhysicalTileInstructionProgram, 16>
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
    for (const wafer::compiler::detail::WholeDAGNodePlacement &placement :
         placements) {
      if (!llvm::is_contained(placement.tiles, wafer::PhysicalTileId(tileId)))
        continue;
      if (placement.node >= sourceLineage.size() ||
          markerIndex >= markers.size())
        return mlir::failure();
      if ((mode == TestLineageMode::MissingFirstNode && placement.node == 0) ||
          (mode == TestLineageMode::MissingLastNode &&
           placement.node + 1 == sourceLineage.size()))
        continue;
      mlir::Operation *marker = markers[markerIndex++].getOperation();
      marker->setLoc(mlir::OpaqueLoc::get<
                     const wafer::CardProgramSourceOperationLineage *>(
          &sourceLineage[placement.node], marker->getLoc()));
    }
    if ((mode == TestLineageMode::FusedFirstTwo ||
         mode == TestLineageMode::Ambiguous) &&
        tileId == 0 && sourceLineage.size() >= 2) {
      mlir::Operation *marker = markers.front().getOperation();
      llvm::SmallVector<mlir::Location, 2> locations;
      for (size_t node = 0; node < 2; ++node)
        locations.push_back(mlir::OpaqueLoc::get<
                            const wafer::CardProgramSourceOperationLineage *>(
            &sourceLineage[node], marker->getLoc()));
      marker->setLoc(mlir::FusedLoc::get(&context, locations));
    }

    mlir::func::FuncOp entry =
        module->lookupSymbol<mlir::func::FuncOp>("entry");
    if (!entry)
      return mlir::failure();
    programs.push_back({wafer::PhysicalTileId(tileId), entry.getOperation()});
    modules.push_back(std::move(module));
  }

  wafer::analysis::WholeCardInstructionProgramCost cost =
      wafer::analysis::analyzeWholeCardInstructionProgramCost(
          programs, wafer::analysis::getTargetScheduleCostPolicy());
  std::vector<wafer::compiler::PhysicalTileExecutable> tiles;
  tiles.reserve(modules.size());
  for (size_t tile = 0; tile < modules.size(); ++tile)
    tiles.push_back(wafer::compiler::ExecutableBundleBuilder::makePhysicalTile(
        wafer::PhysicalCardId(0),
        wafer::PhysicalTileId(static_cast<int64_t>(tile)),
        wafer::LaunchSlotId(static_cast<int64_t>(tile)),
        std::move(modules[tile]), "entry", {},
        wafer::compiler::TransportContract::None, "plan-test"));
  wafer::RuntimeLaunchContract launch =
      llvm::cantFail(wafer::RuntimeLaunchContract::createKernel(
          wafer::KernelLaunchForm::Grid,
          wafer::KernelEntryABI::TileMajorPointerTable,
          {wafer::RuntimeLaunchPhaseRole::Main}));
  return wafer::compiler::detail::AcceptedWholeCardExecutable(
      std::move(tiles), std::move(launch), std::move(cost));
}

static mlir::FailureOr<wafer::compiler::detail::CardDAGAnalysis>
analyzePlanTestDAG(mlir::ModuleOp module) {
  mlir::func::FuncOp function = *module.getOps<mlir::func::FuncOp>().begin();
  return wafer::compiler::detail::CardDAGAnalysis::create(function);
}

static llvm::SmallVector<wafer::CardProgramSourceOperationLineage, 2>
makePlanTestLineage(const wafer::compiler::detail::CardDAGAnalysis &dag) {
  llvm::SmallVector<wafer::CardProgramSourceOperationLineage, 2> lineage;
  lineage.reserve(dag.getNodes().size());
  for (const wafer::compiler::detail::CardDAGNode &node : dag.getNodes())
    lineage.push_back({node.operation, node.id});
  return lineage;
}

static void expectCompletePhysicalTileDomain(
    const wafer::compiler::detail::AcceptedWholeCardExecutable &executable) {
  EXPECT_FALSE(
      wafer::compiler::detail::containsCardProgramSourceOperationLineage(
          executable));
  ASSERT_EQ(executable.tiles.size(), 16u);
  for (size_t index = 0; index < executable.tiles.size(); ++index) {
    const wafer::compiler::PhysicalTileExecutable &tile =
        executable.tiles[index];
    EXPECT_EQ(tile.getPhysicalCardId(), wafer::PhysicalCardId(0));
    EXPECT_EQ(tile.getPhysicalTileId(),
              wafer::PhysicalTileId(static_cast<int64_t>(index)));
    EXPECT_FALSE(tile.getSelectedTileIR().empty());
    EXPECT_NE(tile.getSelectedTileIR().find("wafer.tile.region"),
              llvm::StringRef::npos);
    EXPECT_NE(tile.getSelectedTileIR().find("wafer.tile.load"),
              llvm::StringRef::npos);
    EXPECT_NE(tile.getSelectedTileIR().find("wafer.tile.store"),
              llvm::StringRef::npos);
  }
}

TEST(WholeCardExecutableSynthesisTest,
     AcceptedPlanConservesResidualAndRunsDisjointNodePhasesTogether) {
  ParsedProgram parsed = parseBranchProgram();
  ASSERT_TRUE(parsed.module);
  auto dag = analyzePlanTestDAG(*parsed.module);
  ASSERT_TRUE(mlir::succeeded(dag));
  ASSERT_EQ(dag->getNodes().size(), 2u);
  auto lineage = makePlanTestLineage(*dag);

  llvm::SmallVector<wafer::compiler::detail::WholeDAGNodePlacement, 2>
      placements;
  placements.push_back({0, 0, {wafer::PhysicalTileId(0)}});
  placements.push_back({1, 0, {wafer::PhysicalTileId(1)}});
  auto executable = makePlanTestExecutable(*parsed.context, lineage, placements,
                                           TestLineageMode::Complete);
  ASSERT_TRUE(mlir::succeeded(executable));

  llvm::SmallVector<wafer::analysis::WholeCardInstructionProgramCost, 3>
      phaseCosts;
  std::string failureReason;
  auto plan = wafer::compiler::detail::buildAcceptedWholeDAGSchedulePlan(
      *dag, placements, lineage, *executable, phaseCosts, &failureReason);
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
  EXPECT_TRUE(
      wafer::compiler::detail::containsCardProgramSourceOperationLineage(
          *executable));
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::stripCardProgramSourceOperationLineage(
          *executable, lineage, &failureReason)))
      << failureReason;
  EXPECT_FALSE(
      wafer::compiler::detail::containsCardProgramSourceOperationLineage(
          *executable));
}

TEST(WholeCardExecutableSynthesisTest,
     AcceptedPlanKeepsDependentNodePhasesInDAGOrder) {
  ParsedProgram parsed = parseDependentProgram();
  ASSERT_TRUE(parsed.module);
  auto dag = analyzePlanTestDAG(*parsed.module);
  ASSERT_TRUE(mlir::succeeded(dag));
  ASSERT_EQ(dag->getNodes().size(), 2u);
  ASSERT_EQ(dag->getEdges().size(), 1u);
  auto lineage = makePlanTestLineage(*dag);

  llvm::SmallVector<wafer::compiler::detail::WholeDAGNodePlacement, 2>
      placements;
  placements.push_back({0, 0, {wafer::PhysicalTileId(0)}});
  placements.push_back({1, 0, {wafer::PhysicalTileId(0)}});
  auto executable = makePlanTestExecutable(*parsed.context, lineage, placements,
                                           TestLineageMode::Complete);
  ASSERT_TRUE(mlir::succeeded(executable));

  llvm::SmallVector<wafer::analysis::WholeCardInstructionProgramCost, 3>
      phaseCosts;
  std::string failureReason;
  auto plan = wafer::compiler::detail::buildAcceptedWholeDAGSchedulePlan(
      *dag, placements, lineage, *executable, phaseCosts, &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  ASSERT_EQ(plan->getSteps().size(), 5u);
  for (size_t step = 3; step < plan->getSteps().size(); ++step) {
    EXPECT_EQ(plan->getSteps()[step].kind,
              wafer::analysis::StaticScheduleStep::Kind::Stage);
    EXPECT_EQ(plan->getSteps()[step].stage.independentBranches.size(), 1u);
  }
}

TEST(WholeCardExecutableSynthesisTest,
     AcceptedPlanAllowsEliminatedInternalNodeCoveredByItsSuccessor) {
  ParsedProgram parsed = parseDependentProgram();
  ASSERT_TRUE(parsed.module);
  auto dag = analyzePlanTestDAG(*parsed.module);
  ASSERT_TRUE(mlir::succeeded(dag));
  ASSERT_EQ(dag->getNodes().size(), 2u);
  ASSERT_EQ(dag->getEdges().size(), 1u);
  auto lineage = makePlanTestLineage(*dag);

  llvm::SmallVector<wafer::compiler::detail::WholeDAGNodePlacement, 2>
      placements;
  placements.push_back({0, 0, {wafer::PhysicalTileId(0)}});
  placements.push_back({1, 0, {wafer::PhysicalTileId(0)}});
  auto executable = makePlanTestExecutable(*parsed.context, lineage, placements,
                                           TestLineageMode::MissingFirstNode);
  ASSERT_TRUE(mlir::succeeded(executable));

  llvm::SmallVector<wafer::analysis::WholeCardInstructionProgramCost, 3>
      phaseCosts;
  std::string failureReason;
  auto plan = wafer::compiler::detail::buildAcceptedWholeDAGSchedulePlan(
      *dag, placements, lineage, *executable, phaseCosts, &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  ASSERT_EQ(phaseCosts.size(), 3u);
  EXPECT_EQ(phaseCosts[0].aggregateInstructionCount.value, 47u);
  EXPECT_EQ(phaseCosts[1].aggregateInstructionCount.value, 0u);
  EXPECT_EQ(phaseCosts[2].aggregateInstructionCount.value, 1u);
}

TEST(WholeCardExecutableSynthesisTest,
     AcceptedPlanAttributesFusedLineageOnceToUniqueDownstreamNode) {
  ParsedProgram parsed = parseDependentProgram();
  ASSERT_TRUE(parsed.module);
  auto dag = analyzePlanTestDAG(*parsed.module);
  ASSERT_TRUE(mlir::succeeded(dag));
  ASSERT_EQ(dag->getNodes().size(), 2u);
  ASSERT_EQ(dag->getEdges().size(), 1u);
  auto lineage = makePlanTestLineage(*dag);

  llvm::SmallVector<wafer::compiler::detail::WholeDAGNodePlacement, 2>
      placements;
  placements.push_back({0, 0, {wafer::PhysicalTileId(0)}});
  placements.push_back({1, 0, {wafer::PhysicalTileId(0)}});
  auto executable = makePlanTestExecutable(*parsed.context, lineage, placements,
                                           TestLineageMode::FusedFirstTwo);
  ASSERT_TRUE(mlir::succeeded(executable));

  llvm::SmallVector<wafer::analysis::WholeCardInstructionProgramCost, 3>
      phaseCosts;
  std::string failureReason;
  auto plan = wafer::compiler::detail::buildAcceptedWholeDAGSchedulePlan(
      *dag, placements, lineage, *executable, phaseCosts, &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  ASSERT_EQ(phaseCosts.size(), 3u);
  EXPECT_EQ(phaseCosts[0].aggregateInstructionCount.value, 46u);
  EXPECT_EQ(phaseCosts[1].aggregateInstructionCount.value, 0u);
  EXPECT_EQ(phaseCosts[2].aggregateInstructionCount.value, 2u);
}

TEST(WholeCardExecutableSynthesisTest,
     AcceptedPlanRejectsLostAndAmbiguousSourceLineage) {
  ParsedProgram parsed = parseBranchProgram();
  ASSERT_TRUE(parsed.module);
  auto dag = analyzePlanTestDAG(*parsed.module);
  ASSERT_TRUE(mlir::succeeded(dag));
  ASSERT_EQ(dag->getNodes().size(), 2u);
  auto lineage = makePlanTestLineage(*dag);
  llvm::SmallVector<wafer::compiler::detail::WholeDAGNodePlacement, 2>
      placements;
  placements.push_back({0, 0, {wafer::PhysicalTileId(0)}});
  placements.push_back({1, 0, {wafer::PhysicalTileId(1)}});

  for (TestLineageMode mode :
       {TestLineageMode::MissingLastNode, TestLineageMode::Ambiguous}) {
    auto executable =
        makePlanTestExecutable(*parsed.context, lineage, placements, mode);
    ASSERT_TRUE(mlir::succeeded(executable));
    llvm::SmallVector<wafer::analysis::WholeCardInstructionProgramCost, 3>
        phaseCosts;
    std::string failureReason;
    auto plan = wafer::compiler::detail::buildAcceptedWholeDAGSchedulePlan(
        *dag, placements, lineage, *executable, phaseCosts, &failureReason);
    EXPECT_TRUE(mlir::failed(plan));
    EXPECT_FALSE(failureReason.empty());
    ASSERT_TRUE(mlir::succeeded(
        wafer::compiler::detail::stripCardProgramSourceOperationLineage(
            *executable, lineage, &failureReason)))
        << failureReason;
  }
}

TEST(WholeCardExecutableSynthesisTest,
     NoneMaterializesOneBalancedBaselineThroughExactGates) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::WholeCardExecutableSynthesisStatistics statistics;
  auto executable = wafer::compiler::detail::synthesizeWholeCardExecutable(
      *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::none(), diagnostics, &statistics);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  expectCompletePhysicalTileDomain(*executable);

  EXPECT_EQ(statistics.structuredNodeCount, 1u);
  EXPECT_EQ(statistics.structuredEdgeCount, 0u);
  EXPECT_EQ(statistics.candidateProposals, 1u);
  EXPECT_EQ(statistics.cheapPrunedCandidates, 0u);
  EXPECT_EQ(statistics.shortlistedCandidates, 1u);
  EXPECT_EQ(statistics.materializedCandidates, 1u);
  EXPECT_EQ(statistics.materializationRejections, 0u);
  EXPECT_EQ(statistics.acceptedCandidates, 1u);
  EXPECT_EQ(statistics.plannedCandidates, 1u);
  EXPECT_EQ(statistics.schedulePlanRejections, 0u);
  EXPECT_EQ(statistics.selectedOutputMappingCount, 1u);
  EXPECT_EQ(statistics.selectedUniqueActiveTileCount, 16u);
  EXPECT_EQ(statistics.selectedParallelComponentCount, 1u);
  EXPECT_EQ(statistics.exactGates.preTargetAttempts, 1u);
  EXPECT_EQ(statistics.exactGates.targetGateInvocations, 1u);
  EXPECT_EQ(statistics.exactGates.targetTileGateInvocations, 16u);
  EXPECT_EQ(statistics.selectedExecutableRematerializations, 0u);
  EXPECT_EQ(
      statistics.selectedExecutableRematerializationGates.preTargetAttempts,
      0u);
  EXPECT_NE(diagnosticsText.find("whole-card-search policy=none"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find(
                "selected baseline reuses exact-admitted executable"),
            std::string::npos)
      << diagnosticsText;
}

TEST(WholeCardExecutableSynthesisTest,
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
  ASSERT_TRUE(
      mlir::succeeded(wafer::compiler::detail::materializeSelectedBuffering(
          module, /*requestedBufferCount=*/2, &slotAllocations,
          &failureReason)))
      << failureReason;
  EXPECT_EQ(slotAllocations, 2u);
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

TEST(WholeCardExecutableSynthesisTest,
     SearchUsesFactorizedCoordinatesAndSelectsAcceptedCohort) {
  ParsedProgram parsed = parseProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::WholeCardExecutableSynthesisStatistics statistics;
  auto executable = wafer::compiler::detail::synthesizeWholeCardExecutable(
      *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, &statistics);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  expectCompletePhysicalTileDomain(*executable);

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
  EXPECT_EQ(statistics.exactGates.preTargetAttempts,
            statistics.materializedCandidates);
  EXPECT_EQ(statistics.exactGates.targetGateInvocations,
            statistics.acceptedCandidates);
  EXPECT_EQ(statistics.exactGates.targetTileGateInvocations,
            statistics.acceptedCandidates * 16);
  EXPECT_EQ(statistics.selectedExecutableRematerializations, 1u);
  EXPECT_EQ(
      statistics.selectedExecutableRematerializationGates.preTargetAttempts,
      1u);
  EXPECT_EQ(statistics.selectedStableOrdinal, 0u);
  EXPECT_EQ(statistics.selectedOutputMappingCount, 1u);
  EXPECT_EQ(statistics.selectedUniqueActiveTileCount, 16u);
  EXPECT_EQ(statistics.selectedParallelComponentCount, 1u);
  EXPECT_GT(statistics.enabledDurationTerms, 0u);
  EXPECT_GT(statistics.selectedMakespanPicoseconds, 0u);
  EXPECT_NE(diagnosticsText.find("whole-card-search policy=search"),
            std::string::npos)
      << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("whole-card-selection admitted="),
            std::string::npos)
      << diagnosticsText;

  std::string repeatDiagnosticsText;
  llvm::raw_string_ostream repeatDiagnostics(repeatDiagnosticsText);
  wafer::compiler::detail::WholeCardExecutableSynthesisStatistics
      repeatStatistics;
  auto repeated = wafer::compiler::detail::synthesizeWholeCardExecutable(
      *parsed.module, programMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), repeatDiagnostics,
      &repeatStatistics);
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
  ASSERT_EQ(repeated->tiles.size(), executable->tiles.size());
  for (size_t index = 0; index < executable->tiles.size(); ++index)
    EXPECT_EQ(repeated->tiles[index].getSelectedTileIR(),
              executable->tiles[index].getSelectedTileIR());
}

TEST(WholeCardExecutableSynthesisTest,
     SearchActualizesIndependentComponentPlacementCandidate) {
  ParsedProgram parsed = parseBranchProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::WholeCardExecutableSynthesisStatistics statistics;
  auto executable = wafer::compiler::detail::synthesizeWholeCardExecutable(
      *parsed.module, branchMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, &statistics);
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

TEST(WholeCardExecutableSynthesisTest,
     SearchActualizesBoundedNodePlacementCandidate) {
  ParsedProgram parsed = parseDependentProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::WholeCardExecutableSynthesisStatistics statistics;
  auto executable = wafer::compiler::detail::synthesizeWholeCardExecutable(
      *parsed.module, dependentProgramMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, &statistics);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  EXPECT_FALSE(
      wafer::compiler::detail::containsCardProgramSourceOperationLineage(
          *executable));
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

TEST(WholeCardExecutableSynthesisTest,
     SearchActualizesThreeStagePhysicalTilePipeline) {
  ParsedProgram parsed = parseThreeStageDependentProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::WholeCardExecutableSynthesisStatistics statistics;
  auto executable = wafer::compiler::detail::synthesizeWholeCardExecutable(
      *parsed.module, dependentProgramMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, &statistics);
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

TEST(WholeCardExecutableSynthesisTest,
     SearchActualizesLayoutAndBufferingInTheCommonFrontier) {
  ParsedProgram parsed = parseLayoutPipelineProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::WholeCardExecutableSynthesisStatistics statistics;
  auto executable = wafer::compiler::detail::synthesizeWholeCardExecutable(
      *parsed.module, layoutPipelineProgramMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, &statistics);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  EXPECT_GT(statistics.layoutConversionCandidateProposals, 0u);
  EXPECT_GT(statistics.layoutConversionCandidateMaterializations, 0u);
  EXPECT_GT(statistics.layoutConversionCandidateAcceptances, 0u)
      << diagnosticsText;
  EXPECT_GT(statistics.layoutBufferedCandidateProposals, 0u);
  EXPECT_GT(statistics.layoutBufferedCandidateMaterializations, 0u);
  EXPECT_GT(statistics.bufferedCandidateAcceptances, 0u) << diagnosticsText;
  EXPECT_GT(statistics.rotatingSlotAllocationsMaterialized, 0u)
      << diagnosticsText;
  EXPECT_GT(statistics.applicableFusionLogicalEdges, 0u);
  EXPECT_GT(statistics.actualFusionCandidateAcceptances, 0u) << diagnosticsText;
  EXPECT_GT(statistics.selectedActualFusedLogicalEdges, 0u) << diagnosticsText;
  EXPECT_NE(diagnosticsText.find("layout_buffered_proposals="),
            std::string::npos);
}

TEST(WholeCardExecutableSynthesisTest,
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

TEST(WholeCardExecutableSynthesisTest,
     SearchProposesAndMaterializesMultipleReductionIteratorAxes) {
  ParsedProgram parsed = parseTwoReductionAxisProgram();
  ASSERT_TRUE(parsed.module);

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::WholeCardExecutableSynthesisStatistics statistics;
  auto executable = wafer::compiler::detail::synthesizeWholeCardExecutable(
      *parsed.module, twoReductionAxisProgramMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, &statistics);
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

TEST(WholeCardExecutableSynthesisTest,
     SearchUsesFiniteTemporalTraversalWhenOneWaveExceedsSPM) {
  ParsedProgram baselineProgram = parseLargeTemporalProgram();
  ASSERT_TRUE(baselineProgram.module);
  std::string baselineDiagnosticsText;
  llvm::raw_string_ostream baselineDiagnostics(baselineDiagnosticsText);
  wafer::compiler::detail::WholeCardExecutableSynthesisStatistics
      baselineStatistics;
  auto baseline = wafer::compiler::detail::synthesizeWholeCardExecutable(
      *baselineProgram.module, largeTemporalProgramMetadata(),
      executionConfig(), wafer::OptimizationConfig::none(), baselineDiagnostics,
      &baselineStatistics);
  baselineDiagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(baseline)) << baselineDiagnosticsText;
  // The conservative policy does not predict SPM feasibility from a byte
  // estimate. It follows actual fixed-capacity packing failures through the
  // finite temporal domain until one candidate is accepted.
  EXPECT_GT(baselineStatistics.materializedCandidates, 1u);
  EXPECT_GT(baselineStatistics.materializationRejections, 0u);
  EXPECT_EQ(baselineStatistics.materializedCandidates,
            baselineStatistics.materializationRejections +
                baselineStatistics.acceptedCandidates);
  EXPECT_EQ(baselineStatistics.allocationFeedbackTransitions,
            baselineStatistics.materializationRejections);
  // Exact allocator conflicts can carry several tied structured lineages.
  // The deterministic controller composes their temporal changes in one
  // mutable baseline state. It never creates candidate siblings, progressive
  // promotions, priority selections or lookahead states.
  EXPECT_EQ(baselineStatistics.allocationFeedbackCandidates, 0u);
  EXPECT_GT(baselineStatistics.spmFailureProbeAttempts, 0u);
  EXPECT_EQ(baselineStatistics.allocationFeedbackProgressivePromotions, 0u);
  EXPECT_EQ(baselineStatistics.allocationFeedbackLookaheadCandidates, 0u);
  EXPECT_EQ(baselineStatistics.allocationFeedbackPrioritySelections, 0u);
  EXPECT_EQ(baselineStatistics.spmFailureProbeTilesSkipped,
            baselineStatistics.spmFailureProbeEarlyRejections * 15u);
  EXPECT_NE(baselineDiagnosticsText.find("whole-card-exact-failure-probe"),
            std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_NE(
      baselineDiagnosticsText.find("whole-card-exact-spm-conflict-certificate"),
      std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_EQ(
      baselineDiagnosticsText.find("whole-card-allocation-feedback-joint"),
      std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_EQ(baselineStatistics.acceptedCandidates, 1u);
  EXPECT_EQ(baselineStatistics.selectedExecutableRematerializations, 0u);
  EXPECT_GT(baselineStatistics.selectedTemporalWaveLowerBound, 1u);
  EXPECT_LE(baselineStatistics.selectedPeakAlignedResidencyEstimate,
            3080192u - 65536u);
  for (const wafer::compiler::PhysicalTileExecutable &tile : baseline->tiles) {
    EXPECT_NE(tile.getSelectedTileIR().find("scf.for"), llvm::StringRef::npos)
        << tile.getSelectedTileIR().str();
    const size_t stores =
        countOccurrences(tile.getSelectedTileIR(), "wafer.tile.store");
    EXPECT_GT(stores, 1u);
    EXPECT_LE(stores, 81u);
    EXPECT_NE(tile.getSelectedTileIR().find("memref.dealloc"),
              llvm::StringRef::npos)
        << tile.getSelectedTileIR().str();
  }

  ParsedProgram searchProgram = parseLargeTemporalProgram();
  ASSERT_TRUE(searchProgram.module);
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::detail::WholeCardExecutableSynthesisStatistics statistics;
  auto executable = wafer::compiler::detail::synthesizeWholeCardExecutable(
      *searchProgram.module, largeTemporalProgramMetadata(), executionConfig(),
      wafer::OptimizationConfig::search(), diagnostics, &statistics);
  diagnostics.flush();
  ASSERT_TRUE(mlir::succeeded(executable)) << diagnosticsText;
  ASSERT_EQ(executable->tiles.size(), 16u);
  EXPECT_FALSE(
      wafer::compiler::detail::containsCardProgramSourceOperationLineage(
          *executable));
  EXPECT_EQ(statistics.plannedCandidates,
            statistics.acceptedCandidates - statistics.schedulePlanRejections);
  EXPECT_GE(statistics.candidateProposals, statistics.shortlistedCandidates);
  EXPECT_GT(statistics.materializedCandidates, 0u);
  EXPECT_GT(statistics.selectedTemporalWaveLowerBound, 1u);
  EXPECT_GT(statistics.selectedInstructionExecutionLowerBound, 1u);
  EXPECT_GT(statistics.selectedPeakOutputTileFootprintEstimate, 0u);
  EXPECT_NE(
      baselineDiagnosticsText.find("whole-card-baseline-temporal-refinement"),
      std::string::npos)
      << baselineDiagnosticsText;
  EXPECT_EQ(
      baselineDiagnosticsText.find("whole-card-allocation-feedback-priority"),
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
  for (const wafer::compiler::PhysicalTileExecutable &tile :
       executable->tiles) {
    EXPECT_NE(tile.getSelectedTileIR().find("scf.for"), llvm::StringRef::npos)
        << tile.getSelectedTileIR().str();
    // Up to four iterator dimensions contribute compact
    // prologue/steady/tail sites; runtime wave count is never host-expanded.
    const size_t stores =
        countOccurrences(tile.getSelectedTileIR(), "wafer.tile.store");
    EXPECT_GT(stores, 1u);
    EXPECT_LE(stores, 81u) << tile.getSelectedTileIR().str();
    EXPECT_NE(tile.getSelectedTileIR().find("memref.dealloc"),
              llvm::StringRef::npos)
        << tile.getSelectedTileIR().str();
  }

  ParsedProgram repeatedProgram = parseLargeTemporalProgram();
  ASSERT_TRUE(repeatedProgram.module);
  std::string repeatedDiagnosticsText;
  llvm::raw_string_ostream repeatedDiagnostics(repeatedDiagnosticsText);
  wafer::compiler::detail::WholeCardExecutableSynthesisStatistics
      repeatedStatistics;
  auto repeated = wafer::compiler::detail::synthesizeWholeCardExecutable(
      *repeatedProgram.module, largeTemporalProgramMetadata(),
      executionConfig(), wafer::OptimizationConfig::search(),
      repeatedDiagnostics, &repeatedStatistics);
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
  ASSERT_EQ(repeated->tiles.size(), executable->tiles.size());
  for (size_t index = 0; index < executable->tiles.size(); ++index)
    EXPECT_EQ(repeated->tiles[index].getSelectedTileIR(),
              executable->tiles[index].getSelectedTileIR());
}

} // namespace
