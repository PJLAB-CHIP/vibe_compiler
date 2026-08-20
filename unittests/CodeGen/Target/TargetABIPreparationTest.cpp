//===- TargetABIPreparationTest.cpp - Target ABI preparation tests -------===//

#include "Wafer/Driver/Compilation.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "Wafer/CodeGen/Executable/CardExecutableInternal.h"
#include "Wafer/CodeGen/Target/TargetCodeGenInternal.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Program/ProgramData.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace {

wafer::frontend::ProgramPartitionSlice
singleCardPartitionSlice(llvm::ArrayRef<int64_t> shape) {
  wafer::frontend::ProgramPartitionSlice slice;
  slice.partitionId = 0;
  slice.replicaId = 0;
  slice.offsets.assign(shape.size(), 0);
  slice.sizes.assign(shape.begin(), shape.end());
  slice.strides.assign(shape.size(), 1);
  return slice;
}

wafer::frontend::ProgramBoundaryBinding
shapedBoundary(int64_t index, llvm::ArrayRef<int64_t> shape) {
  wafer::frontend::ProgramBoundaryBinding binding;
  binding.index = index;
  binding.programIndex = index;
  binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
  binding.globalShape.assign(shape.begin(), shape.end());
  binding.localShape.assign(shape.begin(), shape.end());
  binding.dtype = wafer::ProgramElementType::F32;
  binding.partitionSlices.push_back(singleCardPartitionSlice(shape));
  return binding;
}

TEST(TargetABIPreparationTest,
     WorkspaceAlignmentCombinesPolicyAndAllocationRequirements) {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto tensorProgram = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
                  %bias: tensor<4xf32>) -> tensor<4xf32> {
    %tmp = tensor.empty() : tensor<4xf32>
    %first = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
        outs(%tmp : tensor<4xf32>) {
      ^bb0(%a: f32, %b: f32, %old: f32):
        %value = arith.addf %a, %b : f32
        linalg.yield %value : f32
    } -> tensor<4xf32>

    %out = tensor.empty() : tensor<4xf32>
    %second = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%first, %bias : tensor<4xf32>, tensor<4xf32>)
        outs(%out : tensor<4xf32>) {
      ^bb0(%a: f32, %b: f32, %old: f32):
        %value = arith.addf %a, %b : f32
        linalg.yield %value : f32
    } -> tensor<4xf32>
    return %second : tensor<4xf32>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(tensorProgram);

  wafer::frontend::FrontendProgramVerificationResult program;
  program.numPartitions = 1;
  program.programUserInputCount = 3;
  program.distributedInputs = {shapedBoundary(0, {4}), shapedBoundary(1, {4}),
                               shapedBoundary(2, {4})};
  program.distributedOutputs = {shapedBoundary(0, {4})};
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(config));
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  auto cardExecutable = wafer::compiler::detail::buildCardExecutable(
      context, *tensorProgram, std::move(program), *config,
      wafer::OptimizationConfig::none(), diagnostics, std::nullopt,
      programData);
  if (!cardExecutable)
    FAIL() << diagnosticsText << llvm::toString(cardExecutable.takeError());
  tensorProgram = nullptr;
  ASSERT_EQ(cardExecutable->getTileExecutables().size(), 16u);
  for (size_t tileIndex = 0;
       tileIndex < cardExecutable->getTileExecutables().size(); ++tileIndex)
    EXPECT_EQ(cardExecutable->getTileExecutables()[tileIndex].getTileId(),
              wafer::TileId(static_cast<int64_t>(tileIndex)));

  const wafer::compiler::TileExecutable &tile =
      cardExecutable->getTileExecutables().front();
  mlir::func::FuncOp entry =
      tile.getModule().lookupSymbol<mlir::func::FuncOp>(tile.getEntrySymbol());
  ASSERT_TRUE(entry);
  mlir::Builder builder(entry.getContext());
  llvm::SmallVector<mlir::memref::AllocOp, 2> plannedDDRAllocations;
  entry.walk([&](mlir::memref::AllocOp allocation) {
    if (wafer::isWaferDDRMemRefType(allocation.getType()) &&
        allocation->hasAttr(wafer::kWaferDDROffsetAttrName)) {
      allocation->setAttr("alignment", builder.getI64IntegerAttr(384));
      plannedDDRAllocations.push_back(allocation);
    }
  });
  ASSERT_FALSE(plannedDDRAllocations.empty());
  // The structured scheduler fuses the two elementwise tasks, so no
  // intermediate workspace is needed in production.  This ABI-specific test
  // injects one accepted, offset-planned allocation to exercise workspace
  // alignment independently of that scheduling choice.
  mlir::OpBuilder opBuilder(entry.getContext());
  opBuilder.setInsertionPoint(plannedDDRAllocations.front());
  auto workspace = mlir::cast<mlir::memref::AllocOp>(
      opBuilder.clone(*plannedDDRAllocations.front()));
  plannedDDRAllocations.push_back(workspace);

  mlir::FailureOr<wafer::compiler::detail::PreparedTile> prepared =
      wafer::compiler::detail::prepareTargetABI(
          tile, *config, /*transportPreparedBeforeEntry=*/false);
  ASSERT_TRUE(mlir::succeeded(prepared));
  unsigned workspaceSlots = 0;
  for (const wafer::compiler::TileEntryArgument &slot : prepared->slots) {
    if (slot.kind != wafer::compiler::TileEntryArgumentKind::Workspace)
      continue;
    ++workspaceSlots;
    EXPECT_EQ(slot.alignment, 768);
  }
  EXPECT_EQ(workspaceSlots, 1u);

  for (mlir::memref::AllocOp allocation : plannedDDRAllocations)
    allocation->setAttr("alignment", builder.getI64IntegerAttr(
                                         std::numeric_limits<int64_t>::max()));
  std::string overflowDiagnostics;
  mlir::ScopedDiagnosticHandler handler(
      entry.getContext(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(overflowDiagnostics);
        diagnostic.print(stream);
        return mlir::success();
      });
  mlir::FailureOr<wafer::compiler::detail::PreparedTile> overflow =
      wafer::compiler::detail::prepareTargetABI(
          tile, *config, /*transportPreparedBeforeEntry=*/false);
  EXPECT_TRUE(mlir::failed(overflow));
  EXPECT_NE(overflowDiagnostics.find(
                "combined default DDR arena alignment is invalid or exceeds "
                "int64"),
            std::string::npos)
      << overflowDiagnostics;
}

} // namespace
