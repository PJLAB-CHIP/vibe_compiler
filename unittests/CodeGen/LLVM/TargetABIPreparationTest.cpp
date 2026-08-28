//===- TargetABIPreparationTest.cpp - Target ABI preparation tests -------===//

#include "Wafer/Driver/Compilation.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "Wafer/CodeGen/DeviceExecutableInternal.h"
#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"
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

TEST(TargetABIPreparationTest,
     WorkspaceAlignmentCombinesPolicyAndAllocationRequirements) {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %workspace = memref.alloc() {
      alignment = 384 : i64,
      wafer.ddr.offset = #wafer.ddr_offset<0>
    } : memref<1024xf32, #wafer.memory<ddr, tensor>>
    return
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(1);
  ASSERT_TRUE(static_cast<bool>(config));
  wafer::compiler::TileExecutable tile =
      wafer::compiler::DeviceExecutableBuilder::makeTileExecutable(
          wafer::CardId(0), wafer::TileId(0), wafer::LaunchSlotId(0),
          std::move(module), "main", {},
          wafer::compiler::TransportContract::None);
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
