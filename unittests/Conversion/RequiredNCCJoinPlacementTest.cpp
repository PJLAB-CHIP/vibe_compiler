//===- RequiredNCCJoinPlacementTest.cpp ------------------------------------===//

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>

namespace {

static constexpr llvm::StringLiteral kUnsupportedDynamicLoop = R"mlir(
module {
  func.func @dynamic_loop(%upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f16
    %buffer = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [2]
    wafer.instr.fill %buffer, %zero
        {worker = #wafer.ncc_worker<worker0>}
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    scf.for %index = %c0 to %upper step %c1 {
      wafer.instr.fill %buffer, %zero
          {worker = #wafer.ncc_worker<worker1>}
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    }
    return
  }
}
)mlir";

static std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

TEST(RequiredNCCJoinPlacementTest, FailedRebuildLeavesModuleUnchanged) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      kUnsupportedDynamicLoop, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  std::string before = printModule(*module);

  EXPECT_TRUE(mlir::failed(wafer::rebuildRequiredNCCJoins(*module)));

  EXPECT_EQ(printModule(*module), before);
}

TEST(RequiredNCCJoinPlacementTest, FailedRebuildPassLeavesModuleUnchanged) {
  mlir::DialectRegistry registry;
  wafer::registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      kUnsupportedDynamicLoop, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  std::string before = printModule(*module);

  mlir::PassManager manager(&context);
  manager.addNestedPass<mlir::func::FuncOp>(
      wafer::createRebuildRequiredNCCJoinsPass());
  EXPECT_TRUE(mlir::failed(manager.run(*module)));
  EXPECT_EQ(printModule(*module), before);
}

} // namespace
