//===- SelectedBufferMaterializationTest.cpp ------------------------===//

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"

#include "gtest/gtest.h"

namespace {
using namespace wafer::compiler::testing;

TEST(SelectedBufferMaterializationTest,
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
  auto materialized = wafer::compiler::detail::materializeSelectedBuffering(
      std::move(module), /*requestedBufferCount=*/2, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  module = std::move(materialized->module);
  slotAllocations = materialized->slotAllocationCount;
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
  EXPECT_TRUE(
      mlir::failed(wafer::compiler::detail::materializeSelectedBuffering(
          std::move(mismatched), /*requestedBufferCount=*/3, &failureReason)));
}


} // namespace
