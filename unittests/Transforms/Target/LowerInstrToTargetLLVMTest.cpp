#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Transforms/TargetConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <string>

namespace {

void registerTargetConversionDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  wafer::registerAllDialects(registry);
}

template <typename OpT> unsigned countOps(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](OpT) { ++count; });
  return count;
}

TEST(LowerInstrToTargetLLVMTest,
     RejectsResidualElementwiseMapsWithoutMutatingSource) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
        %lhs, %rhs into %dst
        : memref<2x3xf16, #wafer.memory<spm, tensor>>,
          memref<2x3xf16, #wafer.memory<spm, tensor>>
      into memref<2x3xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  wafer::InstrElementwiseOp elementwise;
  source->walk([&](wafer::InstrElementwiseOp op) { elementwise = op; });
  ASSERT_TRUE(elementwise);
  mlir::AffineMapAttr identity = mlir::AffineMapAttr::get(
      mlir::AffineMap::getMultiDimIdentityMap(2, &context));
  elementwise->setAttr(
      "indexing_maps",
      mlir::ArrayAttr::get(&context, {identity, identity, identity}));

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        return mlir::success();
      });
  mlir::PassManager manager(&context);
  manager.enableVerifier(false);
  wafer::TargetConversionRequest request{
      wafer::TargetProfileId::waferTx81SingleCardKernelV1()};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  EXPECT_TRUE(mlir::failed(manager.run(*source)));
  EXPECT_NE(
      diagnostics.find("unsupported_target_instr: terminal elementwise retains "
                       "indexing_maps after instruction legalization"),
      std::string::npos)
      << diagnostics;
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 1u);
  EXPECT_TRUE(elementwise->hasAttr("indexing_maps"));
}

TEST(LowerInstrToTargetLLVMTest,
     LowersStaticallyBoundedLoopIVSubviewToDynamicByteAddress) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(
      %input: memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %row = %c0 to %c4 step %c1 {
      %view = memref.subview %input[%row, 2] [1, 3] [1, 1]
          : memref<4x8xf16, #wafer.memory<ddr, tensor>>
         to memref<1x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
      %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<1x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %view to %spm
          {byte_count = 6 : i64, inner_bytes = 6 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<1x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
         to memref<1x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
    }
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{
      wafer::TargetProfileId::waferTx81SingleCardKernelV1()};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  EXPECT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<mlir::memref::SubViewOp>(*source), 0u);
  EXPECT_EQ(countOps<mlir::scf::ForOp>(*source), 0u);
  EXPECT_EQ(countOps<mlir::LLVM::MulOp>(*source), 1u);
  EXPECT_GE(countOps<mlir::LLVM::AddOp>(*source), 2u);
}

TEST(LowerInstrToTargetLLVMTest,
     RejectsDerivedDynamicSubviewOffsetWithoutMutatingSource) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %shifted = arith.addi %i, %c0 : index
      %view = memref.subview %input[%shifted] [1] [1]
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<1xf16, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        return mlir::success();
      });
  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{
      wafer::TargetProfileId::waferTx81SingleCardKernelV1()};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  EXPECT_TRUE(mlir::failed(manager.run(*source)));
  EXPECT_NE(diagnostics.find("dynamic DDR tensor subview offset #0 must be the "
                             "direct induction variable of scf.for"),
            std::string::npos)
      << diagnostics;
  EXPECT_EQ(countOps<mlir::memref::SubViewOp>(*source), 1u);
  EXPECT_EQ(countOps<mlir::scf::ForOp>(*source), 1u);
  EXPECT_EQ(countOps<mlir::LLVM::LLVMFuncOp>(*source), 0u);
}

TEST(LowerInstrToTargetLLVMTest,
     InjectsPreparedDirectDTELifecycleWithoutLocalDTEOps) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%status: i64) {
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{
      wafer::TargetProfileId::waferTx81SingleCardKernelV1()};
  request.logicalRank = 15;
  request.transportStatusArgumentIndex = 0;
  request.launchABI =
      wafer::TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1();
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  ASSERT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<wafer::InstrDTESendOp>(*source), 0u);
  EXPECT_EQ(countOps<wafer::InstrDTERecvOp>(*source), 0u);
  EXPECT_EQ(countOps<wafer::InstrDTEWaitOp>(*source), 0u);
  EXPECT_EQ(countOps<mlir::LLVM::CallOp>(*source), 2u);
  EXPECT_FALSE(source->lookupSymbol<mlir::LLVM::LLVMFuncOp>(
      "wafer_tx81_direct_dte_begin"));
  EXPECT_TRUE(source->lookupSymbol<mlir::LLVM::LLVMFuncOp>(
      "wafer_tx81_direct_dte_begin_after_prepare"));
  EXPECT_TRUE(source->lookupSymbol<mlir::LLVM::LLVMFuncOp>(
      "wafer_tx81_direct_dte_finish"));

  int64_t rankCount = -1;
  source->walk([&](mlir::LLVM::ConstantOp constant) {
    if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      rankCount = value.getInt();
  });
  EXPECT_EQ(rankCount, 16);
}

} // namespace
