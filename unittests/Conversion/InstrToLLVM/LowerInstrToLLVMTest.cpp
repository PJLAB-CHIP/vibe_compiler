#include "Wafer/Conversion/InstrToLLVM/LowerInstrToTargetLLVMInternal.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Conversion/InstrToLLVM/InstrToLLVM.h"

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
  wafer::registerWaferCoreDialects(registry);
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
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  EXPECT_TRUE(mlir::failed(manager.run(*source)));
  EXPECT_NE(diagnostics.find("does not accept schema-free semantic attribute "
                             "'indexing_maps'"),
            std::string::npos)
      << diagnostics;
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 1u);
  EXPECT_TRUE(elementwise->hasAttr("indexing_maps"));
}

TEST(LowerInstrToTargetLLVMTest,
     LowersStaticallyBoundedDerivedSubviewToDynamicByteAddress) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(
      %input: memref<5x8xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %row = %c0 to %c4 step %c1 {
      %stage_shifted = arith.addi %row, %c1 : index
      %view = memref.subview %input[%stage_shifted, 2] [1, 3] [1, 1]
          : memref<5x8xf16, #wafer.memory<ddr, tensor>>
         to memref<1x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
      %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<1x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %view to %spm
          {byte_count = 6 : i64, inner_bytes = 6 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<1x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
         to memref<1x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
    }
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  EXPECT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<mlir::memref::SubViewOp>(*source), 0u);
  EXPECT_EQ(countOps<mlir::scf::ForOp>(*source), 0u);
  EXPECT_EQ(countOps<mlir::LLVM::MulOp>(*source), 1u);
  EXPECT_GE(countOps<mlir::LLVM::AddOp>(*source), 3u);
}

TEST(LowerInstrToTargetLLVMTest,
     LowersStaticallyBoundedClampedSubviewToDynamicByteAddress) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(
      %input: memref<5x8xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    scf.for %row = %c0 to %c4 step %c1 {
      %lower_clamped = arith.maxsi %row, %c1 : index
      %clamped = arith.minsi %lower_clamped, %c3 : index
      %view = memref.subview %input[%clamped, 2] [1, 3] [1, 1]
          : memref<5x8xf16, #wafer.memory<ddr, tensor>>
         to memref<1x3xf16, strided<[8, 1], offset: ?>,
              #wafer.memory<ddr, tensor>>
      %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<1x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %view to %spm
          {byte_count = 6 : i64, inner_bytes = 6 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<1x3xf16, strided<[8, 1], offset: ?>,
              #wafer.memory<ddr, tensor>>
         to memref<1x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
    }
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  EXPECT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<mlir::memref::SubViewOp>(*source), 0u);
  EXPECT_EQ(countOps<mlir::scf::ForOp>(*source), 0u);
}

TEST(LowerInstrToTargetLLVMTest,
     RejectsUnsupportedDynamicSubviewOffsetWithoutMutatingSource) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>,
      %condition: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %selected = arith.select %condition, %i, %c0 : index
      %view = memref.subview %input[%selected] [1] [1]
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
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  EXPECT_TRUE(mlir::failed(manager.run(*source)));
  EXPECT_NE(diagnostics.find("dynamic tensor subview offset #0 must be a "
                             "supported statically bounded index expression"),
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
      {axes = ["card_partition"], shape = array<i64: 1>}
  func.func @main(%status: i64) {
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  request.cardId = 0;
  request.tileId = 15;
  request.transportStatusArgumentIndex = 0;
  request.transportPreparedBeforeEntry = true;
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

  int64_t tileCount = -1;
  source->walk([&](mlir::LLVM::ConstantOp constant) {
    if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      tileCount = value.getInt();
  });
  EXPECT_EQ(tileCount, 16);
}

TEST(LowerInstrToTargetLLVMTest,
     LowersI8ArithmeticWithoutFormalModelQualification) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<64xi8, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<64xi8, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<64xi8, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
        %lhs, %rhs into %dst
        : memref<64xi8, #wafer.memory<spm, tensor>>,
          memref<64xi8, #wafer.memory<spm, tensor>>
      into memref<64xi8, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));
  EXPECT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 0u);
  EXPECT_GT(countOps<mlir::LLVM::LLVMFuncOp>(*source), 0u);
}

TEST(LowerInstrToTargetLLVMTest,
     LowersF32ElementwiseWithoutFormalModelQualification) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<64xf32, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<64xf32, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<neg> %src into %dst
        : memref<64xf32, #wafer.memory<spm, tensor>>
      into memref<64xf32, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));
  EXPECT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 0u);
  EXPECT_GT(countOps<mlir::LLVM::LLVMFuncOp>(*source), 0u);
}

TEST(LowerInstrToTargetLLVMTest,
     NumericVerificationKeepsQualifiedComputeAndI8MovementIndependent) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(
      %input: memref<64xi8, #wafer.memory<ddr, tensor>>,
      %output: memref<64xi8, #wafer.memory<ddr, tensor>>) {
    %i8 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<64xi8, #wafer.memory<spm, tensor>>
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<64xf16, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<64xf16, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<64xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %input to %i8
        {byte_count = 64 : i64, inner_bytes = 64 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<64xi8, #wafer.memory<ddr, tensor>>
       to memref<64xi8, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
        %lhs, %rhs into %dst
        : memref<64xf16, #wafer.memory<spm, tensor>>,
          memref<64xf16, #wafer.memory<spm, tensor>>
      into memref<64xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %i8 to %output
        {byte_count = 64 : i64, inner_bytes = 64 : i64,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<64xi8, #wafer.memory<spm, tensor>>
       to memref<64xi8, #wafer.memory<ddr, tensor>>
    return
  }
}
  )mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));
  EXPECT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<wafer::InstrRDMAOp>(*source), 0u);
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 0u);
  EXPECT_EQ(countOps<wafer::InstrWDMAOp>(*source), 0u);
}

TEST(LowerInstrToTargetLLVMTest,
     CountsBlockedCTOverTheCompletePhysicalTraversal) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<3x65xf16, #wafer.memory<spm, cx>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<3x65xf16, #wafer.memory<spm, cx>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<abs> %src into %dst
        : memref<3x65xf16, #wafer.memory<spm, cx>>
      into memref<3x65xf16, #wafer.memory<spm, cx>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  wafer::InstrElementwiseOp elementwise;
  source->walk([&](wafer::InstrElementwiseOp op) { elementwise = op; });
  ASSERT_TRUE(elementwise);
  auto destType = mlir::cast<mlir::MemRefType>(elementwise.getDest().getType());
  mlir::FailureOr<int64_t> elements =
      wafer::target_llvm_detail::getPhysicalTraversalElementCount(
          elementwise, destType, "elementwise dest");
  ASSERT_TRUE(mlir::succeeded(elements));
  EXPECT_EQ(*elements, 256);
  EXPECT_NE(*elements, destType.getNumElements());
}

TEST(LowerInstrToTargetLLVMTest,
     PhysicalVerificationAcceptsCompatibleBlockedConvertTraversal) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x2x197xf16, #wafer.memory<spm, ncx>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67584>}
        : memref<2x2x197xbf16, #wafer.memory<spm, ncx>>
    wafer.instr.convert #wafer.instr_convert_kind<fp16_bf16> %src into %dst
        {rounding_mode = 0 : i64}
        : memref<2x2x197xf16, #wafer.memory<spm, ncx>>
       to memref<2x2x197xbf16, #wafer.memory<spm, ncx>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  EXPECT_TRUE(mlir::succeeded(
      wafer::target_llvm_detail::verifyTargetInstructionFormats(*source)));
}

TEST(LowerInstrToTargetLLVMTest,
     PhysicalVerificationRejectsDtypeSpecificBlockedTraversalMismatch) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x2x197xf16, #wafer.memory<spm, ncx>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67584>}
        : memref<2x2x197xf32, #wafer.memory<spm, ncx>>
    wafer.instr.convert #wafer.instr_convert_kind<fp16_fp32> %src into %dst
        : memref<2x2x197xf16, #wafer.memory<spm, ncx>>
       to memref<2x2x197xf32, #wafer.memory<spm, ncx>>
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
  EXPECT_TRUE(mlir::failed(
      wafer::target_llvm_detail::verifyTargetInstructionFormats(*source)));
  EXPECT_NE(diagnostics.find("unsupported_target_physical_traversal: convert"),
            std::string::npos)
      << diagnostics;
}

} // namespace
