#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"
#include "Wafer/Transforms/TargetConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
                  mlir::memref::MemRefDialect>();
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

} // namespace
