//===- ProgramIngestionTest.cpp - StableHLO source boundary tests --------===//

#include "Wafer/Frontend/StableHLO/ProgramIngestion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>

namespace {

TEST(ProgramIngestionTest, AcceptsOnePublicEntryWithPrivatePureHelper) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func private @helper(%arg: tensor<2xf32>) -> tensor<2xf32> {
        return %arg : tensor<2xf32>
      }
      func.func @forward(%arg: tensor<2xf32>) -> tensor<2xf32> {
        %result = func.call @helper(%arg) : (tensor<2xf32>) -> tensor<2xf32>
        return %result : tensor<2xf32>
      }
    }
  )mlir",
                                                          &context);
  ASSERT_TRUE(module);
  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  EXPECT_TRUE(mlir::succeeded(
      wafer::frontend::verifyStableHLOSourceModule(*module, stream)))
      << diagnostics;
}

TEST(ProgramIngestionTest, RejectsRecursivePrivateHelpers) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func private @first() {
        func.call @second() : () -> ()
        return
      }
      func.func private @second() {
        func.call @first() : () -> ()
        return
      }
      func.func @forward() {
        func.call @first() : () -> ()
        return
      }
    }
  )mlir",
                                                          &context);
  ASSERT_TRUE(module);
  std::string diagnostics;
  llvm::raw_string_ostream stream(diagnostics);
  EXPECT_TRUE(mlir::failed(
      wafer::frontend::verifyStableHLOSourceModule(*module, stream)));
}

} // namespace
