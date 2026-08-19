//===- StructuredOperationTileFootprintTest.cpp ----------------------===//

#include "Wafer/Analysis/Structured/StructuredOperationTileFootprint.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

TEST(StructuredOperationTileFootprintTest,
     BaselineAccountsForTensorAndComputeLayoutCopies) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, wafer::WaferDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<16x4096xf16>,
                  %rhs: tensor<4096x4096xf16>,
                  %init: tensor<16x4096xf16>) -> tensor<16x4096xf16> {
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<16x4096xf16>, tensor<4096x4096xf16>)
        outs(%init : tensor<16x4096xf16>) -> tensor<16x4096xf16>
    return %result : tensor<16x4096xf16>
  }
}
)mlir",
                                                         &context);
  ASSERT_TRUE(module);
  mlir::linalg::MatmulOp matmul;
  module->walk([&](mlir::linalg::MatmulOp operation) { matmul = operation; });
  ASSERT_TRUE(matmul);

  const wafer::TargetMemoryPolicy memory =
      wafer::getDefaultWaferTargetPolicy().memory;
  const uint64_t capacity =
      static_cast<uint64_t>(memory.spmLimit - memory.spmBase);
  const llvm::SmallVector<int64_t, 3> localWave{16, 256, 4096};
  std::optional<uint64_t> logical =
      wafer::compiler::detail::estimateStructuredOperationTileResidencyBytes(
          matmul, localWave, memory);
  std::optional<uint64_t> physicalUpperBound =
      wafer::compiler::detail::
          getStructuredOperationLoweringSPMUpperBoundBytes(
              matmul, localWave, localWave, memory);
  ASSERT_TRUE(logical);
  ASSERT_TRUE(physicalUpperBound);
  EXPECT_LE(*logical, capacity);
  EXPECT_GT(*physicalUpperBound, capacity);

  auto legalized = wafer::compiler::detail::
      deriveStructuredOperationTemporalTileShape(matmul, localWave, memory);
  ASSERT_TRUE(mlir::succeeded(legalized));
  EXPECT_NE(*legalized, localWave);
  std::optional<uint64_t> legalizedUpperBound =
      wafer::compiler::detail::
          getStructuredOperationLoweringSPMUpperBoundBytes(
              matmul, localWave, *legalized, memory);
  ASSERT_TRUE(legalizedUpperBound);
  EXPECT_LE(*legalizedUpperBound, capacity);
}

} // namespace
