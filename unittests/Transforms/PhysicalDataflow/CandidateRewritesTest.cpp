//===- CandidateRewritesTest.cpp - Actual-clone producer tests ---------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"
#include "Wafer/Transforms/PhysicalDataflow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

class CandidateRewritesTest : public ::testing::Test {
protected:
  CandidateRewritesTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    wafer::WaferDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    mlir::tensor::registerTilingInterfaceExternalModels(registry);
    wafer::registerTargetImplementationExternalModels(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  }

  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
};

TEST_F(CandidateRewritesTest, MaterializesConsumerLocalPureTensorClone) {
  auto module = parse(R"mlir(
module {
  func.func @fanout(%input: tensor<4xi32>, %producer_out: tensor<4xi32>,
                    %left_out: tensor<4xi32>, %right_out: tensor<4xi32>)
      -> (tensor<4xi32>, tensor<4xi32>) {
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<4xi32>) outs(%producer_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32):
      linalg.yield %value : i32
    } -> tensor<4xi32>
    %left = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<4xi32>) outs(%left_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32):
      linalg.yield %value : i32
    } -> tensor<4xi32>
    %right = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<4xi32>) outs(%right_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32):
      linalg.yield %value : i32
    } -> tensor<4xi32>
    return %left, %right : tensor<4xi32>, tensor<4xi32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("fanout");
  ASSERT_TRUE(function);

  EXPECT_EQ(wafer::materializeConsumerLocalTensorRecomputation(function), 1u);
  unsigned genericCount = 0;
  function.walk([&](mlir::linalg::GenericOp) { ++genericCount; });
  EXPECT_EQ(genericCount, 4u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest, KeepsProducerWithExternallyVisibleUseShared) {
  auto module = parse(R"mlir(
module {
  func.func @visible(%input: tensor<4xi32>, %producer_out: tensor<4xi32>,
                     %left_out: tensor<4xi32>, %right_out: tensor<4xi32>)
      -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<4xi32>) outs(%producer_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32): linalg.yield %value : i32
    } -> tensor<4xi32>
    %left = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<4xi32>) outs(%left_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32): linalg.yield %value : i32
    } -> tensor<4xi32>
    %right = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<4xi32>) outs(%right_out : tensor<4xi32>) {
    ^bb0(%value: i32, %unused: i32): linalg.yield %value : i32
    } -> tensor<4xi32>
    return %producer, %left, %right : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("visible");
  EXPECT_EQ(wafer::materializeConsumerLocalTensorRecomputation(function), 0u);
}

TEST_F(CandidateRewritesTest, HoistsOnlySpeculatableLoopInvariantWork) {
  auto module = parse(R"mlir(
module {
  func.func @licm(%a: i32, %b: i32, %buffer: memref<1xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iv = %c0 to %c4 step %c1 {
      %sum = arith.addi %a, %b : i32
      memref.store %sum, %buffer[%c0] : memref<1xi32>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("licm");
  EXPECT_EQ(wafer::hoistStaticLoopInvariantOperations(function), 1u);
  auto loop = *function.getOps<mlir::scf::ForOp>().begin();
  EXPECT_TRUE(loop.getBody()->getOps<mlir::arith::AddIOp>().empty());
  EXPECT_FALSE(loop.getBody()->getOps<mlir::memref::StoreOp>().empty());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

TEST_F(CandidateRewritesTest, RewritesEachModularIntegerAlgebraVariant) {
  auto module = parse(R"mlir(
module {
  func.func @reassociate(%a: tensor<4xi32>, %b: tensor<4xi32>,
                         %c: tensor<4xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
    ^bb0(%av: i32, %bv: i32, %cv: i32, %unused: i32):
      %ab = arith.addi %av, %bv : i32
      %abc = arith.addi %ab, %cv : i32
      linalg.yield %abc : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
  func.func @tree(%a: tensor<4xi32>, %b: tensor<4xi32>,
                  %c: tensor<4xi32>, %d: tensor<4xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
      ins(%a, %b, %c, %d : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
    ^bb0(%av: i32, %bv: i32, %cv: i32, %dv: i32, %unused: i32):
      %ab = arith.addi %av, %bv : i32
      %abc = arith.addi %ab, %cv : i32
      %abcd = arith.addi %abc, %dv : i32
      linalg.yield %abcd : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
  func.func @distribute(%a: tensor<4xi32>, %b: tensor<4xi32>,
                        %c: tensor<4xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
    ^bb0(%av: i32, %bv: i32, %cv: i32, %unused: i32):
      %bc = arith.addi %bv, %cv : i32
      %r0 = arith.muli %av, %bc : i32
      linalg.yield %r0 : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
  func.func @factor(%a: tensor<4xi32>, %b: tensor<4xi32>,
                    %c: tensor<4xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
    ^bb0(%av: i32, %bv: i32, %cv: i32, %unused: i32):
      %ab = arith.muli %av, %bv : i32
      %ac = arith.muli %av, %cv : i32
      %r0 = arith.addi %ab, %ac : i32
      linalg.yield %r0 : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  EXPECT_EQ(wafer::reassociateIntegerElementwiseExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("reassociate")),
            1u);
  EXPECT_EQ(wafer::balanceIntegerElementwiseReductionTrees(
                module->lookupSymbol<mlir::func::FuncOp>("tree")),
            1u);
  EXPECT_EQ(wafer::distributeIntegerElementwiseExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("distribute")),
            1u);
  EXPECT_EQ(wafer::factorIntegerElementwiseExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("factor")),
            1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::WaferTargetPolicy policy = wafer::getDefaultWaferTargetPolicy();
  wafer::tensor_program_scheduling::SelectionConfig config(policy);
  config.logicalRank = 0;
  wafer::tensor_program_scheduling::CandidateSpec candidate;
  candidate.tileSizes = {4};
  for (llvm::StringRef functionName :
       {"reassociate", "tree", "distribute", "factor"}) {
    auto acceptance =
        wafer::tensor_program_scheduling::evaluateCandidateOnOriginalTask(
            module->lookupSymbol<mlir::func::FuncOp>(functionName), {4},
            candidate, config);
    EXPECT_TRUE(acceptance.failureReason.empty())
        << functionName.str() << ": " << acceptance.failureReason;
    EXPECT_EQ(acceptance.artifactSource,
              wafer::tensor_program_scheduling::CandidateArtifactSource::
                  CompleteTraversalAPI)
        << functionName.str();
  }
}

TEST_F(CandidateRewritesTest, RejectsFloatingAndPoisonChangingAlgebra) {
  auto module = parse(R"mlir(
module {
  func.func @floating(%a: tensor<4xf32>, %b: tensor<4xf32>,
                      %c: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
    ^bb0(%av: f32, %bv: f32, %cv: f32, %unused: f32):
      %ab = arith.addf %av, %bv : f32
      %abc = arith.addf %ab, %cv : f32
      linalg.yield %abc : f32
    } -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
  func.func @no_wrap(%a: tensor<4xi32>, %b: tensor<4xi32>,
                     %c: tensor<4xi32>, %out: tensor<4xi32>)
      -> tensor<4xi32> {
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
    ^bb0(%av: i32, %bv: i32, %cv: i32, %unused: i32):
      %ab = arith.addi %av, %bv overflow<nsw> : i32
      %abc = arith.addi %ab, %cv : i32
      linalg.yield %abc : i32
    } -> tensor<4xi32>
    return %r : tensor<4xi32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  EXPECT_EQ(wafer::reassociateIntegerElementwiseExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("floating")),
            0u);
  EXPECT_EQ(wafer::reassociateIntegerElementwiseExpressions(
                module->lookupSymbol<mlir::func::FuncOp>("no_wrap")),
            0u);
}

} // namespace
