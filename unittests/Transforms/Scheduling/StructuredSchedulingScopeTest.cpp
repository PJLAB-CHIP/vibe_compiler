//===- StructuredSchedulingScopeTest.cpp - Task scope recovery --------===//

#include "Scheduling/StructuredSchedulingScope.h"
#include "Scheduling/ScheduleTensorProgramInternal.h"

#include "Wafer/InitAll.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

#include <memory>

namespace {

using wafer::CandidateTraversalRootCapability;
using wafer::structured_scheduler::ScopeDiscoveryPolicy;
using wafer::structured_scheduler::StructuredSchedulingScope;

class StructuredSchedulingScopeTest : public ::testing::Test {
protected:
  StructuredSchedulingScopeTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    wafer::registerAllDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  CandidateTraversalRootCapability
  getScopeTaskCapability(const StructuredSchedulingScope &scope) {
    mlir::OwningOpRef<mlir::ModuleOp> taskModule =
        wafer::structured_scheduler::cloneScopeToStandaloneModule(scope);
    EXPECT_TRUE(taskModule);
    if (!taskModule)
      return CandidateTraversalRootCapability::Unsupported;
    mlir::func::FuncOp task =
        wafer::structured_scheduler::findSingleTaskFunction(*taskModule);
    EXPECT_TRUE(task);
    if (!task)
      return CandidateTraversalRootCapability::Unsupported;
    return wafer::tensor_program_scheduling::getTaskTraversalRootCapability(
        task);
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredSchedulingScopeTest,
       KeepsShapePreservingAllReduceWithTiledProducer) {
  auto module = parse(R"mlir(
module {
  func.func @terminal_collective(
      %input: tensor<4xf32>, %local_out: tensor<4xf32>,
      %collective_out: tensor<4xf32>) -> tensor<4xf32> {
    %local = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<4xf32>) outs(%local_out : tensor<4xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    } -> tensor<4xf32>
    %collective = wafer.linalg_ext.collective.all_reduce
        ins(%local : tensor<4xf32>)
        outs(%collective_out : tensor<4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {channel_id = 1 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4xf32>
    return %collective : tensor<4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  ScopeDiscoveryPolicy policy{/*includeSharedInputPeers=*/false,
                              /*allowCrossShapeDataflow=*/true,
                              /*cutTerminalFullTraversalOnlyRoots=*/true};
  llvm::SmallVector<StructuredSchedulingScope, 1> scopes;
  ASSERT_TRUE(mlir::succeeded(
      wafer::structured_scheduler::discoverStructuredSchedulingScopes(
          *module, scopes, policy)));
  ASSERT_EQ(scopes.size(), 1u);
  ASSERT_EQ(scopes[0].orderedOps.size(), 2u);

  EXPECT_EQ(wafer::classifyCandidateTraversalRoot(scopes[0].orderedOps[0]),
            CandidateTraversalRootCapability::Tiled);
  EXPECT_EQ(wafer::classifyCandidateTraversalRoot(scopes[0].orderedOps[1]),
            CandidateTraversalRootCapability::Tiled);
  EXPECT_EQ(getScopeTaskCapability(scopes[0]),
            CandidateTraversalRootCapability::Tiled);
}

TEST_F(StructuredSchedulingScopeTest,
       SeparatesTiledCollectiveBeforeDownstreamConsumer) {
  auto module = parse(R"mlir(
module {
  func.func @collective_epilogue(
      %input: tensor<4xf32>, %collective_out: tensor<4xf32>,
      %local_out: tensor<4xf32>) -> tensor<4xf32> {
    %collective = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<4xf32>)
        outs(%collective_out : tensor<4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {channel_id = 2 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4xf32>
    %local = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%collective : tensor<4xf32>) outs(%local_out : tensor<4xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    } -> tensor<4xf32>
    return %local : tensor<4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  ScopeDiscoveryPolicy policy{/*includeSharedInputPeers=*/false,
                              /*allowCrossShapeDataflow=*/true,
                              /*cutTerminalFullTraversalOnlyRoots=*/true};
  llvm::SmallVector<StructuredSchedulingScope, 2> scopes;
  ASSERT_TRUE(mlir::succeeded(
      wafer::structured_scheduler::discoverStructuredSchedulingScopes(
          *module, scopes, policy)));
  ASSERT_EQ(scopes.size(), 2u);
  ASSERT_EQ(scopes[0].orderedOps.size(), 1u);
  ASSERT_EQ(scopes[1].orderedOps.size(), 1u);
  EXPECT_TRUE(mlir::isa<wafer::LinalgExtCollectiveAllReduceOp>(
      scopes[0].orderedOps.front()));
  EXPECT_TRUE(
      mlir::isa<mlir::linalg::GenericOp>(scopes[1].orderedOps.front()));
  EXPECT_EQ(getScopeTaskCapability(scopes[0]),
            CandidateTraversalRootCapability::Tiled);
  EXPECT_EQ(getScopeTaskCapability(scopes[1]),
            CandidateTraversalRootCapability::Tiled);
}

TEST_F(StructuredSchedulingScopeTest,
       SeparatesTiledCollectiveBeforeStaticReshapeConsumer) {
  auto module = parse(R"mlir(
module {
  func.func @collective_reshape_epilogue(
      %input: tensor<4xf32>, %collective_out: tensor<4xf32>,
      %local_out: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %collective = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<4xf32>)
        outs(%collective_out : tensor<4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {channel_id = 20 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4xf32>
    %reshaped = tensor.expand_shape %collective [[0, 1]]
        output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
    %local = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%reshaped : tensor<1x4xf32>)
        outs(%local_out : tensor<1x4xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    } -> tensor<1x4xf32>
    return %local : tensor<1x4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  ScopeDiscoveryPolicy policy{/*includeSharedInputPeers=*/false,
                              /*allowCrossShapeDataflow=*/true,
                              /*cutTerminalFullTraversalOnlyRoots=*/true};
  llvm::SmallVector<StructuredSchedulingScope, 2> scopes;
  ASSERT_TRUE(mlir::succeeded(
      wafer::structured_scheduler::discoverStructuredSchedulingScopes(
          *module, scopes, policy)));
  ASSERT_EQ(scopes.size(), 2u);
  ASSERT_EQ(scopes[0].orderedOps.size(), 1u);
  EXPECT_TRUE(mlir::isa<wafer::LinalgExtCollectiveAllReduceOp>(
      scopes[0].orderedOps.front()));
  ASSERT_EQ(scopes[1].orderedOps.size(), 2u);
  EXPECT_TRUE(
      mlir::isa<mlir::tensor::ExpandShapeOp>(scopes[1].orderedOps.front()));
  EXPECT_TRUE(
      mlir::isa<mlir::linalg::GenericOp>(scopes[1].orderedOps.back()));
  EXPECT_EQ(getScopeTaskCapability(scopes[0]),
            CandidateTraversalRootCapability::Tiled);
  EXPECT_EQ(getScopeTaskCapability(scopes[1]),
            CandidateTraversalRootCapability::Tiled);
}

TEST_F(StructuredSchedulingScopeTest,
       DoesNotExtendCollectiveResidencyThroughReshapeFanout) {
  auto module = parse(R"mlir(
module {
  func.func @collective_reshape_fanout(
      %input: tensor<4xf32>, %collective_out: tensor<4xf32>,
      %first_out: tensor<1x4xf32>, %second_out: tensor<1x4xf32>)
      -> (tensor<1x4xf32>, tensor<1x4xf32>) {
    %collective = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<4xf32>)
        outs(%collective_out : tensor<4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {channel_id = 21 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4xf32>
    %reshaped = tensor.expand_shape %collective [[0, 1]]
        output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
    %first = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%reshaped : tensor<1x4xf32>)
        outs(%first_out : tensor<1x4xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    } -> tensor<1x4xf32>
    %second = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%reshaped : tensor<1x4xf32>)
        outs(%second_out : tensor<1x4xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    } -> tensor<1x4xf32>
    return %first, %second : tensor<1x4xf32>, tensor<1x4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  ScopeDiscoveryPolicy policy{/*includeSharedInputPeers=*/false,
                              /*allowCrossShapeDataflow=*/true,
                              /*cutTerminalFullTraversalOnlyRoots=*/true};
  llvm::SmallVector<StructuredSchedulingScope, 3> scopes;
  ASSERT_TRUE(mlir::succeeded(
      wafer::structured_scheduler::discoverStructuredSchedulingScopes(
          *module, scopes, policy)));
  ASSERT_EQ(scopes.size(), 3u);
  ASSERT_EQ(scopes.front().orderedOps.size(), 1u);
  EXPECT_TRUE(mlir::isa<wafer::LinalgExtCollectiveAllReduceOp>(
      scopes.front().orderedOps.front()));
  EXPECT_EQ(getScopeTaskCapability(scopes.front()),
            CandidateTraversalRootCapability::Tiled);
}

TEST_F(StructuredSchedulingScopeTest,
       DoesNotExtendCollectiveResidencyThroughDynamicReshape) {
  auto module = parse(R"mlir(
module {
  func.func @dynamic_collective_reshape(
      %input: tensor<?x4xf32>, %collective_out: tensor<?x4xf32>,
      %local_out: tensor<?xf32>) -> tensor<?xf32> {
    %collective = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<?x4xf32>)
        outs(%collective_out : tensor<?x4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {channel_id = 22 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<?x4xf32>
    %reshaped = tensor.collapse_shape %collective [[0, 1]]
        : tensor<?x4xf32> into tensor<?xf32>
    %local = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%reshaped : tensor<?xf32>)
        outs(%local_out : tensor<?xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    } -> tensor<?xf32>
    return %local : tensor<?xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  ScopeDiscoveryPolicy policy{/*includeSharedInputPeers=*/false,
                              /*allowCrossShapeDataflow=*/true,
                              /*cutTerminalFullTraversalOnlyRoots=*/true};
  llvm::SmallVector<StructuredSchedulingScope, 2> scopes;
  ASSERT_TRUE(mlir::succeeded(
      wafer::structured_scheduler::discoverStructuredSchedulingScopes(
          *module, scopes, policy)));
  ASSERT_EQ(scopes.size(), 2u);
  ASSERT_EQ(scopes[0].orderedOps.size(), 1u);
  EXPECT_TRUE(mlir::isa<wafer::LinalgExtCollectiveAllReduceOp>(
      scopes[0].orderedOps.front()));
  EXPECT_EQ(getScopeTaskCapability(scopes[0]),
            CandidateTraversalRootCapability::Tiled);
  ASSERT_EQ(scopes[1].orderedOps.size(), 2u);
  EXPECT_TRUE(
      mlir::isa<mlir::tensor::CollapseShapeOp>(scopes[1].orderedOps.front()));
  EXPECT_TRUE(mlir::isa<mlir::linalg::GenericOp>(scopes[1].orderedOps.back()));
}

TEST_F(StructuredSchedulingScopeTest,
       KeepsSingletonShapePreservingAllReduceAsTiledTask) {
  auto module = parse(R"mlir(
module {
  func.func @singleton_collective(
      %input: tensor<4xf32>, %out: tensor<4xf32>) -> tensor<4xf32> {
    %collective = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {channel_id = 3 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4xf32>
    return %collective : tensor<4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  ScopeDiscoveryPolicy policy{/*includeSharedInputPeers=*/false,
                              /*allowCrossShapeDataflow=*/true,
                              /*cutTerminalFullTraversalOnlyRoots=*/true};
  llvm::SmallVector<StructuredSchedulingScope, 1> scopes;
  ASSERT_TRUE(mlir::succeeded(
      wafer::structured_scheduler::discoverStructuredSchedulingScopes(
          *module, scopes, policy)));
  ASSERT_EQ(scopes.size(), 1u);
  ASSERT_EQ(scopes.front().orderedOps.size(), 1u);
  EXPECT_EQ(getScopeTaskCapability(scopes.front()),
            CandidateTraversalRootCapability::Tiled);
  EXPECT_EQ(
      wafer::classifyCandidateTraversalRoot(scopes.front().orderedOps.front()),
      CandidateTraversalRootCapability::Tiled);
  auto returnOp = mlir::cast<mlir::func::ReturnOp>(
      scopes.front().orderedOps.front()->getBlock()->getTerminator());
  EXPECT_EQ(wafer::classifyCandidateTraversalRoot(returnOp.getOperation()),
            CandidateTraversalRootCapability::Unsupported);
}

TEST_F(StructuredSchedulingScopeTest,
       KeepsAxisChangingCollectiveFullTraversalOnly) {
  auto module = parse(R"mlir(
module {
  func.func @singleton_all_gather(
      %input: tensor<4xf32>, %out: tensor<8xf32>) -> tensor<8xf32> {
    %collective = wafer.linalg_ext.collective.all_gather
        ins(%input : tensor<4xf32>) outs(%out : tensor<8xf32>)
        {axis = 0 : i64, channel_id = 4 : i64,
         rank_group = array<i64: 0, 1>} -> tensor<8xf32>
    return %collective : tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  ScopeDiscoveryPolicy policy{/*includeSharedInputPeers=*/false,
                              /*allowCrossShapeDataflow=*/true,
                              /*cutTerminalFullTraversalOnlyRoots=*/true};
  llvm::SmallVector<StructuredSchedulingScope, 1> scopes;
  ASSERT_TRUE(mlir::succeeded(
      wafer::structured_scheduler::discoverStructuredSchedulingScopes(
          *module, scopes, policy)));
  ASSERT_EQ(scopes.size(), 1u);
  ASSERT_EQ(scopes.front().orderedOps.size(), 1u);
  EXPECT_EQ(getScopeTaskCapability(scopes.front()),
            CandidateTraversalRootCapability::FullTraversalOnly);
  EXPECT_EQ(
      wafer::classifyCandidateTraversalRoot(scopes.front().orderedOps.front()),
      CandidateTraversalRootCapability::FullTraversalOnly);
}

TEST_F(StructuredSchedulingScopeTest,
       KeepsShapePreservingCollectivePermuteFullTraversalOnly) {
  auto module = parse(R"mlir(
module {
  func.func @singleton_collective_permute(
      %input: tensor<4xf16>, %out: tensor<4xf16>) -> tensor<4xf16> {
    %collective = wafer.linalg_ext.collective.collective_permute
        ins(%input : tensor<4xf16>) outs(%out : tensor<4xf16>)
        {channel_id = 5 : i64,
         source_target_pairs = array<i64: 0, 1, 1, 0>}
        -> tensor<4xf16>
    return %collective : tensor<4xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);

  ScopeDiscoveryPolicy policy{/*includeSharedInputPeers=*/false,
                              /*allowCrossShapeDataflow=*/true,
                              /*cutTerminalFullTraversalOnlyRoots=*/true};
  llvm::SmallVector<StructuredSchedulingScope, 1> scopes;
  ASSERT_TRUE(mlir::succeeded(
      wafer::structured_scheduler::discoverStructuredSchedulingScopes(
          *module, scopes, policy)));
  ASSERT_EQ(scopes.size(), 1u);
  ASSERT_EQ(scopes.front().orderedOps.size(), 1u);
  EXPECT_EQ(getScopeTaskCapability(scopes.front()),
            CandidateTraversalRootCapability::FullTraversalOnly);
  EXPECT_EQ(
      wafer::classifyCandidateTraversalRoot(scopes.front().orderedOps.front()),
      CandidateTraversalRootCapability::FullTraversalOnly);
}

} // namespace
