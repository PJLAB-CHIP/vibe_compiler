//===- CanonicalSpatialAssignmentTest.cpp ------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"

#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

using wafer::TileId;
using namespace wafer::compiler::detail;

class CanonicalSpatialAssignmentTest : public ::testing::Test {
protected:
  CanonicalSpatialAssignmentTest() {
    wafer::registerWaferCoreDialects(registry);
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  static mlir::func::FuncOp getFunction(mlir::ModuleOp module) {
    return *module.getOps<mlir::func::FuncOp>().begin();
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

constexpr llvm::StringLiteral kFaninProgram = R"mlir(
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @diamond(%input: tensor<5x7xf16>) -> tensor<5x7xf16> {
    %empty0 = tensor.empty() : tensor<5x7xf16>
    %left = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%input : tensor<5x7xf16>)
        outs(%empty0 : tensor<5x7xf16>) {
      ^bb0(%element0: f16, %output0: f16):
        linalg.yield %element0 : f16
    } -> tensor<5x7xf16>
    %empty1 = tensor.empty() : tensor<5x7xf16>
    %right = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%left : tensor<5x7xf16>)
        outs(%empty1 : tensor<5x7xf16>) {
      ^bb0(%element1: f16, %output1: f16):
        linalg.yield %element1 : f16
    } -> tensor<5x7xf16>
    %empty2 = tensor.empty() : tensor<5x7xf16>
    %branch = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%left : tensor<5x7xf16>)
        outs(%empty2 : tensor<5x7xf16>) {
      ^bb0(%element2: f16, %output2: f16):
        linalg.yield %element2 : f16
    } -> tensor<5x7xf16>
    %empty3 = tensor.empty() : tensor<5x7xf16>
    %sum = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%right, %branch : tensor<5x7xf16>, tensor<5x7xf16>)
        outs(%empty3 : tensor<5x7xf16>) {
      ^bb0(%a: f16, %b: f16, %output: f16):
        %value = arith.addf %a, %b : f16
        linalg.yield %value : f16
    } -> tensor<5x7xf16>
    return %sum : tensor<5x7xf16>
  }
}
)mlir";

TEST_F(CanonicalSpatialAssignmentTest,
       DerivesStableDistinctSemanticKeysFromObservableSSAPaths) {
  std::string withUnrelatedOperation = kFaninProgram.str();
  size_t insertion = withUnrelatedOperation.find("    %empty0");
  ASSERT_NE(insertion, std::string::npos);
  withUnrelatedOperation.insert(insertion,
                                "    %unrelated = arith.constant 1 : i32\n");
  mlir::OwningOpRef<mlir::ModuleOp> firstModule = parse(kFaninProgram);
  mlir::OwningOpRef<mlir::ModuleOp> secondModule =
      parse(withUnrelatedOperation);
  ASSERT_TRUE(firstModule && secondModule);

  std::string failureReason;
  auto firstDag =
      StructuredDAGAnalysis::create(getFunction(*firstModule), &failureReason);
  ASSERT_TRUE(mlir::succeeded(firstDag)) << failureReason;
  auto secondDag =
      StructuredDAGAnalysis::create(getFunction(*secondModule), &failureReason);
  ASSERT_TRUE(mlir::succeeded(secondDag)) << failureReason;
  auto first = SemanticRootAnalysis::create(*firstDag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(first)) << failureReason;
  auto second = SemanticRootAnalysis::create(*secondDag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(second)) << failureReason;

  ASSERT_EQ(first->getRoots().size(), 4u);
  ASSERT_EQ(second->getRoots().size(), 4u);
  for (auto [lhs, rhs] :
       llvm::zip_equal(first->getRoots(), second->getRoots())) {
    EXPECT_EQ(lhs.key, rhs.key);
    EXPECT_NE(lhs.operation, rhs.operation);
  }
  EXPECT_LT(first->getRoots()[0].key, first->getRoots()[1].key);
  EXPECT_LT(first->getRoots()[1].key, first->getRoots()[2].key);
  EXPECT_LT(first->getRoots()[2].key, first->getRoots()[3].key);
}

TEST_F(CanonicalSpatialAssignmentTest,
       RejectsStructuredRootsWithoutAnObservableTypedPath) {
  auto module = parse(R"mlir(
#id = affine_map<(d0) -> (d0)>
module {
  func.func @unused(%input: tensor<4xf16>) -> tensor<4xf16> {
    %unused_empty = tensor.empty() : tensor<4xf16>
    %unused = linalg.generic {
        indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%input : tensor<4xf16>) outs(%unused_empty : tensor<4xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<4xf16>
    %result_empty = tensor.empty() : tensor<4xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%input : tensor<4xf16>) outs(%result_empty : tensor<4xf16>) {
      ^bb0(%value: f16, %output: f16):
        linalg.yield %value : f16
    } -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag =
      StructuredDAGAnalysis::create(getFunction(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  EXPECT_TRUE(mlir::failed(SemanticRootAnalysis::create(*dag, &failureReason)));
  EXPECT_NE(failureReason.find("no typed path"), std::string::npos)
      << failureReason;
}

TEST_F(CanonicalSpatialAssignmentTest,
       BuildsMaximumDeterministicMultiAxisAssignmentWithoutIRMutation) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(kFaninProgram);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  std::string before = print(module->getOperation());
  std::string failureReason;
  auto dag =
      StructuredDAGAnalysis::create(getFunction(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;

  auto coordinate = buildCanonicalSpatialAssignment(
      *dag, {TileId(9), TileId(1), TileId(7), TileId(3), TileId(5), TileId(0)},
      &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  ASSERT_EQ(coordinate->plan.nodes.size(), 4u);
  for (const NodeSpatialPlan &node : coordinate->plan.nodes) {
    ASSERT_EQ(node.axes.size(), 2u);
    EXPECT_EQ(node.axes[0].parameter, 3);
    EXPECT_EQ(node.axes[1].parameter, 2);
    EXPECT_TRUE(
        llvm::equal(node.embedding,
                    llvm::ArrayRef<TileId>{TileId(0), TileId(1), TileId(3),
                                           TileId(5), TileId(7), TileId(9)}));
    EXPECT_TRUE(node.reductionMerges.empty());
  }
  for (const NodeExecutionPartition &node : coordinate->assignment.nodes) {
    ASSERT_EQ(node.shards.size(), 6u);
    EXPECT_EQ(node.shards[0].iterationDomain[0], (IteratorInterval{0, 2}));
    EXPECT_EQ(node.shards[4].iterationDomain[0], (IteratorInterval{4, 1}));
    EXPECT_EQ(node.shards[0].iterationDomain[1], (IteratorInterval{0, 4}));
    EXPECT_EQ(node.shards[1].iterationDomain[1], (IteratorInterval{4, 3}));
  }

  auto reordered = buildCanonicalSpatialAssignment(
      *dag, {TileId(0), TileId(5), TileId(3), TileId(9), TileId(1), TileId(7)},
      &failureReason);
  ASSERT_TRUE(mlir::succeeded(reordered)) << failureReason;
  EXPECT_EQ(reordered->plan, coordinate->plan);
  EXPECT_EQ(print(module->getOperation()), before);
}

TEST_F(CanonicalSpatialAssignmentTest,
       LeavesPureReductionAsOneUnpartitionedLogicalCell) {
  auto module = parse(R"mlir(
#input = affine_map<(k) -> (k)>
#output = affine_map<(k) -> ()>
module {
  func.func @reduce(%input: tensor<5xf16>) -> tensor<f16> {
    %empty = tensor.empty() : tensor<f16>
    %result = linalg.generic {
        indexing_maps = [#input, #output], iterator_types = ["reduction"]}
        ins(%input : tensor<5xf16>) outs(%empty : tensor<f16>) {
      ^bb0(%value: f16, %accumulator: f16):
        %sum = arith.addf %value, %accumulator : f16
        linalg.yield %sum : f16
    } -> tensor<f16>
    return %result : tensor<f16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag =
      StructuredDAGAnalysis::create(getFunction(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate = buildCanonicalSpatialAssignment(
      *dag, {TileId(8), TileId(2), TileId(4), TileId(6)}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  ASSERT_EQ(coordinate->plan.nodes.size(), 1u);
  EXPECT_EQ(coordinate->plan.nodes[0].axes[0].parameter, 1);
  EXPECT_TRUE(llvm::equal(coordinate->plan.nodes[0].embedding,
                          llvm::ArrayRef<TileId>{TileId(2)}));
  ASSERT_EQ(coordinate->assignment.nodes[0].shards.size(), 1u);
  EXPECT_EQ(coordinate->assignment.nodes[0].shards[0].iterationDomain[0],
            (IteratorInterval{0, 5}));
}

TEST_F(CanonicalSpatialAssignmentTest,
       AppliesFlashAttentionAndFlashDecodingSpatialContracts) {
  auto module = parse(R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @attention_modes(
      %query: tensor<2x3x4xf16>, %key: tensor<2x5x4xf16>,
      %value: tensor<2x5x6xf16>, %scale: f32)
      -> (tensor<2x3x6xf16>, tensor<2x3x6xf16>) {
    %out0 = tensor.empty() : tensor<2x3x6xf16>
    %fa = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale : tensor<2x3x4xf16>,
            tensor<2x5x4xf16>, tensor<2x5x6xf16>, f32)
        outs(%out0 : tensor<2x3x6xf16>)
        algorithm(<flash_attention>)
        indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x3x6xf16>
    %out1 = tensor.empty() : tensor<2x3x6xf16>
    %fd = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale : tensor<2x3x4xf16>,
            tensor<2x5x4xf16>, tensor<2x5x6xf16>, f32)
        outs(%out1 : tensor<2x3x6xf16>)
        algorithm(<flash_decoding>)
        indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x3x6xf16>
    return %fa, %fd : tensor<2x3x6xf16>, tensor<2x3x6xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  std::string before = print(module->getOperation());
  std::string failureReason;
  auto dag =
      StructuredDAGAnalysis::create(getFunction(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate = buildCanonicalSpatialAssignment(
      *dag, {TileId(7), TileId(0), TileId(4), TileId(2)}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  ASSERT_EQ(coordinate->plan.nodes.size(), 2u);

  const NodeSpatialPlan &fa = coordinate->plan.nodes[0];
  const NodeSpatialPlan &fd = coordinate->plan.nodes[1];
  ASSERT_EQ(fa.root.anchorIndex, 0u);
  ASSERT_EQ(fd.root.anchorIndex, 1u);
  EXPECT_TRUE(llvm::equal(llvm::map_range(fa.axes,
                                          [](const IteratorPartition &axis) {
                                            return axis.parameter;
                                          }),
                          llvm::ArrayRef<int64_t>{2, 2, 1, 1, 1}));
  EXPECT_TRUE(fa.reductionMerges.empty());
  EXPECT_TRUE(llvm::equal(llvm::map_range(fd.axes,
                                          [](const IteratorPartition &axis) {
                                            return axis.parameter;
                                          }),
                          llvm::ArrayRef<int64_t>{2, 1, 1, 2, 1}));
  ASSERT_EQ(fd.reductionMerges.size(), 2u);
  EXPECT_TRUE(llvm::equal(fd.reductionMerges[0].group.parallelCoordinate,
                          llvm::ArrayRef<uint32_t>{0, 0, 0}));
  EXPECT_EQ(fd.reductionMerges[0].tile, TileId(0));
  EXPECT_TRUE(llvm::equal(fd.reductionMerges[1].group.parallelCoordinate,
                          llvm::ArrayRef<uint32_t>{1, 0, 0}));
  EXPECT_EQ(fd.reductionMerges[1].tile, TileId(4));

  ASSERT_EQ(coordinate->assignment.nodes[0].shards.size(), 4u);
  ASSERT_EQ(coordinate->assignment.nodes[1].shards.size(), 4u);
  ASSERT_EQ(coordinate->assignment.nodes[1].reductionGroups.size(), 2u);
  EXPECT_EQ(print(module->getOperation()), before);
}

} // namespace
