//===- AttentionSpatialConstraintsTest.cpp ------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/AttentionSpatialConstraints.h"

#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

using namespace wafer::compiler::detail;

class AttentionSpatialConstraintsTest : public ::testing::Test {
protected:
  AttentionSpatialConstraintsTest() {
    wafer::registerWaferCoreDialects(registry);
    registry.insert<mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseAttentionOps() {
    return mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#multi_q = affine_map<(b, m, k1, k2a, k2b, n) -> (b, m, k1)>
#multi_k = affine_map<(b, m, k1, k2a, k2b, n) -> (b, k2a, k2b, k1)>
#multi_v = affine_map<(b, m, k1, k2a, k2b, n) -> (b, k2a, k2b, n)>
#multi_s = affine_map<(b, m, k1, k2a, k2b, n) -> ()>
#multi_o = affine_map<(b, m, k1, k2a, k2b, n) -> (b, m, n)>
module {
  func.func @attention_modes(
      %query: tensor<2x1024x128xf16>, %key: tensor<2x1024x128xf16>,
      %value: tensor<2x1024x64xf16>, %scale: f32)
      -> (tensor<2x1024x64xf16>, tensor<2x1024x64xf16>) {
    %out0 = tensor.empty() : tensor<2x1024x64xf16>
    %fa = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale :
            tensor<2x1024x128xf16>, tensor<2x1024x128xf16>,
            tensor<2x1024x64xf16>, f32)
        outs(%out0 : tensor<2x1024x64xf16>)
        algorithm(<flash_attention>)
        indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1024x64xf16>
    %out1 = tensor.empty() : tensor<2x1024x64xf16>
    %fd = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale :
            tensor<2x1024x128xf16>, tensor<2x1024x128xf16>,
            tensor<2x1024x64xf16>, f32)
        outs(%out1 : tensor<2x1024x64xf16>)
        algorithm(<flash_decoding>)
        indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1024x64xf16>
    return %fa, %fd : tensor<2x1024x64xf16>, tensor<2x1024x64xf16>
  }

  func.func @multi_k2(
      %query: tensor<2x1025x128xf16>, %key: tensor<2x33x31x128xf16>,
      %value: tensor<2x33x31x64xf16>, %scale: f32)
      -> tensor<2x1025x64xf16> {
    %out = tensor.empty() : tensor<2x1025x64xf16>
    %result = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale : tensor<2x1025x128xf16>,
            tensor<2x33x31x128xf16>, tensor<2x33x31x64xf16>, f32)
        outs(%out : tensor<2x1025x64xf16>)
        algorithm(<flash_decoding>)
        indexing_maps = [#multi_q, #multi_k, #multi_v, #multi_s, #multi_o]
        -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
  }
}
)mlir",
        mlir::ParserConfig(context.get()));
  }

  static llvm::SmallVector<IteratorPartition, 8> unitPartitions(unsigned rank) {
    llvm::SmallVector<IteratorPartition, 8> partitions;
    for (unsigned iterator = 0; iterator < rank; ++iterator)
      partitions.push_back(
          {iterator, IteratorPartitionScheme::BalancedParts, 1});
    return partitions;
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

TEST_F(AttentionSpatialConstraintsTest,
       DerivesModeAndReductionIteratorRolesWithoutMutatingIR) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseAttentionOps();
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  std::string before = print(module->getOperation());

  llvm::SmallVector<wafer::LinalgExtAttentionOp, 3> attentionOps;
  module->walk([&](wafer::LinalgExtAttentionOp attention) {
    attentionOps.push_back(attention);
  });
  ASSERT_EQ(attentionOps.size(), 3u);

  auto fa = deriveAttentionSpatialConstraints(attentionOps[0]);
  auto fd = deriveAttentionSpatialConstraints(attentionOps[1]);
  auto multi = deriveAttentionSpatialConstraints(attentionOps[2]);
  ASSERT_TRUE(mlir::succeeded(fa));
  ASSERT_TRUE(mlir::succeeded(fd));
  ASSERT_TRUE(mlir::succeeded(multi));
  EXPECT_TRUE(
      llvm::equal(fa->queryKeyReductionIterators, llvm::ArrayRef<unsigned>{2}));
  EXPECT_TRUE(
      llvm::equal(fa->keyValueReductionIterators, llvm::ArrayRef<unsigned>{3}));
  EXPECT_EQ(fa->keyValuePartition,
            AttentionKeyValuePartitionRequirement::SingleInterval);
  EXPECT_EQ(fd->keyValuePartition,
            AttentionKeyValuePartitionRequirement::MultipleIntervals);
  EXPECT_TRUE(llvm::equal(multi->queryKeyReductionIterators,
                          llvm::ArrayRef<unsigned>{2}));
  EXPECT_TRUE(llvm::equal(multi->keyValueReductionIterators,
                          llvm::ArrayRef<unsigned>({3, 4})));
  EXPECT_EQ(print(module->getOperation()), before);
}

TEST_F(AttentionSpatialConstraintsTest,
       AppliesOnePredicateToCanonicalAndFullModeChoices) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseAttentionOps();
  ASSERT_TRUE(module);
  llvm::SmallVector<wafer::LinalgExtAttentionOp, 3> attentionOps;
  module->walk([&](wafer::LinalgExtAttentionOp attention) {
    attentionOps.push_back(attention);
  });
  auto fa = deriveAttentionSpatialConstraints(attentionOps[0]);
  auto fd = deriveAttentionSpatialConstraints(attentionOps[1]);
  ASSERT_TRUE(mlir::succeeded(fa));
  ASSERT_TRUE(mlir::succeeded(fd));
  llvm::SmallVector<int64_t, 8> extents{2, 1024, 128, 1024, 64};
  llvm::SmallVector<IteratorPartition, 8> partitions = unitPartitions(5);

  EXPECT_EQ(checkAttentionSpatialConstraints(*fa, extents, partitions),
            AttentionSpatialConstraintViolation::None);
  EXPECT_EQ(checkAttentionSpatialConstraints(*fd, extents, partitions),
            AttentionSpatialConstraintViolation::
                FlashDecodingLeavesKeyValueUnpartitioned);

  partitions[3] = {3, IteratorPartitionScheme::UniformExtent, 1024};
  EXPECT_EQ(checkAttentionSpatialConstraints(*fa, extents, partitions),
            AttentionSpatialConstraintViolation::None);
  EXPECT_EQ(checkAttentionSpatialConstraints(*fd, extents, partitions),
            AttentionSpatialConstraintViolation::
                FlashDecodingLeavesKeyValueUnpartitioned);
  partitions[3] = {3, IteratorPartitionScheme::BalancedParts, 1};

  partitions[2] = {2, IteratorPartitionScheme::BalancedParts, 2};
  EXPECT_EQ(checkAttentionSpatialConstraints(*fa, extents, partitions),
            AttentionSpatialConstraintViolation::None)
      << "K1 remains governed by the general spatial domain";
  partitions[1] = {1, IteratorPartitionScheme::UniformExtent, 128};
  EXPECT_EQ(checkAttentionSpatialConstraints(*fa, extents, partitions),
            AttentionSpatialConstraintViolation::None)
      << "parallel partitioning does not change the attention mode";

  partitions[3] = {3, IteratorPartitionScheme::BalancedParts, 16};
  EXPECT_EQ(
      checkAttentionSpatialConstraints(*fa, extents, partitions),
      AttentionSpatialConstraintViolation::FlashAttentionPartitionsKeyValue);
  EXPECT_EQ(checkAttentionSpatialConstraints(*fd, extents, partitions),
            AttentionSpatialConstraintViolation::None);

  partitions[3] = {3, IteratorPartitionScheme::UniformExtent, 128};
  EXPECT_EQ(checkAttentionSpatialConstraints(*fd, extents, partitions),
            AttentionSpatialConstraintViolation::None);

  llvm::SmallVector<int64_t, 8> raggedExtents{2, 1025, 128, 1031, 64};
  partitions = unitPartitions(5);
  partitions[3] = {3, IteratorPartitionScheme::BalancedParts, 16};
  EXPECT_EQ(
      checkAttentionSpatialConstraints(*fa, raggedExtents, partitions),
      AttentionSpatialConstraintViolation::FlashAttentionPartitionsKeyValue);
  EXPECT_EQ(checkAttentionSpatialConstraints(*fd, raggedExtents, partitions),
            AttentionSpatialConstraintViolation::None);
}

TEST_F(AttentionSpatialConstraintsTest, SupportsMultipleKeyValueIterators) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseAttentionOps();
  ASSERT_TRUE(module);
  wafer::LinalgExtAttentionOp multi;
  module->walk([&](wafer::LinalgExtAttentionOp attention) {
    if (attention.getIterationDomainRank() == 6)
      multi = attention;
  });
  ASSERT_TRUE(multi);
  auto constraints = deriveAttentionSpatialConstraints(multi);
  ASSERT_TRUE(mlir::succeeded(constraints));
  llvm::SmallVector<int64_t, 8> extents{2, 1025, 128, 33, 31, 64};
  llvm::SmallVector<IteratorPartition, 8> partitions = unitPartitions(6);
  EXPECT_EQ(checkAttentionSpatialConstraints(*constraints, extents, partitions),
            AttentionSpatialConstraintViolation::
                FlashDecodingLeavesKeyValueUnpartitioned);
  partitions[4] = {4, IteratorPartitionScheme::UniformExtent, 8};
  EXPECT_EQ(checkAttentionSpatialConstraints(*constraints, extents, partitions),
            AttentionSpatialConstraintViolation::None);
}

TEST_F(AttentionSpatialConstraintsTest, FailsClosedOnMalformedIteratorDomains) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseAttentionOps();
  ASSERT_TRUE(module);
  wafer::LinalgExtAttentionOp attention;
  module->walk([&](wafer::LinalgExtAttentionOp candidate) {
    if (!attention)
      attention = candidate;
  });
  auto constraints = deriveAttentionSpatialConstraints(attention);
  ASSERT_TRUE(mlir::succeeded(constraints));
  llvm::SmallVector<int64_t, 8> extents{2, 1024, 128, 1024, 64};
  llvm::SmallVector<IteratorPartition, 8> partitions = unitPartitions(5);

  partitions.pop_back();
  EXPECT_EQ(checkAttentionSpatialConstraints(*constraints, extents, partitions),
            AttentionSpatialConstraintViolation::IteratorDomainMismatch);
  partitions = unitPartitions(5);
  partitions[3].iterator = 1;
  EXPECT_EQ(checkAttentionSpatialConstraints(*constraints, extents, partitions),
            AttentionSpatialConstraintViolation::IteratorDomainMismatch);
  partitions = unitPartitions(5);
  partitions[3].scheme = static_cast<IteratorPartitionScheme>(255);
  EXPECT_EQ(checkAttentionSpatialConstraints(*constraints, extents, partitions),
            AttentionSpatialConstraintViolation::IteratorDomainMismatch);

  AttentionSpatialConstraints overlapping = *constraints;
  overlapping.keyValueReductionIterators =
      overlapping.queryKeyReductionIterators;
  EXPECT_EQ(
      checkAttentionSpatialConstraints(overlapping, extents, unitPartitions(5)),
      AttentionSpatialConstraintViolation::IteratorDomainMismatch);
}

TEST(IteratorPartitionTest, ReportsExactIntervalCounts) {
  auto alignedBalanced = getIteratorPartitionIntervalCount(
      1024, {0, IteratorPartitionScheme::BalancedParts, 16});
  auto alignedUniform = getIteratorPartitionIntervalCount(
      1024, {0, IteratorPartitionScheme::UniformExtent, 128});
  auto raggedBalanced = getIteratorPartitionIntervalCount(
      1025, {0, IteratorPartitionScheme::BalancedParts, 16});
  auto raggedUniform = getIteratorPartitionIntervalCount(
      1025, {0, IteratorPartitionScheme::UniformExtent, 128});
  ASSERT_TRUE(mlir::succeeded(alignedBalanced));
  ASSERT_TRUE(mlir::succeeded(alignedUniform));
  ASSERT_TRUE(mlir::succeeded(raggedBalanced));
  ASSERT_TRUE(mlir::succeeded(raggedUniform));
  EXPECT_EQ(*alignedBalanced, 16);
  EXPECT_EQ(*alignedUniform, 8);
  EXPECT_EQ(*raggedBalanced, 16);
  EXPECT_EQ(*raggedUniform, 9);
  EXPECT_TRUE(mlir::failed(getIteratorPartitionIntervalCount(
      1024, {0, IteratorPartitionScheme::BalancedParts, 0})));
}

} // namespace
