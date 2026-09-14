//===- TensorResultIndexingTest.cpp --------------------------------------===//

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"

#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect>();
  wafer::registerWaferCoreDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

TEST(TensorResultIndexingTest,
     StaticSupportOperationsShareOneExactTypedRelationBuilder) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @support(%input: tensor<2x1031x128xf16>)
      -> tensor<2x1026x128xf16> {
    %slice = tensor.extract_slice %input[0, 6, 0] [2, 1025, 128]
        [1, 1, 1] : tensor<2x1031x128xf16> to tensor<2x1025x128xf16>
    %collapsed = tensor.collapse_shape %slice [[0, 1], [2]]
        : tensor<2x1025x128xf16> into tensor<2050x128xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1], [2]]
        output_shape [2, 1025, 128]
        : tensor<2050x128xf16> into tensor<2x1025x128xf16>
    %zero = arith.constant 0.0 : f16
    %padded = tensor.pad %expanded low[0, 1, 0] high[0, 0, 0] {
      ^bb0(%b: index, %m: index, %n: index):
        tensor.yield %zero : f16
    } : tensor<2x1025x128xf16> to tensor<2x1026x128xf16>
    return %padded : tensor<2x1026x128xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);

  llvm::SmallVector<mlir::OpResult, 4> results;
  module->walk([&](mlir::tensor::ExtractSliceOp operation) {
    results.push_back(mlir::cast<mlir::OpResult>(operation.getResult()));
  });
  module->walk([&](mlir::tensor::CollapseShapeOp operation) {
    results.push_back(mlir::cast<mlir::OpResult>(operation.getResult()));
  });
  module->walk([&](mlir::tensor::ExpandShapeOp operation) {
    results.push_back(mlir::cast<mlir::OpResult>(operation.getResult()));
  });
  module->walk([&](mlir::tensor::PadOp operation) {
    results.push_back(mlir::cast<mlir::OpResult>(operation.getResult()));
  });
  ASSERT_EQ(results.size(), 4u);

  for (mlir::OpResult result : results) {
    wafer::analysis::TensorResultIndexingResult indexing =
        wafer::analysis::deriveTensorResultIndexing(result);
    ASSERT_TRUE(indexing.isExact()) << indexing.detail;
    ASSERT_EQ(indexing.indexing->operands.size(), 1u);
    EXPECT_EQ(indexing.indexing->operands.front().role,
              wafer::TensorIndexingOperandRole::Source);
    EXPECT_TRUE(indexing.indexing->operands.front()
                    .resultToOperand.isFunctional()
                    .isProvenTrue());
  }

  wafer::analysis::IndexRelationLimits limits;
  limits.maxVariables = 1;
  wafer::analysis::TensorResultIndexingResult bounded =
      wafer::analysis::deriveTensorResultIndexing(results[1], limits);
  EXPECT_EQ(bounded.status,
            wafer::analysis::TensorResultIndexingStatus::ResourceExhausted);
  EXPECT_FALSE(bounded.indexing.has_value());
}

TEST(TensorResultIndexingTest,
     PackOuterPermutationPreservesPartialInnerDemand) {
  for (int64_t extent : {1024, 1025, 1031})
    for (int64_t columns : {128, 129}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(columns);
      auto context = createContext();
      const int64_t outer = (columns + 15) / 16;
      std::string input = "tensor<2x" + std::to_string(extent) + "x" +
                          std::to_string(columns) + "xf16>";
      std::string output = "tensor<" + std::to_string(extent) + "x2x" +
                           std::to_string(outer) + "x16xf16>";
      std::string text;
      llvm::raw_string_ostream b(text);
      b << "module { func.func @pack(%input: " << input << ") -> " << output
        << " { "
           "%zero = arith.constant 0.0 : f16\n%e = tensor.empty() : "
        << output
        << "\n"
           "%p = tensor.pack %input padding_value(%zero : f16) outer_dims_perm "
           "= [1,0,2] "
           "inner_dims_pos = [2] inner_tiles = [16] into %e : "
        << input << " -> " << output << " return %p : " << output << " } }";
      auto module = mlir::parseSourceString<mlir::ModuleOp>(
          b.str(), mlir::ParserConfig(context.get()));
      ASSERT_TRUE(module);
      mlir::tensor::PackOp pack;
      module->walk([&](mlir::tensor::PackOp op) { pack = op; });
      auto uses = llvm::range_size(pack.getSource().getUses());
      auto relation = wafer::analysis::deriveIterationOperandRelation(
          pack->getOpOperand(0), {extent, 2, outer});
      ASSERT_TRUE(relation.isExact()) << relation.reason;
      EXPECT_TRUE(relation.get()->isInjective().isProvenTrue());
      EXPECT_TRUE(relation.get()->contains({extent - 1, 1, outer - 1},
                                           {1, extent - 1, columns - 1}));
      EXPECT_FALSE(relation.get()->contains({extent - 1, 1, outer - 1},
                                            {1, extent - 1, columns}));
      auto main =
          relation.get()->getExactStaticRectangularImage({0, 0, 0}, {64, 2, 8});
      ASSERT_TRUE(main.isExact()) << main.reason;
      EXPECT_EQ(main.domain->sizes,
                (llvm::SmallVector<int64_t, 4>{2, 64, 128}));
      auto tail = relation.get()->getExactStaticRectangularImage(
          {extent - 1, 0, outer - 1}, {1, 2, 1});
      ASSERT_TRUE(tail.isExact()) << tail.reason;
      EXPECT_EQ(tail.domain->offsets, (llvm::SmallVector<int64_t, 4>{
                                          0, extent - 1, (outer - 1) * 16}));
      EXPECT_EQ(tail.domain->sizes,
                (llvm::SmallVector<int64_t, 4>{2, 1, columns == 128 ? 16 : 1}));
      wafer::analysis::IndexRelationLimits limits;
      limits.maxVariables = 1;
      EXPECT_EQ(wafer::analysis::deriveIterationOperandRelation(
                    pack->getOpOperand(0), {extent, 2, outer}, limits)
                    .status,
                wafer::analysis::IndexRelationStatus::ResourceExhausted);
      EXPECT_EQ(llvm::range_size(pack.getSource().getUses()), uses);
    }
}

TEST(TensorResultIndexingTest,
     InsertDemandPartitionsSourceAndUntouchedDestination) {
  using namespace wafer::analysis;
  for (int64_t extent : {1024, 1025, 1031}) {
    auto context = createContext();
    std::string input = "tensor<1x" + std::to_string(extent - 2) + "x64xf16>";
    std::string output = "tensor<2x" + std::to_string(extent) + "x128xf16>";
    std::string text;
    llvm::raw_string_ostream b(text);
    b << "module { func.func @update(%src: " << input << ", %dst: " << output
      << ") -> " << output
      << " { %r = tensor.insert_slice %src into %dst[1, 1, 32] [1, "
      << extent - 2 << ", 64] [1, 1, 1] : " << input << " into " << output
      << " return %r : " << output << " } }";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module);
    mlir::tensor::InsertSliceOp insert;
    module->walk([&](mlir::tensor::InsertSliceOp op) { insert = op; });
    auto transfer = deriveTensorResultIndexing(
        mlir::cast<mlir::OpResult>(insert.getResult()));
    ASSERT_TRUE(transfer.isExact()) << transfer.detail;
    ASSERT_EQ(transfer.indexing->operands.size(), 2u);
    for (const StaticRectangularIndexSet &demand :
         llvm::SmallVector<StaticRectangularIndexSet, 4>{
             {{0, 0, 0}, {2, extent, 128}},
             {{0, 0, 31}, {2, extent, 66}},
             {{1, 1, 32}, {1, extent - 2, 64}},
             {{0, 0, 0}, {1, extent, 128}}}) {
      auto src = getTensorOperandDemand(
          *transfer.indexing, transfer.indexing->operands[0], {demand});
      auto dst = getTensorOperandDemand(
          *transfer.indexing, transfer.indexing->operands[1], {demand});
      ASSERT_TRUE(src.isExact()) << src.reason;
      ASSERT_TRUE(dst.isExact()) << dst.reason;
      auto count = [](llvm::ArrayRef<StaticRectangularIndexSet> boxes,
                      int64_t b, int64_t m, int64_t n) {
        const int64_t points[] = {b, m, n};
        unsigned matches = 0;
        for (const auto &box : boxes) {
          bool inside = true;
          for (auto [point, offset, size] : llvm::zip_equal(
                   llvm::ArrayRef<int64_t>(points), box.offsets, box.sizes))
            inside &= offset <= point && point < offset + size;
          matches += inside;
        }
        return matches;
      };
      // Enumerating logical membership is an independent oracle for exact
      // coverage and disjointness, including multidimensional edge slabs.
      for (int64_t batch = 0; batch < 2; ++batch)
        for (int64_t row = 0; row < extent; ++row)
          for (int64_t col = 0; col < 128; ++col) {
            const bool read = count({demand}, batch, row, col);
            const bool updated = batch == 1 && row >= 1 && row < extent - 1 &&
                                 col >= 32 && col < 96;
            ASSERT_EQ(count(dst.domains, batch, row, col), read && !updated);
            ASSERT_EQ(count(src.domains, batch - 1, row - 1, col - 32),
                      read && updated);
          }
    }
    IndexRelationLimits limits;
    limits.maxRectangularPieces = 1;
    auto bounded = getTensorOperandDemand(
        *transfer.indexing, transfer.indexing->operands[1],
        {{{0, 0, 0}, {2, extent, 128}}}, limits);
    EXPECT_EQ(bounded.status, IndexRelationStatus::ResourceExhausted);
    EXPECT_TRUE(bounded.domains.empty());
    EXPECT_EQ(getTensorOperandDemand(*transfer.indexing,
                                     transfer.indexing->operands[1],
                                     {{{0, 0, 0}, {2, extent + 1, 128}}})
                  .status,
              IndexRelationStatus::Invalid);
  }
}

TEST(TensorResultIndexingTest, ZeroOffsetSubsetIsNotAnEqualVolumeReshape) {
  using namespace wafer::analysis;
  for (int64_t extent : {1024, 1025, 1031}) {
    auto context = createContext();
    for (bool clipped : {false, true}) {
      const int64_t destination = clipped ? extent : extent / 2;
      const int64_t source = clipped ? extent / 2 : extent;
      auto relation = IndexRelation::fromAffineMap(
          mlir::AffineMap::getMultiDimIdentityMap(3, context.get()),
          {2, destination, 128}, {2, source, 128});
      ASSERT_TRUE(relation.isExact()) << relation.reason;
      auto image = relation.get()->getExactStaticRectangularImagePieces(
          {0, 0, 0}, {2, destination, 128});
      ASSERT_TRUE(image.isExact()) << image.reason;
      ASSERT_EQ(image.domains.size(), 1u);
      EXPECT_EQ(image.domains.front().offsets,
                (llvm::SmallVector<int64_t, 4>{0, 0, 0}));
      EXPECT_EQ(image.domains.front().sizes,
                (llvm::SmallVector<int64_t, 4>{2, extent / 2, 128}));
      EXPECT_TRUE(relation.get()->contains({1, extent / 2 - 1, 127},
                                           {1, extent / 2 - 1, 127}));
      EXPECT_FALSE(
          relation.get()->contains({1, extent / 2, 127}, {1, extent / 2, 127}));
    }
  }
}

} // namespace
