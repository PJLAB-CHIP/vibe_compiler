//===- TensorResultIndexingTest.cpp --------------------------------------===//

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"

#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <utility>

namespace {

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::tensor::registerSubsetOpInterfaceExternalModels(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

TEST(TensorResultIndexingTest, ParameterizedLoopGridUsesAffineCoordinates) {
  using namespace wafer::analysis;
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @grid() {
    %lower = arith.constant 0 : index
    %upper = arith.constant 528 : index
    %step = arith.constant 24 : index
    scf.for %iv = %lower to %upper step %step {
      %base = arith.constant 480 : index
      %factor = arith.constant 2 : index
      %product = arith.muli %iv, %factor : index
      %sum = arith.addi %product, %base : index
      %reverse = arith.addi %base, %product : index
      %nonlinear_arith = arith.muli %iv, %iv : index
      %translated = affine.apply affine_map<(d0) -> (d0 + 480)>(%iv)
      %scaled = affine.apply affine_map<(d0) -> (d0 * 2 + 480)>(%iv)
      %nonlinear = affine.apply affine_map<(d0) -> (d0 floordiv 2 + 480)>(%iv)
      %composed = affine.apply affine_map<(d0) -> (d0 + 8)>(%sum)
      scf.yield
    }
    return
  }
}

)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::affine::AffineApplyOp, 4> applies;
  llvm::SmallVector<mlir::arith::AddIOp, 2> additions;
  llvm::SmallVector<mlir::arith::MulIOp, 2> products;
  mlir::scf::ForOp loop;
  module->walk(
      [&](mlir::affine::AffineApplyOp apply) { applies.push_back(apply); });
  module->walk([&](mlir::arith::AddIOp add) { additions.push_back(add); });
  module->walk(
      [&](mlir::arith::MulIOp multiply) { products.push_back(multiply); });
  module->walk([&](mlir::scf::ForOp current) { loop = current; });
  ASSERT_EQ(applies.size(), 4u);
  ASSERT_EQ(additions.size(), 2u);
  ASSERT_EQ(products.size(), 2u);
  ASSERT_TRUE(loop);
  auto plain = queryTensorLoopGrid(loop.getInductionVar()).grid;
  auto translated = queryTensorLoopGrid(applies[0].getResult()).grid;
  auto scaled = queryTensorLoopGrid(applies[1].getResult()).grid;
  ASSERT_TRUE(plain);
  ASSERT_TRUE(translated);
  ASSERT_TRUE(scaled);
  EXPECT_EQ(plain->step, 24);
  EXPECT_EQ(translated->lower, 480);
  EXPECT_EQ(translated->step, 24);
  EXPECT_EQ(scaled->lower, 480);
  EXPECT_EQ(scaled->step, 48);
  EXPECT_EQ(getTensorLoopGridFloor(*scaled, 512), 480);
  EXPECT_EQ(getTensorLoopGridCeil(*scaled, 512), 528);
  EXPECT_EQ(getTensorLoopIndex(*scaled, 528), 24);
  for (auto add : additions) {
    auto equivalent = queryTensorLoopGrid(add.getResult()).grid;
    ASSERT_TRUE(equivalent);
    EXPECT_EQ(equivalent->lower, scaled->lower);
    EXPECT_EQ(equivalent->step, scaled->step);
  }
  auto composed = queryTensorLoopGrid(applies[3].getResult()).grid;
  ASSERT_TRUE(composed);
  EXPECT_EQ(composed->lower, 488);
  EXPECT_EQ(composed->step, 48);
  EXPECT_EQ(queryTensorLoopGrid(applies[2].getResult()).status,
            TensorResultIndexingStatus::Unsupported);
  EXPECT_EQ(queryTensorLoopGrid(products[1].getResult()).status,
            TensorResultIndexingStatus::Unsupported);
  IndexRelationLimits bounded;
  bounded.maxVariables = 1;
  EXPECT_EQ(queryTensorLoopGrid(loop.getInductionVar(), bounded).status,
            TensorResultIndexingStatus::Exact);
  EXPECT_EQ(queryTensorLoopGrid(applies[0].getResult(), bounded).status,
            TensorResultIndexingStatus::ResourceExhausted);
}

TEST(TensorResultIndexingTest, LoopGridNormalizesEquivalentOperandBindings) {
  // Scalar index expressions are a bounded oracle; the loop spans real tiles.
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    func.func @indices() {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1031 = arith.constant 1031 : index
      scf.for %i = %c0 to %c1031 step %c16 {
        %a = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%i)[%c8]
        %b = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%c8, %i)
        %c = arith.addi %i, %c16 : index
        %d = arith.subi %c, %c8 : index
      }
      return
    }
  )mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::Value> values;
  module->walk([&](mlir::affine::AffineApplyOp op) {
    values.push_back(op.getResult());
  });
  module->walk(
      [&](mlir::arith::SubIOp op) { values.push_back(op.getResult()); });
  ASSERT_EQ(values.size(), 3u);
  for (auto value : values) {
    auto result = wafer::analysis::queryTensorLoopGrid(value);
    ASSERT_TRUE(result.grid) << result.detail;
    EXPECT_EQ(result.grid->base, 8);
    EXPECT_EQ(result.grid->scale, 1);
    EXPECT_EQ(result.grid->step, 16);
    EXPECT_EQ(result.grid->upper, 1039);
  }
}

TEST(TensorResultIndexingTest, ComposedViewsKeepASelectedColumnWindowCompact) {
  using namespace wafer::analysis;
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    const std::string n = std::to_string(extent);
    const std::string source = "tensor<3x" + n + "x1x768xf16>";
    const std::string slice = "tensor<1x" + n + "x1x768xf16>";
    const std::string flat = "tensor<" + std::to_string(extent * 768) + "xf16>";
    const std::string view = "tensor<" + n + "x12x64xf16>";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        "module { func.func @read(%input: " + source + ") -> " + view +
            " { %s = tensor.extract_slice %input[2, 0, 0, 0] [1, " + n +
            ", 1, 768] [1, 1, 1, 1] : " + source + " to " + slice +
            "\n%f = tensor.collapse_shape %s [[0, 1, 2, 3]] : " + slice +
            " into " + flat +
            "\n%v = tensor.expand_shape %f [[0, 1, 2]] output_shape [" + n +
            ", 12, 64] : " + flat + " into " + view + "\nreturn %v : " + view +
            "\n}}",
        context.get());
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    auto result = function.getBody().front().getTerminator()->getOperand(0);
    auto chain = deriveTensorViewIndexing(result);
    ASSERT_TRUE(chain.isExact()) << chain.detail;
    ASSERT_EQ(chain.indexing->source, function.getArgument(0));
    for (auto window : {std::pair<int64_t, int64_t>{128, 128},
                        std::pair<int64_t, int64_t>{extent - 7, 7}}) {
      auto image = getTensorViewTileSource(
          *chain.indexing, {extent, 12, 64},
          {{window.first, 10, 0}, {window.second, 2, 64}});
      ASSERT_TRUE(image.isExact()) << image.reason;
      EXPECT_EQ(image.domain->offsets,
                (llvm::SmallVector<int64_t, 4>{2, window.first, 0, 640}));
      EXPECT_EQ(image.domain->sizes,
                (llvm::SmallVector<int64_t, 4>{1, window.second, 1, 128}));
      for (int64_t row = 0; row < window.second; ++row)
        for (int64_t col = 0; col < 64; ++col)
          ASSERT_TRUE(chain.indexing->resultToSource.contains(
              {window.first + row, 11, col},
              {2, window.first + row, 0, 704 + col}));
    }
  }
}

TEST(TensorResultIndexingTest, TransparentReadsStopAtPartialAndUpdatedSources) {
  using namespace wafer::analysis;
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @read(%input: tensor<2x1031x128xf16>,
                      %dest: tensor<2x1031x130xf16>) {
        %zero = arith.constant 0.0 : f16
        %pad = tensor.pad %input low[0, 0, 1] high[0, 0, 1] {
        ^bb0(%b: index, %m: index, %n: index):
          tensor.yield %zero : f16
        } : tensor<2x1031x128xf16> to tensor<2x1031x130xf16>
        %insert = tensor.insert_slice %input into %dest[0, 0, 1]
          [2, 1031, 128] [1, 1, 1] : tensor<2x1031x128xf16> into tensor<2x1031x130xf16>
        %p = tensor.collapse_shape %pad [[0], [1, 2]]
          : tensor<2x1031x130xf16> into tensor<2x134030xf16>
        %i = tensor.collapse_shape %insert [[0], [1, 2]]
          : tensor<2x1031x130xf16> into tensor<2x134030xf16>
        %stride = tensor.extract_slice %input[0, 0, 0] [2, 1031, 64]
          [1, 1, 2] : tensor<2x1031x128xf16> to tensor<2x1031x64xf16>
        return
      }
    })mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  unsigned chains = 0;
  module->walk([&](mlir::tensor::CollapseShapeOp op) {
    auto chain = deriveTensorViewIndexing(op.getResult());
    ASSERT_TRUE(chain.isExact()) << chain.detail;
    EXPECT_EQ(chain.indexing->source, op.getSrc());
    auto leaf =
        deriveTensorResultIndexing(mlir::cast<mlir::OpResult>(op.getSrc()));
    ASSERT_TRUE(leaf.isExact());
    EXPECT_EQ(leaf.indexing->getTransparentSource(), nullptr);
    ++chains;
  });
  EXPECT_EQ(chains, 2u);
  module->walk([&](mlir::tensor::ExtractSliceOp op) {
    auto chain = deriveTensorViewIndexing(op.getResult());
    ASSERT_TRUE(chain.isExact());
    auto tile = getTensorViewTileSource(*chain.indexing, {2, 1031, 64},
                                        {{0, 128, 0}, {2, 128, 64}});
    EXPECT_EQ(tile.status, IndexRelationStatus::Unsupported);
  });
}

TEST(TensorResultIndexingTest, AssemblyCoverageIsIndependentOfUseAndFusion) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    std::string text = R"mlir(
module {
  func.func @assembly(%a: tensor<2x512x64xf16>,
                      %b: tensor<2x512x64xf16>,
                      %c: tensor<2xTAILx64xf16>,
                      %d: tensor<2xTAILx64xf16>)
      -> (tensor<2xEXTENTx128xf16>, tensor<2xEXTENTx128xf16>) {
    %empty = tensor.empty() : tensor<2xEXTENTx128xf16>
    %x0 = tensor.insert_slice %a into %empty[0, 0, 0] [2, 512, 64] [1, 1, 1]
      : tensor<2x512x64xf16> into tensor<2xEXTENTx128xf16>
    %x1 = tensor.insert_slice %b into %x0[0, 0, 64] [2, 512, 64] [1, 1, 1]
      : tensor<2x512x64xf16> into tensor<2xEXTENTx128xf16>
    %x2 = tensor.insert_slice %c into %x1[0, 512, 0] [2, TAIL, 64] [1, 1, 1]
      : tensor<2xTAILx64xf16> into tensor<2xEXTENTx128xf16>
    %x3 = tensor.insert_slice %d into %x2[0, 512, 64] [2, TAIL, 64] [1, 1, 1]
      : tensor<2xTAILx64xf16> into tensor<2xEXTENTx128xf16>
    %overlap = tensor.insert_slice %a into %x3[0, 0, 0] [2, 512, 64] [1, 1, 1]
      : tensor<2x512x64xf16> into tensor<2xEXTENTx128xf16>
    return %x3, %overlap : tensor<2xEXTENTx128xf16>, tensor<2xEXTENTx128xf16>
  }
}

)mlir";
    auto replaceAll = [&](llvm::StringRef needle, const std::string &value) {
      size_t position = 0;
      while ((position = text.find(needle.str(), position)) !=
             std::string::npos) {
        text.replace(position, needle.size(), value);
        position += value.size();
      }
    };
    replaceAll("EXTENT", std::to_string(extent));
    replaceAll("TAIL", std::to_string(extent - 512));
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        text, mlir::ParserConfig(context.get()));
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    auto *terminator = function.getBody().front().getTerminator();
    auto exact = wafer::analysis::queryTensorAssembly(terminator->getOperand(0));
    ASSERT_TRUE(exact.isExact())
        << static_cast<int>(exact.status) << ": " << exact.detail << "\n"
        << text;
    ASSERT_EQ(exact.segments.size(), 4u);
    EXPECT_EQ(exact.segments[0].offsets,
              (llvm::SmallVector<int64_t, 4>{0, 512, 64}));
    EXPECT_EQ(exact.segments[3].offsets,
              (llvm::SmallVector<int64_t, 4>{0, 0, 0}));
    for (const auto &segment : exact.segments) {
      ASSERT_NE(segment.sourceOperand, nullptr);
      EXPECT_EQ(segment.sourceOperand->get(), segment.source);
    }
    auto overlap =
        wafer::analysis::queryTensorAssembly(terminator->getOperand(1));
    EXPECT_EQ(overlap.status,
              wafer::analysis::TensorAssemblyStatus::Unsupported);
    EXPECT_EQ(overlap.detail, "insert assembly rectangles overlap");
    wafer::analysis::StaticRectangularIndexSet whole{{0, 0, 0},
                                                      {2, extent, 128}};
    auto lastWriter = wafer::analysis::queryTensorAssemblyDemand(
        terminator->getOperand(1), whole);
    ASSERT_TRUE(lastWriter.isExact()) << lastWriter.detail;
    ASSERT_EQ(lastWriter.pieces.size(), 4u);
    EXPECT_EQ(lastWriter.pieces[0].resultWindow.offsets,
              (llvm::SmallVector<int64_t, 4>{0, 0, 0}));
    EXPECT_EQ(lastWriter.pieces[0].source, function.getArgument(0));
    EXPECT_NE(lastWriter.pieces[0].sourceOperand,
              exact.segments[3].sourceOperand);
  }
}

TEST(TensorResultIndexingTest,
     AssemblyDemandUsesLastWriterAndReadsOnlyTheOldDestinationRemainder) {
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @assembly(%base: tensor<2x1025x8xf16>,
                      %a: tensor<2x400x8xf16>,
                      %b: tensor<2x300x8xf16>) -> tensor<2x1025x8xf16> {
    %first = tensor.insert_slice %a into %base[0, 100, 0] [2, 400, 8] [1, 1, 1]
      : tensor<2x400x8xf16> into tensor<2x1025x8xf16>
    %last = tensor.insert_slice %b into %first[0, 300, 0] [2, 300, 8] [1, 1, 1]
      : tensor<2x300x8xf16> into tensor<2x1025x8xf16>
    return %last : tensor<2x1025x8xf16>
  }
}
)mlir", mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto value = function.getBody().front().getTerminator()->getOperand(0);
  wafer::analysis::StaticRectangularIndexSet request{{0, 200, 0},
                                                       {2, 600, 8}};
  auto result = wafer::analysis::queryTensorAssemblyDemand(value, request);
  ASSERT_TRUE(result.isExact()) << result.detail;
  ASSERT_EQ(result.pieces.size(), 3u);
  EXPECT_EQ(result.pieces[0].source, function.getArgument(2));
  EXPECT_EQ(result.pieces[0].resultWindow.offsets,
            (llvm::SmallVector<int64_t, 4>{0, 300, 0}));
  EXPECT_EQ(result.pieces[0].sourceWindow.offsets,
            (llvm::SmallVector<int64_t, 4>{0, 0, 0}));
  EXPECT_EQ(result.pieces[1].source, function.getArgument(1));
  EXPECT_EQ(result.pieces[1].resultWindow.offsets,
            (llvm::SmallVector<int64_t, 4>{0, 200, 0}));
  EXPECT_EQ(result.pieces[1].sourceWindow.offsets,
            (llvm::SmallVector<int64_t, 4>{0, 100, 0}));
  EXPECT_EQ(result.pieces[2].source, function.getArgument(0));
  EXPECT_EQ(result.pieces[2].resultWindow.offsets,
            (llvm::SmallVector<int64_t, 4>{0, 600, 0}));
  EXPECT_EQ(result.pieces[2].sourceWindow.offsets,
            (llvm::SmallVector<int64_t, 4>{0, 600, 0}));
  for (int64_t row = 200; row < 800; ++row) {
    unsigned owners = 0;
    for (const auto &piece : result.pieces)
      owners += piece.resultWindow.offsets[1] <= row &&
                row < piece.resultWindow.offsets[1] +
                          piece.resultWindow.sizes[1];
    EXPECT_EQ(owners, 1u) << row;
  }
  wafer::analysis::IndexRelationLimits tight;
  tight.maxRectangularPieces = 2;
  auto bounded =
      wafer::analysis::queryTensorAssemblyDemand(value, request, tight);
  EXPECT_EQ(bounded.status,
            wafer::analysis::TensorAssemblyStatus::ResourceExhausted);
}

TEST(TensorResultIndexingTest,
     AssemblyDemandPreservesRankReducedSourceCoordinates) {
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @assembly(%base: tensor<1x1025x8xf16>,
                      %source: tensor<400x8xf16>)
      -> (tensor<1x1025x8xf16>, tensor<1x1025x8xf16>) {
    %updated = tensor.insert_slice %source into %base[0, 200, 0]
      [1, 400, 8] [1, 1, 1]
      : tensor<400x8xf16> into tensor<1x1025x8xf16>
    %empty = tensor.empty() : tensor<1x1025x8xf16>
    %partial = tensor.insert_slice %source into %empty[0, 200, 0]
      [1, 400, 8] [1, 1, 1]
      : tensor<400x8xf16> into tensor<1x1025x8xf16>
    return %updated, %partial
      : tensor<1x1025x8xf16>, tensor<1x1025x8xf16>
  }
}
)mlir", mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto *terminator = function.getBody().front().getTerminator();
  wafer::analysis::StaticRectangularIndexSet middle{{0, 300, 0},
                                                      {1, 200, 8}};
  auto exact = wafer::analysis::queryTensorAssemblyDemand(
      terminator->getOperand(0), middle);
  ASSERT_TRUE(exact.isExact()) << exact.detail;
  ASSERT_EQ(exact.pieces.size(), 1u);
  EXPECT_EQ(exact.pieces[0].source, function.getArgument(1));
  EXPECT_EQ(exact.pieces[0].sourceWindow.offsets,
            (llvm::SmallVector<int64_t, 4>{100, 0}));
  EXPECT_EQ(exact.pieces[0].sourceWindow.sizes,
            (llvm::SmallVector<int64_t, 4>{200, 8}));
  wafer::analysis::StaticRectangularIndexSet whole{{0, 0, 0},
                                                     {1, 1025, 8}};
  auto undefined = wafer::analysis::queryTensorAssemblyDemand(
      terminator->getOperand(1), whole);
  EXPECT_EQ(undefined.status,
            wafer::analysis::TensorAssemblyStatus::Unsupported);
  EXPECT_EQ(undefined.detail,
            "insert demand reads undefined tensor.empty data");
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
     RankReducedSlicesKeepTheActualDestinationWindow) {
  using namespace wafer::analysis;
  for (int64_t extent : {1024, 1025, 1031}) {
    auto context = createContext();
    const std::string source =
        "tensor<1x" + std::to_string(extent - 2) + "x32xf16>";
    const std::string destination =
        "tensor<2x1x" + std::to_string(extent) + "x64xf16>";
    const std::string result =
        "tensor<1x" + std::to_string(extent) + "x32xf16>";
    const std::string text =
        "module { func.func @update(%src: " + source +
        ", %dst: " + destination + ") -> " + result +
        " { %r = tensor.insert_slice %src into %dst[1, 0, 1, 16] "
        "[1, 1, " +
        std::to_string(extent - 2) + ", 32] [1, 1, 1, 1] : " + source +
        " into " + destination +
        " %s = tensor.extract_slice %r[1, 0, 0, 16] "
        "[1, 1, " +
        std::to_string(extent) + ", 32] [1, 1, 1, 1] : " + destination +
        " to " + result + " return %s : " + result + " } }";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module);
    mlir::tensor::InsertSliceOp insert;
    mlir::tensor::ExtractSliceOp extract;
    module->walk([&](mlir::tensor::InsertSliceOp op) { insert = op; });
    module->walk([&](mlir::tensor::ExtractSliceOp op) { extract = op; });
    auto indexing = deriveTensorResultIndexing(
        mlir::cast<mlir::OpResult>(insert.getResult()));
    ASSERT_TRUE(indexing.isExact()) << indexing.detail;
    auto read = deriveTensorResultIndexing(
        mlir::cast<mlir::OpResult>(extract.getResult()));
    ASSERT_TRUE(read.isExact()) << read.detail;
    auto window =
        getTensorOperandDemand(*read.indexing, read.indexing->operands[0],
                               {{{0, 0, 0}, {1, extent, 32}}});
    ASSERT_TRUE(window.isExact()) << window.reason;
    ASSERT_EQ(window.domains.size(), 1u);
    EXPECT_EQ(window.domains[0].offsets,
              (llvm::SmallVector<int64_t, 4>{1, 0, 0, 16}));
    EXPECT_EQ(window.domains[0].sizes,
              (llvm::SmallVector<int64_t, 4>{1, 1, extent, 32}));
    auto src = getTensorOperandDemand(
        *indexing.indexing, indexing.indexing->operands[0], window.domains);
    auto dst = getTensorOperandDemand(
        *indexing.indexing, indexing.indexing->operands[1], window.domains);
    ASSERT_TRUE(src.isExact()) << src.reason;
    ASSERT_TRUE(dst.isExact()) << dst.reason;
    ASSERT_EQ(src.domains.size(), 1u);
    EXPECT_EQ(src.domains[0].offsets, (llvm::SmallVector<int64_t, 4>{0, 0, 0}));
    EXPECT_EQ(src.domains[0].sizes,
              (llvm::SmallVector<int64_t, 4>{1, extent - 2, 32}));
    ASSERT_EQ(dst.domains.size(), 2u);
    for (auto [index, box] : llvm::enumerate(dst.domains)) {
      EXPECT_EQ(box.offsets, (llvm::SmallVector<int64_t, 4>{
                                 1, 0, index ? extent - 1 : 0, 16}));
      EXPECT_EQ(box.sizes, (llvm::SmallVector<int64_t, 4>{1, 1, 1, 32}));
    }
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

TEST(TensorResultIndexingTest, BoundedAssemblyReadsPreservePartialLastWriters) {
  using namespace wafer::analysis;
  for (int64_t extent : {1024, 1025, 1031}) {
    for (int64_t window : {1, 32}) {
      SCOPED_TRACE(window);
      for (int64_t step : {16, 32, 64}) {
        auto context = createContext();
        std::string full = "tensor<2x" + std::to_string(extent) + "x128xf16>";
        std::string text;
        llvm::raw_string_ostream os(text);
        os << "module { func.func @read(%old: " << full
           << ", %a: tensor<2x600x64xf16>, %b: tensor<400x96xf16>) {\n"
           << " %first = tensor.insert_slice %a into %old[0,100,16] [2,600,64] "
              "[1,1,1] : tensor<2x600x64xf16> into "
           << full << "\n"
           << " %last = tensor.insert_slice %b into %first[1,480,0] [1,400,96] "
              "[1,1,1] : tensor<400x96xf16> into "
           << full << "\n"
           << " %zero = arith.constant 0 : index\n"
           << " %upper = arith.constant " << extent - window + 1 << " : index\n"
           << " %step = arith.constant " << step << " : index\n"
           << " scf.for %i = %zero to %upper step %step {\n"
           << " %read = tensor.extract_slice %last[0,%i,0] [2," << window
           << ",128] [1,1,1] : " << full << " to tensor<2x" << window
           << "x128xf16>\n"
           << " } return } }";
        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
        ASSERT_TRUE(module);
        mlir::tensor::ExtractSliceOp read;
        module->walk([&](mlir::tensor::ExtractSliceOp op) { read = op; });
        auto query = queryTensorAssemblyRead(
            read.getSource(), read.getMixedOffsets(), {2, window, 128});
        ASSERT_TRUE(query.isExact()) << query.detail;
        ASSERT_EQ(query.loops.size(), 1u);
        EXPECT_LT(query.cases.size(), 20u);
        auto function = *module->getOps<mlir::func::FuncOp>().begin();
        unsigned instances = 0;
        for (const auto &part : query.cases) {
          for (int64_t iv = part.lowerBounds[0]; iv < part.upperBounds[0];
               iv += step) {
            ++instances;
            // Independent integer oracle of the original newest-write
            // semantics. Visit every demanded coordinate, including
            // rank-reduced sources.
            for (int64_t batch = 0; batch < 2; ++batch)
              for (int64_t row = 0; row < window; ++row)
                for (int64_t col = 0; col < 128; ++col) {
                  llvm::SmallVector<int64_t, 4> point{batch, row, col};
                  unsigned matches = 0;
                  for (const auto &piece : part.pieces) {
                    bool inside = true;
                    for (unsigned axis = 0; axis < 3; ++axis)
                      inside &=
                          point[axis] >= piece.resultWindow.offsets[axis] &&
                          point[axis] < piece.resultWindow.offsets[axis] +
                                            piece.resultWindow.sizes[axis];
                    if (!inside)
                      continue;
                    ++matches;
                    const bool newest = batch == 1 && iv + row >= 480 &&
                                        iv + row < 880 && col < 96;
                    const bool first = iv + row >= 100 && iv + row < 700 &&
                                       col >= 16 && col < 80;
                    EXPECT_EQ(piece.source, function.getArgument(newest  ? 2
                                                                 : first ? 1
                                                                         : 0));
                    llvm::SmallVector<mlir::Attribute> parameters{
                        mlir::IntegerAttr::get(
                            mlir::IndexType::get(context.get()), iv)},
                        folded;
                    ASSERT_TRUE(mlir::succeeded(
                        piece.sourceOffsets.constantFold(parameters, folded)));
                    llvm::SmallVector<int64_t, 4> source;
                    unsigned sourceAxis = 0;
                    for (unsigned axis = 0; axis < 3; ++axis)
                      if (!newest || axis != 0)
                        source.push_back(
                            mlir::cast<mlir::IntegerAttr>(folded[sourceAxis++])
                                .getInt() +
                            point[axis] - piece.resultWindow.offsets[axis]);
                    auto expected =
                        newest
                            ? llvm::SmallVector<int64_t, 4>{iv + row - 480, col}
                        : first ? llvm::SmallVector<int64_t, 4>{batch,
                                                                iv + row - 100,
                                                                col - 16}
                                : llvm::SmallVector<int64_t, 4>{batch, iv + row,
                                                                col};
                    ASSERT_EQ(source, expected);
                  }
                  ASSERT_EQ(matches, 1u);
                }
          }
        }
        EXPECT_EQ(instances, (extent - window) / step + 1);
        IndexRelationLimits limited;
        limited.maxRectangularPieces = 2;
        auto rejected =
            queryTensorAssemblyRead(read.getSource(), read.getMixedOffsets(),
                                    {2, window, 128}, limited);
        EXPECT_EQ(rejected.status, TensorAssemblyStatus::ResourceExhausted);
        EXPECT_TRUE(rejected.cases.empty());
      }
    }
  }
}

TEST(TensorResultIndexingTest, BoundedAssemblyReadsDoNotDemandUndefinedHoles) {
  using namespace wafer::analysis;
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module { func.func @holes(%a: tensor<2x32x128xf16>, %b: tensor<2x32x128xf16>) {
  %empty = tensor.empty() : tensor<2x2048x128xf16>
  %first = tensor.insert_slice %a into %empty[0,0,0] [2,32,128] [1,1,1] : tensor<2x32x128xf16> into tensor<2x2048x128xf16>
  %last = tensor.insert_slice %b into %first[0,1024,0] [2,32,128] [1,1,1] : tensor<2x32x128xf16> into tensor<2x2048x128xf16>
  %c0 = arith.constant 0 : index
  %c2048 = arith.constant 2048 : index
  %c1024 = arith.constant 1024 : index
  scf.for %iv = %c0 to %c2048 step %c1024 {
    %read = tensor.extract_slice %last[0,%iv,0] [2,32,128] [1,1,1] : tensor<2x2048x128xf16> to tensor<2x32x128xf16>
  }
  return
} }
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  mlir::tensor::ExtractSliceOp read;
  module->walk([&](mlir::tensor::ExtractSliceOp op) { read = op; });
  auto query = queryTensorAssemblyRead(read.getSource(), read.getMixedOffsets(),
                                       {2, 32, 128});
  ASSERT_TRUE(query.isExact()) << query.detail;
  ASSERT_EQ(query.cases.size(), 2u);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  for (unsigned i = 0; i < 2; ++i) {
    ASSERT_EQ(query.cases[i].pieces.size(), 1u);
    EXPECT_EQ(query.cases[i].pieces.front().source, function.getArgument(i));
    EXPECT_EQ(query.cases[i].pieces.front().resultWindow.sizes,
              (llvm::SmallVector<int64_t, 4>{2, 32, 128}));
  }
  auto undefined = queryTensorAssemblyRead(
      read.getSource(), read.getMixedOffsets(), {2, 33, 128});
  EXPECT_EQ(undefined.status, TensorAssemblyStatus::Unsupported);
}

TEST(TensorResultIndexingTest, BoundedAssemblyViewUsesTheWholeCurrentRelation) {
  using namespace wafer::analysis;
  for (int64_t extent : {1024, 1031}) {
    auto context = createContext();
    std::string full = "tensor<2x" + std::to_string(extent) + "x128xf16>";
    std::string view = "tensor<2x" + std::to_string(extent) + "x4x32xf16>";
    std::string text;
    llvm::raw_string_ostream os(text);
    os << "module { func.func @view(%old: " << full
       << ", %new: tensor<2x600x64xf16>) {\n"
       << " %updated = tensor.insert_slice %new into %old[0,100,16] [2,600,64] "
          "[1,1,1] : tensor<2x600x64xf16> into "
       << full << "\n"
       << " %view = tensor.expand_shape %updated [[0], [1], [2,3]] "
          "output_shape [2,"
       << extent << ",4,32] : " << full << " into " << view << "\n"
       << " %zero = arith.constant 0 : index\n %upper = arith.constant "
       << extent - 31 << " : index\n %step = arith.constant 32 : index\n"
       << " scf.for %i = %zero to %upper step %step {\n"
       << " %read = tensor.extract_slice %view[0,%i,1,0] [2,32,2,32] [1,1,1,1] "
          ": "
       << view << " to tensor<2x32x2x32xf16>\n } return } }";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module);
    mlir::tensor::ExtractSliceOp read;
    module->walk([&](mlir::tensor::ExtractSliceOp op) { read = op; });
    auto query = queryTensorAssemblyRead(
        read.getSource(), read.getMixedOffsets(), {2, 32, 2, 32});
    ASSERT_TRUE(query.isExact()) << query.detail;
    EXPECT_EQ(query.shape, (llvm::SmallVector<int64_t, 4>{2, 32, 64}));
    ASSERT_EQ(query.loops.size(), 1u);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    for (const auto &part : query.cases)
      for (int64_t iv = part.lowerBounds[0]; iv < part.upperBounds[0]; iv += 32)
        for (int64_t row = 0; row < 32; ++row)
          for (int64_t col = 0; col < 64; ++col) {
            unsigned matches = 0;
            for (const auto &piece : part.pieces) {
              if (row < piece.resultWindow.offsets[1] ||
                  row >= piece.resultWindow.offsets[1] +
                             piece.resultWindow.sizes[1] ||
                  col < piece.resultWindow.offsets[2] ||
                  col >= piece.resultWindow.offsets[2] +
                             piece.resultWindow.sizes[2])
                continue;
              ++matches;
              bool updated = iv + row >= 100 && iv + row < 700 && col < 48;
              EXPECT_EQ(piece.source, function.getArgument(updated ? 1 : 0));
              llvm::SmallVector<mlir::Attribute> args{mlir::IntegerAttr::get(
                  mlir::IndexType::get(context.get()), iv)},
                  result;
              ASSERT_TRUE(mlir::succeeded(
                  piece.sourceOffsets.constantFold(args, result)));
              EXPECT_EQ(mlir::cast<mlir::IntegerAttr>(result[1]).getInt() +
                            row - piece.resultWindow.offsets[1],
                        iv + row - (updated ? 100 : 0));
              EXPECT_EQ(mlir::cast<mlir::IntegerAttr>(result[2]).getInt() +
                            col - piece.resultWindow.offsets[2],
                        col + (updated ? 16 : 32));
            }
            ASSERT_EQ(matches, 1u);
          }
  }
}

TEST(TensorResultIndexingTest, StaticAssemblyViewPreservesMultipleExactImages) {
  using namespace wafer::analysis;
  for (int64_t extent : {1024, 1025, 1031}) {
    auto context = createContext();
    std::string full = "tensor<2x" + std::to_string(extent) + "x4x32xf16>";
    std::string view = "tensor<2x" + std::to_string(extent * 4) + "x32xf16>";
    std::string text;
    llvm::raw_string_ostream os(text);
    os << "module { func.func @view(%old: " << full
       << ", %new: tensor<2x8x4x32xf16>) {\n"
       << " %updated = tensor.insert_slice %new into %old[0,0,0,0] "
          "[2,8,4,32] [1,1,1,1] : tensor<2x8x4x32xf16> into "
       << full << "\n"
       << " %view = tensor.collapse_shape %updated [[0], [1,2], [3]] : " << full
       << " into " << view << "\n"
       << " %read = tensor.extract_slice %view[0,1,0] [2," << extent
       << ",32] [1,1,1] : " << view << " to tensor<2x" << extent
       << "x32xf16>\n return } }";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module);
    mlir::tensor::ExtractSliceOp read;
    module->walk([&](mlir::tensor::ExtractSliceOp op) { read = op; });
    auto query = queryTensorAssemblyRead(
        read.getSource(), read.getMixedOffsets(), {2, extent, 32});
    ASSERT_TRUE(query.isExact()) << query.detail;
    ASSERT_EQ(query.cases.size(), 1u);
    EXPECT_EQ(query.shape, (llvm::SmallVector<int64_t, 4>{2, extent, 32}));
    EXPECT_GT(query.cases[0].pieces.size(), 1u);
    for (bool pieceBudget : {false, true}) {
      IndexRelationLimits limits;
      if (pieceBudget)
        limits.maxRectangularPieces = 1;
      else
        limits.maxConstraintWork = 0;
      auto limited = queryTensorAssemblyRead(
          read.getSource(), read.getMixedOffsets(), {2, extent, 32}, limits);
      EXPECT_EQ(limited.status, TensorAssemblyStatus::ResourceExhausted)
          << limited.detail;
      EXPECT_TRUE(limited.cases.empty());
    }
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    for (int64_t b = 0; b < 2; ++b)
      for (int64_t row = 0; row < extent; ++row)
        for (int64_t col = 0; col < 32; ++col) {
          llvm::SmallVector<int64_t, 4> point{b, row, col};
          unsigned matches = 0;
          for (const auto &piece : query.cases[0].pieces) {
            int64_t linear = 0;
            bool inside = true;
            for (unsigned axis = 0; axis < point.size(); ++axis) {
              int64_t local = point[axis] - piece.resultWindow.offsets[axis];
              inside &= local >= 0 && local < piece.resultWindow.sizes[axis];
              linear = linear * piece.resultWindow.sizes[axis] + local;
            }
            if (!inside)
              continue;
            ++matches;
            EXPECT_EQ(piece.source, function.getArgument(row + 1 < 32 ? 1 : 0));
            llvm::SmallVector<mlir::Attribute> folded;
            ASSERT_TRUE(
                mlir::succeeded(piece.sourceOffsets.constantFold({}, folded)));
            llvm::SmallVector<int64_t, 4> source(piece.sourceSizes.size());
            for (size_t axis = source.size(); axis-- > 0;) {
              source[axis] =
                  mlir::cast<mlir::IntegerAttr>(folded[axis]).getInt() +
                  linear % piece.sourceSizes[axis];
              linear /= piece.sourceSizes[axis];
            }
            EXPECT_EQ(source, (llvm::SmallVector<int64_t, 4>{
                                  b, (row + 1) / 4, (row + 1) % 4, col}));
          }
          ASSERT_EQ(matches, 1u);
        }
  }
}

static llvm::SmallVector<int64_t>
evaluateSubsetMap(mlir::AffineMap map, llvm::ArrayRef<int64_t> point) {
  llvm::SmallVector<mlir::Attribute> arguments, folded;
  for (int64_t value : point)
    arguments.push_back(
        mlir::IntegerAttr::get(mlir::IndexType::get(map.getContext()), value));
  EXPECT_TRUE(mlir::succeeded(map.constantFold(arguments, folded)));
  llvm::SmallVector<int64_t> values;
  for (auto value : folded)
    values.push_back(mlir::cast<mlir::IntegerAttr>(value).getInt());
  return values;
}

static bool containsSubsetPoint(mlir::IntegerSet domain,
                                llvm::ArrayRef<int64_t> point) {
  auto conditions = evaluateSubsetMap(
      mlir::AffineMap::get(domain.getNumDims(), domain.getNumSymbols(),
                           domain.getConstraints(), domain.getContext()),
      point);
  for (unsigned i = 0; i < conditions.size(); ++i)
    if (domain.isEq(i) ? conditions[i] != 0 : conditions[i] < 0)
      return false;
  return true;
}

TEST(TensorResultIndexingTest,
     SubsetDemandTraversesBothSidesOfTheStructuralDAG) {
  using namespace wafer::analysis;
  for (int64_t extent : {1024, 1025, 1031}) {
    auto context = createContext();
    std::string text = R"mlir(
      func.func @nested(%base: tensor<2xEXTENTx8xf16>,
                        %a: tensor<2x256x8xf16>, %b: tensor<2x256x8xf16>) {
        %empty = tensor.empty() : tensor<2x512x8xf16>
        %first = tensor.insert_slice %a into %empty[0, 0, 0]
          [2, 256, 8] [1, 1, 1] : tensor<2x256x8xf16> into tensor<2x512x8xf16>
        %inner = tensor.insert_slice %b into %first[0, 256, 0]
          [2, 256, 8] [1, 1, 1] : tensor<2x256x8xf16> into tensor<2x512x8xf16>
        %nested = tensor.insert_slice %inner into %base[0, 128, 0]
          [2, 512, 8] [1, 1, 1] : tensor<2x512x8xf16> into tensor<2xEXTENTx8xf16>
        %f0 = tensor.insert_slice %a into %base[0, 128, 0]
          [2, 256, 8] [1, 1, 1] : tensor<2x256x8xf16> into tensor<2xEXTENTx8xf16>
        %flat = tensor.insert_slice %b into %f0[0, 384, 0]
          [2, 256, 8] [1, 1, 1] : tensor<2x256x8xf16> into tensor<2xEXTENTx8xf16>
        %zero = arith.constant 0 : index
        %end = arith.constant END : index
        %step = arith.constant 31 : index
        scf.for %i = %zero to %end step %step {
          %x = tensor.extract_slice %nested[0, %i, 0] [2, 64, 8] [1, 1, 1]
            : tensor<2xEXTENTx8xf16> to tensor<2x64x8xf16>
          %y = tensor.extract_slice %flat[0, %i, 0] [2, 64, 8] [1, 1, 1]
            : tensor<2xEXTENTx8xf16> to tensor<2x64x8xf16>
        }
        return
      }
    )mlir";
    for (auto [key, replacement] :
         {std::pair{std::string("EXTENT"), std::to_string(extent)},
          std::pair{std::string("END"), std::to_string(extent - 63)}}) {
      size_t at = 0;
      while ((at = text.find(key, at)) != std::string::npos) {
        text.replace(at, key.size(), replacement);
        at += replacement.size();
      }
    }
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    llvm::SmallVector<mlir::tensor::ExtractSliceOp> reads;
    module->walk(
        [&](mlir::tensor::ExtractSliceOp read) { reads.push_back(read); });
    ASSERT_EQ(reads.size(), 2u);
    for (auto read : reads) {
      IndexRelationWork work(IndexRelationLimits{});
      auto query = queryTensorSubsetDemand(
          read.getSource(), read.getMixedOffsets(), {2, 64, 8}, read, work);
      ASSERT_TRUE(query.isExact()) << query.detail << "; extent=" << extent;
      ASSERT_EQ(query.demand->parameters.size(), 1u);
      for (int64_t i = 0; i < extent - 63; i += 31)
        for (int64_t row = 0; row < 64; ++row) {
          llvm::SmallVector<int64_t> point{i, 1, row, 7};
          ASSERT_TRUE(containsSubsetPoint(query.demand->domain, point));
          unsigned owners = 0;
          for (const auto &source : query.demand->sources) {
            bool active = true;
            for (auto condition : source.conditions)
              active &= containsSubsetPoint(condition.set, point) !=
                        condition.complement;
            if (!active)
              continue;
            ++owners;
            int64_t global = i + row;
            unsigned argument = global < 128 || global >= 640 ? 0
                                : global < 384                ? 1
                                                              : 2;
            EXPECT_EQ(source.source, function.getArgument(argument));
            auto coordinates = evaluateSubsetMap(source.coordinates, point);
            EXPECT_EQ(coordinates, (llvm::SmallVector<int64_t>{
                                       1,
                                       global - (argument == 1   ? 128
                                                 : argument == 2 ? 384
                                                                 : 0),
                                       7}));
          }
          EXPECT_EQ(owners, 1u) << i << ":" << row;
        }
      IndexRelationLimits limits;
      limits.maxConstraintWork = work.getConsumed();
      IndexRelationWork shared(limits);
      ASSERT_TRUE(queryTensorSubsetDemand(read.getSource(),
                                          read.getMixedOffsets(), {2, 64, 8},
                                          read, shared)
                      .isExact());
      EXPECT_EQ(queryTensorSubsetDemand(read.getSource(),
                                        read.getMixedOffsets(), {2, 64, 8},
                                        read, shared)
                    .status,
                TensorAssemblyStatus::ResourceExhausted);
    }
  }
}

TEST(TensorResultIndexingTest,
     SubsetDemandUsesActualBranchAndEmptyLoopDomains) {
  using namespace wafer::analysis;
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    func.func @holes(%source: tensor<2x512x8xf16>) {
      %empty = tensor.empty() : tensor<2x1031x8xf16>
      %defined = tensor.insert_slice %source into %empty[0, 0, 0]
        [2, 512, 8] [1, 1, 1] : tensor<2x512x8xf16> into tensor<2x1031x8xf16>
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c481 = arith.constant 481 : index
      %c1000 = arith.constant 1000 : index
      scf.for %i = %c0 to %c1000 step %c16 {
        %in = arith.cmpi slt, %i, %c481 : index
        scf.if %in {
          %valid = tensor.extract_slice %defined[0, %i, 0]
            [2, 32, 8] [1, 1, 1] : tensor<2x1031x8xf16> to tensor<2x32x8xf16>
        } else {
          %invalid = tensor.extract_slice %defined[0, %i, 0]
            [2, 32, 8] [1, 1, 1] : tensor<2x1031x8xf16> to tensor<2x32x8xf16>
        }
      }
      scf.for %i = %c0 to %c0 step %c16 {
        %never = tensor.extract_slice %empty[0, 768, 0]
          [2, 32, 8] [1, 1, 1] : tensor<2x1031x8xf16> to tensor<2x32x8xf16>
      }
      return
    }
  )mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::tensor::ExtractSliceOp> reads;
  module->walk(
      [&](mlir::tensor::ExtractSliceOp read) { reads.push_back(read); });
  ASSERT_EQ(reads.size(), 3u);
  for (unsigned i = 0; i < reads.size(); ++i) {
    IndexRelationWork work(IndexRelationLimits{});
    auto read = reads[i];
    auto result = queryTensorSubsetDemand(
        read.getSource(), read.getMixedOffsets(), {2, 32, 8}, read, work);
    if (i == 1) {
      EXPECT_EQ(result.status, TensorAssemblyStatus::Unsupported);
      continue;
    }
    ASSERT_TRUE(result.isExact()) << result.detail;
    if (i == 2)
      EXPECT_TRUE(result.demand->sources.empty());
    else {
      ASSERT_EQ(result.demand->sources.size(), 1u);
      EXPECT_EQ(result.demand->scopeConditions.size(), 1u);
    }
  }
}

TEST(TensorResultIndexingTest,
     ParameterExpressionsKeepCoupledDivisionAndIntegerSemantics) {
  using namespace wafer::analysis;
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    func.func @coordinates() {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c7 = arith.constant 7 : index
      %c17 = arith.constant 17 : index
      %c1031 = arith.constant 1031 : index
      %large = arith.constant 4294967296 : index
      scf.for %i = %c0 to %c1031 step %c7 {
        scf.for %j = %c0 to %c17 step %c1 {
          %a = affine.apply affine_map<(d0)[s0] -> ((d0 * 3 - s0) floordiv 7)>(%i)[%j]
          %b = affine.apply affine_map<(d0, d1) -> ((d1 * 3 - d0) floordiv 7)>(%j, %i)
          %negative = arith.subi %c0, %i : index
          %truncation = arith.divsi %negative, %c7 : index
          %floor = arith.floordivsi %negative, %c7 : index
          %unknown = arith.divsi %i, %j : index
        }
      }
      scf.for %i = %c0 to %large step %c1 {
        %narrow = arith.index_cast %i : index to i32
      }
      return
    }
  )mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::Value> offsets;
  module->walk([&](mlir::affine::AffineApplyOp op) { offsets.push_back(op); });
  for (auto value : offsets) {
    IndexRelationWork work(IndexRelationLimits{});
    auto result = queryTensorIndexExpressions(context.get(), {value}, work);
    ASSERT_TRUE(result.isExact()) << result.detail;
    ASSERT_EQ(result.expressions->parameters.size(), 2u);
    for (int64_t i = 0; i < 1031; i += 7)
      for (int64_t j = 0; j < 17; ++j) {
        llvm::SmallVector<int64_t> point;
        for (auto parameter : result.expressions->parameters) {
          auto loop = mlir::cast<mlir::scf::ForOp>(
              mlir::cast<mlir::BlockArgument>(parameter.induction)
                  .getOwner()
                  ->getParentOp());
          point.push_back(
              mlir::getConstantIntValue(loop.getUpperBound()) == 1031 ? i : j);
        }
        auto actual = evaluateSubsetMap(result.expressions->map, point);
        int64_t numerator = 3 * i - j;
        int64_t expected =
            numerator / 7 - (numerator < 0 && numerator % 7 != 0);
        EXPECT_EQ(actual.front(), expected);
      }
  }
  module->walk([&](mlir::arith::DivSIOp op) {
    IndexRelationWork work(IndexRelationLimits{});
    EXPECT_EQ(queryTensorIndexExpressions(context.get(), {op.getResult()}, work)
                  .status,
              TensorResultIndexingStatus::Unsupported);
  });
  module->walk([&](mlir::arith::FloorDivSIOp op) {
    IndexRelationWork work(IndexRelationLimits{});
    EXPECT_TRUE(
        queryTensorIndexExpressions(context.get(), {op.getResult()}, work)
            .isExact());
  });
  module->walk([&](mlir::arith::IndexCastOp op) {
    IndexRelationWork work(IndexRelationLimits{});
    EXPECT_EQ(queryTensorIndexExpressions(context.get(), {op.getResult()}, work)
                  .status,
              TensorResultIndexingStatus::Unsupported);
  });
}

TEST(TensorResultIndexingTest,
     SubsetDemandComposesPeriodicViewsAndRankProjection) {
  using namespace wafer::analysis;
  for (int64_t extent : {1024, 1025, 1031}) {
    auto context = createContext();
    std::string text = R"mlir(
      func.func @view(%a: tensor<512x8xf16>, %b: tensor<TAILx8xf16>) {
        %empty = tensor.empty() : tensor<1xEXTENTx8xf16>
        %first = tensor.insert_slice %a into %empty[0, 0, 0]
          [1, 512, 8] [1, 1, 1] : tensor<512x8xf16> into tensor<1xEXTENTx8xf16>
        %all = tensor.insert_slice %b into %first[0, 512, 0]
          [1, TAIL, 8] [1, 1, 1] : tensor<TAILx8xf16> into tensor<1xEXTENTx8xf16>
        %flat = tensor.collapse_shape %all [[0, 1, 2]]
          : tensor<1xEXTENTx8xf16> into tensor<FLATxf16>
        %view = tensor.expand_shape %flat [[0, 1, 2]] output_shape [1, ROWS, 4]
          : tensor<FLATxf16> into tensor<1xROWSx4xf16>
        %c0 = arith.constant 0 : index
        %end = arith.constant END : index
        %step = arith.constant 31 : index
        scf.for %i = %c0 to %end step %step {
          %read = tensor.extract_slice %view[0, %i, 0] [1, 32, 4] [1, 1, 1]
            : tensor<1xROWSx4xf16> to tensor<1x32x4xf16>
        }
        %tailStart = arith.constant LAST : index
        %tailEnd = arith.constant ROWS : index
        %tailStep = arith.constant 17 : index
        scf.for %i = %tailStart to %tailEnd step %tailStep {
          %readTail = tensor.extract_slice %view[0, %i, 0] [1, 17, 4] [1, 1, 1]
            : tensor<1xROWSx4xf16> to tensor<1x17x4xf16>
        }
        return
      }
    )mlir";
    for (auto [key, number] :
         {std::pair{"EXTENT", extent}, std::pair{"TAIL", extent - 512},
          std::pair{"FLAT", extent * 8}, std::pair{"ROWS", extent * 2},
          std::pair{"END", extent * 2 - 31},
          std::pair{"LAST", extent * 2 - 17}}) {
      size_t at = 0;
      while ((at = text.find(key, at)) != std::string::npos) {
        auto replacement = std::to_string(number);
        text.replace(at, std::string(key).size(), replacement);
        at += replacement.size();
      }
    }
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    llvm::SmallVector<mlir::tensor::ExtractSliceOp> reads;
    module->walk([&](mlir::tensor::ExtractSliceOp op) { reads.push_back(op); });
    ASSERT_EQ(reads.size(), 2u);
    for (auto read : reads) {
      IndexRelationWork work(IndexRelationLimits{});
      auto query =
          queryTensorSubsetDemand(read.getSource(), read.getMixedOffsets(),
                                  read.getStaticSizes(), read, work);
      ASSERT_TRUE(query.isExact())
          << query.detail << "; work=" << work.getConsumed();
      auto parameter = query.demand->parameters.front();
      for (int64_t i = parameter.lower; i < parameter.upper;
           i += parameter.step)
        for (int64_t row = 0; row < read.getStaticSizes()[1]; ++row)
          for (int64_t col = 0; col < 4; ++col) {
            llvm::SmallVector<int64_t> point{i, 0, row, col};
            unsigned owners = 0;
            for (const auto &source : query.demand->sources) {
              bool active = true;
              for (auto condition : source.conditions)
                active &= containsSubsetPoint(condition.set, point) !=
                          condition.complement;
              if (!active)
                continue;
              ++owners;
              int64_t global = (i + row) * 4 + col;
              unsigned argument = global / 8 < 512 ? 0 : 1;
              EXPECT_EQ(source.source, function.getArgument(argument));
              EXPECT_EQ(evaluateSubsetMap(source.coordinates, point),
                        (llvm::SmallVector<int64_t>{
                            global / 8 - (argument ? 512 : 0), global % 8}));
            }
            EXPECT_EQ(owners, 1u);
          }
    }
  }
}

TEST(TensorResultIndexingTest, SubsetBlockGuardsProveDenseOrderedCopies) {
  using namespace wafer::analysis;
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    func.func @source(%arg: tensor<1x4096x8xf16>) { return }
  )mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto p = mlir::getAffineDimExpr(0, context.get());
  auto u = mlir::getAffineDimExpr(2, context.get());
  auto v = mlir::getAffineDimExpr(3, context.get());
  auto zero = mlir::getAffineConstantExpr(0, context.get());
  auto floor = [](int64_t a, int64_t b) { return a / b - (a % b < 0); };
  // Independent bounded coordinate oracle on a real-sized rank-three source.
  // Negative coefficients, nested floor/mod and an odd block are deliberate.
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned variant = 0; variant < 3; ++variant) {
      TensorSubsetDemand demand;
      demand.shape = {1, extent, 8};
      demand.parameters.push_back({{}, -37, 44, 3});
      demand.domain = mlir::IntegerSet::get(4, 0, {zero}, {true});
      mlir::AffineExpr row = variant == 0 ? 200 - (p - u).floorDiv(7)
                             : variant == 1
                                 ? 200 - (p - u).floorDiv(7).floorDiv(3)
                                 : u * 3 - ((p - u) % 7).floorDiv(3) + 200;
      TensorSubsetSource source{
          function.getArgument(0),
          mlir::AffineMap::get(4, 0, {zero, row, v}, context.get()),
          {}};
      IndexRelationWork work(IndexRelationLimits{});
      uint64_t accepted = 0, rejected = 0;
      for (int64_t block : {1, 2, 3, 17, 32}) {
        auto proof =
            queryTensorSubsetBlock(demand, source, {1, block, 8}, work);
        ASSERT_TRUE(proof.status == TensorSubsetBlockStatus::Copy ||
                    proof.status == TensorSubsetBlockStatus::Subdivide)
            << proof.detail;
        if (!proof.block)
          continue;
        for (int64_t parameter = -37; parameter < 44; parameter += 3)
          for (int64_t origin = 0; origin + block <= extent; origin += 31) {
            llvm::SmallVector<int64_t> point{parameter, 0, origin, 0};
            bool active = true;
            for (auto guard : proof.block->guards)
              active &=
                  containsSubsetPoint(guard.set, point) != guard.complement;
            if (!active) {
              ++rejected;
              continue;
            }
            ++accepted;
            auto offsets = evaluateSubsetMap(proof.block->sourceOffsets, point);
            int64_t volume = 1;
            for (int64_t size : proof.block->sourceSizes)
              volume *= size;
            ASSERT_EQ(volume, block * 8);
            for (int64_t i = 0; i < block; ++i)
              for (int64_t j = 0; j < 8; ++j) {
                int64_t index = origin + i;
                int64_t a = parameter - index;
                int64_t adjustment = variant == 0 ? floor(a, 7)
                                     : variant == 1
                                         ? floor(floor(a, 7), 3)
                                         : floor(a - floor(a, 7) * 7, 3);
                llvm::SmallVector<int64_t> expected{
                    0, (variant == 2 ? index * 3 : 0) - adjustment + 200, j};
                int64_t ordinal = i * 8 + j;
                llvm::SmallVector<int64_t> actual(3);
                for (unsigned axis = 3; axis-- > 0;) {
                  actual[axis] =
                      offsets[axis] + ordinal % proof.block->sourceSizes[axis];
                  ordinal /= proof.block->sourceSizes[axis];
                }
                EXPECT_EQ(actual, expected);
              }
          }
      }
      EXPECT_GT(accepted, 0u);
      EXPECT_GT(rejected, 0u);
      EXPECT_LT(work.getConsumed(), work.getLimits().maxConstraintWork);
    }
}

TEST(TensorResultIndexingTest,
     SubsetBlockRetainsComplementAndRejectsPermutation) {
  using namespace wafer::analysis;
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    func.func @source(%arg: tensor<1x1031x1031xf16>) { return }
  )mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto u = mlir::getAffineDimExpr(1, context.get());
  auto v = mlir::getAffineDimExpr(2, context.get());
  auto zero = mlir::getAffineConstantExpr(0, context.get());
  TensorSubsetDemand demand;
  demand.shape = {1, 1031, 1031};
  demand.domain = mlir::IntegerSet::get(3, 0, {zero}, {true});
  TensorSubsetSource source{
      function.getArgument(0),
      mlir::AffineMap::get(3, 0, {zero, u, v}, context.get()),
      {{mlir::IntegerSet::get(3, 0, {u - 512, v - 400}, {false, false}),
        true}}};
  IndexRelationWork work(IndexRelationLimits{});
  for (int64_t extent : {1, 17, 32}) {
    auto proof = queryTensorSubsetBlock(demand, source, {1, extent, 8}, work);
    ASSERT_EQ(proof.status, TensorSubsetBlockStatus::Copy) << proof.detail;
    for (int64_t row : {0, 499, 511, 512, 1000})
      for (int64_t col : {0, 397, 400, 999}) {
        llvm::SmallVector<int64_t> point{0, row, col};
        bool active = true;
        for (auto guard : proof.block->guards)
          active &= containsSubsetPoint(guard.set, point) != guard.complement;
        if (active) {
          for (int64_t i = 0; i < extent; ++i)
            for (int64_t j = 0; j < 8; ++j)
              EXPECT_TRUE(row + i < 512 || col + j < 400);
        }
      }
  }
  source.conditions.clear();
  source.coordinates = mlir::AffineMap::get(3, 0, {zero, v, u}, context.get());
  auto permutation = queryTensorSubsetBlock(demand, source, {1, 17, 8}, work);
  EXPECT_EQ(permutation.status, TensorSubsetBlockStatus::Subdivide);
  auto unit = queryTensorSubsetBlock(demand, source, {1, 1, 1}, work);
  EXPECT_EQ(unit.status, TensorSubsetBlockStatus::Copy);
  IndexRelationLimits limits;
  limits.maxConstraintWork = 1;
  IndexRelationWork tight(limits);
  EXPECT_EQ(queryTensorSubsetBlock(demand, source, {1, 17, 8}, tight).status,
            TensorSubsetBlockStatus::ResourceExhausted);
}

TEST(TensorResultIndexingTest, SubsetBlockKeepsBatchedContiguousReshape) {
  using namespace wafer::analysis;
  for (int64_t extent : {1024, 1025, 1031}) {
    auto context = createContext();
    std::string text = "func.func @source(%arg: tensor<2x" +
                       std::to_string(extent) + "x4x32xf16>) { return }";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    auto b = mlir::getAffineDimExpr(0, context.get());
    auto r = mlir::getAffineDimExpr(1, context.get());
    auto c = mlir::getAffineDimExpr(2, context.get());
    auto zero = mlir::getAffineConstantExpr(0, context.get());
    TensorSubsetDemand demand;
    demand.shape = {2, extent, 32};
    demand.domain = mlir::IntegerSet::get(3, 0, {zero}, {true});
    TensorSubsetSource source{
        function.getArgument(0),
        mlir::AffineMap::get(3, 0, {b, (r + 1).floorDiv(4), (r + 1) % 4, c},
                             context.get()),
        {}};
    IndexRelationWork work(IndexRelationLimits{});
    auto proof = queryTensorSubsetBlock(demand, source, {2, 32, 32}, work);
    ASSERT_EQ(proof.status, TensorSubsetBlockStatus::Copy) << proof.detail;
    EXPECT_EQ(proof.block->sourceSizes,
              (llvm::SmallVector<int64_t>{2, 8, 4, 32}));
    for (int64_t origin = 0; origin + 32 <= extent; origin += 31) {
      llvm::SmallVector<int64_t> point{0, origin, 0};
      bool active = true;
      for (auto guard : proof.block->guards)
        active &= containsSubsetPoint(guard.set, point) != guard.complement;
      EXPECT_EQ(active, (origin + 1) % 4 == 0);
      if (!active)
        continue;
      auto offsets = evaluateSubsetMap(proof.block->sourceOffsets, point);
      for (int64_t batch = 0; batch < 2; ++batch)
        for (int64_t row = 0; row < 32; ++row)
          for (int64_t col = 0; col < 32; ++col) {
            int64_t ordinal = (batch * 32 + row) * 32 + col;
            llvm::SmallVector<int64_t> actual(4);
            for (unsigned axis = 4; axis-- > 0;) {
              actual[axis] =
                  offsets[axis] + ordinal % proof.block->sourceSizes[axis];
              ordinal /= proof.block->sourceSizes[axis];
            }
            EXPECT_EQ(actual, (llvm::SmallVector<int64_t>{
                                  batch, (origin + row + 1) / 4,
                                  (origin + row + 1) % 4, col}));
          }
    }
  }
}

TEST(TensorResultIndexingTest, SubsetBlockKeepsCoordinateCongruence) {
  using namespace wafer::analysis;
  for (int64_t extent : {1024, 1025, 1031}) {
    auto context = createContext();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        "func.func @source(%arg: tensor<1x" + std::to_string(extent) +
            "x8xf16>) { return }",
        context.get());
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    auto p = mlir::getAffineDimExpr(0, context.get());
    auto r = mlir::getAffineDimExpr(2, context.get());
    auto c = mlir::getAffineDimExpr(3, context.get());
    auto zero = mlir::getAffineConstantExpr(0, context.get());
    TensorSubsetDemand demand;
    demand.parameters.push_back({{}, 0, extent * 2 - 31, 31});
    demand.shape = {1, 32, 4};
    demand.domain = mlir::IntegerSet::get(4, 0, {zero}, {true});
    auto ordinal = (p + r) * 4 + c;
    TensorSubsetSource source{
        function.getArgument(0),
        mlir::AffineMap::get(4, 0, {zero, ordinal.floorDiv(8), ordinal % 8},
                             context.get()),
        {}};
    IndexRelationWork work(IndexRelationLimits{});
    auto proof = queryTensorSubsetBlock(demand, source, {1, 1, 4}, work);
    ASSERT_EQ(proof.status, TensorSubsetBlockStatus::Copy) << proof.detail;
    ASSERT_TRUE(proof.block->guards.empty());
    EXPECT_EQ(proof.block->sourceSizes, (llvm::SmallVector<int64_t>{1, 1, 4}));
    for (int64_t base = 0; base < extent * 2 - 31; base += 31)
      for (int64_t row = 0; row < 32; ++row) {
        auto offsets =
            evaluateSubsetMap(proof.block->sourceOffsets, {base, 0, row, 0});
        for (int64_t col = 0; col < 4; ++col) {
          int64_t index = (base + row) * 4 + col;
          EXPECT_EQ(offsets[1], index / 8);
          EXPECT_EQ(offsets[2] + col, index % 8);
        }
      }
  }
}

} // namespace
