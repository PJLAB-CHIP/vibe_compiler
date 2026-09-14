//===- NormalizeTensorReadsTest.cpp - Explicit Linalg read dependencies ---===//

#include "Wafer/Analysis/Linalg/SemanticRootAnalysis.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace {
class NormalizeTensorReadsTest : public ::testing::Test {
protected:
  NormalizeTensorReadsTest() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  bool normalize(mlir::ModuleOp module) {
    mlir::PassManager manager(context.get());
    manager.addNestedPass<mlir::func::FuncOp>(
        wafer::createNormalizeLinalgTensorReadsPass());
    return mlir::succeeded(manager.run(module)) &&
           mlir::succeeded(mlir::verify(module));
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream,
                     mlir::OpPrintingFlags().elideLargeElementsAttrs(8));
    return text;
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  }

  static mlir::linalg::GenericOp lastGeneric(mlir::ModuleOp module) {
    mlir::linalg::GenericOp result;
    module.walk([&](mlir::linalg::GenericOp generic) { result = generic; });
    return result;
  }

  // Each entry selects an output coordinate; -1 is the zero coordinate of a
  // unit source dimension. This also provides an independent value oracle.
  mlir::OwningOpRef<mlir::ModuleOp> makeRead(int64_t extent,
                                             mlir::Type elementType,
                                             llvm::ArrayRef<int64_t> projection,
                                             bool constant) {
    mlir::OpBuilder builder(context.get());
    auto loc = builder.getUnknownLoc();
    llvm::SmallVector<int64_t> outputShape{2, extent, 8};
    llvm::SmallVector<int64_t> inputShape;
    for (int64_t dimension : projection)
      inputShape.push_back(dimension < 0 ? 1 : outputShape[dimension]);
    auto inputType = mlir::RankedTensorType::get(inputShape, elementType);
    auto outputType = mlir::RankedTensorType::get(outputShape, elementType);
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());
    auto function = builder.create<mlir::func::FuncOp>(
        loc, "read", builder.getFunctionType({inputType}, {outputType}));
    builder.setInsertionPointToStart(function.addEntryBlock());
    mlir::Value input = function.getArgument(0);
    if (constant) {
      llvm::SmallVector<llvm::APInt> bits;
      for (int64_t i = 0; i < inputType.getNumElements(); ++i)
        bits.emplace_back(16, (i * 37 + 0x8000) & 0xffff);
      mlir::DenseElementsAttr data;
      if (auto type = mlir::dyn_cast<mlir::FloatType>(elementType)) {
        llvm::SmallVector<llvm::APFloat> values;
        for (const auto &value : bits)
          values.emplace_back(type.getFloatSemantics(), value);
        data = mlir::DenseElementsAttr::get(inputType, values);
      } else {
        data = mlir::DenseElementsAttr::get(inputType, bits);
      }
      input = builder.create<mlir::arith::ConstantOp>(loc, data);
    }
    auto init =
        builder.create<mlir::tensor::EmptyOp>(loc, outputShape, elementType);
    auto generic = builder.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{outputType}, mlir::ValueRange{},
        mlir::ValueRange{init},
        llvm::ArrayRef<mlir::AffineMap>{builder.getMultiDimIdentityMap(3)},
        llvm::SmallVector<mlir::utils::IteratorType>(
            3, mlir::utils::IteratorType::parallel),
        [&](mlir::OpBuilder &nested, mlir::Location loc,
            mlir::ValueRange arguments) {
          llvm::SmallVector<mlir::Value> indices;
          for (int64_t dimension : projection) {
            if (dimension < 0)
              indices.push_back(
                  nested.create<mlir::arith::ConstantIndexOp>(loc, 0));
            else
              indices.push_back(
                  nested.create<mlir::linalg::IndexOp>(loc, dimension));
          }
          auto value =
              nested.create<mlir::tensor::ExtractOp>(loc, input, indices);
          nested.create<mlir::linalg::YieldOp>(loc, value.getResult());
        });
    builder.create<mlir::func::ReturnOp>(loc, generic.getResults());
    return module;
  }

  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(NormalizeTensorReadsTest, ProjectedReadsPreserveEveryElementBit) {
  mlir::OpBuilder builder(context.get());
  for (int64_t extent : {1024, 1025, 1031}) {
    for (mlir::Type type :
         {mlir::Type(builder.getF16Type()), mlir::Type(builder.getBF16Type()),
          mlir::Type(builder.getI16Type())}) {
      for (llvm::SmallVector<int64_t> projection :
           {llvm::SmallVector<int64_t>{0, 1, 2}, {0, 2, 1}, {1}, {-1, 1, -1}}) {
        SCOPED_TRACE(extent);
        auto module = makeRead(extent, type, projection, true);
        ASSERT_TRUE(module);
        ASSERT_TRUE(normalize(*module));
        auto generic = lastGeneric(*module);
        ASSERT_EQ(generic.getNumDpsInputs(), 1u);
        auto map = generic.getIndexingMapsArray()[0];
        ASSERT_EQ(map.getNumResults(), projection.size());
        for (auto [axis, dimension] : llvm::enumerate(projection))
          EXPECT_EQ(map.getResult(axis),
                    dimension < 0 ? builder.getAffineConstantExpr(0)
                                  : builder.getAffineDimExpr(dimension));
        EXPECT_TRUE(
            generic.getBody()->getOps<mlir::tensor::ExtractOp>().empty());
        auto before = print(*module);
        ASSERT_TRUE(normalize(*module));
        EXPECT_EQ(before, print(*module));

        mlir::PassManager manager(context.get());
        manager.addPass(mlir::createCanonicalizerPass());
        manager.addPass(wafer::createFoldStaticTensorOpsPass());
        ASSERT_TRUE(mlir::succeeded(manager.run(*module)));
        auto function = module->lookupSymbol<mlir::func::FuncOp>("read");
        auto returned = mlir::cast<mlir::func::ReturnOp>(
                            function.getBody().front().getTerminator())
                            .getOperand(0);
        auto constant = returned.getDefiningOp<mlir::arith::ConstantOp>();
        ASSERT_TRUE(constant) << print(*module);
        auto values = mlir::cast<mlir::DenseElementsAttr>(constant.getValue());
        auto inputShape = mlir::cast<mlir::RankedTensorType>(
                              function.getArgument(0).getType())
                              .getShape();
        for (int64_t b = 0; b < 2; ++b)
          for (int64_t m = 0; m < extent; ++m)
            for (int64_t n = 0; n < 8; ++n) {
              int64_t coordinates[] = {b, m, n};
              int64_t inputOffset = 0;
              for (auto [axis, dimension] : llvm::enumerate(projection))
                inputOffset = inputOffset * inputShape[axis] +
                              (dimension < 0 ? 0 : coordinates[dimension]);
              int64_t outputOffset = (b * extent + m) * 8 + n;
              llvm::APInt actual =
                  mlir::isa<mlir::FloatType>(type)
                      ? (*(values.getValues<llvm::APFloat>().begin() +
                           outputOffset))
                            .bitcastToAPInt()
                      : *(values.getValues<llvm::APInt>().begin() +
                          outputOffset);
              ASSERT_EQ(actual.getZExtValue(),
                        (inputOffset * 37 + 0x8000) & 0xffff);
            }
      }
    }
  }
}

TEST_F(NormalizeTensorReadsTest, RealTilingProjectsInputsAndCoversMainAndTail) {
  mlir::OpBuilder builder(context.get());
  for (int64_t extent : {1024, 1025, 1031}) {
    for (llvm::SmallVector<int64_t> projection :
         {llvm::SmallVector<int64_t>{0, 1, 2}, {0, 2, 1}, {1}, {-1, 1, -1}}) {
      SCOPED_TRACE(extent);
      auto module = makeRead(extent, builder.getF16Type(), projection, false);
      ASSERT_TRUE(normalize(*module));
      auto generic = lastGeneric(*module);
      auto input = generic.getInputs()[0];
      auto tiling = mlir::cast<mlir::TilingInterface>(generic.getOperation());
      std::vector<unsigned> coverage(2 * extent * 8, 0);
      unsigned mainTiles = 0, tailTiles = 0;
      builder.setInsertionPoint(generic);
      for (int64_t b = 0; b < 2; ++b)
        for (int64_t m = 0; m < extent; m += 256)
          for (int64_t n = 0; n < 8; n += 4) {
            int64_t offsets[] = {b, m, n};
            int64_t sizes[] = {1, std::min<int64_t>(256, extent - m), 4};
            llvm::SmallVector<mlir::OpFoldResult> offsetAttrs, sizeAttrs;
            for (auto value : offsets)
              offsetAttrs.push_back(builder.getIndexAttr(value));
            for (auto value : sizes)
              sizeAttrs.push_back(builder.getIndexAttr(value));
            auto tiled =
                tiling.getTiledImplementation(builder, offsetAttrs, sizeAttrs);
            ASSERT_TRUE(mlir::succeeded(tiled));
            ASSERT_EQ(tiled->tiledOps.size(), 1u);
            auto tile = mlir::cast<mlir::linalg::GenericOp>(tiled->tiledOps[0]);
            auto slice = tile.getInputs()[0]
                             .getDefiningOp<mlir::tensor::ExtractSliceOp>();
            ASSERT_TRUE(slice);
            EXPECT_EQ(slice.getSource(), input);
            for (auto [axis, dimension] : llvm::enumerate(projection)) {
              EXPECT_EQ(slice.getStaticOffsets()[axis],
                        dimension < 0 ? 0 : offsets[dimension]);
              EXPECT_EQ(slice.getStaticSizes()[axis],
                        dimension < 0 ? 1 : sizes[dimension]);
              EXPECT_EQ(slice.getStaticStrides()[axis], 1);
            }
            EXPECT_TRUE(
                tile.getBody()->getOps<mlir::tensor::ExtractOp>().empty());
            auto outputSlice =
                tile.getOutputs()[0]
                    .getDefiningOp<mlir::tensor::ExtractSliceOp>();
            ASSERT_TRUE(outputSlice);
            EXPECT_EQ(outputSlice.getStaticOffsets(),
                      llvm::ArrayRef<int64_t>(offsets));
            EXPECT_EQ(outputSlice.getStaticSizes(),
                      llvm::ArrayRef<int64_t>(sizes));
            for (int64_t row = m; row < m + sizes[1]; ++row)
              for (int64_t col = n; col < n + sizes[2]; ++col)
                ++coverage[(b * extent + row) * 8 + col];
            sizes[1] == 256 ? ++mainTiles : ++tailTiles;
          }
      EXPECT_EQ(mainTiles, 16u);
      EXPECT_EQ(tailTiles, extent == 1024 ? 0u : 4u);
      EXPECT_TRUE(
          llvm::all_of(coverage, [](unsigned count) { return count == 1; }));
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
  }
}

TEST_F(NormalizeTensorReadsTest, ReusesOnlyMatchingInputsAndPreservesInitTie) {
  auto module = parse(R"mlir(
#id = affine_map<(b,m,n)->(b,m,n)>
module { func.func @read(%a: tensor<2x1024x1024xi32>) -> tensor<2x1024x1024xi32> {
  %r = linalg.generic {indexing_maps=[#id,#id],
      iterator_types=["parallel","parallel","parallel"], read_test = "kept"}
      ins(%a: tensor<2x1024x1024xi32>) outs(%a: tensor<2x1024x1024xi32>) {
    ^bb0(%input:i32,%init:i32):
      %b=linalg.index 0:index
      %m=linalg.index 1:index
      %n=linalg.index 2:index
      %x=tensor.extract %a[%b,%m,%n]:tensor<2x1024x1024xi32>
      %y=tensor.extract %a[%b,%n,%m]:tensor<2x1024x1024xi32>
      %z=tensor.extract %a[%b,%n,%m]:tensor<2x1024x1024xi32>
      %sum=arith.addi %x,%y:i32
      %sum2=arith.addi %sum,%z:i32
      %sum3=arith.addi %sum2,%init:i32
      linalg.yield %sum3:i32
  } -> tensor<2x1024x1024xi32>
  return %r:tensor<2x1024x1024xi32>
}})mlir");
  ASSERT_TRUE(module);
  auto generic = lastGeneric(*module);
  auto initArgument = generic.getBody()->getArgument(1);
  auto init = generic.getOutputs()[0];
  ASSERT_TRUE(normalize(*module));
  ASSERT_EQ(generic.getNumDpsInputs(), 2u);
  EXPECT_EQ(generic.getInputs()[0], init);
  EXPECT_EQ(generic.getInputs()[1], init);
  EXPECT_EQ(generic.getOutputs()[0], init);
  EXPECT_EQ(generic.getBody()->getArgument(2), initArgument);
  EXPECT_EQ(generic->getAttrOfType<mlir::StringAttr>("read_test").getValue(),
            "kept");
  auto adds = llvm::to_vector(generic.getBody()->getOps<mlir::arith::AddIOp>());
  ASSERT_EQ(adds.size(), 3u);
  EXPECT_EQ(adds[0].getLhs(), generic.getBody()->getArgument(0));
  EXPECT_EQ(adds[0].getRhs(), generic.getBody()->getArgument(1));
  EXPECT_EQ(adds[1].getRhs(), generic.getBody()->getArgument(1));
  EXPECT_EQ(adds[2].getRhs(), initArgument);
}

TEST_F(NormalizeTensorReadsTest, CapturedInitIsImmutableInputInReduction) {
  auto module = parse(R"mlir(
#in=affine_map<(b,m,n,k)->(b,m,n,k)>
#out=affine_map<(b,m,n,k)->(b,m,n)>
module {func.func @reduce(%x:tensor<2x1025x8x4xi32>,%init:tensor<2x1025x8xi32>)
    ->tensor<2x1025x8xi32> {
  %r=linalg.generic {indexing_maps=[#in,#out],
      iterator_types=["parallel","parallel","parallel","reduction"]}
      ins(%x:tensor<2x1025x8x4xi32>) outs(%init:tensor<2x1025x8xi32>) {
    ^bb0(%input:i32,%acc:i32):
      %b=linalg.index 0:index
      %m=linalg.index 1:index
      %n=linalg.index 2:index
      %v=tensor.extract %init[%b,%m,%n]:tensor<2x1025x8xi32>
      %a=arith.addi %input,%v:i32
      %s=arith.addi %a,%acc:i32
      linalg.yield %s:i32
  } ->tensor<2x1025x8xi32>
  return %r:tensor<2x1025x8xi32>
}})mlir");
  ASSERT_TRUE(module);
  auto generic = lastGeneric(*module);
  auto accumulator = generic.getBody()->getArgument(1);
  ASSERT_TRUE(normalize(*module));
  ASSERT_EQ(generic.getNumDpsInputs(), 2u);
  EXPECT_EQ(generic.getInputs()[1], generic.getOutputs()[0]);
  EXPECT_EQ(generic.getBody()->getArgument(2), accumulator);
  auto adds = llvm::to_vector(generic.getBody()->getOps<mlir::arith::AddIOp>());
  EXPECT_EQ(adds[0].getRhs(), generic.getBody()->getArgument(1));
  EXPECT_EQ(adds[1].getRhs(), accumulator);
}

TEST_F(NormalizeTensorReadsTest,
       StructuredProducerBecomesObservableDependency) {
  for (int64_t extent : {1024, 1025, 1031}) {
    std::string source = R"mlir(
#ids=affine_map<(b,m)->(b,m)>
#out=affine_map<(b,m,n)->(b,m,n)>
module {func.func @read(%ids:tensor<2xEXTENTxi64>,%table:tensor<4096x8xf16>)
    ->tensor<2xEXTENTx8xf16> {
  %ie=tensor.empty():tensor<2xEXTENTxi32>
  %indices=linalg.generic {indexing_maps=[#ids,#ids],iterator_types=["parallel","parallel"]}
      ins(%ids:tensor<2xEXTENTxi64>) outs(%ie:tensor<2xEXTENTxi32>) {
    ^bb0(%id:i64,%out:i32): %i=arith.trunci %id:i64 to i32
    linalg.yield %i:i32
  } ->tensor<2xEXTENTxi32>
  %empty=tensor.empty():tensor<2xEXTENTx8xf16>
  %r=linalg.generic {indexing_maps=[#out],iterator_types=["parallel","parallel","parallel"]}
      outs(%empty:tensor<2xEXTENTx8xf16>) {
    ^bb0(%out:f16):
      %b=linalg.index 0:index
      %m=linalg.index 1:index
      %n=linalg.index 2:index
      %token=tensor.extract %indices[%b,%m]:tensor<2xEXTENTxi32>
      %index=arith.index_castui %token:i32 to index
      %v=tensor.extract %table[%index,%n]:tensor<4096x8xf16>
      linalg.yield %v:f16
  } ->tensor<2xEXTENTx8xf16>
  return %r:tensor<2xEXTENTx8xf16>
}})mlir";
    for (size_t offset; (offset = source.find("EXTENT")) != std::string::npos;)
      source.replace(offset, 6, std::to_string(extent));
    auto module = parse(source);
    ASSERT_TRUE(module);
    auto function = module->lookupSymbol<mlir::func::FuncOp>("read");
    std::string reason;
    using namespace wafer::compiler::detail;
    {
      auto dag = StructuredDAGAnalysis::create(function, &reason);
      ASSERT_TRUE(mlir::succeeded(dag)) << reason;
      EXPECT_TRUE(dag->getEdges().empty());
      EXPECT_TRUE(mlir::failed(SemanticRootAnalysis::create(*dag, &reason)));
    }
    ASSERT_TRUE(normalize(*module));
    auto dag = StructuredDAGAnalysis::create(function, &reason);
    ASSERT_TRUE(mlir::succeeded(dag)) << reason;
    ASSERT_EQ(dag->getEdges().size(), 1u);
    EXPECT_EQ(dag->getEdges()[0].producer, 0u);
    EXPECT_EQ(dag->getEdges()[0].consumer, 1u);
    EXPECT_EQ(dag->getEdges()[0].consumerOperand, 0u);
    auto roots = SemanticRootAnalysis::create(*dag, &reason);
    ASSERT_TRUE(mlir::succeeded(roots)) << reason;
    EXPECT_EQ(roots->getRoots().size(), 2u);
    auto generic = lastGeneric(*module);
    auto reads =
        llvm::to_vector(generic.getBody()->getOps<mlir::tensor::ExtractOp>());
    ASSERT_EQ(reads.size(), 1u);
    EXPECT_EQ(reads[0].getTensor(), function.getArgument(1));
    auto cast =
        reads[0].getIndices()[0].getDefiningOp<mlir::arith::IndexCastUIOp>();
    ASSERT_TRUE(cast);
    EXPECT_EQ(cast.getIn(), generic.getBody()->getArgument(0));
  }
}

TEST_F(NormalizeTensorReadsTest, UnknownOrNonprojectedReadsStayUnchanged) {
  for (llvm::StringRef read :
       {"%index=arith.index_cast %dynamic:i32 to index\n"
        "%v=tensor.extract %a[%b,%index,%n]:tensor<2x1031x8xi32>",
        "%zero=arith.constant 0:index\n"
        "%v=tensor.extract %a[%b,%zero,%n]:tensor<2x1031x8xi32>",
        "%v=tensor.extract %larger[%b,%m,%n]:tensor<2x2048x8xi32>",
        "%local=tensor.empty():tensor<2x1031x8xi32>\n"
        "%v=tensor.extract %local[%b,%m,%n]:tensor<2x1031x8xi32>"}) {
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << "#id=affine_map<(b,m,n)->(b,m,n)>\n"
              "module {func.func @read(%a:tensor<2x1031x8xi32>,"
              "%larger:tensor<2x2048x8xi32>,%dynamic:i32) "
              "->tensor<2x1031x8xi32> {\n"
              "%empty=tensor.empty():tensor<2x1031x8xi32>\n"
              "%r=linalg.generic {indexing_maps=[#id],"
              "iterator_types=[\"parallel\",\"parallel\",\"parallel\"]} "
              "outs(%empty:tensor<2x1031x8xi32>) {\n^bb0(%out:i32):\n"
              "%b=linalg.index 0:index\n%m=linalg.index "
              "1:index\n%n=linalg.index 2:index\n"
           << read
           << "\nlinalg.yield %v:i32 } ->tensor<2x1031x8xi32>\n"
              "return %r:tensor<2x1031x8xi32> }}";
    auto module = parse(source);
    ASSERT_TRUE(module);
    auto before = print(*module);
    ASSERT_TRUE(normalize(*module));
    EXPECT_EQ(before, print(*module));
  }
}

TEST_F(NormalizeTensorReadsTest, UnknownSourceOrLoopExtentStaysUnchanged) {
  for (bool dynamicLoop : {false, true}) {
    std::string input =
        dynamicLoop ? "tensor<2x1025x8xi32>" : "tensor<2x?x8xi32>";
    std::string output =
        dynamicLoop ? "tensor<2x?x8xi32>" : "tensor<2x1025x8xi32>";
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << "#id=affine_map<(b,m,n)->(b,m,n)>\n"
              "module {func.func @read(%a:"
           << input << ",%size:index) ->" << output
           << " {\n%empty=tensor.empty(" << (dynamicLoop ? "%size" : "")
           << "):" << output
           << "\n%r=linalg.generic {indexing_maps=[#id],"
              "iterator_types=[\"parallel\",\"parallel\",\"parallel\"]} "
              "outs(%empty:"
           << output
           << ") {\n^bb0(%out:i32):\n"
              "%b=linalg.index 0:index\n%m=linalg.index "
              "1:index\n%n=linalg.index 2:index\n"
              "%v=tensor.extract %a[%b,%m,%n]:"
           << input << "\nlinalg.yield %v:i32 } ->" << output
           << "\nreturn %r:" << output << " }}";
    auto module = parse(source);
    ASSERT_TRUE(module);
    auto before = print(*module);
    ASSERT_TRUE(normalize(*module));
    EXPECT_EQ(before, print(*module));
  }
}
} // namespace
