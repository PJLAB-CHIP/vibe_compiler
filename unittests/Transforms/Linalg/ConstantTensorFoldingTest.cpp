//===- ConstantTensorFoldingTest.cpp - Demand-bounded constant reads ----===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>

namespace {
class ConstantTensorFoldingTest : public ::testing::Test {
protected:
  ConstantTensorFoldingTest() : builder(&context) {
    context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                        mlir::linalg::LinalgDialect,
                        mlir::tensor::TensorDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeModule(mlir::Type resultType) {
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
    auto function = builder.create<mlir::func::FuncOp>(
        module.getLoc(), "constant", builder.getFunctionType({}, {resultType}));
    builder.setInsertionPointToStart(function.addEntryBlock());
    return module;
  }

  bool fold(mlir::ModuleOp module) {
    if (mlir::failed(mlir::verify(module)))
      return false;
    mlir::PassManager manager(&context);
    manager.addPass(wafer::createFoldStaticTensorOpsPass());
    return mlir::succeeded(manager.run(module)) &&
           mlir::succeeded(mlir::verify(module));
  }

  mlir::Value returned(mlir::ModuleOp module) {
    auto function = module.lookupSymbol<mlir::func::FuncOp>("constant");
    return mlir::cast<mlir::func::ReturnOp>(
               function.getBody().front().getTerminator())
        .getOperand(0);
  }

  mlir::DenseElementsAttr constantResult(mlir::ModuleOp module) {
    auto constant = returned(module).getDefiningOp<mlir::arith::ConstantOp>();
    return constant
               ? mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue())
               : mlir::DenseElementsAttr{};
  }

  mlir::Value constant(mlir::DenseElementsAttr value) {
    return builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(),
                                                   value);
  }

  mlir::MLIRContext context;
  mlir::OpBuilder builder;
};

TEST_F(ConstantTensorFoldingTest, LargeMappedCompareReadsOnlySelectedElements) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (bool transpose : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(transpose);
      auto inputType =
          mlir::RankedTensorType::get({1, extent, 512}, builder.getI8Type());
      auto outputType = mlir::RankedTensorType::get(
          transpose ? llvm::ArrayRef<int64_t>{1, 512, extent}
                    : llvm::ArrayRef<int64_t>{1, extent, 512},
          builder.getI1Type());
      auto module = makeModule(outputType);
      llvm::SmallVector<llvm::APInt> values;
      values.reserve(inputType.getNumElements());
      for (int64_t row = 0; row < extent; ++row)
        for (int64_t column = 0; column < 512; ++column)
          values.emplace_back(8, (row * 7 + column * 3) % 251);
      auto input = constant(mlir::DenseElementsAttr::get(inputType, values));
      auto rhsType = mlir::RankedTensorType::get({512}, builder.getI8Type());
      values.clear();
      for (int64_t column = 0; column < 512; ++column)
        values.emplace_back(8, column % 197);
      auto rhs = constant(mlir::DenseElementsAttr::get(rhsType, values));
      // The actual init is a large splat, but the body never reads it.
      auto init = constant(mlir::DenseElementsAttr::get(outputType, false));
      auto b = builder.getAffineDimExpr(0);
      auto m = builder.getAffineDimExpr(1);
      auto n = builder.getAffineDimExpr(2);
      auto identity = builder.getMultiDimIdentityMap(3);
      auto result = builder.create<mlir::linalg::GenericOp>(
          builder.getUnknownLoc(), mlir::TypeRange{outputType},
          mlir::ValueRange{input, rhs}, mlir::ValueRange{init},
          llvm::ArrayRef<mlir::AffineMap>{
              identity, mlir::AffineMap::get(3, 0, {n}, &context),
              transpose ? mlir::AffineMap::get(3, 0, {b, n, m}, &context)
                        : identity},
          llvm::SmallVector<mlir::utils::IteratorType>(
              3, mlir::utils::IteratorType::parallel),
          [&](mlir::OpBuilder &nested, mlir::Location loc,
              mlir::ValueRange arguments) {
            auto cmp = nested.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::ugt, arguments[0],
                arguments[1]);
            nested.create<mlir::linalg::YieldOp>(loc, cmp.getResult());
          });
      builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                           result.getResults());
      ASSERT_TRUE(fold(*module));
      auto output = constantResult(*module);
      ASSERT_TRUE(output);
      ASSERT_EQ(output.getType(), outputType);
      auto actual = output.getValues<llvm::APInt>();
      for (int64_t row = 0; row < extent; ++row)
        for (int64_t column = 0; column < 512; ++column) {
          int64_t offset =
              transpose ? column * extent + row : row * 512 + column;
          ASSERT_EQ((*(actual.begin() + offset)).getBoolValue(),
                    (row * 7 + column * 3) % 251 > column % 197)
              << row << ", " << column;
        }
    }
  }
}

TEST_F(ConstantTensorFoldingTest,
       UsedInitAndUnsignedAddRetainIntegerSemantics) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << "#id = affine_map<(b,m,n)->(b,m,n)>\n"
           << "module { func.func @constant() -> tensor<1x" << extent
           << "x4xi8> {\n"
           << "%a = arith.constant dense<120> : tensor<1x" << extent
           << "x4xi8>\n"
           << "%init = arith.constant dense<125> : tensor<1x" << extent
           << "x4xi8>\n"
           << "%r = linalg.generic {indexing_maps=[#id,#id], iterator_types="
              "[\"parallel\",\"parallel\",\"parallel\"]} "
           << "ins(%a : tensor<1x" << extent << "x4xi8>) "
           << "outs(%init : tensor<1x" << extent << "x4xi8>) {\n"
           << "^bb0(%x:i8,%y:i8): %sum=arith.addi %x,%y:i8 "
              "linalg.yield %sum:i8 } -> tensor<1x"
           << extent << "x4xi8>\nreturn %r : tensor<1x" << extent
           << "x4xi8> }}";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(fold(*module));
    auto result = constantResult(*module);
    ASSERT_TRUE(result);
    ASSERT_TRUE(result.isSplat());
    EXPECT_EQ(result.getSplatValue<llvm::APInt>(), llvm::APInt(8, 245));
  }
}

TEST_F(ConstantTensorFoldingTest, SliceAndReshapePreserveFloatingPointBits) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (bool bfloat : {false, true}) {
      for (bool splat : {false, true}) {
        SCOPED_TRACE(extent);
        SCOPED_TRACE(bfloat);
        SCOPED_TRACE(splat);
        auto type = bfloat ? mlir::FloatType(builder.getBF16Type())
                           : mlir::FloatType(builder.getF16Type());
        // A large logical splat and a nonuniform payload use the same window.
        int64_t rows = splat ? 1048576 : 4096;
        auto sourceType = mlir::RankedTensorType::get({1, rows, 8}, type);
        auto outputType = mlir::RankedTensorType::get({1, extent, 2}, type);
        auto module = makeModule(outputType);
        llvm::SmallVector<llvm::APFloat> values;
        llvm::SmallVector<uint16_t> bits =
            bfloat ? llvm::SmallVector<uint16_t>{0x8000, 0x7f80, 0xff80, 0x7fc1,
                                                 0x3f80}
                   : llvm::SmallVector<uint16_t>{0x8000, 0x7c00, 0xfc00, 0x7e01,
                                                 0x3c00};
        for (int64_t i = 0, e = splat ? 1 : sourceType.getNumElements(); i < e;
             ++i)
          values.emplace_back(type.getFloatSemantics(),
                              llvm::APInt(16, bits[i % bits.size()]));
        auto input = constant(mlir::DenseElementsAttr::get(sourceType, values));
        auto indices = [&](llvm::ArrayRef<int64_t> integers) {
          llvm::SmallVector<mlir::OpFoldResult> attributes;
          for (int64_t integer : integers)
            attributes.push_back(builder.getIndexAttr(integer));
          return attributes;
        };
        auto slice = builder.create<mlir::tensor::ExtractSliceOp>(
            builder.getUnknownLoc(), outputType, input, indices({0, 7, 1}),
            indices({1, extent, 2}), indices({1, 3, 2}));
        auto collapsedType = mlir::RankedTensorType::get({1, extent * 2}, type);
        auto collapsed = builder.create<mlir::tensor::CollapseShapeOp>(
            builder.getUnknownLoc(), collapsedType, slice.getResult(),
            llvm::ArrayRef<mlir::ReassociationIndices>{{0}, {1, 2}});
        auto expanded = builder.create<mlir::tensor::ExpandShapeOp>(
            builder.getUnknownLoc(), outputType, collapsed.getResult(),
            llvm::ArrayRef<mlir::ReassociationIndices>{{0}, {1, 2}});
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                             expanded.getResult());
        ASSERT_TRUE(fold(*module));
        auto output = constantResult(*module);
        ASSERT_TRUE(output);
        ASSERT_EQ(output.getType(), outputType);
        auto actual = output.getValues<llvm::APFloat>();
        for (int64_t row = 0; row < extent; ++row)
          for (int64_t column = 0; column < 2; ++column) {
            int64_t original = (7 + row * 3) * 8 + 1 + column * 2;
            EXPECT_EQ(
                (*(actual.begin() + row * 2 + column)).bitcastToAPInt(),
                llvm::APInt(16, bits[splat ? 0 : original % bits.size()]));
          }
      }
    }
  }
}

TEST_F(ConstantTensorFoldingTest, EmptyWindowsProduceEmptyConstants) {
  // A zero extent is the semantic boundary under test. The source still has
  // a real-sized leading dimension, and no nonexistent element may be read.
  for (int64_t extent : {1024, 1025, 1031}) {
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << "module { func.func @constant() -> tensor<1x0x4xf16> {\n"
           << "%a = arith.constant dense<-0.0> : tensor<1x" << extent
           << "x4xf16>\n"
           << "%r = tensor.extract_slice %a[0, " << extent
           << ", 0] [1, 0, 4] [1, 1, 1] : tensor<1x" << extent
           << "x4xf16> to tensor<1x0x4xf16>\n"
           << "return %r : tensor<1x0x4xf16> }}";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(fold(*module));
    auto output = constantResult(*module);
    ASSERT_TRUE(output);
    EXPECT_EQ(output.getNumElements(), 0);
    EXPECT_EQ(output.getType().getElementType(), builder.getF16Type());
  }
}

TEST_F(ConstantTensorFoldingTest, UnknownInputAndScalarBodyRemainActualIR) {
  for (bool unknownInput : {false, true}) {
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << "#id = affine_map<(b,m,n)->(b,m,n)>\n"
           << "module { func.func @constant(%arg: tensor<1x1025x4xi32>) "
              "-> tensor<1x1025x4xi32> {\n"
           << "%a = arith.constant dense<3> : tensor<1x1025x4xi32>\n"
           << "%init = tensor.empty() : tensor<1x1025x4xi32>\n"
           << "%r = linalg.generic {indexing_maps=[#id,#id], iterator_types="
              "[\"parallel\",\"parallel\",\"parallel\"]} ins("
           << (unknownInput ? "%arg" : "%a")
           << " : tensor<1x1025x4xi32>) outs(%init : tensor<1x1025x4xi32>) {\n"
           << "^bb0(%x:i32,%unused:i32): %v = "
           << (unknownInput ? "arith.addi" : "arith.muli")
           << " %x, %x : i32 linalg.yield %v : i32 "
              "} -> tensor<1x1025x4xi32>\n"
           << "return %r : tensor<1x1025x4xi32> }}";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(fold(*module));
    auto generic = returned(*module).getDefiningOp<mlir::linalg::GenericOp>();
    ASSERT_TRUE(generic);
    EXPECT_EQ(generic.getNumDpsInputs(), 1);
    auto &body = generic.getBody()->front();
    EXPECT_EQ(mlir::isa<mlir::arith::AddIOp>(body), unknownInput);
    EXPECT_EQ(mlir::isa<mlir::arith::MulIOp>(body), !unknownInput);
  }
}

TEST_F(ConstantTensorFoldingTest,
       UnsupportedAndBudgetedGenericsRemainActualIR) {
  const char *source = R"mlir(
    #id = affine_map<(b,m,n)->(b,m,n)>
    #diagonal = affine_map<(b,m,n)->(b,m,n,n)>
    module {
      func.func @constant() -> tensor<1x1024x2x2xi32> {
        %a = arith.constant dense<3> : tensor<1x1024x2xi32>
        %init = arith.constant dense<0> : tensor<1x1024x2x2xi32>
        %r = linalg.generic {indexing_maps=[#id,#diagonal], iterator_types=["parallel","parallel","parallel"]}
          ins(%a:tensor<1x1024x2xi32>) outs(%init:tensor<1x1024x2x2xi32>) {
            ^bb0(%x:i32,%y:i32): %sum=arith.addi %x,%x:i32 linalg.yield %sum:i32
        } -> tensor<1x1024x2x2xi32>
        return %r:tensor<1x1024x2x2xi32>
      }
    })mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(fold(*module));
  EXPECT_TRUE(returned(*module).getDefiningOp<mlir::linalg::GenericOp>());

  for (unsigned width : {32, 256}) {
    for (int64_t extent : {1024, 1025}) {
      auto type = mlir::RankedTensorType::get({1, extent, 1024},
                                              builder.getIntegerType(width));
      auto large = makeModule(type);
      auto input =
          constant(mlir::DenseElementsAttr::get(type, llvm::APInt(width, 1)));
      auto init = builder.create<mlir::tensor::EmptyOp>(
          builder.getUnknownLoc(), type.getShape(), type.getElementType());
      auto identity = builder.getMultiDimIdentityMap(3);
      auto generic = builder.create<mlir::linalg::GenericOp>(
          builder.getUnknownLoc(), mlir::TypeRange{type},
          mlir::ValueRange{input}, mlir::ValueRange{init},
          llvm::ArrayRef<mlir::AffineMap>{identity, identity},
          llvm::SmallVector<mlir::utils::IteratorType>(
              3, mlir::utils::IteratorType::parallel),
          [&](mlir::OpBuilder &nested, mlir::Location loc,
              mlir::ValueRange args) {
            mlir::Value value = args[0];
            // Within element/byte bounds, five scalar ops exceed the work
            // bound.
            for (int i = 0; i < 5; ++i)
              value = nested.create<mlir::arith::AddIOp>(loc, value, args[0]);
            nested.create<mlir::linalg::YieldOp>(loc, value);
          });
      builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                           generic.getResults());
      ASSERT_TRUE(fold(*large));
      EXPECT_TRUE(returned(*large).getDefiningOp<mlir::linalg::GenericOp>());
    }
  }
}
} // namespace
