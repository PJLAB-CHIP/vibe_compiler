//===- StaticIndexRangeTest.cpp - Current SSA integer address bounds ------===//

#include "Wafer/Analysis/Instr/StaticIndexRange.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <functional>
#include <string>

namespace {
using namespace wafer::memory_planning::detail;

class StaticIndexRangeTest : public ::testing::Test {
protected:
  StaticIndexRangeTest() {
    context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                        mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  }

  static std::string print(mlir::ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    module.print(stream);
    return text;
  }

  static mlir::Value returned(mlir::ModuleOp module) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    return mlir::cast<mlir::func::ReturnOp>(function.front().getTerminator())
        .getOperand(0);
  }

  mlir::MLIRContext context;
};

TEST_F(StaticIndexRangeTest, ProvesClampedRuntimeLoadsWithoutReadingContents) {
  for (int64_t rows : {1024, 1025, 1031})
    for (unsigned width : {32u, 64u}) {
      SCOPED_TRACE(llvm::formatv("{0}:i{1}", rows, width).str());
      auto text =
          llvm::formatv(R"mlir(
        module {{
          func.func @read(%indices: memref<1x2x{0}xi64>) -> index {{
            %zero = arith.constant 0 : index
            %raw = memref.load %indices[%zero, %zero, %zero]
                : memref<1x2x{0}xi64>
            {2}
            %lo = arith.constant 0 : i{1}
            %hi = arith.constant {3} : i{1}
            %lower = arith.maxsi %{4}, %lo : i{1}
            %clamp = arith.minsi %lower, %hi : i{1}
            %offset = arith.index_cast %clamp : i{1} to index
            return %offset : index
          }
        })mlir",
                        rows, width,
                        width == 32 ? "%narrow = arith.trunci %raw : i64 to i32"
                                    : "",
                        rows - 1, width == 32 ? "narrow" : "raw")
              .str();
      auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      ASSERT_TRUE(module);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      auto before = print(*module);
      auto result = evaluateNonNegativeStaticIndexRange(returned(*module));
      ASSERT_TRUE(result.succeeded());
      EXPECT_FALSE(result.range.empty);
      EXPECT_EQ(result.range.min, 0);
      EXPECT_EQ(result.range.max, rows - 1);
      EXPECT_EQ(print(*module), before);
    }
}

TEST_F(StaticIndexRangeTest, ContainsEveryBitPatternAcrossIntegerSemantics) {
  // Scalar i8 is a bounded exhaustive oracle for integer abstract transfer
  // functions. The same mechanism feeds rank3/1024+ target tests.
  struct Case {
    const char *body;
    std::function<int64_t(llvm::APInt)> expected;
  };
  const Case cases[] = {
      {"%lo = arith.constant 0 : i8\n"
       "%hi = arith.constant 63 : i8\n"
       "%lower = arith.maxsi %x, %lo : i8\n"
       "%v = arith.minsi %lower, %hi : i8\n"
       "%r = arith.index_cast %v : i8 to index",
       [](llvm::APInt x) {
         return std::clamp(x.getSExtValue(), int64_t{0}, int64_t{63});
       }},
      {"%hi = arith.constant 127 : i8\n"
       "%v = arith.minui %x, %hi : i8\n"
       "%r = arith.index_castui %v : i8 to index",
       [](llvm::APInt x) { return std::min(x.getZExtValue(), uint64_t{127}); }},
      {"%v = arith.extui %x : i8 to i32\n"
       "%r = arith.index_cast %v : i32 to index",
       [](llvm::APInt x) { return x.getZExtValue(); }},
      {"%wide = arith.extsi %x : i8 to i32\n"
       "%lo = arith.constant 0 : i32\n"
       "%v = arith.maxsi %wide, %lo : i32\n"
       "%r = arith.index_cast %v : i32 to index",
       [](llvm::APInt x) { return std::max(x.getSExtValue(), int64_t{0}); }},
      {"%bias = arith.constant 120 : i8\n"
       "%sum = arith.addi %x, %bias : i8\n"
       "%mask = arith.constant 31 : i8\n"
       "%v = arith.andi %sum, %mask : i8\n"
       "%r = arith.index_cast %v : i8 to index",
       [](llvm::APInt x) {
         return ((x + llvm::APInt(8, 120)) & llvm::APInt(8, 31)).getSExtValue();
       }},
      {"%wide = arith.extui %x : i8 to i16\n"
       "%factor = arith.constant 17 : i16\n"
       "%product = arith.muli %wide, %factor : i16\n"
       "%narrow = arith.trunci %product : i16 to i8\n"
       "%lo = arith.constant 0 : i8\n"
       "%hi = arith.constant 63 : i8\n"
       "%lower = arith.maxsi %narrow, %lo : i8\n"
       "%v = arith.minsi %lower, %hi : i8\n"
       "%r = arith.index_cast %v : i8 to index",
       [](llvm::APInt x) {
         return std::clamp(
             (x.zext(16) * llvm::APInt(16, 17)).trunc(8).getSExtValue(),
             int64_t{0}, int64_t{63});
       }},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.body);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::formatv("module {{ func.func @oracle(%x: i8) -> index {{ {0}\n"
                      "return %r : index } }",
                      test.body)
            .str(),
        &context);
    ASSERT_TRUE(module);
    auto result = evaluateNonNegativeStaticIndexRange(returned(*module));
    ASSERT_TRUE(result.succeeded());
    int64_t minimum = INT64_MAX;
    int64_t maximum = INT64_MIN;
    for (unsigned bits = 0; bits < 256; ++bits) {
      int64_t actual = test.expected(llvm::APInt(8, bits));
      EXPECT_LE(result.range.min, actual);
      EXPECT_GE(result.range.max, actual);
      minimum = std::min(minimum, actual);
      maximum = std::max(maximum, actual);
    }
    EXPECT_EQ(result.range.min, minimum);
    EXPECT_EQ(result.range.max, maximum);
  }
}

TEST_F(StaticIndexRangeTest, RejectsNegativeUnknownAndWrappedIndices) {
  struct Case {
    const char *body;
    StaticIndexRangeFailureKind failure;
  };
  const Case cases[] = {
      {"%r = arith.index_cast %x : i32 to index",
       StaticIndexRangeFailureKind::NegativeRange},
      {"%hi = arith.constant 1030 : i32\n"
       "%v = arith.minsi %x, %hi : i32\n"
       "%r = arith.index_cast %v : i32 to index",
       StaticIndexRangeFailureKind::NegativeRange},
      {"%lo = arith.constant 0 : i32\n"
       "%hi = arith.constant 255 : i32\n"
       "%lower = arith.maxsi %x, %lo : i32\n"
       "%v = arith.minsi %lower, %hi : i32\n"
       "%narrow = arith.trunci %v : i32 to i8\n"
       "%r = arith.index_cast %narrow : i8 to index",
       StaticIndexRangeFailureKind::NegativeRange},
      {"%lo = arith.constant 0 : i32\n"
       "%hi = arith.constant 1030 : i32\n"
       "%bias = arith.constant 2147483647 : i32\n"
       "%lower = arith.maxsi %x, %lo : i32\n"
       "%v = arith.minsi %lower, %hi : i32\n"
       "%sum = arith.addi %v, %bias : i32\n"
       "%r = arith.index_cast %sum : i32 to index",
       StaticIndexRangeFailureKind::NegativeRange},
      {"%r = arith.addi %unknown, %unknown : index",
       StaticIndexRangeFailureKind::UnsupportedExpression},
      {"%hi = arith.constant 9223372036854775807 : index\n"
       "%one = arith.constant 1 : index\n"
       "%r = arith.addi %hi, %one : index",
       StaticIndexRangeFailureKind::ArithmeticOverflow},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.body);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::formatv("module {{ func.func @negative(%x: i32, %unknown: index) "
                      "-> index {{ {0}\n"
                      "return %r : index } }",
                      test.body)
            .str(),
        &context);
    ASSERT_TRUE(module);
    auto before = print(*module);
    EXPECT_EQ(evaluateNonNegativeStaticIndexRange(returned(*module)).failure,
              test.failure);
    EXPECT_EQ(print(*module), before);
  }
}

TEST_F(StaticIndexRangeTest, HandlesWideIntegersAndFreshMutation) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @wide(%x: i128) -> index {
        %lo = arith.constant 0 : i128
        %hi = arith.constant 1030 : i128
        %lower = arith.maxsi %x, %lo : i128
        %clamp = arith.minsi %lower, %hi : i128
        %r = arith.index_cast %clamp : i128 to index
        return %r : index
      }
    })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  auto result = evaluateNonNegativeStaticIndexRange(returned(*module));
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.range.min, 0);
  EXPECT_EQ(result.range.max, 1030);
  mlir::arith::MinSIOp clamp;
  module->walk([&](mlir::arith::MinSIOp op) { clamp = op; });
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  clamp.getLhsMutable().assign(function.getArgument(0));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(evaluateNonNegativeStaticIndexRange(returned(*module)).failure,
            StaticIndexRangeFailureKind::NegativeRange);
}

TEST_F(StaticIndexRangeTest, BoundsDeepSharedDagWithoutRecursiveExpansion) {
  // The scalar DAG is a work-complexity oracle: 4096 shared nodes represent
  // exponentially many paths. Production rank3 consumers are covered below.
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  mlir::OwningOpRef<mlir::ModuleOp> owner(module);
  auto function = mlir::func::FuncOp::create(
      builder.getUnknownLoc(), "shared",
      builder.getFunctionType({builder.getI32Type()},
                              {builder.getIndexType()}));
  module.push_back(function);
  builder.setInsertionPointToStart(function.addEntryBlock());
  mlir::Value value = function.getArgument(0);
  for (unsigned i = 0; i < 4096; ++i)
    value =
        builder.create<mlir::arith::AddIOp>(function.getLoc(), value, value);
  auto lo =
      builder.create<mlir::arith::ConstantIntOp>(function.getLoc(), 0, 32);
  auto hi =
      builder.create<mlir::arith::ConstantIntOp>(function.getLoc(), 1030, 32);
  value = builder.create<mlir::arith::MaxSIOp>(function.getLoc(), value, lo);
  value = builder.create<mlir::arith::MinSIOp>(function.getLoc(), value, hi);
  value = builder.create<mlir::arith::IndexCastOp>(
      function.getLoc(), builder.getIndexType(), value);
  builder.create<mlir::func::ReturnOp>(function.getLoc(), value);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
  auto result = evaluateNonNegativeStaticIndexRange(value);
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.range.min, 0);
  EXPECT_EQ(result.range.max, 1030);
}

TEST_F(StaticIndexRangeTest,
       UnsignedBranchesDoNotProveSignedAddressesPositive) {
  for (llvm::StringRef predicate : {"ugt", "ult", "sgt"}) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::formatv(R"mlir(module {{
          func.func @branch(%x: i32) {{
            %offset = arith.index_cast %x : i32 to index
            %zero = arith.constant 0 : index
            %limit = arith.constant {1} : index
            %condition = arith.cmpi {0}, %offset, %limit : index
            scf.if %condition {{
              %use = arith.addi %offset, %zero : index
            }
            return
          }
        })mlir",
                      predicate, predicate == "ult" ? -1 : 0)
            .str(),
        &context);
    ASSERT_TRUE(module);
    mlir::arith::AddIOp use;
    module->walk([&](mlir::arith::AddIOp op) { use = op; });
    ASSERT_TRUE(use);
    auto result = evaluateNonNegativeStaticIndexRange(use.getLhs(), use);
    if (predicate == "sgt") {
      ASSERT_TRUE(result.succeeded());
      EXPECT_EQ(result.range.min, 1);
      EXPECT_EQ(result.range.max, INT32_MAX);
    } else {
      EXPECT_EQ(result.failure, StaticIndexRangeFailureKind::NegativeRange);
    }
  }
}
} // namespace
