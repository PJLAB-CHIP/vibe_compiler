//===- StaticIndexRangeTest.cpp - Current SSA integer address bounds ------===//

#include "Wafer/Analysis/Instr/StaticIndexRange.h"
#include "Wafer/Analysis/ControlFlow/IndexValueBounds.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <functional>
#include <string>

namespace {
using namespace wafer::memory_planning::detail;

class StaticIndexRangeTest : public ::testing::Test {
protected:
  StaticIndexRangeTest() {
    mlir::DialectRegistry registry;
    mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
    wafer::analysis::registerIndexValueBoundsModels(registry);
    context.appendDialectRegistry(registry);
    context.loadDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                        mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                        mlir::scf::SCFDialect>();
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

TEST_F(StaticIndexRangeTest, BranchConstraintsFollowEquivalentAffineValues) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    func.func @bounds() {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1031 = arith.constant 1031 : index
      scf.for %i = %c0 to %c1031 step %c1 {
        %lower = affine.apply affine_map<(d0) -> (d0 floordiv 2 - 256)>(%i)
        %upper = affine.apply affine_map<()[s0] -> (300 - s0 floordiv 2)>()[%i]
        %lo = arith.cmpi sge, %lower, %c0 : index
        %hi = arith.cmpi sge, %upper, %c0 : index
        %both = arith.andi %lo, %hi : i1
        scf.if %both {
          %source = affine.apply affine_map<()[s0] -> (s0 floordiv 2 - 256)>()[%i]
          %translated = affine.apply affine_map<(d0) -> (d0 floordiv 2 - 200)>(%i)
          %unrelated = affine.apply affine_map<(d0) -> (d0 mod 7 - 4)>(%i)
        }
        %outside = affine.apply affine_map<(d0) -> (d0 floordiv 2 - 256)>(%i)
      }
      return
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::affine::AffineApplyOp> values;
  module->walk([&](mlir::affine::AffineApplyOp op) { values.push_back(op); });
  ASSERT_EQ(values.size(), 6u);
  auto source = evaluateNonNegativeStaticIndexRange(values[2], values[2]);
  ASSERT_TRUE(source.succeeded());
  EXPECT_EQ(source.range.min, 0);
  EXPECT_EQ(source.range.max, 44);
  auto translated = evaluateNonNegativeStaticIndexRange(values[3], values[3]);
  ASSERT_TRUE(translated.succeeded());
  EXPECT_EQ(translated.range.min, 56);
  EXPECT_EQ(translated.range.max, 100);
  EXPECT_EQ(evaluateNonNegativeStaticIndexRange(values[4], values[4]).failure,
            StaticIndexRangeFailureKind::NegativeRange);
  EXPECT_EQ(evaluateNonNegativeStaticIndexRange(values[5], values[5]).failure,
            StaticIndexRangeFailureKind::NegativeRange);
}

TEST_F(StaticIndexRangeTest, StaticExecutionRequiresAnIndependentPathWitness) {
  for (int64_t extent : {1024, 1025, 1031, 1000000000}) {
    SCOPED_TRACE(extent);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(llvm::formatv(R"mlir(
    func.func @paths(%data: memref<2x{0}x64xf16>, %unknown: i1) {{
      %c0 = arith.constant 0 : index
      %c5 = arith.constant 5 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %end = arith.constant {0} : index
      scf.for %i = %c0 to %end step %c128 {{
        %earlier = arith.cmpi slt, %i, %c256 : index
        scf.if %earlier {{
          %a = memref.load %data[%c0, %i, %c0] {{test.executes = true} : memref<2x{0}x64xf16>
        } else {{
          %b = memref.load %data[%c0, %i, %c0] {{test.executes = true} : memref<2x{0}x64xf16>
        }
        %offGrid = arith.cmpi eq, %i, %c5 : index
        scf.if %offGrid {{
          %c = memref.load %data[%c0, %i, %c0] {{test.executes = false} : memref<2x{0}x64xf16>
        }
        scf.if %unknown {{
          %d = memref.load %data[%c0, %i, %c0] {{test.executes = false} : memref<2x{0}x64xf16>
        }
        scf.for %j = %c0 to %c256 step %c128 {{
          %sum = arith.addi %i, %j : index
          %coupled = arith.cmpi eq, %sum, %c128 : index
          scf.if %coupled {{
            // Independent interval minima are not a joint satisfying tuple.
            // Failing to find a witness must stay unknown, not infer execution.
            %e = memref.load %data[%c0, %i, %c0] {{test.executes = false} : memref<2x{0}x64xf16>
          }
        }
      }
      scf.for %i = %c0 to %c0 step %c128 {{
        %f = memref.load %data[%c0, %i, %c0] {{test.executes = false} : memref<2x{0}x64xf16>
      }
      return
    }
    )mlir",
                                                                        extent)
                                                              .str(),
                                                          &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    std::string before = print(*module);
    unsigned checked = 0;
    module->walk([&](mlir::memref::LoadOp read) {
      auto expected = read->getAttrOfType<mlir::BoolAttr>("test.executes");
      ASSERT_TRUE(expected);
      EXPECT_EQ(proveStaticIndexExecution(
                    read, read->getParentOfType<mlir::func::FuncOp>()),
                expected.getValue() ? StaticIndexExecution::Proven
                                    : StaticIndexExecution::Unknown);
      ++checked;
    });
    EXPECT_EQ(checked, 6u);
    EXPECT_EQ(print(*module), before);
  }
}

TEST_F(StaticIndexRangeTest, BooleanGuardsProveOnlyReachableAddressBounds) {
  // Scalar predicate oracle paired with rank3 subset-to-LLVM coverage.
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    func.func @bounds(%unknown: i1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1031 = arith.constant 1031 : index
      %true = arith.constant true
      scf.for %i = %c0 to %c1031 step %c1 {
        %remainder = affine.apply affine_map<(d0) -> (d0 - (d0 floordiv 7) * 7)>(%i)
        %lo = arith.cmpi sge, %remainder, %c0 : index
        %upper = affine.apply affine_map<(d0) -> (6 - d0 + (d0 floordiv 7) * 7)>(%i)
        %hi = arith.cmpi sge, %upper, %c0 : index
        %always = arith.andi %lo, %hi : i1
        scf.if %always {} else {
          %unreachable = affine.apply affine_map<(d0) -> (d0 - 9999)>(%i)
        }
        %bound = affine.apply affine_map<(d0) -> (d0 - 512)>(%i)
        %test = arith.cmpi sge, %bound, %c0 : index
        %both = arith.andi %always, %test : i1
        %negated = arith.xori %both, %true : i1
        scf.if %negated {
          %bounded = affine.apply affine_map<(d0) -> (d0 + 512)>(%i)
        }
        %ambiguous = arith.andi %unknown, %test : i1
        scf.if %ambiguous {} else {
          %unbounded = affine.apply affine_map<(d0) -> (d0 + 512)>(%i)
        }
      }
      return
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::affine::AffineApplyOp> values;
  module->walk([&](mlir::affine::AffineApplyOp op) { values.push_back(op); });
  ASSERT_EQ(values.size(), 6u);
  auto unreachable = evaluateNonNegativeStaticIndexRange(values[2], values[2]);
  ASSERT_TRUE(unreachable.succeeded());
  EXPECT_TRUE(unreachable.range.empty);
  auto bounded = evaluateNonNegativeStaticIndexRange(values[4], values[4]);
  ASSERT_TRUE(bounded.succeeded());
  EXPECT_FALSE(bounded.range.empty);
  EXPECT_EQ(bounded.range.min, 512);
  EXPECT_EQ(bounded.range.max, 1023);
  auto unknown = evaluateNonNegativeStaticIndexRange(values[5], values[5]);
  ASSERT_TRUE(unknown.succeeded());
  EXPECT_FALSE(unknown.range.empty);
  EXPECT_EQ(unknown.range.max, 1542);
}

// Scalar interface oracle. These bounds also feed the real attention loop
// and physical preparation placement tests.
TEST_F(StaticIndexRangeTest, SignedExtremaProvideOnlyProvenBounds) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(module {
    func.func @entry(%unknown: index) -> (index, index) {
      %negative = arith.constant -7 : index
      %positive = arith.constant 13 : index
      %minimum = arith.minsi %unknown, %negative : index
      %maximum = arith.maxsi %unknown, %positive : index
      return %minimum, %maximum : index, index
    }
  })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto results = function.front().getTerminator()->getOperands();
  using Bounds = mlir::ValueBoundsConstraintSet;
  using BoundType = mlir::presburger::BoundType;
  auto upper = Bounds::computeConstantBound(
      BoundType::UB, Bounds::Variable(results[0]), nullptr, /*closedUB=*/true);
  auto lower =
      Bounds::computeConstantBound(BoundType::LB, Bounds::Variable(results[1]));
  ASSERT_TRUE(mlir::succeeded(upper));
  ASSERT_TRUE(mlir::succeeded(lower));
  EXPECT_EQ(*upper, -7);
  EXPECT_EQ(*lower, 13);
  EXPECT_TRUE(mlir::failed(Bounds::computeConstantBound(
      BoundType::LB, Bounds::Variable(results[0]))));
  EXPECT_TRUE(mlir::failed(Bounds::computeConstantBound(
      BoundType::UB, Bounds::Variable(results[1]))));
}

// Scalar interface oracle; the real causal split below consumes these bounds
// for rank-four payloads and passes them through decomposition/Instr lowering.
TEST_F(StaticIndexRangeTest,
       SignedCeilDivPreservesConstantPositiveDivisorBounds) {
  for (int64_t extent : {1024, 1025, 1031}) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(llvm::formatv(R"mlir(
      module {{
        func.func @entry(%unknown: index, %dynamic: index) -> (index, index, index, index) {{
          %positive = arith.constant {0} : index
          %negative = arith.constant -{0} : index
          %divisor = arith.constant 128 : index
          %minus = arith.constant -128 : index
          %lo = arith.maxsi %unknown, %positive : index
          %hi = arith.minsi %unknown, %negative : index
          %lower = arith.ceildivsi %lo, %divisor : index
          %upper = arith.ceildivsi %hi, %divisor : index
          %unsupported_sign = arith.ceildivsi %lo, %minus : index
          %unsupported_dynamic = arith.ceildivsi %lo, %dynamic : index
          return %lower, %upper, %unsupported_sign, %unsupported_dynamic : index, index, index, index
        }
      }
    )mlir",
                                                                        extent)
                                                              .str(),
                                                          &context);
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    auto results = function.front().getTerminator()->getOperands();
    using Bounds = mlir::ValueBoundsConstraintSet;
    using BoundType = mlir::presburger::BoundType;
    auto lower = Bounds::computeConstantBound(BoundType::LB,
                                              Bounds::Variable(results[0]));
    auto upper = Bounds::computeConstantBound(
        BoundType::UB, Bounds::Variable(results[1]), nullptr, true);
    ASSERT_TRUE(mlir::succeeded(lower));
    ASSERT_TRUE(mlir::succeeded(upper));
    EXPECT_EQ(*lower, (extent + 127) / 128);
    EXPECT_EQ(*upper, -(extent / 128));
    for (auto result : results.drop_front(2))
      EXPECT_TRUE(mlir::failed(Bounds::computeConstantBound(
          BoundType::LB, Bounds::Variable(result))));
  }
}

// Scalar range oracle; actual tensor loops are covered by attention and NCC
// lifetime tests. Bounded upper limits must not imply a constant trip count.
TEST_F(StaticIndexRangeTest, BoundedLoopUpperLimitsRetainExactReachableGrid) {
  for (int64_t extent : {1024, 1025, 1031}) {
    auto module =
        mlir::parseSourceString<mlir::ModuleOp>(llvm::formatv(R"mlir(module {{
          func.func @entry(%unknown: index) {{
            %zero = arith.constant 0 : index
            %end = arith.constant {0} : index
            %query_step = arith.constant 192 : index
            %key_step = arith.constant 128 : index
            scf.for %query = %zero to %end step %query_step {{
              %query_end = arith.addi %query, %query_step : index
              %upper = arith.minsi %end, %query_end : index
              scf.for %key = %zero to %upper step %key_step {{
                %pattern = arith.divui %key, %key_step : index
              }
              scf.for %maybe_empty = %zero to %query step %key_step {{ }
              scf.for %unbounded = %zero to %unknown step %key_step {{ }
            }
            return
          }
        })mlir",
                                                              extent)
                                                    .str(),
                                                &context);
    ASSERT_TRUE(module);
    unsigned inner = 0;
    module->walk([&](mlir::scf::ForOp loop) {
      if (!loop->getParentOfType<mlir::scf::ForOp>())
        return;
      auto position = inner++;
      EXPECT_EQ(proveNonEmptyLoop(loop), position == 0);
      auto range = evaluateNonNegativeStaticIndexRange(loop.getInductionVar());
      if (position == 2) {
        EXPECT_FALSE(range.succeeded());
        return;
      }
      ASSERT_TRUE(range.succeeded());
      EXPECT_FALSE(range.range.empty);
      EXPECT_EQ(range.range.min, 0);
      int64_t largest = -1;
      for (int64_t q = 0; q < extent; q += 192)
        for (int64_t k = 0; k < (position ? q : std::min(extent, q + 192));
             k += 128) {
          EXPECT_LE(k, range.range.max);
          largest = std::max(largest, k);
        }
      EXPECT_EQ(range.range.max, largest);
    });
    EXPECT_EQ(inner, 3u);
    module->walk([&](mlir::arith::DivUIOp divide) {
      auto range = evaluateNonNegativeStaticIndexRange(divide.getResult());
      ASSERT_TRUE(range.succeeded());
      EXPECT_EQ(range.range.min, 0);
      EXPECT_EQ(range.range.max, (extent - 1) / 128);
    });
  }
}

// Scalar interval oracle; rank-three DDR/target witnesses exercise the same
// lower-bound expressions in plan-ddr-memory-dynamic-loop.mlir.
TEST_F(StaticIndexRangeTest, BoundedDynamicLowerPreservesOnlyProvenGrid) {
  for (int64_t extent : {1024, 1025, 1031})
    for (int64_t step : {3, 128})
      for (bool aligned : {false, true}) {
        auto module = mlir::parseSourceString<mlir::ModuleOp>(
            llvm::formatv(R"mlir(
          module {{
            func.func @entry(%unknown: index) {{
              %zero = arith.constant 0 : index
              %base = arith.constant 13 : index
              %end = arith.constant {0} : index
              %query_step = arith.constant 192 : index
              %step = arith.constant {1} : index
              scf.for %q = %zero to %end step %query_step {{
                %rounded = arith.ceildivsi %q, %step : index
                %grid = arith.muli %rounded, %step : index
                %lower = arith.addi {2}, %base : index
                scf.for %key = %lower to %end step %step {{ }
                scf.for %unbounded = %unknown to %end step %step {{ }
              }
              return
            }
          }
        )mlir",
                          extent, step, aligned ? "%grid" : "%q")
                .str(),
            &context);
        ASSERT_TRUE(module);
        unsigned inner = 0;
        module->walk([&](mlir::scf::ForOp loop) {
          if (!loop->getParentOfType<mlir::scf::ForOp>())
            return;
          auto range =
              evaluateNonNegativeStaticIndexRange(loop.getInductionVar());
          if (inner++) {
            EXPECT_EQ(range.failure,
                      StaticIndexRangeFailureKind::DynamicLoopBounds);
            return;
          }
          ASSERT_TRUE(range.succeeded());
          EXPECT_FALSE(range.range.empty);
          EXPECT_EQ(range.range.min, 13);
          if (aligned && step == 128)
            EXPECT_EQ(range.range.max, 13 + (extent - 14) / step * step);
          else
            EXPECT_EQ(range.range.max, extent - 1);
          for (int64_t q = 0; q < extent; q += 192) {
            int64_t lower = 13 + (aligned ? (q + step - 1) / step * step : q);
            for (int64_t k = lower; k < extent; k += step) {
              EXPECT_GE(k, range.range.min);
              EXPECT_LE(k, range.range.max);
            }
          }
        });
        EXPECT_EQ(inner, 2u);
      }
}

TEST_F(StaticIndexRangeTest, PackedAlignmentUsesEveryInductionValue) {
  for (int64_t step : {1, 8}) {
    auto text = llvm::formatv(R"mlir(module {{
      func.func @check(%unknown: index, %source: memref<1x1x1040xi1>) {{
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %end = arith.constant 17 : index
        %step = arith.constant {0} : index
        %aligned = arith.muli %unknown, %c8 : index
        scf.for %i = %c0 to %end step %step {{
          %twice = arith.muli %i, %c2 : index
          %view = memref.subview %source[0, 0, %i] [1, 1, 1024] [1, 1, 1]
            : memref<1x1x1040xi1> to memref<1x1x1024xi1, strided<[1040, 1040, 1], offset: ?>>
        }
        return
      }
    })mlir",
                              step)
                    .str();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    mlir::scf::ForOp loop;
    module->walk([&](mlir::scf::ForOp op) { loop = op; });
    ASSERT_TRUE(loop);
    auto bounds = evaluateNonNegativeStaticIndexRange(loop.getInductionVar());
    ASSERT_TRUE(bounds.succeeded());
    // Both endpoints are aligned even for step=1; this is insufficient.
    EXPECT_EQ(bounds.range.min, 0);
    EXPECT_EQ(bounds.range.max, 16);
    EXPECT_EQ(getKnownIndexRemainder(loop.getInductionVar(), 8),
              step == 8 ? std::optional<uint64_t>(0) : std::nullopt);
    module->walk([&](mlir::memref::SubViewOp view) {
      EXPECT_EQ(mlir::succeeded(proveByteAlignedPackedView(view)), step == 8);
    });
    module->walk([&](mlir::arith::MulIOp multiply) {
      if (!multiply->getParentOfType<mlir::scf::ForOp>()) {
        EXPECT_EQ(getKnownIndexRemainder(multiply, 8),
                  std::optional<uint64_t>(0));
      }
    });
  }
}

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

TEST_F(StaticIndexRangeTest, BoundedBranchOperandsPreserveInductionGrid) {
  for (int64_t extent : {1024, 1025, 1031})
    for (int64_t step : {3, 32, 128}) {
      SCOPED_TRACE(::testing::Message() << extent << '/' << step);
      auto module =
          mlir::parseSourceString<mlir::ModuleOp>(llvm::formatv(R"mlir(
        module {{ func.func @entry(%data: memref<1x{0}x64xf16>) {{
          %zero = arith.constant 0 : index
          %begin = arith.constant 7 : index
          %end = arith.constant {0} : index
          %step = arith.constant {1} : index
          %outer_step = arith.constant 128 : index
          scf.for %outer = %zero to %end step %outer_step {{
            %next_outer = arith.addi %outer, %outer_step : index
            %upper = arith.minsi %next_outer, %end : index
            %limit = arith.subi %upper, %step : index
            scf.for %iv = %begin to %upper step %step {{
              %has_next = arith.cmpi slt, %iv, %limit : index
              scf.if %has_next {{
                %next = arith.addi %iv, %step : index
                %value = memref.load %data[%zero, %next, %zero] : memref<1x{0}x64xf16>
              } else {{
                %last = memref.load %data[%zero, %iv, %zero] : memref<1x{0}x64xf16>
              }
            }
          }
          return
        } }
      )mlir",
                                                                extent, step)
                                                      .str(),
                                                  &context);
      ASSERT_TRUE(module);
      const auto before = print(*module);
      llvm::SmallVector<mlir::memref::LoadOp> loads;
      module->walk([&](mlir::memref::LoadOp op) { loads.push_back(op); });
      ASSERT_EQ(loads.size(), 2u);
      auto next = evaluateNonNegativeStaticIndexRange(loads[0].getIndices()[1],
                                                      loads[0]);
      auto last = evaluateNonNegativeStaticIndexRange(loads[1].getIndices()[1],
                                                      loads[1]);
      ASSERT_TRUE(next.succeeded());
      ASSERT_TRUE(last.succeeded());
      int64_t firstNext = extent, finalNext = -1;
      for (int64_t outer = 0; outer < extent; outer += 128)
        for (int64_t iv = 7, upper = std::min(outer + 128, extent); iv < upper;
             iv += step) {
          if (iv + step < upper) {
            firstNext = std::min(firstNext, iv + step);
            finalNext = std::max(finalNext, iv + step);
          } else {
            EXPECT_LE(last.range.min, iv);
            EXPECT_GE(last.range.max, iv);
          }
        }
      EXPECT_EQ(next.range.min, firstNext);
      EXPECT_EQ(next.range.max, finalNext);
      EXPECT_LT(next.range.max, extent);
      EXPECT_EQ(print(*module), before);
    }
}

TEST_F(StaticIndexRangeTest, BoundedEqualityIntersectsIntervalsNotEndpoints) {
  // Scalar oracle for predicate composition; the rank3 access test above and
  // load-pipeline DDR planning exercise the same query on actual payloads.
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @entry(%a: i32, %b: i32) {
      %zero = arith.constant 0 : i32
      %end = arith.constant 1031 : i32
      %one = arith.constant 1 : i32
      %al = arith.maxsi %a, %zero : i32
      %x = arith.minsi %al, %end : i32
      %bl = arith.maxsi %b, %one : i32
      %y = arith.minsi %bl, %end : i32
      %same = arith.cmpi eq, %x, %y : i32
      %positive = arith.cmpi sgt, %x, %zero : i32
      %both = arith.andi %same, %positive : i1
      scf.if %both {
        %use = arith.index_cast %x : i32 to index
      }
      return
    } }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::arith::IndexCastOp use;
  module->walk([&](mlir::arith::IndexCastOp op) { use = op; });
  auto range = evaluateNonNegativeStaticIndexRange(use.getIn(), use);
  ASSERT_TRUE(range.succeeded());
  EXPECT_FALSE(range.range.empty);
  EXPECT_EQ(range.range.min, 1);
  EXPECT_EQ(range.range.max, 1031);
}
TEST_F(StaticIndexRangeTest, NestedAlternativesRefineBareLoopInduction) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool lowered : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(lowered);
      std::string text = llvm::formatv(R"mlir(module {{
          func.func @entry(%buffer: memref<2x{0}x64xf16>) {{
            %zero = arith.constant 0 : index
            %step = arith.constant 128 : index
            %end = arith.constant {0} : index
            %true = arith.constant true
            scf.for %i = %zero to %end step %step {{
              %a = affine.apply affine_map<(d0)->(128-d0)>(%i)
              %first = arith.cmpi sge, %a, %zero : index
              scf.if %first {{
                %x = memref.load %buffer[%zero, %i, %zero] : memref<2x{0}x64xf16>
              } else {{
                %b = affine.apply affine_map<(d0)->(255-d0)>(%i)
                %before = arith.cmpi sge, %b, %zero : index
                %after = arith.xori %before, %true : i1
                %c = affine.apply affine_map<(d0)->(384-d0)>(%i)
                %limit = arith.cmpi sge, %c, %zero : index
                %second = arith.andi %after, %limit : i1
                scf.if %second {{
                  %y = memref.load %buffer[%zero, %i, %zero] : memref<2x{0}x64xf16>
                } else {{
                  %z = memref.load %buffer[%zero, %i, %zero] : memref<2x{0}x64xf16>
                }
              }
            }
            return
          }
        })mlir",
                                       extent)
                             .str();
      if (lowered) {
        auto replace = [&](llvm::StringRef from, llvm::StringRef to) {
          auto position = text.find(from.str());
          ASSERT_NE(position, std::string::npos);
          text.replace(position, from.size(), to.str());
        };
        replace("%a = affine.apply affine_map<(d0)->(128-d0)>(%i)",
                "%minus = arith.constant -1 : index\n"
                "%negative = arith.muli %i, %minus : index\n"
                "%a = arith.addi %negative, %step : index");
        replace("%b = affine.apply affine_map<(d0)->(255-d0)>(%i)",
                "%c255 = arith.constant 255 : index\n"
                "%b = arith.subi %c255, %i : index");
        replace("%c = affine.apply affine_map<(d0)->(384-d0)>(%i)",
                "%c384 = arith.constant 384 : index\n"
                "%c = arith.addi %negative, %c384 : index");
      }
      auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      ASSERT_TRUE(module);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      llvm::SmallVector<mlir::memref::LoadOp> loads;
      module->walk([&](mlir::memref::LoadOp load) { loads.push_back(load); });
      ASSERT_EQ(loads.size(), 3u);
      const std::array<std::pair<int64_t, int64_t>, 3> expected{
          {{0, 128}, {256, 384}, {512, (extent - 1) / 128 * 128}}};
      for (auto [load, bounds] : llvm::zip_equal(loads, expected)) {
        auto result =
            evaluateNonNegativeStaticIndexRange(load.getIndices()[1], load);
        ASSERT_TRUE(result.succeeded());
        EXPECT_FALSE(result.range.empty);
        EXPECT_EQ(result.range.min, bounds.first);
        EXPECT_EQ(result.range.max, bounds.second);
        EXPECT_EQ(proveStaticIndexExecution(
                      load, load->getParentOfType<mlir::func::FuncOp>()),
                  StaticIndexExecution::Proven);
      }
    }
}

TEST_F(StaticIndexRangeTest, BoundedProductsPreserveSignsAndRejectOverflow) {
  // Bounded scalar oracle for interval multiplication; the load-pipeline
  // matrix consumes count * runtime-positive-step on rank3 1024/1025/1031.
  for (auto bounds : {std::array<int64_t, 4>{0, 29, 32, 33},
                      std::array<int64_t, 4>{-7, -2, -5, -1},
                      std::array<int64_t, 4>{-3, 5, -2, 4},
                      std::array<int64_t, 4>{1, INT64_MAX, 1, 2}}) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::formatv(R"mlir(module {{
          func.func @entry(%a: i64, %b: i64) -> index {{
            %al = arith.constant {0} : i64
            %ah = arith.constant {1} : i64
            %bl = arith.constant {2} : i64
            %bh = arith.constant {3} : i64
            %ax = arith.maxsi %a, %al : i64
            %ac = arith.minsi %ax, %ah : i64
            %bx = arith.maxsi %b, %bl : i64
            %bc = arith.minsi %bx, %bh : i64
            %ai = arith.index_cast %ac : i64 to index
            %bi = arith.index_cast %bc : i64 to index
            %product = arith.muli %ai, %bi : index
            return %product : index
          }
        })mlir",
                      bounds[0], bounds[1], bounds[2], bounds[3])
            .str(),
        &context);
    ASSERT_TRUE(module);
    auto result = evaluateNonNegativeStaticIndexRange(returned(*module));
    if (bounds[1] == INT64_MAX) {
      EXPECT_EQ(result.failure,
                StaticIndexRangeFailureKind::ArithmeticOverflow);
      continue;
    }
    int64_t minimum = INT64_MAX, maximum = INT64_MIN;
    for (int64_t a = bounds[0]; a <= bounds[1]; ++a)
      for (int64_t b = bounds[2]; b <= bounds[3]; ++b) {
        minimum = std::min(minimum, a * b);
        maximum = std::max(maximum, a * b);
      }
    if (minimum < 0) {
      EXPECT_EQ(result.failure, StaticIndexRangeFailureKind::NegativeRange);
      continue;
    }
    ASSERT_TRUE(result.succeeded());
    EXPECT_EQ(result.range.min, minimum);
    EXPECT_EQ(result.range.max, maximum);
  }
}
} // namespace
