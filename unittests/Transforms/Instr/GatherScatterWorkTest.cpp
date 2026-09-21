//===- GatherScatterWorkTest.cpp - Exact GS stream splitting -------------===//

#include "Wafer/Transforms/Instr/GatherScatterWork.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Target/GatherScatter.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

namespace {
using Triple = std::array<int64_t, 3>;
using AddressPair = std::pair<int64_t, int64_t>;

class GatherScatterWorkTest : public ::testing::Test {
protected:
  GatherScatterWorkTest() {
    mlir::DialectRegistry registry;
    wafer::registerWaferCoreDialects(registry);
    context.appendDialectRegistry(registry);
    context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                        mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    context.loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  make(int64_t inner, Triple sourceIterations, Triple sourceStrides,
       Triple destIterations, Triple destStrides, mlir::Type dtype,
       bool shared = false, int64_t sourceBase = 0, int64_t destBase = 0,
       bool dynamic = false) {
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    auto function = mlir::func::FuncOp::create(loc, "entry",
                                               builder.getFunctionType({}, {}));
    module.push_back(function);
    builder.setInsertionPointToStart(function.addEntryBlock());
    auto span = [&](Triple iterations, Triple strides, int64_t base) {
      for (unsigned i = 0; i < 3; ++i)
        base += (iterations[i] - 1) * strides[i];
      return base + inner + 64;
    };
    int64_t maximum =
        std::max(span(sourceIterations, sourceStrides, sourceBase),
                 span(destIterations, destStrides, destBase));
    const int64_t elementBytes = dtype.getIntOrFloatBitWidth() / 8;
    // Real-scale rank-three storage, including room for guards/offsets.
    int64_t channels =
        (maximum + 2 * 1031 * elementBytes - 1) / (2 * 1031 * elementBytes);
    auto type = mlir::MemRefType::get(
        {2, 1031, channels}, dtype, mlir::MemRefLayoutAttrInterface{},
        wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                               wafer::MemLayout::Tensor));
    auto source = builder.create<mlir::memref::AllocOp>(loc, type);
    mlir::Value dest =
        shared ? source.getResult()
               : builder.create<mlir::memref::AllocOp>(loc, type).getResult();
    mlir::Value sourceOffset, destOffset;
    mlir::scf::ForOp outer;
    if (dynamic) {
      auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto c2 = builder.create<mlir::arith::ConstantIndexOp>(loc, 2);
      auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto c32 = builder.create<mlir::arith::ConstantIndexOp>(loc, 32);
      outer = builder.create<mlir::scf::ForOp>(loc, c0, c2, c1);
      builder.setInsertionPointToStart(outer.getBody());
      sourceOffset = builder.create<mlir::arith::MulIOp>(
          loc, outer.getInductionVar(), c32);
      destOffset = sourceOffset;
    }
    int64_t count =
        std::accumulate(sourceIterations.begin(), sourceIterations.end(), inner,
                        std::multiplies<int64_t>());
    builder.create<wafer::InstrGatherScatterOp>(
        loc, source, dest, sourceOffset, destOffset, count, inner,
        dynamic ? mlir::IntegerAttr{} : builder.getI64IntegerAttr(sourceBase),
        dynamic ? mlir::IntegerAttr{} : builder.getI64IntegerAttr(destBase),
        sourceStrides, sourceIterations, destStrides, destIterations,
        wafer::DDRResourceAttr::get(&context, 7), wafer::NCCWorker::Worker1);
    if (dynamic)
      builder.setInsertionPointAfter(outer);
    builder.create<mlir::func::ReturnOp>(loc);
    return mlir::OwningOpRef<mlir::ModuleOp>(module);
  }

  struct Stream {
    std::vector<AddressPair> bytes;
    unsigned commands = 0;
    uint64_t transfers = 0;
  };

  // Independent byte-stream interpreter: execute the actual output loops and
  // descriptor radix arithmetic, not the transformation's slicing algorithm.
  Stream stream(mlir::ModuleOp module, bool bounded) {
    Stream result;
    llvm::DenseMap<mlir::Value, int64_t> values;
    llvm::DenseMap<mlir::Value, llvm::DenseMap<int64_t, int64_t>> memory;
    mlir::Value originalSource;
    std::function<void(mlir::Block &)> run = [&](mlir::Block &block) {
      for (mlir::Operation &operation : block) {
        if (auto c = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(operation))
          values[c] = c.value();
        else if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(operation))
          values[add] = values.at(add.getLhs()) + values.at(add.getRhs());
        else if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(operation))
          values[mul] = values.at(mul.getLhs()) * values.at(mul.getRhs());
        else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
          for (int64_t i = values.at(loop.getLowerBound());
               i < values.at(loop.getUpperBound());
               i += values.at(loop.getStep())) {
            values[loop.getInductionVar()] = i;
            run(*loop.getBody());
          }
        } else if (auto gs =
                       mlir::dyn_cast<wafer::InstrGatherScatterOp>(operation)) {
          if (bounded) {
            EXPECT_TRUE(wafer::target::isGatherScatterIssueBounded(
                gs.getByteCount(), gs.getInnerBytes()));
          }
          EXPECT_EQ(gs.getWorker(), wafer::NCCWorker::Worker1);
          EXPECT_EQ(gs.getDdrResourceAttr().getResourceId(), 7);
          if (!originalSource)
            originalSource = gs.getSource();
          ++result.commands;
          result.transfers += gs.getByteCount() / gs.getInnerBytes();
          auto address = [&](bool source, int64_t ordinal) {
            auto fixed = source ? gs.getSrcOffsetAttr() : gs.getDstOffsetAttr();
            auto dynamic =
                source ? gs.getSrcOffsetValue() : gs.getDstOffsetValue();
            auto iterations =
                source ? gs.getSrcIterations() : gs.getDstIterations();
            auto strides = source ? gs.getSrcStrides() : gs.getDstStrides();
            int64_t value =
                dynamic ? values.at(dynamic) : (fixed ? fixed.getInt() : 0);
            for (unsigned axis = 0; axis < 3; ++axis) {
              value += ordinal % iterations[axis] * strides[axis];
              ordinal /= iterations[axis];
            }
            return value;
          };
          std::vector<AddressPair> writes;
          for (uint64_t n = 0; n < gs.getByteCount() / gs.getInnerBytes();
               ++n) {
            const int64_t source = address(true, n), dest = address(false, n);
            for (uint64_t byte = 0; byte < gs.getInnerBytes(); ++byte) {
              const auto &contents = memory[gs.getSource()];
              auto found = contents.find(source + byte);
              EXPECT_TRUE(found != contents.end() ||
                          gs.getSource() == originalSource)
                  << "read from uninitialized destination byte";
              // Unique byte identities model arbitrary data, including floating
              // bit patterns; this is movement, with no arithmetic oracle.
              int64_t identity =
                  found != contents.end() ? found->second : source + byte;
              writes.emplace_back(identity, dest + byte);
            }
          }
          for (auto [identity, dest] : writes)
            memory[gs.getDest()][dest] = identity;
          result.bytes.insert(result.bytes.end(), writes.begin(), writes.end());
        }
      }
    };
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    run(function.front());
    return result;
  }

  void check(mlir::ModuleOp module, bool downstream = true) {
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
    Stream expected = stream(module, false);
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    mlir::PassManager manager(&context);
    manager.addNestedPass<mlir::func::FuncOp>(
        wafer::createMaterializeGatherScatterWorkPass());
    ASSERT_TRUE(mlir::succeeded(manager.run(module)));
    Stream actual = stream(module, true);
    // Prefix replication changes issue order, while each destination byte must
    // still be written exactly as often with the same source byte identity.
    std::sort(expected.bytes.begin(), expected.bytes.end());
    std::sort(actual.bytes.begin(), actual.bytes.end());
    ASSERT_EQ(actual.bytes, expected.bytes);
    auto repeat = wafer::materializeGatherScatterWork(function);
    ASSERT_TRUE(repeat.succeeded());
    EXPECT_EQ(repeat.rewrittenOperations, 0u);
    Stream repeated = stream(module, true);
    std::sort(repeated.bytes.begin(), repeated.bytes.end());
    EXPECT_EQ(repeated.bytes, expected.bytes);
    if (downstream) {
      ASSERT_TRUE(mlir::succeeded(wafer::rebuildRequiredNCCJoins(function)));
      unsigned joins = 0;
      function.walk([&](wafer::SyncNCCJoinOp join) {
        ++joins;
        EXPECT_FALSE(join->getParentOfType<mlir::scf::ForOp>());
      });
      EXPECT_EQ(joins, 1u);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
    }
  }

  mlir::MLIRContext context;
};

TEST_F(GatherScatterWorkTest, BroadcastMainTailAndDynamicOffsets) {
  for (mlir::Type dtype : {mlir::Type(mlir::Float16Type::get(&context)),
                           mlir::Type(mlir::BFloat16Type::get(&context)),
                           mlir::Type(mlir::Float32Type::get(&context))})
    for (int64_t rows : {1024, 1025, 1031})
      for (bool dynamic : {false, true}) {
        int64_t width = dtype.getIntOrFloatBitWidth() / 8;
        SCOPED_TRACE(::testing::Message()
                     << rows << ':' << width << ':' << dynamic);
        auto module = make(width, {32, rows, 2}, {0, width, 0}, {32, rows, 2},
                           {width, 32 * width, 32 * rows * width}, dtype, false,
                           16, 32, dynamic);
        check(*module);
      }
}

TEST_F(GatherScatterWorkTest, RepeatedInnerAndOuterAxesUseInitializedPrefixes) {
  // Bounded fault oracle: exact original descriptor; realistic rank/length
  // coverage is provided by BroadcastMainTailAndDynamicOffsets above.
  auto module = make(4, {64, 256, 4}, {0, 4, 0}, {64, 256, 4}, {4, 256, 65536},
                     mlir::Float32Type::get(&context));
  check(*module);
  EXPECT_EQ(stream(*module, true).commands, 9u);
  EXPECT_EQ(stream(*module, true).transfers, 1794u);
  unsigned staticGS = 0;
  module->walk([&](wafer::InstrGatherScatterOp) { ++staticGS; });
  EXPECT_EQ(staticGS, 9u);
}

TEST_F(GatherScatterWorkTest, DifferentRadicesAndHolesPreserveOrderedPairs) {
  for (int64_t length : {1024, 1025, 1031}) {
    auto module =
        make(2, {6, length, 4}, {4, 32, 32 * length}, {8, 3 * length, 1},
             {2, 32, 0}, mlir::Float16Type::get(&context), false, 16, 32);
    Stream before = stream(*module, false);
    check(*module);
    EXPECT_EQ(stream(*module, true).bytes, before.bytes);
  }
}

TEST_F(GatherScatterWorkTest, CoalescesCommonContinuousInnerBeforeSplitting) {
  auto module = make(2, {64, 1025, 2}, {2, 128, 131200}, {32, 2050, 2},
                     {2, 64, 131200}, mlir::BFloat16Type::get(&context));
  check(*module);
  EXPECT_EQ(stream(*module, true).commands, 1u);
}

TEST_F(GatherScatterWorkTest, LargeInnerAndPayloadHaveExactTails) {
  for (int64_t size : {1024 * 1024, 1024 * 1024 + 2, 2 * 1024 * 1024 + 62}) {
    auto module = make(size, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {0, 0, 0},
                       mlir::IntegerType::get(&context, 8));
    check(*module);
    EXPECT_EQ(stream(*module, true).commands,
              (size + 1024 * 1024 - 1) / (1024 * 1024));
  }
}

TEST_F(GatherScatterWorkTest, SameBufferDisjointRangesAndAliasingRejection) {
  for (bool disjoint : {false, true}) {
    auto module =
        make(2, {32, 1025, 1}, {0, 2, 0}, {32, 1025, 1}, {2, 64, 0},
             mlir::Float16Type::get(&context), true, 0, disjoint ? 4096 : 0);
    if (disjoint) {
      check(*module);
    } else {
      mlir::ScopedDiagnosticHandler handler(
          &context, [](mlir::Diagnostic &) { return mlir::success(); });
      auto result = wafer::materializeGatherScatterWork(
          *module->getOps<mlir::func::FuncOp>().begin());
      EXPECT_EQ(result.failure,
                wafer::GatherScatterWorkFailure::UnsupportedAliasing);
      EXPECT_EQ(result.rewrittenOperations, 0u);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
  }
}

TEST_F(GatherScatterWorkTest, AlreadyBoundedRepetitionAlsoReducesInnerWork) {
  auto module = make(2, {8, 1024, 1}, {0, 2, 0}, {8, 1024, 1}, {2, 16, 0},
                     mlir::Float16Type::get(&context));
  auto result = wafer::materializeGatherScatterWork(
      *module->getOps<mlir::func::FuncOp>().begin());
  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(result.rewrittenOperations, 1u);
  EXPECT_EQ(stream(*module, true).transfers, 4096u);
  check(*module);
}

TEST_F(GatherScatterWorkTest, NonPowerOfTwoRepetitionAndDifferentRadices) {
  for (int64_t rows : {1024, 1025, 1031}) {
    auto module =
        make(2, {37, rows, 3}, {0, 6, 0}, {37 * rows * 3, 1, 1}, {2, 0, 0},
             mlir::BFloat16Type::get(&context), false, 16, 32);
    const auto before = stream(*module, false).transfers;
    check(*module);
    EXPECT_LT(stream(*module, true).transfers, before);
  }
}

TEST_F(GatherScatterWorkTest, DestinationHolesRetainOrderedBroadcast) {
  auto module = make(1, {17, 1031, 1}, {0, 2, 0}, {17, 1031, 1}, {2, 48, 0},
                     mlir::IntegerType::get(&context, 8));
  Stream before = stream(*module, false);
  check(*module);
  EXPECT_EQ(stream(*module, true).bytes, before.bytes);
}

TEST_F(GatherScatterWorkTest, OversizedPrefixStepsRetainCompactLoops) {
  auto module = make(1, {4, 32768, 1}, {0, 2, 0}, {4, 32768, 1}, {1, 4, 0},
                     mlir::IntegerType::get(&context, 8));
  Stream before = stream(*module, false);
  check(*module);
  EXPECT_LT(stream(*module, true).transfers, before.transfers);
  unsigned staticGS = 0, loops = 0;
  module->walk([&](wafer::InstrGatherScatterOp) { ++staticGS; });
  module->walk([&](mlir::scf::ForOp) { ++loops; });
  // The seed and two growth steps each split into two equal commands. Keep
  // three loop bodies instead of interleaving a separate tree per partition.
  EXPECT_EQ(staticGS, 3u);
  EXPECT_EQ(loops, 3u);
}

TEST_F(GatherScatterWorkTest, BoundedContinuousCopyIsUnchanged) {
  auto module = make(2, {8, 1024, 1}, {2, 16, 0}, {8, 1024, 1}, {2, 16, 0},
                     mlir::Float16Type::get(&context));
  auto result = wafer::materializeGatherScatterWork(
      *module->getOps<mlir::func::FuncOp>().begin());
  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(result.rewrittenOperations, 0u);
  check(*module);
}

TEST_F(GatherScatterWorkTest, UnknownAliasCannotUseDestinationPrefixes) {
  for (int64_t width : {8, 32}) {
    auto module = make(2, {width, 1031, 1}, {0, 2, 0}, {width, 1031, 1},
                       {2, 2 * width, 0}, mlir::Float16Type::get(&context));
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    auto first = *function.getOps<mlir::memref::AllocOp>().begin();
    auto type = first.getType();
    function.setType(mlir::FunctionType::get(&context, {type, type}, {}));
    llvm::SmallVector<mlir::memref::AllocOp> allocations;
    for (auto alloc : function.getOps<mlir::memref::AllocOp>())
      allocations.push_back(alloc);
    for (auto alloc : allocations) {
      auto argument = function.front().addArgument(type, function.getLoc());
      alloc.replaceAllUsesWith(argument);
      alloc.erase();
    }
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    mlir::ScopedDiagnosticHandler handler(
        &context, [](mlir::Diagnostic &) { return mlir::success(); });
    auto result = wafer::materializeGatherScatterWork(function);
    EXPECT_EQ(result.rewrittenOperations, 0u);
    EXPECT_EQ(result.failure,
              width == 8
                  ? wafer::GatherScatterWorkFailure::None
                  : wafer::GatherScatterWorkFailure::UnsupportedAliasing);
  }
}

TEST_F(GatherScatterWorkTest, WideInnerBroadcastKeepsEfficientSingleIssue) {
  auto module = make(4096, {64, 1, 1}, {0, 0, 0}, {64, 1, 1}, {4096, 0, 0},
                     mlir::Float32Type::get(&context));
  auto result = wafer::materializeGatherScatterWork(
      *module->getOps<mlir::func::FuncOp>().begin());
  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(result.rewrittenOperations, 0u);
  check(*module);
}
} // namespace
