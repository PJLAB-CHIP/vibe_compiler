//===- RotatingBuffersTest.cpp ----------------------------------------===//

#include "Wafer/Transforms/Tile/RotatingBuffers.h"
#include "Wafer/Transforms/Tile/StorageInitialization.h"

#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/Instr/MemoryPlanning.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <functional>
#include <memory>
#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::shared_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

TEST(RotatingBuffersTest, CurrentRotatingAllocationFeedsActualMiniMalloc) {
  auto context = createContext();
  mlir::Location loc = mlir::UnknownLoc::get(context.get());
  auto module = mlir::ModuleOp::create(loc);
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto function = moduleBuilder.create<mlir::func::FuncOp>(
      loc, "main",
      moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
  mlir::Block *entry = function.addEntryBlock();
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
  auto region =
      builder.create<TileRegionOp>(loc, mlir::TypeRange{}, mlir::ValueRange{});
  region.getBody().push_back(new mlir::Block());
  mlir::OpBuilder regionBuilder =
      mlir::OpBuilder::atBlockBegin(&region.getBody().front());
  auto type = mlir::MemRefType::get(
      {2, 1031, 128}, regionBuilder.getF16Type(),
      mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Tensor));
  auto allocation = regionBuilder.create<mlir::memref::AllocOp>(loc, type);
  auto lower = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto upper = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 1024);
  auto step = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 128);
  auto zero = regionBuilder.create<mlir::arith::ConstantOp>(
      loc, regionBuilder.getFloatAttr(regionBuilder.getF16Type(), 0.0));
  auto loop = regionBuilder.create<mlir::scf::ForOp>(loc, lower, upper, step);
  mlir::OpBuilder loopBuilder = mlir::OpBuilder::atBlockBegin(loop.getBody());
  auto fill = loopBuilder.create<InstrFillOp>(
      loc, allocation.getResult(), zero, FillDomainAttr(), NCCWorker::Worker0);
  regionBuilder.setInsertionPointAfter(loop);
  regionBuilder.create<mlir::memref::DeallocOp>(loc, allocation);
  regionBuilder.create<TileYieldOp>(loc);
  builder.setInsertionPointAfter(region);
  builder.create<mlir::func::ReturnOp>(loc);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

  StructuredMaterializationRelations relations;
  relations.buffers.push_back(
      {fill, allocation, MaterializedBufferRole::Scratch});
  mlir::OwningOpRef<mlir::ModuleOp> owned(module);
  auto rotated = materializeRotatingAllocations(
      std::move(owned), {{allocation, loop, /*multiplicity=*/2}}, relations);
  ASSERT_TRUE(rotated.succeeded())
      << (rotated.failure ? rotated.failure->detail : "");
  ASSERT_EQ(rotated.materialized->slots.size(), 2u);
  EXPECT_EQ(relations.buffers.size(), 2u);
  ASSERT_TRUE(mlir::succeeded(
      wafer::rebuildRequiredNCCJoins(*rotated.materialized->module)));
  ASSERT_TRUE(mlir::succeeded(wafer::planSPMMemoryModule(
      *rotated.materialized->module, /*spmBase=*/0,
      /*spmLimit=*/3 * 1024 * 1024, /*spmAlignment=*/16)));
  unsigned placed = 0;
  rotated.materialized->module->walk([&](mlir::memref::AllocOp current) {
    if (isWaferSPMMemRefType(current.getType())) {
      ++placed;
      EXPECT_TRUE(current->hasAttr(kWaferSPMOffsetAttrName));
    }
  });
  EXPECT_EQ(placed, 2u);
}

TEST(RotatingBuffersTest, DynamicFinalSlotPreservesEmptyInitialValue) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned multiplicity : {2u, 3u})
      for (bool integerIndex : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << extent << '/' << multiplicity << '/' << integerIndex);
        auto context = createContext();
        auto module = mlir::parseSourceString<mlir::ModuleOp>(
            llvm::formatv(
                R"mlir(
        module {{ func.func @main(%count: i32) {{
          wafer.tile.region(%count : i32) -> () {{
          ^bb0(%n: i32):
            %root = memref.alloc() : memref<1x{0}x64xbf16, #wafer.memory<spm, tensor>>
            %out = memref.alloc() : memref<1x{0}x64xbf16, #wafer.memory<spm, tensor>>
            %initial = arith.constant 1.0 : bf16
            %updated = arith.constant 3.0 : bf16
            %zero = arith.constant 0 : i32
            %max = arith.constant 34 : i32
            %nonnegative = arith.maxsi %n, %zero : i32
            %bounded = arith.minsi %nonnegative, %max : i32
            %trips = {1}
            %start = arith.constant 7 : {2}
            %step = arith.constant 3 : {2}
            %span = arith.muli %trips, %step : {2}
            %end = arith.addi %start, %span : {2}
            wafer.tile.fill %root, %initial : memref<1x{0}x64xbf16, #wafer.memory<spm, tensor>>, bf16
            scf.for %iv = %start to %end step %step {3} {{
              wafer.tile.fill %root, %updated : memref<1x{0}x64xbf16, #wafer.memory<spm, tensor>>, bf16
            }
            memref.copy %root, %out : memref<1x{0}x64xbf16, #wafer.memory<spm, tensor>> to memref<1x{0}x64xbf16, #wafer.memory<spm, tensor>>
            memref.dealloc %root : memref<1x{0}x64xbf16, #wafer.memory<spm, tensor>>
            wafer.tile.yield
          }
          return
        } }
      )mlir",
                extent,
                integerIndex ? "arith.addi %bounded, %zero : i32"
                             : "arith.index_cast %bounded : i32 to index",
                integerIndex ? "i32" : "index", integerIndex ? ": i32" : "")
                .str(),
            context.get());
        ASSERT_TRUE(module);
        TileRegionOp region;
        mlir::scf::ForOp loop;
        ComputeFillOp definition;
        module->walk([&](TileRegionOp op) { region = op; });
        module->walk([&](mlir::scf::ForOp op) { loop = op; });
        loop.walk([&](ComputeFillOp op) { definition = op; });
        auto allocation =
            definition.getDest().getDefiningOp<mlir::memref::AllocOp>();
        StructuredMaterializationRelations relations;
        relations.buffers.push_back(
            {definition, allocation, MaterializedBufferRole::Scratch});
        // Independently parsed negative fixtures exercise the same public
        // preflight before any slot exists. Initial slot0 contents cannot prove
        // per-iteration initialization, and an external alias cannot be rewired
        // by replacing only the original root's direct users.
        if (multiplicity == 2) {
          std::string source;
          llvm::raw_string_ostream stream(source);
          module->print(stream);
          for (unsigned kind = 0; kind != 2; ++kind) {
            auto negative =
                mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
            ASSERT_TRUE(negative);
            mlir::scf::ForOp negativeLoop;
            ComputeFillOp writer;
            mlir::memref::CopyOp observer;
            negative->walk([&](mlir::scf::ForOp op) { negativeLoop = op; });
            negativeLoop.walk([&](ComputeFillOp op) { writer = op; });
            negative->walk([&](mlir::memref::CopyOp op) { observer = op; });
            auto root = writer.getDest().getDefiningOp<mlir::memref::AllocOp>();
            mlir::OpBuilder builder(writer);
            if (kind == 0) {
              builder.create<mlir::memref::CopyOp>(writer.getLoc(), root,
                                                   observer.getTarget());
              writer.erase();
            } else {
              builder.setInsertionPoint(observer);
              auto alias = builder.create<mlir::memref::CastOp>(
                  observer.getLoc(), root.getType(), root);
              observer.getSourceMutable().set(alias);
            }
            ASSERT_TRUE(mlir::succeeded(mlir::verify(*negative)));
            StructuredMaterializationRelations negativeRelations;
            negativeRelations.buffers.push_back(
                {root, root, MaterializedBufferRole::Scratch});
            auto rejected = materializeRotatingAllocations(
                std::move(negative), {{root, negativeLoop, multiplicity}},
                negativeRelations);
            ASSERT_FALSE(rejected.succeeded());
            EXPECT_EQ(rejected.failure->kind,
                      LoopPipeliningFailureKind::Unsupported);
            EXPECT_EQ(negativeRelations.buffers.size(), 1u);
          }
        }
        auto rotated = materializeRotatingAllocations(
            std::move(module), {{allocation, loop, multiplicity}}, relations);
        ASSERT_TRUE(rotated.succeeded()) << rotated.failure->detail;
        ASSERT_EQ(rotated.materialized->slots.size(), multiplicity);
        ASSERT_EQ(relations.buffers.size(), multiplicity);
        module = std::move(rotated.materialized->module);

        for (int64_t n : {0, 1, 2, 3, 7, 34}) {
          llvm::DenseMap<mlir::Value, int64_t> integers;
          llvm::DenseMap<mlir::Value, mlir::Value> aliases;
          llvm::DenseMap<mlir::Value, double> contents;
          integers[region.getBody().front().getArgument(0)] = n;
          mlir::Value published;
          double output = -1;
          std::function<bool(mlir::Block &)> execute = [&](mlir::Block &block) {
            for (auto &op : block.without_terminator()) {
              if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
                aliases[alloc] = alloc;
              } else if (auto fill = mlir::dyn_cast<ComputeFillOp>(op)) {
                auto constant =
                    fill.getValue().getDefiningOp<mlir::arith::ConstantOp>();
                if (!constant)
                  return false;
                contents[aliases.lookup(fill.getDest())] =
                    mlir::cast<mlir::FloatAttr>(constant.getValue())
                        .getValueAsDouble();
              } else if (auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(op)) {
                published = aliases.lookup(copy.getSource());
                if (!contents.contains(published))
                  return false;
                output = contents.lookup(published);
              } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
                for (int64_t iv = integers.lookup(loop.getLowerBound());
                     iv < integers.lookup(loop.getUpperBound());
                     iv += integers.lookup(loop.getStep())) {
                  integers[loop.getInductionVar()] = iv;
                  if (!execute(*loop.getBody()))
                    return false;
                }
              } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
                auto &taken = integers.lookup(branch.getCondition())
                                  ? branch.getThenRegion()
                                  : branch.getElseRegion();
                if (!execute(taken.front()))
                  return false;
                for (auto [result, operand] : llvm::zip_equal(
                         branch.getResults(),
                         taken.front().getTerminator()->getOperands()))
                  aliases[result] = aliases.lookup(operand);
              } else if (auto select =
                             mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
                aliases[select] =
                    aliases.lookup(integers.lookup(select.getCondition())
                                       ? select.getTrueValue()
                                       : select.getFalseValue());
              } else if (auto constant =
                             mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
                if (auto integer =
                        mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
                  integers[constant] = integer.getInt();
              } else if (auto cast =
                             mlir::dyn_cast<mlir::arith::IndexCastOp>(op)) {
                integers[cast] = integers.lookup(cast.getIn());
              } else if (mlir::isa<mlir::memref::DeallocOp>(op)) {
                continue;
              } else if (op.getNumOperands() == 2 && op.getNumResults() == 1) {
                int64_t a = integers.lookup(op.getOperand(0)),
                        b = integers.lookup(op.getOperand(1)), value;
                if (mlir::isa<mlir::arith::AddIOp>(op))
                  value = a + b;
                else if (mlir::isa<mlir::arith::SubIOp>(op))
                  value = a - b;
                else if (mlir::isa<mlir::arith::MulIOp>(op))
                  value = a * b;
                else if (mlir::isa<mlir::arith::MaxSIOp>(op))
                  value = std::max(a, b);
                else if (mlir::isa<mlir::arith::MinSIOp>(op))
                  value = std::min(a, b);
                else if (mlir::isa<mlir::arith::DivUIOp>(op)) {
                  if (a < 0 || b <= 0)
                    return false;
                  value = a / b;
                } else if (mlir::isa<mlir::arith::RemUIOp>(op)) {
                  if (a < 0 || b <= 0)
                    return false;
                  value = a % b;
                } else if (auto compare =
                               mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
                  if (compare.getPredicate() == mlir::arith::CmpIPredicate::eq)
                    value = a == b;
                  else if (compare.getPredicate() ==
                           mlir::arith::CmpIPredicate::slt)
                    value = a < b;
                  else
                    return false;
                } else
                  return false;
                integers[op.getResult(0)] = value;
              } else
                return false;
            }
            return true;
          };
          ASSERT_TRUE(execute(region.getBody().front()));
          EXPECT_EQ(output, n ? 3.0 : 1.0);
          EXPECT_EQ(
              published,
              rotated.materialized->slots[n ? (n - 1) % multiplicity : 0]);
        }
        TileRegionToInstrLoweringSession session(*context);
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
        EXPECT_TRUE(mlir::succeeded(
            planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
      }
}

} // namespace
