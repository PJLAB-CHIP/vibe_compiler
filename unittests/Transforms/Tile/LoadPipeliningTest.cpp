//===- LoadPipeliningTest.cpp ----------------------------------------===//

#include "Wafer/Transforms/Tile/LoadPipelining.h"
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

std::string print(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  stream.flush();
  return text;
}

TEST(LoadPipeliningTest,
     LoadPipelinePreservesEveryWindowAndActualSlotLifetime) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    std::string text;
    llvm::raw_string_ostream out(text);
    const std::string ddr = "memref<2x" + std::to_string(extent) +
                            "x64xf16, #wafer.memory<ddr, tensor>>";
    const std::string input =
        "memref<2x32x64xf16, strided<[" + std::to_string(extent * 64) +
        ", 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>";
    const std::string spm = "memref<2x32x64xf16, #wafer.memory<spm, tensor>>";
    out << "module { func.func @main(%input: " << ddr << ", %output: " << ddr
        << ") { \"wafer.tile.region\"(%input, %output) ({ ^bb0(%src: " << ddr
        << ", %dst: " << ddr << "):\n"
        << "%c0 = arith.constant 0 : index\n%c32 = arith.constant 32 : index\n"
        << "%end = arith.constant " << extent / 32 * 32 << " : index\n"
        << "%unchanged = scf.for %iv = %c0 to %end step %c32 iter_args(%carry "
           "= %dst) -> "
        << ddr << " {\n"
        << "%in = memref.subview %src[0, %iv, 0] [2, 32, 64] [1, 1, 1] : "
        << ddr << " to " << input << "\n"
        << "%out = memref.subview %carry[0, %iv, 0] [2, 32, 64] [1, 1, 1] : "
        << ddr << " to " << input << "\n"
        << "%buffer = memref.alloc() : " << spm << "\n"
        << "wafer.tile.load %in into %buffer : " << input << " into " << spm
        << "\n"
        << "wafer.tile.store %buffer, %out : " << spm << " -> " << input
        << "\nscf.yield %carry : " << ddr << "\n}\n";
    // The remaining rows are copied by the same original non-loop operations.
    if (extent % 32) {
      const std::string tail =
          "memref<2x" + std::to_string(extent % 32) + "x64xf16";
      const std::string td =
          tail + ", strided<[" + std::to_string(extent * 64) +
          ", 64, 1], offset: " + std::to_string(extent / 32 * 32 * 64) +
          ">, #wafer.memory<ddr, tensor>>";
      const std::string ts = tail + ", #wafer.memory<spm, tensor>>";
      for (auto name : {"src", "dst"})
        out << "%" << name << "tail = memref.subview %" << name << "[0, "
            << extent / 32 * 32 << ", 0] [2, " << extent % 32
            << ", 64] [1, 1, 1] : " << ddr << " to " << td << "\n";
      out << "%tailbuf = memref.alloc() : " << ts << "\n"
          << "wafer.tile.load %srctail into %tailbuf : " << td << " into " << ts
          << "\n"
          << "wafer.tile.store %tailbuf, %dsttail : " << ts << " -> " << td
          << "\n";
    }
    out << "\"wafer.tile.yield\"() : () -> ()\n}) : (" << ddr << ", " << ddr
        << ") -> ()\nreturn\n}}\n";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module);
    StructuredMaterializationRelations relations;
    module->walk([&](StorageLoadOp load) {
      relations.buffers.push_back(
          {load, load.getDest(), MaterializedBufferRole::Movement});
    });
    const std::string before = print(*module);
    EXPECT_TRUE(hasDistanceOneLoadPipeline(*module));
    EXPECT_EQ(print(*module), before);
    for (unsigned failure = 0; failure != 5; ++failure) {
      SCOPED_TRACE(failure);
      mlir::OwningOpRef<mlir::ModuleOp> negative(module->clone());
      mlir::scf::ForOp loop;
      negative->walk([&](mlir::scf::ForOp op) { loop = op; });
      auto load = *loop.getBody()->getOps<StorageLoadOp>().begin();
      auto store = *loop.getBody()->getOps<StorageStoreOp>().begin();
      if (failure == 0) {
        loop.setUpperBound(loop.getLowerBound());
      } else if (failure == 1) {
        store->moveBefore(load);
      } else if (failure == 2) {
        mlir::OpBuilder builder(store);
        builder.create<StorageLoadOp>(load.getLoc(), load.getSource(),
                                      load.getDest());
      } else if (failure == 3) {
        auto function = loop->getParentOfType<mlir::func::FuncOp>();
        unsigned index = function.getNumArguments();
        function.insertArgument(index, mlir::IndexType::get(context.get()),
                                mlir::DictionaryAttr{}, loop.getLoc());
        auto owner = loop->getParentOfType<TileRegionOp>();
        owner.getInputsMutable().append(function.getArgument(index));
        auto bound = owner.getBody().front().addArgument(
            mlir::IndexType::get(context.get()), loop.getLoc());
        loop.setUpperBound(bound);
      } else {
        mlir::OpBuilder builder(store);
        builder.create<CommPeerSendOp>(
            load.getLoc(), load.getDest(), 1, 2 * 32 * 64 * 2,
            DTEMessageAttr::get(context.get(), 0, 0, 0));
      }
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*negative)));
      const std::string unchanged = print(*negative);
      EXPECT_FALSE(hasDistanceOneLoadPipeline(*negative));
      EXPECT_EQ(print(*negative), unchanged);
    }
    auto result =
        materializeDistanceOneLoadPipelines(std::move(module), relations);
    ASSERT_TRUE(result.succeeded()) << result.failure->detail;
    module = std::move(result.materialized->module);
    ASSERT_EQ(result.materialized->pipelines.size(), 1u);
    EXPECT_EQ(result.materialized->pipelines.front().stageCount, 2u);
    EXPECT_EQ(result.materialized->pipelines.front().kernelTripCount, 31u);
    module->walk([&](mlir::scf::ForOp loop) {
      EXPECT_EQ(mlir::getConstantIntValue(loop.getUpperBound()), 992);
    });
    TileRegionOp region;
    module->walk([&](TileRegionOp op) { region = op; });
    ASSERT_TRUE(region);
    // Interpret current loop/SSA/select operations. Each stored element must
    // come from the corresponding input window, including prologue and tail.
    struct View {
      mlir::Value root;
      int64_t row = 0;
    };
    llvm::DenseMap<mlir::Value, int64_t> scalars;
    llvm::DenseMap<mlir::Value, View> views;
    llvm::DenseMap<mlir::Value, int64_t> loadedRows;
    std::vector<unsigned> coverage(extent, 0);
    auto source = region.getBody().front().getArgument(0);
    auto destination = region.getBody().front().getArgument(1);
    views[source] = {source};
    views[destination] = {destination};
    auto assign = [&](mlir::Value to, mlir::Value from) {
      if (mlir::isa<mlir::MemRefType>(to.getType()))
        views[to] = views.lookup(from);
      else
        scalars[to] = scalars.lookup(from);
    };
    std::function<bool(mlir::Block &)> execute = [&](mlir::Block &block) {
      for (mlir::Operation &op : block.without_terminator()) {
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
          for (auto [arg, initial] :
               llvm::zip_equal(loop.getRegionIterArgs(), loop.getInitArgs()))
            assign(arg, initial);
          for (int64_t iv = scalars.lookup(loop.getLowerBound());
               iv < scalars.lookup(loop.getUpperBound());
               iv += scalars.lookup(loop.getStep())) {
            scalars[loop.getInductionVar()] = iv;
            if (!execute(*loop.getBody()))
              return false;
            auto yield =
                mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
            for (auto [result, value] :
                 llvm::zip_equal(loop.getResults(), yield.getOperands()))
              assign(result, value);
            for (auto [arg, result] :
                 llvm::zip_equal(loop.getRegionIterArgs(), loop.getResults()))
              assign(arg, result);
          }
        } else if (auto constant =
                       mlir::dyn_cast<mlir::arith::ConstantIndexOp>(op)) {
          scalars[constant] = constant.value();
        } else if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
          views[alloc] = {alloc};
        } else if (auto view = mlir::dyn_cast<mlir::memref::SubViewOp>(op)) {
          auto offset = view.getMixedOffsets()[1];
          auto constant = mlir::getConstantIntValue(offset);
          views[view] = {
              views.lookup(view.getSource()).root,
              views.lookup(view.getSource()).row +
                  (constant ? *constant
                            : scalars.lookup(mlir::cast<mlir::Value>(offset)))};
        } else if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
          assign(select, scalars.lookup(select.getCondition())
                             ? select.getTrueValue()
                             : select.getFalseValue());
        } else if (auto load = mlir::dyn_cast<StorageLoadOp>(op)) {
          View from = views.lookup(load.getSource());
          if (from.root != source)
            return false;
          loadedRows[views.lookup(load.getDest()).root] = from.row;
        } else if (auto store = mlir::dyn_cast<StorageStoreOp>(op)) {
          View to = views.lookup(store.getDest());
          auto found = loadedRows.find(views.lookup(store.getSource()).root);
          if (to.root != destination || found == loadedRows.end() ||
              found->second != to.row)
            return false;
          auto type = mlir::cast<mlir::MemRefType>(store.getSource().getType());
          for (int64_t row = to.row; row < to.row + type.getDimSize(1); ++row) {
            if (row < 0 || row >= extent)
              return false;
            ++coverage[row];
          }
        } else if (op.getNumOperands() == 2 && op.getNumResults() == 1) {
          int64_t a = scalars.lookup(op.getOperand(0)),
                  b = scalars.lookup(op.getOperand(1));
          int64_t value;
          if (mlir::isa<mlir::arith::AddIOp>(op))
            value = a + b;
          else if (mlir::isa<mlir::arith::SubIOp>(op))
            value = a - b;
          else if (mlir::isa<mlir::arith::MulIOp>(op))
            value = a * b;
          else if (mlir::isa<mlir::arith::DivUIOp>(op) && b > 0)
            value = a / b;
          else if (mlir::isa<mlir::arith::RemUIOp>(op) && b > 0)
            value = a % b;
          else if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
            if (cmp.getPredicate() != mlir::arith::CmpIPredicate::eq)
              return false;
            value = a == b;
          } else
            return false;
          scalars[op.getResult(0)] = value;
        } else
          return false;
      }
      return true;
    };
    ASSERT_TRUE(execute(region.getBody().front())) << print(*module);
    EXPECT_TRUE(
        llvm::all_of(coverage, [](unsigned count) { return count == 1; }));
    TileRegionToInstrLoweringSession session(*context);
    ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
    ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
    module->walk([&](SyncNCCJoinOp join) {
      EXPECT_FALSE(join->getParentOfType<mlir::scf::ForOp>());
    });
    ASSERT_TRUE(
        mlir::succeeded(planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
    unsigned placed = 0;
    module->walk([&](mlir::memref::AllocOp allocation) {
      if (isWaferSPMMemRefType(allocation.getType())) {
        ++placed;
        EXPECT_TRUE(allocation->hasAttr(kWaferSPMOffsetAttrName));
      }
    });
    EXPECT_EQ(placed, extent % 32 ? 3u : 2u);
  }
}

} // namespace
