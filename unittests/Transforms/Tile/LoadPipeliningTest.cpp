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
  for (int64_t extent : {1024, 1025, 1031})
    for (int64_t runtimeTrips : {-1, 0, 1, 2, 31})
      for (int conditional : {0, 1, 2, 3})
        for (unsigned variant : {0, 1, 2}) {
          const bool variableStep = variant == 1;
          const bool updateLoadedValue = variant == 2;
          const bool dynamic = runtimeTrips >= 0;
          if (variableStep && (!dynamic || conditional))
            continue;
          if (updateLoadedValue && (runtimeTrips != 2 || conditional))
            continue;
          const int64_t lower = dynamic ? 7 : 0;
          const int64_t tripLimit = variableStep ? 29 : 31;
          const int64_t actualTrips = std::min(runtimeTrips, tripLimit);
          const int64_t stride = variableStep && runtimeTrips > 0 ? 33 : 32;
          SCOPED_TRACE(::testing::Message()
                       << extent << '/' << runtimeTrips << '/' << conditional
                       << '/' << variant);
          auto context = createContext();
          std::string text;
          llvm::raw_string_ostream out(text);
          const std::string ddr = "memref<2x" + std::to_string(extent) +
                                  "x64xf16, #wafer.memory<ddr, tensor>>";
          const std::string input =
              "memref<2x32x64xf16, strided<[" + std::to_string(extent * 64) +
              ", 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>";
          const std::string spm =
              "memref<2x32x64xf16, #wafer.memory<spm, tensor>>";
          out << "module { func.func @main(%input: " << ddr
              << ", %iterations: i32, %enabled: i1) { %output = memref.alloc() "
                 ": "
              << ddr
              << "\n \"wafer.tile.region\"(%input, %output, %iterations, "
                 "%enabled) ({ ^bb0(%src: "
              << ddr << ", %dst: " << ddr << ", %trips: i32, %take: i1):\n"
              << "%c0 = arith.constant 0 : index\n%c32 = arith.constant 32 : "
                 "index\n%c64 = arith.constant 64 : index\n"
              << "%lower = arith.constant " << lower << " : index\n";
          if (dynamic) {
            out << "%zero = arith.constant 0 : i32\n"
                << "%maximum = arith.constant " << tripLimit << " : i32\n"
                << "%nonnegative = arith.maxsi %trips, %zero : i32\n"
                   "%bounded = arith.minsi %nonnegative, %maximum : i32\n"
                   "%count = arith.index_cast %bounded : i32 to index\n";
            if (variableStep)
              out << "%one = arith.constant 1 : i32\n"
                     "%delta = arith.minsi %nonnegative, %one : i32\n"
                     "%delta_index = arith.index_cast %delta : i32 to index\n"
                     "%step = arith.addi %c32, %delta_index : index\n";
            out << "%span = arith.muli %count, "
                << (variableStep ? "%step" : "%c32") << " : index\n"
                << "%end = arith.addi %lower, %span : index\n";
          } else
            out << "%end = arith.constant " << extent / 32 * 32 << " : index\n";
          out << "%unchanged = scf.for %iv = %lower to %end step "
              << (variableStep ? "%step" : "%c32")
              << " iter_args(%carry "
                 "= %dst) -> "
              << ddr << " {\n"
              << "%in = memref.subview %src[0, %iv, 0] [2, 32, 64] [1, 1, 1] : "
              << ddr << " to " << input << "\n"
              << "%out = memref.subview %carry[0, %iv, 0] [2, 32, 64] [1, 1, "
                 "1] "
                 ": "
              << ddr << " to " << input << "\n";
          if (conditional == 1 || conditional == 3)
            out << "scf.if %take {\n";
          if (conditional == 2)
            out << "%parity = arith.remui %iv, %c64 : index\n"
                   "%even = arith.cmpi slt, %parity, %c32 : index\nscf.if "
                   "%even "
                   "{\n";
          out << "%buffer = memref.alloc() : " << spm << "\n"
              << "wafer.tile.load %in into %buffer : " << input << " into "
              << spm << "\n";
          if (updateLoadedValue)
            out << "wafer.tile.elementwise_into <add> %buffer, %buffer into "
                   "%buffer {indexing_maps = [affine_map<(a,b,c)->(a,b,c)>, "
                   "affine_map<(a,b,c)->(a,b,c)>, "
                   "affine_map<(a,b,c)->(a,b,c)>]} : "
                << spm << ", " << spm << " into " << spm << "\n";
          out << "wafer.tile.store %buffer, %out : " << spm << " -> " << input
              << "\n";
          if (conditional == 3)
            out << "} else {\n%alternative = memref.alloc() : " << spm << "\n"
                << "wafer.tile.load %in into %alternative : " << input
                << " into " << spm << "\n"
                << "wafer.tile.store %alternative, %out : " << spm << " -> "
                << input << "\n";
          if (conditional)
            out << "}\n";
          out << "scf.yield %carry : " << ddr << "\n}\n";
          // The remaining rows are copied by the same original non-loop
          // operations.
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
                << "wafer.tile.load %srctail into %tailbuf : " << td << " into "
                << ts << "\n"
                << "wafer.tile.store %tailbuf, %dsttail : " << ts << " -> "
                << td << "\n";
          }
          out << "\"wafer.tile.yield\"() : () -> ()\n}) : (" << ddr << ", "
              << ddr << ", i32, i1) -> ()\nreturn\n}}\n";
          auto module =
              mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
          ASSERT_TRUE(module);
          StructuredMaterializationRelations relations;
          module->walk([&](StorageLoadOp load) {
            relations.buffers.push_back(
                {load, load.getDest(), MaterializedBufferRole::Movement});
          });
          const std::string before = print(*module);
          EXPECT_TRUE(queryDistanceOneLoadPipelines(*module).succeeded());
          EXPECT_EQ(print(*module), before);
          for (unsigned failure = 0; !conditional && failure != 5; ++failure) {
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
              builder.create<mlir::memref::DeallocOp>(load.getLoc(),
                                                      load.getDest());
            } else if (failure == 3) {
              auto function = loop->getParentOfType<mlir::func::FuncOp>();
              unsigned index = function.getNumArguments();
              function.insertArgument(index,
                                      mlir::IndexType::get(context.get()),
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
            EXPECT_FALSE(queryDistanceOneLoadPipelines(*negative).succeeded());
            EXPECT_EQ(print(*negative), unchanged);
          }
          auto result =
              materializeDistanceOneLoadPipelines(std::move(module), relations);
          ASSERT_TRUE(result.succeeded()) << result.failure->detail;
          module = std::move(result.materialized->module);
          ASSERT_EQ(result.materialized->pipelines.size(), 1u);
          EXPECT_EQ(result.materialized->pipelines.front().stageCount, 2u);
          const auto count =
              result.materialized->pipelines.front().kernelTripCount;
          EXPECT_EQ(count.has_value(), !dynamic);
          if (!dynamic) {
            EXPECT_EQ(count, 31u);
            module->walk([&](mlir::scf::ForOp loop) {
              EXPECT_EQ(mlir::getConstantIntValue(loop.getUpperBound()), 992);
            });
          }
          TileRegionOp region;
          module->walk([&](TileRegionOp op) { region = op; });
          ASSERT_TRUE(region);
          // Interpret current loop/SSA/select operations. Each stored element
          // must come from the corresponding input window, including prologue
          // and tail.
          struct View {
            mlir::Value root;
            int64_t row = 0;
          };
          llvm::DenseMap<mlir::Value, int64_t> scalars;
          llvm::DenseMap<mlir::Value, View> views;
          llvm::DenseMap<mlir::Value, int64_t> loadedRows;
          llvm::DenseMap<mlir::Value, unsigned> doublings;
          std::vector<unsigned> coverage(extent, 0);
          auto source = region.getBody().front().getArgument(0);
          auto destination = region.getBody().front().getArgument(1);
          views[source] = {source};
          views[destination] = {destination};
          scalars[region.getBody().front().getArgument(2)] = runtimeTrips;
          scalars[region.getBody().front().getArgument(3)] =
              runtimeTrips % 2 != 0;
          auto assign = [&](mlir::Value to, mlir::Value from) {
            if (mlir::isa<mlir::MemRefType>(to.getType()))
              views[to] = views.lookup(from);
            else
              scalars[to] = scalars.lookup(from);
          };
          std::function<bool(mlir::Block &)> execute = [&](mlir::Block &block) {
            for (mlir::Operation &op : block.without_terminator()) {
              if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
                for (auto [arg, initial] : llvm::zip_equal(
                         loop.getRegionIterArgs(), loop.getInitArgs()))
                  assign(arg, initial);
                for (auto [result, initial] :
                     llvm::zip_equal(loop.getResults(), loop.getInitArgs()))
                  assign(result, initial);
                for (int64_t iv = scalars.lookup(loop.getLowerBound());
                     iv < scalars.lookup(loop.getUpperBound());
                     iv += scalars.lookup(loop.getStep())) {
                  scalars[loop.getInductionVar()] = iv;
                  if (!execute(*loop.getBody()))
                    return false;
                  auto yield = mlir::cast<mlir::scf::YieldOp>(
                      loop.getBody()->getTerminator());
                  for (auto [result, value] :
                       llvm::zip_equal(loop.getResults(), yield.getOperands()))
                    assign(result, value);
                  for (auto [arg, result] : llvm::zip_equal(
                           loop.getRegionIterArgs(), loop.getResults()))
                    assign(arg, result);
                }
              } else if (auto conditional =
                             mlir::dyn_cast<mlir::scf::IfOp>(op)) {
                auto &taken = scalars.lookup(conditional.getCondition())
                                  ? conditional.getThenRegion()
                                  : conditional.getElseRegion();
                if (!taken.empty() && !execute(taken.front()))
                  return false;
              } else if (auto constant =
                             mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
                scalars[constant] =
                    mlir::cast<mlir::IntegerAttr>(constant.getValue()).getInt();
              } else if (auto cast =
                             mlir::dyn_cast<mlir::arith::IndexCastOp>(op)) {
                scalars[cast] = scalars.lookup(cast.getIn());
              } else if (auto alloc =
                             mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
                views[alloc] = {alloc};
              } else if (auto view =
                             mlir::dyn_cast<mlir::memref::SubViewOp>(op)) {
                auto offset = view.getMixedOffsets()[1];
                auto constant = mlir::getConstantIntValue(offset);
                views[view] = {
                    views.lookup(view.getSource()).root,
                    views.lookup(view.getSource()).row +
                        (constant ? *constant
                                  : scalars.lookup(
                                        mlir::cast<mlir::Value>(offset)))};
              } else if (auto select =
                             mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
                assign(select, scalars.lookup(select.getCondition())
                                   ? select.getTrueValue()
                                   : select.getFalseValue());
              } else if (auto load = mlir::dyn_cast<StorageLoadOp>(op)) {
                View from = views.lookup(load.getSource());
                if (from.root != source || from.row < 0 ||
                    from.row + mlir::cast<mlir::MemRefType>(
                                   load.getSource().getType())
                                   .getDimSize(1) >
                        extent)
                  return false;
                loadedRows[views.lookup(load.getDest()).root] = from.row;
                doublings[views.lookup(load.getDest()).root] = 0;
              } else if (auto update =
                             mlir::dyn_cast<ComputeElementwiseIntoOp>(op)) {
                ++doublings[views.lookup(update.getDest()).root];
              } else if (auto store = mlir::dyn_cast<StorageStoreOp>(op)) {
                View to = views.lookup(store.getDest());
                auto found =
                    loadedRows.find(views.lookup(store.getSource()).root);
                if (to.root != destination || found == loadedRows.end() ||
                    found->second != to.row)
                  return false;
                if (doublings.lookup(views.lookup(store.getSource()).root) !=
                    (updateLoadedValue && to.row < extent / 32 * 32 ? 1u : 0u))
                  return false;
                auto type =
                    mlir::cast<mlir::MemRefType>(store.getSource().getType());
                for (int64_t row = to.row; row < to.row + type.getDimSize(1);
                     ++row) {
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
                else if (mlir::isa<mlir::arith::MinSIOp>(op))
                  value = std::min(a, b);
                else if (mlir::isa<mlir::arith::MaxSIOp>(op))
                  value = std::max(a, b);
                else if (mlir::isa<mlir::arith::AndIOp>(op))
                  value = a & b;
                else if (mlir::isa<mlir::arith::DivUIOp>(op) && b > 0)
                  value = a / b;
                else if (mlir::isa<mlir::arith::RemUIOp>(op) && b > 0)
                  value = a % b;
                else if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
                  if (cmp.getPredicate() == mlir::arith::CmpIPredicate::eq)
                    value = a == b;
                  else if (cmp.getPredicate() ==
                           mlir::arith::CmpIPredicate::slt)
                    value = a < b;
                  else
                    return false;
                } else
                  return false;
                scalars[op.getResult(0)] = value;
              } else
                return false;
            }
            return true;
          };
          ASSERT_TRUE(execute(region.getBody().front())) << print(*module);
          for (int64_t row = 0; row < extent; ++row) {
            const int64_t end =
                dynamic ? lower + actualTrips * stride : extent / 32 * 32;
            bool taken =
                !conditional || conditional == 3 ||
                (conditional == 1
                     ? runtimeTrips % 2 != 0
                     : ((lower + ((row - lower) / 32) * 32) % 64) < 32);
            unsigned expected = (row >= lower && row < end && taken &&
                                 (row - lower) % stride < 32) ||
                                row >= extent / 32 * 32;
            EXPECT_EQ(coverage[row], expected) << row;
          }
          TileRegionToInstrLoweringSession session(*context);
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
          ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
          module->walk([&](SyncNCCJoinOp join) {
            EXPECT_FALSE(join->getParentOfType<mlir::scf::ForOp>());
          });
          ASSERT_TRUE(mlir::succeeded(
              planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
          ASSERT_TRUE(mlir::succeeded(
              planDDRMemoryModule(*module, 256, 64 * 1024 * 1024,
                                  64 * 1024 * 1024, 64 * 1024 * 1024)));
          unsigned placed = 0;
          module->walk([&](mlir::memref::AllocOp allocation) {
            if (isWaferSPMMemRefType(allocation.getType())) {
              ++placed;
              EXPECT_TRUE(allocation->hasAttr(kWaferSPMOffsetAttrName));
            }
          });
          EXPECT_EQ(placed, (conditional == 3 ? 4u : 2u) + (extent % 32 != 0));
        }
}

TEST(LoadPipeliningTest, NestedCandidatesKeepOneLiveTransformationScope) {
  for (int64_t extent : {1024, 1025, 1031}) {
    auto context = createContext();
    std::string shape = "2x" + std::to_string(extent) + "x64xf16";
    std::string ddr = "memref<" + shape + ", #wafer.memory<ddr, tensor>>";
    std::string spm = "memref<" + shape + ", #wafer.memory<spm, tensor>>";
    std::string text;
    llvm::raw_string_ostream out(text);
    out << "module { func.func @main(%input: " << ddr << ") {\n"
        << "%output = memref.alloc() : " << ddr << "\n"
        << "wafer.tile.region(%input, %output : " << ddr << ", " << ddr
        << ") -> () {\n"
        << "^bb0(%source: " << ddr << ", %dest: " << ddr << "):\n"
        << "%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : "
           "index\n%c4 = arith.constant 4 : index\n"
        << "scf.for %i = %c0 to %c4 step %c1 {\n"
        << "%outer = memref.alloc() : " << spm << "\n"
        << "wafer.tile.load %source into %outer : " << ddr << " into " << spm
        << "\n"
        << "scf.for %j = %c0 to %c4 step %c1 {\n"
        << "%inner = memref.alloc() : " << spm << "\n"
        << "wafer.tile.load %source into %inner : " << ddr << " into " << spm
        << "\n"
        << "wafer.tile.store %inner, %dest : " << spm << " -> " << ddr
        << "\n}\n"
        << "wafer.tile.store %outer, %dest : " << spm << " -> " << ddr
        << "\n}\n"
        << "wafer.tile.yield } return } }";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module) << text;
    mlir::scf::ForOp outer, inner;
    module->walk([&](mlir::scf::ForOp loop) {
      if (loop->getParentOfType<mlir::scf::ForOp>())
        inner = loop;
      else
        outer = loop;
    });
    auto query = queryDistanceOneLoadPipelines(*module);
    ASSERT_TRUE(query.succeeded()) << query.failure->detail;
    ASSERT_EQ(query.loops.size(), 1u);
    EXPECT_EQ(query.loops.front(), inner);
    auto choice = [](mlir::scf::ForOp loop) {
      TilePipelineChoice result;
      result.loop = loop;
      for (auto &op : loop.getBody()->without_terminator())
        result.operations.push_back(
            {&op,
             mlir::isa<mlir::memref::AllocOp, StorageLoadOp>(op) ? 0u : 1u});
      return result;
    };
    auto overlapping =
        prepareLoopPipelines(*module, {choice(outer), choice(inner)});
    ASSERT_FALSE(overlapping.succeeded());
    EXPECT_EQ(overlapping.failure->kind,
              LoopPipeliningFailureKind::BrokenContract);
    EXPECT_NE(overlapping.failure->detail.find("overlaps"), std::string::npos);
    StructuredMaterializationRelations relations;
    module->walk([&](StorageLoadOp load) {
      relations.buffers.push_back(
          {load, load.getDest(), MaterializedBufferRole::Movement});
    });
    auto pipelined =
        materializeDistanceOneLoadPipelines(std::move(module), relations);
    ASSERT_TRUE(pipelined.succeeded()) << pipelined.failure->detail;
    ASSERT_EQ(pipelined.materialized->pipelines.size(), 1u);
    module = std::move(pipelined.materialized->module);
    EXPECT_TRUE(module->getOperation()->isAncestor(outer));
    TileRegionOp region;
    module->walk([&](TileRegionOp op) { region = op; });
    TileRegionToInstrLoweringSession session(*context);
    ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
    ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
    ASSERT_TRUE(
        mlir::succeeded(planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
  }
}

} // namespace
