//===- LoopPipeliningTest.cpp ----------------------------------------===//

#include "Wafer/Transforms/Tile/LoopPipelining.h"
#include "Wafer/Analysis/ControlFlow/IndexValueBounds.h"
#include "Wafer/Transforms/Tile/StorageInitialization.h"

#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/Instr/MemoryPlanning.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
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
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  wafer::analysis::registerIndexValueBoundsModels(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

struct PipelineInput {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  TilePipelineChoice choice;
};

PipelineInput makeDistanceOneInput(mlir::MLIRContext &context,
                                   uint64_t extent) {
  std::string text = R"mlir(
module {
  func.func @main(%unused: tensor<2xEXTENTx128xf16>) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %seed = arith.constant 0 : i64
    %result = scf.for %iv = %c0 to %c7 step %c1
        iter_args(%carry = %seed) -> i64 {
      %index = arith.index_cast %iv : index to i64
      %offset = arith.addi %index, %index : i64
      %next = arith.addi %carry, %offset : i64
      %scaled = arith.muli %next, %next : i64
      scf.yield %scaled : i64
    }
    return %result : i64
  }
}
)mlir";
  const size_t marker = text.find("EXTENT");
  text.replace(marker, 6, std::to_string(extent));
  PipelineInput input;
  input.module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!input.module)
    return input;
  mlir::arith::IndexCastOp index;
  mlir::arith::AddIOp offset;
  mlir::arith::AddIOp add;
  mlir::arith::MulIOp multiply;
  input.module->walk(
      [&](mlir::scf::ForOp operation) { input.choice.loop = operation; });
  input.module->walk(
      [&](mlir::arith::IndexCastOp operation) { index = operation; });
  input.module->walk([&](mlir::arith::AddIOp operation) {
    if (mlir::isa<mlir::BlockArgument>(operation.getLhs()))
      add = operation;
    else
      offset = operation;
  });
  input.module->walk(
      [&](mlir::arith::MulIOp operation) { multiply = operation; });
  input.choice.operations = {{index, 0}, {offset, 1}, {add, 2}, {multiply, 2}};
  return input;
}

PipelineInput makeFiniteInput(mlir::MLIRContext &context) {
  PipelineInput input;
  input.module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%unused: tensor<2x1025x128xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    scf.for %iv = %c0 to %c7 step %c1 {
      %token = async.execute {
        async.yield
      }
      async.await %token : !async.token
    }
    return
  }
}
)mlir",
                                                         &context);
  if (!input.module)
    return input;
  mlir::async::ExecuteOp issue;
  mlir::async::AwaitOp wait;
  input.module->walk(
      [&](mlir::scf::ForOp operation) { input.choice.loop = operation; });
  input.module->walk(
      [&](mlir::async::ExecuteOp operation) { issue = operation; });
  input.module->walk([&](mlir::async::AwaitOp operation) { wait = operation; });
  input.choice.operations = {{issue, 0}, {wait, 1}};
  input.choice.lowering = TilePipelineLowering::FiniteUnrolled;
  return input;
}

unsigned countLoops(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](mlir::scf::ForOp) { ++count; });
  return count;
}

std::string print(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  stream.flush();
  return text;
}

TEST(LoopPipeliningTest, CurrentOperationBindingsProduceExactSCFPhases) {
  for (uint64_t extent : {uint64_t{1024}, uint64_t{1025}, uint64_t{1031}}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    PipelineInput input = makeDistanceOneInput(*context, extent);
    ASSERT_TRUE(input.module);
    auto prepared = prepareLoopPipelines(*input.module, {input.choice});
    ASSERT_TRUE(prepared.succeeded())
        << (prepared.failure ? prepared.failure->detail : "");
    auto materialized =
        pipelineLoops(std::move(input.module), std::move(*prepared.prepared));
    ASSERT_TRUE(materialized.succeeded())
        << (materialized.failure ? materialized.failure->detail : "");
    ASSERT_EQ(materialized.materialized->pipelines.size(), 1u);
    const PipelinedLoop &pipeline =
        materialized.materialized->pipelines.front();
    EXPECT_EQ(pipeline.stageCount, 3u);
    EXPECT_EQ(pipeline.kernelTripCount, 5u);
    EXPECT_EQ(pipeline.originalOperationCount, 4u);
    EXPECT_EQ(countLoops(materialized.materialized->module->getOperation()),
              1u);
  }
}

TEST(LoopPipeliningTest, EmptyChoiceIsSerializedByteEquivalent) {
  auto context = createContext();
  PipelineInput input = makeDistanceOneInput(*context, 1025);
  ASSERT_TRUE(input.module);
  const std::string before = print(input.module->getOperation());
  auto prepared = prepareLoopPipelines(*input.module, {});
  ASSERT_TRUE(prepared.succeeded());
  auto materialized =
      pipelineLoops(std::move(input.module), std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded());
  EXPECT_TRUE(materialized.materialized->pipelines.empty());
  EXPECT_EQ(print(materialized.materialized->module->getOperation()), before);
}

TEST(LoopPipeliningTest, FiniteUnrollContainsCrossStageAsyncToken) {
  auto context = createContext();
  PipelineInput input = makeFiniteInput(*context);
  ASSERT_TRUE(input.module);
  auto prepared = prepareLoopPipelines(*input.module, {input.choice});
  ASSERT_TRUE(prepared.succeeded())
      << (prepared.failure ? prepared.failure->detail : "");
  auto materialized =
      pipelineLoops(std::move(input.module), std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded())
      << (materialized.failure ? materialized.failure->detail : "");
  EXPECT_EQ(countLoops(materialized.materialized->module->getOperation()), 0u);
  EXPECT_EQ(materialized.materialized->pipelines.front().operations.size(),
            14u);
}

TEST(LoopPipeliningTest,
     PreflightRejectsIncompleteAndDependenceReversingChoices) {
  auto context = createContext();
  PipelineInput incomplete = makeDistanceOneInput(*context, 1025);
  ASSERT_TRUE(incomplete.module);
  incomplete.choice.operations.pop_back();
  auto missing = prepareLoopPipelines(*incomplete.module, {incomplete.choice});
  ASSERT_FALSE(missing.succeeded());
  ASSERT_TRUE(missing.failure);
  EXPECT_EQ(missing.failure->kind, LoopPipeliningFailureKind::BrokenContract);

  PipelineInput reversed = makeDistanceOneInput(*context, 1025);
  ASSERT_TRUE(reversed.module);
  reversed.choice.operations.front().stage = 2;
  auto contradiction =
      prepareLoopPipelines(*reversed.module, {reversed.choice});
  ASSERT_FALSE(contradiction.succeeded());
  ASSERT_TRUE(contradiction.failure);
  EXPECT_EQ(contradiction.failure->kind,
            LoopPipeliningFailureKind::BrokenContract);
}

// Bounded scalar oracle isolates SCF iteration/state semantics; the load test
// separately checks real rank3 payloads through Instr, completion and SPM.
TEST(LoopPipeliningTest, DynamicAndShortDomainsPreserveOrderedState) {
  for (int mode = 0; mode != 5; ++mode) {
    SCOPED_TRACE(mode);
    auto context = createContext();
    std::string text;
    llvm::raw_string_ostream out(text);
    out << R"mlir(module { func.func @main(%shape: tensor<2x1031x128xf16>) {
      %records = memref.alloc() : memref<34xi64>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c34 = arith.constant 34 : index
      %c37 = arith.constant 37 : index
      %c1031 = arith.constant 1031 : index
      %seed = arith.constant 13 : i64
      scf.for %n = %c0 to %c34 step %c1 {
)mlir";
    if (mode == 0)
      out << "%span = arith.muli %n, %c3 : index\n"
             "%end = arith.addi %c37, %span : index\n";
    else if (mode == 1)
      out << "%stride = arith.addi %n, %c1 : index\n";
    else
      out << "%end = arith.constant " << 37 + (mode - 2) * 3 << " : index\n";
    out << "%result = scf.for %iv = %c37 to " << (mode == 1 ? "%c1031" : "%end")
        << " step " << (mode == 1 ? "%stride" : "%c3")
        << R"mlir( iter_args(%state = %seed) -> i64 {
          %index = arith.index_cast %iv : index to i64
          %value = arith.addi %index, %index : i64
          %next = arith.addi %state, %value : i64
          scf.yield %next : i64
        }
        memref.store %result, %records[%n] : memref<34xi64>
      }
      return
    } })mlir";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module) << text;
    TilePipelineChoice choice;
    module->walk([&](mlir::scf::ForOp loop) {
      if (loop->getParentOfType<mlir::scf::ForOp>())
        choice.loop = loop;
    });
    unsigned stage = 0;
    for (auto &op : choice.loop.getBody()->without_terminator())
      choice.operations.push_back({&op, stage++});
    auto prepared = prepareLoopPipelines(*module, {choice});
    ASSERT_TRUE(prepared.succeeded()) << prepared.failure->detail;
    auto result =
        pipelineLoops(std::move(module), std::move(*prepared.prepared));
    ASSERT_TRUE(result.succeeded()) << result.failure->detail;
    EXPECT_EQ(
        result.materialized->pipelines.front().kernelTripCount.has_value(),
        mode >= 2);
    module = std::move(result.materialized->module);
    llvm::DenseMap<mlir::Value, int64_t> values;
    std::vector<int64_t> records(34, -1);
    std::function<bool(mlir::Block &)> execute = [&](mlir::Block &block) {
      for (auto &op : block.without_terminator()) {
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
          llvm::SmallVector<int64_t> state;
          for (auto init : loop.getInitArgs())
            state.push_back(values.lookup(init));
          for (int64_t iv = values.lookup(loop.getLowerBound());
               iv < values.lookup(loop.getUpperBound());
               iv += values.lookup(loop.getStep())) {
            values[loop.getInductionVar()] = iv;
            for (auto [arg, value] :
                 llvm::zip_equal(loop.getRegionIterArgs(), state))
              values[arg] = value;
            if (!execute(*loop.getBody()))
              return false;
            state.clear();
            for (auto value : loop.getBody()->getTerminator()->getOperands())
              state.push_back(values.lookup(value));
          }
          for (auto [value, item] : llvm::zip_equal(loop.getResults(), state))
            values[value] = item;
        } else if (auto cond = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
          auto &region = values.lookup(cond.getCondition())
                             ? cond.getThenRegion()
                             : cond.getElseRegion();
          if (!region.empty() && !execute(region.front()))
            return false;
        } else if (auto constant =
                       mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
          values[constant] =
              mlir::cast<mlir::IntegerAttr>(constant.getValue()).getInt();
        } else if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(op)) {
          values[cast] = values.lookup(cast.getIn());
        } else if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
          values[select] = values.lookup(values.lookup(select.getCondition())
                                             ? select.getTrueValue()
                                             : select.getFalseValue());
        } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
          auto index = values.lookup(store.getIndices().front());
          if (index < 0 || index >= 34)
            return false;
          records[index] = values.lookup(store.getValue());
        } else if (mlir::isa<mlir::memref::AllocOp>(op)) {
          continue;
        } else if (op.getNumOperands() == 2 && op.getNumResults() == 1) {
          int64_t a = values.lookup(op.getOperand(0)),
                  b = values.lookup(op.getOperand(1)), v;
          if (mlir::isa<mlir::arith::AddIOp>(op))
            v = a + b;
          else if (mlir::isa<mlir::arith::SubIOp>(op))
            v = a - b;
          else if (mlir::isa<mlir::arith::MulIOp>(op))
            v = a * b;
          else if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
            if (cmp.getPredicate() == mlir::arith::CmpIPredicate::slt)
              v = a < b;
            else if (cmp.getPredicate() == mlir::arith::CmpIPredicate::eq)
              v = a == b;
            else
              return false;
          } else
            return false;
          values[op.getResult(0)] = v;
        } else
          return false;
      }
      return true;
    };
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    ASSERT_TRUE(execute(function.front())) << print(*module);
    for (int64_t n = 0; n < 34; ++n) {
      int64_t end = mode == 0   ? 37 + n * 3
                    : mode == 1 ? 1031
                                : 37 + (mode - 2) * 3;
      int64_t step = mode == 1 ? n + 1 : 3;
      int64_t expected = 13;
      for (int64_t iv = 37; iv < end; iv += step)
        expected += 2 * iv;
      EXPECT_EQ(records[n], expected) << n << '\n' << print(*module);
    }
  }
}

TEST(LoopPipeliningTest, CrossStageEffectsRequireActualAliasAndRangeProof) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned kind = 0; kind != 4; ++kind) {
      SCOPED_TRACE(::testing::Message() << extent << '/' << kind);
      auto context = createContext();
      auto shape = "1x" + std::to_string(2 * extent) + "x64xf16";
      auto base = "memref<" + shape + ", #wafer.memory<spm, tensor>>";
      auto ddr = "memref<" + shape + ", #wafer.memory<ddr, tensor>>";
      auto compact = "memref<1x" + std::to_string(extent) +
                     "x64xf16, #wafer.memory<spm, tensor>>";
      int64_t offset = kind == 1 ? extent - 1 : extent;
      auto slice = [&](int64_t start) {
        return "memref<1x" + std::to_string(extent) + "x64xf16, strided<[" +
               std::to_string(2 * extent * 64) +
               ", 64, 1], offset: " + std::to_string(start * 64) +
               ">, #wafer.memory<spm, tensor>>";
      };
      auto leftType = slice(0);
      if (kind == 3)
        leftType.replace(leftType.find("spm"), 3, "ddr");
      std::string text;
      llvm::raw_string_ostream out(text);
      out << "module { ";
      if (kind == 3)
        out << "memref.global \"private\" constant @data : " << ddr
            << " = dense<0.0>\n";
      out << "func.func @main("
          << (kind == 2 ? "%a: " + base + ", %b: " + base : "") << ") {\n"
          << (kind == 2 ? "" : "wafer.tile.region() -> () {\n")
          << "%root = memref.alloc() : " << base << "\n";
      if (kind == 3)
        out << "%global = memref.get_global @data : " << ddr << "\n";
      out << "%left = memref.subview %"
          << (kind == 2   ? "a"
              : kind == 3 ? "global"
                          : "root")
          << "[0, 0, 0] [1, " << extent
          << ", 64] [1, 1, 1] : " << (kind == 3 ? ddr : base) << " to "
          << leftType << "\n"
          << "%right = memref.subview %" << (kind == 2 ? "b" : "root") << "[0, "
          << offset << ", 0] [1, " << extent << ", 64] [1, 1, 1] : " << base
          << " to " << slice(offset) << "\n"
          << "%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : "
             "index\n%c4 = arith.constant 4 : index\n"
          << "%zero = arith.constant 0.0 : f16\n"
          << "scf.for %iv = %c0 to %c4 step %c1 {\n"
          << "%tmp = memref.alloc() : " << compact << "\n"
          << "memref.copy %left, %tmp : " << leftType << " to " << compact
          << "\n"
          << "wafer.tile.fill %right, %zero : " << slice(offset) << ", f16\n"
          << "} " << (kind == 2 ? "" : "wafer.tile.yield } ") << "return } }";
      auto module =
          mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
      ASSERT_TRUE(module) << text;
      TilePipelineChoice choice;
      module->walk([&](mlir::scf::ForOp loop) { choice.loop = loop; });
      for (auto &op : choice.loop.getBody()->without_terminator())
        choice.operations.push_back(
            {&op, mlir::isa<ComputeFillOp>(op) ? 1u : 0u});
      const auto original = print(*module);
      auto prepared = prepareLoopPipelines(*module, {choice});
      EXPECT_EQ(print(*module), original);
      if (kind == 1 || kind == 2) {
        ASSERT_FALSE(prepared.succeeded());
        EXPECT_EQ(prepared.failure->kind,
                  LoopPipeliningFailureKind::Unsupported);
        continue;
      }
      ASSERT_TRUE(prepared.succeeded()) << prepared.failure->detail;
      auto result =
          pipelineLoops(std::move(module), std::move(*prepared.prepared));
      ASSERT_TRUE(result.succeeded()) << result.failure->detail;
      module = std::move(result.materialized->module);
      TileRegionOp region;
      module->walk([&](TileRegionOp op) { region = op; });
      TileRegionToInstrLoweringSession session(*context);
      ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
      EXPECT_TRUE(mlir::succeeded(
          planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
      if (kind == 3) {
        EXPECT_TRUE(mlir::succeeded(
            planDDRMemoryModule(*module, 256, 64 * 1024 * 1024,
                                64 * 1024 * 1024, 64 * 1024 * 1024)));
      }
    }
}

TEST(LoopPipeliningTest, StageOffsetsRespectIntegerIndexWidth) {
  // Minimal typed-failure oracle: the original i32 loop is valid, but adding
  // the next-stage offset would wrap. Real payload domains are covered above.
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @main() -> i32 {
      %lower = arith.constant 2147483640 : i32
      %upper = arith.constant 2147483644 : i32
      %step = arith.constant 4 : i32
      %zero = arith.constant 0 : i32
      %result = scf.for %iv = %lower to %upper step %step
          iter_args(%state = %zero) -> i32 : i32 {
        %a = arith.addi %iv, %zero : i32
        %b = arith.addi %state, %a : i32
        scf.yield %b : i32
      }
      return %result : i32
    } }
  )mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  TilePipelineChoice choice;
  module->walk([&](mlir::scf::ForOp loop) { choice.loop = loop; });
  uint32_t stage = 0;
  for (auto &operation : choice.loop.getBody()->without_terminator())
    choice.operations.push_back({&operation, stage++});
  auto original = print(*module);
  auto result = prepareLoopPipelines(*module, {choice});
  ASSERT_FALSE(result.succeeded());
  EXPECT_EQ(result.failure->kind, LoopPipeliningFailureKind::Unsupported);
  EXPECT_NE(result.failure->detail.find("overflow"), std::string::npos);
  EXPECT_EQ(print(*module), original);
}

} // namespace
