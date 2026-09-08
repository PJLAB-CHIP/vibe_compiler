//===- ExecutionStructureMaterializationTest.cpp ---------------------===//

#include "Wafer/Transforms/Tile/ExecutionStructure.h"

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
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

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

TEST(ExecutionStructureMaterializationTest,
     CurrentOperationBindingsProduceExactSCFPhases) {
  for (uint64_t extent : {uint64_t{1024}, uint64_t{1025}, uint64_t{1031}}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    PipelineInput input = makeDistanceOneInput(*context, extent);
    ASSERT_TRUE(input.module);
    auto prepared =
        prepareTileExecutionStructure(*input.module, {input.choice});
    ASSERT_TRUE(prepared.succeeded())
        << (prepared.failure ? prepared.failure->detail : "");
    auto materialized = materializeExecutionStructure(
        std::move(input.module), std::move(*prepared.prepared));
    ASSERT_TRUE(materialized.succeeded())
        << (materialized.failure ? materialized.failure->detail : "");
    ASSERT_EQ(materialized.materialized->pipelines.size(), 1u);
    const MaterializedExecutionPipeline &pipeline =
        materialized.materialized->pipelines.front();
    EXPECT_EQ(pipeline.stageCount, 3u);
    EXPECT_EQ(pipeline.kernelDynamicTripCount, 5u);
    EXPECT_EQ(pipeline.originalOperationCount, 4u);
    EXPECT_EQ(countLoops(materialized.materialized->module->getOperation()),
              1u);
  }
}

TEST(ExecutionStructureMaterializationTest,
     EmptyChoiceIsSerializedByteEquivalent) {
  auto context = createContext();
  PipelineInput input = makeDistanceOneInput(*context, 1025);
  ASSERT_TRUE(input.module);
  const std::string before = print(input.module->getOperation());
  auto prepared = prepareTileExecutionStructure(*input.module, {});
  ASSERT_TRUE(prepared.succeeded());
  auto materialized = materializeExecutionStructure(
      std::move(input.module), std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded());
  EXPECT_TRUE(materialized.materialized->pipelines.empty());
  EXPECT_EQ(print(materialized.materialized->module->getOperation()), before);
}

TEST(ExecutionStructureMaterializationTest,
     FiniteUnrollContainsCrossStageAsyncToken) {
  auto context = createContext();
  PipelineInput input = makeFiniteInput(*context);
  ASSERT_TRUE(input.module);
  auto prepared = prepareTileExecutionStructure(*input.module, {input.choice});
  ASSERT_TRUE(prepared.succeeded())
      << (prepared.failure ? prepared.failure->detail : "");
  auto materialized = materializeExecutionStructure(
      std::move(input.module), std::move(*prepared.prepared));
  ASSERT_TRUE(materialized.succeeded())
      << (materialized.failure ? materialized.failure->detail : "");
  EXPECT_EQ(countLoops(materialized.materialized->module->getOperation()), 0u);
  EXPECT_EQ(materialized.materialized->pipelines.front().operations.size(),
            14u);
}

TEST(ExecutionStructureMaterializationTest,
     AdjacentWritebackPreservesDestinationAndRejectsUnprovenReuse) {
  enum class Case {
    Disjoint,
    InPlace,
    Observer,
    Loop,
    Materialized,
    AliasView,
    PartialOverlap,
    UnknownAlias,
    Mapped,
    DifferentLayout,
    ExtraUse,
    InterveningRead
  };
  for (int64_t extent : {1024, 1025, 1031}) {
    for (Case test : {Case::Disjoint, Case::InPlace, Case::Observer, Case::Loop,
                      Case::Materialized, Case::AliasView, Case::PartialOverlap,
                      Case::UnknownAlias, Case::Mapped, Case::DifferentLayout,
                      Case::ExtraUse, Case::InterveningRead}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(static_cast<int>(test));
      auto context = createContext();
      mlir::Location loc = mlir::UnknownLoc::get(context.get());
      mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
      mlir::OpBuilder builder(context.get());
      auto memory =
          MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Tensor);
      auto type =
          mlir::MemRefType::get({2, extent, 2}, builder.getF16Type(),
                                mlir::MemRefLayoutAttrInterface{}, memory);
      builder.setInsertionPointToStart(module->getBody());
      if (test == Case::UnknownAlias) {
        auto external = builder.create<mlir::func::FuncOp>(
            loc, "buffers", builder.getFunctionType({}, {type, type}));
        external.setPrivate();
      }
      auto function = builder.create<mlir::func::FuncOp>(
          loc, "main", builder.getFunctionType({}, {}));
      auto *entry = function.addEntryBlock();
      builder.setInsertionPointToStart(entry);
      auto region = builder.create<TileRegionOp>(loc, mlir::TypeRange{},
                                                 entry->getArguments());
      auto *body = new mlir::Block();
      region.getBody().push_back(body);
      builder.setInsertionPointToStart(body);
      mlir::Value dest, input;
      if (test == Case::UnknownAlias) {
        auto call = builder.create<mlir::func::CallOp>(
            loc, "buffers", mlir::TypeRange{type, type}, mlir::ValueRange{});
        dest = call.getResult(0);
        input = call.getResult(1);
      } else if (test == Case::PartialOverlap) {
        auto bytes =
            mlir::MemRefType::get({8 * extent + 2}, builder.getI8Type(),
                                  mlir::MemRefLayoutAttrInterface{}, memory);
        auto storage = builder.create<mlir::memref::AllocOp>(loc, bytes);
        auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto two = builder.create<mlir::arith::ConstantIndexOp>(loc, 2);
        dest = builder.create<mlir::memref::ViewOp>(loc, type, storage, zero,
                                                    mlir::ValueRange{});
        input = builder.create<mlir::memref::ViewOp>(loc, type, storage, two,
                                                     mlir::ValueRange{});
      } else {
        auto destType =
            test == Case::DifferentLayout
                ? mlir::MemRefType::get(type.getShape(), type.getElementType(),
                                        mlir::MemRefLayoutAttrInterface{},
                                        MemoryAttr::get(context.get(),
                                                        MemorySpace::SPM,
                                                        MemLayout::Cx))
                : type;
        dest = builder.create<mlir::memref::AllocOp>(loc, destType);
        input = builder.create<mlir::memref::AllocOp>(loc, type);
      }
      mlir::Value observer;
      if (test == Case::Observer || test == Case::AliasView)
        observer = builder.create<mlir::memref::CastOp>(loc, type, dest);
      if (test == Case::AliasView)
        input = observer;
      if (test == Case::InPlace)
        input = dest;
      if (test == Case::Materialized) {
        auto cxType = mlir::MemRefType::get(
            type.getShape(), type.getElementType(),
            mlir::MemRefLayoutAttrInterface{},
            MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Cx));
        dest = builder.create<LayoutMaterializeOp>(loc, cxType, dest);
        input = builder.create<ComputeElementwiseOp>(
            loc, cxType, ComputeElementwiseKind::Add,
            mlir::ValueRange{dest, dest}, mlir::ArrayAttr{});
        type = cxType;
      }
      mlir::scf::ForOp loop;
      if (test == Case::Loop) {
        auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
        auto end = builder.create<mlir::arith::ConstantIndexOp>(loc, 7);
        loop = builder.create<mlir::scf::ForOp>(loc, zero, end, one,
                                                mlir::ValueRange{dest});
        builder.setInsertionPointToStart(loop.getBody());
        dest = loop.getRegionIterArgs().front();
        input = dest;
      }
      mlir::AffineMap identity =
          mlir::AffineMap::getMultiDimIdentityMap(3, context.get());
      mlir::AffineMap inputMap =
          test == Case::Mapped
              ? mlir::AffineMap::getPermutationMap(
                    llvm::ArrayRef<unsigned>{2, 1, 0}, context.get())
              : identity;
      auto maps = builder.getAffineMapArrayAttr({inputMap, inputMap, identity});
      auto value = builder.create<ComputeElementwiseOp>(
          loc, type, ComputeElementwiseKind::Add,
          mlir::ValueRange{input, input}, maps);
      if (test == Case::InterveningRead)
        builder.create<MoveCopyOp>(loc, type, dest, DDRResourceAttr());
      builder.create<MoveCopyIntoOp>(loc, value, dest);
      if (test == Case::ExtraUse)
        builder.create<MoveCopyOp>(loc, type, value, DDRResourceAttr());
      MoveCopyOp reader;
      if (test == Case::Observer)
        reader =
            builder.create<MoveCopyOp>(loc, type, observer, DDRResourceAttr());
      if (loop) {
        builder.create<mlir::scf::YieldOp>(loc, dest);
        builder.setInsertionPointAfter(loop);
      }
      builder.create<TileYieldOp>(loc);
      builder.setInsertionPointAfter(region);
      builder.create<mlir::func::ReturnOp>(loc);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      auto prepared = prepareTileExecutionStructure(*module, {});
      ASSERT_TRUE(prepared.succeeded());
      auto materialized = materializeExecutionStructure(
          std::move(module), std::move(*prepared.prepared));
      ASSERT_TRUE(materialized.succeeded())
          << (materialized.failure ? materialized.failure->detail : "");
      unsigned functional = 0, copies = 0, updates = 0;
      region.walk([&](ComputeElementwiseOp) { ++functional; });
      region.walk([&](MoveCopyIntoOp) { ++copies; });
      region.walk([&](ComputeElementwiseIntoOp update) {
        ++updates;
        EXPECT_EQ(update.getDest(), dest);
      });
      bool eliminate = test == Case::Disjoint || test == Case::InPlace ||
                       test == Case::Observer || test == Case::Loop ||
                       test == Case::Materialized;
      EXPECT_EQ(updates, eliminate ? 1u : 0u);
      EXPECT_EQ(functional, eliminate && test != Case::Materialized ? 0u : 1u);
      EXPECT_EQ(copies, eliminate ? 0u : 1u);
      if (reader) {
        EXPECT_EQ(reader.getSource(), observer);
      }
      if (loop) {
        EXPECT_EQ(
            mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator())
                .getOperand(0),
            dest);
      }
      if (!eliminate)
        continue;
      unsigned beforeAllocations = 0;
      region.walk([&](mlir::memref::AllocOp) { ++beforeAllocations; });
      TileRegionToInstrLoweringSession session(*context);
      ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      unsigned instructions = 0, afterAllocations = 0, transfers = 0;
      InstrElementwiseOp firstInstruction;
      region.walk([&](InstrElementwiseOp instruction) {
        ++instructions;
        if (test == Case::Materialized) {
          if (firstInstruction) {
            EXPECT_EQ(instruction.getDest(),
                      firstInstruction.getInputs().front());
            EXPECT_EQ(instruction.getInputs().front(),
                      firstInstruction.getDest());
          } else {
            firstInstruction = instruction;
          }
        } else {
          EXPECT_EQ(instruction.getDest(), dest);
        }
      });
      region.walk([&](InstrGatherScatterOp) { ++transfers; });
      region.walk([&](mlir::memref::AllocOp) { ++afterAllocations; });
      EXPECT_EQ(instructions, test == Case::Materialized ? 2u : 1u);
      EXPECT_EQ(transfers,
                test == Case::Observer || test == Case::Materialized ? 1u : 0u);
      EXPECT_EQ(afterAllocations,
                beforeAllocations + (test == Case::Materialized ? 2u
                                     : test == Case::Observer   ? 1u
                                                                : 0u));
      ASSERT_TRUE(mlir::succeeded(
          rebuildRequiredNCCJoins(*materialized.materialized->module)));
      EXPECT_TRUE(mlir::succeeded(planSPMMemoryModule(
          *materialized.materialized->module, 0, 3 * 1024 * 1024, 16)));
    }
  }
}

TEST(ExecutionStructureMaterializationTest,
     LoopCarriedElementwiseUsesExplicitExistingDestination) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    mlir::Location loc = mlir::UnknownLoc::get(context.get());
    auto module = mlir::ModuleOp::create(loc);
    mlir::OpBuilder moduleBuilder(module.getBodyRegion());
    auto function = moduleBuilder.create<mlir::func::FuncOp>(
        loc, "main",
        moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
    mlir::Block *entry = function.addEntryBlock();
    mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
    auto region = builder.create<TileRegionOp>(loc, mlir::TypeRange{},
                                               mlir::ValueRange{});
    region.getBody().push_back(new mlir::Block());
    mlir::OpBuilder regionBuilder =
        mlir::OpBuilder::atBlockBegin(&region.getBody().front());
    auto type = mlir::MemRefType::get(
        {2, extent, 128}, regionBuilder.getF32Type(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::NCx));
    auto state = regionBuilder.create<mlir::memref::AllocOp>(loc, type);
    auto input = regionBuilder.create<mlir::memref::AllocOp>(loc, type);
    auto lower = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto upper = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 7);
    auto step = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto loop = regionBuilder.create<mlir::scf::ForOp>(loc, lower, upper, step,
                                                       mlir::ValueRange{state});
    if (!loop.getBody()->empty() &&
        mlir::isa<mlir::scf::YieldOp>(loop.getBody()->back()))
      loop.getBody()->back().erase();
    mlir::OpBuilder loopBuilder = mlir::OpBuilder::atBlockEnd(loop.getBody());
    mlir::AffineMap identity =
        mlir::AffineMap::getMultiDimIdentityMap(type.getRank(), context.get());
    auto maps =
        loopBuilder.getAffineMapArrayAttr({identity, identity, identity});
    auto next = loopBuilder.create<ComputeElementwiseOp>(
        loc, type, ComputeElementwiseKind::Add,
        mlir::ValueRange{loop.getRegionIterArgs().front(), input}, maps);
    loopBuilder.create<mlir::scf::YieldOp>(loc, next.getResult());
    regionBuilder.setInsertionPointAfter(loop);
    regionBuilder.create<TileYieldOp>(loc);
    builder.setInsertionPointAfter(region);
    builder.create<mlir::func::ReturnOp>(loc);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

    auto prepared = prepareTileExecutionStructure(module, {});
    ASSERT_TRUE(prepared.succeeded());
    mlir::OwningOpRef<mlir::ModuleOp> owned(module);
    auto materialized = materializeExecutionStructure(
        std::move(owned), std::move(*prepared.prepared));
    ASSERT_TRUE(materialized.succeeded())
        << (materialized.failure ? materialized.failure->detail : "");
    EXPECT_EQ(countLoops(materialized.materialized->module->getOperation()),
              1u);
    unsigned functional = 0;
    ComputeElementwiseIntoOp update;
    materialized.materialized->module->walk(
        [&](ComputeElementwiseOp) { ++functional; });
    materialized.materialized->module->walk(
        [&](ComputeElementwiseIntoOp operation) { update = operation; });
    EXPECT_EQ(functional, 0u);
    ASSERT_TRUE(update);
    auto currentLoop = update->getParentOfType<mlir::scf::ForOp>();
    ASSERT_TRUE(currentLoop);
    EXPECT_EQ(update.getDest(), currentLoop.getRegionIterArgs().front());
    EXPECT_EQ(update.getInputs().front(), update.getDest());
    EXPECT_EQ(
        mlir::cast<mlir::scf::YieldOp>(currentLoop.getBody()->getTerminator())
            .getOperand(0),
        update.getDest());
    unsigned loopAllocations = 0;
    currentLoop.walk([&](mlir::memref::AllocOp) { ++loopAllocations; });
    EXPECT_EQ(loopAllocations, 0u);
    EXPECT_TRUE(
        mlir::succeeded(mlir::verify(*materialized.materialized->module)));
  }
}

TEST(ExecutionStructureMaterializationTest,
     LoopCarriedLayoutChangeUsesExplicitExistingDestination) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    mlir::Location loc = mlir::UnknownLoc::get(context.get());
    auto module = mlir::ModuleOp::create(loc);
    mlir::OpBuilder moduleBuilder(module.getBodyRegion());
    auto function = moduleBuilder.create<mlir::func::FuncOp>(
        loc, "main",
        moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
    mlir::Block *entry = function.addEntryBlock();
    mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
    auto region = builder.create<TileRegionOp>(loc, mlir::TypeRange{},
                                               mlir::ValueRange{});
    region.getBody().push_back(new mlir::Block());
    mlir::OpBuilder regionBuilder =
        mlir::OpBuilder::atBlockBegin(&region.getBody().front());
    auto tensorType = mlir::MemRefType::get(
        {2, extent, 128}, regionBuilder.getF16Type(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Tensor));
    auto cxType = mlir::MemRefType::get(
        {2, extent, 128}, regionBuilder.getF16Type(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Cx));
    auto source = regionBuilder.create<mlir::memref::AllocOp>(loc, tensorType);
    auto state = regionBuilder.create<mlir::memref::AllocOp>(loc, cxType);
    auto lower = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto upper = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 7);
    auto step = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto loop = regionBuilder.create<mlir::scf::ForOp>(loc, lower, upper, step,
                                                       mlir::ValueRange{state});
    if (!loop.getBody()->empty() &&
        mlir::isa<mlir::scf::YieldOp>(loop.getBody()->back()))
      loop.getBody()->back().erase();
    mlir::OpBuilder loopBuilder = mlir::OpBuilder::atBlockEnd(loop.getBody());
    auto next = loopBuilder.create<LayoutMaterializeOp>(loc, cxType, source);
    loopBuilder.create<mlir::scf::YieldOp>(loc, next.getResult());
    regionBuilder.setInsertionPointAfter(loop);
    regionBuilder.create<TileYieldOp>(loc);
    builder.setInsertionPointAfter(region);
    builder.create<mlir::func::ReturnOp>(loc);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

    auto prepared = prepareTileExecutionStructure(module, {});
    ASSERT_TRUE(prepared.succeeded());
    mlir::OwningOpRef<mlir::ModuleOp> owned(module);
    auto materialized = materializeExecutionStructure(
        std::move(owned), std::move(*prepared.prepared));
    ASSERT_TRUE(materialized.succeeded())
        << (materialized.failure ? materialized.failure->detail : "");
    EXPECT_EQ(countLoops(materialized.materialized->module->getOperation()),
              1u);
    unsigned functional = 0;
    MoveCopyIntoOp update;
    materialized.materialized->module->walk(
        [&](LayoutMaterializeOp) { ++functional; });
    materialized.materialized->module->walk(
        [&](MoveCopyIntoOp operation) { update = operation; });
    EXPECT_EQ(functional, 0u);
    ASSERT_TRUE(update);
    auto currentLoop = update->getParentOfType<mlir::scf::ForOp>();
    ASSERT_TRUE(currentLoop);
    EXPECT_EQ(update.getDest(), currentLoop.getRegionIterArgs().front());
    EXPECT_EQ(getWaferMemoryAttr(
                  mlir::cast<mlir::MemRefType>(update.getSource().getType()))
                  .getLayout(),
              MemLayout::Tensor);
    EXPECT_EQ(getWaferMemoryAttr(
                  mlir::cast<mlir::MemRefType>(update.getDest().getType()))
                  .getLayout(),
              MemLayout::Cx);
    EXPECT_EQ(
        mlir::cast<mlir::scf::YieldOp>(currentLoop.getBody()->getTerminator())
            .getOperand(0),
        update.getDest());
    unsigned loopAllocations = 0;
    currentLoop.walk([&](mlir::memref::AllocOp) { ++loopAllocations; });
    EXPECT_EQ(loopAllocations, 0u);
    EXPECT_TRUE(
        mlir::succeeded(mlir::verify(*materialized.materialized->module)));
  }
}

TEST(ExecutionStructureMaterializationTest,
     PreflightRejectsIncompleteAndDependenceReversingChoices) {
  auto context = createContext();
  PipelineInput incomplete = makeDistanceOneInput(*context, 1025);
  ASSERT_TRUE(incomplete.module);
  incomplete.choice.operations.pop_back();
  auto missing =
      prepareTileExecutionStructure(*incomplete.module, {incomplete.choice});
  ASSERT_FALSE(missing.succeeded());
  ASSERT_TRUE(missing.failure);
  EXPECT_EQ(missing.failure->kind,
            ExecutionStructureFailureKind::BrokenContract);

  PipelineInput reversed = makeDistanceOneInput(*context, 1025);
  ASSERT_TRUE(reversed.module);
  reversed.choice.operations.front().stage = 2;
  auto contradiction =
      prepareTileExecutionStructure(*reversed.module, {reversed.choice});
  ASSERT_FALSE(contradiction.succeeded());
  ASSERT_TRUE(contradiction.failure);
  EXPECT_EQ(contradiction.failure->kind,
            ExecutionStructureFailureKind::BrokenContract);
}

TEST(ExecutionStructureMaterializationTest,
     CurrentRotatingAllocationFeedsActualMiniMalloc) {
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

} // namespace
