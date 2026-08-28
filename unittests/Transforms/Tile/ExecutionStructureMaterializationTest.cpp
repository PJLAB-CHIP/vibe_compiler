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
  loopBuilder.create<InstrFillOp>(loc, allocation.getResult(), zero,
                                  FillDomainAttr(), NCCWorker::Worker0);
  regionBuilder.setInsertionPointAfter(loop);
  regionBuilder.create<mlir::memref::DeallocOp>(loc, allocation);
  regionBuilder.create<TileYieldOp>(loc);
  builder.setInsertionPointAfter(region);
  builder.create<mlir::func::ReturnOp>(loc);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

  StructuredMaterializationRelations relations;
  relations.scratchBuffers.push_back({0, allocation});
  mlir::OwningOpRef<mlir::ModuleOp> owned(module);
  auto rotated = materializeRotatingAllocations(
      std::move(owned), {{allocation, loop, /*multiplicity=*/2}}, relations);
  ASSERT_TRUE(rotated.succeeded())
      << (rotated.failure ? rotated.failure->detail : "");
  ASSERT_EQ(rotated.materialized->slots.size(), 2u);
  EXPECT_EQ(relations.scratchBuffers.size(), 2u);
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
