//===- StructuredBufferRelationsTest.cpp -------------------------------===//

#include "Wafer/Analysis/Structured/StructuredBufferRelations.h"
#include "Wafer/Analysis/Structured/StructuredNodeUseIndex.h"
#include "Wafer/Driver/CompilationInternal.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileWorkStatistics.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

class StructuredBufferRelationsTest : public ::testing::Test {
protected:
  StructuredBufferRelationsTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> createModule() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %token = arith.constant false
    %unused = wafer.tile.region(%token : i1) -> (i1) {
    ^bb0(%tile_token: i1):
      %source = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      %copy = wafer.tile.copy %source
          : memref<4xf16, #wafer.memory<spm, tensor>>
         -> memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.tile.yield %tile_token : i1
    }
    return
  }
}
)mlir",
                                                   context.get());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredBufferRelationsTest,
       ConversionListenerRetargetsCurrentIRBuffer) {
  mlir::OwningOpRef<mlir::ModuleOp> module = createModule();
  ASSERT_TRUE(module);
  wafer::MoveCopyOp copy;
  module->walk([&](wafer::MoveCopyOp operation) { copy = operation; });
  ASSERT_TRUE(copy);

  wafer::StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back(
      {/*structuredNodeId=*/7, /*resultIndex=*/0, copy.getResult()});
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));

  wafer::compiler::detail::StructuredBufferReplacementListener listener(
      relations);
  wafer::TileRegionToInstrLoweringSession loweringSession(*context, &listener);
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  wafer::TileRegionOp region = *function.getOps<wafer::TileRegionOp>().begin();
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstr(region, loweringSession, &listener)));
  EXPECT_TRUE(listener.finalizeAfterRewrite());
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
  EXPECT_FALSE(mlir::isa_and_nonnull<wafer::MoveCopyOp>(
      relations.operationResultBuffers.front().buffer.getDefiningOp()));
  EXPECT_TRUE(llvm::any_of(
      relations.scratchBuffers,
      [](const wafer::StructuredOperationBufferRelation &relation) {
        return relation.structuredNodeId == 7 &&
               mlir::isa_and_nonnull<mlir::memref::AllocOp>(
                   relation.buffer.getDefiningOp());
      }));
  auto emitted = llvm::find_if(
      relations.operationEmissions,
      [](const wafer::StructuredOperationEmissionRelation &relation) {
        return relation.structuredNodeId == 7 &&
               mlir::isa_and_nonnull<wafer::InstrGatherScatterOp>(
                   relation.operation);
      });
  ASSERT_NE(emitted, relations.operationEmissions.end());
  wafer::compiler::detail::StructuredNodeUseIndex nodeUses(relations);
  EXPECT_EQ(nodeUses.collectNodesUsedBy(emitted->operation),
            (llvm::SmallVector<uint32_t, 4>{7}));

  ASSERT_TRUE(mlir::succeeded(wafer::compiler::detail::runPassPipeline(
      *module, "test-required-ncc-join-placement",
      [](mlir::OpPassManager &manager) {
        manager.nest<mlir::func::FuncOp>().addPass(
            wafer::createPlaceRequiredNCCJoinsPass());
      })));
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
}

TEST_F(StructuredBufferRelationsTest,
       ReusesFrequentlyRepeatedExactDescriptorPlansWithinOneRequest) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %token = arith.constant false
    %unused = wafer.tile.region(%token : i1) -> (i1) {
    ^bb0(%tile_token: i1):
      %source = memref.alloc()
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      %first = wafer.tile.materialize_layout %source
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
         -> memref<2x1025x64xf16, #wafer.memory<spm, ncx>>
      %second = wafer.tile.materialize_layout %source
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
         -> memref<2x1025x64xf16, #wafer.memory<spm, ncx>>
      %third = wafer.tile.materialize_layout %source
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
         -> memref<2x1025x64xf16, #wafer.memory<spm, ncx>>
      %fourth = wafer.tile.materialize_layout %source
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
         -> memref<2x1025x64xf16, #wafer.memory<spm, ncx>>
      wafer.tile.yield %tile_token : i1
    }
    return
  }
}
)mlir",
                                              context.get());
  ASSERT_TRUE(module);
  auto work = std::make_shared<wafer::support::CompileWorkStatisticsSession>();
  wafer::support::ScopedCompileWorkStatisticsActivation activation(work);
  wafer::TileRegionToInstrLoweringSession loweringSession(*context);
  wafer::TileRegionOp region;
  module->walk([&](wafer::TileRegionOp candidate) { region = candidate; });
  ASSERT_TRUE(region);
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstr(region, loweringSession)));
  wafer::support::CompileWorkStatistics counts = work->snapshot();
  EXPECT_EQ(counts.tileToInstructionLowerings, 1u);
  EXPECT_EQ(counts.relationDescriptorPlannings, 3u);
}

TEST_F(StructuredBufferRelationsTest,
       ConversionListenerOwnsSupportCopyThroughTypedStoreDataflow) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %dest: memref<1x2x1024xf16, #wafer.memory<ddr, tensor>>) {
    %token = arith.constant false
    %unused = wafer.tile.region(
        %token, %dest : i1,
        memref<1x2x1024xf16, #wafer.memory<ddr, tensor>>) -> (i1) {
    ^bb0(%tile_token: i1,
         %tile_dest: memref<1x2x1024xf16,
             #wafer.memory<ddr, tensor>>):
      %source = memref.alloc()
          : memref<1x2x1024xf16, #wafer.memory<spm, tensor>>
      %copy = wafer.tile.copy %source
          : memref<1x2x1024xf16, #wafer.memory<spm, tensor>>
         -> memref<1x2x1024xf16, #wafer.memory<spm, tensor>>
      wafer.tile.store %copy, %tile_dest
          : memref<1x2x1024xf16, #wafer.memory<spm, tensor>>
         -> memref<1x2x1024xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %tile_token : i1
    }
    return
  }
}
)mlir",
                                              context.get());
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  wafer::TileRegionOp region = *function.getOps<wafer::TileRegionOp>().begin();

  wafer::StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back(
      {/*structuredNodeId=*/7, /*resultIndex=*/0, function.getArgument(0)});
  wafer::compiler::detail::StructuredBufferReplacementListener listener(
      relations);
  wafer::TileRegionToInstrLoweringSession loweringSession(*context, &listener);
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstr(region, loweringSession, &listener)));
  ASSERT_TRUE(listener.finalizeAfterRewrite())
      << listener.getFailureReason().str();
  EXPECT_TRUE(llvm::any_of(
      relations.scratchBuffers,
      [](const wafer::StructuredOperationBufferRelation &relation) {
        return relation.structuredNodeId == 7 &&
               mlir::isa_and_nonnull<mlir::memref::AllocOp>(
                   relation.buffer.getDefiningOp());
      }));
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
}

TEST_F(StructuredBufferRelationsTest,
       ConversionListenerDropsOnlyAnExplicitlyErasedDeadAllocationRelation) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %token = arith.constant false
    %unused = wafer.tile.region(%token : i1) -> (i1) {
    ^bb0(%tile_token: i1):
      %dead = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      %zero = arith.constant 0.0 : f16
      wafer.tile.fill %dead, %zero
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.yield %tile_token : i1
    }
    return
  }
}
)mlir",
                                              context.get());
  ASSERT_TRUE(module);
  mlir::memref::AllocOp allocation;
  module->walk(
      [&](mlir::memref::AllocOp operation) { allocation = operation; });
  ASSERT_TRUE(allocation);

  wafer::StructuredMaterializationRelations relations;
  relations.operandBuffers.push_back(
      {/*structuredNodeId=*/7, allocation.getResult()});
  wafer::compiler::detail::StructuredBufferReplacementListener listener(
      relations);
  wafer::TileRegionToInstrLoweringSession loweringSession(*context, &listener);
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  wafer::TileRegionOp region = *function.getOps<wafer::TileRegionOp>().begin();
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstr(region, loweringSession, &listener)));
  EXPECT_TRUE(listener.finalizeAfterRewrite());
  EXPECT_TRUE(relations.operandBuffers.empty());
  EXPECT_EQ(module->getOperation()->getNumRegions(), 1u);
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
}

TEST_F(StructuredBufferRelationsTest,
       CurrentIRCheckRejectsUntrackedReplacementWithoutDereference) {
  mlir::OwningOpRef<mlir::ModuleOp> module = createModule();
  ASSERT_TRUE(module);
  wafer::MoveCopyOp copy;
  module->walk([&](wafer::MoveCopyOp operation) { copy = operation; });
  ASSERT_TRUE(copy);

  wafer::StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back(
      {/*structuredNodeId=*/7, /*resultIndex=*/0, copy.getResult()});
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  wafer::TileRegionOp region = *function.getOps<wafer::TileRegionOp>().begin();
  ASSERT_TRUE(mlir::succeeded(wafer::convertTileRegionToInstr(region)));

  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));

  // The check itself has no silent retain: a relation whose buffer left the
  // current IR stays a typed failure until the caller explicitly completes a
  // cleanup epoch.
  EXPECT_EQ(relations.operationResultBuffers.size(), 1u);
  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
  wafer::compiler::detail::retainCurrentStructuredBufferRelations(
      module->getOperation(), relations);
  EXPECT_TRUE(relations.operationResultBuffers.empty());
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
}

TEST_F(StructuredBufferRelationsTest,
       AttributionRelationsRebaseToOneCurrentStorageRoot) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %storage = memref.alloc()
        : memref<1024xf16, #wafer.memory<spm, tensor>>
    %view = memref.subview %storage[0] [1024] [1]
        : memref<1024xf16, #wafer.memory<spm, tensor>>
          to memref<1024xf16, strided<[1]>, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                              context.get());
  ASSERT_TRUE(module);
  mlir::memref::AllocOp allocation;
  mlir::memref::SubViewOp view;
  module->walk(
      [&](mlir::memref::AllocOp operation) { allocation = operation; });
  module->walk([&](mlir::memref::SubViewOp operation) { view = operation; });
  ASSERT_TRUE(allocation && view);

  wafer::StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back(
      {/*structuredNodeId=*/7, /*resultIndex=*/0, view.getResult()});
  relations.operandBuffers.push_back(
      {/*structuredNodeId=*/3, view.getResult()});
  relations.outputBuffers.push_back({/*outputIndex=*/0, view.getResult()});
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::rebaseStructuredBufferRelationsToStorageRoots(
          relations)));
  EXPECT_EQ(relations.operationResultBuffers.front().buffer,
            allocation.getResult());
  EXPECT_EQ(relations.operandBuffers.front().buffer, allocation.getResult());
  EXPECT_EQ(relations.outputBuffers.front().buffer, allocation.getResult());
}

TEST_F(StructuredBufferRelationsTest,
       ReplacementListenerTracksMultipleReplacementHops) {
  mlir::OwningOpRef<mlir::ModuleOp> module = createModule();
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  auto first = *function.getOps<mlir::arith::ConstantOp>().begin();
  mlir::OpBuilder builder(function.getBody().front().getTerminator());
  auto second = builder.create<mlir::arith::ConstantIntOp>(
      first.getLoc(), /*value=*/true, /*width=*/1);
  auto third = builder.create<mlir::arith::ConstantIntOp>(
      first.getLoc(), /*value=*/false, /*width=*/1);

  wafer::StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back(
      {/*structuredNodeId=*/7, /*resultIndex=*/0, first.getResult()});
  wafer::compiler::detail::StructuredBufferReplacementListener listener(
      relations);
  listener.notifyOperationReplaced(first.getOperation(), second.getResult());
  EXPECT_EQ(relations.operationResultBuffers.front().buffer,
            second.getResult());
  listener.notifyOperationReplaced(second.getOperation(), third.getResult());
  EXPECT_EQ(relations.operationResultBuffers.front().buffer, third.getResult());
  EXPECT_TRUE(listener.finalizeAfterRewrite());
}

TEST_F(StructuredBufferRelationsTest,
       StrictRemapFailsClosedOnUnmappedRelationAndReportsTheIssue) {
  mlir::OwningOpRef<mlir::ModuleOp> module = createModule();
  ASSERT_TRUE(module);
  wafer::MoveCopyOp copy;
  module->walk([&](wafer::MoveCopyOp operation) { copy = operation; });
  ASSERT_TRUE(copy);

  wafer::StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back(
      {/*structuredNodeId=*/7, /*resultIndex=*/0, copy.getResult()});
  relations.operandBuffers.push_back(
      {/*structuredNodeId=*/3, copy.getSource()});
  relations.outputBuffers.push_back({/*outputIndex=*/1, copy.getResult()});

  // A complete mapping remaps everything and succeeds.
  mlir::IRMapping completeMapping;
  completeMapping.map(copy.getResult(), copy.getSource());
  completeMapping.map(copy.getSource(), copy.getResult());
  auto complete =
      wafer::compiler::detail::remapStructuredBufferRelationsComplete(
          relations, completeMapping);
  ASSERT_TRUE(mlir::succeeded(complete));
  EXPECT_EQ(complete->operationResultBuffers.front().buffer, copy.getSource());
  EXPECT_EQ(complete->operandBuffers.front().buffer, copy.getResult());
  EXPECT_EQ(complete->outputBuffers.front().buffer, copy.getSource());

  // Any unmapped relation fails the remap and reports the dropped entry.
  mlir::IRMapping partialMapping;
  partialMapping.map(copy.getSource(), copy.getResult());
  wafer::compiler::detail::StructuredRelationRemapIssue issue;
  auto partial =
      wafer::compiler::detail::remapStructuredBufferRelationsComplete(
          relations, partialMapping, &issue);
  EXPECT_TRUE(mlir::failed(partial));
  ASSERT_EQ(issue.unmappedResultBuffers.size(), 1u);
  EXPECT_EQ(issue.unmappedResultBuffers.front().structuredNodeId, 7u);
  EXPECT_EQ(issue.unmappedResultBuffers.front().resultIndex, 0u);
  EXPECT_EQ(issue.unmappedOperandBuffers.size(), 0u);
  ASSERT_EQ(issue.unmappedOutputBuffers.size(), 1u);
  EXPECT_EQ(issue.unmappedOutputBuffers.front().outputIndex, 1u);

  // Reusing the caller-owned issue for a later successful remap must not
  // retain stale failure entries from the previous attempt.
  auto recovered =
      wafer::compiler::detail::remapStructuredBufferRelationsComplete(
          relations, completeMapping, &issue);
  ASSERT_TRUE(mlir::succeeded(recovered));
  EXPECT_TRUE(issue.empty());

  // The loose remap keeps omitting for one-shot call sites.
  wafer::StructuredMaterializationRelations loose =
      wafer::compiler::detail::remapStructuredBufferRelations(relations,
                                                              partialMapping);
  EXPECT_TRUE(loose.operationResultBuffers.empty());
  EXPECT_EQ(loose.operandBuffers.size(), 1u);
  EXPECT_TRUE(loose.outputBuffers.empty());
}

TEST_F(StructuredBufferRelationsTest,
       StructuredNodeUseIndexMatchesSharedStorageRoots) {
  mlir::OwningOpRef<mlir::ModuleOp> module = createModule();
  ASSERT_TRUE(module);
  wafer::MoveCopyOp copy;
  module->walk([&](wafer::MoveCopyOp operation) { copy = operation; });
  ASSERT_TRUE(copy);

  wafer::StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back(
      {/*structuredNodeId=*/7, /*resultIndex=*/0, copy.getResult()});
  relations.operandBuffers.push_back(
      {/*structuredNodeId=*/3, copy.getSource()});

  wafer::compiler::detail::StructuredNodeUseIndex nodeUses(relations);
  EXPECT_EQ(nodeUses.collectNodesUsedBy(copy.getOperation()),
            (llvm::SmallVector<uint32_t, 4>{3, 7}));
}

TEST_F(StructuredBufferRelationsTest,
       StructuredNodeUseIndexMemoizesLongSharedViewPrefixes) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%buffer: memref<4xf16, #wafer.memory<spm, tensor>>) {
    return
  }
}
)mlir",
                                              context.get());
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  mlir::Block &body = function.getBody().front();
  mlir::OpBuilder builder(body.getTerminator());
  mlir::Value value = function.getArgument(0);
  wafer::StructuredMaterializationRelations relations;
  mlir::memref::CastOp last;
  for (unsigned index = 0; index < 1024; ++index) {
    last = builder.create<mlir::memref::CastOp>(function.getLoc(),
                                                value.getType(), value);
    value = last.getResult();
    relations.operationResultBuffers.push_back(
        {/*structuredNodeId=*/9, /*resultIndex=*/0, value});
  }

  wafer::compiler::detail::StructuredNodeUseIndex nodeUses(relations);
  EXPECT_EQ(nodeUses.collectNodesUsedBy(last.getOperation()),
            (llvm::SmallVector<uint32_t, 4>{9}));
}

} // namespace
