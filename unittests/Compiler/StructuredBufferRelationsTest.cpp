//===- StructuredBufferRelationsTest.cpp -------------------------------===//

#include "../../lib/Wafer/Compiler/StructuredBufferRelations.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Pipelines/Pipelines.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

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
      {/*structuredNodeId=*/7, copy.getResult()});
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));

  wafer::compiler::detail::StructuredBufferReplacementListener listener(
      relations);
  wafer::TileRegionToInstrLoweringSession loweringSession(*context);
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  wafer::TileRegionOp region = *function.getOps<wafer::TileRegionOp>().begin();
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstr(region, loweringSession, &listener)));
  EXPECT_TRUE(listener.preservedAllRelations());
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
  EXPECT_FALSE(mlir::isa_and_nonnull<wafer::MoveCopyOp>(
      relations.operationResultBuffers.front().buffer.getDefiningOp()));

  ASSERT_TRUE(mlir::succeeded(wafer::compiler::detail::runPassPipeline(
      *module, "test-required-ncc-join-placement",
      [](mlir::OpPassManager &manager) {
        wafer::addRequiredNCCJoinPlacementPass(
            manager.nest<mlir::func::FuncOp>());
      })));
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
      {/*structuredNodeId=*/7, copy.getResult()});
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  wafer::TileRegionOp region = *function.getOps<wafer::TileRegionOp>().begin();
  ASSERT_TRUE(mlir::succeeded(wafer::convertTileRegionToInstr(region)));

  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));

  // The probe/final evidence contract has no silent retain: a relation whose
  // buffer left the current IR stays a typed failure until the caller
  // re-derives or remaps it through an explicit mapping.
  EXPECT_EQ(relations.operationResultBuffers.size(), 1u);
  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
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
      {/*structuredNodeId=*/7, first.getResult()});
  wafer::compiler::detail::StructuredBufferReplacementListener listener(
      relations);
  listener.notifyOperationReplaced(first.getOperation(), second.getResult());
  EXPECT_EQ(relations.operationResultBuffers.front().buffer,
            second.getResult());
  listener.notifyOperationReplaced(second.getOperation(), third.getResult());
  EXPECT_EQ(relations.operationResultBuffers.front().buffer, third.getResult());
  EXPECT_TRUE(listener.preservedAllRelations());
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
      {/*structuredNodeId=*/7, copy.getResult()});
  relations.operandBuffers.push_back({/*structuredNodeId=*/3, copy.getSource()});
  relations.outputBuffers.push_back({/*outputIndex=*/1, copy.getResult()});

  // A complete mapping remaps everything and succeeds.
  mlir::IRMapping completeMapping;
  completeMapping.map(copy.getResult(), copy.getSource());
  completeMapping.map(copy.getSource(), copy.getResult());
  auto complete = wafer::compiler::detail::
      remapStructuredBufferRelationsComplete(relations, completeMapping);
  ASSERT_TRUE(mlir::succeeded(complete));
  EXPECT_EQ(complete->operationResultBuffers.front().buffer,
            copy.getSource());
  EXPECT_EQ(complete->operandBuffers.front().buffer, copy.getResult());
  EXPECT_EQ(complete->outputBuffers.front().buffer, copy.getSource());

  // Any unmapped relation fails the remap and reports the dropped entry.
  mlir::IRMapping partialMapping;
  partialMapping.map(copy.getSource(), copy.getResult());
  wafer::compiler::detail::StructuredRelationRemapIssue issue;
  auto partial = wafer::compiler::detail::
      remapStructuredBufferRelationsComplete(relations, partialMapping, &issue);
  EXPECT_TRUE(mlir::failed(partial));
  ASSERT_EQ(issue.unmappedResultBuffers.size(), 1u);
  EXPECT_EQ(issue.unmappedResultBuffers.front().structuredNodeId, 7u);
  EXPECT_EQ(issue.unmappedOperandBuffers.size(), 0u);
  ASSERT_EQ(issue.unmappedOutputBuffers.size(), 1u);
  EXPECT_EQ(issue.unmappedOutputBuffers.front().outputIndex, 1u);

  // Reusing the caller-owned issue for a later successful remap must not
  // retain stale failure entries from the previous attempt.
  auto recovered = wafer::compiler::detail::
      remapStructuredBufferRelationsComplete(relations, completeMapping,
                                             &issue);
  ASSERT_TRUE(mlir::succeeded(recovered));
  EXPECT_TRUE(issue.empty());

  // The loose remap keeps omitting for one-shot call sites.
  wafer::StructuredMaterializationRelations loose =
      wafer::compiler::detail::remapStructuredBufferRelations(
          relations, partialMapping);
  EXPECT_TRUE(loose.operationResultBuffers.empty());
  EXPECT_EQ(loose.operandBuffers.size(), 1u);
  EXPECT_TRUE(loose.outputBuffers.empty());
}

} // namespace
