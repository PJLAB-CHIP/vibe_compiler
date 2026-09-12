//===- StructuredBufferRelationsTest.cpp -------------------------------===//

#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

class StructuredBufferRelationsTest : public ::testing::Test {
protected:
  StructuredBufferRelationsTest() {
    registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        text, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredBufferRelationsTest,
       RebuildKeepsDistinctOwnersAndDeduplicatesRepeatedOperands) {
  for (int64_t length : {1024, 1025, 1031}) {
    auto module = parse(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry() {
      %buffer = memref.alloc()
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      return
    }
  }
}
)mlir");
    ASSERT_TRUE(module);
    mlir::memref::AllocOp allocation;
    module->walk([&](mlir::memref::AllocOp op) { allocation = op; });
    auto type = allocation.getType();
    allocation.getResult().setType(
        mlir::MemRefType::get({2, length, 64}, type.getElementType(),
                              type.getLayout(), type.getMemorySpace()));
    mlir::OpBuilder builder(allocation->getBlock()->getTerminator());
    llvm::SmallVector<mlir::Operation *> copies;
    for (unsigned index = 0; index < 1024; ++index)
      copies.push_back(builder.create<mlir::memref::CopyOp>(
          allocation.getLoc(), allocation, allocation));
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

    StructuredMaterializationRelations relations;
    for (unsigned repetition = 0; repetition < 2; ++repetition) {
      rebuildCurrentBufferOwnerRelations(*module, relations);
      ASSERT_EQ(relations.buffers.size(), copies.size() + 1);
      EXPECT_EQ(relations.buffers.front().owner, allocation);
      EXPECT_EQ(relations.buffers.front().role,
                MaterializedBufferRole::Scratch);
      for (auto [index, copy] : llvm::enumerate(copies)) {
        const auto &relation = relations.buffers[index + 1];
        EXPECT_EQ(relation.owner, copy);
        EXPECT_EQ(relation.buffer, allocation.getResult());
        EXPECT_EQ(relation.role, MaterializedBufferRole::Movement);
      }
      EXPECT_TRUE(mlir::succeeded(
          checkStructuredBufferRelationsCurrent(*module, relations)));
    }
    copies.back()->erase();
    rebuildCurrentBufferOwnerRelations(*module, relations);
    EXPECT_EQ(relations.buffers.size(), copies.size());
    EXPECT_TRUE(mlir::succeeded(
        checkStructuredBufferRelationsCurrent(*module, relations)));
  }
}

TEST_F(StructuredBufferRelationsTest,
       LoopStorageRootsIncludeBackedgesIndependentOfQueryOrder) {
  auto module = parse(R"mlir(
module {
  func.func @entry() {
    %a = memref.alloc() : memref<2x1031x64xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<2x1031x64xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %end = arith.constant 33 : index
    %results:2 = scf.for %iv = %zero to %end step %one
        iter_args(%left = %a, %right = %b)
        -> (memref<2x1031x64xf16, #wafer.memory<spm, tensor>>,
            memref<2x1031x64xf16, #wafer.memory<spm, tensor>>) {
      scf.yield %right, %left
          : memref<2x1031x64xf16, #wafer.memory<spm, tensor>>,
            memref<2x1031x64xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::Value> allocations;
  mlir::scf::ForOp loop;
  module->walk([&](mlir::memref::AllocOp op) { allocations.push_back(op); });
  module->walk([&](mlir::scf::ForOp op) { loop = op; });
  ASSERT_EQ(allocations.size(), 2u);
  ASSERT_TRUE(loop);
  for (bool reverse : {false, true}) {
    StorageRootMemo memo;
    llvm::SmallVector<mlir::Value> queries;
    queries.append(loop.getRegionIterArgs().begin(),
                   loop.getRegionIterArgs().end());
    queries.append(loop.getResults().begin(), loop.getResults().end());
    if (reverse)
      std::reverse(queries.begin(), queries.end());
    for (auto value : queries) {
      const auto &roots = memo.getStorageRoots(value);
      ASSERT_EQ(roots.size(), 2u);
      for (auto allocation : allocations)
        EXPECT_TRUE(roots.contains(allocation));
    }
  }
}

TEST_F(StructuredBufferRelationsTest,
       ListenerRetargetsActualValueWithoutSourceNodeParity) {
  auto module = parse(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry() {
      %source = memref.alloc()
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      %first = wafer.tile.materialize_layout %source
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
         -> memref<2x1024x64xf16, #wafer.memory<spm, ncx>>
      %second = wafer.tile.materialize_layout %first
          : memref<2x1024x64xf16, #wafer.memory<spm, ncx>>
         -> memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      %c0 = arith.constant 0 : index
      %value = memref.load %second[%c0, %c0, %c0]
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      return
    }
  }
}
)mlir");
  ASSERT_TRUE(module);
  llvm::SmallVector<LayoutMaterializeOp, 2> layouts;
  module->walk([&](LayoutMaterializeOp op) { layouts.push_back(op); });
  ASSERT_EQ(layouts.size(), 2u);
  mlir::memref::LoadOp load;
  module->walk([&](mlir::memref::LoadOp op) { load = op; });
  ASSERT_TRUE(load);

  StructuredMaterializationRelations relations;
  relations.buffers.push_back(
      {load, layouts.front().getResult(), MaterializedBufferRole::Operand});
  StructuredBufferReplacementListener listener(relations);
  listener.notifyOperationReplaced(layouts.front(), layouts.back().getResult());
  EXPECT_TRUE(listener.finalizeAfterRewrite());
  ASSERT_EQ(relations.buffers.size(), 1u);
  EXPECT_EQ(relations.buffers.front().buffer, layouts.back().getResult());
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));
}

TEST_F(StructuredBufferRelationsTest,
       LoweringRecorderTransfersScratchToLiveActualOperation) {
  auto module = parse(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry() {
      %zero = arith.constant 0.000000e+00 : f16
      %buffer = memref.alloc()
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      wafer.tile.fill %buffer, %zero
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.fill %buffer, %zero
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>, f16
      return
    }
  }
}
)mlir");
  ASSERT_TRUE(module);
  ComputeFillOp source;
  InstrFillOp lowered;
  mlir::memref::AllocOp allocation;
  module->walk([&](ComputeFillOp op) { source = op; });
  module->walk([&](InstrFillOp op) { lowered = op; });
  module->walk([&](mlir::memref::AllocOp op) { allocation = op; });
  ASSERT_TRUE(source && lowered && allocation);

  StructuredMaterializationRelations relations;
  StructuredBufferReplacementListener listener(relations);
  listener.recordScratchAllocation(source, allocation);
  listener.recordLoweredOperation(source, lowered);
  EXPECT_TRUE(listener.finalizeAfterRewrite())
      << listener.getFailureReason().str();
  EXPECT_TRUE(llvm::any_of(relations.buffers, [&](const auto &relation) {
    return relation.owner == lowered && relation.buffer == allocation &&
           relation.role == MaterializedBufferRole::Scratch;
  }));
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));
}

TEST_F(StructuredBufferRelationsTest,
       CurrentViewRelationRemainsLiveAndFailsOnStaleOwner) {
  auto module = parse(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry() {
      %buffer = memref.alloc()
          : memref<2x1031x64xf16, #wafer.memory<spm, tensor>>
      %view = memref.subview %buffer[0, 0, 0] [1, 1031, 64] [1, 1, 1]
          : memref<2x1031x64xf16, #wafer.memory<spm, tensor>>
            to memref<1x1031x64xf16, strided<[65984, 64, 1]>,
                      #wafer.memory<spm, tensor>>
      %c0 = arith.constant 0 : index
      %value = memref.load %view[%c0, %c0, %c0]
          : memref<1x1031x64xf16, strided<[65984, 64, 1]>,
                   #wafer.memory<spm, tensor>>
      return
    }
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::memref::SubViewOp view;
  mlir::memref::LoadOp load;
  module->walk([&](mlir::memref::SubViewOp op) { view = op; });
  module->walk([&](mlir::memref::LoadOp op) { load = op; });
  ASSERT_TRUE(view && load);

  StructuredMaterializationRelations relations;
  relations.buffers.push_back(
      {load, view.getResult(), MaterializedBufferRole::Operand});
  EXPECT_EQ(relations.buffers.front().buffer, view.getResult());
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));

  load->erase();
  EXPECT_TRUE(
      mlir::failed(checkStructuredBufferRelationsCurrent(*module, relations)));
  retainCurrentStructuredBufferRelations(*module, relations);
  EXPECT_TRUE(relations.buffers.empty());
}

TEST_F(StructuredBufferRelationsTest,
       ConditionalBufferRelationRetainsAllCurrentStorageRoots) {
  auto module = parse(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%condition: i1) {
      %selected = scf.if %condition
          -> (memref<2x1025x64xf16, #wafer.memory<spm, tensor>>) {
        %then = memref.alloc()
            : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
        scf.yield %then
            : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      } else {
        %else = memref.alloc()
            : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
        scf.yield %else
            : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      }
      %c0 = arith.constant 0 : index
      %value = memref.load %selected[%c0, %c0, %c0]
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      return
    }
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::scf::IfOp branch;
  mlir::memref::LoadOp load;
  module->walk([&](mlir::scf::IfOp op) { branch = op; });
  module->walk([&](mlir::memref::LoadOp op) { load = op; });
  ASSERT_TRUE(branch && load);

  StructuredMaterializationRelations relations;
  relations.buffers.push_back(
      {load, branch.getResult(0), MaterializedBufferRole::Operand});
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));

  StorageRootMemo roots;
  EXPECT_EQ(roots.getStorageRoots(branch.getResult(0)).size(), 2u);
  EXPECT_EQ(relations.buffers.front().buffer, branch.getResult(0));
}

TEST_F(StructuredBufferRelationsTest,
       CrossTileAndOutputEndpointsMustRemainLiveAndUnique) {
  auto module = parse(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x1024x64xf16>) {
      %source = wafer.tile.region(
          %input : tensor<2x1024x64xf16>)
          -> (tensor<2x1024x64xf16>) {
      ^bb0(%arg0: tensor<2x1024x64xf16>):
        wafer.tile.yield %arg0 : tensor<2x1024x64xf16>
      }
      return
    }
  }
  wafer.tile.module card_id = 0 tile_id = 1 {
    func.func @entry(%input: tensor<2x1024x64xf16>) {
      %result = wafer.tile.region(
          %input : tensor<2x1024x64xf16>)
          -> (tensor<2x1024x64xf16>) {
      ^bb0(%arg0: tensor<2x1024x64xf16>):
        wafer.tile.yield %arg0 : tensor<2x1024x64xf16>
      }
      return
    }
  }
}
)mlir");
  ASSERT_TRUE(module);
  llvm::SmallVector<TileRegionOp, 2> regions;
  module->walk([&](TileRegionOp op) { regions.push_back(op); });
  ASSERT_EQ(regions.size(), 2u);
  StructuredMaterializationRelations relations;
  relations.boundaryRelations.push_back(
      {regions[0].getResult(0), regions[1].getBody().getArgument(0)});
  relations.structuralOutputs.push_back({0, regions[1].getResult(0)});
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));
  relations.boundaryRelations.push_back(relations.boundaryRelations.front());
  EXPECT_TRUE(
      mlir::failed(checkStructuredBufferRelationsCurrent(*module, relations)));
}

} // namespace
