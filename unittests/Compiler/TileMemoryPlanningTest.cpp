//===- TileMemoryPlanningTest.cpp ---------------------------------===//

#include "../../lib/Wafer/Compiler/TileMemoryPlanning.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileWorkStatistics.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

class TileMemoryPlanningTest : public ::testing::Test {
protected:
  TileMemoryPlanningTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> candidateWithSPMElements(int64_t elements) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  func.func @main(%boundary: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    %unused = wafer.tile.region(%boundary
        : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%ddr: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>):
      %zero = arith.constant 0.000000e+00 : f16
      %spm = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %ddr into %spm
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
        into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.fill %spm, %zero
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.yield %ddr
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> candidateWithDDRPlacement() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    return
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> candidateWithSPMPlacement() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> candidateWithStaleMidRegionJoin() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %boundary: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %unused = wafer.tile.region(%boundary
        : memref<4xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%ddr: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %a = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      %b = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      %zero = arith.constant 0.000000e+00 : f16
      wafer.instr.fill %a, %zero
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.ncc_join [0]
      wafer.instr.wdma %a to %ddr
          {byte_count = 8 : i64, dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>, inner_bytes = 8 : i64}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      wafer.instr.fill %b, %zero
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.wdma %b to %ddr
          {byte_count = 8 : i64, dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>, inner_bytes = 8 : i64}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %ddr
          : memref<4xf16, #wafer.memory<ddr, tensor>>
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

TEST_F(TileMemoryPlanningTest,
       RejectsTileDataflowAtCanonicalInstrActionBoundary) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      candidateWithSPMElements(/*elements=*/128);
  ASSERT_TRUE(module);
  ASSERT_TRUE(wafer::containsTileDataflowOperations(module->getOperation()));

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(diagnostics);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });
  wafer::compiler::detail::TileMemoryPlanningFailure failure;
  auto memoryPlanned = wafer::compiler::detail::planTileMemory(
      std::move(module), &failure);
  EXPECT_TRUE(mlir::failed(memoryPlanned));
  EXPECT_EQ(
      failure.kind,
      wafer::compiler::detail::TileMemoryPlanningFailureKind::Contract);
  EXPECT_NE(diagnostics.find(
                "tile_memory_planning_requires_canonical_instr_ir"),
            std::string::npos)
      << diagnostics;
}

TEST_F(TileMemoryPlanningTest, ReportsSPMFailureForOwnedTileModule) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      candidateWithSPMElements(/*elements=*/2'000'000);
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(wafer::convertTileRegionToInstrModule(*module)));
  ASSERT_FALSE(wafer::containsTileDataflowOperations(module->getOperation()));

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(diagnostics);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });
  wafer::compiler::detail::TileMemoryPlanningFailure failure;
  auto memoryPlanned = wafer::compiler::detail::planTileMemory(
      std::move(module), &failure);
  EXPECT_TRUE(mlir::failed(memoryPlanned));
  EXPECT_EQ(failure.kind,
            wafer::compiler::detail::TileMemoryPlanningFailureKind::
                SPMAllocation);
  EXPECT_TRUE(failure.spmCapacityOverflow);
  EXPECT_EQ(failure.spmPlanningFailureKind,
            wafer::SPMMemoryPlanningFailureKind::CapacityOverflow);
  EXPECT_TRUE(static_cast<bool>(failure.spmLargestDemandLocation));
  EXPECT_EQ(failure.spmLargestDemandBytes, 4'000'000u);
  ASSERT_FALSE(failure.spmLargestDemands.empty());
  EXPECT_EQ(failure.spmLargestDemands.front().location,
            failure.spmLargestDemandLocation);
  EXPECT_EQ(failure.spmLargestDemands.front().bytes,
            failure.spmLargestDemandBytes);
  ASSERT_EQ(failure.spmCapacityConflictDemands.size(), 1u);
  EXPECT_EQ(failure.spmCapacityConflictDemands.front().location,
            failure.spmLargestDemandLocation);
  EXPECT_EQ(failure.spmCapacityConflictDemands.front().bytes,
            failure.spmLargestDemandBytes);
  ASSERT_EQ(failure.spmIndividuallyOversizedDemands.size(), 1u);
  EXPECT_EQ(failure.spmIndividuallyOversizedDemands.front().location,
            failure.spmLargestDemandLocation);
  EXPECT_EQ(failure.spmIndividuallyOversizedDemands.front().bytes,
            failure.spmLargestDemandBytes);
  EXPECT_NE(diagnostics.find("capacity_overflow"), std::string::npos)
      << diagnostics;
  EXPECT_NE(diagnostics.find("capacity_conflict_demands=1"), std::string::npos)
      << diagnostics;
  EXPECT_NE(diagnostics.find("individually_oversized_demands=1"),
            std::string::npos)
      << diagnostics;
}

TEST_F(TileMemoryPlanningTest,
       RejectsCardPlacementBeforeTileMemoryPlanning) {
  mlir::OwningOpRef<mlir::ModuleOp> module = candidateWithDDRPlacement();
  ASSERT_TRUE(module);

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(diagnostics);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });
  wafer::compiler::detail::TileMemoryPlanningFailure failure;
  auto memoryPlanned = wafer::compiler::detail::planTileMemory(
      std::move(module), &failure);
  EXPECT_TRUE(mlir::failed(memoryPlanned));
  EXPECT_EQ(failure.kind,
            wafer::compiler::detail::TileMemoryPlanningFailureKind::
                PreexistingPlacementFacts);
  EXPECT_NE(diagnostics.find(
                "tile_memory_planning_contains_preexisting_placement_facts"),
            std::string::npos)
      << diagnostics;
}

TEST_F(TileMemoryPlanningTest,
       RejectsPreexistingSPMAssignmentInsteadOfScrubbingIt) {
  mlir::OwningOpRef<mlir::ModuleOp> module = candidateWithSPMPlacement();
  ASSERT_TRUE(module);

  wafer::compiler::detail::TileMemoryPlanningFailure failure;
  auto memoryPlanned = wafer::compiler::detail::planTileMemory(
      std::move(module), &failure);
  EXPECT_TRUE(mlir::failed(memoryPlanned));
  EXPECT_EQ(failure.kind,
            wafer::compiler::detail::TileMemoryPlanningFailureKind::
                PreexistingPlacementFacts);
}

TEST_F(TileMemoryPlanningTest,
       RebuildsRegionRootJoinAfterBufferizationFromCurrentEffects) {
  mlir::OwningOpRef<mlir::ModuleOp> module = candidateWithStaleMidRegionJoin();
  ASSERT_TRUE(module);
  auto workSession =
      std::make_shared<wafer::support::CompileWorkStatisticsSession>();
  wafer::support::ScopedCompileWorkStatisticsActivation workActivation(
      workSession);
  auto memoryPlanned =
      wafer::compiler::detail::planTileMemory(std::move(module));
  ASSERT_TRUE(mlir::succeeded(memoryPlanned));

  llvm::SmallVector<wafer::SyncNCCJoinOp, 2> joins;
  (*memoryPlanned)->walk([&](wafer::SyncNCCJoinOp join) {
    joins.push_back(join);
  });
  ASSERT_EQ(joins.size(), 1u);
  EXPECT_EQ(joins.front().getParticipants(), (llvm::ArrayRef<int64_t>{0}));
  EXPECT_TRUE(mlir::isa<wafer::TileYieldOp>(joins.front()->getNextNode()));
  EXPECT_TRUE(joins.front()->getParentOfType<wafer::TileRegionOp>());
  const wafer::support::CompileWorkStatistics work = workSession->snapshot();
  EXPECT_EQ(work.tileMemoryPlanningInvocations, 1u);
  EXPECT_EQ(work.tileToInstructionLowerings, 0u);
  EXPECT_EQ(work.spmPlanningInvocations, 1u);
}

TEST_F(TileMemoryPlanningTest,
       ExactSelectedBufferMaterializationUsesOwnedModuleInPlace) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %iv = %c0 to %c2 step %c1 {
      %slot = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %input to %slot
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %slot, %slot into %slot
          : memref<4xf16, #wafer.memory<spm, tensor>>,
            memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
      scf.yield
    }
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  mlir::func::FuncOp function =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(function);
  mlir::scf::ForOp loop;
  mlir::memref::AllocOp slot;
  function.walk([&](mlir::scf::ForOp candidate) { loop = candidate; });
  function.walk([&](mlir::memref::AllocOp allocation) { slot = allocation; });
  ASSERT_TRUE(loop);
  ASSERT_TRUE(slot);

  wafer::StructuredMaterializationRelations relations;
  relations.operandBuffers.push_back(
      {/*structuredNodeId=*/1, function.getArgument(0)});
  relations.operationResultBuffers.push_back(
      {/*structuredNodeId=*/2, slot.getResult()});
  wafer::compiler::detail::SelectedBufferRequest request;
  request.producerNode = 1;
  request.consumerNode = 2;
  request.bufferCount = 2;
  request.requireLocalDataflow = true;

  mlir::Operation *originalModule = module->getOperation();
  unsigned slotAllocations = 0;
  wafer::compiler::detail::SelectedBufferMaterializationFailure failure;
  auto materialized = wafer::compiler::detail::materializeSelectedBuffering(
      std::move(module), llvm::ArrayRef{request}, std::move(relations),
      &failure);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failure.detail;
  module = std::move(materialized->module);
  relations = std::move(materialized->materializationRelations);
  slotAllocations = materialized->slotAllocationCount;

  EXPECT_EQ(module->getOperation(), originalModule);
  EXPECT_EQ(slotAllocations, 2u);
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

} // namespace
