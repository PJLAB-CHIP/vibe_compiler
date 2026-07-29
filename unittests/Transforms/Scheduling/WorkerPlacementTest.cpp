//===- WorkerPlacementTest.cpp - Typed NCC worker alternatives ----------===//

#include "Wafer/InitAll.h"
#include "Wafer/Transforms/WorkerPlacement.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>

namespace {

class WorkerPlacementTest : public testing::Test {
protected:
  WorkerPlacementTest() {
    wafer::registerAllDialects(registry);
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
};

TEST_F(WorkerPlacementTest,
       AssignsThreeDisjointComponentsAndRebuildsTerminalParticipants) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %c = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %a, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %b, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %c, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  std::string failure;
  auto candidate = wafer::deriveDisjointNCCWorkerPlacementCandidate(
      *source, &failure);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failure;
  EXPECT_EQ(candidate->dependencyLaneCount, 3u);
  EXPECT_EQ(candidate->participantMask, UINT32_C(0x7));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*candidate->module)));

  std::set<uint32_t> workers;
  candidate->module->walk([&](mlir::Operation *operation) {
    if (std::optional<wafer::NCCWorker> worker =
            wafer::getNCCIssueWorker(operation))
      workers.insert(static_cast<uint32_t>(*worker));
  });
  EXPECT_EQ(workers, (std::set<uint32_t>{0, 1, 2}));

  llvm::SmallVector<wafer::SyncNCCJoinOp, 2> joins;
  candidate->module->walk(
      [&](wafer::SyncNCCJoinOp join) { joins.push_back(join); });
  ASSERT_EQ(joins.size(), 1u);
  EXPECT_EQ(joins.front().getParticipants(),
            (llvm::ArrayRef<int64_t>{0, 1, 2}));

  // The generation parent is immutable.
  std::set<uint32_t> sourceWorkers;
  source->walk([&](mlir::Operation *operation) {
    if (std::optional<wafer::NCCWorker> worker =
            wafer::getNCCIssueWorker(operation))
      sourceWorkers.insert(static_cast<uint32_t>(*worker));
  });
  EXPECT_EQ(sourceWorkers, (std::set<uint32_t>{0}));
}

TEST_F(WorkerPlacementTest,
       UsesStaticSubviewRangesButKeepsOverlappingWritesTogether) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %root = memref.alloc() : memref<16xf16, #wafer.memory<spm, tensor>>
    %other = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %first = memref.subview %root[0] [6] [1]
        : memref<16xf16, #wafer.memory<spm, tensor>>
       to memref<6xf16, strided<[1], offset: 0>,
                 #wafer.memory<spm, tensor>>
    %overlap = memref.subview %root[4] [6] [1]
        : memref<16xf16, #wafer.memory<spm, tensor>>
       to memref<6xf16, strided<[1], offset: 4>,
                 #wafer.memory<spm, tensor>>
    %disjoint = memref.subview %root[12] [4] [1]
        : memref<16xf16, #wafer.memory<spm, tensor>>
       to memref<4xf16, strided<[1], offset: 12>,
                 #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %first, %zero
        : memref<6xf16, strided<[1], offset: 0>,
                 #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %overlap, %zero
        : memref<6xf16, strided<[1], offset: 4>,
                 #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %disjoint, %zero
        : memref<4xf16, strided<[1], offset: 12>,
                 #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %other, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  std::string failure;
  auto candidate = wafer::deriveDisjointNCCWorkerPlacementCandidate(
      *source, &failure);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failure;
  EXPECT_EQ(candidate->dependencyLaneCount, 3u);
  EXPECT_EQ(candidate->participantMask, UINT32_C(0x7));

  llvm::SmallVector<wafer::NCCWorker, 4> workers;
  candidate->module->walk([&](wafer::InstrFillOp fill) {
    workers.push_back(fill.getWorker());
  });
  ASSERT_EQ(workers.size(), 4u);
  EXPECT_EQ(workers[0], workers[1]);
  EXPECT_NE(workers[1], workers[2]);
  EXPECT_NE(workers[2], workers[3]);
}

TEST_F(WorkerPlacementTest,
       KeepsRawWarWawComponentOnOneWorkerAndRejectsPlacedParents) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %c = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %a, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.elementwise <neg> %a into %b
        : memref<4xf16, #wafer.memory<spm, tensor>>
          into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %c, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  std::string failure;
  auto candidate = wafer::deriveDisjointNCCWorkerPlacementCandidate(
      *source, &failure);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failure;
  EXPECT_EQ(candidate->dependencyLaneCount, 2u);

  llvm::SmallVector<wafer::NCCWorker, 3> workers;
  candidate->module->walk([&](mlir::Operation *operation) {
    if (std::optional<wafer::NCCWorker> worker =
            wafer::getNCCIssueWorker(operation))
      workers.push_back(*worker);
  });
  ASSERT_EQ(workers.size(), 3u);
  EXPECT_EQ(workers[0], workers[1]);
  EXPECT_NE(workers[1], workers[2]);

  source->walk([](mlir::memref::AllocOp allocation) {
    allocation->setAttr(
        wafer::kWaferSPMOffsetAttrName,
        wafer::SPMOffsetAttr::get(allocation.getContext(), 65536));
  });
  auto rejected = wafer::deriveDisjointNCCWorkerPlacementCandidate(
      *source, &failure);
  EXPECT_TRUE(mlir::failed(rejected));
  EXPECT_NE(failure.find("unplaced"), std::string::npos);
}

TEST_F(WorkerPlacementTest,
       DoesNotProveFreshAllocationDisjointFromUnknownRoot) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %unknown: memref<4xf16, #wafer.memory<spm, tensor>>) {
    %fresh = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %unknown, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %fresh, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  std::string failure;
  auto candidate = wafer::deriveDisjointNCCWorkerPlacementCandidate(
      *source, &failure);
  EXPECT_TRUE(mlir::failed(candidate));
  EXPECT_NE(failure.find("only one dependency lane"), std::string::npos)
      << failure;
}

TEST_F(WorkerPlacementTest,
       KeepsEveryLoopCarriedRootInTheAliasDependencyLane) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %a = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    %result = scf.for %iv = %c0 to %c2 step %c1
        iter_args(%current = %a)
        -> (memref<4xf16, #wafer.memory<spm, tensor>>) {
      wafer.instr.fill %current, %zero
          : memref<4xf16, #wafer.memory<spm, tensor>>, f16
      scf.yield %b : memref<4xf16, #wafer.memory<spm, tensor>>
    }
    wafer.instr.fill %b, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  std::string failure;
  auto candidate = wafer::deriveDisjointNCCWorkerPlacementCandidate(
      *source, &failure);
  EXPECT_TRUE(mlir::failed(candidate));
  EXPECT_NE(failure.find("only one dependency lane"), std::string::npos)
      << failure;
}

TEST_F(WorkerPlacementTest,
       PlacesNCCWorkersAroundExactDirectDTEWithoutChangingDTE) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %c = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %a, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %b, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    %sent = wafer.instr.dte_send %a
        {peer = 1 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 5, phase = peer_dataflow, round = 0, slice = 0>}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %sent : !async.token
    wafer.instr.elementwise <add> %a, %b into %c
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
          into memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  std::string failure;
  auto candidate = wafer::deriveDisjointNCCWorkerPlacementCandidate(
      *source, &failure);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failure;
  EXPECT_GE(candidate->dependencyLaneCount, 2u);
  EXPECT_GE(llvm::popcount(candidate->participantMask), 2);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*candidate->module)));

  std::set<uint32_t> workers;
  llvm::SmallVector<wafer::InstrDTESendOp, 1> sends;
  llvm::SmallVector<wafer::InstrDTEWaitOp, 1> waits;
  candidate->module->walk([&](mlir::Operation *operation) {
    if (std::optional<wafer::NCCWorker> worker =
            wafer::getNCCIssueWorker(operation))
      workers.insert(static_cast<uint32_t>(*worker));
    if (auto send = mlir::dyn_cast<wafer::InstrDTESendOp>(operation))
      sends.push_back(send);
    if (auto wait = mlir::dyn_cast<wafer::InstrDTEWaitOp>(operation))
      waits.push_back(wait);
  });
  EXPECT_GE(workers.size(), 2u);
  EXPECT_NE(workers.find(1), workers.end());
  ASSERT_EQ(sends.size(), 1u);
  ASSERT_EQ(waits.size(), 1u);
  EXPECT_EQ(sends.front().getPeer(), 1);
  EXPECT_EQ(sends.front().getBytes(), 8);
  ASSERT_TRUE(sends.front().getToken().hasOneUse());
  EXPECT_EQ(*sends.front().getToken().getUsers().begin(),
            waits.front().getOperation());
  EXPECT_EQ(sends.front()->getBlock(), waits.front()->getBlock());
  EXPECT_TRUE(sends.front()->isBeforeInBlock(waits.front()));

  unsigned sourceDTEIssues = 0;
  unsigned sourceDTEWaits = 0;
  source->walk([&](mlir::Operation *operation) {
    sourceDTEIssues += mlir::isa<wafer::InstrDTESendOp,
                                 wafer::InstrDTERecvOp>(operation);
    sourceDTEWaits += mlir::isa<wafer::InstrDTEWaitOp>(operation);
  });
  EXPECT_EQ(sourceDTEIssues, 1u);
  EXPECT_EQ(sourceDTEWaits, 1u);
}

TEST_F(WorkerPlacementTest, RejectsDirectDTEWithoutOneExactSameBlockWait) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %a, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %b, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    %sent = wafer.instr.dte_send %a
        {peer = 1 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 5, phase = peer_dataflow, round = 0, slice = 0>}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  std::string failure;
  auto candidate = wafer::deriveDisjointNCCWorkerPlacementCandidate(
      *source, &failure);
  EXPECT_TRUE(mlir::failed(candidate));
  EXPECT_NE(failure.find("one exact same-block wait"), std::string::npos)
      << failure;
}

TEST_F(WorkerPlacementTest, RejectsStaleNonzeroWorkerAssignment) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %a, %zero
        {worker = #wafer.ncc_worker<worker1>}
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %b, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir",
                                                        &context);
  ASSERT_TRUE(source);

  std::string failure;
  auto candidate = wafer::deriveDisjointNCCWorkerPlacementCandidate(
      *source, &failure);
  EXPECT_TRUE(mlir::failed(candidate));
  EXPECT_NE(failure.find("already has a nonzero worker assignment"),
            std::string::npos)
      << failure;
}

} // namespace
