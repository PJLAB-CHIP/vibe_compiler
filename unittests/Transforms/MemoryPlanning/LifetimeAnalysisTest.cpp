#include "MemoryPlanning/LifetimeAnalysis.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <limits>
#include <memory>

namespace {

using namespace wafer::memory_planning::detail;

class LifetimeAnalysisTest : public ::testing::Test {
protected:
  LifetimeAnalysisTest() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
    wafer::registerAllDialects(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(&context));
  }

  static mlir::func::FuncOp getOnlyFunction(mlir::ModuleOp module) {
    auto functions = module.getOps<mlir::func::FuncOp>();
    return *functions.begin();
  }

  mlir::MLIRContext context;
};

TEST(PathConditionTest, IntersectionImplicationAndSubtractionAreExact) {
  PathCondition root = PathCondition::root();
  std::optional<PathCondition> thenPath = root.withDecision(0, true);
  std::optional<PathCondition> elsePath = root.withDecision(0, false);
  ASSERT_TRUE(thenPath);
  ASSERT_TRUE(elsePath);

  EXPECT_FALSE(thenPath->compatibleWith(*elsePath));
  EXPECT_FALSE(thenPath->intersect(*elsePath));
  EXPECT_TRUE(thenPath->implies(root));
  EXPECT_FALSE(root.implies(*thenPath));

  llvm::SmallVector<PathCondition, 2> remaining;
  root.subtract(*thenPath, remaining);
  ASSERT_EQ(remaining.size(), 1u);
  EXPECT_EQ(remaining.front(), *elsePath);

  remaining.clear();
  thenPath->subtract(root, remaining);
  EXPECT_TRUE(remaining.empty());
}

TEST(PathConditionTest, RepeatableDecisionCannotProvePackingExclusion) {
  PathCondition root = PathCondition::root();
  std::optional<PathCondition> thenPath =
      root.withDecision(0, true, /*repeatable=*/true);
  std::optional<PathCondition> elsePath =
      root.withDecision(0, false, /*repeatable=*/true);
  ASSERT_TRUE(thenPath);
  ASSERT_TRUE(elsePath);

  EXPECT_FALSE(thenPath->compatibleWith(*elsePath));
  EXPECT_TRUE(thenPath->compatibleForPacking(*elsePath));
  EXPECT_EQ(thenPath->withoutRepeatableDecisions(), root);
  EXPECT_EQ(elsePath->withoutRepeatableDecisions(), root);
}

TEST_F(LifetimeAnalysisTest, TimelineModelsIfAndZeroTripLoopPaths) {
  auto module = parse(R"mlir(
module {
  func.func @control(%condition: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.if %condition {
      %then = arith.constant 1 : i32
    } else {
      %else = arith.constant 2 : i32
    }
    scf.for %index = %c0 to %c1 step %c1 {
      %body = arith.constant 3 : i32
    }
    return
  }
}

)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);

  mlir::scf::IfOp ifOp;
  mlir::scf::ForOp forOp;
  function.walk([&](mlir::scf::IfOp op) { ifOp = op; });
  function.walk([&](mlir::scf::ForOp op) { forOp = op; });
  ASSERT_TRUE(ifOp);
  ASSERT_TRUE(forOp);

  TimelineFailure failure;
  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function, &failure);
  ASSERT_TRUE(mlir::succeeded(timeline));

  mlir::Operation *thenOperation = &ifOp.getThenRegion().front().front();
  mlir::Operation *elseOperation = &ifOp.getElseRegion().front().front();
  mlir::Operation *bodyOperation = &forOp.getRegion().front().front();
  std::optional<ProgramPoint> thenPoint = timeline->lookup(thenOperation);
  std::optional<ProgramPoint> elsePoint = timeline->lookup(elseOperation);
  std::optional<ProgramPoint> loopPoint = timeline->lookup(forOp);
  std::optional<ProgramPoint> bodyPoint = timeline->lookup(bodyOperation);
  ASSERT_TRUE(thenPoint);
  ASSERT_TRUE(elsePoint);
  ASSERT_TRUE(loopPoint);
  ASSERT_TRUE(bodyPoint);

  EXPECT_FALSE(thenPoint->path.compatibleWith(elsePoint->path));
  EXPECT_NE(loopPoint->path, bodyPoint->path);
  EXPECT_TRUE(bodyPoint->path.implies(loopPoint->path));

  llvm::SmallVector<PathCondition, 2> zeroTripPaths;
  loopPoint->path.subtract(bodyPoint->path, zeroTripPaths);
  ASSERT_EQ(zeroTripPaths.size(), 1u);
  EXPECT_FALSE(zeroTripPaths.front().compatibleWith(bodyPoint->path));
}

TEST_F(LifetimeAnalysisTest, TimelineRejectsMultiBlockControlFlow) {
  auto module = parse(R"mlir(
module {
  func.func @multi_block(%condition: i1) {
    cf.cond_br %condition, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb2
  ^bb2:
    return
  }
}

)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);

  TimelineFailure failure;
  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function, &failure);
  EXPECT_TRUE(mlir::failed(timeline));
  EXPECT_EQ(failure.kind, TimelineFailureKind::UnsupportedRegionControlFlow);
  EXPECT_EQ(failure.origin, function.getOperation());
}

TEST_F(LifetimeAnalysisTest, TimelineRejectsUnsupportedParallelRegion) {
  auto module = parse(R"mlir(
module {
  func.func @parallel() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%index) = (%c0) to (%c1) step (%c1) {
      scf.reduce
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  mlir::scf::ParallelOp parallel;
  function.walk([&](mlir::scf::ParallelOp op) { parallel = op; });
  ASSERT_TRUE(parallel);

  TimelineFailure failure;
  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function, &failure);
  EXPECT_TRUE(mlir::failed(timeline));
  EXPECT_EQ(failure.kind, TimelineFailureKind::UnsupportedRegionControlFlow);
  EXPECT_EQ(failure.origin, parallel.getOperation());
}

TEST_F(LifetimeAnalysisTest,
       UnrelatedLoopCarriedAsyncTokenDoesNotBlockMemoryDataflow) {
  auto module = parse(R"mlir(
module {
  func.func @unrelated_token(%token: !async.token) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %looped = scf.for %index = %c0 to %c1 step %c1
        iter_args(%iter = %token) -> (!async.token) {
      scf.yield %iter : !async.token
    }
    async.await %looped : !async.token
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 0> demands;
  LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return wafer::isWaferDDRMemRefType(type);
  });
  LifetimeFailure failure;
  EXPECT_TRUE(mlir::succeeded(dataflow.run(function, nullptr, &failure)));
}

TEST_F(LifetimeAnalysisTest, DataflowPropagatesIfAndLoopCarriedRoots) {
  auto module = parse(R"mlir(
module {
  func.func @aliases(%condition: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %allocation = memref.alloc() : memref<16xi8>
    %selected = scf.if %condition -> (memref<16xi8>) {
      scf.yield %allocation : memref<16xi8>
    } else {
      scf.yield %allocation : memref<16xi8>
    }
    %looped = scf.for %index = %c0 to %c1 step %c1
        iter_args(%current = %selected) -> (memref<16xi8>) {
      scf.yield %current : memref<16xi8>
    }
    memref.dealloc %looped : memref<16xi8>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);

  mlir::memref::AllocOp allocation;
  mlir::scf::ForOp forOp;
  mlir::memref::DeallocOp dealloc;
  function.walk([&](mlir::memref::AllocOp op) { allocation = op; });
  function.walk([&](mlir::scf::ForOp op) { forOp = op; });
  function.walk([&](mlir::memref::DeallocOp op) { dealloc = op; });
  ASSERT_TRUE(allocation);
  ASSERT_TRUE(forOp);
  ASSERT_TRUE(dealloc);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 1> demands{
      LifetimeDemand{allocation, 16, 8, 0}};
  LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return mlir::isa<mlir::MemRefType>(type);
  });
  LifetimeFailure failure;
  ASSERT_TRUE(mlir::succeeded(dataflow.run(function, nullptr, &failure)));

  std::optional<ProgramPoint> usePoint = timeline->lookup(dealloc);
  ASSERT_TRUE(usePoint);
  llvm::SmallVector<RootRef, 2> refs =
      dataflow.rootsAt(forOp.getResult(0), usePoint->path);
  ASSERT_FALSE(refs.empty());
  EXPECT_TRUE(
      llvm::all_of(refs, [](RootRef ref) { return ref.demandIndex == 0; }));

  std::optional<int64_t> loopEnd = timeline->lookupSubtreeEnd(forOp);
  ASSERT_TRUE(loopEnd);
  EXPECT_TRUE(
      llvm::any_of(demands.front().segments, [&](const LiveSegment &segment) {
        return segment.endEvent == *loopEnd;
      }));
}

TEST_F(LifetimeAnalysisTest, DataflowPropagatesSelectExternalOrigins) {
  auto module = parse(R"mlir(
module {
  func.func @select_origin(%condition: i1, %lhs: memref<16xi8>,
                           %rhs: memref<16xi8>) {
    %selected = arith.select %condition, %lhs, %rhs : memref<16xi8>
    memref.dealloc %selected : memref<16xi8>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  mlir::arith::SelectOp select;
  mlir::memref::DeallocOp use;
  function.walk([&](mlir::arith::SelectOp op) { select = op; });
  function.walk([&](mlir::memref::DeallocOp op) { use = op; });
  ASSERT_TRUE(select);
  ASSERT_TRUE(use);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 0> demands;
  LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return mlir::isa<mlir::MemRefType>(type);
  });
  LifetimeFailure failure;
  ASSERT_TRUE(mlir::succeeded(dataflow.run(function, nullptr, &failure)));

  std::optional<ProgramPoint> usePoint = timeline->lookup(use);
  ASSERT_TRUE(usePoint);
  llvm::SmallVector<ValueOriginRef, 2> origins =
      dataflow.originsAt(select.getResult(), usePoint->path);
  ASSERT_EQ(origins.size(), 2u);
  EXPECT_TRUE(llvm::any_of(origins, [&](ValueOriginRef origin) {
    return origin.root == function.getArgument(1);
  }));
  EXPECT_TRUE(llvm::any_of(origins, [&](ValueOriginRef origin) {
    return origin.root == function.getArgument(2);
  }));
}

TEST_F(LifetimeAnalysisTest,
       DataflowRecomputesLoopCarriedOriginsThroughBodyViews) {
  auto module = parse(R"mlir(
module {
  func.func @loop_origin(
      %lhs: memref<16xi8>, %rhs: memref<16xi8>,
      %lower: index, %upper: index, %step: index) {
    %looped = scf.for %index = %lower to %upper step %step
        iter_args(%iter = %lhs) -> (memref<16xi8>) {
      %view = memref.subview %iter[0] [16] [1]
          : memref<16xi8> to memref<16xi8, strided<[1]>>
      %c0 = arith.constant 0 : index
      %unused = memref.load %view[%c0]
          : memref<16xi8, strided<[1]>>
      scf.yield %rhs : memref<16xi8>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  mlir::memref::SubViewOp subview;
  mlir::memref::LoadOp load;
  function.walk([&](mlir::memref::SubViewOp op) { subview = op; });
  function.walk([&](mlir::memref::LoadOp op) { load = op; });
  ASSERT_TRUE(subview);
  ASSERT_TRUE(load);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 0> demands;
  LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return mlir::isa<mlir::MemRefType>(type);
  });
  LifetimeFailure failure;
  ASSERT_TRUE(mlir::succeeded(dataflow.run(function, nullptr, &failure)));

  std::optional<ProgramPoint> usePoint = timeline->lookup(load);
  ASSERT_TRUE(usePoint);
  llvm::SmallVector<ValueOriginRef, 2> origins =
      dataflow.originsAt(subview.getResult(), usePoint->path);
  EXPECT_TRUE(llvm::any_of(origins, [&](ValueOriginRef origin) {
    return origin.root == function.getArgument(0);
  }));
  EXPECT_TRUE(llvm::any_of(origins, [&](ValueOriginRef origin) {
    return origin.root == function.getArgument(1);
  }));
}

TEST_F(LifetimeAnalysisTest, DataflowRejectsUnknownTrackedRootProducer) {
  auto module = parse(R"mlir(
module {
  func.func @unsupported_root() {
    %value = memref.alloca() : memref<16xi8>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  mlir::memref::AllocaOp alloca;
  function.walk([&](mlir::memref::AllocaOp op) { alloca = op; });
  ASSERT_TRUE(alloca);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 0> demands;
  LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return mlir::isa<mlir::MemRefType>(type);
  });
  LifetimeFailure failure;
  EXPECT_TRUE(mlir::failed(dataflow.run(function, nullptr, &failure)));
  EXPECT_EQ(failure.kind, LifetimeFailureKind::UnsupportedTrackedValueProducer);
  EXPECT_EQ(failure.origin, alloca.getOperation());
}

TEST_F(LifetimeAnalysisTest, DDRLocalFenceExtendsManagedRootLifetime) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %ddr = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %spm to %ddr
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.local_fence
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  mlir::memref::AllocOp ddrAllocation;
  wafer::SyncLocalFenceOp fence;
  function.walk([&](mlir::memref::AllocOp op) {
    if (wafer::isWaferDDRMemRefType(op.getType()))
      ddrAllocation = op;
  });
  function.walk([&](wafer::SyncLocalFenceOp op) { fence = op; });
  ASSERT_TRUE(ddrAllocation);
  ASSERT_TRUE(fence);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 1> demands{
      LifetimeDemand{ddrAllocation, 256, 256, 0}};
  LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return wafer::isWaferDDRMemRefType(type);
  });
  LocalCompletionTracker completion(wafer::WaferResourceKind::DDR);
  LifetimeFailure failure;
  ASSERT_TRUE(mlir::succeeded(dataflow.run(function, &completion, &failure)));

  std::optional<ProgramPoint> fencePoint = timeline->lookup(fence);
  ASSERT_TRUE(fencePoint);
  EXPECT_TRUE(
      llvm::any_of(demands.front().segments, [&](const LiveSegment &segment) {
        return segment.endEvent == fencePoint->event;
      }));
}

TEST_F(LifetimeAnalysisTest, ExternalDDRLocalIssueStillRequiresFence) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %ddr: memref<128xf16, #wafer.memory<ddr, tensor>>) {
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %spm to %ddr
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  wafer::InstrWDMAOp issue;
  function.walk([&](wafer::InstrWDMAOp op) { issue = op; });
  ASSERT_TRUE(issue);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 0> demands;
  LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return wafer::isWaferDDRMemRefType(type);
  });
  LocalCompletionTracker completion(wafer::WaferResourceKind::DDR);
  LifetimeFailure failure;
  EXPECT_TRUE(mlir::failed(dataflow.run(function, &completion, &failure)));
  EXPECT_EQ(failure.kind, LifetimeFailureKind::MissingLocalCompletion);
  EXPECT_EQ(failure.origin, issue.getOperation());
}

TEST_F(LifetimeAnalysisTest, LoopBodyIssueMustCompleteBeforeBackedge) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %ddr = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    scf.for %index = %c0 to %c2 step %c1 {
      wafer.instr.wdma %spm to %ddr
          {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
          : memref<128xf16, #wafer.memory<spm, tensor>>
         to memref<128xf16, #wafer.memory<ddr, tensor>>
    }
    wafer.instr.local_fence
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  mlir::memref::AllocOp ddrAllocation;
  wafer::InstrWDMAOp issue;
  function.walk([&](mlir::memref::AllocOp op) {
    if (wafer::isWaferDDRMemRefType(op.getType()))
      ddrAllocation = op;
  });
  function.walk([&](wafer::InstrWDMAOp op) { issue = op; });
  ASSERT_TRUE(ddrAllocation);
  ASSERT_TRUE(issue);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 1> demands{
      LifetimeDemand{ddrAllocation, 256, 256, 0}};
  LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return wafer::isWaferDDRMemRefType(type);
  });
  LocalCompletionTracker completion(wafer::WaferResourceKind::DDR);
  LifetimeFailure failure;
  EXPECT_TRUE(mlir::failed(dataflow.run(function, &completion, &failure)));
  EXPECT_EQ(failure.kind, LifetimeFailureKind::LoopBackedgeCompletion);
  EXPECT_EQ(failure.origin, issue.getOperation());
}

TEST_F(LifetimeAnalysisTest, LoopOnlyFenceDoesNotCoverZeroTripPath) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ddr = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %spm to %ddr
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    scf.for %index = %c0 to %c0 step %c1 {
      wafer.instr.local_fence
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  mlir::memref::AllocOp ddrAllocation;
  function.walk([&](mlir::memref::AllocOp op) {
    if (wafer::isWaferDDRMemRefType(op.getType()))
      ddrAllocation = op;
  });
  ASSERT_TRUE(ddrAllocation);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 1> demands{
      LifetimeDemand{ddrAllocation, 256, 256, 0}};
  LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return wafer::isWaferDDRMemRefType(type);
  });
  LocalCompletionTracker completion(wafer::WaferResourceKind::DDR);
  LifetimeFailure failure;
  EXPECT_TRUE(mlir::failed(dataflow.run(function, &completion, &failure)));
  EXPECT_EQ(failure.kind, LifetimeFailureKind::MissingLocalCompletion);
}

TEST_F(LifetimeAnalysisTest, LocalIssueWithoutTrackedEffectIsIgnored) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %zero = arith.constant 0.000000e+00 : f16
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %spm, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 0> demands;
  LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return wafer::isWaferDDRMemRefType(type);
  });
  LocalCompletionTracker completion(wafer::WaferResourceKind::DDR);
  LifetimeFailure failure;
  EXPECT_TRUE(mlir::succeeded(dataflow.run(function, &completion, &failure)));
}

TEST_F(LifetimeAnalysisTest, WeightedFirstFitUsesPathAwareConflicts) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() : memref<256xi8>
    %b = memref.alloc() : memref<256xi8>
    %c = memref.alloc() : memref<256xi8>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::memref::AllocOp, 3> allocations;
  module->walk([&](mlir::memref::AllocOp op) { allocations.push_back(op); });
  ASSERT_EQ(allocations.size(), 3u);

  PathCondition root = PathCondition::root();
  llvm::SmallVector<LifetimeDemand, 3> demands{
      LifetimeDemand{allocations[0],
                     256,
                     256,
                     0,
                     ProgramPoint{0, root},
                     {LiveSegment{0, 10, root}}},
      LifetimeDemand{allocations[1],
                     256,
                     256,
                     1,
                     ProgramPoint{1, root},
                     {LiveSegment{0, 5, root}}},
      LifetimeDemand{allocations[2],
                     256,
                     256,
                     2,
                     ProgramPoint{2, root},
                     {LiveSegment{6, 10, root}}},
  };

  PackingResult packing = packFirstFit(demands, ArenaRange{0, 512});
  ASSERT_TRUE(packing.succeeded());
  ASSERT_EQ(packing.placements.size(), 3u);
  EXPECT_EQ(packing.placements.front().demandIndex, 0u);
  EXPECT_EQ(packing.placements.front().offsetBytes, 0);

  auto placementFor = [&](unsigned demandIndex) -> const Placement & {
    return *llvm::find_if(packing.placements, [&](const Placement &placement) {
      return placement.demandIndex == demandIndex;
    });
  };
  EXPECT_EQ(placementFor(1).offsetBytes, 256);
  EXPECT_EQ(placementFor(2).offsetBytes, 256);

  std::optional<PathCondition> thenPath = root.withDecision(0, true);
  std::optional<PathCondition> elsePath = root.withDecision(0, false);
  ASSERT_TRUE(thenPath);
  ASSERT_TRUE(elsePath);
  demands[0].segments = {LiveSegment{0, 10, *thenPath}};
  demands[1].segments = {LiveSegment{0, 10, *elsePath}};
  demands.resize(2);
  packing = packFirstFit(demands, ArenaRange{0, 256});
  ASSERT_TRUE(packing.succeeded());
  ASSERT_EQ(packing.placements.size(), 2u);
  EXPECT_EQ(packing.placements[0].offsetBytes, 0);
  EXPECT_EQ(packing.placements[1].offsetBytes, 0);
}

TEST_F(LifetimeAnalysisTest, FirstFitReportsNoFitWithoutWritingIR) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() : memref<256xi8>
    %b = memref.alloc() : memref<256xi8>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::memref::AllocOp, 2> allocations;
  module->walk([&](mlir::memref::AllocOp op) { allocations.push_back(op); });
  ASSERT_EQ(allocations.size(), 2u);
  llvm::SmallVector<mlir::DictionaryAttr, 2> attributesBefore;
  llvm::transform(allocations, std::back_inserter(attributesBefore),
                  [](mlir::memref::AllocOp allocation) {
                    return allocation->getAttrDictionary();
                  });

  PathCondition root = PathCondition::root();
  llvm::SmallVector<LifetimeDemand, 2> demands{
      LifetimeDemand{allocations[0],
                     256,
                     256,
                     0,
                     ProgramPoint{0, root},
                     {LiveSegment{0, 10, root}}},
      LifetimeDemand{allocations[1],
                     256,
                     256,
                     1,
                     ProgramPoint{1, root},
                     {LiveSegment{0, 10, root}}},
  };
  PackingResult packing = packFirstFit(demands, ArenaRange{0, 256});
  ASSERT_FALSE(packing.succeeded());
  ASSERT_TRUE(packing.failure);
  EXPECT_EQ(packing.failure->kind, PackingFailureKind::NoFit);
  for (auto [allocation, attributes] : llvm::zip(allocations, attributesBefore))
    EXPECT_EQ(allocation->getAttrDictionary(), attributes);
}

TEST(LifetimeAnalysisUtilitiesTest,
     AlignmentRequirementsUseLeastCommonMultiple) {
  EXPECT_EQ(combineAlignmentRequirements(384, 256), 768);
  EXPECT_EQ(combineAlignmentRequirements(256, 384), 768);
  EXPECT_EQ(combineAlignmentRequirements(256, 512), 512);
  EXPECT_FALSE(combineAlignmentRequirements(0, 256));
  EXPECT_FALSE(
      combineAlignmentRequirements(std::numeric_limits<int64_t>::max(), 2));
}

} // namespace
