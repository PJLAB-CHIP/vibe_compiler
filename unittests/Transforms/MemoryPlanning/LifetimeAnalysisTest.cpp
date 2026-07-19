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

TEST(PathConditionTest, DecisionDomainScalesPastMachineWordWidth) {
  PathCondition root = PathCondition::root();
  std::optional<PathCondition> selected = root.withDecision(130, true);
  std::optional<PathCondition> rejected = root.withDecision(130, false);
  ASSERT_TRUE(selected);
  ASSERT_TRUE(rejected);
  EXPECT_FALSE(selected->compatibleWith(*rejected));
  EXPECT_FALSE(selected->compatibleForPacking(*rejected));

  llvm::SmallVector<PathCondition, 2> remaining;
  root.subtract(*selected, remaining);
  ASSERT_EQ(remaining.size(), 1u);
  EXPECT_EQ(remaining.front(), *rejected);

  std::optional<PathCondition> repeatableSelected =
      root.withDecision(1024, true, /*repeatable=*/true);
  std::optional<PathCondition> repeatableRejected =
      root.withDecision(1024, false, /*repeatable=*/true);
  ASSERT_TRUE(repeatableSelected);
  ASSERT_TRUE(repeatableRejected);
  EXPECT_FALSE(repeatableSelected->compatibleWith(*repeatableRejected));
  EXPECT_TRUE(repeatableSelected->compatibleForPacking(*repeatableRejected));
  EXPECT_EQ(repeatableSelected->withoutRepeatableDecisions(), root);
  EXPECT_EQ(repeatableRejected->withoutRepeatableDecisions(), root);
}

TEST(PathConditionTest, DenseDecisionConjunctionRemainsExactPastWordWidth) {
  PathCondition root = PathCondition::root();
  PathCondition prefix = root;
  PathCondition suffix = root;
  PathCondition complete = root;
  for (uint64_t decision = 0; decision < 130; ++decision) {
    bool selected = (decision % 3) != 0;
    std::optional<PathCondition> next =
        complete.withDecision(decision, selected);
    ASSERT_TRUE(next);
    complete = *next;
    if (decision < 65) {
      next = prefix.withDecision(decision, selected);
      ASSERT_TRUE(next);
      prefix = *next;
    } else {
      next = suffix.withDecision(decision, selected);
      ASSERT_TRUE(next);
      suffix = *next;
    }
  }

  std::optional<PathCondition> rebuilt = prefix.intersect(suffix);
  ASSERT_TRUE(rebuilt);
  EXPECT_EQ(*rebuilt, complete);
  EXPECT_TRUE(complete.implies(prefix));
  EXPECT_TRUE(complete.implies(suffix));
  EXPECT_FALSE(prefix.implies(complete));

  std::optional<PathCondition> conflicting =
      complete.withDecision(97, /*selected=*/false);
  EXPECT_FALSE(conflicting);

  llvm::SmallVector<PathCondition, 4> remaining;
  root.subtract(complete, remaining);
  ASSERT_EQ(remaining.size(), 130u);
  EXPECT_TRUE(llvm::all_of(remaining, [&](const PathCondition &path) {
    return !path.compatibleWith(complete);
  }));
}

TEST_F(LifetimeAnalysisTest, TimelinePreservesHighDecisionBranchExclusivity) {
  std::string source = R"mlir(
module {
  func.func @many_branches(%condition: i1) {
)mlir";
  for (unsigned index = 0; index < 96; ++index) {
    source += R"mlir(
    scf.if %condition {
      scf.yield
    } else {
      scf.yield
    }
)mlir";
  }
  source += R"mlir(
    return
  }
}
)mlir";

  auto module = parse(source);
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  llvm::SmallVector<mlir::scf::IfOp, 96> branches;
  function.walk([&](mlir::scf::IfOp op) { branches.push_back(op); });
  ASSERT_EQ(branches.size(), 96u);

  TimelineFailure failure;
  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function, &failure);
  ASSERT_TRUE(mlir::succeeded(timeline));

  auto branchPoint = [&](mlir::scf::IfOp branch, bool thenBranch) {
    mlir::Region &region =
        thenBranch ? branch.getThenRegion() : branch.getElseRegion();
    return timeline->lookup(region.front().getTerminator());
  };
  std::optional<ProgramPoint> firstThen = branchPoint(branches.front(), true);
  std::optional<ProgramPoint> highThen = branchPoint(branches[80], true);
  std::optional<ProgramPoint> highElse = branchPoint(branches[80], false);
  ASSERT_TRUE(firstThen);
  ASSERT_TRUE(highThen);
  ASSERT_TRUE(highElse);
  EXPECT_FALSE(highThen->path.compatibleWith(highElse->path));
  EXPECT_TRUE(firstThen->path.compatibleWith(highThen->path));
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

TEST_F(LifetimeAnalysisTest,
       NestedLoopDecisionCannotProvePackingExclusionAcrossOuterIterations) {
  auto module = parse(R"mlir(
module {
  func.func @nested_loops(%outerUpper: index, %innerUpper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %outer = %c0 to %outerUpper step %c1 {
      scf.for %inner = %c0 to %innerUpper step %c1 {
        %value = arith.constant 1 : i32
      }
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  mlir::scf::ForOp innerLoop;
  function.walk([&](mlir::scf::ForOp op) {
    if (op->getParentOfType<mlir::scf::ForOp>())
      innerLoop = op;
  });
  ASSERT_TRUE(innerLoop);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  std::optional<ProgramPoint> innerLoopPoint = timeline->lookup(innerLoop);
  std::optional<ProgramPoint> innerBodyPoint =
      timeline->lookup(&innerLoop.getBody()->front());
  ASSERT_TRUE(innerLoopPoint);
  ASSERT_TRUE(innerBodyPoint);

  llvm::SmallVector<PathCondition, 2> zeroTripPaths;
  innerLoopPoint->path.subtract(innerBodyPoint->path, zeroTripPaths);
  ASSERT_EQ(zeroTripPaths.size(), 1u);
  EXPECT_FALSE(zeroTripPaths.front().compatibleWith(innerBodyPoint->path));
  EXPECT_TRUE(zeroTripPaths.front().compatibleForPacking(innerBodyPoint->path));
  EXPECT_EQ(innerBodyPoint->path.withoutRepeatableDecisions(),
            innerLoopPoint->path);
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
  LifetimeDataflow dataflow(
      *timeline, demands,
      [](mlir::Type type) { return mlir::isa<mlir::MemRefType>(type); },
      /*resolveValue=*/{},
      /*isExplicitRoot=*/[](mlir::Value) { return true; });
  LifetimeFailure failure;
  EXPECT_TRUE(mlir::failed(dataflow.run(function, nullptr, &failure)));
  EXPECT_EQ(failure.kind, LifetimeFailureKind::UnsupportedTrackedValueProducer);
  EXPECT_EQ(failure.origin, alloca.getOperation());
}

TEST_F(LifetimeAnalysisTest, DataflowAcceptsPredicateApprovedGlobalRoot) {
  auto module = parse(R"mlir(
module {
  memref.global "private" constant @weights : memref<16xi8> = dense<1>
  func.func @read_global() {
    %c0 = arith.constant 0 : index
    %weights = memref.get_global @weights : memref<16xi8>
    %unused = memref.load %weights[%c0] : memref<16xi8>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  mlir::memref::GetGlobalOp getGlobal;
  mlir::memref::LoadOp load;
  function.walk([&](mlir::memref::GetGlobalOp op) { getGlobal = op; });
  function.walk([&](mlir::memref::LoadOp op) { load = op; });
  ASSERT_TRUE(getGlobal);
  ASSERT_TRUE(load);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 0> demands;
  LifetimeDataflow dataflow(
      *timeline, demands,
      [](mlir::Type type) { return mlir::isa<mlir::MemRefType>(type); },
      /*resolveValue=*/{},
      [&](mlir::Value value) { return value == getGlobal.getResult(); });
  LifetimeFailure failure;
  ASSERT_TRUE(mlir::succeeded(dataflow.run(function, nullptr, &failure)));

  std::optional<ProgramPoint> usePoint = timeline->lookup(load);
  ASSERT_TRUE(usePoint);
  llvm::SmallVector<ValueOriginRef, 2> origins =
      dataflow.originsAt(getGlobal.getResult(), usePoint->path);
  ASSERT_EQ(origins.size(), 1u);
  EXPECT_EQ(origins.front().root, getGlobal.getResult());
}

TEST_F(LifetimeAnalysisTest,
       DataflowPropagatesRootAcrossSiblingTileRegionBoundary) {
  auto module = parse(R"mlir(
module {
  func.func @sibling_regions() {
    %zero = arith.constant 0 : i8
    %resident = wafer.tile.region(%zero : i8) ->
        (memref<16xi8, #wafer.memory<spm, tensor>>) {
    ^bb0(%unused: i8):
      %allocation = memref.alloc()
          : memref<16xi8, #wafer.memory<spm, tensor>>
      wafer.tile.yield %allocation
          : memref<16xi8, #wafer.memory<spm, tensor>>
    }
    %forwarded = wafer.tile.region(%resident
        : memref<16xi8, #wafer.memory<spm, tensor>>) ->
        (memref<16xi8, #wafer.memory<spm, tensor>>) {
    ^bb0(%input: memref<16xi8, #wafer.memory<spm, tensor>>):
      %c0 = arith.constant 0 : index
      %unused = memref.load %input[%c0]
          : memref<16xi8, #wafer.memory<spm, tensor>>
      wafer.tile.yield %input
          : memref<16xi8, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = getOnlyFunction(*module);
  mlir::memref::AllocOp allocation;
  mlir::memref::LoadOp load;
  function.walk([&](mlir::memref::AllocOp op) { allocation = op; });
  function.walk([&](mlir::memref::LoadOp op) { load = op; });
  ASSERT_TRUE(allocation);
  ASSERT_TRUE(load);

  mlir::FailureOr<StructuredTimeline> timeline =
      StructuredTimeline::build(function);
  ASSERT_TRUE(mlir::succeeded(timeline));
  llvm::SmallVector<LifetimeDemand, 1> demands{
      LifetimeDemand{allocation, 16, 16, 0}};
  LifetimeDataflow dataflow(*timeline, demands, [](mlir::Type type) {
    return wafer::isWaferSPMMemRefType(type);
  });
  LifetimeFailure failure;
  ASSERT_TRUE(mlir::succeeded(dataflow.run(function, nullptr, &failure)));

  std::optional<ProgramPoint> loadPoint = timeline->lookup(load);
  ASSERT_TRUE(loadPoint);
  llvm::SmallVector<RootRef, 2> roots =
      dataflow.rootsAt(load.getMemRef(), loadPoint->path);
  ASSERT_EQ(roots.size(), 1u);
  EXPECT_EQ(roots.front().demandIndex, 0u);
  EXPECT_TRUE(
      llvm::any_of(demands.front().segments, [&](const LiveSegment &segment) {
        // The consumer region also forwards the same alias through its result,
        // so the root must reach at least the load and may remain live through
        // the later tile.yield.
        return segment.endEvent >= loadPoint->event;
      }));
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

TEST_F(LifetimeAnalysisTest,
       HighDecisionLoopBodyIssueStillFailsClosedAtBackedge) {
  std::string source = R"mlir(
module {
  func.func @main(%condition: i1) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
)mlir";
  for (unsigned index = 0; index < 80; ++index) {
    source += R"mlir(
    scf.if %condition {
      scf.yield
    } else {
      scf.yield
    }
)mlir";
  }
  source += R"mlir(
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
)mlir";

  auto module = parse(source);
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

TEST(LifetimeAnalysisUtilitiesTest,
     LifetimeOverlapHonorsPathAndRepeatableDecisions) {
  PathCondition root = PathCondition::root();
  LifetimeDemand lhs;
  LifetimeDemand rhs;
  lhs.segments = {LiveSegment{0, 5, root}, LiveSegment{10, 15, root}};
  rhs.segments = {LiveSegment{6, 9, root}};
  EXPECT_FALSE(lifetimesOverlap(lhs, rhs));
  rhs.segments = {LiveSegment{4, 7, root}};
  EXPECT_TRUE(lifetimesOverlap(lhs, rhs));

  std::optional<PathCondition> thenPath = root.withDecision(130, true);
  std::optional<PathCondition> elsePath = root.withDecision(130, false);
  ASSERT_TRUE(thenPath);
  ASSERT_TRUE(elsePath);
  lhs.segments = {LiveSegment{0, 10, *thenPath}};
  rhs.segments = {LiveSegment{0, 10, *elsePath}};
  EXPECT_FALSE(lifetimesOverlap(lhs, rhs));

  std::optional<PathCondition> repeatableThen =
      root.withDecision(1024, true, /*repeatable=*/true);
  std::optional<PathCondition> repeatableElse =
      root.withDecision(1024, false, /*repeatable=*/true);
  ASSERT_TRUE(repeatableThen);
  ASSERT_TRUE(repeatableElse);
  lhs.segments = {LiveSegment{0, 10, *repeatableThen}};
  rhs.segments = {LiveSegment{0, 10, *repeatableElse}};
  EXPECT_TRUE(lifetimesOverlap(lhs, rhs));
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
