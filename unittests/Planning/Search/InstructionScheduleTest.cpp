//===- InstructionScheduleTest.cpp -----------------------------------===//

#include "Wafer/Planning/Search/InstructionSchedule.h"

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <optional>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::shared_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext &context,
                                        llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source,
                                                 mlir::ParserConfig(&context));
}

constexpr llvm::StringLiteral kIndependent = R"mlir(
module {
  func.func @main() {
    %a = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %a, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %b, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    return
  }
}
)mlir";

TEST(InstructionScheduleTest,
     DomainMatchesIndependentOrderAndWorkerCartesianReference) {
  auto context = createContext();
  auto module = parse(*context, kIndependent);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto domain = CardInstructionScheduleDomain::create(
      llvm::ArrayRef<TileInstructionModule>{
          TileInstructionModule{TileId(0), *module}},
      &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  CardInstructionScheduleAssignment current = domain->getFirstAssignment();
  std::optional<CardInstructionScheduleAssignment> selected;
  unsigned count = 0;
  while (true) {
    ASSERT_TRUE(domain->contains(current));
    ++count;
    if (current.windows.size() == 1 &&
        current.windows.front().operations.size() == 2 &&
        current.windows.front().operations.back()->isBeforeInBlock(
            current.windows.front().operations.front()) &&
        current.workers[0].worker == NCCWorker::Worker0 &&
        current.workers[1].worker == NCCWorker::Worker1)
      selected = current;
    auto next = domain->getNextAssignment(current);
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    current = std::move(**next);
  }
  EXPECT_EQ(count, 18u);
  ASSERT_TRUE(selected);

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  modules.push_back(std::move(module));
  auto scheduled = applyInstructionSchedule(std::move(modules),
                                            llvm::ArrayRef<TileId>{TileId(0)},
                                            *domain, *selected, &failureReason);
  ASSERT_TRUE(mlir::succeeded(scheduled)) << failureReason;
  ASSERT_EQ(scheduled->modules.size(), 1u);
  llvm::SmallVector<InstrFillOp, 2> fills;
  scheduled->modules.front()->walk(
      [&](InstrFillOp operation) { fills.push_back(operation); });
  ASSERT_EQ(fills.size(), 2u);
  EXPECT_EQ(fills[0].getIssueWorker(), NCCWorker::Worker1);
  EXPECT_EQ(fills[1].getIssueWorker(), NCCWorker::Worker0);
  auto resources = analyzeInstructionResources(
      llvm::ArrayRef<TileInstructionModule>{
          TileInstructionModule{TileId(0), *scheduled->modules.front()}},
      &failureReason);
  ASSERT_TRUE(mlir::succeeded(resources)) << failureReason;
  EXPECT_FALSE(resources->overlapWitnesses.empty());
}

TEST(InstructionScheduleTest, AliasDependencyRemovesTheReverseOrder) {
  auto context = createContext();
  auto module = parse(*context, R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %buffer, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %buffer, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto domain = CardInstructionScheduleDomain::create(
      llvm::ArrayRef<TileInstructionModule>{
          TileInstructionModule{TileId(0), *module}});
  ASSERT_TRUE(mlir::succeeded(domain));
  CardInstructionScheduleAssignment current = domain->getFirstAssignment();
  unsigned count = 0;
  while (true) {
    ++count;
    auto next = domain->getNextAssignment(current);
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    current = std::move(**next);
  }
  EXPECT_EQ(count, 9u);
}

TEST(InstructionScheduleTest,
     FanoutConsumersEnumerateBothOrdersAfterTheirProducer) {
  auto context = createContext();
  auto module = parse(*context, R"mlir(
module {
  func.func @main() {
    %source = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %left = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %right = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %source, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.elementwise <add> %source, %source into %left
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <add> %source, %source into %right
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto domain = CardInstructionScheduleDomain::create(
      llvm::ArrayRef<TileInstructionModule>{
          TileInstructionModule{TileId(0), *module}});
  ASSERT_TRUE(mlir::succeeded(domain));
  CardInstructionScheduleAssignment current = domain->getFirstAssignment();
  unsigned count = 0;
  while (true) {
    ASSERT_EQ(current.windows.size(), 1u);
    ASSERT_EQ(current.windows.front().operations.size(), 3u);
    EXPECT_TRUE(
        mlir::isa<InstrFillOp>(current.windows.front().operations.front()));
    ++count;
    auto next = domain->getNextAssignment(current);
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    current = std::move(**next);
  }
  EXPECT_EQ(count, 54u);
}

TEST(InstructionScheduleTest, MutationInvalidatesTheBorrowedDomain) {
  auto context = createContext();
  auto module = parse(*context, kIndependent);
  ASSERT_TRUE(module);
  auto domain = CardInstructionScheduleDomain::create(
      llvm::ArrayRef<TileInstructionModule>{
          TileInstructionModule{TileId(0), *module}});
  ASSERT_TRUE(mlir::succeeded(domain));
  CardInstructionScheduleAssignment first = domain->getFirstAssignment();
  InstrFillOp fill;
  module->walk([&](InstrFillOp operation) {
    if (!fill)
      fill = operation;
  });
  ASSERT_TRUE(fill);
  fill.setWorkerAttr(NCCWorkerAttr::get(context.get(), NCCWorker::Worker2));
  EXPECT_FALSE(domain->contains(first));
}

TEST(InstructionScheduleTest,
     DirectDTEIssueComputeWaitProducesActualOverlapWitness) {
  auto context = createContext();
  auto module = parse(*context, R"mlir(
module {
  func.func @main() {
    %send_buffer = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %compute_buffer = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %sent = wafer.instr.dte_send %send_buffer
        {peer = 1 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 7, round = 0, slice = 0>}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.elementwise <add> %compute_buffer, %compute_buffer
        into %compute_buffer
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
        into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.dte_wait %sent : !async.token
    wafer.instr.ncc_join [0]
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto resources = analyzeInstructionResources(
      llvm::ArrayRef<TileInstructionModule>{
          TileInstructionModule{TileId(0), *module}},
      &failureReason);
  ASSERT_TRUE(mlir::succeeded(resources)) << failureReason;
  EXPECT_TRUE(llvm::any_of(resources->overlapWitnesses, [](const auto &entry) {
    return mlir::isa<InstrDTESendOp>(entry.pendingIssue) &&
           mlir::isa<InstrElementwiseOp>(entry.independentOperation);
  }));
  auto domain = CardInstructionScheduleDomain::create(
      llvm::ArrayRef<TileInstructionModule>{
          TileInstructionModule{TileId(0), *module}});
  ASSERT_TRUE(mlir::succeeded(domain));
  CardInstructionScheduleAssignment current = domain->getFirstAssignment();
  while (true) {
    const auto &order = current.windows.front().operations;
    auto wait = llvm::find_if(order, [](mlir::Operation *operation) {
      return mlir::isa<InstrDTEWaitOp>(operation);
    });
    auto send = llvm::find_if(order, [](mlir::Operation *operation) {
      return mlir::isa<InstrDTESendOp>(operation);
    });
    ASSERT_NE(wait, order.end());
    ASSERT_NE(send, order.end());
    EXPECT_LT(send - order.begin(), wait - order.begin());
    auto next = domain->getNextAssignment(current);
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    current = std::move(**next);
  }
}

TEST(InstructionScheduleTest, DetectsSharedDDRAndDirectedPeerLinkResources) {
  auto context = createContext();
  auto sender = parse(*context, R"mlir(
module {
  func.func @main(%ddr: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %spm = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %ddr to %spm
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    %sent = wafer.instr.dte_send %spm
        {peer = 1 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 9, round = 0, slice = 0>}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %sent : !async.token
    return
  }
}
)mlir");
  auto receiver = parse(*context, R"mlir(
module {
  func.func @main(%ddr: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %spm = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %ddr to %spm
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    %received = wafer.instr.dte_recv %spm
        {peer = 0 : i64, bytes = 8 : i64,
         message = #wafer.dte_message<communication = 9, round = 0, slice = 0>}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %received : !async.token
    return
  }
}
)mlir");
  ASSERT_TRUE(sender);
  ASSERT_TRUE(receiver);
  std::string failureReason;
  llvm::SmallVector<TileInstructionModule, 2> modules{
      TileInstructionModule{TileId(0), *sender},
      TileInstructionModule{TileId(1), *receiver}};
  auto resources = analyzeInstructionResources(modules, &failureReason);
  ASSERT_TRUE(mlir::succeeded(resources)) << failureReason;
  EXPECT_TRUE(llvm::any_of(resources->sharedResources, [](const auto &entry) {
    return entry.resource.kind == InstructionResourceKind::CardDDR &&
           entry.users.size() == 2;
  }));
  EXPECT_TRUE(llvm::any_of(resources->sharedResources, [](const auto &entry) {
    return entry.resource.kind == InstructionResourceKind::DirectedPeerLink &&
           entry.resource.first == 0 && entry.resource.second == 1 &&
           entry.users.size() == 2;
  }));
}

} // namespace
