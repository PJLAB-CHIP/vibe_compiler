#include "Wafer/Analysis/DirectCallGraphAnalysis.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <memory>

namespace {

class ObserveDirectCallGraphPass
    : public mlir::PassWrapper<ObserveDirectCallGraphPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  ObserveDirectCallGraphPass() = default;
  ObserveDirectCallGraphPass(llvm::SmallVectorImpl<const void *> *addresses,
                             llvm::SmallVectorImpl<size_t> *functionCounts,
                             bool preserve, bool addFunction)
      : addresses(addresses), functionCounts(functionCounts),
        preserve(preserve), addFunction(addFunction) {}

  void runOnOperation() override {
    auto &analysis =
        getAnalysis<wafer::analysis::DirectCallGraphAnalysis>();
    addresses->push_back(&analysis);
    functionCounts->push_back(analysis.getFunctions().size());
    if (addFunction) {
      mlir::OpBuilder builder(getOperation().getBodyRegion());
      auto function = builder.create<mlir::func::FuncOp>(
          getOperation().getLoc(), "inserted_after_analysis",
          builder.getFunctionType({}, {}));
      function.setPrivate();
    }
    if (preserve)
      markAnalysesPreserved<wafer::analysis::DirectCallGraphAnalysis>();
  }

private:
  llvm::SmallVectorImpl<const void *> *addresses = nullptr;
  llvm::SmallVectorImpl<size_t> *functionCounts = nullptr;
  bool preserve = false;
  bool addFunction = false;
};

class DirectCallGraphAnalysisTest : public ::testing::Test {
protected:
  DirectCallGraphAnalysisTest() {
    registry.insert<mlir::func::FuncDialect>();
    wafer::registerWaferCoreDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(DirectCallGraphAnalysisTest, ResolvesTypedEdgesRootsAndCalleeFirstOrder) {
  auto module = parse(R"mlir(
module {
  func.func private @leaf() { return }
  func.func @entry() {
    func.call @leaf() : () -> ()
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  wafer::analysis::DirectCallGraphAnalysis graph(module->getOperation());
  ASSERT_TRUE(graph.hasModuleScope());
  ASSERT_EQ(graph.getFunctions().size(), 2u);
  EXPECT_TRUE(graph.getUnresolvedCalls().empty());
  EXPECT_TRUE(graph.getUnsupportedCallOperations().empty());
  EXPECT_FALSE(graph.hasRecursiveCycle());

  mlir::func::FuncOp entry = module->lookupSymbol<mlir::func::FuncOp>("entry");
  mlir::func::FuncOp leaf = module->lookupSymbol<mlir::func::FuncOp>("leaf");
  ASSERT_EQ(graph.getCalls(entry).size(), 1u);
  EXPECT_EQ(graph.getCallee(graph.getCalls(entry).front()), leaf);
  ASSERT_EQ(graph.getRootFunctions().size(), 1u);
  EXPECT_EQ(graph.getRootFunctions().front(), entry);
  ASSERT_EQ(graph.getCalleeFirstOrder().size(), 2u);
  EXPECT_EQ(graph.getCalleeFirstOrder().front(), leaf);
  EXPECT_EQ(graph.getCalleeFirstOrder().back(), entry);
}

TEST_F(DirectCallGraphAnalysisTest,
       AnalysisManagerPreservesAndInvalidatesTheModuleFacts) {
  auto module = parse(R"mlir(
module {
  func.func private @leaf() { return }
  func.func @entry() {
    func.call @leaf() : () -> ()
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  llvm::SmallVector<const void *, 4> addresses;
  llvm::SmallVector<size_t, 4> functionCounts;
  mlir::PassManager manager(context.get());
  manager.addPass(std::make_unique<ObserveDirectCallGraphPass>(
      &addresses, &functionCounts, /*preserve=*/true,
      /*addFunction=*/false));
  manager.addPass(std::make_unique<ObserveDirectCallGraphPass>(
      &addresses, &functionCounts, /*preserve=*/true,
      /*addFunction=*/false));
  manager.addPass(std::make_unique<ObserveDirectCallGraphPass>(
      &addresses, &functionCounts, /*preserve=*/false,
      /*addFunction=*/true));
  manager.addPass(std::make_unique<ObserveDirectCallGraphPass>(
      &addresses, &functionCounts, /*preserve=*/true,
      /*addFunction=*/false));
  ASSERT_TRUE(mlir::succeeded(manager.run(*module)));

  ASSERT_EQ(addresses.size(), 4u);
  EXPECT_EQ(addresses[0], addresses[1]);
  EXPECT_EQ(functionCounts,
            (llvm::SmallVector<size_t, 4>{2u, 2u, 2u, 3u}));
}

} // namespace
