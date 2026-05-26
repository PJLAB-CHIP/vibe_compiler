#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

namespace {

TEST(WaferDialectTest, ParsesMemorySpaceAttr) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(module attributes {wafer.memory_space = #wafer.memory_space<spm>} {})mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto attr =
      module->getOperation()->getAttrOfType<wafer::MemorySpaceAttr>("wafer.memory_space");
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getValue(), wafer::MemorySpace::SPM);
}

} // namespace
