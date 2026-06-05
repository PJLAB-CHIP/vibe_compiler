#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <optional>

namespace {

TEST(WaferDialectTest, ParsesMemoryAttrAndComputesPhysicalInfo) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(module attributes {wafer.memory = #wafer.memory<spm, cx>} {})mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(module);

  auto attr =
      module->getOperation()->getAttrOfType<wafer::MemoryAttr>("wafer.memory");
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getSpace(), wafer::MemorySpace::SPM);
  EXPECT_EQ(attr.getLayout(), wafer::MemLayout::Cx);

  auto memrefType = mlir::MemRefType::get(
      {2, 65}, mlir::Float16Type::get(&context),
      mlir::MemRefLayoutAttrInterface{}, attr);
  std::optional<wafer::WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  ASSERT_TRUE(info);
  EXPECT_EQ(info->compactBytes, 260);
  EXPECT_EQ(info->physicalBytes, 512);
  EXPECT_EQ(info->cBlock, 64);
  EXPECT_EQ(info->alignedC, 128);
  EXPECT_EQ(info->tailC, 1);
}

} // namespace
