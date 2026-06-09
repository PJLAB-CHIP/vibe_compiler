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

  auto memrefType =
      mlir::MemRefType::get({2, 65}, mlir::Float16Type::get(&context),
                            mlir::MemRefLayoutAttrInterface{}, attr);
  std::optional<wafer::WaferPhysicalTensorInfo> info =
      wafer::computeWaferPhysicalTensorInfo(memrefType);
  ASSERT_TRUE(info);
  EXPECT_EQ(info->compactBytes, 260);
  EXPECT_EQ(info->physicalBytes, 512);
  EXPECT_EQ(info->cBlock, 64);
  EXPECT_EQ(info->alignedC, 68);
  EXPECT_EQ(info->tailC, 4);
}

TEST(WaferDialectTest, ComputesCxAndNCxBlockMajorOffsetsForLargeC) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  auto cxAttr = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                       wafer::MemLayout::Cx);
  auto ncxAttr = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                        wafer::MemLayout::NCx);
  mlir::Type f16 = mlir::Float16Type::get(&context);

  auto cxType = mlir::MemRefType::get(
      {3, 1000}, f16, mlir::MemRefLayoutAttrInterface{}, cxAttr);
  std::optional<wafer::WaferPhysicalTensorInfo> cxInfo =
      wafer::computeWaferPhysicalTensorInfo(cxType);
  ASSERT_TRUE(cxInfo);
  EXPECT_EQ(cxInfo->cBlock, 64);
  EXPECT_EQ(cxInfo->cxBlocks, 16);
  EXPECT_EQ(cxInfo->c0, 0);
  EXPECT_EQ(cxInfo->alignedC, 1024);
  EXPECT_EQ(cxInfo->physicalBytes, 6144);

  std::optional<int64_t> cxOffset =
      wafer::computeWaferPhysicalElementByteOffset(cxType, {1, 64});
  ASSERT_TRUE(cxOffset);
  EXPECT_EQ(*cxOffset, 512);

  auto ncxType = mlir::MemRefType::get(
      {2, 3, 1000}, f16, mlir::MemRefLayoutAttrInterface{}, ncxAttr);
  std::optional<wafer::WaferPhysicalTensorInfo> ncxInfo =
      wafer::computeWaferPhysicalTensorInfo(ncxType);
  ASSERT_TRUE(ncxInfo);
  EXPECT_EQ(ncxInfo->cBlock, 64);
  EXPECT_EQ(ncxInfo->cxBlocks, 16);
  EXPECT_EQ(ncxInfo->c0, 0);
  EXPECT_EQ(ncxInfo->alignedC, 1024);
  EXPECT_EQ(ncxInfo->batchElements, 3072);
  EXPECT_EQ(ncxInfo->physicalBytes, 12288);

  std::optional<int64_t> ncxOffset =
      wafer::computeWaferPhysicalElementByteOffset(ncxType, {1, 2, 64});
  ASSERT_TRUE(ncxOffset);
  EXPECT_EQ(*ncxOffset, 6784);
}

} // namespace
