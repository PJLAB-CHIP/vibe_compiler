#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace {

enum class TestElementKind { I8, F16, F32 };

struct SlowLayoutCase {
  const char *name;
  std::vector<int64_t> shape;
  wafer::MemLayout layout;
  TestElementKind elementKind;
  int64_t elementBytes;
  bool usesInt8Block;
  std::vector<int64_t> compactStrides;
  int64_t compactTypeOffset = 0;
};

struct SlowPhysicalLayout {
  int64_t physicalElements = 0;
  int64_t cBlock = 0;
  int64_t fullBlocks = 0;
  int64_t tailWidth = 0;
  int64_t alignedC = 0;
  int64_t outerElements = 0;
  int64_t hwElements = 0;
  int64_t batchElements = 0;
};

int64_t slowProduct(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values)
    result *= value;
  return result;
}

int64_t slowAlignUp(int64_t value, int64_t alignment) {
  if (value == 0)
    return 0;
  return ((value - 1) / alignment + 1) * alignment;
}

int64_t slowTailWidth(int64_t remainder) {
  for (int64_t width : {4, 8, 16, 32, 64})
    if (remainder <= width)
      return width;
  return 0;
}

std::vector<int64_t> slowCompactStrides(const SlowLayoutCase &testCase) {
  if (!testCase.compactStrides.empty())
    return testCase.compactStrides;
  std::vector<int64_t> strides(testCase.shape.size(), 1);
  for (int64_t index = static_cast<int64_t>(testCase.shape.size()) - 2;
       index >= 0; --index)
    strides[index] = strides[index + 1] * testCase.shape[index + 1];
  return strides;
}

SlowPhysicalLayout slowPhysicalLayout(const SlowLayoutCase &testCase) {
  SlowPhysicalLayout result;
  if (testCase.layout != wafer::MemLayout::Cx &&
      testCase.layout != wafer::MemLayout::NCx) {
    if (testCase.shape.empty()) {
      result.physicalElements = 1;
      return result;
    }
    if (llvm::is_contained(testCase.shape, int64_t{0}))
      return result;
    std::vector<int64_t> strides = slowCompactStrides(testCase);
    result.physicalElements = 1;
    for (auto [dim, stride] : llvm::zip_equal(testCase.shape, strides))
      result.physicalElements += (dim - 1) * stride;
    return result;
  }

  int64_t logicalC = testCase.shape.back();
  result.cBlock = testCase.usesInt8Block ? 128 : 64;
  int64_t retainThreshold = testCase.usesInt8Block ? 64 : 32;
  result.fullBlocks = logicalC / result.cBlock;
  int64_t remainder = logicalC % result.cBlock;
  if (remainder == 0) {
    result.tailWidth = 0;
  } else if (remainder <= retainThreshold) {
    result.tailWidth = slowTailWidth(remainder);
  } else {
    ++result.fullBlocks;
    result.tailWidth = 0;
  }
  result.alignedC = result.fullBlocks * result.cBlock + result.tailWidth;
  int64_t bankElements = 256 / testCase.elementBytes;
  if (testCase.layout == wafer::MemLayout::Cx) {
    result.outerElements =
        slowProduct(llvm::ArrayRef<int64_t>(testCase.shape).drop_back());
    result.hwElements = result.outerElements;
    result.batchElements =
        slowAlignUp(result.outerElements * result.alignedC, bankElements);
    result.physicalElements = result.batchElements;
    return result;
  }

  result.hwElements = slowProduct(
      llvm::ArrayRef<int64_t>(testCase.shape).drop_front().drop_back());
  result.outerElements = testCase.shape.front() * result.hwElements;
  result.batchElements =
      slowAlignUp(result.hwElements * result.alignedC, bankElements);
  result.physicalElements = testCase.shape.front() * result.batchElements;
  return result;
}

int64_t slowRowMajorIndex(llvm::ArrayRef<int64_t> shape,
                          llvm::ArrayRef<int64_t> indices) {
  int64_t linear = 0;
  for (auto [dim, index] : llvm::zip_equal(shape, indices))
    linear = linear * dim + index;
  return linear;
}

int64_t slowPhysicalElementOffset(const SlowLayoutCase &testCase,
                                  const SlowPhysicalLayout &layout,
                                  llvm::ArrayRef<int64_t> indices) {
  if (testCase.layout != wafer::MemLayout::Cx &&
      testCase.layout != wafer::MemLayout::NCx) {
    int64_t result = 0;
    for (auto [index, stride] :
         llvm::zip_equal(indices, slowCompactStrides(testCase)))
      result += index * stride;
    return result;
  }

  int64_t logicalC = indices.back();
  int64_t fullC = layout.fullBlocks * layout.cBlock;
  bool isTail = logicalC >= fullC;
  int64_t channelOffset = 0;
  if (isTail)
    channelOffset = logicalC - fullC;
  else
    channelOffset = logicalC % layout.cBlock;

  if (testCase.layout == wafer::MemLayout::Cx) {
    int64_t outer =
        slowRowMajorIndex(llvm::ArrayRef<int64_t>(testCase.shape).drop_back(),
                          indices.drop_back());
    if (isTail)
      return layout.fullBlocks * layout.outerElements * layout.cBlock +
             outer * layout.tailWidth + channelOffset;
    int64_t block = logicalC / layout.cBlock;
    return block * layout.outerElements * layout.cBlock +
           outer * layout.cBlock + channelOffset;
  }

  int64_t n = indices.front();
  int64_t hw = slowRowMajorIndex(
      llvm::ArrayRef<int64_t>(testCase.shape).drop_front().drop_back(),
      indices.drop_front().drop_back());
  int64_t batchBase = n * layout.batchElements;
  if (isTail)
    return batchBase + layout.fullBlocks * layout.hwElements * layout.cBlock +
           hw * layout.tailWidth + channelOffset;
  int64_t block = logicalC / layout.cBlock;
  return batchBase + block * layout.hwElements * layout.cBlock +
         hw * layout.cBlock + channelOffset;
}

template <typename Callback>
void forEachCoordinate(llvm::ArrayRef<int64_t> shape, Callback callback) {
  if (llvm::is_contained(shape, int64_t{0}))
    return;
  std::vector<int64_t> indices(shape.size(), 0);
  if (shape.empty()) {
    callback(indices);
    return;
  }
  while (true) {
    callback(indices);
    int64_t dim = static_cast<int64_t>(shape.size()) - 1;
    for (; dim >= 0; --dim) {
      if (++indices[dim] < shape[dim])
        break;
      indices[dim] = 0;
    }
    if (dim < 0)
      return;
  }
}

mlir::Type getElementType(mlir::MLIRContext &context, TestElementKind kind) {
  switch (kind) {
  case TestElementKind::I8:
    return mlir::IntegerType::get(&context, 8);
  case TestElementKind::F16:
    return mlir::Float16Type::get(&context);
  case TestElementKind::F32:
    return mlir::Float32Type::get(&context);
  }
  llvm_unreachable("unknown test element kind");
}

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

TEST(WaferDialectTest, SingleBatchNCxIsPhysicallyEquivalentToCx) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  auto cxMemory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                         wafer::MemLayout::Cx);
  auto ncxMemory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                          wafer::MemLayout::NCx);
  mlir::Type f16 = mlir::Float16Type::get(&context);

  // C=129 exercises two complete 64-lane blocks and a retained C0 tail.
  constexpr int64_t m = 3;
  constexpr int64_t c = 129;
  auto cxType = mlir::MemRefType::get(
      {m, c}, f16, mlir::MemRefLayoutAttrInterface{}, cxMemory);
  auto singleBatchNCxType = mlir::MemRefType::get(
      {1, m, c}, f16, mlir::MemRefLayoutAttrInterface{}, ncxMemory);
  std::optional<wafer::WaferPhysicalTensorInfo> cxInfo =
      wafer::computeWaferPhysicalTensorInfo(cxType);
  std::optional<wafer::WaferPhysicalTensorInfo> ncxInfo =
      wafer::computeWaferPhysicalTensorInfo(singleBatchNCxType);
  ASSERT_TRUE(cxInfo);
  ASSERT_TRUE(ncxInfo);
  ASSERT_EQ(cxInfo->cxBlocks, 2);
  ASSERT_EQ(cxInfo->c0, 4);
  EXPECT_EQ(ncxInfo->cxBlocks, cxInfo->cxBlocks);
  EXPECT_EQ(ncxInfo->c0, cxInfo->c0);
  EXPECT_EQ(ncxInfo->physicalElements, cxInfo->physicalElements);
  EXPECT_EQ(ncxInfo->physicalBytes, cxInfo->physicalBytes);

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t channel = 0; channel < c; ++channel) {
      SCOPED_TRACE(testing::Message()
                   << "row=" << row << " channel=" << channel);
      std::optional<int64_t> cxOffset =
          wafer::computeWaferPhysicalElementByteOffset(cxType, {row, channel});
      std::optional<int64_t> ncxOffset =
          wafer::computeWaferPhysicalElementByteOffset(singleBatchNCxType,
                                                       {0, row, channel});
      ASSERT_TRUE(cxOffset);
      ASSERT_TRUE(ncxOffset);
      EXPECT_EQ(*ncxOffset, *cxOffset);
    }
  }
}

TEST(WaferDialectTest, PhysicalLayoutMatchesIndependentSlowCoordinateOracle) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  const std::vector<SlowLayoutCase> cases = {
      {"tensor-rank0",
       {},
       wafer::MemLayout::Tensor,
       TestElementKind::F32,
       4,
       false},
      {"tensor-contiguous",
       {2, 3},
       wafer::MemLayout::Tensor,
       TestElementKind::F32,
       4,
       false},
      {"ntensor-contiguous",
       {2, 3, 5},
       wafer::MemLayout::NTensor,
       TestElementKind::F16,
       2,
       false},
      {"tensor-strided-view",
       {2, 3},
       wafer::MemLayout::Tensor,
       TestElementKind::F32,
       4,
       false,
       {5, 1},
       7},
      {"cx-f16-tail-align-4-minus-one",
       {3, 3},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-tail-align-4",
       {3, 4},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-tail-align-4-plus-one",
       {3, 5},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-tail-align-8-minus-one",
       {3, 7},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-tail-align-8",
       {3, 8},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-tail-align-8-plus-one",
       {3, 9},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-tail-align-16-minus-one",
       {3, 15},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-tail-align-16",
       {3, 16},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-tail-align-16-plus-one",
       {3, 17},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-tail-threshold-minus-one",
       {3, 31},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-tail-threshold",
       {3, 32},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-tail-threshold-plus-one",
       {3, 33},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-block-minus-one",
       {3, 63},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-block",
       {3, 64},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-block-plus-one",
       {3, 65},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-retained-tail",
       {3, 96},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-f16-folded-tail",
       {3, 97},
       wafer::MemLayout::Cx,
       TestElementKind::F16,
       2,
       false},
      {"cx-i8-tail-threshold-minus-one",
       {2, 63},
       wafer::MemLayout::Cx,
       TestElementKind::I8,
       1,
       true},
      {"cx-i8-tail-threshold",
       {2, 64},
       wafer::MemLayout::Cx,
       TestElementKind::I8,
       1,
       true},
      {"cx-i8-tail-threshold-plus-one",
       {2, 65},
       wafer::MemLayout::Cx,
       TestElementKind::I8,
       1,
       true},
      {"cx-i8-block-minus-one",
       {2, 127},
       wafer::MemLayout::Cx,
       TestElementKind::I8,
       1,
       true},
      {"cx-i8-block",
       {2, 128},
       wafer::MemLayout::Cx,
       TestElementKind::I8,
       1,
       true},
      {"cx-i8-block-plus-one",
       {2, 129},
       wafer::MemLayout::Cx,
       TestElementKind::I8,
       1,
       true},
      {"cx-i8-retained-tail",
       {2, 192},
       wafer::MemLayout::Cx,
       TestElementKind::I8,
       1,
       true},
      {"cx-i8-folded-tail",
       {2, 193},
       wafer::MemLayout::Cx,
       TestElementKind::I8,
       1,
       true},
      {"ncx-f32-rank3-block-minus-one",
       {2, 3, 63},
       wafer::MemLayout::NCx,
       TestElementKind::F32,
       4,
       false},
      {"ncx-f32-rank3-block",
       {2, 3, 64},
       wafer::MemLayout::NCx,
       TestElementKind::F32,
       4,
       false},
      {"ncx-f32-rank3-block-plus-one",
       {2, 3, 65},
       wafer::MemLayout::NCx,
       TestElementKind::F32,
       4,
       false},
      {"ncx-f32-rank3-retained-tail",
       {2, 3, 96},
       wafer::MemLayout::NCx,
       TestElementKind::F32,
       4,
       false},
      {"ncx-f32-rank3-folded-tail",
       {2, 3, 97},
       wafer::MemLayout::NCx,
       TestElementKind::F32,
       4,
       false},
      {"ncx-i8-rank4-block-minus-one",
       {2, 2, 3, 127},
       wafer::MemLayout::NCx,
       TestElementKind::I8,
       1,
       true},
      {"ncx-i8-rank4-block",
       {2, 2, 3, 128},
       wafer::MemLayout::NCx,
       TestElementKind::I8,
       1,
       true},
      {"ncx-i8-rank4-block-plus-one",
       {2, 2, 3, 129},
       wafer::MemLayout::NCx,
       TestElementKind::I8,
       1,
       true},
      {"ncx-i8-rank4-retained-tail",
       {2, 2, 3, 192},
       wafer::MemLayout::NCx,
       TestElementKind::I8,
       1,
       true},
      {"ncx-i8-rank4-folded-tail",
       {2, 2, 3, 193},
       wafer::MemLayout::NCx,
       TestElementKind::I8,
       1,
       true},
  };

  for (const SlowLayoutCase &testCase : cases) {
    SCOPED_TRACE(testCase.name);
    auto memory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                         testCase.layout);
    mlir::MemRefLayoutAttrInterface typeLayout;
    if (!testCase.compactStrides.empty())
      typeLayout = mlir::StridedLayoutAttr::get(
          &context, testCase.compactTypeOffset, testCase.compactStrides);
    auto type = mlir::MemRefType::get(
        testCase.shape, getElementType(context, testCase.elementKind),
        typeLayout, memory);
    SlowPhysicalLayout expectedLayout = slowPhysicalLayout(testCase);
    std::optional<wafer::WaferPhysicalTensorInfo> actualLayout =
        wafer::computeWaferPhysicalTensorInfo(type);
    ASSERT_TRUE(actualLayout);
    EXPECT_EQ(actualLayout->physicalElements, expectedLayout.physicalElements);
    EXPECT_EQ(actualLayout->physicalBytes,
              expectedLayout.physicalElements * testCase.elementBytes);
    if (testCase.layout == wafer::MemLayout::Cx ||
        testCase.layout == wafer::MemLayout::NCx) {
      EXPECT_EQ(actualLayout->cBlock, expectedLayout.cBlock);
      EXPECT_EQ(actualLayout->cxBlocks, expectedLayout.fullBlocks);
      EXPECT_EQ(actualLayout->c0, expectedLayout.tailWidth);
      EXPECT_EQ(actualLayout->alignedC, expectedLayout.alignedC);
      EXPECT_EQ(actualLayout->outerElements, expectedLayout.outerElements);
      EXPECT_EQ(actualLayout->hwElements, expectedLayout.hwElements);
      EXPECT_EQ(actualLayout->batchElements, expectedLayout.batchElements);
    }

    std::optional<wafer::WaferStaticPhysicalOffsetCalculator> calculator =
        wafer::WaferStaticPhysicalOffsetCalculator::create(type);
    ASSERT_TRUE(calculator);

    std::set<int64_t> occupiedOffsets;
    forEachCoordinate(testCase.shape, [&](llvm::ArrayRef<int64_t> indices) {
      std::optional<int64_t> actual =
          wafer::computeWaferPhysicalElementByteOffset(type, indices);
      ASSERT_TRUE(actual);
      int64_t expected =
          slowPhysicalElementOffset(testCase, expectedLayout, indices) *
          testCase.elementBytes;
      EXPECT_EQ(*actual, expected);
      if (calculator) {
        EXPECT_EQ(calculator->getByteOffset(indices), expected);
        EXPECT_EQ(calculator->getByteOffsetForValidIndices(indices), expected);
      }
      EXPECT_GE(*actual, 0);
      EXPECT_LE(*actual + testCase.elementBytes, actualLayout->physicalBytes);
      EXPECT_TRUE(occupiedOffsets.insert(*actual).second);
    });

    std::vector<int64_t> wrongRank(testCase.shape.size() + 1, 0);
    EXPECT_FALSE(wafer::computeWaferPhysicalElementByteOffset(type, wrongRank));
    for (size_t dim = 0; dim < testCase.shape.size(); ++dim) {
      std::vector<int64_t> invalid(testCase.shape.size(), 0);
      invalid[dim] = -1;
      EXPECT_FALSE(wafer::computeWaferPhysicalElementByteOffset(type, invalid));
      invalid[dim] = testCase.shape[dim];
      EXPECT_FALSE(wafer::computeWaferPhysicalElementByteOffset(type, invalid));
    }
  }
}

TEST(WaferDialectTest, ComputesBitpackedOffsetsWithoutGuessingBitOrder) {
  mlir::DialectRegistry registry;
  wafer::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<wafer::WaferDialect>();

  auto tensorMemory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                             wafer::MemLayout::Tensor);
  mlir::Type i1 = mlir::IntegerType::get(&context, 1);
  auto contiguous = mlir::MemRefType::get(
      {2, 9}, i1, mlir::MemRefLayoutAttrInterface{}, tensorMemory);
  EXPECT_FALSE(wafer::WaferStaticPhysicalOffsetCalculator::create(contiguous));
  std::optional<wafer::WaferPhysicalTensorInfo> contiguousInfo =
      wafer::computeWaferPhysicalTensorInfo(contiguous);
  ASSERT_TRUE(contiguousInfo);
  EXPECT_TRUE(contiguousInfo->bitPackedElement);
  EXPECT_EQ(contiguousInfo->physicalElements, 18);
  EXPECT_EQ(contiguousInfo->physicalBytes, 3);
  EXPECT_FALSE(
      wafer::computeWaferPhysicalElementByteOffset(contiguous, {0, 0}));

  std::set<int64_t> occupiedBits;
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t column = 0; column < 9; ++column) {
      std::optional<int64_t> bit = wafer::computeWaferPhysicalElementBitOffset(
          contiguous, {row, column});
      ASSERT_TRUE(bit);
      EXPECT_EQ(*bit, row * 9 + column);
      EXPECT_TRUE(occupiedBits.insert(*bit).second);
    }
  }

  auto stridedLayout = mlir::StridedLayoutAttr::get(&context, 7, {5, 1});
  auto strided = mlir::MemRefType::get({2, 3}, i1, stridedLayout, tensorMemory);
  EXPECT_EQ(wafer::computeWaferPhysicalElementBitOffset(strided, {0, 0}), 0);
  EXPECT_EQ(wafer::computeWaferPhysicalElementBitOffset(strided, {0, 2}), 2);
  EXPECT_EQ(wafer::computeWaferPhysicalElementBitOffset(strided, {1, 0}), 5);
  EXPECT_EQ(wafer::computeWaferPhysicalElementBitOffset(strided, {1, 2}), 7);
  EXPECT_FALSE(wafer::computeWaferPhysicalElementBitOffset(strided, {2, 0}));

  auto cxMemory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                         wafer::MemLayout::Cx);
  auto cx = mlir::MemRefType::get({2, 9}, i1, mlir::MemRefLayoutAttrInterface{},
                                  cxMemory);
  EXPECT_FALSE(wafer::computeWaferPhysicalElementBitOffset(cx, {0, 0}));

  auto f16 =
      mlir::MemRefType::get({2, 3}, mlir::Float16Type::get(&context),
                            mlir::MemRefLayoutAttrInterface{}, tensorMemory);
  EXPECT_EQ(wafer::computeWaferPhysicalElementByteOffset(f16, {1, 2}), 10);
  EXPECT_EQ(wafer::computeWaferPhysicalElementBitOffset(f16, {1, 2}), 80);
}

} // namespace
