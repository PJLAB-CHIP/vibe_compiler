#include "Wafer/Conversion/InstrToLLVM/InstrToLLVM.h"
#include "Wafer/Conversion/InstrToLLVM/LowerInstrToTargetLLVMInternal.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Support/CompileWorkStatistics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <map>
#include <string>
#include <vector>

namespace {

void registerTargetConversionDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  wafer::registerWaferCoreDialects(registry);
}

template <typename OpT> unsigned countOps(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](OpT) { ++count; });
  return count;
}

TEST(LowerInstrToTargetLLVMTest,
     NativeReductionDoesNotDependOnOrderedExpansionBudget) {
  auto check = [](int64_t rows, int64_t width, llvm::StringRef type,
                  llvm::StringRef kind, llvm::StringRef identity) {
    SCOPED_TRACE(llvm::formatv("rows={0} width={1} type={2} kind={3}", rows,
                               width, type, kind)
                     .str());
    mlir::DialectRegistry registry;
    registerTargetConversionDialects(registry);
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    auto text = llvm::formatv(R"mlir(
module {{
  func.func @main() {{
    %token = arith.constant false
    %unused = wafer.tile.region(%token : i1) -> (i1) {{
    ^bb0(%done: i1):
      %input = memref.alloc() : memref<1x{0}x{1}x{2}, #wafer.memory<spm, ncx>>
      %result = wafer.tile.reduce <{3}> %input
          {{dimensions = array<i64: 2>, init_value = {4} : {2}}
          : (memref<1x{0}x{1}x{2}, #wafer.memory<spm, ncx>>)
         -> memref<1x{0}x{2}, #wafer.memory<spm, cx>>
      wafer.tile.yield %done : i1
    }
    return
  }
}
)mlir",
                              rows, width, type, kind, identity)
                    .str();
    auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(source) << text;
    wafer::TileRegionOp region;
    source->walk([&](wafer::TileRegionOp op) { region = op; });
    wafer::TileRegionToInstrLoweringSession session(context);
    ASSERT_TRUE(
        mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
    EXPECT_EQ(countOps<wafer::ComputeReduceOp>(*source), 0u);
    ASSERT_EQ(countOps<wafer::InstrReduceOp>(*source), 1u);
    EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 0u);
    EXPECT_EQ(countOps<wafer::InstrFillOp>(*source), 0u);
    EXPECT_EQ(countOps<mlir::scf::ForOp>(*source), 0u);
    wafer::InstrReduceOp reduce;
    source->walk([&](wafer::InstrReduceOp op) { reduce = op; });
    EXPECT_EQ(reduce.getDimAttr().getInt(), 0);
    auto expectedKind = kind == "sum"   ? wafer::InstrReduceKind::Sum
                        : kind == "max" ? wafer::InstrReduceKind::Max
                                        : wafer::InstrReduceKind::Min;
    EXPECT_EQ(reduce.getKind(), expectedKind);
    auto nativeType = mlir::cast<mlir::MemRefType>(reduce.getDest().getType());
    EXPECT_EQ(nativeType.getShape(), (llvm::ArrayRef<int64_t>{1, rows, 1}));
    EXPECT_EQ(wafer::getWaferMemoryAttr(nativeType).getLayout(),
              wafer::MemLayout::NCx);
    EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*source),
              rows == 1024 ? 1u : 2u);
    std::map<int64_t, int64_t> actualBytes;
    std::map<int64_t, int64_t> expectedBytes;
    source->walk([&](wafer::InstrGatherScatterOp move) {
      EXPECT_EQ(move.getSource(), reduce.getDest());
      auto resultType = mlir::cast<mlir::MemRefType>(move.getDest().getType());
      EXPECT_EQ(resultType.getShape(), (llvm::ArrayRef<int64_t>{1, rows}));
      EXPECT_EQ(wafer::getWaferMemoryAttr(resultType).getLayout(),
                wafer::MemLayout::Cx);
      for (int64_t row = 0; row < rows; ++row) {
        auto src = wafer::computeWaferPhysicalElementByteOffset(nativeType,
                                                                {0, row, 0});
        auto dst =
            wafer::computeWaferPhysicalElementByteOffset(resultType, {0, row});
        ASSERT_TRUE(src && dst);
        for (int64_t byte = 0; byte < (type == "f32" ? 4 : 2); ++byte)
          expectedBytes[*dst + byte] = *src + byte;
      }
      auto expand = [&](mlir::DenseI64ArrayAttr iterations,
                        mlir::DenseI64ArrayAttr strides,
                        mlir::IntegerAttr offset) {
        std::vector<int64_t> bytes;
        auto counts = iterations.asArrayRef();
        auto steps = strides.asArrayRef();
        int64_t base = offset ? offset.getInt() : 0;
        for (int64_t k = 0; k < counts[2]; ++k)
          for (int64_t j = 0; j < counts[1]; ++j)
            for (int64_t i = 0; i < counts[0]; ++i)
              for (int64_t byte = 0; byte < move.getInnerBytesAttr().getInt();
                   ++byte)
                bytes.push_back(base + i * steps[0] + j * steps[1] +
                                k * steps[2] + byte);
        return bytes;
      };
      auto src = expand(move.getSrcIterationsAttr(), move.getSrcStridesAttr(),
                        move.getSrcOffsetAttr());
      auto dst = expand(move.getDstIterationsAttr(), move.getDstStridesAttr(),
                        move.getDstOffsetAttr());
      ASSERT_EQ(src.size(), dst.size());
      EXPECT_EQ(src.size(),
                static_cast<size_t>(move.getByteCountAttr().getInt()));
      for (size_t i = 0; i < src.size(); ++i)
        EXPECT_TRUE(actualBytes.emplace(dst[i], src[i]).second);
    });
    EXPECT_EQ(actualBytes, expectedBytes);
  };
  for (int64_t rows : {1024, 1025, 1031})
    for (llvm::StringRef type : {"f16", "bf16", "f32"}) {
      check(rows, 512, type, "sum", "0.0");
      check(rows, 512, type, "max",
            type == "f32"   ? "0xFF800000"
            : type == "f16" ? "0xFC00"
                            : "0xFF80");
      check(rows, 512, type, "min",
            type == "f32"   ? "0x7F800000"
            : type == "f16" ? "0x7C00"
                            : "0x7F80");
    }
  for (int64_t width : {1, 8, 1023, 1024, 1025, 1031})
    check(1024, width, "f32", "sum", "0.0");
}

TEST(LowerInstrToTargetLLVMTest,
     StoresPreserveDynamicStandardSubviewAddresses) {
  for (int64_t extent : {1024, 1025, 1031})
    for (auto layout : {wafer::MemLayout::Tensor, wafer::MemLayout::Cx,
                        wafer::MemLayout::NCx}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(static_cast<unsigned>(layout));
      mlir::DialectRegistry registry;
      registerTargetConversionDialects(registry);
      mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
      mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
      mlir::MLIRContext context(registry);
      context.loadAllAvailableDialects();
      auto text = llvm::formatv(R"mlir(
        module {{
          func.func @main(%output: memref<1x1x{0}xf16, #wafer.memory<ddr, tensor>>) {{
            %token = arith.constant false
            %unused = wafer.tile.region(%output, %token : memref<1x1x{0}xf16, #wafer.memory<ddr, tensor>>, i1) -> (i1) {{
            ^bb0(%destination: memref<1x1x{0}xf16, #wafer.memory<ddr, tensor>>, %done: i1):
              wafer.tile.yield %done : i1
            }
            return
          }
        }
      )mlir",
                                extent + 4)
                      .str();
      auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      ASSERT_TRUE(module);
      wafer::TileRegionOp region;
      module->walk([&](wafer::TileRegionOp op) { region = op; });
      mlir::OpBuilder builder(region.getBody().front().getTerminator());
      auto loc = region.getLoc();
      auto memory =
          wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM, layout);
      auto type =
          mlir::MemRefType::get({1, 1, extent + 6}, builder.getF16Type(),
                                mlir::MemRefLayoutAttrInterface{}, memory);
      auto allocation = builder.create<mlir::memref::AllocOp>(loc, type);
      auto source = builder.create<mlir::memref::SubViewOp>(
          loc, allocation, llvm::ArrayRef<int64_t>{0, 0, 3},
          llvm::ArrayRef<int64_t>{1, 1, extent},
          llvm::ArrayRef<int64_t>{1, 1, 1});
      auto destination = builder.create<mlir::memref::SubViewOp>(
          loc, region.getBody().getArgument(0),
          llvm::ArrayRef<int64_t>{0, 0, 2},
          llvm::ArrayRef<int64_t>{1, 1, extent},
          llvm::ArrayRef<int64_t>{1, 1, 1});
      auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto end =
          builder.create<mlir::arith::ConstantIndexOp>(loc, extent / 128 * 128);
      auto step = builder.create<mlir::arith::ConstantIndexOp>(loc, 128);
      auto loop = builder.create<mlir::scf::ForOp>(loc, zero, end, step);
      auto store = [&](int64_t size, mlir::OpFoldResult offset) {
        llvm::SmallVector<mlir::OpFoldResult> offsets{
            builder.getIndexAttr(0), builder.getIndexAttr(0), offset};
        auto sizes = mlir::getAsIndexOpFoldResult(&context, {1, 1, size});
        auto strides = mlir::getAsIndexOpFoldResult(&context, {1, 1, 1});
        auto src = builder.create<mlir::memref::SubViewOp>(loc, source, offsets,
                                                           sizes, strides);
        auto dst = builder.create<mlir::memref::SubViewOp>(
            loc, destination, offsets, sizes, strides);
        builder.create<wafer::StorageStoreOp>(loc, src, dst);
      };
      builder.setInsertionPoint(loop.getBody()->getTerminator());
      store(128, loop.getInductionVar());
      builder.setInsertionPointAfter(loop);
      if (extent % 128)
        store(extent % 128, builder.getIndexAttr(extent / 128 * 128));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      wafer::TileRegionToInstrLoweringSession session(context);
      bool standard = layout == wafer::MemLayout::Tensor;
      bool lowered =
          mlir::succeeded(wafer::convertTileRegionToInstr(region, session));
      EXPECT_EQ(lowered, standard);
      if (!standard) {
        EXPECT_EQ(countOps<wafer::StorageStoreOp>(*module),
                  extent % 128 ? 2u : 1u);
        continue;
      }
      ASSERT_TRUE(lowered);
      EXPECT_EQ(countOps<wafer::StorageStoreOp>(*module), 0u);
      EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 1u);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      std::map<int64_t, int64_t> actual, expected;
      for (int64_t byte = 0; byte < 2 * extent; ++byte)
        expected[4 + byte] = 6 + byte;
      auto address = [&](auto &&self, mlir::Value value,
                         int64_t iv) -> int64_t {
        auto view = value.getDefiningOp<mlir::memref::SubViewOp>();
        if (!view)
          return 0;
        llvm::SmallVector<int64_t> strides;
        int64_t ignored;
        EXPECT_TRUE(mlir::succeeded(
            mlir::getStridesAndOffset(view.getSourceType(), strides, ignored)));
        int64_t result = self(self, view.getSource(), iv);
        for (auto [axis, offset] : llvm::enumerate(view.getMixedOffsets())) {
          auto constant = mlir::getConstantIntValue(offset);
          if (!constant) {
            EXPECT_EQ(mlir::cast<mlir::Value>(offset), loop.getInductionVar());
          }
          result += 2 * strides[axis] * (constant ? *constant : iv);
        }
        return result;
      };
      module->walk([&](wafer::InstrWDMAOp dma) {
        bool repeated =
            static_cast<bool>(dma->getParentOfType<mlir::scf::ForOp>());
        for (int64_t iv = 0; iv < (repeated ? extent / 128 * 128 : 1);
             iv += 128) {
          int64_t src =
              address(address, dma.getSource(), iv) +
              (dma.getSrcOffsetAttr() ? dma.getSrcOffsetAttr().getInt() : 0);
          int64_t dst =
              address(address, dma.getDest(), iv) +
              (dma.getDstOffsetAttr() ? dma.getDstOffsetAttr().getInt() : 0);
          auto n = dma.getDstIterations();
          auto strides = dma.getDstStrides();
          int64_t read = 0;
          for (int64_t k = 0; k < n[2]; ++k)
            for (int64_t j = 0; j < n[1]; ++j)
              for (int64_t i = 0; i < n[0]; ++i)
                for (int64_t byte = 0; byte < dma.getInnerBytesAttr().getInt();
                     ++byte)
                  EXPECT_TRUE(actual
                                  .emplace(dst + k * strides[2] +
                                               j * strides[1] + i * strides[0] +
                                               byte,
                                           src + read++)
                                  .second);
          EXPECT_EQ(read, dma.getByteCountAttr().getInt());
        }
      });
      EXPECT_EQ(actual, expected);
    }
}

TEST(LowerInstrToTargetLLVMTest, CopySubviewEndpointsUseTheirBaseCoordinates) {
  for (int64_t extent : {1024, 1025, 1031})
    for (wafer::MemLayout layout :
         {wafer::MemLayout::Tensor, wafer::MemLayout::Cx,
          wafer::MemLayout::NCx})
      for (unsigned views : {1u, 2u, 3u}) {
        SCOPED_TRACE(llvm::formatv("extent={0} layout={1} views={2}", extent,
                                   static_cast<unsigned>(layout), views)
                         .str());
        mlir::DialectRegistry registry;
        registerTargetConversionDialects(registry);
        mlir::MLIRContext context(registry);
        context.loadAllAvailableDialects();
        auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
          module {
            func.func @main() {
              %token = arith.constant false
              %unused = wafer.tile.region(%token : i1) -> (i1) {
              ^bb0(%done: i1):
                wafer.tile.yield %done : i1
              }
              return
            }
          })mlir",
                                                              &context);
        ASSERT_TRUE(module);
        wafer::TileRegionOp region;
        module->walk([&](wafer::TileRegionOp op) { region = op; });
        mlir::OpBuilder builder(region.getBody().front().getTerminator());
        auto loc = region.getLoc();
        auto memory =
            wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM, layout);
        bool reduced = views == 3;
        llvm::SmallVector<int64_t> shape =
            reduced ? llvm::SmallVector<int64_t>{1, extent, 16}
                    : llvm::SmallVector<int64_t>{1, 1, extent, 16};
        auto makeEndpoint = [&](bool useView) {
          auto baseShape =
              useView ? llvm::SmallVector<int64_t>{1, 1, 2 * extent + 7, 16}
                      : shape;
          auto type =
              mlir::MemRefType::get(baseShape, builder.getF16Type(),
                                    mlir::MemRefLayoutAttrInterface{}, memory);
          mlir::Value base = builder.create<mlir::memref::AllocOp>(loc, type);
          mlir::Value value = base;
          if (useView) {
            auto outer = builder.create<mlir::memref::SubViewOp>(
                loc, base, llvm::ArrayRef<int64_t>{0, 0, 2, 0},
                llvm::ArrayRef<int64_t>{1, 1, extent + 2, 16},
                llvm::ArrayRef<int64_t>{1, 1, 2, 1});
            llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
            for (int64_t n : {0, 0, 1, 0})
              offsets.push_back(builder.getIndexAttr(n));
            for (int64_t n : {int64_t(1), int64_t(1), extent, int64_t(16)})
              sizes.push_back(builder.getIndexAttr(n));
            strides.assign(4, builder.getIndexAttr(1));
            auto viewType = mlir::cast<mlir::MemRefType>(
                mlir::memref::SubViewOp::inferRankReducedResultType(
                    shape, outer.getType(), offsets, sizes, strides));
            value = builder.create<mlir::memref::SubViewOp>(
                loc, viewType, outer, offsets, sizes, strides);
          }
          return std::make_pair(base, value);
        };
        auto [sourceBase, source] = makeEndpoint(views & 1);
        auto [destBase, dest] = makeEndpoint(views & 2);
        builder.create<mlir::memref::CopyOp>(loc, source, dest);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        wafer::TileRegionToInstrLoweringSession session(context);
        ASSERT_TRUE(
            mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        EXPECT_EQ(countOps<mlir::memref::CopyOp>(*module), 0u);
        EXPECT_EQ(countOps<mlir::memref::SubViewOp>(*module), 0u);
        EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 2u);
        std::map<int64_t, int64_t> actual, expected;
        auto physicalOffset = [&](mlir::Value base, bool hasView, int64_t row,
                                  int64_t column) {
          llvm::SmallVector<int64_t> index =
              hasView ? llvm::SmallVector<int64_t>{0, 0, 4 + 2 * row, column}
                      : llvm::SmallVector<int64_t>{0, 0, row, column};
          if (reduced && !hasView)
            index.erase(index.begin());
          return wafer::computeWaferPhysicalElementByteOffset(
              mlir::cast<mlir::MemRefType>(base.getType()), index);
        };
        // The oracle enumerates logical points directly; it does not use the
        // production slice relation or descriptor partitioning algorithm.
        for (int64_t row = 0; row < extent; ++row)
          for (int64_t column = 0; column < 16; ++column) {
            auto src = physicalOffset(sourceBase, views & 1, row, column);
            auto dst = physicalOffset(destBase, views & 2, row, column);
            ASSERT_TRUE(src && dst);
            for (int64_t byte : {0, 1})
              expected[*dst + byte] = *src + byte;
          }
        module->walk([&](wafer::InstrGatherScatterOp move) {
          EXPECT_EQ(move.getSource(), sourceBase);
          EXPECT_EQ(move.getDest(), destBase);
          auto expand = [&](mlir::DenseI64ArrayAttr iterations,
                            mlir::DenseI64ArrayAttr strides,
                            mlir::IntegerAttr offset) {
            std::vector<int64_t> result;
            auto n = iterations.asArrayRef();
            auto stride = strides.asArrayRef();
            int64_t base = offset ? offset.getInt() : 0;
            for (int64_t k = 0; k < n[2]; ++k)
              for (int64_t j = 0; j < n[1]; ++j)
                for (int64_t i = 0; i < n[0]; ++i)
                  for (int64_t b = 0; b < move.getInnerBytesAttr().getInt();
                       ++b)
                    result.push_back(base + i * stride[0] + j * stride[1] +
                                     k * stride[2] + b);
            return result;
          };
          auto src = expand(move.getSrcIterationsAttr(),
                            move.getSrcStridesAttr(), move.getSrcOffsetAttr());
          auto dst = expand(move.getDstIterationsAttr(),
                            move.getDstStridesAttr(), move.getDstOffsetAttr());
          ASSERT_EQ(src.size(), dst.size());
          EXPECT_EQ(src.size(),
                    static_cast<size_t>(move.getByteCountAttr().getInt()));
          for (size_t i = 0; i < src.size(); ++i)
            EXPECT_TRUE(actual.emplace(dst[i], src[i]).second);
        });
        EXPECT_EQ(actual, expected);
      }
}

TEST(LowerInstrToTargetLLVMTest, RejectsUnprovenCopySubviewEndpoints) {
  for (unsigned kind : {0u, 1u, 2u}) {
    bool dynamic = kind != 0;
    mlir::DialectRegistry registry;
    registerTargetConversionDialects(registry);
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    std::string text = R"mlir(
      module {
        func.func @main(%offset: index) {
          %unused = wafer.tile.region(%offset : index) -> (index) {
          ^bb0(%position: index):
            %base = memref.alloc() : memref<1x1025x32xf16, #wafer.memory<spm, ncx>>
            %source = memref.alloc() : memref<1x1024x32xf16, #wafer.memory<spm, ncx>>
            %view = memref.subview %base[0, OFFSET, 0] [1, 1024, 32] [1, 1, 1]
              : memref<1x1025x32xf16, #wafer.memory<spm, ncx>>
                to memref<1x1024x32xf16, strided<[32800, 32, 1], offset: BYTEOFFSET>, #wafer.memory<spm, ncx>>
            memref.copy %source, %view
              : memref<1x1024x32xf16, #wafer.memory<spm, ncx>>
                to memref<1x1024x32xf16, strided<[32800, 32, 1], offset: BYTEOFFSET>, #wafer.memory<spm, ncx>>
            wafer.tile.yield %position : index
          }
          return
        }
      })mlir";
    text.replace(text.find("OFFSET"), 6, dynamic ? "%position" : "2");
    for (size_t i; (i = text.find("BYTEOFFSET")) != std::string::npos;)
      text.replace(i, 10, dynamic ? "?" : "64");
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    wafer::TileRegionOp region;
    module->walk([&](wafer::TileRegionOp op) { region = op; });
    if (kind == 2) {
      module->walk([&](mlir::memref::CopyOp copy) {
        mlir::OpBuilder builder(copy);
        mlir::IRMapping mapping;
        auto *view = builder.clone(*copy.getTarget().getDefiningOp(), mapping);
        copy.getSourceMutable().assign(view->getResult(0));
      });
    }
    wafer::TileRegionToInstrLoweringSession session(context);
    EXPECT_EQ(mlir::succeeded(wafer::convertTileRegionToInstr(region, session)),
              kind == 2);
    EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
    EXPECT_EQ(countOps<mlir::memref::CopyOp>(*module), kind == 2 ? 0u : 1u);
  }
}

TEST(LowerInstrToTargetLLVMTest, DescriptorReusePreservesEachActualMovement) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  std::string text;
  llvm::raw_string_ostream out(text);
  out << "module { func.func @main() { %token = arith.constant false\n";
  for (unsigned index = 0; index < 36; ++index) {
    int64_t extent = llvm::ArrayRef<int64_t>{1024, 1025, 1031}[index / 12];
    out << llvm::formatv(R"mlir(
      %result{0} = wafer.tile.region(%token : i1) -> (i1) {{
      ^bb0(%done: i1):
        %source = memref.alloc() : memref<1x{1}x65xf16, #wafer.memory<spm, ncx>>
        %dest = memref.alloc() : memref<1x{1}x65xf16, #wafer.memory<spm, ncx>>
        %output = memref.alloc() : memref<1x{1}x65xf16, #wafer.memory<ddr, tensor>>
        memref.copy %source, %dest
          : memref<1x{1}x65xf16, #wafer.memory<spm, ncx>> to memref<1x{1}x65xf16, #wafer.memory<spm, ncx>>
        wafer.tile.copy_into %source into %dest
          : memref<1x{1}x65xf16, #wafer.memory<spm, ncx>> into memref<1x{1}x65xf16, #wafer.memory<spm, ncx>>
        wafer.tile.store %dest, %output
          : memref<1x{1}x65xf16, #wafer.memory<spm, ncx>> -> memref<1x{1}x65xf16, #wafer.memory<ddr, tensor>>
        wafer.tile.yield %done : i1
      }
    )mlir",
                         index, extent);
  }
  out << "return } }";
  auto lower = [&](bool shared) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    EXPECT_TRUE(module);
    auto work =
        std::make_shared<wafer::support::CompileWorkStatisticsSession>();
    wafer::support::ScopedCompileWorkStatisticsActivation activation(work);
    wafer::TileRegionToInstrLoweringSession sharedSession(context);
    module->walk([&](wafer::TileRegionOp region) {
      if (shared) {
        EXPECT_TRUE(mlir::succeeded(
            wafer::convertTileRegionToInstr(region, sharedSession)));
      } else {
        wafer::TileRegionToInstrLoweringSession independent(context);
        EXPECT_TRUE(mlir::succeeded(
            wafer::convertTileRegionToInstr(region, independent)));
      }
    });
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 108u);
    EXPECT_GT(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
    EXPECT_GT(countOps<wafer::InstrWDMAOp>(*module), 0u);
    std::string result;
    llvm::raw_string_ostream stream(result);
    module->print(stream);
    return std::make_pair(result, work->snapshot().relationDescriptorPlannings);
  };
  auto independent = lower(false);
  auto shared = lower(true);
  EXPECT_EQ(shared.first, independent.first);
  EXPECT_LT(shared.second, independent.second);
  EXPECT_LE(shared.second, 18u); // Three warm-up queries for each of six keys.
  EXPECT_EQ(independent.second, 108u);
}

TEST(LowerInstrToTargetLLVMTest, DescriptorReuseDoesNotCacheUnsupportedCopies) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  wafer::TileRegionToInstrLoweringSession session(context);
  auto work = std::make_shared<wafer::support::CompileWorkStatisticsSession>();
  wafer::support::ScopedCompileWorkStatisticsActivation activation(work);
  for (unsigned repeat = 0; repeat < 4; ++repeat) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
      module { func.func @main() {
        %token = arith.constant false
        %unused = wafer.tile.region(%token : i1) -> (i1) {
        ^bb0(%done: i1):
          %a = memref.alloc() : memref<1x1024x65xi1, #wafer.memory<spm, tensor>>
          %b = memref.alloc() : memref<1x1024x65xi1, #wafer.memory<spm, tensor>>
          memref.copy %a, %b : memref<1x1024x65xi1, #wafer.memory<spm, tensor>>
                           to memref<1x1024x65xi1, #wafer.memory<spm, tensor>>
          wafer.tile.yield %done : i1
        }
        return
      } })mlir",
                                                          &context);
    ASSERT_TRUE(module);
    wafer::TileRegionOp region;
    module->walk([&](wafer::TileRegionOp op) { region = op; });
    EXPECT_TRUE(mlir::failed(wafer::convertTileRegionToInstr(region, session)));
    EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
    EXPECT_EQ(countOps<mlir::memref::CopyOp>(*module), 1u);
  }
  EXPECT_EQ(work->snapshot().relationDescriptorPlannings, 4u);
}

TEST(LowerInstrToTargetLLVMTest,
     ElementwiseKeepsCompatibleOperandLayoutWithoutMovement) {
  for (int64_t rows : {1024, 1025, 1031})
    for (llvm::StringRef layout : {"tensor", "ntensor", "cx", "ncx"}) {
      SCOPED_TRACE(llvm::formatv("rows={0} layout={1}", rows, layout).str());
      mlir::DialectRegistry registry;
      registerTargetConversionDialects(registry);
      mlir::MLIRContext context(registry);
      context.loadAllAvailableDialects();
      auto text = llvm::formatv(R"mlir(
module {{
  func.func @main() {{
    %token = arith.constant false
    %unused = wafer.tile.region(%token : i1) -> (i1) {{
    ^bb0(%done: i1):
      %lhs = memref.alloc() : memref<1x{0}x65xf16, #wafer.memory<spm, {1}>>
      %rhs = memref.alloc() : memref<1x{0}x65xf16, #wafer.memory<spm, {1}>>
      %result = wafer.tile.elementwise <add> %lhs, %rhs
          : (memref<1x{0}x65xf16, #wafer.memory<spm, {1}>>,
             memref<1x{0}x65xf16, #wafer.memory<spm, {1}>>)
         -> memref<1x{0}x65xf16, #wafer.memory<spm, {1}>>
      wafer.tile.yield %done : i1
    }
    return
  }
}
)mlir",
                                rows, layout)
                      .str();
      auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      ASSERT_TRUE(source) << text;
      wafer::TileRegionOp region;
      wafer::ComputeElementwiseOp original;
      source->walk([&](wafer::TileRegionOp op) { region = op; });
      source->walk([&](wafer::ComputeElementwiseOp op) { original = op; });
      llvm::SmallVector<mlir::Value> inputs(original.getInputs());
      mlir::Type outputType = original.getResult().getType();
      wafer::TileRegionToInstrLoweringSession session(context);
      ASSERT_TRUE(
          mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
      EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*source), 0u);
      EXPECT_EQ(countOps<wafer::LayoutMaterializeOp>(*source), 0u);
      ASSERT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 1u);
      source->walk([&](wafer::InstrElementwiseOp op) {
        EXPECT_EQ(op.getInputs(), mlir::ValueRange(inputs));
        EXPECT_EQ(op.getDest().getType(), outputType);
      });
    }
}

TEST(LowerInstrToTargetLLVMTest,
     StridedFillPreservesEveryHoleAndLowersToTarget) {
  for (int64_t length : {1024, 1025, 1031})
    for (bool rankFour : {false, true})
      for (int64_t step : {1, 2}) {
        SCOPED_TRACE(llvm::formatv("length={0} rank4={1} step={2}", length,
                                   rankFour, step)
                         .str());
        mlir::DialectRegistry registry;
        registerTargetConversionDialects(registry);
        mlir::MLIRContext context(registry);
        context.loadAllAvailableDialects();
        int64_t columns = 64 * step + 5;
        std::string prefix = rankFour ? "1x" : "";
        std::string outer = rankFour ? "0, " : "";
        std::string unit = rankFour ? "1, " : "";
        std::string strides =
            rankFour ? llvm::formatv("{0}, ", 2 * length * columns).str() : "";
        auto text = llvm::formatv(R"mlir(
module {{
  func.func @main(%output: memref<{0}2x{1}x{2}xf16, #wafer.memory<ddr, tensor>>) {{
    %result = wafer.tile.region(%output : memref<{0}2x{1}x{2}xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<{0}2x{1}x{2}xf16, #wafer.memory<ddr, tensor>>) {{
    ^bb0(%ddr: memref<{0}2x{1}x{2}xf16, #wafer.memory<ddr, tensor>>):
      %all = memref.alloc() {{wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<{0}2x{1}x{2}xf16, #wafer.memory<spm, tensor>>
      %guard = arith.constant -7.0 : f16
      %one = arith.constant 1.0 : f16
      wafer.tile.fill %all, %guard : memref<{0}2x{1}x{2}xf16, #wafer.memory<spm, tensor>>, f16
      %slice = memref.subview %all[{3}0, 0, 3] [{4}2, {1}, 64] [{4}1, 1, {5}]
          : memref<{0}2x{1}x{2}xf16, #wafer.memory<spm, tensor>>
         to memref<{0}2x{1}x64xf16, strided<[{6}{7}, {2}, {5}], offset: 3>, #wafer.memory<spm, tensor>>
      wafer.tile.fill %slice, %one
          : memref<{0}2x{1}x64xf16, strided<[{6}{7}, {2}, {5}], offset: 3>, #wafer.memory<spm, tensor>>, f16
      wafer.tile.store %all, %ddr : memref<{0}2x{1}x{2}xf16, #wafer.memory<spm, tensor>>
          -> memref<{0}2x{1}x{2}xf16, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %ddr : memref<{0}2x{1}x{2}xf16, #wafer.memory<ddr, tensor>>
    }
    wafer.instr.ncc_join [0]
    return
  }
}
)mlir",
                                  prefix, length, columns, outer, unit, step,
                                  strides, length * columns)
                        .str();
        auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(source) << text;
        wafer::TileRegionOp region;
        source->walk([&](wafer::TileRegionOp op) { region = op; });
        wafer::TileRegionToInstrLoweringSession session(context);
        ASSERT_TRUE(
            mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
        EXPECT_EQ(countOps<wafer::ComputeFillOp>(*source), 0u);
        EXPECT_EQ(countOps<wafer::InstrFillOp>(*source), 2u);
        EXPECT_EQ(countOps<mlir::memref::AllocOp>(*source), 1u);
        EXPECT_EQ(countOps<wafer::SyncNCCJoinOp>(*source), 1u);
        wafer::InstrFillOp fill;
        source->walk([&](wafer::InstrFillOp op) {
          if (op.getDest().getDefiningOp<mlir::memref::SubViewOp>())
            fill = op;
        });
        ASSERT_TRUE(fill);
        auto segment = fill.getDest().getDefiningOp<mlir::memref::SubViewOp>();
        auto view =
            segment.getSource().getDefiningOp<mlir::memref::SubViewOp>();
        ASSERT_TRUE(view);
        auto viewType = view.getType();
        auto [viewStrides, base] = mlir::getStridesAndOffset(viewType);
        auto segmentType =
            mlir::cast<mlir::MemRefType>(fill.getDest().getType());
        int64_t count = segmentType.getNumElements();
        EXPECT_EQ(count, step == 1 ? 64 : 1);
        std::vector<unsigned> writes(2 * length * columns, 0);
        llvm::SmallVector<mlir::scf::ForOp> loops;
        for (auto *parent = fill->getParentOp();
             parent != region.getOperation(); parent = parent->getParentOp())
          if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent))
            loops.push_back(loop);
        EXPECT_EQ(loops.size(), step == 1 ? 2u : 3u);
        llvm::DenseMap<mlir::Value, int64_t> indices;
        auto execute = [&](auto &&self, size_t level) -> void {
          if (level < loops.size()) {
            auto loop = loops[level];
            auto lowerOp = loop.getLowerBound()
                               .getDefiningOp<mlir::arith::ConstantIndexOp>();
            auto upperOp = loop.getUpperBound()
                               .getDefiningOp<mlir::arith::ConstantIndexOp>();
            auto stepOp =
                loop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
            ASSERT_TRUE(lowerOp && upperOp && stepOp);
            int64_t lower = lowerOp.value(), upper = upperOp.value(),
                    increment = stepOp.value();
            ASSERT_GT(increment, 0);
            for (int64_t i = lower; i < upper; i += increment) {
              indices[loop.getInductionVar()] = i;
              self(self, level + 1);
            }
            return;
          }
          int64_t address = base;
          for (auto [axis, offset] :
               llvm::enumerate(segment.getMixedOffsets())) {
            int64_t index = 0;
            if (auto value = mlir::dyn_cast<mlir::Value>(offset)) {
              ASSERT_TRUE(indices.count(value));
              index = indices.lookup(value);
            } else {
              index = mlir::cast<mlir::IntegerAttr>(
                          mlir::cast<mlir::Attribute>(offset))
                          .getInt();
            }
            address += index * viewStrides[axis];
          }
          ASSERT_GE(address, 0);
          ASSERT_LE(address + count, static_cast<int64_t>(writes.size()));
          for (int64_t i = 0; i < count; ++i)
            ++writes[address + i];
        };
        execute(execute, 0);
        for (size_t address = 0; address < writes.size(); ++address) {
          int64_t column = address % columns;
          bool selected =
              column >= 3 && column < 3 + 64 * step && (column - 3) % step == 0;
          ASSERT_EQ(writes[address], selected ? 1u : 0u) << address;
        }
        mlir::PassManager manager(&context);
        manager.addPass(wafer::createLowerInstrToTargetLLVMPass({}));
        ASSERT_TRUE(mlir::succeeded(manager.run(*source)));
        EXPECT_EQ(countOps<mlir::memref::SubViewOp>(*source), 0u);
        EXPECT_EQ(countOps<wafer::InstrFillOp>(*source), 0u);
        unsigned memsets = 0;
        source->walk([&](mlir::LLVM::CallOp call) {
          if (call.getCallee() != "wafer_tx81_memset")
            return;
          auto scalar =
              call.getOperand(1).getDefiningOp<mlir::LLVM::ConstantOp>();
          auto elements =
              call.getOperand(2).getDefiningOp<mlir::LLVM::ConstantOp>();
          ASSERT_TRUE(scalar && elements);
          if (mlir::cast<mlir::IntegerAttr>(scalar.getValue()).getInt() ==
              0x3c00)
            EXPECT_EQ(
                mlir::cast<mlir::IntegerAttr>(elements.getValue()).getInt(),
                count);
          else
            EXPECT_EQ(
                mlir::cast<mlir::IntegerAttr>(elements.getValue()).getInt(),
                static_cast<int64_t>(writes.size()));
          ++memsets;
        });
        EXPECT_EQ(memsets, 2u);
      }
}

TEST(LowerInstrToTargetLLVMTest,
     RejectsResidualElementwiseMapsWithoutMutatingSource) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
        %lhs, %rhs into %dst
        : memref<2x3xf16, #wafer.memory<spm, tensor>>,
          memref<2x3xf16, #wafer.memory<spm, tensor>>
      into memref<2x3xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  wafer::InstrElementwiseOp elementwise;
  source->walk([&](wafer::InstrElementwiseOp op) { elementwise = op; });
  ASSERT_TRUE(elementwise);
  mlir::AffineMapAttr identity = mlir::AffineMapAttr::get(
      mlir::AffineMap::getMultiDimIdentityMap(2, &context));
  elementwise->setAttr(
      "indexing_maps",
      mlir::ArrayAttr::get(&context, {identity, identity, identity}));

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        return mlir::success();
      });
  mlir::PassManager manager(&context);
  manager.enableVerifier(false);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  EXPECT_TRUE(mlir::failed(manager.run(*source)));
  EXPECT_NE(diagnostics.find("does not accept schema-free semantic attribute "
                             "'indexing_maps'"),
            std::string::npos)
      << diagnostics;
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 1u);
  EXPECT_TRUE(elementwise->hasAttr("indexing_maps"));
}

TEST(LowerInstrToTargetLLVMTest,
     LowersStaticallyBoundedDerivedSubviewToDynamicByteAddress) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(
      %input: memref<5x8xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %row = %c0 to %c4 step %c1 {
      %stage_shifted = arith.addi %row, %c1 : index
      %view = memref.subview %input[%stage_shifted, 2] [1, 3] [1, 1]
          : memref<5x8xf16, #wafer.memory<ddr, tensor>>
         to memref<1x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
      %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<1x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %view to %spm
          {byte_count = 6 : i64, inner_bytes = 6 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<1x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
         to memref<1x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
    }
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  EXPECT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<mlir::memref::SubViewOp>(*source), 0u);
  EXPECT_EQ(countOps<mlir::scf::ForOp>(*source), 0u);
  EXPECT_EQ(countOps<mlir::LLVM::MulOp>(*source), 1u);
  EXPECT_GE(countOps<mlir::LLVM::AddOp>(*source), 3u);
}

TEST(LowerInstrToTargetLLVMTest,
     LowersStaticallyBoundedClampedSubviewToDynamicByteAddress) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(
      %input: memref<5x8xf16, #wafer.memory<ddr, tensor>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    scf.for %row = %c0 to %c4 step %c1 {
      %lower_clamped = arith.maxsi %row, %c1 : index
      %clamped = arith.minsi %lower_clamped, %c3 : index
      %view = memref.subview %input[%clamped, 2] [1, 3] [1, 1]
          : memref<5x8xf16, #wafer.memory<ddr, tensor>>
         to memref<1x3xf16, strided<[8, 1], offset: ?>,
              #wafer.memory<ddr, tensor>>
      %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<1x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %view to %spm
          {byte_count = 6 : i64, inner_bytes = 6 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<1x3xf16, strided<[8, 1], offset: ?>,
              #wafer.memory<ddr, tensor>>
         to memref<1x3xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
    }
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  EXPECT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<mlir::memref::SubViewOp>(*source), 0u);
  EXPECT_EQ(countOps<mlir::scf::ForOp>(*source), 0u);
}

TEST(LowerInstrToTargetLLVMTest,
     RejectsUnsupportedDynamicSubviewOffsetWithoutMutatingSource) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>,
      %condition: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %selected = arith.select %condition, %i, %c0 : index
      %view = memref.subview %input[%selected] [1] [1]
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<1xf16, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        return mlir::success();
      });
  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  EXPECT_TRUE(mlir::failed(manager.run(*source)));
  EXPECT_NE(diagnostics.find("dynamic tensor subview offset #0 must be a "
                             "supported statically bounded index expression"),
            std::string::npos)
      << diagnostics;
  EXPECT_EQ(countOps<mlir::memref::SubViewOp>(*source), 1u);
  EXPECT_EQ(countOps<mlir::scf::ForOp>(*source), 1u);
  EXPECT_EQ(countOps<mlir::LLVM::LLVMFuncOp>(*source), 0u);
}

TEST(LowerInstrToTargetLLVMTest,
     InjectsPreparedDirectDTELifecycleWithoutLocalDTEOps) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card_partition"], shape = array<i64: 1>}
  func.func @main(%status: i64) {
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  request.cardId = 0;
  request.tileId = 15;
  request.transportStatusArgumentIndex = 0;
  request.transportPreparedBeforeEntry = true;
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));

  ASSERT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<wafer::InstrDTESendOp>(*source), 0u);
  EXPECT_EQ(countOps<wafer::InstrDTERecvOp>(*source), 0u);
  EXPECT_EQ(countOps<wafer::InstrDTEWaitOp>(*source), 0u);
  EXPECT_EQ(countOps<mlir::LLVM::CallOp>(*source), 2u);
  EXPECT_FALSE(source->lookupSymbol<mlir::LLVM::LLVMFuncOp>(
      "wafer_tx81_direct_dte_begin"));
  EXPECT_TRUE(source->lookupSymbol<mlir::LLVM::LLVMFuncOp>(
      "wafer_tx81_direct_dte_begin_after_prepare"));
  EXPECT_TRUE(source->lookupSymbol<mlir::LLVM::LLVMFuncOp>(
      "wafer_tx81_direct_dte_finish"));

  int64_t tileCount = -1;
  source->walk([&](mlir::LLVM::ConstantOp constant) {
    if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      tileCount = value.getInt();
  });
  EXPECT_EQ(tileCount, 16);
}

TEST(LowerInstrToTargetLLVMTest,
     LowersI8ArithmeticWithoutFormalModelQualification) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<64xi8, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<64xi8, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<64xi8, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
        %lhs, %rhs into %dst
        : memref<64xi8, #wafer.memory<spm, tensor>>,
          memref<64xi8, #wafer.memory<spm, tensor>>
      into memref<64xi8, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));
  EXPECT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 0u);
  EXPECT_GT(countOps<mlir::LLVM::LLVMFuncOp>(*source), 0u);
}

TEST(LowerInstrToTargetLLVMTest,
     LowersF32ElementwiseWithoutFormalModelQualification) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<64xf32, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<64xf32, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<neg> %src into %dst
        : memref<64xf32, #wafer.memory<spm, tensor>>
      into memref<64xf32, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));
  EXPECT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 0u);
  EXPECT_GT(countOps<mlir::LLVM::LLVMFuncOp>(*source), 0u);
}

TEST(LowerInstrToTargetLLVMTest,
     NumericVerificationKeepsQualifiedComputeAndI8MovementIndependent) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(
      %input: memref<64xi8, #wafer.memory<ddr, tensor>>,
      %output: memref<64xi8, #wafer.memory<ddr, tensor>>) {
    %i8 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<64xi8, #wafer.memory<spm, tensor>>
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<64xf16, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<64xf16, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<64xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %input to %i8
        {byte_count = 64 : i64, inner_bytes = 64 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<64xi8, #wafer.memory<ddr, tensor>>
       to memref<64xi8, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add>
        %lhs, %rhs into %dst
        : memref<64xf16, #wafer.memory<spm, tensor>>,
          memref<64xf16, #wafer.memory<spm, tensor>>
      into memref<64xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %i8 to %output
        {byte_count = 64 : i64, inner_bytes = 64 : i64,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<64xi8, #wafer.memory<spm, tensor>>
       to memref<64xi8, #wafer.memory<ddr, tensor>>
    return
  }
}
  )mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  mlir::PassManager manager(&context);
  wafer::TargetConversionRequest request{};
  manager.addPass(wafer::createLowerInstrToTargetLLVMPass(request));
  EXPECT_TRUE(mlir::succeeded(manager.run(*source)));
  EXPECT_EQ(countOps<wafer::InstrRDMAOp>(*source), 0u);
  EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 0u);
  EXPECT_EQ(countOps<wafer::InstrWDMAOp>(*source), 0u);
}

TEST(LowerInstrToTargetLLVMTest,
     CountsBlockedCTOverTheCompletePhysicalTraversal) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<3x65xf16, #wafer.memory<spm, cx>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<3x65xf16, #wafer.memory<spm, cx>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<abs> %src into %dst
        : memref<3x65xf16, #wafer.memory<spm, cx>>
      into memref<3x65xf16, #wafer.memory<spm, cx>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  wafer::InstrElementwiseOp elementwise;
  source->walk([&](wafer::InstrElementwiseOp op) { elementwise = op; });
  ASSERT_TRUE(elementwise);
  auto destType = mlir::cast<mlir::MemRefType>(elementwise.getDest().getType());
  mlir::FailureOr<int64_t> elements =
      wafer::target_llvm_detail::getPhysicalTraversalElementCount(
          elementwise, destType, "elementwise dest");
  ASSERT_TRUE(mlir::succeeded(elements));
  EXPECT_EQ(*elements, 256);
  EXPECT_NE(*elements, destType.getNumElements());
}

TEST(LowerInstrToTargetLLVMTest,
     PhysicalVerificationAcceptsCompatibleBlockedConvertTraversal) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x2x197xf16, #wafer.memory<spm, ncx>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67584>}
        : memref<2x2x197xbf16, #wafer.memory<spm, ncx>>
    wafer.instr.convert #wafer.instr_convert_kind<fp16_bf16> %src into %dst
        {rounding_mode = 0 : i64}
        : memref<2x2x197xf16, #wafer.memory<spm, ncx>>
       to memref<2x2x197xbf16, #wafer.memory<spm, ncx>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);
  EXPECT_TRUE(mlir::succeeded(
      wafer::target_llvm_detail::verifyTargetInstructionFormats(*source)));
}

TEST(LowerInstrToTargetLLVMTest,
     PhysicalVerificationRejectsDtypeSpecificBlockedTraversalMismatch) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main() {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x2x197xf16, #wafer.memory<spm, ncx>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67584>}
        : memref<2x2x197xf32, #wafer.memory<spm, ncx>>
    wafer.instr.convert #wafer.instr_convert_kind<fp16_fp32> %src into %dst
        : memref<2x2x197xf16, #wafer.memory<spm, ncx>>
       to memref<2x2x197xf32, #wafer.memory<spm, ncx>>
    return
  }
}
)mlir",
      mlir::ParserConfig(&context));
  ASSERT_TRUE(source);

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        return mlir::success();
      });
  EXPECT_TRUE(mlir::failed(
      wafer::target_llvm_detail::verifyTargetInstructionFormats(*source)));
  EXPECT_NE(diagnostics.find("unsupported_target_physical_traversal: convert"),
            std::string::npos)
      << diagnostics;
}

} // namespace
