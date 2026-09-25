#include "Wafer/Conversion/InstrToLLVM/InstrToLLVM.h"
#include "Wafer/Conversion/InstrToLLVM/LowerInstrToTargetLLVMInternal.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Support/CompileWorkStatistics.h"
#include "Wafer/Transforms/Tile/MovementFusion.h"

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

#include <functional>
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

TEST(LowerInstrToTargetLLVMTest, RelationsLowerWithExplicitResultFormat) {
  for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (llvm::StringRef kind : {"eq", "ne", "ge", "gt", "le", "lt"})
        for (unsigned form = 0; form < 3; ++form)
          for (bool numeric : {false, true}) {
            SCOPED_TRACE(llvm::formatv("{0}/{1}/{2}/{3}/{4}", dtype, extent,
                                       kind, form, numeric)
                             .str());
            mlir::DialectRegistry registry;
            registerTargetConversionDialects(registry);
            mlir::MLIRContext context(registry);
            context.loadAllAvailableDialects();
            auto type = [&](llvm::StringRef element) {
              return llvm::formatv(
                         "memref<2x32x{0}x{1}, #wafer.memory<spm, tensor>>",
                         extent, element)
                  .str();
            };
            auto input = type(dtype), output = type(numeric ? dtype : "i1");
            auto rhs =
                form == 1 ? dtype.str()
                : form == 2
                    ? llvm::formatv(
                          "memref<32x{0}, #wafer.memory<spm, tensor>>", dtype)
                          .str()
                    : input;
            std::string text;
            llvm::raw_string_ostream out(text);
            out << "module { func.func @entry() {\n"
                << "%lhs = memref.alloc() {wafer.spm.offset = "
                   "#wafer.spm_offset<65536>} : "
                << input << "\n";
            if (form == 1)
              out << "%rhs = arith.constant 1.0 : " << dtype << "\n";
            else
              out << "%rhs = memref.alloc() {wafer.spm.offset = "
                     "#wafer.spm_offset<589824>} : "
                  << rhs << "\n";
            out << "%dst = memref.alloc() {wafer.spm.offset = "
                   "#wafer.spm_offset<1114112>} : "
                << output << "\nwafer.instr.elementwise <" << kind
                << "> %lhs, %rhs into %dst "
                << (form == 2 ? "{rhs_unit_elements = 32 : i64}" : "") << " : "
                << input << ", " << rhs << " into " << output
                << "\nreturn\n}}\n";
            auto module =
                mlir::parseSourceString<mlir::ModuleOp>(text, &context);
            ASSERT_TRUE(module) << text;
            mlir::PassManager manager(&context);
            manager.addPass(wafer::createLowerInstrToTargetLLVMPass({}));
            ASSERT_TRUE(mlir::succeeded(manager.run(*module)));
            ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
            unsigned calls = 0;
            module->walk([&](mlir::LLVM::CallOp call) {
              if (call.getCallee() != ("wafer_tx81_elementwise_" + kind).str())
                return;
              ++calls;
              ASSERT_EQ(call.getNumOperands(), 9u);
              auto integer = [&](unsigned index) -> int64_t {
                auto value = call.getOperand(index)
                                 .getDefiningOp<mlir::LLVM::ConstantOp>();
                EXPECT_TRUE(value);
                return value ? mlir::cast<mlir::IntegerAttr>(value.getValue())
                                   .getInt()
                             : -1;
              };
              EXPECT_EQ(integer(0), 65536);
              EXPECT_EQ(integer(1), form == 1 ? (dtype == "f16"    ? 0x3c00
                                                 : dtype == "bf16" ? 0x3f80
                                                                   : 0x3f800000)
                                              : 589824);
              EXPECT_EQ(integer(2), 1114112);
              EXPECT_EQ(integer(3), 64 * extent);
              EXPECT_EQ(integer(5), form == 2 ? 32 : 0);
              EXPECT_EQ(integer(6), form == 1);
              EXPECT_EQ(integer(7), numeric);
              EXPECT_EQ(integer(8), 0);
            });
            EXPECT_EQ(calls, 1u);
          }
}

TEST(LowerInstrToTargetLLVMTest,
     NumericPredicatesKeepTheOriginalInputSnapshot) {
  for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (unsigned variant = 0; variant < 9; ++variant) {
        SCOPED_TRACE(
            llvm::formatv("{0}/{1}/{2}", dtype, extent, variant).str());
        mlir::DialectRegistry registry;
        registerTargetConversionDialects(registry);
        mlir::MLIRContext context(registry);
        context.loadAllAvailableDialects();
        auto type = [&](llvm::StringRef element, int64_t unit = 0) {
          return llvm::formatv("memref<{0}{1}x{2}, #wafer.memory<spm, tensor>>",
                               unit ? "" : "2x32x", unit ? unit : extent,
                               element)
              .str();
        };
        std::string input = type(dtype), boolean = type("i1");
        std::string numeric =
            type(variant == 6 ? (dtype == "f32" ? "f16" : "f32") : dtype);
        std::string rhs =
            variant == 1 ? dtype.str() : type(dtype, variant == 2 ? 1 : 0);
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { func.func @entry() {\n%token = arith.constant false\n"
            << "%r = wafer.tile.region(%token : i1) -> (i1) { ^bb0(%done: "
               "i1):\n"
            << "%lhs = memref.alloc() : " << input << "\n"
            << "%zero = arith.constant 0.0 : " << dtype << "\n";
        if (variant == 1)
          out << "%rhs = arith.constant 1.0 : " << dtype << "\n";
        else
          out << "%rhs = memref.alloc() : " << rhs << "\n";
        out << "%predicate = memref.alloc() : " << boolean << "\n";
        bool fill = variant == 3 || variant == 8;
        if (fill)
          out << "%bit = arith.constant " << (variant == 3 ? "true" : "false")
              << "\nwafer.instr.fill %predicate, %bit : " << boolean
              << ", i1\n";
        else
          out << "wafer.instr.elementwise <eq> %lhs, %rhs into %predicate "
              << (variant == 2 ? "{rhs_unit_elements = 1 : i64}" : "") << " : "
              << input << ", " << rhs << " into " << boolean << "\n";
        if (variant == 7)
          out << "wafer.instr.fill %lhs, %zero : " << input << ", " << dtype
              << "\n";
        out << "%numeric = memref.alloc() : " << numeric << "\n"
            << "%sink = memref.alloc() : " << numeric << "\n";
        if (variant == 5)
          out << "wafer.instr.elementwise <neg> %numeric into %sink : "
              << numeric << " into " << numeric << "\n";
        out << "wafer.instr.bit2fp %predicate into %numeric : " << boolean
            << " to " << numeric << "\n";
        if (variant == 4)
          out << "%other = memref.alloc() : " << boolean
              << "\nwafer.instr.elementwise <logic_not> %predicate into %other "
                 ": "
              << boolean << " into " << boolean << "\n";
        out << "wafer.instr.elementwise <neg> %numeric into %sink : " << numeric
            << " into " << numeric
            << "\nwafer.tile.yield %done : i1\n}\nreturn\n}}\n";
        auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(module) << text;
        wafer::TileRegionOp region;
        wafer::InstrBit2FpOp conversion;
        wafer::InstrElementwiseOp comparison;
        wafer::InstrFillOp predicateFill, mutation;
        module->walk([&](wafer::TileRegionOp op) { region = op; });
        module->walk([&](wafer::InstrBit2FpOp op) { conversion = op; });
        module->walk([&](wafer::InstrElementwiseOp op) {
          if (op.getKind() == wafer::InstrElementwiseKind::Eq)
            comparison = op;
        });
        module->walk([&](wafer::InstrFillOp op) {
          if (op.getDest().getType().getElementType().isInteger(1))
            predicateFill = op;
          else
            mutation = op;
        });
        mlir::Value destination = conversion.getDest();
        wafer::TileRegionToInstrLoweringSession session(context);
        ASSERT_TRUE(
            mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        bool folded = variant != 4 && variant != 5 && variant != 6;
        EXPECT_EQ(countOps<wafer::InstrBit2FpOp>(*module), folded ? 0u : 1u);
        if (!folded)
          continue;
        if (fill) {
          EXPECT_EQ(predicateFill.getDest(), destination);
          auto constant =
              predicateFill.getValue().getDefiningOp<mlir::arith::ConstantOp>();
          ASSERT_TRUE(constant);
          auto value =
              mlir::cast<mlir::FloatAttr>(constant.getValue()).getValue();
          EXPECT_EQ(value.convertToDouble(), variant == 3 ? 1.0 : 0.0);
        } else {
          EXPECT_EQ(comparison.getDest(), destination);
          EXPECT_EQ(comparison.getRhsUnitElements(), variant == 2 ? 1 : 0);
          EXPECT_TRUE(destination.getDefiningOp()->isBeforeInBlock(comparison));
          if (mutation) {
            EXPECT_TRUE(comparison->isBeforeInBlock(mutation));
          }
        }
        unsigned booleanAllocations = 0;
        module->walk([&](mlir::memref::AllocOp allocation) {
          booleanAllocations +=
              allocation.getType().getElementType().isInteger(1);
        });
        EXPECT_EQ(booleanAllocations, 0u);
      }
}

TEST(LowerInstrToTargetLLVMTest,
     DivisionUsesReciprocalProductWithSafeAliasing) {
  for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
    for (llvm::StringRef layout : {"tensor", "cx", "ncx"})
      for (int64_t extent : {1024, 1025, 1031})
        for (unsigned form = 0; form < 4; ++form) {
          SCOPED_TRACE(
              llvm::formatv("{0}:{1}:{2}:{3}", dtype, layout, extent, form)
                  .str());
          mlir::DialectRegistry registry;
          registerTargetConversionDialects(registry);
          mlir::MLIRContext context(registry);
          context.loadAllAvailableDialects();
          auto type =
              llvm::formatv("memref<1x2x{0}x{1}, #wafer.memory<spm, {2}>>",
                            extent, dtype, layout)
                  .str();
          std::string compute;
          if (form == 0)
            compute = llvm::formatv("%result = wafer.tile.elementwise <div> "
                                    "%lhs, %rhs : ({0}, {0}) -> {0}",
                                    type)
                          .str();
          else {
            if (form == 1)
              compute = "%dest = memref.alloc() : " + type + "\n";
            compute += llvm::formatv("wafer.tile.elementwise_into <div> %lhs, "
                                     "%rhs into %{1} : {0}, {0} into {0}",
                                     type,
                                     form == 1   ? "dest"
                                     : form == 2 ? "lhs"
                                                 : "rhs")
                           .str();
          }
          auto text = llvm::formatv(R"mlir(module {{
            func.func @main() {{
              %token = arith.constant false
              %unused = wafer.tile.region(%token : i1) -> (i1) {{
              ^bb0(%done: i1):
                %lhs = memref.alloc() : {0}
                %rhs = memref.alloc() : {0}
                {1}
                wafer.tile.yield %done : i1
              }
              return
            }
          })mlir",
                                    type, compute)
                          .str();
          auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
          ASSERT_TRUE(source) << text;
          wafer::TileRegionOp region;
          mlir::Value lhs, rhs, destination;
          source->walk([&](wafer::TileRegionOp op) { region = op; });
          source->walk([&](wafer::ComputeElementwiseOp op) {
            lhs = op.getInputs()[0];
            rhs = op.getInputs()[1];
          });
          source->walk([&](wafer::ComputeElementwiseIntoOp op) {
            lhs = op.getInputs()[0];
            rhs = op.getInputs()[1];
            destination = op.getDest();
          });
          wafer::TileRegionToInstrLoweringSession session(context);
          ASSERT_TRUE(mlir::succeeded(
              wafer::convertTileRegionToInstr(region, session)));
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
          EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(*source), 0u);
          EXPECT_EQ(countOps<wafer::ComputeElementwiseIntoOp>(*source), 0u);
          llvm::SmallVector<wafer::InstrElementwiseOp> operations;
          source->walk(
              [&](wafer::InstrElementwiseOp op) { operations.push_back(op); });
          ASSERT_EQ(operations.size(), 2u);
          auto reciprocal = operations[0], multiply = operations[1];
          EXPECT_EQ(reciprocal.getKind(), wafer::InstrElementwiseKind::Recip);
          EXPECT_EQ(reciprocal.getInputs().front(), rhs);
          EXPECT_EQ(multiply.getKind(), wafer::InstrElementwiseKind::Mul);
          EXPECT_EQ(multiply.getInputs()[0], lhs);
          EXPECT_EQ(multiply.getInputs()[1], reciprocal.getDest());
          EXPECT_TRUE(reciprocal->isBeforeInBlock(multiply));
          EXPECT_NE(reciprocal.getDest(), lhs);
          EXPECT_NE(reciprocal.getDest(), rhs);
          EXPECT_NE(reciprocal.getDest(), multiply.getDest());
          if (destination) {
            EXPECT_EQ(multiply.getDest(), destination);
          }
          EXPECT_EQ(reciprocal.getDest().getType(),
                    multiply.getDest().getType());
          EXPECT_EQ(countOps<mlir::memref::AllocOp>(*source),
                    form < 2 ? 4u : 3u);
          EXPECT_EQ(countOps<wafer::InstrFillOp>(*source), 0u);
          EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*source), 0u);
          EXPECT_EQ(countOps<wafer::InstrBit2FpOp>(*source), 0u);
          EXPECT_EQ(countOps<wafer::InstrMaskMoveOp>(*source), 0u);
        }
}

TEST(LowerInstrToTargetLLVMTest, DivisionSupportsMappedInputAndAliasedSubview) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool mapped : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(mapped);
      mlir::DialectRegistry registry;
      registerTargetConversionDialects(registry);
      mlir::MLIRContext context(registry);
      context.loadAllAvailableDialects();
      auto type = llvm::formatv(
                      "memref<1x2x{0}xf32, #wafer.memory<spm, tensor>>", extent)
                      .str();
      auto viewType =
          llvm::formatv("memref<1x2x{0}xf32, strided<[{1}, {0}, 1], offset: "
                        "{1}>, #wafer.memory<spm, tensor>>",
                        extent, 2 * extent)
              .str();
      auto computation = mapped ? llvm::formatv(R"mlir(
        %rhs = memref.alloc() : memref<1x2xf32, #wafer.memory<spm, tensor>>
        %result = wafer.tile.elementwise <div> %lhs, %rhs {{indexing_maps = [
          affine_map<(b,m,n)->(b,m,n)>, affine_map<(b,m,n)->(b,m)>, affine_map<(b,m,n)->(b,m,n)>]}
          : ({0}, memref<1x2xf32, #wafer.memory<spm, tensor>>) -> {0}
      )mlir",
                                                type)
                                      .str()
                                : llvm::formatv(R"mlir(
        %parent = memref.alloc() : memref<2x2x{0}xf32, #wafer.memory<spm, tensor>>
        %rhs = memref.subview %parent[1, 0, 0] [1, 2, {0}] [1, 1, 1]
          : memref<2x2x{0}xf32, #wafer.memory<spm, tensor>> to {2}
        wafer.tile.elementwise_into <div> %lhs, %rhs into %rhs : {1}, {2} into {2}
      )mlir",
                                                extent, type, viewType)
                                      .str();
      auto text = llvm::formatv(R"mlir(module {{ func.func @main() {{
        %token = arith.constant false
        %unused = wafer.tile.region(%token : i1) -> (i1) {{
        ^bb0(%done: i1):
          %lhs = memref.alloc() : {0}
          {1}
          wafer.tile.yield %done : i1
        }
        return
      } })mlir",
                                type, computation)
                      .str();
      auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      ASSERT_TRUE(source) << text;
      wafer::TileRegionOp region;
      mlir::Value rhs;
      source->walk([&](wafer::TileRegionOp op) { region = op; });
      source->walk(
          [&](wafer::ComputeElementwiseOp op) { rhs = op.getInputs()[1]; });
      source->walk(
          [&](wafer::ComputeElementwiseIntoOp op) { rhs = op.getInputs()[1]; });
      wafer::TileRegionToInstrLoweringSession session(context);
      ASSERT_TRUE(
          mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
      llvm::SmallVector<wafer::InstrElementwiseOp> operations;
      source->walk(
          [&](wafer::InstrElementwiseOp op) { operations.push_back(op); });
      ASSERT_EQ(operations.size(), 2u);
      auto reciprocal = operations[0], multiply = operations[1];
      EXPECT_EQ(reciprocal.getKind(), wafer::InstrElementwiseKind::Recip);
      EXPECT_EQ(multiply.getKind(), wafer::InstrElementwiseKind::Mul);
      EXPECT_EQ(multiply.getInputs()[1], reciprocal.getDest());
      if (mapped) {
        EXPECT_GT(countOps<wafer::InstrGatherScatterOp>(*source), 0u);
        source->walk([&](wafer::InstrGatherScatterOp move) {
          EXPECT_EQ(move.getSource(), rhs);
          EXPECT_EQ(move.getDest(), reciprocal.getInputs().front());
        });
      } else {
        EXPECT_EQ(reciprocal.getInputs().front(), rhs);
        EXPECT_EQ(multiply.getDest(), rhs);
        EXPECT_NE(reciprocal.getDest(), rhs);
        EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*source), 0u);
      }
    }
}

TEST(LowerInstrToTargetLLVMTest,
     NativeReductionDoesNotDependOnOrderedExpansionBudget) {
  auto check = [](int64_t rows, int64_t width, llvm::StringRef type,
                  llvm::StringRef kind, llvm::StringRef identity,
                  unsigned extraUnits = 0) {
    SCOPED_TRACE(llvm::formatv("rows={0} width={1} type={2} kind={3}", rows,
                               width, type, kind)
                     .str());
    mlir::DialectRegistry registry;
    registerTargetConversionDialects(registry);
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    std::string prefix;
    for (unsigned i = 0; i < extraUnits; ++i)
      prefix += "1x";
    auto resultLayout =
        extraUnits ? wafer::MemLayout::NCx : wafer::MemLayout::Cx;
    auto text = llvm::formatv(R"mlir(
module {{
  func.func @main() {{
    %token = arith.constant false
    %unused = wafer.tile.region(%token : i1) -> (i1) {{
    ^bb0(%done: i1):
      %input = memref.alloc() : memref<{5}1x{0}x{1}x{2}, #wafer.memory<spm, ncx>>
      %result = wafer.tile.reduce <{3}> %input
          {{dimensions = array<i64: {6}>, init_value = {4} : {2}}
          : (memref<{5}1x{0}x{1}x{2}, #wafer.memory<spm, ncx>>)
         -> memref<{5}1x{0}x{2}, #wafer.memory<spm, {7}>>
      wafer.tile.yield %done : i1
    }
    return
  }
}
)mlir",
                              rows, width, type, kind, identity, prefix,
                              2 + extraUnits, extraUnits ? "ncx" : "cx")
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
    EXPECT_EQ(nativeType.getShape(), (llvm::ArrayRef<int64_t>{1, 1, rows, 1}));
    EXPECT_EQ(wafer::getWaferMemoryAttr(nativeType).getLayout(),
              wafer::MemLayout::NCx);
    if (extraUnits) {
      auto view =
          reduce.getInput().getDefiningOp<mlir::memref::ReinterpretCastOp>();
      ASSERT_TRUE(view);
      auto originalType =
          mlir::cast<mlir::MemRefType>(view.getSource().getType());
      auto inputType = mlir::cast<mlir::MemRefType>(view.getType());
      for (int64_t row = 0; row < rows; ++row)
        for (int64_t column = 0; column < width; ++column) {
          llvm::SmallVector<int64_t> original(extraUnits + 1, 0);
          original.append({row, column});
          auto before = wafer::computeWaferPhysicalElementByteOffset(
              originalType, original);
          auto after = wafer::computeWaferPhysicalElementByteOffset(
              inputType, {0, 0, row, column});
          ASSERT_TRUE(before && after);
          ASSERT_EQ(*before, *after);
        }
    }
    if (!extraUnits) {
      EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*source),
                rows == 1024 ? 1u : 2u);
    }
    std::map<int64_t, int64_t> actualBytes;
    std::map<int64_t, int64_t> expectedBytes;
    source->walk([&](wafer::InstrGatherScatterOp move) {
      EXPECT_EQ(move.getSource(), reduce.getDest());
      auto resultType = mlir::cast<mlir::MemRefType>(move.getDest().getType());
      llvm::SmallVector<int64_t> resultShape(extraUnits + 1, 1);
      resultShape.push_back(rows);
      EXPECT_EQ(resultType.getShape(), llvm::ArrayRef<int64_t>(resultShape));
      EXPECT_EQ(wafer::getWaferMemoryAttr(resultType).getLayout(),
                resultLayout);
      for (int64_t row = 0; row < rows; ++row) {
        auto src = wafer::computeWaferPhysicalElementByteOffset(nativeType,
                                                                {0, 0, row, 0});
        llvm::SmallVector<int64_t> coordinates(extraUnits + 1, 0);
        coordinates.push_back(row);
        auto dst = wafer::computeWaferPhysicalElementByteOffset(resultType,
                                                                coordinates);
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
  for (unsigned extraUnits : {2u, 3u})
    for (int64_t rows : {1024, 1025, 1031}) {
      check(rows, 32, "f32", "sum", "0.0", extraUnits);
      check(rows, 32, "f32", "max", "0xFF800000", extraUnits);
    }
}

TEST(LowerInstrToTargetLLVMTest,
     NativeReductionDoesNotDropNonUnitOrReducedPrefix) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned variant = 0; variant < 3; ++variant) {
      mlir::DialectRegistry registry;
      registerTargetConversionDialects(registry);
      mlir::MLIRContext context(registry);
      context.loadAllAvailableDialects();
      std::string input =
          llvm::formatv("memref<{0}x1x1x{1}x32xf32, #wafer.memory<spm, ncx>>",
                        variant == 0 ? 2 : 1, extent)
              .str();
      std::string output =
          variant == 1
              ? llvm::formatv("memref<1x1x{0}x32xf32, #wafer.memory<spm, ncx>>",
                              extent)
                    .str()
              : llvm::formatv(
                    "memref<{0}x1x1x{1}xf32, #wafer.memory<spm, ncx>>",
                    variant == 0 ? 2 : 1, extent)
                    .str();
      auto text = llvm::formatv(R"mlir(module {{ func.func @main() {{
        wafer.tile.region() -> () {{
          %input = memref.alloc() : {0}
          %result = wafer.tile.reduce <sum> %input
            {{dimensions = array<i64: {2}>, init_value = {3} : f32}
            : ({0}) -> {1}
          wafer.tile.yield
        }
        return
      } })mlir",
                                input, output, variant == 1 ? 0 : 4,
                                variant == 2 ? "1.0" : "0.0")
                      .str();
      auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      ASSERT_TRUE(source) << text;
      wafer::TileRegionOp region;
      source->walk([&](wafer::TileRegionOp op) { region = op; });
      wafer::TileRegionToInstrLoweringSession session(context);
      ASSERT_TRUE(
          mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
      EXPECT_EQ(countOps<wafer::InstrReduceOp>(*source), 0u);
    }
}

TEST(LowerInstrToTargetLLVMTest,
     NativeReductionChecksTheActualNHWCRegisterLimits) {
  for (int64_t rows : {4096, 4097}) {
    SCOPED_TRACE(rows);
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
      %input = memref.alloc() : memref<1x1x{0}x2xf16, #wafer.memory<spm, ncx>>
      %result = wafer.tile.reduce <sum> %input
          {{dimensions = array<i64: 3>, init_value = 0.0 : f16}
          : (memref<1x1x{0}x2xf16, #wafer.memory<spm, ncx>>)
         -> memref<1x1x{0}xf16, #wafer.memory<spm, ncx>>
      wafer.tile.yield %done : i1
    }
    return
  }
}
)mlir",
                              rows)
                    .str();
    auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(source);
    wafer::TileRegionOp region;
    source->walk([&](wafer::TileRegionOp op) { region = op; });
    wafer::TileRegionToInstrLoweringSession session(context);
    ASSERT_TRUE(
        mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*source)));
    EXPECT_EQ(countOps<wafer::ComputeReduceOp>(*source), 0u);
    EXPECT_EQ(countOps<wafer::InstrReduceOp>(*source), rows == 4096 ? 1u : 0u);
    EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source),
              rows == 4096 ? 0u : 1u);
  }
}

TEST(LowerInstrToTargetLLVMTest,
     NativeReductionMaterializesExactNHWCInputGeometry) {
  auto check = [](int64_t rows, int64_t channels, llvm::StringRef dtype) {
    SCOPED_TRACE(
        llvm::formatv("rows={0}, C={1}, dtype={2}", rows, channels, dtype)
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
      %input = memref.alloc() : memref<16x{0}x{1}x{2}, #wafer.memory<spm, ncx>>
      %result = wafer.tile.reduce <sum> %input
          {{dimensions = array<i64: 0>, init_value = 0.0 : {2}}
          : (memref<16x{0}x{1}x{2}, #wafer.memory<spm, ncx>>)
         -> memref<{0}x{1}x{2}, #wafer.memory<spm, cx>>
      wafer.tile.yield %done : i1
    }
    return
  }
}
)mlir",
                              rows, channels, dtype)
                    .str();
    auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(source);
    mlir::memref::AllocOp original;
    source->walk([&](mlir::memref::AllocOp op) { original = op; });
    wafer::TileRegionOp region;
    source->walk([&](wafer::TileRegionOp op) { region = op; });
    wafer::TileRegionToInstrLoweringSession session(context);
    ASSERT_TRUE(
        mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
    ASSERT_EQ(countOps<wafer::InstrReduceOp>(*source), 1u);
    wafer::InstrReduceOp reduce;
    source->walk([&](wafer::InstrReduceOp op) { reduce = op; });
    EXPECT_EQ(reduce.getDimAttr().getInt(), 2);
    auto packed = mlir::cast<mlir::MemRefType>(reduce.getInput().getType());
    EXPECT_EQ(packed.getShape(),
              (llvm::ArrayRef<int64_t>{1, 16, rows, channels}));
    EXPECT_EQ(
        mlir::cast<mlir::MemRefType>(reduce.getDest().getType()).getShape(),
        (llvm::ArrayRef<int64_t>{1, 1, rows, channels}));
    // Independently enumerate logical elements under the two physical types.
    // The descriptor must cover exactly these bytes, without touching padding.
    auto packedInfo = wafer::computeWaferPhysicalTensorInfo(packed);
    auto originalInfo =
        wafer::computeWaferPhysicalTensorInfo(original.getType());
    ASSERT_TRUE(packedInfo && originalInfo);
    std::vector<int64_t> expected(packedInfo->physicalBytes, -1);
    const int64_t elementBytes = dtype == "f32" ? 4 : 2;
    bool identical = packedInfo->physicalBytes == originalInfo->physicalBytes;
    for (int64_t h = 0; h < 16; ++h)
      for (int64_t w = 0; w < rows; ++w)
        for (int64_t c = 0; c < channels; ++c) {
          auto src = wafer::computeWaferPhysicalElementByteOffset(
              original.getType(), *originalInfo, {h, w, c});
          auto dst = wafer::computeWaferPhysicalElementByteOffset(
              packed, *packedInfo, {0, h, w, c});
          ASSERT_TRUE(src && dst);
          identical &= *src == *dst;
          for (int64_t byte = 0; byte < elementBytes; ++byte)
            expected[*dst + byte] = *src + byte;
        }
    if (identical) {
      auto view =
          reduce.getInput().getDefiningOp<mlir::memref::ReinterpretCastOp>();
      ASSERT_TRUE(view);
      EXPECT_EQ(view.getSource(), original.getResult());
    } else {
      ASSERT_TRUE(reduce.getInput().getDefiningOp<mlir::memref::AllocOp>());
      std::vector<bool> seen(expected.size(), false);
      uint64_t transferred = 0;
      unsigned movements = 0;
      source->walk([&](wafer::InstrGatherScatterOp move) {
        if (move.getDest() != reduce.getInput())
          return;
        ++movements;
        ASSERT_EQ(move.getSource(), original.getResult());
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
        ASSERT_EQ(src.size(),
                  static_cast<size_t>(move.getByteCountAttr().getInt()));
        bool exact = true;
        for (size_t i = 0; i < src.size(); ++i) {
          if (dst[i] < 0 || static_cast<size_t>(dst[i]) >= expected.size() ||
              seen[dst[i]] || expected[dst[i]] != src[i]) {
            exact = false;
            break;
          }
          seen[dst[i]] = true;
        }
        EXPECT_TRUE(exact);
        transferred += src.size();
      });
      EXPECT_GT(movements, 0u);
      EXPECT_EQ(transferred,
                static_cast<uint64_t>(16 * rows * channels * elementBytes));
      for (size_t i = 0; i < expected.size(); ++i)
        if (seen[i] != (expected[i] >= 0)) {
          ADD_FAILURE() << "missing or padded destination byte " << i;
          break;
        }
    }
    EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*source), 0u);
    EXPECT_EQ(countOps<mlir::scf::ForOp>(*source), 0u);
  };
  for (int64_t rows : {1024, 1025, 1031})
    for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
      check(rows, 1, dtype);
  check(1031, 65, "f16");
}

TEST(LowerInstrToTargetLLVMTest,
     DynamicGatherScatterOffsetsAreRelativeToNestedViews) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    mlir::DialectRegistry registry;
    registerTargetConversionDialects(registry);
    mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(module {
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
    auto tensor = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                         wafer::MemLayout::Tensor);
    auto ncx = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                      wafer::MemLayout::NCx);
    auto type =
        mlir::MemRefType::get({1, extent, 128}, builder.getF16Type(),
                              mlir::MemRefLayoutAttrInterface{}, tensor);
    auto input = builder.create<mlir::memref::AllocOp>(loc, type);
    auto output = builder.create<mlir::memref::AllocOp>(loc, type);
    auto view = [&](mlir::Value base, int64_t channel) {
      return builder.create<mlir::memref::SubViewOp>(
          loc, base, llvm::ArrayRef<int64_t>{0, 0, channel},
          llvm::ArrayRef<int64_t>{1, extent, 64},
          llvm::ArrayRef<int64_t>{1, 1, 1});
    };
    auto source = view(input, 64);
    auto destination = view(output, 32);
    auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto end = builder.create<mlir::arith::ConstantIndexOp>(loc, extent);
    auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto loop = builder.create<mlir::scf::ForOp>(loc, zero, end, one);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    llvm::SmallVector<mlir::OpFoldResult> offsets{builder.getIndexAttr(0),
                                                  loop.getInductionVar(),
                                                  builder.getIndexAttr(0)};
    auto sizes = mlir::getAsIndexOpFoldResult(&context, {1, 1, 64});
    auto strides = mlir::getAsIndexOpFoldResult(&context, {1, 1, 1});
    auto from = builder.create<mlir::memref::SubViewOp>(loc, source, offsets,
                                                        sizes, strides);
    auto to = builder.create<mlir::memref::SubViewOp>(loc, destination, offsets,
                                                      sizes, strides);
    auto packedType =
        mlir::MemRefType::get({1, 1, 64}, builder.getF16Type(),
                              mlir::MemRefLayoutAttrInterface{}, ncx);
    auto packed =
        builder.create<wafer::LayoutMaterializeOp>(loc, packedType, from);
    builder.create<wafer::MoveCopyIntoOp>(loc, packed, to);
    wafer::TileRegionToInstrLoweringSession session(context);
    ASSERT_TRUE(
        mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    ASSERT_TRUE(mlir::succeeded(
        wafer::target_llvm_detail::verifyTargetInstructionFormats(*module)));
    std::function<int64_t(mlir::Value, int64_t)> evaluate =
        [&](mlir::Value value, int64_t row) -> int64_t {
      if (!value)
        return 0;
      if (value == loop.getInductionVar())
        return row;
      if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
        return constant.value();
      if (auto add = value.getDefiningOp<mlir::arith::AddIOp>())
        return evaluate(add.getLhs(), row) + evaluate(add.getRhs(), row);
      if (auto mul = value.getDefiningOp<mlir::arith::MulIOp>())
        return evaluate(mul.getLhs(), row) * evaluate(mul.getRhs(), row);
      ADD_FAILURE() << "unexpected actual offset expression";
      return -1;
    };
    unsigned reads = 0, writes = 0;
    module->walk([&](wafer::InstrGatherScatterOp move) {
      bool read = move.getSource() == source.getResult();
      bool write = move.getDest() == destination.getResult();
      ASSERT_NE(read, write);
      reads += read;
      writes += write;
      EXPECT_EQ(move.getByteCount(), 128);
      EXPECT_EQ(move.getInnerBytes(), 128);
      auto counts = read ? move.getSrcIterations() : move.getDstIterations();
      EXPECT_EQ(counts, (llvm::ArrayRef<int64_t>{1, 1, 1}));
      for (int64_t row = 0; row < extent; ++row) {
        int64_t relative = evaluate(
            read ? move.getSrcOffsetValue() : move.getDstOffsetValue(), row);
        for (int64_t byte = 0; byte < 128; ++byte)
          EXPECT_EQ((read ? 128 : 64) + relative + byte,
                    2 * (row * 128 + (read ? 64 : 32)) + byte);
      }
    });
    EXPECT_EQ(reads, 1u);
    EXPECT_EQ(writes, 1u);
  }
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

TEST(LowerInstrToTargetLLVMTest,
     PackedReadExtractsExactBitsAcrossUnalignedRows) {
  for (int64_t rows : {1024, 1025, 1031})
    for (int64_t column = 0; column < 8; ++column)
      for (bool dynamic : {false, true}) {
        SCOPED_TRACE(llvm::formatv("rows={0}, column={1}, dynamic={2}", rows,
                                   column, dynamic)
                         .str());
        mlir::DialectRegistry registry;
        registerTargetConversionDialects(registry);
        mlir::MLIRContext context(registry);
        context.loadAllAvailableDialects();
        auto text = llvm::formatv(R"mlir(module {{
          func.func @main(%input: memref<1x{0}x1031xi1, #wafer.memory<ddr, tensor>>) {{
            wafer.tile.region(%input : memref<1x{0}x1031xi1, #wafer.memory<ddr, tensor>>) -> () {{
            ^bb0(%src: memref<1x{0}x1031xi1, #wafer.memory<ddr, tensor>>):
              wafer.tile.yield
            }
            return
          }
        })mlir",
                                  rows)
                        .str();
        auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(module);
        wafer::TileRegionOp region;
        module->walk([&](wafer::TileRegionOp op) { region = op; });
        mlir::OpBuilder builder(region.getBody().front().getTerminator());
        auto loc = region.getLoc();
        auto copy = [&](int64_t count, mlir::OpFoldResult row) {
          auto view = builder.create<mlir::memref::SubViewOp>(
              loc, region.getBody().getArgument(0),
              llvm::ArrayRef<mlir::OpFoldResult>{builder.getIndexAttr(0), row,
                                                 builder.getIndexAttr(column)},
              mlir::getAsIndexOpFoldResult(&context, {1, count, 1024}),
              mlir::getAsIndexOpFoldResult(&context, {1, 1, 1}));
          mlir::Value source = view;
          llvm::SmallVector<int64_t> shape{1, count, 1024};
          if (dynamic) {
            source = builder.create<mlir::memref::CollapseShapeOp>(
                loc, source,
                llvm::ArrayRef<mlir::ReassociationIndices>{{0, 1}, {2}});
            shape = {count, 1024};
          }
          auto type = mlir::MemRefType::get(
              shape, builder.getI1Type(), mlir::MemRefLayoutAttrInterface{},
              wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                     wafer::MemLayout::Tensor));
          auto output = builder.create<mlir::memref::AllocOp>(loc, type);
          builder.create<wafer::StorageLoadOp>(loc, source, output);
        };
        mlir::scf::ForOp loop;
        if (dynamic) {
          auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
          auto end = builder.create<mlir::arith::ConstantIndexOp>(loc, 1024);
          auto step = builder.create<mlir::arith::ConstantIndexOp>(loc, 256);
          loop = builder.create<mlir::scf::ForOp>(loc, zero, end, step);
          builder.setInsertionPoint(loop.getBody()->getTerminator());
          copy(256, loop.getInductionVar());
          builder.setInsertionPointAfter(loop);
        } else {
          for (int64_t row = 0; row < 1024; row += 256)
            copy(256, builder.getIndexAttr(row));
        }
        if (rows != 1024)
          copy(rows - 1024, builder.getIndexAttr(1024));
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        wafer::TileRegionToInstrLoweringSession session(context);
        ASSERT_TRUE(
            mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        EXPECT_EQ(countOps<wafer::StorageLoadOp>(*module), 0u);
        EXPECT_EQ(countOps<wafer::InstrWDMAOp>(*module), 0u);
        unsigned expectedReads = dynamic ? 1 : 4;
        expectedReads += rows != 1024;
        EXPECT_EQ(countOps<wafer::InstrRDMAOp>(*module), expectedReads);
        bool hasAlignedSingleRow = rows == 1025 && column == 0;
        EXPECT_EQ(countOps<wafer::InstrBit2FpOp>(*module),
                  expectedReads - hasAlignedSingleRow);

        // Execute actual byte descriptors using the original bit address as
        // each bit's identity. This checks every coordinate, including holes,
        // all eight bit residues, dynamic main blocks and partial final rows.
        llvm::SmallVector<wafer::InstrRDMAOp> reads;
        module->walk([&](wafer::InstrRDMAOp read) { reads.push_back(read); });
        int64_t firstRow = 0;
        for (auto read : reads) {
          auto expand =
              llvm::find_if(read.getDest().getUsers(), [](mlir::Operation *op) {
                return mlir::isa<wafer::InstrBit2FpOp>(op);
              });
          if (expand == read.getDest().user_end() && hasAlignedSingleRow) {
            ASSERT_EQ(firstRow, 1024);
            ASSERT_EQ(read.getByteCount(), 128u);
            llvm::SmallVector<int64_t> strides;
            int64_t offset;
            ASSERT_TRUE(mlir::succeeded(mlir::getStridesAndOffset(
                mlir::cast<mlir::MemRefType>(read.getSource().getType()),
                strides, offset)));
            ASSERT_EQ(offset, 1024 * 1031);
            ++firstRow;
            continue;
          }
          ASSERT_NE(expand, read.getDest().user_end());
          auto window =
              read.getSource().getDefiningOp<mlir::memref::SubViewOp>();
          ASSERT_TRUE(window);
          auto expanded = mlir::cast<wafer::InstrBit2FpOp>(*expand).getDest();
          llvm::SmallVector<wafer::InstrGatherScatterOp> gathers;
          for (auto *user : expanded.getUsers())
            if (auto gather = mlir::dyn_cast<wafer::InstrGatherScatterOp>(user))
              gathers.push_back(gather);
          ASSERT_FALSE(gathers.empty());
          auto compact = gathers.front().getDest();
          auto compactType = mlir::cast<mlir::MemRefType>(compact.getType());
          auto count = compactType.getNumElements();
          unsigned repeats = read->getParentOfType<mlir::scf::ForOp>() ? 4 : 1;
          for (unsigned repeat = 0; repeat < repeats; ++repeat) {
            std::function<int64_t(mlir::Value)> evaluate =
                [&](mlir::Value value) -> int64_t {
              if (auto constant = mlir::getConstantIntValue(value))
                return *constant;
              if (loop && value == loop.getInductionVar())
                return repeat * 256;
              auto *operation = value.getDefiningOp();
              EXPECT_TRUE(operation);
              if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(operation))
                return evaluate(add.getLhs()) + evaluate(add.getRhs());
              if (auto sub = mlir::dyn_cast<mlir::arith::SubIOp>(operation))
                return evaluate(sub.getLhs()) - evaluate(sub.getRhs());
              if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(operation))
                return evaluate(mul.getLhs()) * evaluate(mul.getRhs());
              ADD_FAILURE() << "unexpected address expression";
              return -1;
            };
            auto offset = window.getMixedOffsets().front();
            auto constant = mlir::getConstantIntValue(offset);
            int64_t windowStart =
                constant ? *constant
                         : evaluate(mlir::cast<mlir::Value>(offset));
            ASSERT_GE(windowStart, 0);
            ASSERT_EQ(windowStart % 8, 0);
            ASSERT_LE(windowStart / 8 + read.getByteCount(),
                      (rows * 1031 + 7) / 8);
            std::vector<int64_t> actual(count, -1);
            for (auto gather : gathers) {
              ASSERT_FALSE(gather.getSrcOffsetValue());
              ASSERT_FALSE(gather.getDstOffsetValue());
              auto srcStrides = gather.getSrcStrides();
              auto dstStrides = gather.getDstStrides();
              auto srcCounts = gather.getSrcIterations();
              auto dstCounts = gather.getDstIterations();
              // The contiguous destination may have a different descriptor
              // nesting from the strided source. Enumerate both independently.
              auto enumerate = [&](auto strides, auto counts, int64_t offset) {
                std::vector<int64_t> addresses;
                for (int64_t k = 0; k < counts[2]; ++k)
                  for (int64_t j = 0; j < counts[1]; ++j)
                    for (int64_t i = 0; i < counts[0]; ++i)
                      for (int64_t byte = 0;
                           byte < static_cast<int64_t>(gather.getInnerBytes());
                           byte += 2)
                        addresses.push_back(offset + k * strides[2] +
                                            j * strides[1] + i * strides[0] +
                                            byte);
                return addresses;
              };
              auto source = enumerate(srcStrides, srcCounts,
                                      gather.getSrcOffset().value_or(0));
              auto dest = enumerate(dstStrides, dstCounts,
                                    gather.getDstOffset().value_or(0));
              ASSERT_EQ(source.size(), dest.size());
              for (auto [src, dst] : llvm::zip_equal(source, dest)) {
                ASSERT_EQ(src % 2, 0);
                ASSERT_EQ(dst % 2, 0);
                ASSERT_GE(src, 0);
                ASSERT_LT(src / 2, read.getByteCount() * 8);
                ASSERT_GE(dst, 0);
                ASSERT_LT(dst / 2, count);
                ASSERT_EQ(actual[dst / 2], -1);
                actual[dst / 2] = windowStart + src / 2;
              }
            }
            for (int64_t element = 0; element < count; ++element)
              ASSERT_EQ(actual[element], (firstRow + element / 1024) * 1031 +
                                             column + element % 1024);
            firstRow += count / 1024;
          }
        }
        EXPECT_EQ(firstRow, rows);
      }
}

TEST(LowerInstrToTargetLLVMTest, PackedDmaCoversRowsAndPreservesAdjacentBytes) {
  for (bool local : {false, true})
    for (int64_t rows : {1024, 1025, 1031}) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(local);
      mlir::DialectRegistry registry;
      registerTargetConversionDialects(registry);
      mlir::MLIRContext context(registry);
      context.loadAllAvailableDialects();
      auto text = llvm::formatv(R"mlir(module {{
      func.func @main(%input: memref<2x{0}x1024xi1, #wafer.memory<ddr, tensor>>,
                      %output: memref<2x{0}x1024xi1, #wafer.memory<ddr, tensor>>) {{
        wafer.tile.region(%input, %output : memref<2x{0}x1024xi1, #wafer.memory<ddr, tensor>>,
          memref<2x{0}x1024xi1, #wafer.memory<ddr, tensor>>) -> () {{
        ^bb0(%src: memref<2x{0}x1024xi1, #wafer.memory<ddr, tensor>>,
             %dst: memref<2x{0}x1024xi1, #wafer.memory<ddr, tensor>>):
          wafer.tile.yield
        }
        return
      }
    })mlir",
                                rows)
                      .str();
      auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      ASSERT_TRUE(module);
      wafer::TileRegionOp region;
      module->walk([&](wafer::TileRegionOp op) { region = op; });
      mlir::OpBuilder builder(region.getBody().front().getTerminator());
      auto loc = region.getLoc();
      llvm::SmallVector<mlir::Value> endpoints{region.getBody().getArgument(0),
                                               region.getBody().getArgument(1)};
      if (local) {
        auto memory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                             wafer::MemLayout::Tensor);
        auto type =
            mlir::MemRefType::get({2, rows, 1024}, builder.getI1Type(),
                                  mlir::MemRefLayoutAttrInterface{}, memory);
        for (auto &endpoint : endpoints)
          endpoint = builder.create<mlir::memref::AllocOp>(loc, type);
      }
      auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto end =
          builder.create<mlir::arith::ConstantIndexOp>(loc, rows / 256 * 256);
      auto step = builder.create<mlir::arith::ConstantIndexOp>(loc, 256);
      auto loop = builder.create<mlir::scf::ForOp>(loc, zero, end, step);
      auto copy = [&](int64_t count, mlir::OpFoldResult row) {
        auto sizes = mlir::getAsIndexOpFoldResult(&context, {2, count, 512});
        auto strides = mlir::getAsIndexOpFoldResult(&context, {1, 1, 1});
        auto view = [&](unsigned argument, int64_t channel) {
          llvm::SmallVector<mlir::OpFoldResult> offsets{
              builder.getIndexAttr(0), row, builder.getIndexAttr(channel)};
          return builder.create<mlir::memref::SubViewOp>(
              loc, endpoints[argument], offsets, sizes, strides);
        };
        auto src = view(0, 512), dst = view(1, 0);
        auto memory = wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                             wafer::MemLayout::Tensor);
        auto type =
            mlir::MemRefType::get({2, count, 512}, builder.getI1Type(),
                                  mlir::MemRefLayoutAttrInterface{}, memory);
        auto buffer = builder.create<mlir::memref::AllocOp>(loc, type);
        if (local) {
          builder.create<mlir::memref::CopyOp>(loc, src, buffer);
          builder.create<wafer::MoveCopyIntoOp>(loc, buffer, dst);
        } else {
          builder.create<wafer::StorageLoadOp>(loc, src, buffer);
          builder.create<wafer::StorageStoreOp>(loc, buffer, dst);
        }
      };
      builder.setInsertionPoint(loop.getBody()->getTerminator());
      copy(256, loop.getInductionVar());
      builder.setInsertionPointAfter(loop);
      if (rows % 256)
        copy(rows % 256, builder.getIndexAttr(rows / 256 * 256));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      wafer::TileRegionToInstrLoweringSession session(context);
      ASSERT_TRUE(
          mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      EXPECT_EQ(countOps<wafer::StorageLoadOp>(*module), 0u);
      EXPECT_EQ(countOps<wafer::StorageStoreOp>(*module), 0u);

      // Execute the actual byte descriptors with address identities as data.
      // Untouched bytes retain a sentinel; every destination is written once.
      std::vector<int64_t> actual(2 * rows * 128, -1),
          expected(actual.size(), -1);
      unsigned moves = 0;
      auto check = [&](auto read, auto write) {
        ASSERT_TRUE(write);
        ++moves;
        bool repeated = static_cast<bool>(
            read->template getParentOfType<mlir::scf::ForOp>());
        auto enumerate = [&](mlir::Value value, int64_t iv, int64_t inner,
                             llvm::ArrayRef<int64_t> counts,
                             llvm::ArrayRef<int64_t> strides) {
          auto subview = value.getDefiningOp<mlir::memref::SubViewOp>();
          EXPECT_TRUE(subview);
          llvm::SmallVector<int64_t> logicalStrides;
          int64_t ignored;
          EXPECT_TRUE(mlir::succeeded(mlir::getStridesAndOffset(
              subview.getSourceType(), logicalStrides, ignored)));
          int64_t bits = 0;
          for (auto [axis, offset] :
               llvm::enumerate(subview.getMixedOffsets())) {
            auto constant = mlir::getConstantIntValue(offset);
            if (!constant) {
              EXPECT_EQ(mlir::cast<mlir::Value>(offset),
                        loop.getInductionVar());
            }
            bits += (constant ? *constant : iv) * logicalStrides[axis];
          }
          EXPECT_EQ(bits % 8, 0);
          std::vector<int64_t> addresses;
          for (int64_t k = 0; k < counts[2]; ++k)
            for (int64_t j = 0; j < counts[1]; ++j)
              for (int64_t i = 0; i < counts[0]; ++i)
                for (int64_t byte = 0; byte < inner; ++byte)
                  addresses.push_back(bits / 8 + k * strides[2] +
                                      j * strides[1] + i * strides[0] + byte);
          return addresses;
        };
        for (int64_t iv = 0; iv < (repeated ? rows / 256 * 256 : 1);
             iv += 256) {
          auto source =
              enumerate(read.getSource(), iv, read.getInnerBytes(),
                        read.getSrcIterations(), read.getSrcStrides());
          auto destination =
              enumerate(write.getDest(), iv, write.getInnerBytes(),
                        write.getDstIterations(), write.getDstStrides());
          ASSERT_EQ(source.size(), destination.size());
          EXPECT_EQ(source.size(), static_cast<size_t>(read.getByteCount()));
          EXPECT_EQ(destination.size(),
                    static_cast<size_t>(write.getByteCount()));
          for (auto [src, dst] : llvm::zip_equal(source, destination)) {
            ASSERT_GE(src, 0);
            ASSERT_LT(src, static_cast<int64_t>(actual.size()));
            ASSERT_GE(dst, 0);
            ASSERT_LT(dst, static_cast<int64_t>(actual.size()));
            ASSERT_EQ(actual[dst], -1);
            actual[dst] = src;
          }
        }
      };
      module->walk([&](wafer::InstrRDMAOp read) {
        wafer::InstrWDMAOp write;
        for (auto *user : read.getDest().getUsers())
          if (auto dma = mlir::dyn_cast<wafer::InstrWDMAOp>(user))
            write = dma;
        check(read, write);
      });
      module->walk([&](wafer::InstrGatherScatterOp read) {
        if (!read.getDest().getDefiningOp<mlir::memref::AllocOp>())
          return;
        wafer::InstrGatherScatterOp write;
        for (auto *user : read.getDest().getUsers())
          if (auto dma = mlir::dyn_cast<wafer::InstrGatherScatterOp>(user);
              dma && dma.getSource() == read.getDest())
            write = dma;
        ASSERT_TRUE(write);
        auto expectCompact = [&](llvm::ArrayRef<int64_t> counts,
                                 llvm::ArrayRef<int64_t> strides, int64_t inner,
                                 int64_t bytes) {
          int64_t ordinal = 0;
          for (int64_t k = 0; k < counts[2]; ++k)
            for (int64_t j = 0; j < counts[1]; ++j)
              for (int64_t i = 0; i < counts[0]; ++i)
                for (int64_t byte = 0; byte < inner; ++byte)
                  EXPECT_EQ(k * strides[2] + j * strides[1] + i * strides[0] +
                                byte,
                            ordinal++);
          EXPECT_EQ(ordinal, bytes);
        };
        expectCompact(read.getDstIterations(), read.getDstStrides(),
                      read.getInnerBytes(), read.getByteCount());
        expectCompact(write.getSrcIterations(), write.getSrcStrides(),
                      write.getInnerBytes(), write.getByteCount());
        check(read, write);
      });
      EXPECT_EQ(moves, rows % 256 ? 2u : 1u);
      for (int64_t row = 0; row < 2 * rows; ++row)
        for (int64_t byte = 0; byte < 64; ++byte)
          expected[row * 128 + byte] = row * 128 + 64 + byte;
      EXPECT_EQ(actual, expected);
    }
}

TEST(LowerInstrToTargetLLVMTest, DMAOffsetsRequireAnInBoundsPhysicalRange) {
  for (bool read : {false, true})
    for (unsigned mode : {0u, 1u, 2u}) {
      mlir::DialectRegistry registry;
      registerTargetConversionDialects(registry);
      mlir::MLIRContext context(registry);
      context.loadAllAvailableDialects();
      std::string ddr = "memref<1x1024x128xbf16, #wafer.memory<ddr, ncx>>";
      std::string spm = "memref<1x1024x128xbf16, #wafer.memory<spm, tensor>>";
      std::string text;
      llvm::raw_string_ostream out(text);
      out << "module { func.func @entry(%ddr: " << ddr
          << ", %unknown: index) {\n"
          << "%local = memref.alloc() {wafer.spm.offset = "
             "#wafer.spm_offset<0>} : "
          << spm << "\n%offset = arith.constant " << (mode == 1 ? 262144 : 128)
          << " : index\n"
          << "wafer.instr."
          << (read ? "rdma %ddr to %local src" : "wdma %local to %ddr dst")
          << "_offset_value(" << (mode == 2 ? "%unknown" : "%offset") << ") {"
          << "byte_count = 256 : i64, inner_bytes = 128 : i64, "
          << (read ? "dst" : "src") << "_offset = 0 : i64, "
          << (read ? "src" : "dst") << "_strides = array<i64: 128, 0, 0>, "
          << (read ? "src" : "dst")
          << "_iterations = array<i64: 2, 1, 1>} : " << (read ? ddr : spm)
          << " to " << (read ? spm : ddr) << "\nreturn\n}}";
      auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      ASSERT_TRUE(module);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      std::string diagnostic;
      mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &d) {
        llvm::raw_string_ostream stream(diagnostic);
        d.print(stream);
        return mlir::success();
      });
      EXPECT_EQ(mlir::succeeded(
                    wafer::target_llvm_detail::verifyTargetInstructionFormats(
                        *module)),
                mode == 0);
      if (mode) {
        EXPECT_NE(diagnostic.find(mode == 1
                                      ? "target_geometry_mismatch"
                                      : "unsupported_target_dynamic_offset"),
                  std::string::npos)
            << diagnostic;
      }
    }
}

TEST(LowerInstrToTargetLLVMTest, BlockedDDRWindowsCopyExactPhysicalBytes) {
  for (int64_t rows : {1024, 1025, 1031})
    for (bool bf16 : {false, true})
      for (unsigned mode : {0u, 1u, 2u, 3u})
        for (bool fuse : {false, true}) {
          SCOPED_TRACE(
              llvm::formatv("rows={0} bf16={1} mode={2}", rows, bf16, mode)
                  .str());
          mlir::DialectRegistry registry;
          registerTargetConversionDialects(registry);
          mlir::MLIRContext context(registry);
          context.loadAllAvailableDialects();
          auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
          module { func.func @entry() {
            wafer.tile.region() -> () { wafer.tile.yield }
            return
          } })mlir",
                                                                &context);
          ASSERT_TRUE(module);
          wafer::TileRegionOp region;
          module->walk([&](wafer::TileRegionOp op) { region = op; });
          auto function = region->getParentOfType<mlir::func::FuncOp>();
          mlir::OpBuilder builder(region.getBody().front().getTerminator());
          auto loc = region.getLoc();
          auto element = bf16 ? builder.getBF16Type() : builder.getF16Type();
          const int64_t channels = mode == 2 ? 131 : 384;
          const int64_t width = mode == 2 ? 67 : mode == 3 ? 64 : 128;
          llvm::SmallVector<int64_t> rootShape{2, rows + 4, channels};
          if (mode == 1)
            rootShape.insert(rootShape.begin() + 1, 1);
          auto type = mlir::MemRefType::get(
              rootShape, element, mlir::MemRefLayoutAttrInterface{},
              wafer::MemoryAttr::get(&context, wafer::MemorySpace::DDR,
                                     wafer::MemLayout::NCx));
          llvm::SmallVector<mlir::Value> roots;
          for (unsigned i = 0; i < 2; ++i) {
            function.insertArgument(i, type, mlir::DictionaryAttr{}, loc);
            region.getInputsMutable().append(function.getArgument(i));
            roots.push_back(region.getBody().front().addArgument(type, loc));
          }
          auto emitWindow = [&](mlir::Value row, int64_t length,
                                mlir::Value channel = {}) {
            auto shape = rootShape;
            shape[shape.size() - 2] = length;
            shape.back() = width;
            auto localType = mlir::MemRefType::get(
                shape, element, mlir::MemRefLayoutAttrInterface{},
                wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                       (mode == 2 || fuse)
                                           ? wafer::MemLayout::NCx
                                           : wafer::MemLayout::Tensor));
            auto local = builder.create<mlir::memref::AllocOp>(loc, localType);
            llvm::SmallVector<mlir::OpFoldResult> offsets(
                shape.size(), builder.getIndexAttr(0));
            offsets[shape.size() - 2] = row;
            offsets.back() = builder.getIndexAttr(64);
            if (channel)
              offsets.back() = channel;
            llvm::SmallVector<mlir::OpFoldResult> sizes, strides;
            for (int64_t extent : shape) {
              sizes.push_back(builder.getIndexAttr(extent));
              strides.push_back(builder.getIndexAttr(1));
            }
            auto src = builder.create<mlir::memref::SubViewOp>(
                loc, roots[0], offsets, sizes, strides);
            auto dst = builder.create<mlir::memref::SubViewOp>(
                loc, roots[1], offsets, sizes, strides);
            if (fuse) {
              auto stagingType = mlir::MemRefType::get(
                  shape, element, mlir::MemRefLayoutAttrInterface{},
                  wafer::MemoryAttr::get(&context, wafer::MemorySpace::SPM,
                                         wafer::MemLayout::Tensor));
              auto staging =
                  builder.create<mlir::memref::AllocOp>(loc, stagingType);
              local.erase();
              builder.create<wafer::StorageLoadOp>(loc, src, staging);
              auto compute = builder.create<wafer::LayoutMaterializeOp>(
                  loc, localType, staging);
              auto publication = builder.create<wafer::LayoutMaterializeOp>(
                  loc, stagingType, compute);
              builder.create<wafer::StorageStoreOp>(loc, publication, dst);
            } else {
              builder.create<wafer::StorageLoadOp>(loc, src, local);
              builder.create<wafer::StorageStoreOp>(loc, local, dst);
            }
          };
          if (mode == 3) {
            auto row = builder.create<mlir::arith::ConstantIndexOp>(loc, 2);
            auto begin = builder.create<mlir::arith::ConstantIndexOp>(loc, 64);
            auto end = builder.create<mlir::arith::ConstantIndexOp>(loc, 192);
            auto step = builder.create<mlir::arith::ConstantIndexOp>(loc, 64);
            auto loop = builder.create<mlir::scf::ForOp>(loc, begin, end, step);
            builder.setInsertionPointToStart(loop.getBody());
            emitWindow(row, rows, loop.getInductionVar());
          } else if (mode == 1) {
            auto begin = builder.create<mlir::arith::ConstantIndexOp>(loc, 2);
            auto end = builder.create<mlir::arith::ConstantIndexOp>(
                loc, 2 + rows / 256 * 256);
            auto step = builder.create<mlir::arith::ConstantIndexOp>(loc, 256);
            auto loop = builder.create<mlir::scf::ForOp>(loc, begin, end, step);
            builder.setInsertionPointToStart(loop.getBody());
            emitWindow(loop.getInductionVar(), 256);
            builder.setInsertionPointAfter(loop);
            if (rows % 256)
              emitWindow(end, rows % 256);
          } else {
            auto row = builder.create<mlir::arith::ConstantIndexOp>(loc, 2);
            emitWindow(row, rows);
          }
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
          if (fuse) {
            mlir::IRRewriter rewriter(&context);
            wafer::compiler::detail::fuseDMALayoutMovements(*module, rewriter);
            EXPECT_EQ(countOps<wafer::LayoutMaterializeOp>(*module), 0u);
          }
          wafer::TileRegionToInstrLoweringSession session(context);
          ASSERT_TRUE(mlir::succeeded(
              wafer::convertTileRegionToInstr(region, session)));
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
          EXPECT_EQ(countOps<wafer::StorageLoadOp>(*module), 0u);
          EXPECT_EQ(countOps<wafer::StorageStoreOp>(*module), 0u);
          EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
          // Full blocks stay symbolic. A narrow three-element tail has two
          // padding bytes per SPM row: compact DMA cannot skip those bytes.
          unsigned maxMoves = mode == 2 ? 2 * (rows + 1) : 8;
          EXPECT_LE(countOps<wafer::InstrRDMAOp>(*module), maxMoves);
          EXPECT_LE(countOps<wafer::InstrWDMAOp>(*module), maxMoves);
          auto geometry = wafer::computeWaferPhysicalTensorInfo(type);
          ASSERT_TRUE(geometry);
          std::vector<int64_t> actual(geometry->physicalBytes, -1);
          std::vector<int64_t> expected(actual.size(), -1);
          llvm::DenseMap<mlir::Value, int64_t> values;
          llvm::DenseMap<mlir::Value, std::vector<int64_t>> buffers;
          std::function<void(mlir::Operation *)> run = [&](mlir::Operation
                                                               *op) {
            if (auto c = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(op))
              values[c] = c.value();
            else if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(op))
              values[add] = values.at(add.getLhs()) + values.at(add.getRhs());
            else if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(op))
              values[mul] = values.at(mul.getLhs()) * values.at(mul.getRhs());
            else if (auto div = mlir::dyn_cast<mlir::arith::DivUIOp>(op))
              values[div] = values.at(div.getLhs()) / values.at(div.getRhs());
            else if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
              auto info =
                  wafer::computeWaferPhysicalTensorInfo(alloc.getType());
              ASSERT_TRUE(info);
              buffers[alloc] = std::vector<int64_t>(info->physicalBytes, -1);
            } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
              for (int64_t i = values.at(loop.getLowerBound());
                   i < values.at(loop.getUpperBound());
                   i += values.at(loop.getStep())) {
                values[loop.getInductionVar()] = i;
                for (auto &nested : *loop.getBody())
                  run(&nested);
              }
            } else if (mlir::isa<wafer::InstrRDMAOp, wafer::InstrWDMAOp>(op)) {
              auto execute = [&](auto move, bool read,
                                 llvm::ArrayRef<int64_t> counts,
                                 llvm::ArrayRef<int64_t> strides) {
                auto offset = [&](bool source) {
                  auto dynamic = source ? move.getSrcOffsetValue()
                                        : move.getDstOffsetValue();
                  auto fixed = source ? move.getSrcOffsetAttr()
                                      : move.getDstOffsetAttr();
                  return dynamic ? values.at(dynamic)
                                 : (fixed ? fixed.getInt() : 0);
                };
                EXPECT_EQ(read ? move.getSource() : move.getDest(),
                          roots[read ? 0 : 1]);
                auto buffer =
                    buffers.find(read ? move.getDest() : move.getSource());
                ASSERT_NE(buffer, buffers.end());
                auto &local = buffer->second;
                for (int64_t b = 0; b < int64_t(move.getByteCount()); ++b) {
                  int64_t n = b / move.getInnerBytes();
                  int64_t ddr = offset(read) + b % move.getInnerBytes();
                  for (unsigned axis = 0; axis < 3; ++axis) {
                    ddr += n % counts[axis] * strides[axis];
                    n /= counts[axis];
                  }
                  int64_t spm = offset(!read) + b;
                  ASSERT_GE(ddr, 0);
                  ASSERT_LT(ddr, int64_t(actual.size()));
                  ASSERT_GE(spm, 0);
                  ASSERT_LT(spm, int64_t(local.size()));
                  if (read) {
                    ASSERT_EQ(local[spm], -1);
                    local[spm] = ddr;
                  } else {
                    ASSERT_NE(local[spm], -1);
                    ASSERT_EQ(actual[ddr], -1);
                    actual[ddr] = local[spm];
                  }
                }
              };
              if (auto read = mlir::dyn_cast<wafer::InstrRDMAOp>(op))
                execute(read, true, read.getSrcIterations(),
                        read.getSrcStrides());
              else {
                auto write = mlir::cast<wafer::InstrWDMAOp>(op);
                execute(write, false, write.getDstIterations(),
                        write.getDstStrides());
              }
            } else {
              for (auto &r : op->getRegions())
                for (auto &block : r)
                  for (auto &nested : block)
                    run(&nested);
            }
          };
          run(*module);
          auto calculator =
              wafer::WaferStaticPhysicalOffsetCalculator::create(type);
          ASSERT_TRUE(calculator);
          for (int64_t batch = 0; batch < 2; ++batch)
            for (int64_t row = 2; row < rows + 2; ++row)
              for (int64_t c = 64; c < 64 + (mode == 3 ? 128 : width); ++c) {
                llvm::SmallVector<int64_t> point{batch, row, c};
                if (mode == 1)
                  point.insert(point.begin() + 1, 0);
                auto offset = calculator->getByteOffset(point);
                ASSERT_TRUE(offset);
                expected[*offset] = *offset;
                expected[*offset + 1] = *offset + 1;
              }
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

TEST(LowerInstrToTargetLLVMTest, PackedCopyUsesOnlyCompleteBlockedAllocations) {
  for (llvm::StringRef layout : {"cx", "ncx"})
    for (int64_t extent : {1024, 1025, 1031})
      for (bool partial : {false, true}) {
        SCOPED_TRACE(
            llvm::formatv("{0}:{1}:{2}", layout, extent, partial).str());
        mlir::DialectRegistry registry;
        registerTargetConversionDialects(registry);
        mlir::MLIRContext context(registry);
        context.loadAllAvailableDialects();
        auto type =
            llvm::formatv("memref<2x{0}x197xi1, #wafer.memory<spm, {1}>>",
                          extent, layout)
                .str();
        auto destType =
            partial ? llvm::formatv("memref<2x{0}x197xi1, "
                                    "strided<[{1},197,1], offset: {1}>, "
                                    "#wafer.memory<spm, {2}>>",
                                    extent, extent * 197, layout)
                          .str()
                    : type;
        auto destination = partial ? llvm::formatv(R"mlir(
          %base = memref.alloc() : memref<3x{0}x197xi1, #wafer.memory<spm, {1}>>
          %dest = memref.subview %base[1,0,0] [2,{0},197] [1,1,1]
            : memref<3x{0}x197xi1, #wafer.memory<spm, {1}>> to {2}
        )mlir",
                                                   extent, layout, destType)
                                         .str()
                                   : "%dest = memref.alloc() : " + type;
        auto text = llvm::formatv(R"mlir(module {{ func.func @main() {{
          %token = arith.constant false
          %unused = wafer.tile.region(%token : i1) -> (i1) {{
          ^bb0(%done: i1):
            %source = memref.alloc() : {0}
            {1}
            wafer.tile.copy_into %source into %dest : {0} into {2}
            wafer.tile.yield %done : i1
          }
          return
        } })mlir",
                                  type, destination, destType)
                        .str();
        auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(module) << text;
        wafer::TileRegionOp region;
        module->walk([&](wafer::TileRegionOp op) { region = op; });
        wafer::TileRegionToInstrLoweringSession session(context);
        EXPECT_EQ(
            mlir::succeeded(wafer::convertTileRegionToInstr(region, session)),
            !partial);
        EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module),
                  partial ? 0u : 1u);
        if (partial)
          continue;
        module->walk([&](wafer::InstrGatherScatterOp op) {
          auto src = mlir::cast<mlir::MemRefType>(op.getSource().getType());
          auto info = wafer::computeWaferPhysicalTensorInfo(src);
          ASSERT_TRUE(info);
          EXPECT_EQ(op.getByteCountAttr().getInt(), info->physicalBytes);
          EXPECT_EQ(op.getInnerBytesAttr().getInt(), info->physicalBytes);
          EXPECT_EQ(op.getSrcIterationsAttr().asArrayRef(),
                    (llvm::ArrayRef<int64_t>{1, 1, 1}));
          EXPECT_EQ(op.getDstIterationsAttr().asArrayRef(),
                    (llvm::ArrayRef<int64_t>{1, 1, 1}));
          EXPECT_FALSE(op.getSrcOffsetAttr());
          EXPECT_FALSE(op.getDstOffsetAttr());
        });
        EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
        EXPECT_TRUE(mlir::succeeded(
            wafer::target_llvm_detail::verifyTargetInstructionFormats(
                *module)));
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
  // Unlike a complete same-encoding byte copy, this conversion needs packed
  // bit movement between different physical mappings and remains unsupported.
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
          %a = memref.alloc() : memref<1x1024x65xi1, #wafer.memory<spm, ncx>>
          %b = memref.alloc() : memref<1x1024x65xi1, #wafer.memory<spm, tensor>>
          memref.copy %a, %b : memref<1x1024x65xi1, #wafer.memory<spm, ncx>>
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
     DifferentLayoutFamiliesUseProvenTraversalWithoutMovement) {
  for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (bool reverse : {false, true})
        for (bool mapped : {false, true}) {
          SCOPED_TRACE(
              llvm::formatv("{0}:{1}:{2}:{3}", dtype, extent, reverse, mapped)
                  .str());
          mlir::DialectRegistry registry;
          registerTargetConversionDialects(registry);
          mlir::MLIRContext context(registry);
          context.loadAllAvailableDialects();
          auto from =
              llvm::formatv("memref<1x{0}x128x{1}, #wafer.memory<spm, {2}>>",
                            extent, dtype, reverse ? "cx" : "ncx")
                  .str();
          auto to =
              llvm::formatv("memref<1x{0}x128x{1}, #wafer.memory<spm, {2}>>",
                            extent, dtype, reverse ? "ncx" : "cx")
                  .str();
          llvm::StringRef maps =
              mapped ? "{indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1,d2)>, "
                       "affine_map<(d0,d1,d2)->(d0,d1,d2)>]}"
                     : "";
          auto text = llvm::formatv(R"mlir(module {{ func.func @main() {{
          wafer.tile.region() -> () {{
            %src = memref.alloc() : {0}
            %result = wafer.tile.elementwise <exp> %src {2} : ({0}) -> {1}
            wafer.tile.yield
          }
          return
        } })mlir",
                                    from, to, maps)
                          .str();
          auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
          ASSERT_TRUE(module) << text;
          wafer::TileRegionOp region;
          module->walk([&](wafer::TileRegionOp op) { region = op; });
          wafer::TileRegionToInstrLoweringSession session(context);
          ASSERT_TRUE(mlir::succeeded(
              wafer::convertTileRegionToInstr(region, session)));
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
          EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
          EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 2u);
          EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*module), 1u);
          EXPECT_TRUE(mlir::succeeded(
              wafer::target_llvm_detail::verifyTargetInstructionFormats(
                  *module)));
        }
}

TEST(LowerInstrToTargetLLVMTest,
     ConvertUsesSelectedDifferentFamilyWithoutMovement) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool reverse : {false, true}) {
      mlir::DialectRegistry registry;
      registerTargetConversionDialects(registry);
      mlir::MLIRContext context(registry);
      context.loadAllAvailableDialects();
      auto source =
          llvm::formatv("memref<1x{0}x128x{1}, #wafer.memory<spm, ncx>>",
                        extent, reverse ? "bf16" : "f16")
              .str();
      auto dest = llvm::formatv("memref<1x{0}x128x{1}, #wafer.memory<spm, cx>>",
                                extent, reverse ? "f16" : "bf16")
                      .str();
      auto text = llvm::formatv(R"mlir(module {{ func.func @main() {{
        wafer.tile.region() -> () {{
          %src = memref.alloc() : {0}
          %result = wafer.tile.compute.convert %src : {0} to {1}
          wafer.tile.yield
        }
        return
      } })mlir",
                                source, dest)
                      .str();
      auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      ASSERT_TRUE(module) << text;
      wafer::TileRegionOp region;
      module->walk([&](wafer::TileRegionOp op) { region = op; });
      wafer::TileRegionToInstrLoweringSession session(context);
      ASSERT_TRUE(
          mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
      EXPECT_EQ(countOps<wafer::InstrConvertOp>(*module), 1u);
      EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
      EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 2u);
      EXPECT_TRUE(mlir::succeeded(
          wafer::target_llvm_detail::verifyTargetInstructionFormats(*module)));
    }
}

TEST(LowerInstrToTargetLLVMTest,
     LinearCTValidatesActualTraversalAcrossFamiliesAndDtypes) {
  enum class Case { Compatible, Padding, Stride, Dtype };
  for (int64_t extent : {1024, 1025, 1031})
    for (llvm::StringRef kind :
         {"elementwise", "convert", "bit2fp", "mask_move"})
      for (Case test :
           {Case::Compatible, Case::Padding, Case::Stride, Case::Dtype}) {
        if (test == Case::Dtype && kind != "convert")
          continue;
        SCOPED_TRACE(
            llvm::formatv("{0}:{1}:{2}", extent, kind, static_cast<int>(test))
                .str());
        mlir::DialectRegistry registry;
        registerTargetConversionDialects(registry);
        mlir::MLIRContext context(registry);
        context.loadAllAvailableDialects();
        auto shape = test == Case::Compatible || test == Case::Stride
                         ? llvm::formatv("1x1x{0}", extent).str()
                         : llvm::formatv("2x{0}x197", extent).str();
        auto source =
            llvm::formatv("memref<{0}x{1}, {2}#wafer.memory<spm, {3}>>", shape,
                          kind == "bit2fp" ? "i1" : "f16",
                          test == Case::Stride ? "strided<[4096, 4096, 2]>, "
                                               : "",
                          test == Case::Stride ? "tensor" : "ncx")
                .str();
        auto dest =
            llvm::formatv("memref<{0}x{1}, #wafer.memory<spm, {2}>>", shape,
                          kind == "convert"
                              ? (test == Case::Dtype ? "f32" : "bf16")
                              : "f16",
                          test == Case::Stride ? "tensor" : "cx")
                .str();
        std::string instruction;
        if (kind == "elementwise")
          instruction =
              llvm::formatv(
                  "wafer.instr.elementwise #wafer.instr_elementwise_kind<abs> "
                  "%src into %dst : {0} into {1}",
                  source, dest)
                  .str();
        else if (kind == "convert")
          instruction =
              llvm::formatv("wafer.instr.convert "
                            "#wafer.instr_convert_kind<{0}> %src into "
                            "%dst {1} : {2} to {3}",
                            test == Case::Dtype ? "fp16_fp32" : "fp16_bf16",
                            test == Case::Dtype ? ""
                                                : "{rounding_mode = 0 : i64}",
                            source, dest)
                  .str();
        else if (kind == "bit2fp")
          instruction =
              llvm::formatv("wafer.instr.bit2fp %src into %dst : {0} to {1}",
                            source, dest)
                  .str();
        else
          instruction = llvm::formatv("wafer.instr.mask_move %src, %src into "
                                      "%dst : {0}, {0} into {1}",
                                      source, dest)
                            .str();
        auto text =
            llvm::formatv(
                "module {{ func.func @main() {{ %src = memref.alloc() : {0}\n"
                "%dst = memref.alloc() : {1}\n{2}\nreturn } }",
                source, dest, instruction)
                .str();
        auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(module) << text;
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        std::string diagnostics;
        mlir::ScopedDiagnosticHandler handler(
            &context, [&](mlir::Diagnostic &diagnostic) {
              llvm::raw_string_ostream stream(diagnostics);
              diagnostic.print(stream);
              return mlir::success();
            });
        auto result =
            wafer::target_llvm_detail::verifyTargetInstructionFormats(*module);
        EXPECT_EQ(mlir::succeeded(result), test == Case::Compatible)
            << diagnostics;
        if (test != Case::Compatible) {
          EXPECT_NE(diagnostics.find("unsupported_target_physical_traversal"),
                    std::string::npos)
              << diagnostics;
        }
      }
}

TEST(LowerInstrToTargetLLVMTest, ScalarRHSHasNoStorageOrMovement) {
  for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (llvm::StringRef kind : {"add", "mul", "sub", "max", "min", "lt"}) {
        mlir::DialectRegistry registry;
        registerTargetConversionDialects(registry);
        mlir::MLIRContext context(registry);
        context.loadAllAvailableDialects();
        auto text =
            llvm::formatv(R"mlir(module {{ func.func @main() {{
          %token = arith.constant false
          %unused = wafer.tile.region(%token : i1) -> (i1) {{
          ^bb0(%done: i1):
            %lhs = memref.alloc() : memref<2x{0}x32x{1}, #wafer.memory<spm, tensor>>
            %scalar = arith.constant 2.0 : {1}
            %result = wafer.tile.elementwise <{2}> %lhs, %scalar
              : (memref<2x{0}x32x{1}, #wafer.memory<spm, tensor>>, {1})
                -> memref<2x{0}x32x{3}, #wafer.memory<spm, tensor>>
            wafer.tile.yield %done : i1
          }
          return
        } })mlir",
                          extent, dtype, kind, kind == "lt" ? "i1" : dtype)
                .str();
        auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(module) << text;
        wafer::TileRegionOp region;
        module->walk([&](wafer::TileRegionOp op) { region = op; });
        wafer::TileRegionToInstrLoweringSession session(context);
        ASSERT_TRUE(
            mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        EXPECT_EQ(countOps<mlir::memref::AllocOp>(*module), 2u);
        EXPECT_EQ(countOps<wafer::InstrFillOp>(*module), 0u);
        EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*module), 0u);
        EXPECT_EQ(countOps<wafer::InstrElementwiseOp>(*module), 1u);
        module->walk([&](wafer::InstrElementwiseOp op) {
          EXPECT_TRUE(mlir::isa<mlir::FloatType>(op.getInputs()[1].getType()));
          EXPECT_EQ(op.getRhsUnitElements(), 0);
          op.setRhsUnitElements(1);
          mlir::ScopedDiagnosticHandler handler(&context,
                                                [](mlir::Diagnostic &) {});
          EXPECT_TRUE(mlir::failed(mlir::verify(op)));
          op.setRhsUnitElements(0);
          auto lhs = op.getInputs()[0], rhs = op.getInputs()[1];
          op->setOperand(0, rhs);
          op->setOperand(1, lhs);
          EXPECT_TRUE(mlir::failed(mlir::verify(op)));
          op->setOperand(0, lhs);
          op->setOperand(1, rhs);
          mlir::OpBuilder builder(op);
          mlir::Type otherType = rhs.getType().isF32()
                                     ? mlir::Type(builder.getBF16Type())
                                     : mlir::Type(builder.getF32Type());
          auto wrongScalar = builder.create<mlir::arith::ConstantOp>(
              op.getLoc(), builder.getFloatAttr(otherType, 2.0));
          op->setOperand(1, wrongScalar.getResult());
          EXPECT_TRUE(mlir::failed(mlir::verify(op)));
          op->setOperand(1, rhs);
          wrongScalar.erase();
          EXPECT_TRUE(mlir::succeeded(mlir::verify(op)));
        });
      }
}

TEST(LowerInstrToTargetLLVMTest, MappedUnitBroadcastKeepsOnlyActualRHSStorage) {
  for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (int64_t unit : {1, 32, 64})
        for (llvm::StringRef kind : {"add", "sub", "div"}) {
          SCOPED_TRACE(
              llvm::formatv("{0}:{1}:{2}:{3}", dtype, extent, unit, kind)
                  .str());
          mlir::DialectRegistry registry;
          registerTargetConversionDialects(registry);
          mlir::MLIRContext context(registry);
          context.loadAllAvailableDialects();
          auto vectorType =
              llvm::formatv("memref<2x{0}x{1}x{2}, #wafer.memory<spm, tensor>>",
                            extent, unit, dtype)
                  .str();
          auto unitType =
              llvm::formatv("memref<{0}x{1}, #wafer.memory<spm, tensor>>", unit,
                            dtype)
                  .str();
          auto text = llvm::formatv(R"mlir(module {{
            func.func @main() {{
              %token = arith.constant false
              %unused = wafer.tile.region(%token : i1) -> (i1) {{
              ^bb0(%done: i1):
                %lhs = memref.alloc() : {0}
                %rhs = memref.alloc() : {1}
                %result = wafer.tile.elementwise <{2}> %lhs, %rhs {{
                  indexing_maps = [affine_map<(b,m,n)->(b,m,n)>,
                                   affine_map<(b,m,n)->(n)>,
                                   affine_map<(b,m,n)->(b,m,n)>]
                } : ({0}, {1}) -> {0}
                wafer.tile.yield %done : i1
              }
              return
            }
          })mlir",
                                    vectorType, unitType, kind)
                          .str();
          auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
          ASSERT_TRUE(source) << text;
          wafer::TileRegionOp region;
          mlir::Value rhs;
          source->walk([&](wafer::TileRegionOp op) { region = op; });
          source->walk(
              [&](wafer::ComputeElementwiseOp op) { rhs = op.getInputs()[1]; });
          wafer::TileRegionToInstrLoweringSession session(context);
          ASSERT_TRUE(mlir::succeeded(
              wafer::convertTileRegionToInstr(region, session)));
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
          EXPECT_EQ(countOps<wafer::InstrGatherScatterOp>(*source), 0u);
          unsigned unitOperations = 0;
          source->walk([&](wafer::InstrElementwiseOp op) {
            if (op.getKind() == wafer::InstrElementwiseKind::Recip) {
              EXPECT_EQ(op.getInputs().front(), rhs);
              EXPECT_EQ(op.getDest().getType(), rhs.getType());
              return;
            }
            ++unitOperations;
            EXPECT_EQ(op.getRhsUnitElements(), unit);
            EXPECT_EQ(op.getInputs()[1].getType(), rhs.getType());
            if (kind != "div") {
              EXPECT_EQ(op.getInputs()[1], rhs);
            }
          });
          EXPECT_EQ(unitOperations, 1u);
        }
}

TEST(LowerInstrToTargetLLVMTest,
     PermutedPhysicalMovementsCopyExactSourceBytes) {
  for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (unsigned mode = 0; mode < 3; ++mode) {
        mlir::DialectRegistry registry;
        registerTargetConversionDialects(registry);
        mlir::MLIRContext context(registry);
        context.loadAllAvailableDialects();
        auto sourceType =
            llvm::formatv("memref<2x128x{0}x{1}, #wafer.memory<spm, {2}>>",
                          extent, dtype,
                          mode == 0   ? "tensor"
                          : mode == 1 ? "cx"
                                      : "ncx")
                .str();
        auto destType =
            llvm::formatv("memref<2x{0}x128x{1}, #wafer.memory<spm, {2}>>",
                          extent, dtype, mode == 0 ? "ncx" : "tensor")
                .str();
        auto text = llvm::formatv(R"mlir(
module {{ func.func @entry() {{
  wafer.tile.region() -> () {{
    %lhs = memref.alloc() : {0}
    %rhs = memref.alloc() : {1}
    %dest = memref.alloc() : {0}
    wafer.tile.elementwise_into <add> %lhs, %rhs into %dest
      {{indexing_maps = [affine_map<(b,m,n)->(b,m,n)>, affine_map<(b,m,n)->(b,n,m)>, affine_map<(b,m,n)->(b,m,n)>]}
      : {0}, {1} into {0}
    wafer.tile.yield
  }
  return
}})mlir",
                                  destType, sourceType)
                        .str();
        if (mode)
          text = llvm::formatv(R"mlir(module {{ func.func @entry() {{
          wafer.tile.region() -> () {{
            %rhs = memref.alloc() : {0}
            %result = wafer.tile.transpose %rhs {{permutation = array<i64: 0, 2, 1>}
              : {0} -> {1}
            wafer.tile.yield
          }
          return
        } })mlir",
                               sourceType, destType)
                     .str();
        auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(module) << text;
        wafer::TileRegionOp region;
        module->walk([&](wafer::TileRegionOp op) { region = op; });
        mlir::Value rhs;
        module->walk([&](wafer::ComputeElementwiseIntoOp op) {
          rhs = op.getInputs()[1];
        });
        module->walk([&](wafer::MoveTransposeOp op) { rhs = op.getSource(); });
        auto originalType = mlir::cast<mlir::MemRefType>(rhs.getType());
        auto originalOffsets =
            wafer::WaferStaticPhysicalOffsetCalculator::create(originalType);
        ASSERT_TRUE(originalOffsets);
        wafer::TileRegionToInstrLoweringSession session(context);
        ASSERT_TRUE(
            mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        mlir::Value copied;
        module->walk(
            [&](wafer::InstrElementwiseOp op) { copied = op.getInputs()[1]; });
        if (mode)
          module->walk(
              [&](wafer::InstrGatherScatterOp op) { copied = op.getDest(); });
        ASSERT_TRUE(copied && copied != rhs);
        auto target = mlir::cast<mlir::MemRefType>(copied.getType());
        int64_t width = dtype == "f32" ? 4 : 2;
        std::vector<int64_t> actual(2 * extent * 128 * width, -1);
        llvm::DenseMap<mlir::Value, int64_t> values;
        std::function<void(mlir::Operation *)> run = [&](mlir::Operation
                                                             *operation) {
          if (auto c = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(operation))
            values[c] = c.value();
          else if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(operation))
            values[add] = values.at(add.getLhs()) + values.at(add.getRhs());
          else if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(operation))
            values[mul] = values.at(mul.getLhs()) * values.at(mul.getRhs());
          else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
            for (int64_t i = values.at(loop.getLowerBound());
                 i < values.at(loop.getUpperBound());
                 i += values.at(loop.getStep())) {
              values[loop.getInductionVar()] = i;
              for (auto &nested : *loop.getBody())
                run(&nested);
            }
          } else if (auto move = mlir::dyn_cast<wafer::InstrGatherScatterOp>(
                         operation)) {
            EXPECT_EQ(move.getSource(), rhs);
            EXPECT_EQ(move.getDest(), copied);
            auto address = [&](bool source, int64_t ordinal) {
              auto fixed =
                  source ? move.getSrcOffsetAttr() : move.getDstOffsetAttr();
              auto dynamic =
                  source ? move.getSrcOffsetValue() : move.getDstOffsetValue();
              auto counts =
                  source ? move.getSrcIterations() : move.getDstIterations();
              auto strides =
                  source ? move.getSrcStrides() : move.getDstStrides();
              int64_t offset =
                  dynamic ? values.at(dynamic) : (fixed ? fixed.getInt() : 0);
              for (unsigned i = 0; i < 3; ++i) {
                offset += ordinal % counts[i] * strides[i];
                ordinal /= counts[i];
              }
              return offset;
            };
            for (uint64_t n = 0; n < move.getByteCount() / move.getInnerBytes();
                 ++n)
              for (uint64_t byte = 0; byte < move.getInnerBytes(); ++byte) {
                int64_t dst = address(false, n) + byte;
                ASSERT_GE(dst, 0);
                ASSERT_LT(dst, int64_t(actual.size()));
                ASSERT_EQ(actual[dst], -1);
                actual[dst] = address(true, n) + byte;
              }
          } else
            for (auto &r : operation->getRegions())
              for (auto &block : r)
                for (auto &nested : block)
                  run(&nested);
        };
        run(module->getOperation());
        for (int64_t b = 0; b < 2; ++b)
          for (int64_t m = 0; m < extent; ++m)
            for (int64_t n = 0; n < 128; ++n) {
              auto dst = wafer::computeWaferPhysicalElementByteOffset(
                  target, {b, m, n});
              ASSERT_TRUE(dst);
              auto src = originalOffsets->getByteOffset({b, n, m});
              ASSERT_TRUE(src);
              for (int64_t byte = 0; byte < width; ++byte)
                ASSERT_EQ(actual[*dst + byte], *src + byte)
                    << b << "/" << m << "/" << n;
            }
        EXPECT_EQ(std::count(actual.begin(), actual.end(), -1), 0);
      }
}

// Values and their reciprocals are exactly representable in all three formats.
// This oracle executes the emitted physical views/commands, then compares
// logical results; it does not use the broadcast proof's slice list.
TEST(LowerInstrToTargetLLVMTest, SplitBroadcastExecutesTailsAndBatchGaps) {
  for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (int64_t channels : {257, 263, 289})
        for (llvm::StringRef kind :
             {"sub", "add", "mul", "max", "min", "div"}) {
          // Sub spans the geometry matrix; the other arithmetic branches share
          // its address proof and exercise a real-size padded tail in each
          // dtype.
          if (kind != "sub" && (extent != 1031 || channels != 263))
            continue;
          SCOPED_TRACE(
              llvm::formatv("{0}/{1}/{2}/{3}", dtype, extent, channels, kind)
                  .str());
          mlir::DialectRegistry registry;
          registerTargetConversionDialects(registry);
          mlir::MLIRContext context(registry);
          context.loadAllAvailableDialects();
          auto text = llvm::formatv(R"mlir(module {{ func.func @main() {{
          wafer.tile.region() -> () {{
            %lhs = memref.alloc() : memref<2x{0}x{1}x{2}, #wafer.memory<spm, ncx>>
            %rhs = memref.alloc() : memref<2x{1}x{2}, #wafer.memory<spm, tensor>>
            wafer.tile.elementwise_into <{3}> %lhs, %rhs into %lhs
              {{indexing_maps = [affine_map<(b,r,c)->(b,r,c)>,
                  affine_map<(b,r,c)->(b,c)>, affine_map<(b,r,c)->(b,r,c)>]}
              : memref<2x{0}x{1}x{2}, #wafer.memory<spm, ncx>>,
                memref<2x{1}x{2}, #wafer.memory<spm, tensor>>
                into memref<2x{0}x{1}x{2}, #wafer.memory<spm, ncx>>
            wafer.tile.yield
          }
          return
        } })mlir",
                                    extent, channels, dtype, kind)
                          .str();
          auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
          ASSERT_TRUE(module) << text;
          wafer::TileRegionOp region;
          wafer::ComputeElementwiseIntoOp source;
          module->walk([&](wafer::TileRegionOp op) { region = op; });
          module->walk(
              [&](wafer::ComputeElementwiseIntoOp op) { source = op; });
          auto lhs = source.getInputs()[0], rhs = source.getInputs()[1];
          auto lhsType = mlir::cast<mlir::MemRefType>(lhs.getType());
          auto info = wafer::computeWaferPhysicalTensorInfo(lhsType);
          ASSERT_TRUE(info);
          const int64_t width = lhsType.getElementTypeBitWidth() / 8;
          llvm::DenseMap<mlir::Value, std::vector<float>> buffers;
          buffers[lhs] = std::vector<float>(info->physicalElements, 3);
          buffers[rhs].resize(2 * channels);
          for (int64_t i = 0; i < 2 * channels; ++i)
            buffers[rhs][i] = 1 << (i % 4);
          wafer::TileRegionToInstrLoweringSession session(context);
          ASSERT_TRUE(mlir::succeeded(
              wafer::convertTileRegionToInstr(region, session)));
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
          auto resolve = [&](mlir::Value value) {
            int64_t offset = 0;
            if (auto view =
                    value.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
              offset = view.getStaticOffsets()[0];
              value = view.getSource();
            }
            if (auto cast =
                    value.getDefiningOp<mlir::memref::MemorySpaceCastOp>())
              value = cast.getSource();
            return std::make_pair(value, offset);
          };
          unsigned groups = 0, tails = 0, copies = 0, reciprocals = 0;
          module->walk([&](mlir::Operation *op) {
            if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
              if (!buffers.count(alloc.getResult())) {
                EXPECT_LE(alloc.getType().getNumElements(),
                          kind == "div" ? 2 * channels : 64);
                buffers[alloc.getResult()].resize(
                    alloc.getType().getNumElements());
              }
            } else if (auto fill = mlir::dyn_cast<wafer::InstrFillOp>(op)) {
              auto [base, offset] = resolve(fill.getDest());
              std::fill(buffers[base].begin(), buffers[base].end(), 0);
            } else if (auto copy =
                           mlir::dyn_cast<wafer::InstrGatherScatterOp>(op)) {
              ++copies;
              EXPECT_LE(copy.getByteCount(), 64 * width);
              EXPECT_EQ(copy.getInnerBytes(), copy.getByteCount());
              auto [src, s] = resolve(copy.getSource());
              auto [dst, d] = resolve(copy.getDest());
              s += copy.getSrcOffset().value_or(0) / width;
              d += copy.getDstOffset().value_or(0) / width;
              std::copy_n(buffers[src].begin() + s, copy.getByteCount() / width,
                          buffers[dst].begin() + d);
            } else if (auto compute =
                           mlir::dyn_cast<wafer::InstrElementwiseOp>(op)) {
              auto [left, l] = resolve(compute.getInputs()[0]);
              auto [dest, d] = resolve(compute.getDest());
              int64_t count =
                  mlir::cast<mlir::MemRefType>(compute.getDest().getType())
                      .getNumElements();
              if (compute.getKind() == wafer::InstrElementwiseKind::Recip) {
                ++reciprocals;
                EXPECT_EQ(count, 2 * channels);
                for (int64_t i = 0; i < count; ++i)
                  buffers[dest][d + i] = 1.0f / buffers[left][l + i];
                return;
              }
              auto [right, r] = resolve(compute.getInputs()[1]);
              int64_t unit = compute.getRhsUnitElements();
              int64_t group = compute.getRhsGroupElements();
              if (group) {
                ++groups;
                EXPECT_EQ(unit, 64);
              } else {
                ++tails;
                EXPECT_LE(unit, 64);
              }
              for (int64_t i = 0; i < count; ++i) {
                int64_t ri = (group ? (i / group) * unit : 0) + i % unit;
                ASSERT_LT(r + ri, int64_t(buffers[right].size()));
                ASSERT_LT(d + i, int64_t(buffers[dest].size()));
                float a = buffers[left][l + i], b = buffers[right][r + ri];
                float value = 0;
                switch (compute.getKind()) {
                case wafer::InstrElementwiseKind::Sub:
                  value = a - b;
                  break;
                case wafer::InstrElementwiseKind::Add:
                  value = a + b;
                  break;
                case wafer::InstrElementwiseKind::Mul:
                  value = a * b;
                  break;
                case wafer::InstrElementwiseKind::Max:
                  value = std::max(a, b);
                  break;
                case wafer::InstrElementwiseKind::Min:
                  value = std::min(a, b);
                  break;
                default:
                  FAIL() << "Unexpected split arithmetic opcode";
                }
                buffers[dest][d + i] = value;
              }
            }
          });
          EXPECT_EQ(groups, 2u);
          EXPECT_EQ(tails, 2u);
          EXPECT_EQ(copies, 2u);
          EXPECT_EQ(reciprocals, kind == "div" ? 1u : 0u);
          for (int64_t b = 0; b < 2; ++b)
            for (int64_t row = 0; row < extent; ++row)
              for (int64_t c = 0; c < channels; ++c) {
                auto offset = wafer::computeWaferPhysicalElementByteOffset(
                    lhsType, {b, row, c});
                ASSERT_TRUE(offset);
                float right = 1 << ((b * channels + c) % 4);
                float expected = kind == "sub"   ? 3 - right
                                 : kind == "add" ? 3 + right
                                 : kind == "mul" ? 3 * right
                                 : kind == "max" ? std::max(3.0f, right)
                                 : kind == "min" ? std::min(3.0f, right)
                                                 : 3 / right;
                ASSERT_EQ(buffers[lhs][*offset / width], expected);
              }
          // Each actual allocation gets its own disjoint address. The
          // production allocator is exercised separately by the pipeline lit
          // test.
          int64_t address = 65536;
          module->walk([&](mlir::memref::AllocOp alloc) {
            alloc->setAttr("wafer.spm.offset",
                           wafer::SPMOffsetAttr::get(&context, address));
            address += 3145728;
          });
          mlir::PassManager manager(&context);
          manager.addPass(wafer::createLowerInstrToTargetLLVMPass({}));
          ASSERT_TRUE(mlir::succeeded(manager.run(*module)));
          EXPECT_EQ(countOps<mlir::memref::MemorySpaceCastOp>(*module), 0u);
        }
}

TEST(LowerInstrToTargetLLVMTest, SplitBroadcastRetainsAliasedRHSSnapshot) {
  for (int64_t extent : {1024, 1025, 1031}) {
    mlir::DialectRegistry registry;
    registerTargetConversionDialects(registry);
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    auto text = llvm::formatv(R"mlir(module {{ func.func @main() {{
      wafer.tile.region() -> () {{
        %lhs = memref.alloc() : memref<2x{0}x263xbf16, #wafer.memory<spm, ncx>>
        %physical = memref.memory_space_cast %lhs :
          memref<2x{0}x263xbf16, #wafer.memory<spm, ncx>> to
          memref<2x{0}x263xbf16, #wafer.memory<spm, tensor>>
        %rhs = memref.reinterpret_cast %physical to offset: [0], sizes: [2, 263], strides: [263, 1] :
          memref<2x{0}x263xbf16, #wafer.memory<spm, tensor>> to
          memref<2x263xbf16, #wafer.memory<spm, tensor>>
        wafer.tile.elementwise_into <sub> %lhs, %rhs into %lhs
          {{indexing_maps = [affine_map<(b,r,c)->(b,r,c)>,
              affine_map<(b,r,c)->(b,c)>, affine_map<(b,r,c)->(b,r,c)>]}
          : memref<2x{0}x263xbf16, #wafer.memory<spm, ncx>>,
            memref<2x263xbf16, #wafer.memory<spm, tensor>>
            into memref<2x{0}x263xbf16, #wafer.memory<spm, ncx>>
        wafer.tile.yield
      }
      return
    } })mlir",
                              extent)
                    .str();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(module) << text;
    wafer::TileRegionOp region;
    module->walk([&](wafer::TileRegionOp op) { region = op; });
    wafer::TileRegionToInstrLoweringSession session(context);
    ASSERT_TRUE(
        mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    unsigned computations = 0;
    module->walk([&](wafer::InstrElementwiseOp op) {
      ++computations;
      EXPECT_EQ(op.getRhsUnitElements(), 0);
      EXPECT_EQ(op.getRhsGroupElements(), 0);
      auto snapshot = op.getInputs()[1].getDefiningOp<mlir::memref::AllocOp>();
      ASSERT_TRUE(snapshot);
      EXPECT_EQ(
          snapshot.getType().getShape(),
          mlir::cast<mlir::MemRefType>(op.getDest().getType()).getShape());
      unsigned copies = 0;
      module->walk([&](wafer::InstrGatherScatterOp copy) {
        if (copy.getDest() == snapshot.getResult()) {
          ++copies;
          auto *owner = copy.getOperation();
          while (owner->getBlock() != op->getBlock())
            owner = owner->getParentOp();
          EXPECT_TRUE(owner->isBeforeInBlock(op));
        }
      });
      EXPECT_GT(copies, 0u);
    });
    EXPECT_EQ(computations, 1u);
  }
}

TEST(LowerInstrToTargetLLVMTest, GroupedBroadcastHasExplicitTargetGeometry) {
  for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (llvm::StringRef kind : {"add", "sub", "mul", "max", "min"}) {
        mlir::DialectRegistry registry;
        registerTargetConversionDialects(registry);
        mlir::MLIRContext context(registry);
        context.loadAllAvailableDialects();
        std::string full =
            llvm::formatv("memref<2x{0}x64x{1}, #wafer.memory<spm, tensor>>",
                          extent, dtype)
                .str();
        std::string rhs =
            llvm::formatv("memref<2x64x{0}, #wafer.memory<spm, tensor>>", dtype)
                .str();
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { func.func @entry() {\n"
            << "%lhs = memref.alloc() {wafer.spm.offset = "
               "#wafer.spm_offset<65536>} : "
            << full << "\n"
            << "%rhs = memref.alloc() {wafer.spm.offset = "
               "#wafer.spm_offset<655360>} : "
            << rhs << "\n"
            << "wafer.instr.elementwise <" << kind << "> %lhs, %rhs into %lhs "
            << "{rhs_unit_elements = 64 : i64, rhs_group_elements = "
            << extent * 64 << " : i64} : " << full << ", " << rhs << " into "
            << full << "\nreturn\n}}\n";
        auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(module) << text;
        wafer::InstrElementwiseOp elementwise;
        module->walk([&](wafer::InstrElementwiseOp op) { elementwise = op; });
        {
          mlir::ScopedDiagnosticHandler handler(&context,
                                                [](mlir::Diagnostic &) {});
          for (int64_t invalid : {-1, 63, 65, 2147483647}) {
            elementwise.setRhsGroupElements(invalid);
            EXPECT_TRUE(mlir::failed(mlir::verify(*module)));
          }
          elementwise.setRhsGroupElements(extent * 64);
          elementwise.setRhsUnitElements(32);
          EXPECT_TRUE(mlir::failed(mlir::verify(*module)));
          elementwise.setRhsUnitElements(64);
        }
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        mlir::PassManager manager(&context);
        manager.addPass(wafer::createLowerInstrToTargetLLVMPass({}));
        ASSERT_TRUE(mlir::succeeded(manager.run(*module)));
        unsigned calls = 0;
        module->walk([&](mlir::LLVM::CallOp call) {
          if (call.getCallee() != ("wafer_tx81_elementwise_" + kind).str())
            return;
          ++calls;
          ASSERT_EQ(call.getNumOperands(), 9u);
          const std::pair<unsigned, int64_t> fields[] = {
              {3, 2 * extent * 64}, {5, 64}, {6, 0}, {7, extent * 64}, {8, 0}};
          for (auto [index, expected] : fields) {
            auto constant =
                call.getOperand(index).getDefiningOp<mlir::LLVM::ConstantOp>();
            ASSERT_TRUE(constant);
            EXPECT_EQ(
                mlir::cast<mlir::IntegerAttr>(constant.getValue()).getInt(),
                expected);
          }
        });
        EXPECT_EQ(calls, 1u);
      }
}

TEST(LowerInstrToTargetLLVMTest, UnitBroadcastVerifierRejectsInvalidContracts) {
  mlir::DialectRegistry registry;
  registerTargetConversionDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @main() {
      %lhs = memref.alloc() : memref<2x1025x32xf32, #wafer.memory<spm, tensor>>
      %rhs = memref.alloc() : memref<32xf32, #wafer.memory<spm, tensor>>
      %dest = memref.alloc() : memref<2x1025x32xf32, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %lhs, %rhs into %dest {rhs_unit_elements = 32 : i64}
        : memref<2x1025x32xf32, #wafer.memory<spm, tensor>>,
          memref<32xf32, #wafer.memory<spm, tensor>>
        into memref<2x1025x32xf32, #wafer.memory<spm, tensor>>
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  wafer::InstrElementwiseOp operation;
  module->walk([&](wafer::InstrElementwiseOp op) { operation = op; });
  std::string diagnostic;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &value) { diagnostic += value.str(); });
  for (int64_t unit : {-1, 65, 31}) {
    operation.setRhsUnitElements(unit);
    diagnostic.clear();
    EXPECT_TRUE(mlir::failed(mlir::verify(*module)));
    EXPECT_NE(diagnostic.find("rhs_unit_elements"), std::string::npos);
  }
  operation.setRhsUnitElements(32);
  for (auto kind : {wafer::InstrElementwiseKind::Recip,
                    wafer::InstrElementwiseKind::LogicAnd}) {
    operation.setKind(kind);
    diagnostic.clear();
    EXPECT_TRUE(mlir::failed(mlir::verify(*module)));
    EXPECT_NE(diagnostic.find("floating binary"), std::string::npos);
  }
}

TEST(LowerInstrToTargetLLVMTest, NonperiodicRowBroadcastRetainsMovement) {
  for (int64_t extent : {1024, 1025, 1031}) {
    mlir::DialectRegistry registry;
    registerTargetConversionDialects(registry);
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    auto text = llvm::formatv(R"mlir(module {{
      func.func @main() {{
        %token = arith.constant false
        %unused = wafer.tile.region(%token : i1) -> (i1) {{
        ^bb0(%done: i1):
          %lhs = memref.alloc() : memref<2x32x{0}xf32, #wafer.memory<spm, tensor>>
          %rhs = memref.alloc() : memref<32xf32, #wafer.memory<spm, tensor>>
          %result = wafer.tile.elementwise <sub> %lhs, %rhs {{
            indexing_maps = [affine_map<(b,m,n)->(b,m,n)>,
                             affine_map<(b,m,n)->(m)>,
                             affine_map<(b,m,n)->(b,m,n)>]
          } : (memref<2x32x{0}xf32, #wafer.memory<spm, tensor>>,
               memref<32xf32, #wafer.memory<spm, tensor>>)
           -> memref<2x32x{0}xf32, #wafer.memory<spm, tensor>>
          wafer.tile.yield %done : i1
        }
        return
      }
    })mlir",
                              extent)
                    .str();
    auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(source);
    wafer::TileRegionOp region;
    source->walk([&](wafer::TileRegionOp op) { region = op; });
    wafer::TileRegionToInstrLoweringSession session(context);
    ASSERT_TRUE(
        mlir::succeeded(wafer::convertTileRegionToInstr(region, session)));
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
    EXPECT_GT(countOps<wafer::InstrGatherScatterOp>(*source), 0u);
    source->walk([&](wafer::InstrElementwiseOp op) {
      EXPECT_EQ(op.getRhsUnitElements(), 0);
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

TEST(LowerInstrToTargetLLVMTest,
     PreservesClampedRuntimeIndicesAndExactRankThreeByteAddresses) {
  for (int64_t rows : {1024, 1025, 1031})
    for (llvm::StringRef dtype : {"f16", "bf16"})
      for (unsigned width : {32u, 64u}) {
        SCOPED_TRACE(llvm::formatv("{0}:{1}:i{2}", rows, dtype, width).str());
        mlir::DialectRegistry registry;
        registerTargetConversionDialects(registry);
        mlir::MLIRContext context(registry);
        context.loadAllAvailableDialects();
        auto text = llvm::formatv(R"mlir(
          module {{
            func.func @read(%input: memref<2x{0}x64x{1}, #wafer.memory<ddr, tensor>>) {{
              %zero = arith.constant 0 : index
              %one = arith.constant 1 : index
              %end = arith.constant {0} : index
              scf.for %position = %zero to %end step %one {{
              %token = arith.index_cast %position : index to i{2}
              %lo = arith.constant 0 : i{2}
              %hi = arith.constant {3} : i{2}
              %lower = arith.maxsi %token, %lo : i{2}
              %clamp = arith.minsi %lower, %hi : i{2}
              %row = arith.index_cast %clamp : i{2} to index
              %view = memref.subview %input[1, %row, 8] [1, 1, 16] [1, 1, 1]
                  : memref<2x{0}x64x{1}, #wafer.memory<ddr, tensor>>
                 to memref<1x1x16x{1}, strided<[{4}, 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>
              %local = memref.alloc() {{wafer.spm.offset = #wafer.spm_offset<65536>}
                  : memref<1x1x16x{1}, #wafer.memory<spm, tensor>>
              wafer.instr.rdma %view to %local
                  {{byte_count = 32 : i64, inner_bytes = 32 : i64,
                    src_strides = array<i64: 0, 0, 0>, src_iterations = array<i64: 1, 1, 1>}
                  : memref<1x1x16x{1}, strided<[{4}, 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>
                 to memref<1x1x16x{1}, #wafer.memory<spm, tensor>>
              wafer.instr.ncc_join [0]
              }
              return
            }
          })mlir",
                                  rows, dtype, width, rows - 1, rows * 64)
                        .str();
        auto source = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
        ASSERT_TRUE(source) << text;
        mlir::PassManager manager(&context);
        manager.addPass(wafer::createLowerInstrToTargetLLVMPass({}));
        ASSERT_TRUE(mlir::succeeded(manager.run(*source)));
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
        EXPECT_EQ(countOps<mlir::memref::SubViewOp>(*source), 0u);
        EXPECT_EQ(countOps<mlir::LLVM::SMaxOp>(*source), 1u);
        EXPECT_EQ(countOps<mlir::LLVM::SMinOp>(*source), 1u);
        auto function = source->lookupSymbol<mlir::LLVM::LLVMFuncOp>("read");
        ASSERT_TRUE(function);
        mlir::LLVM::CallOp rdma;
        source->walk([&](mlir::LLVM::CallOp call) {
          if (call.getCallee() == "wafer_tx81_rdma")
            rdma = call;
        });
        ASSERT_TRUE(rdma);
        auto address = rdma.getOperand(0).getDefiningOp<mlir::LLVM::AddOp>();
        ASSERT_TRUE(address);
        auto base = address.getLhs().getDefiningOp<mlir::LLVM::AddOp>();
        auto delta = address.getRhs().getDefiningOp<mlir::LLVM::MulOp>();
        ASSERT_TRUE(base);
        ASSERT_TRUE(delta);
        EXPECT_EQ(base.getLhs(), function.getArgument(0));
        auto constant = [](mlir::Value value) -> std::optional<int64_t> {
          auto op = value.getDefiningOp<mlir::LLVM::ConstantOp>();
          if (op)
            if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(op.getValue()))
              return attr.getInt();
          return std::nullopt;
        };
        EXPECT_EQ(constant(base.getRhs()), rows * 128 + 16);
        EXPECT_EQ(constant(delta.getRhs()), 128);
        mlir::Value row = delta.getLhs();
        if (width == 32) {
          auto extend = row.getDefiningOp<mlir::LLVM::SExtOp>();
          ASSERT_TRUE(extend);
          row = extend.getArg();
        }
        auto clamp = row.getDefiningOp<mlir::LLVM::SMinOp>();
        ASSERT_TRUE(clamp);
        auto lower = clamp.getOperand(0).getDefiningOp<mlir::LLVM::SMaxOp>();
        ASSERT_TRUE(lower);
        mlir::Value token = lower.getOperand(0);
        if (width == 32) {
          auto truncate = token.getDefiningOp<mlir::LLVM::TruncOp>();
          ASSERT_TRUE(truncate);
          token = truncate.getArg();
        }
        EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(token));
        EXPECT_NE(token.getParentBlock(), &function.getBody().front());
        EXPECT_EQ(constant(lower.getOperand(1)), 0);
        EXPECT_EQ(constant(clamp.getOperand(1)), rows - 1);
      }
}

TEST(LowerInstrToTargetLLVMTest,
     RejectsOutOfBoundsRuntimeClampWithoutMutatingInput) {
  for (int64_t rows : {1024, 1025, 1031}) {
    mlir::DialectRegistry registry;
    registerTargetConversionDialects(registry);
    mlir::MLIRContext context(registry);
    context.loadAllAvailableDialects();
    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::formatv(R"mlir(
        module {{ func.func @outside(%input: memref<2x{0}x64xf16, #wafer.memory<ddr, tensor>>) {{
          %zero = arith.constant 0 : index
          %one = arith.constant 1 : index
          %end = arith.constant {2} : index
          scf.for %position = %zero to %end step %one {{
          %x = arith.index_cast %position : index to i32
          %lo = arith.constant 0 : i32
          %hi = arith.constant {0} : i32
          %lower = arith.maxsi %x, %lo : i32
          %clamp = arith.minsi %lower, %hi : i32
          %row = arith.index_cast %clamp : i32 to index
          %view = memref.subview %input[1, %row, 8] [1, 1, 16] [1, 1, 1]
              : memref<2x{0}x64xf16, #wafer.memory<ddr, tensor>>
             to memref<1x1x16xf16, strided<[{1}, 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          }
          return
        } })mlir",
                      rows, rows * 64, rows + 1)
            .str(),
        &context);
    ASSERT_TRUE(source);
    auto print = [&]() {
      std::string text;
      llvm::raw_string_ostream stream(text);
      source->print(stream);
      return text;
    };
    std::string before = print();
    std::string diagnostics;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &d) {
      llvm::raw_string_ostream stream(diagnostics);
      d.print(stream);
      return mlir::success();
    });
    mlir::PassManager manager(&context);
    manager.addPass(wafer::createLowerInstrToTargetLLVMPass({}));
    EXPECT_TRUE(mlir::failed(manager.run(*source)));
    EXPECT_NE(diagnostics.find(llvm::formatv(
                                  "target_geometry_mismatch: dynamic tensor "
                                  "subview dimension #1 may access source "
                                  "coordinate {0} outside static extent {0}",
                                  rows)
                                  .str()),
              std::string::npos)
        << diagnostics;
    EXPECT_EQ(before, print());
  }
}

} // namespace
