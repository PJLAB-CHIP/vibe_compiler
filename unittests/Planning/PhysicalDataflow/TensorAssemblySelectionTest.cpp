//===- TensorAssemblySelectionTest.cpp ---------------------------------===//
#include "Wafer/Planning/PhysicalDataflow/TensorAssemblySelection.h"

#include "Wafer/InitWaferDialects.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

namespace {
using namespace wafer;
using namespace wafer::compiler::detail;

class TensorAssemblySelectionTest : public ::testing::Test {
protected:
  TensorAssemblySelectionTest() {
    mlir::DialectRegistry registry;
    registerWaferCoreDialects(registry);
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }
  mlir::OwningOpRef<mlir::ModuleOp> parse(int64_t extent) {
    // These are choice-correspondence tests. Materialization and numerical
    // coverage use the actual Temporal/Instr tests, not these unused reads.
    std::string text = R"mlir(module {
      func.func @entry(%input: tensor<2xEXTENTx64xf16>) {
        %a = tensor.cast %input : tensor<2xEXTENTx64xf16> to tensor<2xEXTENTx64xf16>
        %b = tensor.cast %input : tensor<2xEXTENTx64xf16> to tensor<2xEXTENTx64xf16>
        %r0 = tensor.extract_slice %a[0, 0, 0] [2, EXTENT, 16] [1, 1, 1]
          : tensor<2xEXTENTx64xf16> to tensor<2xEXTENTx16xf16>
        %r1 = tensor.extract_slice %a[0, 0, 8] [2, EXTENT, 16] [1, 1, 1]
          : tensor<2xEXTENTx64xf16> to tensor<2xEXTENTx16xf16>
        %r2 = tensor.extract_slice %b[0, 0, 0] [2, EXTENT, 16] [1, 1, 1]
          : tensor<2xEXTENTx64xf16> to tensor<2xEXTENTx16xf16>
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %cend = arith.constant EXTENT : index
        scf.for %iv = %c0 to %cend step %c1 {
          %r3 = tensor.extract_slice %a[0, %iv, 0] [2, 1, 16] [1, 1, 1]
            : tensor<2xEXTENTx64xf16> to tensor<2x1x16xf16>
        }
        scf.for %iv = %c0 to %cend step %c16 {
          %r4 = tensor.extract_slice %a[0, %iv, 8] [2, 1, 16] [1, 1, 1]
            : tensor<2xEXTENTx64xf16> to tensor<2x1x16xf16>
        }
        scf.for %iv = %c0 to %cend step %c1 {
          %r5 = tensor.extract_slice %b[0, %iv, 0] [2, 1, 16] [1, 1, 1]
            : tensor<2xEXTENTx64xf16> to tensor<2x1x16xf16>
        }
        return
      }
    })mlir";
    for (size_t at; (at = text.find("EXTENT")) != std::string::npos;)
      text.replace(at, 6, std::to_string(extent));
    return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  }
  llvm::SmallVector<mlir::tensor::ExtractSliceOp> reads(mlir::ModuleOp module) {
    llvm::SmallVector<mlir::tensor::ExtractSliceOp> result;
    module.walk(
        [&](mlir::tensor::ExtractSliceOp read) { result.push_back(read); });
    return result;
  }
  mlir::MLIRContext context;
};

TEST_F(TensorAssemblySelectionTest, SourceAndReadScopesSurviveExactClone) {
  for (int64_t extent : {1024, 1025, 1031}) {
    auto parent = parse(extent);
    ASSERT_TRUE(parent);
    auto parentReads = reads(*parent);
    ASSERT_EQ(parentReads.size(), 6u);
    auto identity = mlir::DistinctAttr::create(mlir::UnitAttr::get(&context));
    auto scope = IterationCoordinatesAttr::get(
        &context, {IterationCoordinateAttr::get(&context, identity, 1)});
    for (auto read : llvm::ArrayRef(parentReads).drop_front(3))
      read->getParentOp()->setAttr(kIterationCoordinatesAttrName, scope);
    auto original = TensorChoiceBindings::capture(parent->getOperation());
    auto expected = groupTensorAssemblyReads(parentReads, original);
    ASSERT_TRUE(expected.unavailable.empty());
    ASSERT_EQ(expected.families.size(), 5u);
    EXPECT_EQ(expected.families[3].reads.size(), 2u);
    EXPECT_FALSE(expected.families[3].selection ==
                 expected.families[4].selection);
    EXPECT_FALSE(expected.families[0].selection ==
                 expected.families[1].selection);
    for (unsigned repetition = 0; repetition < 2; ++repetition) {
      mlir::IRMapping mapping;
      auto candidate = mlir::OwningOpRef<mlir::ModuleOp>(
          mlir::cast<mlir::ModuleOp>(parent->getOperation()->clone(mapping)));
      auto bound = original.clone(mapping);
      ASSERT_TRUE(mlir::succeeded(bound));
      auto actual = groupTensorAssemblyReads(reads(*candidate), *bound);
      ASSERT_TRUE(actual.unavailable.empty());
      ASSERT_EQ(actual.families.size(), expected.families.size());
      for (const auto &family : expected.families) {
        auto found = llvm::find_if(actual.families, [&](const auto &current) {
          return current.selection == family.selection;
        });
        ASSERT_NE(found, actual.families.end());
        for (auto read : family.reads)
          EXPECT_TRUE(llvm::is_contained(
              found->reads, mlir::cast<mlir::tensor::ExtractSliceOp>(
                                mapping.lookup(read.getOperation()))));
      }
      TensorAssemblyIntent a{
          {expected.families[0].selection, expected.families[4].selection}};
      TensorAssemblyIntent b{
          {expected.families[4].selection, expected.families[0].selection}};
      EXPECT_EQ(a, b);
      b.selections.pop_back();
      EXPECT_FALSE(a == b);
      TensorAssemblyIntent shared;
      auto choices = extendTensorAssemblyIntent(shared, a.selections);
      ASSERT_EQ(choices.size(), 3u);
      EXPECT_EQ(choices[0], a);
      EXPECT_EQ(
          choices[1].selections,
          (llvm::SmallVector<TensorAssemblySelection, 4>{a.selections[0]}));
      EXPECT_EQ(
          choices[2].selections,
          (llvm::SmallVector<TensorAssemblySelection, 4>{a.selections[1]}));
      auto extended = extendTensorAssemblyIntent(choices[1], a.selections);
      ASSERT_EQ(extended.size(), 1u);
      EXPECT_EQ(extended.front(), a);
      EXPECT_TRUE(extendTensorAssemblyIntent(a, a.selections).empty());
      EXPECT_TRUE(extendTensorAssemblyIntent(a, b.selections).empty());
    }
  }
}

TEST_F(TensorAssemblySelectionTest,
       ReplacementMergeAndErasureInvalidateExactly) {
  auto parent = parse(1031);
  ASSERT_TRUE(parent);
  auto original = TensorChoiceBindings::capture(parent->getOperation());
  mlir::IRMapping mapping;
  auto candidate = mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::cast<mlir::ModuleOp>(parent->getOperation()->clone(mapping)));
  auto bound = original.clone(mapping);
  ASSERT_TRUE(mlir::succeeded(bound));
  auto currentReads = reads(*candidate);
  auto sourceA = currentReads[0].getSource();
  auto sourceB = currentReads[2].getSource();
  auto anchorA = bound->getAnchors(sourceA).front();
  auto anchorB = bound->getAnchors(sourceB).front();
  mlir::IRRewriter rewriter(&context, &*bound);
  rewriter.replaceOp(sourceB.getDefiningOp(), sourceA.getDefiningOp());
  EXPECT_EQ(bound->getAnchors(sourceA).size(), 2u);
  EXPECT_TRUE(llvm::is_contained(bound->getAnchors(sourceA), anchorA));
  EXPECT_TRUE(llvm::is_contained(bound->getAnchors(sourceA), anchorB));
  rewriter.eraseOp(currentReads[3]->getParentOp());
  rewriter.eraseOp(currentReads[0]);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*candidate)));
  mlir::IRMapping nextMapping;
  auto next = mlir::OwningOpRef<mlir::ModuleOp>(mlir::cast<mlir::ModuleOp>(
      candidate->getOperation()->clone(nextMapping)));
  // A stale result or deleted induction scope would make cloning fail.
  auto nextBindings = bound->clone(nextMapping);
  ASSERT_TRUE(mlir::succeeded(nextBindings));
  EXPECT_EQ(nextBindings->getAnchors(nextMapping.lookup(sourceA)).size(), 2u);
  auto emptyBindings = TensorChoiceBindings{};
  auto missing = captureTensorAssemblyRead(reads(*next).front(), emptyBindings);
  EXPECT_TRUE(
      std::holds_alternative<TensorAssemblyBindingUnavailable>(missing));
}
} // namespace
