//===- TensorProgramAlternativeTest.cpp --------------------------------===//

#include "Wafer/Compiler/Search/TensorProgramAlternative.h"

#include "Wafer/Compiler/Pipeline/CompilationInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using wafer::compiler::detail::TensorProgramAlternativeAssignment;
using wafer::compiler::detail::TensorProgramAlternativeKind;
using wafer::compiler::detail::TensorProgramAlternativeMaterializationKind;

class TensorProgramAlternativeTest : public ::testing::Test {
protected:
  TensorProgramAlternativeTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseDecodeProgram() {
    std::string path = std::string(WAFER_TEST_SOURCE_DIR) +
                       "/unittests/Compiler/Search/Inputs/functional-decode.mlir";
    return mlir::parseSourceFile<mlir::ModuleOp>(
        path, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

template <typename OpTy> static size_t countOps(mlir::Operation *root) {
  size_t count = 0;
  root->walk([&](OpTy) { ++count; });
  return count;
}

static bool hasTensorResultShape(mlir::Operation *root,
                                 llvm::ArrayRef<int64_t> shape) {
  bool found = false;
  root->walk([&](mlir::Operation *operation) {
    for (mlir::Value result : operation->getResults()) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
      found |= type && type.hasStaticShape() &&
               llvm::equal(type.getShape(), shape);
    }
  });
  return found;
}

static std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

TEST_F(TensorProgramAlternativeTest,
       DecodeDomainIsCompleteCompactAndDeterministic) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseDecodeProgram();
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  auto domain = wafer::compiler::detail::getTensorProgramAlternativeDomain(
      *source);
  ASSERT_TRUE(mlir::succeeded(domain));
  EXPECT_EQ(domain->getReductionExtent(), 7);
  EXPECT_TRUE(domain->supportsSplitKeyValue());
  auto count = domain->getAssignmentCount();
  ASSERT_TRUE(mlir::succeeded(count));
  EXPECT_EQ(*count, 22u);

  std::vector<TensorProgramAlternativeAssignment> assignments;
  std::optional<TensorProgramAlternativeAssignment> current =
      domain->getFirstAssignment();
  while (current) {
    assignments.push_back(*current);
    auto next = domain->getNextAssignment(*current);
    ASSERT_TRUE(mlir::succeeded(next));
    current = *next;
  }
  EXPECT_EQ(assignments.size(), *count);
  EXPECT_TRUE(std::is_sorted(assignments.begin(), assignments.end()));
  EXPECT_EQ(std::set<TensorProgramAlternativeAssignment>(assignments.begin(),
                                                         assignments.end())
                .size(),
            assignments.size());
  EXPECT_TRUE(domain->contains(
      TensorProgramAlternativeAssignment::splitKeyValueAttention(4, 2)));
  EXPECT_FALSE(domain->contains(
      TensorProgramAlternativeAssignment::splitKeyValueAttention(5, 2)));
}

TEST_F(TensorProgramAlternativeTest,
       MaterializesPureOnlineAndSplitRootsWithoutChangingSource) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseDecodeProgram();
  ASSERT_TRUE(source);
  std::string sourceBefore = printModule(*source);
  mlir::func::FuncOp sourceFunction =
      source->lookupSymbol<mlir::func::FuncOp>("functional_decode");
  ASSERT_TRUE(sourceFunction);
  mlir::FunctionType sourceType = sourceFunction.getFunctionType();

  for (TensorProgramAlternativeAssignment assignment :
       {TensorProgramAlternativeAssignment::onlineAttention(3),
        TensorProgramAlternativeAssignment::splitKeyValueAttention(3, 2)}) {
    auto result = wafer::compiler::detail::materializeTensorProgramAlternative(
        *source, assignment);
    ASSERT_EQ(result.getKind(),
              TensorProgramAlternativeMaterializationKind::Materialized)
        << result.getDiagnostic().str();
    mlir::OwningOpRef<mlir::ModuleOp> alternative = result.takeModule();
    ASSERT_TRUE(alternative);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*alternative)));
    mlir::func::FuncOp function =
        alternative->lookupSymbol<mlir::func::FuncOp>("functional_decode");
    ASSERT_TRUE(function);
    EXPECT_EQ(function.getFunctionType(), sourceType);
    EXPECT_FALSE(hasTensorResultShape(*alternative, {2, 7}));
    EXPECT_EQ(countOps<mlir::memref::AllocOp>(*alternative), 0u);
    EXPECT_EQ(countOps<wafer::TileRegionOp>(*alternative), 0u);
    EXPECT_GT(countOps<mlir::linalg::GenericOp>(*alternative), 0u);
    if (assignment.kind ==
        TensorProgramAlternativeKind::SplitKeyValueAttention)
      EXPECT_EQ(countOps<mlir::math::LogOp>(*alternative), 2u);
  }
  EXPECT_EQ(printModule(*source), sourceBefore);
}

TEST_F(TensorProgramAlternativeTest,
       MaterializesOriginalAsAnIndependentUnchangedRoot) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseDecodeProgram();
  ASSERT_TRUE(source);
  std::string sourceBefore = printModule(*source);

  auto result = wafer::compiler::detail::materializeTensorProgramAlternative(
      *source, TensorProgramAlternativeAssignment::original());
  ASSERT_EQ(result.getKind(),
            TensorProgramAlternativeMaterializationKind::Materialized);
  mlir::OwningOpRef<mlir::ModuleOp> original = result.takeModule();
  ASSERT_TRUE(original);
  EXPECT_EQ(printModule(*original), sourceBefore);
  EXPECT_EQ(printModule(*source), sourceBefore);
}

TEST_F(TensorProgramAlternativeTest,
       PrecomputedScoresAreNotAdvertisedAsAnOnlineAlgorithm) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseDecodeProgram();
  ASSERT_TRUE(source);
  mlir::linalg::MatmulOp score;
  source->walk([&](mlir::linalg::MatmulOp operation) { score = operation; });
  ASSERT_TRUE(score);
  auto init = score.getDpsInits().front().getDefiningOp<mlir::linalg::FillOp>();
  ASSERT_TRUE(init);
  mlir::Value precomputed = init.getDpsInits().front();
  score.getResult(0).replaceAllUsesWith(precomputed);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));

  auto domain = wafer::compiler::detail::getTensorProgramAlternativeDomain(
      *source);
  ASSERT_TRUE(mlir::succeeded(domain));
  auto count = domain->getAssignmentCount();
  ASSERT_TRUE(mlir::succeeded(count));
  EXPECT_EQ(*count, 1u);
  EXPECT_EQ(domain->getFirstAssignment(),
            TensorProgramAlternativeAssignment::original());
}

TEST_F(TensorProgramAlternativeTest,
       InvalidAssignmentIsTypedAndDoesNotCreateAnActualRoot) {
  mlir::OwningOpRef<mlir::ModuleOp> source = parseDecodeProgram();
  ASSERT_TRUE(source);
  std::string sourceBefore = printModule(*source);
  auto result = wafer::compiler::detail::materializeTensorProgramAlternative(
      *source,
      TensorProgramAlternativeAssignment::splitKeyValueAttention(5, 2));
  EXPECT_EQ(result.getKind(),
            TensorProgramAlternativeMaterializationKind::InvalidAssignment);
  EXPECT_EQ(printModule(*source), sourceBefore);
}

} // namespace
