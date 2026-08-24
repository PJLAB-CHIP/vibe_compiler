//===- UnifiedPhysicalDataflowTest.cpp -------------------------------===//

#include "Wafer/Planning/Search/UnifiedPhysicalDataflow.h"

#include "Wafer/CodeGen/Executable/CardExecutableCompilation.h"

#include "TestSupport/CodeGen/CardExecutableTestSupport.h"

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;
using namespace wafer::compiler::testing;

template <typename OpT> unsigned countOps(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](OpT) { ++count; });
  return count;
}

TEST(UnifiedPhysicalDataflowTest,
     FirstJointAssignmentMaterializesAndPassesTheCompleteExecutableGate) {
  ParsedProgram parsed = parseDependentProgram();
  ASSERT_TRUE(parsed.module);
  std::string sourceBefore;
  llvm::raw_string_ostream sourceStream(sourceBefore);
  parsed.module->print(sourceStream);
  sourceStream.flush();

  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  wafer::compiler::ProgramDataHandoff programData;
  auto analysis = analyzeCardProgram(*parsed.module, dependentProgramMetadata(),
                                     executionConfig(), diagnostics);
  ASSERT_TRUE(mlir::succeeded(analysis)) << diagnosticsText;
  auto domain = UnifiedPhysicalDataflowDomain::create(**analysis, CardId(0));
  ASSERT_TRUE(mlir::succeeded(domain));
  std::string failureReason;
  auto first = domain->getFirstAssignment(&failureReason);
  ASSERT_TRUE(mlir::succeeded(first)) << failureReason;
  EXPECT_TRUE(domain->contains(*first));
  auto next = domain->getNextAssignment(*first, &failureReason);
  ASSERT_TRUE(mlir::succeeded(next)) << failureReason;
  EXPECT_TRUE(*next);
  EXPECT_TRUE(domain->contains(**next));

  auto materialized =
      domain->materialize(*parsed.module, *first, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<TileRegionOp>(materialized->module->getOperation()), 2u);
  CardExecutablePreparation preparation;
  auto compiled = compileCardModuleToExecutable(
      std::move(materialized->module), CardId(0), (*analysis)->availableTileIds,
      materialized->relations, preparation, dependentProgramMetadata(),
      executionConfig(), diagnostics, programData, /*statistics=*/nullptr,
      /*tilePipelineParallelism=*/0, /*captureTileIRTrace=*/false);
  diagnostics.flush();
  ASSERT_TRUE(compiled.isAccepted())
      << compiled.gate << ": " << compiled.detail << "\n"
      << diagnosticsText;
  EXPECT_EQ(compiled.executable->tiles.size(), 16u);

  std::string sourceAfter;
  llvm::raw_string_ostream afterStream(sourceAfter);
  parsed.module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(sourceAfter, sourceBefore);
}

} // namespace
