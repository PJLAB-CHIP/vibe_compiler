//===- SystemCTargetModelDTEIntegrationTest.cpp - SystemC DTE gate -------===//

#include "SystemCTargetModelTestSupport.h"

#include "Wafer/Model/SystemCTargetModel.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

TEST(SystemCTargetModelDTEIntegrationTest,
     ExecutesDTEPrepareThenNCCJoinThenIssueAndExactDTEWait) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModuleBundle> bundle =
      test::buildDirectDTETargetBundle(
          diagnostics, TargetProfileId::waferTx81SingleCardKernelV3());
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());
  llvm::Expected<test::NCCJoinRewriteResult> rewrite =
      test::rewriteNCCJoinsAfter(*bundle,
                                 TargetCallBuiltin::DirectDTESendPrepare);
  ASSERT_TRUE(static_cast<bool>(rewrite))
      << llvm::toString(rewrite.takeError());
  EXPECT_GT(rewrite->erasedJoinCount, 0u);
  EXPECT_GT(rewrite->insertedJoinCount, 0u);
  EXPECT_GT(rewrite->insertedTerminalJoinCount, 0u);
  llvm::Expected<test::DirectDTEInvocationData> invocation =
      test::buildDirectDTEInvocationData(*bundle);
  ASSERT_TRUE(static_cast<bool>(invocation))
      << llvm::toString(invocation.takeError());

  llvm::Expected<TargetCallExecutable> frontend =
      prepareTargetCallFrontend(*bundle, invocation->arguments);
  ASSERT_TRUE(static_cast<bool>(frontend))
      << llvm::toString(frontend.takeError());
  llvm::Expected<TargetModelResult> result = executeSystemCTargetModel(
      std::move(*frontend), invocation->inputBindings,
      TargetModelKernelBudget::create(FormalNumericWorkBudget::create(
                                          /*maximumScalarEvaluations=*/1024,
                                          /*maximumFusedMultiplyAdds=*/1024),
                                      /*maximumMovementBytes=*/4096,
                                      /*maximumMovementSegments=*/1024));
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->completedRankCount, 16);
  EXPECT_GE(result->issuedTransactionCount, 16u * 8u);
  EXPECT_GE(result->systemCThreadProcessCount, 17u);
  EXPECT_GT(result->finalDeltaCount, 0u);
  EXPECT_EQ(result->schedulerIdentity, "untimed-delta-worker-aware-ncc-v2");
  ASSERT_EQ(result->outputs.size(), 16u);
  for (const TargetModelOutput &output : result->outputs) {
    ASSERT_GE(output.logicalRank, 0);
    ASSERT_LT(output.logicalRank, 16);
    const int64_t peer = output.logicalRank ^ 1;
    EXPECT_EQ(output.bytes,
              invocation->inputBytesByRank[static_cast<size_t>(peer)]);
  }
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
