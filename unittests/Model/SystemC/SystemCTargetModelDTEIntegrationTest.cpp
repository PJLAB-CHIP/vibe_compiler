//===- SystemCTargetModelDTEIntegrationTest.cpp - SystemC DTE gate -------===//

#include "SystemCTargetModelTestSupport.h"

#include "Wafer/Model/SystemC/SystemCTargetModel.h"

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
  llvm::Expected<TargetLLVMModules> targetLLVMModules =
      test::compileDirectDTETargetModules(
          diagnostics, TargetIdentityId::waferTx81SingleCard());
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  llvm::Expected<test::NCCJoinRewriteResult> rewrite =
      test::rewriteNCCJoinsAfter(*targetLLVMModules,
                                 TargetCallBuiltin::DirectDTESendPrepare);
  ASSERT_TRUE(static_cast<bool>(rewrite))
      << llvm::toString(rewrite.takeError());
  EXPECT_GT(rewrite->erasedJoinCount, 0u);
  EXPECT_GT(rewrite->insertedJoinCount, 0u);
  EXPECT_GT(rewrite->insertedTerminalJoinCount, 0u);
  llvm::Expected<test::DirectDTEInvocationData> invocation =
      test::buildDirectDTEInvocationData(*targetLLVMModules);
  ASSERT_TRUE(static_cast<bool>(invocation))
      << llvm::toString(invocation.takeError());

  llvm::Expected<TargetCallExecutable> frontend =
      createTargetCallExecutable(*targetLLVMModules, invocation->arguments);
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
  EXPECT_EQ(result->completedTileCount, 16);
  EXPECT_GE(result->issuedCommandCount, 16u * 8u);
  EXPECT_GE(result->systemCThreadProcessCount, 17u);
  EXPECT_GT(result->finalDeltaCount, 0u);
  EXPECT_EQ(result->schedulerIdentity, "untimed-delta-worker-aware-ncc");
  ASSERT_EQ(result->outputs.size(), 1u);
  EXPECT_EQ(result->outputs.front().bytes, invocation->expectedOutputBytes);
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
