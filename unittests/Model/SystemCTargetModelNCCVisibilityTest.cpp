//===- SystemCTargetModelNCCVisibilityTest.cpp - NCC visibility ----------===//

#include "SystemCTargetModelTestSupport.h"

#include "Wafer/Model/SystemCTargetModel.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <string>
#include <utility>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

TEST(SystemCTargetModelNCCVisibilityTest,
     RejectsLegacyDTEAutoIssueWhenNCCJoinIsAfterTheWait) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModules> targetLLVMModules =
      test::compileDirectDTETargetModules(
          diagnostics, TargetIdentityId::waferTx81SingleCard());
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());

  // Remove production joins and put each replacement after its DTE wait. A
  // participant join after issue cannot retroactively order the transfer, so
  // the issue-time visibility gate must reject the pending NCC source write
  // before making peer readiness visible.
  llvm::Expected<test::NCCJoinRewriteResult> rewrite =
      test::rewriteNCCJoinsAfter(*targetLLVMModules,
                                 TargetCallBuiltin::DirectDTEWait);
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
  ASSERT_FALSE(static_cast<bool>(result));
  const std::string error = llvm::toString(result.takeError());
  EXPECT_NE(error.find("dte-issue-order"), std::string::npos) << error;
  EXPECT_NE(error.find("pending NCC source write"), std::string::npos) << error;
  EXPECT_NE(
      error.find("matching participant join must precede the Direct DTE issue"),
      std::string::npos)
      << error;
  EXPECT_EQ(error.find("no-progress"), std::string::npos) << error;
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
