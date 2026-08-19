//===- SystemCTargetModelDTELateJoinTest.cpp - DTE issue ordering --------===//

#include "SystemCTargetModelTestSupport.h"

#include "Wafer/Model/SystemC/SystemCTargetModel.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>

#ifndef WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE
#error "WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE must select the test scenario"
#endif

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

constexpr test::LateJoinDTEAccessMode getAccessMode() {
#if WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE == 0
  return test::LateJoinDTEAccessMode::SourceWrite;
#elif WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE == 1
  return test::LateJoinDTEAccessMode::DestinationRead;
#elif WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE == 2
  return test::LateJoinDTEAccessMode::DestinationWrite;
#else
#error "unknown WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE"
#endif
}

constexpr const char *getExpectedConflict() {
#if WAFER_TEST_DTE_LATE_JOIN_ACCESS_MODE == 0
  return "pending NCC source write";
#else
  return "pending NCC destination read/write";
#endif
}

TEST(SystemCTargetModelDTELateJoinTest,
     ParticipantJoinAfterIssueCannotRetroactivelyOrderTransfer) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModules> targetLLVMModules =
      test::compileDirectDTETargetModules(
          diagnostics, TargetIdentityId::waferTx81SingleCard());
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());

  llvm::Expected<test::LateJoinDTERewriteResult> rewrite =
      test::insertPendingComputeWithLateJoin(*targetLLVMModules,
                                             getAccessMode());
  ASSERT_TRUE(static_cast<bool>(rewrite))
      << llvm::toString(rewrite.takeError());
  EXPECT_EQ(rewrite->insertedComputeCount, 16u);
  EXPECT_EQ(rewrite->insertedSetupJoinCount, 16u);
  if (getAccessMode() == test::LateJoinDTEAccessMode::SourceWrite)
    EXPECT_EQ(rewrite->removedPreIssueJoinCount, 0u);
  else
    EXPECT_GT(rewrite->removedPreIssueJoinCount, 0u);
  EXPECT_EQ(rewrite->insertedLateJoinCount, 16u);

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
                                          /*maximumScalarEvaluations=*/4096,
                                          /*maximumFusedMultiplyAdds=*/4096),
                                      /*maximumMovementBytes=*/8192,
                                      /*maximumMovementSegments=*/2048));
  ASSERT_FALSE(static_cast<bool>(result));
  const std::string error = llvm::toString(result.takeError());
  EXPECT_NE(error.find("dte-issue-order"), std::string::npos) << error;
  EXPECT_NE(error.find(getExpectedConflict()), std::string::npos) << error;
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
