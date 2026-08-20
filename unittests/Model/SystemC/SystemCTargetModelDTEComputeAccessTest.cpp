//===- SystemCTargetModelDTEComputeAccessTest.cpp - NCC/DTE ranges ------===//

#include "SystemCTargetModelTestSupport.h"

#include "Wafer/Model/SystemC/SystemCTargetModel.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>

#ifndef WAFER_TEST_DTE_COMPUTE_ACCESS_MODE
#error "WAFER_TEST_DTE_COMPUTE_ACCESS_MODE must select the test scenario"
#endif

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

constexpr test::PendingComputeDTEAccessMode getAccessMode() {
#if WAFER_TEST_DTE_COMPUTE_ACCESS_MODE == 0
  return test::PendingComputeDTEAccessMode::Disjoint;
#elif WAFER_TEST_DTE_COMPUTE_ACCESS_MODE == 1
  return test::PendingComputeDTEAccessMode::OverlapWithoutJoin;
#elif WAFER_TEST_DTE_COMPUTE_ACCESS_MODE == 2
  return test::PendingComputeDTEAccessMode::OverlapWithPreIssueJoin;
#else
#error "unknown WAFER_TEST_DTE_COMPUTE_ACCESS_MODE"
#endif
}

TEST(SystemCTargetModelDTEComputeAccessTest,
     EnforcesTypedElementwiseAndGemmPendingRanges) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModules> targetLLVMModules =
      test::compileDirectDTETargetModules(
          diagnostics, TargetIdentityId::waferTx81SingleCard());
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  llvm::Expected<test::PendingComputeDTERewriteResult> rewrite =
      test::insertPendingComputeBeforeDTEReceive(*targetLLVMModules,
                                                 getAccessMode());
  ASSERT_TRUE(static_cast<bool>(rewrite))
      << llvm::toString(rewrite.takeError());
  EXPECT_EQ(rewrite->elementwiseCount, 8u);
  EXPECT_EQ(rewrite->gemmCount, 8u);
  EXPECT_EQ(rewrite->insertedSetupJoinCount, 16u);
  EXPECT_GT(rewrite->removedInterveningJoinCount, 0u);
  EXPECT_EQ(rewrite->insertedJoinCount,
            getAccessMode() ==
                    test::PendingComputeDTEAccessMode::OverlapWithPreIssueJoin
                ? 16u
                : 0u);
  EXPECT_EQ(rewrite->overlappingReadCount,
            getAccessMode() == test::PendingComputeDTEAccessMode::Disjoint
                ? 0u
                : 16u);

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
  if (getAccessMode() ==
      test::PendingComputeDTEAccessMode::OverlapWithoutJoin) {
    ASSERT_FALSE(static_cast<bool>(result));
    const std::string error = llvm::toString(result.takeError());
    EXPECT_NE(error.find("dte-issue-order"), std::string::npos) << error;
    EXPECT_NE(error.find("pending NCC destination read/write"),
              std::string::npos)
        << error;
    EXPECT_EQ(error.find("no-progress"), std::string::npos) << error;
    return;
  }

  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->completedTileCount, 16);
  ASSERT_EQ(result->outputs.size(), 1u);
  EXPECT_EQ(result->outputs.front().bytes, invocation->expectedOutputBytes);
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
