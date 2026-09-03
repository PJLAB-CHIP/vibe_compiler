//===- SystemCTargetModelDTEMultiSendTest.cpp - Native DTE model gate ---===//

#include "SystemCTargetModelTestSupport.h"

#include "Wafer/Simulator/SystemC/SystemCTargetModel.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <string>
#include <utility>

#ifndef WAFER_TEST_DTE_MULTISEND_KIND
#error "WAFER_TEST_DTE_MULTISEND_KIND must select broadcast or scatter"
#endif

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

TEST(SystemCTargetModelDTEMultiSendTest,
     ExecutesOneNativeIssueWithTwoExactReceivers) {
  constexpr bool scatter = WAFER_TEST_DTE_MULTISEND_KIND == 1;
  std::string diagnostics;
  llvm::Expected<TargetLLVMModules> targetLLVMModules =
      scatter ? test::compileNativeScatterTargetModules(diagnostics)
              : test::compileNativeBroadcastTargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());
  llvm::Expected<test::DirectDTEInvocationData> invocation =
      scatter ? test::buildNativeScatterInvocationData(*targetLLVMModules)
              : test::buildNativeBroadcastInvocationData(*targetLLVMModules);
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
  ASSERT_EQ(result->outputs.size(), 1u);
  EXPECT_EQ(result->outputs.front().bytes, invocation->expectedOutputBytes);
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
