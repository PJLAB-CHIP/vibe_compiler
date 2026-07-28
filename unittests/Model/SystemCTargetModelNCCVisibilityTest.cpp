//===- SystemCTargetModelNCCVisibilityTest.cpp - NCC publication gate ----===//

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
     RejectsDTEObservationWhenNCCJoinIsAfterTheWait) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModuleBundle> bundle =
      test::buildDirectDTETargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());

  // Remove production joins and put each replacement after its DTE wait. An
  // eager-memory model would copy the pending NCC bytes, let the wait return,
  // and then execute this too-late join. The worker-aware visibility model must
  // leave the endpoint pending, so the late join is unreachable.
  llvm::Expected<test::NCCJoinRewriteResult> rewrite =
      test::rewriteNCCJoinsAfter(*bundle, TargetCallBuiltin::DirectDTEWait);
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
  ASSERT_FALSE(static_cast<bool>(result));
  const std::string error = llvm::toString(result.takeError());
  EXPECT_NE(error.find("no-progress"), std::string::npos) << error;
  EXPECT_NE(error.find("unresolved Direct DTE/event state"), std::string::npos)
      << error;
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
