//===- SystemCTargetModelDTEPreIssueTest.cpp - DTE issue-point gate -------===//

#include "SystemCTargetModelTestSupport.h"

#include "Wafer/Model/SystemC/SystemCTargetModel.h"
#include "Wafer/Target/Core/TargetCall.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include <string>
#include <utility>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

TEST(SystemCTargetModelDTEPreIssueTest,
     SendPrepareAloneCannotEstablishTransportEffect) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModules> targetLLVMModules =
      test::compileDirectDTETargetModules(
          diagnostics, TargetIdentityId::waferTx81SingleCard());
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());

  const llvm::StringRef issueSymbol =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTESendIssue).symbol;
  size_t erasedIssueCount = 0;
  for (const TargetLLVMModule &targetModule : targetLLVMModules->getModules()) {
    llvm::Module &module = const_cast<llvm::Module &>(targetModule.getModule());
    llvm::SmallVector<llvm::CallInst *, 4> issueCalls;
    for (llvm::Function &function : module)
      for (llvm::BasicBlock &block : function)
        for (llvm::Instruction &instruction : block)
          if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction))
            if (llvm::Function *callee = call->getCalledFunction())
              if (callee->getName() == issueSymbol)
                issueCalls.push_back(call);
    erasedIssueCount += issueCalls.size();
    for (llvm::CallInst *call : issueCalls)
      call->eraseFromParent();
  }
  EXPECT_GT(erasedIssueCount, 0u);

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
  EXPECT_NE(error.find("stage=dte-wait"), std::string::npos) << error;
  EXPECT_NE(error.find("prepared send that was not issued"), std::string::npos)
      << error;
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
