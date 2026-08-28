//===- SystemCTargetModelDTEReadinessTest.cpp - DTE peer-ready gate -------===//

#include "SystemCTargetModelTestSupport.h"

#include "Wafer/Simulator/SystemC/SystemCTargetModel.h"
#include "Wafer/Target/TargetCall.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

static llvm::Expected<size_t>
cloneIndependentNCCAfterSendIssue(TargetLLVMModules &targetLLVMModules) {
  const llvm::StringRef issueSymbol =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTESendIssue).symbol;
  size_t inserted = 0;
  for (const TargetLLVMModule &targetModule : targetLLVMModules.getModules()) {
    llvm::Module &module = const_cast<llvm::Module &>(targetModule.getModule());
    llvm::SmallVector<llvm::CallInst *, 4> sendIssues;
    llvm::CallInst *nccTemplate = nullptr;
    for (llvm::Function &function : module)
      for (llvm::BasicBlock &block : function)
        for (llvm::Instruction &instruction : block) {
          auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
          llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
          if (!callee)
            continue;
          if (callee->getName() == issueSymbol)
            sendIssues.push_back(call);
          const TargetCallDescriptor *descriptor =
              findTargetCallDescriptor(callee->getName());
          if (!nccTemplate && descriptor && descriptor->issueDomain &&
              descriptor->issueDomain->completionBehavior ==
                  TargetNCCCompletionBehavior::OrderedAsynchronousIssue)
            nccTemplate = call;
        }
    if (sendIssues.empty())
      continue;
    if (!nccTemplate)
      return llvm::createStringError(
          "Direct-DTE sender module has no independent NCC call to clone");
    for (llvm::CallInst *issue : sendIssues) {
      auto *clone = llvm::cast<llvm::CallInst>(nccTemplate->clone());
      clone->insertAfter(issue);
      ++inserted;
    }
    std::string verification;
    llvm::raw_string_ostream stream(verification);
    if (llvm::verifyModule(module, &stream))
      return llvm::createStringError(
          "late-receiver test produced invalid LLVM IR: " + stream.str());
  }
  if (inserted == 0)
    return llvm::createStringError(
        "Direct-DTE targetLLVMModules has no send issue to exercise");
  return inserted;
}

TEST(SystemCTargetModelDTEReadinessTest,
     LateReceiverReadinessPrecedesIndependentNCCAndWait) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModules> targetLLVMModules =
      test::compileDirectDTETargetModules(diagnostics);
  ASSERT_TRUE(static_cast<bool>(targetLLVMModules))
      << diagnostics << llvm::toString(targetLLVMModules.takeError());

  // Clone a legal independent NCC issue immediately after every explicit DTE
  // send issue. With no receiver-readiness check, an early sender can expose
  // this pending worker before its peer prepares the receive; matching then
  // defers on NCC visibility and the following DTE wait deadlocks. The real
  // issue contract blocks until receive prepare, so matching completes before
  // this cloned NCC issue becomes visible.
  llvm::Expected<size_t> inserted =
      cloneIndependentNCCAfterSendIssue(*targetLLVMModules);
  ASSERT_TRUE(static_cast<bool>(inserted))
      << llvm::toString(inserted.takeError());
  EXPECT_GT(*inserted, 0u);

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
                                          /*maximumScalarEvaluations=*/2048,
                                          /*maximumFusedMultiplyAdds=*/2048),
                                      /*maximumMovementBytes=*/8192,
                                      /*maximumMovementSegments=*/2048));
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->completedTileCount, 16);
  EXPECT_GT(result->finalDeltaCount, 0u);
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
