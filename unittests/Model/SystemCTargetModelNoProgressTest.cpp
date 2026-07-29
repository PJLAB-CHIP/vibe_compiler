//===- SystemCTargetModelNoProgressTest.cpp - SystemC failure gate -------===//

#include "SystemCTargetModelTestSupport.h"

#include "Wafer/Model/SystemCTargetModel.h"
#include "Wafer/Target/TargetCall.h"

#include "gtest/gtest.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>

namespace {

using namespace wafer;
using namespace wafer::compiler;
using namespace wafer::model;

llvm::CallInst *findSendIssueCall(llvm::Module &module) {
  const llvm::StringRef symbol =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTESendIssue).symbol;
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &instruction : block)
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction))
          if (llvm::Function *callee = call->getCalledFunction())
            if (callee->getName() == symbol)
              return call;
  return nullptr;
}

TEST(SystemCTargetModelNoProgressTest,
     UnmatchedSourceProducedEndpointWakesWaitersWithoutPartialResult) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModuleBundle> bundle =
      test::buildDirectDTETargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());

  llvm::Module &rankZeroModule =
      const_cast<llvm::Module &>(bundle->getModules().front().getModule());
  llvm::CallInst *send = findSendIssueCall(rankZeroModule);
  ASSERT_NE(send, nullptr);
  ASSERT_EQ(send->arg_size(), 7u);
  auto *fsm = llvm::dyn_cast<llvm::ConstantInt>(send->getArgOperand(5));
  ASSERT_NE(fsm, nullptr);
  const uint64_t unmatchedFSM = (fsm->getZExtValue() + 1) % 4;
  send->setArgOperand(5, llvm::ConstantInt::get(
                             send->getArgOperand(5)->getType(), unmatchedFSM));

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
  bool sawNoProgress = false;
  std::string detail;
  llvm::Error remaining = llvm::handleErrors(
      result.takeError(), [&](const SystemCTargetModelError &error) {
        sawNoProgress =
            error.getCode() == SystemCTargetModelErrorCode::NoProgress;
        detail = error.getDetail().str();
      });
  ASSERT_FALSE(static_cast<bool>(remaining))
      << llvm::toString(std::move(remaining));
  EXPECT_TRUE(sawNoProgress) << detail;
  EXPECT_NE(detail.find("unresolved Direct DTE/event state"), std::string::npos)
      << detail;
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
