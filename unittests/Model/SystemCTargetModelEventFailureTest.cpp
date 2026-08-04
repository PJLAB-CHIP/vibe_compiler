//===- SystemCTargetModelEventFailureTest.cpp - SystemC event failure ---===//

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

llvm::CallInst *findWaitCall(llvm::Module &module) {
  const llvm::StringRef symbol =
      getTargetCallDescriptor(TargetCallBuiltin::DirectDTEWait)
          .symbol;
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &instruction : block)
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction))
          if (llvm::Function *callee = call->getCalledFunction())
            if (callee->getName() == symbol)
              return call;
  return nullptr;
}

TEST(SystemCTargetModelEventFailureTest,
     UnknownSourceProducedWaitEventFailsAtomicallyAndWakesPeers) {
  std::string diagnostics;
  llvm::Expected<TargetLLVMModuleBundle> bundle =
      test::buildDirectDTETargetBundle(diagnostics);
  ASSERT_TRUE(static_cast<bool>(bundle))
      << diagnostics << llvm::toString(bundle.takeError());

  llvm::CallInst *wait = nullptr;
  for (const TargetLLVMModule &targetModule : bundle->getModules()) {
    llvm::Module &module = const_cast<llvm::Module &>(targetModule.getModule());
    wait = findWaitCall(module);
    if (wait)
      break;
  }
  ASSERT_NE(wait, nullptr);
  ASSERT_EQ(wait->arg_size(), 1u);
  wait->setArgOperand(0,
                      llvm::ConstantInt::get(wait->getArgOperand(0)->getType(),
                                             UINT64_C(0xdeadbeef)));

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
  EXPECT_NE(error.find("stage=dte-wait"), std::string::npos) << error;
  EXPECT_NE(error.find("unknown, foreign, or already released event"),
            std::string::npos)
      << error;
}

} // namespace

extern "C" int sc_main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
