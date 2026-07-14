//===- SystemCTargetModelTestSupport.h - Source DTE test support -*- C++
//-*-===//

#ifndef WAFER_UNITTESTS_MODEL_SYSTEMCTARGETMODELTESTSUPPORT_H
#define WAFER_UNITTESTS_MODEL_SYSTEMCTARGETMODELTESTSUPPORT_H

#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Model/TargetModelMemory.h"

#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace wafer::model::test {

struct DirectDTEInvocationData {
  std::vector<compiler::TargetCallRankArguments> arguments;
  std::vector<TargetModelInputBinding> inputBindings;
  std::vector<std::vector<uint8_t>> inputBytesByRank;
};

llvm::Expected<compiler::TargetLLVMModuleBundle>
buildDirectDTETargetBundle(std::string &diagnosticText);

llvm::Expected<DirectDTEInvocationData>
buildDirectDTEInvocationData(const compiler::TargetLLVMModuleBundle &bundle);

} // namespace wafer::model::test

#endif // WAFER_UNITTESTS_MODEL_SYSTEMCTARGETMODELTESTSUPPORT_H
