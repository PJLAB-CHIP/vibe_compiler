//===- SystemCTargetModelTestSupport.h - Source DTE test support -*- C++
//-*-===//

#ifndef WAFER_UNITTESTS_MODEL_SYSTEMCTARGETMODELTESTSUPPORT_H
#define WAFER_UNITTESTS_MODEL_SYSTEMCTARGETMODELTESTSUPPORT_H

#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Model/TargetModelMemory.h"
#include "Wafer/Target/TargetCall.h"

#include "llvm/Support/Error.h"

#include <cstddef>
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

struct NCCJoinRewriteResult {
  size_t erasedJoinCount = 0;
  size_t insertedJoinCount = 0;
  size_t insertedTerminalJoinCount = 0;
};

/// Removes every pre-existing local-fence/NCC-join call and inserts one typed
/// worker-0 join immediately after every selected Direct-DTE call plus one
/// terminal join before each entry return.
llvm::Expected<NCCJoinRewriteResult> rewriteNCCJoinsAfter(
    compiler::TargetLLVMModuleBundle &bundle, TargetCallBuiltin anchor);

} // namespace wafer::model::test

#endif // WAFER_UNITTESTS_MODEL_SYSTEMCTARGETMODELTESTSUPPORT_H
