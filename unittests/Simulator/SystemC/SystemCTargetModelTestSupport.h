//===- SystemCTargetModelTestSupport.h - Source DTE test support -*- C++
//-*-===//

#ifndef WAFER_UNITTESTS_MODEL_SYSTEMCTARGETMODELTESTSUPPORT_H
#define WAFER_UNITTESTS_MODEL_SYSTEMCTARGETMODELTESTSUPPORT_H

#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Simulator/Memory/TargetModelMemory.h"
#include "Wafer/Target/TargetCall.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>
#include <vector>

namespace wafer::model::test {

struct DirectDTEInvocationData {
  std::vector<compiler::TargetCallTileArguments> arguments;
  std::vector<TargetModelInputBinding> inputBindings;
  std::vector<uint8_t> expectedOutputBytes;
};

llvm::Expected<compiler::TargetLLVMModules> compileDirectDTETargetModules(
    std::string &diagnosticText,
    TargetIdentityId targetIdentity = TargetIdentityId::waferTx81SingleCard());

llvm::Expected<DirectDTEInvocationData> buildDirectDTEInvocationData(
    const compiler::TargetLLVMModules &targetLLVMModules);

struct NCCJoinRewriteResult {
  size_t erasedJoinCount = 0;
  size_t insertedJoinCount = 0;
  size_t insertedTerminalJoinCount = 0;
};

/// Removes every pre-existing local-fence/NCC-join call and inserts one typed
/// worker-0 join immediately after every selected Direct-DTE call plus one
/// terminal join before each entry return.
llvm::Expected<NCCJoinRewriteResult>
rewriteNCCJoinsAfter(compiler::TargetLLVMModules &targetLLVMModules,
                     TargetCallBuiltin anchor);

enum class PendingComputeDTEAccessMode : uint8_t {
  Disjoint,
  OverlapWithoutJoin,
  OverlapWithPreIssueJoin,
};

struct PendingComputeDTERewriteResult {
  size_t elementwiseCount = 0;
  size_t gemmCount = 0;
  size_t insertedSetupJoinCount = 0;
  size_t removedInterveningJoinCount = 0;
  size_t insertedJoinCount = 0;
  size_t overlappingReadCount = 0;
};

/// Inserts worker-0 elementwise and GEMM calls immediately before Direct
/// DTE receive preparation. Even ranks use elementwise and odd ranks use GEMM,
/// so one all-rank invocation exercises both typed read footprints.
llvm::Expected<PendingComputeDTERewriteResult>
insertPendingComputeBeforeDTEReceive(
    compiler::TargetLLVMModules &targetLLVMModules,
    PendingComputeDTEAccessMode accessMode);

enum class LateJoinDTEAccessMode : uint8_t {
  SourceWrite,
  DestinationRead,
  DestinationWrite,
};

struct LateJoinDTERewriteResult {
  size_t insertedComputeCount = 0;
  size_t insertedSetupJoinCount = 0;
  size_t removedPreIssueJoinCount = 0;
  size_t insertedLateJoinCount = 0;
};

/// Leaves one overlapping worker-0 elementwise command pending when the
/// Direct-DTE send is issued, then places the matching participant join
/// immediately after that issue. This intentionally-invalid ordering covers
/// source-write and destination-read/write hazards independently.
llvm::Expected<LateJoinDTERewriteResult>
insertPendingComputeWithLateJoin(compiler::TargetLLVMModules &targetLLVMModules,
                                 LateJoinDTEAccessMode accessMode);

} // namespace wafer::model::test

#endif // WAFER_UNITTESTS_MODEL_SYSTEMCTARGETMODELTESTSUPPORT_H
