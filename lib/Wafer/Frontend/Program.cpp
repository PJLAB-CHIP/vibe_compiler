//===- Program.cpp - Frontend program-directory orchestration ------------===//

#include "Wafer/Frontend/Program.h"

#include "ProgramInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

using namespace mlir;

namespace wafer::frontend {

using namespace program_detail;

static LogicalResult
verifyProgramDirectoryImpl(ModuleOp module, llvm::StringRef programPath,
                           llvm::raw_ostream &diagnostics,
                           FrontendProgramVerificationResult *result,
                           const ProgramPayloadResolver *resolver) {
  FrontendProgramVerificationResult verified;

  if (failed(verifyFrontendProgram(module, diagnostics, &verified)))
    return failure();

  std::string metaPath =
      program_detail::programPath(programPath, {"functions", "forward.meta"});
  FailureOr<ProgramMetadata> meta = parseProgramMetadata(metaPath, diagnostics);
  if (failed(meta))
    return failure();

  bool rejected = verifyProgramMetadata(module, programPath, *meta, diagnostics,
                                        &verified, resolver);
  if (!rejected) {
    FailureOr<func::FuncOp> func = findSingleFunction(module, diagnostics);
    if (failed(func))
      return failure();
    bool postSpmdMarker =
        hasSpmdParameterShardings(module) ||
        fileExists(program_detail::programPath(
            programPath, {"functions", "forward.parameter_shards.json"}));
    rejected |= verifyDistributedBoundary(module, *meta, *func, postSpmdMarker,
                                          diagnostics, &verified);
    if (!rejected)
      rejected |= verifyParameterShards(module, programPath, *meta, *func,
                                        diagnostics, resolver, &verified);
  }
  if (rejected)
    return failure();
  if (result)
    *result = std::move(verified);
  return success();
}

LogicalResult
verifyProgramDirectory(ModuleOp module, llvm::StringRef programPath,
                       llvm::raw_ostream &diagnostics,
                       FrontendProgramVerificationResult *result,
                       const ProgramPayloadResolver *resolver) {
  return verifyProgramDirectoryImpl(module, programPath, diagnostics, result,
                                    resolver);
}

} // namespace wafer::frontend
