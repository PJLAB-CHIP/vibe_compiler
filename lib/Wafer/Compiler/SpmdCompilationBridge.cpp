//===- SpmdCompilationBridge.cpp - External SPMD helper bridge -----------===//

#include "CompilationInternal.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Program.h"

#include <string>

namespace wafer::compiler::detail {

bool runSpmdHelper(llvm::StringRef helper,
                   llvm::StringRef inputProgramDirectory,
                   llvm::StringRef outputProgramDirectory,
                   const ExecutionConfig &config,
                   llvm::raw_ostream &diagnostics) {
  std::string helperStorage = helper.str();
  std::string inputStorage = inputProgramDirectory.str();
  std::string outputStorage = outputProgramDirectory.str();
  std::string rankCountStorage = std::to_string(config.getRankCount());
  llvm::SmallVector<llvm::StringRef, 9> arguments = {
      helperStorage,    "--input-program-dir",
      inputStorage,     "--output-program-dir",
      outputStorage,    "--entry-function",
      "forward",        "--logical-rank-count",
      rankCountStorage,
  };
  int exitCode = llvm::sys::ExecuteAndWait(helperStorage, arguments);
  if (exitCode == 0)
    return false;
  std::string message = "XLA SPMD partitioner helper failed";
  if (exitCode > 0)
    message += " with exit code " + std::to_string(exitCode);
  return reject(diagnostics, message);
}

} // namespace wafer::compiler::detail
