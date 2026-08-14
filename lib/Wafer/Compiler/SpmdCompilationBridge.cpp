//===- SpmdCompilationBridge.cpp - External SPMD helper bridge -----------===//

#include "CompilationInternal.h"

#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Program.h"

#include <string>

namespace wafer::compiler::detail {

mlir::LogicalResult runSpmdHelper(llvm::StringRef helper,
                                  llvm::StringRef inputProgramDirectory,
                                  llvm::StringRef outputProgramDirectory,
                                  const ExecutionConfig &config,
                                  llvm::raw_ostream &diagnostics) {
  wafer::support::ScopedCompileTimingSpan timing(
      "external-pipeline", "source-to-tensor-program", "xla-spmd-partitioning");
  std::string helperStorage = helper.str();
  std::string inputStorage = inputProgramDirectory.str();
  std::string outputStorage = outputProgramDirectory.str();
  std::string partitionCountStorage = std::to_string(config.getNumPartitions());
  llvm::SmallVector<llvm::StringRef, 9> arguments = {
      helperStorage,
      "--input-program-dir",
      inputStorage,
      "--output-program-dir",
      outputStorage,
      "--entry-function",
      "forward",
      "--num-partitions",
      partitionCountStorage,
  };
  int exitCode = llvm::sys::ExecuteAndWait(helperStorage, arguments);
  if (exitCode == 0)
    return mlir::success();
  timing.markFailed();
  std::string message = "XLA SPMD partitioner helper failed";
  if (exitCode > 0)
    message += " with exit code " + std::to_string(exitCode);
  reject(diagnostics, message);
  return mlir::failure();
}

} // namespace wafer::compiler::detail
