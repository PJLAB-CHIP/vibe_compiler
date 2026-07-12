//===- Compilation.h - Typed Wafer compiler request ------------*- C++ -*-===//

#ifndef WAFER_COMPILER_COMPILATION_H
#define WAFER_COMPILER_COMPILATION_H

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::compiler {

/// Validated execution facts for the current single-card compiler boundary.
/// There is deliberately no default configuration: callers must choose the
/// one-rank or complete 16-rank domain explicitly.
class ExecutionConfig {
public:
  static llvm::Expected<ExecutionConfig>
  createForSingleCard(int64_t executionRankCount);

  int64_t getRankCount() const { return executionRankCount; }

private:
  explicit ExecutionConfig(int64_t executionRankCount)
      : executionRankCount(executionRankCount) {}

  int64_t executionRankCount;
};

/// Move-only semantic input to the compiler driver. Output locations,
/// toolchain helper paths, pass names and per-rank loop indices are
/// orchestration details and intentionally do not belong to this value.
class CompilationRequest {
public:
  static llvm::Expected<CompilationRequest>
  create(llvm::StringRef sourceProgramDirectory,
         ExecutionConfig executionConfig);

  CompilationRequest(CompilationRequest &&) = default;
  CompilationRequest &operator=(CompilationRequest &&) = default;
  CompilationRequest(const CompilationRequest &) = delete;
  CompilationRequest &operator=(const CompilationRequest &) = delete;

  llvm::StringRef getSourceProgramDirectory() const {
    return sourceProgramDirectory;
  }
  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }

private:
  CompilationRequest(llvm::StringRef sourceProgramDirectory,
                     ExecutionConfig executionConfig)
      : sourceProgramDirectory(sourceProgramDirectory.str()),
        executionConfig(executionConfig) {}

  std::string sourceProgramDirectory;
  ExecutionConfig executionConfig;
};

/// Runs the current production prefix through verified logical groups. The
/// input is snapshotted before parsing and the output directory becomes
/// visible only after every frontend, SPMD and lowering gate succeeds. The
/// executable-bundle stage extends this driver past the grouped-program
/// boundary.
mlir::LogicalResult compileToGroupedProgram(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper, llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler

#endif // WAFER_COMPILER_COMPILATION_H
