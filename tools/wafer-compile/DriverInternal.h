//===- DriverInternal.h - Wafer compiler driver internals -------*- C++ -*-===//

#ifndef WAFER_TOOLS_WAFER_COMPILE_DRIVERINTERNAL_H
#define WAFER_TOOLS_WAFER_COMPILE_DRIVERINTERNAL_H

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"
#ifdef WAFER_ENABLE_SYSTEMC_MODEL
#include "Wafer/Model/SystemCTargetModel.h"
#endif

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compile_driver {

struct CommandLineOptions {
  std::optional<std::string> inputProgramDirectory;
  std::optional<std::string> outputProgramDirectory;
  std::optional<std::string> executionRanks;
  std::optional<std::string> runtimeLaunchKind;
  std::optional<std::string> compilerIRDumpDirectory;
  std::optional<std::string> optimizationPreset;
  bool compileTiming = false;
  bool profile = false;
  std::vector<std::string> modelInputs;
  std::vector<std::string> modelExpected;
  std::optional<std::string> modelAtol;
  std::optional<std::string> modelRtol;
  bool modelReportNumericStatistics = false;
  bool targetModel = false;
  std::optional<std::string> targetModelMaximumScalarEvaluations;
  std::optional<std::string> targetModelMaximumFusedMultiplyAdds;
  std::optional<std::string> targetModelMaximumMovementBytes;
  std::optional<std::string> targetModelMaximumMovementSegments;
  std::optional<std::string> targetModelNumericPolicy;
  std::vector<std::string> targetModelBulkRecords;
  std::optional<std::string> targetModelMaximumBulkTotalBytes;
  std::optional<std::string> targetModelMaximumBulkScratchpadBytes;
  std::optional<std::string> targetModelMaximumBulkReorderBytes;
};

struct IndexedPath {
  int64_t index = -1;
  std::string path;
};

void printHelp();
bool parseCommandLine(int argc, char **argv, CommandLineOptions &options);
bool requireOption(const std::optional<std::string> &value,
                   llvm::StringRef option);
std::string resolveSpmdPartitionerHelperPath();
std::optional<std::vector<IndexedPath>>
parseIndexedPaths(llvm::ArrayRef<std::string> values, llvm::StringRef option);
std::optional<double> parseTolerance(const std::optional<std::string> &value,
                                     llvm::StringRef option,
                                     double defaultValue);
std::optional<OptimizationConfig>
parseOptimizationConfig(const CommandLineOptions &options);
std::optional<uint64_t>
parsePositiveCount(const std::optional<std::string> &value,
                   llvm::StringRef option);

/// Writes the accepted instruction modules and their exact Target LLVM
/// translations for compiler inspection. The destination must not exist.
bool dumpCompilerIR(
    llvm::StringRef destination,
    const wafer::compiler::TargetCompilationProduct &product,
    llvm::raw_ostream &diagnostics);

#ifdef WAFER_ENABLE_SYSTEMC_MODEL
bool runTargetModelGate(
    const CommandLineOptions &options,
    const wafer::compiler::TargetCompilationProduct &product,
    llvm::ArrayRef<IndexedPath> inputPaths,
    llvm::ArrayRef<IndexedPath> expectedPaths, double atol, double rtol,
    wafer::model::TargetModelKernelBudget budget,
    wafer::model::TargetModelExecutionPolicy executionPolicy);
#endif

} // namespace wafer::compile_driver

#endif // WAFER_TOOLS_WAFER_COMPILE_DRIVERINTERNAL_H
