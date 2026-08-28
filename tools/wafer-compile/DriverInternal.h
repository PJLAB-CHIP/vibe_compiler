//===- DriverInternal.h - Wafer compiler driver internals -------*- C++ -*-===//

#ifndef WAFER_TOOLS_WAFER_COMPILE_DRIVERINTERNAL_H
#define WAFER_TOOLS_WAFER_COMPILE_DRIVERINTERNAL_H

#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Driver/Compilation.h"
#include "Wafer/Driver/CompilationResult.h"
#ifdef WAFER_ENABLE_SYSTEMC_MODEL
#include "Wafer/Simulator/SystemC/SystemCTargetModel.h"
#endif

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compile_driver {

struct CommandLineOptions {
  std::optional<std::string> inputProgramDirectory;
  std::optional<std::string> outputDirectory;
  std::optional<std::string> numPartitions;
  std::optional<std::string> compilerIRDumpDirectory;
  std::optional<std::string> optimizationPolicy;
  bool compileTiming = false;
  bool profile = false;
  // The following fields belong to the internal qualification/debug entry
  // (wafer-compile-test) and are never parsed by the production compiler.
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
  std::vector<std::string> targetModelOneDNNRecords;
  std::optional<std::string> targetModelMaximumOneDNNTotalBytes;
  std::optional<std::string> targetModelMaximumOneDNNScratchpadBytes;
  std::optional<std::string> targetModelMaximumOneDNNReorderBytes;
};

/// Validated external tool facts for one invocation. The single resolver
/// discovers installed resources relative to the executable and toolchain
/// interpreters via PATH; every fact is validated before use and the driver
/// never falls back to searching source or build tree paths.
struct DriverToolFacts {
  std::string spmdPartitionerHelper;
  std::string pythonExecutable;
  std::string deviceLinkerScript;
  std::string llvmClangXX;
  std::string tx8DepsRoot;
  std::string waferIncludeDir;
  std::string waferCrtSource;
  std::string waferCrtIncludeDir;
};

struct IndexedPath {
  int64_t index = -1;
  std::string path;
};

void printHelp();
bool parseCommandLine(int argc, char **argv, CommandLineOptions &options);
bool requireOption(const std::optional<std::string> &value,
                   llvm::StringRef option);
/// Single external tool resolver: test-only environment override, installed
/// resource next to the executable, then PATH for toolchain interpreters.
llvm::Expected<DriverToolFacts> resolveDriverToolFacts();
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
bool dumpCompilerIR(llvm::StringRef destination,
                    const wafer::compiler::CompiledProgram &compiledProgram,
                    llvm::raw_ostream &diagnostics);

#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
#ifdef WAFER_ENABLE_SYSTEMC_MODEL
bool runTargetModelGate(
    const CommandLineOptions &options,
    const wafer::compiler::CompiledProgram &compiledProgram,
    llvm::ArrayRef<IndexedPath> inputPaths,
    llvm::ArrayRef<IndexedPath> expectedPaths, double atol, double rtol,
    wafer::model::TargetModelKernelBudget budget,
    wafer::model::TargetModelExecutionPolicy executionPolicy);
#endif
#endif

} // namespace wafer::compile_driver

#endif // WAFER_TOOLS_WAFER_COMPILE_DRIVERINTERNAL_H
