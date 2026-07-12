//===- wafer-compile.cpp - Wafer user compiler driver --------------------===//

#include "Wafer/Compiler/Compilation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

#ifndef WAFER_XLA_SPMD_PARTITIONER_HELPER
#define WAFER_XLA_SPMD_PARTITIONER_HELPER ""
#endif

namespace {

struct CommandLineOptions {
  std::optional<std::string> inputProgramDirectory;
  std::optional<std::string> outputProgramDirectory;
  std::optional<std::string> executionRanks;
};

void printHelp() {
  llvm::outs() << "usage: wafer-compile --input-program-dir <dir> "
                  "--output-program-dir <dir> --execution-ranks <1|16>\n";
}

bool setOption(std::optional<std::string> &slot, llvm::StringRef option,
               llvm::StringRef value) {
  if (slot) {
    llvm::errs() << "wafer-compile: duplicate option: " << option << "\n";
    return true;
  }
  slot = value.str();
  return false;
}

bool parseValueOption(int argc, char **argv, int &index, llvm::StringRef arg,
                      llvm::StringRef option,
                      std::optional<std::string> &slot) {
  if (arg == option) {
    if (index + 1 >= argc) {
      llvm::errs() << "wafer-compile: missing value for " << option << "\n";
      return true;
    }
    return setOption(slot, option, argv[++index]);
  }

  std::string prefix = (option + "=").str();
  if (arg.starts_with(prefix))
    return setOption(slot, option, arg.drop_front(prefix.size()));
  return false;
}

bool parseCommandLine(int argc, char **argv, CommandLineOptions &options) {
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef arg(argv[index]);
    if (arg == "--help") {
      printHelp();
      return true;
    }
    if (arg == "--input-program-dir" ||
        arg.starts_with("--input-program-dir=")) {
      if (parseValueOption(argc, argv, index, arg, "--input-program-dir",
                           options.inputProgramDirectory))
        return false;
      continue;
    }
    if (arg == "--output-program-dir" ||
        arg.starts_with("--output-program-dir=")) {
      if (parseValueOption(argc, argv, index, arg, "--output-program-dir",
                           options.outputProgramDirectory))
        return false;
      continue;
    }
    if (arg == "--execution-ranks" || arg.starts_with("--execution-ranks=")) {
      if (parseValueOption(argc, argv, index, arg, "--execution-ranks",
                           options.executionRanks))
        return false;
      continue;
    }

    llvm::errs() << "wafer-compile: unknown argument: " << arg << "\n";
    return false;
  }
  return true;
}

bool requireOption(const std::optional<std::string> &value,
                   llvm::StringRef option) {
  if (value)
    return true;
  llvm::errs() << "wafer-compile: missing required " << option << "\n";
  return false;
}

std::string resolveSpmdPartitionerHelperPath() {
#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
  if (const char *environment =
          std::getenv("WAFER_TEST_XLA_SPMD_PARTITIONER_HELPER"))
    return environment;
#endif
  return WAFER_XLA_SPMD_PARTITIONER_HELPER;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && llvm::StringRef(argv[1]) == "--help") {
    printHelp();
    return 0;
  }

  CommandLineOptions options;
  if (!parseCommandLine(argc, argv, options))
    return 1;
  if (!requireOption(options.inputProgramDirectory, "--input-program-dir") ||
      !requireOption(options.outputProgramDirectory, "--output-program-dir") ||
      !requireOption(options.executionRanks, "--execution-ranks"))
    return 1;

  int64_t rankCount = 0;
  llvm::StringRef rankValue(*options.executionRanks);
  if (rankValue.getAsInteger(10, rankCount)) {
    llvm::errs() << "wafer-compile: invalid --execution-ranks value: "
                 << rankValue << "\n";
    return 1;
  }

  llvm::Expected<wafer::compiler::ExecutionConfig> executionConfig =
      wafer::compiler::ExecutionConfig::createForSingleCard(rankCount);
  if (!executionConfig) {
    llvm::errs() << "wafer-compile: "
                 << llvm::toString(executionConfig.takeError()) << "\n";
    return 1;
  }

  llvm::Expected<wafer::compiler::CompilationRequest> request =
      wafer::compiler::CompilationRequest::create(
          *options.inputProgramDirectory, std::move(*executionConfig));
  if (!request) {
    llvm::errs() << "wafer-compile: " << llvm::toString(request.takeError())
                 << "\n";
    return 1;
  }

  std::string helperPath = resolveSpmdPartitionerHelperPath();
  if (helperPath.empty()) {
    llvm::errs()
        << "wafer-compile: no XLA SPMD partitioner helper configured\n";
    return 1;
  }

  if (mlir::failed(wafer::compiler::compileToGroupedProgram(
          std::move(*request), *options.outputProgramDirectory, helperPath,
          llvm::errs())))
    return 1;

  llvm::outs() << "wafer-compile: compiled grouped program with "
                  "execution-ranks="
               << rankCount << ": " << *options.outputProgramDirectory << "\n";
  return 0;
}
