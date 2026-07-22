//===- DriverOptions.cpp - Wafer compiler driver options -----------------===//

#include "DriverInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>

#ifndef WAFER_XLA_SPMD_PARTITIONER_HELPER
#define WAFER_XLA_SPMD_PARTITIONER_HELPER ""
#endif

namespace wafer::compile_driver {

void printHelp() {
  llvm::outs() << "usage: wafer-compile --input-program-dir <dir> "
                  "--output-program-dir <dir> --execution-ranks <1|16> "
                  "--target-profile <registered-id> "
                  "--launch-abi <registered-id> "
                  "[--target-model "
                  "--model-input <index>=<npy> "
                  "--model-expected <index>=<npy> "
                  "[--model-atol <value>] [--model-rtol <value>] "
                  "[--model-report-numeric-statistics] "
                  "--target-model-max-scalar-evaluations <count> "
                  "--target-model-max-fused-multiply-adds <count> "
                  "--target-model-max-movement-bytes <bytes> "
                  "--target-model-max-movement-segments <count> "
                  "[--target-model-numeric-policy "
                  "<formal|prefer-admitted|managed-reference> "
                  "[--target-model-bulk-record <record>] "
                  "--target-model-max-bulk-total-bytes <bytes> "
                  "--target-model-max-bulk-scratchpad-bytes <bytes> "
                  "--target-model-max-bulk-reorder-bytes <bytes>]]\n";
}

namespace {

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

bool parseRepeatedValueOption(int argc, char **argv, int &index,
                              llvm::StringRef arg, llvm::StringRef option,
                              std::vector<std::string> &values) {
  if (arg == option) {
    if (index + 1 >= argc) {
      llvm::errs() << "wafer-compile: missing value for " << option << "\n";
      return true;
    }
    values.emplace_back(argv[++index]);
    return false;
  }
  std::string prefix = (option + "=").str();
  if (arg.starts_with(prefix)) {
    values.push_back(arg.drop_front(prefix.size()).str());
    return false;
  }
  return false;
}

} // namespace

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
    if (arg == "--target-profile" || arg.starts_with("--target-profile=")) {
      if (parseValueOption(argc, argv, index, arg, "--target-profile",
                           options.targetProfile))
        return false;
      continue;
    }
    if (arg == "--launch-abi" || arg.starts_with("--launch-abi=")) {
      if (parseValueOption(argc, argv, index, arg, "--launch-abi",
                           options.targetLaunchABI))
        return false;
      continue;
    }
    if (arg == "--target-model") {
      if (options.targetModel) {
        llvm::errs() << "wafer-compile: duplicate option: --target-model\n";
        return false;
      }
      options.targetModel = true;
      continue;
    }
    if (arg == "--model-report-numeric-statistics") {
      if (options.modelReportNumericStatistics) {
        llvm::errs() << "wafer-compile: duplicate option: "
                        "--model-report-numeric-statistics\n";
        return false;
      }
      options.modelReportNumericStatistics = true;
      continue;
    }
    if (arg == "--target-model-max-scalar-evaluations" ||
        arg.starts_with("--target-model-max-scalar-evaluations=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-scalar-evaluations",
                           options.targetModelMaximumScalarEvaluations))
        return false;
      continue;
    }
    if (arg == "--target-model-max-fused-multiply-adds" ||
        arg.starts_with("--target-model-max-fused-multiply-adds=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-fused-multiply-adds",
                           options.targetModelMaximumFusedMultiplyAdds))
        return false;
      continue;
    }
    if (arg == "--target-model-max-movement-bytes" ||
        arg.starts_with("--target-model-max-movement-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-movement-bytes",
                           options.targetModelMaximumMovementBytes))
        return false;
      continue;
    }
    if (arg == "--target-model-max-movement-segments" ||
        arg.starts_with("--target-model-max-movement-segments=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-movement-segments",
                           options.targetModelMaximumMovementSegments))
        return false;
      continue;
    }
    if (arg == "--target-model-numeric-policy" ||
        arg.starts_with("--target-model-numeric-policy=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-numeric-policy",
                           options.targetModelNumericPolicy))
        return false;
      continue;
    }
    if (arg == "--target-model-bulk-record" ||
        arg.starts_with("--target-model-bulk-record=")) {
      if (parseRepeatedValueOption(argc, argv, index, arg,
                                   "--target-model-bulk-record",
                                   options.targetModelBulkRecords))
        return false;
      continue;
    }
    if (arg == "--target-model-max-bulk-total-bytes" ||
        arg.starts_with("--target-model-max-bulk-total-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-bulk-total-bytes",
                           options.targetModelMaximumBulkTotalBytes))
        return false;
      continue;
    }
    if (arg == "--target-model-max-bulk-scratchpad-bytes" ||
        arg.starts_with("--target-model-max-bulk-scratchpad-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-bulk-scratchpad-bytes",
                           options.targetModelMaximumBulkScratchpadBytes))
        return false;
      continue;
    }
    if (arg == "--target-model-max-bulk-reorder-bytes" ||
        arg.starts_with("--target-model-max-bulk-reorder-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-bulk-reorder-bytes",
                           options.targetModelMaximumBulkReorderBytes))
        return false;
      continue;
    }
    if (arg == "--model-input" || arg.starts_with("--model-input=")) {
      if (parseRepeatedValueOption(argc, argv, index, arg, "--model-input",
                                   options.modelInputs))
        return false;
      continue;
    }
    if (arg == "--model-expected" || arg.starts_with("--model-expected=")) {
      if (parseRepeatedValueOption(argc, argv, index, arg, "--model-expected",
                                   options.modelExpected))
        return false;
      continue;
    }
    if (arg == "--model-atol" || arg.starts_with("--model-atol=")) {
      if (parseValueOption(argc, argv, index, arg, "--model-atol",
                           options.modelAtol))
        return false;
      continue;
    }
    if (arg == "--model-rtol" || arg.starts_with("--model-rtol=")) {
      if (parseValueOption(argc, argv, index, arg, "--model-rtol",
                           options.modelRtol))
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

std::optional<std::vector<IndexedPath>>
parseIndexedPaths(llvm::ArrayRef<std::string> values, llvm::StringRef option) {
  std::vector<IndexedPath> parsed;
  for (const std::string &storage : values) {
    llvm::StringRef value(storage);
    auto [indexText, path] = value.split('=');
    int64_t index = -1;
    if (path.empty() || indexText.getAsInteger(10, index) || index < 0) {
      llvm::errs() << "wafer-compile: invalid " << option << " value: " << value
                   << "\n";
      return std::nullopt;
    }
    if (llvm::any_of(parsed, [&](const IndexedPath &item) {
          return item.index == index;
        })) {
      llvm::errs() << "wafer-compile: duplicate " << option
                   << " index: " << index << "\n";
      return std::nullopt;
    }
    parsed.push_back({index, path.str()});
  }
  return parsed;
}

std::optional<double> parseTolerance(const std::optional<std::string> &value,
                                     llvm::StringRef option,
                                     double defaultValue) {
  if (!value)
    return defaultValue;
  errno = 0;
  char *end = nullptr;
  double parsed = std::strtod(value->c_str(), &end);
  if (errno != 0 || end != value->c_str() + value->size() ||
      !std::isfinite(parsed) || parsed < 0.0) {
    llvm::errs() << "wafer-compile: invalid " << option << " value: " << *value
                 << "\n";
    return std::nullopt;
  }
  return parsed;
}

std::optional<uint64_t>
parsePositiveCount(const std::optional<std::string> &value,
                   llvm::StringRef option) {
  if (!value) {
    llvm::errs() << "wafer-compile: target model requires " << option << "\n";
    return std::nullopt;
  }
  uint64_t parsed = 0;
  if (llvm::StringRef(*value).getAsInteger(10, parsed) || parsed == 0) {
    llvm::errs() << "wafer-compile: invalid " << option << " value: " << *value
                 << "\n";
    return std::nullopt;
  }
  return parsed;
}

} // namespace wafer::compile_driver
