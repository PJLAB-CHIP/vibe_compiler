//===- XlaSpmdDriver.cpp - XLA SPMD helper command line --------------===//

#include "XlaSpmdPartitionerInternal.h"

#include "absl/strings/str_cat.h"
#include "tsl/platform/statusor.h"

#include <cstdlib>
#include <iostream>

namespace wafer::xla_spmd_helper {

void printUsage() {
  std::cerr << "wafer_xla_spmd_partitioner "
               "--input-program-dir <dir> --output-program-dir <dir> "
               "--entry-function forward --num-partitions <n>\n";
}

absl::StatusOr<Options> parseOptions(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto requireValue = [&](std::string_view name) -> absl::StatusOr<char *> {
      if (i + 1 >= argc)
        return absl::InvalidArgumentError(
            absl::StrCat("missing value for ", name));
      return argv[++i];
    };

    if (arg == "--input-program-dir") {
      TF_ASSIGN_OR_RETURN(char *value, requireValue(arg));
      options.inputProgramDir = value;
    } else if (arg == "--output-program-dir") {
      TF_ASSIGN_OR_RETURN(char *value, requireValue(arg));
      options.outputProgramDir = value;
    } else if (arg == "--entry-function") {
      TF_ASSIGN_OR_RETURN(char *value, requireValue(arg));
      options.entryFunction = value;
    } else if (arg == "--num-partitions") {
      TF_ASSIGN_OR_RETURN(char *value, requireValue(arg));
      try {
        options.numPartitions = std::stoll(value);
      } catch (...) {
        return absl::InvalidArgumentError(
            "--num-partitions must be an integer");
      }
    } else if (arg == "--help") {
      printUsage();
      std::exit(0);
    } else {
      return absl::InvalidArgumentError(absl::StrCat("unknown option: ", arg));
    }
  }

  if (options.inputProgramDir.empty())
    return absl::InvalidArgumentError("missing --input-program-dir");
  if (options.outputProgramDir.empty())
    return absl::InvalidArgumentError("missing --output-program-dir");
  if (options.entryFunction.empty())
    return absl::InvalidArgumentError("missing --entry-function");
  if (options.entryFunction != "forward")
    return absl::InvalidArgumentError(
        "--entry-function must be the canonical entry 'forward'");
  if (options.numPartitions <= 0)
    return absl::InvalidArgumentError("--num-partitions must be positive");
  return options;
}

} // namespace wafer::xla_spmd_helper
