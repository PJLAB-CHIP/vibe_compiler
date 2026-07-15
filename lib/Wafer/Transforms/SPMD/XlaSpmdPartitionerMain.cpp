//===- XlaSpmdPartitionerMain.cpp - Wafer XLA SPMD helper driver -------===//

#include "XlaSpmdPartitionerInternal.h"

#include <iostream>

int main(int argc, char **argv) {
  using namespace wafer::xla_spmd_helper;
  absl::StatusOr<Options> options = parseOptions(argc, argv);
  if (!options.ok()) {
    std::cerr << "wafer_xla_spmd_partitioner: " << options.status() << "\n";
    printUsage();
    return 1;
  }
  absl::Status status = run(*options);
  if (!status.ok()) {
    std::cerr << "wafer_xla_spmd_partitioner: " << status << "\n";
    return 1;
  }
  return 0;
}
