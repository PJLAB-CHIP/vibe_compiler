//===- DeviceExecutable.cpp - Device executable ownership ----------------===//

#include "Wafer/CodeGen/DeviceExecutableInternal.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <utility>
#include <vector>

namespace wafer::compiler {

llvm::Expected<ExecutionConfig>
ExecutionConfig::createForSingleCard(int64_t numPartitions) {
  if (numPartitions != 1)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "num-partitions must be exactly 1 for the single-card compiler");
  return ExecutionConfig(numPartitions, kSingleCardTileCount);
}

DeviceExecutable::DeviceExecutable(DeviceExecutable &&) = default;
DeviceExecutable &DeviceExecutable::operator=(DeviceExecutable &&) = default;
DeviceExecutable::~DeviceExecutable() = default;

DeviceExecutable::DeviceExecutable(
    ExecutionConfig executionConfig,
    RuntimeLaunchContract runtimeLaunchContract,
    std::shared_ptr<mlir::MLIRContext> context,
    std::vector<TileExecutable> tiles,
    std::unique_ptr<ProgramDataHandoff> programData)
    : executionConfig(executionConfig),
      runtimeLaunchContract(std::move(runtimeLaunchContract)),
      context(std::move(context)), tiles(std::move(tiles)),
      programData(std::move(programData)) {}

const ProgramDataHandoff &DeviceExecutable::getProgramDataHandoff() const {
  return *programData;
}

} // namespace wafer::compiler
