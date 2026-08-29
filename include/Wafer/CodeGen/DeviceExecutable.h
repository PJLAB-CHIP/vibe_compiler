//===- DeviceExecutable.h - Accepted device executable --------*- C++ -*-===//

#ifndef WAFER_CODEGEN_DEVICEEXECUTABLE_H
#define WAFER_CODEGEN_DEVICEEXECUTABLE_H

#include "Wafer/Frontend/Program.h"
#include "Wafer/Frontend/ProgramElementType.h"
#include "Wafer/Target/RuntimeLaunchContract.h"
#include "Wafer/Target/TargetIdentity.h"
#include "Wafer/Target/TopologyIds.h"
#include "Wafer/Target/TransportContract.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

class ProgramDataHandoff;

enum class ProgramResourceRole { UserInput, Parameter, Constant, Output };

/// Stable identity of one logical program tensor inside a compilation.
struct ProgramTensorId {
  ProgramResourceRole role = ProgramResourceRole::UserInput;
  int64_t roleIndex = -1;

  friend bool operator==(const ProgramTensorId &lhs,
                         const ProgramTensorId &rhs) {
    return lhs.role == rhs.role && lhs.roleIndex == rhs.roleIndex;
  }
  friend bool operator!=(const ProgramTensorId &lhs,
                         const ProgramTensorId &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const ProgramTensorId &lhs,
                        const ProgramTensorId &rhs) {
    return std::make_pair(static_cast<uint8_t>(lhs.role), lhs.roleIndex) <
           std::make_pair(static_cast<uint8_t>(rhs.role), rhs.roleIndex);
  }
};

/// Validated execution facts for the current single-card compiler boundary.
class ExecutionConfig {
public:
  static constexpr int64_t kSingleCardTileCount = 16;

  static llvm::Expected<ExecutionConfig>
  createForSingleCard(int64_t numPartitions);

  int64_t getNumPartitions() const { return numPartitions; }
  int64_t getTileCount() const { return tileCount; }
  TargetIdentityId getTargetIdentityId() const {
    return TargetIdentityId::waferTx81SingleCard();
  }

  friend bool operator==(const ExecutionConfig &lhs,
                         const ExecutionConfig &rhs) {
    return lhs.numPartitions == rhs.numPartitions &&
           lhs.tileCount == rhs.tileCount;
  }
  friend bool operator!=(const ExecutionConfig &lhs,
                         const ExecutionConfig &rhs) {
    return !(lhs == rhs);
  }

private:
  ExecutionConfig(int64_t numPartitions, int64_t tileCount)
      : numPartitions(numPartitions), tileCount(tileCount) {}

  int64_t numPartitions;
  int64_t tileCount;
};

enum class EntryLocalCompletionKind { ReturnAfterLocalDrain };
enum class DDRAllocationContract { DefaultArenaRelativeOffsets };

/// A verified program-boundary resource projected to one card partition.
struct ProgramResourceBinding {
  ProgramResourceRole role;
  ProgramTensorId programTensorId;
  int64_t index;
  int64_t programIndex;
  std::string name;
  ProgramElementType dtype = ProgramElementType::F32;
  frontend::ProgramDistributionKind distribution;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  frontend::ProgramPartitionSlice slice;
};

/// Optional same-invocation compiler inspection output.
struct TileIRTrace {
  CardId cardId{0};
  TileId tileId{0};
  LaunchSlotId launchSlotId{0};
  std::string tileDataflowIR;
};

struct CompilationIRTrace {
  std::vector<TileIRTrace> tiles;
};

/// One independently lowered and accepted Tile executable.
class TileExecutable {
public:
  TileExecutable(TileExecutable &&) = default;
  TileExecutable &operator=(TileExecutable &&) = default;
  TileExecutable(const TileExecutable &) = delete;
  TileExecutable &operator=(const TileExecutable &) = delete;

  CardId getCardId() const { return cardId; }
  TileId getTileId() const { return tileId; }
  LaunchSlotId getLaunchSlotId() const { return launchSlotId; }
  llvm::StringRef getEntrySymbol() const { return entrySymbol; }
  mlir::ModuleOp getModule() const { return *module; }
  const std::vector<ProgramResourceBinding> &getProgramBindings() const {
    return programBindings;
  }
  EntryLocalCompletionKind getEntryLocalCompletionKind() const {
    return entryLocalCompletionKind;
  }
  TransportContract getTransportContract() const { return transportContract; }
  DDRAllocationContract getDDRAllocationContract() const {
    return ddrAllocationContract;
  }

private:
  friend struct DeviceExecutableBuilder;

  TileExecutable(CardId cardId, TileId tileId, LaunchSlotId launchSlotId,
                 mlir::OwningOpRef<mlir::ModuleOp> module,
                 llvm::StringRef entrySymbol,
                 std::vector<ProgramResourceBinding> programBindings,
                 TransportContract transportContract)
      : cardId(cardId), tileId(tileId), launchSlotId(launchSlotId),
        module(std::move(module)), entrySymbol(entrySymbol.str()),
        programBindings(std::move(programBindings)),
        entryLocalCompletionKind(
            EntryLocalCompletionKind::ReturnAfterLocalDrain),
        transportContract(transportContract),
        ddrAllocationContract(
            DDRAllocationContract::DefaultArenaRelativeOffsets) {}

  CardId cardId;
  TileId tileId;
  LaunchSlotId launchSlotId;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string entrySymbol;
  std::vector<ProgramResourceBinding> programBindings;
  EntryLocalCompletionKind entryLocalCompletionKind;
  TransportContract transportContract;
  DDRAllocationContract ddrAllocationContract;
};

/// Owns the complete accepted executable and its source payload handoff.
class DeviceExecutable {
public:
  DeviceExecutable(DeviceExecutable &&);
  DeviceExecutable &operator=(DeviceExecutable &&);
  DeviceExecutable(const DeviceExecutable &) = delete;
  DeviceExecutable &operator=(const DeviceExecutable &) = delete;
  ~DeviceExecutable();

  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }
  const std::vector<TileExecutable> &getTileExecutables() const {
    return tiles;
  }
  const RuntimeLaunchContract &getRuntimeLaunchContract() const {
    return runtimeLaunchContract;
  }
  const ProgramDataHandoff &getProgramDataHandoff() const;

private:
  friend struct DeviceExecutableBuilder;

  DeviceExecutable(ExecutionConfig executionConfig,
                   RuntimeLaunchContract runtimeLaunchContract,
                   std::shared_ptr<mlir::MLIRContext> context,
                   std::vector<TileExecutable> tiles,
                   std::unique_ptr<ProgramDataHandoff> programData);

  ExecutionConfig executionConfig;
  RuntimeLaunchContract runtimeLaunchContract;
  std::shared_ptr<mlir::MLIRContext> context;
  std::vector<TileExecutable> tiles;
  std::unique_ptr<ProgramDataHandoff> programData;
};

} // namespace wafer::compiler

#endif // WAFER_CODEGEN_DEVICEEXECUTABLE_H
