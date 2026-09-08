//===- BoardRuntime.cpp - Verified package board execution --------------===//

#include "Wafer/Runtime/Board/BoardRuntime.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Package/Manifest/PackageManifest.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace wafer::runtime {
namespace {

constexpr uint64_t boardRuntimeFreeMemoryReserve = 64ULL * 1024 * 1024;
constexpr uint64_t kProgramDataReadWindowBytes = 1 << 20;

constexpr CardId noCard{-1};
constexpr TileId noTile{-1};

struct DiagnosticLocation {
  CardId cardId = noCard;
  TileId tileId = noTile;
  LaunchSlotId launchSlot;
  EntryId entry;
};

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(llvm::errc::invalid_argument, message);
}

llvm::Error boardError(
    BoardRuntimeStage stage, DiagnosticLocation location, llvm::Twine detail,
    BoardRuntimeContextState contextState = BoardRuntimeContextState::Usable) {
  return llvm::make_error<BoardRuntimeError>(
      stage, location.cardId, location.tileId, location.launchSlot,
      location.entry, detail.str(), contextState);
}

llvm::Error wrapDriverError(
    BoardRuntimeStage stage, DiagnosticLocation location, llvm::Error error,
    BoardRuntimeContextState contextState = BoardRuntimeContextState::Usable) {
  return boardError(stage, location, llvm::toString(std::move(error)),
                    contextState);
}

DiagnosticLocation locationFor(const RuntimeSessionPlan &tile) {
  return {tile.cardId, tile.tileId, tile.launchSlot, tile.entry};
}

DiagnosticLocation locationFor(const PackageEntrypointRecord &entry) {
  return {entry.cardId, entry.tileId, entry.launchSlot, entry.id};
}

bool hasCompleteQualification(const BoardDeviceQualification &qualification) {
  return qualification.runtimeVersion != 0 && qualification.tileCount != 0 &&
         !qualification.name.empty() && !qualification.pciBusId.empty() &&
         !qualification.runtimeLibraryDigest.empty();
}

bool qualificationMatches(const BoardDeviceQualification &lhs,
                          const BoardDeviceQualification &rhs) {
  return lhs.runtimeVersion == rhs.runtimeVersion &&
         lhs.tileCount == rhs.tileCount && lhs.name == rhs.name &&
         lhs.pciBusId == rhs.pciBusId &&
         lhs.runtimeLibraryDigest == rhs.runtimeLibraryDigest;
}

bool deviceMatchesQualification(const BoardDeviceInfo &device,
                                const BoardDeviceQualification &qualification) {
  return device.runtimeVersion == qualification.runtimeVersion &&
         device.tileCount == qualification.tileCount &&
         device.name == qualification.name &&
         device.pciBusId == qualification.pciBusId &&
         device.runtimeLibraryDigest == qualification.runtimeLibraryDigest;
}

llvm::Expected<BoardDeviceInfo>
qualifyBoardDevice(uint32_t deviceId, uint32_t requiredTileCount,
                   const BoardDeviceQualification &qualification,
                   BoardRuntimeDriver &driver) {
  if (!hasCompleteQualification(qualification))
    return boardError(BoardRuntimeStage::Validation, {},
                      "board execution requires complete explicit device "
                      "qualification facts");
  if (requiredTileCount == 0 || requiredTileCount != qualification.tileCount)
    return boardError(BoardRuntimeStage::Validation, {},
                      "complete package Tile domain does not exactly match "
                      "the explicit device qualification");
  if (driver.getContextState() == BoardRuntimeContextState::Poisoned)
    return boardError(BoardRuntimeStage::DeviceSelection, {},
                      "TX provider is already quarantined",
                      BoardRuntimeContextState::Poisoned);

  llvm::Expected<uint32_t> deviceCount = driver.getDeviceCount();
  if (!deviceCount)
    return wrapDriverError(BoardRuntimeStage::DeviceSelection, {},
                           deviceCount.takeError(), driver.getContextState());
  if (deviceId >= *deviceCount)
    return boardError(BoardRuntimeStage::DeviceSelection, {},
                      "requested device is not present");
  if (llvm::Error error = driver.selectDevice(deviceId))
    return wrapDriverError(BoardRuntimeStage::DeviceSelection, {},
                           std::move(error), driver.getContextState());
  llvm::Expected<BoardDeviceInfo> device = driver.getDeviceInfo(deviceId);
  if (!device)
    return wrapDriverError(BoardRuntimeStage::DeviceSelection, {},
                           device.takeError(), driver.getContextState());
  if (device->deviceId != deviceId)
    return boardError(BoardRuntimeStage::DeviceSelection, {},
                      "TX inventory identifies a different selected device");
  if (!deviceMatchesQualification(*device, qualification))
    return boardError(BoardRuntimeStage::DeviceSelection, {},
                      "live TX inventory does not match the explicit board "
                      "qualification");
  if (device->freeMemoryBytes > device->totalMemoryBytes)
    return boardError(BoardRuntimeStage::DeviceSelection, {},
                      "TX runtime reported free memory greater than total "
                      "memory");

  uint32_t availableTiles = 0;
  llvm::DenseSet<int64_t> tileIds;
  llvm::DenseSet<uint64_t> launchSlots;
  llvm::DenseSet<uint64_t> availableLaunchSlots;
  std::set<std::pair<uint32_t, uint32_t>> tileCoordinates;
  for (const BoardDeviceInfo::Tile &tile : device->tiles) {
    if (tile.tileId.getValue() < 0 ||
        !tileIds.insert(tile.tileId.getValue()).second)
      return boardError(BoardRuntimeStage::DeviceSelection, {},
                        "TX inventory contains an invalid or duplicate "
                        "Tile ID");
    if (!tile.launchSlot.isValid() ||
        !launchSlots.insert(tile.launchSlot.getValue()).second)
      return boardError(BoardRuntimeStage::DeviceSelection, {},
                        "TX inventory contains an invalid or duplicate "
                        "launch slot");
    if (!tile.available)
      continue;
    ++availableTiles;
    availableLaunchSlots.insert(tile.launchSlot.getValue());
    if (!tileCoordinates.emplace(tile.physicalX, tile.physicalY).second)
      return boardError(BoardRuntimeStage::DeviceSelection, {},
                        "TX inventory maps available tiles to duplicate "
                        "physical coordinates");
  }
  if (availableTiles != device->tileCount ||
      availableTiles != requiredTileCount)
    return boardError(BoardRuntimeStage::DeviceSelection, {},
                      "TX tile availability is not exactly the complete "
                      "qualified launch-slot domain");
  for (uint32_t launchSlot = 0; launchSlot < requiredTileCount; ++launchSlot)
    if (!availableLaunchSlots.contains(launchSlot))
      return boardError(BoardRuntimeStage::DeviceSelection, {},
                        "TX inventory does not contain the complete qualified "
                        "launch-slot domain");
  return std::move(*device);
}

llvm::Error verifyPlannedTileBindings(const BoardDeviceInfo &device,
                                      const RuntimeInvocationPlan &plan) {
  for (const RuntimeSessionPlan &tile : plan.tiles) {
    auto inventory = llvm::find_if(device.tiles, [&](const auto &candidate) {
      return candidate.available && candidate.tileId == tile.tileId &&
             candidate.launchSlot == tile.launchSlot;
    });
    if (inventory == device.tiles.end())
      return boardError(BoardRuntimeStage::Validation, locationFor(tile),
                        "package Tile/launch-slot binding is absent from the "
                        "qualified device inventory");
  }
  return llvm::Error::success();
}

struct VerifiedModuleSnapshot {
  const PackageModuleRecord *module = nullptr;
  DiagnosticLocation diagnosticLocation;
  std::vector<uint8_t> bytes;
};

struct LiveModule {
  const PackageModuleRecord *moduleRecord = nullptr;
  BoardModuleHandle module;
};

} // namespace

char BoardRuntimeError::ID = 0;

llvm::StringRef stringifyBoardRuntimeStage(BoardRuntimeStage stage) {
  switch (stage) {
  case BoardRuntimeStage::Validation:
    return "validation";
  case BoardRuntimeStage::DeviceSelection:
    return "device-selection";
  case BoardRuntimeStage::ResourceAllocation:
    return "resource-allocation";
  case BoardRuntimeStage::HostToDevice:
    return "host-to-device";
  case BoardRuntimeStage::ModuleLoad:
    return "module-load";
  case BoardRuntimeStage::EntryResolve:
    return "entry-resolve";
  case BoardRuntimeStage::Launch:
    return "launch";
  case BoardRuntimeStage::Completion:
    return "completion";
  case BoardRuntimeStage::DeviceToHost:
    return "device-to-host";
  case BoardRuntimeStage::Cleanup:
    return "cleanup";
  }
  llvm_unreachable("unknown board runtime stage");
}

llvm::StringRef
stringifyBoardRuntimeContextState(BoardRuntimeContextState state) {
  switch (state) {
  case BoardRuntimeContextState::Usable:
    return "usable";
  case BoardRuntimeContextState::Poisoned:
    return "poisoned";
  }
  llvm_unreachable("unknown board runtime context state");
}

void BoardRuntimeError::log(llvm::raw_ostream &stream) const {
  stream << "board runtime " << stringifyBoardRuntimeStage(stage) << " failed";
  if (cardId.getValue() >= 0)
    stream << " for card " << cardId.getValue();
  if (tileId.getValue() >= 0)
    stream << " Tile " << tileId.getValue();
  if (launchSlot.isValid())
    stream << " launch-slot " << launchSlot.getValue();
  if (entry.isValid())
    stream << " entry " << entry.getValue();
  stream << " (context=" << stringifyBoardRuntimeContextState(contextState)
         << "): " << detail;
}

std::error_code BoardRuntimeError::convertToErrorCode() const {
  return llvm::make_error_code(llvm::errc::io_error);
}

namespace {

struct BoardLiveAllocation {
  enum class Kind { ProgramData, Invocation };
  Kind kind = Kind::Invocation;
  DiagnosticLocation location;
  BoardDeviceMemory memory;
};

llvm::Expected<BoardRuntimeInvocationResult> executeBoardInvocationImpl(
    const ExecutablePackage &package, BoardRuntimeInvocationRequest request,
    BoardRuntimeDriver &driver, const BoardDeviceInfo *qualifiedDevice,
    uint32_t qualifiedTileCount, bool *qualifiedSessionUsable) {
  const PackageManifest &manifest = package.getManifest();
  const KernelRuntimeLaunchContract &kernelLaunch = manifest.launch.getKernel();
  if (!hasCompleteQualification(request.qualification))
    return boardError(BoardRuntimeStage::Validation, {},
                      "board execution requires complete explicit device "
                      "qualification facts");
  if (request.completionTimeoutMilliseconds == 0 ||
      request.completionTimeoutMilliseconds >
          kMaximumBoardCompletionTimeoutMilliseconds)
    return boardError(BoardRuntimeStage::Validation, {},
                      "board completion timeout is outside the supported "
                      "range");

  llvm::DenseSet<int64_t> directDTETiles;
  for (const PackageEntrypointRecord &entry : manifest.entries)
    if (std::holds_alternative<DirectDTETransportRequirements>(entry.transport))
      directDTETiles.insert(entry.tileId.getValue());
  const bool hasDirectDTETransport = !directDTETiles.empty();

  // Caller input bindings: all-and-only external input ports with exact
  // target byte counts.
  llvm::DenseMap<uint64_t, const BoardRuntimeBinding *> bindingsByPort;
  for (const BoardRuntimeBinding &binding : request.bindings) {
    uint64_t id = binding.port.getValue();
    if (!binding.port.isValid() || bindingsByPort.count(id))
      return boardError(BoardRuntimeStage::Validation, {},
                        "invocation has a duplicate or invalid input port");
    if (id >= manifest.inputs.size())
      return boardError(BoardRuntimeStage::Validation, {},
                        "invocation binds an unknown input port");
    if (binding.bytes.size() != manifest.inputs[id].bytes)
      return boardError(BoardRuntimeStage::Validation, {},
                        "invocation buffer byte count is not exact");
    bindingsByPort[id] = &binding;
  }
  for (uint64_t port = 0; port < manifest.inputs.size(); ++port)
    if (!bindingsByPort.count(port))
      return boardError(BoardRuntimeStage::Validation, {},
                        "invocation omits an input port binding");
  if (bindingsByPort.size() != request.bindings.size())
    return boardError(BoardRuntimeStage::Validation, {},
                      "invocation bindings are not all-and-only for package");

  // Profiler records: present in the package exactly when the invocation
  // provides the exact record bytes for all 16 Tiles.
  std::vector<const PackageEntrypointRecord *> entriesByLaunchSlot(
      manifest.tileCount, nullptr);
  for (const PackageEntrypointRecord &entry : manifest.entries)
    entriesByLaunchSlot[entry.launchSlot.getValue()] = &entry;
  const auto profileArgument = [](const PackageEntrypointRecord &entry)
      -> const ProfileRecordArgument * {
    for (const TileEntryArgumentRecord &argument : entry.arguments)
      if (const auto *profile =
              std::get_if<ProfileRecordArgument>(&argument.reference))
        return profile;
    return nullptr;
  };
  size_t packageProfileRecords = 0;
  for (const PackageEntrypointRecord *entry : entriesByLaunchSlot)
    if (profileArgument(*entry))
      ++packageProfileRecords;
  const size_t tileCount = static_cast<size_t>(manifest.tileCount);
  if (packageProfileRecords != 0 && packageProfileRecords != tileCount)
    return boardError(BoardRuntimeStage::Validation, {},
                      "package profile records do not cover the complete "
                      "Tile domain");
  if (request.profilerRecordBytes) {
    if (packageProfileRecords != tileCount ||
        request.profilerRecordBytes->size() != tileCount)
      return boardError(BoardRuntimeStage::Validation, {},
                        "invocation provides profiler records for a package "
                        "without the complete record domain");
    for (int64_t launchSlot = 0; launchSlot < manifest.tileCount; ++launchSlot)
      if (profileArgument(*entriesByLaunchSlot[launchSlot])->bytes !=
          (*request.profilerRecordBytes)[launchSlot].size())
        return boardError(BoardRuntimeStage::Validation,
                          locationFor(*entriesByLaunchSlot[launchSlot]),
                          "profiler invocation buffer byte count is not exact");
  } else if (packageProfileRecords != 0) {
    return boardError(BoardRuntimeStage::Validation, {},
                      "package profile records require exact invocation "
                      "record bytes");
  }

  std::vector<RuntimeInvocationBinding> invocationBindings;
  invocationBindings.reserve(manifest.inputs.size());
  for (const ExternalPortRecord &port : manifest.inputs) {
    invocationBindings.push_back({port.id, port.bytes, port.alignment});
  }

  const RuntimeEnvironment &providerEnvironment =
      driver.getProviderEnvironment();
  llvm::Expected<RuntimeInvocationPlan> semanticPlan = planRuntimeInvocation(
      package.getVerifiedManifest(), invocationBindings, providerEnvironment);
  if (!semanticPlan)
    return wrapDriverError(BoardRuntimeStage::Validation, {},
                           semanticPlan.takeError());

  std::vector<VerifiedModuleSnapshot> moduleSnapshots;
  moduleSnapshots.reserve(manifest.modules.size());
  if (package.getModuleBuffers().size() != manifest.modules.size())
    return boardError(BoardRuntimeStage::Validation, {},
                      "owned package module domain is incomplete");
  for (auto [moduleIndex, module] : llvm::enumerate(manifest.modules)) {
    auto firstTile = llvm::find_if(semanticPlan->tiles, [&](const auto &tile) {
      return tile.module == module.id;
    });
    if (firstTile == semanticPlan->tiles.end())
      return boardError(BoardRuntimeStage::Validation, {},
                        "package contains an unreferenced module");
    llvm::StringRef owned =
        package.getModuleBuffers()[moduleIndex]->getBuffer();
    std::vector<uint8_t> bytes(reinterpret_cast<const uint8_t *>(owned.data()),
                               reinterpret_cast<const uint8_t *>(owned.data()) +
                                   owned.size());
    moduleSnapshots.push_back(
        {&module, locationFor(*firstTile), std::move(bytes)});
  }
  llvm::sort(moduleSnapshots, [](const auto &lhs, const auto &rhs) {
    return lhs.diagnosticLocation.launchSlot.getValue() <
           rhs.diagnosticLocation.launchSlot.getValue();
  });

  BoardDeviceInfo device;
  if (qualifiedDevice) {
    if (!qualifiedSessionUsable || !*qualifiedSessionUsable)
      return boardError(BoardRuntimeStage::DeviceSelection, {},
                        "qualified board runtime session is no longer usable",
                        BoardRuntimeContextState::Poisoned);
    if (manifest.tileCount != static_cast<int64_t>(qualifiedTileCount))
      return boardError(
          BoardRuntimeStage::Validation, {},
          "package Tile domain does not match the qualified board runtime "
          "session");
    if (driver.getContextState() == BoardRuntimeContextState::Poisoned) {
      *qualifiedSessionUsable = false;
      return boardError(BoardRuntimeStage::DeviceSelection, {},
                        "TX provider is already quarantined",
                        BoardRuntimeContextState::Poisoned);
    }
    device = *qualifiedDevice;
  } else {
    llvm::Expected<BoardDeviceInfo> qualified = qualifyBoardDevice(
        request.deviceId, static_cast<uint32_t>(manifest.tileCount),
        request.qualification, driver);
    if (!qualified)
      return qualified.takeError();
    device = std::move(*qualified);
  }

  RuntimeEnvironment capacityEnvironment = providerEnvironment;
  capacityEnvironment.maxResourceBytes = device.freeMemoryBytes;
  llvm::Expected<RuntimeInvocationPlan> capacityPlan = planRuntimeInvocation(
      package.getVerifiedManifest(), invocationBindings, capacityEnvironment);
  if (!capacityPlan)
    return wrapDriverError(BoardRuntimeStage::Validation, {},
                           capacityPlan.takeError());
  if (llvm::Error error = verifyPlannedTileBindings(device, *capacityPlan))
    return std::move(error);

  uint64_t allocationBytes = 0;
  auto accumulate = [&](uint64_t bytes, llvm::StringRef what) -> llvm::Error {
    if (bytes > std::numeric_limits<uint64_t>::max() - allocationBytes)
      return boardError(
          BoardRuntimeStage::Validation, {},
          (llvm::Twine("aggregate board ") + what + " byte count overflows")
              .str());
    allocationBytes += bytes;
    return llvm::Error::success();
  };
  if (llvm::Error error =
          accumulate(capacityPlan->programDataBytes, "program data"))
    return std::move(error);
  if (llvm::Error error =
          accumulate(capacityPlan->invocationBytes, "invocation"))
    return std::move(error);
  for (const VerifiedModuleSnapshot &snapshot : moduleSnapshots) {
    if (llvm::Error error = accumulate(snapshot.bytes.size(), "module"))
      return std::move(error);
  }
  if (device.freeMemoryBytes <= boardRuntimeFreeMemoryReserve ||
      allocationBytes > device.freeMemoryBytes - boardRuntimeFreeMemoryReserve)
    return boardError(
        BoardRuntimeStage::Validation, {},
        "aggregate board allocation demand exceeds qualified free memory "
        "after the runtime safety reserve");

  BoardRuntimeInvocationResult result;
  result.device = device;
  for (const RuntimeSessionPlan &tile : capacityPlan->tiles)
    result.tiles.push_back({tile.entry, tile.cardId, tile.tileId,
                            tile.launchSlot, tile.module, tile.completion});
  result.completedStages.push_back(BoardRuntimeStage::Validation);
  result.completedStages.push_back(BoardRuntimeStage::DeviceSelection);

  std::vector<BoardLiveAllocation> allocations;
  std::vector<uint64_t> tileRowAddresses;
  std::vector<LiveModule> liveModules;
  bool submissionLive = false;
  auto observeProviderState = [&]() {
    BoardRuntimeContextState state = driver.getContextState();
    if (state == BoardRuntimeContextState::Poisoned && qualifiedSessionUsable)
      *qualifiedSessionUsable = false;
    return state;
  };
  auto cleanup = [&]() -> llvm::Error {
    llvm::Error cleanupError = llvm::Error::success();
    if (submissionLive) {
      if (llvm::Error error = driver.releaseSubmission()) {
        BoardRuntimeContextState state = observeProviderState();
        llvm::Error wrapped = wrapDriverError(BoardRuntimeStage::Cleanup, {},
                                              std::move(error), state);
        if (state == BoardRuntimeContextState::Poisoned)
          return llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
        cleanupError =
            llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
      }
      submissionLive = false;
    }
    for (LiveModule &liveModule : llvm::reverse(liveModules)) {
      if (llvm::Error error = driver.unloadModule(liveModule.module)) {
        BoardRuntimeContextState state = observeProviderState();
        llvm::Error wrapped = wrapDriverError(BoardRuntimeStage::Cleanup, {},
                                              std::move(error), state);
        if (state == BoardRuntimeContextState::Poisoned)
          return llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
        cleanupError =
            llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
      }
    }
    liveModules.clear();
    for (BoardLiveAllocation &allocation : llvm::reverse(allocations)) {
      if (llvm::Error error = driver.free(allocation.memory)) {
        BoardRuntimeContextState state = observeProviderState();
        llvm::Error wrapped =
            wrapDriverError(BoardRuntimeStage::Cleanup, allocation.location,
                            std::move(error), state);
        if (state == BoardRuntimeContextState::Poisoned)
          return llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
        cleanupError =
            llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
      }
    }
    allocations.clear();
    return cleanupError;
  };
  auto fail =
      [&](BoardRuntimeStage stage, DiagnosticLocation location,
          llvm::Error error) -> llvm::Expected<BoardRuntimeInvocationResult> {
    BoardRuntimeContextState state = observeProviderState();
    llvm::Error primary =
        wrapDriverError(stage, location, std::move(error), state);
    if (state == BoardRuntimeContextState::Poisoned)
      return std::move(primary);
    return llvm::joinErrors(std::move(primary), cleanup());
  };

  // Program data: one contiguous read-only allocation with one whole-file
  // H2D when the package owns TargetTensors; no provider call otherwise.
  std::optional<BoardDeviceMemory> programDataMemory;
  std::vector<uint8_t> programDataBytes;
  if (capacityPlan->programDataRequired) {
    llvm::StringRef owned = package.getProgramDataBuffer().getBuffer();
    programDataBytes.assign(reinterpret_cast<const uint8_t *>(owned.data()),
                            reinterpret_cast<const uint8_t *>(owned.data()) +
                                owned.size());
    llvm::Expected<BoardDeviceMemory> memory = driver.allocate(
        capacityPlan->programDataBytes, capacityPlan->programDataAlignment);
    if (!memory)
      return fail(BoardRuntimeStage::ResourceAllocation, {},
                  memory.takeError());
    allocations.push_back(
        {BoardLiveAllocation::Kind::ProgramData, {}, *memory});
    if (llvm::Error error = driver.copyHostToDevice(*memory, programDataBytes))
      return fail(BoardRuntimeStage::HostToDevice, {}, std::move(error));
    programDataMemory = *memory;
  }

  // Invocation: one allocation for inputs, outputs, per-Tile workspace,
  // profile records, transport status and (TileRow) pointer rows.
  llvm::Expected<BoardDeviceMemory> invocationMemory = driver.allocate(
      capacityPlan->invocationBytes, capacityPlan->invocationAlignment);
  if (!invocationMemory)
    return fail(BoardRuntimeStage::ResourceAllocation, {},
                invocationMemory.takeError());
  allocations.push_back(
      {BoardLiveAllocation::Kind::Invocation, {}, *invocationMemory});
  result.completedStages.push_back(BoardRuntimeStage::ResourceAllocation);

  uint64_t invocationBase = invocationMemory->value;
  uint64_t programDataBase = programDataMemory ? programDataMemory->value : 0;
  auto checkedInvocationAddress = [&](const RuntimePlannedRange &range) {
    return invocationBase + range.offset;
  };

  // Host-to-device: exact input bytes, profiler records, and the Direct-DTE
  // status poison pattern. Workspace is allocation-only storage.
  for (const ExternalPortRecord &port : manifest.inputs) {
    const RuntimePlannedRange &range =
        capacityPlan->inputRanges[port.id.getValue()];
    const BoardRuntimeBinding *binding =
        bindingsByPort.lookup(port.id.getValue());
    if (llvm::Error error = driver.copyHostToDevice(
            BoardDeviceMemory{checkedInvocationAddress(range)}, binding->bytes))
      return fail(BoardRuntimeStage::HostToDevice, {}, std::move(error));
  }
  for (const RuntimePlannedRange &range :
       capacityPlan->zeroInitializedSharedWorkspaceRanges) {
    std::vector<uint8_t> zero(range.bytes, 0);
    if (llvm::Error error = driver.copyHostToDevice(
            BoardDeviceMemory{checkedInvocationAddress(range)}, zero))
      return fail(BoardRuntimeStage::HostToDevice, {}, std::move(error));
  }
  if (request.profilerRecordBytes) {
    for (auto [tile, ranges, image] :
         llvm::zip(capacityPlan->tiles, capacityPlan->tileRanges,
                   *request.profilerRecordBytes)) {
      if (!ranges.profileRecord)
        return fail(BoardRuntimeStage::HostToDevice, locationFor(tile),
                    invalid("profiler record range is missing"));
      if (llvm::Error error = driver.copyHostToDevice(
              BoardDeviceMemory{
                  checkedInvocationAddress(*ranges.profileRecord)},
              image))
        return fail(BoardRuntimeStage::HostToDevice, locationFor(tile),
                    std::move(error));
    }
  }
  if (hasDirectDTETransport) {
    std::vector<uint8_t> poison(runtime::kDirectDTEStatusStorageBytes, 0xff);
    for (auto [tile, ranges] :
         llvm::zip(capacityPlan->tiles, capacityPlan->tileRanges)) {
      if (!ranges.transportStatus)
        return fail(BoardRuntimeStage::HostToDevice, locationFor(tile),
                    invalid("transport status range is missing"));
      if (llvm::Error error = driver.copyHostToDevice(
              BoardDeviceMemory{
                  checkedInvocationAddress(*ranges.transportStatus)},
              poison))
        return fail(BoardRuntimeStage::HostToDevice, locationFor(tile),
                    std::move(error));
    }
  }

  // Pointer rows: every Tile argument resolves to program-data base +
  // file offset or invocation base + planned offset. TileRow rows live in
  // device child ranges; TileMajor rows travel in the kernel packet.
  const auto resolveAddress =
      [&](const RuntimeArgumentAddress &argument) -> uint64_t {
    if (argument.base == RuntimeArgumentAddressBase::ProgramData)
      return programDataBase + argument.offset;
    return invocationBase + argument.offset;
  };
  if (kernelLaunch.entryABI == KernelEntryABI::TileRowPointerTable) {
    if (capacityPlan->pointerRows.size() != capacityPlan->tiles.size())
      return fail(BoardRuntimeStage::HostToDevice, {},
                  invalid("Tile-row allocation domain is incomplete"));
    tileRowAddresses.reserve(capacityPlan->tiles.size());
    for (auto [tileIndex, tile] : llvm::enumerate(capacityPlan->tiles)) {
      std::vector<uint64_t> row;
      row.reserve(tile.argumentAddresses.size());
      for (const RuntimeArgumentAddress &argument : tile.argumentAddresses)
        row.push_back(resolveAddress(argument));
      const RuntimePlannedRange &range = capacityPlan->pointerRows[tileIndex];
      if (range.bytes != row.size() * sizeof(uint64_t))
        return fail(BoardRuntimeStage::HostToDevice, locationFor(tile),
                    invalid("Tile-row plan does not match the "
                            "resolved argument count"));
      tileRowAddresses.push_back(checkedInvocationAddress(range));
      llvm::ArrayRef<uint8_t> rowBytes(
          reinterpret_cast<const uint8_t *>(row.data()),
          row.size() * sizeof(uint64_t));
      if (llvm::Error error = driver.copyHostToDevice(
              BoardDeviceMemory{tileRowAddresses.back()}, rowBytes))
        return fail(BoardRuntimeStage::HostToDevice, locationFor(tile),
                    std::move(error));
    }
  }
  result.completedStages.push_back(BoardRuntimeStage::HostToDevice);

  llvm::ArrayRef<RuntimeLaunchPhaseRole> launchPhases =
      manifest.launch.getPhases();
  if (capacityPlan->tiles.empty() || launchPhases.empty())
    return fail(BoardRuntimeStage::Validation, {},
                invalid("runtime launch has no planned Tile or phase"));

  for (const VerifiedModuleSnapshot &snapshot : moduleSnapshots) {
    llvm::Expected<BoardModuleHandle> loaded =
        driver.loadModule(snapshot.bytes);
    if (!loaded)
      return fail(BoardRuntimeStage::ModuleLoad, snapshot.diagnosticLocation,
                  loaded.takeError());
    liveModules.push_back({snapshot.module, *loaded});
  }
  result.completedStages.push_back(BoardRuntimeStage::ModuleLoad);

  std::vector<BoardTileLaunch> baseLaunches;
  baseLaunches.reserve(capacityPlan->tiles.size());
  for (auto [tileIndex, tile] : llvm::enumerate(capacityPlan->tiles)) {
    BoardTileLaunch launch;
    launch.cardId = tile.cardId;
    launch.tileId = tile.tileId;
    launch.launchSlot = tile.launchSlot;
    launch.entry = tile.entry;
    if (kernelLaunch.entryABI == KernelEntryABI::TileRowPointerTable) {
      if (tileIndex >= tileRowAddresses.size())
        return fail(BoardRuntimeStage::Launch, locationFor(tile),
                    invalid("Tile-row launch storage is missing"));
      launch.arguments.push_back(tileRowAddresses[tileIndex]);
      baseLaunches.push_back(std::move(launch));
      continue;
    }
    launch.arguments.reserve(tile.argumentAddresses.size());
    for (const RuntimeArgumentAddress &argument : tile.argumentAddresses)
      launch.arguments.push_back(resolveAddress(argument));
    baseLaunches.push_back(std::move(launch));
  }

  std::optional<std::chrono::steady_clock::time_point> submissionBegin;
  std::optional<BoardCompletionDeadline> deadline;
  auto beginSubmissionWindow = [&] {
    if (submissionBegin)
      return;
    submissionBegin = std::chrono::steady_clock::now();
    deadline = *submissionBegin +
               std::chrono::milliseconds(request.completionTimeoutMilliseconds);
  };
  uint64_t maximumPollGapNanoseconds = 0;
  uint64_t hostSubmitNanoseconds = 0;
  std::optional<uint64_t> deviceExecutionNanoseconds;
  if (request.deviceTimingPolicy == BoardDeviceTimingPolicy::StreamEvents)
    deviceExecutionNanoseconds = 0;
  auto accumulateHostSubmit =
      [&](std::chrono::steady_clock::time_point submitBegin,
          std::chrono::steady_clock::time_point submitEnd) -> llvm::Error {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             submitEnd - submitBegin)
                             .count();
    if (elapsed < 0)
      return invalid(
          "host steady clock moved backwards during provider submission");
    const uint64_t elapsedNanoseconds = static_cast<uint64_t>(elapsed);
    if (elapsedNanoseconds >
        std::numeric_limits<uint64_t>::max() - hostSubmitNanoseconds)
      return invalid("aggregate provider submission time overflows");
    hostSubmitNanoseconds += elapsedNanoseconds;
    return llvm::Error::success();
  };
  auto accumulateDeviceTiming =
      [&](const BoardCompletionObservation &observation) -> llvm::Error {
    const bool requested =
        request.deviceTimingPolicy == BoardDeviceTimingPolicy::StreamEvents;
    if (requested != observation.deviceExecutionNanoseconds.has_value())
      return invalid(
          requested
              ? "provider omitted requested same-stream device timing"
              : "provider returned same-stream device timing when disabled");
    if (!requested)
      return llvm::Error::success();
    if (*observation.deviceExecutionNanoseconds >
        std::numeric_limits<uint64_t>::max() - *deviceExecutionNanoseconds)
      return invalid("aggregate device execution time overflows");
    *deviceExecutionNanoseconds += *observation.deviceExecutionNanoseconds;
    return llvm::Error::success();
  };
  {
    std::vector<llvm::DenseMap<uint64_t, BoardFunctionHandle>> functionsByPhase;
    functionsByPhase.reserve(launchPhases.size());
    for (auto [phaseIndex, phaseRole] : llvm::enumerate(launchPhases)) {
      llvm::DenseMap<uint64_t, BoardFunctionHandle> functionsByModule;
      for (const LiveModule &liveModule : liveModules) {
        auto firstTile =
            llvm::find_if(capacityPlan->tiles, [&](const auto &tile) {
              return tile.module == liveModule.moduleRecord->id;
            });
        if (firstTile == capacityPlan->tiles.end())
          return fail(BoardRuntimeStage::EntryResolve, {},
                      invalid("loaded module has no typed Tile interface"));
        if (firstTile->phases.size() != launchPhases.size() ||
            firstTile->phases[phaseIndex].role != phaseRole)
          return fail(
              BoardRuntimeStage::EntryResolve, locationFor(*firstTile),
              invalid("Tile launch phase plan does not match manifest"));
        llvm::Expected<BoardFunctionHandle> function = driver.resolveEntry(
            liveModule.module, firstTile->phases[phaseIndex].symbol);
        if (!function)
          return fail(BoardRuntimeStage::EntryResolve, locationFor(*firstTile),
                      function.takeError());
        functionsByModule[liveModule.moduleRecord->id.getValue()] = *function;
      }
      functionsByPhase.push_back(std::move(functionsByModule));
    }

    std::vector<std::vector<BoardTileLaunch>> launchesByPhase;
    launchesByPhase.reserve(launchPhases.size());
    for (auto [phaseIndex, phaseRole] : llvm::enumerate(launchPhases)) {
      std::vector<BoardTileLaunch> phaseLaunches = baseLaunches;
      for (auto [tileIndex, tile] : llvm::enumerate(capacityPlan->tiles)) {
        if (tile.phases.size() != launchPhases.size() ||
            tile.phases[phaseIndex].role != phaseRole)
          return fail(
              BoardRuntimeStage::EntryResolve, locationFor(tile),
              invalid("Tile launch phase plan does not match manifest"));
        auto function =
            functionsByPhase[phaseIndex].find(tile.module.getValue());
        if (function == functionsByPhase[phaseIndex].end())
          return fail(BoardRuntimeStage::EntryResolve, locationFor(tile),
                      invalid("Tile phase export was not resolved"));
        phaseLaunches[tileIndex].function = function->second;
      }
      launchesByPhase.push_back(std::move(phaseLaunches));
    }

    for (auto [phaseIndex, phaseRole] : llvm::enumerate(launchPhases)) {
      beginSubmissionWindow();
      const auto submitBegin = std::chrono::steady_clock::now();
      llvm::Error submitError = driver.submitKernelPhase(
          kernelLaunch.form, phaseRole, launchesByPhase[phaseIndex],
          request.deviceTimingPolicy);
      const auto submitEnd = std::chrono::steady_clock::now();
      if (submitError)
        return fail(BoardRuntimeStage::Launch, {}, std::move(submitError));
      submissionLive = true;
      if (llvm::Error error = accumulateHostSubmit(submitBegin, submitEnd)) {
        driver.quarantine();
        return fail(BoardRuntimeStage::Launch, {}, std::move(error));
      }
      llvm::Expected<BoardCompletionObservation> observation =
          driver.waitCurrentSubmission(*deadline,
                                       request.completionObservationPolicy);
      if (!observation)
        return fail(BoardRuntimeStage::Completion, {}, observation.takeError());
      maximumPollGapNanoseconds = std::max(
          maximumPollGapNanoseconds, observation->maximumPollGapNanoseconds);
      if (llvm::Error error = accumulateDeviceTiming(*observation))
        return fail(BoardRuntimeStage::Completion, {}, std::move(error));
    }
  }

  if (!submissionBegin)
    return fail(BoardRuntimeStage::Launch, {},
                invalid("runtime launch did not submit any phase"));
  const auto completionEnd = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           completionEnd - *submissionBegin)
                           .count();
  if (elapsed < 0)
    return fail(BoardRuntimeStage::Completion, {},
                invalid("host steady clock moved backwards"));
  result.launchToCompletionNanoseconds = static_cast<uint64_t>(elapsed);
  result.hostSubmitNanoseconds = hostSubmitNanoseconds;
  result.deviceExecutionNanoseconds = deviceExecutionNanoseconds;
  result.completionObservationResolutionNanoseconds = maximumPollGapNanoseconds;
  result.completedStages.push_back(BoardRuntimeStage::EntryResolve);
  result.completedStages.push_back(BoardRuntimeStage::Launch);
  result.completedStages.push_back(BoardRuntimeStage::Completion);

  if (hasDirectDTETransport) {
    size_t observedStatuses = 0;
    for (auto [tile, ranges] :
         llvm::zip(capacityPlan->tiles, capacityPlan->tileRanges)) {
      if (!ranges.transportStatus ||
          !directDTETiles.contains(tile.tileId.getValue()))
        continue;
      ++observedStatuses;
      std::vector<uint8_t> statusBytes(ranges.transportStatus->bytes);
      if (llvm::Error error = driver.copyDeviceToHost(
              statusBytes, BoardDeviceMemory{checkedInvocationAddress(
                               *ranges.transportStatus)})) {
        driver.quarantine();
        return fail(BoardRuntimeStage::DeviceToHost, locationFor(tile),
                    std::move(error));
      }
      uint32_t status = kDirectDTEStatusPoison;
      if (statusBytes.size() == kDirectDTEStatusStorageBytes)
        std::memcpy(&status, statusBytes.data() + kDirectDTEStatusValueOffset,
                    kDirectDTEStatusValueBytes);
      if (status != static_cast<uint32_t>(DirectDTEStatusValue::Success)) {
        driver.quarantine();
        return fail(BoardRuntimeStage::Completion, locationFor(tile),
                    invalid("Direct DTE terminal status is not success: " +
                            llvm::Twine(status)));
      }
    }
    if (observedStatuses != directDTETiles.size()) {
      driver.quarantine();
      return fail(BoardRuntimeStage::Completion, {},
                  invalid("Direct DTE terminal status domain is incomplete"));
    }
  }

  for (const ExternalPortRecord &port : manifest.outputs) {
    const RuntimePlannedRange &range =
        capacityPlan->outputRanges[port.id.getValue()];
    BoardRuntimeOutput output{port.id, std::vector<uint8_t>(range.bytes)};
    if (llvm::Error error = driver.copyDeviceToHost(
            output.bytes, BoardDeviceMemory{checkedInvocationAddress(range)})) {
      if (hasDirectDTETransport)
        driver.quarantine();
      return fail(BoardRuntimeStage::DeviceToHost, {}, std::move(error));
    }
    result.outputs.push_back(std::move(output));
  }
  if (request.profilerRecordBytes) {
    for (auto [tile, ranges] :
         llvm::zip(capacityPlan->tiles, capacityPlan->tileRanges)) {
      if (!ranges.profileRecord)
        return fail(BoardRuntimeStage::DeviceToHost, locationFor(tile),
                    invalid("profiler record range is missing"));
      BoardRuntimeProfilerOutput output{
          tile.launchSlot, std::vector<uint8_t>(ranges.profileRecord->bytes)};
      if (llvm::Error error = driver.copyDeviceToHost(
              output.bytes, BoardDeviceMemory{checkedInvocationAddress(
                                *ranges.profileRecord)})) {
        if (hasDirectDTETransport)
          driver.quarantine();
        return fail(BoardRuntimeStage::DeviceToHost, locationFor(tile),
                    std::move(error));
      }
      result.profilerOutputs.push_back(std::move(output));
    }
  }
  result.completedStages.push_back(BoardRuntimeStage::DeviceToHost);

  if (llvm::Error error = cleanup())
    return std::move(error);
  result.completedStages.push_back(BoardRuntimeStage::Cleanup);
  return result;
}

} // namespace

QualifiedBoardRuntimeSession::QualifiedBoardRuntimeSession(
    BoardRuntimeDriver &driver, uint32_t deviceId, uint32_t qualifiedTileCount,
    BoardDeviceQualification qualification, BoardDeviceInfo device)
    : driver(&driver), deviceId(deviceId),
      qualifiedTileCount(qualifiedTileCount),
      qualification(std::move(qualification)), device(std::move(device)),
      usable(true) {}

QualifiedBoardRuntimeSession::QualifiedBoardRuntimeSession(
    QualifiedBoardRuntimeSession &&other) noexcept
    : driver(std::exchange(other.driver, nullptr)),
      deviceId(std::exchange(other.deviceId, 0)),
      qualifiedTileCount(std::exchange(other.qualifiedTileCount, 0)),
      qualification(std::move(other.qualification)),
      device(std::move(other.device)),
      usable(std::exchange(other.usable, false)) {}

QualifiedBoardRuntimeSession &QualifiedBoardRuntimeSession::operator=(
    QualifiedBoardRuntimeSession &&other) noexcept {
  if (this == &other)
    return *this;
  driver = std::exchange(other.driver, nullptr);
  deviceId = std::exchange(other.deviceId, 0);
  qualifiedTileCount = std::exchange(other.qualifiedTileCount, 0);
  qualification = std::move(other.qualification);
  device = std::move(other.device);
  usable = std::exchange(other.usable, false);
  return *this;
}

llvm::Expected<BoardRuntimeInvocationResult>
executeBoardInvocationInSession(const ExecutablePackage &package,
                                BoardRuntimeInvocationRequest request,
                                QualifiedBoardRuntimeSession &session) {
  if (!session.driver || !session.usable)
    return boardError(BoardRuntimeStage::DeviceSelection, {},
                      "qualified board runtime session is no longer usable",
                      BoardRuntimeContextState::Poisoned);
  if (request.deviceId != session.deviceId ||
      !qualificationMatches(request.qualification, session.qualification))
    return boardError(
        BoardRuntimeStage::Validation, {},
        "board invocation does not match the qualified session identity");
  if (package.getManifest().tileCount !=
      static_cast<int64_t>(session.qualifiedTileCount))
    return boardError(
        BoardRuntimeStage::Validation, {},
        "package Tile domain does not match the qualified board runtime "
        "session");
  if (session.driver->getContextState() == BoardRuntimeContextState::Poisoned) {
    session.usable = false;
    return boardError(BoardRuntimeStage::DeviceSelection, {},
                      "TX provider is already quarantined",
                      BoardRuntimeContextState::Poisoned);
  }
  return executeBoardInvocationImpl(
      package, std::move(request), *session.driver, &session.device,
      session.qualifiedTileCount, &session.usable);
}

llvm::Expected<
    std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
executeBoardInvocationAndStartSession(const ExecutablePackage &package,
                                      BoardRuntimeInvocationRequest request,
                                      BoardRuntimeDriver &driver) {
  const uint32_t deviceId = request.deviceId;
  const uint32_t tileCount =
      static_cast<uint32_t>(package.getManifest().tileCount);
  BoardDeviceQualification qualification = request.qualification;
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocationImpl(package, std::move(request), driver,
                                 /*qualifiedDevice=*/nullptr,
                                 /*qualifiedTileCount=*/0,
                                 /*qualifiedSessionUsable=*/nullptr);
  if (!result)
    return result.takeError();
  QualifiedBoardRuntimeSession session(
      driver, deviceId, tileCount, std::move(qualification), result->device);
  return std::pair(std::move(*result), std::move(session));
}

llvm::Expected<BoardRuntimeInvocationResult>
executeBoardInvocation(const ExecutablePackage &package,
                       BoardRuntimeInvocationRequest request,
                       BoardRuntimeDriver &driver) {
  return executeBoardInvocationImpl(package, std::move(request), driver,
                                    /*qualifiedDevice=*/nullptr,
                                    /*qualifiedTileCount=*/0,
                                    /*qualifiedSessionUsable=*/nullptr);
}

} // namespace wafer::runtime
