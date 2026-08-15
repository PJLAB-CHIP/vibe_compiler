//===- BoardRuntime.cpp - Verified package board execution --------------===//

#include "Wafer/Runtime/BoardRuntime.h"

#include "PackageManifestInternal.h"
#include "Wafer/ABI/Tx81ProfilerABI.h"

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
#include <vector>

namespace wafer::runtime {
namespace {

constexpr uint64_t boardRuntimeFreeMemoryReserve = 64ULL * 1024 * 1024;

constexpr CardId noCard{-1};
constexpr TileId noTile{-1};

struct DiagnosticLocation {
  CardId cardId = noCard;
  TileId tileId = noTile;
  LaunchSlotId launchSlot;
  EntryId entry;
};

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

DiagnosticLocation locationFor(const PackageResourceRecord &resource,
                               EntryId entry = {}) {
  if (const auto *card = std::get_if<CardResourceScope>(&resource.scope))
    return {card->cardId, noTile, LaunchSlotId(), entry};
  const auto &tile = std::get<TileResourceScope>(resource.scope);
  return {tile.cardId, tile.tileId, LaunchSlotId(), entry};
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
      return boardError(
          BoardRuntimeStage::Validation, locationFor(tile),
          "package Tile/launch-slot binding is absent from the "
          "qualified device inventory");
  }
  return llvm::Error::success();
}

bool isProfilerWorkspace(const PackageResourceRecord &resource) {
  const bool exactRecordBytes =
      resource.bytes == WAFER_TX81_PROFILER_MIN_BUFFER_BYTES ||
      resource.bytes == WAFER_TX81_PROFILER_TRACE_BUFFER_BYTES;
  return !resource.hostVisible &&
         resource.role == PackageResourceRole::Workspace &&
         resource.roleIndex == 1 && resource.type.dtype == "u8" &&
         resource.bytes <=
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
         resource.type.shape ==
             std::vector<int64_t>{static_cast<int64_t>(resource.bytes)} &&
         exactRecordBytes &&
         resource.alignment == WAFER_TX81_PROFILER_BUFFER_ALIGNMENT &&
         resource.access == PackageAccessMode::ReadWrite;
}

llvm::Expected<std::vector<uint8_t>>
readVerifiedModule(llvm::StringRef packageRoot,
                   const PackageModuleRecord &module,
                   DiagnosticLocation location) {
  llvm::SmallString<256> path(packageRoot);
  llvm::sys::path::append(path, module.relativePath);
  if (llvm::sys::fs::get_file_type(path, /*Follow=*/false) !=
      llvm::sys::fs::file_type::regular_file)
    return boardError(BoardRuntimeStage::ModuleLoad, location,
                      "package module is no longer a regular file");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return boardError(BoardRuntimeStage::ModuleLoad, location,
                      "failed to reopen verified package module: " +
                          buffer.getError().message());
  llvm::StringRef bytes = (*buffer)->getBuffer();
  llvm::SHA256 hasher;
  hasher.update(bytes);
  std::string digest =
      "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
  if (digest != module.digest)
    return boardError(BoardRuntimeStage::ModuleLoad, location,
                      "package module changed after manifest verification");
  return std::vector<uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()),
                              reinterpret_cast<const uint8_t *>(bytes.data()) +
                                  bytes.size());
}

struct LiveAllocation {
  const PackageResourceRecord *resource = nullptr;
  DiagnosticLocation location;
  BoardDeviceMemory memory;
};

struct LiveTileArgumentRow {
  DiagnosticLocation location;
  BoardDeviceMemory memory;
};

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

llvm::Expected<BoardRuntimeInvocationResult> executeBoardInvocationImpl(
    const VerifiedPackageManifest &package, llvm::StringRef packageRoot,
    BoardRuntimeInvocationRequest request, BoardRuntimeDriver &driver,
    const BoardDeviceInfo *qualifiedDevice, uint32_t qualifiedTileCount,
    bool *qualifiedSessionUsable) {
  const PackageManifest &manifest = package.getManifest();
  const KernelRuntimeLaunchContract *kernelLaunch = manifest.launch.getKernel();
  const bool modelLaunch = manifest.launch.getModel() != nullptr;
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

  llvm::DenseMap<uint64_t, BoardRuntimeBinding *> bindingsByResource;
  for (BoardRuntimeBinding &binding : request.bindings) {
    uint64_t id = binding.resource.getValue();
    if (!binding.resource.isValid() || bindingsByResource.count(id))
      return boardError(BoardRuntimeStage::Validation, {},
                        "invocation has a duplicate or invalid ResourceId");
    const PackageResourceRecord *resource =
        detail::findResource(manifest.resources, binding.resource);
    if (!resource || !resource->hostVisible)
      return boardError(BoardRuntimeStage::Validation, {},
                        "invocation binds an unknown or internal ResourceId");
    if (binding.bytes.size() != resource->bytes)
      return boardError(BoardRuntimeStage::Validation, locationFor(*resource),
                        "invocation buffer byte count is not exact");
    bindingsByResource[id] = &binding;
  }

  llvm::DenseSet<uint64_t> profilerResourceIds;
  llvm::DenseSet<int64_t> profilerResourceTiles;
  auto isBoundAsFinalProfilerWorkspace =
      [&](const PackageResourceRecord &resource) {
        if (!isProfilerWorkspace(resource))
          return false;
        const auto &scope = std::get<TileResourceScope>(resource.scope);
        auto entry = llvm::find_if(
            manifest.entries, [&](const PackageEntrypointRecord &candidate) {
              return candidate.cardId == scope.cardId &&
                     candidate.tileId == scope.tileId;
            });
        return entry != manifest.entries.end() && !entry->slots.empty() &&
               entry->slots.back().resource == resource.id &&
               entry->slots.back().access == resource.access;
      };
  for (const PackageResourceRecord &resource : manifest.resources) {
    if (!isBoundAsFinalProfilerWorkspace(resource))
      continue;
    profilerResourceIds.insert(resource.id.getValue());
    profilerResourceTiles.insert(
        std::get<TileResourceScope>(resource.scope).tileId.getValue());
  }

  llvm::DenseMap<uint64_t, BoardRuntimeBinding *> profilerByResource;
  llvm::DenseSet<int64_t> profilerTiles;
  for (BoardRuntimeBinding &binding : request.profilerBindings) {
    uint64_t id = binding.resource.getValue();
    if (!binding.resource.isValid() || profilerByResource.count(id) ||
        bindingsByResource.count(id))
      return boardError(
          BoardRuntimeStage::Validation, {},
          "profiler invocation has a duplicate or invalid ResourceId");
    const PackageResourceRecord *resource =
        detail::findResource(manifest.resources, binding.resource);
    if (!resource || !isBoundAsFinalProfilerWorkspace(*resource))
      return boardError(
          BoardRuntimeStage::Validation, {},
          "profiler invocation binds a resource outside the exact internal "
          "record contract");
    if (binding.bytes.size() != resource->bytes)
      return boardError(BoardRuntimeStage::Validation, locationFor(*resource),
                        "profiler invocation buffer byte count is not exact");
    const auto &scope = std::get<TileResourceScope>(resource->scope);
    if (!profilerTiles.insert(scope.tileId.getValue()).second)
      return boardError(BoardRuntimeStage::Validation, locationFor(*resource),
                        "profiler invocation binds more than one record for "
                        "one Tile");
    profilerByResource[id] = &binding;
  }
  if (!profilerResourceIds.empty() || !request.profilerBindings.empty()) {
    if (manifest.tileCount != WAFER_TX81_PROFILER_TILE_COUNT ||
        profilerResourceIds.size() != WAFER_TX81_PROFILER_TILE_COUNT ||
        profilerResourceTiles.size() != WAFER_TX81_PROFILER_TILE_COUNT ||
        request.profilerBindings.size() != WAFER_TX81_PROFILER_TILE_COUNT ||
        profilerTiles.size() != WAFER_TX81_PROFILER_TILE_COUNT)
      return boardError(
          BoardRuntimeStage::Validation, {},
          "profiler invocation requires all-and-only Tiles 0..15");
    for (int64_t tileId = 0; tileId < WAFER_TX81_PROFILER_TILE_COUNT; ++tileId)
      if (!profilerResourceTiles.contains(tileId) ||
          !profilerTiles.contains(tileId))
        return boardError(
            BoardRuntimeStage::Validation,
            {CardId(0), TileId(tileId), LaunchSlotId(), {}},
            "profiler invocation requires all-and-only Tiles 0..15");
    for (uint64_t resourceId : profilerResourceIds)
      if (!profilerByResource.count(resourceId))
        return boardError(
            BoardRuntimeStage::Validation, {},
            "profiler invocation omits an internal profiler ResourceId");
  }

  std::vector<RuntimeInvocationBinding> invocationBindings;
  for (const PackageResourceRecord &resource : manifest.resources) {
    if (!resource.hostVisible)
      continue;
    if (!bindingsByResource.count(resource.id.getValue()))
      return boardError(BoardRuntimeStage::Validation, locationFor(resource),
                        "invocation omits a host-visible ResourceId");
    invocationBindings.push_back({resource.id, resource.bytes,
                                  resource.alignment, resource.access, true});
  }
  if (invocationBindings.size() != request.bindings.size())
    return boardError(BoardRuntimeStage::Validation, {},
                      "invocation bindings are not all-and-only for package");

  const RuntimeEnvironment &providerEnvironment =
      driver.getProviderEnvironment();
  llvm::Expected<RuntimeInvocationPlan> semanticPlan =
      planRuntimeInvocation(package, invocationBindings, providerEnvironment);
  if (!semanticPlan)
    return wrapDriverError(BoardRuntimeStage::Validation, {},
                           semanticPlan.takeError());

  std::vector<VerifiedModuleSnapshot> moduleSnapshots;
  moduleSnapshots.reserve(manifest.modules.size());
  for (const PackageModuleRecord &module : manifest.modules) {
    auto firstTile = llvm::find_if(semanticPlan->tiles, [&](const auto &tile) {
      return tile.module == module.id;
    });
    if (firstTile == semanticPlan->tiles.end())
      return boardError(BoardRuntimeStage::Validation, {},
                        "package contains an unreferenced module");
    llvm::Expected<std::vector<uint8_t>> bytes =
        readVerifiedModule(packageRoot, module, locationFor(*firstTile));
    if (!bytes)
      return bytes.takeError();
    moduleSnapshots.push_back(
        {&module, locationFor(*firstTile), std::move(*bytes)});
  }
  llvm::sort(moduleSnapshots, [](const auto &lhs, const auto &rhs) {
    return lhs.diagnosticLocation.launchSlot.getValue() <
           rhs.diagnosticLocation.launchSlot.getValue();
  });
  auto findSnapshot = [&](ModuleId module) -> const VerifiedModuleSnapshot * {
    auto iterator = llvm::find_if(moduleSnapshots, [&](const auto &snapshot) {
      return snapshot.module->id == module;
    });
    return iterator == moduleSnapshots.end() ? nullptr : &*iterator;
  };

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
  llvm::Expected<RuntimeInvocationPlan> capacityPlan =
      planRuntimeInvocation(package, invocationBindings, capacityEnvironment);
  if (!capacityPlan)
    return wrapDriverError(BoardRuntimeStage::Validation, {},
                           capacityPlan.takeError());
  if (llvm::Error error = verifyPlannedTileBindings(device, *capacityPlan))
    return std::move(error);

  auto findFirstResourceUse =
      [&](ResourceId resource) -> const RuntimeSessionPlan * {
    auto tile = llvm::find_if(capacityPlan->tiles, [&](const auto &candidate) {
      return llvm::is_contained(candidate.launchOrder, resource);
    });
    return tile == capacityPlan->tiles.end() ? nullptr : &*tile;
  };

  uint64_t allocationBytes = 0;
  for (const PackageResourceRecord &resource : manifest.resources) {
    const RuntimeSessionPlan *firstUse = findFirstResourceUse(resource.id);
    if (!firstUse)
      return boardError(BoardRuntimeStage::Validation, locationFor(resource),
                        "package resource has no runtime launch consumer");
    if (resource.bytes > std::numeric_limits<uint64_t>::max() - allocationBytes)
      return boardError(BoardRuntimeStage::Validation, locationFor(*firstUse),
                        "aggregate board allocation byte count overflows");
    allocationBytes += resource.bytes;
  }
  if (kernelLaunch &&
      kernelLaunch->entryABI == KernelEntryABI::TileRowPointerTable) {
    for (const RuntimeSessionPlan &tile : capacityPlan->tiles) {
      const uint64_t rowBytes =
          static_cast<uint64_t>(tile.launchOrder.size()) * sizeof(uint64_t);
      if (rowBytes > std::numeric_limits<uint64_t>::max() - allocationBytes)
        return boardError(BoardRuntimeStage::Validation, locationFor(tile),
                          "aggregate Tile-row pointer storage overflows");
      allocationBytes += rowBytes;
    }
  }
  for (const VerifiedModuleSnapshot &snapshot : moduleSnapshots) {
    uint64_t moduleBytes = snapshot.bytes.size();
    if (moduleBytes > std::numeric_limits<uint64_t>::max() - allocationBytes)
      return boardError(BoardRuntimeStage::Validation, {},
                        "aggregate board module byte count overflows");
    allocationBytes += moduleBytes;
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

  std::vector<LiveAllocation> allocations;
  std::vector<LiveTileArgumentRow> tileArgumentRows;
  std::vector<LiveModule> liveModules;
  std::optional<BoardGraphHandle> liveGraph;
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
    if (liveGraph) {
      if (llvm::Error error = driver.unloadGraph(*liveGraph)) {
        BoardRuntimeContextState state = observeProviderState();
        llvm::Error wrapped = wrapDriverError(BoardRuntimeStage::Cleanup, {},
                                              std::move(error), state);
        if (state == BoardRuntimeContextState::Poisoned)
          return llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
        cleanupError =
            llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
      }
      liveGraph.reset();
    }
    for (LiveTileArgumentRow &row : llvm::reverse(tileArgumentRows)) {
      if (llvm::Error error = driver.free(row.memory)) {
        BoardRuntimeContextState state = observeProviderState();
        llvm::Error wrapped = wrapDriverError(
            BoardRuntimeStage::Cleanup, row.location, std::move(error), state);
        if (state == BoardRuntimeContextState::Poisoned)
          return llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
        cleanupError =
            llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
      }
    }
    tileArgumentRows.clear();
    for (LiveAllocation &allocation : llvm::reverse(allocations)) {
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

  llvm::DenseMap<uint64_t, BoardDeviceMemory> memoryByResource;
  for (const PackageResourceRecord &resource : manifest.resources) {
    const RuntimeSessionPlan *firstUse = findFirstResourceUse(resource.id);
    if (!firstUse)
      return fail(
          BoardRuntimeStage::ResourceAllocation, locationFor(resource),
          detail::invalid("package resource has no runtime launch consumer"));
    llvm::Expected<BoardDeviceMemory> memory =
        driver.allocate(resource.bytes, resource.alignment);
    if (!memory)
      return fail(BoardRuntimeStage::ResourceAllocation, locationFor(*firstUse),
                  memory.takeError());
    allocations.push_back({&resource, locationFor(*firstUse), *memory});
    memoryByResource[resource.id.getValue()] = *memory;
  }
  if (kernelLaunch &&
      kernelLaunch->entryABI == KernelEntryABI::TileRowPointerTable) {
    tileArgumentRows.reserve(capacityPlan->tiles.size());
    for (const RuntimeSessionPlan &tile : capacityPlan->tiles) {
      const uint64_t rowBytes =
          static_cast<uint64_t>(tile.launchOrder.size()) * sizeof(uint64_t);
      llvm::Expected<BoardDeviceMemory> memory =
          driver.allocate(rowBytes, alignof(uint64_t));
      if (!memory)
        return fail(BoardRuntimeStage::ResourceAllocation, locationFor(tile),
                    memory.takeError());
      tileArgumentRows.push_back({locationFor(tile), *memory});
    }
  }
  result.completedStages.push_back(BoardRuntimeStage::ResourceAllocation);

  for (const LiveAllocation &allocation : allocations) {
    const PackageResourceRecord &resource = *allocation.resource;
    // Workspace is allocation-only storage.  Materializing a host-sized zero
    // buffer and uploading it is both semantically unnecessary and
    // prohibitive for large compiler-managed DDR arenas.
    if (resource.role == PackageResourceRole::Workspace) {
      auto profiler = profilerByResource.find(resource.id.getValue());
      if (profiler == profilerByResource.end())
        continue;
      if (llvm::Error error = driver.copyHostToDevice(allocation.memory,
                                                      profiler->second->bytes))
        return fail(BoardRuntimeStage::HostToDevice, allocation.location,
                    std::move(error));
      continue;
    }
    llvm::ArrayRef<uint8_t> source;
    std::vector<uint8_t> zeros;
    auto binding = bindingsByResource.find(resource.id.getValue());
    if (binding != bindingsByResource.end()) {
      source = binding->second->bytes;
    } else {
      const uint8_t initialValue =
          resource.role == PackageResourceRole::TransportStatus ? 0xff : 0;
      zeros.assign(static_cast<size_t>(resource.bytes), initialValue);
      source = zeros;
    }
    if (llvm::Error error = driver.copyHostToDevice(allocation.memory, source))
      return fail(BoardRuntimeStage::HostToDevice, allocation.location,
                  std::move(error));
  }
  if (!tileArgumentRows.empty()) {
    if (tileArgumentRows.size() != capacityPlan->tiles.size())
      return fail(BoardRuntimeStage::HostToDevice, {},
                  detail::invalid("Tile-row allocation domain is incomplete"));
    for (auto [tileIndex, tile] : llvm::enumerate(capacityPlan->tiles)) {
      std::vector<uint64_t> row;
      row.reserve(tile.launchOrder.size());
      for (ResourceId resource : tile.launchOrder) {
        auto memory = memoryByResource.find(resource.getValue());
        if (memory == memoryByResource.end())
          return fail(
              BoardRuntimeStage::HostToDevice, locationFor(tile),
              detail::invalid("Tile-row slot has no device allocation"));
        row.push_back(static_cast<uint64_t>(memory->second.value));
      }
      llvm::ArrayRef<uint8_t> rowBytes(
          reinterpret_cast<const uint8_t *>(row.data()),
          row.size() * sizeof(uint64_t));
      if (llvm::Error error = driver.copyHostToDevice(
              tileArgumentRows[tileIndex].memory, rowBytes))
        return fail(BoardRuntimeStage::HostToDevice, locationFor(tile),
                    std::move(error));
    }
  }
  result.completedStages.push_back(BoardRuntimeStage::HostToDevice);

  llvm::ArrayRef<RuntimeLaunchPhaseRole> launchPhases =
      manifest.launch.getPhases();
  if (capacityPlan->tiles.empty() || launchPhases.empty())
    return fail(BoardRuntimeStage::Validation, {},
                detail::invalid("runtime launch has no planned Tile or phase"));

  if (modelLaunch) {
    const RuntimeSessionPlan &firstTile = capacityPlan->tiles.front();
    if (firstTile.phases.size() != launchPhases.size() ||
        firstTile.phases.front().role != launchPhases.front())
      return fail(
          BoardRuntimeStage::Validation, locationFor(firstTile),
          detail::invalid("model launch phase plan does not match manifest"));
    std::vector<BoardGraphModuleSnapshot> graphModules;
    graphModules.reserve(capacityPlan->tiles.size());
    for (const RuntimeSessionPlan &tile : capacityPlan->tiles) {
      const PackageModuleRecord *module =
          detail::findModule(manifest.modules, tile.module);
      const VerifiedModuleSnapshot *snapshot = findSnapshot(tile.module);
      if (!module || !snapshot)
        return fail(BoardRuntimeStage::ModuleLoad, locationFor(tile),
                    detail::invalid("model graph module snapshot is missing"));
      graphModules.push_back({tile.cardId, tile.tileId, tile.launchSlot,
                              tile.module, module->digest, snapshot->bytes});
    }
    llvm::Expected<BoardGraphHandle> loaded =
        driver.loadGraph(graphModules, firstTile.phases.front().symbol);
    if (!loaded)
      return fail(BoardRuntimeStage::ModuleLoad, {}, loaded.takeError());
    liveGraph = *loaded;
  } else {
    for (const VerifiedModuleSnapshot &snapshot : moduleSnapshots) {
      llvm::Expected<BoardModuleHandle> loaded =
          driver.loadModule(snapshot.bytes);
      if (!loaded)
        return fail(BoardRuntimeStage::ModuleLoad, snapshot.diagnosticLocation,
                    loaded.takeError());
      liveModules.push_back({snapshot.module, *loaded});
    }
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
    if (kernelLaunch &&
        kernelLaunch->entryABI == KernelEntryABI::TileRowPointerTable) {
      if (tileIndex >= tileArgumentRows.size())
        return fail(BoardRuntimeStage::Launch, locationFor(tile),
                    detail::invalid("Tile-row launch storage is missing"));
      launch.arguments.push_back(
          static_cast<uint64_t>(tileArgumentRows[tileIndex].memory.value));
      baseLaunches.push_back(std::move(launch));
      continue;
    }
    launch.arguments.reserve(tile.launchOrder.size());
    for (ResourceId resource : tile.launchOrder) {
      auto memory = memoryByResource.find(resource.getValue());
      if (memory == memoryByResource.end())
        return fail(BoardRuntimeStage::Launch, locationFor(tile),
                    detail::invalid("launch slot has no device allocation"));
      launch.arguments.push_back(static_cast<uint64_t>(memory->second.value));
    }
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
      return detail::invalid(
          "host steady clock moved backwards during provider submission");
    const uint64_t elapsedNanoseconds = static_cast<uint64_t>(elapsed);
    if (elapsedNanoseconds >
        std::numeric_limits<uint64_t>::max() - hostSubmitNanoseconds)
      return detail::invalid("aggregate provider submission time overflows");
    hostSubmitNanoseconds += elapsedNanoseconds;
    return llvm::Error::success();
  };
  auto accumulateDeviceTiming =
      [&](const BoardCompletionObservation &observation) -> llvm::Error {
    const bool requested =
        request.deviceTimingPolicy == BoardDeviceTimingPolicy::StreamEvents;
    if (requested != observation.deviceExecutionNanoseconds.has_value())
      return detail::invalid(
          requested
              ? "provider omitted requested same-stream device timing"
              : "provider returned same-stream device timing when disabled");
    if (!requested)
      return llvm::Error::success();
    if (*observation.deviceExecutionNanoseconds >
        std::numeric_limits<uint64_t>::max() - *deviceExecutionNanoseconds)
      return detail::invalid("aggregate device execution time overflows");
    *deviceExecutionNanoseconds += *observation.deviceExecutionNanoseconds;
    return llvm::Error::success();
  };
  if (kernelLaunch) {
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
          return fail(
              BoardRuntimeStage::EntryResolve, {},
              detail::invalid("loaded module has no typed Tile interface"));
        if (firstTile->phases.size() != launchPhases.size() ||
            firstTile->phases[phaseIndex].role != phaseRole)
          return fail(BoardRuntimeStage::EntryResolve, locationFor(*firstTile),
                      detail::invalid(
                          "Tile launch phase plan does not match manifest"));
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
          return fail(BoardRuntimeStage::EntryResolve, locationFor(tile),
                      detail::invalid(
                          "Tile launch phase plan does not match manifest"));
        auto function =
            functionsByPhase[phaseIndex].find(tile.module.getValue());
        if (function == functionsByPhase[phaseIndex].end())
          return fail(BoardRuntimeStage::EntryResolve, locationFor(tile),
                      detail::invalid("Tile phase export was not resolved"));
        phaseLaunches[tileIndex].function = function->second;
      }
      launchesByPhase.push_back(std::move(phaseLaunches));
    }

    for (auto [phaseIndex, phaseRole] : llvm::enumerate(launchPhases)) {
      beginSubmissionWindow();
      const auto submitBegin = std::chrono::steady_clock::now();
      llvm::Error submitError = driver.submitKernelPhase(
          kernelLaunch->form, phaseRole, launchesByPhase[phaseIndex],
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
  } else if (modelLaunch) {
    std::vector<BoardModelTensorLaunch> tensors;
    for (const RuntimeSessionPlan &tile : capacityPlan->tiles)
      for (auto [slotOrdinal, resourceId] : llvm::enumerate(tile.launchOrder)) {
        const PackageResourceRecord *resource =
            detail::findResource(manifest.resources, resourceId);
        auto memory = memoryByResource.find(resourceId.getValue());
        if (!resource || memory == memoryByResource.end())
          return fail(
              BoardRuntimeStage::Launch, locationFor(tile),
              detail::invalid(
                  "model launch slot has no typed resource allocation"));
        tensors.push_back({tile.cardId, tile.tileId, tile.launchSlot,
                           slotOrdinal, resource->role, memory->second,
                           resource->bytes, resource->type.dtype,
                           resource->type.shape});
      }
    beginSubmissionWindow();
    const auto submitBegin = std::chrono::steady_clock::now();
    llvm::Error submitError =
        driver.submitModel(*liveGraph, tensors, request.deviceTimingPolicy);
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
    maximumPollGapNanoseconds = observation->maximumPollGapNanoseconds;
    if (llvm::Error error = accumulateDeviceTiming(*observation))
      return fail(BoardRuntimeStage::Completion, {}, std::move(error));
  } else {
    return fail(BoardRuntimeStage::Launch, {},
                detail::invalid("package has an unknown runtime launch kind"));
  }

  if (!submissionBegin)
    return fail(BoardRuntimeStage::Launch, {},
                detail::invalid("runtime launch did not submit any phase"));
  const auto completionEnd = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           completionEnd - *submissionBegin)
                           .count();
  if (elapsed < 0)
    return fail(BoardRuntimeStage::Completion, {},
                detail::invalid("host steady clock moved backwards"));
  result.launchToCompletionNanoseconds = static_cast<uint64_t>(elapsed);
  result.hostSubmitNanoseconds = hostSubmitNanoseconds;
  result.deviceExecutionNanoseconds = deviceExecutionNanoseconds;
  result.completionObservationResolutionNanoseconds = maximumPollGapNanoseconds;
  result.completedStages.push_back(BoardRuntimeStage::EntryResolve);
  result.completedStages.push_back(BoardRuntimeStage::Launch);
  result.completedStages.push_back(BoardRuntimeStage::Completion);

  if (hasDirectDTETransport) {
    size_t observedStatuses = 0;
    for (const LiveAllocation &allocation : allocations) {
      const PackageResourceRecord &resource = *allocation.resource;
      if (resource.role != PackageResourceRole::TransportStatus ||
          !directDTETiles.contains(
              std::get<TileResourceScope>(resource.scope).tileId.getValue()))
        continue;
      ++observedStatuses;
      std::vector<uint8_t> statusBytes(resource.bytes);
      if (llvm::Error error =
              driver.copyDeviceToHost(statusBytes, allocation.memory)) {
        driver.quarantine();
        return fail(BoardRuntimeStage::DeviceToHost, allocation.location,
                    std::move(error));
      }
      uint32_t status = kDirectDTEStatusPoison;
      if (statusBytes.size() == kDirectDTEStatusStorageBytes)
        std::memcpy(&status, statusBytes.data() + kDirectDTEStatusValueOffset,
                    kDirectDTEStatusValueBytes);
      if (status != static_cast<uint32_t>(DirectDTEStatusValue::Success)) {
        driver.quarantine();
        return fail(
            BoardRuntimeStage::Completion, allocation.location,
            detail::invalid("Direct DTE terminal status is not success: " +
                            llvm::Twine(status)));
      }
    }
    if (observedStatuses != directDTETiles.size()) {
      driver.quarantine();
      return fail(
          BoardRuntimeStage::Completion, {},
          detail::invalid("Direct DTE terminal status domain is incomplete"));
    }
  }

  for (const LiveAllocation &allocation : allocations) {
    const PackageResourceRecord &resource = *allocation.resource;
    if (!resource.hostVisible || resource.access == PackageAccessMode::ReadOnly)
      continue;
    BoardRuntimeOutput output{resource.id,
                              std::vector<uint8_t>(resource.bytes)};
    if (llvm::Error error =
            driver.copyDeviceToHost(output.bytes, allocation.memory)) {
      if (hasDirectDTETransport)
        driver.quarantine();
      return fail(BoardRuntimeStage::DeviceToHost, allocation.location,
                  std::move(error));
    }
    result.outputs.push_back(std::move(output));
  }
  for (const LiveAllocation &allocation : allocations) {
    const PackageResourceRecord &resource = *allocation.resource;
    if (!profilerByResource.count(resource.id.getValue()))
      continue;
    BoardRuntimeOutput output{resource.id,
                              std::vector<uint8_t>(resource.bytes)};
    if (llvm::Error error =
            driver.copyDeviceToHost(output.bytes, allocation.memory)) {
      if (hasDirectDTETransport)
        driver.quarantine();
      return fail(BoardRuntimeStage::DeviceToHost, allocation.location,
                  std::move(error));
    }
    result.profilerOutputs.push_back(std::move(output));
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
executeBoardInvocationInSession(const VerifiedPackageManifest &package,
                                llvm::StringRef packageRoot,
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
      package, packageRoot, std::move(request), *session.driver,
      &session.device, session.qualifiedTileCount, &session.usable);
}

llvm::Expected<
    std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
executeBoardInvocationAndStartSession(const VerifiedPackageManifest &package,
                                      llvm::StringRef packageRoot,
                                      BoardRuntimeInvocationRequest request,
                                      BoardRuntimeDriver &driver) {
  const uint32_t deviceId = request.deviceId;
  const uint32_t tileCount =
      static_cast<uint32_t>(package.getManifest().tileCount);
  BoardDeviceQualification qualification = request.qualification;
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocationImpl(package, packageRoot, std::move(request),
                                 driver, /*qualifiedDevice=*/nullptr,
                                 /*qualifiedTileCount=*/0,
                                 /*qualifiedSessionUsable=*/nullptr);
  if (!result)
    return result.takeError();
  QualifiedBoardRuntimeSession session(
      driver, deviceId, tileCount, std::move(qualification), result->device);
  return std::pair(std::move(*result), std::move(session));
}

llvm::Expected<BoardRuntimeInvocationResult> executeBoardInvocation(
    const VerifiedPackageManifest &package, llvm::StringRef packageRoot,
    BoardRuntimeInvocationRequest request, BoardRuntimeDriver &driver) {
  return executeBoardInvocationImpl(package, packageRoot, std::move(request),
                                    driver, /*qualifiedDevice=*/nullptr,
                                    /*qualifiedTileCount=*/0,
                                    /*qualifiedSessionUsable=*/nullptr);
}

} // namespace wafer::runtime
