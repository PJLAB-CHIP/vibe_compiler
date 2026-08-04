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

llvm::Error boardError(
    BoardRuntimeStage stage, int64_t logicalRank, EntryId entry,
    llvm::Twine detail,
    BoardRuntimeContextState contextState = BoardRuntimeContextState::Usable) {
  return llvm::make_error<BoardRuntimeError>(stage, logicalRank, entry,
                                             detail.str(), contextState);
}

llvm::Error wrapDriverError(
    BoardRuntimeStage stage, int64_t logicalRank, EntryId entry,
    llvm::Error error,
    BoardRuntimeContextState contextState = BoardRuntimeContextState::Usable) {
  return boardError(stage, logicalRank, entry, llvm::toString(std::move(error)),
                    contextState);
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
qualifyBoardDevice(uint32_t deviceId, uint32_t requiredLogicalRankCount,
                   const BoardDeviceQualification &qualification,
                   BoardRuntimeDriver &driver) {
  const EntryId noEntry;
  if (!hasCompleteQualification(qualification))
    return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                      "board execution requires complete explicit device "
                      "qualification facts");
  if (requiredLogicalRankCount == 0 ||
      requiredLogicalRankCount > qualification.tileCount)
    return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                      "qualified logical-rank count is outside the explicit "
                      "device qualification");
  if (driver.getContextState() == BoardRuntimeContextState::Poisoned)
    return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                      "TX provider is already quarantined",
                      BoardRuntimeContextState::Poisoned);

  llvm::Expected<uint32_t> deviceCount = driver.getDeviceCount();
  if (!deviceCount)
    return wrapDriverError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                           deviceCount.takeError(), driver.getContextState());
  if (deviceId >= *deviceCount)
    return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                      "requested device is not present");
  if (llvm::Error error = driver.selectDevice(deviceId))
    return wrapDriverError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                           std::move(error), driver.getContextState());
  llvm::Expected<BoardDeviceInfo> device = driver.getDeviceInfo(deviceId);
  if (!device)
    return wrapDriverError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                           device.takeError(), driver.getContextState());
  if (device->deviceId != deviceId)
    return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                      "TX inventory identifies a different selected device");
  if (!deviceMatchesQualification(*device, qualification))
    return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                      "live TX inventory does not match the explicit board "
                      "qualification");
  if (device->freeMemoryBytes > device->totalMemoryBytes)
    return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                      "TX runtime reported free memory greater than total "
                      "memory");

  uint32_t availableTiles = 0;
  llvm::DenseSet<uint32_t> logicalTiles;
  llvm::DenseSet<uint32_t> availableLogicalTiles;
  std::set<std::pair<uint32_t, uint32_t>> physicalTiles;
  for (const BoardDeviceInfo::Tile &tile : device->tiles) {
    if (!logicalTiles.insert(tile.logicalIndex).second)
      return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                        "TX inventory contains duplicate logical tile indices");
    if (!tile.available)
      continue;
    ++availableTiles;
    availableLogicalTiles.insert(tile.logicalIndex);
    if (!physicalTiles.emplace(tile.physicalX, tile.physicalY).second)
      return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                        "TX inventory maps available tiles to duplicate "
                        "physical coordinates");
  }
  if (availableTiles != device->tileCount ||
      availableTiles < requiredLogicalRankCount)
    return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                      "TX tile availability does not cover the qualified "
                      "logical-rank domain");
  for (uint32_t logicalRank = 0; logicalRank < requiredLogicalRankCount;
       ++logicalRank)
    if (!availableLogicalTiles.contains(logicalRank))
      return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                        "TX inventory does not contain the complete qualified "
                        "logical-rank domain");
  return std::move(*device);
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
                   const PackageModuleRecord &module, int64_t logicalRank,
                   EntryId entry) {
  llvm::SmallString<256> path(packageRoot);
  llvm::sys::path::append(path, module.relativePath);
  if (llvm::sys::fs::get_file_type(path, /*Follow=*/false) !=
      llvm::sys::fs::file_type::regular_file)
    return boardError(BoardRuntimeStage::ModuleLoad, logicalRank, entry,
                      "package module is no longer a regular file");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return boardError(BoardRuntimeStage::ModuleLoad, logicalRank, entry,
                      "failed to reopen verified package module: " +
                          buffer.getError().message());
  llvm::StringRef bytes = (*buffer)->getBuffer();
  llvm::SHA256 hasher;
  hasher.update(bytes);
  std::string digest =
      "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
  if (digest != module.digest)
    return boardError(BoardRuntimeStage::ModuleLoad, logicalRank, entry,
                      "package module changed after manifest verification");
  return std::vector<uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()),
                              reinterpret_cast<const uint8_t *>(bytes.data()) +
                                  bytes.size());
}

struct LiveAllocation {
  const PackageResourceRecord *resource = nullptr;
  EntryId entry;
  BoardDeviceMemory memory;
};

struct LiveRankArgumentRow {
  int64_t logicalRank = -1;
  EntryId entry;
  BoardDeviceMemory memory;
};

struct VerifiedModuleSnapshot {
  const PackageModuleRecord *module = nullptr;
  int64_t diagnosticRank = -1;
  EntryId diagnosticEntry;
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
  case BoardRuntimeStage::Preflight:
    return "preflight";
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
  if (logicalRank >= 0)
    stream << " for rank " << logicalRank;
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
    const BoardDeviceInfo *qualifiedDevice, uint32_t qualifiedLogicalRankCount,
    bool *qualifiedSessionUsable) {
  const PackageManifest &manifest = package.getManifest();
  const EntryId noEntry;
  const KernelRuntimeLaunchContract *kernelLaunch = manifest.launch.getKernel();
  const bool modelLaunch = manifest.launch.getModel() != nullptr;
  if (!hasCompleteQualification(request.qualification))
    return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                      "board execution requires complete explicit device "
                      "qualification facts");
  if (request.completionTimeoutMilliseconds == 0 ||
      request.completionTimeoutMilliseconds >
          kMaximumBoardCompletionTimeoutMilliseconds)
    return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                      "board completion timeout is outside the supported "
                      "range");
  if (request.deviceTimingPolicy == BoardDeviceTimingPolicy::StreamEvents &&
      kernelLaunch && kernelLaunch->form == KernelLaunchForm::PerRank &&
      manifest.rankCount > 1)
    return boardError(
        BoardRuntimeStage::Preflight, -1, noEntry,
        "same-stream device timing does not support a multi-rank per-rank "
        "launch");

  llvm::DenseSet<int64_t> directDTERanks;
  for (const PackageEntrypointRecord &entry : manifest.entries)
    if (std::holds_alternative<DirectDTETransportRequirements>(entry.transport))
      directDTERanks.insert(entry.logicalRank);
  const bool hasDirectDTETransport = !directDTERanks.empty();

  llvm::DenseMap<uint64_t, BoardRuntimeBinding *> bindingsByResource;
  for (BoardRuntimeBinding &binding : request.bindings) {
    uint64_t id = binding.resource.getValue();
    if (!binding.resource.isValid() || bindingsByResource.count(id))
      return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                        "invocation has a duplicate or invalid ResourceId");
    const PackageResourceRecord *resource =
        detail::findResource(manifest.resources, binding.resource);
    if (!resource || !resource->hostVisible)
      return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                        "invocation binds an unknown or internal ResourceId");
    if (binding.bytes.size() != resource->bytes)
      return boardError(BoardRuntimeStage::Preflight, resource->logicalRank,
                        noEntry, "invocation buffer byte count is not exact");
    bindingsByResource[id] = &binding;
  }

  llvm::DenseSet<uint64_t> profilerResourceIds;
  llvm::DenseSet<int64_t> profilerResourceRanks;
  auto isBoundAsFinalProfilerWorkspace =
      [&](const PackageResourceRecord &resource) {
        if (!isProfilerWorkspace(resource))
          return false;
        auto entry = llvm::find_if(
            manifest.entries, [&](const PackageEntrypointRecord &candidate) {
              return candidate.logicalRank == resource.logicalRank;
            });
        return entry != manifest.entries.end() && !entry->slots.empty() &&
               entry->slots.back().resource == resource.id &&
               entry->slots.back().access == resource.access;
      };
  for (const PackageResourceRecord &resource : manifest.resources) {
    if (!isBoundAsFinalProfilerWorkspace(resource))
      continue;
    profilerResourceIds.insert(resource.id.getValue());
    profilerResourceRanks.insert(resource.logicalRank);
  }

  llvm::DenseMap<uint64_t, BoardRuntimeBinding *> profilerByResource;
  llvm::DenseSet<int64_t> profilerRanks;
  for (BoardRuntimeBinding &binding : request.profilerBindings) {
    uint64_t id = binding.resource.getValue();
    if (!binding.resource.isValid() || profilerByResource.count(id) ||
        bindingsByResource.count(id))
      return boardError(
          BoardRuntimeStage::Preflight, -1, noEntry,
          "profiler invocation has a duplicate or invalid ResourceId");
    const PackageResourceRecord *resource =
        detail::findResource(manifest.resources, binding.resource);
    if (!resource || !isBoundAsFinalProfilerWorkspace(*resource))
      return boardError(
          BoardRuntimeStage::Preflight, -1, noEntry,
          "profiler invocation binds a resource outside the exact internal "
          "record contract");
    if (binding.bytes.size() != resource->bytes)
      return boardError(BoardRuntimeStage::Preflight, resource->logicalRank,
                        noEntry,
                        "profiler invocation buffer byte count is not exact");
    if (!profilerRanks.insert(resource->logicalRank).second)
      return boardError(BoardRuntimeStage::Preflight, resource->logicalRank,
                        noEntry,
                        "profiler invocation binds more than one record for "
                        "one logical rank");
    profilerByResource[id] = &binding;
  }
  if (!profilerResourceIds.empty() || !request.profilerBindings.empty()) {
    if (manifest.rankCount != WAFER_TX81_PROFILER_TILE_COUNT ||
        profilerResourceIds.size() != WAFER_TX81_PROFILER_TILE_COUNT ||
        profilerResourceRanks.size() != WAFER_TX81_PROFILER_TILE_COUNT ||
        request.profilerBindings.size() != WAFER_TX81_PROFILER_TILE_COUNT ||
        profilerRanks.size() != WAFER_TX81_PROFILER_TILE_COUNT)
      return boardError(
          BoardRuntimeStage::Preflight, -1, noEntry,
          "profiler invocation requires all-and-only logical tiles 0..15");
    for (int64_t logicalRank = 0; logicalRank < WAFER_TX81_PROFILER_TILE_COUNT;
         ++logicalRank)
      if (!profilerResourceRanks.contains(logicalRank) ||
          !profilerRanks.contains(logicalRank))
        return boardError(
            BoardRuntimeStage::Preflight, logicalRank, noEntry,
            "profiler invocation requires all-and-only logical tiles 0..15");
    for (uint64_t resourceId : profilerResourceIds)
      if (!profilerByResource.count(resourceId))
        return boardError(
            BoardRuntimeStage::Preflight, -1, noEntry,
            "profiler invocation omits an internal profiler ResourceId");
  }

  std::vector<RuntimeInvocationBinding> preflightBindings;
  for (const PackageResourceRecord &resource : manifest.resources) {
    if (!resource.hostVisible)
      continue;
    if (!bindingsByResource.count(resource.id.getValue()))
      return boardError(BoardRuntimeStage::Preflight, resource.logicalRank,
                        noEntry, "invocation omits a host-visible ResourceId");
    preflightBindings.push_back({resource.id, resource.bytes,
                                 resource.alignment, resource.access, true});
  }
  if (preflightBindings.size() != request.bindings.size())
    return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                      "invocation bindings are not all-and-only for package");

  const RuntimeEnvironment &providerEnvironment =
      driver.getProviderEnvironment();
  llvm::Expected<RuntimeInvocationPlan> semanticPlan =
      preflightNoCardRuntimeInvocation(package, preflightBindings,
                                       providerEnvironment);
  if (!semanticPlan)
    return wrapDriverError(BoardRuntimeStage::Preflight, -1, noEntry,
                           semanticPlan.takeError());

  std::vector<VerifiedModuleSnapshot> moduleSnapshots;
  moduleSnapshots.reserve(manifest.modules.size());
  for (const PackageModuleRecord &module : manifest.modules) {
    auto firstRank = llvm::find_if(semanticPlan->ranks, [&](const auto &rank) {
      return rank.module == module.id;
    });
    if (firstRank == semanticPlan->ranks.end())
      return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                        "package contains an unreferenced module");
    llvm::Expected<std::vector<uint8_t>> bytes = readVerifiedModule(
        packageRoot, module, firstRank->logicalRank, firstRank->entry);
    if (!bytes)
      return bytes.takeError();
    moduleSnapshots.push_back(
        {&module, firstRank->logicalRank, firstRank->entry, std::move(*bytes)});
  }
  llvm::sort(moduleSnapshots, [](const auto &lhs, const auto &rhs) {
    return lhs.diagnosticRank < rhs.diagnosticRank;
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
      return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                        "qualified board runtime session is no longer usable",
                        BoardRuntimeContextState::Poisoned);
    if (manifest.rankCount != static_cast<int64_t>(qualifiedLogicalRankCount))
      return boardError(
          BoardRuntimeStage::Preflight, -1, noEntry,
          "package rank domain does not match the qualified board runtime "
          "session");
    if (driver.getContextState() == BoardRuntimeContextState::Poisoned) {
      *qualifiedSessionUsable = false;
      return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                        "TX provider is already quarantined",
                        BoardRuntimeContextState::Poisoned);
    }
    device = *qualifiedDevice;
  } else {
    llvm::Expected<BoardDeviceInfo> qualified = qualifyBoardDevice(
        request.deviceId, static_cast<uint32_t>(manifest.rankCount),
        request.qualification, driver);
    if (!qualified)
      return qualified.takeError();
    device = std::move(*qualified);
  }

  RuntimeEnvironment capacityEnvironment = providerEnvironment;
  capacityEnvironment.maxResourceBytes = device.freeMemoryBytes;
  llvm::Expected<RuntimeInvocationPlan> capacityPlan =
      preflightNoCardRuntimeInvocation(package, preflightBindings,
                                       capacityEnvironment);
  if (!capacityPlan)
    return wrapDriverError(BoardRuntimeStage::Preflight, -1, noEntry,
                           capacityPlan.takeError());

  uint64_t allocationBytes = 0;
  for (const RuntimeSessionPlan &rank : capacityPlan->ranks)
    for (const PlannedRuntimeResource &resource : rank.resources) {
      if (resource.bytes >
          std::numeric_limits<uint64_t>::max() - allocationBytes)
        return boardError(BoardRuntimeStage::Preflight, rank.logicalRank,
                          rank.entry,
                          "aggregate board allocation byte count overflows");
      allocationBytes += resource.bytes;
    }
  if (kernelLaunch &&
      kernelLaunch->entryABI == KernelEntryABI::RankRowPointerTable) {
    for (const RuntimeSessionPlan &rank : capacityPlan->ranks) {
      const uint64_t rowBytes =
          static_cast<uint64_t>(rank.launchOrder.size()) * sizeof(uint64_t);
      if (rowBytes > std::numeric_limits<uint64_t>::max() - allocationBytes)
        return boardError(BoardRuntimeStage::Preflight, rank.logicalRank,
                          rank.entry,
                          "aggregate rank-row pointer storage overflows");
      allocationBytes += rowBytes;
    }
  }
  for (const VerifiedModuleSnapshot &snapshot : moduleSnapshots) {
    uint64_t moduleBytes = snapshot.bytes.size();
    if (moduleBytes > std::numeric_limits<uint64_t>::max() - allocationBytes)
      return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                        "aggregate board module byte count overflows");
    allocationBytes += moduleBytes;
  }
  if (device.freeMemoryBytes <= boardRuntimeFreeMemoryReserve ||
      allocationBytes > device.freeMemoryBytes - boardRuntimeFreeMemoryReserve)
    return boardError(
        BoardRuntimeStage::Preflight, -1, noEntry,
        "aggregate board allocation demand exceeds qualified free memory "
        "after the runtime safety reserve");

  BoardRuntimeInvocationResult result;
  result.device = device;
  for (const RuntimeSessionPlan &rank : capacityPlan->ranks)
    result.ranks.push_back(
        {rank.entry, rank.logicalRank, rank.module, rank.terminalCompletion});
  result.completedStages.push_back(BoardRuntimeStage::Preflight);
  result.completedStages.push_back(BoardRuntimeStage::DeviceSelection);

  std::vector<LiveAllocation> allocations;
  std::vector<LiveRankArgumentRow> rankArgumentRows;
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
        llvm::Error wrapped = wrapDriverError(BoardRuntimeStage::Cleanup, -1,
                                              noEntry, std::move(error), state);
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
        llvm::Error wrapped = wrapDriverError(BoardRuntimeStage::Cleanup, -1,
                                              noEntry, std::move(error), state);
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
        llvm::Error wrapped = wrapDriverError(BoardRuntimeStage::Cleanup, -1,
                                              noEntry, std::move(error), state);
        if (state == BoardRuntimeContextState::Poisoned)
          return llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
        cleanupError =
            llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
      }
      liveGraph.reset();
    }
    for (LiveRankArgumentRow &row : llvm::reverse(rankArgumentRows)) {
      if (llvm::Error error = driver.free(row.memory)) {
        BoardRuntimeContextState state = observeProviderState();
        llvm::Error wrapped =
            wrapDriverError(BoardRuntimeStage::Cleanup, row.logicalRank,
                            row.entry, std::move(error), state);
        if (state == BoardRuntimeContextState::Poisoned)
          return llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
        cleanupError =
            llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
      }
    }
    rankArgumentRows.clear();
    for (LiveAllocation &allocation : llvm::reverse(allocations)) {
      if (llvm::Error error = driver.free(allocation.memory)) {
        BoardRuntimeContextState state = observeProviderState();
        llvm::Error wrapped = wrapDriverError(
            BoardRuntimeStage::Cleanup, allocation.resource->logicalRank,
            allocation.entry, std::move(error), state);
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
      [&](BoardRuntimeStage stage, int64_t logicalRank, EntryId entry,
          llvm::Error error) -> llvm::Expected<BoardRuntimeInvocationResult> {
    BoardRuntimeContextState state = observeProviderState();
    llvm::Error primary =
        wrapDriverError(stage, logicalRank, entry, std::move(error), state);
    if (state == BoardRuntimeContextState::Poisoned)
      return std::move(primary);
    return llvm::joinErrors(std::move(primary), cleanup());
  };

  llvm::DenseMap<uint64_t, BoardDeviceMemory> memoryByResource;
  for (const RuntimeSessionPlan &rank : capacityPlan->ranks)
    for (ResourceId resourceId : rank.launchOrder) {
      const PackageResourceRecord *resource =
          detail::findResource(manifest.resources, resourceId);
      llvm::Expected<BoardDeviceMemory> memory =
          driver.allocate(resource->bytes, resource->alignment);
      if (!memory)
        return fail(BoardRuntimeStage::ResourceAllocation, rank.logicalRank,
                    rank.entry, memory.takeError());
      allocations.push_back({resource, rank.entry, *memory});
      memoryByResource[resource->id.getValue()] = *memory;
    }
  if (kernelLaunch &&
      kernelLaunch->entryABI == KernelEntryABI::RankRowPointerTable) {
    rankArgumentRows.reserve(capacityPlan->ranks.size());
    for (const RuntimeSessionPlan &rank : capacityPlan->ranks) {
      const uint64_t rowBytes =
          static_cast<uint64_t>(rank.launchOrder.size()) * sizeof(uint64_t);
      llvm::Expected<BoardDeviceMemory> memory =
          driver.allocate(rowBytes, alignof(uint64_t));
      if (!memory)
        return fail(BoardRuntimeStage::ResourceAllocation, rank.logicalRank,
                    rank.entry, memory.takeError());
      rankArgumentRows.push_back({rank.logicalRank, rank.entry, *memory});
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
        return fail(BoardRuntimeStage::HostToDevice, resource.logicalRank,
                    allocation.entry, std::move(error));
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
      return fail(BoardRuntimeStage::HostToDevice,
                  allocation.resource->logicalRank, allocation.entry,
                  std::move(error));
  }
  if (!rankArgumentRows.empty()) {
    if (rankArgumentRows.size() != capacityPlan->ranks.size())
      return fail(BoardRuntimeStage::HostToDevice, -1, noEntry,
                  detail::invalid("rank-row allocation domain is incomplete"));
    for (auto [rankIndex, rank] : llvm::enumerate(capacityPlan->ranks)) {
      std::vector<uint64_t> row;
      row.reserve(rank.launchOrder.size());
      for (ResourceId resource : rank.launchOrder) {
        auto memory = memoryByResource.find(resource.getValue());
        if (memory == memoryByResource.end())
          return fail(
              BoardRuntimeStage::HostToDevice, rank.logicalRank, rank.entry,
              detail::invalid("rank-row slot has no device allocation"));
        row.push_back(static_cast<uint64_t>(memory->second.value));
      }
      llvm::ArrayRef<uint8_t> rowBytes(
          reinterpret_cast<const uint8_t *>(row.data()),
          row.size() * sizeof(uint64_t));
      if (llvm::Error error = driver.copyHostToDevice(
              rankArgumentRows[rankIndex].memory, rowBytes))
        return fail(BoardRuntimeStage::HostToDevice, rank.logicalRank,
                    rank.entry, std::move(error));
    }
  }
  result.completedStages.push_back(BoardRuntimeStage::HostToDevice);

  llvm::ArrayRef<RuntimeLaunchPhaseRole> launchPhases =
      manifest.launch.getPhases();
  if (capacityPlan->ranks.empty() || launchPhases.empty())
    return fail(BoardRuntimeStage::Preflight, -1, noEntry,
                detail::invalid("runtime launch has no planned rank or phase"));

  if (modelLaunch) {
    const RuntimeSessionPlan &firstRank = capacityPlan->ranks.front();
    if (firstRank.phases.size() != launchPhases.size() ||
        firstRank.phases.front().role != launchPhases.front())
      return fail(
          BoardRuntimeStage::Preflight, firstRank.logicalRank, firstRank.entry,
          detail::invalid("model launch phase plan does not match manifest"));
    std::vector<BoardGraphModuleSnapshot> graphModules;
    graphModules.reserve(capacityPlan->ranks.size());
    for (size_t index = 0; index < capacityPlan->ranks.size(); ++index) {
      const RuntimeSessionPlan &rank = capacityPlan->ranks[index];
      const PackageModuleRecord *module =
          detail::findModule(manifest.modules, rank.module);
      const VerifiedModuleSnapshot *snapshot = findSnapshot(rank.module);
      if (!module || !snapshot)
        return fail(BoardRuntimeStage::ModuleLoad, rank.logicalRank, rank.entry,
                    detail::invalid("model graph module snapshot is missing"));
      graphModules.push_back({static_cast<uint16_t>(rank.logicalRank),
                              rank.module, module->digest, snapshot->bytes});
    }
    llvm::Expected<BoardGraphHandle> loaded =
        driver.loadGraph(graphModules, firstRank.phases.front().symbol);
    if (!loaded)
      return fail(BoardRuntimeStage::ModuleLoad, -1, noEntry,
                  loaded.takeError());
    liveGraph = *loaded;
  } else {
    for (const VerifiedModuleSnapshot &snapshot : moduleSnapshots) {
      llvm::Expected<BoardModuleHandle> loaded =
          driver.loadModule(snapshot.bytes);
      if (!loaded)
        return fail(BoardRuntimeStage::ModuleLoad, snapshot.diagnosticRank,
                    snapshot.diagnosticEntry, loaded.takeError());
      liveModules.push_back({snapshot.module, *loaded});
    }
  }
  result.completedStages.push_back(BoardRuntimeStage::ModuleLoad);

  std::vector<BoardRankLaunch> baseLaunches;
  baseLaunches.reserve(capacityPlan->ranks.size());
  for (auto [rankIndex, rank] : llvm::enumerate(capacityPlan->ranks)) {
    BoardRankLaunch launch;
    launch.logicalRank = rank.logicalRank;
    launch.entry = rank.entry;
    if (kernelLaunch &&
        kernelLaunch->entryABI == KernelEntryABI::RankRowPointerTable) {
      if (rankIndex >= rankArgumentRows.size())
        return fail(BoardRuntimeStage::Launch, rank.logicalRank, rank.entry,
                    detail::invalid("rank-row launch storage is missing"));
      launch.arguments.push_back(
          static_cast<uint64_t>(rankArgumentRows[rankIndex].memory.value));
      baseLaunches.push_back(std::move(launch));
      continue;
    }
    launch.arguments.reserve(rank.launchOrder.size());
    for (ResourceId resource : rank.launchOrder) {
      auto memory = memoryByResource.find(resource.getValue());
      if (memory == memoryByResource.end())
        return fail(BoardRuntimeStage::Launch, rank.logicalRank, rank.entry,
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
        auto firstRank =
            llvm::find_if(capacityPlan->ranks, [&](const auto &rank) {
              return rank.module == liveModule.moduleRecord->id;
            });
        if (firstRank == capacityPlan->ranks.end())
          return fail(
              BoardRuntimeStage::EntryResolve, -1, noEntry,
              detail::invalid("loaded module has no typed rank interface"));
        if (firstRank->phases.size() != launchPhases.size() ||
            firstRank->phases[phaseIndex].role != phaseRole)
          return fail(BoardRuntimeStage::EntryResolve, firstRank->logicalRank,
                      firstRank->entry,
                      detail::invalid(
                          "rank launch phase plan does not match manifest"));
        llvm::Expected<BoardFunctionHandle> function = driver.resolveEntry(
            liveModule.module, firstRank->phases[phaseIndex].symbol);
        if (!function)
          return fail(BoardRuntimeStage::EntryResolve, firstRank->logicalRank,
                      firstRank->entry, function.takeError());
        functionsByModule[liveModule.moduleRecord->id.getValue()] = *function;
      }
      functionsByPhase.push_back(std::move(functionsByModule));
    }

    std::vector<std::vector<BoardRankLaunch>> launchesByPhase;
    launchesByPhase.reserve(launchPhases.size());
    for (auto [phaseIndex, phaseRole] : llvm::enumerate(launchPhases)) {
      std::vector<BoardRankLaunch> phaseLaunches = baseLaunches;
      for (auto [rankIndex, rank] : llvm::enumerate(capacityPlan->ranks)) {
        if (rank.phases.size() != launchPhases.size() ||
            rank.phases[phaseIndex].role != phaseRole)
          return fail(BoardRuntimeStage::EntryResolve, rank.logicalRank,
                      rank.entry,
                      detail::invalid(
                          "rank launch phase plan does not match manifest"));
        auto function =
            functionsByPhase[phaseIndex].find(rank.module.getValue());
        if (function == functionsByPhase[phaseIndex].end())
          return fail(BoardRuntimeStage::EntryResolve, rank.logicalRank,
                      rank.entry,
                      detail::invalid("rank phase export was not resolved"));
        phaseLaunches[rankIndex].function = function->second;
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
        return fail(BoardRuntimeStage::Launch, -1, noEntry,
                    std::move(submitError));
      submissionLive = true;
      if (llvm::Error error = accumulateHostSubmit(submitBegin, submitEnd)) {
        driver.quarantine();
        return fail(BoardRuntimeStage::Launch, -1, noEntry, std::move(error));
      }
      llvm::Expected<BoardCompletionObservation> observation =
          driver.waitCurrentSubmission(*deadline,
                                       request.completionObservationPolicy);
      if (!observation)
        return fail(BoardRuntimeStage::Completion, -1, noEntry,
                    observation.takeError());
      maximumPollGapNanoseconds = std::max(
          maximumPollGapNanoseconds, observation->maximumPollGapNanoseconds);
      if (llvm::Error error = accumulateDeviceTiming(*observation))
        return fail(BoardRuntimeStage::Completion, -1, noEntry,
                    std::move(error));
    }
  } else if (modelLaunch) {
    std::vector<BoardModelTensorLaunch> tensors;
    for (const RuntimeSessionPlan &rank : capacityPlan->ranks)
      for (auto [slotOrdinal, resourceId] : llvm::enumerate(rank.launchOrder)) {
        const PackageResourceRecord *resource =
            detail::findResource(manifest.resources, resourceId);
        auto memory = memoryByResource.find(resourceId.getValue());
        if (!resource || memory == memoryByResource.end())
          return fail(
              BoardRuntimeStage::Launch, rank.logicalRank, rank.entry,
              detail::invalid(
                  "model launch slot has no typed resource allocation"));
        tensors.push_back({rank.logicalRank, slotOrdinal, resource->role,
                           memory->second, resource->bytes,
                           resource->type.dtype, resource->type.shape});
      }
    beginSubmissionWindow();
    const auto submitBegin = std::chrono::steady_clock::now();
    llvm::Error submitError =
        driver.submitModel(*liveGraph, tensors, request.deviceTimingPolicy);
    const auto submitEnd = std::chrono::steady_clock::now();
    if (submitError)
      return fail(BoardRuntimeStage::Launch, -1, noEntry,
                  std::move(submitError));
    submissionLive = true;
    if (llvm::Error error = accumulateHostSubmit(submitBegin, submitEnd)) {
      driver.quarantine();
      return fail(BoardRuntimeStage::Launch, -1, noEntry, std::move(error));
    }
    llvm::Expected<BoardCompletionObservation> observation =
        driver.waitCurrentSubmission(*deadline,
                                     request.completionObservationPolicy);
    if (!observation)
      return fail(BoardRuntimeStage::Completion, -1, noEntry,
                  observation.takeError());
    maximumPollGapNanoseconds = observation->maximumPollGapNanoseconds;
    if (llvm::Error error = accumulateDeviceTiming(*observation))
      return fail(BoardRuntimeStage::Completion, -1, noEntry, std::move(error));
  } else {
    return fail(BoardRuntimeStage::Launch, -1, noEntry,
                detail::invalid("package has an unknown runtime launch kind"));
  }

  if (!submissionBegin)
    return fail(BoardRuntimeStage::Launch, -1, noEntry,
                detail::invalid("runtime launch did not submit any phase"));
  const auto completionEnd = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           completionEnd - *submissionBegin)
                           .count();
  if (elapsed < 0)
    return fail(BoardRuntimeStage::Completion, -1, noEntry,
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
          !directDTERanks.contains(resource.logicalRank))
        continue;
      ++observedStatuses;
      std::vector<uint8_t> statusBytes(resource.bytes);
      if (llvm::Error error =
              driver.copyDeviceToHost(statusBytes, allocation.memory)) {
        driver.quarantine();
        return fail(BoardRuntimeStage::DeviceToHost, resource.logicalRank,
                    allocation.entry, std::move(error));
      }
      uint32_t status = kDirectDTEStatusPoison;
      if (statusBytes.size() == kDirectDTEStatusStorageBytes)
        std::memcpy(&status, statusBytes.data() + kDirectDTEStatusValueOffset,
                    kDirectDTEStatusValueBytes);
      if (status != static_cast<uint32_t>(DirectDTEStatusValue::Success)) {
        driver.quarantine();
        return fail(
            BoardRuntimeStage::Completion, resource.logicalRank,
            allocation.entry,
            detail::invalid("Direct DTE terminal status is not success: " +
                            llvm::Twine(status)));
      }
    }
    if (observedStatuses != directDTERanks.size()) {
      driver.quarantine();
      return fail(
          BoardRuntimeStage::Completion, -1, noEntry,
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
      if (directDTERanks.contains(resource.logicalRank))
        driver.quarantine();
      return fail(BoardRuntimeStage::DeviceToHost, resource.logicalRank,
                  allocation.entry, std::move(error));
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
      if (directDTERanks.contains(resource.logicalRank))
        driver.quarantine();
      return fail(BoardRuntimeStage::DeviceToHost, resource.logicalRank,
                  allocation.entry, std::move(error));
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
    BoardRuntimeDriver &driver, uint32_t deviceId,
    uint32_t qualifiedLogicalRankCount, BoardDeviceQualification qualification,
    BoardDeviceInfo device)
    : driver(&driver), deviceId(deviceId),
      qualifiedLogicalRankCount(qualifiedLogicalRankCount),
      qualification(std::move(qualification)), device(std::move(device)),
      usable(true) {}

QualifiedBoardRuntimeSession::QualifiedBoardRuntimeSession(
    QualifiedBoardRuntimeSession &&other) noexcept
    : driver(std::exchange(other.driver, nullptr)),
      deviceId(std::exchange(other.deviceId, 0)),
      qualifiedLogicalRankCount(
          std::exchange(other.qualifiedLogicalRankCount, 0)),
      qualification(std::move(other.qualification)),
      device(std::move(other.device)),
      usable(std::exchange(other.usable, false)) {}

QualifiedBoardRuntimeSession &QualifiedBoardRuntimeSession::operator=(
    QualifiedBoardRuntimeSession &&other) noexcept {
  if (this == &other)
    return *this;
  driver = std::exchange(other.driver, nullptr);
  deviceId = std::exchange(other.deviceId, 0);
  qualifiedLogicalRankCount = std::exchange(other.qualifiedLogicalRankCount, 0);
  qualification = std::move(other.qualification);
  device = std::move(other.device);
  usable = std::exchange(other.usable, false);
  return *this;
}

llvm::Expected<QualifiedBoardRuntimeSession>
qualifyBoardRuntimeSession(uint32_t deviceId, uint32_t requiredLogicalRankCount,
                           const BoardDeviceQualification &qualification,
                           BoardRuntimeDriver &driver) {
  llvm::Expected<BoardDeviceInfo> device = qualifyBoardDevice(
      deviceId, requiredLogicalRankCount, qualification, driver);
  if (!device)
    return device.takeError();
  return QualifiedBoardRuntimeSession(driver, deviceId,
                                      requiredLogicalRankCount, qualification,
                                      std::move(*device));
}

llvm::Expected<BoardRuntimeInvocationResult>
executeBoardInvocationInSession(const VerifiedPackageManifest &package,
                                llvm::StringRef packageRoot,
                                BoardRuntimeInvocationRequest request,
                                QualifiedBoardRuntimeSession &session) {
  const EntryId noEntry;
  if (!session.driver || !session.usable)
    return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                      "qualified board runtime session is no longer usable",
                      BoardRuntimeContextState::Poisoned);
  if (request.deviceId != session.deviceId ||
      !qualificationMatches(request.qualification, session.qualification))
    return boardError(
        BoardRuntimeStage::Preflight, -1, noEntry,
        "board invocation does not match the qualified session identity");
  if (package.getManifest().rankCount !=
      static_cast<int64_t>(session.qualifiedLogicalRankCount))
    return boardError(
        BoardRuntimeStage::Preflight, -1, noEntry,
        "package rank domain does not match the qualified board runtime "
        "session");
  if (session.driver->getContextState() == BoardRuntimeContextState::Poisoned) {
    session.usable = false;
    return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                      "TX provider is already quarantined",
                      BoardRuntimeContextState::Poisoned);
  }
  return executeBoardInvocationImpl(
      package, packageRoot, std::move(request), *session.driver,
      &session.device, session.qualifiedLogicalRankCount, &session.usable);
}

llvm::Expected<
    std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
executeBoardInvocationAndStartSession(const VerifiedPackageManifest &package,
                                      llvm::StringRef packageRoot,
                                      BoardRuntimeInvocationRequest request,
                                      BoardRuntimeDriver &driver) {
  const uint32_t deviceId = request.deviceId;
  const uint32_t rankCount =
      static_cast<uint32_t>(package.getManifest().rankCount);
  BoardDeviceQualification qualification = request.qualification;
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocationImpl(package, packageRoot, std::move(request),
                                 driver, /*qualifiedDevice=*/nullptr,
                                 /*qualifiedLogicalRankCount=*/0,
                                 /*qualifiedSessionUsable=*/nullptr);
  if (!result)
    return result.takeError();
  QualifiedBoardRuntimeSession session(
      driver, deviceId, rankCount, std::move(qualification), result->device);
  return std::pair(std::move(*result), std::move(session));
}

llvm::Expected<BoardRuntimeInvocationResult> executeBoardInvocation(
    const VerifiedPackageManifest &package, llvm::StringRef packageRoot,
    BoardRuntimeInvocationRequest request, BoardRuntimeDriver &driver) {
  return executeBoardInvocationImpl(package, packageRoot, std::move(request),
                                    driver, /*qualifiedDevice=*/nullptr,
                                    /*qualifiedLogicalRankCount=*/0,
                                    /*qualifiedSessionUsable=*/nullptr);
}

llvm::Expected<BoardRuntimeResult>
executeBoardEntry(const VerifiedPackageManifest &package,
                  llvm::StringRef packageRoot, BoardRuntimeRequest request,
                  BoardRuntimeDriver &driver) {
  const PackageManifest &manifest = package.getManifest();
  if (manifest.rankCount != 1 || manifest.entries.size() != 1 ||
      manifest.entries.front().id != request.entry)
    return boardError(
        BoardRuntimeStage::Preflight, -1, request.entry,
        "rank-one entry execution requires the unique entry of a "
        "rank-count=1 package");

  BoardRuntimeInvocationRequest invocation;
  invocation.deviceId = request.deviceId;
  invocation.completionTimeoutMilliseconds =
      request.completionTimeoutMilliseconds;
  invocation.qualification = std::move(request.qualification);
  invocation.bindings = std::move(request.bindings);
  llvm::Expected<BoardRuntimeInvocationResult> result = executeBoardInvocation(
      package, packageRoot, std::move(invocation), driver);
  if (!result)
    return result.takeError();

  const BoardRuntimeRankResult &rank = result->ranks.front();
  BoardRuntimeResult entryResult;
  entryResult.device = std::move(result->device);
  entryResult.entry = rank.entry;
  entryResult.logicalRank = rank.logicalRank;
  entryResult.module = rank.module;
  entryResult.terminalCompletion = rank.terminalCompletion;
  entryResult.completedStages = std::move(result->completedStages);
  entryResult.outputs = std::move(result->outputs);
  return entryResult;
}

} // namespace wafer::runtime
