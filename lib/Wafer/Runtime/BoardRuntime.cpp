//===- BoardRuntime.cpp - Verified package board execution --------------===//

#include "Wafer/Runtime/BoardRuntime.h"

#include "PackageManifestInternal.h"

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

#include <limits>
#include <cstring>
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

llvm::Expected<BoardRuntimeInvocationResult> executeBoardInvocation(
    const VerifiedPackageManifest &package, llvm::StringRef packageRoot,
    BoardRuntimeInvocationRequest request, BoardRuntimeDriver &driver) {
  const PackageManifest &manifest = package.getManifest();
  const EntryId noEntry;
  const bool perRankLaunch =
      manifest.launchABI == TargetLaunchABIId::perRankPointerBlockV1();
  const bool kernelGridLaunch = manifest.launchABI ==
                                TargetLaunchABIId::
                                    tx81KernelGridPointerTableV1();
  const bool modelLaunch =
      manifest.launchABI == TargetLaunchABIId::tx81ModelBootParamV1();
  const bool clusterDirectDTELaunch =
      manifest.launchABI ==
      TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1();
  if (request.qualification.runtimeVersion == 0 ||
      request.qualification.tileCount == 0 ||
      request.qualification.name.empty() ||
      request.qualification.pciBusId.empty() ||
      request.qualification.runtimeLibraryDigest.empty())
    return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                      "board execution requires complete explicit device "
                      "qualification facts");
  if (request.completionTimeoutMilliseconds == 0 ||
      request.completionTimeoutMilliseconds >
          kMaximumBoardCompletionTimeoutMilliseconds)
    return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                      "board completion timeout is outside the supported "
                      "range");
  for (const PackageEntrypointRecord &entry : manifest.entries) {
    const bool directDTE =
        std::holds_alternative<DirectDTETransportRequirements>(
            entry.transport);
    if (directDTE != clusterDirectDTELaunch)
      return boardError(
          BoardRuntimeStage::Preflight, entry.logicalRank, entry.id,
          "TX board Direct DTE requires the closed cluster prepare/main "
          "launch ABI");
  }

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

  if (driver.getContextState() == BoardRuntimeContextState::Poisoned)
    return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                      "TX provider is already quarantined",
                      BoardRuntimeContextState::Poisoned);

  llvm::Expected<uint32_t> deviceCount = driver.getDeviceCount();
  if (!deviceCount)
    return wrapDriverError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                           deviceCount.takeError(), driver.getContextState());
  if (request.deviceId >= *deviceCount)
    return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                      "requested device is not present");
  if (llvm::Error error = driver.selectDevice(request.deviceId))
    return wrapDriverError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                           std::move(error), driver.getContextState());
  llvm::Expected<BoardDeviceInfo> device =
      driver.getDeviceInfo(request.deviceId);
  if (!device)
    return wrapDriverError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                           device.takeError(), driver.getContextState());
  if (device->runtimeVersion != request.qualification.runtimeVersion ||
      device->tileCount != request.qualification.tileCount ||
      device->name != request.qualification.name ||
      device->pciBusId != request.qualification.pciBusId ||
      device->runtimeLibraryDigest !=
          request.qualification.runtimeLibraryDigest)
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
      availableTiles < static_cast<uint32_t>(manifest.rankCount))
    return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                      "TX tile availability does not cover the package rank "
                      "domain");
  if (manifest.rankCount == 16) {
    for (uint32_t logicalRank = 0; logicalRank < 16; ++logicalRank)
      if (!availableLogicalTiles.contains(logicalRank))
        return boardError(BoardRuntimeStage::DeviceSelection, -1, noEntry,
                          "TX inventory does not contain logical tiles 0..15");
  }

  RuntimeEnvironment capacityEnvironment = providerEnvironment;
  capacityEnvironment.maxResourceBytes = device->freeMemoryBytes;
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
  for (const VerifiedModuleSnapshot &snapshot : moduleSnapshots) {
    uint64_t moduleBytes = snapshot.bytes.size();
    if (moduleBytes > std::numeric_limits<uint64_t>::max() - allocationBytes)
      return boardError(BoardRuntimeStage::Preflight, -1, noEntry,
                        "aggregate board module byte count overflows");
    allocationBytes += moduleBytes;
  }
  if (device->freeMemoryBytes <= boardRuntimeFreeMemoryReserve ||
      allocationBytes > device->freeMemoryBytes - boardRuntimeFreeMemoryReserve)
    return boardError(
        BoardRuntimeStage::Preflight, -1, noEntry,
        "aggregate board allocation demand exceeds qualified free memory "
        "after the runtime safety reserve");

  BoardRuntimeInvocationResult result;
  result.device = std::move(*device);
  for (const RuntimeSessionPlan &rank : capacityPlan->ranks)
    result.ranks.push_back(
        {rank.entry, rank.logicalRank, rank.module, rank.terminalCompletion});
  result.completedStages.push_back(BoardRuntimeStage::Preflight);
  result.completedStages.push_back(BoardRuntimeStage::DeviceSelection);

  std::vector<LiveAllocation> allocations;
  std::vector<LiveModule> liveModules;
  std::optional<BoardGraphHandle> liveGraph;
  bool submissionLive = false;
  auto cleanup = [&]() -> llvm::Error {
    llvm::Error cleanupError = llvm::Error::success();
    if (submissionLive) {
      if (llvm::Error error = driver.releaseSubmission()) {
        BoardRuntimeContextState state = driver.getContextState();
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
        BoardRuntimeContextState state = driver.getContextState();
        llvm::Error wrapped = wrapDriverError(
            BoardRuntimeStage::Cleanup, -1, noEntry, std::move(error), state);
        if (state == BoardRuntimeContextState::Poisoned)
          return llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
        cleanupError =
            llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
      }
    }
    liveModules.clear();
    if (liveGraph) {
      if (llvm::Error error = driver.unloadGraph(*liveGraph)) {
        BoardRuntimeContextState state = driver.getContextState();
        llvm::Error wrapped = wrapDriverError(BoardRuntimeStage::Cleanup, -1,
                                              noEntry, std::move(error), state);
        if (state == BoardRuntimeContextState::Poisoned)
          return llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
        cleanupError =
            llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
      }
      liveGraph.reset();
    }
    for (LiveAllocation &allocation : llvm::reverse(allocations)) {
      if (llvm::Error error = driver.free(allocation.memory)) {
        BoardRuntimeContextState state = driver.getContextState();
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
    BoardRuntimeContextState state = driver.getContextState();
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
  result.completedStages.push_back(BoardRuntimeStage::ResourceAllocation);

  for (const LiveAllocation &allocation : allocations) {
    const PackageResourceRecord &resource = *allocation.resource;
    // Workspace is allocation-only storage.  Materializing a host-sized zero
    // buffer and uploading it is both semantically unnecessary and
    // prohibitive for large compiler-managed DDR arenas.
    if (resource.role == PackageResourceRole::Workspace)
      continue;
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
  result.completedStages.push_back(BoardRuntimeStage::HostToDevice);

  if (modelLaunch) {
    std::vector<BoardGraphModuleSnapshot> graphModules;
    graphModules.reserve(capacityPlan->ranks.size());
    for (size_t index = 0; index < capacityPlan->ranks.size(); ++index) {
      const RuntimeSessionPlan &rank = capacityPlan->ranks[index];
      const PackageModuleRecord *module =
          detail::findModule(manifest.modules, rank.module);
      const VerifiedModuleSnapshot *snapshot = findSnapshot(rank.module);
      if (!module || !snapshot)
        return fail(BoardRuntimeStage::ModuleLoad, rank.logicalRank,
                    rank.entry,
                    detail::invalid("model graph module snapshot is missing"));
      graphModules.push_back(
          {static_cast<uint16_t>(rank.logicalRank), rank.module,
           module->digest, snapshot->bytes});
    }
    llvm::Expected<BoardGraphHandle> loaded =
        driver.loadGraph(graphModules, capacityPlan->ranks.front().mainSymbol);
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
                    snapshot.diagnosticEntry,
                    loaded.takeError());
      liveModules.push_back({snapshot.module, *loaded});
    }
  }
  result.completedStages.push_back(BoardRuntimeStage::ModuleLoad);

  llvm::DenseMap<uint64_t, BoardFunctionHandle> mainFunctionsByModule;
  std::optional<BoardFunctionHandle> clusterPrepareFunction;
  if (!modelLaunch) {
    for (const LiveModule &liveModule : liveModules) {
      auto firstRank =
          llvm::find_if(capacityPlan->ranks, [&](const auto &rank) {
            return rank.module == liveModule.moduleRecord->id;
          });
      if (firstRank == capacityPlan->ranks.end())
        return fail(BoardRuntimeStage::EntryResolve, -1, noEntry,
                    detail::invalid(
                        "loaded module has no typed rank interface"));
      llvm::Expected<BoardFunctionHandle> function =
          driver.resolveEntry(liveModule.module, firstRank->mainSymbol);
      if (!function)
        return fail(BoardRuntimeStage::EntryResolve,
                    firstRank->logicalRank, firstRank->entry,
                    function.takeError());
      mainFunctionsByModule[liveModule.moduleRecord->id.getValue()] = *function;
      if (clusterDirectDTELaunch) {
        const PackageModuleExportRecord *prepare = detail::findModuleExport(
            *liveModule.moduleRecord, PackageModuleExportRole::Prepare);
        if (!prepare)
          return fail(BoardRuntimeStage::EntryResolve,
                      firstRank->logicalRank, firstRank->entry,
                      detail::invalid(
                          "cluster module has no typed prepare export"));
        llvm::Expected<BoardFunctionHandle> resolvedPrepare =
            driver.resolveEntry(liveModule.module, prepare->symbol);
        if (!resolvedPrepare)
          return fail(BoardRuntimeStage::EntryResolve,
                      firstRank->logicalRank, firstRank->entry,
                      resolvedPrepare.takeError());
        clusterPrepareFunction = *resolvedPrepare;
      }
    }
  }
  result.completedStages.push_back(BoardRuntimeStage::EntryResolve);

  std::vector<BoardRankLaunch> launches;
  launches.reserve(capacityPlan->ranks.size());
  for (const RuntimeSessionPlan &rank : capacityPlan->ranks) {
    BoardRankLaunch launch;
    launch.logicalRank = rank.logicalRank;
    launch.entry = rank.entry;
    if (!modelLaunch) {
      auto function = mainFunctionsByModule.find(rank.module.getValue());
      if (function == mainFunctionsByModule.end())
        return fail(BoardRuntimeStage::EntryResolve, rank.logicalRank,
                    rank.entry,
                    detail::invalid("rank main export was not resolved"));
      launch.function = function->second;
    }
    launch.arguments.reserve(rank.launchOrder.size());
    for (ResourceId resource : rank.launchOrder) {
      auto memory = memoryByResource.find(resource.getValue());
      if (memory == memoryByResource.end())
        return fail(BoardRuntimeStage::Launch, rank.logicalRank, rank.entry,
                    detail::invalid("launch slot has no device allocation"));
      launch.arguments.push_back(static_cast<uint64_t>(memory->second.value));
    }
    launches.push_back(std::move(launch));
  }
  llvm::Error submissionError = [&]() -> llvm::Error {
    if (perRankLaunch)
      return driver.submitAll(launches);
    if (kernelGridLaunch)
      return driver.submitKernelGrid(launches);
    if (clusterDirectDTELaunch) {
      if (!clusterPrepareFunction)
        return detail::invalid(
            "cluster prepare export was not resolved");
      return driver.submitClusterPrepareMain(*clusterPrepareFunction,
                                             launches);
    }
    if (modelLaunch) {
      std::vector<BoardModelTensorLaunch> tensors;
      for (const RuntimeSessionPlan &rank : capacityPlan->ranks)
        for (auto [slotOrdinal, resourceId] :
             llvm::enumerate(rank.launchOrder)) {
          const PackageResourceRecord *resource =
              detail::findResource(manifest.resources, resourceId);
          auto memory = memoryByResource.find(resourceId.getValue());
          if (!resource || memory == memoryByResource.end())
            return detail::invalid(
                "model launch slot has no typed resource allocation");
          tensors.push_back({rank.logicalRank, slotOrdinal, resource->role,
                             memory->second, resource->bytes,
                             resource->type.dtype, resource->type.shape});
        }
      return driver.submitModel(*liveGraph, tensors);
    }
    return detail::invalid("package has an unknown launch ABI");
  }();
  if (submissionError)
    return fail(BoardRuntimeStage::Launch, -1, noEntry,
                std::move(submissionError));
  submissionLive = true;
  result.completedStages.push_back(BoardRuntimeStage::Launch);

  if (llvm::Error error = driver.waitAll(request.completionTimeoutMilliseconds))
    return fail(BoardRuntimeStage::Completion, -1, noEntry, std::move(error));
  result.completedStages.push_back(BoardRuntimeStage::Completion);

  if (clusterDirectDTELaunch) {
    size_t observedStatuses = 0;
    for (const LiveAllocation &allocation : allocations) {
      const PackageResourceRecord &resource = *allocation.resource;
      if (resource.role != PackageResourceRole::TransportStatus)
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
        std::memcpy(&status,
                    statusBytes.data() + kDirectDTEStatusValueOffset,
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
    if (observedStatuses != static_cast<size_t>(manifest.rankCount)) {
      driver.quarantine();
      return fail(BoardRuntimeStage::Completion, -1, noEntry,
                  detail::invalid(
                      "Direct DTE terminal status domain is incomplete"));
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
      if (clusterDirectDTELaunch)
        driver.quarantine();
      return fail(BoardRuntimeStage::DeviceToHost, resource.logicalRank,
                  allocation.entry, std::move(error));
    }
    result.outputs.push_back(std::move(output));
  }
  result.completedStages.push_back(BoardRuntimeStage::DeviceToHost);

  if (llvm::Error error = cleanup())
    return std::move(error);
  result.completedStages.push_back(BoardRuntimeStage::Cleanup);
  return result;
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
        "single-entry compatibility execution requires the unique entry of a "
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
