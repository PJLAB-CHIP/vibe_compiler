//===- BoardRuntime.cpp - Verified package board execution --------------===//

#include "Wafer/Runtime/BoardRuntime.h"

#include "PackageManifestInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <limits>
#include <optional>
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

const PackageResourceRecord *
findResourceForEntry(const PackageManifest &manifest,
                     const PackageEntrypointRecord &entry,
                     ResourceId resourceId) {
  const PackageResourceRecord *resource =
      detail::findResource(manifest.resources, resourceId);
  if (!resource || resource->logicalRank != entry.logicalRank)
    return nullptr;
  return resource;
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
  BoardDeviceMemory memory;
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
  stream << "board runtime " << stringifyBoardRuntimeStage(stage)
         << " failed for rank " << logicalRank << " entry " << entry.getValue()
         << " (context=" << stringifyBoardRuntimeContextState(contextState)
         << "): " << detail;
}

std::error_code BoardRuntimeError::convertToErrorCode() const {
  return llvm::make_error_code(llvm::errc::io_error);
}

llvm::Expected<BoardRuntimeResult>
executeBoardEntry(const VerifiedPackageManifest &package,
                  llvm::StringRef packageRoot, BoardRuntimeRequest request,
                  BoardRuntimeDriver &driver) {
  const PackageManifest &manifest = package.getManifest();
  auto entryIterator = llvm::find_if(manifest.entries, [&](const auto &entry) {
    return entry.id == request.entry;
  });
  if (entryIterator == manifest.entries.end())
    return boardError(BoardRuntimeStage::Preflight, -1, request.entry,
                      "entry ID is not present in package");
  const PackageEntrypointRecord &entry = *entryIterator;
  if (request.qualification.runtimeVersion == 0 ||
      request.qualification.tileCount == 0 ||
      request.qualification.name.empty() ||
      request.qualification.pciBusId.empty() ||
      request.qualification.runtimeLibraryDigest.empty())
    return boardError(BoardRuntimeStage::Preflight, entry.logicalRank, entry.id,
                      "board execution requires complete explicit device "
                      "qualification facts");
  if (!std::holds_alternative<NoTransportRequirements>(entry.transport))
    return boardError(BoardRuntimeStage::Preflight, entry.logicalRank, entry.id,
                      "single-entry execution cannot satisfy Direct DTE");
  const PackageModuleRecord *module =
      detail::findModule(manifest.modules, entry.module);
  if (!module)
    return boardError(BoardRuntimeStage::Preflight, entry.logicalRank, entry.id,
                      "entry module is missing");

  llvm::DenseMap<uint64_t, BoardRuntimeBinding *> bindingsByResource;
  for (BoardRuntimeBinding &binding : request.bindings) {
    uint64_t id = binding.resource.getValue();
    if (!binding.resource.isValid() || bindingsByResource.count(id))
      return boardError(BoardRuntimeStage::Preflight, entry.logicalRank,
                        entry.id,
                        "invocation has a duplicate or invalid ResourceId");
    const PackageResourceRecord *resource =
        findResourceForEntry(manifest, entry, binding.resource);
    if (!resource || !resource->hostVisible)
      return boardError(BoardRuntimeStage::Preflight, entry.logicalRank,
                        entry.id,
                        "invocation binds an unknown or internal ResourceId");
    if (binding.bytes.size() != resource->bytes)
      return boardError(BoardRuntimeStage::Preflight, entry.logicalRank,
                        entry.id, "invocation buffer byte count is not exact");
    bindingsByResource[id] = &binding;
  }

  std::vector<RuntimeInvocationBinding> preflightBindings;
  for (const PackageABISlotBinding &slot : entry.slots) {
    const PackageResourceRecord *resource =
        findResourceForEntry(manifest, entry, slot.resource);
    if (!resource)
      return boardError(BoardRuntimeStage::Preflight, entry.logicalRank,
                        entry.id, "entry slot resource is missing");
    if (!resource->hostVisible)
      continue;
    if (!bindingsByResource.count(resource->id.getValue()))
      return boardError(BoardRuntimeStage::Preflight, entry.logicalRank,
                        entry.id, "invocation omits a host-visible ResourceId");
    preflightBindings.push_back({resource->id, resource->bytes,
                                 resource->alignment, resource->access, true});
  }
  if (preflightBindings.size() != request.bindings.size())
    return boardError(BoardRuntimeStage::Preflight, entry.logicalRank, entry.id,
                      "invocation bindings are not all-and-only for entry");

  RuntimeEnvironment semanticEnvironment{
      manifest.targetProfile, manifest.targetIdentity, manifest.runtimeABI,
      manifest.moduleFormat, std::numeric_limits<uint64_t>::max()};
  llvm::Expected<RuntimeSessionPlan> semanticPlan =
      preflightNoCardRuntimeSession(package, entry.id, preflightBindings,
                                    semanticEnvironment);
  if (!semanticPlan)
    return wrapDriverError(BoardRuntimeStage::Preflight, entry.logicalRank,
                           entry.id, semanticPlan.takeError());

  llvm::Expected<std::vector<uint8_t>> moduleBytes =
      readVerifiedModule(packageRoot, *module, entry.logicalRank, entry.id);
  if (!moduleBytes)
    return moduleBytes.takeError();

  llvm::Expected<uint32_t> deviceCount = driver.getDeviceCount();
  if (!deviceCount)
    return wrapDriverError(BoardRuntimeStage::DeviceSelection,
                           entry.logicalRank, entry.id, deviceCount.takeError(),
                           driver.getContextState());
  if (request.deviceId >= *deviceCount)
    return boardError(BoardRuntimeStage::DeviceSelection, entry.logicalRank,
                      entry.id, "requested device is not present");
  if (llvm::Error error = driver.selectDevice(request.deviceId))
    return wrapDriverError(BoardRuntimeStage::DeviceSelection,
                           entry.logicalRank, entry.id, std::move(error),
                           driver.getContextState());
  llvm::Expected<BoardDeviceInfo> device =
      driver.getDeviceInfo(request.deviceId);
  if (!device)
    return wrapDriverError(BoardRuntimeStage::DeviceSelection,
                           entry.logicalRank, entry.id, device.takeError(),
                           driver.getContextState());
  if (device->runtimeVersion != request.qualification.runtimeVersion ||
      device->tileCount != request.qualification.tileCount ||
      device->name != request.qualification.name ||
      device->pciBusId != request.qualification.pciBusId ||
      device->runtimeLibraryDigest !=
          request.qualification.runtimeLibraryDigest)
    return boardError(BoardRuntimeStage::DeviceSelection, entry.logicalRank,
                      entry.id,
                      "live TX inventory does not match the explicit board "
                      "qualification");
  if (device->freeMemoryBytes > device->totalMemoryBytes)
    return boardError(BoardRuntimeStage::DeviceSelection, entry.logicalRank,
                      entry.id,
                      "TX runtime reported free memory greater than total "
                      "memory");
  RuntimeEnvironment capacityEnvironment{
      manifest.targetProfile, manifest.targetIdentity, manifest.runtimeABI,
      manifest.moduleFormat, device->freeMemoryBytes};
  llvm::Expected<RuntimeSessionPlan> capacityPlan =
      preflightNoCardRuntimeSession(package, entry.id, preflightBindings,
                                    capacityEnvironment);
  if (!capacityPlan)
    return wrapDriverError(BoardRuntimeStage::Preflight, entry.logicalRank,
                           entry.id, capacityPlan.takeError());

  uint64_t allocationBytes = 0;
  for (const PlannedRuntimeResource &resource : capacityPlan->resources) {
    if (resource.bytes > std::numeric_limits<uint64_t>::max() - allocationBytes)
      return boardError(BoardRuntimeStage::Preflight, entry.logicalRank,
                        entry.id,
                        "aggregate board allocation byte count overflows");
    allocationBytes += resource.bytes;
  }
  if (device->freeMemoryBytes <= boardRuntimeFreeMemoryReserve ||
      allocationBytes > device->freeMemoryBytes - boardRuntimeFreeMemoryReserve)
    return boardError(
        BoardRuntimeStage::Preflight, entry.logicalRank, entry.id,
        "aggregate board allocation demand exceeds qualified free memory "
        "after the runtime safety reserve");

  BoardRuntimeResult result;
  result.device = std::move(*device);
  result.entry = entry.id;
  result.logicalRank = entry.logicalRank;
  result.module = module->id;
  result.terminalCompletion = entry.terminalCompletion;
  result.completedStages.push_back(BoardRuntimeStage::Preflight);
  result.completedStages.push_back(BoardRuntimeStage::DeviceSelection);

  std::vector<LiveAllocation> allocations;
  std::optional<BoardModuleHandle> liveModule;
  auto cleanup = [&]() -> llvm::Error {
    llvm::Error cleanupError = llvm::Error::success();
    if (liveModule) {
      if (llvm::Error error = driver.unloadModule(*liveModule)) {
        BoardRuntimeContextState state = driver.getContextState();
        llvm::Error wrapped =
            wrapDriverError(BoardRuntimeStage::Cleanup, entry.logicalRank,
                            entry.id, std::move(error), state);
        if (state == BoardRuntimeContextState::Poisoned)
          return llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
        cleanupError =
            llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
      }
      liveModule.reset();
    }
    for (LiveAllocation &allocation : llvm::reverse(allocations)) {
      if (llvm::Error error = driver.free(allocation.memory)) {
        BoardRuntimeContextState state = driver.getContextState();
        llvm::Error wrapped =
            wrapDriverError(BoardRuntimeStage::Cleanup, entry.logicalRank,
                            entry.id, std::move(error), state);
        if (state == BoardRuntimeContextState::Poisoned)
          return llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
        cleanupError =
            llvm::joinErrors(std::move(cleanupError), std::move(wrapped));
      }
    }
    allocations.clear();
    return cleanupError;
  };
  auto fail = [&](BoardRuntimeStage stage,
                  llvm::Error error) -> llvm::Expected<BoardRuntimeResult> {
    BoardRuntimeContextState state = driver.getContextState();
    llvm::Error primary = wrapDriverError(stage, entry.logicalRank, entry.id,
                                          std::move(error), state);
    if (state == BoardRuntimeContextState::Poisoned)
      return std::move(primary);
    return llvm::joinErrors(std::move(primary), cleanup());
  };

  llvm::DenseMap<uint64_t, BoardDeviceMemory> memoryByResource;
  for (const PackageABISlotBinding &slot : entry.slots) {
    const PackageResourceRecord *resource =
        findResourceForEntry(manifest, entry, slot.resource);
    llvm::Expected<BoardDeviceMemory> memory =
        driver.allocate(resource->bytes, resource->alignment);
    if (!memory)
      return fail(BoardRuntimeStage::ResourceAllocation, memory.takeError());
    allocations.push_back({resource, *memory});
    memoryByResource[resource->id.getValue()] = *memory;
  }
  result.completedStages.push_back(BoardRuntimeStage::ResourceAllocation);

  for (const LiveAllocation &allocation : allocations) {
    const PackageResourceRecord &resource = *allocation.resource;
    if (resource.access == PackageAccessMode::WriteOnly)
      continue;
    llvm::ArrayRef<uint8_t> source;
    std::vector<uint8_t> zeros;
    auto binding = bindingsByResource.find(resource.id.getValue());
    if (binding != bindingsByResource.end()) {
      source = binding->second->bytes;
    } else {
      zeros.assign(static_cast<size_t>(resource.bytes), 0);
      source = zeros;
    }
    if (llvm::Error error = driver.copyHostToDevice(allocation.memory, source))
      return fail(BoardRuntimeStage::HostToDevice, std::move(error));
  }
  result.completedStages.push_back(BoardRuntimeStage::HostToDevice);

  llvm::Expected<BoardModuleHandle> loaded = driver.loadModule(*moduleBytes);
  if (!loaded)
    return fail(BoardRuntimeStage::ModuleLoad, loaded.takeError());
  liveModule = *loaded;
  result.completedStages.push_back(BoardRuntimeStage::ModuleLoad);

  llvm::Expected<BoardFunctionHandle> function =
      driver.resolveEntry(*liveModule, entry.symbol);
  if (!function)
    return fail(BoardRuntimeStage::EntryResolve, function.takeError());
  result.completedStages.push_back(BoardRuntimeStage::EntryResolve);

  std::vector<uint64_t> arguments;
  arguments.reserve(entry.slots.size());
  for (const PackageABISlotBinding &slot : entry.slots) {
    auto memory = memoryByResource.find(slot.resource.getValue());
    if (memory == memoryByResource.end())
      return fail(BoardRuntimeStage::Launch,
                  detail::invalid("launch slot has no device allocation"));
    arguments.push_back(static_cast<uint64_t>(memory->second.value));
  }
  if (llvm::Error error = driver.launch(*function, arguments))
    return fail(BoardRuntimeStage::Launch, std::move(error));
  result.completedStages.push_back(BoardRuntimeStage::Launch);

  if (llvm::Error error = driver.synchronize())
    return fail(BoardRuntimeStage::Completion, std::move(error));
  result.completedStages.push_back(BoardRuntimeStage::Completion);

  for (const LiveAllocation &allocation : allocations) {
    const PackageResourceRecord &resource = *allocation.resource;
    if (!resource.hostVisible || resource.access == PackageAccessMode::ReadOnly)
      continue;
    BoardRuntimeOutput output{resource.id,
                              std::vector<uint8_t>(resource.bytes)};
    if (llvm::Error error =
            driver.copyDeviceToHost(output.bytes, allocation.memory))
      return fail(BoardRuntimeStage::DeviceToHost, std::move(error));
    result.outputs.push_back(std::move(output));
  }
  result.completedStages.push_back(BoardRuntimeStage::DeviceToHost);

  if (llvm::Error error = cleanup())
    return std::move(error);
  result.completedStages.push_back(BoardRuntimeStage::Cleanup);
  return result;
}

} // namespace wafer::runtime
