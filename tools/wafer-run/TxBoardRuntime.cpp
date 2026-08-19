//===- TxBoardRuntime.cpp - TX public-runtime board provider ------------===//

#include "Wafer/Runtime/Board/TxBoardRuntime.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include "tx_runtime.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wafer::runtime {
namespace {

struct TxApi {
  decltype(&txGetDeviceCount) getDeviceCount = nullptr;
  decltype(&txSetDevice) setDevice = nullptr;
  decltype(&txGetDeviceProperty) getDeviceProperty = nullptr;
  decltype(&txGetDeviceAllTileInfo) getDeviceAllTileInfo = nullptr;
  decltype(&txMemGetInfo) memGetInfo = nullptr;
  decltype(&txRuntimeGetVersion) runtimeGetVersion = nullptr;
  decltype(&txGetDevicePCIBusId) getDevicePCIBusId = nullptr;
  decltype(&txMalloc) malloc = nullptr;
  decltype(&txFree) free = nullptr;
  decltype(&txMemcpy) memcpy = nullptr;
  decltype(&txModuleLoad) moduleLoad = nullptr;
  decltype(&txModuleUnload) moduleUnload = nullptr;
  decltype(&txModuleGetFunction) moduleGetFunction = nullptr;
  decltype(&txLaunchKernel) launchKernel = nullptr;
  decltype(&txLaunchClusterKernel) launchClusterKernel = nullptr;
  decltype(&txStreamCreate) streamCreate = nullptr;
  decltype(&txStreamDestroy) streamDestroy = nullptr;
  decltype(&txStreamQuery) streamQuery = nullptr;
  decltype(&txEventCreate) eventCreate = nullptr;
  decltype(&txEventDestroy) eventDestroy = nullptr;
  decltype(&txEventRecord) eventRecord = nullptr;
  decltype(&txEventQuery) eventQuery = nullptr;
  decltype(&txEventElapsedTime) eventElapsedTime = nullptr;
};

RuntimeEnvironment makeTxProviderEnvironment() {
  RuntimeEnvironment environment(TargetIdentityId::waferTx81SingleCard(),
                                 KernelRuntimeABIId::waferTx81Kernel(),
                                 kCurrentTargetModuleFormat);
  environment.supportedKernelLaunchForms = {
      KernelLaunchForm::Grid,
      KernelLaunchForm::Cluster,
  };
  environment.supportedKernelEntryABIs = {
      KernelEntryABI::TileMajorPointerTable,
      KernelEntryABI::TileRowPointerTable,
  };
  environment.supportsDirectDTE = true;
  environment.directDTEStatusABI = kDirectDTEStatusABI.str();
  environment.supportsHostWatchdog = true;
  return environment;
}

enum class SubmissionKind { None, Kernel };

class ScopedFD {
public:
  explicit ScopedFD(int descriptor) : descriptor(descriptor) {}
  ~ScopedFD() {
    if (descriptor >= 0)
      close(descriptor);
  }

  ScopedFD(const ScopedFD &) = delete;
  ScopedFD &operator=(const ScopedFD &) = delete;

  int get() const { return descriptor; }

private:
  int descriptor = -1;
};

template <typename Function>
llvm::Expected<Function> resolveTxSymbol(void *handle, llvm::StringRef name,
                                         dev_t expectedDevice,
                                         ino_t expectedInode) {
  dlerror();
  void *address = dlsym(handle, name.str().c_str());
  const char *error = dlerror();
  if (error || !address)
    return llvm::createStringError(
        llvm::errc::invalid_argument, "failed to resolve TX symbol %s: %s",
        name.str().c_str(), error ? error : "null address");
  Dl_info information{};
  if (dladdr(address, &information) == 0 || !information.dli_fname)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "dladdr failed for TX symbol %s",
                                   name.str().c_str());
  struct stat providerStatus{};
  if (stat(information.dli_fname, &providerStatus) != 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to stat provider for TX symbol %s", name.str().c_str());
  if (providerStatus.st_dev != expectedDevice ||
      providerStatus.st_ino != expectedInode)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "TX symbol %s resolved from an unqualified provider",
        name.str().c_str());
  return reinterpret_cast<Function>(address);
}

llvm::Expected<std::string> hashOpenRuntimeLibrary(int descriptor) {
  if (lseek(descriptor, 0, SEEK_SET) < 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to rewind the configured TX runtime library");
  llvm::SHA256 hasher;
  std::array<char, 64 * 1024> buffer;
  while (true) {
    ssize_t count = read(descriptor, buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR)
      continue;
    if (count < 0)
      return llvm::createStringError(
          std::error_code(errno, std::generic_category()),
          "failed to read the configured TX runtime library");
    if (count == 0)
      break;
    hasher.update(llvm::StringRef(buffer.data(), static_cast<size_t>(count)));
  }
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

class TxBoardRuntimeDriver final : public BoardRuntimeDriver {
public:
  TxBoardRuntimeDriver(void *library, TxApi api,
                       std::string runtimeLibraryDigest)
      : library(library), api(api),
        runtimeLibraryDigest(std::move(runtimeLibraryDigest)),
        providerEnvironment(makeTxProviderEnvironment()) {}

  // The board CLI is a one-shot process and exits with std::_Exit after the
  // explicit TX lifecycle. The handle intentionally remains process-owned so
  // neither success nor quarantine invokes unqualified provider finalizers.
  ~TxBoardRuntimeDriver() override = default;

  BoardRuntimeContextState getContextState() const override {
    return contextState;
  }

  void quarantine() override {
    contextState = BoardRuntimeContextState::Poisoned;
  }

  const RuntimeEnvironment &getProviderEnvironment() const override {
    return providerEnvironment;
  }

  llvm::Expected<uint32_t> getDeviceCount() override {
    if (llvm::Error error = requireUsable("txGetDeviceCount"))
      return std::move(error);
    uint32_t count = 0;
    txError_t status = api.getDeviceCount(&count);
    if (status != TX_SUCCESS)
      return txError("txGetDeviceCount", status);
    return count;
  }

  llvm::Error selectDevice(uint32_t deviceId) override {
    if (llvm::Error error = requireUsable("txSetDevice"))
      return error;
    return check("txSetDevice", api.setDevice(deviceId));
  }

  llvm::Expected<BoardDeviceInfo> getDeviceInfo(uint32_t deviceId) override {
    if (llvm::Error error = requireUsable("getDeviceInfo"))
      return std::move(error);
    txDeviceProperty property{};
    txError_t status = api.getDeviceProperty(deviceId, &property);
    if (status != TX_SUCCESS)
      return txError("txGetDeviceProperty", status);
    uint64_t freeBytes = 0;
    uint64_t totalBytes = 0;
    status = api.memGetInfo(&freeBytes, &totalBytes);
    if (status != TX_SUCCESS)
      return txError("txMemGetInfo", status);
    uint32_t runtimeVersion = 0;
    status = api.runtimeGetVersion(&runtimeVersion);
    if (status != TX_SUCCESS)
      return txError("txRuntimeGetVersion", status);
    char pciBusId[32] = {};
    status = api.getDevicePCIBusId(pciBusId, sizeof(pciBusId), deviceId);
    if (status != TX_SUCCESS)
      return txError("txGetDevicePCIBusId", status);
    tileTotalInfo tileInfo{};
    status = api.getDeviceAllTileInfo(deviceId, &tileInfo);
    if (status != TX_SUCCESS)
      return txError("txGetDeviceAllTileInfo", status);

    BoardDeviceInfo info;
    info.deviceId = deviceId;
    info.runtimeVersion = runtimeVersion;
    info.freeMemoryBytes = freeBytes;
    info.totalMemoryBytes = totalBytes;
    info.tileCount = property.tileProp.tileNum;
    info.name.assign(property.devProp.devName,
                     strnlen(property.devProp.devName, NPU_NAME_LENGTH));
    info.pciBusId.assign(pciBusId, strnlen(pciBusId, sizeof(pciBusId)));
    info.runtimeLibraryDigest = runtimeLibraryDigest;
    info.tiles.reserve(NPU_TILE_COUNT_MAX);
    for (const tileFullInfo &tile : tileInfo.tilesFullInfo) {
      if (tile.phyTilex >= 4 || tile.phyTiley >= 4)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "TX inventory contains an out-of-range Tile coordinate");
      const int64_t tileId =
          static_cast<int64_t>(tile.phyTiley) * 4 + tile.phyTilex;
      info.tiles.push_back({TileId(tileId),
                            LaunchSlotId(tile.index), tile.isAvailable == 1,
                            tile.phyTilex, tile.phyTiley});
    }
    return info;
  }

  llvm::Expected<BoardDeviceMemory> allocate(uint64_t bytes,
                                             uint64_t alignment) override {
    if (llvm::Error error = requireUsable("txMalloc"))
      return std::move(error);
    if (alignment == 0)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "TX allocation alignment is zero");
    void *pointer = nullptr;
    txError_t status = api.malloc(&pointer, bytes);
    if (status != TX_SUCCESS)
      return txError("txMalloc", status);
    uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    if (address == 0 || address % alignment != 0)
      return poisonContractViolation(
          "txMalloc returned a null or misaligned address");
    return BoardDeviceMemory{address};
  }

  llvm::Error free(BoardDeviceMemory memory) override {
    if (llvm::Error error = requireUsable("txFree"))
      return error;
    return check("txFree", api.free(reinterpret_cast<void *>(memory.value)));
  }

  llvm::Error copyHostToDevice(BoardDeviceMemory destination,
                               llvm::ArrayRef<uint8_t> source) override {
    if (llvm::Error error = requireUsable("txMemcpy(H2D)"))
      return error;
    return check("txMemcpy(H2D)",
                 api.memcpy(reinterpret_cast<void *>(destination.value),
                            source.data(), source.size(),
                            txMemcpyHostToDevice));
  }

  llvm::Error copyDeviceToHost(llvm::MutableArrayRef<uint8_t> destination,
                               BoardDeviceMemory source) override {
    if (llvm::Error error = requireUsable("txMemcpy(D2H)"))
      return error;
    return check("txMemcpy(D2H)",
                 api.memcpy(destination.data(),
                            reinterpret_cast<const void *>(source.value),
                            destination.size(), txMemcpyDeviceToHost));
  }

  llvm::Expected<BoardModuleHandle>
  loadModule(llvm::ArrayRef<uint8_t> moduleBytes) override {
    if (llvm::Error error = requireUsable("txModuleLoad"))
      return std::move(error);
    if (!api.moduleLoad)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "TX module loading is unavailable");
    if (moduleBytes.empty() ||
        moduleBytes.size() > std::numeric_limits<uint32_t>::max())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX module byte count must be nonzero and fit uint32_t");
    llvm::SHA256 hasher;
    hasher.update(
        llvm::StringRef(reinterpret_cast<const char *>(moduleBytes.data()),
                        moduleBytes.size()));
    std::string digest =
        "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
    txModule_t module = nullptr;
    txError_t status = api.moduleLoad(
        &module, reinterpret_cast<const char *>(moduleBytes.data()),
        static_cast<uint32_t>(moduleBytes.size()));
    if (status != TX_SUCCESS)
      return txError("txModuleLoad", status);
    if (!module)
      return poisonContractViolation(
          "txModuleLoad returned success with a null module");
    uintptr_t handle = reinterpret_cast<uintptr_t>(module);
    auto iterator = loadedModules.find(handle);
    if (iterator == loadedModules.end()) {
      loadedModules.emplace(handle, LoadedModuleState{std::move(digest), 1});
    } else {
      if (iterator->second.digest != digest)
        return poisonContractViolation(
            "txModuleLoad aliased different code objects to one module handle");
      if (iterator->second.referenceCount ==
          std::numeric_limits<uint64_t>::max())
        return poisonContractViolation(
            "TX module ownership reference count overflowed");
      ++iterator->second.referenceCount;
    }
    return BoardModuleHandle{handle};
  }

  llvm::Error unloadModule(BoardModuleHandle module) override {
    if (llvm::Error error = requireUsable("txModuleUnload"))
      return error;
    if (!api.moduleUnload)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "TX module unloading is unavailable");
    auto iterator = loadedModules.find(module.value);
    if (iterator == loadedModules.end())
      return poisonContractViolation(
          "TX module ownership is missing during unload");
    if (iterator->second.referenceCount > 1) {
      --iterator->second.referenceCount;
      return llvm::Error::success();
    }
    if (llvm::Error error =
            check("txModuleUnload",
                  api.moduleUnload(reinterpret_cast<txModule_t>(module.value))))
      return error;
    loadedModules.erase(iterator);
    return llvm::Error::success();
  }

  llvm::Expected<BoardFunctionHandle>
  resolveEntry(BoardModuleHandle module, llvm::StringRef symbol) override {
    if (llvm::Error error = requireUsable("txModuleGetFunction"))
      return std::move(error);
    if (!api.moduleGetFunction || symbol.empty() || symbol.contains('\0'))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX module entry resolution has an invalid symbol");
    txFunction_t function = nullptr;
    std::string ownedSymbol = symbol.str();
    txError_t status = api.moduleGetFunction(
        &function, reinterpret_cast<txModule_t>(module.value),
        ownedSymbol.c_str());
    if (status != TX_SUCCESS)
      return txError("txModuleGetFunction", status);
    if (!function)
      return poisonContractViolation(
          "txModuleGetFunction returned success with a null function");
    return BoardFunctionHandle{reinterpret_cast<uintptr_t>(function)};
  }

  llvm::Error submitKernelPhase(KernelLaunchForm form,
                                RuntimeLaunchPhaseRole phaseRole,
                                llvm::ArrayRef<BoardTileLaunch> launches,
                                BoardDeviceTimingPolicy timingPolicy) override {
    if (llvm::Error error = requireUsable("kernel phase submission"))
      return error;
    const bool cluster = form == KernelLaunchForm::Cluster;
    if ((cluster && !api.launchClusterKernel) ||
        (!cluster && !api.launchKernel))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX %s kernel launch form is unavailable",
          stringifyKernelLaunchForm(form).str().c_str());
    if (launches.empty() ||
        launches.size() > std::numeric_limits<uint32_t>::max())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX kernel phase requires a nonempty uint32_t Tile domain");

    const bool firstPhase = !submissionActive;
    if (firstPhase) {
      if (submissionKind != SubmissionKind::None || phaseSubmitted ||
          activeKernelForm || activeKernelPhase || !activeStreams.empty() ||
          !completedStreams.empty() || !submissionArgumentBlocks.empty() ||
          !submissionEntries.empty() || activeDeviceTimingPolicy ||
          timingStartEvent || timingEndEvent)
        return poisonContractViolation(
            "TX provider has stale state before a kernel submission");
    } else if (submissionKind != SubmissionKind::Kernel || !activeKernelForm ||
               *activeKernelForm != form || phaseSubmitted ||
               !activeDeviceTimingPolicy ||
               *activeDeviceTimingPolicy != timingPolicy ||
               activeStreams.empty() ||
               completedStreams.size() != activeStreams.size() ||
               !llvm::all_of(completedStreams,
                             [](bool complete) { return complete; })) {
      return poisonContractViolation(
          "TX later kernel phase requires a terminal prior phase with the "
          "same launch form");
    }

    const uintptr_t sharedFunction = launches.front().function.value;
    const size_t sharedSlotCount = launches.front().arguments.size();
    std::vector<std::vector<uint64_t>> argumentBlocks(1);

    std::array<bool, 16> seenTileIds{};
    for (auto [launchIndex, launch] : llvm::enumerate(launches)) {
      const int64_t tileId = launch.tileId.getValue();
      if (launch.cardId != CardId(0) || tileId < 0 || tileId >= 16 ||
          seenTileIds[tileId] ||
          launch.launchSlot != LaunchSlotId(launchIndex) ||
          !launch.entry.isValid() || launch.function.value == 0)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "TX kernel phase is not a canonical Tile/function "
            "domain");
      seenTileIds[tileId] = true;
      if (launch.function.value != sharedFunction ||
          launch.arguments.size() != sharedSlotCount)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "TX shared kernel phase does not use one function and slot shape");
      argumentBlocks.front().insert(argumentBlocks.front().end(),
                                    launch.arguments.begin(),
                                    launch.arguments.end());
    }

    const uint64_t byteLimit = cluster ? kTx81ClusterKernelArgumentBytesMax
                                       : kTx81KernelArgumentBytesMax;
    if (argumentBlocks.front().size() > byteLimit / sizeof(uint64_t))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX shared Tile-major argument table exceeds the qualified packet "
          "limit");

    if (firstPhase) {
      submissionArgumentBlocks = argumentBlocks;
      submissionEntries.reserve(launches.size());
      for (const BoardTileLaunch &launch : launches)
        submissionEntries.push_back(launch.entry);

      activeStreams.reserve(1);
      txStream_t stream = nullptr;
      txError_t status = api.streamCreate(&stream);
      if (status != TX_SUCCESS)
        return txError("txStreamCreate(kernel)", status);
      if (!stream)
        return poisonContractViolation(
            "txStreamCreate(kernel) returned a null stream");
      activeStreams.push_back(stream);
      if (llvm::Error error = initializeDeviceTiming(timingPolicy, "kernel"))
        return error;
      submissionKind = SubmissionKind::Kernel;
      activeKernelForm = form;
      submissionActive = true;
    } else {
      if (submissionEntries.size() != launches.size() ||
          submissionArgumentBlocks != argumentBlocks)
        return poisonContractViolation(
            "TX later kernel phase changed Tile identity or argument storage");
      for (auto [index, launch] : llvm::enumerate(launches))
        if (submissionEntries[index] != launch.entry)
          return poisonContractViolation(
              "TX later kernel phase changed Tile entry identity");
    }

    completedStreams.assign(activeStreams.size(), false);
    activeKernelPhase = phaseRole;
    phaseSubmitted = true;
    const std::string phaseName =
        stringifyRuntimeLaunchPhaseRole(phaseRole).str();
    dim3 blockDim = {1, 1, 1};
    if (llvm::Error error =
            recordDeviceTimingStart(activeStreams.front(), "kernel"))
      return error;
    dim3 gridDim = {static_cast<uint32_t>(launches.size()), 1, 1};
    txError_t status;
    if (cluster) {
      dim3 clusterDim = {1, 1, 1};
      status = api.launchClusterKernel(
          reinterpret_cast<txFunction_t>(sharedFunction), clusterDim, gridDim,
          blockDim, submissionArgumentBlocks.front().data(),
          static_cast<uint32_t>(submissionArgumentBlocks.front().size() *
                                sizeof(uint64_t)),
          0, activeStreams.front());
      if (status != TX_SUCCESS)
        return txError("txLaunchClusterKernel(" + phaseName + ")", status);
    } else {
      status = api.launchKernel(
          reinterpret_cast<txFunction_t>(sharedFunction), gridDim, blockDim,
          submissionArgumentBlocks.front().data(),
          static_cast<uint32_t>(submissionArgumentBlocks.front().size() *
                                sizeof(uint64_t)),
          0, activeStreams.front());
      if (status != TX_SUCCESS)
        return txError("txLaunchKernel(grid:" + phaseName + ")", status);
    }
    if (llvm::Error error =
            recordDeviceTimingEnd(activeStreams.front(), "kernel"))
      return error;
    return llvm::Error::success();
  }

  llvm::Expected<BoardCompletionObservation> waitCurrentSubmission(
      BoardCompletionDeadline deadline,
      BoardCompletionObservationPolicy observationPolicy) override {
    if (llvm::Error error = requireUsable("submission phase completion"))
      return error;
    if (!submissionActive || submissionKind == SubmissionKind::None ||
        !phaseSubmitted || activeStreams.empty() ||
        completedStreams.size() != activeStreams.size() ||
        !activeDeviceTimingPolicy)
      return poisonContractViolation(
          "TX phase completion has no live submitted phase");
    const bool deviceTimingEnabled =
        *activeDeviceTimingPolicy == BoardDeviceTimingPolicy::StreamEvents;
    const bool deviceTimingStateIsValid =
        deviceTimingEnabled ? timingStartEvent && timingEndEvent
                            : !timingStartEvent && !timingEndEvent;
    if (!deviceTimingStateIsValid)
      return poisonContractViolation(
          "TX phase completion has inconsistent device-timing state");

    const auto waitBegin = std::chrono::steady_clock::now();
    std::vector<std::chrono::steady_clock::time_point> lastObservations(
        activeStreams.size(), waitBegin);
    std::optional<std::chrono::steady_clock::time_point> lastTimingObservation;
    uint64_t maximumPollGapNanoseconds = 0;
    auto recordObservation =
        [&](size_t streamIndex,
            std::chrono::steady_clock::time_point observationTime) {
          const auto observedGap =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  observationTime - lastObservations[streamIndex])
                  .count();
          if (observedGap > 0)
            maximumPollGapNanoseconds = std::max(
                maximumPollGapNanoseconds, static_cast<uint64_t>(observedGap));
          lastObservations[streamIndex] = observationTime;
        };
    auto deadlineExceeded = [&]() -> llvm::Error {
      contextState = BoardRuntimeContextState::Poisoned;
      std::string submissionPhase = "kernel";
      if (submissionKind == SubmissionKind::Kernel && activeKernelForm &&
          activeKernelPhase)
        submissionPhase = (stringifyKernelLaunchForm(*activeKernelForm) + ":" +
                           stringifyRuntimeLaunchPhaseRole(*activeKernelPhase))
                              .str();
      return llvm::createStringError(
          llvm::errc::io_error, "TX %s completion exceeded the host deadline",
          submissionPhase.c_str());
    };
    while (true) {
      bool allComplete = true;
      for (size_t index = 0; index < activeStreams.size(); ++index) {
        if (completedStreams[index])
          continue;
        if (std::chrono::steady_clock::now() >= deadline)
          return deadlineExceeded();
        txError_t status = api.streamQuery(activeStreams[index]);
        const auto observationTime = std::chrono::steady_clock::now();
        recordObservation(index, observationTime);
        if (status == TX_SUCCESS) {
          completedStreams[index] = true;
        } else if (status == TX_ERROR_NOT_READY) {
          allComplete = false;
        } else {
          return txError("txStreamQuery", status);
        }
        // A completion observation counts only when the query itself returned
        // before the host deadline. Once expired, this local quarantine is the
        // final action and no later stream is queried.
        if (observationTime >= deadline)
          return deadlineExceeded();
      }
      if (allComplete) {
        if (!deviceTimingEnabled) {
          phaseSubmitted = false;
          return BoardCompletionObservation{maximumPollGapNanoseconds,
                                            std::nullopt};
        }
        if (std::chrono::steady_clock::now() >= deadline)
          return deadlineExceeded();
        txError_t status = api.eventQuery(timingEndEvent);
        const auto observationTime = std::chrono::steady_clock::now();
        if (lastTimingObservation) {
          const auto observedGap =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  observationTime - *lastTimingObservation)
                  .count();
          if (observedGap > 0)
            maximumPollGapNanoseconds = std::max(
                maximumPollGapNanoseconds, static_cast<uint64_t>(observedGap));
        }
        lastTimingObservation = observationTime;
        if (status == TX_SUCCESS) {
          llvm::Expected<std::optional<uint64_t>> deviceTiming =
              readDeviceTiming();
          if (!deviceTiming)
            return deviceTiming.takeError();
          if (!deviceTiming->has_value())
            return poisonContractViolation(
                "TX stream-event completion omitted device timing");
          if (std::chrono::steady_clock::now() >= deadline)
            return deadlineExceeded();
          phaseSubmitted = false;
          return BoardCompletionObservation{maximumPollGapNanoseconds,
                                            **deviceTiming};
        }
        if (status == TX_ERROR_NOT_READY) {
          allComplete = false;
        } else {
          return txError("txEventQuery(end)", status);
        }
        if (observationTime >= deadline)
          return deadlineExceeded();
      }
      if (observationPolicy == BoardCompletionObservationPolicy::Normal)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  llvm::Error releaseSubmission() override {
    if (llvm::Error error = requireUsable("txStreamDestroy"))
      return error;
    if (!submissionActive || phaseSubmitted ||
        !llvm::all_of(completedStreams, [](bool complete) { return complete; }))
      return poisonContractViolation(
          "TX submission release requires a terminal current phase");
    if (llvm::Error error = destroyDeviceTiming())
      return error;
    while (!activeStreams.empty()) {
      if (llvm::Error error =
              check("txStreamDestroy", api.streamDestroy(activeStreams.back())))
        return error;
      activeStreams.pop_back();
    }
    submissionArgumentBlocks.clear();
    submissionEntries.clear();
    completedStreams.clear();
    activeKernelForm.reset();
    activeKernelPhase.reset();
    submissionKind = SubmissionKind::None;
    phaseSubmitted = false;
    submissionActive = false;
    return llvm::Error::success();
  }

private:
  llvm::Error initializeDeviceTiming(BoardDeviceTimingPolicy timingPolicy,
                                     llvm::StringRef operation) {
    if (activeDeviceTimingPolicy || timingStartEvent || timingEndEvent)
      return poisonContractViolation(
          "TX provider has stale device-timing state");
    activeDeviceTimingPolicy = timingPolicy;
    if (timingPolicy == BoardDeviceTimingPolicy::Disabled)
      return llvm::Error::success();

    txError_t status = api.eventCreate(&timingStartEvent);
    if (status != TX_SUCCESS)
      return txError((operation + " txEventCreate(start)").str(), status);
    if (!timingStartEvent)
      return poisonContractViolation(
          (operation + " txEventCreate(start) returned a null event").str());
    status = api.eventCreate(&timingEndEvent);
    if (status != TX_SUCCESS)
      return txError((operation + " txEventCreate(end)").str(), status);
    if (!timingEndEvent)
      return poisonContractViolation(
          (operation + " txEventCreate(end) returned a null event").str());
    return llvm::Error::success();
  }

  llvm::Error recordDeviceTimingStart(txStream_t stream,
                                      llvm::StringRef operation) {
    if (!activeDeviceTimingPolicy)
      return poisonContractViolation(
          "TX device-timing policy is missing before submission");
    if (*activeDeviceTimingPolicy == BoardDeviceTimingPolicy::Disabled) {
      if (timingStartEvent || timingEndEvent)
        return poisonContractViolation(
            "TX disabled device timing owns event handles");
      return llvm::Error::success();
    }
    if (!timingStartEvent || !timingEndEvent)
      return poisonContractViolation(
          "TX stream-event timing is missing event handles");
    txError_t status = api.eventRecord(timingStartEvent, stream);
    if (status != TX_SUCCESS)
      return txError((operation + " txEventRecord(start)").str(), status);
    return llvm::Error::success();
  }

  llvm::Error recordDeviceTimingEnd(txStream_t stream,
                                    llvm::StringRef operation) {
    if (!activeDeviceTimingPolicy)
      return poisonContractViolation(
          "TX device-timing policy is missing after submission");
    if (*activeDeviceTimingPolicy == BoardDeviceTimingPolicy::Disabled)
      return llvm::Error::success();
    if (!timingStartEvent || !timingEndEvent)
      return poisonContractViolation(
          "TX stream-event timing is missing event handles");
    txError_t status = api.eventRecord(timingEndEvent, stream);
    if (status != TX_SUCCESS)
      return txError((operation + " txEventRecord(end)").str(), status);
    return llvm::Error::success();
  }

  llvm::Expected<std::optional<uint64_t>> readDeviceTiming() {
    if (!activeDeviceTimingPolicy)
      return poisonContractViolation(
          "TX device-timing policy is missing at completion");
    if (*activeDeviceTimingPolicy == BoardDeviceTimingPolicy::Disabled) {
      if (timingStartEvent || timingEndEvent)
        return poisonContractViolation(
            "TX disabled device timing owns event handles");
      return std::optional<uint64_t>();
    }
    if (!timingStartEvent || !timingEndEvent)
      return poisonContractViolation(
          "TX stream-event timing is missing event handles");

    float elapsedMilliseconds = 0.0f;
    txError_t status = api.eventElapsedTime(&elapsedMilliseconds,
                                            timingStartEvent, timingEndEvent);
    if (status != TX_SUCCESS)
      return txError("txEventElapsedTime", status);
    if (!std::isfinite(elapsedMilliseconds) || elapsedMilliseconds < 0.0f)
      return poisonContractViolation(
          "txEventElapsedTime returned a non-finite or negative duration");

    // The TX event API reports float milliseconds. Round to the nearest
    // integer nanosecond after checking the full long-double conversion
    // domain. A valid 0.0f remains a valid, resolution-quantized zero sample.
    const long double elapsedNanoseconds =
        static_cast<long double>(elapsedMilliseconds) * 1000000.0L;
    const long double roundedNanoseconds = std::round(elapsedNanoseconds);
    const long double uint64UpperExclusive = std::ldexp(1.0L, 64);
    if (roundedNanoseconds >= uint64UpperExclusive)
      return poisonContractViolation(
          "txEventElapsedTime duration overflows nanoseconds");
    return std::optional<uint64_t>{static_cast<uint64_t>(roundedNanoseconds)};
  }

  llvm::Error destroyDeviceTiming() {
    if (!activeDeviceTimingPolicy)
      return poisonContractViolation(
          "TX submission release is missing device-timing state");
    if (*activeDeviceTimingPolicy == BoardDeviceTimingPolicy::Disabled) {
      if (timingStartEvent || timingEndEvent)
        return poisonContractViolation(
            "TX disabled device timing owns event handles");
      activeDeviceTimingPolicy.reset();
      return llvm::Error::success();
    }
    if (!timingStartEvent || !timingEndEvent)
      return poisonContractViolation(
          "TX stream-event timing is missing event handles");
    if (llvm::Error error =
            check("txEventDestroy(end)", api.eventDestroy(timingEndEvent)))
      return error;
    timingEndEvent = nullptr;
    if (llvm::Error error =
            check("txEventDestroy(start)", api.eventDestroy(timingStartEvent)))
      return error;
    timingStartEvent = nullptr;
    activeDeviceTimingPolicy.reset();
    return llvm::Error::success();
  }

  llvm::Error requireUsable(llvm::StringRef operation) const {
    if (contextState == BoardRuntimeContextState::Usable)
      return llvm::Error::success();
    return llvm::createStringError(
        llvm::errc::io_error,
        "%s suppressed because the TX context is poisoned",
        operation.str().c_str());
  }

  llvm::Error txError(llvm::StringRef operation, txError_t status) {
    contextState = BoardRuntimeContextState::Poisoned;
    return llvm::createStringError(llvm::errc::io_error, "%s returned 0x%08x",
                                   operation.str().c_str(),
                                   static_cast<unsigned>(status));
  }

  llvm::Error check(llvm::StringRef operation, txError_t status) {
    if (status == TX_SUCCESS)
      return llvm::Error::success();
    return txError(operation, status);
  }

  llvm::Error poisonContractViolation(llvm::StringRef detail) {
    contextState = BoardRuntimeContextState::Poisoned;
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   detail.str().c_str());
  }

  BoardRuntimeContextState contextState = BoardRuntimeContextState::Usable;
  struct LoadedModuleState {
    std::string digest;
    uint64_t referenceCount = 0;
  };
  void *library = nullptr;
  TxApi api;
  std::string runtimeLibraryDigest;
  RuntimeEnvironment providerEnvironment;
  std::unordered_map<uintptr_t, LoadedModuleState> loadedModules;
  std::vector<txStream_t> activeStreams;
  std::vector<bool> completedStreams;
  std::vector<std::vector<uint64_t>> submissionArgumentBlocks;
  std::vector<EntryId> submissionEntries;
  SubmissionKind submissionKind = SubmissionKind::None;
  std::optional<KernelLaunchForm> activeKernelForm;
  std::optional<RuntimeLaunchPhaseRole> activeKernelPhase;
  std::optional<BoardDeviceTimingPolicy> activeDeviceTimingPolicy;
  txEvent_t timingStartEvent = nullptr;
  txEvent_t timingEndEvent = nullptr;
  bool phaseSubmitted = false;
  bool submissionActive = false;
};

} // namespace

llvm::Expected<std::unique_ptr<BoardRuntimeDriver>>
createTxBoardRuntimeDriver(llvm::StringRef expectedRuntimeLibraryDigest) {
  llvm::StringRef configuredLibrary = WAFER_TX_RUNTIME_LIBRARY_PATH;
  int descriptor =
      open(configuredLibrary.str().c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to open the configured TX runtime library");
  ScopedFD ownedDescriptor(descriptor);
  struct stat beforeStatus{};
  if (fstat(descriptor, &beforeStatus) != 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to stat the configured TX runtime library");
  if (!S_ISREG(beforeStatus.st_mode))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "configured TX runtime library is not a regular file");
  llvm::Expected<std::string> digest = hashOpenRuntimeLibrary(descriptor);
  if (!digest)
    return digest.takeError();
  if (*digest != expectedRuntimeLibraryDigest)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "configured TX runtime library digest does not match the explicit "
        "qualification");

  if (lseek(descriptor, 0, SEEK_SET) < 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to rewind the qualified TX runtime library before loading");
  std::string descriptorPath = "/proc/self/fd/" + std::to_string(descriptor);
  void *library = dlopen(descriptorPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "failed to load configured TX runtime: %s",
                                   dlerror());
  llvm::Expected<std::string> loadedDigest = hashOpenRuntimeLibrary(descriptor);
  if (!loadedDigest)
    return loadedDigest.takeError();
  struct stat afterStatus{};
  if (fstat(descriptor, &afterStatus) != 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to restat the loaded TX runtime library");
  if (*loadedDigest != *digest || afterStatus.st_dev != beforeStatus.st_dev ||
      afterStatus.st_ino != beforeStatus.st_ino ||
      afterStatus.st_size != beforeStatus.st_size)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "TX runtime library changed while it was being qualified and loaded");
  TxApi api;
#define WAFER_RESOLVE_TX_API(field, symbol)                                    \
  do {                                                                         \
    llvm::Expected<decltype(api.field)> resolved =                             \
        resolveTxSymbol<decltype(api.field)>(                                  \
            library, #symbol, beforeStatus.st_dev, beforeStatus.st_ino);       \
    if (!resolved)                                                             \
      return resolved.takeError();                                             \
    api.field = *resolved;                                                     \
  } while (false)
  WAFER_RESOLVE_TX_API(getDeviceCount, txGetDeviceCount);
  WAFER_RESOLVE_TX_API(setDevice, txSetDevice);
  WAFER_RESOLVE_TX_API(getDeviceProperty, txGetDeviceProperty);
  WAFER_RESOLVE_TX_API(getDeviceAllTileInfo, txGetDeviceAllTileInfo);
  WAFER_RESOLVE_TX_API(memGetInfo, txMemGetInfo);
  WAFER_RESOLVE_TX_API(runtimeGetVersion, txRuntimeGetVersion);
  WAFER_RESOLVE_TX_API(getDevicePCIBusId, txGetDevicePCIBusId);
  WAFER_RESOLVE_TX_API(malloc, txMalloc);
  WAFER_RESOLVE_TX_API(free, txFree);
  WAFER_RESOLVE_TX_API(memcpy, txMemcpy);
  WAFER_RESOLVE_TX_API(moduleLoad, txModuleLoad);
  WAFER_RESOLVE_TX_API(moduleUnload, txModuleUnload);
  WAFER_RESOLVE_TX_API(moduleGetFunction, txModuleGetFunction);
  WAFER_RESOLVE_TX_API(launchKernel, txLaunchKernel);
  WAFER_RESOLVE_TX_API(launchClusterKernel, txLaunchClusterKernel);
  WAFER_RESOLVE_TX_API(streamCreate, txStreamCreate);
  WAFER_RESOLVE_TX_API(streamDestroy, txStreamDestroy);
  WAFER_RESOLVE_TX_API(streamQuery, txStreamQuery);
  WAFER_RESOLVE_TX_API(eventCreate, txEventCreate);
  WAFER_RESOLVE_TX_API(eventDestroy, txEventDestroy);
  WAFER_RESOLVE_TX_API(eventRecord, txEventRecord);
  WAFER_RESOLVE_TX_API(eventQuery, txEventQuery);
  WAFER_RESOLVE_TX_API(eventElapsedTime, txEventElapsedTime);
#undef WAFER_RESOLVE_TX_API
  return std::make_unique<TxBoardRuntimeDriver>(library, api,
                                                std::move(*digest));
}

} // namespace wafer::runtime
