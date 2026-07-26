//===- TxBoardRuntime.cpp - TX public-runtime board provider ------------===//

#include "Wafer/Runtime/TxBoardRuntime.h"
#include "Wafer/Runtime/Tx81ModelABI.h"

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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <memory>
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
  decltype(&txLoadGraph) loadGraph = nullptr;
  decltype(&txUnloadGraph) unloadGraph = nullptr;
  decltype(&txLaunchModel) launchModel = nullptr;
  decltype(&txStreamCreate) streamCreate = nullptr;
  decltype(&txStreamDestroy) streamDestroy = nullptr;
  decltype(&txStreamQuery) streamQuery = nullptr;
};

RuntimeEnvironment makeTxProviderEnvironment(TargetLaunchABIId launchABI) {
  const TargetProfileRecord &profile =
      getTargetProfileRecord(TargetProfileId::waferTx81SingleCardKernelV1());
  RuntimeEnvironment environment(profile.id, profile.targetIdentity,
                                 profile.kernelRuntimeABI,
                                 launchABI,
                                 profile.moduleFormat);
  environment.supportsHostWatchdog = true;
  if (launchABI ==
      TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1()) {
    environment.supportsDirectDTE = true;
    environment.directDTEStatusABI = kDirectDTEStatusABI.str();
  }
  return environment;
}

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

llvm::Error writeGraphModuleFile(llvm::StringRef path,
                                 llvm::ArrayRef<uint8_t> bytes) {
  std::string storage = path.str();
  int descriptor = open(storage.c_str(),
                        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                        0600);
  if (descriptor < 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to create invocation-private TX graph module");
  ScopedFD ownedDescriptor(descriptor);
  size_t written = 0;
  while (written < bytes.size()) {
    ssize_t count =
        ::write(descriptor, bytes.data() + written, bytes.size() - written);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return llvm::createStringError(
          count < 0 ? std::error_code(errno, std::generic_category())
                    : llvm::make_error_code(llvm::errc::io_error),
          "failed to write invocation-private TX graph module");
    written += static_cast<size_t>(count);
  }
  if (::fsync(descriptor) != 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to fsync invocation-private TX graph module");
  return llvm::Error::success();
}

struct StagedGraphDirectory {
  std::string root;
  std::string providerPath;
  int descriptor = -1;
};

llvm::Expected<StagedGraphDirectory> stageGraphModules(
    llvm::ArrayRef<BoardGraphModuleSnapshot> modules) {
  if (modules.size() != 16)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "TX model graph requires exactly 16 tile module snapshots");
  std::string pattern = "/tmp/wafer-tx-graph-XXXXXX";
  if (!::mkdtemp(pattern.data()))
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to create invocation-private TX graph directory");

  auto fail = [&](llvm::Error error) -> llvm::Expected<StagedGraphDirectory> {
    llvm::sys::fs::remove_directories(pattern);
    return std::move(error);
  };
  uint64_t aggregateModuleBytes = 0;
  for (auto [tile, module] : llvm::enumerate(modules)) {
    if (module.logicalTile != tile || !module.module.isValid() ||
        module.bytes.empty() ||
        module.bytes.size() > std::numeric_limits<uint32_t>::max() ||
        module.digest.empty() ||
        module.bytes.size() >
            std::numeric_limits<uint64_t>::max() - aggregateModuleBytes)
      return fail(llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX graph module snapshots have an invalid tile domain or size"));
    aggregateModuleBytes += module.bytes.size();
    llvm::SHA256 hasher;
    hasher.update(llvm::StringRef(
        reinterpret_cast<const char *>(module.bytes.data()),
        module.bytes.size()));
    std::string digest =
        "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
    if (digest != module.digest)
      return fail(llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX graph module snapshot digest does not match its bytes"));
    llvm::SmallString<256> tileDirectory(pattern);
    llvm::sys::path::append(tileDirectory,
                            "tile" + std::to_string(tile));
    std::string tileStorage = tileDirectory.str().str();
    if (::mkdir(tileStorage.c_str(), 0700) != 0)
      return fail(llvm::createStringError(
          std::error_code(errno, std::generic_category()),
          "failed to create invocation-private TX tile directory"));
    llvm::SmallString<256> modulePath(tileDirectory);
    llvm::sys::path::append(modulePath, "kcore_fw.so");
    if (llvm::Error error = writeGraphModuleFile(modulePath, module.bytes))
      return fail(std::move(error));
  }

  // The vendor hashes the graph path string verbatim as the model-module
  // identity. A bare /proc/self/fd/N repeats across one-shot processes, so add
  // a private, invocation-unique no-op path component while retaining the
  // directory-fd pin and the same tileN lookup root.
  std::string identity = llvm::sys::path::filename(pattern).str();
  llvm::SmallString<256> identityDirectory(pattern);
  llvm::sys::path::append(identityDirectory, identity);
  std::string identityStorage = identityDirectory.str().str();
  if (::mkdir(identityStorage.c_str(), 0700) != 0)
    return fail(llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to create invocation-private TX graph identity directory"));

  int descriptor = open(pattern.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                             O_NOFOLLOW);
  if (descriptor < 0)
    return fail(llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to pin invocation-private TX graph directory"));
  return StagedGraphDirectory{
      pattern,
      "/proc/self/fd/" + std::to_string(descriptor) + "/" + identity + "/..",
      descriptor};
}

class TxBoardRuntimeDriver final : public BoardRuntimeDriver {
public:
  TxBoardRuntimeDriver(void *library, TxApi api,
                       std::string runtimeLibraryDigest,
                       TargetLaunchABIId launchABI)
      : library(library), api(api),
        runtimeLibraryDigest(std::move(runtimeLibraryDigest)),
        providerEnvironment(makeTxProviderEnvironment(launchABI)) {}

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
    for (const tileFullInfo &tile : tileInfo.tilesFullInfo)
      info.tiles.push_back(
          {tile.index, tile.isAvailable == 1, tile.phyTilex, tile.phyTiley});
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
    if (providerEnvironment.launchABI ==
            TargetLaunchABIId::tx81ModelBootParamV1() ||
        !api.moduleLoad)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX module loading is unavailable for the selected launch ABI");
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
    auto iterator = moduleOwnership.find(handle);
    if (iterator == moduleOwnership.end()) {
      moduleOwnership.emplace(handle, ModuleOwnership{std::move(digest), 1});
    } else {
      if (iterator->second.digest != digest)
        return poisonContractViolation(
            "txModuleLoad aliased different code objects to one module handle");
      if (iterator->second.logicalReferences ==
          std::numeric_limits<uint64_t>::max())
        return poisonContractViolation(
            "TX logical module ownership count overflowed");
      ++iterator->second.logicalReferences;
    }
    return BoardModuleHandle{handle};
  }

  llvm::Error unloadModule(BoardModuleHandle module) override {
    if (llvm::Error error = requireUsable("txModuleUnload"))
      return error;
    if (providerEnvironment.launchABI ==
            TargetLaunchABIId::tx81ModelBootParamV1() ||
        !api.moduleUnload)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX module unloading is unavailable for the selected launch ABI");
    auto iterator = moduleOwnership.find(module.value);
    if (iterator == moduleOwnership.end())
      return poisonContractViolation(
          "logical TX module ownership is missing during unload");
    if (iterator->second.logicalReferences > 1) {
      --iterator->second.logicalReferences;
      return llvm::Error::success();
    }
    if (llvm::Error error =
            check("txModuleUnload",
                  api.moduleUnload(reinterpret_cast<txModule_t>(module.value))))
      return error;
    moduleOwnership.erase(iterator);
    return llvm::Error::success();
  }

  llvm::Expected<BoardFunctionHandle>
  resolveEntry(BoardModuleHandle module, llvm::StringRef symbol) override {
    if (llvm::Error error = requireUsable("txModuleGetFunction"))
      return std::move(error);
    if (providerEnvironment.launchABI ==
            TargetLaunchABIId::tx81ModelBootParamV1() ||
        !api.moduleGetFunction || symbol.empty() || symbol.contains('\0'))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX module entry resolution has an invalid launch ABI or symbol");
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

  llvm::Expected<BoardGraphHandle>
  loadGraph(llvm::ArrayRef<BoardGraphModuleSnapshot> modules,
            llvm::StringRef symbol) override {
    if (llvm::Error error = requireUsable("txLoadGraph"))
      return std::move(error);
    if (providerEnvironment.launchABI !=
            TargetLaunchABIId::tx81ModelBootParamV1() ||
        !api.loadGraph)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX graph loading requires the model BootParam launch ABI");
    if (!graphs.empty() || symbol.empty() || symbol.size() >= 128 ||
        symbol.contains('\0'))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX provider already owns a graph or the graph symbol does not fit "
          "the qualified 128-byte loader field");
    llvm::Expected<StagedGraphDirectory> staged = stageGraphModules(modules);
    if (!staged)
      return staged.takeError();

    std::string ownedSymbol = symbol.str();
    txError_t status =
        api.loadGraph(staged->providerPath.c_str(), ownedSymbol.c_str());
    if (status != TX_SUCCESS) {
      // A type-6 failure has an unknown accepted subset. Preserve the pinned
      // directory and issue no unload/reset/power operation in quarantine.
      return txError("txLoadGraph(type-6)", status);
    }
    uintptr_t handle = nextGraphHandle++;
    if (handle == 0)
      return poisonContractViolation("TX graph handle identity overflowed");
    std::string moduleName = std::to_string(
        std::hash<std::string>{}(staged->providerPath));
    graphs.emplace(handle,
                   GraphOwnership{std::move(staged->root),
                                  std::move(staged->providerPath),
                                  std::move(moduleName), staged->descriptor});
    return BoardGraphHandle{handle};
  }

  llvm::Error unloadGraph(BoardGraphHandle graph) override {
    if (llvm::Error error = requireUsable("txUnloadGraph"))
      return error;
    if (providerEnvironment.launchABI !=
            TargetLaunchABIId::tx81ModelBootParamV1() ||
        !api.unloadGraph)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX graph unloading is unavailable for the selected launch ABI");
    if (submissionActive)
      return poisonContractViolation(
          "TX graph unload was requested with a live submission");
    auto iterator = graphs.find(graph.value);
    if (iterator == graphs.end())
      return poisonContractViolation("TX graph ownership is missing");
    txError_t status = api.unloadGraph(iterator->second.providerPath.c_str());
    if (status != TX_SUCCESS)
      return txError("txUnloadGraph(type-8)", status);

    std::string stagingRoot = std::move(iterator->second.stagingRoot);
    int descriptor = iterator->second.directoryDescriptor;
    graphs.erase(iterator);
    if (descriptor >= 0 && ::close(descriptor) != 0)
      return llvm::createStringError(
          std::error_code(errno, std::generic_category()),
          "failed to close invocation-private TX graph directory");
    if (std::error_code error = llvm::sys::fs::remove_directories(stagingRoot))
      return llvm::createStringError(
          error, "failed to remove invocation-private TX graph directory");
    return llvm::Error::success();
  }

  llvm::Error submitAll(llvm::ArrayRef<BoardRankLaunch> launches) override {
    if (llvm::Error error = requireUsable("all-rank submission"))
      return error;
    if (providerEnvironment.launchABI !=
            TargetLaunchABIId::perRankPointerBlockV1() ||
        !api.launchKernel)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX per-rank submission is unavailable for the selected launch ABI");
    if (submissionActive || !activeStreams.empty() ||
        !submissionArgumentBlocks.empty())
      return poisonContractViolation(
          "TX provider already owns a live all-rank submission");
    if (launches.empty())
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "all-rank submission is empty");

    for (const BoardRankLaunch &launch : launches) {
      if (launch.logicalRank < 0 || !launch.entry.isValid() ||
          launch.function.value == 0)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "all-rank submission contains invalid rank/function identity");
      if (launch.arguments.size() >
          kTx81KernelArgumentBytesMax / sizeof(uint64_t))
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "TX launch argument block exceeds the "
                                       "qualified V5.6 packet limit");
    }

    submissionArgumentBlocks.reserve(launches.size());
    for (const BoardRankLaunch &launch : launches)
      submissionArgumentBlocks.emplace_back(launch.arguments.begin(),
                                            launch.arguments.end());

    activeStreams.reserve(launches.size());
    completedStreams.assign(launches.size(), false);
    for (size_t index = 0; index < launches.size(); ++index) {
      txStream_t stream = nullptr;
      txError_t status = api.streamCreate(&stream);
      if (status != TX_SUCCESS)
        return txError("txStreamCreate", status);
      if (!stream)
        return poisonContractViolation(
            "txStreamCreate returned success with a null stream");
      activeStreams.push_back(stream);
    }

    dim3 gridDim = {1, 1, 1};
    dim3 blockDim = {1, 1, 1};
    for (auto [index, launch] : llvm::enumerate(launches)) {
      txError_t status = api.launchKernel(
          reinterpret_cast<txFunction_t>(launch.function.value), gridDim,
          blockDim, submissionArgumentBlocks[index].data(),
          static_cast<uint32_t>(submissionArgumentBlocks[index].size() *
                                sizeof(uint64_t)),
          0, activeStreams[index]);
      if (status != TX_SUCCESS)
        return txError("txLaunchKernel(all-rank)", status);
    }
    submissionActive = true;
    return llvm::Error::success();
  }

  llvm::Error
  submitKernelGrid(llvm::ArrayRef<BoardRankLaunch> launches) override {
    if (llvm::Error error = requireUsable("kernel-grid submission"))
      return error;
    if (providerEnvironment.launchABI !=
            TargetLaunchABIId::tx81KernelGridPointerTableV1() ||
        !api.launchKernel)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX kernel-grid submission requires its explicit launch ABI");
    if (submissionActive || !activeStreams.empty() ||
        !submissionArgumentBlocks.empty() || launches.size() != 16)
      return poisonContractViolation(
          "TX kernel-grid provider has invalid submission ownership");

    const uintptr_t function = launches.front().function.value;
    const size_t slotsPerRank = launches.front().arguments.size();
    if (function == 0 || slotsPerRank == 0)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "TX kernel-grid launch is empty");
    if (slotsPerRank > kTx81KernelArgumentBytesMax / sizeof(uint64_t) /
                           launches.size())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX kernel-grid rank-major argument table exceeds the qualified "
          "V5.6 packet limit");

    std::vector<uint64_t> arguments;
    arguments.reserve(slotsPerRank * launches.size());
    for (auto [rank, launch] : llvm::enumerate(launches)) {
      if (launch.logicalRank != static_cast<int64_t>(rank) ||
          !launch.entry.isValid() || launch.function.value != function ||
          launch.arguments.size() != slotsPerRank)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "TX kernel-grid launches are not a canonical shared-function "
            "rank domain");
      arguments.insert(arguments.end(), launch.arguments.begin(),
                       launch.arguments.end());
    }

    txStream_t stream = nullptr;
    txError_t status = api.streamCreate(&stream);
    if (status != TX_SUCCESS)
      return txError("txStreamCreate(kernel-grid)", status);
    if (!stream)
      return poisonContractViolation(
          "txStreamCreate(kernel-grid) returned a null stream");
    activeStreams.push_back(stream);
    completedStreams.assign(1, false);
    // V5.6 queues the caller's host pointer and copies the argument bytes only
    // when the asynchronous command is submitted. Keep this rank-major table
    // alive through terminal completion and stream destruction.
    submissionArgumentBlocks.push_back(std::move(arguments));

    dim3 gridDim = {16, 1, 1};
    dim3 blockDim = {1, 1, 1};
    status = api.launchKernel(
        reinterpret_cast<txFunction_t>(function), gridDim, blockDim,
        submissionArgumentBlocks.front().data(),
        static_cast<uint32_t>(submissionArgumentBlocks.front().size() *
                              sizeof(uint64_t)),
        0, stream);
    if (status != TX_SUCCESS)
      return txError("txLaunchKernel(grid.x=16)", status);
    submissionActive = true;
    return llvm::Error::success();
  }

  llvm::Error submitClusterPrepareMain(
      BoardFunctionHandle prepare,
      llvm::ArrayRef<BoardRankLaunch> mainLaunches) override {
    if (llvm::Error error = requireUsable("cluster prepare submission"))
      return error;
    if (providerEnvironment.launchABI !=
            TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1() ||
        !api.launchClusterKernel)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX cluster submission requires its explicit Direct-DTE launch "
          "ABI");
    if (submissionActive || !activeStreams.empty() ||
        !submissionArgumentBlocks.empty() || clusterMainFunction != 0 ||
        mainLaunches.size() != 16)
      return poisonContractViolation(
          "TX cluster provider has invalid submission ownership");

    const uintptr_t mainFunction = mainLaunches.front().function.value;
    const size_t slotsPerRank = mainLaunches.front().arguments.size();
    if (prepare.value == 0 || mainFunction == 0 || slotsPerRank == 0)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "TX cluster launch is empty");
    if (slotsPerRank > kTx81ClusterKernelArgumentBytesMax / sizeof(uint64_t) /
                           mainLaunches.size())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX cluster rank-major argument table exceeds the qualified V5.6 "
          "C-INS packet limit");

    std::vector<uint64_t> arguments;
    arguments.reserve(slotsPerRank * mainLaunches.size());
    for (auto [rank, launch] : llvm::enumerate(mainLaunches)) {
      if (launch.logicalRank != static_cast<int64_t>(rank) ||
          !launch.entry.isValid() || launch.function.value != mainFunction ||
          launch.arguments.size() != slotsPerRank)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "TX cluster launches are not a canonical shared-function rank "
            "domain");
      arguments.insert(arguments.end(), launch.arguments.begin(),
                       launch.arguments.end());
    }

    txStream_t stream = nullptr;
    txError_t status = api.streamCreate(&stream);
    if (status != TX_SUCCESS)
      return txError("txStreamCreate(cluster)", status);
    if (!stream)
      return poisonContractViolation(
          "txStreamCreate(cluster) returned a null stream");
    activeStreams.push_back(stream);
    completedStreams.assign(1, false);
    submissionArgumentBlocks.push_back(std::move(arguments));
    clusterMainFunction = mainFunction;

    dim3 clusterDim = {1, 1, 1};
    dim3 gridDim = {16, 1, 1};
    dim3 blockDim = {1, 1, 1};
    status = api.launchClusterKernel(
        reinterpret_cast<txFunction_t>(prepare.value), clusterDim, gridDim,
        blockDim, submissionArgumentBlocks.front().data(),
        static_cast<uint32_t>(submissionArgumentBlocks.front().size() *
                              sizeof(uint64_t)),
        0, stream);
    if (status != TX_SUCCESS)
      return txError("txLaunchClusterKernel(prepare)", status);
    submissionActive = true;
    return llvm::Error::success();
  }

  llvm::Error
  submitModel(BoardGraphHandle graph,
              llvm::ArrayRef<BoardModelTensorLaunch> tensors) override {
    if (llvm::Error error = requireUsable("model submission"))
      return error;
    if (providerEnvironment.launchABI !=
            TargetLaunchABIId::tx81ModelBootParamV1() ||
        !api.launchModel)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX model submission requires its explicit launch ABI");
    if (submissionActive || !activeStreams.empty() ||
        !submissionArgumentBlocks.empty() || !submissionMetadata.empty())
      return poisonContractViolation(
          "TX model provider already owns submission state");
    auto graphIterator = graphs.find(graph.value);
    if (graphIterator == graphs.end())
      return poisonContractViolation("TX model graph ownership is missing");

    std::vector<Tx81ModelTensorDescriptor> descriptors;
    descriptors.reserve(tensors.size());
    for (const BoardModelTensorLaunch &tensor : tensors) {
      Tx81ModelTensorClass tensorClass;
      switch (tensor.role) {
      case PackageResourceRole::UserInput:
        tensorClass = Tx81ModelTensorClass::Input;
        break;
      case PackageResourceRole::Output:
        tensorClass = Tx81ModelTensorClass::Output;
        break;
      case PackageResourceRole::Parameter:
      case PackageResourceRole::Constant:
        tensorClass = Tx81ModelTensorClass::Parameter;
        break;
      case PackageResourceRole::Workspace:
      case PackageResourceRole::TransportStatus:
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "TX model launch contains an unsupported resource role");
      }
      descriptors.push_back({tensorClass, tensor.logicalRank,
                             tensor.slotOrdinal,
                             static_cast<uint64_t>(tensor.memory.value),
                             tensor.bytes, tensor.dtype, tensor.shape});
    }

    llvm::Expected<std::vector<uint8_t>> dynMods =
        buildTx81DynlibRunModules(graphIterator->second.moduleName);
    if (!dynMods)
      return dynMods.takeError();
    llvm::Expected<Tx81ModelBootParamImage> validation =
        buildTx81ModelBootParam(descriptors, /*dynamicTLVDeviceAddress=*/8);
    if (!validation)
      return validation.takeError();

    auto upload = [&](llvm::ArrayRef<uint8_t> bytes,
                      llvm::StringRef operation)
        -> llvm::Expected<uint64_t> {
      void *pointer = nullptr;
      txError_t status = api.malloc(&pointer, bytes.size());
      if (status != TX_SUCCESS)
        return txError((operation + " txMalloc").str(), status);
      if (!pointer || reinterpret_cast<uintptr_t>(pointer) %
                              alignof(uint64_t) !=
                          0)
        return poisonContractViolation(
            (operation + " txMalloc returned null or misaligned").str());
      status = api.memcpy(pointer, bytes.data(), bytes.size(),
                          txMemcpyHostToDevice);
      if (status != TX_SUCCESS)
        return txError((operation + " txMemcpy(H2D)").str(), status);
      submissionMetadata.push_back(pointer);
      return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pointer));
    };

    llvm::Expected<uint64_t> dynModsAddress = upload(*dynMods, "type-7 DynMods");
    if (!dynModsAddress)
      return dynModsAddress.takeError();
    llvm::Expected<std::vector<uint8_t>> tlv =
        buildTx81DynlibRunTLV(*dynModsAddress);
    if (!tlv)
      return tlv.takeError();
    llvm::Expected<uint64_t> tlvAddress = upload(*tlv, "type-7 TLV");
    if (!tlvAddress)
      return tlvAddress.takeError();
    llvm::Expected<Tx81ModelBootParamImage> bootParam =
        buildTx81ModelBootParam(descriptors, *tlvAddress);
    if (!bootParam)
      return bootParam.takeError();
    llvm::Expected<uint64_t> bootParamAddress =
        upload(bootParam->bytes, "model BootParam");
    if (!bootParamAddress)
      return bootParamAddress.takeError();

    txStream_t stream = nullptr;
    txError_t status = api.streamCreate(&stream);
    if (status != TX_SUCCESS)
      return txError("txStreamCreate(model)", status);
    if (!stream)
      return poisonContractViolation(
          "txStreamCreate(model) returned a null stream");
    activeStreams.push_back(stream);
    completedStreams.assign(1, false);
    status = api.launchModel(*bootParamAddress, stream);
    if (status != TX_SUCCESS)
      return txError("txLaunchModel(type-7)", status);
    submissionActive = true;
    return llvm::Error::success();
  }

  llvm::Expected<BoardCompletionObservation>
  waitAll(uint64_t timeoutMilliseconds,
          BoardCompletionObservationPolicy observationPolicy) override {
    if (llvm::Error error = requireUsable("all-rank completion"))
      return error;
    if (!submissionActive || activeStreams.empty() ||
        completedStreams.size() != activeStreams.size())
      return poisonContractViolation(
          "TX all-rank completion has no live submission");
    if (timeoutMilliseconds == 0)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "TX completion timeout must be positive");

    const auto waitBegin = std::chrono::steady_clock::now();
    const auto deadline =
        waitBegin + std::chrono::milliseconds(timeoutMilliseconds);
    std::vector<std::chrono::steady_clock::time_point> lastObservations(
        activeStreams.size(), waitBegin);
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
      llvm::StringRef submissionPhase = "all-rank";
      if (providerEnvironment.launchABI ==
          TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1())
        submissionPhase =
            clusterMainFunction != 0 ? "cluster prepare" : "cluster main";
      return llvm::createStringError(
          llvm::errc::io_error, "TX %s completion exceeded the host deadline",
          submissionPhase.str().c_str());
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
        if (clusterMainFunction != 0) {
          if (std::chrono::steady_clock::now() >= deadline)
            return deadlineExceeded();
          dim3 clusterDim = {1, 1, 1};
          dim3 gridDim = {16, 1, 1};
          dim3 blockDim = {1, 1, 1};
          txError_t status = api.launchClusterKernel(
              reinterpret_cast<txFunction_t>(clusterMainFunction), clusterDim,
              gridDim, blockDim, submissionArgumentBlocks.front().data(),
              static_cast<uint32_t>(submissionArgumentBlocks.front().size() *
                                    sizeof(uint64_t)),
              0, activeStreams.front());
          if (status != TX_SUCCESS)
            return txError("txLaunchClusterKernel(main)", status);
          clusterMainFunction = 0;
          completedStreams.assign(1, false);
          lastObservations.assign(activeStreams.size(),
                                  std::chrono::steady_clock::now());
          continue;
        }
        return BoardCompletionObservation{maximumPollGapNanoseconds};
      }
      if (observationPolicy == BoardCompletionObservationPolicy::Normal)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      else
        std::this_thread::yield();
    }
  }

  llvm::Error releaseSubmission() override {
    if (llvm::Error error = requireUsable("txStreamDestroy"))
      return error;
    if (!submissionActive ||
        !llvm::all_of(completedStreams, [](bool complete) { return complete; }))
      return poisonContractViolation(
          "TX submission release requires all ranks to be terminal");
    while (!activeStreams.empty()) {
      if (llvm::Error error =
              check("txStreamDestroy", api.streamDestroy(activeStreams.back())))
        return error;
      activeStreams.pop_back();
    }
    while (!submissionMetadata.empty()) {
      if (llvm::Error error =
              check("txFree(model-metadata)",
                    api.free(submissionMetadata.back())))
        return error;
      submissionMetadata.pop_back();
    }
    submissionArgumentBlocks.clear();
    completedStreams.clear();
    clusterMainFunction = 0;
    submissionActive = false;
    return llvm::Error::success();
  }

private:
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
  struct ModuleOwnership {
    std::string digest;
    uint64_t logicalReferences = 0;
  };
  struct GraphOwnership {
    std::string stagingRoot;
    std::string providerPath;
    std::string moduleName;
    int directoryDescriptor = -1;
  };
  void *library = nullptr;
  TxApi api;
  std::string runtimeLibraryDigest;
  RuntimeEnvironment providerEnvironment;
  std::unordered_map<uintptr_t, ModuleOwnership> moduleOwnership;
  std::unordered_map<uintptr_t, GraphOwnership> graphs;
  uintptr_t nextGraphHandle = 1;
  std::vector<txStream_t> activeStreams;
  std::vector<bool> completedStreams;
  std::vector<std::vector<uint64_t>> submissionArgumentBlocks;
  std::vector<void *> submissionMetadata;
  uintptr_t clusterMainFunction = 0;
  bool submissionActive = false;
};

} // namespace

llvm::Expected<std::unique_ptr<BoardRuntimeDriver>>
createTxBoardRuntimeDriver(llvm::StringRef expectedRuntimeLibraryDigest,
                           TargetLaunchABIId launchABI) {
  if (launchABI != TargetLaunchABIId::perRankPointerBlockV1() &&
      launchABI != TargetLaunchABIId::tx81KernelGridPointerTableV1() &&
      launchABI != TargetLaunchABIId::tx81ModelBootParamV1() &&
      launchABI !=
          TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "TX launch ABI is not registered");
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
  if (launchABI == TargetLaunchABIId::tx81ModelBootParamV1()) {
    WAFER_RESOLVE_TX_API(loadGraph, txLoadGraph);
    WAFER_RESOLVE_TX_API(unloadGraph, txUnloadGraph);
    WAFER_RESOLVE_TX_API(launchModel, txLaunchModel);
  } else {
    WAFER_RESOLVE_TX_API(moduleLoad, txModuleLoad);
    WAFER_RESOLVE_TX_API(moduleUnload, txModuleUnload);
    WAFER_RESOLVE_TX_API(moduleGetFunction, txModuleGetFunction);
    if (launchABI ==
        TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1())
      WAFER_RESOLVE_TX_API(launchClusterKernel, txLaunchClusterKernel);
    else
      WAFER_RESOLVE_TX_API(launchKernel, txLaunchKernel);
  }
  WAFER_RESOLVE_TX_API(streamCreate, txStreamCreate);
  WAFER_RESOLVE_TX_API(streamDestroy, txStreamDestroy);
  WAFER_RESOLVE_TX_API(streamQuery, txStreamQuery);
#undef WAFER_RESOLVE_TX_API
  return std::make_unique<TxBoardRuntimeDriver>(library, api,
                                                std::move(*digest), launchABI);
}

} // namespace wafer::runtime
