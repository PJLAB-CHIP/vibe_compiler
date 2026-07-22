//===- TxBoardRuntime.cpp - TX public-runtime board provider ------------===//

#include "Wafer/Runtime/TxBoardRuntime.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include "tx_runtime.h"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace wafer::runtime {
namespace {

struct TxApi {
  decltype(&txGetDeviceCount) getDeviceCount = nullptr;
  decltype(&txSetDevice) setDevice = nullptr;
  decltype(&txGetDeviceProperty) getDeviceProperty = nullptr;
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
  decltype(&txStreamSynchronize) streamSynchronize = nullptr;
};

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
        runtimeLibraryDigest(std::move(runtimeLibraryDigest)) {}

  // The board CLI is a one-shot process and exits with std::_Exit after the
  // explicit TX lifecycle. The handle intentionally remains process-owned so
  // neither success nor quarantine invokes unqualified provider finalizers.
  ~TxBoardRuntimeDriver() override = default;

  BoardRuntimeContextState getContextState() const override {
    return contextState;
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
    if (moduleBytes.empty() ||
        moduleBytes.size() > std::numeric_limits<uint32_t>::max())
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "TX module byte count must be nonzero and fit uint32_t");
    txModule_t module = nullptr;
    txError_t status = api.moduleLoad(
        &module, reinterpret_cast<const char *>(moduleBytes.data()),
        static_cast<uint32_t>(moduleBytes.size()));
    if (status != TX_SUCCESS)
      return txError("txModuleLoad", status);
    if (!module)
      return poisonContractViolation(
          "txModuleLoad returned success with a null module");
    return BoardModuleHandle{reinterpret_cast<uintptr_t>(module)};
  }

  llvm::Error unloadModule(BoardModuleHandle module) override {
    if (llvm::Error error = requireUsable("txModuleUnload"))
      return error;
    return check("txModuleUnload",
                 api.moduleUnload(reinterpret_cast<txModule_t>(module.value)));
  }

  llvm::Expected<BoardFunctionHandle>
  resolveEntry(BoardModuleHandle module, llvm::StringRef symbol) override {
    if (llvm::Error error = requireUsable("txModuleGetFunction"))
      return std::move(error);
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

  llvm::Error launch(BoardFunctionHandle function,
                     llvm::ArrayRef<uint64_t> arguments) override {
    if (llvm::Error error = requireUsable("txLaunchKernel"))
      return error;
    if (arguments.size() >
        std::numeric_limits<uint32_t>::max() / sizeof(uint64_t))
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "TX launch argument block is too large");
    dim3 gridDim = {1, 1, 1};
    dim3 blockDim = {1, 1, 1};
    return check("txLaunchKernel",
                 api.launchKernel(
                     reinterpret_cast<txFunction_t>(function.value), gridDim,
                     blockDim, const_cast<uint64_t *>(arguments.data()),
                     static_cast<uint32_t>(arguments.size() * sizeof(uint64_t)),
                     0, nullptr));
  }

  llvm::Error synchronize() override {
    if (llvm::Error error = requireUsable("txStreamSynchronize"))
      return error;
    return check("txStreamSynchronize", api.streamSynchronize(nullptr));
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
  void *library = nullptr;
  TxApi api;
  std::string runtimeLibraryDigest;
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
  WAFER_RESOLVE_TX_API(streamSynchronize, txStreamSynchronize);
#undef WAFER_RESOLVE_TX_API
  return std::make_unique<TxBoardRuntimeDriver>(library, api,
                                                std::move(*digest));
}

} // namespace wafer::runtime
