//===- BulkExecutionEnvironment.cpp - Managed oneDNN environment ----===//

#include "BulkTensorNumericInternal.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_version_hash.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <cfenv>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <gnu/libc-version.h>
#include <sched.h>
#include <sys/utsname.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifndef WAFER_BULK_DEPENDENCY_RECORD_SHA256
#error "managed bulk dependency record identity is required"
#endif
#ifndef WAFER_BULK_ONEDNN_LIBRARY_SHA256
#error "managed oneDNN library identity is required"
#endif
#ifndef WAFER_BULK_ONEDNN_VERSION
#error "managed oneDNN version is required"
#endif
#ifndef WAFER_BULK_ONEDNN_COMMIT
#error "managed oneDNN commit is required"
#endif

namespace wafer::bulk_detail {

uint32_t readMXCSR() {
#if defined(__x86_64__) || defined(_M_X64)
  return _mm_getcsr();
#else
  return 0;
#endif
}

} // namespace wafer::bulk_detail

namespace wafer {

using namespace bulk_detail;

namespace {

std::string makeFeaturesDigest() {
  llvm::StringMap<bool, llvm::MallocAllocator> features =
      llvm::sys::getHostCPUFeatures();
  std::vector<std::pair<std::string, bool>> ordered;
  ordered.reserve(features.size());
  for (const auto &feature : features)
    ordered.emplace_back(feature.getKey().str(), feature.getValue());
  llvm::sort(ordered, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  llvm::SmallString<1024> payload;
  llvm::raw_svector_ostream stream(payload);
  for (const auto &[name, enabled] : ordered)
    stream << name << '=' << (enabled ? '1' : '0') << '\n';
  return sha256(payload);
}

std::string readHostText(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/true,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return "unavailable";
  return (*buffer)->getBuffer().trim().str();
}

std::string makeStableCPUFingerprint() {
  std::string cpuInfo = readHostText("/proc/cpuinfo");
  if (cpuInfo == "unavailable")
    return cpuInfo;
  llvm::SmallString<512> selected;
  llvm::raw_svector_ostream stream(selected);
  for (llvm::StringRef line : llvm::split(llvm::StringRef(cpuInfo), '\n')) {
    llvm::StringRef key = line.take_front(line.find(':')).trim();
    if (key == "vendor_id" || key == "cpu family" || key == "model" ||
        key == "stepping" || key == "microcode")
      stream << line.trim() << '\n';
    if (key == "microcode")
      break;
  }
  return selected.empty() ? "unavailable" : sha256(selected);
}

std::string makeLoaderDigest() {
  std::string maps = readHostText("/proc/self/maps");
  if (maps == "unavailable")
    return maps;
  for (llvm::StringRef line : llvm::split(llvm::StringRef(maps), '\n')) {
    size_t separator = line.rfind(' ');
    if (separator == llvm::StringRef::npos)
      continue;
    llvm::StringRef path = line.drop_front(separator + 1).trim();
    if (!path.contains("ld-linux") && !path.contains("/ld-"))
      continue;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false);
    if (!buffer)
      return "unavailable";
    return sha256((*buffer)->getBuffer());
  }
  return "unavailable";
}

std::string makeHostPlatformDigest() {
  llvm::SmallString<2048> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "schema", "wafer-host-platform");
  appendField(stream, "process_triple", llvm::sys::getProcessTriple());
  appendField(stream, "cpu_fingerprint", makeStableCPUFingerprint());
  appendField(stream, "cpu_online",
              readHostText("/sys/devices/system/cpu/online"));
  appendField(stream, "numa_online",
              readHostText("/sys/devices/system/node/online"));
  appendField(stream, "loader", makeLoaderDigest());
#if defined(__linux__)
  struct utsname systemInfo{};
  if (::uname(&systemInfo) == 0) {
    appendField(stream, "kernel_sysname", systemInfo.sysname);
    appendField(stream, "kernel_release", systemInfo.release);
    appendField(stream, "kernel_version", systemInfo.version);
    appendField(stream, "kernel_machine", systemInfo.machine);
  } else {
    appendField(stream, "kernel", "unavailable");
  }
  appendField(stream, "libc_version", gnu_get_libc_version());
  appendField(stream, "libc_release", gnu_get_libc_release());
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (::sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
    appendField(
        stream, "affinity",
        sha256(llvm::ArrayRef<uint8_t>(
            reinterpret_cast<const uint8_t *>(&affinity), sizeof(affinity))));
  else
    appendField(stream, "affinity", "unavailable");
#else
  appendField(stream, "kernel", "unavailable");
  appendField(stream, "libc", "unavailable");
  appendField(stream, "affinity", "unavailable");
#endif
  return sha256(payload);
}

uint32_t readMXCSRControl() {
  // Sticky exception status bits 0..5 are invocation state, not an execution
  // policy. All control bits, including masks/RC/FTZ/DAZ, are identity.
  return readMXCSR() & ~UINT32_C(0x3f);
}

} // namespace

llvm::Expected<BulkExecutionEnvironment>
createManagedBulkExecutionEnvironment() {
  static std::mutex mutex;
  static std::optional<BulkExecutionEnvironment> cached;
  static std::optional<std::string> failure;
  std::lock_guard<std::mutex> lock(mutex);
  if (cached) {
    if (std::fegetround() != cached->getFloatingRoundingMode() ||
        readMXCSRControl() != cached->getMXCSR())
      return bulkError(BulkTensorNumericErrorCode::EnvironmentMismatch,
                       "current caller fenv/MXCSR control no longer matches "
                       "the managed sequential environment");
    if (llvm::sys::getHostCPUName() != cached->getHostCPUName() ||
        makeFeaturesDigest() != cached->getHostFeaturesDigest() ||
        makeHostPlatformDigest() != cached->getHostPlatformDigest())
      return bulkError(BulkTensorNumericErrorCode::EnvironmentMismatch,
                       "current host CPU/platform/affinity no longer matches "
                       "the managed sequential environment");
    return *cached;
  }
  if (failure)
    return bulkError(BulkTensorNumericErrorCode::BackendConfigurationFailure,
                     *failure);
  auto fail = [&](const llvm::Twine &detail)
      -> llvm::Expected<BulkExecutionEnvironment> {
    failure = detail.str();
    return bulkError(BulkTensorNumericErrorCode::BackendConfigurationFailure,
                     *failure);
  };
  if (std::fegetround() != FE_TONEAREST)
    return fail("initial sequential profile requires FE_TONEAREST");
#if !defined(__x86_64__) && !defined(_M_X64)
  return fail("initial sequential profile requires x86-64 MXCSR control");
#endif
  const uint32_t mxcsr = readMXCSRControl();
  if ((mxcsr & UINT32_C(0x8040)) != 0 || (mxcsr & UINT32_C(0x6000)) != 0 ||
      (mxcsr & UINT32_C(0x1f80)) != UINT32_C(0x1f80))
    return fail("initial sequential profile requires MXCSR RNE, FTZ=0, "
                "DAZ=0 and all exceptions masked");
  if (dnnl::set_max_cpu_isa(dnnl::cpu_isa::isa_default) !=
      dnnl::status::success)
    return fail(
        "oneDNN max CPU ISA policy was already initialized or rejected");
  if (dnnl::set_cpu_isa_hints(dnnl::cpu_isa_hints::no_hints) !=
      dnnl::status::success)
    return fail(
        "oneDNN CPU ISA hints policy was already initialized or rejected");
  try {
    dnnl::set_primitive_cache_capacity(0);
  } catch (const dnnl::error &error) {
    return fail(llvm::Twine("oneDNN primitive cache policy failed: ") +
                error.what());
  }
  const dnnl::version_t *runtimeVersion = dnnl::version();
  if (!runtimeVersion || runtimeVersion->major != DNNL_VERSION_MAJOR ||
      runtimeVersion->minor != DNNL_VERSION_MINOR ||
      runtimeVersion->patch != DNNL_VERSION_PATCH || !runtimeVersion->hash ||
      llvm::StringRef(runtimeVersion->hash) != DNNL_VERSION_HASH ||
      runtimeVersion->cpu_runtime != DNNL_RUNTIME_SEQ)
    return fail("loaded oneDNN version/commit/runtime identity mismatch");
  dnnl::cpu_isa effectiveISA = dnnl::get_effective_cpu_isa();
  const char *isaName =
      dnnl_cpu_isa2str(static_cast<dnnl_cpu_isa_t>(effectiveISA));
  if (!isaName || !*isaName)
    return fail("oneDNN returned an empty effective CPU ISA");

  llvm::SmallString<512> backendPayload;
  llvm::raw_svector_ostream backendStream(backendPayload);
  appendField(backendStream, "schema", "wafer-bulk-backend");
  appendField(backendStream, "name", "oneDNN");
  appendField(backendStream, "version", WAFER_BULK_ONEDNN_VERSION);
  appendField(backendStream, "commit", WAFER_BULK_ONEDNN_COMMIT);
  const std::string dependencyRecordDigest =
      (llvm::Twine("sha256:") + WAFER_BULK_DEPENDENCY_RECORD_SHA256).str();
  const std::string libraryDigest =
      (llvm::Twine("sha256:") + WAFER_BULK_ONEDNN_LIBRARY_SHA256).str();
  appendField(backendStream, "dependency_record", dependencyRecordDigest);
  appendField(backendStream, "library", libraryDigest);
  BulkBackendDescriptor backend(
      "oneDNN", WAFER_BULK_ONEDNN_VERSION, WAFER_BULK_ONEDNN_COMMIT,
      dependencyRecordDigest, libraryDigest, sha256(backendPayload));

  std::string cpuName = llvm::sys::getHostCPUName().str();
  std::string featuresDigest = makeFeaturesDigest();
  std::string hostPlatformDigest = makeHostPlatformDigest();
  llvm::SmallString<1024> environmentPayload;
  llvm::raw_svector_ostream environmentStream(environmentPayload);
  appendField(environmentStream, "schema", "wafer-bulk-environment");
  appendField(environmentStream, "backend", backend.getDigest());
  appendField(environmentStream, "host_cpu", cpuName);
  appendField(environmentStream, "host_features", featuresDigest);
  appendField(environmentStream, "host_platform", hostPlatformDigest);
  appendField(environmentStream, "effective_isa", isaName);
  appendField(environmentStream, "fenv_round", std::fegetround());
  appendField(environmentStream, "mxcsr", mxcsr);
  appendField(environmentStream, "thread_runtime", "sequential-caller-worker");
  cached = BulkExecutionEnvironment(
      std::move(backend), std::move(cpuName), std::move(featuresDigest),
      std::move(hostPlatformDigest), isaName, std::fegetround(), mxcsr,
      "sequential-caller-worker", sha256(environmentPayload));
  return *cached;
}

} // namespace wafer
