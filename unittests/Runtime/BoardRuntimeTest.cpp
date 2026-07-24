//===- BoardRuntimeTest.cpp - Board provider lifecycle tests ------------===//

#include "Wafer/Runtime/BoardRuntime.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

llvm::Error injected(llvm::StringRef operation) {
  return llvm::createStringError(llvm::errc::io_error, "injected %s failure",
                                 operation.str().c_str());
}

class FakeBoardDriver final : public wafer::runtime::BoardRuntimeDriver {
public:
  explicit FakeBoardDriver(
      wafer::TargetLaunchABIId launchABI =
          wafer::TargetLaunchABIId::perRankPointerBlockV1())
      : providerEnvironment(makeProviderEnvironment(launchABI)) {}

  wafer::runtime::BoardRuntimeContextState getContextState() const override {
    return contextState;
  }

  void quarantine() override {
    calls.push_back("quarantine");
    contextState = wafer::runtime::BoardRuntimeContextState::Poisoned;
  }

  const wafer::runtime::RuntimeEnvironment &
  getProviderEnvironment() const override {
    return providerEnvironment;
  }

  llvm::Expected<uint32_t> getDeviceCount() override {
    calls.push_back("get-device-count");
    if (failOperation == "get-device-count")
      return injectedFailure();
    return 1;
  }

  llvm::Error selectDevice(uint32_t deviceId) override {
    calls.push_back("select-device");
    selectedDevice = deviceId;
    if (failOperation == "select-device")
      return injectedFailure();
    return llvm::Error::success();
  }

  llvm::Expected<wafer::runtime::BoardDeviceInfo>
  getDeviceInfo(uint32_t deviceId) override {
    calls.push_back("device-info");
    if (failOperation == "device-info")
      return injectedFailure();
    wafer::runtime::BoardDeviceInfo info;
    info.deviceId = deviceId;
    info.runtimeVersion = 0x514;
    info.freeMemoryBytes = freeMemoryBytes;
    info.totalMemoryBytes = totalMemoryBytes;
    info.tileCount = 16;
    info.name = "/dev/accel/dev-0";
    info.pciBusId = "0000:00:00.0";
    info.runtimeLibraryDigest = runtimeLibraryDigest;
    for (uint16_t logicalIndex = 0; logicalIndex < 16; ++logicalIndex)
      info.tiles.push_back(
          {logicalIndex,
           static_cast<int64_t>(logicalIndex) != unavailableLogicalTile,
           static_cast<uint32_t>(logicalIndex % 8),
           static_cast<uint32_t>(logicalIndex / 8)});
    return info;
  }

  llvm::Expected<wafer::runtime::BoardDeviceMemory>
  allocate(uint64_t bytes, uint64_t alignment) override {
    calls.push_back("allocate");
    if (shouldFail("allocate"))
      return injectedFailure();
    nextAddress = (nextAddress + alignment - 1) & ~(alignment - 1);
    uintptr_t address = nextAddress;
    nextAddress += static_cast<uintptr_t>(bytes + alignment);
    allocations.push_back({address, std::vector<uint8_t>(bytes)});
    allocatedAddresses.push_back(address);
    return wafer::runtime::BoardDeviceMemory{address};
  }

  llvm::Error free(wafer::runtime::BoardDeviceMemory memory) override {
    calls.push_back("free:" + std::to_string(memory.value));
    if (shouldFail("free"))
      return injectedFailure();
    auto iterator = find(memory.value);
    if (iterator == allocations.end())
      return injected("unknown-free");
    freedAddresses.push_back(memory.value);
    allocations.erase(iterator);
    return llvm::Error::success();
  }

  llvm::Error copyHostToDevice(wafer::runtime::BoardDeviceMemory destination,
                               llvm::ArrayRef<uint8_t> source) override {
    calls.push_back("h2d");
    if (shouldFail("h2d"))
      return injectedFailure();
    auto iterator = find(destination.value);
    if (iterator == allocations.end() ||
        iterator->bytes.size() != source.size())
      return injected("invalid-h2d");
    h2dPayloads.emplace_back(source.begin(), source.end());
    std::copy(source.begin(), source.end(), iterator->bytes.begin());
    return llvm::Error::success();
  }

  llvm::Error
  copyDeviceToHost(llvm::MutableArrayRef<uint8_t> destination,
                   wafer::runtime::BoardDeviceMemory source) override {
    calls.push_back("d2h");
    if (shouldFail("d2h"))
      return injectedFailure();
    auto iterator = find(source.value);
    if (iterator == allocations.end() ||
        iterator->bytes.size() != destination.size())
      return injected("invalid-d2h");
    std::copy(iterator->bytes.begin(), iterator->bytes.end(),
              destination.begin());
    return llvm::Error::success();
  }

  llvm::Expected<wafer::runtime::BoardModuleHandle>
  loadModule(llvm::ArrayRef<uint8_t> moduleBytes) override {
    calls.push_back("load-module");
    if (shouldFail("load-module"))
      return injectedFailure();
    if (moduleBytes.empty())
      return injected("empty-module");
    uintptr_t handle = 0x7000 + loadedHandles.size();
    loadedHandles.push_back(handle);
    liveModules.push_back(handle);
    return wafer::runtime::BoardModuleHandle{handle};
  }

  llvm::Error unloadModule(wafer::runtime::BoardModuleHandle module) override {
    calls.push_back("unload-module:" + std::to_string(module.value));
    if (shouldFail("unload-module"))
      return injectedFailure();
    auto iterator =
        std::find(liveModules.begin(), liveModules.end(), module.value);
    if (iterator == liveModules.end())
      return injected("invalid-module");
    unloadedHandles.push_back(module.value);
    liveModules.erase(iterator);
    return llvm::Error::success();
  }

  llvm::Expected<wafer::runtime::BoardFunctionHandle>
  resolveEntry(wafer::runtime::BoardModuleHandle module,
               llvm::StringRef symbol) override {
    calls.push_back("resolve-entry");
    if (shouldFail("resolve-entry"))
      return injectedFailure();
    if (std::find(liveModules.begin(), liveModules.end(), module.value) ==
            liveModules.end() ||
        (symbol != "main" && symbol != "prepare"))
      return injected("invalid-entry");
    return wafer::runtime::BoardFunctionHandle{
        module.value + (symbol == "prepare" ? 0x1000 : 0x2000)};
  }

  llvm::Expected<wafer::runtime::BoardGraphHandle> loadGraph(
      llvm::ArrayRef<wafer::runtime::BoardGraphModuleSnapshot> modules,
      llvm::StringRef symbol) override {
    calls.push_back("load-graph");
    if (shouldFail("load-graph"))
      return injectedFailure();
    if (graphLive || modules.size() != 16 || symbol != "main")
      return injected("invalid-graph-load");
    for (auto [tile, module] : llvm::enumerate(modules))
      if (module.logicalTile != tile || module.bytes.empty() ||
          module.digest.empty())
        return injected("invalid-graph-module-domain");
    graphLive = true;
    return wafer::runtime::BoardGraphHandle{0xa000};
  }

  llvm::Error unloadGraph(wafer::runtime::BoardGraphHandle graph) override {
    calls.push_back("unload-graph");
    if (shouldFail("unload-graph"))
      return injectedFailure();
    if (!graphLive || graph.value != 0xa000)
      return injected("invalid-graph-unload");
    graphLive = false;
    return llvm::Error::success();
  }

  llvm::Error
  submitAll(llvm::ArrayRef<wafer::runtime::BoardRankLaunch> launches) override {
    calls.push_back("submit-all");
    if (shouldFail("submit-all"))
      return injectedFailure();
    if (submissionLive || launches.empty())
      return injected("invalid-submit-state");
    submittedLaunches.assign(launches.begin(), launches.end());
    for (auto [index, launch] : llvm::enumerate(launches)) {
      if (launch.logicalRank != static_cast<int64_t>(index) ||
          launch.function.value == 0 ||
          launch.arguments.size() != 3)
        return injected("invalid-canonical-submit");
      auto input = find(launch.arguments[0]);
      auto output = find(launch.arguments[1]);
      auto workspace = find(launch.arguments[2]);
      if (input == allocations.end() || output == allocations.end() ||
          workspace == allocations.end() ||
          input->bytes.size() != output->bytes.size())
        return injected("invalid-submit-buffers");
      for (size_t byte = 0; byte < input->bytes.size(); ++byte)
        output->bytes[byte] =
            input->bytes[byte] ^ static_cast<uint8_t>(launch.logicalRank);
    }
    submissionLive = true;
    return llvm::Error::success();
  }

  llvm::Error submitKernelGrid(
      llvm::ArrayRef<wafer::runtime::BoardRankLaunch> launches) override {
    calls.push_back("submit-kernel-grid");
    if (shouldFail("submit-kernel-grid"))
      return injectedFailure();
    if (submissionLive || launches.size() != 16)
      return injected("invalid-kernel-grid-state");
    submittedLaunches.assign(launches.begin(), launches.end());
    for (auto [rank, launch] : llvm::enumerate(launches)) {
      if (launch.logicalRank != static_cast<int64_t>(rank) ||
          launch.function.value != 0x9000 || launch.arguments.size() < 2)
        return injected("invalid-kernel-grid-launch");
      auto input = find(launch.arguments[0]);
      auto output = find(launch.arguments[1]);
      if (input == allocations.end() || output == allocations.end() ||
          input->bytes.size() != output->bytes.size())
        return injected("invalid-kernel-grid-buffers");
      for (size_t byte = 0; byte < input->bytes.size(); ++byte)
        output->bytes[byte] =
            input->bytes[byte] ^ static_cast<uint8_t>(rank);
    }
    submissionLive = true;
    return llvm::Error::success();
  }

  llvm::Error submitClusterPrepareMain(
      wafer::runtime::BoardFunctionHandle prepare,
      llvm::ArrayRef<wafer::runtime::BoardRankLaunch> launches) override {
    calls.push_back("submit-cluster-prepare-main");
    if (shouldFail("submit-cluster-prepare-main"))
      return injectedFailure();
    if (submissionLive || launches.size() != 16 || prepare.value != 0x8000)
      return injected("invalid-cluster-state");
    submittedLaunches.assign(launches.begin(), launches.end());
    for (auto [rank, launch] : llvm::enumerate(launches)) {
      if (launch.logicalRank != static_cast<int64_t>(rank) ||
          launch.function.value != 0x9000 || launch.arguments.size() < 3)
        return injected("invalid-cluster-launch");
      auto input = find(launch.arguments[0]);
      auto output = find(launch.arguments[1]);
      auto status = find(launch.arguments.back());
      if (input == allocations.end() || output == allocations.end() ||
          status == allocations.end() ||
          input->bytes.size() != output->bytes.size() ||
          status->bytes.size() !=
              wafer::runtime::kDirectDTEStatusStorageBytes)
        return injected("invalid-cluster-buffers");
      for (size_t byte = 0; byte < input->bytes.size(); ++byte)
        output->bytes[byte] =
            input->bytes[byte] ^ static_cast<uint8_t>(rank);
      const uint32_t terminalStatus =
          clusterStatusOverride.value_or(static_cast<uint32_t>(
              wafer::runtime::DirectDTEStatusValue::Success));
      std::memcpy(status->bytes.data() +
                      wafer::runtime::kDirectDTEStatusValueOffset,
                  &terminalStatus,
                  sizeof(terminalStatus));
    }
    submissionLive = true;
    return llvm::Error::success();
  }

  llvm::Error submitModel(
      wafer::runtime::BoardGraphHandle graph,
      llvm::ArrayRef<wafer::runtime::BoardModelTensorLaunch> tensors) override {
    calls.push_back("submit-model");
    if (shouldFail("submit-model"))
      return injectedFailure();
    if (submissionLive || !graphLive || graph.value != 0xa000 ||
        tensors.empty())
      return injected("invalid-model-submit-state");
    submittedModelTensors.assign(tensors.begin(), tensors.end());
    for (int64_t rank = 0; rank < 16; ++rank) {
      auto input = llvm::find_if(tensors, [&](const auto &tensor) {
        return tensor.logicalRank == rank &&
               tensor.role ==
                   wafer::runtime::PackageResourceRole::UserInput;
      });
      auto output = llvm::find_if(tensors, [&](const auto &tensor) {
        return tensor.logicalRank == rank &&
               tensor.role == wafer::runtime::PackageResourceRole::Output;
      });
      if (input == tensors.end() || output == tensors.end())
        return injected("invalid-model-tensor-domain");
      auto inputAllocation = find(input->memory.value);
      auto outputAllocation = find(output->memory.value);
      if (inputAllocation == allocations.end() ||
          outputAllocation == allocations.end() ||
          inputAllocation->bytes.size() != outputAllocation->bytes.size())
        return injected("invalid-model-buffers");
      for (size_t byte = 0; byte < inputAllocation->bytes.size(); ++byte)
        outputAllocation->bytes[byte] =
            inputAllocation->bytes[byte] ^ static_cast<uint8_t>(rank);
    }
    submissionLive = true;
    return llvm::Error::success();
  }

  llvm::Error waitAll(uint64_t timeoutMilliseconds) override {
    calls.push_back("wait-all");
    observedTimeoutMilliseconds = timeoutMilliseconds;
    if (shouldFail("wait-all"))
      return injectedFailure();
    if (!submissionLive)
      return injected("wait-without-submit");
    return llvm::Error::success();
  }

  llvm::Error releaseSubmission() override {
    calls.push_back("release-submission");
    if (shouldFail("release-submission")) {
      if (contextState == wafer::runtime::BoardRuntimeContextState::Usable)
        submissionLive = false;
      return injectedFailure();
    }
    if (!submissionLive)
      return injected("release-without-submit");
    submissionLive = false;
    return llvm::Error::success();
  }

  std::string failOperation;
  bool poisonOnFailure = false;
  size_t failIndex = 0;
  std::vector<std::string> calls;
  std::vector<uintptr_t> allocatedAddresses;
  std::vector<uintptr_t> freedAddresses;
  std::vector<uintptr_t> loadedHandles;
  std::vector<uintptr_t> unloadedHandles;
  std::vector<std::vector<uint8_t>> h2dPayloads;
  std::vector<wafer::runtime::BoardRankLaunch> submittedLaunches;
  std::vector<wafer::runtime::BoardModelTensorLaunch> submittedModelTensors;
  uint64_t observedTimeoutMilliseconds = 0;
  uint32_t selectedDevice = std::numeric_limits<uint32_t>::max();
  int64_t unavailableLogicalTile = -1;
  uint64_t freeMemoryBytes = 128ULL * 1024 * 1024;
  uint64_t totalMemoryBytes = 256ULL * 1024 * 1024;
  std::string runtimeLibraryDigest =
      "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  std::optional<uint32_t> clusterStatusOverride;

private:
  static wafer::runtime::RuntimeEnvironment
  makeProviderEnvironment(wafer::TargetLaunchABIId launchABI) {
    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    wafer::runtime::RuntimeEnvironment environment{
        target.id, target.targetIdentity, target.kernelRuntimeABI, launchABI,
        target.moduleFormat};
    if (launchABI == wafer::TargetLaunchABIId::
                         tx81ClusterDirectDTEPrepareMainV1()) {
      environment.supportsDirectDTE = true;
      environment.directDTEStatusABI =
          wafer::runtime::kDirectDTEStatusABI.str();
      environment.supportsHostWatchdog = true;
    }
    return environment;
  }

  bool shouldFail(llvm::StringRef operation) {
    if (failOperation != operation)
      return false;
    return matchingFailureCall++ == failIndex;
  }

  llvm::Error injectedFailure() {
    if (poisonOnFailure)
      contextState = wafer::runtime::BoardRuntimeContextState::Poisoned;
    return injected(failOperation);
  }

  struct Allocation {
    uintptr_t address;
    std::vector<uint8_t> bytes;
  };

  std::vector<Allocation>::iterator find(uintptr_t address) {
    return std::find_if(allocations.begin(), allocations.end(),
                        [&](const Allocation &allocation) {
                          return allocation.address == address;
                        });
  }

  uintptr_t nextAddress = 0x1000;
  std::vector<Allocation> allocations;
  std::vector<uintptr_t> liveModules;
  bool graphLive = false;
  bool submissionLive = false;
  size_t matchingFailureCall = 0;
  wafer::runtime::RuntimeEnvironment providerEnvironment;
  wafer::runtime::BoardRuntimeContextState contextState =
      wafer::runtime::BoardRuntimeContextState::Usable;
};

class BoardRuntimeTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(
        llvm::sys::fs::createUniqueDirectory("wafer-board-runtime-test", root));
    llvm::SmallString<256> modules(root);
    llvm::sys::path::append(modules, "modules");
    ASSERT_FALSE(llvm::sys::fs::create_directories(modules));
    modulePath = modules;
    llvm::sys::path::append(modulePath, "rank_00000.so");
    std::error_code error;
    llvm::raw_fd_ostream output(modulePath, error, llvm::sys::fs::OF_None);
    ASSERT_FALSE(error);
    output << moduleBytes;
  }

  void TearDown() override { llvm::sys::fs::remove_directories(root); }

  wafer::runtime::PackageManifest makeManifest() const {
    using namespace wafer::runtime;
    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    llvm::SHA256 hasher;
    hasher.update(moduleBytes);
    PackageManifest manifest(target.id, target.targetIdentity,
                             target.kernelRuntimeABI,
                             wafer::TargetLaunchABIId::perRankPointerBlockV1(),
                             target.moduleFormat);
    manifest.program = ProgramId(0);
    manifest.rankCount = 1;
    manifest.resources = {{ResourceId(0),
                           0,
                           PackageResourceRole::UserInput,
                           0,
                           "input",
                           {"f32", {16}},
                           64,
                           256,
                           PackageAccessMode::ReadOnly,
                           true},
                          {ResourceId(1),
                           0,
                           PackageResourceRole::Output,
                           0,
                           "output",
                           {"f32", {16}},
                           64,
                           256,
                           PackageAccessMode::WriteOnly,
                           true},
                          {ResourceId(2),
                           0,
                           PackageResourceRole::Workspace,
                           0,
                           "default_ddr_arena",
                           {"u8", {512}},
                           512,
                           256,
                           PackageAccessMode::ReadWrite,
                           false}};
    manifest.modules = {
        {ModuleId(0), "modules/rank_00000.so",
         "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true),
         target.moduleFormat.str(),
         {{PackageModuleExportRole::Main, "main"}}}};
    manifest.entries = {{EntryId(0),
                         0,
                         ModuleId(0),
                         {{0, ResourceId(0), PackageAccessMode::ReadOnly},
                          {1, ResourceId(1), PackageAccessMode::WriteOnly},
                          {2, ResourceId(2), PackageAccessMode::ReadWrite}},
                         CompletionId(0)}};
    manifest.completions = {{CompletionId(0), 0, "entry_return"}};
    return manifest;
  }

  void createRankModules(int64_t rankCount) const {
    for (int64_t rank = 1; rank < rankCount; ++rank) {
      std::string rankText = std::to_string(rank);
      llvm::SmallString<256> path(root);
      llvm::sys::path::append(path, "modules",
                              "rank_" + std::string(5 - rankText.size(), '0') +
                                  rankText + ".so");
      std::error_code error;
      llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
      ASSERT_FALSE(error);
      output << moduleBytes;
      output.close();
      ASSERT_FALSE(output.has_error());
    }
  }

  std::string moduleDigest() const {
    llvm::SHA256 hasher;
    hasher.update(moduleBytes);
    return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
  }

  wafer::runtime::PackageManifest makeRank16Manifest(
      wafer::TargetLaunchABIId launchABI =
          wafer::TargetLaunchABIId::perRankPointerBlockV1()) const {
    using namespace wafer::runtime;
    const wafer::TargetProfileRecord &target = wafer::getTargetProfileRecord(
        wafer::TargetProfileId::waferTx81SingleCardKernelV1());
    PackageManifest manifest(target.id, target.targetIdentity,
                             target.kernelRuntimeABI, launchABI,
                             target.moduleFormat);
    manifest.program = ProgramId(0);
    manifest.rankCount = 16;
    const bool sharedModule =
        launchABI ==
            wafer::TargetLaunchABIId::tx81KernelGridPointerTableV1() ||
        launchABI == wafer::TargetLaunchABIId::
                         tx81ClusterDirectDTEPrepareMainV1();
    const bool cluster =
        launchABI == wafer::TargetLaunchABIId::
                         tx81ClusterDirectDTEPrepareMainV1();

    uint64_t nextResource = 0;
    for (int64_t rank = 0; rank < 16; ++rank) {
      std::string rankText = std::to_string(rank);
      std::string moduleName =
          "rank_" + std::string(5 - rankText.size(), '0') + rankText + ".so";
      ModuleId module(sharedModule
                          ? 0
                          : static_cast<uint64_t>((rank * 7 + 5) % 16));
      EntryId entry(static_cast<uint64_t>((rank * 5 + 3) % 16));
      CompletionId completion(static_cast<uint64_t>((rank * 9 + 1) % 16));
      if (!sharedModule || rank == 0)
        manifest.modules.push_back(
            {module, "modules/" + moduleName, moduleDigest(),
             target.moduleFormat.str(),
             cluster
                 ? std::vector<PackageModuleExportRecord>{
                       {PackageModuleExportRole::Prepare, "prepare"},
                       {PackageModuleExportRole::Main, "main"}}
                 : std::vector<PackageModuleExportRecord>{
                       {PackageModuleExportRole::Main, "main"}}});
      manifest.completions.push_back({completion, rank, "entry_return"});

      std::vector<PackageABISlotBinding> slots;
      auto addResource = [&](PackageResourceRole role, llvm::StringRef name,
                             llvm::StringRef dtype, std::vector<int64_t> shape,
                             uint64_t bytes, PackageAccessMode access,
                             bool hostVisible, uint64_t alignment = 256) {
        ResourceId resource(nextResource++);
        manifest.resources.push_back(
            {resource,
             rank,
             role,
             0,
             (llvm::Twine(name) + "_rank_" + llvm::Twine(rank)).str(),
             {dtype.str(), std::move(shape)},
             bytes,
             alignment,
             access,
             hostVisible});
        slots.push_back({slots.size(), resource, access});
      };
      addResource(PackageResourceRole::UserInput, "input", "f32", {16}, 64,
                  PackageAccessMode::ReadOnly, true);
      addResource(PackageResourceRole::Output, "output", "f32", {16}, 64,
                  PackageAccessMode::WriteOnly, true);
      if (launchABI != wafer::TargetLaunchABIId::tx81ModelBootParamV1())
        addResource(PackageResourceRole::Workspace, "workspace", "u8", {512},
                    512, PackageAccessMode::ReadWrite, false);
      TransportRequirements transport = NoTransportRequirements{};
      if (cluster) {
        ResourceId status(nextResource++);
        manifest.resources.push_back(
            {status, rank, PackageResourceRole::TransportStatus, 0,
             (llvm::Twine("direct_dte_status_rank_") + llvm::Twine(rank))
                 .str(),
             {"u32", {1}}, kDirectDTEStatusStorageBytes,
             kDirectDTEStatusStorageAlignment, PackageAccessMode::ReadWrite,
             false});
        slots.push_back(
            {slots.size(), status, PackageAccessMode::ReadWrite});
        transport = DirectDTETransportRequirements{
            status, kDirectDTEStatusABI.str(), true};
      }
      manifest.entries.push_back({entry, rank, module, std::move(slots),
                                  completion, std::move(transport)});
    }

    // Identity order is intentionally unrelated to logical-rank order. The
    // verified package owns dense typed identities; the invocation owner must
    // still establish one canonical rank 0..15 submission.
    std::reverse(manifest.resources.begin(), manifest.resources.end());
    std::reverse(manifest.modules.begin(), manifest.modules.end());
    std::reverse(manifest.entries.begin(), manifest.entries.end());
    std::reverse(manifest.completions.begin(), manifest.completions.end());
    return manifest;
  }

  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verifyRank16(
      wafer::TargetLaunchABIId launchABI =
          wafer::TargetLaunchABIId::perRankPointerBlockV1()) const {
    const bool sharedModule =
        launchABI ==
            wafer::TargetLaunchABIId::tx81KernelGridPointerTableV1() ||
        launchABI == wafer::TargetLaunchABIId::
                         tx81ClusterDirectDTEPrepareMainV1();
    if (sharedModule) {
      for (int64_t rank = 1; rank < 16; ++rank) {
        std::string rankText = std::to_string(rank);
        llvm::SmallString<256> path(root);
        llvm::sys::path::append(
            path, "modules",
            "rank_" + std::string(5 - rankText.size(), '0') + rankText +
                ".so");
        if (llvm::sys::fs::exists(path))
          EXPECT_FALSE(llvm::sys::fs::remove(path));
      }
    } else {
      createRankModules(16);
    }
    return wafer::runtime::verifyPackageManifest(makeRank16Manifest(launchABI),
                                                 root);
  }

  static uint8_t rankInputByte(int64_t rank, size_t byte) {
    return static_cast<uint8_t>(rank * 17 + byte);
  }

  wafer::runtime::BoardRuntimeInvocationRequest
  makeRank16Request(const wafer::runtime::PackageManifest &manifest) const {
    using namespace wafer::runtime;
    BoardRuntimeInvocationRequest request;
    request.deviceId = 0;
    request.completionTimeoutMilliseconds = 4321;
    request.qualification = {
        0x514,
        16,
        "/dev/accel/dev-0",
        "0000:00:00.0",
        "sha256:"
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    };
    for (const PackageResourceRecord &resource : manifest.resources) {
      if (!resource.hostVisible)
        continue;
      std::vector<uint8_t> bytes(resource.bytes, 0);
      if (resource.role == PackageResourceRole::UserInput)
        for (size_t byte = 0; byte < bytes.size(); ++byte)
          bytes[byte] = rankInputByte(resource.logicalRank, byte);
      request.bindings.push_back({resource.id, std::move(bytes)});
    }
    return request;
  }

  static const wafer::runtime::PackageEntrypointRecord &
  findRankEntry(const wafer::runtime::PackageManifest &manifest,
                int64_t logicalRank) {
    auto iterator = llvm::find_if(manifest.entries, [&](const auto &entry) {
      return entry.logicalRank == logicalRank;
    });
    EXPECT_NE(iterator, manifest.entries.end());
    return *iterator;
  }

  llvm::Expected<wafer::runtime::VerifiedPackageManifest> verify() const {
    return wafer::runtime::verifyPackageManifest(makeManifest(), root);
  }

  wafer::runtime::BoardRuntimeRequest makeRequest() const {
    std::vector<uint8_t> input(64);
    for (size_t index = 0; index < input.size(); ++index)
      input[index] = static_cast<uint8_t>(index);
    wafer::runtime::BoardRuntimeRequest request;
    request.deviceId = 0;
    request.entry = wafer::runtime::EntryId(0);
    request.qualification = {
        0x514,
        16,
        "/dev/accel/dev-0",
        "0000:00:00.0",
        "sha256:"
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    };
    request.bindings = {
        {wafer::runtime::ResourceId(0), std::move(input)},
        {wafer::runtime::ResourceId(1), std::vector<uint8_t>(64)}};
    return request;
  }

  llvm::SmallString<256> root;
  llvm::SmallString<256> modulePath;
  const llvm::StringRef moduleBytes = "\x7f"
                                      "ELFboard-runtime-test";
};

TEST_F(BoardRuntimeTest, ExecutesTypedLifecycleAndCopiesCompleteOutput) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, makeRequest(), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(driver.selectedDevice, 0u);
  ASSERT_EQ(result->outputs.size(), 1u);
  ASSERT_EQ(result->outputs.front().bytes.size(), 64u);
  for (size_t index = 0; index < result->outputs.front().bytes.size(); ++index)
    EXPECT_EQ(result->outputs.front().bytes[index], index);
  EXPECT_EQ(result->completedStages.back(),
            wafer::runtime::BoardRuntimeStage::Cleanup);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "allocate"),
            3);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "h2d"), 2);
  ASSERT_EQ(driver.h2dPayloads.size(), 2u);
  EXPECT_TRUE(llvm::all_of(driver.h2dPayloads,
                           [](const auto &bytes) {
                             return bytes.size() == 64;
                           }));
  auto unload = std::find_if(
      driver.calls.begin(), driver.calls.end(),
      [](const std::string &call) { return call.find("unload-module:") == 0; });
  ASSERT_NE(unload, driver.calls.end());
  ASSERT_NE(unload + 1, driver.calls.end());
  EXPECT_TRUE((unload + 1)->find("free:") == 0);
}

TEST_F(BoardRuntimeTest,
       ExecutesCanonicalRank16InvocationWithOneProviderSubmission) {
  createRankModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  FakeBoardDriver driver;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(manifest), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  ASSERT_EQ(result->ranks.size(), 16u);
  ASSERT_EQ(driver.submittedLaunches.size(), 16u);
  for (int64_t rank = 0; rank < 16; ++rank) {
    const wafer::runtime::PackageEntrypointRecord &entry =
        findRankEntry(manifest, rank);
    EXPECT_EQ(result->ranks[rank].logicalRank, rank);
    EXPECT_EQ(result->ranks[rank].entry, entry.id);
    EXPECT_EQ(driver.submittedLaunches[rank].logicalRank, rank);
    EXPECT_EQ(driver.submittedLaunches[rank].entry, entry.id);
  }

  ASSERT_EQ(result->outputs.size(), 16u);
  for (const wafer::runtime::BoardRuntimeOutput &output : result->outputs) {
    auto resource =
        llvm::find_if(manifest.resources, [&](const auto &candidate) {
          return candidate.id == output.resource;
        });
    ASSERT_NE(resource, manifest.resources.end());
    EXPECT_EQ(resource->role, wafer::runtime::PackageResourceRole::Output);
    ASSERT_EQ(output.bytes.size(), 64u);
    for (size_t byte = 0; byte < output.bytes.size(); ++byte)
      EXPECT_EQ(output.bytes[byte],
                rankInputByte(resource->logicalRank, byte) ^
                    static_cast<uint8_t>(resource->logicalRank));
  }
  EXPECT_EQ(result->completedStages.back(),
            wafer::runtime::BoardRuntimeStage::Cleanup);
  EXPECT_EQ(driver.observedTimeoutMilliseconds, 4321u);

  auto submit =
      std::find(driver.calls.begin(), driver.calls.end(), "submit-all");
  ASSERT_NE(submit, driver.calls.end());
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "submit-all"),
            1);
  struct BeforeSubmit {
    llvm::StringRef operation;
    size_t expectedCount;
  };
  const BeforeSubmit beforeSubmit[] = {{"allocate", 48},
                                       {"h2d", 32},
                                       {"load-module", 16},
                                       {"resolve-entry", 16}};
  for (const BeforeSubmit &expected : beforeSubmit) {
    SCOPED_TRACE(expected.operation.str());
    EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                         expected.operation.str()),
              expected.expectedCount);
    EXPECT_EQ(std::find(submit, driver.calls.end(), expected.operation.str()),
              driver.calls.end());
  }

  auto release =
      std::find(driver.calls.begin(), driver.calls.end(), "release-submission");
  auto firstUnload = std::find_if(
      driver.calls.begin(), driver.calls.end(),
      [](const std::string &call) { return call.find("unload-module:") == 0; });
  auto firstFree = std::find_if(
      driver.calls.begin(), driver.calls.end(),
      [](const std::string &call) { return call.find("free:") == 0; });
  ASSERT_NE(release, driver.calls.end());
  ASSERT_NE(firstUnload, driver.calls.end());
  ASSERT_NE(firstFree, driver.calls.end());
  EXPECT_LT(release, firstUnload);
  EXPECT_LT(firstUnload, firstFree);

  std::vector<uintptr_t> expectedModules = driver.loadedHandles;
  std::reverse(expectedModules.begin(), expectedModules.end());
  EXPECT_EQ(driver.unloadedHandles, expectedModules);
  std::vector<uintptr_t> expectedAllocations = driver.allocatedAddresses;
  std::reverse(expectedAllocations.begin(), expectedAllocations.end());
  EXPECT_EQ(driver.freedAddresses, expectedAllocations);
}

TEST_F(BoardRuntimeTest,
       KernelGridLoadsOneSharedModuleAndSubmitsOneCanonicalGrid) {
  using wafer::TargetLaunchABIId;
  const TargetLaunchABIId launchABI =
      TargetLaunchABIId::tx81KernelGridPointerTableV1();
  createRankModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(launchABI);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  FakeBoardDriver driver(launchABI);
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(manifest), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "load-module"),
            1);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "resolve-entry"), 1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-grid"),
            1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "submit-all"),
            0);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "load-graph"),
            0);
  ASSERT_EQ(driver.submittedLaunches.size(), 16u);
  for (auto [rank, launch] : llvm::enumerate(driver.submittedLaunches)) {
    EXPECT_EQ(launch.logicalRank, static_cast<int64_t>(rank));
    EXPECT_EQ(launch.function.value, 0x9000u);
  }

  ASSERT_EQ(result->outputs.size(), 16u);
  for (const wafer::runtime::BoardRuntimeOutput &output : result->outputs) {
    auto resource = llvm::find_if(manifest.resources, [&](const auto &candidate) {
      return candidate.id == output.resource;
    });
    ASSERT_NE(resource, manifest.resources.end());
    for (size_t byte = 0; byte < output.bytes.size(); ++byte)
      EXPECT_EQ(output.bytes[byte],
                rankInputByte(resource->logicalRank, byte) ^
                    static_cast<uint8_t>(resource->logicalRank));
  }
}

TEST_F(BoardRuntimeTest,
       ClusterDirectDTEUsesTypedPrepareMainAndChecksStatusBeforeOutputs) {
  using wafer::TargetLaunchABIId;
  const TargetLaunchABIId launchABI =
      TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1();
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(launchABI);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  FakeBoardDriver driver(launchABI);
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(manifest), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "load-module"),
            1);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "resolve-entry"), 2);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-cluster-prepare-main"),
            1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "d2h"), 32);
  EXPECT_EQ(std::count_if(driver.h2dPayloads.begin(),
                          driver.h2dPayloads.end(), [](const auto &bytes) {
                            return bytes.size() ==
                                       wafer::runtime::
                                           kDirectDTEStatusStorageBytes &&
                                   llvm::all_of(bytes, [](uint8_t byte) {
                                     return byte == 0xff;
                                   });
                          }),
            16);
  ASSERT_EQ(result->outputs.size(), 16u);
  for (const wafer::runtime::BoardRuntimeOutput &output : result->outputs) {
    auto resource =
        llvm::find_if(manifest.resources, [&](const auto &candidate) {
          return candidate.id == output.resource;
        });
    ASSERT_NE(resource, manifest.resources.end());
    EXPECT_EQ(resource->role, wafer::runtime::PackageResourceRole::Output);
    for (size_t byte = 0; byte < output.bytes.size(); ++byte)
      EXPECT_EQ(output.bytes[byte],
                rankInputByte(resource->logicalRank, byte) ^
                    static_cast<uint8_t>(resource->logicalRank));
  }
}

TEST_F(BoardRuntimeTest,
       ClusterDirectDTEBadStatusQuarantinesBeforeAnyUserOutputReadback) {
  using wafer::TargetLaunchABIId;
  const TargetLaunchABIId launchABI =
      TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1();
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(launchABI);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver(launchABI);
  driver.clusterStatusOverride = static_cast<uint32_t>(
      wafer::runtime::DirectDTEStatusValue::Pending);
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  bool sawPoison = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const wafer::runtime::BoardRuntimeError &error) {
        sawPoison = true;
        EXPECT_EQ(error.getStage(),
                  wafer::runtime::BoardRuntimeStage::Completion);
        EXPECT_EQ(error.getContextState(),
                  wafer::runtime::BoardRuntimeContextState::Poisoned);
      });
  EXPECT_TRUE(sawPoison);
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back(), "quarantine");
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "d2h"), 1);
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(),
                      "release-submission"),
            driver.calls.end());
  EXPECT_TRUE(driver.unloadedHandles.empty());
  EXPECT_TRUE(driver.freedAddresses.empty());
}

TEST_F(BoardRuntimeTest,
       ClusterDirectDTEStatusReadbackFailureQuarantinesWithoutCleanup) {
  using wafer::TargetLaunchABIId;
  const TargetLaunchABIId launchABI =
      TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1();
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(launchABI);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver(launchABI);
  driver.failOperation = "d2h";
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_GE(driver.calls.size(), 2u);
  EXPECT_EQ(driver.calls[driver.calls.size() - 2], "d2h");
  EXPECT_EQ(driver.calls.back(), "quarantine");
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(),
                      "release-submission"),
            driver.calls.end());
  EXPECT_TRUE(driver.unloadedHandles.empty());
  EXPECT_TRUE(driver.freedAddresses.empty());
}

TEST_F(BoardRuntimeTest,
       ClusterDirectDTEOutputReadbackFailureQuarantinesWithoutCleanup) {
  using wafer::TargetLaunchABIId;
  const TargetLaunchABIId launchABI =
      TargetLaunchABIId::tx81ClusterDirectDTEPrepareMainV1();
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(launchABI);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver(launchABI);
  driver.failOperation = "d2h";
  driver.failIndex = 16;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_GE(driver.calls.size(), 2u);
  EXPECT_EQ(driver.calls[driver.calls.size() - 2], "d2h");
  EXPECT_EQ(driver.calls.back(), "quarantine");
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "d2h"), 17);
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(),
                      "release-submission"),
            driver.calls.end());
  EXPECT_TRUE(driver.unloadedHandles.empty());
  EXPECT_TRUE(driver.freedAddresses.empty());
}

TEST_F(BoardRuntimeTest,
       ModelLoadsCompleteGraphAndSubmitsTypedTensorDomain) {
  using wafer::TargetLaunchABIId;
  const TargetLaunchABIId launchABI =
      TargetLaunchABIId::tx81ModelBootParamV1();
  createRankModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(launchABI);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  FakeBoardDriver driver(launchABI);
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(manifest), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "load-graph"),
            1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "submit-model"),
            1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "load-module"),
            0);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "resolve-entry"), 0);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "submit-all"),
            0);
  ASSERT_EQ(driver.submittedModelTensors.size(), 32u);
  for (int64_t rank = 0; rank < 16; ++rank)
    EXPECT_EQ(llvm::count_if(driver.submittedModelTensors,
                             [&](const auto &tensor) {
                               return tensor.logicalRank == rank;
                             }),
              2);

  auto release =
      std::find(driver.calls.begin(), driver.calls.end(), "release-submission");
  auto unloadGraph =
      std::find(driver.calls.begin(), driver.calls.end(), "unload-graph");
  auto firstFree = std::find_if(
      driver.calls.begin(), driver.calls.end(),
      [](const std::string &call) { return call.find("free:") == 0; });
  ASSERT_NE(release, driver.calls.end());
  ASSERT_NE(unloadGraph, driver.calls.end());
  ASSERT_NE(firstFree, driver.calls.end());
  EXPECT_LT(release, unloadGraph);
  EXPECT_LT(unloadGraph, firstFree);

  ASSERT_EQ(result->outputs.size(), 16u);
  for (const wafer::runtime::BoardRuntimeOutput &output : result->outputs) {
    auto resource = llvm::find_if(manifest.resources, [&](const auto &candidate) {
      return candidate.id == output.resource;
    });
    ASSERT_NE(resource, manifest.resources.end());
    for (size_t byte = 0; byte < output.bytes.size(); ++byte)
      EXPECT_EQ(output.bytes[byte],
                rankInputByte(resource->logicalRank, byte) ^
                    static_cast<uint8_t>(resource->logicalRank));
  }
}

TEST_F(BoardRuntimeTest,
       MultiTilePoisonedProviderFailureIsTheFinalDriverCall) {
  using wafer::TargetLaunchABIId;
  struct Scenario {
    TargetLaunchABIId launchABI;
    llvm::StringLiteral operation;
  };
  const Scenario scenarios[] = {
      {TargetLaunchABIId::tx81KernelGridPointerTableV1(),
       "submit-kernel-grid"},
      {TargetLaunchABIId::tx81ModelBootParamV1(), "load-graph"},
      {TargetLaunchABIId::tx81ModelBootParamV1(), "submit-model"},
  };
  createRankModules(16);
  for (const Scenario &scenario : scenarios) {
    SCOPED_TRACE(scenario.operation.str());
    llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
        verifyRank16(scenario.launchABI);
    ASSERT_TRUE(static_cast<bool>(package))
        << llvm::toString(package.takeError());
    FakeBoardDriver driver(scenario.launchABI);
    driver.failOperation = scenario.operation.str();
    driver.poisonOnFailure = true;
    llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
        wafer::runtime::executeBoardInvocation(
            *package, root, makeRank16Request(package->getManifest()), driver);
    ASSERT_FALSE(static_cast<bool>(result));
    llvm::consumeError(result.takeError());
    ASSERT_FALSE(driver.calls.empty());
    EXPECT_EQ(driver.calls.back(), scenario.operation);
    EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(),
                        "release-submission"),
              driver.calls.end());
    EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(),
                        "unload-graph"),
              driver.calls.end());
    EXPECT_EQ(std::find_if(driver.calls.begin(), driver.calls.end(),
                           [](const std::string &call) {
                             return call.find("unload-module:") == 0 ||
                                    call.find("free:") == 0;
                           }),
              driver.calls.end());
  }
}

TEST_F(BoardRuntimeTest,
       ModelManifestRejectsUnqualifiedBootParamBeforeDriverCreation) {
  using wafer::TargetLaunchABIId;
  using wafer::runtime::PackageManifest;
  const TargetLaunchABIId launchABI =
      TargetLaunchABIId::tx81ModelBootParamV1();
  createRankModules(16);

  auto expectRejected = [&](PackageManifest manifest) {
    llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
        wafer::runtime::verifyPackageManifest(std::move(manifest), root);
    EXPECT_FALSE(static_cast<bool>(package));
    if (!package)
      llvm::consumeError(package.takeError());
  };

  PackageManifest zeroExtent = makeRank16Manifest(launchABI);
  for (auto &resource : zeroExtent.resources)
    if (resource.role == wafer::runtime::PackageResourceRole::UserInput)
      resource.type.shape = {0};
  expectRejected(std::move(zeroExtent));

  PackageManifest byteMismatch = makeRank16Manifest(launchABI);
  for (auto &resource : byteMismatch.resources)
    if (resource.role == wafer::runtime::PackageResourceRole::UserInput)
      resource.bytes -= sizeof(float);
  expectRejected(std::move(byteMismatch));

  PackageManifest insufficientAlignment = makeRank16Manifest(launchABI);
  for (auto &resource : insufficientAlignment.resources)
    resource.alignment = 1;
  expectRejected(std::move(insufficientAlignment));

  PackageManifest longSymbol = makeRank16Manifest(launchABI);
  for (auto &module : longSymbol.modules)
    module.exports.front().symbol.assign(128, 'm');
  expectRejected(std::move(longSymbol));

  PackageManifest nulSymbol = makeRank16Manifest(launchABI);
  for (auto &module : nulSymbol.modules)
    module.exports.front().symbol = std::string("ma\0in", 5);
  expectRejected(std::move(nulSymbol));
}

TEST_F(BoardRuntimeTest, Rank16RequiresCompleteUniqueTileInventory) {
  createRankModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.unavailableLogicalTile = 15;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  bool sawDeviceSelection = false;
  llvm::handleAllErrors(
      result.takeError(), [&](const wafer::runtime::BoardRuntimeError &error) {
        sawDeviceSelection = true;
        EXPECT_EQ(error.getStage(),
                  wafer::runtime::BoardRuntimeStage::DeviceSelection);
      });
  EXPECT_TRUE(sawDeviceSelection);
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(), "allocate"),
            driver.calls.end());
}

TEST_F(BoardRuntimeTest,
       Rank15RecoverableFailuresReturnNoPartialInvocationAndCleanup) {
  createRankModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  const wafer::runtime::EntryId rank15Entry = findRankEntry(manifest, 15).id;
  struct Scenario {
    llvm::StringRef operation;
    size_t failIndex;
    wafer::runtime::BoardRuntimeStage stage;
  };
  const Scenario scenarios[] = {
      {"allocate", 45, wafer::runtime::BoardRuntimeStage::ResourceAllocation},
      {"h2d", 31, wafer::runtime::BoardRuntimeStage::HostToDevice},
      {"load-module", 15, wafer::runtime::BoardRuntimeStage::ModuleLoad},
      {"resolve-entry", 15, wafer::runtime::BoardRuntimeStage::EntryResolve},
      {"d2h", 15, wafer::runtime::BoardRuntimeStage::DeviceToHost},
  };
  for (const Scenario &scenario : scenarios) {
    SCOPED_TRACE(scenario.operation.str());
    FakeBoardDriver driver;
    driver.failOperation = scenario.operation.str();
    driver.failIndex = scenario.failIndex;
    llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
        wafer::runtime::executeBoardInvocation(
            *package, root, makeRank16Request(manifest), driver);
    ASSERT_FALSE(static_cast<bool>(result));
    bool sawRank15Failure = false;
    llvm::handleAllErrors(
        result.takeError(),
        [&](const wafer::runtime::BoardRuntimeError &error) {
          sawRank15Failure = true;
          EXPECT_EQ(error.getStage(), scenario.stage);
          EXPECT_EQ(error.getLogicalRank(), 15);
          EXPECT_EQ(error.getEntry(), rank15Entry);
          EXPECT_EQ(error.getContextState(),
                    wafer::runtime::BoardRuntimeContextState::Usable);
        });
    EXPECT_TRUE(sawRank15Failure);
    EXPECT_EQ(driver.getContextState(),
              wafer::runtime::BoardRuntimeContextState::Usable);
    EXPECT_FALSE(driver.freedAddresses.empty());
    std::vector<uintptr_t> expectedFreed = driver.allocatedAddresses;
    std::reverse(expectedFreed.begin(), expectedFreed.end());
    EXPECT_EQ(driver.freedAddresses, expectedFreed);
    if (!driver.loadedHandles.empty()) {
      std::vector<uintptr_t> expectedUnloaded = driver.loadedHandles;
      std::reverse(expectedUnloaded.begin(), expectedUnloaded.end());
      EXPECT_EQ(driver.unloadedHandles, expectedUnloaded);
    }
    if (scenario.operation == "d2h") {
      EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "d2h"),
                16);
      EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                           "release-submission"),
                1);
    }
  }
}

TEST_F(BoardRuntimeTest,
       Rank15PoisonedFailuresAreTheFinalProviderCallWithoutCleanup) {
  createRankModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  const wafer::runtime::EntryId rank15Entry = findRankEntry(manifest, 15).id;
  struct Scenario {
    llvm::StringRef operation;
    size_t failIndex;
    wafer::runtime::BoardRuntimeStage stage;
  };
  const Scenario scenarios[] = {
      {"allocate", 45, wafer::runtime::BoardRuntimeStage::ResourceAllocation},
      {"h2d", 31, wafer::runtime::BoardRuntimeStage::HostToDevice},
      {"load-module", 15, wafer::runtime::BoardRuntimeStage::ModuleLoad},
      {"resolve-entry", 15, wafer::runtime::BoardRuntimeStage::EntryResolve},
      {"d2h", 15, wafer::runtime::BoardRuntimeStage::DeviceToHost},
  };
  for (const Scenario &scenario : scenarios) {
    SCOPED_TRACE(scenario.operation.str());
    FakeBoardDriver driver;
    driver.failOperation = scenario.operation.str();
    driver.failIndex = scenario.failIndex;
    driver.poisonOnFailure = true;
    llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
        wafer::runtime::executeBoardInvocation(
            *package, root, makeRank16Request(manifest), driver);
    ASSERT_FALSE(static_cast<bool>(result));
    bool sawRank15Failure = false;
    llvm::handleAllErrors(
        result.takeError(),
        [&](const wafer::runtime::BoardRuntimeError &error) {
          sawRank15Failure = true;
          EXPECT_EQ(error.getStage(), scenario.stage);
          EXPECT_EQ(error.getLogicalRank(), 15);
          EXPECT_EQ(error.getEntry(), rank15Entry);
          EXPECT_EQ(error.getContextState(),
                    wafer::runtime::BoardRuntimeContextState::Poisoned);
        });
    EXPECT_TRUE(sawRank15Failure);
    ASSERT_FALSE(driver.calls.empty());
    EXPECT_EQ(driver.calls.back(), scenario.operation);
    EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(),
                        "release-submission"),
              driver.calls.end());
    EXPECT_EQ(std::find_if(driver.calls.begin(), driver.calls.end(),
                           [](const std::string &call) {
                             return call.find("unload-module:") == 0;
                           }),
              driver.calls.end());
    EXPECT_EQ(std::find_if(driver.calls.begin(), driver.calls.end(),
                           [](const std::string &call) {
                             return call.find("free:") == 0;
                           }),
              driver.calls.end());
  }
}

TEST_F(BoardRuntimeTest,
       WholeDomainSubmitAndWaitFailuresNeverReturnPartialResults) {
  createRankModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  struct Scenario {
    llvm::StringRef operation;
    wafer::runtime::BoardRuntimeStage stage;
    bool expectsRelease;
  };
  const Scenario scenarios[] = {
      {"submit-all", wafer::runtime::BoardRuntimeStage::Launch, false},
      {"wait-all", wafer::runtime::BoardRuntimeStage::Completion, true},
  };
  for (const Scenario &scenario : scenarios) {
    SCOPED_TRACE(scenario.operation.str());
    FakeBoardDriver driver;
    driver.failOperation = scenario.operation.str();
    llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
        wafer::runtime::executeBoardInvocation(
            *package, root, makeRank16Request(package->getManifest()), driver);
    ASSERT_FALSE(static_cast<bool>(result));
    bool sawWholeDomainFailure = false;
    llvm::handleAllErrors(result.takeError(),
                          [&](const wafer::runtime::BoardRuntimeError &error) {
                            sawWholeDomainFailure = true;
                            EXPECT_EQ(error.getStage(), scenario.stage);
                            EXPECT_EQ(error.getLogicalRank(), -1);
                            EXPECT_FALSE(error.getEntry().isValid());
                          });
    EXPECT_TRUE(sawWholeDomainFailure);
    EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                         "release-submission"),
              scenario.expectsRelease ? 1 : 0);
    EXPECT_EQ(driver.freedAddresses.size(), driver.allocatedAddresses.size());
    EXPECT_EQ(driver.unloadedHandles.size(), driver.loadedHandles.size());
  }
}

TEST_F(BoardRuntimeTest, Rank16PoisonedWaitIsTheFinalProviderCall) {
  createRankModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.failOperation = "wait-all";
  driver.poisonOnFailure = true;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back(), "wait-all");
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(), "d2h"),
            driver.calls.end());
  EXPECT_EQ(
      std::find(driver.calls.begin(), driver.calls.end(), "release-submission"),
      driver.calls.end());
  EXPECT_TRUE(driver.freedAddresses.empty());
  EXPECT_TRUE(driver.unloadedHandles.empty());
}

TEST_F(BoardRuntimeTest, PreflightFailureHasNoDriverSideEffects) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  wafer::runtime::BoardRuntimeRequest request = makeRequest();
  request.bindings.pop_back();
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, std::move(request),
                                        driver);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("preflight"),
            std::string::npos);
  EXPECT_TRUE(driver.calls.empty());
}

TEST_F(BoardRuntimeTest, QualificationMismatchStopsBeforeAllocation) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  wafer::runtime::BoardRuntimeRequest request = makeRequest();
  request.qualification.runtimeVersion = 0x513;
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, std::move(request),
                                        driver);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("qualification"),
            std::string::npos);
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(), "allocate"),
            driver.calls.end());
}

TEST_F(BoardRuntimeTest, AggregateDemandUsesCurrentFreeMemory) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.freeMemoryBytes = 64ULL * 1024 * 1024 + 511;
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, makeRequest(), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("aggregate"),
            std::string::npos);
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(), "allocate"),
            driver.calls.end());
}

TEST_F(BoardRuntimeTest, ResolveFailureSuppressesLaunchAndCleansInReverse) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.failOperation = "resolve-entry";
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, makeRequest(), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  bool sawTypedError = false;
  llvm::handleAllErrors(
      result.takeError(), [&](const wafer::runtime::BoardRuntimeError &error) {
        sawTypedError = true;
        EXPECT_EQ(error.getStage(),
                  wafer::runtime::BoardRuntimeStage::EntryResolve);
      });
  EXPECT_TRUE(sawTypedError);
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(), "submit-all"),
            driver.calls.end());
  EXPECT_NE(std::find_if(driver.calls.begin(), driver.calls.end(),
                         [](const std::string &call) {
                           return call.find("unload-module:") == 0;
                         }),
            driver.calls.end());
}

TEST_F(BoardRuntimeTest, RecoverableFailuresReportEveryProviderStage) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  struct Scenario {
    llvm::StringRef operation;
    wafer::runtime::BoardRuntimeStage stage;
  };
  const Scenario scenarios[] = {
      {"get-device-count", wafer::runtime::BoardRuntimeStage::DeviceSelection},
      {"select-device", wafer::runtime::BoardRuntimeStage::DeviceSelection},
      {"device-info", wafer::runtime::BoardRuntimeStage::DeviceSelection},
      {"allocate", wafer::runtime::BoardRuntimeStage::ResourceAllocation},
      {"load-module", wafer::runtime::BoardRuntimeStage::ModuleLoad},
  };
  for (const Scenario &scenario : scenarios) {
    SCOPED_TRACE(scenario.operation.str());
    FakeBoardDriver driver;
    driver.failOperation = scenario.operation.str();
    llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
        wafer::runtime::executeBoardEntry(*package, root, makeRequest(),
                                          driver);
    ASSERT_FALSE(static_cast<bool>(result));
    bool sawExpectedStage = false;
    llvm::handleAllErrors(result.takeError(),
                          [&](const wafer::runtime::BoardRuntimeError &error) {
                            sawExpectedStage |=
                                error.getStage() == scenario.stage;
                          });
    EXPECT_TRUE(sawExpectedStage);
  }
}

TEST_F(BoardRuntimeTest, PoisonedCompletionFailureStopsAllProviderCalls) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.failOperation = "wait-all";
  driver.poisonOnFailure = true;
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, makeRequest(), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  bool sawTypedError = false;
  llvm::handleAllErrors(
      result.takeError(), [&](const wafer::runtime::BoardRuntimeError &error) {
        sawTypedError = true;
        EXPECT_EQ(error.getStage(),
                  wafer::runtime::BoardRuntimeStage::Completion);
        EXPECT_EQ(error.getContextState(),
                  wafer::runtime::BoardRuntimeContextState::Poisoned);
      });
  EXPECT_TRUE(sawTypedError);
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back(), "wait-all");
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(), "d2h"),
            driver.calls.end());
  EXPECT_EQ(std::find_if(driver.calls.begin(), driver.calls.end(),
                         [](const std::string &call) {
                           return call.find("unload-module:") == 0;
                         }),
            driver.calls.end());
  EXPECT_EQ(std::find_if(driver.calls.begin(), driver.calls.end(),
                         [](const std::string &call) {
                           return call.find("free:") == 0;
                         }),
            driver.calls.end());
}

TEST_F(BoardRuntimeTest, PoisonedHostToDeviceFailureIsLastProviderCall) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.failOperation = "h2d";
  driver.poisonOnFailure = true;
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, makeRequest(), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back(), "h2d");
}

TEST_F(BoardRuntimeTest, PoisonedLaunchFailureIsLastProviderCall) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.failOperation = "submit-all";
  driver.poisonOnFailure = true;
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, makeRequest(), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back(), "submit-all");
}

TEST_F(BoardRuntimeTest, PoisonedDeviceToHostFailureIsLastProviderCall) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.failOperation = "d2h";
  driver.poisonOnFailure = true;
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, makeRequest(), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back(), "d2h");
}

TEST_F(BoardRuntimeTest, PoisonedUnloadFailureStopsRemainingCleanup) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.failOperation = "unload-module";
  driver.poisonOnFailure = true;
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, makeRequest(), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back().find("unload-module:"), 0u);
}

TEST_F(BoardRuntimeTest, PoisonedFreeFailureStopsRemainingCleanup) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.failOperation = "free";
  driver.poisonOnFailure = true;
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, makeRequest(), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back().find("free:"), 0u);
  EXPECT_EQ(std::count_if(driver.calls.begin(), driver.calls.end(),
                          [](const std::string &call) {
                            return call.find("free:") == 0;
                          }),
            1);
}

} // namespace
