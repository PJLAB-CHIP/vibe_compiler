//===- BoardRuntimeTest.cpp - Board provider lifecycle tests ------------===//

#include "Wafer/Runtime/BoardRuntime.h"
#include "Wafer/ABI/Tx81ProfilerABI.h"

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
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

static_assert(!std::is_copy_constructible_v<
              wafer::runtime::QualifiedBoardRuntimeSession>);
static_assert(
    !std::is_copy_assignable_v<wafer::runtime::QualifiedBoardRuntimeSession>);
static_assert(
    std::is_move_constructible_v<wafer::runtime::QualifiedBoardRuntimeSession>);

llvm::Error injected(llvm::StringRef operation) {
  return llvm::createStringError(llvm::errc::io_error, "injected %s failure",
                                 operation.str().c_str());
}

enum class TestLaunchContractCase {
  PerRank,
  Grid,
  GridRankRows,
  Cluster,
  Model,
};

wafer::RuntimeLaunchContract
makeLaunchContract(TestLaunchContractCase launchCase) {
  using namespace wafer;
  switch (launchCase) {
  case TestLaunchContractCase::PerRank:
    return llvm::cantFail(RuntimeLaunchContract::createKernel(
        KernelLaunchForm::PerRank, KernelEntryABI::RankLocalPointerBlock,
        {RuntimeLaunchPhaseRole::Main}));
  case TestLaunchContractCase::Grid:
    return llvm::cantFail(RuntimeLaunchContract::createKernel(
        KernelLaunchForm::Grid, KernelEntryABI::RankMajorPointerTable,
        {RuntimeLaunchPhaseRole::Main}));
  case TestLaunchContractCase::GridRankRows:
    return llvm::cantFail(RuntimeLaunchContract::createKernel(
        KernelLaunchForm::Grid, KernelEntryABI::RankRowPointerTable,
        {RuntimeLaunchPhaseRole::Main}));
  case TestLaunchContractCase::Cluster:
    return llvm::cantFail(RuntimeLaunchContract::createKernel(
        KernelLaunchForm::Cluster, KernelEntryABI::RankMajorPointerTable,
        {RuntimeLaunchPhaseRole::Prepare, RuntimeLaunchPhaseRole::Main}));
  case TestLaunchContractCase::Model:
    return llvm::cantFail(RuntimeLaunchContract::createModel(
        ModelEntryABI::Tx81ModelBootParam, {RuntimeLaunchPhaseRole::Main}));
  }
  llvm_unreachable("unknown test launch contract case");
}

class FakeBoardDriver final : public wafer::runtime::BoardRuntimeDriver {
public:
  FakeBoardDriver() : providerEnvironment(makeProviderEnvironment()) {}

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

  llvm::Expected<wafer::runtime::BoardGraphHandle>
  loadGraph(llvm::ArrayRef<wafer::runtime::BoardGraphModuleSnapshot> modules,
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

  llvm::Error submitKernelPhase(
      wafer::KernelLaunchForm form, wafer::RuntimeLaunchPhaseRole phaseRole,
      llvm::ArrayRef<wafer::runtime::BoardRankLaunch> launches,
      wafer::runtime::BoardDeviceTimingPolicy timingPolicy) override {
    const std::string operation =
        (llvm::Twine("submit-kernel-phase:") +
         wafer::stringifyKernelLaunchForm(form) + ":" +
         wafer::stringifyRuntimeLaunchPhaseRole(phaseRole))
            .str();
    calls.push_back(operation);
    observedDeviceTimingPolicies.push_back(timingPolicy);
    if (submitEntered == std::chrono::steady_clock::time_point{})
      submitEntered = std::chrono::steady_clock::now();
    if (submitDelay.count() != 0)
      std::this_thread::sleep_for(submitDelay);
    if (shouldFail(operation))
      return injectedFailure();
    if (launches.empty() || phaseSubmitted ||
        (submissionLive && activeKernelForm != form))
      return injected("invalid-kernel-phase-state");
    if (!submissionLive) {
      submissionLive = true;
      activeKernelForm = form;
    }

    submittedLaunches.assign(launches.begin(), launches.end());
    submittedKernelPhases.push_back(phaseRole);
    const uintptr_t sharedFunction = launches.front().function.value;
    for (auto [rank, launch] : llvm::enumerate(launches)) {
      std::vector<uint64_t> decodedArguments;
      llvm::ArrayRef<uint64_t> arguments = launch.arguments;
      if (decodeRankRowArguments) {
        if (launch.arguments.size() != 1)
          return injected("invalid-rank-row-launch-packet");
        auto row = find(launch.arguments.front());
        if (row == allocations.end() || row->bytes.empty() ||
            row->bytes.size() % sizeof(uint64_t) != 0)
          return injected("invalid-rank-row-storage");
        decodedArguments.resize(row->bytes.size() / sizeof(uint64_t));
        std::memcpy(decodedArguments.data(), row->bytes.data(),
                    row->bytes.size());
        arguments = decodedArguments;
      }
      if (launch.logicalRank != static_cast<int64_t>(rank) ||
          launch.function.value == 0 || arguments.size() < 2 ||
          (form != wafer::KernelLaunchForm::PerRank &&
           launch.function.value != sharedFunction))
        return injected("invalid-canonical-kernel-phase");
      auto input = find(arguments[0]);
      auto output = find(arguments[1]);
      if (input == allocations.end() || output == allocations.end() ||
          input->bytes.size() != output->bytes.size())
        return injected("invalid-kernel-phase-buffers");
      if (phaseRole != wafer::RuntimeLaunchPhaseRole::Main)
        continue;
      for (size_t byte = 0; byte < input->bytes.size(); ++byte)
        output->bytes[byte] = input->bytes[byte] ^ static_cast<uint8_t>(rank);
      auto status = find(arguments.back());
      if (status != allocations.end() && arguments.size() >= 4 &&
          status->bytes.size() ==
              wafer::runtime::kDirectDTEStatusStorageBytes) {
        const uint32_t terminalStatus =
            transportStatusOverride.value_or(static_cast<uint32_t>(
                wafer::runtime::DirectDTEStatusValue::Success));
        std::memcpy(status->bytes.data() +
                        wafer::runtime::kDirectDTEStatusValueOffset,
                    &terminalStatus, sizeof(terminalStatus));
      }
    }
    phaseSubmitted = true;
    activeKernelPhase = phaseRole;
    return llvm::Error::success();
  }

  llvm::Error
  submitModel(wafer::runtime::BoardGraphHandle graph,
              llvm::ArrayRef<wafer::runtime::BoardModelTensorLaunch> tensors,
              wafer::runtime::BoardDeviceTimingPolicy timingPolicy) override {
    calls.push_back("submit-model");
    observedDeviceTimingPolicies.push_back(timingPolicy);
    if (shouldFail("submit-model"))
      return injectedFailure();
    if (submissionLive || phaseSubmitted || !graphLive ||
        graph.value != 0xa000 || tensors.empty())
      return injected("invalid-model-submit-state");
    submittedModelTensors.assign(tensors.begin(), tensors.end());
    for (int64_t rank = 0; rank < 16; ++rank) {
      auto input = llvm::find_if(tensors, [&](const auto &tensor) {
        return tensor.logicalRank == rank &&
               tensor.role == wafer::runtime::PackageResourceRole::UserInput;
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
    phaseSubmitted = true;
    activeKernelPhase.reset();
    return llvm::Error::success();
  }

  llvm::Expected<wafer::runtime::BoardCompletionObservation>
  waitCurrentSubmission(wafer::runtime::BoardCompletionDeadline deadline,
                        wafer::runtime::BoardCompletionObservationPolicy
                            observationPolicy) override {
    calls.push_back("wait-current-submission");
    observedDeadlines.push_back(deadline);
    observedCompletionObservationPolicy = observationPolicy;
    if (shouldFail("wait-current-submission")) {
      llvm::Error error = injectedFailure();
      if (contextState == wafer::runtime::BoardRuntimeContextState::Usable)
        phaseSubmitted = false;
      return error;
    }
    if (!submissionLive || !phaseSubmitted)
      return injected("wait-without-submit");
    const auto waitBegin = std::chrono::steady_clock::now();
    if (waitDelay.count() != 0)
      std::this_thread::sleep_for(waitDelay);
    completionObserved = std::chrono::steady_clock::now();
    const auto resolution =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            completionObserved - waitBegin)
            .count();
    phaseSubmitted = false;
    const size_t observationIndex = observedDeadlines.size() - 1;
    std::optional<uint64_t> deviceExecutionNanoseconds;
    if (observationIndex < deviceExecutionNanosecondsByWait.size())
      deviceExecutionNanoseconds =
          deviceExecutionNanosecondsByWait[observationIndex];
    return wafer::runtime::BoardCompletionObservation{
        resolution > 0 ? static_cast<uint64_t>(resolution) : 0,
        deviceExecutionNanoseconds};
  }

  llvm::Error releaseSubmission() override {
    calls.push_back("release-submission");
    if (shouldFail("release-submission")) {
      if (contextState == wafer::runtime::BoardRuntimeContextState::Usable)
        submissionLive = false;
      return injectedFailure();
    }
    if (!submissionLive || phaseSubmitted)
      return injected("release-without-submit");
    submissionLive = false;
    activeKernelForm.reset();
    activeKernelPhase.reset();
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
  std::vector<wafer::RuntimeLaunchPhaseRole> submittedKernelPhases;
  std::vector<wafer::runtime::BoardModelTensorLaunch> submittedModelTensors;
  std::vector<wafer::runtime::BoardCompletionDeadline> observedDeadlines;
  std::vector<wafer::runtime::BoardDeviceTimingPolicy>
      observedDeviceTimingPolicies;
  std::vector<std::optional<uint64_t>> deviceExecutionNanosecondsByWait;
  wafer::runtime::BoardCompletionObservationPolicy
      observedCompletionObservationPolicy =
          wafer::runtime::BoardCompletionObservationPolicy::Normal;
  uint32_t selectedDevice = std::numeric_limits<uint32_t>::max();
  int64_t unavailableLogicalTile = -1;
  uint64_t freeMemoryBytes = 128ULL * 1024 * 1024;
  uint64_t totalMemoryBytes = 256ULL * 1024 * 1024;
  std::chrono::milliseconds submitDelay{0};
  std::chrono::milliseconds waitDelay{0};
  std::chrono::steady_clock::time_point submitEntered;
  std::chrono::steady_clock::time_point completionObserved;
  std::string runtimeLibraryDigest =
      "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  std::optional<uint32_t> transportStatusOverride;
  bool decodeRankRowArguments = false;

private:
  static wafer::runtime::RuntimeEnvironment makeProviderEnvironment() {
    wafer::runtime::RuntimeEnvironment environment{
        wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
        wafer::kCurrentTargetModuleFormat};
    environment.supportedKernelLaunchForms = {wafer::KernelLaunchForm::PerRank,
                                              wafer::KernelLaunchForm::Grid,
                                              wafer::KernelLaunchForm::Cluster};
    environment.supportedKernelEntryABIs = {
        wafer::KernelEntryABI::RankLocalPointerBlock,
        wafer::KernelEntryABI::RankMajorPointerTable,
        wafer::KernelEntryABI::RankRowPointerTable};
    environment.supportedModelEntryABIs = {
        wafer::ModelEntryABI::Tx81ModelBootParam};
    environment.supportsDirectDTE = true;
    environment.directDTEStatusABI = wafer::runtime::kDirectDTEStatusABI.str();
    environment.supportsHostWatchdog = true;
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
  bool phaseSubmitted = false;
  std::optional<wafer::KernelLaunchForm> activeKernelForm;
  std::optional<wafer::RuntimeLaunchPhaseRole> activeKernelPhase;
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
    llvm::SHA256 hasher;
    hasher.update(moduleBytes);
    PackageManifest manifest(
        wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
        makeLaunchContract(TestLaunchContractCase::PerRank),
        wafer::kCurrentTargetModuleFormat);
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
        {ModuleId(0),
         "modules/rank_00000.so",
         "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true),
         wafer::kCurrentTargetModuleFormat.str(),
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
      TestLaunchContractCase launchCase = TestLaunchContractCase::PerRank,
      bool withProfiler = false,
      std::optional<bool> directDTEOverride = std::nullopt) const {
    using namespace wafer::runtime;
    PackageManifest manifest(
        wafer::kCurrentTargetIdentity, wafer::kCurrentKernelRuntimeABI,
        makeLaunchContract(launchCase), wafer::kCurrentTargetModuleFormat);
    manifest.program = ProgramId(0);
    manifest.rankCount = 16;
    const bool sharedModule =
        launchCase == TestLaunchContractCase::Grid ||
        launchCase == TestLaunchContractCase::GridRankRows ||
        launchCase == TestLaunchContractCase::Cluster;
    const bool cluster = launchCase == TestLaunchContractCase::Cluster;
    const bool model = launchCase == TestLaunchContractCase::Model;
    const bool directDTE = directDTEOverride.value_or(cluster);

    uint64_t nextResource = 0;
    for (int64_t rank = 0; rank < 16; ++rank) {
      std::string rankText = std::to_string(rank);
      std::string moduleName =
          "rank_" + std::string(5 - rankText.size(), '0') + rankText + ".so";
      ModuleId module(
          sharedModule ? 0 : static_cast<uint64_t>((rank * 7 + 5) % 16));
      EntryId entry(static_cast<uint64_t>((rank * 5 + 3) % 16));
      CompletionId completion(static_cast<uint64_t>((rank * 9 + 1) % 16));
      if (!sharedModule || rank == 0)
        manifest.modules.push_back(
            {module, "modules/" + moduleName, moduleDigest(),
             wafer::kCurrentTargetModuleFormat.str(),
             cluster ? std::vector<
                           PackageModuleExportRecord>{{PackageModuleExportRole::
                                                           Prepare,
                                                       "prepare"},
                                                      {PackageModuleExportRole::
                                                           Main,
                                                       "main"}}
                     : std::vector<PackageModuleExportRecord>{
                           {PackageModuleExportRole::Main, "main"}}});
      manifest.completions.push_back({completion, rank, "entry_return"});

      std::vector<PackageABISlotBinding> slots;
      auto addResource = [&](PackageResourceRole role, llvm::StringRef name,
                             llvm::StringRef dtype, std::vector<int64_t> shape,
                             uint64_t bytes, PackageAccessMode access,
                             bool hostVisible, uint64_t alignment = 256,
                             int64_t roleIndex = 0) {
        ResourceId resource(nextResource++);
        manifest.resources.push_back(
            {resource,
             rank,
             role,
             roleIndex,
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
      if (!model)
        addResource(PackageResourceRole::Workspace, "workspace", "u8", {512},
                    512, PackageAccessMode::ReadWrite, false);
      if (withProfiler)
        addResource(PackageResourceRole::Workspace, "profiler_record", "u8",
                    {WAFER_TX81_PROFILER_MIN_BUFFER_BYTES},
                    WAFER_TX81_PROFILER_MIN_BUFFER_BYTES,
                    PackageAccessMode::ReadWrite, false,
                    WAFER_TX81_PROFILER_BUFFER_ALIGNMENT, /*roleIndex=*/1);
      TransportRequirements transport = NoTransportRequirements{};
      if (directDTE) {
        ResourceId status(nextResource++);
        manifest.resources.push_back(
            {status,
             rank,
             PackageResourceRole::TransportStatus,
             0,
             (llvm::Twine("direct_dte_status_rank_") + llvm::Twine(rank)).str(),
             {"u32", {1}},
             kDirectDTEStatusStorageBytes,
             kDirectDTEStatusStorageAlignment,
             PackageAccessMode::ReadWrite,
             false});
        slots.push_back({slots.size(), status, PackageAccessMode::ReadWrite});
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
      TestLaunchContractCase launchCase = TestLaunchContractCase::PerRank,
      bool withProfiler = false,
      std::optional<bool> directDTEOverride = std::nullopt) const {
    const bool sharedModule =
        launchCase == TestLaunchContractCase::Grid ||
        launchCase == TestLaunchContractCase::GridRankRows ||
        launchCase == TestLaunchContractCase::Cluster;
    if (sharedModule) {
      for (int64_t rank = 1; rank < 16; ++rank) {
        std::string rankText = std::to_string(rank);
        llvm::SmallString<256> path(root);
        llvm::sys::path::append(
            path, "modules",
            "rank_" + std::string(5 - rankText.size(), '0') + rankText + ".so");
        if (llvm::sys::fs::exists(path))
          EXPECT_FALSE(llvm::sys::fs::remove(path));
      }
    } else {
      createRankModules(16);
    }
    return wafer::runtime::verifyPackageManifest(
        makeRank16Manifest(launchCase, withProfiler, directDTEOverride), root);
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
  EXPECT_EQ(driver.observedCompletionObservationPolicy,
            wafer::runtime::BoardCompletionObservationPolicy::Normal);
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
  EXPECT_TRUE(llvm::all_of(driver.h2dPayloads, [](const auto &bytes) {
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
  ASSERT_EQ(driver.observedDeadlines.size(), 1u);
  EXPECT_GT(driver.observedDeadlines.front(), driver.submitEntered);

  auto submit = std::find(driver.calls.begin(), driver.calls.end(),
                          "submit-kernel-phase:per-rank:main");
  ASSERT_NE(submit, driver.calls.end());
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:per-rank:main"),
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
  createRankModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  FakeBoardDriver driver;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(manifest), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "load-module"),
            1);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "resolve-entry"), 1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:grid:main"),
            1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:per-rank:main"),
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
    auto resource =
        llvm::find_if(manifest.resources, [&](const auto &candidate) {
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
       KernelGridRankRowsOwnDeviceStorageAndSubmitOnePointerPerRank) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::GridRankRows);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  FakeBoardDriver driver;
  driver.decodeRankRowArguments = true;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(manifest), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  ASSERT_EQ(driver.submittedLaunches.size(), 16u);
  ASSERT_EQ(driver.allocatedAddresses.size(), 64u);
  ASSERT_GE(driver.h2dPayloads.size(), 16u);
  const size_t firstRowPayload = driver.h2dPayloads.size() - 16;
  for (size_t rank = 0; rank < 16; ++rank) {
    const auto &launch = driver.submittedLaunches[rank];
    ASSERT_EQ(launch.arguments.size(), 1u);
    EXPECT_EQ(launch.arguments.front(), driver.allocatedAddresses[48 + rank]);

    const std::vector<uint8_t> &payload =
        driver.h2dPayloads[firstRowPayload + rank];
    ASSERT_EQ(payload.size(), 3 * sizeof(uint64_t));
    std::array<uint64_t, 3> row{};
    std::memcpy(row.data(), payload.data(), payload.size());
    for (size_t slot = 0; slot < row.size(); ++slot)
      EXPECT_EQ(row[slot], driver.allocatedAddresses[rank * 3 + slot]);
  }
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:grid:main"),
            1);
  EXPECT_EQ(result->outputs.size(), 16u);
  EXPECT_EQ(driver.freedAddresses.size(), driver.allocatedAddresses.size());
}

TEST_F(BoardRuntimeTest,
       ProfilerWorkspaceIsInitializedAndReturnedOutsideUserOutputs) {
  using namespace wafer::runtime;
  createRankModules(16);
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid, /*withProfiler=*/true);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());

  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  for (const PackageResourceRecord &resource :
       package->getManifest().resources) {
    if (resource.role != PackageResourceRole::Workspace ||
        resource.roleIndex != 1)
      continue;
    std::vector<uint8_t> bytes(resource.bytes,
                               static_cast<uint8_t>(resource.logicalRank));
    request.profilerBindings.push_back({resource.id, std::move(bytes)});
  }
  ASSERT_EQ(request.profilerBindings.size(), 16u);

  FakeBoardDriver driver;
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->outputs.size(), 16u);
  EXPECT_EQ(std::count_if(driver.h2dPayloads.begin(), driver.h2dPayloads.end(),
                          [](const auto &bytes) {
                            return bytes.size() ==
                                   WAFER_TX81_PROFILER_MIN_BUFFER_BYTES;
                          }),
            16);
  EXPECT_EQ(
      std::count_if(driver.h2dPayloads.begin(), driver.h2dPayloads.end(),
                    [](const auto &bytes) { return bytes.size() == 512; }),
      0);
  ASSERT_EQ(result->profilerOutputs.size(), 16u);
  for (const BoardRuntimeOutput &output : result->profilerOutputs) {
    auto resource = llvm::find_if(
        package->getManifest().resources,
        [&](const auto &candidate) { return candidate.id == output.resource; });
    ASSERT_NE(resource, package->getManifest().resources.end());
    EXPECT_EQ(resource->role, PackageResourceRole::Workspace);
    EXPECT_EQ(resource->roleIndex, 1);
    EXPECT_EQ(llvm::find_if(result->outputs,
                            [&](const auto &userOutput) {
                              return userOutput.resource == output.resource;
                            }),
              result->outputs.end());
    EXPECT_TRUE(llvm::all_of(output.bytes, [&](uint8_t byte) {
      return byte == static_cast<uint8_t>(resource->logicalRank);
    }));
  }
}

TEST_F(BoardRuntimeTest, ProfilerWorkspaceRequiresAllSixteenTiles) {
  using namespace wafer::runtime;
  createRankModules(16);
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid, /*withProfiler=*/true);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());

  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  for (const PackageResourceRecord &resource :
       package->getManifest().resources) {
    if (resource.role == PackageResourceRole::Workspace &&
        resource.roleIndex == 1)
      request.profilerBindings.push_back(
          {resource.id, std::vector<uint8_t>(resource.bytes)});
  }
  request.profilerBindings.pop_back();

  FakeBoardDriver driver;
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("all-and-only"),
            std::string::npos);
  EXPECT_TRUE(driver.calls.empty());
}

TEST_F(BoardRuntimeTest, ProfilerWorkspaceRequiresExactRecordByteCount) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid, /*withProfiler=*/true);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());

  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  for (const PackageResourceRecord &resource :
       package->getManifest().resources) {
    if (resource.role == PackageResourceRole::Workspace &&
        resource.roleIndex == 1)
      request.profilerBindings.push_back(
          {resource.id, std::vector<uint8_t>(resource.bytes)});
  }
  ASSERT_EQ(request.profilerBindings.size(), 16u);
  request.profilerBindings.front().bytes.pop_back();

  FakeBoardDriver driver;
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("byte count"),
            std::string::npos);
  EXPECT_TRUE(driver.calls.empty());
}

TEST_F(BoardRuntimeTest, ProfilerWorkspaceRejectsUnregisteredAlignedSize) {
  using namespace wafer::runtime;
  createRankModules(1);
  PackageManifest manifest =
      makeRank16Manifest(TestLaunchContractCase::Grid, /*withProfiler=*/true);
  const uint64_t unregisteredBytes = WAFER_TX81_PROFILER_MIN_BUFFER_BYTES +
                                     WAFER_TX81_PROFILER_BUFFER_ALIGNMENT;
  for (PackageResourceRecord &resource : manifest.resources) {
    if (resource.role != PackageResourceRole::Workspace ||
        resource.roleIndex != 1)
      continue;
    resource.bytes = unregisteredBytes;
    resource.type.shape = {static_cast<int64_t>(unregisteredBytes)};
  }
  llvm::Expected<VerifiedPackageManifest> package =
      verifyPackageManifest(std::move(manifest), root);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());

  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  for (const PackageResourceRecord &resource :
       package->getManifest().resources) {
    if (resource.role == PackageResourceRole::Workspace &&
        resource.roleIndex == 1)
      request.profilerBindings.push_back(
          {resource.id, std::vector<uint8_t>(resource.bytes)});
  }

  FakeBoardDriver driver;
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("exact internal"),
            std::string::npos);
  EXPECT_TRUE(driver.calls.empty());
}

TEST_F(BoardRuntimeTest,
       OrdinaryWorkspaceCannotBeInjectedThroughProfilerBindings) {
  using namespace wafer::runtime;
  createRankModules(16);
  llvm::Expected<VerifiedPackageManifest> package = verifyRank16();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());

  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  auto workspace =
      llvm::find_if(package->getManifest().resources, [](const auto &resource) {
        return resource.role == PackageResourceRole::Workspace &&
               resource.roleIndex == 0;
      });
  ASSERT_NE(workspace, package->getManifest().resources.end());
  request.profilerBindings.push_back(
      {workspace->id, std::vector<uint8_t>(workspace->bytes)});

  FakeBoardDriver driver;
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("exact internal"),
            std::string::npos);
  EXPECT_TRUE(driver.calls.empty());
}

TEST_F(BoardRuntimeTest,
       QualifiedSessionReusesOneInventoryAcrossSerialInvocations) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest seed =
      makeRank16Request(package->getManifest());

  FakeBoardDriver driver;
  driver.waitDelay = std::chrono::milliseconds(1);
  llvm::Expected<QualifiedBoardRuntimeSession> qualified =
      qualifyBoardRuntimeSession(seed.deviceId, /*requiredLogicalRankCount=*/16,
                                 seed.qualification, driver);
  ASSERT_TRUE(static_cast<bool>(qualified))
      << llvm::toString(qualified.takeError());
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "get-device-count"),
      1);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "select-device"), 1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "device-info"),
            1);

  QualifiedBoardRuntimeSession session = std::move(*qualified);
  EXPECT_FALSE(qualified->isUsable());
  for (unsigned iteration = 0; iteration < 2; ++iteration) {
    llvm::Expected<BoardRuntimeInvocationResult> result =
        executeBoardInvocationInSession(
            *package, root, makeRank16Request(package->getManifest()), session);
    ASSERT_TRUE(static_cast<bool>(result))
        << llvm::toString(result.takeError());
    EXPECT_GT(result->launchToCompletionNanoseconds, 0u);
    EXPECT_GT(result->completionObservationResolutionNanoseconds, 0u);
    EXPECT_EQ(driver.observedCompletionObservationPolicy,
              BoardCompletionObservationPolicy::Normal);
    EXPECT_EQ(result->completedStages.back(), BoardRuntimeStage::Cleanup);
  }

  EXPECT_TRUE(session.isUsable());
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "get-device-count"),
      1);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "select-device"), 1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "device-info"),
            1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:grid:main"),
            2);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "wait-current-submission"),
            2);
}

TEST_F(BoardRuntimeTest,
       StartSessionFirstInvocationIsTheOrdinaryNormalOneShotPath) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());

  FakeBoardDriver ordinaryDriver;
  llvm::Expected<BoardRuntimeInvocationResult> ordinary =
      executeBoardInvocation(*package, root,
                             makeRank16Request(package->getManifest()),
                             ordinaryDriver);
  ASSERT_TRUE(static_cast<bool>(ordinary))
      << llvm::toString(ordinary.takeError());

  FakeBoardDriver sessionDriver;
  llvm::Expected<
      std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
      started = executeBoardInvocationAndStartSession(
          *package, root, makeRank16Request(package->getManifest()),
          sessionDriver);
  ASSERT_TRUE(static_cast<bool>(started))
      << llvm::toString(started.takeError());
  EXPECT_TRUE(started->second.isUsable());
  EXPECT_EQ(sessionDriver.observedCompletionObservationPolicy,
            BoardCompletionObservationPolicy::Normal);
  EXPECT_EQ(sessionDriver.calls, ordinaryDriver.calls);
  EXPECT_EQ(started->first.completedStages, ordinary->completedStages);
  EXPECT_EQ(started->first.ranks.size(), ordinary->ranks.size());
  EXPECT_EQ(started->first.outputs.size(), ordinary->outputs.size());
  EXPECT_EQ(std::count(sessionDriver.calls.begin(), sessionDriver.calls.end(),
                       "get-device-count"),
            1);
  EXPECT_EQ(std::count(sessionDriver.calls.begin(), sessionDriver.calls.end(),
                       "select-device"),
            1);
  EXPECT_EQ(std::count(sessionDriver.calls.begin(), sessionDriver.calls.end(),
                       "device-info"),
            1);
}

TEST_F(BoardRuntimeTest, StartSessionAcceptsHighResolutionFirstObservation) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  request.completionObservationPolicy =
      BoardCompletionObservationPolicy::ProfileHighResolution;

  FakeBoardDriver driver;
  driver.waitDelay = std::chrono::milliseconds(2);
  llvm::Expected<
      std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
      started = executeBoardInvocationAndStartSession(
          *package, root, std::move(request), driver);
  ASSERT_TRUE(static_cast<bool>(started))
      << llvm::toString(started.takeError());
  EXPECT_TRUE(started->second.isUsable());
  EXPECT_EQ(driver.observedCompletionObservationPolicy,
            BoardCompletionObservationPolicy::ProfileHighResolution);
  EXPECT_GT(started->first.launchToCompletionNanoseconds, 0u);
  EXPECT_GT(started->first.completionObservationResolutionNanoseconds, 0u);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "get-device-count"),
      1);
}

TEST_F(BoardRuntimeTest,
       StartSessionFailureReturnsNoCapabilityAndPoisonStopsLaterCalls) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());

  FakeBoardDriver firstFailureDriver;
  firstFailureDriver.failOperation = "wait-current-submission";
  llvm::Expected<
      std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
      failed = executeBoardInvocationAndStartSession(
          *package, root, makeRank16Request(package->getManifest()),
          firstFailureDriver);
  ASSERT_FALSE(static_cast<bool>(failed));
  llvm::consumeError(failed.takeError());

  FakeBoardDriver driver;
  llvm::Expected<
      std::pair<BoardRuntimeInvocationResult, QualifiedBoardRuntimeSession>>
      started = executeBoardInvocationAndStartSession(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_TRUE(static_cast<bool>(started))
      << llvm::toString(started.takeError());
  QualifiedBoardRuntimeSession session = std::move(started->second);
  driver.failOperation = "wait-current-submission";
  driver.poisonOnFailure = true;
  llvm::Expected<BoardRuntimeInvocationResult> poisoned =
      executeBoardInvocationInSession(
          *package, root, makeRank16Request(package->getManifest()), session);
  ASSERT_FALSE(static_cast<bool>(poisoned));
  llvm::consumeError(poisoned.takeError());
  EXPECT_FALSE(session.isUsable());
  const size_t callsAfterPoison = driver.calls.size();

  driver.failOperation.clear();
  driver.poisonOnFailure = false;
  llvm::Expected<BoardRuntimeInvocationResult> rejected =
      executeBoardInvocationInSession(
          *package, root, makeRank16Request(package->getManifest()), session);
  ASSERT_FALSE(static_cast<bool>(rejected));
  llvm::consumeError(rejected.takeError());
  EXPECT_EQ(driver.calls.size(), callsAfterPoison);
}

TEST_F(BoardRuntimeTest,
       ProfileCompletionObservationCoversSubmitAndForwardsHighResolution) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  request.completionObservationPolicy =
      BoardCompletionObservationPolicy::ProfileHighResolution;

  FakeBoardDriver driver;
  driver.submitDelay = std::chrono::milliseconds(2);
  driver.waitDelay = std::chrono::milliseconds(2);
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  EXPECT_EQ(driver.observedCompletionObservationPolicy,
            BoardCompletionObservationPolicy::ProfileHighResolution);
  EXPECT_GT(result->completionObservationResolutionNanoseconds, 0u);
  ASSERT_NE(driver.submitEntered, std::chrono::steady_clock::time_point{});
  ASSERT_NE(driver.completionObserved, std::chrono::steady_clock::time_point{});
  const auto providerSubmitThroughCompletion =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          driver.completionObserved - driver.submitEntered)
          .count();
  ASSERT_GT(providerSubmitThroughCompletion, 0);
  EXPECT_GE(result->launchToCompletionNanoseconds,
            static_cast<uint64_t>(providerSubmitThroughCompletion));
  EXPECT_GE(result->launchToCompletionNanoseconds,
            result->completionObservationResolutionNanoseconds);
}

TEST_F(BoardRuntimeTest,
       StreamEventTimingSeparatesDeviceSubmitAndHostEnvelope) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  request.deviceTimingPolicy = BoardDeviceTimingPolicy::StreamEvents;

  FakeBoardDriver driver;
  driver.submitDelay = std::chrono::milliseconds(2);
  driver.waitDelay = std::chrono::milliseconds(2);
  driver.deviceExecutionNanosecondsByWait = {750000};
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  ASSERT_TRUE(result->deviceExecutionNanoseconds.has_value());
  EXPECT_EQ(*result->deviceExecutionNanoseconds, 750000u);
  EXPECT_GE(result->hostSubmitNanoseconds, 2000000u);
  EXPECT_GE(result->launchToCompletionNanoseconds,
            result->hostSubmitNanoseconds);
  ASSERT_EQ(driver.observedDeviceTimingPolicies.size(), 1u);
  EXPECT_EQ(driver.observedDeviceTimingPolicies.front(),
            BoardDeviceTimingPolicy::StreamEvents);
}

TEST_F(BoardRuntimeTest,
       StreamEventTimingRejectsMultiRankPerRankBeforeProviderEffect) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::PerRank);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  request.deviceTimingPolicy = BoardDeviceTimingPolicy::StreamEvents;

  FakeBoardDriver driver;
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  const std::string message = llvm::toString(result.takeError());
  EXPECT_NE(message.find("board runtime preflight failed"), std::string::npos);
  EXPECT_NE(
      message.find(
          "same-stream device timing does not support a multi-rank per-rank "
          "launch"),
      std::string::npos);
  EXPECT_TRUE(driver.calls.empty());
  EXPECT_TRUE(driver.observedDeviceTimingPolicies.empty());
  EXPECT_TRUE(driver.allocatedAddresses.empty());
  EXPECT_EQ(driver.selectedDevice, std::numeric_limits<uint32_t>::max());
}

TEST_F(BoardRuntimeTest, StreamEventTimingSupportsSingleRankPerRank) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package = verify();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  request.deviceTimingPolicy = BoardDeviceTimingPolicy::StreamEvents;

  FakeBoardDriver driver;
  driver.deviceExecutionNanosecondsByWait = {125};
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  ASSERT_TRUE(result->deviceExecutionNanoseconds.has_value());
  EXPECT_EQ(*result->deviceExecutionNanoseconds, 125u);
  ASSERT_EQ(driver.observedDeviceTimingPolicies.size(), 1u);
  EXPECT_EQ(driver.observedDeviceTimingPolicies.front(),
            BoardDeviceTimingPolicy::StreamEvents);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:per-rank:main"),
            1);
}

TEST_F(BoardRuntimeTest, StreamEventTimingSumsEveryKernelPhase) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Cluster);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  request.deviceTimingPolicy = BoardDeviceTimingPolicy::StreamEvents;

  FakeBoardDriver driver;
  driver.deviceExecutionNanosecondsByWait = {250, 750};
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  ASSERT_TRUE(result->deviceExecutionNanoseconds.has_value());
  EXPECT_EQ(*result->deviceExecutionNanoseconds, 1000u);
  ASSERT_EQ(driver.observedDeviceTimingPolicies.size(), 2u);
  EXPECT_TRUE(llvm::all_of(
      driver.observedDeviceTimingPolicies, [](BoardDeviceTimingPolicy policy) {
        return policy == BoardDeviceTimingPolicy::StreamEvents;
      }));
}

TEST_F(BoardRuntimeTest,
       StreamEventTimingPropagatesQuantizedZeroThroughModelSubmission) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Model);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  request.deviceTimingPolicy = BoardDeviceTimingPolicy::StreamEvents;

  FakeBoardDriver driver;
  driver.deviceExecutionNanosecondsByWait = {0};
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  ASSERT_TRUE(result->deviceExecutionNanoseconds.has_value());
  EXPECT_EQ(*result->deviceExecutionNanoseconds, 0u);
  ASSERT_EQ(driver.observedDeviceTimingPolicies.size(), 1u);
  EXPECT_EQ(driver.observedDeviceTimingPolicies.front(),
            BoardDeviceTimingPolicy::StreamEvents);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "submit-model"), 1);
}

TEST_F(BoardRuntimeTest, StreamEventTimingRejectsMissingProviderObservation) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  request.deviceTimingPolicy = BoardDeviceTimingPolicy::StreamEvents;

  FakeBoardDriver driver;
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocation(*package, root, std::move(request), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError())
                .find("omitted requested same-stream device timing"),
            std::string::npos);
}

TEST_F(BoardRuntimeTest, DisabledTimingRejectsUnexpectedProviderObservation) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());

  FakeBoardDriver driver;
  driver.deviceExecutionNanosecondsByWait = {1};
  llvm::Expected<BoardRuntimeInvocationResult> result = executeBoardInvocation(
      *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(
      llvm::toString(result.takeError()).find("device timing when disabled"),
      std::string::npos);
}

TEST_F(BoardRuntimeTest,
       QualifiedSessionRejectsDifferentInvocationIdentityWithoutProviderCall) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package = verifyRank16();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest seed =
      makeRank16Request(package->getManifest());
  FakeBoardDriver driver;
  llvm::Expected<QualifiedBoardRuntimeSession> session =
      qualifyBoardRuntimeSession(seed.deviceId, /*requiredLogicalRankCount=*/16,
                                 seed.qualification, driver);
  ASSERT_TRUE(static_cast<bool>(session))
      << llvm::toString(session.takeError());
  const size_t callsAfterQualification = driver.calls.size();

  BoardRuntimeInvocationRequest mismatched =
      makeRank16Request(package->getManifest());
  mismatched.qualification.runtimeLibraryDigest =
      "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
  llvm::Expected<BoardRuntimeInvocationResult> result =
      executeBoardInvocationInSession(*package, root, std::move(mismatched),
                                      *session);
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("session identity"),
            std::string::npos);
  EXPECT_EQ(driver.calls.size(), callsAfterQualification);
  EXPECT_TRUE(session->isUsable());
}

TEST_F(BoardRuntimeTest,
       PoisonedQualifiedSessionCannotIssueAnotherProviderCall) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package = verifyRank16();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest seed =
      makeRank16Request(package->getManifest());
  FakeBoardDriver driver;
  llvm::Expected<QualifiedBoardRuntimeSession> session =
      qualifyBoardRuntimeSession(seed.deviceId, /*requiredLogicalRankCount=*/16,
                                 seed.qualification, driver);
  ASSERT_TRUE(static_cast<bool>(session))
      << llvm::toString(session.takeError());

  driver.failOperation = "wait-current-submission";
  driver.poisonOnFailure = true;
  llvm::Expected<BoardRuntimeInvocationResult> failed =
      executeBoardInvocationInSession(
          *package, root, makeRank16Request(package->getManifest()), *session);
  ASSERT_FALSE(static_cast<bool>(failed));
  llvm::consumeError(failed.takeError());
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back(), "wait-current-submission");
  EXPECT_FALSE(session->isUsable());
  const size_t callsAfterPoison = driver.calls.size();

  driver.failOperation.clear();
  driver.poisonOnFailure = false;
  llvm::Expected<BoardRuntimeInvocationResult> rejected =
      executeBoardInvocationInSession(
          *package, root, makeRank16Request(package->getManifest()), *session);
  ASSERT_FALSE(static_cast<bool>(rejected));
  bool sawPoison = false;
  llvm::handleAllErrors(rejected.takeError(),
                        [&](const BoardRuntimeError &error) {
                          sawPoison = error.getContextState() ==
                                      BoardRuntimeContextState::Poisoned;
                        });
  EXPECT_TRUE(sawPoison);
  EXPECT_EQ(driver.calls.size(), callsAfterPoison);
}

TEST_F(BoardRuntimeTest, QualifiedSessionRequiresCompleteLogicalTileDomain) {
  using namespace wafer::runtime;
  llvm::Expected<VerifiedPackageManifest> package = verifyRank16();
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  BoardRuntimeInvocationRequest request =
      makeRank16Request(package->getManifest());
  FakeBoardDriver driver;
  driver.unavailableLogicalTile = 15;
  llvm::Expected<QualifiedBoardRuntimeSession> session =
      qualifyBoardRuntimeSession(request.deviceId,
                                 /*requiredLogicalRankCount=*/16,
                                 request.qualification, driver);
  ASSERT_FALSE(static_cast<bool>(session));
  EXPECT_NE(llvm::toString(session.takeError()).find("availability"),
            std::string::npos);
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(), "allocate"),
            driver.calls.end());
}

TEST_F(BoardRuntimeTest,
       ClusterDirectDTEUsesTypedPrepareMainAndChecksStatusBeforeOutputs) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Cluster);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  FakeBoardDriver driver;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(manifest), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "load-module"),
            1);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "resolve-entry"), 2);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:cluster:prepare"),
            1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:cluster:main"),
            1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "wait-current-submission"),
            2);
  ASSERT_EQ(driver.observedDeadlines.size(), 2u);
  EXPECT_EQ(driver.observedDeadlines[0], driver.observedDeadlines[1]);
  const std::vector<std::string> phaseSequence = {
      "resolve-entry",
      "resolve-entry",
      "submit-kernel-phase:cluster:prepare",
      "wait-current-submission",
      "submit-kernel-phase:cluster:main",
      "wait-current-submission"};
  auto call = driver.calls.begin();
  for (const std::string &expected : phaseSequence) {
    call = std::find(call, driver.calls.end(), expected);
    ASSERT_NE(call, driver.calls.end()) << expected;
    ++call;
  }
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "d2h"), 32);
  EXPECT_EQ(
      std::count_if(driver.h2dPayloads.begin(), driver.h2dPayloads.end(),
                    [](const auto &bytes) {
                      return bytes.size() ==
                                 wafer::runtime::kDirectDTEStatusStorageBytes &&
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
       ClusterMainSubmitFailureIsLaunchStageAfterPrepareTerminal) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Cluster);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.failOperation = "submit-kernel-phase:cluster:main";
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  bool sawLaunchFailure = false;
  llvm::handleAllErrors(
      result.takeError(), [&](const wafer::runtime::BoardRuntimeError &error) {
        sawLaunchFailure = true;
        EXPECT_EQ(error.getStage(), wafer::runtime::BoardRuntimeStage::Launch);
        EXPECT_EQ(error.getContextState(),
                  wafer::runtime::BoardRuntimeContextState::Usable);
      });
  EXPECT_TRUE(sawLaunchFailure);
  const std::vector<std::string> phaseSequence = {
      "submit-kernel-phase:cluster:prepare", "wait-current-submission",
      "submit-kernel-phase:cluster:main", "release-submission"};
  auto call = driver.calls.begin();
  for (const std::string &expected : phaseSequence) {
    call = std::find(call, driver.calls.end(), expected);
    ASSERT_NE(call, driver.calls.end()) << expected;
    ++call;
  }
  ASSERT_EQ(driver.observedDeadlines.size(), 1u);
}

TEST_F(BoardRuntimeTest, DirectDTEStatusHandlingIsIndependentOfGridLaunchForm) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Grid, /*withProfiler=*/false,
                   /*directDTEOverride=*/true);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:grid:main"),
            1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "d2h"), 32);
}

TEST_F(BoardRuntimeTest,
       ClusterResolvesEveryPhaseBeforeTheFirstProviderSubmission) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Cluster);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.failOperation = "resolve-entry";
  driver.failIndex = 1;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  bool sawResolveFailure = false;
  llvm::handleAllErrors(
      result.takeError(), [&](const wafer::runtime::BoardRuntimeError &error) {
        sawResolveFailure = true;
        EXPECT_EQ(error.getStage(),
                  wafer::runtime::BoardRuntimeStage::EntryResolve);
      });
  EXPECT_TRUE(sawResolveFailure);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "resolve-entry"), 2);
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(),
                      "submit-kernel-phase:cluster:prepare"),
            driver.calls.end());
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(),
                      "submit-kernel-phase:cluster:main"),
            driver.calls.end());
}

TEST_F(BoardRuntimeTest,
       ClusterLaunchWithoutDirectDTEHasNoTransportStatusReadback) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Cluster, /*withProfiler=*/false,
                   /*directDTEOverride=*/false);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:cluster:prepare"),
            1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:cluster:main"),
            1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "d2h"), 16);
}

TEST_F(BoardRuntimeTest,
       ClusterDirectDTEBadStatusQuarantinesBeforeAnyUserOutputReadback) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Cluster);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.transportStatusOverride =
      static_cast<uint32_t>(wafer::runtime::DirectDTEStatusValue::Pending);
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  bool sawPoison = false;
  llvm::handleAllErrors(
      result.takeError(), [&](const wafer::runtime::BoardRuntimeError &error) {
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
  EXPECT_EQ(
      std::find(driver.calls.begin(), driver.calls.end(), "release-submission"),
      driver.calls.end());
  EXPECT_TRUE(driver.unloadedHandles.empty());
  EXPECT_TRUE(driver.freedAddresses.empty());
}

TEST_F(BoardRuntimeTest,
       ClusterDirectDTEStatusReadbackFailureQuarantinesWithoutCleanup) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Cluster);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
  driver.failOperation = "d2h";
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_GE(driver.calls.size(), 2u);
  EXPECT_EQ(driver.calls[driver.calls.size() - 2], "d2h");
  EXPECT_EQ(driver.calls.back(), "quarantine");
  EXPECT_EQ(
      std::find(driver.calls.begin(), driver.calls.end(), "release-submission"),
      driver.calls.end());
  EXPECT_TRUE(driver.unloadedHandles.empty());
  EXPECT_TRUE(driver.freedAddresses.empty());
}

TEST_F(BoardRuntimeTest,
       ClusterDirectDTEOutputReadbackFailureQuarantinesWithoutCleanup) {
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Cluster);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  FakeBoardDriver driver;
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
  EXPECT_EQ(
      std::find(driver.calls.begin(), driver.calls.end(), "release-submission"),
      driver.calls.end());
  EXPECT_TRUE(driver.unloadedHandles.empty());
  EXPECT_TRUE(driver.freedAddresses.empty());
}

TEST_F(BoardRuntimeTest, ModelLoadsCompleteGraphAndSubmitsTypedTensorDomain) {
  createRankModules(16);
  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      verifyRank16(TestLaunchContractCase::Model);
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  FakeBoardDriver driver;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(manifest), driver);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());

  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "load-graph"),
            1);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "submit-model"), 1);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(), "load-module"),
            0);
  EXPECT_EQ(
      std::count(driver.calls.begin(), driver.calls.end(), "resolve-entry"), 0);
  EXPECT_EQ(std::count(driver.calls.begin(), driver.calls.end(),
                       "submit-kernel-phase:per-rank:main"),
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
    auto resource =
        llvm::find_if(manifest.resources, [&](const auto &candidate) {
          return candidate.id == output.resource;
        });
    ASSERT_NE(resource, manifest.resources.end());
    for (size_t byte = 0; byte < output.bytes.size(); ++byte)
      EXPECT_EQ(output.bytes[byte],
                rankInputByte(resource->logicalRank, byte) ^
                    static_cast<uint8_t>(resource->logicalRank));
  }
}

TEST_F(BoardRuntimeTest, MultiTilePoisonedProviderFailureIsTheFinalDriverCall) {
  struct Scenario {
    TestLaunchContractCase launchCase;
    llvm::StringLiteral operation;
  };
  const Scenario scenarios[] = {
      {TestLaunchContractCase::Grid, "submit-kernel-phase:grid:main"},
      {TestLaunchContractCase::Model, "load-graph"},
      {TestLaunchContractCase::Model, "submit-model"},
  };
  createRankModules(16);
  for (const Scenario &scenario : scenarios) {
    SCOPED_TRACE(scenario.operation.str());
    llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
        verifyRank16(scenario.launchCase);
    ASSERT_TRUE(static_cast<bool>(package))
        << llvm::toString(package.takeError());
    FakeBoardDriver driver;
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
    EXPECT_EQ(
        std::find(driver.calls.begin(), driver.calls.end(), "unload-graph"),
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
  using wafer::runtime::PackageManifest;
  createRankModules(16);

  auto expectRejected = [&](PackageManifest manifest) {
    llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
        wafer::runtime::verifyPackageManifest(std::move(manifest), root);
    EXPECT_FALSE(static_cast<bool>(package));
    if (!package)
      llvm::consumeError(package.takeError());
  };

  PackageManifest zeroExtent =
      makeRank16Manifest(TestLaunchContractCase::Model);
  for (auto &resource : zeroExtent.resources)
    if (resource.role == wafer::runtime::PackageResourceRole::UserInput)
      resource.type.shape = {0};
  expectRejected(std::move(zeroExtent));

  PackageManifest byteMismatch =
      makeRank16Manifest(TestLaunchContractCase::Model);
  for (auto &resource : byteMismatch.resources)
    if (resource.role == wafer::runtime::PackageResourceRole::UserInput)
      resource.bytes -= sizeof(float);
  expectRejected(std::move(byteMismatch));

  PackageManifest insufficientAlignment =
      makeRank16Manifest(TestLaunchContractCase::Model);
  for (auto &resource : insufficientAlignment.resources)
    resource.alignment = 1;
  expectRejected(std::move(insufficientAlignment));

  PackageManifest longSymbol =
      makeRank16Manifest(TestLaunchContractCase::Model);
  for (auto &module : longSymbol.modules)
    module.exports.front().symbol.assign(128, 'm');
  expectRejected(std::move(longSymbol));

  PackageManifest nulSymbol = makeRank16Manifest(TestLaunchContractCase::Model);
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
      {"submit-kernel-phase:per-rank:main",
       wafer::runtime::BoardRuntimeStage::Launch, false},
      {"wait-current-submission", wafer::runtime::BoardRuntimeStage::Completion,
       true},
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
  driver.failOperation = "wait-current-submission";
  driver.poisonOnFailure = true;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(
          *package, root, makeRank16Request(package->getManifest()), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back(), "wait-current-submission");
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
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(),
                      "submit-kernel-phase:per-rank:main"),
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
  driver.failOperation = "wait-current-submission";
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
  EXPECT_EQ(driver.calls.back(), "wait-current-submission");
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
  driver.failOperation = "submit-kernel-phase:per-rank:main";
  driver.poisonOnFailure = true;
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, makeRequest(), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back(), "submit-kernel-phase:per-rank:main");
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
