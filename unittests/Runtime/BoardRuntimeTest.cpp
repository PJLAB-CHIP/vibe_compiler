//===- BoardRuntimeTest.cpp - Board provider lifecycle tests ------------===//

#include "Wafer/Runtime/BoardRuntime.h"

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
#include <limits>
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
  wafer::runtime::BoardRuntimeContextState getContextState() const override {
    return contextState;
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
    return info;
  }

  llvm::Expected<wafer::runtime::BoardDeviceMemory>
  allocate(uint64_t bytes, uint64_t alignment) override {
    calls.push_back("allocate");
    if (failOperation == "allocate" && allocations.size() == failIndex)
      return injectedFailure();
    nextAddress = (nextAddress + alignment - 1) & ~(alignment - 1);
    uintptr_t address = nextAddress;
    nextAddress += static_cast<uintptr_t>(bytes + alignment);
    allocations.push_back({address, std::vector<uint8_t>(bytes)});
    return wafer::runtime::BoardDeviceMemory{address};
  }

  llvm::Error free(wafer::runtime::BoardDeviceMemory memory) override {
    calls.push_back("free:" + std::to_string(memory.value));
    if (failOperation == "free")
      return injectedFailure();
    auto iterator = find(memory.value);
    if (iterator == allocations.end())
      return injected("unknown-free");
    allocations.erase(iterator);
    return llvm::Error::success();
  }

  llvm::Error copyHostToDevice(wafer::runtime::BoardDeviceMemory destination,
                               llvm::ArrayRef<uint8_t> source) override {
    calls.push_back("h2d");
    if (failOperation == "h2d")
      return injectedFailure();
    auto iterator = find(destination.value);
    if (iterator == allocations.end() ||
        iterator->bytes.size() != source.size())
      return injected("invalid-h2d");
    std::copy(source.begin(), source.end(), iterator->bytes.begin());
    return llvm::Error::success();
  }

  llvm::Error
  copyDeviceToHost(llvm::MutableArrayRef<uint8_t> destination,
                   wafer::runtime::BoardDeviceMemory source) override {
    calls.push_back("d2h");
    if (failOperation == "d2h")
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
    if (failOperation == "load-module")
      return injectedFailure();
    if (moduleBytes.empty())
      return injected("empty-module");
    moduleLive = true;
    return wafer::runtime::BoardModuleHandle{7};
  }

  llvm::Error unloadModule(wafer::runtime::BoardModuleHandle module) override {
    calls.push_back("unload-module");
    if (failOperation == "unload-module")
      return injectedFailure();
    if (!moduleLive || module.value != 7)
      return injected("invalid-module");
    moduleLive = false;
    return llvm::Error::success();
  }

  llvm::Expected<wafer::runtime::BoardFunctionHandle>
  resolveEntry(wafer::runtime::BoardModuleHandle module,
               llvm::StringRef symbol) override {
    calls.push_back("resolve-entry");
    if (failOperation == "resolve-entry")
      return injectedFailure();
    if (!moduleLive || module.value != 7 || symbol != "main")
      return injected("invalid-entry");
    return wafer::runtime::BoardFunctionHandle{9};
  }

  llvm::Error launch(wafer::runtime::BoardFunctionHandle function,
                     llvm::ArrayRef<uint64_t> arguments) override {
    calls.push_back("launch");
    if (failOperation == "launch")
      return injectedFailure();
    if (function.value != 9 || arguments.size() != 3)
      return injected("invalid-launch");
    auto input = find(arguments[0]);
    auto output = find(arguments[1]);
    if (input == allocations.end() || output == allocations.end() ||
        input->bytes.size() != output->bytes.size())
      return injected("invalid-launch-buffers");
    output->bytes = input->bytes;
    return llvm::Error::success();
  }

  llvm::Error synchronize() override {
    calls.push_back("synchronize");
    if (failOperation == "synchronize")
      return injectedFailure();
    return llvm::Error::success();
  }

  std::string failOperation;
  bool poisonOnFailure = false;
  size_t failIndex = 0;
  std::vector<std::string> calls;
  uint32_t selectedDevice = std::numeric_limits<uint32_t>::max();
  uint64_t freeMemoryBytes = 128ULL * 1024 * 1024;
  uint64_t totalMemoryBytes = 256ULL * 1024 * 1024;
  std::string runtimeLibraryDigest =
      "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

private:
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
  bool moduleLive = false;
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
                             target.kernelRuntimeABI, target.moduleFormat);
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
        {ModuleId(0), 0, "modules/rank_00000.so",
         "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true),
         target.moduleFormat.str()}};
    manifest.entries = {{EntryId(0),
                         0,
                         ModuleId(0),
                         "main",
                         {{0, ResourceId(0), PackageAccessMode::ReadOnly},
                          {1, ResourceId(1), PackageAccessMode::WriteOnly},
                          {2, ResourceId(2), PackageAccessMode::ReadWrite}},
                         CompletionId(0)}};
    manifest.completions = {{CompletionId(0), 0, "entry_return"}};
    return manifest;
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
  auto unload =
      std::find(driver.calls.begin(), driver.calls.end(), "unload-module");
  ASSERT_NE(unload, driver.calls.end());
  ASSERT_NE(unload + 1, driver.calls.end());
  EXPECT_TRUE((unload + 1)->find("free:") == 0);
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
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(), "launch"),
            driver.calls.end());
  EXPECT_NE(
      std::find(driver.calls.begin(), driver.calls.end(), "unload-module"),
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
  driver.failOperation = "synchronize";
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
  EXPECT_EQ(driver.calls.back(), "synchronize");
  EXPECT_EQ(std::find(driver.calls.begin(), driver.calls.end(), "d2h"),
            driver.calls.end());
  EXPECT_EQ(
      std::find(driver.calls.begin(), driver.calls.end(), "unload-module"),
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
  driver.failOperation = "launch";
  driver.poisonOnFailure = true;
  llvm::Expected<wafer::runtime::BoardRuntimeResult> result =
      wafer::runtime::executeBoardEntry(*package, root, makeRequest(), driver);
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::consumeError(result.takeError());
  ASSERT_FALSE(driver.calls.empty());
  EXPECT_EQ(driver.calls.back(), "launch");
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
  EXPECT_EQ(driver.calls.back(), "unload-module");
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
