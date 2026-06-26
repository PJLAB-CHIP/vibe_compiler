#include "Wafer/Runtime/HostRuntime.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

#ifndef WAFER_FAKE_TX_RUNTIME_PATH
#error "WAFER_FAKE_TX_RUNTIME_PATH must be defined by CMake"
#endif

namespace {

std::string packageJson() {
  return R"json({
    "schema_version": 2,
    "name": "model_package_sample",
    "runtime": {
      "mode": "tx",
      "completion_source": "runtime_stream_wait"
    },
    "model": {
      "id": "model_package_sample",
      "abi": "wafer-cabi-v0",
      "interface": {
        "inputs": [],
        "outputs": [],
        "parameters": [],
        "workspace": [],
        "resident_constants": []
      },
      "resources": {
        "spm_bytes": 0,
        "ddr_external_input_bytes": 0,
        "ddr_external_output_bytes": 0,
        "workspace_bytes": 0,
        "resident_constant_bytes": 0
      }
    },
    "modules": [
      {
        "name": "kernel",
        "format": "tx.kcore",
        "path": "model_package.so"
      }
    ],
    "entrypoints": [
      {
        "name": "debug_kernel",
        "executor": "tx.module",
        "module": "kernel",
        "function": "model_package_sample_abi",
        "binding_order": []
      }
    ]
  })json";
}

TEST(HostRuntimeTest, ParsesPackageEntrypointWithoutRuntimeHandles) {
  llvm::Expected<wafer::runtime::RuntimePackage> package =
      wafer::runtime::parseRuntimePackageMetadata(packageJson());
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());

  EXPECT_EQ(package->name, "model_package_sample");
  EXPECT_EQ(package->runtimeMode, "tx");
  EXPECT_EQ(package->completionSource, "runtime_stream_wait");
  ASSERT_EQ(package->modules.size(), 1u);
  EXPECT_EQ(package->modules[0].name, "kernel");
  EXPECT_EQ(package->modules[0].format, "tx.kcore");
  EXPECT_EQ(package->modules[0].path, "model_package.so");

  const wafer::runtime::RuntimeEntrypoint *entrypoint =
      package->findEntrypoint("debug_kernel");
  ASSERT_NE(entrypoint, nullptr);
  EXPECT_EQ(entrypoint->executor, wafer::runtime::EntrypointExecutor::TxModule);
  EXPECT_EQ(entrypoint->module, "kernel");
  EXPECT_EQ(entrypoint->function, "model_package_sample_abi");
}

TEST(HostRuntimeTest, ReportsEntrypointSpecificMissingSymbols) {
  llvm::Expected<wafer::runtime::RuntimePackage> package =
      wafer::runtime::parseRuntimePackageMetadata(packageJson());
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::RuntimeEntrypoint *entrypoint =
      package->findEntrypoint("debug_kernel");
  ASSERT_NE(entrypoint, nullptr);

  std::vector<std::string> available = {"txSetDevice",
                                        "txMalloc",
                                        "txFree",
                                        "txMemcpy",
                                        "txStreamSynchronize",
                                        "txModuleLoad",
                                        "txModuleGetFunction"};
  std::vector<std::string> missing =
      wafer::runtime::missingRequiredTxRuntimeSymbols(*entrypoint, available);

  ASSERT_EQ(missing.size(), 1u);
  EXPECT_EQ(missing[0], "txLaunchKernel");
}

TEST(HostRuntimeTest, LoadsRuntimeLibraryAndBindsSelectedExecutorSymbols) {
  llvm::Expected<wafer::runtime::RuntimePackage> package =
      wafer::runtime::parseRuntimePackageMetadata(packageJson());
  ASSERT_TRUE(static_cast<bool>(package))
      << llvm::toString(package.takeError());
  const wafer::runtime::RuntimeEntrypoint *entrypoint =
      package->findEntrypoint("debug_kernel");
  ASSERT_NE(entrypoint, nullptr);

  llvm::Expected<wafer::runtime::TxRuntimeLibrary> library =
      wafer::runtime::TxRuntimeLibrary::load(WAFER_FAKE_TX_RUNTIME_PATH);
  ASSERT_TRUE(static_cast<bool>(library))
      << llvm::toString(library.takeError());

  llvm::Error error = library->validateRequiredSymbols(*entrypoint);
  EXPECT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
}

} // namespace
