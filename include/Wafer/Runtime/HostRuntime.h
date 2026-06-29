//===- HostRuntime.h - Wafer host runtime -----------------------*- C++ -*-===//

#ifndef WAFER_RUNTIME_HOSTRUNTIME_H
#define WAFER_RUNTIME_HOSTRUNTIME_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace wafer {
namespace runtime {

enum class EntrypointExecutor {
  TxModel,
  TxGraph,
  TxModule,
  TxCluster,
  LegacyPackage,
};

struct RuntimeModule {
  std::string name;
  std::string format;
  std::string path;
};

struct RuntimeEntrypoint {
  std::string name;
  EntrypointExecutor executor = EntrypointExecutor::TxModule;
  std::string module;
  std::string function;
  std::string bpmState;
  std::string modSymbol;
  std::vector<std::string> bindingOrder;
};

struct RuntimeBinding {
  std::string name;
  std::string role;
  std::uint64_t bytes = 0;
  std::vector<std::string> lifecycle;
  bool readOnly = false;
  bool hostVisible = false;
  std::string source;
};

struct RuntimeLaunchArg {
  std::size_t index = 0;
  std::string bindingName;
  std::string role;
  std::uint64_t bytes = 0;
};

struct RuntimeSession {
  std::string packageName;
  std::string runtimeMode;
  std::string completionSource;
  std::vector<RuntimeBinding> bindings;
  std::vector<RuntimeLaunchArg> launchArgs;
  std::string entrypointName;
  EntrypointExecutor executor = EntrypointExecutor::TxModule;
  std::string launchApi;
  bool hasModule = false;
  std::string moduleName;
  std::string moduleFormat;
  std::string modulePath;
  std::string function;
  std::string bpmState;
  std::string modSymbol;
  std::uint64_t argBytes = 0;
};

struct RuntimePackage {
  std::string name;
  std::string modelId;
  std::string modelAbi;
  std::string runtimeMode;
  std::string completionSource;
  std::vector<RuntimeBinding> bindings;
  std::vector<RuntimeModule> modules;
  std::vector<RuntimeEntrypoint> entrypoints;

  const RuntimeModule *findModule(llvm::StringRef moduleName) const;
  const RuntimeEntrypoint *findEntrypoint(llvm::StringRef entrypointName) const;
};

llvm::Expected<RuntimePackage>
parseRuntimePackageMetadata(llvm::StringRef metadata);

llvm::Expected<RuntimePackage>
loadRuntimePackageMetadataFile(llvm::StringRef path);

llvm::Expected<EntrypointExecutor>
parseEntrypointExecutor(llvm::StringRef executor);

llvm::StringRef stringifyEntrypointExecutor(EntrypointExecutor executor);

llvm::StringRef launchApiForEntrypointExecutor(EntrypointExecutor executor);

llvm::Expected<RuntimeSession>
buildRuntimeSession(const RuntimePackage &package,
                    const RuntimeEntrypoint &entrypoint);

llvm::Error validateTxHostRuntimePackage(const RuntimePackage &package);

std::vector<std::string>
requiredTxRuntimeSymbols(const RuntimeEntrypoint *entrypoint = nullptr);

std::vector<std::string>
missingRequiredTxRuntimeSymbols(const RuntimeEntrypoint &entrypoint,
                                llvm::ArrayRef<std::string> available);

class TxRuntimeLibrary {
public:
  TxRuntimeLibrary(TxRuntimeLibrary &&other) noexcept;
  TxRuntimeLibrary &operator=(TxRuntimeLibrary &&other) noexcept;
  TxRuntimeLibrary(const TxRuntimeLibrary &) = delete;
  TxRuntimeLibrary &operator=(const TxRuntimeLibrary &) = delete;
  ~TxRuntimeLibrary();

  static llvm::Expected<TxRuntimeLibrary> load(llvm::StringRef path);

  bool hasSymbol(llvm::StringRef symbol) const;
  llvm::Error
  validateRequiredSymbols(const RuntimeEntrypoint &entrypoint) const;

private:
  TxRuntimeLibrary(std::string path, void *handle);

  std::string path;
  void *handle = nullptr;
};

} // namespace runtime
} // namespace wafer

#endif // WAFER_RUNTIME_HOSTRUNTIME_H
