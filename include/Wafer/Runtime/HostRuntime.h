//===- HostRuntime.h - Wafer host runtime -----------------------*- C++ -*-===//

#ifndef WAFER_RUNTIME_HOSTRUNTIME_H
#define WAFER_RUNTIME_HOSTRUNTIME_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

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

struct RuntimePackage {
  std::string name;
  std::string runtimeMode;
  std::string completionSource;
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
