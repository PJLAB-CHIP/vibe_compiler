//===- HostRuntime.cpp - Wafer host runtime package boundary --------------===//

#include "Wafer/Runtime/HostRuntime.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <dlfcn.h>

namespace wafer {
namespace runtime {
namespace {

llvm::Error makeError(llvm::Twine message) {
  return llvm::createStringError(message);
}

llvm::Expected<std::string> requireString(const llvm::json::Object &object,
                                          llvm::StringRef field) {
  if (std::optional<llvm::StringRef> value = object.getString(field))
    return value->str();
  return makeError("package metadata field '" + field + "' must be a string");
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef field) {
  const llvm::json::Object *nested = object.getObject(field);
  if (nested != nullptr)
    return nested;
  return makeError("package metadata field '" + field + "' must be an object");
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef field) {
  const llvm::json::Array *array = object.getArray(field);
  if (array != nullptr)
    return array;
  return makeError("package metadata field '" + field + "' must be an array");
}

llvm::Expected<std::vector<std::string>>
parseStringArray(const llvm::json::Object &object, llvm::StringRef field) {
  const llvm::json::Array *array = object.getArray(field);
  if (array == nullptr)
    return std::vector<std::string>();

  std::vector<std::string> values;
  values.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    if (std::optional<llvm::StringRef> string = value.getAsString()) {
      values.push_back(string->str());
      continue;
    }
    return makeError("package metadata array '" + field +
                     "' must contain only strings");
  }
  return values;
}

llvm::Expected<RuntimeModule> parseModule(const llvm::json::Value &value) {
  const llvm::json::Object *object = value.getAsObject();
  if (object == nullptr)
    return makeError("module descriptor must be an object");

  RuntimeModule module;
  if (llvm::Expected<std::string> name = requireString(*object, "name"))
    module.name = *name;
  else
    return name.takeError();
  if (llvm::Expected<std::string> format = requireString(*object, "format"))
    module.format = *format;
  else
    return format.takeError();
  if (llvm::Expected<std::string> path = requireString(*object, "path"))
    module.path = *path;
  else
    return path.takeError();
  return module;
}

llvm::Expected<RuntimeEntrypoint>
parseEntrypoint(const llvm::json::Value &value) {
  const llvm::json::Object *object = value.getAsObject();
  if (object == nullptr)
    return makeError("entrypoint descriptor must be an object");

  RuntimeEntrypoint entrypoint;
  if (llvm::Expected<std::string> name = requireString(*object, "name"))
    entrypoint.name = *name;
  else
    return name.takeError();

  llvm::Expected<std::string> executorText = requireString(*object, "executor");
  if (!executorText)
    return executorText.takeError();
  llvm::Expected<EntrypointExecutor> executor =
      parseEntrypointExecutor(*executorText);
  if (!executor)
    return executor.takeError();
  entrypoint.executor = *executor;

  if (std::optional<llvm::StringRef> module = object->getString("module"))
    entrypoint.module = module->str();
  if (std::optional<llvm::StringRef> function = object->getString("function"))
    entrypoint.function = function->str();
  if (std::optional<llvm::StringRef> modSymbol =
          object->getString("mod_symbol"))
    entrypoint.modSymbol = modSymbol->str();
  if (const llvm::json::Object *bpmDescriptor =
          object->getObject("bpm_descriptor")) {
    if (std::optional<llvm::StringRef> state =
            bpmDescriptor->getString("state"))
      entrypoint.bpmState = state->str();
  }

  llvm::Expected<std::vector<std::string>> bindingOrder =
      parseStringArray(*object, "binding_order");
  if (!bindingOrder)
    return bindingOrder.takeError();
  entrypoint.bindingOrder = std::move(*bindingOrder);
  return entrypoint;
}

std::string joinStrings(llvm::ArrayRef<std::string> values) {
  std::string result;
  llvm::raw_string_ostream os(result);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << values[i];
  }
  return os.str();
}

} // namespace

const RuntimeModule *
RuntimePackage::findModule(llvm::StringRef moduleName) const {
  auto it = std::find_if(
      modules.begin(), modules.end(),
      [&](const RuntimeModule &module) { return module.name == moduleName; });
  return it == modules.end() ? nullptr : &*it;
}

const RuntimeEntrypoint *
RuntimePackage::findEntrypoint(llvm::StringRef entrypointName) const {
  auto it = std::find_if(entrypoints.begin(), entrypoints.end(),
                         [&](const RuntimeEntrypoint &entrypoint) {
                           return entrypoint.name == entrypointName;
                         });
  return it == entrypoints.end() ? nullptr : &*it;
}

llvm::Expected<EntrypointExecutor>
parseEntrypointExecutor(llvm::StringRef executor) {
  if (executor == "tx.model")
    return EntrypointExecutor::TxModel;
  if (executor == "tx.graph")
    return EntrypointExecutor::TxGraph;
  if (executor == "tx.module")
    return EntrypointExecutor::TxModule;
  if (executor == "tx.cluster")
    return EntrypointExecutor::TxCluster;
  if (executor == "legacy.tsm")
    return EntrypointExecutor::LegacyPackage;
  return makeError("unsupported entrypoint executor: " + executor);
}

llvm::StringRef stringifyEntrypointExecutor(EntrypointExecutor executor) {
  switch (executor) {
  case EntrypointExecutor::TxModel:
    return "tx.model";
  case EntrypointExecutor::TxGraph:
    return "tx.graph";
  case EntrypointExecutor::TxModule:
    return "tx.module";
  case EntrypointExecutor::TxCluster:
    return "tx.cluster";
  case EntrypointExecutor::LegacyPackage:
    return "legacy.tsm";
  }
  llvm_unreachable("unknown entrypoint executor");
}

llvm::Expected<RuntimePackage>
parseRuntimePackageMetadata(llvm::StringRef metadata) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(metadata);
  if (!parsed)
    return parsed.takeError();

  const llvm::json::Object *root = parsed->getAsObject();
  if (root == nullptr)
    return makeError("package metadata must be a JSON object");

  RuntimePackage package;
  if (llvm::Expected<std::string> name = requireString(*root, "name"))
    package.name = *name;
  else
    return name.takeError();

  llvm::Expected<const llvm::json::Object *> runtime =
      requireObject(*root, "runtime");
  if (!runtime)
    return runtime.takeError();
  if (llvm::Expected<std::string> mode = requireString(**runtime, "mode"))
    package.runtimeMode = *mode;
  else
    return mode.takeError();
  if (llvm::Expected<std::string> completion =
          requireString(**runtime, "completion_source"))
    package.completionSource = *completion;
  else
    return completion.takeError();

  llvm::Expected<const llvm::json::Array *> modules =
      requireArray(*root, "modules");
  if (!modules)
    return modules.takeError();
  for (const llvm::json::Value &value : **modules) {
    llvm::Expected<RuntimeModule> module = parseModule(value);
    if (!module)
      return module.takeError();
    package.modules.push_back(std::move(*module));
  }

  llvm::Expected<const llvm::json::Array *> entrypoints =
      requireArray(*root, "entrypoints");
  if (!entrypoints)
    return entrypoints.takeError();
  for (const llvm::json::Value &value : **entrypoints) {
    llvm::Expected<RuntimeEntrypoint> entrypoint = parseEntrypoint(value);
    if (!entrypoint)
      return entrypoint.takeError();
    package.entrypoints.push_back(std::move(*entrypoint));
  }
  return package;
}

llvm::Expected<RuntimePackage>
loadRuntimePackageMetadataFile(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read package metadata: " + path);
  return parseRuntimePackageMetadata((*buffer)->getBuffer());
}

std::vector<std::string>
requiredTxRuntimeSymbols(const RuntimeEntrypoint *entrypoint) {
  std::vector<std::string> symbols = {"txSetDevice", "txMalloc", "txFree",
                                      "txMemcpy", "txStreamSynchronize"};
  if (entrypoint == nullptr)
    return symbols;

  switch (entrypoint->executor) {
  case EntrypointExecutor::TxModule:
    symbols.push_back("txModuleLoad");
    symbols.push_back("txModuleGetFunction");
    symbols.push_back("txLaunchKernel");
    break;
  case EntrypointExecutor::TxCluster:
    symbols.push_back("txModuleLoad");
    symbols.push_back("txModuleGetFunction");
    symbols.push_back("txLaunchClusterKernel");
    break;
  case EntrypointExecutor::TxModel:
    symbols.push_back("txLaunchModel");
    symbols.push_back("txLaunchModelSync");
    break;
  case EntrypointExecutor::TxGraph:
    symbols.push_back("txLoadGraph");
    symbols.push_back("txUnloadGraph");
    break;
  case EntrypointExecutor::LegacyPackage:
    break;
  }
  return symbols;
}

std::vector<std::string>
missingRequiredTxRuntimeSymbols(const RuntimeEntrypoint &entrypoint,
                                llvm::ArrayRef<std::string> available) {
  llvm::StringSet<> availableSet;
  for (llvm::StringRef symbol : available)
    availableSet.insert(symbol);

  std::vector<std::string> missing;
  for (const std::string &symbol : requiredTxRuntimeSymbols(&entrypoint)) {
    if (!availableSet.contains(symbol))
      missing.push_back(symbol);
  }
  return missing;
}

TxRuntimeLibrary::TxRuntimeLibrary(std::string path, void *handle)
    : path(std::move(path)), handle(handle) {}

TxRuntimeLibrary::TxRuntimeLibrary(TxRuntimeLibrary &&other) noexcept
    : path(std::move(other.path)), handle(other.handle) {
  other.handle = nullptr;
}

TxRuntimeLibrary &
TxRuntimeLibrary::operator=(TxRuntimeLibrary &&other) noexcept {
  if (this == &other)
    return *this;
  if (handle != nullptr)
    dlclose(handle);
  path = std::move(other.path);
  handle = other.handle;
  other.handle = nullptr;
  return *this;
}

TxRuntimeLibrary::~TxRuntimeLibrary() {
  if (handle != nullptr)
    dlclose(handle);
}

llvm::Expected<TxRuntimeLibrary> TxRuntimeLibrary::load(llvm::StringRef path) {
  void *handle = dlopen(path.str().c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char *message = dlerror();
    return makeError("failed to load tx runtime library '" + path + "': " +
                     (message == nullptr ? "unknown dlopen error" : message));
  }
  return TxRuntimeLibrary(path.str(), handle);
}

bool TxRuntimeLibrary::hasSymbol(llvm::StringRef symbol) const {
  if (handle == nullptr)
    return false;
  return dlsym(handle, symbol.str().c_str()) != nullptr;
}

llvm::Error TxRuntimeLibrary::validateRequiredSymbols(
    const RuntimeEntrypoint &entrypoint) const {
  std::vector<std::string> missing;
  for (const std::string &symbol : requiredTxRuntimeSymbols(&entrypoint)) {
    if (!hasSymbol(symbol))
      missing.push_back(symbol);
  }
  if (missing.empty())
    return llvm::Error::success();

  return makeError("tx runtime library is missing required symbol(s) for " +
                   stringifyEntrypointExecutor(entrypoint.executor) + ": " +
                   joinStrings(missing));
}

} // namespace runtime
} // namespace wafer
