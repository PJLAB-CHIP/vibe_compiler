//===- HostRuntime.cpp - Wafer host runtime package boundary --------------===//

#include "Wafer/Runtime/HostRuntime.h"

#include "llvm/ADT/STLExtras.h"
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

llvm::Expected<std::uint64_t>
requirePositiveInteger(const llvm::json::Object &object,
                       llvm::StringRef field) {
  std::optional<int64_t> value = object.getInteger(field);
  if (!value || *value <= 0)
    return makeError("package metadata field '" + field +
                     "' must be a positive integer");
  return static_cast<std::uint64_t>(*value);
}

llvm::Expected<bool> requireBool(const llvm::json::Object &object,
                                 llvm::StringRef field) {
  std::optional<bool> value = object.getBoolean(field);
  if (value)
    return *value;
  return makeError("package metadata field '" + field + "' must be a boolean");
}

std::vector<std::string>
lifecycle(std::initializer_list<llvm::StringRef> items) {
  std::vector<std::string> result;
  result.reserve(items.size());
  for (llvm::StringRef item : items)
    result.push_back(item.str());
  return result;
}

llvm::Expected<RuntimeBinding>
parseExternalBinding(const llvm::json::Value &value, llvm::StringRef role,
                     std::vector<std::string> bindingLifecycle) {
  const llvm::json::Object *object = value.getAsObject();
  if (object == nullptr)
    return makeError("model interface " + role + " binding must be an object");

  RuntimeBinding binding;
  binding.role = role.str();
  binding.lifecycle = std::move(bindingLifecycle);
  if (llvm::Expected<std::string> name = requireString(*object, "name"))
    binding.name = *name;
  else
    return name.takeError();

  llvm::Expected<const llvm::json::Object *> bindingObject =
      requireObject(*object, "binding");
  if (!bindingObject)
    return bindingObject.takeError();
  if (llvm::Expected<std::uint64_t> bytes =
          requirePositiveInteger(**bindingObject, "bytes"))
    binding.bytes = *bytes;
  else
    return bytes.takeError();
  if (llvm::Expected<bool> readOnly = requireBool(**bindingObject, "read_only"))
    binding.readOnly = *readOnly;
  else
    return readOnly.takeError();
  if (llvm::Expected<bool> hostVisible =
          requireBool(**bindingObject, "host_visible"))
    binding.hostVisible = *hostVisible;
  else
    return hostVisible.takeError();
  return binding;
}

llvm::Expected<RuntimeBinding>
parseWorkspaceBinding(const llvm::json::Value &value) {
  const llvm::json::Object *object = value.getAsObject();
  if (object == nullptr)
    return makeError("model interface workspace binding must be an object");

  RuntimeBinding binding;
  binding.role = "workspace";
  binding.lifecycle = lifecycle({"allocate", "query", "bind"});
  if (llvm::Expected<std::string> name = requireString(*object, "name"))
    binding.name = *name;
  else
    return name.takeError();
  if (llvm::Expected<std::uint64_t> bytes =
          requirePositiveInteger(*object, "bytes"))
    binding.bytes = *bytes;
  else
    return bytes.takeError();
  binding.readOnly = false;
  binding.hostVisible = false;
  return binding;
}

llvm::Expected<RuntimeBinding>
parseResidentConstantBinding(const llvm::json::Value &value) {
  const llvm::json::Object *object = value.getAsObject();
  if (object == nullptr)
    return makeError(
        "model interface resident constant binding must be an object");

  RuntimeBinding binding;
  binding.role = "resident_constant";
  binding.lifecycle = lifecycle({"allocate", "query", "bind", "copy_h2d"});
  if (llvm::Expected<std::string> name = requireString(*object, "name"))
    binding.name = *name;
  else
    return name.takeError();
  if (llvm::Expected<std::uint64_t> bytes =
          requirePositiveInteger(*object, "bytes"))
    binding.bytes = *bytes;
  else
    return bytes.takeError();

  llvm::Expected<std::string> source = requireString(*object, "source");
  if (!source)
    return source.takeError();
  binding.source = *source;
  if (*source == "launch_input" || *source == "parameter") {
    llvm::Expected<std::string> sourceBinding =
        requireString(*object, "source_binding");
    if (!sourceBinding)
      return sourceBinding.takeError();
    binding.source += ":";
    binding.source += *sourceBinding;
  }
  binding.readOnly = true;
  binding.hostVisible = false;
  return binding;
}

template <typename ParseFn>
llvm::Error
parseBindingArray(const llvm::json::Object &object, llvm::StringRef field,
                  std::vector<RuntimeBinding> &bindings, ParseFn parseBinding) {
  llvm::Expected<const llvm::json::Array *> array = requireArray(object, field);
  if (!array)
    return array.takeError();
  for (const llvm::json::Value &value : **array) {
    llvm::Expected<RuntimeBinding> binding = parseBinding(value);
    if (!binding)
      return binding.takeError();
    bindings.push_back(std::move(*binding));
  }
  return llvm::Error::success();
}

llvm::Error parseModelInterface(const llvm::json::Object &root,
                                RuntimePackage &package) {
  llvm::Expected<const llvm::json::Object *> model =
      requireObject(root, "model");
  if (!model)
    return model.takeError();
  if (llvm::Expected<std::string> modelId = requireString(**model, "id"))
    package.modelId = *modelId;
  else
    return modelId.takeError();
  if (llvm::Expected<std::string> modelAbi = requireString(**model, "abi"))
    package.modelAbi = *modelAbi;
  else
    return modelAbi.takeError();

  llvm::Expected<const llvm::json::Object *> interface =
      requireObject(**model, "interface");
  if (!interface)
    return interface.takeError();

  if (llvm::Error error = parseBindingArray(
          **interface, "inputs", package.bindings,
          [](const llvm::json::Value &value) {
            return parseExternalBinding(
                value, "input",
                lifecycle({"import_or_allocate", "query", "bind", "copy_h2d"}));
          }))
    return error;
  if (llvm::Error error = parseBindingArray(
          **interface, "outputs", package.bindings,
          [](const llvm::json::Value &value) {
            return parseExternalBinding(
                value, "output",
                lifecycle({"allocate", "query", "bind", "copy_d2h"}));
          }))
    return error;
  if (llvm::Error error = parseBindingArray(
          **interface, "parameters", package.bindings,
          [](const llvm::json::Value &value) {
            return parseExternalBinding(
                value, "parameter",
                lifecycle({"import_or_allocate", "query", "bind", "copy_h2d"}));
          }))
    return error;
  if (llvm::Error error = parseBindingArray(
          **interface, "workspace", package.bindings, parseWorkspaceBinding))
    return error;
  if (llvm::Error error =
          parseBindingArray(**interface, "resident_constants", package.bindings,
                            parseResidentConstantBinding))
    return error;
  return llvm::Error::success();
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

llvm::Expected<const RuntimeBinding *>
findRuntimeBinding(llvm::ArrayRef<RuntimeBinding> bindings,
                   llvm::StringRef name) {
  const RuntimeBinding *result = nullptr;
  for (const RuntimeBinding &binding : bindings) {
    if (binding.name != name)
      continue;
    if (result != nullptr)
      return makeError("runtime binding name is duplicated: " + name);
    result = &binding;
  }
  if (result == nullptr)
    return makeError("runtime binding was not found: " + name);
  return result;
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

llvm::StringRef launchApiForEntrypointExecutor(EntrypointExecutor executor) {
  switch (executor) {
  case EntrypointExecutor::TxModule:
    return "txLaunchKernel";
  case EntrypointExecutor::TxCluster:
    return "txLaunchClusterKernel";
  case EntrypointExecutor::TxModel:
    return "txLaunchModel";
  case EntrypointExecutor::TxGraph:
    return "txLoadGraph";
  case EntrypointExecutor::LegacyPackage:
    return "legacyRun";
  }
  llvm_unreachable("unknown entrypoint executor");
}

llvm::Expected<RuntimeSession>
buildRuntimeSession(const RuntimePackage &package,
                    const RuntimeEntrypoint &entrypoint) {
  RuntimeSession session;
  session.packageName = package.name;
  session.runtimeMode = package.runtimeMode;
  session.completionSource = package.completionSource;
  session.bindings = package.bindings;
  session.entrypointName = entrypoint.name;
  session.executor = entrypoint.executor;
  session.launchApi = launchApiForEntrypointExecutor(entrypoint.executor).str();
  session.function = entrypoint.function;
  session.bpmState = entrypoint.bpmState;
  session.modSymbol = entrypoint.modSymbol;

  if (entrypoint.executor == EntrypointExecutor::TxModule ||
      entrypoint.executor == EntrypointExecutor::TxCluster ||
      entrypoint.executor == EntrypointExecutor::TxGraph) {
    const RuntimeModule *module = package.findModule(entrypoint.module);
    if (module == nullptr)
      return makeError("entrypoint " + entrypoint.name +
                       " references missing module: " + entrypoint.module);
    session.hasModule = true;
    session.moduleName = module->name;
    session.moduleFormat = module->format;
    session.modulePath = module->path;
  }

  if ((entrypoint.executor == EntrypointExecutor::TxModule ||
       entrypoint.executor == EntrypointExecutor::TxCluster) &&
      entrypoint.function.empty())
    return makeError("entrypoint " + entrypoint.name + " is missing function");
  if (entrypoint.executor == EntrypointExecutor::TxModel &&
      entrypoint.bpmState != "materialized")
    return makeError("tx.model entrypoint " + entrypoint.name +
                     " requires a materialized BPM descriptor");

  for (auto item : llvm::enumerate(entrypoint.bindingOrder)) {
    llvm::Expected<const RuntimeBinding *> binding =
        findRuntimeBinding(package.bindings, item.value());
    if (!binding) {
      llvm::consumeError(binding.takeError());
      return makeError("entrypoint " + entrypoint.name + " binding_order[" +
                       llvm::Twine(item.index()) +
                       "] does not name a runtime binding: " + item.value());
    }
    RuntimeLaunchArg arg;
    arg.index = item.index();
    arg.bindingName = (*binding)->name;
    arg.role = (*binding)->role;
    arg.bytes = (*binding)->bytes;
    session.argBytes += arg.bytes;
    session.launchArgs.push_back(std::move(arg));
  }
  return session;
}

llvm::Error validateTxHostRuntimePackage(const RuntimePackage &package) {
  if (package.runtimeMode != "tx")
    return makeError("tx-host backend requires runtime.mode tx");

  if (package.completionSource != "runtime_stream_wait" &&
      package.completionSource != "runtime_command_completion" &&
      package.completionSource != "kcore_local_drain")
    return makeError("completion source is not supported by tx-host backend: " +
                     package.completionSource);

  return llvm::Error::success();
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

  if (llvm::Error error = parseModelInterface(*root, package))
    return std::move(error);

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
