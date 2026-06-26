//===- wafer-run.cpp - Wafer host runtime driver --------------------------===//

#include "Wafer/Runtime/HostRuntime.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

struct Options {
  std::string packageMetadataPath;
  std::string entrypointName;
  std::string runtimeLibraryPath;
  bool showHelp = false;
};

void printUsage(llvm::raw_ostream &os) {
  os << "usage: wafer-run --package-metadata <path> --entrypoint <name> "
        "--runtime-library <path>\n";
}

bool parseArgs(int argc, char **argv, Options &options) {
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef arg(argv[index]);
    auto requireValue = [&](std::string &field) -> bool {
      if (index + 1 >= argc)
        return false;
      field = argv[++index];
      return true;
    };

    if (arg == "--package-metadata") {
      if (!requireValue(options.packageMetadataPath))
        return false;
      continue;
    }
    if (arg == "--entrypoint") {
      if (!requireValue(options.entrypointName))
        return false;
      continue;
    }
    if (arg == "--runtime-library") {
      if (!requireValue(options.runtimeLibraryPath))
        return false;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      options.showHelp = true;
      return true;
    }
    llvm::errs() << "error: unknown argument: " << arg << "\n";
    return false;
  }

  if (options.packageMetadataPath.empty()) {
    llvm::errs() << "error: --package-metadata is required\n";
    return false;
  }
  if (options.entrypointName.empty()) {
    llvm::errs() << "error: --entrypoint is required\n";
    return false;
  }
  if (options.runtimeLibraryPath.empty()) {
    llvm::errs() << "error: --runtime-library is required\n";
    return false;
  }
  return true;
}

int fail(llvm::Error error) {
  llvm::errs() << "error: " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);

  Options options;
  if (!parseArgs(argc, argv, options))
    return 1;
  if (options.showHelp) {
    printUsage(llvm::outs());
    return 0;
  }

  llvm::Expected<wafer::runtime::RuntimePackage> package =
      wafer::runtime::loadRuntimePackageMetadataFile(
          options.packageMetadataPath);
  if (!package)
    return fail(package.takeError());

  const wafer::runtime::RuntimeEntrypoint *entrypoint =
      package->findEntrypoint(options.entrypointName);
  if (entrypoint == nullptr) {
    llvm::errs() << "error: entrypoint was not found in package metadata: "
                 << options.entrypointName << "\n";
    return 1;
  }
  if (entrypoint->executor ==
      wafer::runtime::EntrypointExecutor::LegacyPackage) {
    llvm::errs() << "error: legacy.tsm entrypoint " << entrypoint->name
                 << " requires the legacy board gate\n";
    return 1;
  }

  llvm::Expected<wafer::runtime::TxRuntimeLibrary> runtimeLibrary =
      wafer::runtime::TxRuntimeLibrary::load(options.runtimeLibraryPath);
  if (!runtimeLibrary)
    return fail(runtimeLibrary.takeError());

  if (llvm::Error error =
          runtimeLibrary->validateRequiredSymbols(*entrypoint)) {
    return fail(std::move(error));
  }

  llvm::outs() << "backend: tx-host\n";
  llvm::outs() << "package: " << package->name << "\n";
  llvm::outs() << "runtime_mode: " << package->runtimeMode << "\n";
  llvm::outs() << "completion_source: " << package->completionSource << "\n";
  llvm::outs() << "selected_entrypoint: " << entrypoint->name << " "
               << wafer::runtime::stringifyEntrypointExecutor(
                      entrypoint->executor)
               << "\n";
  llvm::outs() << "tx_runtime_library: " << options.runtimeLibraryPath << "\n";
  llvm::outs() << "tx_runtime_symbols: ok\n";
  llvm::outs() << "board_launch_gate: not executed\n";
  return 0;
}
