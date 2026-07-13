//===- wafer-run.cpp - Verified Wafer package inspection ----------------===//

#include "Wafer/Runtime/PackageManifest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::string packageDirectory;
  uint64_t entryId = std::numeric_limits<uint64_t>::max();
  uint64_t maxResourceBytes = std::numeric_limits<uint64_t>::max();
  bool noCard = false;
};

void printUsage(llvm::raw_ostream &output) {
  output << "usage: wafer-run --package-dir <path> --entry-id <id> "
            "--no-card [--max-resource-bytes <bytes>]\n";
}

llvm::Expected<Options> parseOptions(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef argument = argv[index];
    auto requireValue = [&]() -> llvm::Expected<llvm::StringRef> {
      if (index + 1 >= argc)
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "missing value for " + argument);
      return llvm::StringRef(argv[++index]);
    };
    if (argument == "--package-dir") {
      llvm::Expected<llvm::StringRef> value = requireValue();
      if (!value)
        return value.takeError();
      options.packageDirectory = value->str();
      continue;
    }
    if (argument == "--entry-id") {
      llvm::Expected<llvm::StringRef> value = requireValue();
      if (!value)
        return value.takeError();
      if (value->getAsInteger(10, options.entryId))
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "--entry-id must be an integer");
      continue;
    }
    if (argument == "--max-resource-bytes") {
      llvm::Expected<llvm::StringRef> value = requireValue();
      if (!value)
        return value.takeError();
      if (value->getAsInteger(10, options.maxResourceBytes))
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "--max-resource-bytes must be an integer");
      continue;
    }
    if (argument == "--no-card") {
      options.noCard = true;
      continue;
    }
    if (argument == "--help" || argument == "-h") {
      printUsage(llvm::outs());
      std::exit(0);
    }
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown option: " + argument);
  }
  if (options.packageDirectory.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "--package-dir is required");
  if (options.entryId == std::numeric_limits<uint64_t>::max())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "--entry-id is required");
  if (!options.noCard)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "wafer-run requires explicit --no-card; board execution is not "
        "implemented");
  return options;
}

int fail(llvm::Error error) {
  llvm::errs() << "wafer-run: " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  llvm::Expected<Options> options = parseOptions(argc, argv);
  if (!options)
    return fail(options.takeError());

  llvm::Expected<wafer::runtime::VerifiedPackageManifest> package =
      wafer::runtime::loadVerifiedPackageManifest(options->packageDirectory);
  if (!package)
    return fail(package.takeError());
  const wafer::runtime::PackageManifest &manifest = package->getManifest();
  wafer::runtime::EntryId entryId(options->entryId);
  auto entry = llvm::find_if(manifest.entries, [&](const auto &candidate) {
    return candidate.id == entryId;
  });
  if (entry == manifest.entries.end())
    return fail(llvm::createStringError(llvm::errc::invalid_argument,
                                        "entry ID is not present in package"));

  std::vector<wafer::runtime::RuntimeInvocationBinding> bindings;
  for (const wafer::runtime::PackageResourceRecord &resource :
       manifest.resources) {
    if (resource.logicalRank != entry->logicalRank || !resource.hostVisible)
      continue;
    bindings.push_back({resource.id, resource.bytes, resource.alignment,
                        resource.access, true});
  }
  wafer::runtime::RuntimeEnvironment environment{
      manifest.targetIdentity, manifest.runtimeABI, manifest.moduleFormat,
      options->maxResourceBytes};
  llvm::Expected<wafer::runtime::RuntimeSessionPlan> plan =
      wafer::runtime::preflightNoCardRuntimeSession(*package, entryId, bindings,
                                                    environment);
  if (!plan)
    return fail(plan.takeError());

  llvm::outs() << "package: id=" << manifest.program.getValue()
               << " schema=" << manifest.schemaVersion
               << " ranks=" << manifest.rankCount << "\n";
  llvm::outs() << "target: " << manifest.targetIdentity
               << " runtime_abi=" << manifest.runtimeABI
               << " module_format=" << manifest.moduleFormat << "\n";
  llvm::outs() << "entry: " << plan->entry.getValue()
               << " rank=" << plan->logicalRank
               << " symbol=" << plan->entrySymbol << "\n";
  llvm::outs() << "module: " << plan->module.getValue()
               << " path=" << plan->modulePath << "\n";
  for (auto [ordinal, resource] : llvm::enumerate(plan->resources)) {
    llvm::outs() << "launch_slot: " << ordinal
                 << " resource=" << resource.resource.getValue() << " role="
                 << wafer::runtime::stringifyPackageResourceRole(resource.role)
                 << " bytes=" << resource.bytes
                 << " alignment=" << resource.alignment << " access="
                 << wafer::runtime::stringifyPackageAccessMode(resource.access)
                 << " externally_bound="
                 << (resource.externallyBound ? "true" : "false") << "\n";
  }
  llvm::outs() << "terminal_completion: " << plan->terminalCompletion.getValue()
               << " kind=entry_return\n";
  llvm::outs() << "board_execution: false\n";
  return 0;
}
