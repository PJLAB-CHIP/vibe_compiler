//===- wafer-run.cpp - Verified Wafer package execution -----------------===//

#include "Wafer/Runtime/BoardRuntime.h"
#include "Wafer/Runtime/PackageManifest.h"
#if defined(WAFER_ENABLE_BOARD_RUNTIME)
#include "Wafer/Runtime/TxBoardRuntime.h"
#endif

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
  struct ResourceFile {
    uint64_t resourceId = std::numeric_limits<uint64_t>::max();
    std::string path;
  };

  std::string packageDirectory;
  uint64_t entryId = std::numeric_limits<uint64_t>::max();
  bool allRanks = false;
  uint64_t maxResourceBytes = std::numeric_limits<uint64_t>::max();
  uint64_t completionTimeoutMilliseconds =
      wafer::runtime::kDefaultBoardCompletionTimeoutMilliseconds;
  bool completionTimeoutSpecified = false;
  uint32_t deviceId = 0;
  bool deviceIdSpecified = false;
  uint32_t expectedRuntimeVersion = 0;
  uint32_t expectedTileCount = 0;
  std::string expectedDeviceName;
  std::string expectedPCIBusId;
  std::string expectedRuntimeLibraryDigest;
  std::optional<std::string> directDTEStatusABI;
  std::vector<ResourceFile> resourceFiles;
  std::vector<ResourceFile> expectedFiles;
  bool supportsHostWatchdog = false;
  bool noCard = false;
  bool board = false;
};

void printUsage(llvm::raw_ostream &output) {
  output << "usage:\n"
            "  wafer-run --package-dir <path> --entry-id <id> --no-card "
            "[--max-resource-bytes <bytes>] [--direct-dte-status-abi <abi> "
            "--supports-host-watchdog]\n"
            "  wafer-run --package-dir <path> --all-ranks --no-card "
            "[--max-resource-bytes <bytes>] [--direct-dte-status-abi <abi> "
            "--supports-host-watchdog]\n"
            "  wafer-run --package-dir <path> (--all-ranks | --entry-id <id>) "
            "--board "
            "[--device-id <id>] --expected-runtime-version <decimal> "
            "--expected-device-name <name> --expected-pci-bus-id <bdf> "
            "--expected-tile-count <count> "
            "--expected-runtime-library-sha256 <hex> "
            "[--completion-timeout-ms <milliseconds>] "
            "--resource <ResourceId=raw-file>... "
            "--expected <ResourceId=raw-file>...\n";
}

llvm::Expected<Options::ResourceFile>
parseResourceFile(llvm::StringRef value, llvm::StringRef option) {
  auto split = value.split('=');
  Options::ResourceFile result;
  if (split.first.empty() || split.second.empty() ||
      split.first.getAsInteger(10, result.resourceId))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        (option + " must use ResourceId=raw-file").str());
  result.path = split.second.str();
  return result;
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
    if (argument == "--all-ranks") {
      options.allRanks = true;
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
    if (argument == "--device-id") {
      llvm::Expected<llvm::StringRef> value = requireValue();
      if (!value)
        return value.takeError();
      if (value->getAsInteger(10, options.deviceId))
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "--device-id must be an integer");
      options.deviceIdSpecified = true;
      continue;
    }
    if (argument == "--completion-timeout-ms") {
      llvm::Expected<llvm::StringRef> value = requireValue();
      if (!value)
        return value.takeError();
      if (value->getAsInteger(10, options.completionTimeoutMilliseconds) ||
          options.completionTimeoutMilliseconds == 0 ||
          options.completionTimeoutMilliseconds >
              wafer::runtime::kMaximumBoardCompletionTimeoutMilliseconds)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "--completion-timeout-ms is outside the supported range");
      options.completionTimeoutSpecified = true;
      continue;
    }
    if (argument == "--expected-runtime-version" ||
        argument == "--expected-tile-count") {
      llvm::Expected<llvm::StringRef> value = requireValue();
      if (!value)
        return value.takeError();
      uint32_t &destination = argument == "--expected-runtime-version"
                                  ? options.expectedRuntimeVersion
                                  : options.expectedTileCount;
      if (value->getAsInteger(10, destination) || destination == 0)
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            argument + " must be a positive decimal integer");
      continue;
    }
    if (argument == "--expected-device-name" ||
        argument == "--expected-pci-bus-id") {
      llvm::Expected<llvm::StringRef> value = requireValue();
      if (!value)
        return value.takeError();
      if (value->empty())
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       argument + " must not be empty");
      std::string &destination = argument == "--expected-device-name"
                                     ? options.expectedDeviceName
                                     : options.expectedPCIBusId;
      destination = value->str();
      continue;
    }
    if (argument == "--expected-runtime-library-sha256") {
      llvm::Expected<llvm::StringRef> value = requireValue();
      if (!value)
        return value.takeError();
      if (value->size() != 64 || !llvm::all_of(*value, [](char character) {
            return llvm::isHexDigit(character);
          }))
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "--expected-runtime-library-sha256 must be 64 hex digits");
      options.expectedRuntimeLibraryDigest = "sha256:" + value->lower();
      continue;
    }
    if (argument == "--resource" || argument == "--expected") {
      llvm::Expected<llvm::StringRef> value = requireValue();
      if (!value)
        return value.takeError();
      llvm::Expected<Options::ResourceFile> parsed =
          parseResourceFile(*value, argument);
      if (!parsed)
        return parsed.takeError();
      if (argument == "--resource")
        options.resourceFiles.push_back(std::move(*parsed));
      else
        options.expectedFiles.push_back(std::move(*parsed));
      continue;
    }
    if (argument == "--direct-dte-status-abi") {
      llvm::Expected<llvm::StringRef> value = requireValue();
      if (!value)
        return value.takeError();
      if (value->empty())
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "--direct-dte-status-abi must not be empty");
      options.directDTEStatusABI = value->str();
      continue;
    }
    if (argument == "--supports-host-watchdog") {
      options.supportsHostWatchdog = true;
      continue;
    }
    if (argument == "--no-card") {
      options.noCard = true;
      continue;
    }
    if (argument == "--board") {
      options.board = true;
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
  const bool hasEntry = options.entryId != std::numeric_limits<uint64_t>::max();
  if (hasEntry == options.allRanks)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "exactly one of --entry-id or --all-ranks "
                                   "is required");
  if (options.noCard == options.board)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "wafer-run requires explicit --no-card or --board, but not both");
  if (options.noCard &&
      (!options.resourceFiles.empty() || !options.expectedFiles.empty() ||
       options.expectedRuntimeVersion != 0 || options.expectedTileCount != 0 ||
       options.deviceIdSpecified || !options.expectedDeviceName.empty() ||
       !options.expectedPCIBusId.empty() ||
       !options.expectedRuntimeLibraryDigest.empty() ||
       options.completionTimeoutSpecified))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "board binding and qualification options require --board");
  if (options.board &&
      (options.maxResourceBytes != std::numeric_limits<uint64_t>::max() ||
       options.directDTEStatusABI || options.supportsHostWatchdog))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "no-card admission options are not valid with --board");
  if (options.board &&
      (options.expectedRuntimeVersion == 0 || options.expectedTileCount == 0 ||
       options.expectedDeviceName.empty() || options.expectedPCIBusId.empty() ||
       options.expectedRuntimeLibraryDigest.empty()))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "--board requires complete explicit TX qualification options");
  return options;
}

int fail(llvm::Error error) {
  llvm::errs() << "wafer-run: " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

llvm::Expected<std::vector<uint8_t>> readRawFile(llvm::StringRef path,
                                                 uint64_t expectedBytes) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read raw tensor file: " + path);
  llvm::StringRef bytes = (*buffer)->getBuffer();
  if (bytes.size() != expectedBytes)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "raw tensor file byte count does not match ResourceId: " + path);
  return std::vector<uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()),
                              reinterpret_cast<const uint8_t *>(bytes.data()) +
                                  bytes.size());
}

llvm::Expected<llvm::DenseMap<uint64_t, std::string>>
indexResourceFiles(llvm::ArrayRef<Options::ResourceFile> files,
                   llvm::StringRef option) {
  llvm::DenseMap<uint64_t, std::string> indexed;
  for (const Options::ResourceFile &file : files) {
    if (!indexed.try_emplace(file.resourceId, file.path).second)
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "duplicate ResourceId for " + option);
  }
  return indexed;
}

void printNoCardRankPlan(const wafer::runtime::RuntimeSessionPlan &plan) {
  llvm::outs() << "entry: " << plan.entry.getValue()
               << " rank=" << plan.logicalRank << " symbol=" << plan.entrySymbol
               << "\n";
  llvm::outs() << "module: " << plan.module.getValue()
               << " path=" << plan.modulePath << "\n";
  for (auto [ordinal, resource] : llvm::enumerate(plan.resources)) {
    llvm::outs() << "launch_slot: " << ordinal
                 << " resource=" << resource.resource.getValue() << " role="
                 << wafer::runtime::stringifyPackageResourceRole(resource.role)
                 << " bytes=" << resource.bytes
                 << " alignment=" << resource.alignment << " access="
                 << wafer::runtime::stringifyPackageAccessMode(resource.access)
                 << " externally_bound="
                 << (resource.externallyBound ? "true" : "false") << "\n";
  }
  llvm::outs() << "terminal_completion: " << plan.terminalCompletion.getValue()
               << " kind=entry_return\n";
}

int runNoCard(const Options &options,
              const wafer::runtime::VerifiedPackageManifest &package,
              const wafer::runtime::PackageEntrypointRecord *selectedEntry) {
  const wafer::runtime::PackageManifest &manifest = package.getManifest();
  std::vector<wafer::runtime::RuntimeInvocationBinding> bindings;
  for (const wafer::runtime::PackageResourceRecord &resource :
       manifest.resources) {
    if (!resource.hostVisible ||
        (selectedEntry && resource.logicalRank != selectedEntry->logicalRank))
      continue;
    bindings.push_back({resource.id, resource.bytes, resource.alignment,
                        resource.access, true});
  }
  wafer::runtime::RuntimeEnvironment environment{
      manifest.targetProfile, manifest.targetIdentity, manifest.runtimeABI,
      manifest.moduleFormat, options.maxResourceBytes};
  if (options.directDTEStatusABI) {
    environment.supportsDirectDTE = true;
    environment.directDTEStatusABI = *options.directDTEStatusABI;
  }
  environment.supportsHostWatchdog = options.supportsHostWatchdog;
  std::optional<wafer::runtime::RuntimeSessionPlan> selectedPlan;
  std::optional<wafer::runtime::RuntimeInvocationPlan> invocationPlan;
  if (selectedEntry) {
    llvm::Expected<wafer::runtime::RuntimeSessionPlan> plan =
        wafer::runtime::preflightNoCardRuntimeSession(
            package, selectedEntry->id, bindings, environment);
    if (!plan)
      return fail(plan.takeError());
    selectedPlan.emplace(std::move(*plan));
  } else {
    llvm::Expected<wafer::runtime::RuntimeInvocationPlan> plan =
        wafer::runtime::preflightNoCardRuntimeInvocation(package, bindings,
                                                         environment);
    if (!plan)
      return fail(plan.takeError());
    invocationPlan.emplace(std::move(*plan));
  }

  llvm::outs() << "package: id=" << manifest.program.getValue()
               << " schema=" << manifest.schemaVersion
               << " ranks=" << manifest.rankCount << "\n";
  llvm::outs() << "target_profile: "
               << wafer::stringifyTargetProfileId(manifest.targetProfile)
               << "\n";
  llvm::outs() << "target: "
               << wafer::stringifyTargetIdentityId(manifest.targetIdentity)
               << " runtime_abi="
               << wafer::stringifyKernelRuntimeABIId(manifest.runtimeABI)
               << " module_format=" << manifest.moduleFormat << "\n";
  if (selectedEntry) {
    printNoCardRankPlan(*selectedPlan);
  } else {
    for (const wafer::runtime::RuntimeSessionPlan &rank : invocationPlan->ranks)
      printNoCardRankPlan(rank);
    llvm::outs() << "invocation_ranks: " << invocationPlan->rankCount << "\n";
  }
  llvm::outs() << "board_execution: false\n";
  return 0;
}

#if defined(WAFER_ENABLE_BOARD_RUNTIME)
int runBoard(const Options &options,
             const wafer::runtime::VerifiedPackageManifest &package) {
  const wafer::runtime::PackageManifest &manifest = package.getManifest();
  llvm::Expected<llvm::DenseMap<uint64_t, std::string>> resources =
      indexResourceFiles(options.resourceFiles, "--resource");
  if (!resources)
    return fail(resources.takeError());
  llvm::Expected<llvm::DenseMap<uint64_t, std::string>> expected =
      indexResourceFiles(options.expectedFiles, "--expected");
  if (!expected)
    return fail(expected.takeError());

  wafer::runtime::BoardRuntimeInvocationRequest request;
  request.deviceId = options.deviceId;
  request.completionTimeoutMilliseconds = options.completionTimeoutMilliseconds;
  request.qualification = {
      options.expectedRuntimeVersion,       options.expectedTileCount,
      options.expectedDeviceName,           options.expectedPCIBusId,
      options.expectedRuntimeLibraryDigest,
  };
  llvm::DenseMap<uint64_t, std::vector<uint8_t>> expectedBytes;
  for (const wafer::runtime::PackageResourceRecord &resource :
       manifest.resources) {
    if (!resource.hostVisible)
      continue;
    std::vector<uint8_t> bytes(resource.bytes, 0);
    auto source = resources->find(resource.id.getValue());
    if (resource.access != wafer::runtime::PackageAccessMode::WriteOnly) {
      if (source == resources->end())
        return fail(llvm::createStringError(
            llvm::errc::invalid_argument,
            "--resource omits readable ResourceId " +
                std::to_string(resource.id.getValue())));
      llvm::Expected<std::vector<uint8_t>> loaded =
          readRawFile(source->second, resource.bytes);
      if (!loaded)
        return fail(loaded.takeError());
      bytes = std::move(*loaded);
      resources->erase(source);
    } else if (source != resources->end()) {
      return fail(llvm::createStringError(
          llvm::errc::invalid_argument,
          "--resource must not initialize write-only ResourceId " +
              std::to_string(resource.id.getValue())));
    }
    if (resource.access != wafer::runtime::PackageAccessMode::ReadOnly) {
      auto reference = expected->find(resource.id.getValue());
      if (reference == expected->end())
        return fail(llvm::createStringError(
            llvm::errc::invalid_argument,
            "--expected omits writable ResourceId " +
                std::to_string(resource.id.getValue())));
      llvm::Expected<std::vector<uint8_t>> loaded =
          readRawFile(reference->second, resource.bytes);
      if (!loaded)
        return fail(loaded.takeError());
      expectedBytes[resource.id.getValue()] = std::move(*loaded);
      expected->erase(reference);
    }
    request.bindings.push_back({resource.id, std::move(bytes)});
  }
  if (!resources->empty() || !expected->empty())
    return fail(llvm::createStringError(
        llvm::errc::invalid_argument,
        "board invocation contains ResourceIds outside the package"));

  llvm::Expected<std::unique_ptr<wafer::runtime::BoardRuntimeDriver>> driver =
      wafer::runtime::createTxBoardRuntimeDriver(
          options.expectedRuntimeLibraryDigest);
  if (!driver) {
    llvm::errs() << "wafer-run: " << llvm::toString(driver.takeError())
                 << "\nwafer-run: ending TX provider setup without running "
                    "provider finalizers\n";
    llvm::outs().flush();
    llvm::errs().flush();
    std::_Exit(1);
  }
  auto terminateBoardProcess = [&](llvm::Error error, bool poisoned) -> int {
    llvm::errs() << "wafer-run: " << llvm::toString(std::move(error));
    if (poisoned)
      llvm::errs() << "\nwafer-run: TX context quarantined; no further "
                      "provider call, reset, power operation, or retry was "
                      "attempted";
    llvm::errs()
        << "\nwafer-run: ending the one-shot board process without running "
           "TX provider finalizers\n";
    llvm::outs().flush();
    llvm::errs().flush();
    std::_Exit(1);
  };
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      wafer::runtime::executeBoardInvocation(package, options.packageDirectory,
                                             std::move(request), **driver);
  if (!result) {
    llvm::Error error = result.takeError();
    bool poisoned = (*driver)->getContextState() ==
                    wafer::runtime::BoardRuntimeContextState::Poisoned;
    return terminateBoardProcess(std::move(error), poisoned);
  }
  for (const wafer::runtime::BoardRuntimeOutput &output : result->outputs) {
    auto reference = expectedBytes.find(output.resource.getValue());
    if (reference == expectedBytes.end())
      return terminateBoardProcess(
          llvm::createStringError(
              llvm::errc::result_out_of_range,
              "board returned an unexpected writable ResourceId " +
                  std::to_string(output.resource.getValue())),
          /*poisoned=*/false);
    if (reference->second != output.bytes) {
      auto mismatch = llvm::mismatch(reference->second, output.bytes);
      size_t offset =
          static_cast<size_t>(mismatch.first - reference->second.begin());
      return terminateBoardProcess(
          llvm::createStringError(
              llvm::errc::result_out_of_range,
              "complete board output differs for ResourceId " +
                  std::to_string(output.resource.getValue()) + " at byte " +
                  std::to_string(offset) + ": expected=0x" +
                  llvm::utohexstr(*mismatch.first) + " actual=0x" +
                  llvm::utohexstr(*mismatch.second)),
          /*poisoned=*/false);
    }
  }
  if (result->outputs.size() != expectedBytes.size())
    return terminateBoardProcess(
        llvm::createStringError(
            llvm::errc::invalid_argument,
            "board result did not return all-and-only writable resources"),
        /*poisoned=*/false);

  llvm::outs() << "board_device: id=" << result->device.deviceId
               << " name=" << result->device.name
               << " pci_bus_id=" << result->device.pciBusId
               << " runtime_version=0x"
               << llvm::utohexstr(result->device.runtimeVersion)
               << " tiles=" << result->device.tileCount
               << " free_memory_bytes=" << result->device.freeMemoryBytes
               << " total_memory_bytes=" << result->device.totalMemoryBytes
               << " runtime_library_digest="
               << result->device.runtimeLibraryDigest << "\n";
  for (const auto &tile : result->device.tiles)
    llvm::outs() << "board_tile: logical=" << tile.logicalIndex
                 << " available=" << (tile.available ? "true" : "false")
                 << " physical_x=" << tile.physicalX
                 << " physical_y=" << tile.physicalY << "\n";
  if (!options.allRanks) {
    const wafer::runtime::BoardRuntimeRankResult &rank = result->ranks.front();
    llvm::outs() << "entry: " << rank.entry.getValue()
                 << " rank=" << rank.logicalRank << "\n";
  } else {
    for (const wafer::runtime::BoardRuntimeRankResult &rank : result->ranks)
      llvm::outs() << "entry: " << rank.entry.getValue()
                   << " rank=" << rank.logicalRank
                   << " module=" << rank.module.getValue() << "\n";
  }
  for (wafer::runtime::BoardRuntimeStage stage : result->completedStages)
    llvm::outs() << "board_stage: "
                 << wafer::runtime::stringifyBoardRuntimeStage(stage) << "\n";
  for (const wafer::runtime::BoardRuntimeOutput &output : result->outputs)
    llvm::outs() << "output_compare: resource=" << output.resource.getValue()
                 << " bytes=" << output.bytes.size() << " exact=true\n";
  if (!options.allRanks) {
    llvm::outs() << "terminal_completion: "
                 << result->ranks.front().terminalCompletion.getValue()
                 << " kind=entry_return\n";
  } else {
    for (const wafer::runtime::BoardRuntimeRankResult &rank : result->ranks)
      llvm::outs() << "terminal_completion: "
                   << rank.terminalCompletion.getValue()
                   << " kind=entry_return rank=" << rank.logicalRank << "\n";
    llvm::outs() << "invocation_ranks: " << result->ranks.size() << "\n";
    llvm::outs() << "physical_rank_mapping: unclaimed\n";
  }
  llvm::outs() << "board_execution: true\n";
  llvm::outs().flush();
  llvm::errs().flush();
  std::_Exit(0);
}
#endif

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
  const wafer::runtime::PackageEntrypointRecord *selectedEntry = nullptr;
  if (!options->allRanks) {
    wafer::runtime::EntryId entryId(options->entryId);
    auto entry = llvm::find_if(manifest.entries, [&](const auto &candidate) {
      return candidate.id == entryId;
    });
    if (entry == manifest.entries.end())
      return fail(llvm::createStringError(
          llvm::errc::invalid_argument, "entry ID is not present in package"));
    selectedEntry = &*entry;
  }
  if (options->board && !options->allRanks && manifest.rankCount != 1)
    return fail(llvm::createStringError(
        llvm::errc::invalid_argument,
        "multi-rank board execution requires --all-ranks"));

  if (options->noCard)
    return runNoCard(*options, *package, selectedEntry);
#if defined(WAFER_ENABLE_BOARD_RUNTIME)
  return runBoard(*options, *package);
#else
  return fail(llvm::createStringError(
      llvm::errc::not_supported,
      "board runtime support is not enabled in this build"));
#endif
}
