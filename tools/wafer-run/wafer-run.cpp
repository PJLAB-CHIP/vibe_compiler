//===- wafer-run.cpp - Verified Wafer package execution -----------------===//

#include "Wafer/Runtime/Board/BoardRuntime.h"
#include "Wafer/Package/Manifest/PackageManifest.h"
#include "Wafer/Runtime/Profile/ProfileInstrumentation.h"
#include "WaferProfileCollection.h"
#include "WaferRunBoardIO.h"
#if defined(WAFER_ENABLE_BOARD_RUNTIME)
#include "Wafer/Runtime/Board/TxBoardRuntime.h"
#endif

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
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
  std::string packageDirectory;
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
  std::vector<wafer::runtime::cli::PortFile> resourceFiles;
  std::vector<wafer::runtime::cli::PortFile> expectedFiles;
  std::vector<wafer::runtime::cli::PortFile> relaxedF16ExpectedFiles;
  std::vector<wafer::runtime::cli::PortFile> outputFiles;
  bool supportsHostWatchdog = false;
  bool deviceTiming = false;
  bool noCard = false;
  bool board = false;
};

void printUsage(llvm::raw_ostream &output) {
  output << "usage:\n"
            "  wafer-run --package-dir <path> --no-card "
            "[--max-resource-bytes <bytes>] [--direct-dte-status-abi <abi> "
            "--supports-host-watchdog]\n"
            "  wafer-run --package-dir <path> --board "
            "[--device-id <id>] --expected-runtime-version <decimal> "
            "--expected-device-name <name> --expected-pci-bus-id <bdf> "
            "--expected-tile-count <count> "
            "--expected-runtime-library-sha256 <hex> "
            "[--completion-timeout-ms <milliseconds>] "
            "[--device-timing] "
            "--resource <ResourceId=raw-file>... "
            "[--expected <ResourceId=raw-file>]... "
            "[--expected-f16-relaxed <ResourceId=raw-file>]... "
            "[--output <ResourceId=raw-file>]...\n";
}

llvm::Expected<wafer::runtime::cli::PortFile>
parseResourceFile(llvm::StringRef value, llvm::StringRef option) {
  auto split = value.split('=');
  wafer::runtime::cli::PortFile result;
  if (split.first.empty() || split.second.empty() ||
      split.first.getAsInteger(10, result.portId))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        (option + " must use PortId=raw-file").str());
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
    if (argument == "--resource" || argument == "--expected" ||
        argument == "--expected-f16-relaxed" || argument == "--output") {
      llvm::Expected<llvm::StringRef> value = requireValue();
      if (!value)
        return value.takeError();
      llvm::Expected<wafer::runtime::cli::PortFile> parsed =
          parseResourceFile(*value, argument);
      if (!parsed)
        return parsed.takeError();
      if (argument == "--resource")
        options.resourceFiles.push_back(std::move(*parsed));
      else if (argument == "--expected")
        options.expectedFiles.push_back(std::move(*parsed));
      else if (argument == "--expected-f16-relaxed")
        options.relaxedF16ExpectedFiles.push_back(std::move(*parsed));
      else
        options.outputFiles.push_back(std::move(*parsed));
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
    if (argument == "--device-timing") {
      options.deviceTiming = true;
      continue;
    }
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "unknown option: " + argument);
  }
  if (options.packageDirectory.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "--package-dir is required");
  if (options.noCard == options.board)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "wafer-run requires explicit --no-card or --board, but not both");
  if (options.noCard &&
      (!options.resourceFiles.empty() || !options.expectedFiles.empty() ||
       !options.relaxedF16ExpectedFiles.empty() ||
       !options.outputFiles.empty() || options.expectedRuntimeVersion != 0 ||
       options.expectedTileCount != 0 || options.deviceIdSpecified ||
       !options.expectedDeviceName.empty() ||
       !options.expectedPCIBusId.empty() ||
       !options.expectedRuntimeLibraryDigest.empty() ||
       options.completionTimeoutSpecified || options.deviceTiming))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "board binding and qualification options require --board");
  if (options.board &&
      (options.maxResourceBytes != std::numeric_limits<uint64_t>::max() ||
       options.directDTEStatusABI || options.supportsHostWatchdog))
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "no-card verification options are not valid with --board");
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

void printRuntimeLaunchContract(const wafer::RuntimeLaunchContract &launch) {
  llvm::outs() << "launch: kind=kernel";
  const wafer::KernelRuntimeLaunchContract &kernel = launch.getKernel();
  llvm::outs() << " form=" << wafer::stringifyKernelLaunchForm(kernel.form)
               << " entry_abi="
               << wafer::stringifyKernelEntryABI(kernel.entryABI);
  llvm::outs() << " phases=";
  for (auto [index, phase] : llvm::enumerate(launch.getPhases())) {
    if (index != 0)
      llvm::outs() << ",";
    llvm::outs() << wafer::stringifyRuntimeLaunchPhaseRole(phase);
  }
  llvm::outs() << "\n";
}

void printNoCardTilePlan(const wafer::runtime::RuntimeSessionPlan &plan) {
  llvm::outs() << "entry: " << plan.entry.getValue()
               << " card_id=" << plan.cardId.getValue()
               << " tile_id=" << plan.tileId.getValue()
               << " launch_slot=" << plan.launchSlot.getValue() << "\n";
  llvm::outs() << "module: " << plan.module.getValue()
               << " path=" << plan.modulePath << "\n";
  for (const wafer::runtime::PlannedRuntimeLaunchPhase &phase : plan.phases)
    llvm::outs() << "launch_phase: role="
                 << wafer::stringifyRuntimeLaunchPhaseRole(phase.role)
                 << " symbol=" << phase.symbol << "\n";
  for (auto [ordinal, address] : llvm::enumerate(plan.argumentAddresses)) {
    llvm::outs() << "launch_argument: " << ordinal << " base="
                 << (address.base ==
                             wafer::runtime::RuntimeArgumentAddressBase::ProgramData
                         ? "program_data"
                         : "invocation")
                 << " offset=" << address.offset << "\n";
  }
  llvm::outs() << "completion: "
               << wafer::runtime::stringifyPackageEntryCompletionKind(
                      plan.completion)
               << "\n";
}

int runNoCard(
    const Options &options,
    const wafer::runtime::ExecutablePackage &package,
    const std::optional<wafer::runtime::VerifiedProfileInstrumentation>
        &profileInstrumentation) {
  const wafer::runtime::PackageManifest &manifest = package.getManifest();
  std::vector<wafer::runtime::RuntimeInvocationBinding> bindings;
  for (const wafer::runtime::ExternalPortRecord &port : manifest.inputs)
    bindings.push_back({port.id, port.bytes, port.alignment});
  wafer::runtime::RuntimeEnvironment environment{
      manifest.targetIdentity, manifest.runtimeABI, manifest.moduleFormat,
      options.maxResourceBytes};
  const wafer::KernelRuntimeLaunchContract &kernel = manifest.launch.getKernel();
  environment.supportedKernelLaunchForms.push_back(kernel.form);
  environment.supportedKernelEntryABIs.push_back(kernel.entryABI);
  if (options.directDTEStatusABI) {
    environment.supportsDirectDTE = true;
    environment.directDTEStatusABI = *options.directDTEStatusABI;
  }
  environment.supportsHostWatchdog = options.supportsHostWatchdog;
  llvm::Expected<wafer::runtime::RuntimeInvocationPlan> invocationPlan =
      wafer::runtime::planRuntimeInvocation(package.getVerifiedManifest(),
                                            bindings, environment);
  if (!invocationPlan)
    return fail(invocationPlan.takeError());

  llvm::outs() << "package: id=" << manifest.program.getValue()
               << " cards=" << manifest.cardCount
               << " tiles=" << manifest.tileCount << "\n";
  llvm::outs() << "target_identity: "
               << wafer::stringifyTargetIdentityId(manifest.targetIdentity)
               << "\n";
  llvm::outs() << "target: "
               << wafer::stringifyTargetIdentityId(manifest.targetIdentity)
               << " runtime_abi="
               << wafer::stringifyKernelRuntimeABIId(manifest.runtimeABI)
               << " module_format=" << manifest.moduleFormat << "\n";
  printRuntimeLaunchContract(manifest.launch);
  for (const wafer::runtime::RuntimeSessionPlan &tile : invocationPlan->tiles)
    printNoCardTilePlan(tile);
  llvm::outs() << "invocation_tiles: " << invocationPlan->tileCount << "\n";
  if (profileInstrumentation)
    llvm::outs() << "profile_instrumentation: ready cards="
                 << profileInstrumentation->getCardCount()
                 << " tiles=" << profileInstrumentation->getTileCount()
                 << " captures=" << profileInstrumentation->getCaptures().size()
                 << " target_call_sites="
                 << profileInstrumentation->getSiteCount() << "\n";
  // No-card is an invocation/package contract check only. It does not run
  // target arithmetic, device completion, or numeric readback; callers must
  // not reuse these lines as a board result.
  llvm::outs() << "arithmetic_execution: false\n"
               << "completion_execution: false\n"
               << "numeric_readback: false\n";
  llvm::outs() << "board_execution: false\n";
  return 0;
}

#if defined(WAFER_ENABLE_BOARD_RUNTIME)
int runBoard(const Options &options,
             const wafer::runtime::ExecutablePackage &package,
             const std::optional<wafer::runtime::VerifiedProfileInstrumentation>
                 &profileInstrumentation) {
  const wafer::runtime::PackageManifest &manifest = package.getManifest();
  wafer::runtime::BoardRuntimeInvocationRequest request;
  request.deviceId = options.deviceId;
  request.completionTimeoutMilliseconds = options.completionTimeoutMilliseconds;
  if (options.deviceTiming)
    request.deviceTimingPolicy =
        wafer::runtime::BoardDeviceTimingPolicy::StreamEvents;
  request.qualification = {
      options.expectedRuntimeVersion,       options.expectedTileCount,
      options.expectedDeviceName,           options.expectedPCIBusId,
      options.expectedRuntimeLibraryDigest,
  };
  llvm::Expected<wafer::runtime::cli::BoardInvocationFilePlan> filePlan =
      wafer::runtime::cli::prepareBoardInvocationFiles(
          manifest, std::move(request), options.resourceFiles,
          options.expectedFiles, options.outputFiles,
          options.relaxedF16ExpectedFiles);
  if (!filePlan)
    return fail(filePlan.takeError());

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
  wafer::runtime::cli::BoardInvocationFilePlan completedPlan;
  std::string profileRunDirectory;
  llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> result =
      [&]() -> llvm::Expected<wafer::runtime::BoardRuntimeInvocationResult> {
    if (profileInstrumentation) {
      llvm::Expected<wafer::runtime::cli::BoardProfileCollectionResult>
          collection = wafer::runtime::cli::runBoardProfileCollection(
              *profileInstrumentation, manifest, *filePlan, **driver);
      if (!collection)
        return collection.takeError();
      completedPlan = std::move(collection->finalPlan);
      profileRunDirectory = std::move(collection->runDirectory);
      return std::move(collection->finalResult);
    }
    completedPlan = std::move(*filePlan);
    return wafer::runtime::executeBoardInvocation(
        package, std::move(completedPlan.request), **driver);
  }();
  if (!result) {
    llvm::Error error = result.takeError();
    bool poisoned = (*driver)->getContextState() ==
                    wafer::runtime::BoardRuntimeContextState::Poisoned;
    return terminateBoardProcess(std::move(error), poisoned);
  }
  if (llvm::Error error = wafer::runtime::cli::validateAndWriteBoardOutputs(
          result->outputs, completedPlan))
    return terminateBoardProcess(std::move(error), /*poisoned=*/false);

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
  printRuntimeLaunchContract(manifest.launch);
  for (const auto &tile : result->device.tiles)
    llvm::outs() << "board_tile: tile_id=" << tile.tileId.getValue()
                 << " launch_slot=" << tile.launchSlot.getValue()
                 << " available=" << (tile.available ? "true" : "false")
                 << " physical_x=" << tile.physicalX
                 << " physical_y=" << tile.physicalY << "\n";
  for (const wafer::runtime::BoardRuntimeTileResult &tile : result->tiles)
    llvm::outs() << "entry: " << tile.entry.getValue()
                 << " card_id=" << tile.cardId.getValue()
                 << " tile_id=" << tile.tileId.getValue()
                 << " launch_slot=" << tile.launchSlot.getValue()
                 << " module=" << tile.module.getValue() << "\n";
  for (wafer::runtime::BoardRuntimeStage stage : result->completedStages)
    llvm::outs() << "board_stage: "
                 << wafer::runtime::stringifyBoardRuntimeStage(stage) << "\n";
  for (const wafer::runtime::BoardRuntimeOutput &output : result->outputs) {
    auto comparison =
        completedPlan.expectedComparisons.find(output.port.getValue());
    if (comparison != completedPlan.expectedComparisons.end()) {
      llvm::outs() << "output_compare: port=" << output.port.getValue()
                   << " bytes=" << output.bytes.size();
      if (comparison->second ==
          wafer::runtime::cli::BoardOutputComparisonKind::Exact) {
        llvm::outs() << " exact=true\n";
      } else {
        llvm::outs()
            << " policy=f16-relaxed abs=0.0009765625 rel=0.001 max_ulp="
            << wafer::runtime::cli::kRelaxedF16MaximumUlp
            << " signed_zero_equal=true\n";
      }
    }
    auto capture = completedPlan.outputPaths.find(output.port.getValue());
    if (capture != completedPlan.outputPaths.end())
      llvm::outs() << "output_capture: port=" << output.port.getValue()
                   << " bytes=" << output.bytes.size()
                   << " path=" << capture->second << "\n";
  }
  for (const wafer::runtime::BoardRuntimeTileResult &tile : result->tiles)
    llvm::outs() << "completion: "
                 << wafer::runtime::stringifyPackageEntryCompletionKind(
                        tile.completion)
                 << " tile_id=" << tile.tileId.getValue() << "\n";
  llvm::outs() << "invocation_tiles: " << result->tiles.size() << "\n";
  if (options.deviceTiming && result->deviceExecutionNanoseconds)
    llvm::outs() << "board_timing: kind=tx-stream-events device_elapsed_ns="
                 << *result->deviceExecutionNanoseconds << "\n";
  const wafer::KernelRuntimeLaunchContract &kernel = manifest.launch.getKernel();
  if (kernel.form == wafer::KernelLaunchForm::Grid) {
    llvm::outs() << "launch_pattern: kernel-grid-x" << manifest.tileCount
                 << "\n";
    llvm::outs() << "physical_tile_execution_basis: "
                    "scheduler-pid-x-and-exact-tile-slices\n";
  } else {
    llvm::outs() << "launch_pattern: cluster-x" << manifest.tileCount << "\n";
    llvm::outs() << "physical_tile_execution_basis: "
                    "cluster-pid-and-exact-tile-slices\n";
  }
  llvm::outs() << "physical_tile_domain: 0.." << manifest.tileCount - 1 << "\n";
  llvm::outs() << "physical_execution_claim: none\n";
  if (!profileRunDirectory.empty())
    llvm::outs() << "profile_run: " << profileRunDirectory
                 << "\nprofile_report: " << profileRunDirectory
                 << "/index.html\n";
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

  llvm::Expected<wafer::runtime::ExecutablePackage> package =
      wafer::runtime::loadExecutablePackage(options->packageDirectory);
  if (!package)
    return fail(package.takeError());
  llvm::Expected<std::optional<wafer::runtime::VerifiedProfileInstrumentation>>
      loaded = wafer::runtime::loadSiblingProfileInstrumentationIfPresent(
          options->packageDirectory);
  if (!loaded)
    return fail(loaded.takeError());
  std::optional<wafer::runtime::VerifiedProfileInstrumentation>
      profileInstrumentation;
  if (*loaded)
    profileInstrumentation.emplace(std::move(**loaded));
  if (options->noCard) {
    return runNoCard(*options, *package, profileInstrumentation);
  }
#if defined(WAFER_ENABLE_BOARD_RUNTIME)
  return runBoard(*options, *package, profileInstrumentation);
#else
  return fail(llvm::createStringError(
      llvm::errc::not_supported,
      "board runtime support is not enabled in this build"));
#endif
}
