//===- TargetDeviceLink.cpp - Target LLVM emission and device link -------===//

#include "TargetArtifactInternal.h"
#include "Wafer/Support/OptimizationMechanism.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <string>
#include <system_error>

namespace wafer::compiler::detail {
namespace {

struct BackendObservation {
  MechanismKey key;
  bool succeeded = false;
  std::vector<std::string> argv;
  OptimizationDigest inputSnapshotDigest{};
  OptimizationDigest toolDigest{};
  OptimizationDigest outputDigest{};
};

static bool parseDigest(llvm::StringRef text, OptimizationDigest &digest) {
  if (text.size() != digest.size() * 2)
    return false;
  auto nibble = [](char character) -> std::optional<uint8_t> {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return character - 'a' + 10;
    return std::nullopt;
  };
  for (size_t index = 0; index < digest.size(); ++index) {
    std::optional<uint8_t> high = nibble(text[index * 2]);
    std::optional<uint8_t> low = nibble(text[index * 2 + 1]);
    if (!high || !low)
      return false;
    digest[index] = static_cast<uint8_t>((*high << 4) | *low);
  }
  return true;
}

static llvm::Expected<std::vector<BackendObservation>>
readBackendObservations(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read device action observation");
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return parsed.takeError();
  llvm::json::Array *array = parsed->getAsArray();
  if (!array)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "device action observation is not an array");

  std::vector<BackendObservation> observations;
  for (const llvm::json::Value &value : *array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object || object->size() != 6)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "device action observation has unknown or missing fields");
    std::optional<llvm::StringRef> kind = object->getString("action_kind");
    std::optional<bool> succeeded = object->getBoolean("succeeded");
    const llvm::json::Array *argv = object->getArray("argv");
    std::optional<llvm::StringRef> inputDigest =
        object->getString("input_snapshot_digest");
    std::optional<llvm::StringRef> toolDigest =
        object->getString("tool_digest");
    std::optional<llvm::StringRef> outputDigest =
        object->getString("output_digest");
    if (!kind || !succeeded || !argv || argv->empty() || !inputDigest ||
        !toolDigest || !outputDigest)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "device action observation has an invalid field value");

    BackendObservation observation;
    observation.succeeded = *succeeded;
    if (!parseDigest(*inputDigest, observation.inputSnapshotDigest) ||
        !parseDigest(*toolDigest, observation.toolDigest) ||
        !parseDigest(*outputDigest, observation.outputDigest))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "device action observation has a malformed SHA-256 digest");
    if (*kind == "device-object-compilation")
      observation.key = mechanism::DeviceObjectCompilation;
    else if (*kind == "device-runtime-compilation")
      observation.key = mechanism::DeviceRuntimeCompilation;
    else if (*kind == "device-garbage-collection-link")
      observation.key = mechanism::DeviceGarbageCollectionLink;
    else
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "unknown device action kind");
    for (const llvm::json::Value &argument : *argv) {
      std::optional<llvm::StringRef> string = argument.getAsString();
      if (!string)
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "device action argv is not textual");
      observation.argv.push_back(string->str());
    }
    observations.push_back(std::move(observation));
  }
  return observations;
}

static llvm::SmallVector<llvm::StringRef, 20>
asArgumentRefs(const std::vector<std::string> &storage) {
  llvm::SmallVector<llvm::StringRef, 20> arguments;
  arguments.reserve(storage.size());
  for (const std::string &argument : storage)
    arguments.push_back(argument);
  return arguments;
}

static std::vector<std::string> buildActionDriverArguments(
    llvm::StringRef python, llvm::StringRef script, llvm::StringRef llvmIR,
    llvm::StringRef module, llvm::StringRef object, llvm::StringRef crtObject,
    llvm::StringRef stagingDirectory, unsigned actionIndex,
    llvm::StringRef evidencePath, bool plan) {
  return {python.str(),
          script.str(),
          "--llvm-ir",
          llvmIR.str(),
          "--output",
          module.str(),
          "--object-output",
          object.str(),
          "--crt-object-output",
          crtObject.str(),
          "--execution-staging-directory",
          stagingDirectory.str(),
          "--backend-action-index",
          std::to_string(actionIndex),
          plan ? "--backend-action-plan-output"
               : "--optimization-observation-output",
          evidencePath.str()};
}

static llvm::Error runDeviceLinkAction(
    llvm::StringRef python, llvm::StringRef script, llvm::StringRef llvmIR,
    llvm::StringRef module, llvm::StringRef object, llvm::StringRef crtObject,
    llvm::StringRef stagingDirectory, unsigned actionIndex,
    MechanismKey expectedKey, uint64_t invocationOrdinal) {
  llvm::SmallString<256> planPath(stagingDirectory);
  llvm::sys::path::append(planPath,
                          "action-" + std::to_string(actionIndex) + ".plan.json");
  llvm::SmallString<256> terminalPath(stagingDirectory);
  llvm::sys::path::append(
      terminalPath,
      "action-" + std::to_string(actionIndex) + ".terminal.json");

  std::vector<std::string> planStorage = buildActionDriverArguments(
      python, script, llvmIR, module, object, crtObject, stagingDirectory,
      actionIndex, planPath, /*plan=*/true);
  int planExit =
      llvm::sys::ExecuteAndWait(python, asArgumentRefs(planStorage));
  if (planExit != 0)
    return llvm::createStringError(
        llvm::errc::io_error,
        "device action %u planning failed with exit code %d", actionIndex,
        planExit);
  llvm::Expected<std::vector<BackendObservation>> planned =
      readBackendObservations(planPath);
  (void)llvm::sys::fs::remove(planPath);
  if (!planned)
    return planned.takeError();
  if (planned->size() != 1 || planned->front().key != expectedKey ||
      planned->front().succeeded)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "device action plan has the wrong key or terminal state");
  BackendObservation plan = std::move(planned->front());

  OptimizationInvocationTokenV1 token;
  std::string diagnostic;
  if (!beginOptimizationInvocationV1(
          expectedKey, OptimizationCutPoint::DevicePublicationTransaction,
          invocationOrdinal, plan.inputSnapshotDigest, token,
          &diagnostic))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "cannot begin device action: %s",
                                   diagnostic.c_str());

  std::vector<std::string> executeStorage = buildActionDriverArguments(
      python, script, llvmIR, module, object, crtObject, stagingDirectory,
      actionIndex, terminalPath, /*plan=*/false);
  int actionExit =
      llvm::sys::ExecuteAndWait(python, asArgumentRefs(executeStorage));
  llvm::Expected<std::vector<BackendObservation>> actual =
      readBackendObservations(terminalPath);
  (void)llvm::sys::fs::remove(terminalPath);

  BackendObservation terminal = plan;
  bool exactTerminal = false;
  std::string terminalError;
  if (!actual) {
    terminalError = llvm::toString(actual.takeError());
  } else if (actual->size() != 1) {
    terminalError = "device action emitted zero or multiple terminals";
  } else {
    BackendObservation &observed = actual->front();
    terminal.toolDigest = observed.toolDigest;
    terminal.outputDigest = observed.outputDigest;
    terminal.succeeded = observed.succeeded;
    exactTerminal = observed.key == expectedKey &&
                    observed.argv == plan.argv &&
                    observed.inputSnapshotDigest == plan.inputSnapshotDigest &&
                    observed.toolDigest == plan.toolDigest &&
                    observed.succeeded == (actionExit == 0);
    if (!exactTerminal)
      terminalError = "device action terminal disagrees with its plan";
  }

  OptimizationInvocationTelemetry telemetry;
  telemetry.key = expectedKey;
  telemetry.cutPoint = OptimizationCutPoint::DevicePublicationTransaction;
  telemetry.invocationOrdinal = invocationOrdinal;
  telemetry.outcome = exactTerminal && terminal.succeeded
                          ? InvocationOutcome::Applied
                          : InvocationOutcome::Invalid;
  telemetry.successfulBackendActionCount =
      telemetry.outcome == InvocationOutcome::Applied ? 1 : 0;
  telemetry.actionExecutor = plan.argv.front();
  telemetry.actionArgv = plan.argv;
  telemetry.inputSnapshotDigest = plan.inputSnapshotDigest;
  telemetry.observedToolDigest = terminal.toolDigest;
  telemetry.observedOutputDigest = terminal.outputDigest;
  telemetry.workUnits = plan.argv.size();
  if (!commitOptimizationInvocationV1(token, telemetry, &diagnostic))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "invalid device action terminal: %s",
                                   diagnostic.c_str());
  if (!exactTerminal)
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   terminalError.c_str());
  if (!terminal.succeeded)
    return llvm::createStringError(
        llvm::errc::io_error,
        "device action %u failed with exit code %d", actionIndex, actionExit);
  return llvm::Error::success();
}

} // namespace

llvm::Error writeLLVMIR(const llvm::Module &module, llvm::StringRef path) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "failed to open target LLVM IR");
  module.print(output, nullptr);
  output.close();
  if (output.has_error())
    return llvm::createStringError(llvm::errc::io_error,
                                   "failed to write target LLVM IR");
  return llvm::Error::success();
}

llvm::Error runDeviceLink(const TargetToolchain &toolchain,
                          llvm::StringRef llvmIR, llvm::StringRef module,
                          llvm::StringRef object, llvm::StringRef crtObject,
                          int64_t logicalRank) {
  if (logicalRank < 0 || logicalRank >= 16)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "device action logical rank is outside the single-card domain");
  llvm::SmallString<256> stagingPrefix(module);
  stagingPrefix.append(".device-link-staging");
  llvm::SmallString<256> stagingDirectory;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          stagingPrefix, stagingDirectory))
    return llvm::createStringError(error,
                                   "failed to create device link staging");
  auto cleanup = llvm::make_scope_exit(
      [&] { (void)llvm::sys::fs::remove_directories(stagingDirectory); });

  const std::array<MechanismKey, 3> keys = {
      mechanism::DeviceObjectCompilation,
      mechanism::DeviceRuntimeCompilation,
      mechanism::DeviceGarbageCollectionLink,
  };
  for (unsigned actionIndex = 0; actionIndex < keys.size(); ++actionIndex)
    if (llvm::Error error = runDeviceLinkAction(
            toolchain.getPythonExecutable(), toolchain.getDeviceLinkerScript(),
            llvmIR, module, object, crtObject, stagingDirectory, actionIndex,
            keys[actionIndex], static_cast<uint64_t>(logicalRank)))
      return error;
  return llvm::Error::success();
}

} // namespace wafer::compiler::detail
