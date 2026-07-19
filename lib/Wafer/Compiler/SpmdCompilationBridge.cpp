//===- SpmdCompilationBridge.cpp - External SPMD helper bridge -----------===//

#include "CompilationInternal.h"

#include "Wafer/Support/OptimizationArtifactDigest.h"
#include "Wafer/Support/OptimizationMechanism.h"
#include "Wafer/Target/TargetProfile.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/Program.h"

#include <string>

namespace wafer::compiler::detail {

namespace {

static void hashU64(llvm::SHA256 &hash, uint64_t value) {
  std::array<uint8_t, 8> bytes{};
  for (unsigned index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<uint8_t>(value >> (56 - index * 8));
  hash.update(bytes);
}

static std::optional<OptimizationDigest>
digestSpmdInput(llvm::StringRef inputProgramDirectory,
                const ExecutionConfig &config) {
  std::optional<OptimizationDigest> directory =
      digestOptimizationArtifactDirectoryV1(
      inputProgramDirectory, "wafer.spmd-input-program-directory-v1");
  if (!directory)
    return std::nullopt;
  llvm::SHA256 hash;
  hash.update("wafer.spmd-invocation-input-snapshot-v1");
  const uint8_t separator = 0;
  hash.update(llvm::ArrayRef<uint8_t>(&separator, 1));
  hash.update(*directory);
  hashU64(hash, config.getRankCount());
  hash.update(stringifyTargetProfileId(config.getTargetProfileId()));
  return hash.final();
}

} // namespace

bool runSpmdHelper(llvm::StringRef helper,
                   llvm::StringRef inputProgramDirectory,
                   llvm::StringRef outputProgramDirectory,
                   const ExecutionConfig &config,
                   llvm::raw_ostream &diagnostics) {
  std::string helperStorage = helper.str();
  std::string inputStorage = inputProgramDirectory.str();
  std::string outputStorage = outputProgramDirectory.str();
  std::string rankCountStorage = std::to_string(config.getRankCount());
  llvm::SmallVector<llvm::StringRef, 9> arguments = {
      helperStorage,    "--input-program-dir",
      inputStorage,     "--output-program-dir",
      outputStorage,    "--entry-function",
      "forward",        "--logical-rank-count",
      rankCountStorage,
  };
  std::optional<OptimizationDigest> inputSnapshot =
      digestSpmdInput(inputProgramDirectory, config);
  std::optional<OptimizationDigest> toolDigest =
      digestOptimizationArtifactFileV1(helper,
                                       "wafer.observed-backend-tool-v1");
  if (!inputSnapshot || !toolDigest)
    return reject(diagnostics,
                  "cannot establish SPMD helper input/tool digest");
  OptimizationInvocationTokenV1 invocationToken;
  std::string telemetryDiagnostic;
  if (!beginOptimizationInvocationV1(
          mechanism::SpmdPartition,
          OptimizationCutPoint::SPMDPartitionTransaction,
          /*invocationOrdinal=*/0, *inputSnapshot, invocationToken,
          &telemetryDiagnostic))
    return reject(diagnostics,
                  "cannot begin SPMD optimization invocation: " +
                      telemetryDiagnostic);
  int exitCode = llvm::sys::ExecuteAndWait(helperStorage, arguments);
  OptimizationInvocationTelemetry telemetry;
  telemetry.key = mechanism::SpmdPartition;
  telemetry.cutPoint = OptimizationCutPoint::SPMDPartitionTransaction;
  telemetry.outcome =
      exitCode == 0 ? InvocationOutcome::Applied : InvocationOutcome::Invalid;
  telemetry.successfulBackendActionCount = exitCode == 0 ? 1 : 0;
  telemetry.inputSnapshotDigest = *inputSnapshot;
  telemetry.observedToolDigest = *toolDigest;
  std::optional<OptimizationDigest> outputDigest =
      digestOptimizationArtifactDirectoryV1(
          outputProgramDirectory, "wafer.spmd-output-program-directory-v1");
  if (outputDigest)
    telemetry.observedOutputDigest = *outputDigest;
  else {
    llvm::SHA256 emptyOutput;
    emptyOutput.update("wafer.spmd-missing-output-v1");
    telemetry.observedOutputDigest = emptyOutput.final();
  }
  telemetry.actionExecutor = helperStorage;
  telemetry.actionArgv.reserve(arguments.size());
  for (llvm::StringRef argument : arguments)
    telemetry.actionArgv.push_back(argument.str());
  if (!commitOptimizationInvocationV1(invocationToken, telemetry,
                                      &telemetryDiagnostic))
    return reject(diagnostics,
                  "invalid SPMD optimization telemetry: " +
                      telemetryDiagnostic);
  if (exitCode == 0)
    return false;
  std::string message = "XLA SPMD partitioner helper failed";
  if (exitCode > 0)
    message += " with exit code " + std::to_string(exitCode);
  return reject(diagnostics, message);
}

} // namespace wafer::compiler::detail
