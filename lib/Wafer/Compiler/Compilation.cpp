//===- Compilation.cpp - Typed Wafer compiler facade --------------------===//

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"
#include "Wafer/Compiler/Testing.h"
#include "Wafer/IR/WaferDialect.h"

#include "CompilationInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler {
namespace {

detail::WholeVariantSelectionMode
getSelectionMode(testing::CollectiveCharacterizationAlgorithm algorithm) {
  using Algorithm = testing::CollectiveCharacterizationAlgorithm;
  using Mode = detail::WholeVariantSelectionMode;
  switch (algorithm) {
  case Algorithm::AllGatherDirect:
    return Mode::CharacterizeAllGatherDirect;
  case Algorithm::AllGatherRing:
    return Mode::CharacterizeAllGatherRing;
  case Algorithm::ReduceScatterDirect:
    return Mode::CharacterizeReduceScatterDirect;
  case Algorithm::ReduceScatterRing:
    return Mode::CharacterizeReduceScatterRing;
  case Algorithm::AllReduceRing:
    return Mode::CharacterizeAllReduceRing;
  case Algorithm::NoCResidentAllReduceRing:
    return Mode::QualifyNoCResidentAllReduceRing;
  case Algorithm::AllReduceTree:
    return Mode::CharacterizeAllReduceTree;
  }
  llvm_unreachable("unknown collective characterization algorithm");
}

mlir::LogicalResult writeCollectiveCharacterizationReport(
    const ExecutableBundle &bundle, detail::WholeVariantSelectionMode mode,
    llvm::StringRef reportPath, llvm::raw_ostream &diagnostics) {
  struct MessageRecord {
    std::string direction;
    int64_t peer = 0;
    int64_t communicationId = 0;
    std::string phase;
    int64_t round = 0;
    int64_t payloadSlice = 0;
    int64_t issueBytes = 0;
    int64_t constantLoopMultiplicity = 0;
    int64_t executedBytes = 0;
  };

  if (reportPath.empty()) {
    detail::reject(diagnostics,
                   "collective characterization report path must not be empty");
    return mlir::failure();
  }
  if (detail::pathEntryExists(reportPath)) {
    detail::reject(diagnostics, "refusing to replace existing collective "
                                "characterization report");
    return mlir::failure();
  }

  llvm::SmallString<256> temporaryPattern(reportPath);
  temporaryPattern += ".tmp-%%%%%%";
  int descriptor = -1;
  llvm::SmallString<256> temporaryPath;
  if (std::error_code error = llvm::sys::fs::createUniqueFile(
          temporaryPattern, descriptor, temporaryPath)) {
    detail::reject(diagnostics,
                   "failed to stage collective characterization report: " +
                       error.message());
    return mlir::failure();
  }
  auto cleanup = llvm::make_scope_exit(
      [&] { (void)llvm::sys::fs::remove(temporaryPath); });

  bool invalidAccounting = false;
  llvm::raw_fd_ostream output(descriptor, /*shouldClose=*/true);
  llvm::json::OStream json(output, /*IndentSize=*/2);
  json.object([&] {
    json.attribute("schema_version", int64_t(2));
    json.attribute("requested_alternative",
                   detail::getCollectiveCharacterizationAlternative(mode));
    json.attribute("rank_count", bundle.getExecutionConfig().getRankCount());
    json.attribute("collection_semantics",
                   "sorted-message-tuples-and-constant-loop-weighted-bytes");
    json.attributeArray("ranks", [&] {
      for (const RankExecutable &rank : bundle.getRankExecutables()) {
        std::set<std::string> phases;
        std::set<int64_t> communicationIds;
        std::set<int64_t> rounds;
        std::set<int64_t> peers;
        std::set<int64_t> payloadSlices;
        std::vector<MessageRecord> messages;
        uint64_t sendBytes = 0;
        uint64_t recvBytes = 0;
        auto record = [&](llvm::StringRef direction, auto op, uint64_t &bytes) {
          phases.insert(
              stringifyDTEProtocolPhase(op.getMessage().getPhase()).str());
          communicationIds.insert(op.getMessage().getCommunicationId());
          rounds.insert(op.getMessage().getRound());
          peers.insert(op.getPeer());
          payloadSlices.insert(op.getMessage().getPayloadSlice());
          uint64_t multiplicity = 1;
          for (mlir::Operation *parent = op->getParentOp(); parent;
               parent = parent->getParentOp()) {
            auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent);
            if (!loop)
              continue;
            std::optional<int64_t> lower = mlir::getConstantIntValue(
                mlir::getAsOpFoldResult(loop.getLowerBound()));
            std::optional<int64_t> upper = mlir::getConstantIntValue(
                mlir::getAsOpFoldResult(loop.getUpperBound()));
            std::optional<int64_t> step = mlir::getConstantIntValue(
                mlir::getAsOpFoldResult(loop.getStep()));
            if (!lower || !upper || !step || *step <= 0) {
              invalidAccounting = true;
              return;
            }
            uint64_t trips = 0;
            if (*upper > *lower) {
              uint64_t span =
                  static_cast<uint64_t>(*upper) - static_cast<uint64_t>(*lower);
              trips = 1 + (span - 1) / static_cast<uint64_t>(*step);
            }
            if (trips != 0 &&
                multiplicity > std::numeric_limits<uint64_t>::max() / trips) {
              invalidAccounting = true;
              return;
            }
            multiplicity *= trips;
          }
          uint64_t peer = op.getPeer();
          uint64_t issueBytes = op.getBytes();
          if (multiplicity == 0 || issueBytes == 0 ||
              peer >
                  static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
              issueBytes >
                  static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            invalidAccounting = true;
            return;
          }
          if (multiplicity != 0 &&
              issueBytes >
                  std::numeric_limits<uint64_t>::max() / multiplicity) {
            invalidAccounting = true;
            return;
          }
          uint64_t executedBytes = issueBytes * multiplicity;
          if (executedBytes >
                  static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
              multiplicity >
                  static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
              executedBytes > std::numeric_limits<uint64_t>::max() - bytes) {
            invalidAccounting = true;
            return;
          }
          bytes += executedBytes;
          messages.push_back(MessageRecord{
              direction.str(),
              static_cast<int64_t>(peer),
              op.getMessage().getCommunicationId(),
              stringifyDTEProtocolPhase(op.getMessage().getPhase()).str(),
              op.getMessage().getRound(),
              op.getMessage().getPayloadSlice(),
              static_cast<int64_t>(issueBytes),
              static_cast<int64_t>(multiplicity),
              static_cast<int64_t>(executedBytes),
          });
        };
        rank.getModule().walk(
            [&](InstrDTESendOp op) { record("send", op, sendBytes); });
        rank.getModule().walk(
            [&](InstrDTERecvOp op) { record("recv", op, recvBytes); });
        if (sendBytes >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            recvBytes >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
          invalidAccounting = true;
        std::sort(
            messages.begin(), messages.end(),
            [](const MessageRecord &left, const MessageRecord &right) {
              return std::tie(left.direction, left.peer, left.communicationId,
                              left.phase, left.round, left.payloadSlice,
                              left.issueBytes, left.constantLoopMultiplicity) <
                     std::tie(right.direction, right.peer,
                              right.communicationId, right.phase, right.round,
                              right.payloadSlice, right.issueBytes,
                              right.constantLoopMultiplicity);
            });

        json.object([&] {
          json.attribute("rank", rank.getLogicalRank());
          json.attributeArray("phases", [&] {
            for (const std::string &phase : phases)
              json.value(phase);
          });
          json.attributeArray("communication_ids", [&] {
            for (int64_t communicationId : communicationIds)
              json.value(communicationId);
          });
          json.attributeArray("rounds", [&] {
            for (int64_t round : rounds)
              json.value(round);
          });
          json.attributeArray("peers", [&] {
            for (int64_t peer : peers)
              json.value(peer);
          });
          json.attributeArray("payload_slices", [&] {
            for (int64_t payloadSlice : payloadSlices)
              json.value(payloadSlice);
          });
          json.attributeArray("messages", [&] {
            for (const MessageRecord &message : messages) {
              json.object([&] {
                json.attribute("direction", message.direction);
                json.attribute("peer", message.peer);
                json.attribute("communication_id", message.communicationId);
                json.attribute("phase", message.phase);
                json.attribute("round", message.round);
                json.attribute("payload_slice", message.payloadSlice);
                json.attribute("issue_bytes", message.issueBytes);
                json.attribute("constant_loop_multiplicity",
                               message.constantLoopMultiplicity);
                json.attribute("executed_bytes", message.executedBytes);
              });
            }
          });
          json.attribute("send_bytes", static_cast<int64_t>(sendBytes));
          json.attribute("recv_bytes", static_cast<int64_t>(recvBytes));
        });
      }
    });
  });
  output << "\n";
  output.close();
  if (invalidAccounting || output.has_error()) {
    detail::reject(diagnostics,
                   invalidAccounting
                       ? "collective characterization message accounting is "
                         "not statically representable"
                       : "failed to write collective characterization report");
    return mlir::failure();
  }
  if (std::error_code error =
          llvm::sys::fs::rename(temporaryPath, reportPath)) {
    detail::reject(diagnostics, "failed to atomically publish collective "
                                "characterization report: " +
                                    error.message());
    return mlir::failure();
  }
  cleanup.release();
  return mlir::success();
}

} // namespace

llvm::Expected<ExecutionConfig>
ExecutionConfig::createForSingleCard(int64_t executionRankCount,
                                     TargetProfileId targetProfile,
                                     RuntimeLaunchKind runtimeLaunchKind) {
  if (executionRankCount != 1 && executionRankCount != 16)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "execution-ranks must be exactly 1 or 16 for the single-card compiler");
  if (runtimeLaunchKind == RuntimeLaunchKind::Model && executionRankCount != 16)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "model runtime launch requires execution-ranks=16");
  if (runtimeLaunchKind == RuntimeLaunchKind::Model &&
      targetProfile != TargetProfileId::waferTx81SingleCardKernelV1())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "model runtime launch is not qualified for the selected target "
        "profile");
  return ExecutionConfig(executionRankCount, targetProfile, runtimeLaunchKind);
}

llvm::Expected<CompilationRequest>
CompilationRequest::create(llvm::StringRef sourceProgramDirectory,
                           ExecutionConfig executionConfig) {
  if (sourceProgramDirectory.empty())
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "source program directory must not be empty");
  return CompilationRequest(sourceProgramDirectory, executionConfig);
}

llvm::Expected<CompilationOptions>
CompilationOptions::profile(const ExecutionConfig &executionConfig) {
  if (executionConfig.getRankCount() != 16 ||
      executionConfig.getRuntimeLaunchKind() != RuntimeLaunchKind::Kernel)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "profile compilation requires a 16-rank kernel launch");
  return CompilationOptions(/*profileCompanion=*/true);
}

llvm::Expected<ExecutableBundle>
compileTensorProgramToExecutableBundle(llvm::StringRef tensorProgramDirectory,
                                       ExecutionConfig executionConfig,
                                       llvm::raw_ostream &diagnostics) {
  return detail::compileTensorProgramToExecutableBundleImpl(
      tensorProgramDirectory, executionConfig, diagnostics, std::nullopt,
      detail::WholeVariantSelectionMode::Production);
}

mlir::FailureOr<ExecutableBundle> compileProgram(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  return compileProgram(std::move(request), outputProgramDirectory,
                        xlaSpmdPartitionerHelper, targetToolchain,
                        CompilationOptions::standard(), diagnostics);
}

mlir::FailureOr<ExecutableBundle>
compileProgram(CompilationRequest request,
               llvm::StringRef outputProgramDirectory,
               llvm::StringRef xlaSpmdPartitionerHelper,
               const TargetToolchain &targetToolchain,
               CompilationOptions options, llvm::raw_ostream &diagnostics) {
  std::optional<ExecutableBundle> retainedExecutableBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics,
          detail::WholeVariantSelectionMode::Production, options, std::nullopt,
          std::nullopt, std::nullopt, &retainedExecutableBundle, nullptr)))
    return mlir::failure();
  if (!retainedExecutableBundle) {
    detail::reject(
        diagnostics,
        "successful compilation did not retain its executable bundle");
    return mlir::failure();
  }
  return std::move(*retainedExecutableBundle);
}

mlir::FailureOr<TargetCompilationProduct> compileProgramWithTargetLLVMBundle(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  std::optional<ExecutableBundle> retainedExecutableBundle;
  std::optional<TargetLLVMModuleBundle> retainedTargetLLVMModuleBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics,
          detail::WholeVariantSelectionMode::Production,
          CompilationOptions::standard(), std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle,
          &retainedTargetLLVMModuleBundle)))
    return mlir::failure();
  if (!retainedExecutableBundle || !retainedTargetLLVMModuleBundle) {
    detail::reject(diagnostics,
                   "successful compilation did not retain its complete target "
                   "compilation product");
    return mlir::failure();
  }
  return TargetCompilationProduct(std::move(*retainedExecutableBundle),
                                  std::move(*retainedTargetLLVMModuleBundle));
}

mlir::LogicalResult testing::compileProgramWithRankFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLogicalRank,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLogicalRank < 0 ||
      failAfterLogicalRank >= request.getExecutionConfig().getRankCount()) {
    detail::reject(diagnostics,
                   "test-only failure rank is outside ExecutionConfig");
    return mlir::failure();
  }
  return detail::runCompilationTransaction(
      std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, diagnostics,
      detail::WholeVariantSelectionMode::Production,
      CompilationOptions::standard(), failAfterLogicalRank, std::nullopt,
      std::nullopt, nullptr, nullptr);
}

mlir::LogicalResult testing::compileProgramWithTargetRankFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLogicalRank,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLogicalRank < 0 ||
      failAfterLogicalRank >= request.getExecutionConfig().getRankCount()) {
    detail::reject(diagnostics,
                   "test-only target failure rank is outside ExecutionConfig");
    return mlir::failure();
  }
  return detail::runCompilationTransaction(
      std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, diagnostics,
      detail::WholeVariantSelectionMode::Production,
      CompilationOptions::standard(), std::nullopt, failAfterLogicalRank,
      std::nullopt, nullptr, nullptr);
}

mlir::LogicalResult testing::compileProgramWithPackageRankFailure(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, int64_t failAfterLogicalRank,
    llvm::raw_ostream &diagnostics) {
  if (failAfterLogicalRank < 0 ||
      failAfterLogicalRank >= request.getExecutionConfig().getRankCount()) {
    detail::reject(diagnostics,
                   "test-only package failure rank is outside ExecutionConfig");
    return mlir::failure();
  }
  return detail::runCompilationTransaction(
      std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
      targetToolchain, diagnostics,
      detail::WholeVariantSelectionMode::Production,
      CompilationOptions::standard(), std::nullopt, std::nullopt,
      failAfterLogicalRank, nullptr, nullptr);
}

mlir::FailureOr<ExecutableBundle> testing::compileProgramWithReservedBaseline(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  std::optional<ExecutableBundle> retainedExecutableBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics,
          detail::WholeVariantSelectionMode::ReservedBaseline,
          CompilationOptions::standard(), std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle, nullptr)))
    return mlir::failure();
  if (!retainedExecutableBundle) {
    detail::reject(diagnostics,
                   "successful reserved-baseline compilation did not retain "
                   "its executable bundle");
    return mlir::failure();
  }
  return std::move(*retainedExecutableBundle);
}

mlir::FailureOr<TargetCompilationProduct>
testing::compileProgramWithReservedBaselineTargetCompilation(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  std::optional<ExecutableBundle> retainedExecutableBundle;
  std::optional<TargetLLVMModuleBundle> retainedTargetLLVMModuleBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics,
          detail::WholeVariantSelectionMode::ReservedBaseline,
          CompilationOptions::standard(), std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle,
          &retainedTargetLLVMModuleBundle)))
    return mlir::failure();
  if (!retainedExecutableBundle || !retainedTargetLLVMModuleBundle) {
    detail::reject(
        diagnostics,
        "successful reserved-baseline target compilation did not retain its "
        "complete target compilation product");
    return mlir::failure();
  }
  return TargetCompilationProduct(std::move(*retainedExecutableBundle),
                                  std::move(*retainedTargetLLVMModuleBundle));
}

mlir::FailureOr<ExecutableBundle>
testing::compileProgramForStaticFixedSlotQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  std::optional<ExecutableBundle> retainedExecutableBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics,
          detail::WholeVariantSelectionMode::QualifyStaticFixedSlot,
          CompilationOptions::standard(), std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle, nullptr)))
    return mlir::failure();
  if (!retainedExecutableBundle) {
    detail::reject(diagnostics,
                   "successful static fixed-slot qualification did not retain "
                   "its executable bundle");
    return mlir::failure();
  }
  return std::move(*retainedExecutableBundle);
}

mlir::FailureOr<TargetCompilationProduct>
testing::compileProgramForStaticFixedSlotTargetQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  std::optional<ExecutableBundle> retainedExecutableBundle;
  std::optional<TargetLLVMModuleBundle> retainedTargetLLVMModuleBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics,
          detail::WholeVariantSelectionMode::QualifyStaticFixedSlot,
          CompilationOptions::standard(), std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle,
          &retainedTargetLLVMModuleBundle)))
    return mlir::failure();
  if (!retainedExecutableBundle || !retainedTargetLLVMModuleBundle) {
    detail::reject(
        diagnostics,
        "successful static fixed-slot target qualification did not retain "
        "its complete target compilation product");
    return mlir::failure();
  }
  return TargetCompilationProduct(std::move(*retainedExecutableBundle),
                                  std::move(*retainedTargetLLVMModuleBundle));
}

mlir::FailureOr<ExecutableBundle>
testing::compileProgramForWorkerPlacementQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  std::optional<ExecutableBundle> retainedExecutableBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics,
          detail::WholeVariantSelectionMode::QualifyWorkerPlacement,
          CompilationOptions::standard(), std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle, nullptr)))
    return mlir::failure();
  if (!retainedExecutableBundle) {
    detail::reject(diagnostics,
                   "successful worker-placement qualification did not retain "
                   "its executable bundle");
    return mlir::failure();
  }
  return std::move(*retainedExecutableBundle);
}

mlir::FailureOr<TargetCompilationProduct>
testing::compileProgramForWorkerPlacementTargetQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  std::optional<ExecutableBundle> retainedExecutableBundle;
  std::optional<TargetLLVMModuleBundle> retainedTargetLLVMModuleBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics,
          detail::WholeVariantSelectionMode::QualifyWorkerPlacement,
          CompilationOptions::standard(), std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle,
          &retainedTargetLLVMModuleBundle)))
    return mlir::failure();
  if (!retainedExecutableBundle || !retainedTargetLLVMModuleBundle) {
    detail::reject(
        diagnostics,
        "successful worker-placement target qualification did not retain "
        "its complete target compilation product");
    return mlir::failure();
  }
  return TargetCompilationProduct(std::move(*retainedExecutableBundle),
                                  std::move(*retainedTargetLLVMModuleBundle));
}

mlir::FailureOr<ExecutableBundle>
testing::compileProgramForNoCResidentFixedSlotWorkerQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  std::optional<ExecutableBundle> retainedExecutableBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics,
          detail::WholeVariantSelectionMode::
              QualifyNoCResidentFixedSlotWorker,
          CompilationOptions::standard(), std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle, nullptr)))
    return mlir::failure();
  if (!retainedExecutableBundle) {
    detail::reject(
        diagnostics,
        "successful NoC-resident fixed-slot worker qualification did not "
        "retain its executable bundle");
    return mlir::failure();
  }
  return std::move(*retainedExecutableBundle);
}

mlir::FailureOr<TargetCompilationProduct>
testing::compileProgramForNoCResidentFixedSlotWorkerTargetQualification(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, llvm::raw_ostream &diagnostics) {
  std::optional<ExecutableBundle> retainedExecutableBundle;
  std::optional<TargetLLVMModuleBundle> retainedTargetLLVMModuleBundle;
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics,
          detail::WholeVariantSelectionMode::
              QualifyNoCResidentFixedSlotWorker,
          CompilationOptions::standard(), std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle,
          &retainedTargetLLVMModuleBundle)))
    return mlir::failure();
  if (!retainedExecutableBundle || !retainedTargetLLVMModuleBundle) {
    detail::reject(
        diagnostics,
        "successful NoC-resident fixed-slot worker target qualification did "
        "not retain its complete target compilation product");
    return mlir::failure();
  }
  return TargetCompilationProduct(std::move(*retainedExecutableBundle),
                                  std::move(*retainedTargetLLVMModuleBundle));
}

mlir::FailureOr<ExecutableBundle>
testing::compileProgramForCollectiveCharacterization(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain,
    CollectiveCharacterizationAlgorithm algorithm, llvm::StringRef reportPath,
    llvm::raw_ostream &diagnostics) {
  if (reportPath.empty()) {
    detail::reject(diagnostics,
                   "collective characterization report path must not be empty");
    return mlir::failure();
  }
  if (detail::pathEntryExists(reportPath)) {
    detail::reject(diagnostics, "refusing to replace existing collective "
                                "characterization report");
    return mlir::failure();
  }
  std::optional<ExecutableBundle> retainedExecutableBundle;
  detail::WholeVariantSelectionMode selectionMode = getSelectionMode(algorithm);
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics, selectionMode,
          CompilationOptions::standard(), std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle, nullptr)))
    return mlir::failure();
  if (!retainedExecutableBundle) {
    detail::reject(
        diagnostics,
        "successful collective characterization compilation did not retain "
        "its executable bundle");
    return mlir::failure();
  }
  if (mlir::failed(writeCollectiveCharacterizationReport(
          *retainedExecutableBundle, selectionMode, reportPath, diagnostics)))
    return mlir::failure();
  return std::move(*retainedExecutableBundle);
}

mlir::FailureOr<TargetCompilationProduct>
testing::compileProgramForCollectiveCharacterizationTargetCompilation(
    CompilationRequest request, llvm::StringRef outputProgramDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain,
    CollectiveCharacterizationAlgorithm algorithm, llvm::StringRef reportPath,
    llvm::raw_ostream &diagnostics) {
  if (reportPath.empty()) {
    detail::reject(diagnostics,
                   "collective characterization report path must not be empty");
    return mlir::failure();
  }
  if (detail::pathEntryExists(reportPath)) {
    detail::reject(diagnostics, "refusing to replace existing collective "
                                "characterization report");
    return mlir::failure();
  }
  std::optional<ExecutableBundle> retainedExecutableBundle;
  std::optional<TargetLLVMModuleBundle> retainedTargetLLVMModuleBundle;
  detail::WholeVariantSelectionMode selectionMode = getSelectionMode(algorithm);
  if (mlir::failed(detail::runCompilationTransaction(
          std::move(request), outputProgramDirectory, xlaSpmdPartitionerHelper,
          targetToolchain, diagnostics, selectionMode,
          CompilationOptions::standard(), std::nullopt, std::nullopt,
          std::nullopt, &retainedExecutableBundle,
          &retainedTargetLLVMModuleBundle)))
    return mlir::failure();
  if (!retainedExecutableBundle || !retainedTargetLLVMModuleBundle) {
    detail::reject(
        diagnostics,
        "successful collective characterization target compilation did not "
        "retain its complete target compilation product");
    return mlir::failure();
  }
  if (mlir::failed(writeCollectiveCharacterizationReport(
          *retainedExecutableBundle, selectionMode, reportPath, diagnostics)))
    return mlir::failure();
  return TargetCompilationProduct(std::move(*retainedExecutableBundle),
                                  std::move(*retainedTargetLLVMModuleBundle));
}

} // namespace wafer::compiler
