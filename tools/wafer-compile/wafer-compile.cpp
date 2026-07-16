//===- wafer-compile.cpp - Wafer user compiler driver --------------------===//

#include "DriverInternal.h"

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/TargetArtifact.h"
#ifdef WAFER_ENABLE_SYSTEMC_MODEL
#include "Wafer/Model/SystemCTargetModel.h"
#endif
#ifdef WAFER_ENABLE_TARGET_BULK_MODEL
#include "Wafer/Model/TargetBulkModel.h"
#endif
#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
#include "Wafer/Compiler/Testing.h"
#endif

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#ifndef WAFER_PYTHON_EXECUTABLE
#define WAFER_PYTHON_EXECUTABLE ""
#endif
#ifndef WAFER_DEVICE_LINKER_SCRIPT
#define WAFER_DEVICE_LINKER_SCRIPT ""
#endif

using namespace wafer::compile_driver;

#ifdef WAFER_ENABLE_SYSTEMC_MODEL
extern "C" int sc_main(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
  if (argc == 2 && llvm::StringRef(argv[1]) == "--help") {
    printHelp();
    return 0;
  }

  CommandLineOptions options;
  if (!parseCommandLine(argc, argv, options))
    return 1;
  if (!requireOption(options.inputProgramDirectory, "--input-program-dir") ||
      !requireOption(options.outputProgramDirectory, "--output-program-dir") ||
      !requireOption(options.executionRanks, "--execution-ranks") ||
      !requireOption(options.targetProfile, "--target-profile"))
    return 1;

  bool modelInvocationRequested = !options.modelInputs.empty() ||
                                  !options.modelExpected.empty() ||
                                  options.modelAtol || options.modelRtol;
  const bool targetModelOptionsProvided =
      options.targetModelMaximumScalarEvaluations ||
      options.targetModelMaximumFusedMultiplyAdds ||
      options.targetModelMaximumMovementBytes ||
      options.targetModelMaximumMovementSegments ||
      options.targetModelNumericPolicy ||
      !options.targetModelBulkRecords.empty() ||
      options.targetModelMaximumBulkTotalBytes ||
      options.targetModelMaximumBulkScratchpadBytes ||
      options.targetModelMaximumBulkReorderBytes;
  if (targetModelOptionsProvided && !options.targetModel) {
    llvm::errs() << "wafer-compile: target model budget options require "
                    "--target-model\n";
    return 1;
  }
  if (options.targetModel && !modelInvocationRequested) {
    llvm::errs() << "wafer-compile: target model execution requires complete "
                    "--model-input and --model-expected bindings\n";
    return 1;
  }
  if (modelInvocationRequested && !options.targetModel) {
    llvm::errs() << "wafer-compile: model input/expected options require "
                    "--target-model\n";
    return 1;
  }
#ifndef WAFER_ENABLE_SYSTEMC_MODEL
  if (options.targetModel) {
    llvm::errs() << "wafer-compile: target model support is not configured\n";
    return 1;
  }
#endif
  if (modelInvocationRequested &&
      (options.modelInputs.empty() || options.modelExpected.empty())) {
    llvm::errs() << "wafer-compile: target model execution requires both "
                    "--model-input and --model-expected\n";
    return 1;
  }
  std::optional<double> modelAtol =
      parseTolerance(options.modelAtol, "--model-atol", 0.0);
  std::optional<double> modelRtol =
      parseTolerance(options.modelRtol, "--model-rtol", 0.0);
  if (!modelAtol || !modelRtol)
    return 1;
  auto modelInputs = parseIndexedPaths(options.modelInputs, "--model-input");
  auto modelExpected =
      parseIndexedPaths(options.modelExpected, "--model-expected");
  if (!modelInputs || !modelExpected)
    return 1;

#ifdef WAFER_ENABLE_SYSTEMC_MODEL
  std::optional<wafer::model::TargetModelKernelBudget> targetModelBudget;
  std::optional<wafer::model::TargetModelExecutionPolicy>
      targetModelExecutionPolicy;
#ifdef WAFER_ENABLE_TARGET_BULK_MODEL
  std::unique_ptr<wafer::model::TargetModelBulkBackend> targetModelBulkBackend;
  std::unique_ptr<wafer::model::ManagedReferenceTargetModelBackend>
      targetModelManagedReferenceBackend;
#endif
  if (options.targetModel) {
    auto scalar =
        parsePositiveCount(options.targetModelMaximumScalarEvaluations,
                           "--target-model-max-scalar-evaluations");
    auto fusedMultiplyAdds =
        parsePositiveCount(options.targetModelMaximumFusedMultiplyAdds,
                           "--target-model-max-fused-multiply-adds");
    auto movementBytes =
        parsePositiveCount(options.targetModelMaximumMovementBytes,
                           "--target-model-max-movement-bytes");
    auto movementSegments =
        parsePositiveCount(options.targetModelMaximumMovementSegments,
                           "--target-model-max-movement-segments");
    if (!scalar || !fusedMultiplyAdds || !movementBytes || !movementSegments)
      return 1;
    targetModelBudget.emplace(wafer::model::TargetModelKernelBudget::create(
        wafer::FormalNumericWorkBudget::create(*scalar, *fusedMultiplyAdds),
        *movementBytes, *movementSegments));

    llvm::StringRef numericPolicy =
        options.targetModelNumericPolicy
            ? llvm::StringRef(*options.targetModelNumericPolicy)
            : llvm::StringRef("formal");
    const bool hasBulkConfiguration =
        !options.targetModelBulkRecords.empty() ||
        options.targetModelMaximumBulkTotalBytes ||
        options.targetModelMaximumBulkScratchpadBytes ||
        options.targetModelMaximumBulkReorderBytes;
    if (numericPolicy == "formal") {
      if (hasBulkConfiguration) {
        llvm::errs() << "wafer-compile: bulk model options require "
                        "--target-model-numeric-policy=prefer-admitted or "
                        "managed-reference\n";
        return 1;
      }
      targetModelExecutionPolicy.emplace(
          wafer::model::TargetModelExecutionPolicy::formalOnly());
    } else if (numericPolicy == "prefer-admitted" ||
               numericPolicy == "managed-reference") {
#ifdef WAFER_ENABLE_TARGET_BULK_MODEL
      if (numericPolicy == "prefer-admitted" &&
          options.targetModelBulkRecords.empty()) {
        llvm::errs() << "wafer-compile: prefer-admitted GEMM requires at "
                        "least one --target-model-bulk-record\n";
        return 1;
      }
      if (numericPolicy == "managed-reference" &&
          !options.targetModelBulkRecords.empty()) {
        llvm::errs() << "wafer-compile: managed-reference GEMM does not "
                        "consume exact --target-model-bulk-record entries\n";
        return 1;
      }
      auto maximumTotalBytes =
          parsePositiveCount(options.targetModelMaximumBulkTotalBytes,
                             "--target-model-max-bulk-total-bytes");
      auto maximumScratchpadBytes =
          parsePositiveCount(options.targetModelMaximumBulkScratchpadBytes,
                             "--target-model-max-bulk-scratchpad-bytes");
      auto maximumReorderBytes =
          parsePositiveCount(options.targetModelMaximumBulkReorderBytes,
                             "--target-model-max-bulk-reorder-bytes");
      if (!maximumTotalBytes || !maximumScratchpadBytes || !maximumReorderBytes)
        return 1;
      const wafer::BulkNumericWorkBudget bulkBudget =
          wafer::BulkNumericWorkBudget::create(*maximumTotalBytes,
                                               *maximumScratchpadBytes,
                                               *maximumReorderBytes);
      if (numericPolicy == "prefer-admitted") {
        auto qualified = wafer::model::QualifiedTargetModelBulkBackend::create(
            options.targetModelBulkRecords, bulkBudget);
        if (!qualified) {
          llvm::errs() << "wafer-compile: "
                       << llvm::toString(qualified.takeError()) << "\n";
          return 1;
        }
        targetModelBulkBackend = std::move(*qualified);
        targetModelExecutionPolicy.emplace(
            wafer::model::TargetModelExecutionPolicy::preferAdmitted(
                *targetModelBulkBackend));
      } else {
        auto managed = wafer::model::ManagedReferenceTargetModelBackend::create(
            bulkBudget);
        if (!managed) {
          llvm::errs() << "wafer-compile: "
                       << llvm::toString(managed.takeError()) << "\n";
          return 1;
        }
        targetModelManagedReferenceBackend = std::move(*managed);
        targetModelExecutionPolicy.emplace(
            wafer::model::TargetModelExecutionPolicy::managedReference(
                *targetModelManagedReferenceBackend,
                *targetModelManagedReferenceBackend));
      }
#else
      llvm::errs() << "wafer-compile: bulk model support is not "
                      "configured\n";
      return 1;
#endif
    } else {
      llvm::errs() << "wafer-compile: invalid "
                      "--target-model-numeric-policy value: "
                   << numericPolicy << "\n";
      return 1;
    }
  }
#endif

  int64_t rankCount = 0;
  llvm::StringRef rankValue(*options.executionRanks);
  if (rankValue.getAsInteger(10, rankCount)) {
    llvm::errs() << "wafer-compile: invalid --execution-ranks value: "
                 << rankValue << "\n";
    return 1;
  }

  llvm::Expected<wafer::TargetProfileId> targetProfile =
      wafer::parseTargetProfileId(*options.targetProfile);
  if (!targetProfile) {
    llvm::errs() << "wafer-compile: "
                 << llvm::toString(targetProfile.takeError()) << "\n";
    return 1;
  }

  llvm::Expected<wafer::compiler::ExecutionConfig> executionConfig =
      wafer::compiler::ExecutionConfig::createForSingleCard(rankCount,
                                                            *targetProfile);
  if (!executionConfig) {
    llvm::errs() << "wafer-compile: "
                 << llvm::toString(executionConfig.takeError()) << "\n";
    return 1;
  }

  llvm::Expected<wafer::compiler::CompilationRequest> request =
      wafer::compiler::CompilationRequest::create(
          *options.inputProgramDirectory, std::move(*executionConfig));
  if (!request) {
    llvm::errs() << "wafer-compile: " << llvm::toString(request.takeError())
                 << "\n";
    return 1;
  }

  std::string helperPath = resolveSpmdPartitionerHelperPath();
  if (helperPath.empty()) {
    llvm::errs()
        << "wafer-compile: no XLA SPMD partitioner helper configured\n";
    return 1;
  }
  llvm::Expected<wafer::compiler::TargetToolchain> targetToolchain =
      wafer::compiler::TargetToolchain::create(WAFER_PYTHON_EXECUTABLE,
                                               WAFER_DEVICE_LINKER_SCRIPT);
  if (!targetToolchain) {
    llvm::errs() << "wafer-compile: "
                 << llvm::toString(targetToolchain.takeError()) << "\n";
    return 1;
  }

  mlir::LogicalResult compilationStatus = mlir::failure();
  std::optional<wafer::compiler::ExecutableBundle> executableBundle;
  std::optional<wafer::compiler::TargetCompilationProduct>
      targetCompilationProduct;
#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
  const char *failureRank = std::getenv("WAFER_TEST_FAIL_AFTER_LOGICAL_RANK");
  const char *targetFailureRank =
      std::getenv("WAFER_TEST_FAIL_AFTER_TARGET_LOGICAL_RANK");
  const char *packageFailureRank =
      std::getenv("WAFER_TEST_FAIL_AFTER_PACKAGE_LOGICAL_RANK");
  unsigned failureInjectionCount = (failureRank ? 1u : 0u) +
                                   (targetFailureRank ? 1u : 0u) +
                                   (packageFailureRank ? 1u : 0u);
  if (failureInjectionCount > 1) {
    llvm::errs() << "wafer-compile: multiple test-only failure injections "
                    "are not allowed\n";
    return 1;
  }
  if (failureRank) {
    int64_t parsedFailureRank = -1;
    if (llvm::StringRef(failureRank).getAsInteger(10, parsedFailureRank)) {
      llvm::errs() << "wafer-compile: invalid test-only failure rank\n";
      return 1;
    }
    compilationStatus = wafer::compiler::testing::compileProgramWithRankFailure(
        std::move(*request), *options.outputProgramDirectory, helperPath,
        *targetToolchain, parsedFailureRank, llvm::errs());
  } else if (targetFailureRank) {
    int64_t parsedFailureRank = -1;
    if (llvm::StringRef(targetFailureRank)
            .getAsInteger(10, parsedFailureRank)) {
      llvm::errs() << "wafer-compile: invalid test-only target failure rank\n";
      return 1;
    }
    compilationStatus =
        wafer::compiler::testing::compileProgramWithTargetRankFailure(
            std::move(*request), *options.outputProgramDirectory, helperPath,
            *targetToolchain, parsedFailureRank, llvm::errs());
  } else if (packageFailureRank) {
    int64_t parsedFailureRank = -1;
    if (llvm::StringRef(packageFailureRank)
            .getAsInteger(10, parsedFailureRank)) {
      llvm::errs() << "wafer-compile: invalid test-only package failure rank\n";
      return 1;
    }
    compilationStatus =
        wafer::compiler::testing::compileProgramWithPackageRankFailure(
            std::move(*request), *options.outputProgramDirectory, helperPath,
            *targetToolchain, parsedFailureRank, llvm::errs());
  } else
#endif
  {
    if (options.targetModel) {
      mlir::FailureOr<wafer::compiler::TargetCompilationProduct>
          compiledProgram = wafer::compiler::compileProgramWithTargetLLVMBundle(
              std::move(*request), *options.outputProgramDirectory, helperPath,
              *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        targetCompilationProduct.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    } else {
      mlir::FailureOr<wafer::compiler::ExecutableBundle> compiledProgram =
          wafer::compiler::compileProgram(
              std::move(*request), *options.outputProgramDirectory, helperPath,
              *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        executableBundle.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    }
  }
  if (mlir::failed(compilationStatus))
    return 1;

  llvm::outs() << "wafer-compile: published verified package with "
                  "execution-ranks="
               << rankCount << ": " << *options.outputProgramDirectory << "\n";
#ifdef WAFER_ENABLE_SYSTEMC_MODEL
  if (options.targetModel) {
    llvm::outs().flush();
    if (!targetCompilationProduct || !targetModelBudget ||
        !targetModelExecutionPolicy) {
      llvm::errs() << "wafer-compile: target model compilation product or "
                      "budget is missing\n";
      return 1;
    }
    if (runTargetModelGate(options, *targetCompilationProduct, *modelInputs,
                           *modelExpected, *modelAtol, *modelRtol,
                           *targetModelBudget, *targetModelExecutionPolicy))
      return 1;
  }
#endif
  return 0;
}
