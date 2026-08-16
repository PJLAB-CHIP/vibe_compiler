//===- wafer-compile.cpp - Wafer user compiler driver --------------------===//

#include "DriverInternal.h"

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/Package.h"
#include "Wafer/Compiler/TargetCodeGen.h"
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
#include <optional>
#include <string>
#include <utility>

using namespace wafer::compile_driver;

namespace {

/// Renders a failed typed compilation to the CLI status: the detailed
/// diagnostics already went to stderr during the transaction, so the CLI only
/// classifies the failure and exits non-zero without touching any output.
int reportCompilationFailure(llvm::Error error) {
  llvm::errs() << "wafer-compile: " << llvm::toString(std::move(error))
               << "\n";
  return 1;
}

int reportSuccess(const wafer::compiler::CompilationResult &result,
                  int64_t numPartitions) {
  llvm::outs() << "wafer-compile: wrote verified package with "
                  "num-partitions="
               << numPartitions << " tiles="
               << wafer::compiler::ExecutionConfig::kSingleCardTileCount
               << ": " << result.getPackage().getRootDirectory() << "\n";
  if (result.getProfileInstrumentation())
    llvm::outs() << "wafer-compile: wrote profile instrumentation: "
                 << result.getProfileInstrumentation()->getRootDirectory()
                 << "\n";
  return 0;
}

} // namespace

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
      !requireOption(options.outputPackageDirectory, "--output-package-dir") ||
      !requireOption(options.numPartitions, "--num-partitions"))
    return 1;

#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
  if (options.profile && options.targetModel) {
    llvm::errs()
        << "wafer-compile: --profile cannot be combined with --target-model\n";
    return 1;
  }
  bool modelInvocationRequested = !options.modelInputs.empty() ||
                                  !options.modelExpected.empty() ||
                                  options.modelAtol || options.modelRtol;
  const bool targetModelOptionsProvided =
      options.targetModelMaximumScalarEvaluations ||
      options.targetModelMaximumFusedMultiplyAdds ||
      options.targetModelMaximumMovementBytes ||
      options.targetModelMaximumMovementSegments ||
      options.modelReportNumericStatistics ||
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
#endif

  int64_t numPartitions = 0;
  llvm::StringRef partitionValue(*options.numPartitions);
  if (partitionValue.getAsInteger(10, numPartitions)) {
    llvm::errs() << "wafer-compile: invalid --num-partitions value: "
                 << partitionValue << "\n";
    return 1;
  }

  llvm::Expected<wafer::compiler::ExecutionConfig> executionConfig =
      wafer::compiler::ExecutionConfig::createForSingleCard(numPartitions);
  if (!executionConfig) {
    llvm::errs() << "wafer-compile: "
                 << llvm::toString(executionConfig.takeError()) << "\n";
    return 1;
  }
  std::optional<wafer::OptimizationConfig> optimizationConfig =
      parseOptimizationConfig(options);
  if (!optimizationConfig)
    return 1;
  const auto timingMode =
      options.compileTiming ? wafer::compiler::CompilationTimingMode::Detailed
                            : wafer::compiler::CompilationTimingMode::Disabled;
  wafer::compiler::CompilationOptions compilationOptions =
      wafer::compiler::CompilationOptions::standard(*optimizationConfig,
                                                    timingMode);
  if (options.profile) {
    llvm::Expected<wafer::compiler::CompilationOptions> profileOptions =
        wafer::compiler::CompilationOptions::profile(
            *executionConfig, *optimizationConfig, timingMode);
    if (!profileOptions) {
      llvm::errs() << "wafer-compile: "
                   << llvm::toString(profileOptions.takeError()) << "\n";
      return 1;
    }
    compilationOptions = *profileOptions;
  }

  llvm::Expected<wafer::compiler::CompilationRequest> request =
      wafer::compiler::CompilationRequest::create(
          *options.inputProgramDirectory, std::move(*executionConfig));
  if (!request) {
    llvm::errs() << "wafer-compile: " << llvm::toString(request.takeError())
                 << "\n";
    return 1;
  }

  llvm::Expected<DriverToolFacts> toolFacts = resolveDriverToolFacts();
  if (!toolFacts) {
    llvm::errs() << "wafer-compile: "
                 << llvm::toString(toolFacts.takeError()) << "\n";
    return 1;
  }
  llvm::Expected<wafer::compiler::TargetToolchain> targetToolchain =
      wafer::compiler::TargetToolchain::create(
          toolFacts->pythonExecutable, toolFacts->deviceLinkerScript,
          toolFacts->llvmClangXX, toolFacts->tx8DepsRoot,
          toolFacts->waferIncludeDir, toolFacts->waferCrtSource,
          toolFacts->waferCrtIncludeDir);
  if (!targetToolchain) {
    llvm::errs() << "wafer-compile: "
                 << llvm::toString(targetToolchain.takeError()) << "\n";
    return 1;
  }

#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
  const char *executableFailureSlot =
      std::getenv("WAFER_TEST_FAIL_AFTER_EXECUTABLE_LAUNCH_SLOT");
  const char *targetFailureSlot =
      std::getenv("WAFER_TEST_FAIL_AFTER_TARGET_LAUNCH_SLOT");
  const char *packageFailureSlot =
      std::getenv("WAFER_TEST_FAIL_AFTER_PACKAGE_LAUNCH_SLOT");
  unsigned failureInjectionCount = (executableFailureSlot ? 1u : 0u) +
                                   (targetFailureSlot ? 1u : 0u) +
                                   (packageFailureSlot ? 1u : 0u);
  if (*optimizationConfig != wafer::OptimizationConfig::search() &&
      failureInjectionCount != 0) {
    llvm::errs()
        << "wafer-compile: explicit optimization configuration cannot be "
           "combined with test-only compilation controls\n";
    return 1;
  }
  if (options.profile && failureInjectionCount != 0) {
    llvm::errs() << "wafer-compile: --profile cannot be combined with "
                    "test-only compilation controls\n";
    return 1;
  }
  if (failureInjectionCount > 1) {
    llvm::errs() << "wafer-compile: multiple test-only failure injections "
                    "are not allowed\n";
    return 1;
  }
  if (executableFailureSlot) {
    int64_t parsedFailureSlot = -1;
    if (llvm::StringRef(executableFailureSlot)
            .getAsInteger(10, parsedFailureSlot)) {
      llvm::errs()
          << "wafer-compile: invalid test-only executable launch slot\n";
      return 1;
    }
    mlir::LogicalResult injectionStatus =
        wafer::compiler::testing::compileProgramWithExecutableLaunchSlotFailure(
            std::move(*request), *options.outputPackageDirectory,
            toolFacts->spmdPartitionerHelper, *targetToolchain,
            parsedFailureSlot, llvm::errs());
    if (mlir::failed(injectionStatus))
      return 1;
    llvm::outs() << "wafer-compile: wrote verified package with "
                    "num-partitions="
                 << numPartitions << " tiles="
                 << wafer::compiler::ExecutionConfig::kSingleCardTileCount
                 << ": " << *options.outputPackageDirectory << "\n";
    return 0;
  }
  if (targetFailureSlot) {
    int64_t parsedFailureSlot = -1;
    if (llvm::StringRef(targetFailureSlot)
            .getAsInteger(10, parsedFailureSlot)) {
      llvm::errs() << "wafer-compile: invalid test-only target launch slot\n";
      return 1;
    }
    mlir::LogicalResult injectionStatus =
        wafer::compiler::testing::compileProgramWithTargetLaunchSlotFailure(
            std::move(*request), *options.outputPackageDirectory,
            toolFacts->spmdPartitionerHelper, *targetToolchain,
            parsedFailureSlot, llvm::errs());
    if (mlir::failed(injectionStatus))
      return 1;
    llvm::outs() << "wafer-compile: wrote verified package with "
                    "num-partitions="
                 << numPartitions << " tiles="
                 << wafer::compiler::ExecutionConfig::kSingleCardTileCount
                 << ": " << *options.outputPackageDirectory << "\n";
    return 0;
  }
  if (packageFailureSlot) {
    int64_t parsedFailureSlot = -1;
    if (llvm::StringRef(packageFailureSlot)
            .getAsInteger(10, parsedFailureSlot)) {
      llvm::errs() << "wafer-compile: invalid test-only package launch slot\n";
      return 1;
    }
    mlir::LogicalResult injectionStatus =
        wafer::compiler::testing::compileProgramWithPackageLaunchSlotFailure(
            std::move(*request), *options.outputPackageDirectory,
            toolFacts->spmdPartitionerHelper, *targetToolchain,
            parsedFailureSlot, llvm::errs());
    if (mlir::failed(injectionStatus))
      return 1;
    llvm::outs() << "wafer-compile: wrote verified package with "
                    "num-partitions="
                 << numPartitions << " tiles="
                 << wafer::compiler::ExecutionConfig::kSingleCardTileCount
                 << ": " << *options.outputPackageDirectory << "\n";
    return 0;
  }
  if (options.targetModel || options.compilerIRDumpDirectory) {
    llvm::Expected<wafer::compiler::CompiledProgram> compiledProgram =
        wafer::compiler::compileProgramWithTargetLLVMModules(
            std::move(*request), *options.outputPackageDirectory,
            toolFacts->spmdPartitionerHelper, *targetToolchain,
            compilationOptions, llvm::errs());
    if (!compiledProgram)
      return reportCompilationFailure(compiledProgram.takeError());
    if (options.compilerIRDumpDirectory) {
      if (!dumpCompilerIR(*options.compilerIRDumpDirectory, *compiledProgram,
                          llvm::errs()))
        return 1;
      llvm::outs() << "wafer-compile: dumped compiler IR: "
                   << *options.compilerIRDumpDirectory << "\n";
    }
    llvm::outs() << "wafer-compile: wrote verified package with "
                    "num-partitions="
                 << numPartitions << " tiles="
                 << wafer::compiler::ExecutionConfig::kSingleCardTileCount
                 << ": " << *options.outputPackageDirectory << "\n";
#ifdef WAFER_ENABLE_SYSTEMC_MODEL
    if (options.targetModel) {
      std::optional<wafer::model::TargetModelKernelBudget> targetModelBudget;
      std::optional<wafer::model::TargetModelExecutionPolicy>
          targetModelExecutionPolicy;
#ifdef WAFER_ENABLE_TARGET_BULK_MODEL
      std::unique_ptr<wafer::model::TargetModelBulkBackend>
          targetModelBulkBackend;
      std::unique_ptr<wafer::model::ManagedReferenceTargetModelBackend>
          targetModelManagedReferenceBackend;
#endif
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
                          "--target-model-numeric-policy=bulk-then-formal or "
                          "managed-reference\n";
          return 1;
        }
        targetModelExecutionPolicy.emplace(
            wafer::model::TargetModelExecutionPolicy::formalOnly());
      } else if (numericPolicy == "bulk-then-formal" ||
                 numericPolicy == "managed-reference") {
#ifdef WAFER_ENABLE_TARGET_BULK_MODEL
        if (numericPolicy == "bulk-then-formal" &&
            options.targetModelBulkRecords.empty()) {
          llvm::errs() << "wafer-compile: bulk-then-formal GEMM requires at "
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
        if (!maximumTotalBytes || !maximumScratchpadBytes ||
            !maximumReorderBytes)
          return 1;
        const wafer::BulkNumericWorkBudget bulkBudget =
            wafer::BulkNumericWorkBudget::create(*maximumTotalBytes,
                                                 *maximumScratchpadBytes,
                                                 *maximumReorderBytes);
        if (numericPolicy == "bulk-then-formal") {
          auto qualified = wafer::model::QualifiedTargetModelBulkBackend::create(
              options.targetModelBulkRecords, bulkBudget);
          if (!qualified) {
            llvm::errs() << "wafer-compile: "
                         << llvm::toString(qualified.takeError()) << "\n";
            return 1;
          }
          targetModelBulkBackend = std::move(*qualified);
          targetModelExecutionPolicy.emplace(
              wafer::model::TargetModelExecutionPolicy::bulkThenFormal(
                  *targetModelBulkBackend));
        } else {
          auto managed =
              wafer::model::ManagedReferenceTargetModelBackend::create(
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
      llvm::outs().flush();
      if (!targetModelBudget || !targetModelExecutionPolicy) {
        llvm::errs() << "wafer-compile: compiled program or "
                        "budget is missing\n";
        return 1;
      }
      if (runTargetModelGate(options, *compiledProgram, *modelInputs,
                             *modelExpected, *modelAtol, *modelRtol,
                             *targetModelBudget, *targetModelExecutionPolicy))
        return 1;
    }
#endif
    return 0;
  }
#endif

  llvm::Expected<wafer::compiler::CompilationResult> result =
      wafer::compiler::compileProgram(
          std::move(*request), *options.outputPackageDirectory,
          toolFacts->spmdPartitionerHelper, *targetToolchain,
          compilationOptions, llvm::errs());
  if (!result)
    return reportCompilationFailure(result.takeError());
  return reportSuccess(*result, numPartitions);
}
