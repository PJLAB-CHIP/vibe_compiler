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
#ifndef WAFER_DEVICE_CLANGXX
#define WAFER_DEVICE_CLANGXX ""
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
      !requireOption(options.runtimeLaunchKind, "--launch-kind"))
    return 1;
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

  llvm::Expected<wafer::RuntimeLaunchKind> runtimeLaunchKind =
      wafer::parseRuntimeLaunchKind(*options.runtimeLaunchKind);
  if (!runtimeLaunchKind) {
    llvm::errs() << "wafer-compile: "
                 << llvm::toString(runtimeLaunchKind.takeError()) << "\n";
    return 1;
  }

  llvm::Expected<wafer::compiler::ExecutionConfig> executionConfig =
      wafer::compiler::ExecutionConfig::createForSingleCard(
          rankCount, *runtimeLaunchKind);
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

  std::string helperPath = resolveSpmdPartitionerHelperPath();
  if (helperPath.empty()) {
    llvm::errs()
        << "wafer-compile: no XLA SPMD partitioner helper configured\n";
    return 1;
  }
  llvm::Expected<wafer::compiler::TargetToolchain> targetToolchain =
      wafer::compiler::TargetToolchain::create(WAFER_PYTHON_EXECUTABLE,
                                               WAFER_DEVICE_LINKER_SCRIPT,
                                               WAFER_DEVICE_CLANGXX);
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
  const char *reservedBaseline =
      std::getenv("WAFER_TEST_SELECT_RESERVED_BASELINE");
  const char *staticFixedSlot =
      std::getenv("WAFER_TEST_SELECT_STATIC_FIXED_SLOT");
  const char *directDTEComputeOverlap =
      std::getenv("WAFER_TEST_SELECT_DIRECT_DTE_COMPUTE_OVERLAP");
  const char *serializedDirectDTECompute =
      std::getenv("WAFER_TEST_SELECT_SERIALIZED_DIRECT_DTE_COMPUTE");
  const char *workerPlacement =
      std::getenv("WAFER_TEST_SELECT_WORKER_PLACEMENT");
  const char *noCResidentFixedSlotWorker =
      std::getenv("WAFER_TEST_SELECT_NOC_RESIDENT_FIXED_SLOT_WORKER");
  const char *collectiveAlternative =
      std::getenv("WAFER_TEST_COLLECTIVE_CHARACTERIZATION_ALTERNATIVE");
  const char *collectiveReport =
      std::getenv("WAFER_TEST_COLLECTIVE_CHARACTERIZATION_REPORT");
  unsigned failureInjectionCount = (failureRank ? 1u : 0u) +
                                   (targetFailureRank ? 1u : 0u) +
                                   (packageFailureRank ? 1u : 0u);
  const bool hasTestOnlyCompilationControl =
      failureInjectionCount != 0 || reservedBaseline || staticFixedSlot ||
      directDTEComputeOverlap || serializedDirectDTECompute ||
      workerPlacement || noCResidentFixedSlotWorker || collectiveAlternative ||
      collectiveReport;
  if (*optimizationConfig != wafer::OptimizationConfig::production() &&
      hasTestOnlyCompilationControl) {
    llvm::errs()
        << "wafer-compile: explicit optimization configuration cannot be "
           "combined with test-only compilation controls\n";
    return 1;
  }
  if (options.profile &&
      (failureInjectionCount != 0 || reservedBaseline || staticFixedSlot ||
       directDTEComputeOverlap || workerPlacement ||
       serializedDirectDTECompute || noCResidentFixedSlotWorker ||
       collectiveAlternative || collectiveReport)) {
    llvm::errs() << "wafer-compile: --profile cannot be combined with "
                    "test-only compilation controls\n";
    return 1;
  }
  if (failureInjectionCount > 1) {
    llvm::errs() << "wafer-compile: multiple test-only failure injections "
                    "are not allowed\n";
    return 1;
  }
  if (reservedBaseline && failureInjectionCount != 0) {
    llvm::errs()
        << "wafer-compile: test-only reserved-baseline selection cannot be "
           "combined with failure injection\n";
    return 1;
  }
  if (staticFixedSlot && failureInjectionCount != 0) {
    llvm::errs()
        << "wafer-compile: test-only static fixed-slot qualification cannot "
           "be combined with failure injection\n";
    return 1;
  }
  if (directDTEComputeOverlap && failureInjectionCount != 0) {
    llvm::errs() << "wafer-compile: test-only Direct-DTE compute-overlap "
                    "qualification cannot be combined with failure injection\n";
    return 1;
  }
  if (serializedDirectDTECompute && failureInjectionCount != 0) {
    llvm::errs() << "wafer-compile: test-only serialized Direct-DTE compute "
                    "selection cannot be combined with failure injection\n";
    return 1;
  }
  if (workerPlacement && failureInjectionCount != 0) {
    llvm::errs()
        << "wafer-compile: test-only worker-placement qualification cannot "
           "be combined with failure injection\n";
    return 1;
  }
  if (noCResidentFixedSlotWorker && failureInjectionCount != 0) {
    llvm::errs() << "wafer-compile: test-only NoC-resident fixed-slot worker "
                    "qualification cannot be combined with failure injection\n";
    return 1;
  }
  if (collectiveAlternative && failureInjectionCount != 0) {
    llvm::errs()
        << "wafer-compile: test-only collective characterization cannot be "
           "combined with failure injection\n";
    return 1;
  }
  unsigned wholeVariantSelectionCount =
      (reservedBaseline ? 1u : 0u) + (staticFixedSlot ? 1u : 0u) +
      (directDTEComputeOverlap ? 1u : 0u) +
      (serializedDirectDTECompute ? 1u : 0u) + (workerPlacement ? 1u : 0u) +
      (noCResidentFixedSlotWorker ? 1u : 0u) +
      (collectiveAlternative ? 1u : 0u);
  if (wholeVariantSelectionCount > 1) {
    llvm::errs()
        << "wafer-compile: test-only whole-variant selections are mutually "
           "exclusive\n";
    return 1;
  }
  if (static_cast<bool>(collectiveAlternative) !=
      static_cast<bool>(collectiveReport)) {
    llvm::errs()
        << "wafer-compile: test-only collective characterization requires "
           "both alternative and report path\n";
    return 1;
  }
  if (reservedBaseline && llvm::StringRef(reservedBaseline) != "1") {
    llvm::errs()
        << "wafer-compile: invalid test-only reserved-baseline selection\n";
    return 1;
  }
  if (staticFixedSlot && llvm::StringRef(staticFixedSlot) != "1") {
    llvm::errs() << "wafer-compile: invalid test-only static fixed-slot "
                    "qualification\n";
    return 1;
  }
  if (directDTEComputeOverlap &&
      llvm::StringRef(directDTEComputeOverlap) != "1") {
    llvm::errs()
        << "wafer-compile: invalid test-only Direct-DTE compute-overlap "
           "qualification\n";
    return 1;
  }
  if (serializedDirectDTECompute &&
      llvm::StringRef(serializedDirectDTECompute) != "1") {
    llvm::errs()
        << "wafer-compile: invalid test-only serialized Direct-DTE compute "
           "selection\n";
    return 1;
  }
  if (workerPlacement && llvm::StringRef(workerPlacement) != "1") {
    llvm::errs() << "wafer-compile: invalid test-only worker-placement "
                    "qualification\n";
    return 1;
  }
  if (noCResidentFixedSlotWorker &&
      llvm::StringRef(noCResidentFixedSlotWorker) != "1") {
    llvm::errs()
        << "wafer-compile: invalid test-only NoC-resident fixed-slot worker "
           "qualification\n";
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
  } else if (reservedBaseline) {
    if (options.targetModel) {
      mlir::FailureOr<wafer::compiler::TargetCompilationProduct>
          compiledProgram = wafer::compiler::testing::
              compileProgramWithReservedBaselineTargetCompilation(
                  std::move(*request), *options.outputProgramDirectory,
                  helperPath, *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        targetCompilationProduct.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    } else {
      mlir::FailureOr<wafer::compiler::ExecutableBundle> compiledProgram =
          wafer::compiler::testing::compileProgramWithReservedBaseline(
              std::move(*request), *options.outputProgramDirectory, helperPath,
              *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        executableBundle.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    }
  } else if (staticFixedSlot) {
    if (options.targetModel) {
      mlir::FailureOr<wafer::compiler::TargetCompilationProduct>
          compiledProgram = wafer::compiler::testing::
              compileProgramForStaticFixedSlotTargetQualification(
                  std::move(*request), *options.outputProgramDirectory,
                  helperPath, *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        targetCompilationProduct.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    } else {
      mlir::FailureOr<wafer::compiler::ExecutableBundle> compiledProgram =
          wafer::compiler::testing::
              compileProgramForStaticFixedSlotQualification(
                  std::move(*request), *options.outputProgramDirectory,
                  helperPath, *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        executableBundle.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    }
  } else if (directDTEComputeOverlap) {
    if (options.targetModel) {
      mlir::FailureOr<wafer::compiler::TargetCompilationProduct>
          compiledProgram = wafer::compiler::testing::
              compileProgramForDirectDTEComputeOverlapTargetQualification(
                  std::move(*request), *options.outputProgramDirectory,
                  helperPath, *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        targetCompilationProduct.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    } else {
      mlir::FailureOr<wafer::compiler::ExecutableBundle> compiledProgram =
          wafer::compiler::testing::
              compileProgramForDirectDTEComputeOverlapQualification(
                  std::move(*request), *options.outputProgramDirectory,
                  helperPath, *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        executableBundle.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    }
  } else if (serializedDirectDTECompute) {
    if (options.targetModel) {
      llvm::errs()
          << "wafer-compile: serialized Direct-DTE compute selection does "
             "not support --target-model\n";
      return 1;
    }
    mlir::FailureOr<wafer::compiler::ExecutableBundle> compiledProgram = wafer::
        compiler::testing::compileProgramForSerializedDirectDTEComputeBaseline(
            std::move(*request), *options.outputProgramDirectory, helperPath,
            *targetToolchain, llvm::errs());
    if (mlir::succeeded(compiledProgram)) {
      executableBundle.emplace(std::move(*compiledProgram));
      compilationStatus = mlir::success();
    }
  } else if (workerPlacement) {
    if (options.targetModel) {
      mlir::FailureOr<wafer::compiler::TargetCompilationProduct>
          compiledProgram = wafer::compiler::testing::
              compileProgramForWorkerPlacementTargetQualification(
                  std::move(*request), *options.outputProgramDirectory,
                  helperPath, *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        targetCompilationProduct.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    } else {
      mlir::FailureOr<wafer::compiler::ExecutableBundle> compiledProgram =
          wafer::compiler::testing::
              compileProgramForWorkerPlacementQualification(
                  std::move(*request), *options.outputProgramDirectory,
                  helperPath, *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        executableBundle.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    }
  } else if (noCResidentFixedSlotWorker) {
    if (options.targetModel) {
      mlir::FailureOr<wafer::compiler::TargetCompilationProduct>
          compiledProgram = wafer::compiler::testing::
              compileProgramForNoCResidentFixedSlotWorkerTargetQualification(
                  std::move(*request), *options.outputProgramDirectory,
                  helperPath, *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        targetCompilationProduct.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    } else {
      mlir::FailureOr<wafer::compiler::ExecutableBundle> compiledProgram =
          wafer::compiler::testing::
              compileProgramForNoCResidentFixedSlotWorkerQualification(
                  std::move(*request), *options.outputProgramDirectory,
                  helperPath, *targetToolchain, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        executableBundle.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    }
  } else if (collectiveAlternative) {
    using Algorithm =
        wafer::compiler::testing::CollectiveCharacterizationAlgorithm;
    std::optional<Algorithm> algorithm;
    llvm::StringRef alternative(collectiveAlternative);
    if (alternative == "all-gather-direct")
      algorithm = Algorithm::AllGatherDirect;
    else if (alternative == "all-gather-ring")
      algorithm = Algorithm::AllGatherRing;
    else if (alternative == "reduce-scatter-direct")
      algorithm = Algorithm::ReduceScatterDirect;
    else if (alternative == "reduce-scatter-ring")
      algorithm = Algorithm::ReduceScatterRing;
    else if (alternative == "all-reduce-ring")
      algorithm = options.targetModel ? Algorithm::NoCResidentAllReduceRing
                                      : Algorithm::AllReduceRing;
    else if (alternative == "all-reduce-tree")
      algorithm = Algorithm::AllReduceTree;
    if (!algorithm) {
      llvm::errs()
          << "wafer-compile: invalid test-only collective characterization "
             "alternative\n";
      return 1;
    }
    if (options.targetModel) {
      mlir::FailureOr<wafer::compiler::TargetCompilationProduct>
          compiledProgram = wafer::compiler::testing::
              compileProgramForCollectiveCharacterizationTargetCompilation(
                  std::move(*request), *options.outputProgramDirectory,
                  helperPath, *targetToolchain, *algorithm, collectiveReport,
                  llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        targetCompilationProduct.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    } else {
      mlir::FailureOr<wafer::compiler::ExecutableBundle> compiledProgram =
          wafer::compiler::testing::compileProgramForCollectiveCharacterization(
              std::move(*request), *options.outputProgramDirectory, helperPath,
              *targetToolchain, *algorithm, collectiveReport, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        executableBundle.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    }
  } else
#endif
  {
    if (options.targetModel || options.compilerIRDumpDirectory) {
      mlir::FailureOr<wafer::compiler::TargetCompilationProduct>
          compiledProgram = wafer::compiler::compileProgramWithTargetLLVMBundle(
              std::move(*request), *options.outputProgramDirectory, helperPath,
              *targetToolchain, compilationOptions, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        targetCompilationProduct.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    } else {
      mlir::FailureOr<wafer::compiler::ExecutableBundle> compiledProgram =
          wafer::compiler::compileProgram(
              std::move(*request), *options.outputProgramDirectory, helperPath,
              *targetToolchain, compilationOptions, llvm::errs());
      if (mlir::succeeded(compiledProgram)) {
        executableBundle.emplace(std::move(*compiledProgram));
        compilationStatus = mlir::success();
      }
    }
  }
  if (mlir::failed(compilationStatus))
    return 1;

  if (options.compilerIRDumpDirectory) {
    if (!targetCompilationProduct ||
        !dumpCompilerIR(*options.compilerIRDumpDirectory,
                        *targetCompilationProduct, llvm::errs()))
      return 1;
    llvm::outs() << "wafer-compile: dumped compiler IR: "
                 << *options.compilerIRDumpDirectory << "\n";
  }

  llvm::outs() << "wafer-compile: published verified package with "
                  "execution-ranks="
               << rankCount << ": " << *options.outputProgramDirectory << "\n";
  if (options.profile)
    llvm::outs() << "wafer-compile: published profile companion: "
                 << *options.outputProgramDirectory << ".profile\n";
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
