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

  bool referenceRequested = !options.referenceInputs.empty() ||
                            !options.referenceExpected.empty() ||
                            options.referenceAtol || options.referenceRtol;
  const bool targetModelOptionsProvided =
      options.targetModelMaximumScalarEvaluations ||
      options.targetModelMaximumFusedMultiplyAdds ||
      options.targetModelMaximumMovementBytes ||
      options.targetModelMaximumMovementSegments ||
      options.targetModelGemmBackend || options.targetModelOracle ||
      !options.targetModelBulkRecords.empty() ||
      options.targetModelMaximumBulkTotalBytes ||
      options.targetModelMaximumBulkScratchpadBytes ||
      options.targetModelMaximumBulkReorderBytes;
  if (targetModelOptionsProvided && !options.targetModel) {
    llvm::errs() << "wafer-compile: target model budget options require "
                    "--target-model\n";
    return 1;
  }
  if (options.targetModel && !referenceRequested) {
    llvm::errs() << "wafer-compile: target model execution requires complete "
                    "--reference-input and --reference-expected bindings\n";
    return 1;
  }
  llvm::StringRef targetModelOracle =
      options.targetModelOracle ? llvm::StringRef(*options.targetModelOracle)
                                : llvm::StringRef("reference");
  if (options.targetModelOracle && !options.targetModel) {
    llvm::errs() << "wafer-compile: --target-model-oracle requires "
                    "--target-model\n";
    return 1;
  }
  if (targetModelOracle != "reference" && targetModelOracle != "external") {
    llvm::errs() << "wafer-compile: invalid --target-model-oracle value: "
                 << targetModelOracle << "\n";
    return 1;
  }
#ifndef WAFER_ENABLE_SYSTEMC_MODEL
  if (options.targetModel) {
    llvm::errs() << "wafer-compile: target model support is not configured\n";
    return 1;
  }
#endif
  if (referenceRequested &&
      (options.referenceInputs.empty() || options.referenceExpected.empty())) {
    llvm::errs() << "wafer-compile: reference execution requires both "
                    "--reference-input and --reference-expected\n";
    return 1;
  }
  std::optional<double> referenceAtol =
      parseTolerance(options.referenceAtol, "--reference-atol", 0.0);
  std::optional<double> referenceRtol =
      parseTolerance(options.referenceRtol, "--reference-rtol", 0.0);
  if (!referenceAtol || !referenceRtol)
    return 1;
  auto referenceInputs =
      parseIndexedPaths(options.referenceInputs, "--reference-input");
  auto referenceExpected =
      parseIndexedPaths(options.referenceExpected, "--reference-expected");
  if (!referenceInputs || !referenceExpected)
    return 1;

#ifdef WAFER_ENABLE_SYSTEMC_MODEL
  std::optional<wafer::model::TargetModelKernelBudget> targetModelBudget;
  std::optional<wafer::model::TargetModelExecutionPolicy>
      targetModelExecutionPolicy;
#ifdef WAFER_ENABLE_TARGET_BULK_MODEL
  std::unique_ptr<wafer::model::QualifiedTargetModelBulkBackend>
      targetModelBulkBackend;
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

    llvm::StringRef gemmBackend =
        options.targetModelGemmBackend
            ? llvm::StringRef(*options.targetModelGemmBackend)
            : llvm::StringRef("formal");
    const bool hasBulkConfiguration =
        !options.targetModelBulkRecords.empty() ||
        options.targetModelMaximumBulkTotalBytes ||
        options.targetModelMaximumBulkScratchpadBytes ||
        options.targetModelMaximumBulkReorderBytes;
    if (gemmBackend == "formal") {
      if (hasBulkConfiguration) {
        llvm::errs() << "wafer-compile: bulk model options require "
                        "--target-model-gemm-backend=prefer-admitted\n";
        return 1;
      }
      targetModelExecutionPolicy.emplace(
          wafer::model::TargetModelExecutionPolicy::formalOnly());
    } else if (gemmBackend == "prefer-admitted") {
#ifdef WAFER_ENABLE_TARGET_BULK_MODEL
      if (options.targetModelBulkRecords.empty()) {
        llvm::errs() << "wafer-compile: prefer-admitted GEMM requires at "
                        "least one --target-model-bulk-record\n";
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
      auto backend = wafer::model::QualifiedTargetModelBulkBackend::create(
          options.targetModelBulkRecords,
          wafer::BulkNumericWorkBudget::create(*maximumTotalBytes,
                                               *maximumScratchpadBytes,
                                               *maximumReorderBytes));
      if (!backend) {
        llvm::errs() << "wafer-compile: " << llvm::toString(backend.takeError())
                     << "\n";
        return 1;
      }
      targetModelBulkBackend = std::move(*backend);
      targetModelExecutionPolicy.emplace(
          wafer::model::TargetModelExecutionPolicy::preferAdmitted(
              *targetModelBulkBackend));
#else
      llvm::errs() << "wafer-compile: qualified bulk model support is not "
                      "configured\n";
      return 1;
#endif
    } else {
      llvm::errs() << "wafer-compile: invalid "
                      "--target-model-gemm-backend value: "
                   << gemmBackend << "\n";
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
  const bool executeReferenceGate =
      referenceRequested &&
      (!options.targetModel || targetModelOracle == "reference");
  if (executeReferenceGate) {
    // The package boundary is complete before the downstream numeric gate.
    // Keep mixed stdout/stderr diagnostics in that semantic order as well.
    llvm::outs().flush();
    const wafer::compiler::ExecutableBundle *referenceBundle =
        executableBundle
            ? &*executableBundle
            : (targetCompilationProduct
                   ? &targetCompilationProduct->getExecutableBundle()
                   : nullptr);
    if (!referenceBundle) {
      llvm::errs() << "wafer-compile: reference execution cannot be combined "
                      "with test-only compilation failure injection\n";
      return 1;
    }
    if (runReferenceGate(options, *referenceBundle, *referenceInputs,
                         *referenceExpected, *referenceAtol, *referenceRtol))
      return 1;
    llvm::outs() << "wafer-compile: reference outputs matched\n";
  }
#ifdef WAFER_ENABLE_SYSTEMC_MODEL
  if (options.targetModel) {
    llvm::outs().flush();
    if (!targetCompilationProduct || !targetModelBudget ||
        !targetModelExecutionPolicy) {
      llvm::errs() << "wafer-compile: target model compilation product or "
                      "budget is missing\n";
      return 1;
    }
    if (runTargetModelGate(options, *targetCompilationProduct, *referenceInputs,
                           *referenceExpected, *referenceAtol, *referenceRtol,
                           *targetModelBudget, *targetModelExecutionPolicy))
      return 1;
  }
#endif
  return 0;
}
