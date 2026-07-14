//===- wafer-compile.cpp - Wafer user compiler driver --------------------===//

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/ReferenceExecutor.h"
#include "Wafer/Compiler/TargetArtifact.h"
#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
#include "Wafer/Compiler/Testing.h"
#endif

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifndef WAFER_XLA_SPMD_PARTITIONER_HELPER
#define WAFER_XLA_SPMD_PARTITIONER_HELPER ""
#endif
#ifndef WAFER_PYTHON_EXECUTABLE
#define WAFER_PYTHON_EXECUTABLE ""
#endif
#ifndef WAFER_DEVICE_LINKER_SCRIPT
#define WAFER_DEVICE_LINKER_SCRIPT ""
#endif

namespace {

struct CommandLineOptions {
  std::optional<std::string> inputProgramDirectory;
  std::optional<std::string> outputProgramDirectory;
  std::optional<std::string> executionRanks;
  std::optional<std::string> targetProfile;
  std::vector<std::string> referenceInputs;
  std::vector<std::string> referenceExpected;
  std::optional<std::string> referenceAtol;
  std::optional<std::string> referenceRtol;
};

void printHelp() {
  llvm::outs() << "usage: wafer-compile --input-program-dir <dir> "
                  "--output-program-dir <dir> --execution-ranks <1|16> "
                  "--target-profile <registered-id> "
                  "[--reference-input <index>=<npy>] "
                  "[--reference-expected <index>=<npy>] "
                  "[--reference-atol <value>] [--reference-rtol <value>]\n";
}

bool setOption(std::optional<std::string> &slot, llvm::StringRef option,
               llvm::StringRef value) {
  if (slot) {
    llvm::errs() << "wafer-compile: duplicate option: " << option << "\n";
    return true;
  }
  slot = value.str();
  return false;
}

bool parseValueOption(int argc, char **argv, int &index, llvm::StringRef arg,
                      llvm::StringRef option,
                      std::optional<std::string> &slot) {
  if (arg == option) {
    if (index + 1 >= argc) {
      llvm::errs() << "wafer-compile: missing value for " << option << "\n";
      return true;
    }
    return setOption(slot, option, argv[++index]);
  }

  std::string prefix = (option + "=").str();
  if (arg.starts_with(prefix))
    return setOption(slot, option, arg.drop_front(prefix.size()));
  return false;
}

bool parseRepeatedValueOption(int argc, char **argv, int &index,
                              llvm::StringRef arg, llvm::StringRef option,
                              std::vector<std::string> &values) {
  if (arg == option) {
    if (index + 1 >= argc) {
      llvm::errs() << "wafer-compile: missing value for " << option << "\n";
      return true;
    }
    values.emplace_back(argv[++index]);
    return false;
  }
  std::string prefix = (option + "=").str();
  if (arg.starts_with(prefix)) {
    values.push_back(arg.drop_front(prefix.size()).str());
    return false;
  }
  return false;
}

bool parseCommandLine(int argc, char **argv, CommandLineOptions &options) {
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef arg(argv[index]);
    if (arg == "--help") {
      printHelp();
      return true;
    }
    if (arg == "--input-program-dir" ||
        arg.starts_with("--input-program-dir=")) {
      if (parseValueOption(argc, argv, index, arg, "--input-program-dir",
                           options.inputProgramDirectory))
        return false;
      continue;
    }
    if (arg == "--output-program-dir" ||
        arg.starts_with("--output-program-dir=")) {
      if (parseValueOption(argc, argv, index, arg, "--output-program-dir",
                           options.outputProgramDirectory))
        return false;
      continue;
    }
    if (arg == "--execution-ranks" || arg.starts_with("--execution-ranks=")) {
      if (parseValueOption(argc, argv, index, arg, "--execution-ranks",
                           options.executionRanks))
        return false;
      continue;
    }
    if (arg == "--target-profile" || arg.starts_with("--target-profile=")) {
      if (parseValueOption(argc, argv, index, arg, "--target-profile",
                           options.targetProfile))
        return false;
      continue;
    }
    if (arg == "--reference-input" || arg.starts_with("--reference-input=")) {
      if (parseRepeatedValueOption(argc, argv, index, arg, "--reference-input",
                                   options.referenceInputs))
        return false;
      continue;
    }
    if (arg == "--reference-expected" ||
        arg.starts_with("--reference-expected=")) {
      if (parseRepeatedValueOption(argc, argv, index, arg,
                                   "--reference-expected",
                                   options.referenceExpected))
        return false;
      continue;
    }
    if (arg == "--reference-atol" || arg.starts_with("--reference-atol=")) {
      if (parseValueOption(argc, argv, index, arg, "--reference-atol",
                           options.referenceAtol))
        return false;
      continue;
    }
    if (arg == "--reference-rtol" || arg.starts_with("--reference-rtol=")) {
      if (parseValueOption(argc, argv, index, arg, "--reference-rtol",
                           options.referenceRtol))
        return false;
      continue;
    }

    llvm::errs() << "wafer-compile: unknown argument: " << arg << "\n";
    return false;
  }
  return true;
}

bool requireOption(const std::optional<std::string> &value,
                   llvm::StringRef option) {
  if (value)
    return true;
  llvm::errs() << "wafer-compile: missing required " << option << "\n";
  return false;
}

std::string resolveSpmdPartitionerHelperPath() {
#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
  if (const char *environment =
          std::getenv("WAFER_TEST_XLA_SPMD_PARTITIONER_HELPER"))
    return environment;
#endif
  return WAFER_XLA_SPMD_PARTITIONER_HELPER;
}

struct IndexedPath {
  int64_t index = -1;
  std::string path;
};

std::optional<std::vector<IndexedPath>>
parseIndexedPaths(llvm::ArrayRef<std::string> values, llvm::StringRef option) {
  std::vector<IndexedPath> parsed;
  for (const std::string &storage : values) {
    llvm::StringRef value(storage);
    auto [indexText, path] = value.split('=');
    int64_t index = -1;
    if (path.empty() || indexText.getAsInteger(10, index) || index < 0) {
      llvm::errs() << "wafer-compile: invalid " << option << " value: " << value
                   << "\n";
      return std::nullopt;
    }
    if (llvm::any_of(parsed, [&](const IndexedPath &item) {
          return item.index == index;
        })) {
      llvm::errs() << "wafer-compile: duplicate " << option
                   << " index: " << index << "\n";
      return std::nullopt;
    }
    parsed.push_back({index, path.str()});
  }
  return parsed;
}

std::optional<double> parseTolerance(const std::optional<std::string> &value,
                                     llvm::StringRef option,
                                     double defaultValue) {
  if (!value)
    return defaultValue;
  errno = 0;
  char *end = nullptr;
  double parsed = std::strtod(value->c_str(), &end);
  if (errno != 0 || end != value->c_str() + value->size() ||
      !std::isfinite(parsed) || parsed < 0.0) {
    llvm::errs() << "wafer-compile: invalid " << option << " value: " << *value
                 << "\n";
    return std::nullopt;
  }
  return parsed;
}

bool compareReferenceTensor(const wafer::compiler::ReferenceTensor &actual,
                            const IndexedPath &expectedPath, double atol,
                            double rtol,
                            std::optional<int64_t> logicalRank = std::nullopt) {
  auto expected = wafer::compiler::ReferenceTensor::loadNpy(expectedPath.path);
  if (!expected) {
    llvm::errs() << "wafer-compile: " << llvm::toString(expected.takeError())
                 << "\n";
    return true;
  }
  auto printRank = [&] {
    if (logicalRank)
      llvm::errs() << " rank " << *logicalRank;
  };
  if (actual.getDType() != expected->getDType() ||
      actual.getShape() != expected->getShape()) {
    llvm::errs() << "wafer-compile: reference output type mismatch at index "
                 << expectedPath.index;
    printRank();
    llvm::errs() << "\n";
    return true;
  }
  llvm::ArrayRef<uint8_t> actualBytes = actual.getBytes();
  llvm::ArrayRef<uint8_t> expectedBytes = expected->getBytes();
  if (actualBytes.size() != expectedBytes.size()) {
    llvm::errs() << "wafer-compile: reference output byte count mismatch";
    printRank();
    llvm::errs() << "\n";
    return true;
  }
  if (actual.getDType() != "f32") {
    if (actualBytes != expectedBytes) {
      llvm::errs() << "wafer-compile: reference output differs at index "
                   << expectedPath.index;
      printRank();
      llvm::errs() << "\n";
      return true;
    }
    return false;
  }
  std::optional<size_t> firstMismatch;
  float firstActual = 0.0f;
  float firstExpected = 0.0f;
  size_t worstElement = 0;
  double worstAbsoluteError = 0.0;
  double worstToleranceRatio = 0.0;
  for (size_t offset = 0; offset < actualBytes.size(); offset += 4) {
    float actualValue = 0.0f;
    float expectedValue = 0.0f;
    std::memcpy(&actualValue, actualBytes.data() + offset, 4);
    std::memcpy(&expectedValue, expectedBytes.data() + offset, 4);
    double tolerance = atol + rtol * std::abs(expectedValue);
    double absoluteError =
        std::abs(static_cast<double>(actualValue) - expectedValue);
    double toleranceRatio = tolerance == 0.0
                                ? (absoluteError == 0.0 ? 0.0 : INFINITY)
                                : absoluteError / tolerance;
    if (toleranceRatio > worstToleranceRatio) {
      worstElement = offset / 4;
      worstAbsoluteError = absoluteError;
      worstToleranceRatio = toleranceRatio;
    }
    if ((!std::isfinite(actualValue) || !std::isfinite(expectedValue) ||
         absoluteError > tolerance) &&
        !firstMismatch) {
      firstMismatch = offset / 4;
      firstActual = actualValue;
      firstExpected = expectedValue;
    }
  }
  if (!firstMismatch)
    return false;
  llvm::errs() << "wafer-compile: reference output mismatch at index "
               << expectedPath.index << " element " << *firstMismatch;
  printRank();
  llvm::errs() << ": actual=" << firstActual << " expected=" << firstExpected
               << "; worst_element=" << worstElement
               << " max_abs_error=" << worstAbsoluteError
               << " tolerance_ratio=" << worstToleranceRatio << "\n";
  return true;
}

template <typename OutputBinding>
bool compareReferenceOutputs(llvm::ArrayRef<OutputBinding> outputs,
                             llvm::ArrayRef<IndexedPath> expectedPaths,
                             double atol, double rtol) {
  if (outputs.size() != expectedPaths.size()) {
    llvm::errs() << "wafer-compile: reference expected output domain is "
                    "incomplete\n";
    return true;
  }
  for (const IndexedPath &expectedPath : expectedPaths) {
    const OutputBinding *actual = nullptr;
    for (const OutputBinding &candidate : outputs)
      if (candidate.index == expectedPath.index) {
        if (actual) {
          llvm::errs() << "wafer-compile: duplicate reference output index\n";
          return true;
        }
        actual = &candidate;
      }
    if (!actual) {
      llvm::errs() << "wafer-compile: missing reference output index "
                   << expectedPath.index << "\n";
      return true;
    }
    if (compareReferenceTensor(actual->tensor, expectedPath, atol, rtol))
      return true;
  }
  return false;
}

bool compareReplicatedRankOutputs(
    const wafer::compiler::ExecutableBundle &bundle,
    const wafer::compiler::ReferenceMultiRankExecutionResult &result,
    llvm::ArrayRef<IndexedPath> expectedPaths, double atol, double rtol) {
  const auto &ranks = bundle.getRankExecutables();
  const auto &rankResults = result.getRankResults();
  if (ranks.size() != rankResults.size()) {
    llvm::errs() << "wafer-compile: reference rank result domain is "
                    "incomplete\n";
    return true;
  }
  for (size_t rankIndex = 0; rankIndex < ranks.size(); ++rankIndex) {
    int64_t logicalRank = ranks[rankIndex].getLogicalRank();
    if (rankResults[rankIndex].getLogicalRank() != logicalRank) {
      llvm::errs() << "wafer-compile: reference rank result order is not "
                      "canonical\n";
      return true;
    }
    for (const wafer::compiler::RankProgramBinding &binding :
         ranks[rankIndex].getProgramBindings()) {
      if (binding.role != wafer::compiler::ProgramResourceRole::Output ||
          binding.distribution !=
              wafer::frontend::ProgramDistributionKind::Replicated)
        continue;
      const IndexedPath *expected = nullptr;
      for (const IndexedPath &candidate : expectedPaths)
        if (candidate.index == binding.programIndex)
          expected = &candidate;
      const wafer::compiler::ReferenceOutputBinding *actual = nullptr;
      for (const auto &candidate : rankResults[rankIndex].getOutputs())
        if (candidate.index == binding.programIndex)
          actual = &candidate;
      if (!expected || !actual) {
        llvm::errs() << "wafer-compile: replicated reference output domain is "
                        "incomplete\n";
        return true;
      }
      if (compareReferenceTensor(actual->tensor, *expected, atol, rtol,
                                 logicalRank))
        return true;
    }
  }
  return false;
}

bool runReferenceGate(const CommandLineOptions &options,
                      const wafer::compiler::ExecutableBundle &bundle,
                      llvm::ArrayRef<IndexedPath> inputPaths,
                      llvm::ArrayRef<IndexedPath> expectedPaths, double atol,
                      double rtol) {
  std::vector<wafer::compiler::ReferenceGlobalInputBinding> globalInputs;
  for (const IndexedPath &inputPath : inputPaths) {
    auto tensor = wafer::compiler::ReferenceTensor::loadNpy(inputPath.path);
    if (!tensor) {
      llvm::errs() << "wafer-compile: " << llvm::toString(tensor.takeError())
                   << "\n";
      return true;
    }
    globalInputs.push_back({inputPath.index, std::move(*tensor)});
  }
  auto invocations = wafer::compiler::prepareReferenceInvocations(
      bundle, *options.outputProgramDirectory, globalInputs);
  if (!invocations) {
    llvm::errs() << "wafer-compile: " << llvm::toString(invocations.takeError())
                 << "\n";
    return true;
  }
  if (bundle.getExecutionConfig().getRankCount() == 1) {
    if (invocations->size() != 1 || invocations->front().logicalRank != 0) {
      llvm::errs() << "wafer-compile: single-rank reference invocation is "
                      "not canonical\n";
      return true;
    }
    auto result = wafer::compiler::executeReferenceRank(
        bundle, 0, invocations->front().inputs);
    if (!result) {
      llvm::errs() << "wafer-compile: " << llvm::toString(result.takeError())
                   << "\n";
      return true;
    }
    return compareReferenceOutputs<wafer::compiler::ReferenceOutputBinding>(
        result->getOutputs(), expectedPaths, atol, rtol);
  }
  auto result = wafer::compiler::executeReferenceBundle(bundle, *invocations);
  if (!result) {
    llvm::errs() << "wafer-compile: " << llvm::toString(result.takeError())
                 << "\n";
    return true;
  }
  if (compareReferenceOutputs<wafer::compiler::ReferenceGlobalOutputBinding>(
          result->getGlobalOutputs(), expectedPaths, atol, rtol))
    return true;
  return compareReplicatedRankOutputs(bundle, *result, expectedPaths, atol,
                                      rtol);
}

} // namespace

int main(int argc, char **argv) {
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
      wafer::compiler::ExecutionConfig::createForSingleCard(
          rankCount, *targetProfile);
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
    mlir::FailureOr<wafer::compiler::ExecutableBundle> compiledProgram =
        wafer::compiler::compileProgram(
            std::move(*request), *options.outputProgramDirectory, helperPath,
            *targetToolchain, llvm::errs());
    if (mlir::succeeded(compiledProgram)) {
      executableBundle.emplace(std::move(*compiledProgram));
      compilationStatus = mlir::success();
    }
  }
  if (mlir::failed(compilationStatus))
    return 1;

  llvm::outs() << "wafer-compile: published verified package with "
                  "execution-ranks="
               << rankCount << ": " << *options.outputProgramDirectory << "\n";
  if (referenceRequested) {
    // The package boundary is complete before the downstream numeric gate.
    // Keep mixed stdout/stderr diagnostics in that semantic order as well.
    llvm::outs().flush();
    if (!executableBundle) {
      llvm::errs() << "wafer-compile: reference execution cannot be combined "
                      "with test-only compilation failure injection\n";
      return 1;
    }
    if (runReferenceGate(options, *executableBundle, *referenceInputs,
                         *referenceExpected, *referenceAtol, *referenceRtol))
      return 1;
    llvm::outs() << "wafer-compile: reference outputs matched\n";
  }
  return 0;
}
