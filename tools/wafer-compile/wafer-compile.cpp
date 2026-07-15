//===- wafer-compile.cpp - Wafer user compiler driver --------------------===//

#include "Wafer/Compiler/Compilation.h"
#include "Wafer/Compiler/ProgramInvocation.h"
#include "Wafer/Compiler/ReferenceExecutor.h"
#include "Wafer/Compiler/TargetArtifact.h"
#ifdef WAFER_ENABLE_SYSTEMC_MODEL
#include "Wafer/Model/SystemCTargetModel.h"
#include "Wafer/Model/TargetModelInvocation.h"
#endif
#ifdef WAFER_ENABLE_TARGET_BULK_MODEL
#include "Wafer/Model/TargetBulkModel.h"
#endif
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
#include <memory>
#include <optional>
#include <set>
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
  bool targetModel = false;
  std::optional<std::string> targetModelMaximumScalarEvaluations;
  std::optional<std::string> targetModelMaximumFusedMultiplyAdds;
  std::optional<std::string> targetModelMaximumMovementBytes;
  std::optional<std::string> targetModelMaximumMovementSegments;
  std::optional<std::string> targetModelGemmBackend;
  std::optional<std::string> targetModelOracle;
  std::vector<std::string> targetModelBulkRecords;
  std::optional<std::string> targetModelMaximumBulkTotalBytes;
  std::optional<std::string> targetModelMaximumBulkScratchpadBytes;
  std::optional<std::string> targetModelMaximumBulkReorderBytes;
};

void printHelp() {
  llvm::outs() << "usage: wafer-compile --input-program-dir <dir> "
                  "--output-program-dir <dir> --execution-ranks <1|16> "
                  "--target-profile <registered-id> "
                  "[--reference-input <index>=<npy>] "
                  "[--reference-expected <index>=<npy>] "
                  "[--reference-atol <value>] [--reference-rtol <value>] "
                  "[--target-model "
                  "--target-model-oracle <reference|external> "
                  "--target-model-max-scalar-evaluations <count> "
                  "--target-model-max-fused-multiply-adds <count> "
                  "--target-model-max-movement-bytes <bytes> "
                  "--target-model-max-movement-segments <count> "
                  "[--target-model-gemm-backend <formal|prefer-admitted> "
                  "--target-model-bulk-record <record> "
                  "--target-model-max-bulk-total-bytes <bytes> "
                  "--target-model-max-bulk-scratchpad-bytes <bytes> "
                  "--target-model-max-bulk-reorder-bytes <bytes>]]\n";
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
    if (arg == "--target-model") {
      if (options.targetModel) {
        llvm::errs() << "wafer-compile: duplicate option: --target-model\n";
        return false;
      }
      options.targetModel = true;
      continue;
    }
    if (arg == "--target-model-max-scalar-evaluations" ||
        arg.starts_with("--target-model-max-scalar-evaluations=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-scalar-evaluations",
                           options.targetModelMaximumScalarEvaluations))
        return false;
      continue;
    }
    if (arg == "--target-model-max-fused-multiply-adds" ||
        arg.starts_with("--target-model-max-fused-multiply-adds=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-fused-multiply-adds",
                           options.targetModelMaximumFusedMultiplyAdds))
        return false;
      continue;
    }
    if (arg == "--target-model-max-movement-bytes" ||
        arg.starts_with("--target-model-max-movement-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-movement-bytes",
                           options.targetModelMaximumMovementBytes))
        return false;
      continue;
    }
    if (arg == "--target-model-max-movement-segments" ||
        arg.starts_with("--target-model-max-movement-segments=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-movement-segments",
                           options.targetModelMaximumMovementSegments))
        return false;
      continue;
    }
    if (arg == "--target-model-gemm-backend" ||
        arg.starts_with("--target-model-gemm-backend=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-gemm-backend",
                           options.targetModelGemmBackend))
        return false;
      continue;
    }
    if (arg == "--target-model-oracle" ||
        arg.starts_with("--target-model-oracle=")) {
      if (parseValueOption(argc, argv, index, arg, "--target-model-oracle",
                           options.targetModelOracle))
        return false;
      continue;
    }
    if (arg == "--target-model-bulk-record" ||
        arg.starts_with("--target-model-bulk-record=")) {
      if (parseRepeatedValueOption(argc, argv, index, arg,
                                   "--target-model-bulk-record",
                                   options.targetModelBulkRecords))
        return false;
      continue;
    }
    if (arg == "--target-model-max-bulk-total-bytes" ||
        arg.starts_with("--target-model-max-bulk-total-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-bulk-total-bytes",
                           options.targetModelMaximumBulkTotalBytes))
        return false;
      continue;
    }
    if (arg == "--target-model-max-bulk-scratchpad-bytes" ||
        arg.starts_with("--target-model-max-bulk-scratchpad-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-bulk-scratchpad-bytes",
                           options.targetModelMaximumBulkScratchpadBytes))
        return false;
      continue;
    }
    if (arg == "--target-model-max-bulk-reorder-bytes" ||
        arg.starts_with("--target-model-max-bulk-reorder-bytes=")) {
      if (parseValueOption(argc, argv, index, arg,
                           "--target-model-max-bulk-reorder-bytes",
                           options.targetModelMaximumBulkReorderBytes))
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

std::optional<uint64_t>
parsePositiveCount(const std::optional<std::string> &value,
                   llvm::StringRef option) {
  if (!value) {
    llvm::errs() << "wafer-compile: target model requires " << option << "\n";
    return std::nullopt;
  }
  uint64_t parsed = 0;
  if (llvm::StringRef(*value).getAsInteger(10, parsed) || parsed == 0) {
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

#ifdef WAFER_ENABLE_SYSTEMC_MODEL
bool compareModelTensor(const wafer::compiler::ProgramTensor &actual,
                        const wafer::compiler::ProgramTensor &expected,
                        int64_t programIndex, int64_t logicalRank, double atol,
                        double rtol) {
  if (actual.getDType() != expected.getDType() ||
      actual.getShape() != expected.getShape() ||
      actual.getBytes().size() != expected.getBytes().size()) {
    llvm::errs() << "wafer-compile: target model output type mismatch at index "
                 << programIndex << " rank " << logicalRank << "\n";
    return true;
  }
  if (actual.getDType() != "f32") {
    if (actual.getBytes() == expected.getBytes())
      return false;
    llvm::errs() << "wafer-compile: target model output differs at index "
                 << programIndex << " rank " << logicalRank << "\n";
    return true;
  }
  llvm::ArrayRef<uint8_t> actualBytes = actual.getBytes();
  llvm::ArrayRef<uint8_t> expectedBytes = expected.getBytes();
  for (size_t offset = 0; offset < actualBytes.size(); offset += 4) {
    float actualValue = 0.0f;
    float expectedValue = 0.0f;
    std::memcpy(&actualValue, actualBytes.data() + offset, 4);
    std::memcpy(&expectedValue, expectedBytes.data() + offset, 4);
    const double absoluteError =
        std::abs(static_cast<double>(actualValue) - expectedValue);
    const double tolerance = atol + rtol * std::abs(expectedValue);
    if (!std::isfinite(actualValue) || !std::isfinite(expectedValue) ||
        absoluteError > tolerance) {
      llvm::errs() << "wafer-compile: target model output mismatch at index "
                   << programIndex << " rank " << logicalRank << " element "
                   << offset / 4 << ": actual=" << actualValue
                   << " expected=" << expectedValue
                   << " abs_error=" << absoluteError
                   << " tolerance=" << tolerance << "\n";
      return true;
    }
  }
  return false;
}

bool runTargetModelGate(
    const CommandLineOptions &options,
    const wafer::compiler::TargetCompilationProduct &product,
    llvm::ArrayRef<IndexedPath> inputPaths,
    llvm::ArrayRef<IndexedPath> expectedPaths, double atol, double rtol,
    wafer::model::TargetModelKernelBudget budget,
    wafer::model::TargetModelExecutionPolicy executionPolicy) {
  std::vector<wafer::compiler::ProgramGlobalInputBinding> globalInputs;
  for (const IndexedPath &inputPath : inputPaths) {
    auto tensor = wafer::compiler::ProgramTensor::loadNpy(inputPath.path);
    if (!tensor) {
      llvm::errs() << "wafer-compile: " << llvm::toString(tensor.takeError())
                   << "\n";
      return true;
    }
    globalInputs.push_back({inputPath.index, std::move(*tensor)});
  }
  std::vector<std::pair<int64_t, wafer::compiler::ProgramTensor>>
      expectedTensors;
  for (const IndexedPath &expectedPath : expectedPaths) {
    auto tensor = wafer::compiler::ProgramTensor::loadNpy(expectedPath.path);
    if (!tensor) {
      llvm::errs() << "wafer-compile: " << llvm::toString(tensor.takeError())
                   << "\n";
      return true;
    }
    expectedTensors.emplace_back(expectedPath.index, std::move(*tensor));
  }

  auto programInvocations = wafer::compiler::prepareProgramInvocations(
      product.getExecutableBundle(), *options.outputProgramDirectory,
      globalInputs);
  if (!programInvocations) {
    llvm::errs() << "wafer-compile: "
                 << llvm::toString(programInvocations.takeError()) << "\n";
    return true;
  }
  auto invocation = wafer::model::prepareTargetModelInvocation(
      product.getExecutableBundle(), product.getTargetLLVMModuleBundle(),
      *programInvocations);
  if (!invocation) {
    llvm::errs() << "wafer-compile: " << llvm::toString(invocation.takeError())
                 << "\n";
    return true;
  }
  auto result = wafer::model::executeSystemCTargetModel(
      std::move(invocation->getExecutable()), invocation->getInputBindings(),
      budget, executionPolicy);
  if (!result) {
    llvm::errs() << "wafer-compile: " << llvm::toString(result.takeError())
                 << "\n";
    return true;
  }

  const auto &rankExecutables =
      product.getExecutableBundle().getRankExecutables();
  const auto &targetModules = product.getTargetLLVMModuleBundle().getModules();
  size_t expectedOutputCount = 0;
  for (const auto &module : targetModules)
    expectedOutputCount +=
        llvm::count_if(module.getKernelABISlots(), [](const auto &slot) {
          return slot.role == wafer::compiler::KernelABISlotRole::Output;
        });
  if (result->completedRankCount !=
          static_cast<int64_t>(rankExecutables.size()) ||
      result->outputs.size() != expectedOutputCount) {
    llvm::errs() << "wafer-compile: target model output rank/domain is "
                    "incomplete\n";
    return true;
  }

  std::set<std::pair<int64_t, int64_t>> seenOutputs;
  for (const wafer::model::TargetModelOutput &output : result->outputs) {
    if (output.logicalRank < 0 ||
        static_cast<size_t>(output.logicalRank) >= targetModules.size()) {
      llvm::errs() << "wafer-compile: target model output rank is invalid\n";
      return true;
    }
    if (!seenOutputs.emplace(output.logicalRank, output.slotOrdinal).second) {
      llvm::errs() << "wafer-compile: target model output is duplicated\n";
      return true;
    }
    const auto &module = targetModules[static_cast<size_t>(output.logicalRank)];
    const wafer::compiler::KernelABISlot *slot = nullptr;
    for (const auto &candidate : module.getKernelABISlots())
      if (candidate.ordinal == output.slotOrdinal)
        slot = &candidate;
    if (!slot || slot->role != wafer::compiler::KernelABISlotRole::Output ||
        slot->resourceIndex != output.resourceIndex) {
      llvm::errs() << "wafer-compile: target model output disagrees with the "
                      "Kernel ABI\n";
      return true;
    }
    const auto &rank = rankExecutables[static_cast<size_t>(output.logicalRank)];
    const wafer::compiler::RankProgramBinding *binding = nullptr;
    for (const auto &candidate : rank.getProgramBindings())
      if (candidate.role == wafer::compiler::ProgramResourceRole::Output &&
          candidate.index == slot->resourceIndex)
        binding = &candidate;
    if (!binding) {
      llvm::errs() << "wafer-compile: target model output has no typed program "
                      "binding\n";
      return true;
    }
    const wafer::compiler::ProgramTensor *globalExpected = nullptr;
    for (const auto &candidate : expectedTensors)
      if (candidate.first == binding->programIndex)
        globalExpected = &candidate.second;
    if (!globalExpected) {
      llvm::errs() << "wafer-compile: missing target model expected output "
                   << binding->programIndex << "\n";
      return true;
    }
    auto expected = wafer::compiler::sliceProgramTensorForBinding(
        *globalExpected, *binding);
    auto actual =
        wafer::model::decodeTargetModelProgramTensor(*slot, output.bytes);
    if (!expected || !actual) {
      llvm::Error errors = llvm::Error::success();
      if (!expected)
        errors = llvm::joinErrors(std::move(errors), expected.takeError());
      if (!actual)
        errors = llvm::joinErrors(std::move(errors), actual.takeError());
      llvm::errs() << "wafer-compile: " << llvm::toString(std::move(errors))
                   << "\n";
      return true;
    }
    if (compareModelTensor(*actual, *expected, binding->programIndex,
                           output.logicalRank, atol, rtol))
      return true;
  }
  llvm::outs() << "wafer-compile: target model outputs matched; ranks="
               << result->completedRankCount
               << " transactions=" << result->issuedTransactionCount
               << " systemc_threads=" << result->systemCThreadProcessCount
               << " final_delta=" << result->finalDeltaCount
               << " formal_commands=" << result->formalNumericCommandCount
               << " bulk_commands=" << result->bulkNumericCommandCount
               << " bulk_matmuls=" << result->bulkMatmulInvocationCount
               << " bulk_reorders=" << result->bulkReorderInvocationCount
               << " bulk_formal_fmas="
               << result->bulkFormalFusedMultiplyAddCount
               << " scheduler=" << result->schedulerIdentity << "\n";
  for (llvm::StringRef digest : result->bulkAdmissionRecordDigests)
    llvm::outs() << "wafer-compile: target model bulk admission=" << digest
                 << "\n";
  return false;
}
#endif

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
