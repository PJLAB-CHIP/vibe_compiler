//===- ReferenceGate.cpp - Wafer reference execution gate ----------------===//

#include "DriverInternal.h"

#include "Wafer/Compiler/ProgramInvocation.h"
#include "Wafer/Compiler/ReferenceExecutor.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::compile_driver {
namespace {

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

} // namespace

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

} // namespace wafer::compile_driver
