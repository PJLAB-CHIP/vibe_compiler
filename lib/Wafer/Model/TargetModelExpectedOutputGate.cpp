//===- TargetModelExpectedOutputGate.cpp - Source-backed model gate ------===//

#include "Wafer/Model/TargetModelExpectedOutputGate.h"

#include "Wafer/Compiler/ProgramInvocation.h"
#include "Wafer/Compiler/ProgramTensorComparison.h"
#include "Wafer/Model/TargetModelInvocation.h"
#include "Wafer/Model/Testing.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace wafer::model {
namespace {

static llvm::Error gateError(const llvm::Twine &message) {
  return llvm::createStringError(message.str());
}

static llvm::Expected<TargetModelResult>
executeAndCompareTargetModelExpectedOutputsImpl(
    const compiler::ExecutableBundle &executableBundle,
    const compiler::TargetLLVMModuleBundle &targetLLVMModuleBundle,
    llvm::StringRef outputProgramDirectory,
    llvm::ArrayRef<TargetModelTensorFileBinding> inputPaths,
    llvm::ArrayRef<TargetModelTensorFileBinding> expectedPaths, double atol,
    double rtol, TargetModelKernelBudget budget,
    TargetModelExecutionPolicy executionPolicy,
    std::optional<int64_t> terminalFailureRank,
    llvm::raw_ostream *numericStatistics) {
  std::vector<compiler::ProgramGlobalInputBinding> globalInputs;
  for (const TargetModelTensorFileBinding &inputPath : inputPaths) {
    auto tensor = compiler::ProgramTensor::loadNpy(inputPath.path);
    if (!tensor)
      return tensor.takeError();
    globalInputs.push_back({inputPath.index, std::move(*tensor)});
  }
  std::vector<std::pair<int64_t, compiler::ProgramTensor>> expectedTensors;
  for (const TargetModelTensorFileBinding &expectedPath : expectedPaths) {
    auto tensor = compiler::ProgramTensor::loadNpy(expectedPath.path);
    if (!tensor)
      return tensor.takeError();
    expectedTensors.emplace_back(expectedPath.index, std::move(*tensor));
  }

  auto programInvocations = compiler::prepareProgramInvocations(
      executableBundle, outputProgramDirectory, globalInputs);
  if (!programInvocations)
    return programInvocations.takeError();
  auto invocation = prepareTargetModelInvocation(
      executableBundle, targetLLVMModuleBundle, *programInvocations);
  if (!invocation)
    return invocation.takeError();
  llvm::Expected<TargetModelResult> result =
      terminalFailureRank
          ? testing::executeSystemCTargetModelWithTerminalFailure(
                std::move(invocation->getExecutable()),
                invocation->getInputBindings(), budget, executionPolicy,
                *terminalFailureRank)
          : executeSystemCTargetModel(std::move(invocation->getExecutable()),
                                      invocation->getInputBindings(), budget,
                                      executionPolicy);
  if (!result)
    return result.takeError();

  const auto &rankExecutables = executableBundle.getRankExecutables();
  const auto &targetModules = targetLLVMModuleBundle.getModules();
  size_t expectedOutputCount = 0;
  for (const auto &module : targetModules)
    expectedOutputCount +=
        llvm::count_if(module.getKernelABISlots(), [](const auto &slot) {
          return slot.role == compiler::KernelABISlotRole::Output;
        });
  if (result->completedRankCount !=
          static_cast<int64_t>(rankExecutables.size()) ||
      result->outputs.size() != expectedOutputCount)
    return gateError("target model output rank/domain is incomplete");

  std::set<std::pair<int64_t, int64_t>> seenOutputs;
  for (const TargetModelOutput &output : result->outputs) {
    if (output.logicalRank < 0 ||
        static_cast<size_t>(output.logicalRank) >= targetModules.size())
      return gateError("target model output rank is invalid");
    if (!seenOutputs.emplace(output.logicalRank, output.slotOrdinal).second)
      return gateError("target model output is duplicated");
    const auto &module = targetModules[static_cast<size_t>(output.logicalRank)];
    const compiler::KernelABISlot *slot = nullptr;
    for (const auto &candidate : module.getKernelABISlots())
      if (candidate.ordinal == output.slotOrdinal)
        slot = &candidate;
    if (!slot || slot->role != compiler::KernelABISlotRole::Output ||
        slot->resourceIndex != output.resourceIndex)
      return gateError(
          "target model output disagrees with the Kernel Runtime ABI");
    const auto &rank = rankExecutables[static_cast<size_t>(output.logicalRank)];
    const compiler::RankProgramBinding *binding = nullptr;
    for (const auto &candidate : rank.getProgramBindings())
      if (candidate.role == compiler::ProgramResourceRole::Output &&
          candidate.index == slot->resourceIndex)
        binding = &candidate;
    if (!binding)
      return gateError("target model output has no typed program binding");
    const compiler::ProgramTensor *globalExpected = nullptr;
    for (const auto &candidate : expectedTensors)
      if (candidate.first == binding->programIndex)
        globalExpected = &candidate.second;
    if (!globalExpected)
      return gateError("missing target model expected output " +
                       llvm::Twine(binding->programIndex));
    auto expected =
        compiler::sliceProgramTensorForBinding(*globalExpected, *binding);
    auto actual = decodeTargetModelProgramTensor(*slot, output.bytes);
    if (!expected || !actual) {
      llvm::Error errors = llvm::Error::success();
      if (!expected)
        errors = llvm::joinErrors(std::move(errors), expected.takeError());
      if (!actual)
        errors = llvm::joinErrors(std::move(errors), actual.takeError());
      return std::move(errors);
    }
    if (numericStatistics) {
      auto statistics = compiler::computeProgramTensorComparisonStatistics(
          *actual, *expected);
      if (!statistics)
        return statistics.takeError();
      *numericStatistics
          << "target model numeric statistics index=" << binding->programIndex
          << " rank=" << output.logicalRank << " dtype=" << actual->getDType()
          << " elements=" << statistics->elementCount
          << " exact=" << statistics->exactElementCount << " exact_fraction="
          << llvm::format("%.17g", statistics->exactFraction) << " mean_abs="
          << llvm::format("%.17g", statistics->meanAbsoluteError)
          << " p99_abs=" << llvm::format("%.17g", statistics->p99AbsoluteError)
          << " p999_abs="
          << llvm::format("%.17g", statistics->p999AbsoluteError) << " max_abs="
          << llvm::format("%.17g", statistics->maximumAbsoluteError)
          << " mean_ulp=" << llvm::format("%.17g", statistics->meanUlpDistance)
          << " p99_ulp=" << statistics->p99UlpDistance
          << " p999_ulp=" << statistics->p999UlpDistance
          << " max_ulp=" << statistics->maximumUlpDistance << "\n";
    }
    if (llvm::Error comparison = compiler::compareProgramTensorExpectedOutput(
            *actual, *expected, atol, rtol))
      return llvm::joinErrors(
          gateError("target model output differs at index " +
                    llvm::Twine(binding->programIndex) + " rank " +
                    llvm::Twine(output.logicalRank)),
          std::move(comparison));
  }
  return std::move(*result);
}

} // namespace

llvm::Expected<TargetModelResult> executeAndCompareTargetModelExpectedOutputs(
    const compiler::ExecutableBundle &executableBundle,
    const compiler::TargetLLVMModuleBundle &targetLLVMModuleBundle,
    llvm::StringRef outputProgramDirectory,
    llvm::ArrayRef<TargetModelTensorFileBinding> inputPaths,
    llvm::ArrayRef<TargetModelTensorFileBinding> expectedPaths, double atol,
    double rtol, TargetModelKernelBudget budget,
    TargetModelExecutionPolicy executionPolicy,
    llvm::raw_ostream *numericStatistics) {
  return executeAndCompareTargetModelExpectedOutputsImpl(
      executableBundle, targetLLVMModuleBundle, outputProgramDirectory,
      inputPaths, expectedPaths, atol, rtol, budget, executionPolicy,
      std::nullopt, numericStatistics);
}

llvm::Expected<TargetModelResult>
testing::executeAndCompareTargetModelExpectedOutputsWithTerminalFailure(
    const compiler::ExecutableBundle &executableBundle,
    const compiler::TargetLLVMModuleBundle &targetLLVMModuleBundle,
    llvm::StringRef outputProgramDirectory,
    llvm::ArrayRef<TargetModelTensorFileBinding> inputPaths,
    llvm::ArrayRef<TargetModelTensorFileBinding> expectedPaths, double atol,
    double rtol, TargetModelKernelBudget budget,
    TargetModelExecutionPolicy executionPolicy, int64_t failureRank,
    llvm::raw_ostream *numericStatistics) {
  return executeAndCompareTargetModelExpectedOutputsImpl(
      executableBundle, targetLLVMModuleBundle, outputProgramDirectory,
      inputPaths, expectedPaths, atol, rtol, budget, executionPolicy,
      failureRank, numericStatistics);
}

} // namespace wafer::model
