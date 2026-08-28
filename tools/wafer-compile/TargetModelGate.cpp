//===- TargetModelGate.cpp - Wafer target-model execution gate -----------===//

#include "DriverInternal.h"

#ifdef WAFER_ENABLE_SYSTEMC_MODEL

#include "Wafer/Model/Core/TargetModelInvocation.h"
#include "Wafer/Model/SystemC/SystemCTargetModel.h"
#include "Wafer/Program/ProgramInvocation.h"
#include "Wafer/Program/ProgramTensorComparison.h"
#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
#include "Wafer/Model/TestSupport/Testing.h"
#endif

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

namespace wafer::compile_driver {
bool runTargetModelGate(
    const CommandLineOptions &options,
    const wafer::compiler::CompiledProgram &compiledProgram,
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
      compiledProgram.getDeviceExecutable(), globalInputs);
  if (!programInvocations) {
    llvm::errs() << "wafer-compile: "
                 << llvm::toString(programInvocations.takeError()) << "\n";
    return true;
  }
  auto invocation = wafer::model::prepareTargetModelInvocation(
      compiledProgram.getDeviceExecutable(),
      compiledProgram.getTargetLLVMModules(), *programInvocations);
  if (!invocation) {
    llvm::errs() << "wafer-compile: " << llvm::toString(invocation.takeError())
                 << "\n";
    return true;
  }
  llvm::Expected<wafer::model::TargetModelResult> result = [&]() {
#ifdef WAFER_ENABLE_TEST_HELPER_OVERRIDE
    if (const char *failureLaunchSlot =
            std::getenv("WAFER_TEST_FAIL_TARGET_MODEL_TILE_COMPLETION_SLOT")) {
      int64_t parsedFailureLaunchSlot = -1;
      if (llvm::StringRef(failureLaunchSlot)
              .getAsInteger(10, parsedFailureLaunchSlot))
        return llvm::Expected<wafer::model::TargetModelResult>(
            llvm::createStringError(
                "invalid test-only target model Tile completion failure "
                "launch slot"));
      return wafer::model::testing::
          executeSystemCTargetModelWithTileCompletionFailure(
              std::move(invocation->getExecutable()),
              invocation->getInputBindings(), budget, executionPolicy,
              parsedFailureLaunchSlot);
    }
#endif
    return wafer::model::executeSystemCTargetModel(
        std::move(invocation->getExecutable()), invocation->getInputBindings(),
        budget, executionPolicy);
  }();
  if (!result) {
    llvm::errs() << "wafer-compile: " << llvm::toString(result.takeError())
                 << "\n";
    return true;
  }

  const auto &deviceExecutable =
      compiledProgram.getDeviceExecutable().getTileExecutables();
  const auto &targetModules =
      compiledProgram.getTargetLLVMModules().getModules();
  const size_t expectedOutputCount = llvm::count_if(
      targetModules.front().getTileEntryArguments(), [](const auto &slot) {
        return slot.kind ==
               wafer::compiler::TileEntryArgumentKind::ExternalOutput;
      });
  if (result->completedTileCount !=
          static_cast<int64_t>(deviceExecutable.size()) ||
      result->outputs.size() != expectedOutputCount) {
    llvm::errs() << "wafer-compile: target model output Tile domain is "
                    "incomplete\n";
    return true;
  }

  std::set<int64_t> seenOutputs;
  for (const wafer::model::TargetModelOutput &output : result->outputs) {
    const int64_t launchSlot = output.launchSlotId.getValue();
    if (launchSlot < 0 ||
        static_cast<size_t>(launchSlot) >= targetModules.size()) {
      llvm::errs() << "wafer-compile: target model output launch slot is "
                      "invalid\n";
      return true;
    }
    if (!seenOutputs.insert(output.resourceIndex).second) {
      llvm::errs() << "wafer-compile: target model output is duplicated\n";
      return true;
    }
    const auto &module = targetModules[static_cast<size_t>(launchSlot)];
    if (module.getCardId() != output.cardId ||
        module.getTileId() != output.tileId ||
        module.getLaunchSlotId() != output.launchSlotId) {
      llvm::errs() << "wafer-compile: target model output Tile "
                      "identity is invalid\n";
      return true;
    }
    const wafer::compiler::TileEntryArgument *slot = nullptr;
    for (const auto &candidate : module.getTileEntryArguments())
      if (candidate.ordinal == output.slotOrdinal)
        slot = &candidate;
    if (!slot ||
        slot->kind != wafer::compiler::TileEntryArgumentKind::ExternalOutput ||
        slot->resourceIndex != output.resourceIndex ||
        output.resource != wafer::model::getTargetModelResourceId(
                               module.getCardId(), module.getTileId(),
                               slot->kind, slot->resourceIndex)) {
      llvm::errs() << "wafer-compile: target model output disagrees with the "
                      "Kernel ABI\n";
      return true;
    }
    const auto &tile = deviceExecutable[static_cast<size_t>(launchSlot)];
    const wafer::compiler::ProgramResourceBinding *binding = nullptr;
    for (const auto &candidate : tile.getProgramBindings())
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
    if (options.modelReportNumericStatistics) {
      auto statistics =
          wafer::compiler::computeProgramTensorComparisonStatistics(*actual,
                                                                    *expected);
      if (!statistics) {
        llvm::errs() << "wafer-compile: target model numeric statistics "
                        "failed at index "
                     << binding->programIndex << " launch slot " << launchSlot
                     << ": " << llvm::toString(statistics.takeError()) << "\n";
        return true;
      }
      llvm::outs()
          << "wafer-compile: target model numeric statistics index="
          << binding->programIndex << " launch_slot=" << launchSlot
          << " dtype=" << stringifyProgramElementType(actual->getDType())
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
    if (llvm::Error comparison =
            wafer::compiler::compareProgramTensorExpectedOutput(
                *actual, *expected, atol, rtol)) {
      llvm::errs() << "wafer-compile: target model output differs at index "
                   << binding->programIndex << " launch slot " << launchSlot
                   << ": " << llvm::toString(std::move(comparison)) << "\n";
      return true;
    }
  }
  llvm::outs() << "wafer-compile: target model outputs matched; tiles="
               << result->completedTileCount
               << " commands=" << result->issuedCommandCount
               << " systemc_threads=" << result->systemCThreadProcessCount
               << " final_delta=" << result->finalDeltaCount
               << " formal_operations=" << result->formalNumericOperationCount
               << " managed_reference_commands="
               << result->managedReferenceNumericOperationCount
               << " managed_reference_scalars="
               << result->managedReferenceScalarEvaluationCount
               << " onednn_operations=" << result->onednnNumericOperationCount
               << " onednn_matmuls=" << result->onednnMatmulInvocationCount
               << " onednn_reorders=" << result->onednnReorderInvocationCount
               << " onednn_formal_fmas="
               << result->onednnFormalFusedMultiplyAddCount
               << " scheduler=" << result->schedulerIdentity << "\n";
  for (llvm::StringRef digest : result->onednnQualificationRecordDigests)
    llvm::outs() << "wafer-compile: target model onednn qualification="
                 << digest << "\n";
  for (llvm::StringRef digest :
       result->onednnManagedReferenceEnvironmentDigests)
    llvm::outs() << "wafer-compile: target model managed-reference environment="
                 << digest << "\n";
  for (llvm::StringRef digest :
       result->managedReferenceTensorEnvironmentDigests)
    llvm::outs()
        << "wafer-compile: target model managed-reference tensor environment="
        << digest << "\n";
  for (llvm::StringRef implementation :
       result->managedReferenceTensorImplementations)
    llvm::outs() << "wafer-compile: target model managed-reference tensor "
                    "implementation="
                 << implementation << "\n";
  return false;
}

} // namespace wafer::compile_driver

#endif // WAFER_ENABLE_SYSTEMC_MODEL
