//===- RuntimeInvocationPlanning.cpp - Runtime invocation planning --------===//

#include "Wafer/Package/Manifest/PackageManifest.h"

#include "llvm/Support/Errc.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

namespace wafer::runtime {


namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(llvm::errc::invalid_argument, message);
}


bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedAlign(uint64_t value, uint64_t alignment, uint64_t &result) {
  if (alignment == 0)
    return false;
  const uint64_t remainder = value % alignment;
  const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
  return checkedAdd(value, padding, result);
}

bool checkedAlignment(uint64_t value, uint64_t alignment) {
  if (alignment == 0)
    return false;
  return value % alignment == 0;
}

llvm::Error validateRuntimeEnvironment(const PackageManifest &manifest,
                                       const RuntimeEnvironment &environment) {
  if (environment.targetIdentity != manifest.targetIdentity ||
      environment.runtimeABI != manifest.runtimeABI ||
      environment.moduleFormat != manifest.moduleFormat)
    return invalid("runtime environment is incompatible with package target");
  const KernelRuntimeLaunchContract &kernel = manifest.launch.getKernel();
  if (!llvm::is_contained(environment.supportedKernelLaunchForms,
                          kernel.form) ||
      !llvm::is_contained(environment.supportedKernelEntryABIs,
                          kernel.entryABI))
    return invalid(
        "runtime environment does not support the package kernel launch "
        "contract");
  return llvm::Error::success();
}

PackageModuleExportRole exportRoleForPhase(RuntimeLaunchPhaseRole phase) {
  return phase == RuntimeLaunchPhaseRole::Prepare
             ? PackageModuleExportRole::Prepare
             : PackageModuleExportRole::Main;
}

} // namespace

llvm::Expected<RuntimeInvocationPlan> planRuntimeInvocation(
    const VerifiedPackageManifest &package,
    llvm::ArrayRef<RuntimeInvocationBinding> invocationBindings,
    const RuntimeEnvironment &environment) {
  const PackageManifest &manifest = package.getManifest();
  if (llvm::Error error = validateRuntimeEnvironment(manifest, environment))
    return std::move(error);

  std::vector<const PackageEntrypointRecord *> entriesByLaunchSlot(
      manifest.tileCount, nullptr);
  for (const PackageEntrypointRecord &entry : manifest.entries)
    entriesByLaunchSlot[entry.launchSlot.getValue()] = &entry;
  if (llvm::is_contained(entriesByLaunchSlot, nullptr))
    return invalid("runtime invocation launch-slot domain is incomplete");

  const size_t transportKind =
      entriesByLaunchSlot.front()->transport.index();
  if (llvm::any_of(entriesByLaunchSlot, [&](const auto *entry) {
        return entry->transport.index() != transportKind;
      }))
    return invalid("runtime invocation contains mixed transport requirements");
  for (const PackageEntrypointRecord *entry : entriesByLaunchSlot)
    if (const auto *requirements = std::get_if<DirectDTETransportRequirements>(
            &entry->transport)) {
      if (!environment.supportsDirectDTE ||
          environment.directDTEStatusABI != requirements->statusABI ||
          (requirements->hostWatchdogRequired &&
           !environment.supportsHostWatchdog))
        return invalid(
            "runtime environment does not satisfy Direct DTE transport "
            "requirements");
    }

  // Caller bindings: all-and-only external input ports, with exact byte
  // counts. Output ports are prepared by the runtime and never bound.
  std::vector<const RuntimeInvocationBinding *> bindingsByPort(
      manifest.inputs.size(), nullptr);
  for (const RuntimeInvocationBinding &binding : invocationBindings) {
    if (!binding.port.isValid() ||
        binding.port.getValue() >= manifest.inputs.size() ||
        bindingsByPort[binding.port.getValue()])
      return invalid(
          "runtime invocation contains duplicate or unknown input binding");
    const ExternalPortRecord &port =
        manifest.inputs[binding.port.getValue()];
    if (binding.bytes != port.bytes ||
        binding.alignment == 0 ||
        !checkedAlignment(binding.alignment, port.alignment) ||
        binding.alignment < port.alignment)
      return invalid("runtime invocation binding does not satisfy input "
                     "port");
    bindingsByPort[binding.port.getValue()] = &binding;
  }
  for (uint64_t port = 0; port < manifest.inputs.size(); ++port)
    if (!bindingsByPort[port])
      return invalid("runtime invocation is missing an input port binding");

  RuntimeInvocationPlan plan;
  plan.cardCount = manifest.cardCount;
  plan.tileCount = manifest.tileCount;

  // Program data: one contiguous read-only allocation carrying every
  // TargetTensor at its manifest file offset.
  plan.programDataRequired = manifest.programData.totalBytes > 0;
  plan.programDataBytes = manifest.programData.totalBytes;
  plan.programDataAlignment = manifest.programData.baseAlignment;
  if (plan.programDataBytes > environment.maxResourceBytes)
    return invalid("runtime environment program data capacity is "
                   "insufficient");
  plan.targetTensorRanges.reserve(manifest.targetTensors.size());
  for (const TargetTensorRecord &tensor : manifest.targetTensors)
    plan.targetTensorRanges.push_back(
        {tensor.fileOffset, tensor.bytes});

  // Invocation allocation: deterministic packing of inputs, outputs,
  // per-Tile entry-local ranges, and (for TileRowPointerTable) per-Tile
  // pointer rows.
  const KernelRuntimeLaunchContract &kernel = manifest.launch.getKernel();
  uint64_t nextOffset = 0;
  uint64_t invocationAlignment = 1;
  auto placeRange = [&](uint64_t bytes,
                        uint64_t alignment) -> llvm::Expected<RuntimePlannedRange> {
    if (bytes == 0 || alignment == 0)
      return invalid("runtime child range has zero bytes or alignment");
    uint64_t offset = 0;
    if (!checkedAlign(nextOffset, alignment, offset) ||
        !checkedAdd(offset, bytes, nextOffset))
      return invalid("runtime invocation child range overflows");
    if (alignment > invocationAlignment)
      invocationAlignment = alignment;
    return RuntimePlannedRange{offset, bytes};
  };

  plan.inputRanges.reserve(manifest.inputs.size());
  for (const ExternalPortRecord &port : manifest.inputs) {
    llvm::Expected<RuntimePlannedRange> range =
        placeRange(port.bytes, port.alignment);
    if (!range)
      return range.takeError();
    plan.inputRanges.push_back(*range);
  }
  plan.outputRanges.reserve(manifest.outputs.size());
  for (const ExternalPortRecord &port : manifest.outputs) {
    llvm::Expected<RuntimePlannedRange> range =
        placeRange(port.bytes, port.alignment);
    if (!range)
      return range.takeError();
    plan.outputRanges.push_back(*range);
  }
  plan.tileRanges.reserve(entriesByLaunchSlot.size());
  for (const PackageEntrypointRecord *entry : entriesByLaunchSlot) {
    RuntimeEntryLocalRanges ranges;
    for (const TileEntryArgumentRecord &argument : entry->arguments) {
      if (const auto *workspace =
              std::get_if<WorkspaceArgument>(&argument.reference)) {
        if (ranges.workspace)
          return invalid("runtime entry has more than one workspace");
        llvm::Expected<RuntimePlannedRange> range =
            placeRange(workspace->bytes, workspace->alignment);
        if (!range)
          return range.takeError();
        ranges.workspace = *range;
      } else if (const auto *profile =
                     std::get_if<ProfileRecordArgument>(
                         &argument.reference)) {
        if (ranges.profileRecord)
          return invalid("runtime entry has more than one profile record");
        llvm::Expected<RuntimePlannedRange> range =
            placeRange(profile->bytes, profile->alignment);
        if (!range)
          return range.takeError();
        ranges.profileRecord = *range;
      } else if (const auto *status = std::get_if<TransportStatusArgument>(
                     &argument.reference)) {
        if (ranges.transportStatus)
          return invalid("runtime entry has more than one transport status");
        llvm::Expected<RuntimePlannedRange> range =
            placeRange(status->bytes, status->alignment);
        if (!range)
          return range.takeError();
        ranges.transportStatus = *range;
      }
    }
    plan.tileRanges.push_back(ranges);
  }
  if (kernel.entryABI == KernelEntryABI::TileRowPointerTable) {
    plan.pointerRows.reserve(entriesByLaunchSlot.size());
    for (const PackageEntrypointRecord *entry : entriesByLaunchSlot) {
      const uint64_t rowBytes =
          static_cast<uint64_t>(entry->arguments.size()) * sizeof(uint64_t);
      llvm::Expected<RuntimePlannedRange> range =
          placeRange(rowBytes, alignof(uint64_t));
      if (!range)
        return range.takeError();
      plan.pointerRows.push_back(*range);
    }
  }
  plan.invocationBytes = nextOffset;
  plan.invocationAlignment = invocationAlignment;
  if (plan.invocationBytes > environment.maxResourceBytes)
    return invalid("runtime environment invocation capacity is insufficient");

  // Resolve every Tile argument to a checked address.
  plan.tiles.reserve(entriesByLaunchSlot.size());
  for (int64_t launchSlot = 0; launchSlot < manifest.tileCount; ++launchSlot) {
    const PackageEntrypointRecord &entry = *entriesByLaunchSlot[launchSlot];
    const PackageModuleRecord *module =
        findModule(manifest.modules, entry.module);
    if (!module)
      return invalid("runtime entry references a missing module");

    RuntimeSessionPlan session;
    session.entry = entry.id;
    session.cardId = entry.cardId;
    session.tileId = entry.tileId;
    session.launchSlot = entry.launchSlot;
    session.module = module->id;
    session.modulePath = module->relativePath;
    for (RuntimeLaunchPhaseRole phase : manifest.launch.getPhases()) {
      const PackageModuleExportRecord *moduleExport =
          findModuleExport(*module, exportRoleForPhase(phase));
      if (!moduleExport)
        return invalid(
            "runtime entry module is missing a required typed launch phase");
      session.phases.push_back({phase, moduleExport->symbol});
    }
    session.completion = entry.completion;
    session.transport = entry.transport;
    session.argumentAddresses.reserve(entry.arguments.size());
    for (const TileEntryArgumentRecord &argument : entry.arguments) {
      if (const auto *input =
              std::get_if<ExternalInputArgument>(&argument.reference)) {
        session.argumentAddresses.push_back(
            {RuntimeArgumentAddressBase::Invocation,
             plan.inputRanges[input->port.getValue()].offset});
      } else if (const auto *tensor =
                     std::get_if<TargetTensorArgument>(&argument.reference)) {
        session.argumentAddresses.push_back(
            {RuntimeArgumentAddressBase::ProgramData,
             manifest.targetTensors[tensor->tensor.getValue()].fileOffset});
      } else if (const auto *output = std::get_if<ExternalOutputArgument>(
                     &argument.reference)) {
        session.argumentAddresses.push_back(
            {RuntimeArgumentAddressBase::Invocation,
             plan.outputRanges[output->port.getValue()].offset});
      } else if (std::holds_alternative<WorkspaceArgument>(
                     argument.reference)) {
        if (!plan.tileRanges[launchSlot].workspace)
          return invalid("runtime entry workspace range is missing");
        session.argumentAddresses.push_back(
            {RuntimeArgumentAddressBase::Invocation,
             plan.tileRanges[launchSlot].workspace->offset});
      } else if (std::holds_alternative<ProfileRecordArgument>(
                     argument.reference)) {
        if (!plan.tileRanges[launchSlot].profileRecord)
          return invalid("runtime entry profile record range is missing");
        session.argumentAddresses.push_back(
            {RuntimeArgumentAddressBase::Invocation,
             plan.tileRanges[launchSlot].profileRecord->offset});
      } else {
        if (!plan.tileRanges[launchSlot].transportStatus)
          return invalid("runtime entry transport status range is missing");
        session.argumentAddresses.push_back(
            {RuntimeArgumentAddressBase::Invocation,
             plan.tileRanges[launchSlot].transportStatus->offset});
      }
    }
    plan.tiles.push_back(std::move(session));
  }
  return plan;
}

} // namespace wafer::runtime
