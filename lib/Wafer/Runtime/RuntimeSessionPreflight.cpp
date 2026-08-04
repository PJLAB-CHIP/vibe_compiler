//===- RuntimeSessionPreflight.cpp - Side-effect-free runtime planning ----===//

#include "PackageManifestInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <vector>

namespace wafer::runtime {

using detail::findModule;
using detail::findModuleExport;
using detail::findResource;
using detail::invalid;

namespace {

llvm::Error validateRuntimeEnvironment(const PackageManifest &manifest,
                                       const RuntimeEnvironment &environment) {
  if (environment.targetIdentity != manifest.targetIdentity ||
      environment.runtimeABI != manifest.runtimeABI ||
      environment.moduleFormat != manifest.moduleFormat)
    return invalid("runtime environment is incompatible with package target");
  if (const auto *kernel = manifest.launch.getKernel()) {
    if (!llvm::is_contained(environment.supportedKernelLaunchForms,
                            kernel->form) ||
        !llvm::is_contained(environment.supportedKernelEntryABIs,
                            kernel->entryABI))
      return invalid(
          "runtime environment does not support the package kernel launch "
          "contract");
  } else {
    const auto *model = manifest.launch.getModel();
    if (!llvm::is_contained(environment.supportedModelEntryABIs,
                            model->entryABI))
      return invalid(
          "runtime environment does not support the package model launch "
          "contract");
  }
  return llvm::Error::success();
}

PackageModuleExportRole exportRoleForPhase(RuntimeLaunchPhaseRole phase) {
  return phase == RuntimeLaunchPhaseRole::Prepare
             ? PackageModuleExportRole::Prepare
             : PackageModuleExportRole::Main;
}

} // namespace

llvm::Expected<RuntimeSessionPlan> preflightNoCardRuntimeSession(
    const VerifiedPackageManifest &package, EntryId entryId,
    llvm::ArrayRef<RuntimeInvocationBinding> invocationBindings,
    const RuntimeEnvironment &environment) {
  const PackageManifest &manifest = package.getManifest();
  if (llvm::Error error = validateRuntimeEnvironment(manifest, environment))
    return std::move(error);
  auto entryIterator = llvm::find_if(
      manifest.entries, [&](const auto &entry) { return entry.id == entryId; });
  if (entryIterator == manifest.entries.end())
    return invalid("runtime entry ID is not present in package");
  const PackageEntrypointRecord &entry = *entryIterator;
  const PackageModuleRecord *module =
      findModule(manifest.modules, entry.module);
  if (!module)
    return invalid("runtime entry references a missing module");
  std::vector<const RuntimeInvocationBinding *> bindingsByResource(
      manifest.resources.size(), nullptr);
  for (const RuntimeInvocationBinding &binding : invocationBindings) {
    if (!binding.resource.isValid() ||
        binding.resource.getValue() >= bindingsByResource.size() ||
        bindingsByResource[binding.resource.getValue()])
      return invalid(
          "runtime invocation contains duplicate or unknown resource");
    bindingsByResource[binding.resource.getValue()] = &binding;
  }

  RuntimeSessionPlan plan;
  plan.entry = entry.id;
  plan.logicalRank = entry.logicalRank;
  plan.module = module->id;
  plan.modulePath = module->relativePath;
  for (RuntimeLaunchPhaseRole phase : manifest.launch.getPhases()) {
    const PackageModuleExportRecord *moduleExport =
        findModuleExport(*module, exportRoleForPhase(phase));
    if (!moduleExport)
      return invalid(
          "runtime entry module is missing a required typed launch phase");
    plan.phases.push_back({phase, moduleExport->symbol});
  }
  plan.terminalCompletion = entry.terminalCompletion;
  plan.transport = entry.transport;
  if (auto *requirements =
          std::get_if<DirectDTETransportRequirements>(&entry.transport)) {
    if (!environment.supportsDirectDTE ||
        environment.directDTEStatusABI != requirements->statusABI ||
        (requirements->hostWatchdogRequired &&
         !environment.supportsHostWatchdog))
      return invalid(
          "runtime environment does not satisfy Direct DTE transport "
          "requirements");
  }
  for (const PackageABISlotBinding &slot : entry.slots) {
    const PackageResourceRecord *resource =
        findResource(manifest.resources, slot.resource);
    if (!resource)
      return invalid("runtime ABI slot references a missing resource");
    if (resource->bytes > environment.maxResourceBytes)
      return invalid("runtime environment resource capacity is insufficient");
    const RuntimeInvocationBinding *binding =
        bindingsByResource[resource->id.getValue()];
    if (resource->hostVisible) {
      if (!binding || binding->bytes < resource->bytes ||
          binding->alignment < resource->alignment ||
          binding->alignment % resource->alignment != 0 ||
          binding->access != resource->access || !binding->hostVisible)
        return invalid("runtime invocation binding does not satisfy resource");
    } else if (binding) {
      return invalid("runtime invocation must not bind internal resource");
    }
    plan.resources.push_back({resource->id, resource->role, resource->bytes,
                              resource->alignment, resource->access,
                              binding != nullptr});
    plan.launchOrder.push_back(resource->id);
  }
  for (const RuntimeInvocationBinding &binding : invocationBindings) {
    const PackageResourceRecord *resource =
        findResource(manifest.resources, binding.resource);
    if (!resource || resource->logicalRank != entry.logicalRank ||
        !resource->hostVisible)
      return invalid("runtime invocation contains an extra binding");
  }
  plan.executesBoard = false;
  return plan;
}

llvm::Expected<RuntimeInvocationPlan> preflightNoCardRuntimeInvocation(
    const VerifiedPackageManifest &package,
    llvm::ArrayRef<RuntimeInvocationBinding> invocationBindings,
    const RuntimeEnvironment &environment) {
  const PackageManifest &manifest = package.getManifest();
  if (llvm::Error error = validateRuntimeEnvironment(manifest, environment))
    return std::move(error);

  std::vector<const PackageEntrypointRecord *> entriesByRank(manifest.rankCount,
                                                             nullptr);
  for (const PackageEntrypointRecord &entry : manifest.entries)
    entriesByRank[entry.logicalRank] = &entry;

  const size_t transportKind = entriesByRank.front()->transport.index();
  if (llvm::any_of(entriesByRank, [&](const auto *entry) {
        return entry->transport.index() != transportKind;
      }))
    return invalid("runtime invocation contains mixed transport requirements");

  std::vector<std::vector<RuntimeInvocationBinding>> bindingsByRank(
      manifest.rankCount);
  std::vector<bool> seenBindings(manifest.resources.size(), false);
  for (const RuntimeInvocationBinding &binding : invocationBindings) {
    if (!binding.resource.isValid() ||
        binding.resource.getValue() >= manifest.resources.size() ||
        seenBindings[binding.resource.getValue()])
      return invalid(
          "runtime invocation contains duplicate or unknown resource");
    const PackageResourceRecord &resource =
        manifest.resources[binding.resource.getValue()];
    seenBindings[binding.resource.getValue()] = true;
    if (!resource.hostVisible)
      return invalid("runtime invocation contains an extra binding");
    bindingsByRank[resource.logicalRank].push_back(binding);
  }
  for (const PackageResourceRecord &resource : manifest.resources)
    if (resource.hostVisible && !seenBindings[resource.id.getValue()])
      return invalid(
          "runtime invocation is missing a host-visible resource binding");

  RuntimeInvocationPlan plan;
  plan.rankCount = manifest.rankCount;
  plan.ranks.reserve(manifest.rankCount);
  for (int64_t logicalRank = 0; logicalRank < manifest.rankCount;
       ++logicalRank) {
    const PackageEntrypointRecord &entry = *entriesByRank[logicalRank];
    llvm::Expected<RuntimeSessionPlan> session = preflightNoCardRuntimeSession(
        package, entry.id, bindingsByRank[logicalRank], environment);
    if (!session)
      return session.takeError();
    plan.ranks.push_back(std::move(*session));
  }
  return plan;
}

} // namespace wafer::runtime
