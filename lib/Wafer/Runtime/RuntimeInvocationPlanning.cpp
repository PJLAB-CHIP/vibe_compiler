//===- RuntimeInvocationPlanning.cpp - Runtime invocation planning --------===//

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

llvm::Expected<RuntimeSessionPlan> buildRuntimeTilePlan(
    const PackageManifest &manifest, const PackageEntrypointRecord &entry,
    llvm::ArrayRef<const RuntimeInvocationBinding *> bindingsByResource,
    const RuntimeEnvironment &environment) {
  const PackageModuleRecord *module =
      findModule(manifest.modules, entry.module);
  if (!module)
    return invalid("runtime entry references a missing module");

  RuntimeSessionPlan plan;
  plan.entry = entry.id;
  plan.cardId = entry.cardId;
  plan.tileId = entry.tileId;
  plan.launchSlot = entry.launchSlot;
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
  plan.completion = entry.completion;
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
  return plan;
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

  const size_t transportKind = entriesByLaunchSlot.front()->transport.index();
  if (llvm::any_of(entriesByLaunchSlot, [&](const auto *entry) {
        return entry->transport.index() != transportKind;
      }))
    return invalid("runtime invocation contains mixed transport requirements");

  std::vector<const RuntimeInvocationBinding *> bindingsByResource(
      manifest.resources.size(), nullptr);
  for (const RuntimeInvocationBinding &binding : invocationBindings) {
    if (!binding.resource.isValid() ||
        binding.resource.getValue() >= manifest.resources.size() ||
        bindingsByResource[binding.resource.getValue()])
      return invalid(
          "runtime invocation contains duplicate or unknown resource");
    const PackageResourceRecord &resource =
        manifest.resources[binding.resource.getValue()];
    if (!resource.hostVisible)
      return invalid("runtime invocation contains an extra binding");
    bindingsByResource[binding.resource.getValue()] = &binding;
  }
  for (const PackageResourceRecord &resource : manifest.resources)
    if (resource.hostVisible && !bindingsByResource[resource.id.getValue()])
      return invalid(
          "runtime invocation is missing a host-visible resource binding");

  RuntimeInvocationPlan plan;
  plan.cardCount = manifest.cardCount;
  plan.tileCount = manifest.tileCount;
  plan.tiles.reserve(manifest.tileCount);
  for (int64_t launchSlot = 0; launchSlot < manifest.tileCount; ++launchSlot) {
    const PackageEntrypointRecord &entry = *entriesByLaunchSlot[launchSlot];
    llvm::Expected<RuntimeSessionPlan> tile =
        buildRuntimeTilePlan(manifest, entry, bindingsByResource, environment);
    if (!tile)
      return tile.takeError();
    plan.tiles.push_back(std::move(*tile));
  }
  return plan;
}

} // namespace wafer::runtime
