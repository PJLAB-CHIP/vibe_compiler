//===- RuntimeSessionPreflight.cpp - Side-effect-free runtime planning ----===//

#include "PackageManifestInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <vector>

namespace wafer::runtime {

using detail::findModule;
using detail::findResource;
using detail::invalid;

llvm::Expected<RuntimeSessionPlan> preflightNoCardRuntimeSession(
    const VerifiedPackageManifest &package, EntryId entryId,
    llvm::ArrayRef<RuntimeInvocationBinding> invocationBindings,
    const RuntimeEnvironment &environment) {
  const PackageManifest &manifest = package.getManifest();
  if (environment.targetProfile != manifest.targetProfile ||
      environment.targetIdentity != manifest.targetIdentity ||
      environment.runtimeABI != manifest.runtimeABI ||
      environment.moduleFormat != manifest.moduleFormat)
    return invalid("runtime environment is incompatible with package target");
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
  plan.entrySymbol = entry.symbol;
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

} // namespace wafer::runtime
