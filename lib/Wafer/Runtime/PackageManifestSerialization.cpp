//===- PackageManifestSerialization.cpp - Canonical manifest JSON --------===//

#include "PackageManifestInternal.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <variant>

namespace wafer::runtime {

std::string
serializeCanonicalPackageJson(const VerifiedPackageManifest &verified) {
  const PackageManifest &manifest = verified.getManifest();
  std::string storage;
  llvm::raw_string_ostream output(storage);
  llvm::json::OStream json(output, /*IndentSize=*/2);
  json.object([&] {
    json.attributeObject("program", [&] {
      json.attribute("id", int64_t(manifest.program.getValue()));
    });
    json.attributeObject("target", [&] {
      json.attribute("identity",
                     stringifyTargetIdentityId(manifest.targetIdentity));
      json.attribute("runtime_abi",
                     stringifyKernelRuntimeABIId(manifest.runtimeABI));
      json.attributeObject("launch", [&] {
        json.attribute("kind",
                       stringifyRuntimeLaunchKind(manifest.launch.getKind()));
        if (const auto *kernel = manifest.launch.getKernel()) {
          json.attribute("form", stringifyKernelLaunchForm(kernel->form));
          json.attribute("entry_abi",
                         stringifyKernelEntryABI(kernel->entryABI));
        } else {
          const auto *model = manifest.launch.getModel();
          json.attribute("entry_abi", stringifyModelEntryABI(model->entryABI));
        }
        json.attributeArray("phases", [&] {
          for (RuntimeLaunchPhaseRole phase : manifest.launch.getPhases())
            json.value(stringifyRuntimeLaunchPhaseRole(phase));
        });
      });
      json.attribute("module_format", manifest.moduleFormat);
    });
    json.attribute("card_count", manifest.cardCount);
    json.attribute("tile_count", manifest.tileCount);
    json.attributeArray("resources", [&] {
      for (const PackageResourceRecord &resource : manifest.resources)
        json.object([&] {
          json.attribute("id", int64_t(resource.id.getValue()));
          json.attributeObject("scope", [&] {
            if (const auto *card =
                    std::get_if<CardResourceScope>(&resource.scope)) {
              json.attribute("kind", "card");
              json.attribute("card_id", card->cardId.getValue());
              return;
            }
            const auto &tile = std::get<TileResourceScope>(resource.scope);
            json.attribute("kind", "tile");
            json.attribute("card_id", tile.cardId.getValue());
            json.attribute("tile_id", tile.tileId.getValue());
          });
          json.attribute("role", stringifyPackageResourceRole(resource.role));
          json.attribute("role_index", resource.roleIndex);
          json.attribute("name", resource.name);
          json.attributeObject("type", [&] {
            json.attribute("dtype", resource.type.dtype);
            json.attributeArray("shape", [&] {
              for (int64_t dimension : resource.type.shape)
                json.value(dimension);
            });
          });
          json.attribute("bytes", int64_t(resource.bytes));
          json.attribute("alignment", int64_t(resource.alignment));
          json.attribute("access", stringifyPackageAccessMode(resource.access));
          json.attribute("host_visible", resource.hostVisible);
        });
    });
    json.attributeArray("modules", [&] {
      for (const PackageModuleRecord &module : manifest.modules)
        json.object([&] {
          json.attribute("id", int64_t(module.id.getValue()));
          json.attribute("path", module.relativePath);
          json.attribute("digest", module.digest);
          json.attribute("format", module.format);
          json.attributeArray("exports", [&] {
            for (const PackageModuleExportRecord &moduleExport : module.exports)
              json.object([&] {
                json.attribute("role", stringifyPackageModuleExportRole(
                                           moduleExport.role));
                json.attribute("symbol", moduleExport.symbol);
              });
          });
        });
    });
    json.attributeArray("entries", [&] {
      for (const PackageEntrypointRecord &entry : manifest.entries)
        json.object([&] {
          json.attribute("id", int64_t(entry.id.getValue()));
          json.attribute("card_id", entry.cardId.getValue());
          json.attribute("tile_id", entry.tileId.getValue());
          json.attribute("launch_slot",
                         int64_t(entry.launchSlot.getValue()));
          json.attribute("module", int64_t(entry.module.getValue()));
          json.attributeArray("slots", [&] {
            for (const PackageABISlotBinding &slot : entry.slots)
              json.object([&] {
                json.attribute("ordinal", int64_t(slot.ordinal));
                json.attribute("resource", int64_t(slot.resource.getValue()));
                json.attribute("access",
                               stringifyPackageAccessMode(slot.access));
              });
          });
          json.attribute("completion",
                         stringifyPackageEntryCompletionKind(
                             entry.completion));
          json.attributeObject("transport", [&] {
            if (std::holds_alternative<NoTransportRequirements>(
                    entry.transport)) {
              json.attribute("kind", "none");
              return;
            }
            const auto &requirements =
                std::get<DirectDTETransportRequirements>(entry.transport);
            json.attribute("kind", "direct_dte");
            json.attribute("status_resource",
                           int64_t(requirements.statusResource.getValue()));
            json.attribute("status_abi", requirements.statusABI);
            json.attribute("host_watchdog_required",
                           requirements.hostWatchdogRequired);
          });
        });
    });
  });
  output << '\n';
  output.flush();
  return storage;
}

} // namespace wafer::runtime
